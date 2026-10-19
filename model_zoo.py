"""Shared model construction / forward / zero-shot transfer for the problem drivers (train_scalar.py,
train_allen_cahn.py, train_elasticity.py).  One place defines every model, so every class is compared with the
same code.

Level data D (a dict built by the driver) must hold:
    H          NestedHierarchy of the level
    FEAT       (n, n_all_nodes, cin) node features (normalised with training-level statistics)
    X_int      torch (n, nI, cin)  features at the unknowns, H.nodes[L] order      (oa-ll)
    X_dom      torch (n, n_dom, cin) features at every domain node                 (transolver, gnn)
    dom, take_dom  domain node ids / positions of the unknowns inside X_dom
    UC         (n, nI * cout) coarse-solve input (added back to the prediction)
Models:
    oa-ll      learnable lifting on the mesh hierarchy, one predict/update per level (the paper's model so far)
    oa-ll-s    the same with ONE predict/update shared by all levels -> runs on a finer hierarchy unchanged
    oa-ll-e    oa-ll with edge-conditioned detail evolution (MeshGraphNet-type messages); oa-ll-e4: 4 layers;
               oa-ll-se: shared lifting + edge evolution  (see mesh_lifting.variant)
    gnn        MeshGraphNet-style encode-process-decode on the mesh edges
    transolver mesh-native transformer (Wu et al. 2024)
    lino, wno, fno   grid models through the mesh<->grid bridge; 'wno' is our light WNO layer
    wno-off    the published WNO (Tripura & Chakraborty, Version 2.0.0 of their repository)
"""
import numpy as np, torch, torch.nn as nn
from mesh_grid_bridge import MeshGridBridge

GRID_KINDS = ('lino', 'wno', 'wno-off', 'fno', 'lino-op', 'wno-op', 'fno-op')
# error-equation input kinds (residual D^-1 r / sigma instead of u_c, output x sigma); OP_W_KINDS also get the operator
# weights.  fno-op / wno-op / lino-op / transolver-op (3 Oct): standard NOs given the SAME residual input.
OP_W_KINDS = ('oa-ll-op', 'oa-ll-opl', 'oa-ll-op-lin', 'oa-ll-op-nomlp', 'gnn-op')
OP_KINDS = OP_W_KINDS + ('transolver-op', 'lino-op', 'wno-op', 'fno-op')
MESH_KINDS = ('oa-ll', 'oa-ll-s', 'oa-ll-e', 'oa-ll-e4', 'oa-ll-se', 'oa-ll-ep', 'oa-ll-p', 'oa-ll-p8', 'oa-ll-ep8', 'oa-ll-sep', 'oa-ll-epr', 'oa-ll-epm', 'oa-ll-op', 'oa-ll-opl', 'oa-ll-op-lin', 'oa-ll-op-nomlp', 'gnn-op', 'gnn', 'gnn30', 'transolver', 'transolver-op')


def n_params(m): return sum(p.numel() for p in m.parameters())


class Zoo:
    def __init__(self, cin, dev, cout=1, grid_ext='nearest', batch=8, scale=1.0, U_test=None):
        self.cin, self.dev, self.cout, self.grid_ext, self.batch, self.scale = cin, dev, cout, grid_ext, batch, scale
        self.U_test = U_test

    # ---------------------------------------------------------------- grids
    def grid_data(self, D, ng=None, verbose=False, te=None, op=False):
        key = ('grid_z' if op else 'grid', ng)
        if key not in D:
            H_ = D['H']; BR = MeshGridBridge(H_, ng=ng)
            full = (D['FEAT_z'] if op else D['FEAT']).copy(); full[:, ~H_.in_domain] = 0
            chans = [BR.node_field_to_grid(full[..., c], self.grid_ext) for c in range(full.shape[-1])]
            g = np.stack(chans + [np.broadcast_to(BR.mask.astype(float), chans[0].shape)], -1)
            co = BR.to_nodes.tocoo()
            TN = torch.sparse_coo_tensor(np.vstack([co.row, co.col]), co.data.astype('float32'), co.shape,
                                         device=self.dev).coalesce()
            D[key] = (BR, torch.tensor(g, dtype=torch.float32, device=self.dev), TN)
        return D[key]

    def grid_floor(self, D, U_te, ng=None):
        """round trip mesh -> grid -> mesh of exact fields (n_te, nI*cout), per extension; mean rel-l2"""
        BR = self.grid_data(D, ng)[0]; out = {}
        c = self.cout; nI = U_te.shape[1] // c
        for ext in ('zero', 'nearest'):
            rt = np.stack([BR.round_trip_floor(U_te.reshape(len(U_te), nI, c)[..., j], ext)[1] for j in range(c)], -1)
            rt = rt.reshape(len(U_te), -1)
            out[ext] = (rt, float(np.mean(np.linalg.norm(rt - U_te, axis=1) / np.linalg.norm(U_te, axis=1))))
        return out

    # ---------------------------------------------------------------- error-equation inputs (2 Oct)
    def op_setup(self, D, A_list, F, tr):
        """oa-ll-op / gnn-op data for a level: residual of the coarse solve r = F - A u_c as D^-1 r normalised per
        sample (replaces the u_c input channel, which must be the LAST feature channel), the per-sample scale sigma
        (output multiplied back: exact linearity in the scale of r), and the operator weights a_ij / a_ii on the fine
        graph (oa-ll-op) and on the domain graph (gnn-op; 0 on edges touching a non-unknown node).  Scalar problems."""
        from mesh_lifting import MeshHierarchyGeometry
        from baselines_extra import mesh_edges
        H = D['H']; L = H.L
        if 'MG' not in D: D['MG'] = MeshHierarchyGeometry(H, self.dev)
        fs, fd = np.array(D['MG'].fine['src'].cpu().numpy()), np.array(D['MG'].fine['dst'].cpu().numpy())
        ed, ef = mesh_edges(H.coords, H.elem[H.keep], D['dom'])
        posL = -np.ones(len(H.coords), dtype=np.int64); posL[H.nodes[L]] = np.arange(len(H.nodes[L]))
        gs, gd = posL[D['dom'][ed[0]]], posL[D['dom'][ed[1]]]; ok = (gs >= 0) & (gd >= 0)
        Z, W, WG = [], [], []
        for i, A in enumerate(A_list):
            A = A.tocsr(); dg = A.diagonal()
            Z.append((F - A @ D['UC'][i]) / dg)
            W.append(np.asarray(A[fd, fs]).ravel() / dg[fd])
            wg = np.zeros(len(gs)); wg[ok] = np.asarray(A[np.array(gd[ok]), np.array(gs[ok])]).ravel() / dg[gd[ok]]
            WG.append(wg)
        Z = np.array(Z); sig = np.sqrt(np.mean(Z ** 2, 1)) + 1e-30
        T = lambda v: torch.tensor(v, dtype=torch.float32, device=self.dev)
        D['SIG'] = T(sig / sig[tr].mean())
        D['X_int_z'] = D['X_int'].clone(); D['X_int_z'][..., -1] = T(Z / sig[:, None])
        zf = np.zeros((len(Z), len(H.coords))); zf[:, H.nodes[L]] = Z / sig[:, None]
        D['X_dom_z'] = D['X_dom'].clone(); D['X_dom_z'][..., -1] = T(zf[:, D['dom']])
        D['FEAT_z'] = D['FEAT'].copy(); D['FEAT_z'][..., -1] = zf          # all nodes (grid models)
        D['OPW'], D['OPWG'] = T(np.array(W)), T(np.array(WG))

    def op_setup_block(self, D, A_list, F, tr, b=2):
        """The same for a vector problem with b dofs per node (interleaved, dof b*p + c for unknown p): the nodal
        diagonal blocks D_i = A_ii (b x b, SPD as principal submatrices of A) define block Jacobi,
            z_i = D_i^-1 r_i  (input, replaces the b u_c channels, which must be the LAST b feature channels),
            W_ij = D_i^-1 A_ij  (b x b per edge, row block i = destination) on the fine graph (oa-ll-op) and on the
                   domain graph (gnn-op, flattened to b*b edge features; 0 on edges touching a non-unknown node).
        Block Jacobi rather than point Jacobi: lambda_max(D^-1 A) <= 3 for ANY coefficient on P1 triangles (A is a sum
        of element matrices, each <= 3 x its nodal block diagonal), so one damping omega = 2/3 serves every sample; and
        the blocks rotate with the frame (Q^T W_ij Q), the point diagonal does not.  See README 3 Oct."""
        from mesh_lifting import MeshHierarchyGeometry
        from baselines_extra import mesh_edges
        H = D['H']; L = H.L; nI = len(H.nodes[L])
        if 'MG' not in D: D['MG'] = MeshHierarchyGeometry(H, self.dev)
        fs, fd = D['MG'].fine['src'].cpu().numpy(), D['MG'].fine['dst'].cpu().numpy()
        ed, ef = mesh_edges(H.coords, H.elem[H.keep], D['dom'])
        posL = -np.ones(len(H.coords), dtype=np.int64); posL[H.nodes[L]] = np.arange(nI)
        gs, gd = posL[D['dom'][ed[0]]], posL[D['dom'][ed[1]]]; ok = (gs >= 0) & (gd >= 0)
        ii = np.arange(nI)
        def blocks(A, I, J):                                   # (len(I), b, b): A[b I + p, b J + q]
            out = np.empty((len(I), b, b))
            for p in range(b):
                for q in range(b):
                    out[:, p, q] = np.asarray(A[b * I + p, b * J + q]).ravel()
            return out
        Z, W, WG = [], [], []
        for i, A in enumerate(A_list):
            A = A.tocsr()
            Dinv = np.linalg.inv(blocks(A, ii, ii))                # (nI, b, b)
            r = (F - A @ D['UC'][i]).reshape(nI, b)
            Z.append(np.einsum('ipq,iq->ip', Dinv, r).ravel())
            W.append(np.einsum('epq,eqr->epr', Dinv[fd], blocks(A, fd, fs)))
            wg = np.zeros((len(gs), b, b))
            wg[ok] = np.einsum('epq,eqr->epr', Dinv[gd[ok]], blocks(A, gd[ok], gs[ok]))
            WG.append(wg)
        Z = np.array(Z); sig = np.sqrt(np.mean(Z ** 2, 1)) + 1e-30
        T = lambda v: torch.tensor(v, dtype=torch.float32, device=self.dev)
        D['SIG'] = T(sig / sig[tr].mean())
        Zn = (Z / sig[:, None]).reshape(len(Z), nI, b)
        D['X_int_z'] = D['X_int'].clone(); D['X_int_z'][..., -b:] = T(Zn)
        zf = np.zeros((len(Z), len(H.coords), b)); zf[:, H.nodes[L]] = Zn
        D['X_dom_z'] = D['X_dom'].clone(); D['X_dom_z'][..., -b:] = T(zf[:, D['dom']])
        D['FEAT_z'] = D['FEAT'].copy(); D['FEAT_z'][..., -b:] = zf          # all nodes (grid models)
        D['OPW'], D['OPWG'] = T(np.array(W)), T(np.array(WG))

    # ---------------------------------------------------------------- models
    def make(self, kind, w, D0, ng0=None, ll_output='latent'):
        cin, co = self.cin, self.cout
        if kind in ('transolver-op', 'lino-op', 'wno-op', 'fno-op'):
            kind = kind[:-3]                        # same architecture; the inputs differ (see forward)
        if kind == 'gnn-op':
            from baselines_extra import MeshGNN, mesh_edges
            H_ = D0['H']; ed, ef = mesh_edges(H_.coords, H_.elem[H_.keep], D0['dom'])
            return MeshGNN(cin, ed, ef, width=w, steps=10, out_channels=co, op_edge=True, op_dim=co * co)
        if kind.startswith('oa-ll'):
            from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant
            if 'MG' not in D0: D0['MG'] = MeshHierarchyGeometry(D0['H'], self.dev)
            if kind == 'oa-ll-opl':                    # lifting only on the levels the coarse solve does not cover
                if 'MG_loc' not in D0:
                    D0['MG_loc'] = MeshHierarchyGeometry(D0['H'], self.dev, max_levels=max(1, D0['H'].L - D0.get('k0', 0))
                                                         if D0.get('k0', 0) else 4)
                return MeshLiftingNO(D0['MG_loc'], cin, width=w, out_channels=co, output=ll_output,
                                     **dict(variant(kind), **({'op_block': co} if co > 1 else {})))
            kw = variant(kind)
            if kw.get('post_op') and co > 1: kw['op_block'] = co       # vector problem: 2x2-block smoother
            return MeshLiftingNO(D0['MG'], cin, width=w, out_channels=co, output=ll_output, **kw)
        if kind == 'transolver':
            from baselines_extra import Transolver
            return Transolver(cin, hidden=w, out_channels=co)
        if kind in ('gnn', 'gnn30'):
            from baselines_extra import MeshGNN, mesh_edges
            H_ = D0['H']; ed, ef = mesh_edges(H_.coords, H_.elem[H_.keep], D0['dom'])
            return MeshGNN(cin, ed, ef, width=w, steps=30 if kind == 'gnn30' else 10, out_channels=co)
        c2 = cin + 1                                           # + domain mask channel
        if kind == 'lino':
            from baselines_extra import LiNO2D
            return LiNO2D(c2, hidden=w, levels=3, out_channels=co)
        if kind == 'wno-off':
            from baselines_extra import WNO2dOfficial
            return WNO2dOfficial(c2, width=w, size=(ng0, ng0), out_channels=co)
        if kind in ('fno', 'wno'):
            from ofwno_nets import RealFNO2D, RealWNO2D
            m = RealFNO2D(c2, hidden=w, modes=min(12, ng0 // 2), n_layers=4) if kind == 'fno' else \
                RealWNO2D(c2, hidden=w, n_layers=4, J=2)
            m.output_proj[2] = nn.Conv2d(w, co, 1)
            return m
        raise ValueError(kind)

    def fit_width(self, kind, budget, D0, ng0=None, **kw):
        """largest width whose model has <= budget parameters (None if even the smallest does not fit)"""
        lo, hi = 2, 256
        while hi - lo > 1:
            mid = (lo + hi) // 2
            try: ok = n_params(self.make(kind, mid, D0, ng0, **kw)) <= budget
            except Exception: ok = False
            if ok: lo = mid
            else: hi = mid
        return lo if n_params(self.make(kind, lo, D0, ng0, **kw)) <= budget else None

    def forward(self, kind, model, D, idx, ng=None):
        """(B, nI*cout) prediction, unknown ordering H.nodes[L] (dofs interleaved when cout > 1)"""
        B = len(idx)
        if kind in OP_W_KINDS and kind != 'gnn-op':
            model.set_op(D['OPW'][idx])
            return model(D['X_int_z'][idx]).reshape(B, -1) * D['SIG'][idx][:, None]
        if kind == 'transolver-op':
            return model(D['X_dom_z'][idx])[:, D['take_dom']].reshape(B, -1) * D['SIG'][idx][:, None]
        if kind in ('lino-op', 'wno-op', 'fno-op'):
            sig = D['SIG'][idx][:, None]
            return self._grid_out(kind[:-3], model, D, idx, ng, op=True) * sig
        if kind == 'gnn-op':
            model.set_op(D['OPWG'][idx])
            return model(D['X_dom_z'][idx])[:, D['take_dom']].reshape(B, -1) * D['SIG'][idx][:, None]
        if kind.startswith('oa-ll'):
            return model(D['X_int'][idx]).reshape(B, -1)
        if kind in ('transolver', 'gnn', 'gnn30'):
            return model(D['X_dom'][idx])[:, D['take_dom']].reshape(B, -1)
        return self._grid_out(kind, model, D, idx, ng)

    def _grid_out(self, kind, model, D, idx, ng, op=False):
        B = len(idx)
        BR, Xg, TN = self.grid_data(D, ng, op=op)
        out = model(Xg[idx])
        if kind in ('fno', 'wno') and self.cout > 1:
            out = out.permute(0, 2, 3, 1)                      # (B, c, n, n) -> (B, n, n, c)
        n2 = TN.shape[1]; c = self.cout
        flat = out.reshape(B, n2, c).permute(1, 0, 2).reshape(n2, B * c)
        return torch.sparse.mm(TN, flat).reshape(-1, B, c).permute(1, 0, 2).reshape(B, -1)

    def rebind(self, kind, model, D):
        """point a trained mesh model at another level's mesh"""
        if kind.startswith('oa-ll'):
            from mesh_lifting import MeshHierarchyGeometry
            model.set_geometry(MeshHierarchyGeometry(D['H'], self.dev))
        if kind in ('gnn', 'gnn30', 'gnn-op'):
            from baselines_extra import mesh_edges
            H_ = D['H']; ed, ef = mesh_edges(H_.coords, H_.elem[H_.keep], D['dom'])
            model.set_graph(ed, ef)

    def predict(self, kind, model, D, idx_all, ng=None):
        model.eval(); P = []
        with torch.no_grad():
            for idx in np.array_split(idx_all, int(np.ceil(len(idx_all) / self.batch))):
                P.append(self.forward(kind, model, D, torch.tensor(idx, device=self.dev), ng))
        return torch.cat(P).cpu().numpy().astype(float) * self.scale + D['UC'][idx_all]

    def note(self, kind, D0, ng0=None):
        if kind.startswith('oa-ll'):
            from mesh_lifting import MeshHierarchyGeometry
            if 'MG' not in D0: D0['MG'] = MeshHierarchyGeometry(D0['H'], self.dev)
            if kind == 'oa-ll-opl' and 'MG_loc' in D0: return D0['MG_loc'].describe() + ', no global attention'
            return D0['MG'].describe() + (', levels share one predict/update' if kind == 'oa-ll-s' else '')
        if kind in GRID_KINDS: return self.grid_data(D0, None)[0].report() + (', error-equation inputs' if kind.endswith('-op') else '')
        return f"{len(D0['dom'])} mesh nodes, native"
