"""Learnable lifting neural operator on an unstructured nested mesh hierarchy ('oa-ll').

LiNO (Pandey et al. 2026) learns predict/update steps on dyadic Cartesian grids and lists
unstructured meshes as future work.  This model runs the same idea on our second-generation
hierarchy: at every level k the unknowns of the level-k mesh split into the coarse nodes (level k-1)
and the new nodes (level k); a learned PREDICT maps coarse features to the new nodes (graph message
passing over coarse neighbours within ~h_{k-1}, geometric interpolation as the base) and a learned
UPDATE corrects the coarse features from the details.  The detail coefficients of every level are
evolved by a shared local graph operator; the coarsest level gets global attention; the exact inverse
lifting maps back.  Exactly invertible for any parameters (predict/update are additive).
Per-sample cost: assembly-level (no operator factorisation) -- a mesh-native surrogate."""
import math, numpy as np, torch, torch.nn as nn
from scipy.spatial import cKDTree


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class MeshHierarchyGeometry:
    """Index tensors for the lifting levels, built once per mesh (shared by all samples)."""
    def __init__(self, H, device, min_coarse=16, max_levels=4, r_pred=1.1, r_evo=1.1):
        self.levels = []
        L = H.L
        k = L
        while k >= 1 and len(self.levels) < max_levels and len(H.nodes[k - 1]) >= min_coarse:
            nc, nk = len(H.nodes[k - 1]), len(H.nodes[k])
            m = nk - nc
            if m <= 0:
                k -= 1; continue
            cx, fx, hk1, hk = H.level_coords(k - 1), H.new_coords(k), H.h[k - 1], H.h[k]
            tree = cKDTree(cx)
            nb = tree.query_ball_point(fx, r=r_pred * hk1)
            for i, s in enumerate(nb):                            # guarantee >= 1 coarse neighbour
                if not s: nb[i] = [int(tree.query(fx[i])[1])]
            fi = np.concatenate([np.full(len(s), i) for i, s in enumerate(nb)])
            ci = np.concatenate([np.asarray(s, int) for s in nb])
            rel = (fx[fi] - cx[ci]) / hk1
            ef = np.column_stack([rel, np.linalg.norm(rel, axis=1)])
            # detail graph: new nodes of level k within ~h_k of each other (self included)
            pairs = cKDTree(fx).query_ball_point(fx, r=r_evo * hk)
            ai = np.concatenate([np.full(len(s), i) for i, s in enumerate(pairs)])
            bi = np.concatenate([np.asarray(s, int) for s in pairs])
            erl = (fx[bi] - fx[ai]) / hk                            # edge offsets in units of h_k (for evo='edge')
            erel = np.column_stack([erl, np.linalg.norm(erl, axis=1)])
            T = lambda z, dt=torch.long: torch.tensor(z, dtype=dt, device=device)
            Pn = H.P[k].tocsr()[nc:, :].tocoo()                   # geometric interpolation rows (new <- coarse)
            Pk = torch.sparse_coo_tensor(np.vstack([Pn.row, Pn.col]), Pn.data.astype('float32'),
                                         (m, nc), device=device).coalesce()
            self.levels.append(dict(
                k=k, nc=nc, m=m, fi=T(fi), ci=T(ci), ef=T(ef, torch.float32),
                deg_f=T(np.bincount(fi, minlength=m).clip(min=1), torch.float32),
                deg_c=T(np.bincount(ci, minlength=nc).clip(min=1), torch.float32),
                ai=T(ai), bi=T(bi), deg_a=T(np.bincount(ai, minlength=m).clip(min=1), torch.float32),
                erel=T(erel, torch.float32),
                Pk=Pk))
            k -= 1
        self.n_coarse = self.levels[-1]['nc'] if self.levels else len(H.nodes[L])
        # the fine mesh graph among the unknowns (model order H.nodes[L]), for the nodal post-processor (post > 0)
        pos = -np.ones(len(H.coords), dtype=np.int64); pos[H.nodes[L]] = np.arange(len(H.nodes[L]))
        el = H.elem[H.keep]
        e = np.unique(np.sort(np.concatenate([el[:, [0, 1]], el[:, [1, 2]], el[:, [2, 0]]]), 1), axis=0)
        e = e[(pos[e[:, 0]] >= 0) & (pos[e[:, 1]] >= 0)]
        src = np.concatenate([pos[e[:, 0]], pos[e[:, 1]]]); dst = np.concatenate([pos[e[:, 1]], pos[e[:, 0]]])
        xyL = H.coords[H.nodes[L]]; off = (xyL[src] - xyL[dst]) / H.h[L]
        T = lambda z, dt=torch.long: torch.tensor(z, dtype=dt, device=device)
        self.fine = dict(n=len(H.nodes[L]), src=T(src), dst=T(dst),
                         erel=T(np.column_stack([off, np.linalg.norm(off, axis=1)]), torch.float32),
                         deg=T(np.bincount(dst, minlength=len(H.nodes[L])).clip(min=1), torch.float32))

    def describe(self):
        return (f"{len(self.levels)} lifting levels "
                f"({', '.join(str(l['m']) for l in self.levels)} detail nodes), coarse {self.n_coarse}")


def _agg(vals, idx, n, deg):                                    # mean of edge values per target node
    out = vals.new_zeros(vals.shape[0], n, vals.shape[2]).index_add_(1, idx, vals)
    return out / deg[None, :, None]


class MeshLiftingNO(nn.Module):
    def __init__(self, geom, in_channels, width=24, heads=4, evo_layers=2, scaling='none', output='latent', out_channels=1, ablate='none',
                 share_levels=False, evo='mean', post=0, post_shared=False, post_edge=False, post_op=False, op_block=1,
                 linear_only=False, post_mlp=True, oa_predict=False):
        super().__init__()
        # oa_predict (3 Oct, evening): the base of the lifting's predict is the sample's OPERATOR-ADAPTED predict
        #  Phat_k X_C (route V, Phat_k ~ -A_FF^-1 A_FC of the level-k operator, per sample, set with set_oa) instead of
        #  the mean over the coarse neighbours; the learned message term is kept on top.  Still exactly invertible
        #  (the predict depends on X_C only); with zero learned terms the lifting IS the route-V change of basis
        #  x -> (x_C, x_F - Phat x_C).  No new parameters.  mathcheck/check_oa_transpose_pair.log, cheap_oa_predict.log.
        self.oa_predict, self._oa = oa_predict, None
        # Ablations of oa-ll-op (3 Oct, requested by both external reviews):
        #  linear_only: ONLY the linear path -- z = D^-1 r / sigma -> enc_lin -> nu steps s <- s + L([s, sum_j W_ij V s_j])
        #               -> dec_lin.  No lifting, no attention, no MLPs: a TRAINED polynomial of degree nu in the stencil
        #               W (= a learned fixed-coefficient smoother on the error equation, the class's linear core).
        #  post_mlp=False: oa-ll-op with the smoother's MLP correction removed (hierarchy and attention kept).
        self.linear_only, self.post_mlp = linear_only, post_mlp
        if linear_only:
            if not post_op: raise ValueError('linear_only needs post_op')
            C = max(op_block, width - width % op_block); self.C, self.g, self.op_block, self.post_op = C, geom, op_block, True
            self.post_steps, self.post_shared, self.cout, self._opw = post, False, out_channels, None
            self.enc_lin = nn.Linear(op_block, C, bias=False)               # input: the residual channel(s) only
            self.post_V = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(post)])
            self.post_L = nn.ModuleList([nn.Linear(2 * C, C, bias=False) for _ in range(post)])
            self.dec_lin = nn.Linear(C, out_channels, bias=False)
            return
        # op_block (3 Oct, vector problems): b dofs per node (b = 2 for plane elasticity).  The operator weights are
        # then the b x b blocks W_ij = D_i^-1 A_ij (D_i = A_ii, the nodal diagonal block) and the latent of every node
        # is read as C/b vectors of b components: the linear path forms sum_j W_ij (V s_j) block by block, so with the
        # MLPs at zero the smoother is weighted BLOCK Jacobi on A e = r (mathcheck/check_op_block_representability.py).
        # op_block = 1 is the scalar oa-ll-op, bit for bit.
        self.op_block = op_block
        # post_op (2 Oct, 'oa-ll-op'): OPERATOR-CONSISTENT smoother.  Each fine-mesh step has a LINEAR path through
        # the sample's own stiffness,  s_i <- s_i + L([s_i, sum_j w_ij V s_j]) + MLP([s_i, sum_j w_ij V s_j,
        # mean_j m_ij]),  w_ij = a_ij / a_ii  (set per batch with set_op), plus linear skips in the encoder and the
        # decoder.  With the MLPs at zero and the lifting at identity the network IS weighted Jacobi on the error
        # equation A e = r (input channel D^-1 r), so the class contains the two-grid method: see README 2 Oct.
        self.post_op = post_op; self._opw = None
        # post_edge: the fine-mesh smoother keeps a PERSISTENT edge latent, MeshGraphNet-style (Pfaff et al. 2021):
        # e_ij <- e_ij + MLP([e_ij, s_i, s_j]),  s_i <- s_i + MLP([s_i, mean_j e_ij]), MLPs ending in LayerNorm.
        # The edge state carries a discrete gradient across the steps -- the one GNN ingredient oa-ll-ep lacked
        # (its messages are recomputed from (s_i, s_j, x_j - x_i) at every step).  False reproduces every result.
        self.post_edge = post_edge
        # post_shared: ONE message/update pair iterated `post` times (a smoother is the same operator applied
        # repeatedly) -- depth without parameters, so the width is not squeezed by the budget
        self.post_shared = post_shared
        # post: number of edge-conditioned message-passing steps on the FINE mesh (nodal space, all unknowns) after
        # the inverse lifting -- the multigrid 'smoother' of the V-cycle this network resembles.  The lifted space is
        # block-diagonal by level, but the fine operator couples each new node to its coarse neighbours; only a
        # nodal-space operator can form that stencil.  0 (default) reproduces every earlier result.
        self.post_steps = post
        # evo: detail evolution operator.  'mean' (default, all results up to 29 Sep): MLP([d_i, mean_j d_j, log s_i])
        # -- an isotropic node-wise aggregation that cannot weight a neighbour by a PAIR quantity, so it cannot form
        # a stencil sum_j a_ij (d_j - d_i).  'edge': edge-conditioned messages m_ij = MLP([d_i, d_j, (x_j - x_i)/h, |.|]),
        # averaged over j, then d_i += MLP([d_i, mean_j m_ij, log s_i]) -- the MeshGraphNet-type local operator.
        self.evo_kind = evo
        # share_levels: ONE predict / update network and no per-level embedding for all lifting levels.  The edge
        # features are already scale-free (offsets in units of h_{k-1}), so the same weights apply to any number of
        # levels: a model trained on the level-L hierarchy runs unchanged on level L+1 (zero-shot refinement;
        # swap the geometry with set_geometry).  Default False reproduces every earlier result.
        self.share = share_levels
        self.cout = out_channels
        # ablations for the paper: 'fixed-lift' (geometric predict/update, nothing learned in the lifting),
        # 'no-attn' (no global attention on the coarsest level), 'no-stiffness' (no log-stiffness feature)
        self.ablate = ablate
        self.scaling, self.output = scaling, output
        C = max(heads, (width // heads) * heads)
        if post_op and op_block > 1:                    # whole b-vectors per node, and still a multiple of heads
            q = math.lcm(op_block, heads); C = max(q, C - C % q)
        self.g, self.C = geom, C
        nl = 1 if share_levels else len(geom.levels)
        self.enc = _mlp(in_channels, 2 * C, C)
        self.P = nn.ModuleList([_mlp(C + 3, C, C) for _ in range(nl)])          # per-level predict
        self.U = nn.ModuleList([_mlp(C + 3, C, C) for _ in range(nl)])          # per-level update
        self.evo = nn.ModuleList([_mlp(2 * C + 1, C, C) for _ in range(evo_layers)])  # shared across levels
        if evo == 'edge':
            self.evo_msg = nn.ModuleList([_mlp(2 * C + 3, C, C) for _ in range(evo_layers)])
        elif evo != 'mean':
            raise ValueError(evo)
        self.lvl = nn.Parameter(torch.zeros(nl, C))
        if ablate != 'no-attn':                         # (no-attn: the module is not built -> no dead parameters)
            self.n1, self.att = nn.LayerNorm(C), nn.MultiheadAttention(C, heads, batch_first=True)
        self.n2, self.cm = nn.LayerNorm(C), _mlp(C, 2 * C, C)
        self.dec = _mlp(C, C, out_channels)
        npost = min(post, 1) if post_shared else post
        if post_op:
            self.enc_lin = nn.Linear(in_channels, C, bias=False)
            self.dec_lin = nn.Linear(C, out_channels, bias=False)
            self.post_V = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(npost)])
            self.post_L = nn.ModuleList([nn.Linear(2 * C, C, bias=False) for _ in range(npost)])
            if post_mlp:
                self.post_msg = nn.ModuleList([_mlp(2 * C + 3 + op_block ** 2, C, C) for _ in range(npost)])
                self.post_upd = nn.ModuleList([_mlp(3 * C, C, C) for _ in range(npost)])
            else:                                       # placeholders without parameters (never called)
                self.post_msg = nn.ModuleList([nn.Identity() for _ in range(npost)])
                self.post_upd = nn.ModuleList([nn.Identity() for _ in range(npost)])
        elif post_edge:
            lnm = lambda i, o: nn.Sequential(nn.Linear(i, C), nn.GELU(), nn.Linear(C, o), nn.LayerNorm(o))
            self.post_enc_e = lnm(3, C)
            self.post_msg = nn.ModuleList([lnm(3 * C, C) for _ in range(npost)])
            self.post_upd = nn.ModuleList([lnm(2 * C, C) for _ in range(npost)])
        else:
            self.post_msg = nn.ModuleList([_mlp(2 * C + 3, C, C) for _ in range(npost)])
            self.post_upd = nn.ModuleList([_mlp(2 * C, C, C) for _ in range(npost)])
        self.head_c = _mlp(C, C, out_channels)          # coarsest-level coefficient(s)
        self.head_d = _mlp(C + 1, C, out_channels)      # detail coefficient(s) (shared over levels, + level id)
        self.synth_out = output == 'synth'

    def set_oa(self, vals, pats=None):
        """per-batch operator-adapted predict values on fixed per-level patterns: vals[l] (B, nnz_l); pats[l] = (rows,
        cols) long tensors (rows: new-node index 0..m-1, cols: coarse-node index 0..nc-1), set once."""
        if pats is not None: self._oa_pats = pats
        self._oa = vals

    def set_op(self, w):
        """per-batch operator weights on the fine graph edges, edge order of geom.fine: w_ij = a_ij / a_ii, shape
        (B, E), for scalar problems; the blocks W_ij = D_i^-1 A_ij, shape (B, E, b, b), for op_block = b > 1"""
        self._opw = w

    def set_geometry(self, geom):
        """run the trained network on another hierarchy (e.g. one level finer); needs share_levels=True unless
        the number of lifting levels is unchanged."""
        if not self.share and len(geom.levels) != len(self.P):
            raise ValueError('set_geometry to a different number of levels needs share_levels=True')
        self.g = geom

    def _predict(self, l, Xc):
        L = self.g.levels[l]; B = Xc.shape[0]
        xc = Xc[:, L['ci']]                                                     # (B, E, C)
        if self.ablate == 'fixed-lift':
            return _agg(xc, L['fi'], L['m'], L['deg_f'])                        # plain neighbour mean
        if self.oa_predict:                                                     # Phat X_C + learned messages
            rows, cols = self._oa_pats[l]
            base = Xc.new_zeros(B, L['m'], Xc.shape[2]).index_add_(1, rows, self._oa[l][..., None] * Xc[:, cols])
            msg = self.P[0 if self.share else l](torch.cat([xc, L['ef'][None].expand(B, -1, -1)], -1))
            return base + _agg(msg, L['fi'], L['m'], L['deg_f'])
        msg = xc + self.P[0 if self.share else l](torch.cat([xc, L['ef'][None].expand(B, -1, -1)], -1))
        return _agg(msg, L['fi'], L['m'], L['deg_f'])

    def _update(self, l, d):
        L = self.g.levels[l]; B = d.shape[0]
        df = d[:, L['fi']]
        if self.ablate == 'fixed-lift':
            return d.new_zeros(d.shape[0], L['nc'], d.shape[2])                   # no update step
        msg = self.U[0 if self.share else l](torch.cat([df, -L['ef'][None].expand(B, -1, -1)], -1))
        return _agg(msg, L['ci'], L['nc'], L['deg_c'])

    def _evolve(self, l, d, sk=None):
        """Detail evolution.  With the sample's operator scaling sk = diag(A_FF)^-1 (normalised per
        sample), each increment is scaled like a Jacobi step -- the operator information that gives
        oa-hb its energy-norm accuracy; log sk is also a feature (local stiffness)."""
        L = self.g.levels[l]
        d = d + self.lvl[0 if self.share else l]
        if sk is None:
            sk = d.new_ones(d.shape[0], d.shape[1])
        lsk = torch.log(sk)[..., None]
        if self.ablate == 'no-stiffness': lsk = torch.zeros_like(lsk)
        for j, f in enumerate(self.evo):
            if self.evo_kind == 'edge':
                B = d.shape[0]
                msg = self.evo_msg[j](torch.cat([d[:, L['ai']], d[:, L['bi']], L['erel'][None].expand(B, -1, -1)], -1))
                nb = _agg(msg, L['ai'], L['m'], L['deg_a'])
            else:
                nb = _agg(d[:, L['bi']], L['ai'], L['m'], L['deg_a'])
            d = d + f(torch.cat([d, nb, lsk], -1))
        return d

    def _stencil(self, vs, B):
        G = self.g.fine; bb = self.op_block
        if bb == 1:
            return self._opw[..., None] * vs[:, G['src']]
        vj = vs[:, G['src']].reshape(B, -1, self.C // bb, bb)
        return torch.einsum('nepq,necq->necp', self._opw, vj).reshape(B, -1, self.C)

    def _forward_linear(self, x):
        G = self.g.fine; B = x.shape[0]
        s = self.enc_lin(x[..., -self.op_block:])
        for fv, fl in zip(self.post_V, self.post_L):
            opa = s.new_zeros(s.shape).index_add_(1, G['dst'], self._stencil(fv(s), B))
            s = s + fl(torch.cat([s, opa], -1))
        out = self.dec_lin(s)
        return out[..., 0] if self.cout == 1 else out

    def forward(self, x, scal=None):                            # x: (B, N_L, cin) in H.nodes[L] order
        if self.linear_only:
            return self._forward_linear(x)
        lsk_scal = scal                                          # log-stiffness features always used
        if self.scaling == 'none': scal = None
        """scal: optional dict level k -> (B, m_k) Jacobi scaling diag(A_FF)^-1/2 of that level's new
        nodes (from the sample's hierarchical-basis operator data); None = purely data-driven."""
        # energy-normalised details: physical detail = w_k * coefficient, w_k = diag(A_FF)^-1/2
        # (normalised per sample).  Analysis divides, synthesis multiplies -> still exactly invertible,
        # and the synthesised field is automatically flat where the medium is stiff.
        def wk(l):
            if scal is None: return None
            v = scal[self.g.levels[l]['k']]
            return (v / v.median(1, keepdim=True).values)[..., None]
        X = self.enc(x) + (self.enc_lin(x) if self.post_op else 0); dets = []
        for l, L in enumerate(self.g.levels):                   # forward lifting, fine -> coarse
            Xc, Xf = X[:, :L['nc']], X[:, L['nc']:]
            d = Xf - self._predict(l, Xc)
            X = Xc + self._update(l, d)
            dets.append(d if scal is None else d / wk(l))
        s = X                                                    # coarsest coefficients
        if self.ablate != 'no-attn':
            s = s + self.att(*(3 * [self.n1(s)]), need_weights=False)[0]
        s = s + self.cm(self.n2(s))
        def sk(l):
            if lsk_scal is None: return None
            v = lsk_scal[self.g.levels[l]['k']] ** 2                  # diag(A_FF)^-1
            return v / v.median(1, keepdim=True).values
        dets = [self._evolve(l, d, sk(l)) for l, d in enumerate(dets)]
        if self.synth_out:
            # linear synthesis in the hierarchical FE basis, per output channel:
            # u_k = [u_{k-1}; P_new u_{k-1}] + [0; w_k c_k]  (P_new reproduces linear fields, hence rigid modes)
            u = self.head_c(s)                                     # (B, n_coarse, cout)
            for l in reversed(range(len(self.g.levels))):
                L = self.g.levels[l]; B, nc_, co = u.shape
                lvl = torch.full_like(dets[l][..., :1], 0.0 if self.share else float(l) / max(1, len(self.g.levels)))
                c = self.head_d(torch.cat([dets[l], lvl], -1))        # (B, m, cout)
                if scal is not None: c = c * wk(l)
                interp = torch.sparse.mm(L['Pk'], u.permute(1, 0, 2).reshape(nc_, B * co))
                interp = interp.reshape(-1, B, co).permute(1, 0, 2)  # (B, m, cout)
                u = torch.cat([u, interp + c], 1)
            return u[..., 0] if self.cout == 1 else u
        for l in reversed(range(len(self.g.levels))):            # exact inverse lifting (latent decoder)
            dl = dets[l] if scal is None else dets[l] * wk(l)
            Xc = s - self._update(l, dl)
            s = torch.cat([Xc, dl + self._predict(l, Xc)], 1)
        if self.post_steps:
            G = self.g.fine; B = s.shape[0]
            steps = ([(self.post_msg[0], self.post_upd[0])] * self.post_steps if self.post_shared
                     else list(zip(self.post_msg, self.post_upd)))
            if self.post_op:
                bb = self.op_block
                if bb == 1:
                    w = self._opw[..., None]                              # (B, E, 1): a_ij / a_ii, row i = dst
                    ef = torch.cat([G['erel'][None].expand(B, -1, -1), w], -1)
                    stencil = lambda vs: w * vs[:, G['src']]
                else:
                    W = self._opw                                         # (B, E, b, b): D_i^-1 A_ij, row block i = dst
                    ef = torch.cat([G['erel'][None].expand(B, -1, -1), W.reshape(B, W.shape[1], bb * bb)], -1)
                    def stencil(vs):                                      # (W_ij v_j) for each of the C/b vectors
                        vj = vs[:, G['src']].reshape(B, -1, self.C // bb, bb)
                        return torch.einsum('nepq,necq->necp', W, vj).reshape(B, -1, self.C)
                Vs = list(zip(self.post_V, self.post_L)) * (self.post_steps if self.post_shared else 1)
                for (fm, fu), (fv, fl) in zip(steps, Vs):
                    vs = fv(s)
                    opa = s.new_zeros(s.shape).index_add_(1, G['dst'], stencil(vs))   # sum_j W_ij V s_j
                    nl = _agg(fm(torch.cat([s[:, G['src']], s[:, G['dst']], ef], -1)), G['dst'], G['n'], G['deg']) if self.post_mlp else None
                    s = s + fl(torch.cat([s, opa], -1)) + (fu(torch.cat([s, opa, nl], -1)) if self.post_mlp else 0)
            elif self.post_edge:
                e = self.post_enc_e(G['erel'])[None].expand(B, -1, -1)
                for fm, fu in steps:
                    e = e + fm(torch.cat([e, s[:, G['src']], s[:, G['dst']]], -1))
                    s = s + fu(torch.cat([s, _agg(e, G['dst'], G['n'], G['deg'])], -1))
            else:
                for fm, fu in steps:
                    msg = fm(torch.cat([s[:, G['src']], s[:, G['dst']], G['erel'][None].expand(B, -1, -1)], -1))
                    s = s + fu(torch.cat([s, _agg(msg, G['dst'], G['n'], G['deg'])], -1))
        out = self.dec(s) + (self.dec_lin(s) if self.post_op else 0)
        return out[..., 0] if self.cout == 1 else out


def variant(kind):
    """constructor options of the oa-ll family by model name: oa-ll (per-level lifting, mean evolution),
    oa-ll-s (shared lifting), oa-ll-e (edge-conditioned evolution), oa-ll-e4 (edge, 4 evolution layers),
    oa-ll-se (shared + edge), oa-ll-ep (edge + 4 fine-mesh post steps), oa-ll-p (mean evolution + 4 post steps)"""
    table = {'oa-ll': {}, 'oa-ll-s': dict(share_levels=True), 'oa-ll-e': dict(evo='edge'),
             'oa-ll-e4': dict(evo='edge', evo_layers=4), 'oa-ll-se': dict(share_levels=True, evo='edge'),
             'oa-ll-ep': dict(evo='edge', post=4), 'oa-ll-p': dict(post=4),
             # 8 post steps; 2 attention heads so the width is not rounded to a multiple of 4 (uses the budget fully)
             'oa-ll-p8': dict(post=8, heads=2), 'oa-ll-ep8': dict(evo='edge', post=8, heads=2),
             # shared lifting + edge evolution + 4 post steps: the transferable version of oa-ll-ep
             'oa-ll-sep': dict(share_levels=True, evo='edge', post=4),
             # edge evolution + a 10-step fine-mesh smoother with ONE shared step (the GNN's depth, not its parameters)
             'oa-ll-epr': dict(evo='edge', post=10, post_shared=True),
             # oa-ll-ep with a MeshGraphNet-type smoother: persistent edge latents + LayerNorm (2 Oct)
             'oa-ll-epm': dict(evo='edge', post=4, post_edge=True),
             # operator-consistent smoother (contains weighted Jacobi on A e = r exactly); needs set_op + residual input
             'oa-ll-op': dict(evo='edge', post=4, post_op=True),
             # ablations of oa-ll-op (3 Oct): linear path only (a trained degree-4 polynomial smoother on A e = r), and
             # the full model without the smoother's MLP correction
             'oa-ll-op-lin': dict(post=4, post_op=True, linear_only=True),
             'oa-ll-op-nomlp': dict(evo='edge', post=4, post_op=True, post_mlp=False),
             # oa-ll-op with the operator-adapted (route V) predict as the lifting's base (3 Oct, evening)
             'oa-ll-op-oa': dict(evo='edge', post=4, post_op=True, oa_predict=True),
             # ablation: oa-ll-ep architecture with the residual input of oa-ll-op (input vs operator structure)
             'oa-ll-epz': dict(evo='edge', post=4),
             # oa-ll-op with the hierarchy COMPLEMENTARY to the exact coarse space (3 Oct): lifting only on levels
             # k0+1..L (geometry built with max_levels = L - k0 by the driver) and no global attention -- after an
             # exact coarse solve the correction is local (LOD), so coarser levels / global coupling are redundant
             'oa-ll-opl': dict(evo='edge', post=4, post_op=True, ablate='no-attn', heads=1)}  # heads=1: no attention, so no width rounding
    if kind not in table: raise ValueError(kind)
    return table[kind]


OA_LL_KINDS = ('oa-ll', 'oa-ll-s', 'oa-ll-e', 'oa-ll-e4', 'oa-ll-se', 'oa-ll-ep', 'oa-ll-p', 'oa-ll-p8', 'oa-ll-ep8', 'oa-ll-sep', 'oa-ll-epr')
