"""oa_layer_torch.py -- the trainable OA-lifting neural-operator layer (formulation section 5).

    v_{t+1} = GELU( W v_t  +  T_a  diag(d_k) R~_theta diag(d_k)  T_a^T  M v_t  +  b )

  T_a        operator-adapted lifting transform of the sample (per-sample update C_k(a)),
             applied matrix-free: synth / dual exactly as in oa_lifting.OATransform
  d_k        Jacobi scaling diag(D_k)^-1/2 of the sample          (section 5.2, rule 2)
  R~_theta   ONE set of weights shared by all levels                 (rule 1)
             = local graph kernel on the level-k nodes, radius r h_k
             + low-rank global term  U U^T,  U = phi(node features) (rule 3, section 7)
  M          lumped mass (= the f = 1 load vector)

Batching: all per-sample sparse matrices of a batch are stacked block-diagonally,
so one layer is a handful of sparse matmuls regardless of batch size.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch
import warnings
warnings.filterwarnings("ignore", message="Sparse invariant checks")
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.spatial import cKDTree

from oa_lifting import OATransform


def _to_torch_sparse(M, device, dtype=torch.float32):
    M = sp.coo_matrix(M)
    idx = torch.tensor(np.vstack([M.row, M.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(M.data, dtype=dtype), M.shape,
                                   device=device).coalesce()


# ---------------------------------------------------------------------------
# per-sample operator data
# ---------------------------------------------------------------------------
def nodal_log_a(H, a_elem):
    """element log-a averaged to the interior finest nodes."""
    L = H.L
    n_all = len(H.coords)
    s = np.zeros(n_all); c = np.zeros(n_all)
    kp = getattr(H, 'keep', np.ones(len(H.elem), bool))       # only elements inside the domain
    la = np.log(np.asarray(a_elem)[kp])
    for j in range(3):
        np.add.at(s, H.elem[kp, j], la); np.add.at(c, H.elem[kp, j], 1)
    return (s / np.maximum(c, 1))[H.nodes[L]]


def frozen_C(H, A_ref, rho=2, route='V'):
    """Adaptation matrices of the frozen transform: identical for every sample, built once."""
    from oa_lifting import _local_C
    C, Ak = {}, A_ref.tocsc()
    for k in range(H.L, 0, -1):
        P, E = H.P[k], H.E[k]
        nc = P.shape[1]
        if route == 'V':
            A_FF = (E.T @ Ak @ E).tocsc(); A_FC = (E.T @ Ak)[:, :nc].tocsc()
            C[k] = _local_C(A_FF, A_FC, H.new_coords(k), H.level_coords(k - 1),
                            rho * H.h[k - 1]).tocsr()
            Phi = sp.vstack([sp.identity(nc, format='csr'), C[k]]).tocsc()
            Ak = (Phi.T @ Ak @ Phi).tocsc()
        else:
            Ac = (P.T @ Ak @ P).tocsc()
            C[k] = _local_C(Ac, (P.T @ Ak @ E).tocsc(), H.level_coords(k - 1), H.new_coords(k),
                            rho * H.h[k - 1]).tocsr()
            Ak = Ac
    return C


def sample_operator_data(H, a_elem, mode='local', rho=2, C_frozen=None, route='V'):
    """What the layer needs for one sample: update matrices C_k and the Jacobi scaling
    d_k = diag(D_k)^-1/2.  Only diag(D_k) is formed (D_k itself is never needed here).
    mode: 'local' (per-sample Algorithm 0, radius rho), 'hb' (C = 0), 'frozen' (C_frozen given)."""
    from oa_lifting import _local_C
    Ak = H.A_fine(a_elem).tocsc()
    C, d = {}, {}
    for k in range(H.L, 0, -1):
        P, E = H.P[k], H.E[k]
        nc = P.shape[1]
        if route == 'V':
            # operator in the PREDICT: Phi = [I; Phat], Psi = E, D_k = A_FF
            A_FF = (E.T @ Ak @ E).tocsc()
            A_FC = (E.T @ Ak)[:, :nc].tocsc()
            if mode == 'local':
                Ck = _local_C(A_FF, A_FC, H.new_coords(k), H.level_coords(k - 1),
                              rho * H.h[k - 1]).tocsr()
            elif mode == 'hb':
                Ck = (E.T @ P).tocsr()                     # geometric predict
            elif mode == 'frozen':
                Ck = C_frozen[k]
            else:
                raise ValueError(mode)
            d[k] = 1.0 / np.sqrt(A_FF.diagonal())
            C[k] = sp.coo_matrix(Ck)
            Phi = sp.vstack([sp.identity(nc, format='csr'), Ck]).tocsc()
            Ak = (Phi.T @ Ak @ Phi).tocsc()                # = Schur complement for the exact predict
            continue
        # route W: operator in the UPDATE
        Ac = (P.T @ Ak @ P).tocsc()
        if mode == 'local':
            Ck = _local_C(Ac, (P.T @ Ak @ E).tocsc(), H.level_coords(k - 1), H.new_coords(k),
                          rho * H.h[k - 1])
        elif mode == 'hb':
            Ck = sp.csr_matrix((Ac.shape[0], E.shape[1]))
        elif mode == 'frozen':
            Ck = C_frozen[k]
        else:
            raise ValueError(mode)
        Psi = (E + P @ Ck).tocsc()
        d[k] = 1.0 / np.sqrt(np.asarray((Ak @ Psi).multiply(Psi).sum(0)).ravel())
        C[k] = sp.coo_matrix(Ck)
        Ak = Ac
    d[0] = 1.0 / np.sqrt(Ak.diagonal()) if Ak.shape[0] else np.zeros(0)
    A = H.A_fine(a_elem).tocoo()          # kept for the energy-norm loss (same pattern every sample)
    return dict(C=C, d=d, loga=nodal_log_a(H, a_elem), route=route,
                A=(A.row.astype(np.int32), A.col.astype(np.int32), A.data))


class SharedGeometry:
    """Shared (sample-independent) sparse structure: P_k, E_k, neighbour graphs."""

    def __init__(self, H, radius=2.0, device='cpu'):
        self.H, self.L, self.device = H, H.L, device
        self.n = {k: len(H.nodes[k]) for k in range(self.L + 1)}
        self.m = {k: (len(H.new[k]) if k > 0 else len(H.nodes[0])) for k in range(self.L + 1)}
        self.P = {k: H.P[k].tocsr() for k in range(1, self.L + 1)}
        # coarse selection [I 0] (route V uses Phi = R0^T + E Phat instead of the geometric P)
        self.R0 = {k: sp.eye(self.n[k], format='csr')[:self.n[k - 1]].tocsr()
                   for k in range(1, self.L + 1)}
        self.E = {k: H.E[k].tocsr() for k in range(1, self.L + 1)}
        self.mass = H.F.copy()                                  # lumped mass = f=1 load
        self.scale = 4.0 ** self.L                              # h_L^-2: keeps dual coefficients O(1)
        # global ids of the level-k block (for node features) and coordinates
        fine = {g: i for i, g in enumerate(H.nodes[self.L])}
        self.block_ids = {0: np.array([fine[g] for g in H.nodes[0]], int)}
        for k in range(1, self.L + 1):
            self.block_ids[k] = np.array([fine[g] for g in H.new[k]], int)
        self.edges = {}
        for k in range(self.L + 1):
            xy = H.coords[H.nodes[0]] if k == 0 else H.new_coords(k)
            hk = H.h[max(k, 1)]
            if len(xy) == 0:
                self.edges[k] = (np.zeros((2, 0), int), np.zeros((0, 2))); continue
            tr = cKDTree(xy)
            pairs = tr.query_pairs(radius * hk + 1e-12, output_type='ndarray')
            src = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(len(xy))])
            dst = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(len(xy))])
            rel = (xy[src] - xy[dst]) / hk
            self.edges[k] = (np.vstack([src, dst]), rel)


class OABatch:
    """Block-diagonal stacking of B samples -> torch sparse operators."""

    def __init__(self, G: SharedGeometry, samples, device='cpu'):
        B = len(samples); L = G.L; dev = device
        self.B, self.G = B, G
        # shared prolongation / injection, block-stacked once per batch size and cached
        cache = G.__dict__.setdefault('_kron_cache', {})
        if B not in cache:
            I = sp.identity(B, format='csr')
            cache[B] = dict(
                P={k: _to_torch_sparse(sp.kron(I, G.P[k]), dev) for k in range(1, L + 1)},
                Pt={k: _to_torch_sparse(sp.kron(I, G.P[k].T), dev) for k in range(1, L + 1)},
                E={k: _to_torch_sparse(sp.kron(I, G.E[k]), dev) for k in range(1, L + 1)},
                Et={k: _to_torch_sparse(sp.kron(I, G.E[k].T), dev) for k in range(1, L + 1)},
                R0={k: _to_torch_sparse(sp.kron(I, G.R0[k]), dev) for k in range(1, L + 1)},
                R0t={k: _to_torch_sparse(sp.kron(I, G.R0[k].T), dev) for k in range(1, L + 1)})
        self.P, self.Pt, self.E, self.Et = (cache[B][x] for x in ('P', 'Pt', 'E', 'Et'))
        self.R0, self.R0t = cache[B]['R0'], cache[B]['R0t']
        self.route = samples[0].get('route', 'W')
        # per-sample update matrices: concatenate COO triplets with block offsets
        self.C, self.Ct = {}, {}
        for k in range(1, L + 1):
            # route W: C_k is (n_{k-1} x m_k); route V: Phat_k is (m_k x n_{k-1})
            nc, mk = (G.n[k - 1], G.m[k]) if samples[0].get('route', 'W') == 'W' else (G.m[k], G.n[k - 1])
            rows = np.concatenate([s['C'][k].row + i * nc for i, s in enumerate(samples)])
            cols = np.concatenate([s['C'][k].col + i * mk for i, s in enumerate(samples)])
            vals = torch.tensor(np.concatenate([s['C'][k].data for s in samples]),
                                dtype=torch.float32, device=dev)
            r_t = torch.tensor(rows, dtype=torch.long, device=dev)
            c_t = torch.tensor(cols, dtype=torch.long, device=dev)
            self.C[k] = torch.sparse_coo_tensor(torch.stack([r_t, c_t]), vals, (B * nc, B * mk)).coalesce()
            self.Ct[k] = torch.sparse_coo_tensor(torch.stack([c_t, r_t]), vals, (B * mk, B * nc)).coalesce()
        self.d = {k: torch.tensor(np.concatenate([s['d'][k] for s in samples]),
                                  dtype=torch.float32, device=dev)[:, None]
                  for k in range(L + 1)}
        self.mass = torch.tensor(np.tile(G.mass * G.scale, B), dtype=torch.float32, device=dev)[:, None]
        loga = np.stack([s['loga'] for s in samples])          # (B, N)
        self.loga = torch.tensor(loga, dtype=torch.float32, device=dev)
        # per-level node features: (log a at node, log d_k), stacked over the batch
        self.feat = {}
        for k in range(L + 1):
            ids = G.block_ids[k]
            la = loga[:, ids].reshape(-1)
            ld = np.log(np.concatenate([s['d'][k] for s in samples]) + 1e-30)
            self.feat[k] = torch.tensor(np.stack([la, ld], 1), dtype=torch.float32, device=dev)
        # shared edge tensors (same graph for every sample)
        if not hasattr(G, '_edges_t'):
            G._edges_t = {k: (torch.tensor(G.edges[k][0][0], dtype=torch.long, device=dev),
                              torch.tensor(G.edges[k][0][1], dtype=torch.long, device=dev),
                              torch.tensor(G.edges[k][1], dtype=torch.float32, device=dev)[None])
                          for k in range(L + 1)}
        self.G_edges_t = G._edges_t
        self.batch_of = {k: torch.arange(B, device=dev).repeat_interleave(G.m[k]) for k in range(L + 1)}
        # fine stiffness of each sample, stacked block-diagonally (energy-norm loss)
        if 'A' in samples[0]:
            N = G.n[L]
            r = np.concatenate([s['A'][0] + i * N for i, s in enumerate(samples)])
            c = np.concatenate([s['A'][1] + i * N for i, s in enumerate(samples)])
            v = np.concatenate([s['A'][2] for s in samples])
            self.A = torch.sparse_coo_tensor(
                torch.tensor(np.vstack([r, c]), dtype=torch.long, device=dev),
                torch.tensor(v, dtype=torch.float32, device=dev), (B * N, B * N)).coalesce()
        else:
            self.A = None

    def energy_sq(self, v):
        """per-sample v^T A v for v of shape (B*N, 1)"""
        return (v * torch.sparse.mm(self.A, v)).view(self.B, -1).sum(1)

    # T^T r   (r: (B*N, C))  -> dict of (B*m_k, C)
    def dual(self, r):
        out = {}
        for k in range(self.G.L, 0, -1):
            if self.route == 'V':                      # Phi^T r = R0 r + Phat^T (E^T r)
                out[k] = torch.sparse.mm(self.Et[k], r)
                r = torch.sparse.mm(self.R0[k], r) + torch.sparse.mm(self.Ct[k], out[k])
            else:
                Pr = torch.sparse.mm(self.Pt[k], r)
                out[k] = torch.sparse.mm(self.Et[k], r) + torch.sparse.mm(self.Ct[k], Pr)
                r = Pr
        out[0] = r
        return out

    # T w
    def synth(self, w):
        s = w[0]
        for k in range(1, self.G.L + 1):
            if self.route == 'V':                      # Phi c + E d = R0^T c + E (Phat c + d)
                s = torch.sparse.mm(self.R0t[k], s) + \
                    torch.sparse.mm(self.E[k], torch.sparse.mm(self.C[k], s) + w[k])
            else:
                s = torch.sparse.mm(self.P[k], s + torch.sparse.mm(self.C[k], w[k])) + \
                    torch.sparse.mm(self.E[k], w[k])
        return s


# ---------------------------------------------------------------------------
class LevelKernel(nn.Module):
    """R~: local graph kernel + low-rank global term, shared across levels."""

    def __init__(self, width, rank=16, hidden=32, n_levels=8, rank_scale=None):
        """rank       : fixed rank for every level, or the CAP when rank_scale is set
        rank_scale : if set, level k uses r_k = min(rank, ceil(rank_scale * sqrt(m_k))).
                     Measured (exp_level_spectra, connected channels, levels 4-6): the number of
                     small Jacobi-scaled eigenvalues grows ~ sqrt(m_k) (11, 34, 63 for m_k = 176,
                     736, 3008), i.e. with channel LENGTH in nodes; rank_scale ~ 1.2 covers it."""
        super().__init__()
        self.rank = rank
        self.rank_scale = rank_scale
        self.level_emb = nn.Embedding(n_levels + 1, 4)
        fin = 2 + 4
        self.edge_mlp = nn.Sequential(nn.Linear(2 + 2 * fin, hidden), nn.GELU(),
                                      nn.Linear(hidden, width))
        self.W_nb = nn.Linear(width, width, bias=False)
        self.W_self = nn.Linear(width, width, bias=False)
        if rank > 0:
            self.phi = nn.Sequential(nn.Linear(fin, hidden), nn.GELU(), nn.Linear(hidden, rank))
            self.W_lr_in = nn.Linear(width, width, bias=False)

    def forward(self, g, k, batch: OABatch):
        B, m = batch.B, batch.G.m[k]
        C = g.shape[1]
        feat = batch.feat[k]
        emb = self.level_emb.weight[k].expand(feat.shape[0], -1)
        f = torch.cat([feat, emb], 1).view(B, m, -1)            # every sample has the same m_k
        gb = g.view(B, m, C)
        y = self.W_self(gb)
        src, dst, rel = batch.G_edges_t[k]
        if src.numel():
            gate = self.edge_mlp(torch.cat([rel.expand(B, -1, -1), f[:, src], f[:, dst]], 2))  # (B,E,C)
            msg = gate * self.W_nb(gb)[:, src]
            y = y.index_add(1, dst, msg)
        if self.rank > 0:
            U = self.phi(f)                                       # (B, m, r)
            if self.rank_scale is not None:
                rk = min(self.rank, int(np.ceil(self.rank_scale * np.sqrt(m))))
                U = U[..., :rk]
            coeff = torch.einsum('bmr,bmc->brc', U, self.W_lr_in(gb)) / m
            y = y + torch.einsum('bmr,brc->bmc', U, coeff)
        return y.reshape(B * m, C)


class OALiftingLayer(nn.Module):
    def __init__(self, width, rank=16, hidden=32, n_levels=8, rank_scale=None):
        super().__init__()
        self.R = LevelKernel(width, rank, hidden, n_levels, rank_scale)
        self.W = nn.Linear(width, width)

    def forward(self, v, batch: OABatch):
        g = batch.dual(v * batch.mass)
        w = {k: batch.d[k] * self.R(batch.d[k] * g[k], k, batch) for k in g if g[k].shape[0]}
        for k in g:
            if k not in w: w[k] = g[k]
        return Fn.gelu(self.W(v) + batch.synth(w))


class OALiftingNO(nn.Module):
    """P (pointwise) -> n_layers OA-lifting layers -> Q (pointwise)."""

    def __init__(self, width=32, n_layers=4, rank=16, hidden=32, n_levels=8, rank_scale=None,
                 extra_in=0):
        """extra_in: extra per-node input channels, e.g. the exact coarse solution (--k0)."""
        super().__init__()
        self.extra_in = extra_in
        self.P = nn.Sequential(nn.Linear(4 + extra_in, width), nn.GELU(), nn.Linear(width, width))
        self.layers = nn.ModuleList([OALiftingLayer(width, rank, hidden, n_levels, rank_scale)
                                     for _ in range(n_layers)])
        self.Q = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, 1))

    def forward(self, batch: OABatch, xy, la_mean, la_std, extra=None):
        B = batch.B
        la = (batch.loga - la_mean) / la_std                     # (B, N)
        x = torch.cat([la.reshape(-1, 1), xy.repeat(B, 1),
                       torch.ones(la.numel(), 1, device=la.device)], 1)
        if self.extra_in:                       # e.g. the exact coarse solution (--k0)
            x = torch.cat([x, extra.reshape(-1, self.extra_in)], 1)
        v = self.P(x)
        for layer in self.layers:
            v = layer(v, batch)
        return self.Q(v).reshape(B, -1)


def n_params(m):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in m.parameters())
