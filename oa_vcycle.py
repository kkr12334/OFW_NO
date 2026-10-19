"""oa-vc: a learned ONE-PASS V-cycle with operator-adapted transfers (3 Oct 2026, evening).

Why this structure (mathcheck/check_oa_transpose_pair.log).  Route V on the nested hierarchy,
    Phi_k = [I; Phat_k],  Phat_k ~ -A_FF^-1 A_FC,  A_{k-1} = Phi_k^T A_k Phi_k,
block-diagonalises A when Phat_k is exact, and A^-1 = W D^-1 W^T: restriction is the TRANSPOSE of prolongation.  An
exactly invertible lifting (oa-ll, LiNO) analyses with the INVERSE of its synthesis transform; with the restricted
(level-block-diagonal) latent map of oa-ll it cannot represent W^-1 A^-1 W = D^-1 W^T W (oa-ll-op-oa: +5% only).  This is a
statement about that architecture class, not about invertible transforms in general (independent review C, 5 Oct).
Note (review, 5 Oct): OpSmoother(mlp=False) still applies the encoder / decoder MLPs and the sigma scaling, so it is
positively homogeneous but NOT linear in the residual; it is not a residual-linear control.  With the cheap Phat (m Jacobi steps of prolongator smoothing from the geometric predict) the block-diagonal
pass neglects the inter-level coupling Phat leaves; the multiplicative form -- a V-cycle with Galerkin level operators --
does not (one classical V(4,4)-cycle from zero: barriers 0.138, channels 0.141 energy; geometric transfers 0.343 / 0.230).

The network.  Exact, non-learned: the transfers Phi_k (per sample), the Galerkin level operators A_k, the residual
updates b - A_k x, the coarsest solve A_0^-1 (28 nodes, dense).  Learned: the pre- and post-smoother at every level, an
OPERATOR-CONSISTENT network on A_k x = b (the oa-ll-op smoother block: linear path s <- s + L([s, sum_j W_ij V s_j]) with
W_ij = A_ij / A_ii, plus an MLP message term), positively homogeneous through its own input scaling:
    S(b) = sigma_b * net(D^-1 b / sigma_b),  sigma_b = rms(D^-1 b).
So the whole map r -> e is positively homogeneous in r, and the class CONTAINS the classical V(nu,nu)-cycle with these
transfers and nu-step weighted Jacobi (or Chebyshev, C >= 2) smoothers -- the Proposition of oa-ll-op applied per level.
"""
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn
from torch.utils.checkpoint import checkpoint
from scipy.sparse.linalg import splu


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


# ---------------------------------------------------------------------------------------------------------------
# per-sample operator data (numpy / scipy)
# ---------------------------------------------------------------------------------------------------------------
def oa_hierarchy(H, A, n_levels, m=4, omega=2 / 3):
    """levels l = 0 .. n_levels-1 (l = 0 finest, level k = L - l): (A_k csr, Phat_k csr (F x C)); returns (levels, A_0)."""
    Ak = A.tocsr(); out = []
    for l in range(n_levels):
        k = H.L - l; nc = H.P[k].shape[1]
        A_FF, A_FC = Ak[nc:, nc:].tocsr(), Ak[nc:, :nc].tocsr(); X = H.P[k][nc:].tocsr()
        if m:
            Dinv = sp.diags(1 / A_FF.diagonal())
            for _ in range(m):
                X = (X + omega * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
        out.append((Ak, X))
        Phi = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr(); Ak = (Phi.T @ Ak @ Phi).tocsr()
    return out, Ak


def level_graphs(H, n_levels):
    """boolean adjacency (with diagonal) of the geometric level-k meshes, l = 0 .. n_levels-1 (k = L - l): the
    nested-P1 Galerkin pattern P_k^T G_k P_k is exactly the level-(k-1) mesh graph."""
    one = lambda M: (abs(M) > 0).astype(np.float64).tocsr()
    G = one(H.A_fine(np.ones(H.n_elem))); out = []
    for l in range(n_levels):
        k = H.L - l; out.append(G); G = one(H.P[k].T @ G @ H.P[k])
    return out


def loc_patterns(H, n_levels, r):
    """localisation patterns for Phat_k (F x C of level k): coarse node c may interpolate to fine node f iff their
    graph distance on the level-k mesh is <= r.  r is in level-k mesh widths (r = 1: the geometric predict's own
    support), so the support of every basis function -- and the stencil width of every Galerkin operator -- is bounded
    independently of the number of levels (Owhadi's localisation of gamblets to rho-patches; in AMG terms
    energy-minimising prolongation on a prescribed sparsity pattern)."""
    one = lambda M: (abs(M) > 0).astype(np.float64).tocsr()
    out = []
    for l, G in enumerate(level_graphs(H, n_levels)):
        k = H.L - l; nc = H.P[k].shape[1]; R = G.copy()
        for _ in range(r - 1): R = one(R @ G)
        out.append(one(R[nc:, :nc]))
    return out


def oa_hierarchy_loc(H, A, n_levels, m=4, omega=2 / 3, r=2, masks=None):
    """oa_hierarchy with every Jacobi step of the prolongator smoothing restricted to the localisation pattern
    (loc_patterns) and projected so that each row of the update keeps the row sum the UNRESTRICTED step would give
    (Euclidean projection onto {row sums fixed} within the pattern): constants are interpolated exactly wherever the
    unrestricted iteration does (A 1 = 0 rows), Dirichlet rows keep their behaviour.  Galerkin levels unchanged."""
    masks = masks if masks is not None else loc_patterns(H, n_levels, r)
    Ak = A.tocsr(); out = []
    for l in range(n_levels):
        k = H.L - l; nc = H.P[k].shape[1]; M = masks[l]
        A_FF, A_FC = Ak[nc:, nc:].tocsr(), Ak[nc:, :nc].tocsr(); X = H.P[k][nc:].tocsr()
        cnt = np.asarray(M.sum(1)).ravel()
        if m:
            dinv = 1 / A_FF.diagonal(); Dinv = sp.diags(dinv)
            for _ in range(m):
                D = (omega * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
                Dm = D.multiply(M).tocsr()
                c = (np.asarray(D.sum(1)).ravel() - np.asarray(Dm.sum(1)).ravel()) / cnt
                X = (X + Dm + sp.diags(c) @ M).tocsr()
        out.append((Ak, X))
        Phi = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr(); Ak = (Phi.T @ Ak @ Phi).tocsr()
    return out, Ak


def filtered(Ak, G):
    """A^F: the entries of A_k on the level-k mesh graph G (boolean, with diagonal); every dropped entry is added to the
    diagonal of its row, so row sums -- and hence A^F 1 = A 1 -- are kept (smoothed aggregation's filtered matrix,
    Vanek-Mandel-Brezina 1996).  On the fine level of a P1 discretisation A^F = A."""
    Ak = Ak.tocsr(); Af = Ak.multiply(G).tocsr()
    drop = np.asarray(Ak.sum(1)).ravel() - np.asarray(Af.sum(1)).ravel()
    return (Af + sp.diags(drop)).tocsr()


def strength_graph(Ak, theta):
    """boolean graph of the strong connections |A_ij| >= theta sqrt(A_ii A_jj) (with the diagonal)."""
    C = Ak.tocoo(); d = Ak.diagonal()
    keep = (C.row == C.col) | (np.abs(C.data) >= theta * np.sqrt(d[C.row] * d[C.col]))
    return sp.csr_matrix((np.ones(keep.sum()), (C.row[keep], C.col[keep])), shape=Ak.shape)


def oa_hierarchy_filt(H, A, n_levels, m=4, omega=2 / 3, graphs=None, theta=None):
    """oa_hierarchy with the prolongator smoothing done with the FILTERED operator A^F_k (filtered): Phat_k has support
    within m + 1 level-k mesh widths of its coarse node on every level, constants are interpolated exactly (A^F 1 = A 1),
    the fine level is unchanged (A^F = A there).  Galerkin levels A_{k-1} = Phi^T A_k Phi with the unfiltered A_k."""
    graphs = graphs if (graphs is not None or theta is not None) else level_graphs(H, n_levels)
    Ak = A.tocsr(); out = []
    for l in range(n_levels):
        k = H.L - l; nc = H.P[k].shape[1]
        Af = filtered(Ak, strength_graph(Ak, theta) if theta is not None else graphs[l])
        A_FF, A_FC = Af[nc:, nc:].tocsr(), Af[nc:, :nc].tocsr(); X = H.P[k][nc:].tocsr()
        if m:
            Dinv = sp.diags(1 / A_FF.diagonal())
            for _ in range(m):
                X = (X + omega * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
        out.append((Ak, X))
        Phi = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr(); Ak = (Phi.T @ Ak @ Phi).tocsr()
    return out, Ak


def structural_patterns(H, n_levels, m=4, A_struct=None):
    """fixed per-level sparsity patterns (boolean products, so no sample has an entry outside them).
    A_struct (5 Oct): a fine operator with the full structural pattern of the problem class (e.g. a random TENSOR
    coefficient, whose stiffness is nonzero on edges where every isotropic one cancels); default: random isotropic a."""
    one = lambda M: (abs(M) > 0).astype(np.float64).tocsr()
    # random positive coefficient: entries that cancel only for special a (e.g. a = 1, cot t1 + cot t2 = 0) must be in
    # the pattern; entries zero for EVERY a (both opposite angles right) are dropped from the sample values too
    As = one(A_struct if A_struct is not None else H.A_fine(np.random.default_rng(12345).uniform(1, 2, H.n_elem))); pats = []
    for l in range(n_levels):
        k = H.L - l; nc = H.P[k].shape[1]
        Xs = one(H.P[k][nc:])
        for _ in range(m):
            Xs = one(Xs + As[nc:, :nc] + As[nc:, nc:] @ Xs)
        pats.append((As.tocoo(), Xs.tocoo()))
        Phi = sp.vstack([sp.identity(nc, format='csr'), Xs]).tocsr(); As = one(Phi.T @ As @ Phi)
    return pats, As.shape[0]


def _on_pattern(M, pat):
    """values of sparse M on the (sorted) pattern pat (coo); every entry of M must lie in the pattern."""
    keys = pat.row.astype(np.int64) * pat.shape[1] + pat.col
    o = np.argsort(keys); ks = keys[o]
    C = M.tocoo(); nz = C.data != 0; C = sp.coo_matrix((C.data[nz], (C.row[nz], C.col[nz])), C.shape)
    kk = C.row.astype(np.int64) * pat.shape[1] + C.col
    pos = np.searchsorted(ks, kk)
    assert np.all(ks[np.minimum(pos, len(ks) - 1)] == kk), 'entry outside the structural pattern'
    v = np.zeros(len(ks)); v[pos] = C.data
    return v, o


def near_kernel_basis(A, theta=0.1, tau=0.05):
    """Near-kernel vectors of A from A alone (4 Oct): connected components of the STRONG-connection graph
    (-A_ij >= theta sqrt(A_ii A_jj)) that contain no STRONGLY Dirichlet-coupled row (row sum >= theta A_ii; P1 rows away
    from Dirichlet nodes sum to zero).  Such a component is a region coupled to the rest only through weak connections, so its indicator z
    has z^T A z / z^T D z ~ (weak / strong): a floating island of a high-contrast medium.  Kept only if that Rayleigh
    quotient is < tau (the definition of near-kernel; small clusters of low-conductivity nodes have ~0.5 and are dropped).
    An island that splits into weakly coupled pieces is spanned by the pieces' indicators.  Returns Z (N x k), k >= 0."""
    from scipy.sparse.csgraph import connected_components
    A = A.tocsr(); d = A.diagonal(); C = A.tocoo()
    off = C.row != C.col
    strong = off & (-C.data >= theta * np.sqrt(d[C.row] * d[C.col]))
    S = sp.csr_matrix((np.ones(strong.sum()), (C.row[strong], C.col[strong])), shape=A.shape)
    n_, lab = connected_components(S, directed=False)
    rs = np.asarray(A.sum(1)).ravel()
    # anchored = STRONGLY coupled to a Dirichlet node (lost row mass >= theta * A_ii); a coupling through a
    # low-conductivity element (row sum ~ contrast * A_ii) leaves the component floating
    anchored = np.zeros(n_, bool); anchored[lab[rs >= theta * d]] = True
    cols = []
    for c in np.nonzero(~anchored)[0]:                 # keep only genuine near-kernel vectors: z^T A z / z^T D z < tau
        z = (lab == c).astype(float)
        if z @ (A @ z) < tau * (z @ (d * z)): cols.append(z)
    return np.array(cols).T if cols else np.zeros((A.shape[0], 0))


class OAData:
    """everything oa-vc needs, for all samples, as tensors on fixed patterns."""
    def __init__(self, H, a_list, n_levels, m, device, coords_fn, near_kernel=False, A_list=None, A_struct=None):
        # A_list / A_struct (5 Oct): the fine operators themselves (any SPD stiffness on H.nodes[L], e.g. a tensor
        # coefficient) instead of H.A_fine(a) for a in a_list; default behaviour unchanged.
        pats, n0 = structural_patterns(H, n_levels, m, A_struct)
        self.n_levels, self.m, self.L, self.lv = n_levels, m, [], []
        orders = []
        for l, (pA, pX) in enumerate(pats):
            k = H.L - l
            kA = pA.row.astype(np.int64) * pA.shape[1] + pA.col; oA = np.argsort(kA)
            kX = pX.row.astype(np.int64) * pX.shape[1] + pX.col; oX = np.argsort(kX)
            r, c = pA.row[oA], pA.col[oA]; off = r != c
            xy = coords_fn(k); rel = (xy[c] - xy[r]) / H.h[k]
            T = lambda z, dt=torch.long: torch.tensor(np.ascontiguousarray(z), dtype=dt, device=device)
            self.lv.append(dict(k=k, n=pA.shape[0], nc=H.P[k].shape[1],
                                Ar=T(r), Ac=T(c), off=T(np.nonzero(off)[0]), diag=T(np.nonzero(~off)[0]),
                                diag_row=T(r[~off]),
                                erel=T(np.column_stack([rel, np.linalg.norm(rel, axis=1)])[off], torch.float32),
                                deg=T(np.bincount(r[off], minlength=pA.shape[0]).clip(min=1), torch.float32),
                                Xr=T(pX.row[oX]), Xc=T(pX.col[oX])))
        Avals = [[] for _ in pats]; Xvals = [[] for _ in pats]; A0inv = []; Zs = []
        for A_ in (A_list if A_list is not None else (H.A_fine(a) for a in a_list)):
            levs, A0 = oa_hierarchy(H, A_, n_levels, m)
            if near_kernel: Zs.append(near_kernel_basis(levs[0][0]))
            for l, ((Ak, X), (pA, pX)) in enumerate(zip(levs, pats)):
                Avals[l].append(_on_pattern(Ak, pA)[0]); Xvals[l].append(_on_pattern(X, pX)[0])
            A0inv.append(np.linalg.inv(A0.toarray()))
        f32 = lambda z: torch.tensor(np.array(z), dtype=torch.float32, device=device)
        self.Av = [f32(v) for v in Avals]; self.Xv = [f32(v) for v in Xvals]; self.A0inv = f32(A0inv); self.n0 = n0
        # 5 Oct (review, point 3): the fine-level operator ALSO in float64, exactly as assembled (Avals[0] are the values of
        # the scipy matrix on its pattern).  The output safeguard is evaluated with it, not with the float32 copy the cycle
        # uses: casting the rounded float32 values back to double would not recover the assembled operator.
        self.A64 = torch.tensor(np.array(Avals[0]), dtype=torch.float64, device=device)
        self.Z, self.k_nk = None, [z.shape[1] for z in Zs]
        if near_kernel:                                        # padded with zero columns (masked in the projection)
            km = max(1, max(self.k_nk)); N = pats[0][0].shape[0]
            Zp = np.zeros((len(Zs), N, km))
            for i, z in enumerate(Zs): Zp[i, :, :z.shape[1]] = z
            self.Z = f32(Zp)

    def batch(self, idx):
        return [v[idx] for v in self.Av], [v[idx] for v in self.Xv], self.A0inv[idx]

    def batch_z(self, idx):
        return None if self.Z is None else self.Z[idx]

    def batch_a64(self, idx):
        return self.A64[idx]


# ---------------------------------------------------------------------------------------------------------------
# the network
# ---------------------------------------------------------------------------------------------------------------
class OpSmoother(nn.Module):
    """b -> correction x for A x = b on one level: S(b) = sigma * net(D^-1 b / sigma), operator-consistent steps."""
    def __init__(self, C, nu, mlp=True, linear=False):
        super().__init__()
        # linear (5 Oct, review point 5): the RESIDUAL-LINEAR control.  Only the linear path is kept -- linear encoder of
        # z / sigma alone (the log-diagonal feature would add an affine offset), the operator-weighted linear steps, linear
        # decoder -- so S(b) = sigma L(z / sigma) = L(D^-1 b): linear in b for every A, A-dependent through W_ij and D, and
        # it still contains nu-step weighted Jacobi.  (mlp=False alone is NOT linear: encoder / decoder MLPs remain.)
        self.C, self.nu, self.linear = C, nu, linear
        self.mlp = mlp and not linear
        self.enc_lin = nn.Linear(2, C, bias=False); self.enc = None if linear else _mlp(2, 2 * C, C)
        self.V = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(nu)])
        self.Lw = nn.ModuleList([nn.Linear(2 * C, C, bias=False) for _ in range(nu)])
        if mlp:
            self.msg = nn.ModuleList([_mlp(2 * C + 4, C, C) for _ in range(nu)])
            self.upd = nn.ModuleList([_mlp(3 * C, C, C) for _ in range(nu)])
        self.dec_lin = nn.Linear(C, 1, bias=False); self.dec = None if linear else _mlp(C, C, 1)

    def forward(self, b, lv, Avals):
        B = b.shape[0]; d = Avals[:, lv['diag']]                               # (B, n) diagonal, pattern order = rows
        dg = b.new_zeros(B, lv['n']).index_add_(1, lv['diag_row'], d)
        z = b / dg
        # sigma = scaled rms (5 Oct): max|z| * rms(z / max|z|) -- no float32 underflow of z^2 for tiny residuals.  An exactly
        # zero right-hand side is handled explicitly: S(0) = 0 (positive homogeneity at lambda -> 0), with safe values in the
        # masked branch so that the backward pass stays finite (check_projection_precision.log, 4b).
        mz = z.abs().amax(1, keepdim=True); nz = mz > 0
        mz = torch.where(nz, mz, torch.ones_like(mz))
        q = (z / mz).pow(2).mean(1, keepdim=True)                               # >= 1/n when z != 0
        sig = mz * torch.where(nz, q, torch.ones_like(q)).sqrt()                # = 1 on the masked (z = 0) rows
        lsd = torch.log(dg / dg.median(1, keepdim=True).values)
        x = torch.stack([z / sig, torch.zeros_like(lsd) if self.linear else lsd], -1)   # (B, n, 2)
        s = self.enc_lin(x) if self.linear else self.enc(x) + self.enc_lin(x)
        r_, c_ = lv['Ar'][lv['off']], lv['Ac'][lv['off']]
        w = (Avals[:, lv['off']] / dg[:, r_])[..., None]                        # W_ij = A_ij / A_ii (row i)
        ef = torch.cat([lv['erel'][None].expand(B, -1, -1), w], -1)
        for t in range(self.nu):
            opa = s.new_zeros(s.shape).index_add_(1, r_, w * self.V[t](s)[:, c_])
            ds = self.Lw[t](torch.cat([s, opa], -1))
            if self.mlp:
                nl = s.new_zeros(s.shape).index_add_(1, r_, self.msg[t](torch.cat([s[:, r_], s[:, c_], ef], -1))) / lv['deg'][None, :, None]
                ds = ds + self.upd[t](torch.cat([s, opa, nl], -1))
            s = s + ds
        y = self.dec_lin(s) if self.linear else self.dec_lin(s) + self.dec(s)
        return torch.where(nz, y[..., 0] * sig, torch.zeros_like(b))


class OAVCycle(nn.Module):
    def __init__(self, data: OAData, width=16, nu=4, share=True, mlp=True, project=False, deflate=False, ckpt=False,
                 linear=False):
        super().__init__()
        # ckpt (4 Oct): activation checkpointing of every smoother call during training -- the smoother's activations are
        #   recomputed in the backward pass instead of stored.  Same map, same gradients (to round-off); memory ~1 smoother
        #   instead of 2 x n_levels smoothers (L4 batch 8 ran out of a 16 GB GPU without it); ~30% more training time.
        self.ckpt = ckpt
        # project (4 Oct): output = Galerkin projection of the error onto span{pre-smoother part, coarse-correction part,
        #   post-smoother part (fine level), near-kernel vectors}: e = x0 + W y, W^T A W y = W^T (r - A x0).  Energy-optimal
        #   in that span, so the energy error never exceeds that of x0 (and of u_c); positively homogeneous; no new
        #   parameters.  deflate: x0 = Z (Z^T A Z)^-1 Z^T r before the cycle (Z = near-kernel vectors, OAData).
        #   project='eval' (oa-vcd): trained without the projection (it makes the loss blind to the scale of each part and
        #   training stalls: L2 channels 0.039 vs 0.0057), applied at evaluation, where it can only lower the energy error.
        #   project='both' (oa-vcb): trained on loss(projected) + loss(plain cycle) -- the plain term keeps every part
        #   well scaled, the projected term teaches the parts to complement each other; evaluation uses the projection.
        self.project, self.deflate, self._Z = project, deflate, None
        self.lv, self.n_levels = data.lv, data.n_levels
        nets = 1 if share else self.n_levels
        self.pre = nn.ModuleList([OpSmoother(width, nu, mlp, linear) for _ in range(nets)])
        self.post = nn.ModuleList([OpSmoother(width, nu, mlp, linear) for _ in range(nets)])
        self.share = share

    def set_ops(self, Av, Xv, A0inv, Z=None, A64=None):
        # A64: float64 values of the assembled fine operator (OAData.batch_a64).  Without it the safeguard uses the stored
        # operator cast to double, i.e. it holds in the norm of the ROUNDED operator only (mathchecks with float64 data).
        self._A, self._X, self._A0inv, self._Z, self._A64 = Av, Xv, A0inv, Z, A64

    def _matvec64(self, x):
        lv = self.lv[0]; Av = self._A64 if getattr(self, '_A64', None) is not None else self._A[0].double()
        return x.new_zeros(x.shape[0], lv['n']).index_add_(1, lv['Ar'], Av * x[:, lv['Ac']])

    def _matvec(self, l, x):
        lv = self.lv[l]
        return x.new_zeros(x.shape[0], lv['n']).index_add_(1, lv['Ar'], self._A[l] * x[:, lv['Ac']])

    def _restrict(self, l, r):                                                  # Phi^T r = r_C + Phat^T r_F
        lv = self.lv[l]; nc = lv['nc']
        return r[:, :nc].index_add(1, lv['Xc'], self._X[l] * r[:, nc:][:, lv['Xr']])

    def _prolong(self, l, e):                                                   # Phi e = [e; Phat e]
        lv = self.lv[l]
        f = e.new_zeros(e.shape[0], lv['n'] - lv['nc']).index_add_(1, lv['Xr'], self._X[l] * e[:, lv['Xc']])
        return torch.cat([e, f], 1)

    def _smooth(self, S, b, l):
        if self.ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(lambda b_: S(b_, self.lv[l], self._A[l]), b, use_reentrant=False)
        return S(b, self.lv[l], self._A[l])

    def _cycle(self, l, b, parts=False):
        j = 0 if self.share else l
        x = self._smooth(self.pre[j], b, l)
        rc = self._restrict(l, b - self._matvec(l, x))
        if l + 1 < self.n_levels:
            ec = self._cycle(l + 1, rc)
        else:
            ec = torch.bmm(self._A0inv, rc[..., None])[..., 0]
        xc = self._prolong(l, ec)
        xp = self._smooth(self.post[j], b - self._matvec(l, x + xc), l)
        return (x, xc, xp) if parts else x + xc + xp

    def _galerkin(self, W, b):
        """Correction d = W y, y from the Jacobi-scaled, ridge-regularised normal equations of min ||e - W y||_A, A e = b.
        Ideal (exact arithmetic, no ridge): d is the A-orthogonal projection of e onto span W (exact minimisation).
        Implemented (5 Oct, after the independent review, finding B and its follow-up):
          * everything in float64: W, b (the caller passes the float64 residual), the fine operator (OAData.A64, the
            assembled values), AW, G = W^T A W (symmetrised), g = W^T b, the solve and d = W y;
          * ridge (G_s + 1e-10 I) y_s = g_s on the Jacobi-scaled G (zero / padding columns masked): with an accurate PSD G
            the energy change is -y^T G y - 2 y^T R y <= 0 -- non-increase, but NOT exact minimisation over span W;
          * acceptance on the correction itself, against the float64 operator:  2 d^T b - d^T A d >= 0  (energy
            non-increase).  If it fails (round-off in a near-singular solve), d is replaced by the exact line-search step
            d * max(d^T b, 0) / d^T A d, whose change -(d^T b)_+^2 / d^T A d is never positive; this may shrink or reject
            the correction (alpha = 0) and is then not the minimiser either.
        No eigendecomposition in the forward pass (its gradient is ill-defined at repeated eigenvalues / rank changes);
        the numerical rank of G_s is only measured, without gradient, when self.collect_stats is set.
        Returns d in float64."""
        B, N, m = W.shape
        Wd, bd = W.double(), b.double()
        AW = torch.stack([self._matvec64(Wd[..., j]) for j in range(m)], -1)
        G = Wd.transpose(1, 2) @ AW; g = Wd.transpose(1, 2) @ bd[..., None]
        G = 0.5 * (G + G.transpose(1, 2))
        dg = torch.diagonal(G, dim1=1, dim2=2); live = dg > 1e-30 * dg.amax(1, keepdim=True).clamp_min(1e-300)
        # "double where": the masked branch gets a safe input, otherwise rsqrt'(0) = inf and 0 * inf = NaN in backward
        # (mathcheck/check_projection_precision.log 4b: NaN parameter gradients at a zero residual before this)
        sc = torch.where(live, torch.where(live, dg, torch.ones_like(dg)).rsqrt(), torch.zeros_like(dg))
        Gs = sc[:, :, None] * G * sc[:, None, :] + torch.diag_embed((~live).double() + 1e-10)
        y = sc[..., None] * torch.linalg.solve(Gs, sc[..., None] * g)
        d = (Wd @ y)[..., 0]
        db = (d * bd).sum(1); dAd = (d * self._matvec64(d)).sum(1)
        bad = (2 * db - dAd) < 0
        alpha = torch.where(bad, db.clamp_min(0) / torch.where(bad, dAd, torch.ones_like(dAd)), torch.ones_like(db))   # bad => dAd > 0
        st = self.__dict__.setdefault('stats', dict(calls=0, samples=0, backstop=0, rejected=0, cols=0, rank_deficit=0))
        st['calls'] += 1; st['samples'] += B; st['backstop'] += int(bad.sum()); st['rejected'] += int((bad & (db <= 0)).sum())
        if getattr(self, 'collect_stats', False):
            with torch.no_grad():
                ev = torch.linalg.eigvalsh(Gs - torch.diag_embed((~live).double() + 1e-10))
                rk = (ev > 1e-8 * ev.amax(1, keepdim=True).clamp_min(1e-300)).sum(1)
                st['cols'] += int(live.sum()); st['rank_deficit'] += int((live.sum(1) - rk).clamp_min(0).sum())
        return alpha[:, None] * d

    def forward(self, r):
        # r may be float64 (drivers since 5 Oct: the exact residual); the cycle runs in the network's dtype, deflation and
        # the Galerkin output in float64 against the assembled operator; the output has r's dtype.
        pdt = next(self.parameters()).dtype
        rn = r.to(pdt)
        if not (self.project or self.deflate):
            return self._cycle(0, rn).to(r.dtype)
        r64 = r.double()
        Z = self._Z if (self.deflate and self._Z is not None) else None
        x0 = self._galerkin(Z, r64) if Z is not None else torch.zeros_like(r64)
        b64 = r64 - self._matvec64(x0); bn = b64.to(pdt)
        self.aux_out = None
        if not self.project or (self.project == 'eval' and self.training):
            return (x0 + self._cycle(0, bn).double()).to(r.dtype)   # project='eval': plain cycle in training
        parts = self._cycle(0, bn, parts=True)
        W = torch.stack(parts, -1)
        if Z is not None: W = torch.cat([W, Z.to(W.dtype)], -1)
        if self.project == 'both' and self.training:
            self.aux_out = (x0 + (parts[0] + parts[1] + parts[2]).double()).to(r.dtype)   # plain cycle: second loss term
        return (x0 + self._galerkin(W, b64)).to(r.dtype)


# ---------------------------------------------------------------------------------------------------------------
# the classical member of the class (numpy): one V(nu,nu)-cycle, Chebyshev smoothers, same transfers
# ---------------------------------------------------------------------------------------------------------------
def classical_vcycle(levs, A0, b, nu=4, q=10, fine_lmax=3.0):
    def lam_max(Ak, its=30):
        dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
        for _ in range(its):
            w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
        return 1.05 * lam

    def cheb(Ak, rhs, x, lmax):
        dinv = 1 / Ak.diagonal(); lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(nu - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        return x
    lm = [fine_lmax if (l == 0 and fine_lmax) else lam_max(Ak) for l, (Ak, _) in enumerate(levs)]   # fine_lmax None: power iteration

    def cyc(l, rhs):
        Ak, X = levs[l]; nc = X.shape[1]
        x = cheb(Ak, rhs, np.zeros_like(rhs), lm[l]); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else splu(A0.tocsc()).solve(rc)
        x = x + np.concatenate([ec, X @ ec])
        return cheb(Ak, rhs, x, lm[l])
    return cyc(0, b)


def classical_vcycle_proj(levs, A0, b, Z=None, nu=4, q=10, project=True, fine_lmax=3.0):
    """numpy member of oa-vcp's class: deflation by Z, one V(nu,nu) Chebyshev cycle on the deflated residual, Galerkin
    projection onto span{pre part, coarse part, post part, Z} at the fine level."""
    A = levs[0][0]
    def gal(W, rhs):
        if W.shape[1] == 0: return np.zeros_like(rhs)
        G = W.T @ (A @ W); dg = np.diag(G); live = dg > 1e-30 * dg.max()
        sc = np.where(live, 1 / np.sqrt(np.where(live, dg, 1)), 0)
        y = sc * np.linalg.solve(sc[:, None] * G * sc[None, :] + np.diag((~live) + 1e-10), sc * (W.T @ rhs))
        return W @ y
    x0 = gal(Z, b) if Z is not None and Z.shape[1] else np.zeros_like(b)
    bb = b - A @ x0
    def lam_max(Ak, its=30):
        dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
        for _ in range(its):
            w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
        return 1.05 * lam
    def cheb(Ak, rhs, x, lmax):
        dinv = 1 / Ak.diagonal(); lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(nu - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        return x
    lm = [fine_lmax if (l == 0 and fine_lmax) else lam_max(Ak) for l, (Ak, _) in enumerate(levs)]   # fine_lmax None: power iteration
    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]
        x = cheb(Ak, rhs, np.zeros_like(rhs), lm[l]); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else splu(A0.tocsc()).solve(rc)
        xc = np.concatenate([ec, X @ ec])
        xp = cheb(Ak, rhs - Ak @ (x + xc), np.zeros_like(rhs), lm[l])
        return (x, xc, xp) if parts else x + xc + xp
    if not project:
        return x0 + cyc(0, bb)
    W = np.column_stack(list(cyc(0, bb, parts=True)) + ([Z] if Z is not None and Z.shape[1] else []))
    return x0 + gal(W, bb)

