"""oa_lifting.py -- the operator-adapted lifting transform in two-level form.

Companion to the project doc `oa-lifting-layer-formulation.md` (section numbers
below refer to it).  Built on ofwno_core.MeshHierarchy / PoissonAssembler and
grf_field.ElementwiseOperatorAssembly; nothing in the existing files is changed.

Notation (per level k = 1..L, interior nodes only -- Dirichlet nodes are
REMOVED at every level, not replaced by identity rows):
    A_k      Galerkin stiffness at level k   (A_{k-1} = P_k^T A_k P_k)
    P_k      prolongation level k-1 -> k     (nodal interpolation)
    E_k      injection of the new level-k nodes
    B_k      P_k^T A_k E_k
    C_k      update (lifting) matrix:  Psi_k = E_k + P_k C_k
    D_k      Psi_k^T A_k Psi_k            (the level-k wavelet block)

Modes for C_k:
    'exact'  C_k = -A_{k-1}^{-1} B_k          (global; block-diagonalises exactly)
    'local'  C_k from patch solves of radius rho * h_{k-1}  (Algorithm 0)
    'frozen' C_k computed from a reference coefficient (default a = 1)
    'hb'     C_k = 0                                    (plain hierarchical basis)

The two-level form orthogonalises each wavelet against V_{k-1} in ONE solve.
Your WoodburyPatchLocalSolver does Gram-Schmidt against each coarser level
separately with a per-level patch; for the exact mode the two are identical,
for the local mode they are different truncations.
"""
from __future__ import annotations
import time
import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree

from ofwno_core import MeshHierarchy, PoissonAssembler, build_channel_assignment
from grf_field import ElementwiseOperatorAssembly, sample_grf, grf_at_elements
from frozen_basis import sample_a_lognormal


# ---------------------------------------------------------------------------
# mesh hierarchy in nested-level form
# ---------------------------------------------------------------------------
class NestedHierarchy:
    """Per-level interior node sets, prolongations and injections."""

    def __init__(self, L, domain='square', bc='dirichlet', coarse=None, neumann=None):
        """domain: 'square' (unit square) or 'L' (unit square minus [1/2,1]^2, re-entrant corner at
        (1/2,1/2)).  bc: 'dirichlet' (u=0 on the whole boundary) or 'mixed' (homogeneous Neumann on
        the outer top edge y=1 and outer right edge x=1, Dirichlet elsewhere incl. the re-entrant
        edges).  The L-domain is resolved exactly from mesh level 1 on, so the hierarchy, the predict
        and the grid bijection are unchanged; removed elements simply carry no stiffness."""
        if coarse is not None: domain = 'mesh'            # arbitrary coarse triangulation
        self.L, self.domain, self.bc = L, domain, bc
        self.mesh = MeshHierarchy(*coarse) if coarse is not None else MeshHierarchy()
        self.mesh.refine_to_level(L)
        m = self.mesh
        self.coords = np.asarray(m.coords)
        self.elem = np.asarray([np.asarray(e, int) for e in m.elements])
        self.centroids = self.coords[self.elem].mean(1)
        self.n_elem = len(self.elem)
        cx, cy = self.centroids[:, 0], self.centroids[:, 1]
        self.keep = ~((cx > 0.5) & (cy > 0.5)) if domain == 'L' else np.ones(self.n_elem, bool)
        if domain not in ('square', 'L', 'mesh'): raise ValueError(domain)
        if bc not in ('dirichlet', 'mixed'): raise ValueError(bc)
        neumann_fn = neumann
        in_dom = np.zeros(len(self.coords), bool); in_dom[self.elem[self.keep].ravel()] = True
        x, y = self.coords[:, 0], self.coords[:, 1]
        close = np.isclose
        on_bnd = np.zeros(len(self.coords), bool)
        on_bnd[self.mesh.get_boundary_nodes()] = True        # topological: any domain, holes included
        if domain == 'L':
            on_bnd |= (close(x, 0.5) & (y >= 0.5 - 1e-12)) | (close(y, 0.5) & (x >= 0.5 - 1e-12))
        neumann = np.zeros_like(on_bnd)
        if domain == 'mesh':
            if neumann_fn is not None:
                neumann = on_bnd & np.array([bool(neumann_fn(px, py)) for px, py in self.coords])
            elif bc != 'dirichlet':
                raise ValueError("domain='mesh' with bc != 'dirichlet' needs a neumann=f(x, y) predicate")
        elif bc == 'mixed':
            xr = 0.5 if domain == 'L' else 1.0         # extent of the outer top / right edges
            neumann = (close(y, 1) & (x > 1e-12) & (x < xr - 1e-12)) | \
                      (close(x, 1) & (y > 1e-12) & (y < xr - 1e-12))
        dirichlet = (on_bnd & ~neumann) | ~in_dom
        self.neumann_nodes = np.nonzero(neumann & in_dom)[0]
        bnd = set(np.nonzero(dirichlet)[0].tolist())
        self.boundary = np.array(sorted(bnd), int)
        self.in_domain = in_dom
        # unknown nodes present at level k (global ids), coarse-first ordering
        self.nodes = {}
        acc = []
        for k in range(L + 1):
            acc = acc + [n for n in m.level_nodes[k] if n not in bnd]
            self.nodes[k] = np.array(acc, int)
        self.new = {k: np.array([n for n in m.level_nodes[k] if n not in bnd], int)
                    for k in range(1, L + 1)}
        self.P, self.E, self.h = {}, {}, {}
        if domain == 'mesh':                              # mean coarse edge length sets the patch scale
            c0 = np.asarray(MeshHierarchy(*coarse).coords)
            e0 = np.asarray(MeshHierarchy(*coarse).elements)
            h0 = float(np.mean([np.linalg.norm(c0[t[i]] - c0[t[(i + 1) % 3]]) for t in e0 for i in range(3)]))
        else:
            h0 = 1.0                                      # unchanged for the square / L set-ups
        for k in range(1, L + 1):
            fine = self.nodes[k]; coarse = self.nodes[k - 1]
            pos_f = {g: i for i, g in enumerate(fine)}
            pos_c = {g: i for i, g in enumerate(coarse)}
            I, J, V = [], [], []
            for g, ic in pos_c.items():
                I.append(pos_f[g]); J.append(ic); V.append(1.0)
            for g in self.new[k]:
                for p in m.node_parents[g]:
                    if p in pos_c:
                        I.append(pos_f[g]); J.append(pos_c[p]); V.append(0.5)
            self.P[k] = sp.csr_matrix((V, (I, J)), shape=(len(fine), len(coarse)))
            nn = np.array([pos_f[g] for g in self.new[k]], int)
            self.E[k] = sp.csr_matrix((np.ones(len(nn)), (nn, np.arange(len(nn)))),
                                      shape=(len(fine), len(nn)))
            self.h[k] = h0 / 2 ** k           # fine spacing of the level-k mesh
        self.h[0] = h0
        self.asm = ElementwiseOperatorAssembly(self.mesh, PoissonAssembler)
        # load f = 1 on the domain only: each P1 element gives area/3 to each vertex
        xy = self.coords[self.elem]
        area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                            - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
        F = np.zeros(len(self.coords))
        for j in range(3):
            np.add.at(F, self.elem[self.keep, j], area[self.keep] / 3.0)
        self.F = F[self.nodes[L]]

    def A_fine(self, a_elem):
        if self.domain == 'square':
            K = self.asm.K_raw(a_elem).tocsr()
        else:                                  # removed elements carry no stiffness (K_raw rejects a=0)
            s_ = self.asm; a = np.where(self.keep, np.asarray(a_elem).ravel(), 0.0)
            K = sp.coo_matrix((s_.V * a[s_.E], (s_.I, s_.J)), shape=(s_.n, s_.n)).tocsr()
        idx = self.nodes[self.L]
        return K[idx][:, idx].tocsc()

    def new_coords(self, k):   return self.coords[self.new[k]]
    def level_coords(self, k): return self.coords[self.nodes[k]]


# ---------------------------------------------------------------------------
# coefficient fields
# ---------------------------------------------------------------------------
def field_blocks(H, nc, contrast, rng):
    """Your existing 'channelized' benchmark: nc x nc blocks, log-uniform values."""
    ch = build_channel_assignment(H.mesh, nc, nc)
    mu = sample_a_lognormal(1, nc * nc, contrast, rng)[0]
    return mu[ch]


def field_grf(H, contrast, rng, corr_length=0.05, nu=0.5):
    n = 2 ** H.L + 1
    G = sample_grf(1, n, corr_length=corr_length, nu=nu, contrast=contrast, rng=rng)
    return grf_at_elements(G, H.mesh, n)[0]


def field_channels(H, contrast, rng, n_channels=4, width=1 / 16, amp=0.08, background=1.0):
    """CONNECTED high-permeability channels on a low background (SPE10-like).
    Each channel is a meandering band crossing the whole domain, horizontal or
    vertical, of the given width in domain units (1/16 = 4 cells at level 6).
    This is the 'routing' regime flagged in where-the-project-stands.md, and the
    one where the level blocks D_k become ill-conditioned (formulation section 7)."""
    xc, yc = H.centroids[:, 0], H.centroids[:, 1]
    a = np.full(H.n_elem, background)
    for c in range(n_channels):
        horiz = rng.random() < 0.5
        s, t = (xc, yc) if horiz else (yc, xc)
        t0 = rng.uniform(0.15, 0.85); f = rng.uniform(0.5, 2.0); ph = rng.uniform(0, 2 * np.pi)
        centre = t0 + amp * np.sin(2 * np.pi * f * s + ph)
        a[np.abs(t - centre) < width / 2] = background * contrast
    return a


def make_field(H, kind, contrast, rng, **kw):
    if kind == 'blocks8':   return field_blocks(H, 8, contrast, rng)
    if kind == 'blocks16':  return field_blocks(H, 16, contrast, rng)
    if kind == 'blocks4':   return field_blocks(H, 4, contrast, rng)
    if kind == 'grf':       return field_grf(H, contrast, rng, **kw)
    if kind == 'channels':  return field_channels(H, contrast, rng, **kw)
    if kind == 'const':     return np.ones(H.n_elem)
    raise ValueError(kind)


# ---------------------------------------------------------------------------
# the transform
# ---------------------------------------------------------------------------
def _local_C(Ac, Bk, coarse_xy, new_xy, radius):
    """Algorithm 0: for each new node t, solve Ac[S,S] c = -B[S,t] on the coarse nodes within
    `radius` of t.  Patches of equal size are solved as one batched dense solve, which is what
    makes the build cheap enough to sit inside a neural-operator layer."""
    tree = cKDTree(coarse_xy)
    patches = tree.query_ball_point(new_xy, r=radius + 1e-12)
    Ac = Ac.tocsr(); Bc = Bk.tocsc()
    indptr, indices, data = Bc.indptr, Bc.indices, Bc.data
    nc, mk = Ac.shape[0], Bk.shape[1]
    by_size = {}
    for t, S in enumerate(patches):
        if S: by_size.setdefault(len(S), []).append(t)
    rows, cols, vals = [], [], []
    Ad = None
    for size, ts in by_size.items():
        idx = np.array([sorted(patches[t]) for t in ts], dtype=np.int64)      # (nt, size)
        nt = len(ts)
        if Ad is None and nc <= 6000:        # 6000^2 float64 = 288 MB; above this, slice sparsely
            # one dense view of the coarse matrix, built by scattering its nonzeros into a
            # lazily-allocated zero array (cheaper than csr_todense, which writes every cell):
            # patches overlap heavily, so fancy-indexing this beats slicing the sparse matrix
            coo = Ac.tocoo()
            Ad = np.zeros((nc, nc), dtype=np.float64)
            Ad[coo.row, coo.col] = coo.data
        if Ad is not None:
            Apatch = Ad[idx[:, :, None], idx[:, None, :]]                     # (nt, size, size)
        else:
            Apatch = np.stack([Ac[i][:, i].toarray() for i in idx])   # fallback for very coarse levels
        # gather the right-hand sides straight from the CSC arrays (scipy's per-column
        # fancy indexing dominated the build before this)
        b = np.zeros((nt, size))
        for j, t in enumerate(ts):
            lo, hi = indptr[t], indptr[t + 1]
            if lo == hi:
                continue
            r = indices[lo:hi]
            pos = np.searchsorted(idx[j], r)
            pos[pos >= size] = 0
            m = idx[j][pos] == r
            b[j, pos[m]] = data[lo:hi][m]
        keep = np.abs(b).max(1) > 0
        if not keep.any():
            continue
        c = np.linalg.solve(Apatch[keep], -b[keep][:, :, None])[:, :, 0]      # batched
        ii = idx[keep]
        tt = np.asarray(ts)[keep]
        rows.append(ii.ravel())
        cols.append(np.repeat(tt, size))
        vals.append(c.ravel())
    if not rows:
        return sp.csr_matrix((nc, mk))
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(nc, mk))
    C.data[np.abs(C.data) < 1e-14] = 0.0
    C.eliminate_zeros()
    return C


class OATransform:
    """Multilevel transform T = [Phi_0 | Psi_1 | ... | Psi_L], applied matrix-free,
    together with its level blocks D_k = Psi_k^T A Psi_k.

    Two routes put the operator into different halves of the lifting step
    (foundations doc section 3):

      route 'W'  (operator in the UPDATE; Amaratunga-Sudarshan-D'Heedene, the original code)
          Phi_k = P_k                (geometric prolongation)
          Psi_k = E_k + P_k C_k,     C_k = -A_{k-1}^{-1} B_k,  B_k = P_k^T A_k E_k
          D_k   = Schur complement A_FF - B^T A_{k-1}^{-1} B;  coarse block = Galerkin A_{k0}
      route 'V'  (operator in the PREDICT; energy-minimising / ideal interpolation)
          Phi_k = [I; Phat_k],       Phat_k = -A_FF^{-1} A_FC   (a-harmonic interpolation)
          Psi_k = E_k                (wavelets are the plain nodal functions at the new nodes)
          D_k   = A_FF;              coarse block = Schur complement A_CC - A_CF A_FF^-1 A_FC

    Both block-diagonalise A exactly when built exactly.  Route V's detail blocks are the
    finest-level nodal blocks A_FF.  Their Jacobi-scaled condition number is NOT bounded
    independently of the coefficient in general (corrected 5 Oct, independent review G):
    mathcheck/check_stocktake.log measures 11.5 for the channel benchmark and 15.7 for
    log-uniform coefficients constant on coarse elements at L3, but 693 for an adversarial
    P1 coefficient (middle child 1e4, corner children 1) and 11194 for its elasticity analogue
    at L2.  An earlier bound "3+sqrt(3) for any elementwise-constant coefficient" was wrong
    (it holds for the Q1 case only); coarse-level blocks of the Galerkin hierarchy have their
    own spectra.  Route W's blocks are Schur complements, contrast-robust only when the
    coefficient is resolved by a coarse level.

    mode : 'exact' | 'local' (patch radius rho, in coarse mesh widths) | 'frozen' | 'hb'
           'hb' means no operator adaptation at all (C = 0 / Phat = geometric), identical
           for both routes.
    """

    def __init__(self, H, A, mode='exact', rho=2, A_ref=None, dense_D_max=6000,
                 frozen_local_rho=None, route='W', verbose=False, phat_fn=None):
        self.H, self.mode, self.rho, self.route = H, mode, rho, route
        t0 = time.time()
        L = H.L
        self.A = {L: A.tocsc()}
        Aref = {L: (A_ref if A_ref is not None else A).tocsc()} if mode == 'frozen' else None
        self.C, self.B, self.D, self.Phi, self.Psi = {}, {}, {}, {}, {}
        for k in range(L, 0, -1):
            P, E, Ak = H.P[k], H.E[k], self.A[k]
            nc, mk = P.shape[1], E.shape[1]
            if route == 'V':
                A_FF = (E.T @ Ak @ E).tocsc()
                A_FC = (E.T @ Ak @ P[:, :nc]).tocsc() if False else (E.T @ Ak)[:, :nc].tocsc()
                A_CC = Ak[:nc, :nc].tocsc()
                A_CF = A_FC.T.tocsc()
                if mode == 'hb':
                    Phat = None                                   # geometric predict
                elif mode == 'exact':
                    Phat = sp.csr_matrix(-splu(A_FF).solve(A_FC.toarray()))
                elif mode == 'local':
                    Phat = _local_C(A_FF, A_FC, H.new_coords(k), H.level_coords(k - 1),
                                    rho * H.h[k - 1])
                elif mode == 'given':                             # externally supplied predict
                    Phat = sp.csr_matrix(phat_fn(k, A_FF, A_FC, H))  # (learned / smoothed / ...)
                elif mode == 'frozen':
                    Rk = Aref[k]
                    R_FF = (E.T @ Rk @ E).tocsc(); R_FC = (E.T @ Rk)[:, :nc].tocsc()
                    Phat = (sp.csr_matrix(-splu(R_FF).solve(R_FC.toarray()))
                            if frozen_local_rho is None else
                            _local_C(R_FF, R_FC, H.new_coords(k), H.level_coords(k - 1),
                                     frozen_local_rho * H.h[k - 1]))
                    Rphi = sp.vstack([sp.identity(nc, format='csr'), Phat]).tocsc()
                    Aref[k - 1] = (Rphi.T @ Rk @ Rphi).tocsc()
                else:
                    raise ValueError(mode)
                if Phat is None:
                    Phi = P.tocsr()
                else:
                    Phi = sp.vstack([sp.identity(nc, format='csr'), Phat]).tocsr()
                Ac = (Phi.T @ Ak @ Phi).tocsc()   # = A_CC - A_CF A_FF^-1 A_FC for the exact predict
                self.Phi[k], self.Psi[k] = Phi, E.tocsr()
                self.C[k] = Phat if Phat is not None else sp.csr_matrix((mk, nc))
                Dk = A_FF
                if Dk.shape[0] <= dense_D_max: Dk = Dk.toarray(); Dk = 0.5 * (Dk + Dk.T)
                self.D[k], self.A[k - 1] = Dk, Ac
                continue

            # ---- route W ----
            Ac = (P.T @ Ak @ P).tocsc()
            Bk = (P.T @ Ak @ E).tocsc()
            self.A[k - 1], self.B[k] = Ac, Bk
            if mode == 'hb':
                Ck = sp.csr_matrix((Ac.shape[0], E.shape[1]))
            elif mode == 'exact':
                Ck = -splu(Ac).solve(Bk.toarray())           # dense n_{k-1} x m_k
            elif mode == 'local':
                Ck = _local_C(Ac, Bk, H.level_coords(k - 1), H.new_coords(k),
                              rho * H.h[k - 1])
            elif mode == 'frozen':
                Rk = Aref[k]
                Rc = (P.T @ Rk @ P).tocsc(); Rb = (P.T @ Rk @ E).tocsc()
                Aref[k - 1] = Rc
                Ck = (-splu(Rc).solve(Rb.toarray()) if frozen_local_rho is None else
                      _local_C(Rc, Rb, H.level_coords(k - 1), H.new_coords(k),
                               frozen_local_rho * H.h[k - 1]))
            else:
                raise ValueError(mode)
            self.C[k] = Ck
            self.Phi[k] = P.tocsr()
            self.Psi[k] = (E + P @ Ck) if sp.issparse(Ck) else None   # dense C: applied implicitly
            # D_k = Psi^T A Psi = A_ff + B^T C + C^T B + C^T A_c C  (general C)
            Aff = (E.T @ Ak @ E)
            BtC = Bk.T @ Ck
            if sp.issparse(Ck):
                Dk = (Aff + BtC + BtC.T + Ck.T @ (Ac @ Ck)).tocsc()
                if Dk.shape[0] <= dense_D_max: Dk = Dk.toarray()
            else:
                Aff = Aff.toarray()
                Dk = Aff + BtC + BtC.T + Ck.T @ (Ac @ Ck)
            if not sp.issparse(Dk): Dk = 0.5 * (Dk + Dk.T)
            self.D[k] = Dk
        self.D[0] = self.A[0].toarray()
        self.build_time = time.time() - t0
        self._fac = None
        if verbose:
            print(f"  [OATransform route {route} / {mode}] built in {self.build_time:.1f}s")

    # -- block sizes / levels
    @property
    def levels(self): return list(range(self.H.L + 1))
    def size(self, k): return self.D[k].shape[0]

    # -- synthesis T c   (coefficients by level -> fine nodal)
    def synth(self, coeffs):
        s = np.asarray(coeffs[0], float)
        for k in range(1, self.H.L + 1):
            w = np.asarray(coeffs[k], float)
            if self.route == 'V':
                s = self.Phi[k] @ s + self.H.E[k] @ w
            else:
                s = self.H.P[k] @ (s + self.C[k] @ w) + self.H.E[k] @ w
        return s

    # -- dual analysis T^T r   (fine functional -> dual coefficients by level)
    def dual(self, r):
        out = {}
        r = np.asarray(r, float)
        for k in range(self.H.L, 0, -1):
            if self.route == 'V':
                out[k] = self.H.E[k].T @ r
                r = self.Phi[k].T @ r
            else:
                Pr = self.H.P[k].T @ r
                out[k] = self.H.E[k].T @ r + self.C[k].T @ Pr
                r = Pr
        out[0] = r
        return out

    # -- primal analysis T^{-1} v  (fine nodal function -> coefficients), no solve
    def analysis(self, v):
        out = {}
        v = np.asarray(v, float)
        for k in range(self.H.L, 0, -1):
            nc = self.H.nodes[k - 1]
            s = v[:len(nc)]                               # coarse nodes come first
            if self.route == 'V':
                out[k] = self.H.E[k].T @ v - self.H.E[k].T @ (self.Phi[k] @ s)
                v = s
                continue
            w = self.H.E[k].T @ v - self.H.E[k].T @ (self.H.P[k] @ s)   # predict
            s = s - self.C[k] @ w                          # undo update
            out[k] = w; v = s
        out[0] = v
        return out

    # -- block-Jacobi preconditioner  M = T diag(D)^-1 T^T
    def _factor(self):
        if self._fac is None:
            self._fac = {}
            for k, Dk in self.D.items():
                if sp.issparse(Dk): self._fac[k] = ('lu', splu(Dk.tocsc()))
                else:               self._fac[k] = ('ch', sla.cho_factor(Dk))
        return self._fac

    def block_solve(self, k, g):
        kind, f = self._factor()[k]
        return f.solve(g) if kind == 'lu' else sla.cho_solve(f, g)

    def precond(self, r):
        g = self.dual(r)
        return self.synth({k: self.block_solve(k, g[k]) for k in g})

    def apply_R(self, r, R):
        """T R T^T r for a dict R[k] of (dense or sparse) matrices / callables."""
        g = self.dual(r)
        out = {}
        for k in g:
            Rk = R[k]
            out[k] = Rk(g[k]) if callable(Rk) else Rk @ g[k]
        return self.synth(out)


# ---------------------------------------------------------------------------
# PCG with a Lanczos estimate of kappa(M A)
# ---------------------------------------------------------------------------
def pcg_kappa(A, b, M, tol=1e-10, maxiter=5000):
    """Preconditioned CG; returns (x, iterations, kappa estimate of M A).
    kappa comes from the Lanczos tridiagonal built from the CG coefficients --
    the standard, essentially free estimate of the extreme Ritz values."""
    x = np.zeros_like(b); r = b.copy(); z = M(r); p = z.copy()
    rz = r @ z; nb = np.linalg.norm(b)
    alphas, betas = [], []
    it = 0
    while it < maxiter:
        Ap = A @ p
        alpha = rz / (p @ Ap)
        x += alpha * p; r -= alpha * Ap
        it += 1; alphas.append(alpha)
        if np.linalg.norm(r) <= tol * nb: break
        z = M(r); rz_new = r @ z
        beta = rz_new / rz; betas.append(beta)
        p = z + beta * p; rz = rz_new
    n = len(alphas)
    d = np.empty(n); e = np.empty(max(n - 1, 0))
    d[0] = 1 / alphas[0]
    for j in range(1, n):
        d[j] = 1 / alphas[j] + betas[j - 1] / alphas[j - 1]
        e[j - 1] = np.sqrt(betas[j - 1]) / alphas[j - 1]
    ev = sla.eigh_tridiagonal(d, e, eigvals_only=True) if n > 1 else d
    return x, it, float(ev[-1] / ev[0])


def ritz_extremes(A, M, n_iter=60, seed=0):
    """Extreme eigenvalues of M A (M = preconditioner, SPD) by Lanczos from a RANDOM start, with full
    reorthogonalisation in the M^{-1} inner product. Ritz values lie inside the spectrum, so
    lmin_ritz >= lmin and lmax_ritz <= lmax; with n_iter >= ~30 they are tight for these sizes.
    Used per sample for the certificate constants: sqrt(lmin) <= eta / ||e||_A <= sqrt(lmax)."""
    n = A.shape[0]; m = min(n_iter, n)
    r = np.random.default_rng(seed).standard_normal(n)
    # Lanczos on the A-symmetric operator M A, inner product <x, y>_A
    Q = np.zeros((n, m)); AQ = np.zeros((n, m)); al = np.zeros(m); be = np.zeros(m)
    q = r / np.sqrt(r @ (A @ r)); k = 0
    for k in range(m):
        Q[:, k] = q; Aq = A @ q; AQ[:, k] = Aq
        w = M(Aq)
        al[k] = w @ Aq
        w -= Q[:, :k + 1] @ (AQ[:, :k + 1].T @ w)            # full reorthogonalisation (A-inner product)
        w -= Q[:, :k + 1] @ (AQ[:, :k + 1].T @ w)
        nb = np.sqrt(max(w @ (A @ w), 0.0))
        if k == m - 1 or nb < 1e-12 * abs(al[k]): break
        be[k] = nb; q = w / nb
    kk = k + 1
    ev = sla.eigh_tridiagonal(al[:kk], be[:kk - 1], eigvals_only=True) if kk > 1 else al[:1]
    return float(ev[0]), float(ev[-1])


# ---------------------------------------------------------------------------
# spectral tools for a level block
# ---------------------------------------------------------------------------
def jacobi_scaled(Dk):
    Dk = Dk.toarray() if sp.issparse(Dk) else Dk
    d = 1.0 / np.sqrt(np.diag(Dk))
    return Dk * d[:, None] * d[None, :], d


def block_spectrum(Dk, gap=20.0):
    """kappa raw, kappa Jacobi-scaled, number of eigenvalues below lambda_max/gap,
    and kappa after deflating them. Dense; call only for blocks you can afford."""
    Dd = Dk.toarray() if sp.issparse(Dk) else Dk
    ev_raw = np.linalg.eigvalsh(Dd)
    Dj, _ = jacobi_scaled(Dd)
    ev = np.linalg.eigvalsh(Dj)
    r = int((ev < ev[-1] / gap).sum())
    kap_defl = ev[-1] / ev[r] if r < len(ev) else np.nan
    return dict(n=len(ev), kappa_raw=ev_raw[-1] / ev_raw[0], kappa_jac=ev[-1] / ev[0],
                n_small=r, kappa_deflated=kap_defl,
                n_small_50=int((ev < ev[-1] / 50).sum()), n_small_10=int((ev < ev[-1] / 10).sum()))


def energy_err(A, u, uh):
    e = u - uh
    return float(np.sqrt(e @ (A @ e) / (u @ (A @ u))))


def rel_l2(u, uh):
    return float(np.linalg.norm(u - uh) / np.linalg.norm(u))
