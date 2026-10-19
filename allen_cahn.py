"""Allen-Cahn on the nested (unstructured, perforated) hierarchy -- the third problem class.

    u_t = eps^2 Lap u - f(u),   f(u) = u^3 - u,   grad u . n = 0 on the whole boundary (outer edges and pores),
    u(0) = u0  (Gaussian random field, clipped to [-1, 1]);   the surrogate learns  u0 -> u(T).

Discretisation (P1, lumped mass M, stiffness K; both from the level-L mesh and restricted by Galerkin to
coarser levels, which is exact on the nested hierarchy):

    ((1 + S dt) M + dt eps^2 K) u^{n+1} = M ((1 + S dt) u^n - dt f~(u^n))           (stabilised semi-implicit)

f~ is f truncated outside [-1, 1] to be globally Lipschitz with |f~'| <= 2 (Shen & Yang 2010): for S >= 1 the
scheme is unconditionally energy stable for

    E(u) = eps^2 / 2 u^T K u + sum_i m_i F~(u_i),   F~' = f~,   F~(u) = (u^2 - 1)^2 / 4 on [-1, 1].

(The stiffness is NOT an M-matrix on this mesh -- some angles exceed 90 deg -- so no discrete maximum
principle is claimed; energy stability does not need one.)  The system matrix does not depend on the
sample, so it is factorised once per level and all samples are advanced together as columns.

Level-k solve (the coarse input for the networks): the same scheme with M_k = lumped(P^T M P) and
K_k = P^T K P on the level-k mesh, started from the nodal values of u0 at the level-k nodes (the P1
interpolant; nested nodes), and prolonged to the fine mesh by P.
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from scalar_problems import unit_matrices


def neumann_hierarchy(L, coarse_n=9):
    """Perforated square, homogeneous Neumann on every boundary (all nodes are unknowns)."""
    return NestedHierarchy(L, coarse=perforated_square(n=coarse_n), bc='mixed', neumann=lambda x, y: True)


def f_tilde(u):
    return np.where(u > 1, 2 * (u - 1), np.where(u < -1, 2 * (u + 1), u ** 3 - u))


def F_tilde(u):
    return np.where(u > 1, (u - 1) ** 2, np.where(u < -1, (u + 1) ** 2, 0.25 * (u ** 2 - 1) ** 2))


class ACOperators:
    """Lumped mass and stiffness on every level k (unknown ordering H.nodes[k])."""

    def __init__(self, H):
        self.H = H
        L = H.L
        Mu, area = unit_matrices(H.coords, H.elem)                  # (ne, 3, 3, 3) tensor pieces
        Ke = Mu[:, 0] + Mu[:, 2]                                    # isotropic Laplacian element matrices
        e = H.elem; n = len(H.coords)
        I = np.repeat(e, 3, axis=1).ravel(); J = np.tile(e, (1, 3)).ravel()
        K = sp.coo_matrix((Ke.ravel(), (I, J)), shape=(n, n)).tocsr()
        m = np.zeros(n)
        for j in range(3): np.add.at(m, e[:, j], area / 3)          # lumped P1 mass: area / 3 per vertex
        idx = H.nodes[L]
        self.K = {L: K[idx][:, idx].tocsr()}
        self.m = {L: m[idx]}
        self.Pc = {L: sp.identity(len(idx), format='csr')}          # level k -> level L prolongation
        for k in range(L, 0, -1):
            P = H.P[k]
            self.K[k - 1] = (P.T @ self.K[k] @ P).tocsr()
            self.m[k - 1] = P.T @ self.m[k]                         # row sums of the Galerkin mass = lumped mass
            self.Pc[k - 1] = (self.Pc[k] @ P).tocsr()
        self._fac = {}

    def mass_M(self, k=None):
        return sp.diags(self.m[self.H.L if k is None else k])

    def energy(self, u, eps, k=None):
        k = self.H.L if k is None else k
        u = np.atleast_2d(u)
        grad = np.einsum('bi,bi->b', u, (self.K[k] @ u.T).T)
        return 0.5 * eps ** 2 * grad + F_tilde(u) @ self.m[k]

    def solve(self, u0, k, eps, T, dt, S=2.0, record_energy=False):
        """Advance u0 (n_samples, n_k) or (n_k,) on level k to time T. Returns u(T) (and the energy history)."""
        nsteps = int(round(T / dt))
        if abs(nsteps * dt - T) > 1e-9 * T: raise ValueError('T must be a multiple of dt')
        key = (k, eps, dt, S)
        if key not in self._fac:
            A = sp.diags((1 + S * dt) * self.m[k]) + dt * eps ** 2 * self.K[k]
            self._fac[key] = splu(A.tocsc())
        lu = self._fac[key]
        U = np.array(np.atleast_2d(u0), float).T                   # (n_k, B)
        mk = self.m[k][:, None]
        hist = [self.energy(U.T, eps, k)] if record_energy else None
        for _ in range(nsteps):
            U = lu.solve(mk * ((1 + S * dt) * U - dt * f_tilde(U)))
            if record_energy: hist.append(self.energy(U.T, eps, k))
        U = U.T
        out = U[0] if np.ndim(u0) == 1 else U
        return (out, np.array(hist)) if record_energy else out

    def coarse_solve(self, u0_fine, k0, eps, T, dt, S=2.0):
        """AC solve on the level-k0 mesh from the fine initial data (nodal injection), prolonged to level L."""
        n0 = len(self.H.nodes[k0])
        u0c = np.atleast_2d(u0_fine)[:, :n0]                       # coarse nodes come first at every level
        uc = self.solve(u0c, k0, eps, T, dt, S)
        return (self.Pc[k0] @ np.atleast_2d(uc).T).T

    # norms used for the errors
    def h1_sq(self, v, k=None):
        k = self.H.L if k is None else k
        v = np.atleast_2d(v); return np.einsum('bi,bi->b', v, (self.K[k] @ v.T).T)

    def l2_sq(self, v, k=None):
        k = self.H.L if k is None else k
        v = np.atleast_2d(v); return (v ** 2) @ self.m[k]


# ------------------------------------------------------------------------------------ initial conditions
def grf_initial(H, n, rng, tau=15.0, alpha=2.0, std=0.6, ng=256):
    """u0 = clip(std * g, -1, 1), g a unit-variance Gaussian random field on the unit square with spectrum
    (4 pi^2 |k|^2 + tau^2)^(-alpha) (the FNO/WNO covariance family), sampled by FFT on an ng x ng periodic
    grid and interpolated bilinearly to the mesh nodes -- the same u0 whatever the mesh level."""
    kx = np.fft.fftfreq(ng, d=1.0 / ng)
    KX, KY = np.meshgrid(kx, kx, indexing='ij')
    sq = (4 * np.pi ** 2 * (KX ** 2 + KY ** 2) + tau ** 2) ** (-alpha / 2)
    sq[0, 0] = 0.0                                                   # zero-mean field
    xy = H.coords[H.nodes[H.L]]
    gx = xy[:, 0] * ng; gy = xy[:, 1] * ng
    i0 = np.floor(gx).astype(int); j0 = np.floor(gy).astype(int); fx = gx - i0; fy = gy - j0
    i0 %= ng; j0 %= ng; i1 = (i0 + 1) % ng; j1 = (j0 + 1) % ng
    out = np.empty((n, len(xy)))
    for s in range(n):
        z = rng.standard_normal((ng, ng)) + 1j * rng.standard_normal((ng, ng))
        g = np.real(np.fft.ifft2(z * sq)); g /= g.std()
        v = (g[i0, j0] * (1 - fx) * (1 - fy) + g[i1, j0] * fx * (1 - fy)
             + g[i0, j1] * (1 - fx) * fy + g[i1, j1] * fx * fy)
        out[s] = np.clip(std * v, -1.0, 1.0)
    return out
