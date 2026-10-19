"""Per-sample operator data computed from the stiffness matrix A alone (no learned parameters; Algorithm 1 of the paper).

* operator-adapted transfers  Phi_k = [I; Phat_k],  Phat_k ~ -A_FF^{-1} A_FC  by m weighted-Jacobi steps from the geometric
  predict (eq. 4); Galerkin level operators A_{k-1} = Phi_k^T A_k Phi_k,
* fixed sparsity patterns so that every sample's operator data are values on one pattern (batched tensors),
* near-kernel (floating-island) vectors Z detected from A alone.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components


def choose_levels(H, max_levels=4, min_coarsest=16):
    """Number of V-cycle levels: add a level while the next coarser mesh has at least `min_coarsest` unknowns."""
    n, k = 0, H.L
    while k >= 1 and n < max_levels and len(H.nodes[k - 1]) >= min_coarsest:
        n += 1; k -= 1
    return n


def oa_hierarchy(H, A, n_levels, m=4, omega=2 / 3):
    """Levels l = 0 .. n_levels-1 (l = 0 finest, level k = L - l).  Returns ([(A_k csr, Phat_k csr (F x C))], A_0)
    with A_0 the exact-solve operator below the last level.  m = 0 gives the geometric transfers."""
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


def structural_patterns(H, n_levels, m=4, A_struct=None):
    """Fixed per-level sparsity patterns (boolean products, so no sample has an entry outside them).
    A_struct: a fine operator with the full structural pattern of the problem class; default a random isotropic a."""
    one = lambda M: (abs(M) > 0).astype(np.float64).tocsr()
    As = one(A_struct if A_struct is not None else H.A_fine(np.random.default_rng(12345).uniform(1, 2, H.n_elem)))
    pats = []
    for l in range(n_levels):
        k = H.L - l; nc = H.P[k].shape[1]
        Xs = one(H.P[k][nc:])
        for _ in range(m):
            Xs = one(Xs + As[nc:, :nc] + As[nc:, nc:] @ Xs)
        pats.append((As.tocoo(), Xs.tocoo()))
        Phi = sp.vstack([sp.identity(nc, format='csr'), Xs]).tocsr(); As = one(Phi.T @ As @ Phi)
    return pats, As.shape[0]


def on_pattern(M, pat):
    """Values of sparse M on the (sorted) pattern `pat` (coo); every nonzero of M must lie in the pattern."""
    keys = pat.row.astype(np.int64) * pat.shape[1] + pat.col
    o = np.argsort(keys); ks = keys[o]
    C = M.tocoo(); nz = C.data != 0; C = sp.coo_matrix((C.data[nz], (C.row[nz], C.col[nz])), C.shape)
    kk = C.row.astype(np.int64) * pat.shape[1] + C.col
    pos = np.searchsorted(ks, kk)
    assert np.all(ks[np.minimum(pos, len(ks) - 1)] == kk), 'entry outside the structural pattern'
    v = np.zeros(len(ks)); v[pos] = C.data
    return v, o


def near_kernel_basis(A, theta=0.1, tau=0.05):
    """Near-kernel vectors of A from A alone (Section 3.4).  Connected components of the strong-connection graph
    (-A_ij >= theta sqrt(A_ii A_jj)) that contain no row strongly coupled to the Dirichlet boundary (row sum >= theta A_ii;
    interior rows of a diffusion matrix sum to zero) are floating regions; the indicator z of each is kept if its Rayleigh
    quotient z^T A z / z^T D z is below tau.  Returns Z (N x k), k >= 0.  A heuristic for scalar diffusion matrices."""
    A = A.tocsr(); d = A.diagonal(); C = A.tocoo()
    off = C.row != C.col
    strong = off & (-C.data >= theta * np.sqrt(d[C.row] * d[C.col]))
    S = sp.csr_matrix((np.ones(strong.sum()), (C.row[strong], C.col[strong])), shape=A.shape)
    n_, lab = connected_components(S, directed=False)
    rs = np.asarray(A.sum(1)).ravel()
    anchored = np.zeros(n_, bool); anchored[lab[rs >= theta * d]] = True
    cols = []
    for c in np.nonzero(~anchored)[0]:
        z = (lab == c).astype(float)
        if z @ (A @ z) < tau * (z @ (d * z)):
            cols.append(z)
    return np.array(cols).T if cols else np.zeros((A.shape[0], 0))
