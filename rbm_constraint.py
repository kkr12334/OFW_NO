"""Rigid-body-mode constraint for truncated (local) operator-adapted predicts in 2D elasticity.

A patch-truncated energy-minimising predict Phat (fine dofs x coarse dofs) loses the rigid-body modes,
and then the adapted coarse space is WORSE than the plain Galerkin one.  The standard remedy from
energy-minimising AMG (Mandel-Brezina-Vanek 1999; Xu-Zikatanov 2004; Olson-Schroder-Tuminaro 2011):
keep each row's sparsity and project it, with the smallest Euclidean change, onto the affine set
    Phat[i, S_i] B_C[S_i, :] = B_F[i, :]
so the coarse space reproduces translations and the infinitesimal rotation exactly.
Dofs are interleaved (2g, 2g+1)."""
import numpy as np, scipy.sparse as sp


def rbm(xy_nodes):
    """(2n, 3) rigid-body modes at interleaved dofs: t_x, t_y, rotation about the centroid."""
    c = xy_nodes - xy_nodes.mean(0) if len(xy_nodes) else xy_nodes
    n = len(xy_nodes); B = np.zeros((2 * n, 3))
    B[0::2, 0] = 1.0; B[1::2, 1] = 1.0
    B[0::2, 2] = -c[:, 1]; B[1::2, 2] = c[:, 0]
    return B


def constrain_rows(Phat, B_F, B_C, rows=None, min_support=3):
    """Project the rows of Phat (sparse, nF x nC) onto Phat B_C = B_F, keeping the sparsity."""
    P = sp.csr_matrix(Phat, copy=True); P.sort_indices()
    rows = range(P.shape[0]) if rows is None else rows
    for i in rows:
        lo, hi = P.indptr[i], P.indptr[i + 1]
        if hi - lo < min_support: continue
        S = P.indices[lo:hi]; p = P.data[lo:hi]; BS = B_C[S]            # (s, 3)
        r = B_F[i] - p @ BS                                              # constraint violation (3,)
        G = BS.T @ BS
        try:
            lam = np.linalg.solve(G + 1e-12 * np.trace(G) * np.eye(3), r)
        except np.linalg.LinAlgError:
            continue
        P.data[lo:hi] = p + BS @ lam
    return P
