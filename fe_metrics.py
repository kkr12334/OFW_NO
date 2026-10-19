"""5 Oct (independent review): helpers shared by the drivers.
  fe_mass(H, L)       consistent P1 mass matrix of the level-L mesh, restricted to the unknowns H.nodes[L] (the Dirichlet
                      values are zero), so  e^T M e = ||e_h||_{L2}^2  for the FE function -- the FE L2 norm (finding H: the
                      tables' "rel-L2" is the nodal Euclidean norm, which over-weights dense node clusters).
  fe_mass_vector(M)   the same for 2-vector fields with interleaved (x, y) dofs per node: M (x) I_2.
  atomic_pickle(o, p) write to a temporary file and os.replace it (finding E): a reader never sees a partial file.
                      Concurrent jobs must still use separate output directories."""
import os, pickle, numpy as np, scipy.sparse as sp


def fe_mass(H, L=None):
    L = H.L if L is None else L
    el = H.elem[H.keep]; xy = H.coords[el]
    area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                        - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    Mloc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
    M = sp.csr_matrix((np.einsum('e,ij->eij', area, Mloc).ravel(),
                       (np.repeat(el, 3, axis=1).ravel(), np.tile(el, (1, 3)).ravel())),
                      shape=(len(H.coords), len(H.coords)))
    return M[H.nodes[L]][:, H.nodes[L]].tocsr()


def fe_mass_vector(M):
    return sp.kron(M, sp.identity(2), format='csr')


def rel_norm(M, e, u):
    """per-row relative norm sqrt(e^T M e / u^T M u) for stacked rows (n, N)."""
    e, u = np.atleast_2d(e), np.atleast_2d(u)
    return np.sqrt(np.einsum('bi,bi->b', e, (M @ e.T).T) / np.einsum('bi,bi->b', u, (M @ u.T).T))


def atomic_pickle(obj, path):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, 'wb') as fh:
        pickle.dump(obj, fh)
    os.replace(tmp, path)
