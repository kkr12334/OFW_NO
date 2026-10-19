"""Plane-strain linear elasticity on the nested (unstructured) hierarchy -- the second problem class.

    -div sigma(u) = f,   sigma = C(E(x), nu) : eps(u),   u = 0 on the outer boundary,
    sigma n = 0 on the hole boundaries (traction-free pores).

Material: fibre composite -- random circular stiff fibres (E_f = contrast) in a soft matrix (E_m = 1),
optionally on the perforated (porous) domain.  P1 elements, two dofs per node.

VectorHierarchy wraps NestedHierarchy so that everything algebraic (OATransform, certificate, Galerkin
coarse solves) runs unchanged: dofs are interleaved (2g, 2g+1) in the scalar node order, so
P_k -> kron(P_k, I_2), E_k -> kron(E_k, I_2), coordinates are duplicated per dof."""
import numpy as np, scipy.sparse as sp


def unit_element_matrices(coords, elem, nu=0.3):
    """K1[e] (6x6) for E = 1, plane strain; dof order (u0x,u0y,u1x,u1y,u2x,u2y)."""
    p = coords[elem]                                            # (ne,3,2)
    x, y = p[..., 0], p[..., 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    area = 0.5 * np.abs(det)
    b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / det[:, None]   # dN/dx
    c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / det[:, None]   # dN/dy
    B = np.zeros((len(elem), 3, 6))
    B[:, 0, 0::2] = b; B[:, 1, 1::2] = c; B[:, 2, 0::2] = c; B[:, 2, 1::2] = b
    f = 1.0 / ((1 + nu) * (1 - 2 * nu))
    D = f * np.array([[1 - nu, nu, 0], [nu, 1 - nu, 0], [0, 0, (1 - 2 * nu) / 2]])
    return area[:, None, None] * np.einsum('eki,kl,elj->eij', B, D, B), area


class VectorHierarchy:
    def __init__(self, H, body_force=(0.0, -1.0), nu=0.3):
        self.H, self.L, self.h = H, H.L, H.h
        il = lambda g: np.stack([2 * np.asarray(g), 2 * np.asarray(g) + 1], 1).ravel()
        self.nodes = {k: il(v) for k, v in H.nodes.items()}
        self.new = {k: il(v) for k, v in H.new.items()}
        I2 = sp.identity(2, format='csr')
        self.P = {k: sp.kron(H.P[k], I2, format='csr') for k in H.P}
        self.E = {k: sp.kron(H.E[k], I2, format='csr') for k in H.E}
        self.elem = H.elem; self.keep = H.keep; self.n_elem = H.n_elem
        self.coords = np.repeat(H.coords, 2, axis=0)
        self.K1, area = unit_element_matrices(H.coords, H.elem, nu)
        self.edofs = np.stack([2 * H.elem[:, j // 2] + (j % 2) for j in range(6)], 1)     # (ne,6)
        nd = 2 * len(H.coords)
        I = np.repeat(self.edofs, 6, axis=1).ravel(); J = np.tile(self.edofs, (1, 6)).ravel()
        self._IJ = (I, J, nd)
        F = np.zeros(nd)
        for j in range(3):
            np.add.at(F, 2 * H.elem[self.keep, j], body_force[0] * area[self.keep] / 3)
            np.add.at(F, 2 * H.elem[self.keep, j] + 1, body_force[1] * area[self.keep] / 3)
        self.F = F[self.nodes[self.L]]

    def new_coords(self, k):   return np.repeat(self.H.new_coords(k), 2, axis=0)
    def level_coords(self, k): return np.repeat(self.H.level_coords(k), 2, axis=0)

    def A_fine(self, E_elem):
        I, J, nd = self._IJ
        w = np.where(self.keep, np.asarray(E_elem, float), 0.0)
        V = (self.K1 * w[:, None, None]).ravel()
        K = sp.coo_matrix((V, (I, J)), shape=(nd, nd)).tocsr()
        idx = self.nodes[self.L]
        return K[idx][:, idx].tocsc()


def fibre_composite(H, contrast, rng, vf=0.3, r=(0.025, 0.05)):
    """Random circular stiff fibres (overlaps allowed) to volume fraction ~vf; E_m = 1, E_f = contrast."""
    xc = H.centroids
    E = np.ones(H.n_elem)
    lo, hi = xc.min(0), xc.max(0)
    area_tot = np.prod(hi - lo); covered = 0.0
    while covered < vf * area_tot:
        c = rng.uniform(lo, hi); rad = rng.uniform(*r)
        E[np.hypot(xc[:, 0] - c[0], xc[:, 1] - c[1]) < rad] = contrast
        covered += np.pi * rad ** 2
    return E
