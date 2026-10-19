"""Scalar elliptic problems with a TENSOR coefficient on the nested unstructured hierarchy.

    -div( K(x) grad u ) = f  in the perforated domain,   u = 0 on the outer boundary,
    either u = 0 on the pores ('dirichlet') or K grad u . n = 0 on the pores ('insulated').

K is constant per element, K = [[k11, k12], [k12, k22]], P1 elements.  The element stiffness is linear in
(k11, k12, k22):  K_e = k11 M1 + k12 M2 + k22 M3  with M1 = a gx gx^T, M2 = a (gx gy^T + gy gx^T),
M3 = a gy gy^T (a = area, gx/gy = basis-function gradients), so assembly and the element-wise energy
e^T K_e e used by the loss are cheap.

Coefficient: 'tows' -- a woven / tow-reinforced composite.  Meandering fibre tows (the connected channels of
the Darcy benchmark) are strongly anisotropic, conducting along the local tow direction
(k_L = contrast) and moderately across it (k_T); the matrix is isotropic (k = 1).
Isotropic scalar fields (Darcy) are the special case k12 = 0, k11 = k22 = a."""
import numpy as np, scipy.sparse as sp


def unit_matrices(coords, elem):
    p = coords[elem]; x, y = p[..., 0], p[..., 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    area = 0.5 * np.abs(det)
    gx = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1) / det[:, None]
    gy = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1) / det[:, None]
    o = lambda u, v: np.einsum('ei,ej->eij', u, v)
    M = np.stack([o(gx, gx), o(gx, gy) + o(gy, gx), o(gy, gy)], 1) * area[:, None, None, None]   # (ne,3,3,3)
    return M, area


class TensorProblem:
    def __init__(self, H):
        self.H = H
        self.M, self.area = unit_matrices(H.coords, H.elem)
        e = H.elem
        self._I = np.repeat(e, 3, axis=1).ravel(); self._J = np.tile(e, (1, 3)).ravel()
        n = len(H.coords)
        F = np.zeros(n)
        for j in range(3): np.add.at(F, e[H.keep, j], H.keep[H.keep] * self.area[H.keep] / 3)
        self.F = F[H.nodes[H.L]]

    def elem_matrices(self, K):                         # K: (ne, 3) = (k11, k12, k22)
        K = np.where(self.H.keep[:, None], np.asarray(K, float), 0.0)
        return np.einsum('ec,ecij->eij', K, self.M)

    def A_fine(self, K):
        n = len(self.H.coords)
        A = sp.coo_matrix((self.elem_matrices(K).ravel(), (self._I, self._J)), shape=(n, n)).tocsr()
        idx = self.H.nodes[self.H.L]
        return A[idx][:, idx].tocsc()


def iso(a):
    a = np.asarray(a, float).ravel(); return np.stack([a, np.zeros_like(a), a], 1)


def field_tows(H, contrast, rng, k_T=5.0, n_channels=4, width=1 / 16, amp=0.08):
    """Anisotropic tows along the meandering channel curves: K = k_T I + (k_L - k_T) t t^T, k_L = contrast."""
    xc, yc = H.centroids[:, 0], H.centroids[:, 1]
    K = np.tile([1.0, 0.0, 1.0], (H.n_elem, 1))
    for c in range(n_channels):
        horiz = rng.random() < 0.5
        s, t = (xc, yc) if horiz else (yc, xc)
        t0 = rng.uniform(0.15, 0.85); f = rng.uniform(0.5, 2.0); ph = rng.uniform(0, 2 * np.pi)
        centre = t0 + amp * np.sin(2 * np.pi * f * s + ph)
        slope = amp * 2 * np.pi * f * np.cos(2 * np.pi * f * s + ph)          # d centre / d s
        tx, ty = (np.ones_like(s), slope) if horiz else (slope, np.ones_like(s))
        nrm = np.hypot(tx, ty); tx, ty = tx / nrm, ty / nrm
        m = np.abs(t - centre) < width / 2
        kL = contrast
        K[m, 0] = k_T + (kL - k_T) * tx[m] ** 2
        K[m, 1] = (kL - k_T) * tx[m] * ty[m]
        K[m, 2] = k_T + (kL - k_T) * ty[m] ** 2
    return K


def tensor_features(K):
    """Per element: mean log-eigenvalue, and log-anisotropy x (cos 2theta, sin 2theta) of the major axis."""
    k11, k12, k22 = K[:, 0], K[:, 1], K[:, 2]
    tr_, det = k11 + k22, k11 * k22 - k12 ** 2
    disc = np.sqrt(np.maximum((k11 - k22) ** 2 / 4 + k12 ** 2, 0))
    l1, l2 = tr_ / 2 + disc, np.maximum(tr_ / 2 - disc, 1e-30)
    th2 = np.arctan2(2 * k12, k11 - k22)                 # 2 theta of the major axis
    an = np.log(l1 / l2)
    return np.stack([0.5 * np.log(np.maximum(det, 1e-30)), an * np.cos(th2), an * np.sin(th2)], 1)
