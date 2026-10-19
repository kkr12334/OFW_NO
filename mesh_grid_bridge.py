"""Mesh <-> uniform grid transfer for the unstructured-domain experiments.

FNO and WNO need a uniform grid.  On an unstructured mesh that costs an interpolation in each
direction -- which is exactly the practical penalty the paper is claiming, so it has to be done in the
standard, fair way (as in Geo-FNO): inputs are interpolated mesh -> grid, the model predicts on the
grid, and the prediction is interpolated grid -> mesh nodes, where the loss and all errors are
computed.  Both transfers are linear, sparse and precomputed once.
"""
import numpy as np, scipy.sparse as sp
from matplotlib.tri import Triangulation


class MeshGridBridge:
    def __init__(self, H, ng=None):
        self.H = H
        self.coords = np.asarray(H.coords)
        elems = H.elem[H.keep]
        lo = self.coords.min(0); hi = self.coords.max(0)
        n_int = len(H.nodes[H.L])
        if ng is None:                       # grid at least as fine as the mesh (>= as many points)
            ng = int(np.ceil(np.sqrt(len(self.coords))))
            ng = max(ng, 8)
        self.ng = ng
        self.gx = np.linspace(lo[0], hi[0], ng); self.gy = np.linspace(lo[1], hi[1], ng)
        self.gxy = np.stack(np.meshgrid(self.gx, self.gy, indexing='ij'), -1)
        self.hx = self.gx[1] - self.gx[0]; self.hy = self.gy[1] - self.gy[0]

        tri = Triangulation(self.coords[:, 0], self.coords[:, 1], elems)
        finder = tri.get_trifinder()
        G = self.gxy.reshape(-1, 2)
        t = finder(G[:, 0], G[:, 1])                       # -1 outside the domain
        self.mask = (t >= 0).reshape(ng, ng)
        # mesh nodes -> grid points: barycentric weights inside the found triangle
        I, J, V = [], [], []
        for g in np.nonzero(t >= 0)[0]:
            v = elems[t[g]]; p = self.coords[v]
            T = np.array([[p[0, 0] - p[2, 0], p[1, 0] - p[2, 0]],
                          [p[0, 1] - p[2, 1], p[1, 1] - p[2, 1]]])
            l = np.linalg.solve(T, G[g] - p[2]); w = np.array([l[0], l[1], 1 - l[0] - l[1]])
            I += [g] * 3; J += list(v); V += list(w)
        self.to_grid = sp.csr_matrix((V, (I, J)), shape=(ng * ng, len(self.coords)))

        # grid -> mesh nodes (interior unknowns only): bilinear
        nodes = H.nodes[H.L]; P = self.coords[nodes]
        ix = np.clip(((P[:, 0] - lo[0]) / self.hx).astype(int), 0, ng - 2)
        iy = np.clip(((P[:, 1] - lo[1]) / self.hy).astype(int), 0, ng - 2)
        fx = (P[:, 0] - self.gx[ix]) / self.hx; fy = (P[:, 1] - self.gy[iy]) / self.hy
        I, J, V = [], [], []
        for k, (i, j, a, b) in enumerate(zip(ix, iy, fx, fy)):
            for di, dj, w in ((0, 0, (1 - a) * (1 - b)), (1, 0, a * (1 - b)),
                              (0, 1, (1 - a) * b), (1, 1, a * b)):
                I.append(k); J.append((i + di) * ng + (j + dj)); V.append(w)
        self.to_nodes = sp.csr_matrix((V, (I, J)), shape=(len(nodes), ng * ng))
        # nearest domain node of every grid point outside the domain (for extend='nearest')
        from scipy.spatial import cKDTree
        dom_nodes = np.nonzero(H.in_domain)[0]
        out_pts = G[~self.mask.ravel()]
        self.ext_nodes = dom_nodes[cKDTree(self.coords[dom_nodes]).query(out_pts)[1]] if len(out_pts) else np.zeros(0, int)

    def node_field_to_grid(self, f_nodes_all, extend='zero'):
        """(..., n_all_nodes) -> (..., ng, ng).  Outside the domain: extend='zero' (the original behaviour;
        exact where the field vanishes on the boundary, e.g. Dirichlet pores) or 'nearest' (value of the
        nearest domain node).  'nearest' is the fair choice when the field is NOT zero on the boundary
        (insulated / traction-free / Neumann pores): zero padding then puts a jump at every pore edge, and the
        mesh->grid->mesh round trip of the exact solution alone loses ~15% rel-L2 (anisotropic, insulated, L3),
        against 0.75% with the nearest extension."""
        f = np.asarray(f_nodes_all)
        out = (self.to_grid @ f.reshape(-1, f.shape[-1]).T).T.reshape(*f.shape[:-1], self.ng * self.ng)
        if extend == 'zero':
            out[..., ~self.mask.ravel()] = 0.0
        elif extend == 'nearest':
            out[..., ~self.mask.ravel()] = f[..., self.ext_nodes]
        else:
            raise ValueError(extend)
        return out.reshape(*f.shape[:-1], self.ng, self.ng)

    def interior_to_grid(self, f_int, extend='zero'):
        """(..., n_interior) -> (..., ng, ng): unknowns scattered to all nodes (0 on Dirichlet)."""
        f = np.asarray(f_int)
        full = np.zeros(f.shape[:-1] + (len(self.coords),))
        full[..., self.H.nodes[self.H.L]] = f
        return self.node_field_to_grid(full, extend)

    def round_trip_floor(self, U_int, extend='zero'):
        """mesh -> grid -> mesh of exact fields (n, n_interior): per-sample relative l2 of what is lost."""
        rt = (self.to_nodes @ self.interior_to_grid(U_int, extend).reshape(len(U_int), -1).T).T
        return np.linalg.norm(rt - U_int, axis=1) / np.linalg.norm(U_int, axis=1), rt

    def report(self):
        return (f"grid {self.ng}x{self.ng} ({int(self.mask.sum())} points inside the domain) for "
                f"{len(self.H.nodes[self.H.L])} mesh unknowns")
