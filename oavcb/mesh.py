"""Meshes and P1 finite elements.

* coarse triangulations (jittered-grid Delaunay: perforated square, polygon, L-shape; or the two-triangle square),
* uniform red refinement with nested node sets, coarse nodes ordered first on every level,
* P1 stiffness assembly for an element-wise coefficient, load vector and mass matrix.

Dirichlet conditions (u = 0) on the whole boundary, holes included; Dirichlet nodes are removed at every level.
"""
import numpy as np
import scipy.sparse as sp
from scipy.spatial import Delaunay


# ---------------------------------------------------------------------------------------------------------------
# coarse triangulations
# ---------------------------------------------------------------------------------------------------------------
def _triangulate(points, inside, min_angle_deg=15.0):
    """Delaunay of `points`, keeping triangles whose centroid satisfies `inside` and whose minimum angle is
    >= min_angle_deg."""
    pts = np.asarray(points, float)
    tri = Delaunay(pts)
    keep = []
    for t in tri.simplices:
        p = pts[t]
        if not inside(p.mean(0)):
            continue
        ang = []
        for i in range(3):
            u = p[(i + 1) % 3] - p[i]; v = p[(i + 2) % 3] - p[i]
            c = (u @ v) / (np.linalg.norm(u) * np.linalg.norm(v) + 1e-300)
            ang.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
        if min(ang) >= min_angle_deg:
            keep.append(t)
    elems = np.asarray(keep, int)
    used = np.unique(elems)
    remap = -np.ones(len(pts), int); remap[used] = np.arange(len(used))
    return pts[used], remap[elems]


def unit_square():
    return np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]), np.array([[0, 1, 2], [0, 2, 3]])


def perforated_square(n=9, holes=((0.30, 0.32, 0.12), (0.68, 0.62, 0.15), (0.30, 0.72, 0.10)),
                      jitter=0.25, seed=0, min_angle_deg=15.0):
    """Unit square with circular holes (jittered grid + Delaunay); each hole boundary is a ring of mesh edges."""
    rng = np.random.default_rng(seed)
    h = 1.0 / (n - 1)
    g = np.stack(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij'), -1).reshape(-1, 2)
    edge = (np.isclose(g[:, 0], 0) | np.isclose(g[:, 0], 1) | np.isclose(g[:, 1], 0) | np.isclose(g[:, 1], 1))
    g[~edge] += jitter * h * rng.standard_normal((np.sum(~edge), 2))

    def d_hole(p):                     # signed distance to the nearest hole (> 0 outside every hole)
        return min((np.hypot(p[0] - cx, p[1] - cy) - r) for cx, cy, r in holes)

    pts = [p for p in g if d_hole(p) > 0.45 * h]
    for cx, cy, r in holes:
        m = max(8, int(round(2 * np.pi * r / h)))
        th = np.arange(m) * 2 * np.pi / m
        pts += [np.array([cx + r * np.cos(t), cy + r * np.sin(t)]) for t in th]
    return _triangulate(np.array(pts), lambda c: d_hole(c) > 0, min_angle_deg)


def polygon_domain(vertices, n=10, jitter=0.25, seed=0, min_angle_deg=15.0):
    """Unstructured mesh of a simple polygon (list of (x, y), counter-clockwise)."""
    V = np.asarray(vertices, float)
    lo, hi = V.min(0), V.max(0)
    h = max(hi - lo) / (n - 1)
    rng = np.random.default_rng(seed)

    def inside(p):                                   # ray casting
        x, y = p; c = False; m = len(V)
        for i in range(m):
            x1, y1 = V[i]; x2, y2 = V[(i + 1) % m]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                c = not c
        return c

    pts = []
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        m = max(1, int(round(np.linalg.norm(b - a) / h)))
        pts += [a + (b - a) * j / m for j in range(m)]
    for x in np.arange(lo[0] + h / 2, hi[0], h):
        for y in np.arange(lo[1] + h / 2, hi[1], h):
            p = np.array([x, y]) + jitter * h * rng.standard_normal(2)
            if inside(p) and all(np.linalg.norm(p - q) > 0.5 * h for q in pts):
                pts.append(p)
    return _triangulate(np.array(pts), inside, min_angle_deg)


POLYGON = [(0, 0), (1, 0), (1, .4), (.6, .6), (1, .8), (1, 1), (0, 1)]


def coarse_mesh(name, n=9):
    """(coords, elements) of the named coarse triangulation."""
    if name == 'square':
        return unit_square()
    if name == 'perforated':
        return perforated_square(n=n)
    if name == 'polygon':
        return polygon_domain(POLYGON, n=n)
    if name == 'lshape':
        return polygon_domain([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)], n=max(5, n // 2))
    raise ValueError(f"unknown coarse mesh '{name}'")


# ---------------------------------------------------------------------------------------------------------------
# uniform refinement
# ---------------------------------------------------------------------------------------------------------------
class RefinedMesh:
    """Red refinement (each triangle -> 4) with bookkeeping of the nodes created on each level and their parents."""

    def __init__(self, coords, elements):
        self.coords = [list(map(float, c)) for c in np.asarray(coords).tolist()]
        self.elements = [list(map(int, e)) for e in np.asarray(elements).tolist()]
        self.current_level = 0
        self.level_nodes = {0: list(range(len(self.coords)))}
        self.node_parents = {}
        self.edge_map = {}

    def _midpoint(self, n1, n2, new_level):
        edge = tuple(sorted((n1, n2)))
        if edge in self.edge_map:
            return self.edge_map[edge]
        c1, c2 = np.array(self.coords[n1]), np.array(self.coords[n2])
        idx = len(self.coords)
        self.coords.append(((c1 + c2) / 2.0).tolist())
        self.edge_map[edge] = idx
        self.node_parents[idx] = (n1, n2)
        self.level_nodes.setdefault(new_level, []).append(idx)
        return idx

    def refine_one_level(self):
        new_level = self.current_level + 1
        self.edge_map = {}
        new = []
        for n0, n1, n2 in self.elements:
            m01 = self._midpoint(n0, n1, new_level)
            m12 = self._midpoint(n1, n2, new_level)
            m20 = self._midpoint(n2, n0, new_level)
            new.extend([[n0, m01, m20], [n1, m12, m01], [n2, m20, m12], [m01, m12, m20]])
        self.elements = new
        self.current_level = new_level

    def refine_to_level(self, target):
        while self.current_level < target:
            self.refine_one_level()

    def boundary_nodes(self):
        """Nodes on edges that belong to exactly one element (topological: holes and re-entrant corners included)."""
        cnt = {}
        for e in self.elements:
            for u, v in ((e[0], e[1]), (e[1], e[2]), (e[2], e[0])):
                k = (u, v) if u < v else (v, u)
                cnt[k] = cnt.get(k, 0) + 1
        bnd = set()
        for (u, v), c in cnt.items():
            if c == 1:
                bnd.add(u); bnd.add(v)
        return np.array(sorted(bnd), dtype=int)


# ---------------------------------------------------------------------------------------------------------------
# nested finite-element hierarchy
# ---------------------------------------------------------------------------------------------------------------
class Hierarchy:
    """Nested P1 spaces V_0 c ... c V_L on a refined coarse triangulation (interior nodes only).

    nodes[k]  global ids of the unknowns of level k, coarse-first (nodes[k-1] is a prefix of nodes[k])
    P[k]      geometric prolongation level k-1 -> k, P[k] = [I; P_F]  (sparse, len(nodes[k]) x len(nodes[k-1]))
    h[k]      mesh-width scale of level k (mean coarse edge length / 2^k; h0 = 1 for the plain unit square)
    elem, coords, centroids, area   fine mesh;   F  load f = 1 on the unknowns of level L
    """

    def __init__(self, coarse, levels, h0=None):
        coords0, elems0 = coarse
        self.L = L = levels
        mesh = RefinedMesh(coords0, elems0)
        mesh.refine_to_level(L)
        self.coords = np.asarray(mesh.coords)
        self.elem = np.asarray(mesh.elements, int)
        self.centroids = self.coords[self.elem].mean(1)
        self.n_elem = len(self.elem)
        xy = self.coords[self.elem]
        self.area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1])
                                 - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
        self.boundary = mesh.boundary_nodes()
        bnd = set(self.boundary.tolist())
        self.nodes, acc = {}, []
        for k in range(L + 1):
            acc = acc + [n for n in mesh.level_nodes[k] if n not in bnd]
            self.nodes[k] = np.array(acc, int)
        self.new = {k: np.array([n for n in mesh.level_nodes[k] if n not in bnd], int) for k in range(1, L + 1)}
        if h0 is None:
            c0, e0 = np.asarray(coords0, float), np.asarray(elems0, int)
            h0 = float(np.mean([np.linalg.norm(c0[t[i]] - c0[t[(i + 1) % 3]]) for t in e0 for i in range(3)]))
        self.h = {k: h0 / 2 ** k for k in range(L + 1)}
        self.P = {}
        for k in range(1, L + 1):
            fine, coarse = self.nodes[k], self.nodes[k - 1]
            pos_f = {g: i for i, g in enumerate(fine)}
            pos_c = {g: i for i, g in enumerate(coarse)}
            I, J, V = [], [], []
            for g, ic in pos_c.items():
                I.append(pos_f[g]); J.append(ic); V.append(1.0)
            for g in self.new[k]:
                for p in mesh.node_parents[g]:
                    if p in pos_c:
                        I.append(pos_f[g]); J.append(pos_c[p]); V.append(0.5)
            self.P[k] = sp.csr_matrix((V, (I, J)), shape=(len(fine), len(coarse)))
            nc = len(coarse)
            assert (self.P[k][:nc] - sp.identity(nc)).nnz == 0, 'level ordering must list the coarse nodes first'
        # element stiffness data: K(a) = sum_e a_e K_e
        x, y = xy[:, :, 0], xy[:, :, 1]
        b = np.stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]], 1)
        c = np.stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]], 1)
        Ke = (b[:, :, None] * b[:, None, :] + c[:, :, None] * c[:, None, :]) / (4.0 * self.area)[:, None, None]
        self._I = np.repeat(self.elem, 3, axis=1).ravel()
        self._J = np.tile(self.elem, (1, 3)).ravel()
        self._V = Ke.reshape(self.n_elem, 9).ravel()
        self._E = np.repeat(np.arange(self.n_elem), 9)
        F = np.zeros(len(self.coords))
        for j in range(3):
            np.add.at(F, self.elem[:, j], self.area / 3.0)
        self.F = F[self.nodes[L]]

    # -- operators -----------------------------------------------------------------------------------------------
    def A_fine(self, a_elem):
        """Stiffness matrix of  -div(a grad u)  on the unknowns of level L (csc); a_elem: one value per fine element."""
        a = np.asarray(a_elem, float).ravel()
        if a.size != self.n_elem:
            raise ValueError(f"a has {a.size} entries, expected {self.n_elem}")
        if a.min() <= 0:
            raise ValueError(f"non-elliptic sample: min a = {a.min():.4g}")
        n = len(self.coords)
        K = sp.coo_matrix((self._V * a[self._E], (self._I, self._J)), shape=(n, n)).tocsr()
        idx = self.nodes[self.L]
        return K[idx][:, idx].tocsc()

    def load(self, f_elem):
        """Load vector of an element-wise constant source f_elem (nodal quadrature: area / 3 per vertex)."""
        Fv = np.zeros(len(self.coords)); w = np.asarray(f_elem) * self.area / 3.0
        for j in range(3):
            np.add.at(Fv, self.elem[:, j], w)
        return Fv[self.nodes[self.L]]

    def mass_matrix(self):
        """Consistent P1 mass matrix on the unknowns of level L (the FE L2 norm)."""
        Mloc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
        n = len(self.coords)
        M = sp.csr_matrix((np.einsum('e,ij->eij', self.area, Mloc).ravel(),
                           (np.repeat(self.elem, 3, axis=1).ravel(), np.tile(self.elem, (1, 3)).ravel())), shape=(n, n))
        idx = self.nodes[self.L]
        return M[idx][:, idx].tocsr()

    def level_coords(self, k):
        return self.coords[self.nodes[k]]

    def prolongation(self, k_from, k_to):
        """Composite geometric prolongation level k_from -> k_to (k_to > k_from)."""
        P = None
        for k in range(k_from + 1, k_to + 1):
            P = self.P[k] if P is None else self.P[k] @ P
        return P.tocsr()
