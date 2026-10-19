"""Coarse triangulations for the unstructured-domain experiments.

Every generator returns (coords, elements) suitable for MeshHierarchy(coords, elements) and hence for
NestedHierarchy(L, coarse=(coords, elements)).  Uniform refinement keeps the domain exactly, so the
lifting, the block structure and the certificate need no change.  scipy.spatial.Delaunay is the only
dependency (no meshing library).
"""
import numpy as np
from scipy.spatial import Delaunay


def _triangulate(points, inside, min_angle_deg=15.0):
    """Delaunay of `points`, keeping triangles whose centroid satisfies `inside` and that are not
    degenerate slivers (min angle >= min_angle_deg)."""
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


def perforated_square(n=9, holes=((0.30, 0.32, 0.12), (0.68, 0.62, 0.15), (0.30, 0.72, 0.10)),
                      jitter=0.25, seed=0, min_angle_deg=15.0):
    """Unit square with circular holes, meshed unstructured (jittered grid + Delaunay).
    `holes` = ((cx, cy, r), ...).  Points inside a hole are dropped and the hole is ringed with
    nodes, so the hole boundary is a polygon of mesh edges -- a real boundary, not a mask."""
    rng = np.random.default_rng(seed)
    h = 1.0 / (n - 1)
    g = np.stack(np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing='ij'), -1).reshape(-1, 2)
    edge = (np.isclose(g[:, 0], 0) | np.isclose(g[:, 0], 1) |
            np.isclose(g[:, 1], 0) | np.isclose(g[:, 1], 1))
    g[~edge] += jitter * h * rng.standard_normal((np.sum(~edge), 2))

    def d_hole(p):                     # signed distance to the nearest hole (>0 outside every hole)
        return min((np.hypot(p[0] - cx, p[1] - cy) - r) for cx, cy, r in holes)

    pts = [p for p in g if d_hole(p) > 0.45 * h]
    for cx, cy, r in holes:            # ring of nodes on each hole boundary
        m = max(8, int(round(2 * np.pi * r / h)))
        th = np.arange(m) * 2 * np.pi / m
        pts += [np.array([cx + r * np.cos(t), cy + r * np.sin(t)]) for t in th]
    return _triangulate(np.array(pts), lambda c: d_hole(c) > 0, min_angle_deg)


def polygon_domain(vertices, n=10, jitter=0.25, seed=0, min_angle_deg=15.0):
    """Unstructured mesh of an arbitrary simple polygon (list of (x, y), counter-clockwise)."""
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
    for i in range(len(V)):                          # nodes along the polygon boundary
        a, b = V[i], V[(i + 1) % len(V)]
        m = max(1, int(round(np.linalg.norm(b - a) / h)))
        pts += [a + (b - a) * j / m for j in range(m)]
    gx = np.arange(lo[0] + h / 2, hi[0], h); gy = np.arange(lo[1] + h / 2, hi[1], h)
    for x in gx:
        for y in gy:
            p = np.array([x, y]) + jitter * h * rng.standard_normal(2)
            if inside(p) and all(np.linalg.norm(p - q) > 0.5 * h for q in pts):
                pts.append(p)
    return _triangulate(np.array(pts), inside, min_angle_deg)


def l_shape_coarse(n=5, **kw):
    """L-shaped domain as an UNSTRUCTURED coarse mesh (the grid-aligned version lives in
    NestedHierarchy(domain='L'); this one is for the mesh-vs-grid comparison)."""
    return polygon_domain([(0, 0), (1, 0), (1, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)], n=n, **kw)


def mesh_quality(coords, elements):
    """(n_nodes, n_elements, min angle in degrees, max/min element area)."""
    p = np.asarray(coords)[np.asarray(elements)]
    ang, area = [], []
    for t in p:
        a_, b_ = t[1] - t[0], t[2] - t[0]                      # 2-D cross product (np.cross of 2-D vectors is removed in NumPy 2.x)
        area.append(0.5 * abs(a_[0] * b_[1] - a_[1] * b_[0]))
        for i in range(3):
            u = t[(i + 1) % 3] - t[i]; v = t[(i + 2) % 3] - t[i]
            ang.append(np.degrees(np.arccos(np.clip((u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1))))
    return len(coords), len(elements), float(np.min(ang)), float(np.max(area) / np.min(area))
