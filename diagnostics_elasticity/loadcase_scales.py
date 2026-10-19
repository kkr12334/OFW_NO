"""Is the elasticity difficulty caused by the LOAD CASE?  Compare the scale content of the correction
u - u_c (HB level shares, L2) and the coarse-solve error for gravity on a clamped plate (bending-dominated)
vs uniaxial tension (left edge clamped, traction on the right edge, membrane-dominated: the RVE setting)."""
import numpy as np, scipy.sparse as sp, sys; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
L = 3; rng = np.random.default_rng(3)
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
not_left = lambda x, y: not np.isclose(x, 0)
def setup(case):
    if case == 'gravity':
        H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H); return H, V, V.F
    H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=not_left); V = VectorHierarchy(H, body_force=(0, 0))
    F = np.zeros(2 * len(H.coords))
    for a, b in H.mesh.boundary_edges():
        pa, pb = H.coords[a], H.coords[b]
        if np.isclose(pa[0], 1) and np.isclose(pb[0], 1):
            l = np.linalg.norm(pa - pb); F[2 * a] += l / 2; F[2 * b] += l / 2
    return H, V, F[V.nodes[L]]
def hb(P, e):
    out = {}; w = e
    for k in range(L, 0, -1):
        Pk = P[k].tocsr(); nc = Pk.shape[1]; c = w[:nc]; d = w[nc:] - (Pk @ c)[nc:]; out[k] = np.linalg.norm(d)**2; w = c
    out[0] = np.linalg.norm(w)**2; t = sum(out.values()); return [round(out[k]/t, 3) for k in sorted(out)]
for case in ('gravity', 'tension'):
    H, V, F = setup(case)
    for c in (1, 1e2):
        E = np.asarray(field_channels(H, c, rng)).ravel(); A = V.A_fine(E); u = spsolve(A, F)
        P2 = V.P[3]; uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); e = u - uc
        en = np.sqrt(e @ A @ e / (u @ A @ u)); rl = np.linalg.norm(e) / np.linalg.norm(u)
        print(f"{case:8s} contrast {c:>5g}: coarse (level 2) error rel-L2 {rl:.3f} energy {en:.3f};  correction HB share L0..L3 {hb(V.P, e)}")
