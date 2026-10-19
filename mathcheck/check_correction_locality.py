"""Locality of the fine-scale correction after the exact coarse solve (3 Oct).
For a point residual at fine node i the correction is e_i = (I - Pi_H) A^-1 delta_i (Pi_H = A-orthogonal projection
onto the level-k0 space).  Printed: mean fraction of ||e_i||_A^2 carried by nodes FARTHER than rho*H from node i,
rho = 1, 2, 3, 4 (H = level-k0 mesh size).  LOD theory: exponential decay in rho, with a rate that degrades with the
coefficient contrast.  Question: is the correction less local for Darcy channels (1e4) than for aniso tows (1e2)?"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from scalar_problems import TensorProblem, iso, field_tows

L, k0 = 3, 2
def run(name, H, Kfun, n_samp=4, n_pts=25):
    TP = TensorProblem(H); rng = np.random.default_rng(11)
    P = H.P[L].tocsr(); xy = H.level_coords(L); Hk = H.h[k0]
    out = {r: [] for r in (1, 2, 3, 4)}
    for _ in range(n_samp):
        A = TP.A_fine(Kfun(H, rng)).tocsc(); lu = splu(A)
        Ac = (P.T @ A @ P).tocsc(); luc = splu(Ac)
        for i in rng.choice(A.shape[0], n_pts, replace=False):
            d = np.zeros(A.shape[0]); d[i] = 1.0
            u = lu.solve(d); e = u - P @ luc.solve(P.T @ (A @ u))          # (I - Pi_H) A^-1 delta_i
            tot = e @ (A @ e); dist = np.linalg.norm(xy - xy[i], axis=1)
            for r in out:
                eo = np.where(dist > r * Hk, e, 0.0); out[r].append((eo @ (A @ eo)) / tot)
    print(f"{name:<40} energy fraction beyond rho*H:  " + "  ".join(f"rho={r}: {np.mean(v):.2e}" for r, v in out.items()))

Hd = NestedHierarchy(L, coarse=perforated_square(n=9))
run("Darcy channels 1e4 (Dirichlet pores)", Hd, lambda H, rng: iso(field_channels(H, 1e4, rng)))
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
Ha = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer)
run("aniso tows 1e2, k_T 5 (insulated pores)", Ha, lambda H, rng: field_tows(H, 1e2, rng, k_T=5.0))
run("Darcy channels 1e2 (control: same field)", Hd, lambda H, rng: iso(field_channels(H, 1e2, rng)))
