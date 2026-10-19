"""Two-grid ceiling of the oa-ll-op class on the actual problem data (3 Oct).
For each problem: e = u_h - u_c (Galerkin coarse solve k0 = L-1), S = I - omega D^-1 A.
Prints mean ||S^nu e||_A / ||e||_A  -- the guaranteed ceiling of a nu-step oa-ll-op -- plus the fraction of positive
off-diagonal stiffness entries (non-M-matrix couplings) and the spread of |a_ij|/a_ii (anisotropy of the stencil).
Textbook expectation: point Jacobi smooths poorly along strongly coupled directions, so the ceiling is worse for the
anisotropic tow problem than for isotropic Darcy."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from scalar_problems import TensorProblem, iso, field_tows

L, k0, n, omega = 3, 2, 12, 2 / 3
outer = None
def run(name, H, Kfun):
    TP = TensorProblem(H); rng = np.random.default_rng(7)
    Pk = H.P[L].tocsr()
    R = {1: [], 4: [], 10: [], 30: []}; pos, spread = [], []
    for _ in range(n):
        A = TP.A_fine(Kfun(H, rng)).tocsr(); F = TP.F
        u = spsolve(A.tocsc(), F); uc = Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F)
        e = u - uc; r = F - A @ uc; dg = A.diagonal(); en = lambda v: np.sqrt(v @ (A @ v))
        x = np.zeros_like(e)
        for k in range(1, 31):
            x = x + omega * (r - A @ x) / dg
            if k in R: R[k].append(en(e - x) / en(e))
        C = A.tocoo(); off = C.row != C.col
        pos.append(np.mean(C.data[off] > 1e-12 * dg[C.row[off]]))
        w = np.abs(C.data[off]) / dg[C.row[off]]; spread.append(np.percentile(w, 95) / max(np.percentile(w, 50), 1e-30))
    print(f"{name:<38} ceiling nu=1 {np.mean(R[1]):.3f}  nu=4 {np.mean(R[4]):.3f}  nu=10 {np.mean(R[10]):.3f}  "
          f"nu=30 {np.mean(R[30]):.3f} | positive off-diag {np.mean(pos):.1%} | |w| p95/p50 {np.mean(spread):.1f}")

Hd = NestedHierarchy(L, coarse=perforated_square(n=9))
run("Darcy channels 1e4 (Dirichlet pores)", Hd, lambda H, rng: iso(field_channels(H, 1e4, rng)))
# anisotropic tows, insulated pores -- same construction as train_scalar.py (contrast 1e2, k_T 5)
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
Ha = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer)
run("aniso tows 1e2, k_T 5 (insulated pores)", Ha, lambda H, rng: field_tows(H, 1e2, rng, k_T=5.0))
run("isotropic channels 1e2 (insulated pores)", Ha, lambda H, rng: iso(field_channels(H, 1e2, rng)))
