"""Stock-take classical column for the anisotropic class (3 Oct): two-grid PCG (omega-Jacobi 0.6, exact Galerkin
coarse solve) and AMG-PCG, started from the Galerkin coarse solve u_c (k0 2) -- the inputs oa-ll-op gets.  Tows
k_L 1e2, k_T 5, insulated pores, L3, 10 samples, CPU.  Target: oa-ll-op 3-seed mean 2.09e-2 / 0.075; gnn-op 2.28e-2 / 0.058."""
import sys, os, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from scalar_problems import TensorProblem, field_tows
import pyamg
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
L, k0, om = 3, 2, 0.6
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); TP = TensorProblem(H); F = TP.F
Pk = None
for k in range(k0 + 1, L + 1):
    Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr(); PkT = Pk.T.tocsr(); rng = np.random.default_rng(3000)
def pcg(A, x0, Minv, u, maxit=30):
    x = x0.copy(); r = F - A @ x; z = Minv(r); p = z.copy(); rz = r @ z
    en = lambda v: np.sqrt(v @ (A @ v)); h = [(np.linalg.norm(x - u) / np.linalg.norm(u), en(x - u) / en(u))]
    for _ in range(maxit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap; z = Minv(r); rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        h.append((np.linalg.norm(x - u) / np.linalg.norm(u), en(x - u) / en(u)))
    return np.array(h)
HT, HA, tm = [], [], []
for s in range(10):
    A = TP.A_fine(field_tows(H, 1e2, rng, k_T=5.0)).tocsr(); u = spsolve(A.tocsc(), F)
    t0 = time.perf_counter(); solc = factorized((PkT @ A @ Pk).tocsc()); uc = Pk @ solc(PkT @ F); tc = time.perf_counter() - t0
    dinv = 1 / A.diagonal()
    def tg(r):
        x = om * dinv * r; x = x + Pk @ solc(PkT @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    t0 = time.perf_counter(); HT.append(pcg(A, uc, tg, u)); tt = (time.perf_counter() - t0) / 30
    t0 = time.perf_counter(); M = pyamg.smoothed_aggregation_solver(A).aspreconditioner(); ts = time.perf_counter() - t0
    t0 = time.perf_counter(); HA.append(pcg(A, uc, lambda r: M @ r, u)); ta = (time.perf_counter() - t0) / 30
    tm.append((tc, tt, ts, ta))
t = 1e3 * np.median(np.array(tm), 0)
print(f"aniso L3 k0 2, {len(H.nodes[L])} unknowns: coarse solve {t[0]:.1f} ms; per iteration TG-PCG {t[1]:.2f} ms, AMG-PCG {t[3]:.2f} ms (+ setup {t[2]:.1f} ms)")
for name, hs in (('TG-PCG', HT), ('AMG-PCG', HA)):
    m = np.array(hs).mean(0)
    print(f"  {name:<8} from u_c: " + '  '.join(f"{j}:{m[j, 0]:.1e}/{m[j, 1]:.3f}" for j in (0, 1, 2, 3, 4, 5, 8)))
