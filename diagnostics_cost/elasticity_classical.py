"""Stock-take (3 Oct): classical two-grid PCG for the elasticity class, started from the Galerkin coarse solve (the
input with which no network learns) and, for comparison, the AMG (smoothed aggregation, rigid-body near-null space)
PCG.  Bands 1e2, porous plate, L3 k0 2, 6 samples; symmetric two-grid with 2x2 block-Jacobi omega 0.6 smoothing."""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
import pyamg
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
L, k0, om = 3, 2, 0.6
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H); nI = len(H.nodes[L])
Pk = V.P[L].tocsr(); PkT = Pk.T.tocsr(); F = V.F; rng = np.random.default_rng(2000)
def bdinv(A):
    i = np.arange(nI); r = np.concatenate([2*i, 2*i, 2*i+1, 2*i+1]); c = np.concatenate([2*i, 2*i+1, 2*i, 2*i+1])
    Db = np.asarray(A[r, c]).ravel().reshape(4, nI).T.reshape(nI, 2, 2); Di = np.linalg.inv(Db)
    return sp.csr_matrix((Di.reshape(nI, 4).T.ravel(), (r, c)), shape=A.shape)
def pcg(A, x0, Minv, u, maxit=60):
    x = x0.copy(); r = F - A @ x; z = Minv(r); p = z.copy(); rz = r @ z
    en = lambda v: np.sqrt(v @ (A @ v)); h = [(np.linalg.norm(x - u) / np.linalg.norm(u), en(x - u) / en(u))]
    for _ in range(maxit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap; z = Minv(r); rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        h.append((np.linalg.norm(x - u) / np.linalg.norm(u), en(x - u) / en(u)))
    return np.array(h)
H_tg, H_amg, tms = [], [], []
xy = np.repeat(H.coords[H.nodes[L]], 2, axis=0)
Bns = np.zeros((2 * nI, 3)); Bns[0::2, 0] = 1; Bns[1::2, 1] = 1
Bns[0::2, 2] = -H.coords[H.nodes[L]][:, 1]; Bns[1::2, 2] = H.coords[H.nodes[L]][:, 0]   # rigid-body modes
for s in range(6):
    A = V.A_fine(field_channels(H, 1e2, rng)).tocsr()
    t0 = time.perf_counter(); u = spsolve(A.tocsc(), F); td = time.perf_counter() - t0
    solc = factorized((PkT @ A @ Pk).tocsc()); uc = Pk @ solc(PkT @ F); Dm = bdinv(A)
    def tg(r):
        x = om * (Dm @ r); x = x + Pk @ solc(PkT @ (r - A @ x)); return x + om * (Dm @ (r - A @ x))
    t0 = time.perf_counter(); H_tg.append(pcg(A, uc, tg, u)); ttg = (time.perf_counter() - t0) / 60
    t0 = time.perf_counter(); ml = pyamg.smoothed_aggregation_solver(A, B=Bns); M = ml.aspreconditioner(); ts = time.perf_counter() - t0
    t0 = time.perf_counter(); H_amg.append(pcg(A, uc, lambda r: M @ r, u)); ta = (time.perf_counter() - t0) / 60
    tms.append((td, ttg, ts, ta))
t = 1e3 * np.median(np.array(tms), 0)
print(f"elasticity L{L} bands 1e2, {2 * nI} dofs: direct {t[0]:.0f} ms; per iteration TG-PCG {t[1]:.2f} ms, AMG-PCG {t[3]:.2f} ms (+ setup {t[2]:.0f} ms)")
for name, hs in (('TG-PCG', H_tg), ('AMG-PCG', H_amg)):
    m = np.array(hs).mean(0)
    print(f"  {name:<8} from u_c: " + '  '.join(f"{j}:{m[j, 0]:.1e}/{m[j, 1]:.3f}" for j in (0, 1, 2, 4, 8, 16, 32, 60)))
