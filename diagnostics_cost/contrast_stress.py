"""Reviewer request (3 Oct): stress the classical baseline.  Darcy, perforated L3, channels, contrast 1e4 / 1e6 / 1e8,
Galerkin coarse solve (k0 2) as the start.  Two-grid PCG (omega 0.6 Jacobi + exact Galerkin coarse) and AMG-PCG (pyamg
smoothed aggregation, setup included).  Iterations to reach oa-ll-op's 1e4 accuracy (9.99e-3 / 0.024) and to a relative
energy error of 1e-6; coarse-alone error for scale.  10 samples, CPU."""
import sys, os, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
import pyamg
L, k0, om = 3, 2, 0.6
CONTRASTS = (1e4, 1e6, 1e8, 1e-4, 1e-8)   # < 1: the channels are BARRIERS (low conductivity)
H = NestedHierarchy(L, coarse=perforated_square(n=9)); F = H.F
Pk = None
for k in range(k0 + 1, L + 1):
    Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
def pcg(A, x0, Minv, u, maxit=200):
    x = x0.copy(); r = F - A @ x; z = Minv(r); p = z.copy(); rz = r @ z
    en = lambda v: np.sqrt(v @ (A @ v)); eu = en(u); nu = np.linalg.norm(u)
    h = [(np.linalg.norm(x - u) / nu, en(x - u) / eu)]
    for _ in range(maxit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap; z = Minv(r); rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        h.append((np.linalg.norm(x - u) / nu, en(x - u) / eu))
        if h[-1][1] < 1e-7: break
    return np.array(h)
def hit(h, a, b):
    k = np.nonzero((h[:, 0] <= a) & (h[:, 1] <= b))[0]; return int(k[0]) if len(k) else np.nan
for c in CONTRASTS:
    rng = np.random.default_rng(1000); R = []
    for s in range(10):
        A = H.A_fine(field_channels(H, c, rng)).tocsr(); u = spsolve(A.tocsc(), F)
        solc = factorized((PkT @ A @ Pk).tocsc()); uc = Pk @ solc(PkT @ F); dinv = 1 / A.diagonal()
        def tg(r):
            x = om * dinv * r; x = x + Pk @ solc(PkT @ (r - A @ x)); return x + om * dinv * (r - A @ x)
        ht = pcg(A, uc, tg, u)
        t0 = time.perf_counter(); M = pyamg.smoothed_aggregation_solver(A).aspreconditioner(); ts = time.perf_counter() - t0
        ha = pcg(A, uc, lambda r: M @ r, u)
        en0 = ht[0]
        R.append((en0[0], en0[1], hit(ht, 9.99e-3, 0.024), hit(ht, 1, 1e-6), hit(ha, 9.99e-3, 0.024), hit(ha, 1, 1e-6), ts * 1e3))
    R = np.array(R)
    print(f"contrast {c:.0e}: coarse alone {np.mean(R[:, 0]):.3e} / {np.mean(R[:, 1]):.3f};  iterations median [max] -- "
          f"to oa-ll-op's accuracy: TG-PCG {np.nanmedian(R[:, 2]):.0f} [{np.nanmax(R[:, 2]):.0f}], AMG-PCG {np.nanmedian(R[:, 4]):.0f} "
          f"[{np.nanmax(R[:, 4]):.0f}];  to energy 1e-6: TG-PCG {np.nanmedian(R[:, 3]):.0f} [{np.nanmax(R[:, 3]):.0f}], AMG-PCG "
          f"{np.nanmedian(R[:, 5]):.0f} [{np.nanmax(R[:, 5]):.0f}]  (AMG setup {np.median(R[:, 6]):.0f} ms)", flush=True)
