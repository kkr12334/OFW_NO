"""Stock-take idea check (3 Oct): a CHEAP certificate from the symmetric two-grid preconditioner B_TG (omega-Jacobi,
exact Galerkin coarse solve, omega-Jacobi; omega = 0.6 < 2/3 so that the smoother is an A-contraction for every
coefficient, since lambda_max(D^-1 A) <= 3).  Then B_TG A = I - E with E A-selfadjoint and PSD, hence
lambda_max(B_TG A) <= 1 exactly and eta = sqrt(r^T B_TG r) <= ||e||_A for EVERY r (a guaranteed lower bound), while
||e||_A <= eta / sqrt(lambda_min).  Cost: one coarse solve + 3 mat-vecs, against the Route V rho = 2 / 4 transform
(~1.1 s / ~3.9 s per sample at L4).  Darcy, perforated, channels 1e4; Lanczos (60 steps) for lambda_min, lambda_max."""
import sys, os, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import factorized
from oa_lifting import NestedHierarchy, field_channels, ritz_extremes
from coarse_meshes import perforated_square
om = 0.6
for L, k0 in ((3, 2), (4, 3)):
    H = NestedHierarchy(L, coarse=perforated_square(n=9)); rng = np.random.default_rng(1000)
    Pk = None
    for k in range(k0 + 1, L + 1):
        Pk = H.P[k] if Pk is None else H.P[k] @ Pk
    Pk = Pk.tocsr(); PkT = Pk.T.tocsr(); out = []
    for s in range(4):
        A = H.A_fine(field_channels(H, 1e4, rng)).tocsr(); dinv = 1 / A.diagonal()
        t0 = time.perf_counter(); solc = factorized((PkT @ A @ Pk).tocsc()); tf = time.perf_counter() - t0
        def B(r):
            x = om * dinv * r; x = x + Pk @ solc(PkT @ (r - A @ x)); return x + om * dinv * (r - A @ x)
        r = np.random.default_rng(s).standard_normal(A.shape[0]); t0 = time.perf_counter(); B(r); tb = time.perf_counter() - t0
        lo, hi = ritz_extremes(A, B, n_iter=60, seed=s)
        out.append((lo, hi, tf * 1e3, tb * 1e3))
    o = np.array(out)
    print(f"L{L} k0 {k0}: lambda(B_TG A) in [{o[:, 0].min():.3f}, {o[:, 1].max():.4f}] (min over samples / max); "
          f"guaranteed factor 1/sqrt(lambda_min) = {1 / np.sqrt(o[:, 0].min()):.2f}; eta costs {np.median(o[:, 3]):.2f} ms "
          f"(+ coarse factorisation {np.median(o[:, 2]):.1f} ms, already paid by the surrogate pipeline)")
