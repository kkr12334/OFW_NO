"""3 Oct: cost of the operator-adapted (route V local) coarse space vs a direct solve, barrier field 1e-4, one CPU."""
import os, sys, time, numpy as np, scipy.sparse as sp, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels, _local_C
from coarse_meshes import perforated_square
for L in (3, 4):
    H = NestedHierarchy(L, coarse=perforated_square(n=9)); rng = np.random.default_rng(1000)
    nc = H.P[L].shape[1]
    for s in range(3):
        a = field_channels(H, 1e-4, rng); A = H.A_fine(a).tocsr()
        t0 = time.perf_counter(); u = spsolve(A.tocsc(), H.F); td = time.perf_counter() - t0
        A_FF = A[nc:, nc:].tocsc(); A_FC = A[nc:, :nc].tocsr()
        out = []
        for rho in (2, 3):
            t0 = time.perf_counter(); Ph = _local_C(A_FF, A_FC, H.new_coords(L), H.level_coords(L - 1), rho * H.h[L - 1])
            P = sp.vstack([sp.identity(nc, format='csr'), sp.csr_matrix(Ph)]).tocsr(); tb = time.perf_counter() - t0
            t0 = time.perf_counter(); solc = factorized((P.T @ A @ P).tocsc()); uc = P @ solc(P.T @ H.F); ts = time.perf_counter() - t0
            out.append(f"rho{rho}: build {tb*1e3:.0f} ms + Galerkin {ts*1e3:.1f} ms")
        print(f"L{L} ({A.shape[0]} unknowns) sample {s}: direct solve {td*1e3:.1f} ms; " + '; '.join(out))
