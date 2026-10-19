"""3 Oct: classical iterations needed to match oa-ll-op trained on n=1000 (750 train), Darcy L3 perforated, channels 1e4,
k0 2 Galerkin, seed 0: rel-L2 6.894e-3 / energy 1.863e-2.  Same methods and samples as classical_vs_surrogate.py (L3 only),
but printing every iteration 0..8.  Note: the classical errors are on 10 fresh samples (rng 1000), the network's on its
250-sample test set; same field family and contrast."""
import sys, os, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
import pyamg

TGT = {'oa-ll-op n=1000 (seed 0)': (6.894e-3, 1.863e-2), 'gnn-op n=1000 (seed 0)': (1.181e-2, 2.530e-2),
       'oa-ll-op n=400 (3 seeds)': (9.99e-3, 0.024)}


def pcg(A, b, x0, Minv, u, maxit=12):
    x = x0.copy(); r = b - A @ x; z = Minv(r); p = z.copy(); rz = r @ z; hist = []
    en = lambda v: np.sqrt(v @ (A @ v)); nu_ = en(u); nl = np.linalg.norm(u)
    hist.append((np.linalg.norm(x - u) / nl, en(x - u) / nu_))
    for _ in range(maxit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = Minv(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        hist.append((np.linalg.norm(x - u) / nl, en(x - u) / nu_))
    return np.array(hist)


L, k0 = 3, 2
H = NestedHierarchy(L, coarse=perforated_square(n=9)); rng = np.random.default_rng(1000)
Pk = None
for k in range(k0 + 1, L + 1):
    Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr(); PkT = Pk.T.tocsr(); F = H.F; om = 2 / 3
rows = {'TG-PCG': [], 'AMG-PCG': []}
for s in range(10):
    a = field_channels(H, 1e4, rng); A = H.A_fine(a).tocsr(); u = spsolve(A.tocsc(), F)
    Ac = (PkT @ A @ Pk).tocsc(); solc = factorized(Ac); uc = Pk @ solc(PkT @ F); dinv = 1 / A.diagonal()
    def tg(r):
        x = om * dinv * r; x = x + Pk @ solc(PkT @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    rows['TG-PCG'].append(pcg(A, F, uc, tg, u))
    ml = pyamg.smoothed_aggregation_solver(A); M = ml.aspreconditioner(cycle='V')
    rows['AMG-PCG'].append(pcg(A, F, uc, lambda r: M @ r, u))
for name, hs in rows.items():
    hs = np.array(hs); m = hs.mean(0)
    print(f"{name}: mean rel-L2 / energy after j its")
    for j in range(9):
        print(f"   {j}: {m[j, 0]:.3e} / {m[j, 1]:.4f}")
    for tn, t in TGT.items():
        hit = [int(np.nonzero((h[:, 0] <= t[0]) & (h[:, 1] <= t[1]))[0][0]) for h in hs]
        print(f"   to match {tn} {t[0]:.2e}/{t[1]:.4f}: median {np.median(hit):.0f} its (min {min(hit)}, max {max(hit)})")
