"""4 Oct: classical comparators for the k0 = 0 run (no external coarse solve: the networks start from u = 0), Darcy L3
perforated, channels 1e4, the driver's test set (samples 300-399 of rng 1000).  Kaggle GPU seed 0:
oa-ll-op 3.949e-2 / 8.062e-2, oa-vcb 2.489e-3 / 6.783e-3 (rel-L2 / energy).
Same information as oa-vcb (the operator, its operator-adapted hierarchy, the exact 28-node coarsest solve):
  * one classical V(4,4)-cycle from zero (Chebyshev smoothers, same transfers and Galerkin levels);
  * the classical member of oa-vcb's class (the same cycle + near-kernel deflation + Galerkin output projection);
  * PCG from zero preconditioned by that V-cycle: iterations needed to match each network (both norms)."""
import sys, os, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, classical_vcycle, classical_vcycle_proj, near_kernel_basis

NETS = {'oa-ll-op': (3.949e-2, 8.062e-2), 'oa-vcb': (2.489e-3, 6.783e-3)}
L = 3; H = NestedHierarchy(L, coarse=perforated_square(n=9)); F = H.F
rng = np.random.default_rng(1000); A_all = [field_channels(H, 1e4, rng) for _ in range(400)]
rows = {'V(4,4)': [], 'member': []}; hist = []
for i in range(300, 400):
    A = H.A_fine(A_all[i]).tocsr(); u = spsolve(A.tocsc(), F); levs, A0 = oa_hierarchy(H, A, 3, 4)
    en = lambda v: np.sqrt(v @ (A @ v)); nu_, nl = en(u), np.linalg.norm(u)
    err = lambda x: (np.linalg.norm(x - u) / nl, en(x - u) / nu_)
    rows['V(4,4)'].append(err(classical_vcycle(levs, A0, F)))
    rows['member'].append(err(classical_vcycle_proj(levs, A0, F, Z=near_kernel_basis(A), nu=4)))
    M = lambda r: classical_vcycle(levs, A0, r)
    x = np.zeros_like(F); r = F.copy(); z = M(r); p = z.copy(); rz = r @ z; h = [err(x)]
    for _ in range(10):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2; h.append(err(x))
    hist.append(h)
for k, v in rows.items():
    v = np.array(v); print(f"one pass, classical {k:<7}: rel-L2 {v[:, 0].mean():.3e}  energy {v[:, 1].mean():.4f}")
hist = np.array(hist); m = hist.mean(0)
print("PCG from zero, preconditioner = the classical V(4,4)-cycle (mean rel-L2 / energy after j its):")
for j in range(7): print(f"   {j}: {m[j, 0]:.3e} / {m[j, 1]:.5f}")
for n_, t in NETS.items():
    hit = [int(np.nonzero((h[:, 0] <= t[0]) & (h[:, 1] <= t[1]))[0][0]) if ((h[:, 0] <= t[0]) & (h[:, 1] <= t[1])).any() else 99 for h in hist]
    print(f"   to match {n_} {t[0]:.2e} / {t[1]:.4f} per sample: median {np.median(hit):.0f} its (min {min(hit)}, max {max(hit)})")
