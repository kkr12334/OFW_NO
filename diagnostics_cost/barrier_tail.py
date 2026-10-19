"""4 Oct: the no-island tail of oa-vc on the L3 barrier test set (GPU run, seed 0, per-sample energy errors pasted from
the --reeval results).  Per test sample (300-399, regenerated from rng 1000): floating-island flag, the smallest
eigenvalue of D^-1 A (shift-invert Lanczos), the classical operator-adapted V(4,4) error, and the two networks' errors."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, eigsh
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, classical_vcycle

OP = np.array([float(v) for v in """0.1113 0.9315 0.0832 0.1199 0.9910 0.1118 0.9830 0.0304 0.7336 0.6068 0.0660 0.1527 0.0727 1.0130 0.0788 0.1083 0.4167 0.3599 0.9601 0.9447 0.0900 0.5534 0.7282 0.5462 0.5638 0.0640 0.9411 0.5673 0.9875 0.6399 0.1095 0.0944 0.0398 0.7592 0.9739 0.9913 0.0796 0.9647 0.0717 0.1071 0.1024 0.1560 0.0693 0.1840 0.0232 0.9056 0.0427 0.8808 0.1884 0.1282 0.0704 0.0855 0.0578 0.1658 0.0282 0.9932 0.6005 0.9605 0.9833 0.4987 0.6788 0.5142 0.1561 0.0716 0.9402 1.6169 0.9372 0.5177 0.8065 0.0386 0.9216 0.0503 0.1583 0.9834 0.0854 0.4194 0.0783 0.6777 0.9542 0.0727 0.9786 0.1309 0.0946 0.1527 0.1612 0.0798 0.5017 0.1173 0.0588 0.1057 0.1036 0.4230 0.1376 0.0737 0.1601 0.1051 0.8721 0.0900 0.1023 0.1089""".split()])
VC = np.array([float(v) for v in """0.0044 0.6491 0.0036 0.0064 0.9639 0.0061 0.8938 0.0019 0.2537 0.2575 0.0072 0.0055 0.0043 0.7203 0.0045 0.0068 0.0818 0.0672 0.6553 0.7933 0.0555 0.4366 0.6049 0.2984 0.1005 0.0293 1.0752 0.3035 0.9538 0.5583 0.0051 0.0288 0.0026 0.5174 0.9319 0.8428 0.0278 0.8580 0.0124 0.0217 0.0107 0.0112 0.0104 0.0960 0.0019 0.2326 0.0026 0.6383 0.0094 0.0134 0.0077 0.0044 0.0090 0.0830 0.0025 0.9791 0.4452 0.9939 0.8786 2.2239 0.4313 0.1598 0.0047 0.0058 0.7525 0.4198 0.9565 0.2587 0.6253 0.0028 0.6694 0.0050 0.0091 0.9500 0.0177 0.3491 0.0172 0.3603 0.8525 0.0042 0.9365 0.0067 0.0051 0.0370 0.0249 0.0048 0.3966 0.0044 0.0088 0.0072 0.0083 0.1048 0.0701 0.0038 0.0566 0.0083 0.2835 0.0812 0.1350 0.0087""".split()])

L = 3; H = NestedHierarchy(L, coarse=perforated_square(n=9))
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]
P2 = H.P[3].tocsr(); dir_nodes = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L])
def has_island(a):
    cond = H.keep & (a >= 0.5); el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M.T @ M).tocsr(); used = np.unique(el.ravel()); n_, lab = connected_components(G[used][:, used], directed=False)
    touch = np.zeros(n_, bool); touch[lab[np.isin(used, dir_nodes)]] = True
    return not touch.all()
rng = np.random.default_rng(1000); A_all = [field_channels(H, 1e-4, rng) for _ in range(400)]
rows = []
for j, i in enumerate(range(300, 400)):
    a = A_all[i]; A = H.A_fine(a).tocsr(); F = load(a >= 0.5); u = spsolve(A.tocsc(), F)
    d = A.diagonal(); Ds = sp.diags(1 / np.sqrt(d)); S = (Ds @ A @ Ds).tocsc()
    lmin = eigsh(S, k=1, sigma=0, which='LM', return_eigenvectors=False)[0]
    uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); levs, A0 = oa_hierarchy(H, A, 3, 4)
    x = uc + classical_vcycle(levs, A0, F - A @ uc); e = x - u
    rows.append((has_island(a), lmin, np.sqrt(e @ (A @ e) / (u @ (A @ u))), np.abs(u).max()))
isl = np.array([r[0] for r in rows]); lm = np.array([r[1] for r in rows]); cl = np.array([r[2] for r in rows]); um = np.array([r[3] for r in rows])
print(f"islands {isl.sum()}/100;  lambda_min(D^-1 A): with island median {np.median(lm[isl]):.1e}, without {np.median(lm[~isl]):.1e}")
ni = ~isl
o = np.argsort(-VC * ni)[:12]
print("\nthe 12 worst NO-island samples for oa-vc (test index, lambda_min, max|u|, oa-vc, oa-ll-op, classical V(4,4)):")
for k in o:
    print(f"  {k:>3}  {lm[k]:.1e}  {um[k]:.2e}  {VC[k]:.4f}  {OP[k]:.4f}  {cl[k]:.4f}")
from scipy.stats import spearmanr
print(f"\nno-island samples: Spearman(log lambda_min, oa-vc error) = {spearmanr(np.log(lm[ni]), VC[ni])[0]:+.2f}; "
      f"(log lambda_min, classical) = {spearmanr(np.log(lm[ni]), cl[ni])[0]:+.2f}; (classical, oa-vc) = {spearmanr(cl[ni], VC[ni])[0]:+.2f}")
for thr in (1e-3, 3e-4, 1e-4):
    near = ni & (lm < thr)
    print(f"  no-island, lambda_min < {thr:.0e}: {near.sum()} samples, oa-vc mean {VC[near].mean() if near.any() else float('nan'):.3f} vs "
          f"classical {cl[near].mean() if near.any() else float('nan'):.3f};  lambda_min >= {thr:.0e}: {(ni & ~near).sum()} samples, "
          f"oa-vc mean {VC[ni & ~near].mean():.3f} (median {np.median(VC[ni & ~near]):.4f}) vs classical {cl[ni & ~near].mean():.3f}, oa-ll-op {OP[ni & ~near].mean():.3f}")
print(f"\nwith islands: oa-vc beats the classical member on {np.mean(VC[isl] < cl[isl]) * 100:.0f}% of samples; "
      f"oa-vc > 1 (worse than doing nothing) on {np.sum(VC > 1)} samples: {list(np.nonzero(VC > 1)[0])}, island flags {list(isl[VC > 1].astype(int))}")
