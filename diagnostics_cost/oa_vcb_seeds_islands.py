"""4 Oct: island split of oa-vcb seeds 1-2 on the L3 barrier test set (Kaggle GPU runs; per-sample energy errors en_s
read from the results jsonl).  Per test sample (300-399, regenerated from rng 1000): floating-island flag (coefficient
islands not touching the Dirichlet boundary) and the classical member of the class (operator-adapted V(4,4) + near-kernel
deflation + Galerkin projection, from the same geometric u_c), for paired comparisons.
usage: python diagnostics_cost/oa_vcb_seeds_islands.py results_barriers.jsonl"""
import sys, os, json, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, classical_vcycle_proj, near_kernel_basis

runs = []
for line in open(sys.argv[1]):
    d = json.loads(line); c = json.loads(d['cfg'])
    if c['contrast'] < 1 and d['model'] == 'oa-vcb': runs.append((c['seed'], np.array(d['result']['en_s'])))

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
isl, cm = [], []
for i in range(300, 400):
    a = A_all[i]; A = H.A_fine(a).tocsr(); F = load(a >= 0.5); u = spsolve(A.tocsc(), F)
    uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); levs, A0 = oa_hierarchy(H, A, 3, 4)
    x = uc + classical_vcycle_proj(levs, A0, F - A @ uc, Z=near_kernel_basis(A), nu=4); e = x - u
    isl.append(has_island(a)); cm.append(np.sqrt(e @ (A @ e) / (u @ (A @ u))))
isl, cm = np.array(isl), np.array(cm)
print(f"islands {isl.sum()}/100")
print(f"classical member (V(4,4) op.-adapted + deflation + projection): all {cm.mean():.4f}, island {cm[isl].mean():.4f}, "
      f"no island {cm[~isl].mean():.4f}")
for s, en in sorted(runs):
    print(f"oa-vcb seed {s}: all {en.mean():.4f}, island {en[isl].mean():.4f}, no island {en[~isl].mean():.4f};  "
          f"beats the classical member on {np.mean(en < cm) * 100:.0f}% of samples (island {np.mean(en[isl] < cm[isl]) * 100:.0f}%, "
          f"no island {np.mean(en[~isl] < cm[~isl]) * 100:.0f}%);  worst 3: {np.round(np.sort(en)[-3:], 3)}, "
          f"samples > classical x2: {int(np.sum(en > 2 * cm))}")
if len(runs) > 1:
    E = np.array([en for _, en in runs]); print(f"seed-to-seed per-sample spread: mean |log(en_s1/en_s2)| = "
                                              f"{np.mean(np.abs(np.log(E[0] / E[1]))):.3f}")
