"""development diagnostic: input-independent PU robustness index vs contrast (seed-1 development operator 300).
Contrast c: channels K in {1, c}; barriers K in {1, 1/c}.  Same geometry for every c."""
import os, sys, time, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis
import rc_fem as R
H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9))
rng = np.random.default_rng(1001); a4 = [make_field(H, 'channels', 1e4, rng) for _ in range(301)][300]
chan = a4 > 1
for fam in ('channels', 'barriers'):
    for c in (1e0, 1e1, 1e2, 1e3, 1e4, 1e6):
        K = np.where(chan, c if fam == 'channels' else 1 / c, 1.0)
        levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
        Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
        row = [f'{fam:8s} c={c:7.0e} nk={Z.shape[1]}']
        for nm, pu in (('G2', R.pu_geometric(M, 2)), ('G1', R.pu_geometric(M, 1)), ('OA2', R.pu_operator_adapted(M, Phi3, Z))):
            ci = R.pu_robustness_index(M, K, pu[0], pu[1])
            fl, bd = ci[pu[1]], ci[~pu[1]]
            row.append(f'{nm}: floating max {fl.max():8.2f} p95 {np.quantile(fl, 0.95):7.2f} | boundary max {bd.max():7.2f}')
        print(' | '.join(row), flush=True)
