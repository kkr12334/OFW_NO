"""Independent check of ASTRA's shape constant beta_T (Route C review supplement, section 1), using rc_fem's own Alfeld split
(ASTRA's audit assembles the element matrices independently of rc_fem; this is the converse check).
B_T = RT0 fields on the three centroid sub-triangles of parent T with zero normal flux on T's outer edges (3 internal-edge dofs).
beta_T^2 = lambda_min( L H^{-1} L^T / |T| ),  L (2x3) = integrals of the basis, H (3x3) = unweighted mass matrix.
Writes beta_shape.json."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
import rc_fem as R

cm = perforated_square(n=9)
M = R.Mesh(NestedHierarchy(3, coarse=cm), cm)
fl = M.fl; nf = len(M.fe.E); b2 = np.zeros(nf)
for k in range(nf):
    ts = np.arange(3 * k, 3 * k + 3)
    e = fl.edge[ts]; ue, cnt = np.unique(e, return_counts=True); internal = ue[cnt == 2]
    Hm = np.zeros((3, 3)); Lm = np.zeros((2, 3))
    for t in ts:
        ph = M.phi_q[t]                                       # (3 local basis, 3 quadrature points, 2)
        loc = [np.nonzero(internal == fl.edge[t, i])[0] for i in range(3)]
        for i in range(3):
            if len(loc[i]) == 0: continue
            a = loc[i][0]
            Lm[:, a] += (ph[i] * M.wq[t]).sum(0)
            for j in range(3):
                if len(loc[j]) == 0: continue
                Hm[a, loc[j][0]] += np.sum(ph[i] * ph[j], 1) @ np.full(3, M.wq[t])
    b2[k] = np.linalg.eigvalsh(Lm @ np.linalg.solve(Hm, Lm.T) / M.fe.area[k])[0]
coarse_parent = np.arange(nf) // 64                           # L3 = 3 uniform red refinements of the 116 coarse triangles
per_coarse = np.array([b2[coarse_parent == p].min() for p in range(nf // 64)])
spread = float(max(np.ptp(b2[coarse_parent == p]) for p in range(nf // 64)))
out = dict(level=3, parent_triangles=nf, coarse_triangles=int(nf // 64), min_beta2=float(b2.min()), inv_min_beta=float(1 / np.sqrt(b2.min())),
           distinct_values_rounded_1e10=int(len(np.unique(np.round(b2, 10)))), max_spread_within_coarse_triangle=spread,
           min_beta2_over_coarse=float(per_coarse.min()))
print(out)
json.dump(out, open(os.path.join(HERE, 'beta_shape.json'), 'w'), indent=1)
