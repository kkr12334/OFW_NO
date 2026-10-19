"""development diagnostic (input-independent): worst-case efficiency of the GLOBAL moment-constrained RT0(T_b) flux
(the 'joint optimum').  For the residual of any v in V_h, eta_joint^2 = R^T S^{-1}... precisely:
    eta_joint = max_lambda (lambda . R) / sqrt(lambda^T S lambda),   S = G M^{-1} G^T,
where M is the K^{-1}-weighted RT0(T_b) mass matrix and (G sigma)_a = (div sigma, psi_a).  With |||u_h - v||| =
sqrt(R^T A^{-1} R), the worst case over ALL inputs is  C_glob = max eta_joint / |||u_h - v||| = sqrt(lambda_max(A, S)),
i.e. 1 / sqrt(lambda_min(S, A)).  Seed-1 operator 300 geometry, contrast swept."""
import os, sys, time, json, numpy as np, scipy.sparse as sp, scipy.linalg as sl
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
import rc_fem as R
H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9)); fl = M.fl
rng = np.random.default_rng(1001); a4 = [make_field(H, 'channels', 1e4, rng) for _ in range(301)][300]; chan = a4 > 1
nb, nE = len(fl.E), fl.n_edge
D = sp.csr_matrix((fl.div_int.ravel(), (np.repeat(np.arange(nb), 3), fl.edge.ravel())), shape=(nb, nE))
G = (M.Bb @ sp.diags(1 / fl.area) @ D).tocsr()
res = []
for fam in ('channels', 'barriers'):
    for c in (1e0, 1e2, 1e4, 1e6):
        t0 = time.time()
        K = np.where(chan, c if fam == 'channels' else 1 / c, 1.0)
        Mb = R.local_mass(M, K, M.L + 1)
        I = np.repeat(fl.edge, 3, 1).ravel(); J = np.tile(fl.edge, (1, 3)).ravel()
        Mg = sp.csr_matrix((Mb.ravel(), (I, J)), shape=(nE, nE)).tocsc()
        X = splu(Mg).solve(G.T.toarray())
        S = G @ X; S = 0.5 * (S + S.T)
        A = M.stiffness(K).toarray()
        lmin = sl.eigh(S, A, eigvals_only=True, subset_by_index=[0, 0])[0]
        res.append(dict(family=fam, contrast=c, C_glob=float(1 / np.sqrt(lmin))))
        print(f'{fam:8s} c={c:6.0e}: worst-case efficiency of the global moment-constrained RT0(T_b) flux C_glob = {1 / np.sqrt(lmin):.4f}  ({time.time() - t0:.0f}s)', flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'global_stability.json'), 'w'), indent=1)
