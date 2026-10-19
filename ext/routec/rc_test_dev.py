"""development checks of the reference certificate on seed-1 development operators (id 300).  Not confirmatory."""
import os, sys, time, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis
import rc_fem as R

fam = sys.argv[1] if len(sys.argv) > 1 else 'channels'
idx = int(sys.argv[2]) if len(sys.argv) > 2 else 300
contrast = {'channels': 1e4, 'barriers': 1e-4}[fam]
H = NestedHierarchy(3, coarse=perforated_square(n=9))
t0 = time.time(); M = R.Mesh(H, perforated_square(n=9)); print(f'mesh built {time.time() - t0:.1f}s; FE elements {len(M.fe.E)}, flux elements {len(M.fl.E)}, flux edges {M.fl.n_edge}')
rng = np.random.default_rng(1001)
aa = [make_field(H, 'channels', contrast, rng) for _ in range(idx + 1)]
K = aa[idx]
fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
A = M.stiffness(K); A_ref = H.A_fine(K)
print('stiffness matches H.A_fine:', abs(A - A_ref).max() / abs(A_ref).max())
F = M.load(fK)
uh = spsolve(A, F)
nrm = M.energy(K, M.full(uh))
# inputs
P2 = R.geometric_P(M, 2)
uc = P2 @ np.linalg.solve((P2.T @ A @ P2).toarray(), P2.T @ F)
rng2 = np.random.default_rng(7)
noise = rng2.standard_normal(M.N); noise *= 0.1 * nrm / M.energy(K, M.full(noise))
inputs = {'exact u_h': uh, 'u_c (k0=2 Galerkin)': uc, 'u_h + 10% white noise': uh + noise}
# coarse spaces
levs, A0 = oa_hierarchy(H, A_ref, 3, 4)
Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr()
Z = near_kernel_basis(levs[0][0])
pus = {'G2 (geometric level-2 hats)': R.pu_geometric(M, 2), 'G1 (geometric level-1 hats)': R.pu_geometric(M, 1),
       'OA2 (operator-adapted level 2, Shepard, +Z)': R.pu_operator_adapted(M, Phi3, Z)}
print(f'{fam} id {idx}: |||u_h||| = {nrm:.4e}, near-kernel dim {Z.shape[1]}'); Mb = R.local_mass(M, K, M.L + 1)
for name, v in inputs.items():
    e_true = M.energy(K, M.full(uh - v))
    Rv = F - A @ v
    rK = R.lift_residual(M, Rv)
    lift_err = np.max(np.abs(M.Bb @ rK - Rv)) / max(1e-300, np.max(np.abs(Rv))) if np.any(Rv) else 0.0
    t0 = time.time(); Sd, dd = R.flux_disc(M, K, fK, rK, M.full(v), Mb); dd = dd['defect']; td = time.time() - t0
    div_err = np.max(np.abs(Sd.div() - (fK[M.par] - rK)))
    print(f'\n[{name}] |||u_h - v||| = {e_true:.4e} ({e_true / nrm:.2e} rel); lifting defect {lift_err:.1e}; '
          f'disc flux {td:.1f}s, patch defect {dd:.1e}, div defect {div_err:.1e}')
    rows = []
    t0 = time.time(); So = R.flux_alg_oracle(M, K, rK, Mb); rows.append(('oracle global RT0', So, time.time() - t0, {}))
    t0 = time.time(); Sm, im = R.flux_alg_mvy(M, K, A, rK, Rv); rows.append(('MVY multilevel (geometric)', Sm, time.time() - t0, im))
    for pn, (Th, inv, C) in pus.items():
        t0 = time.time(); Sp, ip = R.flux_alg_pu(M, K, A, rK, Rv, Th, inv, C, Mloc_b=Mb); rows.append((pn, Sp, time.time() - t0, ip))
    for nm, S, tt, info in rows:
        eta = R.ps_norm(M, K, S)
        dv = np.max(np.abs(S.div() - rK))
        tot = Field = None
        St = R.Field(M); St += Sd; St += S
        eta_tot = R.ps_norm(M, K, St, M.full(v))
        dvt = np.max(np.abs(St.div() - fK[M.par]))
        ok = eta >= e_true * (1 - 1e-10)
        print(f'   {nm:45s} eta_alg {eta:.4e}  eff {eta / e_true if e_true > 0 else float("nan"):8.3f}  guarantee {ok}  '
              f'div defect {dv:.1e}  compat {info.get("compat", 0):.1e}  eta_total {eta_tot:.4e} (div {dvt:.1e})  {tt:.1f}s')
