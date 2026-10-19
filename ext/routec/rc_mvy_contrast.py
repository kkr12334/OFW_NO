"""development check: is the MVY multilevel overestimation a contrast effect (not an implementation error)?
Same geometry (seed-1 operator 300 channels/barriers), contrast c from 1 to 1e4; input 30 Jacobi-PCG steps and u_c."""
import os, sys, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis
import rc_fem as R
from rc_goal import joint_flux
H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9))
rng = np.random.default_rng(1001); a4 = [make_field(H, 'channels', 1e4, rng) for _ in range(301)][300]; chan = a4 > 1
for fam in ('channels', 'barriers'):
    for c in (1e0, 1e1, 1e2, 1e3, 1e4):
        K = np.where(chan, c if fam == 'channels' else 1 / c, 1.0)
        fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
        A = M.stiffness(K); F = M.load(fK); uh = spsolve(A, F); Mb = R.local_mass(M, K, M.L + 1)
        d = A.diagonal(); x = np.zeros(M.N); r = F.copy(); z = r / d; p = z.copy(); rz = r @ z
        for _ in range(30):
            Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        P2 = R.geometric_P(M, 2); uc = P2 @ np.linalg.solve((P2.T @ A @ P2).toarray(), P2.T @ F)
        levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
        Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
        G2 = R.pu_geometric(M, 2); OA = R.pu_operator_adapted(M, Phi3, Z)
        row = [f'{fam:8s} c={c:6.0e}']
        for nm, v in (('PCG30', x), ('u_c', uc)):
            Rv = F - A @ v; e = M.energy(K, M.full(uh - v)); rb = R.lift_residual(M, Rv)
            mv = R.ps_norm(M, K, R.flux_alg_mvy(M, K, A, rb, Rv)[0]) / e
            ora = R.ps_norm(M, K, R.flux_alg_oracle(M, K, rb, Mb)) / e
            jo = R.ps_norm(M, K, joint_flux(M, K, Rv, Mb)) / e
            g2 = R.ps_norm(M, K, R.flux_alg_pu_liftfree(M, K, A, Rv, *G2, Mloc_b=Mb)[0]) / e
            oa = R.ps_norm(M, K, R.flux_alg_pu_liftfree(M, K, A, Rv, *OA, Mloc_b=Mb)[0]) / e
            row.append(f'{nm}: MVY {mv:7.2f} oracle(MVY lift) {ora:5.2f} G2-LF {g2:6.2f} OA2-LF {oa:5.2f} joint {jo:5.2f}')
        print(' | '.join(row), flush=True)
