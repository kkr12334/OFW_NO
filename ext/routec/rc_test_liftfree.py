"""development: lifting-free moment patches vs relaxed vs joint optimum (seed-1 development operator 300 only)."""
import os, sys, time, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis
import rc_fem as R
from rc_goal import joint_flux
H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9))
for fam, con in (('channels', 1e4), ('barriers', 1e-4)):
    rng = np.random.default_rng(1001); aa = [make_field(H, 'channels', con, rng) for _ in range(301)]; K = aa[300]
    fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
    A = M.stiffness(K); F = M.load(fK); uh = spsolve(A, F); Mb = R.local_mass(M, K, M.L + 1); nrm = M.energy(K, M.full(uh))
    P2 = R.geometric_P(M, 2); uc = P2 @ np.linalg.solve((P2.T @ A @ P2).toarray(), P2.T @ F)
    noise = np.random.default_rng(7).standard_normal(M.N); noise *= 0.1 * nrm / M.energy(K, M.full(noise))
    d = A.diagonal(); x = np.zeros(M.N); r = F.copy(); z = r / d; p = z.copy(); rz = r @ z
    for _ in range(30):
        Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
    levs, A0 = oa_hierarchy(H, H.A_fine(K), 3, 4)
    Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
    pus = {'G2': R.pu_geometric(M, 2), 'G1': R.pu_geometric(M, 1), 'OA2+Z': R.pu_operator_adapted(M, Phi3, Z)}
    print(f'\n{fam} id 300')
    for name, v in (('u_c', uc), ('30 Jacobi-PCG', x), ('u_h + 10% noise', uh + noise)):
        Rv = F - A @ v; e = M.energy(K, M.full(uh - v)); rb = R.lift_residual(M, Rv)
        row = [f'{name:16s} joint {R.ps_norm(M, K, joint_flux(M, K, Rv, Mb)) / e:5.3f}']
        for pn, (Th, inv, C) in pus.items():
            S2, _ = R.flux_alg_pu_relaxed(M, K, A, rb, Rv, Th, inv, C, Mloc_b=Mb)
            t0 = time.time(); S3, i3 = R.flux_alg_pu_liftfree(M, K, A, Rv, Th, inv, C, Mloc_b=Mb); tt = time.time() - t0
            mom = np.max(np.abs(M.Bb @ S3.div() - Rv)) / np.max(np.abs(Rv))
            row.append(f'{pn}: relaxed {R.ps_norm(M, K, S2) / e:5.3f} lift-free {R.ps_norm(M, K, S3) / e:5.3f} (mom {mom:.0e}, disc {i3["disconnected"]}, {tt:.0f}s)')
        print(' | '.join(row), flush=True)
