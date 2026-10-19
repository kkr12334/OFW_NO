"""development stress check at contrast 1e6 (seed-1 operator 300 geometry): OA2 lifting-free vs joint optimum."""
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
    c = 1e6; K = np.where(chan, c if fam == 'channels' else 1 / c, 1.0)
    fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
    A = M.stiffness(K); F = M.load(fK); uh = spsolve(A, F); Mb = R.local_mass(M, K, M.L + 1); nrm = M.energy(K, M.full(uh))
    levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
    Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
    OA = R.pu_operator_adapted(M, Phi3, Z); G2 = R.pu_geometric(M, 2)
    d = A.diagonal()
    def pcg(its):
        x = np.zeros(M.N); r = F.copy(); z = r / d; p = z.copy(); rz = r @ z
        for _ in range(its):
            Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        return x
    P2 = R.geometric_P(M, 2); uc = P2 @ np.linalg.solve((P2.T @ A @ P2).toarray(), P2.T @ F)
    n = np.random.default_rng(7).standard_normal(M.N); n *= 0.1 * nrm / M.energy(K, M.full(n))
    for nm, v in (('PCG10', pcg(10)), ('PCG30', pcg(30)), ('PCG100', pcg(100)), ('u_c', uc), ('noise 10%', uh + n)):
        Rv = F - A @ v; e = M.energy(K, M.full(uh - v))
        jo = R.ps_norm(M, K, joint_flux(M, K, Rv, Mb)) / e
        oa = R.ps_norm(M, K, R.flux_alg_pu_liftfree(M, K, A, Rv, *OA, Mloc_b=Mb)[0]) / e
        g2 = R.ps_norm(M, K, R.flux_alg_pu_liftfree(M, K, A, Rv, *G2, Mloc_b=Mb)[0]) / e
        print(f'{fam:8s} c=1e6 nk={Z.shape[1]} {nm:9s} rel err {e / nrm:.1e}: G2-LF {g2:7.2f}  OA2-LF {oa:6.3f}  joint {jo:6.3f}', flush=True)
