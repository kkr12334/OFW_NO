"""diagnostic: how much of the algebraic-estimator overestimation is due to the residual lifting?
joint oracle = min ||K^{-1/2} sigma|| over RT0(T_b) subject only to the discrete moment conditions (div sigma, psi_a) = R_a
(the lifting is optimised jointly).  Development operators only."""
import os, sys, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
import rc_fem as R

def joint_oracle(M, K, Rv, Mb):
    fl = M.fl; nb, nE = len(fl.E), fl.n_edge
    I = np.repeat(fl.edge, 3, 1).ravel(); J = np.tile(fl.edge, (1, 3)).ravel()
    Mg = sp.csr_matrix((Mb.ravel(), (I, J)), shape=(nE, nE))
    D = sp.csr_matrix((fl.div_int.ravel(), (np.repeat(np.arange(nb), 3), fl.edge.ravel())), shape=(nb, nE))
    G = (M.Bb @ sp.diags(1 / fl.area) @ D).tocsr()            # moments of div sigma against the unknown hats
    sol = spsolve(sp.bmat([[Mg, -G.T], [G, None]]).tocsc(), np.concatenate([np.zeros(nE), Rv]))
    S = R.Field(M).add(M.L + 1, np.arange(nE), sol[:nE])
    return S

if __name__ == '__main__':
    H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9))
    for fam, con in (('channels', 1e4), ('barriers', 1e-4)):
        rng = np.random.default_rng(1001); aa = [make_field(H, 'channels', con, rng) for _ in range(301)]; K = aa[300]
        fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
        A = M.stiffness(K); F = M.load(fK); uh = spsolve(A, F); Mb = R.local_mass(M, K, M.L + 1)
        nrm = M.energy(K, M.full(uh))
        P2 = R.geometric_P(M, 2); uc = P2 @ np.linalg.solve((P2.T @ A @ P2).toarray(), P2.T @ F)
        noise = np.random.default_rng(7).standard_normal(M.N); noise *= 0.1 * nrm / M.energy(K, M.full(noise))
        # a few Jacobi-PCG steps from zero: smooth-ish algebraic error typical of iterative solvers
        d = A.diagonal(); x = np.zeros(M.N); r = F.copy(); z = r / d; p = z.copy(); rz = r @ z
        for _ in range(30):
            Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        for name, v in (('u_c', uc), ('30 Jacobi-PCG steps', x), ('u_h + 10% noise', uh + noise)):
            Rv = F - A @ v; e = M.energy(K, M.full(uh - v)); rb = R.lift_residual(M, Rv)
            so = R.ps_norm(M, K, R.flux_alg_oracle(M, K, rb, Mb)); sj = joint_oracle(M, K, Rv, Mb)
            chk = np.max(np.abs(M.Bb @ sj.div() - Rv)) / np.max(np.abs(Rv))
            print(f'{fam:8s} {name:22s} rel err {e / nrm:.2e}:  oracle (MVY lifting) eff {so / e:7.3f}   joint oracle eff {R.ps_norm(M, K, sj) / e:7.3f}  (moment check {chk:.1e})')
