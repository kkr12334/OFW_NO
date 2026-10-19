"""size of the L3 discretisation error bound for exact solves (development operator 300): eta_total(u_h) / |||u_h|||."""
import os, sys, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
import rc_fem as R
H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9))
for fam, con in (('channels', 1e4), ('barriers', 1e-4)):
    rng = np.random.default_rng(1001); K = [make_field(H, 'channels', con, rng) for _ in range(301)][300]
    fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
    A = M.stiffness(K); uh = spsolve(A, M.load(fK)); Mb = R.local_mass(M, K, M.L + 1)
    S, _ = R.flux_disc(M, K, fK, np.zeros(len(M.fl.E)), M.full(uh), Mb)
    print(f'{fam} id 300 L3: eta_total(u_h) / |||u_h||| = {R.ps_norm(M, K, S, M.full(uh)) / M.energy(K, M.full(uh)):.4f}', flush=True)
