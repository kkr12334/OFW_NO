"""Constrained energy-minimising coarse space (pyamg, Olson-Schroder-Tuminaro 2011): start from the
geometric P (which reproduces the rigid-body modes exactly), minimise the energy of its columns within
a sparsity pattern while keeping P B_C = B_F.  Coarse-only error of the Galerkin projection onto it."""
import numpy as np, scipy.sparse as sp, sys, time; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from pyamg.aggregation.smooth import energy_prolongation_smoother
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, fibre_composite
L = 3
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
P = V.P[3].tocsr()
cen = H.coords[H.nodes[3]].mean(0)
def rbm_c(xy): n = len(xy); B = np.zeros((2*n, 3)); B[0::2,0]=1; B[1::2,1]=1; B[0::2,2]=-(xy[:,1]-cen[1]); B[1::2,2]=xy[:,0]-cen[0]; return B
Bc = rbm_c(H.level_coords(2)); Bf = P @ Bc          # modes as the geometric P reproduces them (exact away from the clamp)
print("true rigid modes vs P B_C, max diff (only at clamp-adjacent nodes):", np.abs(P @ Bc - rbm_c(H.level_coords(3))).max())
def hb(e):
    out = {}; w = e
    for k in range(L, 0, -1):
        Pk = V.P[k].tocsr(); m = Pk.shape[1]; c = w[:m]; d = w[m:] - (Pk @ c)[m:]; out[k] = np.linalg.norm(d)**2; w = c
    out[0] = np.linalg.norm(w)**2; t = sum(out.values()); return [round(float(out[k]/t), 2) for k in sorted(out)]
rng = np.random.default_rng(11)
for name, E in [('bands 1e2', field_channels(H, 1e2, rng)), ('fibres 1e2', fibre_composite(H, 1e2, rng, vf=0.3))]:
    E = np.asarray(E).ravel(); A = V.A_fine(E).tocsr(); u = spsolve(A.tocsc(), V.F)
    pat = A.copy(); pat.data[:] = 1.0
    print(name)
    for deg, its in [(0, 0), (1, 2), (1, 4), (1, 8), (2, 4), (2, 8), (2, 16), (3, 16)]:
        t0 = time.time()
        Pm = P if its == 0 else energy_prolongation_smoother(A, P, pat, Bc, Bf, (False, {}), krylov='cg', maxiter=its, tol=1e-12, degree=deg, weighting='local')
        tb = time.time() - t0; Pm = sp.csr_matrix(Pm)
        uc = Pm @ spsolve((Pm.T @ A @ Pm).tocsc(), Pm.T @ V.F); e = u - uc
        print(f"   degree {deg} its {its:2d}: build {tb*1e3:5.0f} ms, nnz/row {Pm.nnz/Pm.shape[0]:5.1f}, RBM err {np.abs(Pm@Bc-Bf).max():.1e}, "
              f"rel {np.linalg.norm(e)/np.linalg.norm(u):.3f} energy {np.sqrt(e@A@e/(u@A@u)):.3f}  HB share {hb(e)}")
