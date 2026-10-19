import numpy as np, scipy.sparse as sp, sys, time; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels, _local_C
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, fibre_composite
from rbm_constraint import rbm, constrain_rows
L = 3
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
P = V.P[3].tocsr(); nc = P.shape[1]; Ek = V.E[3]
cen = H.coords[H.nodes[3]].mean(0)
B_C = rbm(H.level_coords(2) - cen + H.level_coords(2).mean(0) - H.level_coords(2).mean(0)); B_F = rbm(H.new_coords(3))
# same rotation centre for both sets
def rbm_c(xy): n = len(xy); B = np.zeros((2*n, 3)); B[0::2,0]=1; B[1::2,1]=1; B[0::2,2]=-(xy[:,1]-cen[1]); B[1::2,2]=xy[:,0]-cen[0]; return B
B_C, B_F = rbm_c(H.level_coords(2)), rbm_c(H.new_coords(3))
# rows next to the clamp: fine dofs whose node touches a clamped node
clamped = set(H.boundary.tolist()); near = set()
for e in H.elem:
    if any(v in clamped for v in e): near.update(e.tolist())
newn = H.new[3]; far_rows = [2*i + c for i, g in enumerate(newn) if g not in near for c in (0, 1)]
def hb(e):
    out = {}; w = e
    for k in range(L, 0, -1):
        Pk = V.P[k].tocsr(); m = Pk.shape[1]; c = w[:m]; d = w[m:] - (Pk @ c)[m:]; out[k] = np.linalg.norm(d)**2; w = c
    out[0] = np.linalg.norm(w)**2; t = sum(out.values()); return [round(float(out[k]/t), 2) for k in sorted(out)]
rng = np.random.default_rng(11)
for name, E in [('bands 1e2', field_channels(H, 1e2, rng)), ('fibres 1e2', fibre_composite(H, 1e2, rng, vf=0.3))]:
    E = np.asarray(E).ravel(); A = V.A_fine(E).tocsr(); u = spsolve(A.tocsc(), V.F)
    A_FF = (Ek.T @ A @ Ek).tocsc(); A_FC = (Ek.T @ A)[:, :nc].tocsc()
    print(name)
    for rho in (1, 2, 3):
        t0 = time.time(); Ph = _local_C(A_FF, A_FC, V.new_coords(3), V.level_coords(2), rho * H.h[2]); tl = time.time() - t0
        for lab, Pm in [('plain', Ph), ('RBM all rows', constrain_rows(Ph, B_F, B_C)), ('RBM away from clamp', constrain_rows(Ph, B_F, B_C, rows=far_rows))]:
            t1 = time.time()
            Phi = sp.vstack([sp.identity(nc, format='csr'), sp.csr_matrix(Pm)]).tocsr()
            uc = Phi @ spsolve((Phi.T @ A @ Phi).tocsc(), Phi.T @ V.F); e = u - uc
            print(f"   rho {rho} {lab:20s} local build {tl*1e3:5.0f} ms  rel {np.linalg.norm(e)/np.linalg.norm(u):.3f}  energy {np.sqrt(e@A@e/(u@A@u)):.3f}  HB share {hb(e)}")
