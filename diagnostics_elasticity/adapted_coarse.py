"""Does an operator-adapted coarse space fix the elasticity coarse solve?
One level (L3 -> level 2): Galerkin P vs adapted Phi = [I; Phat], Phat = local (rho) or exact -A_FF^-1 A_FC.
Reports coarse-only error of the Galerkin projection onto each space and the scale content of what is left."""
import numpy as np, scipy.sparse as sp, sys, time; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve, splu
from oa_lifting import NestedHierarchy, field_channels, _local_C
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, fibre_composite
L = 3
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
P = V.P[3].tocsr(); nc = P.shape[1]; Ek = V.E[3]
def hb(e):
    out = {}; w = e
    for k in range(L, 0, -1):
        Pk = V.P[k].tocsr(); m = Pk.shape[1]; c = w[:m]; d = w[m:] - (Pk @ c)[m:]; out[k] = np.linalg.norm(d)**2; w = c
    out[0] = np.linalg.norm(w)**2; t = sum(out.values()); return [round(out[k]/t, 2) for k in sorted(out)]
rng = np.random.default_rng(11)
for name, E in [('bands 1e2', field_channels(H, 1e2, rng)), ('fibres 1e2', fibre_composite(H, 1e2, rng, vf=0.3))]:
    E = np.asarray(E).ravel(); A = V.A_fine(E).tocsr(); u = spsolve(A.tocsc(), V.F)
    A_FF = (Ek.T @ A @ Ek).tocsc(); A_FC = (Ek.T @ A)[:, :nc].tocsc()
    rows = []
    for lab in ['galerkin', 'rho1', 'rho2', 'rho3', 'exact']:
        t0 = time.time()
        if lab == 'galerkin': Phi = P
        else:
            if lab == 'exact': Ph = sp.csr_matrix(-splu(A_FF).solve(A_FC.toarray()))
            else: Ph = _local_C(A_FF, A_FC, V.new_coords(3), V.level_coords(2), int(lab[-1]) * H.h[2])
            Phi = sp.vstack([sp.identity(nc, format='csr'), sp.csr_matrix(Ph)]).tocsr()
        uc = Phi @ spsolve((Phi.T @ A @ Phi).tocsc(), Phi.T @ V.F); tb = time.time() - t0
        e = u - uc
        rows.append(f"{lab:8s} build+solve {tb*1e3:6.0f} ms  rel {np.linalg.norm(e)/np.linalg.norm(u):.3f}  energy {np.sqrt(e@A@e/(u@A@u)):.3f}  HB share {hb(e)}")
    print(name); [print("   " + r) for r in rows]
