import numpy as np, sys; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
H = NestedHierarchy(3, coarse=perforated_square(n=9)); rng = np.random.default_rng(1)
Pk = (H.P[3]).tocsr()
def run(A_of, F, fields, P):
    U = np.stack([spsolve(A_of(a), F) for a in fields])
    UC = np.stack([P @ spsolve((P.T @ A_of(a) @ P).tocsc(), P.T @ F) for a in fields])
    return U, UC
def stats(name, U, UC, A_of, fields):
    T = U - UC; tr, te = range(32), range(32, 40); m = T[list(tr)].mean(0)
    rel = lambda p, i: np.linalg.norm(p - U[i]) / np.linalg.norm(U[i])
    en = lambda p, i: np.sqrt((p-U[i]) @ (A_of(fields[i]) @ (p-U[i])) / (U[i] @ (A_of(fields[i]) @ U[i])))
    print(f"{name}: coarse only {np.mean([rel(UC[i],i) for i in te]):.3f} / {np.mean([en(UC[i],i) for i in te]):.3f}   "
          f"coarse+MEAN correction {np.mean([rel(UC[i]+m,i) for i in te]):.3f} / {np.mean([en(UC[i]+m,i) for i in te]):.3f}  (rel-L2 / energy)")
fa = [np.asarray(make_field(H, 'channels', 1e4, rng)).ravel() for _ in range(40)]
U, UC = run(H.A_fine, H.F, fa, Pk); stats("Darcy channels 1e4   ", U, UC, H.A_fine, fa)
from scipy.sparse import csr_matrix
Hm = NestedHierarchy(3, coarse=perforated_square(n=9), bc='mixed',
     neumann=lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1)))
V = VectorHierarchy(Hm); Pv = V.P[3].tocsr()
from oa_lifting import field_channels
for c in (1e2, 1e3):
    fe = [np.asarray(field_channels(Hm, c, rng)).ravel() for _ in range(40)]
    U, UC = run(V.A_fine, V.F, fe, Pv); stats(f"Elasticity bands {c:.0e}", U, UC, V.A_fine, fe)
