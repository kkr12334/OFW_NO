import numpy as np, sys; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, unit_element_matrices
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
L = 3; rng = np.random.default_rng(5)
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
# coarse-mesh hole vertices = pore nodes present at level 0
lvl0 = set(H.mesh.level_nodes[0]); pore = set(H.neumann_nodes.tolist())
corner = np.array(sorted(pore & lvl0))
print(f"hole polygons: {len(corner)} coarse-level vertices (re-entrant corners of the solid)")
Pk = V.P[3] @ V.P[2] if False else V.P[3]
def elem_energy_share(E, which='elast'):
    A = V.A_fine(E); u = spsolve(A, V.F)
    P = V.P[3].tocsr(); uc = P @ spsolve((P.T @ A @ P).tocsc(), P.T @ V.F)
    e = u - uc
    full = np.zeros(2 * len(H.coords)); full[V.nodes[L]] = e
    ge = full[V.edofs]; K1 = V.K1
    ee = np.asarray(E) * np.einsum('ei,eij,ej->e', ge, K1, ge)
    near = np.any(np.isin(H.elem, corner), axis=1)
    return near.mean(), ee[near].sum() / ee.sum()
for c in (1, 1e2):
    E = np.asarray(field_channels(H, c, rng)).ravel()
    f, s = elem_energy_share(E)
    print(f"elasticity contrast {c:g}: elements touching hole corners = {f*100:.1f}% of elements, carry {s*100:.1f}% of the correction energy")
