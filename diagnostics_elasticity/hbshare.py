import numpy as np, sys; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
L = 3; rng = np.random.default_rng(5)
def hb_shares(P, e, ncomp):
    # HB analysis: level-k detail = e_new - interp(e_coarse); returns rms of details per level and coarse
    out = {}; w = e
    for k in range(L, 0, -1):
        Pk = P[k].tocsr(); nc = Pk.shape[1]
        c = w[:nc]; d = w[nc:] - (Pk @ c)[nc:]
        out[k] = np.linalg.norm(d); w = c
    out[0] = np.linalg.norm(w); tot = sum(v**2 for v in out.values())
    return {k: v**2 / tot for k, v in sorted(out.items())}
def corr(A, F, P, k0):
    u = spsolve(A, F); Pk = None
    for k in range(k0 + 1, L + 1): Pk = P[k] if Pk is None else P[k] @ Pk
    Pk = Pk.tocsr(); uc = Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F); return u - uc, u
Hd = NestedHierarchy(L, coarse=perforated_square(n=9))
for c in (1, 1e4):
    a = np.asarray(field_channels(Hd, c, rng)).ravel(); e, u = corr(Hd.A_fine(a), Hd.F, Hd.P, 2)
    print(f"Darcy      contrast {c:g}: HB L2 share by level", {k: round(v, 3) for k, v in hb_shares(Hd.P, e, 1).items()})
He = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(He)
for c in (1, 1e2):
    E = np.asarray(field_channels(He, c, rng)).ravel(); e, u = corr(V.A_fine(E), V.F, V.P, 2)
    print(f"Elasticity contrast {c:g}: HB L2 share by level", {k: round(v, 3) for k, v in hb_shares(V.P, e, 2).items()})
