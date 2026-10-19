"""Which composite field makes a well-posed learning problem?  For bands vs short fibres (L3, k0 2, 1e2):
coarse-only error, sample-to-sample variation of u, correction HB shares, linear-ridge predictability."""
import numpy as np, sys; sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, fibre_composite
L = 3; n = 60
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
P = V.P[3]
def hb(e):
    out = {}; w = e
    for k in range(L, 0, -1):
        Pk = V.P[k].tocsr(); nc = Pk.shape[1]; c = w[:nc]; d = w[nc:] - (Pk @ c)[nc:]; out[k] = np.linalg.norm(d)**2; w = c
    out[0] = np.linalg.norm(w)**2; t = sum(out.values()); return np.array([out[k]/t for k in sorted(out)])
for name, gen in [('bands 1e2', lambda r: field_channels(H, 1e2, r)), ('fibres 1e2 vf .3', lambda r: fibre_composite(H, 1e2, r, vf=0.3)),
                  ('fibres 1e1 vf .3', lambda r: fibre_composite(H, 1e1, r, vf=0.3))]:
    rng = np.random.default_rng(7); Es, Us, UCs, en, sh = [], [], [], [], []
    for i in range(n):
        E = np.asarray(gen(rng)).ravel(); A = V.A_fine(E); u = spsolve(A, V.F)
        uc = P @ spsolve((P.T @ A @ P).tocsc(), P.T @ V.F); e = u - uc
        Es.append(np.log(E)); Us.append(u); UCs.append(uc); en.append(np.sqrt(e @ A @ e / (u @ A @ u)))
        if i < 10: sh.append(hb(e))
    Es, Us, UCs = map(np.stack, (Es, Us, UCs)); tr, te = slice(0, 45), slice(45, n)
    amp = np.linalg.norm(Us, axis=1)
    rel = lambda p: np.mean(np.linalg.norm(p - Us[te], axis=1) / np.linalg.norm(Us[te], axis=1))
    Xc = Es[tr] - Es[tr].mean(0); T = (Us - UCs)
    W = np.linalg.solve(Xc @ Xc.T + 1.0 * np.eye(45), T[tr] - T[tr].mean(0))
    ridge = UCs[te] + T[tr].mean(0) + (Es[te] - Es[tr].mean(0)) @ Xc.T @ W
    print(f"{name:17s}: |u| range {amp.min():.3f}-{amp.max():.3f} (x{amp.max()/amp.min():.1f}); coarse-only rel {rel(UCs[te]):.3f} energy {np.mean(en[45:]):.3f}; "
          f"+mean corr rel {rel(UCs[te] + T[tr].mean(0)):.3f}; +linear-ridge corr rel {rel(ridge):.3f}; "
          f"correction HB share L0..L3 {np.round(np.mean(sh, 0), 2)}")
