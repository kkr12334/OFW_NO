"""Self-test for oa_lifting.py -- run this first (under a minute).

Checks, on the project's triangle hierarchy with a connected-channel field:
  1. T^T A T is block-diagonal                       (Proposition 2)
  2. A^-1 = T D^-1 T^T                               (Theorem 1)
  3. det T = 1 and T^-1 needs no solve               (Proposition 1)
  4. the matrix-free dual analysis equals T^T
  5. every transform mode gives the exact solution through PCG
Asserts every number it prints.
"""
import sys
import numpy as np
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '.')
from oa_lifting import NestedHierarchy, OATransform, make_field, pcg_kappa, rel_l2

rng = np.random.default_rng(0)
ok = True
for L in (3, 4):
    H = NestedHierarchy(L)
    a = make_field(H, 'channels', 1e4, rng)
    A = H.A_fine(a)
    T = OATransform(H, A, 'exact')
    n = A.shape[0]
    sizes = [T.size(k) for k in T.levels]; offs = np.cumsum([0] + sizes)
    Tm = np.zeros((n, n))
    for j in range(n):
        kk = int(np.searchsorted(offs, j, side='right') - 1)
        co = {k: np.zeros(sizes[k]) for k in T.levels}; co[kk][j - offs[kk]] = 1
        Tm[:, j] = T.synth(co)
    Ah = Tm.T @ A.toarray() @ Tm
    off = Ah.copy()
    for k in T.levels: off[offs[k]:offs[k + 1], offs[k]:offs[k + 1]] = 0
    u = spsolve(A, H.F)
    v = rng.standard_normal(n)
    checks = {
        'block-diagonal': np.linalg.norm(off) / np.linalg.norm(Ah),
        'Green identity': rel_l2(u, T.precond(H.F)),
        '|det T - 1|': abs(np.linalg.det(Tm) - 1),
        'analysis o synth': np.abs(T.synth(T.analysis(v)) - v).max(),
        'dual == T^T': np.abs(Tm.T @ v - np.concatenate([T.dual(v)[k] for k in T.levels])).max(),
    }
    for name, val in checks.items():
        good = val < 1e-9
        ok &= good
        print(f"L={L}  {name:<18} {val:.2e}  {'ok' if good else 'FAIL'}")
    for mode, kw in [('local', dict(rho=2)), ('frozen', dict(A_ref=H.A_fine(np.ones(H.n_elem)))), ('hb', {})]:
        Tl = OATransform(H, A, mode, **kw)
        x, it, kap = pcg_kappa(A, H.F, Tl.precond)
        good = rel_l2(u, x) < 1e-8
        ok &= good
        print(f"L={L}  PCG {mode:<8} {it:4d} it  kappa {kap:9.1f}  err {rel_l2(u, x):.1e}  {'ok' if good else 'FAIL'}")
print("\nALL OK" if ok else "\nSOME CHECKS FAILED")
sys.exit(0 if ok else 1)
