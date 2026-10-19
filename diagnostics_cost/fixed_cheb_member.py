"""11 Oct 2026: is a Chebyshev-smoothed cycle with a FIXED spectral interval contained in the architecture, and how strong is
it?  The smoother steps carry step-dependent linear maps L_t, so nu Richardson steps x <- x + w_t D^-1 (b - A x) with
w_t = 1 / (roots of the degree-nu Chebyshev residual polynomial on [lam/10, lam]) are representable with the SAME weights
on every level (Proposition containment with w_t in place of omega).  Product form == three-term recurrence in exact
arithmetic (checked here).  Compares, on the test samples: adaptive Chebyshev reference (per-level power-iteration lam),
fixed-interval Chebyshev with lam = 3 on all levels, fixed lam* = max of the per-level estimates over the TRAINING samples,
and the Jacobi member.  usage: python diagnostics_cost/fixed_cheb_member.py tag L k0 seed contrast load"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
tag, L, k0, DS, con, load = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
nu, q = 4, 10
S = setup(L, k0, DS, contrast=con, load=load)
def lam_est(Ak, its=30):
    dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam
def weights(lam):
    lo, hi = lam / q, lam; th, de = (hi + lo) / 2, (hi - lo) / 2
    return [1 / (th - de * np.cos((2 * i - 1) * np.pi / (2 * nu))) for i in range(1, nu + 1)]
def smooth(Ak, b, W):
    x = np.zeros_like(b); dinv = 1 / Ak.diagonal()
    for w in W: x = x + w * dinv * (b - Ak @ x)
    return x
def gal(A, Wm, b):
    keep = np.linalg.norm(Wm, axis=0) > 0; Wm = Wm[:, keep]
    return Wm @ np.linalg.solve(Wm.T @ (A @ Wm), Wm.T @ b) if Wm.shape[1] else 0 * b
def member(levs, A0, b, Z, Ws):                       # Ws: per-level step weights (same list on all levels = contained)
    A = levs[0][0]; x0 = gal(A, Z, b); bb = b - A @ x0
    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]; x = smooth(Ak, rhs, Ws[l]); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else np.linalg.solve(A0.toarray(), rc)
        xc = np.concatenate([ec, X @ ec]); xp = smooth(Ak, rhs - Ak @ (x + xc), Ws[l])
        return (x, xc, xp) if parts else x + xc + xp
    return x0 + gal(A, np.column_stack(list(cyc(0, bb, True)) + [Z]), bb)
en = lambda A, x, u: float(np.sqrt(((x - u) @ (A @ (x - u))) / (u @ (A @ u))))
lams = []
for i in S['tr'][:60]:
    levs, _ = oa_hierarchy(S['H'], S['A'][i], S['nl'], 4); lams.append([lam_est(Ak) for Ak, _ in levs])
lams = np.array(lams); lstar = float(lams.max())
print(f"{tag}: per-level power-iteration lam (x1.05) on 60 training samples: max per level {np.round(lams.max(0), 3)}, "
      f"min per level {np.round(lams.min(0), 3)}; lam* = {lstar:.3f}")
E = {k: [] for k in ('adaptive', 'adaptive_product', 'fixed3', 'fixedstar', 'jacobi')}
for i in S['te']:
    A = S['A'][i]; u = S['U'][i]; uc = S['UC'][i]; r = S['Fs'][i] - A @ uc
    levs, A0 = oa_hierarchy(S['H'], A, S['nl'], 4); Z = near_kernel_basis(levs[0][0]); nlv = len(levs)
    E['adaptive'].append(en(A, uc + classical_vcycle_proj(levs, A0, r, Z=Z, nu=4), u))
    lm = [3.0] + [lam_est(Ak) for Ak, _ in levs[1:]]
    E['adaptive_product'].append(en(A, uc + member(levs, A0, r, Z, [weights(x) for x in lm]), u))
    E['fixed3'].append(en(A, uc + member(levs, A0, r, Z, [weights(3.0)] * nlv), u))
    E['fixedstar'].append(en(A, uc + member(levs, A0, r, Z, [weights(lstar)] * nlv), u))
    E['jacobi'].append(en(A, uc + member(levs, A0, r, Z, [[2 / 3] * nu] * nlv), u))
E = {k: np.array(v) for k, v in E.items()}
G = lambda a, b: float(np.exp(np.mean(np.log(a / b))))
print(f"  product form vs recurrence (adaptive): max rel diff {np.max(np.abs(E['adaptive_product'] / E['adaptive'] - 1)):.1e}")
for k in ('fixed3', 'fixedstar', 'jacobi'):
    print(f"  {k}: mean {E[k].mean():.4e}; G({k}/adaptive) {G(E[k], E['adaptive']):.3f}; adaptive better on {np.mean(E['adaptive'] < E[k]):.2f}")
json.dump(dict(tag=tag, lam_star=lstar, lam_levels_max=lams.max(0).tolist(), lam_levels_min=lams.min(0).tolist(),
               sample_ids=[int(i) for i in S['te']], **{k: v.tolist() for k, v in E.items()}),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'fixed_cheb_{tag}.json'), 'w'))
