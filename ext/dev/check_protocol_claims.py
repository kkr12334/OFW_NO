"""Numerical checks of the mathematical statements used in STAGE1_PROTOCOL (16 Oct 2026). Real level matrices from
development data (barriers, seed 1, test id 300 -- an island sample -- and channels id 300)."""
import os, sys, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
rng = np.random.default_rng(0)

def cheb(Ak, rhs, nu=4, q=10, lmax=3.0):          # verbatim from classical_vcycle_proj, from zero
    dinv = 1 / Ak.diagonal(); lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
    x = np.zeros_like(rhs); s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
    for _ in range(nu - 1):
        r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
    return x
def pcg(Ak, b, its=4):
    dinv = 1 / Ak.diagonal(); x = np.zeros_like(b); r = b.copy(); z = dinv * r; p = z.copy(); rz = r @ z
    for _ in range(its):
        if rz == 0: break
        Ap = Ak @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = dinv * r; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
    return x
def sd(Ak, b, its=4):
    dinv = 1 / Ak.diagonal(); x = np.zeros_like(b); r = b.copy()
    for _ in range(its):
        p = dinv * r; Ap = Ak @ p; den = p @ Ap
        if den == 0: break
        al = (p @ r) / den; x += al * p; r -= al * Ap
    return x
def mr(Ak, b, its=4):
    dinv = 1 / Ak.diagonal(); x = np.zeros_like(b); r = b.copy()
    for _ in range(its):
        p = dinv * r; Ap = Ak @ p; den = Ap @ (dinv * Ap)
        if den == 0: break
        al = (Ap @ (dinv * r)) / den; x += al * p; r -= al * Ap
    return x
def k4_min(Ak, b):
    """exact minimiser of ||A^-1 b - x||_A over K4 = span{D^-1 b, B D^-1 b, B^2 D^-1 b, B^3 D^-1 b} (orthonormalised basis)."""
    dinv = 1 / Ak.diagonal(); v = dinv * b; V = [v]
    for _ in range(3): V.append(dinv * (Ak @ V[-1]))
    Q, _ = np.linalg.qr(np.column_stack(V)); G = Q.T @ (Ak @ Q); return Q @ np.linalg.solve(G, Q.T @ b), Q
def eA(Ak, b, x):
    u = sp.linalg.spsolve(Ak.tocsc(), b); e = u - x; return np.sqrt(e @ (Ak @ e))

out = []
for fam, kw in (('barriers', dict(contrast=1e-4, load='no-low')), ('channels', {})):
    S = setup(3, 2, 1, **kw); i = 300; A = S['A'][i]; levs, A0 = oa_hierarchy(S['H'], A, S['nl'], 4)
    r = S['Fs'][i] - A @ S['UC'][i]
    for l, (Ak, _) in enumerate(levs):
        b = r if l == 0 else rng.standard_normal(Ak.shape[0])
        lmax = 3.0 if l == 0 else 2.0
        xk, Q = k4_min(Ak, b); xc, xp, xs, xm = cheb(Ak, b, lmax=lmax), pcg(Ak, b), sd(Ak, b), mr(Ak, b)
        # (a) PCG(4) from zero = the K4 energy minimiser; (b) Chebyshev(4) lies in K4
        Gq = Q.T @ (Ak @ Q)
        def dist(x):                                   # A-distance from K4, relative
            y = Q @ np.linalg.solve(Gq, Q.T @ (Ak @ x)); d = x - y; return np.sqrt(d @ (Ak @ d)) / np.sqrt(x @ (Ak @ x))
        out.append((fam, l, eA(Ak, b, xp) / eA(Ak, b, xk) - 1, dist(xc), dist(xp), dist(xs), dist(xm),
                    eA(Ak, b, xp) <= eA(Ak, b, xc) * (1 + 1e-12), eA(Ak, b, xp) <= min(eA(Ak, b, xs), eA(Ak, b, xm)) * (1 + 1e-12)))
    # (e) oddness / homogeneity of adaptive smoothers; (f) non-additivity
    Ak = levs[0][0]; b1, b2 = rng.standard_normal(Ak.shape[0]), rng.standard_normal(Ak.shape[0])
    for name, f in (('pcg', pcg), ('sd', sd), ('mr', mr), ('cheb', cheb)):
        odd = np.abs(f(Ak, -b1) + f(Ak, b1)).max() / np.abs(f(Ak, b1)).max()
        hom = np.abs(f(Ak, -2.5 * b1) - (-2.5) * f(Ak, b1)).max() / np.abs(f(Ak, b1)).max()
        add = np.linalg.norm(f(Ak, b1 + b2) - f(Ak, b1) - f(Ak, b2)) / np.linalg.norm(f(Ak, b1 + b2))
        print(f"{fam} {name:<4}: oddness defect {odd:.1e}; homogeneity(lambda=-2.5) defect {hom:.1e}; additivity defect {add:.2e}")
    # (g) Galerkin output never increases energy relative to x0, for ANY directions (random W)
    Z = near_kernel_basis(A); worst = -np.inf
    for t in range(200):
        W = np.column_stack([rng.standard_normal(A.shape[0]) for _ in range(3)] + ([Z] if Z.shape[1] else []))
        u = sp.linalg.spsolve(A.tocsc(), r)
        G = W.T @ (A @ W); y = np.linalg.lstsq(G, W.T @ r, rcond=None)[0]; c = W @ y
        e0 = np.sqrt(u @ (A @ u)); e1 = np.sqrt((u - c) @ (A @ (u - c))); worst = max(worst, e1 / e0 - 1)
    print(f"{fam}: Galerkin output with random directions: max (||e-c||_A/||e||_A - 1) over 200 trials = {worst:.2e} (<= 0 expected)")
print("\nfamily level  PCG/K4min-1  dist_A(K4): cheb pcg sd mr   PCG<=Cheb  PCG<=min(SD,MR)")
for o in out: print(f"{o[0]:<9} {o[1]}   {o[2]: .1e}   {o[3]:.1e} {o[4]:.1e} {o[5]:.1e} {o[6]:.1e}   {o[7]}  {o[8]}")
