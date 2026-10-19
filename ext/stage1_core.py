"""Stage 1 core (16 Oct 2026; protocol v1 + amendment 1): operator-adapted V-cycle with pluggable smoothers, the common
classical deflation/output wrapper, spectral-bound policies and a rank-aware Krylov basis.  numpy / scipy, float64.

Lives in ext/, so the legacy top-level code hash f8653612eb84 is unchanged.  Uses oa_vcycle.oa_hierarchy and
near_kernel_basis unchanged."""
import os, sys, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
if ROOT not in sys.path: sys.path.insert(0, ROOT)
from oa_vcycle import oa_hierarchy, near_kernel_basis   # noqa: E402


# ----------------------------------------------------------------------------------------------- spectral bounds
def lam_power(Ak, its=30):
    """legacy CHEB estimator (classical_vcycle_proj): 1.05 x 30-step power iteration on D^-1 A (heuristic)."""
    dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam


def lam_lanczos(Ak, its=10, boost=1.1):
    """CHEB-T estimator (Adams et al. 2003): boost x largest Ritz value of 10 Lanczos steps on H = D^-1/2 A D^-1/2
    (same spectrum as D^-1 A).  Heuristic (Ritz values are lower bounds of lambda_max); not a certificate."""
    s = 1 / np.sqrt(Ak.diagonal()); H = sp.diags(s) @ Ak @ sp.diags(s)
    n = Ak.shape[0]; q = np.random.default_rng(0).standard_normal(n); q /= np.linalg.norm(q)
    Q = [q]; al, be = [], []; qp = np.zeros(n); b = 0.0
    for j in range(min(its, n)):
        w = H @ Q[-1] - b * qp; a = Q[-1] @ w; w -= a * Q[-1]
        for qq in Q: w -= (qq @ w) * qq                          # full reorthogonalisation
        al.append(a); b = np.linalg.norm(w)
        if b < 1e-12: break
        be.append(b); qp = Q[-1]; Q.append(w / b)
    T = np.diag(al) + np.diag(be[:len(al) - 1], 1) + np.diag(be[:len(al) - 1], -1)
    return boost * float(np.linalg.eigvalsh(T).max())


def lam_gershgorin(Ak):
    """GUARANTEED upper bound of lambda_max(D^-1 A): max absolute row sum of H = D^-1/2 A D^-1/2 (H symmetric, so its
    spectral radius is <= its infinity norm)."""
    s = 1 / np.sqrt(Ak.diagonal()); H = (sp.diags(s) @ Ak @ sp.diags(s)).tocsr()
    return float(np.abs(H).sum(1).max())


# ----------------------------------------------------------------------------------------------- smoothers
# a smoother is f(Ak, dinv, rhs, l) -> x, started from zero; every one below performs 4 update stages.
def make_cheb(lams, q=10, nu=4):
    """Chebyshev recurrence of classical_vcycle_proj (verbatim arithmetic), interval [lams[l]/q, lams[l]]."""
    def f(Ak, dinv, rhs, l, cnt=None):
        lo, hi = lams[l] / q, lams[l]; th, de = (hi + lo) / 2, (hi - lo) / 2
        x = np.zeros_like(rhs); s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(nu - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        if cnt is not None: cnt['matvec'][l] += nu; cnt['dinv'][l] += nu
        return x
    return f


def pcg4(Ak, dinv, rhs, l, cnt=None, its=4):
    x = np.zeros_like(rhs); r = rhs.copy(); z = dinv * r; rz = r @ z; p = z.copy(); st = 'ok'
    for _ in range(its):
        if rz == 0: st = 'exact'; break
        Ap = Ak @ p; pAp = p @ Ap
        if not pAp > 0: st = 'curvature'; break
        al = rz / pAp; x += al * p; r -= al * Ap; z = dinv * r; rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        if cnt is not None: cnt['matvec'][l] += 1; cnt['dinv'][l] += 1; cnt['dot'][l] += 2
    if cnt is not None and st != 'ok': cnt['status'][st] = cnt['status'].get(st, 0) + 1
    return x


def sd4(Ak, dinv, rhs, l, cnt=None, its=4):
    x = np.zeros_like(rhs); r = rhs.copy()
    for _ in range(its):
        p = dinv * r; Ap = Ak @ p; den = p @ Ap
        if not den > 0:
            if cnt is not None:
                k = 'exact' if den == 0 else 'curvature'; cnt['status'][k] = cnt['status'].get(k, 0) + 1
            break
        al = (p @ r) / den; x += al * p; r -= al * Ap
        if cnt is not None: cnt['matvec'][l] += 1; cnt['dinv'][l] += 1; cnt['dot'][l] += 2
    return x


def mr4(Ak, dinv, rhs, l, cnt=None, its=4):
    x = np.zeros_like(rhs); r = rhs.copy()
    for _ in range(its):
        p = dinv * r; Ap = Ak @ p; den = Ap @ (dinv * Ap)
        if not den > 0:
            if cnt is not None: cnt['status']['exact'] = cnt['status'].get('exact', 0) + 1
            break
        al = (Ap @ (dinv * r)) / den; x += al * p; r -= al * Ap
        if cnt is not None: cnt['matvec'][l] += 1; cnt['dinv'][l] += 2; cnt['dot'][l] += 2
    return x


def make_poly(coef, scales):
    """x = p(B) D^-1 b, p(t) = (1 / s_l) sum_j coef[l][j] T_j(2 t / s_l - 1), j = 0..3: Chebyshev basis on [0, s_l] with the
    1/s normalisation, so that a coefficient vector means the same scaled polynomial at every scale (a Chebyshev
    smoother on [s/q, s] has scale-independent coefficients).  coef: (n_levels, 4) or (4,) (shared); scales: per level."""
    coef = np.atleast_2d(coef)
    def f(Ak, dinv, rhs, l, cnt=None):
        c = coef[l if coef.shape[0] > 1 else 0]; s = scales[l]
        v = dinv * rhs                                           # T_0 v
        Bv = lambda y: dinv * (Ak @ y)
        t0 = v; t1 = 2 / s * Bv(v) - v                           # T_1(2B/s - I) v
        x = (c[0] * t0 + c[1] * t1) / s
        for j in range(2, 4):
            t2 = 2 * (2 / s * Bv(t1) - t1) - t0; x = x + c[j] * t2 / s; t0, t1 = t1, t2
        if cnt is not None: cnt['matvec'][l] += 3; cnt['dinv'][l] += 4
        return x
    return f


# ----------------------------------------------------------------------------------------------- cycle and wrapper
class Hier:
    def __init__(self, H, A, n_levels, m=4):
        self.levs, self.A0 = oa_hierarchy(H, A, n_levels, m); self.lu = splu(self.A0.tocsc())
        self.dinv = [1 / Ak.diagonal() for Ak, _ in self.levs]; self.A = self.levs[0][0]; self.nl = len(self.levs)

    def new_counter(self):
        return dict(matvec=[0] * self.nl, dinv=[0] * self.nl, dot=[0] * self.nl, transfer=[0] * self.nl, coarsest=0,
                    status={})

    def cycle(self, l, rhs, pre, post, parts=False, cnt=None):
        """V-cycle at level l (0 = finest).  pre/post: smoother functions, applied from zero (pre on rhs, post on the
        updated residual), as in classical_vcycle_proj and OAVCycle._cycle."""
        Ak, X = self.levs[l]; nc = X.shape[1]; di = self.dinv[l]
        x = pre(Ak, di, rhs, l, cnt); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        if l + 1 < self.nl: ec = self.cycle(l + 1, rc, pre, post, cnt=cnt)
        else:
            ec = self.lu.solve(rc)
            if cnt is not None: cnt['coarsest'] += 1
        xc = np.concatenate([ec, X @ ec])
        xp = post(Ak, di, rhs - Ak @ (x + xc), l, cnt)
        if cnt is not None: cnt['matvec'][l] += 2; cnt['transfer'][l] += 2
        return (x, xc, xp) if parts else x + xc + xp


def galerkin(A, W, b, stats=None):
    """Common classical output (amendment 1, A1): mirrors OAVCycle._galerkin in float64 -- symmetrised Gram matrix,
    Jacobi scaling with dead-column masking, 1e-10 ridge, then the energy-acceptance backstop 2 d^T b - d^T A d >= 0
    (else the exact line-search step d * max(d^T b, 0) / d^T A d).  Records backstop / rejection counts in stats."""
    if W.shape[1] == 0: return np.zeros_like(b)
    AW = A @ W; G = W.T @ AW; G = 0.5 * (G + G.T); g = W.T @ b
    dg = np.diag(G); live = dg > 1e-30 * max(dg.max(), 1e-300)
    sc = np.where(live, 1 / np.sqrt(np.where(live, dg, 1)), 0)
    Gs = sc[:, None] * G * sc[None, :] + np.diag((~live) + 1e-10)
    d = W @ (sc * np.linalg.solve(Gs, sc * g))
    db = d @ b; dAd = d @ (A @ d)
    if 2 * db - dAd < 0:
        alpha = max(db, 0) / dAd if dAd > 0 else 0.0
        if stats is not None:
            stats['backstop'] = stats.get('backstop', 0) + 1
            if db <= 0: stats['rejected'] = stats.get('rejected', 0) + 1
        d = alpha * d
    return d


def one_pass(hier, r, Z, pre, post, project=True, cnt=None, stats=None):
    """deflation (x0 = common galerkin on Z) + one cycle on b = r - A x0 + Galerkin output on [x, xc, xp, Z]."""
    A = hier.A
    x0 = galerkin(A, Z, r, stats) if (Z is not None and Z.shape[1]) else np.zeros_like(r)
    bb = r - A @ x0
    parts = hier.cycle(0, bb, pre, post, parts=True, cnt=cnt)
    if not project: return x0 + sum(parts)
    W = np.column_stack(list(parts) + ([Z] if (Z is not None and Z.shape[1]) else []))
    return x0 + galerkin(A, W, bb, stats)


# ----------------------------------------------------------------------------------------------- Krylov diagnostic
def krylov_basis(Ak, b, dim=4, rtol=1e-10):
    """Rank-aware A-orthonormal basis of K(b) = span{D^-1 b, B D^-1 b, ...} (B = D^-1 A), at most `dim` vectors.
    Arnoldi in the A-inner product with two passes of reorthogonalisation; a new vector is dropped (and the iteration
    stops: happy breakdown) when its A-norm after orthogonalisation is <= rtol x its A-norm before.  Returns
    (Q with Q^T A Q = I, retained rank).  b = 0 gives rank 0."""
    di = 1 / Ak.diagonal(); v = di * b; Q = []
    for _ in range(dim):
        nv0 = np.sqrt(max(v @ (Ak @ v), 0.0))
        if nv0 == 0: break
        w = v.copy()
        for _p in range(2):
            for q in Q: w -= (q @ (Ak @ w)) * q
        nw = np.sqrt(max(w @ (Ak @ w), 0.0))
        if nw <= rtol * nv0: break
        Q.append(w / nw); v = di * (Ak @ Q[-1])
    Qm = np.column_stack(Q) if Q else np.zeros((len(b), 0))
    return Qm, Qm.shape[1]


def k4_diagnostic(Ak, b, x_nn, x_pcg=None, rtol=1e-10):
    """D1 (amendment 1, A2): delta = ||x_nn - Pi x_nn||_A / ||x_nn||_A (Pi: A-orthogonal projection onto K), the K energy
    minimiser x_K = Q Q^T b, rho = E(x_nn) / E(x_K) with E(x) = ||A^-1 b - x||_A, and the PCG cross-check.  Zero cases
    are reported as flags, never divided."""
    from scipy.sparse.linalg import spsolve
    Q, rk = krylov_basis(Ak, b, 4, rtol); u = spsolve(Ak.tocsc(), b)
    xK = Q @ (Q.T @ b) if rk else np.zeros_like(b)
    E = lambda x: float(np.sqrt(max((u - x) @ (Ak @ (u - x)), 0.0)))
    nx = float(np.sqrt(max(x_nn @ (Ak @ x_nn), 0.0)))
    out = dict(rank=rk, E_nn=E(x_nn), E_K=E(xK), norm_nn=nx)
    if nx == 0: out['delta'] = None; out['flag'] = 'zero output'
    else:
        px = Q @ (Q.T @ (Ak @ x_nn)) if rk else np.zeros_like(x_nn); dd = x_nn - px
        out['delta'] = float(np.sqrt(max(dd @ (Ak @ dd), 0.0))) / nx
    out['rho'] = None if out['E_K'] == 0 else out['E_nn'] / out['E_K']
    if out['E_K'] == 0: out['flag'] = 'exact convergence in K'
    if x_pcg is not None: out['pcg_vs_K'] = None if out['E_K'] == 0 else E(x_pcg) / out['E_K'] - 1
    return out


# ----------------------------------------------------------------------------------------------- K2-cycle, FCG, PCG-V
def fcg(Aop, b, prec, iterations, x0=None, record=None):
    """Flexible CG, Notay & Vassilevski Alg. 3.1 with m_0 = 0, m_i = 1 (orthogonalise against the previous direction).
    Aop: callable or matrix; prec: callable r -> w (may be nonlinear).  record(i, x) is called after every iteration."""
    mv = Aop if callable(Aop) else (lambda v: Aop @ v)
    x = np.zeros_like(b) if x0 is None else x0.copy(); r = b - mv(x) if x0 is not None else b.copy(); dprev = Adprev = None
    for i in range(iterations):
        w = prec(r); d = w.copy()
        if dprev is not None: d -= (w @ Adprev) / (dprev @ Adprev) * dprev
        Ad = mv(d); dAd = d @ Ad
        if not dAd > 0: break
        a = (d @ r) / dAd; x = x + a * d; r = r - a * Ad; dprev, Adprev = d, Ad
        if record is not None: record(i, x)
    return x


def kcycle(hier, l, rhs, pre, post, gamma=2, parts=False, cnt=None):
    """K2-cycle (amendment 1, A3.6): smoothers at level l; the level-(l+1) coarse system is solved by `gamma` FCG
    iterations preconditioned by kcycle(l+1) -- except at the last smoothed level, whose coarse system (level 0, A_0)
    is solved exactly with the factor computed once per operator."""
    Ak, X = hier.levs[l]; nc = X.shape[1]; di = hier.dinv[l]
    x = pre(Ak, di, rhs, l, cnt); r = rhs - Ak @ x
    rc = r[:nc] + X.T @ r[nc:]
    if l + 1 < hier.nl:
        Ac = hier.levs[l + 1][0]
        ec = fcg(Ac, rc, lambda v: kcycle(hier, l + 1, v, pre, post, gamma, cnt=cnt), gamma)
        if cnt is not None: cnt['matvec'][l + 1] += gamma; cnt['dot'][l + 1] += 4 * gamma
    else:
        ec = hier.lu.solve(rc)
        if cnt is not None: cnt['coarsest'] += 1
    xc = np.concatenate([ec, X @ ec])
    xp = post(Ak, di, rhs - Ak @ (x + xc), l, cnt)
    if cnt is not None: cnt['matvec'][l] += 2; cnt['transfer'][l] += 2
    return (x, xc, xp) if parts else x + xc + xp


def one_pass_k2(hier, r, Z, pre, post, gamma=2, project=True, cnt=None, stats=None):
    A = hier.A
    x0 = galerkin(A, Z, r, stats) if (Z is not None and Z.shape[1]) else np.zeros_like(r)
    bb = r - A @ x0
    parts = kcycle(hier, 0, bb, pre, post, gamma, parts=True, cnt=cnt)
    if not project: return x0 + sum(parts)
    W = np.column_stack(list(parts) + ([Z] if (Z is not None and Z.shape[1]) else []))
    return x0 + galerkin(A, W, bb, stats)


def pcg_v_preconditioner(hier, q=10):
    """PCG-V (amendment 1, A3.2): plain symmetric V(4,4) cycle, no deflation, Chebyshev with the guaranteed Gershgorin
    bound on every level, same polynomial pre and post -> SPD."""
    sm = make_cheb([lam_gershgorin(Ak) for Ak, _ in hier.levs], q=q)
    return lambda r: hier.cycle(0, r, sm, sm)


def pcg(A, b, M, x0, iterations, record=None):
    """ordinary PCG from x0 with a fixed SPD preconditioner M (callable)."""
    x = x0.copy(); r = b - A @ x; z = M(r); p = z.copy(); rz = r @ z
    for i in range(iterations):
        Ap = A @ p; pAp = p @ Ap
        if not pAp > 0 or rz == 0: break
        a = rz / pAp; x = x + a * p; r = r - a * Ap; z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        if record is not None: record(i, x)
    return x


# ----------------------------------------------------------------------------------------------- spatial controls (conditional, v1 §3.3)
def sgs4(Ak, dinv, rhs, l, cnt=None):
    """symmetric Gauss-Seidel, 2 symmetric sweeps from zero (= 4 sweeps, as accuracy_vs_time.py's SGS reference)."""
    from pyamg.relaxation.relaxation import gauss_seidel
    x = np.zeros_like(rhs); gauss_seidel(Ak.tocsr(), x, np.asarray(rhs, float), iterations=2, sweep='symmetric')
    if cnt is not None: cnt['matvec'][l] += 4
    return x


def sgs_pcg4(Ak, dinv, rhs, l, cnt=None, its=4):
    """4 PCG iterations from zero, preconditioned by one symmetric Gauss-Seidel sweep from zero (SPD for SPD A)."""
    from pyamg.relaxation.relaxation import gauss_seidel
    Acsr = Ak.tocsr()
    def M(v):
        z = np.zeros_like(v); gauss_seidel(Acsr, z, np.asarray(v, float), iterations=1, sweep='symmetric'); return z
    x = np.zeros_like(rhs); r = rhs.copy(); z = M(r); rz = r @ z; p = z.copy()
    if cnt is not None: cnt['matvec'][l] += 2                         # initial symmetric sweep = 2 sweeps (matvec-equivalents)
    st = 'ok'
    for _ in range(its):
        if rz == 0: st = 'exact'; break
        Ap = Ak @ p; pAp = p @ Ap
        if not pAp > 0: st = 'curvature'; break
        a = rz / pAp; x += a * p; r -= a * Ap; z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        if cnt is not None: cnt['matvec'][l] += 3; cnt['dot'][l] += 2   # A p + one symmetric sweep (2)
    if cnt is not None and st != 'ok': cnt['status'][st] = cnt['status'].get(st, 0) + 1
    return x
