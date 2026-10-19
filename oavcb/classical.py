"""The matched classical reference (numpy): the same deflated V(nu, nu) cycle, with the same transfers, Galerkin levels, exact
coarsest solve and Galerkin output, but degree-nu Chebyshev smoothers on [lambda_k / q, lambda_k]:
lambda_k = 3 on the finest scalar level, otherwise 1.05 x a 30-step power-iteration estimate of lambda_max(D^-1 A_k).
Also the contained members of Proposition 2 (weighted Jacobi, fixed-interval Chebyshev) and a two-grid PCG for cost studies."""
import numpy as np
from scipy.sparse.linalg import splu


def _lam_max(Ak, its=30):
    dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam


def chebyshev(Ak, rhs, x, lmax, nu=4, q=10):
    """nu Chebyshev steps on D^-1 A_k over [lmax / q, lmax], starting from x."""
    dinv = 1 / Ak.diagonal(); lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
    s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
    for _ in range(nu - 1):
        r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
    return x


def jacobi(Ak, rhs, x, nu=4, omega=2 / 3):
    dinv = 1 / Ak.diagonal()
    for _ in range(nu):
        x = x + omega * dinv * (rhs - Ak @ x)
    return x


def galerkin_projection(A, W, rhs):
    """A-orthogonal projection of A^-1 rhs onto span W, computed as in OAVCycle._galerkin (scaled, ridge 1e-10)."""
    if W.shape[1] == 0:
        return np.zeros_like(rhs)
    G = W.T @ (A @ W); dg = np.diag(G); live = dg > 1e-30 * dg.max()
    sc = np.where(live, 1 / np.sqrt(np.where(live, dg, 1)), 0)
    y = sc * np.linalg.solve(sc[:, None] * G * sc[None, :] + np.diag((~live) + 1e-10), sc * (W.T @ rhs))
    return W @ y


def classical_cycle(levs, A0, b, Z=None, nu=4, q=10, project=True, smoother='chebyshev', fine_lmax=3.0, lmax=None):
    """Classical counterpart of one OA-VCB pass (deflation by Z, one V(nu, nu) cycle on the deflated residual, Galerkin
    projection onto span{pre, coarse, post, Z}).  Returns the correction.
    smoother 'chebyshev' (the classical reference), or 'jacobi' (contained Jacobi member, omega = 2/3).
    lmax: per-level spectral upper bounds to use for Chebyshev (default: fine_lmax on level 0, power iteration below;
    pass fine_lmax=lmax=3.0 for every level to get the contained fixed-interval member)."""
    A = levs[0][0]
    hasZ = Z is not None and Z.shape[1] > 0
    x0 = galerkin_projection(A, Z, b) if hasZ else np.zeros_like(b)
    bb = b - A @ x0
    if lmax is None:
        lm = [fine_lmax if (l == 0 and fine_lmax) else _lam_max(Ak) for l, (Ak, _) in enumerate(levs)]
    else:
        lm = [lmax] * len(levs)
    smooth = ((lambda Ak, r, l: chebyshev(Ak, r, np.zeros_like(r), lm[l], nu, q)) if smoother == 'chebyshev'
              else (lambda Ak, r, l: jacobi(Ak, r, np.zeros_like(r), nu)))
    A0lu = splu(A0.tocsc())

    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]
        x = smooth(Ak, rhs, l); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else A0lu.solve(rc)
        xc = np.concatenate([ec, X @ ec])
        xp = smooth(Ak, rhs - Ak @ (x + xc), l)
        return (x, xc, xp) if parts else x + xc + xp

    if not project:
        return x0 + cyc(0, bb)
    W = np.column_stack(list(cyc(0, bb, parts=True)) + ([Z] if hasZ else []))
    return x0 + galerkin_projection(A, W, bb)


def two_grid_pcg(A, b, P, x0, n_iter=8):
    """PCG with the two-grid preconditioner (Jacobi - coarse Galerkin - Jacobi, omega = 2/3) from x0; returns the iterates."""
    dinv = 1 / A.diagonal(); PT = P.T.tocsr(); solc = splu((PT @ A @ P).tocsc())

    def tg(r):
        y = (2 / 3) * dinv * r; y = y + P @ solc.solve(PT @ (r - A @ y)); return y + (2 / 3) * dinv * (r - A @ y)

    x = x0.copy(); r = b - A @ x; z = tg(r); p = z.copy(); rz = r @ z; its = [x.copy()]
    for _ in range(n_iter):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = tg(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2; its.append(x.copy())
    return its
