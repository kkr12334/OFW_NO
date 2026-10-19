"""3 Oct (evening): the linear skeleton of an operator-adapted multilevel corrector -- route V as a TRANSPOSE pair.

Route V on the nested hierarchy, level k (new nodes F, old nodes C, coarse nodes listed first):
    Phi_k = [I; Phat_k],   Phat_k ~ -A_FF^-1 A_FC,   A_{k-1} = Phi_k^T A_k Phi_k,   Psi_k = E_k (nodal at the new nodes).
With the exact Phat_k, Psi_k^T A_k Phi_k = A_FC + A_FF Phat_k = 0, so in the multilevel basis W (coarsest Phi functions and
all prolonged detail functions) W^T A W = blockdiag(A_0, A_FF^(1), ..., A_FF^(L)) and
    A^-1 = W blockdiag^-1 W^T.
Applied to a residual r this is a one-pass multilevel method:
    down (dual / analysis, the transpose):  d_k = g_k[F],   g_{k-1} = g_k[C] + Phat_k^T g_k[F]          (g_L = r)
    coarsest:                                c_0 = A_0^-1 g_0                                              (28 nodes: dense)
    up (primal / synthesis):                 u_k = [u_{k-1}; Phat_k u_{k-1} + delta_k],  delta_k ~ A_FF^(k)^-1 d_k
Note the pair is W and W^T (dual analysis, primal synthesis), NOT a transform and its inverse.  Wording corrected 5 Oct
(independent review C): an invertible W is perfectly compatible with this solve; what the inverse pair (analysis W^-1,
synthesis W) cannot do is reproduce it with a BLOCK-DIAGONAL latent map, since W^-1 A^-1 W = D^-1 W^T W is not block
diagonal.  oa-ll's lifting, whose latent map is restricted in this way, falls under that statement.

The detail blocks A_FF^(k) are well conditioned on the benchmark coefficients (Jacobi-scaled, P1: kappa ~ 9-16 for the
channels and for log-uniform coefficients constant on coarse elements), so a few local steps solve them; the coarse block
is tiny.  Corrected 5 Oct (review G): this is NOT a coefficient-independent bound -- an adversarial P1 coefficient gives
kappa 693 at L3 (check_stocktake.log).  The approximations are therefore: (i) a cheap Phat (m steps of prolongator smoothing
from the geometric predict), (ii) nu Chebyshev steps on each detail block, (iii) optional post-smoothing on A.

Checks (Darcy perforated L3, 10 samples, barriers 1e-4 --load no-low and channels 1e4):
  1. exactness: exact Phat at every level + exact block solves reproduce A^-1 r to round-off (two samples);
  2. one pass from r = F (no external coarse solve, k0 = 0) and from the geometric Galerkin u_c (k0 = 2), as a function
     of the predict (geo / jac m / exact) and nu.  This is the non-learned member of the class the new network contains."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, splu, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square

L = 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
for k in range(1, L + 1):
    nc = H.P[k].shape[1]
    assert (H.P[k][:nc] - sp.identity(nc)).nnz == 0


def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3):
        np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]


def build(A, kind, m=4):
    """per level k: (nc, Phat, A_FF); plus A_0.  kind: geo | jac | exact"""
    lev = {}; Ak = A.tocsr()
    for k in range(L, 0, -1):
        nc = H.P[k].shape[1]; A_FF = Ak[nc:, nc:].tocsr(); A_FC = Ak[nc:, :nc].tocsr()
        X = H.P[k][nc:].tocsr()
        if kind == 'jac':
            Dinv = sp.diags(1 / A_FF.diagonal())
            for _ in range(m):
                X = (X + (2 / 3) * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
        elif kind == 'exact':
            X = sp.csr_matrix(-splu(A_FF.tocsc()).solve(A_FC.toarray()))
        Phi = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr()
        lev[k] = (nc, X, A_FF)
        Ak = (Phi.T @ Ak @ Phi).tocsr()
    return lev, Ak


def cheb(A, b, nu, x=None, q=10, lmax=3.0):
    dinv = 1 / A.diagonal(); x = np.zeros_like(b) if x is None else x.copy()
    if nu == 0: return x
    lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
    s1 = th / de; r0 = 1 / s1; d = dinv * (b - A @ x) / th; x = x + d
    for _ in range(nu - 1):
        r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (b - A @ x); x = x + d; r0 = r1
    return x


def one_pass(lev, A0, r, nu, q=10):
    g = {L: r}; dk = {}
    for k in range(L, 0, -1):
        nc, X, _ = lev[k]; dk[k] = g[k][nc:]; g[k - 1] = g[k][:nc] + X.T @ g[k][nc:]
    u = spsolve(A0.tocsc(), g[0]) if A0.shape[0] > 1 else g[0] / A0.toarray()[0, 0]
    for k in range(1, L + 1):
        nc, X, A_FF = lev[k]
        delta = spsolve(A_FF.tocsc(), dk[k]) if nu == 'exact' else cheb(A_FF, dk[k], nu, q=q)
        u = np.concatenate([u, X @ u + delta])
    return u


def errs(x, u, A):
    e = x - u
    return np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))


# 1. exactness
rng = np.random.default_rng(7)
for s in range(2):
    a = field_channels(H, 1e-4, rng); A = H.A_fine(a).tocsr(); F = load(a >= 0.5); u = spsolve(A.tocsc(), F)
    lev, A0 = build(A, 'exact'); x = one_pass(lev, A0, F, 'exact')
    print(f"exactness, sample {s}: ||W D^-1 W^T F - A^-1 F|| / ||A^-1 F|| = {np.linalg.norm(x - u) / np.linalg.norm(u):.2e}"
          f"   (coarsest A_0: {A0.shape[0]} nodes)")

# 2. one pass
P2 = None
for k in range(3, L + 1):
    P2 = H.P[k] if P2 is None else H.P[k] @ P2
P2 = P2.tocsr()
CONF = [('geo', 0), ('jac', 2), ('jac', 4), ('jac', 8), ('exact', 0)]
for title, contrast, nolow in (('barriers 1e-4, --load no-low', 1e-4, True), ('channels 1e4', 1e4, False)):
    rng = np.random.default_rng(1000); R = {}
    for s in range(10):
        a = field_channels(H, contrast, rng); A = H.A_fine(a).tocsr(); F = load(a >= 0.5) if nolow else H.F
        u = spsolve(A.tocsc(), F)
        uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); r_c = F - A @ uc
        for kind, m in CONF:
            lev, A0 = build(A, kind, m); tag = kind if kind != 'jac' else f'jac {m}'
            for nu in (2, 4, 'exact'):
                x0 = one_pass(lev, A0, F, nu)                                  # k0 = 0: from r = F
                R.setdefault((tag, nu, 'k0=0'), []).append(errs(x0, u, A))
                R.setdefault((tag, nu, 'k0=0 +cheb4'), []).append(errs(cheb(A, F, 4, x0), u, A))
                x2 = uc + one_pass(lev, A0, r_c, nu)                            # from the geometric u_c
                R.setdefault((tag, nu, 'k0=2'), []).append(errs(x2, u, A))
                R.setdefault((tag, nu, 'k0=2 +cheb4'), []).append(errs(cheb(A, F, 4, x2), u, A))
    print(f"\n=== {title}: one pass of W D~^-1 W^T, L3 -> L0, 10 samples, rel-L2 / energy ===")
    print(f"{'predict':<9}{'nu':>6}{'from F (k0 0)':>20}{'+ Cheb post(4)':>20}{'from u_c (k0 2)':>20}{'+ Cheb post(4)':>20}")
    for kind, m in CONF:
        tag = kind if kind != 'jac' else f'jac {m}'
        for nu in (2, 4, 'exact'):
            row = f"{tag:<9}{str(nu):>6}"
            for col in ('k0=0', 'k0=0 +cheb4', 'k0=2', 'k0=2 +cheb4'):
                v = np.mean(R[(tag, nu, col)], 0); row += f"{v[0]:>12.2e} /{v[1]:>6.3f}"
            print(row)


# 3. the multiplicative alternative: ONE V(nu,nu)-cycle with operator-adapted transfers Phi_k and Galerkin level
#    operators A_{k-1} = Phi_k^T A_k Phi_k (Chebyshev smoothing on A_k, exact coarsest solve).  With the cheap predict the
#    block-diagonal pass above neglects the inter-level coupling that an inexact Phat leaves; the V-cycle does not.
def build_ops(A, kind, m):
    ops = {L: A.tocsr()}; Phis = {}
    for k in range(L, 0, -1):
        Ak = ops[k]; nc = H.P[k].shape[1]; A_FF = Ak[nc:, nc:].tocsr(); A_FC = Ak[nc:, :nc].tocsr(); X = H.P[k][nc:].tocsr()
        if kind == 'jac':
            Dinv = sp.diags(1 / A_FF.diagonal())
            for _ in range(m):
                X = (X + (2 / 3) * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
        elif kind == 'exact':
            X = sp.csr_matrix(-splu(A_FF.tocsc()).solve(A_FC.toarray()))
        Phis[k] = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr(); ops[k - 1] = (Phis[k].T @ Ak @ Phis[k]).tocsr()
    return ops, Phis


def lam_max(Ak, its=30, seed=0):
    # power iteration for lambda_max(D^-1 A_k), x 1.05: the P1 bound 3 is proved for the fine level only; Galerkin
    # operators on smoothed transfers have wider stencils and can exceed it (the first version of this check diverged)
    d = Ak.diagonal(); v = np.random.default_rng(seed).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / d; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam


def vcycle(ops, Phis, k, b, nu, x=None, lm=None):
    if k == 0:
        return spsolve(ops[0].tocsc(), b)
    lmax = 3.0 if k == L else lm[k]
    x = cheb(ops[k], b, nu, x, lmax=lmax)
    rc = Phis[k].T @ (b - ops[k] @ x)
    x = x + Phis[k] @ vcycle(ops, Phis, k - 1, rc, nu, lm=lm)
    return cheb(ops[k], b, nu, x, lmax=lmax)


for title, contrast, nolow in (('barriers 1e-4, --load no-low', 1e-4, True), ('channels 1e4', 1e4, False)):
    rng = np.random.default_rng(1000); R = {}
    for s in range(10):
        a = field_channels(H, contrast, rng); A = H.A_fine(a).tocsr(); F = load(a >= 0.5) if nolow else H.F
        u = spsolve(A.tocsc(), F)
        for kind, m in [('geo', 0), ('jac', 2), ('jac', 4), ('jac', 8), ('exact', 0)]:
            ops, Phis = build_ops(A, kind, m); tag = kind if kind != 'jac' else f'jac {m}'
            lm = {k: lam_max(ops[k]) for k in range(1, L)}
            R.setdefault((tag, 'lmax'), []).append([lm[k] for k in range(1, L)])
            for nu in (2, 4, 8):
                R.setdefault((tag, nu), []).append(errs(vcycle(ops, Phis, L, F, nu, lm=lm), u, A))
    print(f"\n=== {title}: ONE V(nu,nu)-cycle from 0, operator-adapted transfers, Galerkin levels, L3 -> L0, rel-L2 / energy ===")
    print(f"{'predict':<9}" + ''.join(f"{'nu = %d' % nu:>20}" for nu in (2, 4, 8)))
    for kind, m in [('geo', 0), ('jac', 2), ('jac', 4), ('jac', 8), ('exact', 0)]:
        tag = kind if kind != 'jac' else f'jac {m}'
        print(f"{tag:<9}" + ''.join(f"{np.mean(R[(tag, nu)], 0)[0]:>12.2e} /{np.mean(R[(tag, nu)], 0)[1]:>6.3f}" for nu in (2, 4, 8))
              + "   lambda_max(D^-1 A_k), k = 1, 2: " + ', '.join(f"{v:.2f}" for v in np.max(R[(tag, 'lmax')], 0)))
