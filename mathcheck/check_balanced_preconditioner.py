"""5 Oct (Astra, MATHEMATICAL_DIRECTION_REVIEW_2026-10-05.md, sections 2-4): the review's linear SPD construction measured on
OUR operators, with the classical (non-learned) members -- the numbers a learned residual-linear inverse would have to beat.
L3 perforated (3486 unknowns), dense, float64.  For each sample A:
  M_geo  = symmetric V(4,4) Chebyshev cycle, geometric transfers (m = 0)        -- linear, symmetric
  M_oa   = the same with operator-adapted transfers (m = 4 Jacobi steps)          -- oa-vcb's classical member, unprojected
  B_bal  = C + Q M_oa Q^T,  C = Z (Z^T A Z)^-1 Z^T, Q = I - C A,  Z = near-kernel vectors from A (oa_vcycle.near_kernel_basis)
Reported: symmetry ||B - B^T|| / ||B||;  spectrum of A^1/2 B A^1/2 (via Cholesky A = L L^T: eig of L^T B L) -> lambda_min,
lambda_max, kappa = lambda_max / lambda_min, energy-operator defect delta = ||I - A^1/2 B A^1/2||_2 = max|1 - lambda|;
||B A Z - Z|| / ||Z||;  identity  I - B A = Q (I - M A) Q;  PCG iterations to 1e-8 (energy-relative) from 0 for F = 1.
Review section 4: delta < 1 => ||e - B r||_A <= delta ||e||_A for EVERY residual, kappa <= (1 + delta) / (1 - delta)."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.linalg import cholesky, eigvalsh, solve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis

L, NU, Q_CHEB = 3, 4, 10
H = NestedHierarchy(L, coarse=perforated_square(n=9))
nl = 0; k_ = L
while k_ >= 1 and nl < 4 and len(H.nodes[k_ - 1]) >= 16: nl += 1; k_ -= 1


def lam_max(Ak, its=50):
    dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam


def vcycle_matrix(levs, A0):
    """dense matrix of the symmetric V(NU,NU) Chebyshev cycle (same polynomial smoother before and after)."""
    lm = [lam_max(Ak) for Ak, _ in levs]
    def cheb(Ak, rhs, x, lmax):
        dinv = (1 / Ak.diagonal())[:, None]; lo, hi = lmax / Q_CHEB, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(NU - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        return x
    A0d = A0.toarray()
    def cyc(l, rhs):
        Ak, X = levs[l]; nc = X.shape[1]
        x = cheb(Ak, rhs, np.zeros_like(rhs), lm[l]); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else solve(A0d, rc, assume_a='pos')
        x = x + np.vstack([ec, X @ ec])
        return cheb(Ak, rhs, x, lm[l])
    return cyc(0, np.eye(levs[0][0].shape[0]))


def pcg_its(A, B, b, Lc, tol=1e-8, maxit=300):
    u = np.linalg.solve(Lc.T, np.linalg.solve(Lc, b)); nu_ = np.sqrt(u @ (A @ u))
    x = np.zeros_like(b); r = b.copy(); z = B @ r; p = z.copy(); rz = r @ z
    for it in range(1, maxit + 1):
        Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap
        e = u - x
        if np.sqrt(e @ (A @ e)) <= tol * nu_: return it
        z = B @ r; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
    return maxit


def report(name, B, A, Ad, Lc, Z=None, M=None, Qm=None):
    S = Lc.T @ B @ Lc; lam = eigvalsh(0.5 * (S + S.T))
    sym = np.linalg.norm(B - B.T) / np.linalg.norm(B)
    delta = max(abs(1 - lam[0]), abs(1 - lam[-1]))
    extra = ''
    if Z is not None and Z.shape[1]:
        extra += f"  |BAZ-Z|/|Z| {np.linalg.norm(B @ (Ad @ Z) - Z) / np.linalg.norm(Z):.1e}"
        I = np.eye(Ad.shape[0])
        extra += f"  |(I-BA) - Q(I-MA)Q|/|I-BA| {np.linalg.norm((I - B @ Ad) - Qm @ (I - M @ Ad) @ Qm) / np.linalg.norm(I - B @ Ad):.1e}"
    its = pcg_its(A, B, H.F, Lc)
    print(f"    {name:<26} sym {sym:.1e}  lambda [{lam[0]:.3e}, {lam[-1]:.4f}]  kappa {lam[-1] / lam[0]:9.2f}  "
          f"delta {delta:.4f}  PCG its {its:3d}{extra}", flush=True)
    return dict(lmin=lam[0], lmax=lam[-1], delta=delta, its=its)


print(f"L{L}: {len(H.F)} unknowns, {nl} levels, V({NU},{NU}) Chebyshev (q {Q_CHEB}), symmetric")
summary = {}
for cname, contrast, pick in (('channels 1e4', 1e4, None), ('barriers 1e-4', 1e-4, 'islands')):
    rng = np.random.default_rng(1000); a_all = [field_channels(H, contrast, rng) for _ in range(400)]
    cand = list(range(300, 400)); chosen = []
    if pick is None:
        chosen = cand[:3]
    else:                                    # two samples with and two without near-kernel (island) vectors
        w, wo = [], []
        for i in cand:
            k = near_kernel_basis(H.A_fine(a_all[i]).tocsr()).shape[1]
            (w if k else wo).append(i)
            if len(w) >= 2 and len(wo) >= 2: break
        chosen = w[:2] + wo[:2]
    for i in chosen:
        A = H.A_fine(a_all[i]).tocsr(); Ad = A.toarray(); Lc = cholesky(Ad, lower=True)
        Z = near_kernel_basis(A)
        print(f"  {cname}, sample {i}: {Z.shape[1]} near-kernel vector(s)", flush=True)
        Mg = vcycle_matrix(*oa_hierarchy(H, A, nl, 0)); Mo = vcycle_matrix(*oa_hierarchy(H, A, nl, 4))
        res = {'M_geo (geometric V)': report('M_geo (geometric V)', Mg, A, Ad, Lc),
               'M_oa (op.-adapted V)': report('M_oa (op.-adapted V)', Mo, A, Ad, Lc)}
        if Z.shape[1]:
            C = Z @ np.linalg.solve(Z.T @ Ad @ Z, Z.T); Qm = np.eye(len(H.F)) - C @ Ad
            Bb = C + Qm @ Mo @ Qm.T
            res['B_bal (balanced, Z)'] = report('B_bal (balanced, Z)', Bb, A, Ad, Lc, Z=Z, M=Mo, Qm=Qm)
            Bg = C + Qm @ Mg @ Qm.T
            res['B_bal geo (balanced, Z)'] = report('B_bal geo (balanced, Z)', Bg, A, Ad, Lc, Z=Z, M=Mg, Qm=Qm)
        for k, v in res.items(): summary.setdefault((cname, Z.shape[1] > 0, k), []).append(v)
print("\nSUMMARY (mean over samples): delta / kappa / PCG its")
for (cname, isl, k), v in summary.items():
    print(f"  {cname:<14} {'with island' if isl else 'no island  '}  {k:<26} delta {np.mean([x['delta'] for x in v]):.4f}  "
          f"kappa {np.mean([x['lmax'] / x['lmin'] for x in v]):9.2f}  its {np.mean([x['its'] for x in v]):5.1f}")
