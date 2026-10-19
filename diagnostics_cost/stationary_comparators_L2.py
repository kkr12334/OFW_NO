"""Second review (3 Oct): does the LEARNED one-pass corrector beat a CLASSICAL one-pass corrector with the same
information (one Galerkin coarse solve + nu smoothing steps)?  Darcy perforated, channels 1e4, 10 samples.
  post(nu)  : start from u_c (the network's input), nu smoothing steps -- the Proposition's comparator
  V(nu,nu)  : start from 0, nu pre-smoothing, ONE coarse correction, nu post-smoothing (2 nu steps, 1 coarse solve)
Smoothers: weighted Jacobi (best omega of 0.4..1.0 per configuration, chosen on these samples, i.e. optimistic),
Chebyshev on D^-1 A over [b/q, b] with b = 3 (the proved bound; best q of 4, 10, 30, 100), symmetric Gauss-Seidel
(nu/2 forward+backward pairs, i.e. nu triangular sweeps).  Errors rel-L2 / relative energy, means over samples."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized, spsolve_triangular
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
OP = {2: (2.474e-2, 0.045)}   # L2 CPU screen (ablation_linear_L2_cpu.log)
def jac(A, dinv, om):
    return lambda x, b, n: [x := x + om * dinv * (b - A @ x) for _ in range(n)][-1] if n else x
def cheb(A, dinv, q, lmax=3.0):
    a, bb = lmax / q, lmax; th, de = (bb + a) / 2, (bb - a) / 2
    def run(x, b, n):
        if n == 0: return x
        s1 = th / de; r0 = 1 / s1; d = dinv * (b - A @ x) / th; x = x + d
        for _ in range(n - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (b - A @ x); x = x + d; r0 = r1
        return x
    return run
def sgs(A):
    Lw = sp.tril(A, 0, format='csr'); Up = sp.triu(A, 0, format='csr')
    def run(x, b, n):
        for k in range(n):
            T = Lw if k % 2 == 0 else Up
            x = x + spsolve_triangular(T, b - A @ x, lower=(k % 2 == 0))
        return x
    return run
for L, k0 in ((2, 1),):
    H = NestedHierarchy(L, coarse=perforated_square(n=9)); F = H.F; rng = np.random.default_rng(1000)
    Pk = None
    for k in range(k0 + 1, L + 1):
        Pk = H.P[k] if Pk is None else H.P[k] @ Pk
    Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
    samples = []
    for s in range(10 if L == 3 else 6):
        A = H.A_fine(field_channels(H, 1e4, rng)).tocsr(); u = spsolve(A.tocsc(), F)
        solc = factorized((PkT @ A @ Pk).tocsc()); samples.append((A, u, solc, 1 / A.diagonal()))
    def ev(make, mode, nu):
        out = []
        for A, u, solc, dinv in samples:
            sm = make(A, dinv)
            if mode == 'post':
                x = sm(Pk @ solc(PkT @ F), F, nu)
            else:
                x = sm(np.zeros_like(u), F, nu); x = x + Pk @ solc(PkT @ (F - A @ x)); x = sm(x, F, nu)
            e = x - u; out.append((np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))))
        return np.mean(out, 0)
    print(f"\nDarcy L{L} (k0 {k0}); oa-ll-op (trained, one coarse solve): {OP[L][0]:.2e} / {OP[L][1]:.3f};  "
          f"coarse alone {ev(lambda A, d: (lambda x, b, n: x), 'post', 0)[0]:.2e} / {ev(lambda A, d: (lambda x, b, n: x), 'post', 0)[1]:.3f}")
    for mode in ('post', 'V'):
        for nu in (4, 8, 16):
            rj = min((ev(lambda A, d, om=om: jac(A, d, om), mode, nu) for om in (0.4, 0.5, 0.6, 0.67, 0.8, 1.0)), key=lambda t: t[1])
            rc = min((ev(lambda A, d, q=q: cheb(A, d, q), mode, nu) for q in (4, 10, 30, 100)), key=lambda t: t[1])
            rg = ev(lambda A, d: sgs(A), mode, nu)
            print(f"  {mode:<4} nu {nu:>2}: Jacobi(best omega) {rj[0]:.2e} / {rj[1]:.3f} | Chebyshev(best q) {rc[0]:.2e} / {rc[1]:.3f} "
                  f"| sym. Gauss-Seidel {rg[0]:.2e} / {rg[1]:.3f}", flush=True)
