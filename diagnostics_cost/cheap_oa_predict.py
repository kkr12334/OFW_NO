"""3 Oct (evening): can the operator-adapted coarse space be had CHEAPLY, judged by what matters now -- the quality of a
one-pass correction (coarse solve + 4 Chebyshev steps) -- rather than by the conditioning of a full transform (the
criterion of the earlier feasibility study, under which partial smoothing failed)?

Route V ideal predict: Phat = -A_FF^-1 A_FC.  Cheap approximations, all starting from the geometric predict P0 (rows
of the linear interpolation on the new nodes F) and iterating on  A_FF X = -A_FC  column-wise (sparse-matrix iteration,
i.e. prolongator smoothing):
  jac m   X <- X + (2/3) D_FF^-1 (-A_FC - A_FF X),  m steps
  cheb m  m Chebyshev steps for D_FF^-1 A_FF on [3/q, 3] (3 = proved bound of lambda_max(D^-1 A) for P1, which
          holds for every principal submatrix by interlacing), best q of (4, 10, 30)
Starting from P0 keeps constants exact: the residual of P0 is the F-rows' row sums of A, zero away from Dirichlet
nodes.  The support grows by one fine hop per step.  Compared with the route-V local patch solves (rho 3, the space
that fixed the barrier field: Chebyshev post(4) 0.052) and the geometric space (0.197).
Darcy L3 perforated (fine L3, coarse L2, k0 2), 10 samples (rng 1000), barriers with --load no-low and channels 1e4."""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels, _local_C
from coarse_meshes import perforated_square

L = 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
Pg = H.P[L].tocsr(); nc = Pg.shape[1]
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
I_c = sp.identity(nc, format='csr')


def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3):
        np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]


def smoothed(A_FF, A_FC, m, kind, q=10, drop=0.0):
    X = Pg[nc:].tocsr(); Dinv = sp.diags(1 / A_FF.diagonal()); B = -A_FC
    if kind == 'jac':
        for _ in range(m):
            X = (X + (2 / 3) * (Dinv @ (B - A_FF @ X))).tocsr()
    else:                                                      # Chebyshev on D^-1 A_FF over [3/q, 3]
        lo, hi = 3.0 / q, 3.0; th, de = (hi + lo) / 2, (hi - lo) / 2; s1 = th / de; r0 = 1 / s1
        Dk = (Dinv @ (B - A_FF @ X)) / th; X = (X + Dk).tocsr()
        for _ in range(m - 1):
            r1 = 1 / (2 * s1 - r0); Dk = (r1 * r0 * Dk + (2 * r1 / de) * (Dinv @ (B - A_FF @ X))).tocsr()
            X = (X + Dk).tocsr(); r0 = r1
    if drop:                                                   # optional sparsification (relative to row max)
        X = X.tocoo(); rmax = np.zeros(X.shape[0]); np.maximum.at(rmax, X.row, np.abs(X.data))
        keep = np.abs(X.data) >= drop * rmax[X.row]; X = sp.csr_matrix((X.data[keep], (X.row[keep], X.col[keep])), X.shape)
    return sp.vstack([I_c, X]).tocsr()


def cheb_post(A, dinv, q, x, b, n, lmax=3.0):
    lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
    s1 = th / de; r0 = 1 / s1; d = dinv * (b - A @ x) / th; x = x + d
    for _ in range(n - 1):
        r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (b - A @ x); x = x + d; r0 = r1
    return x


def errs(x, u, A):
    e = x - u
    return np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))


SPACES = ['geo'] + [f'jac {m}' for m in (2, 4, 8, 16)] + [f'cheb {m}' for m in (2, 4, 6, 8, 12)] + ['oa rho3']
for title, contrast, nolow in (('barriers 1e-4, --load no-low', 1e-4, True), ('channels 1e4', 1e4, False)):
    rng = np.random.default_rng(1000); R = {}
    for s in range(10):
        a = field_channels(H, contrast, rng); A = H.A_fine(a).tocsr(); F = load(a >= 0.5) if nolow else H.F
        u = spsolve(A.tocsc(), F); dinv = 1 / A.diagonal()
        A_FF = A[nc:, nc:].tocsr(); A_FC = A[nc:, :nc].tocsr()
        for name in SPACES:
            t0 = time.perf_counter()
            if name == 'geo':
                P = Pg
            elif name == 'oa rho3':
                P = sp.vstack([I_c, sp.csr_matrix(_local_C(A_FF.tocsc(), A_FC, H.new_coords(L), H.level_coords(L - 1), 3 * H.h[L - 1]))]).tocsr()
            elif name.startswith('jac'):
                P = smoothed(A_FF, A_FC, int(name.split()[1]), 'jac')
            else:
                m = int(name.split()[1])
                best = None
                for q in (4, 10, 30):
                    Pq = smoothed(A_FF, A_FC, m, 'cheb', q)
                    e_ = np.linalg.norm((A_FF @ Pq[nc:] + A_FC).data)      # residual of the predict equation
                    if best is None or e_ < best[0]: best = (e_, Pq)
                P = best[1]
            tb = time.perf_counter() - t0
            PT = P.T.tocsr(); solc = factorized((PT @ A @ P).tocsc()); uc = P @ solc(PT @ F)
            R.setdefault((name, 'coarse'), []).append(errs(uc, u, A))
            R.setdefault((name, 'cheb4'), []).append(min((errs(cheb_post(A, dinv, q, uc, F, 4), u, A) for q in (4, 10, 30, 100)), key=lambda t: t[1]))
            R.setdefault((name, 'ms'), []).append(tb * 1e3); R.setdefault((name, 'nnz'), []).append(P.nnz / P.shape[0])
    print(f"\n=== {title}: L3 fine / L2 coarse, 10 samples; rel-L2 / energy; build ms (median, CPU) ===")
    print(f"{'predict':<10}{'nnz/row':>8}{'build ms':>10}{'coarse alone':>20}{'+ Chebyshev post(4)':>22}")
    for name in SPACES:
        c = np.mean(R[(name, 'coarse')], 0); k = np.mean(R[(name, 'cheb4')], 0)
        print(f"{name:<10}{np.mean(R[(name, 'nnz')]):>8.1f}{np.median(R[(name, 'ms')]):>10.1f}"
              f"{c[0]:>11.2e} /{c[1]:>6.3f}{k[0]:>13.2e} /{k[1]:>6.3f}")
