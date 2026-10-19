"""3 Oct, after the GPU barrier run (flow around barriers, --load no-low): oa-ll-op 4.39e-1 / 0.406 and gnn-op
4.63e-1 / 0.400 against Chebyshev post(4) 4.68e-1 / 0.394 on the same 100 test samples -- the learned correctors stop
at the classical one-pass level.

Hypothesis.  In oa-ll-op the lifting's predict/update see only GEOMETRY (coarse latent + edge positions); the operator
weights a_ij/a_ii enter only the fine-level smoother.  So the network's coarse-to-fine transfer is coefficient-blind
in the pair sense, like the geometric prolongation of the Galerkin coarse solve -- which is what fails on barriers.
The operator-adapted wavelet construction (route V: predict = -A_FF^-1 A_FC, Amaratunga/Sudarshan lineage; gamblets)
makes the transfer operator-dependent.  If an operator-dependent coarse space fixes barriers CLASSICALLY, the network
failure is an architecture choice (a geometric lifting), not an information limit.

Test, no training: one level transition (L3 fine, L2 coarse = k0 2), coarse space range(P) with
  geo     P = H.P[3]                      (linear interpolation; the Galerkin coarse solve of all runs)
  direct  Phat_ij = a_ij / sum_{k in C_i} a_ik      (classical AMG direct interpolation, row sums 1)
  jacobi  Phat = -D_FF^-1 A_FC            (one Jacobi step of the ideal predict; same sparsity as A_FC)
  ideal   Phat = -A_FF^-1 A_FC            (exact a-harmonic predict = route V exact; the coarse block is the Schur
                                           complement and the remaining error lives on the F nodes only)
  oa rhoR the same predict localised to patches of radius R coarse widths (OATransform route V, mode 'local')
  ('jacobi' does not reproduce constants -- row sums ~0.5 -- and is shown only as the known failure of the
   unnormalised one-step predict; 'direct' is its row-normalised form.)
For each: the coarse solve alone, + best Chebyshev post(4), and two-grid PCG (omega-Jacobi + that coarse solve)
after 1 / 2 / 4 iterations.  Darcy L3 perforated, 10 samples (rng 1000), rel-L2 / relative energy."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized, splu
from oa_lifting import NestedHierarchy, field_channels, _local_C
from coarse_meshes import perforated_square

L = 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
Pg = H.P[L].tocsr(); nc = Pg.shape[1]; nf = Pg.shape[0]
assert np.allclose(Pg[:nc].toarray(), np.eye(nc)), "fine ordering must list the coarse nodes first"
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))


def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3):
        np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]


def prolongations(A):
    A = A.tocsr(); A_FF = A[nc:, nc:].tocsc(); A_FC = A[nc:, :nc].tocsr(); I = sp.identity(nc, format='csr')
    out = {'geo': Pg}
    rs = np.asarray(A_FC.sum(1)).ravel()                       # sum_{k in C_i} a_ik (negative off-diagonals)
    Pd = sp.diags(np.where(rs != 0, 1 / np.where(rs != 0, rs, 1), 0)) @ A_FC
    rows0 = np.nonzero(rs == 0)[0]                             # F nodes with no C neighbour: geometric row
    if len(rows0):
        Pd = Pd.tolil(); Pd[rows0] = Pg[nc:][rows0]; Pd = Pd.tocsr()
    out['direct'] = sp.vstack([I, Pd]).tocsr()
    out['jacobi'] = sp.vstack([I, -sp.diags(1 / A_FF.diagonal()) @ A_FC]).tocsr()
    out['ideal'] = sp.vstack([I, sp.csr_matrix(-splu(A_FF).solve(A_FC.toarray()))]).tocsr()
    for rho in (1, 2, 3):                                      # route V local (the project's OA transform, mode 'local')
        Ph = _local_C(A_FF, A_FC, H.new_coords(L), H.level_coords(L - 1), rho * H.h[L - 1])
        out[f'oa rho{rho}'] = sp.vstack([I, sp.csr_matrix(Ph)]).tocsr()
    return out


def cheb(A, dinv, q, x, b, n, lmax=3.0):
    lo, hi = lmax / q, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
    s1 = th / de; r0 = 1 / s1; d = dinv * (b - A @ x) / th; x = x + d
    for _ in range(n - 1):
        r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (b - A @ x); x = x + d; r0 = r1
    return x


def errs(x, u, A):
    e = x - u
    return np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))


CASES = [('barriers 1e-4, no source in the barriers (--load no-low)', 1e-4, True),
         ('channels 1e4 (control: the paper setting)', 1e4, False)]
for title, contrast, nolow in CASES:
    rng = np.random.default_rng(1000); R = {}
    for s in range(10):
        a = field_channels(H, contrast, rng); A = H.A_fine(a).tocsr()
        F = load(a >= 0.5) if nolow else H.F
        u = spsolve(A.tocsc(), F); dinv = 1 / A.diagonal()
        for name, P in prolongations(A).items():
            PT = P.T.tocsr(); solc = factorized((PT @ A @ P).tocsc()); uc = P @ solc(PT @ F)
            R.setdefault((name, 'coarse'), []).append(errs(uc, u, A))
            R.setdefault((name, 'cheb4'), []).append(min((errs(cheb(A, dinv, q, uc, F, 4), u, A) for q in (4, 10, 30, 100)),
                                                          key=lambda t: t[1]))
            def tg(r):
                y = (2 / 3) * dinv * r; y = y + P @ solc(PT @ (r - A @ y)); return y + (2 / 3) * dinv * (r - A @ y)
            x = uc.copy(); r = F - A @ x; z = tg(r); p = z.copy(); rz = r @ z
            for j in range(1, 5):
                Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
                z = tg(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
                if j in (1, 2, 4):
                    R.setdefault((name, f'tg{j}'), []).append(errs(x, u, A))
            R.setdefault((name, 'nnzP'), []).append(P.nnz / P.shape[0])
    print(f"\n=== {title}: L3 fine, L2 coarse, 10 samples; rel-L2 / energy ===")
    print(f"{'coarse space':<10}{'nnz/row':>8}{'coarse alone':>20}{'+ Chebyshev post(4)':>22}{'TG-PCG 1 it':>20}{'2 its':>20}{'4 its':>20}")
    for name in ('geo', 'direct', 'jacobi', 'oa rho1', 'oa rho2', 'oa rho3', 'ideal'):
        row = f"{name:<10}{np.mean(R[(name, 'nnzP')]):>8.1f}"
        for k in ('coarse', 'cheb4', 'tg1', 'tg2', 'tg4'):
            m = np.mean(R[(name, k)], 0); row += f"{m[0]:>11.2e} /{m[1]:>6.3f}"
        print(row)
