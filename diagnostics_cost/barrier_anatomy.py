"""3 Oct: why do oa-ll-op AND gnn-op both stall at ~0.29 on the barrier field (channels with contrast 1e-4)?
Darcy L3 perforated, k0 2 Galerkin, 10 samples (rng 1000, as contrast_stress.py).

The load is f = 1 everywhere, INCLUDING inside the barriers (conductivity 1e-4).  A source inside an insulator makes a
bump u ~ f w^2 / (8 a) ~ 50 in each strip (w = 1/16), against u ~ 1e-2 in the bulk.  This script measures
  (1) how much of ||u||^2 and ||u||_A^2 lives on the barrier nodes / elements;
  (2) the classical one-pass correctors (coarse solve + nu steps) and the barrier-subdomain exact solve
      u_c + R_S^T A_SS^-1 R_S r  (S = nodes whose stencil is all-barrier);
  (3) TG-PCG / AMG-PCG iterations to the networks' accuracy (op 2.892e-1 / 0.2934, gnn-op 2.867e-1 / 0.2956);
for the load as run ('f=1') and for a source only outside the barriers ('f=1 off barriers'),
the physically meaningful 'flow around barriers' problem."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
import pyamg

NET = (2.892e-1, 0.2934)
L, k0 = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (3, 2)
H = NestedHierarchy(L, coarse=perforated_square(n=9))
Pk = None
for k in range(k0 + 1, L + 1):
    Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))


def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3):
        np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]


def cheb(A, dinv, q, x, b, n, lmax=3.0):
    a_, bb = lmax / q, lmax; th, de = (bb + a_) / 2, (bb - a_) / 2
    s1 = th / de; r0 = 1 / s1; d = dinv * (b - A @ x) / th; x = x + d
    for _ in range(n - 1):
        r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (b - A @ x); x = x + d; r0 = r1
    return x


def pcg(A, b, x0, Minv, u, maxit=40):
    x = x0.copy(); r = b - A @ x; z = Minv(r); p = z.copy(); rz = r @ z
    en = lambda v: np.sqrt(v @ (A @ v)); nu_ = en(u); nl = np.linalg.norm(u); h = [(np.linalg.norm(x - u) / nl, en(x - u) / nu_)]
    for _ in range(maxit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = Minv(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        h.append((np.linalg.norm(x - u) / nl, en(x - u) / nu_))
    return np.array(h)


def errs(x, u, A):
    e = x - u
    return np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))


for name in ('f=1 (as run)', 'f=1 off barriers'):
    rng = np.random.default_rng(1000); R = {}
    def add(k, v): R.setdefault(k, []).append(v)
    for s in range(10):
        a = field_channels(H, 1e-4, rng); A = H.A_fine(a).tocsr(); bar = a < 0.5
        F = load(np.ones(H.n_elem, bool)) if name.startswith('f=1 (') else load(~bar)
        u = spsolve(A.tocsc(), F); dinv = 1 / A.diagonal()
        # barrier nodes: every element touching the node is a barrier element
        nodal_bar = np.ones(len(H.coords), bool)
        for j in range(3):
            np.logical_and.at(nodal_bar, H.elem[H.keep, j], bar[H.keep])
        S = nodal_bar[H.nodes[L]]
        uAu = u @ (A @ u); Ab = H.A_fine(np.where(bar, a, 0.0) + 1e-30).tocsr()
        add('frac_L2', np.sum(u[S] ** 2) / np.sum(u ** 2)); add('frac_A', (u @ (Ab @ u)) / uAu); add('frac_nodes', S.mean())
        add('umax/umed', np.abs(u).max() / np.median(np.abs(u)))
        solc = factorized((PkT @ A @ Pk).tocsc()); uc = Pk @ solc(PkT @ F); add('coarse', errs(uc, u, A))
        for nu in (4, 16):
            add(f'cheb{nu}', min((errs(cheb(A, dinv, q, uc, F, nu), u, A) for q in (4, 10, 30, 100)), key=lambda t: t[1]))
        r = F - A @ uc; iS = np.nonzero(S)[0]; x = uc.copy()
        if len(iS):
            x[iS] += spsolve(A[iS][:, iS].tocsc(), r[iS])
        add('barrier_solve', errs(x, u, A))
        x2 = x + Pk @ solc(PkT @ (F - A @ x)); add('barrier_solve+coarse', errs(x2, u, A))
        def tg(rr):
            y = (2 / 3) * dinv * rr; y = y + Pk @ solc(PkT @ (rr - A @ y)); return y + (2 / 3) * dinv * (rr - A @ y)
        h = pcg(A, F, uc, tg, u); add('tg_hist', h)
        ml = pyamg.smoothed_aggregation_solver(A); M = ml.aspreconditioner(cycle='V'); ha = pcg(A, F, uc, lambda v: M @ v, u); add('amg_hist', ha)
    print(f"\n=== barriers 1e-4, load {name} (10 samples, L{L}, k0 {k0} Galerkin) ===")
    print(f"  barrier nodes {np.mean(R['frac_nodes'])*100:.1f}% of unknowns carry {np.mean(R['frac_L2'])*100:.1f}% of ||u||^2; "
          f"barrier elements carry {np.mean(R['frac_A'])*100:.1f}% of ||u||_A^2;  max|u|/median|u| {np.median(R['umax/umed']):.0f}")
    for k in ('coarse', 'cheb4', 'cheb16', 'barrier_solve', 'barrier_solve+coarse'):
        m = np.mean(R[k], 0); print(f"  {k:<22} rel-L2 {m[0]:.3e}  energy {m[1]:.4f}")
    for k in ('tg_hist', 'amg_hist'):
        hs = np.array(R[k]); m = hs.mean(0)
        hit = [np.nonzero((h[:, 0] <= NET[0]) & (h[:, 1] <= NET[1]))[0] for h in hs]
        hit = [int(t[0]) if len(t) else 99 for t in hit]
        print(f"  {k[:3].upper()}-PCG after 1/2/4/8 its: " + ' '.join(f"{m[j,0]:.2e}/{m[j,1]:.3f}" for j in (1, 2, 4, 8))
              + f";  its to the networks' 2.89e-1/0.293: median {np.median(hit):.0f} [max {max(hit)}]")
