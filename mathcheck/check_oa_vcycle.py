"""3 Oct (evening): checks for oa-vc (oa_vcycle.py).
  1. CONTAINMENT: with the smoother networks set to nu-step weighted Jacobi (linear path only, MLPs zeroed), the torch
     V-cycle equals a numpy V(nu,nu)-cycle with weighted-Jacobi smoothers, the same operator-adapted transfers and
     Galerkin levels, and the exact coarsest solve -- to round-off;
  2. positive homogeneity in r for random weights: f(3 r) = 3 f(r);
  3. the numpy classical member (Chebyshev smoothers) reproduces check_oa_transpose_pair's V(4,4) numbers.
Perforated L3, barriers 1e-4 (--load no-low), float64."""
import sys, os, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, splu
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, classical_vcycle
torch.set_default_dtype(torch.float64)

L, m, nu, om = 3, 4, 4, 2 / 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3):
        np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]

rng = np.random.default_rng(11)
a_list = [field_channels(H, 1e-4, rng) for _ in range(3)]
D = OAData(H, a_list, 3, m, 'cpu', H.level_coords)
D.Av = [v.double() for v in D.Av]; D.Xv = [v.double() for v in D.Xv]; D.A0inv = D.A0inv.double()
# float32 storage above is for training; recompute float64 values for an exact comparison
from oa_vcycle import structural_patterns, _on_pattern
pats, _ = structural_patterns(H, 3, m)
Av, Xv, A0i = [[] for _ in pats], [[] for _ in pats], []
for a in a_list:
    levs, A0 = oa_hierarchy(H, H.A_fine(a), 3, m)
    for l, ((Ak, X), (pA, pX)) in enumerate(zip(levs, pats)):
        Av[l].append(_on_pattern(Ak, pA)[0]); Xv[l].append(_on_pattern(X, pX)[0])
    A0i.append(np.linalg.inv(A0.toarray()))
D.Av = [torch.tensor(np.array(v)) for v in Av]; D.Xv = [torch.tensor(np.array(v)) for v in Xv]; D.A0inv = torch.tensor(np.array(A0i))

net = OAVCycle(D, width=4, nu=nu, share=True, mlp=True)
with torch.no_grad():
    for S in (net.pre[0], net.post[0]):
        for p in S.parameters(): p.zero_()
        S.enc_lin.weight[1, 0] = 1.0                       # channel 1 <- z / sigma ; channel 0 (x) starts at 0
        for t in range(nu):
            S.V[t].weight[0, 0] = 1.0                      # V picks x
            Lw = S.Lw[t].weight                            # dx = omega (z - x - sum_j W_ij x_j)
            Lw[0, 1] = om; Lw[0, 0] = -om; Lw[0, S.C + 0] = -om
        S.dec_lin.weight[0, 0] = 1.0

def np_jacobi(Ak, b, x):
    dinv = 1 / Ak.diagonal()
    for _ in range(nu):
        x = x + om * dinv * (b - Ak @ x)
    return x

def np_vcycle(levs, A0, b):
    def cyc(l, rhs):
        Ak, X = levs[l]; nc = X.shape[1]
        x = np_jacobi(Ak, rhs, np.zeros_like(rhs)); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else np.linalg.solve(A0.toarray(), rc)
        x = x + np.concatenate([ec, X @ ec])
        return np_jacobi(Ak, rhs, x)
    return cyc(0, b)

R = np.array([load(a >= 0.5) for a in a_list])
net.set_ops(*D.batch(np.arange(3)))
with torch.no_grad():
    out = net(torch.tensor(R)).numpy()
worst = 0.0
for i, a in enumerate(a_list):
    levs, A0 = oa_hierarchy(H, H.A_fine(a), 3, m); ref = np_vcycle(levs, A0, R[i])
    worst = max(worst, np.abs(out[i] - ref).max() / np.abs(ref).max())
print(f"1. containment: network with Jacobi weights vs numpy V({nu},{nu}) Jacobi cycle, max rel. diff {worst:.1e}")

torch.manual_seed(0); net2 = OAVCycle(D, width=8, nu=nu); net2.set_ops(*D.batch(np.arange(3)))
with torch.no_grad():
    f1 = net2(torch.tensor(R)); f3 = net2(3 * torch.tensor(R))
print(f"2. positive homogeneity, random weights: max |f(3r) - 3 f(r)| / max |3 f(r)| = {float((f3 - 3 * f1).abs().max() / (3 * f1).abs().max()):.1e}")

rng = np.random.default_rng(1000); E = []
for s in range(10):
    a = field_channels(H, 1e-4, rng); A = H.A_fine(a).tocsr(); F = load(a >= 0.5); u = spsolve(A.tocsc(), F)
    levs, A0 = oa_hierarchy(H, A, 3, m); x = classical_vcycle(levs, A0, F, nu=4)
    e = x - u; E.append((np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))))
print(f"3. classical member (Chebyshev, q 10), barriers, 10 samples: {np.mean(E, 0)[0]:.2e} / {np.mean(E, 0)[1]:.3f}  "
      f"(check_oa_transpose_pair: jac 4, nu 4 -> 1.42e-01 / 0.138)")
print(f"   parameters at width 16, nu 4, shared: {sum(p.numel() for p in OAVCycle(D, width=16, nu=4).parameters()):,}")
