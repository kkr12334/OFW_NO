"""5 Oct: oa-vcb on plane-strain elasticity (train_elasticity.py, dof graph, P_k (x) I_2, stiff bands E_f/E_m 1e2, L3).
  1. rigid-body modes: the operator-adapted prolongation of every level maps the coarse translations / rotation to the
     fine ones on rows away from the Dirichlet boundary (and the geometric one does everywhere);
  2. CONTAINMENT with Jacobi weights (Galerkin output, no deflation) vs a numpy cycle;
  3. ENERGY GUARANTEE for random weights from the Galerkin u_c, and homogeneity;
  4. classical one-pass members on the same samples (from the Galerkin u_c, energy):  V(4,4) op.-adapted, + projection."""
import sys, os, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, structural_patterns, _on_pattern, classical_vcycle, classical_vcycle_proj
torch.set_default_dtype(torch.float64)
L, m, nu, om = 3, 4, 4, 2 / 3
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H); F = V.F
rng = np.random.default_rng(2000); E_all = [field_channels(H, 1e2, rng) for _ in range(400)]
te = list(range(300, 312)); A_list = [V.A_fine(E_all[i]).tocsr() for i in te]

def rbm(k):                                          # translations x, y and rotation on the level-k dofs
    xy = H.level_coords(k); n = len(xy); T = np.zeros((2 * n, 3))
    T[0::2, 0] = 1; T[1::2, 1] = 1; T[0::2, 2] = -xy[:, 1]; T[1::2, 2] = xy[:, 0]; return T
levs, A0 = oa_hierarchy(V, A_list[0], 3, m); worst = []
for l, (Ak, X) in enumerate(levs):
    k = L - l; nc = X.shape[1]; Tc, Tf = rbm(k - 1), rbm(k)
    Phi = sp.vstack([sp.identity(nc), X]).tocsr(); d = Phi @ Tc - Tf
    rs = np.abs(np.asarray(Ak.sum(1)).ravel()) / Ak.diagonal()          # rows coupled to the Dirichlet boundary
    interior = rs < 1e-8; dg = sp.vstack([sp.identity(nc), (V.P[k][nc:])]).tocsr() @ Tc - Tf
    worst.append((np.abs(d[interior]).max() if interior.any() else float('nan'), np.abs(dg).max(), interior.mean(),
                  float(np.median(np.abs(d)))))
print("1. rigid-body modes, per level: max |Phi t_c - t_f| on rows not coupled to the boundary (op.-adapted) / everywhere "
      "(geometric) / share of such rows / median |Phi t_c - t_f| over all rows:",
      [tuple(f"{v:.2f}" if i == 2 else f"{v:.1e}" for i, v in enumerate(w)) for w in worst])

D = OAData(V, None, 3, m, 'cpu', V.level_coords, near_kernel=False, A_list=A_list)
pats, _ = structural_patterns(V, 3, m); Av, Xv, A0i = [[] for _ in pats], [[] for _ in pats], []
for A in A_list:
    lv_, A0_ = oa_hierarchy(V, A, 3, m)
    for l, ((Ak, X), (pA, pX)) in enumerate(zip(lv_, pats)):
        Av[l].append(_on_pattern(Ak, pA)[0]); Xv[l].append(_on_pattern(X, pX)[0])
    A0i.append(np.linalg.inv(A0_.toarray()))
ops = ([torch.tensor(np.array(v)) for v in Av], [torch.tensor(np.array(v)) for v in Xv], torch.tensor(np.array(A0i)))
net = OAVCycle(D, width=4, nu=nu, project=True, deflate=False)
with torch.no_grad():
    for S in (net.pre[0], net.post[0]):
        for p in S.parameters(): p.zero_()
        S.enc_lin.weight[1, 0] = 1.0
        for t in range(nu):
            S.V[t].weight[0, 0] = 1.0; Lw = S.Lw[t].weight; Lw[0, 1] = om; Lw[0, 0] = -om; Lw[0, S.C] = -om
        S.dec_lin.weight[0, 0] = 1.0
net.set_ops(*ops)
def jac(Ak, b):
    x = np.zeros_like(b); dinv = 1 / Ak.diagonal()
    for _ in range(nu): x = x + om * dinv * (b - Ak @ x)
    return x
def np_vcp(levs, A0, b):
    A = levs[0][0]
    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]; x = jac(Ak, rhs); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else np.linalg.solve(A0.toarray(), rc)
        xc = np.concatenate([ec, X @ ec]); xp = jac(Ak, rhs - Ak @ (x + xc))
        return (x, xc, xp) if parts else x + xc + xp
    W = np.column_stack(cyc(0, b, True)); return W @ np.linalg.solve(W.T @ (A @ W), W.T @ b)
P2 = V.P[L].tocsr(); R = np.array([F - A @ (P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F)) for A in A_list])
with torch.no_grad(): out = net(torch.tensor(R)).numpy()
w = max(np.abs(out[i] - np_vcp(*oa_hierarchy(V, A, 3, m), R[i])).max() / np.abs(out[i]).max() for i, A in enumerate(A_list))
print(f"2. containment (Jacobi weights, Galerkin output): max rel. diff {w:.1e}")
viol, rat = 0, []
for seed in range(3):
    torch.manual_seed(seed); n2 = OAVCycle(D, width=8, nu=nu, project=True, deflate=False)
    for i, A in enumerate(A_list):
        u = spsolve(A.tocsc(), F); e = u - (F - R[i] - 0) * 0 - (u - spsolve(A.tocsc(), R[i]))   # e = A^-1 r
        n2.set_ops(*[[v[i:i + 1] for v in o] if isinstance(o, list) else o[i:i + 1] for o in ops])
        with torch.no_grad(): x = n2(torch.tensor(R[i])[None]).numpy()[0]
        en = lambda v: np.sqrt(v @ (A @ v))
        if en(e - x) > en(e) * (1 + 1e-9): viol += 1
        rat.append(en(e - x) / en(e))
    n2.set_ops(*ops)
    with torch.no_grad(): Rt = torch.tensor(R); f1 = n2(Rt); f3 = n2(2.5 * Rt)
print(f"3. energy guarantee, 3 x {len(te)}: {viol} violations, ratio [{min(rat):.3f}, {max(rat):.3f}] (untrained); "
      f"homogeneity {float((f3 - 2.5 * f1).abs().max() / (2.5 * f1).abs().max()):.1e}")
Ec = {'coarse solve alone': [], 'V(4,4) op.-adapted': [], 'classical member (+ projection)': []}
for i, A in enumerate(A_list):
    u = spsolve(A.tocsc(), F); uc = u - spsolve(A.tocsc(), R[i]); lv_, A0_ = oa_hierarchy(V, A, 3, m)
    en = lambda v: np.sqrt(v @ (A @ v) / (u @ (A @ u)))
    Ec['coarse solve alone'].append(en(uc - u))
    Ec['V(4,4) op.-adapted'].append(en(uc + classical_vcycle(lv_, A0_, R[i], fine_lmax=None) - u))
    Ec['classical member (+ projection)'].append(en(uc + classical_vcycle_proj(lv_, A0_, R[i], Z=None, fine_lmax=None) - u))
print("4. classical one-pass, energy (12 samples):", {k: round(float(np.mean(v)), 4) for k, v in Ec.items()},
      "| earlier learned, same protocol (seed 0): oa-ll-op 0.211, gnn-op 0.196; two-grid PCG 4 its 0.040")
