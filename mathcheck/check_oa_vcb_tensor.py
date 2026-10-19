"""5 Oct: oa-vcb on a TENSOR coefficient (anisotropic tows, insulated pores; train_scalar.py --pde aniso).
  0. the structural pattern from a random tensor coefficient covers every tow sample (no entry outside the pattern);
  1. CONTAINMENT: smoothers set to nu-step weighted Jacobi -> the network equals the numpy deflated + projected cycle;
  2. ENERGY GUARANTEE for random weights, from the Galerkin u_c (k0 = L-1), and positive homogeneity;
  3. near-kernel detection from A alone on the tow samples (what it finds with insulated pores, Rayleigh quotients);
  4. the classical one-pass members on the same samples (V(4,4) with power-iteration lambda_max on every level).
Level 3, contrast (k_L) 1e2, k_T 5, as the 3 Oct baselines."""
import sys, os, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from scalar_problems import TensorProblem, field_tows
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, near_kernel_basis, structural_patterns, _on_pattern, \
    classical_vcycle, classical_vcycle_proj
torch.set_default_dtype(torch.float64)

L, m, nu, om = 3, 4, 4, 2 / 3
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); TP = TensorProblem(H); F = TP.F
g = np.random.default_rng(54321)
Ks = np.stack([g.uniform(1, 2, H.n_elem), g.uniform(-0.5, 0.5, H.n_elem), g.uniform(1, 2, H.n_elem)], 1)
A_struct = TP.A_fine(Ks)
rng = np.random.default_rng(3000); K_all = [field_tows(H, 1e2, rng, k_T=5.0) for _ in range(400)]
te = list(range(300, 312)); A_list = [TP.A_fine(K_all[i]).tocsr() for i in te]
D = OAData(H, None, 3, m, 'cpu', H.level_coords, near_kernel=True, A_list=A_list, A_struct=A_struct)
print(f"0. pattern: all {len(te)} tow samples on the structural pattern (an entry outside would have raised); "
      f"near-kernel vectors per sample {D.k_nk}")
pats, _ = structural_patterns(H, 3, m, A_struct); Av, Xv, A0i = [[] for _ in pats], [[] for _ in pats], []
for A in A_list:
    levs, A0 = oa_hierarchy(H, A, 3, m)
    for l, ((Ak, X), (pA, pX)) in enumerate(zip(levs, pats)):
        Av[l].append(_on_pattern(Ak, pA)[0]); Xv[l].append(_on_pattern(X, pX)[0])
    A0i.append(np.linalg.inv(A0.toarray()))
ops = ([torch.tensor(np.array(v)) for v in Av], [torch.tensor(np.array(v)) for v in Xv], torch.tensor(np.array(A0i)), D.Z.double())

net = OAVCycle(D, width=4, nu=nu, project=True, deflate=True)
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
def gal(A, W, b):
    keep = np.linalg.norm(W, axis=0) > 0; W = W[:, keep]
    return W @ np.linalg.solve(W.T @ (A @ W), W.T @ b) if W.shape[1] else 0 * b
def np_vcp(levs, A0, b, Z):
    A = levs[0][0]; x0 = gal(A, Z, b); bb = b - A @ x0
    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]; x = jac(Ak, rhs); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else np.linalg.solve(A0.toarray(), rc)
        xc = np.concatenate([ec, X @ ec]); xp = jac(Ak, rhs - Ak @ (x + xc))
        return (x, xc, xp) if parts else x + xc + xp
    return x0 + gal(A, np.column_stack(list(cyc(0, bb, True)) + [Z]), bb)
P2 = H.P[L].tocsr()
R = []
for A in A_list:
    uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); R.append(F - A @ uc)
R = np.array(R)
with torch.no_grad(): out = net(torch.tensor(R)).numpy()
worst = 0.0
for i, A in enumerate(A_list):
    levs, A0 = oa_hierarchy(H, A, 3, m); ref = np_vcp(levs, A0, R[i], D.Z[i].double().numpy())
    worst = max(worst, np.abs(out[i] - ref).max() / np.abs(ref).max())
print(f"1. containment (Jacobi weights): network vs numpy deflated + projected cycle, max rel. diff {worst:.1e}")

viol, ratios = 0, []
for seed in range(3):
    torch.manual_seed(seed); net2 = OAVCycle(D, width=8, nu=nu, project=True, deflate=True)
    for i, A in enumerate(A_list):
        u = spsolve(A.tocsc(), F); uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); e = u - uc
        net2.set_ops(*[[v[i:i + 1] for v in o] if isinstance(o, list) else o[i:i + 1] for o in ops])
        with torch.no_grad(): x = net2(torch.tensor(R[i])[None]).numpy()[0]
        en = lambda v: np.sqrt(v @ (A @ v)); x0 = gal(A, D.Z[i].double().numpy(), R[i])
        if en(e - x) > en(e) * (1 + 1e-9) or en(e - x) > en(e - x0) * (1 + 1e-9): viol += 1
        ratios.append(en(e - x) / en(e))
    net2.set_ops(*ops)
    with torch.no_grad():
        Rt = torch.tensor(R); f1 = net2(Rt); f3 = net2(2.5 * Rt)
print(f"2. energy guarantee, 3 random weight sets x {len(te)} samples: {viol} violations; ratio in [{min(ratios):.3f}, "
      f"{max(ratios):.3f}] (untrained); homogeneity {float((f3 - 2.5 * f1).abs().max() / (2.5 * f1).abs().max()):.1e}")

nk, rqs = [], []
for i in range(300, 400):
    A = TP.A_fine(K_all[i]).tocsr(); Z = near_kernel_basis(A); nk.append(Z.shape[1]); d = A.diagonal()
    rqs += [Z[:, j] @ (A @ Z[:, j]) / (Z[:, j] @ (d * Z[:, j])) for j in range(Z.shape[1])]
print(f"3. near-kernel detection on 100 tow test samples: {sum(nk)} vectors, {np.mean(np.array(nk) > 0) * 100:.0f}% of samples "
      f"have one" + (f", Rayleigh quotients max {max(rqs):.1e}" if rqs else ""))

E = {'V(4,4) op.-adapted': [], 'classical member (+defl.+proj.)': []}
for i, A in enumerate(A_list):
    u = spsolve(A.tocsc(), F); uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); levs, A0 = oa_hierarchy(H, A, 3, m)
    en = lambda v: np.sqrt(v @ (A @ v) / (u @ (A @ u)))
    E['V(4,4) op.-adapted'].append(en(uc + classical_vcycle(levs, A0, R[i], fine_lmax=None) - u))
    E['classical member (+defl.+proj.)'].append(en(uc + classical_vcycle_proj(levs, A0, R[i], Z=near_kernel_basis(A), fine_lmax=None) - u))
    if i == 0: E.setdefault('coarse solve alone', [])
    E['coarse solve alone'].append(en(uc - u))
print("4. classical one-pass on the same 12 samples (energy, from the Galerkin u_c):",
      {k: round(float(np.mean(v)), 4) for k, v in E.items()})
