"""4 Oct: checks for oa-vcp (oa_vcycle.py with project=True, deflate=True).
  1. CONTAINMENT: smoothers set to nu-step weighted Jacobi -> the network equals a numpy cycle with the same deflation,
     transfers, Galerkin levels, exact coarsest solve and fine-level Galerkin projection (round-off).
  2. ENERGY GUARANTEE for random weights: ||e - x||_A <= ||e||_A (never worse than the input u_c) and
     <= ||e - x0||_A (never worse than deflation alone), every sample; plus positive homogeneity.
  3. ALGEBRAIC NEAR-KERNEL DETECTION (strong-connection components of A without Dirichlet-adjacent rows) vs the
     coefficient-based floating islands, on the L3 barrier test set (samples 300-399, rng 1000): same count, same nodes.
Perforated L3, barriers 1e-4 (--load no-low)."""
import sys, os, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, near_kernel_basis, structural_patterns, _on_pattern
torch.set_default_dtype(torch.float64)

L, m, nu, om = 3, 4, 4, 2 / 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]

rng = np.random.default_rng(1000); A_all = [field_channels(H, 1e-4, rng) for _ in range(400)]
a_list = [A_all[i] for i in range(300, 312)]                       # 12 test samples (several with islands)
D = OAData(H, a_list, 3, m, 'cpu', H.level_coords, near_kernel=True)
pats, _ = structural_patterns(H, 3, m); Av, Xv, A0i = [[] for _ in pats], [[] for _ in pats], []
for a in a_list:                                                   # float64 operator data for exact comparisons
    levs, A0 = oa_hierarchy(H, H.A_fine(a), 3, m)
    for l, ((Ak, X), (pA, pX)) in enumerate(zip(levs, pats)):
        Av[l].append(_on_pattern(Ak, pA)[0]); Xv[l].append(_on_pattern(X, pX)[0])
    A0i.append(np.linalg.inv(A0.toarray()))
ops = ([torch.tensor(np.array(v)) for v in Av], [torch.tensor(np.array(v)) for v in Xv], torch.tensor(np.array(A0i)), D.Z.double())
print(f"near-kernel vectors per sample: {D.k_nk}")

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

R = np.array([load(a >= 0.5) for a in a_list])
with torch.no_grad(): out = net(torch.tensor(R)).numpy()
worst = 0.0
for i, a in enumerate(a_list):
    levs, A0 = oa_hierarchy(H, H.A_fine(a), 3, m); ref = np_vcp(levs, A0, R[i], D.Z[i].double().numpy())
    worst = max(worst, np.abs(out[i] - ref).max() / np.abs(ref).max())
print(f"1. containment (Jacobi weights): network vs numpy deflated + projected cycle, max rel. diff {worst:.1e}")

# 2. energy guarantee and homogeneity, random weights, from the geometric u_c (as in training)
P2 = H.P[3].tocsr(); viol = 0; ratios = []
for seed in range(3):
    torch.manual_seed(seed); net2 = OAVCycle(D, width=8, nu=nu, project=True, deflate=True); net2.set_ops(*ops)
    for i, a in enumerate(a_list):
        A = H.A_fine(a).tocsr(); F = R[i]; u = spsolve(A.tocsc(), F)
        uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); r = F - A @ uc; e = u - uc
        net2.set_ops(*[[v[i:i + 1] for v in o] if isinstance(o, list) else o[i:i + 1] for o in ops])   # this sample's operator
        with torch.no_grad(): x = net2(torch.tensor(r)[None]).numpy()[0]
        en = lambda v: np.sqrt(v @ (A @ v))
        x0 = gal(A, D.Z[i].double().numpy(), r)
        if en(e - x) > en(e) * (1 + 1e-9) or en(e - x) > en(e - x0) * (1 + 1e-9): viol += 1
        ratios.append(en(e - x) / en(e))
    net2.set_ops(*ops)
    with torch.no_grad():
        Rt = torch.tensor(R); f1 = net2(Rt); f3 = net2(2.5 * Rt)
print(f"2. energy guarantee, 3 random weight sets x 12 samples: {viol} violations; ||e - x||_A / ||e||_A in "
      f"[{min(ratios):.3f}, {max(ratios):.3f}] (untrained);  homogeneity |f(2.5 r) - 2.5 f(r)| / |2.5 f(r)| = "
      f"{float((f3 - 2.5 * f1).abs().max() / (2.5 * f1).abs().max()):.1e}")

# 3. algebraic detection vs coefficient islands, all 100 test samples
dirn = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L]); pos = -np.ones(len(H.coords), int); pos[H.nodes[L]] = np.arange(len(H.nodes[L]))
def coef_islands(a):
    cond = H.keep & (a >= 0.5); el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M.T @ M).tocsr(); used = np.unique(el.ravel()); n_, lab = connected_components(G[used][:, used], directed=False)
    touch = np.zeros(n_, bool); touch[lab[np.isin(used, dirn)]] = True
    return [set(pos[used[lab == c]]) for c in np.nonzero(~touch)[0]]
nk, nci, rqs, jac_ = [], [], [], []
for i in range(300, 400):
    a = A_all[i]; Z = near_kernel_basis(H.A_fine(a)); ci = coef_islands(a)
    nk.append(Z.shape[1]); nci.append(len(ci))
    for s_ in ci:
        z = np.zeros(Z.shape[0]); z[list(s_)] = 1.0
        res = z - Z @ np.linalg.lstsq(Z, z, rcond=None)[0] if Z.shape[1] else z
        jac_.append(np.linalg.norm(res) / np.linalg.norm(z))
    A_ = H.A_fine(a); d_ = A_.diagonal()
    rqs += [Z[:, j] @ (A_ @ Z[:, j]) / (Z[:, j] @ (d_ * Z[:, j])) for j in range(Z.shape[1])]
print(f"3. near-kernel detection from A alone vs coefficient islands (100 test samples): {sum(nci)} coefficient islands, "
      f"{sum(nk)} detected vectors; distance of each island indicator from span(Z): max {max(jac_):.1e} "
      f"(0 = contained); Rayleigh quotients of the detected vectors: max {max(rqs):.1e}, median {np.median(rqs):.1e}")

# 3b (5 Oct, review): precision as well as recall.  Ground truth = the coefficient geometry (conductive elements a >= 0.5,
# connected components not touching a Dirichlet node), independent of the A-only detector.  A detected vector counts as a
# true positive if >= 90% of its nodes lie in one coefficient island.
tp = fp = 0; fp_rq = []
for i in range(300, 400):
    a = A_all[i]; A_ = H.A_fine(a); Z = near_kernel_basis(A_); ci = coef_islands(a); d_ = A_.diagonal()
    for j in range(Z.shape[1]):
        S = set(np.nonzero(Z[:, j])[0])
        best = max((len(S & s_) / len(S) for s_ in ci), default=0.0)
        if best >= 0.9: tp += 1
        else: fp += 1; fp_rq.append(Z[:, j] @ (A_ @ Z[:, j]) / (Z[:, j] @ (d_ * Z[:, j])))
covered = sum(1 for d in jac_ if d < 1e-8)
print(f"3b. detector vs coefficient-geometry ground truth: recall {covered}/{len(jac_)} islands exactly in span(Z); precision "
      f"{tp}/{tp + fp} detected vectors inside an island" + (f"; the {fp} others have Rayleigh quotients "
      f"{np.round(fp_rq, 4).tolist()} (near-kernel by definition, not coefficient islands)" if fp else ""))
