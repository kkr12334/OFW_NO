"""11 Oct 2026: containment of FIXED POLYNOMIAL smoothers (step-dependent weights w_t), in particular degree-4 Chebyshev on the
fixed interval [lam/10, lam], lam = 3, same weights on every level.  Derived from check_oa_vcp.py (4 Oct), part 1 only: (oa_vcycle.py with project=True, deflate=True).
  1. CONTAINMENT: smoothers set to nu-step weighted Jacobi -> the network equals a numpy cycle with the same deflation,
     transfers, Galerkin levels, exact coarsest solve and fine-level Galerkin projection (round-off).
  2. ENERGY GUARANTEE for random weights: ||e - x||_A <= ||e||_A (never worse than the input u_c) and
     <= ||e - x0||_A (never worse than deflation alone), every sample; plus positive homogeneity.
  3. ALGEBRAIC NEAR-KERNEL DETECTION (strong-connection components of A without Dirichlet-adjacent rows) vs the
     coefficient-based floating islands, on the L3 barrier test set (samples 300-399, rng 1000): same count, same nodes.
Perforated L3, barriers 1e-4 (--load no-low).
11 Oct (Astra polynomial review): 1b. general fixed correction polynomial via the Horner construction (random coefficients whose
residual polynomial has complex roots), network vs numpy; 1c. direct vector comparison of the product-form fixed-interval
Chebyshev smoother with the three-term Chebyshev recurrence on every level operator (no projection involved).
Parts 2 and 3 of the docstring above are not part of this script (see check_oa_vcp.py)."""
import sys, os, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, near_kernel_basis, structural_patterns, _on_pattern
torch.set_default_dtype(torch.float64)

L, m, nu = 3, 4, 4
lam, q = 3.0, 10
th, de = (lam + lam / q) / 2, (lam - lam / q) / 2
WT = [1 / (th - de * np.cos((2 * i - 1) * np.pi / (2 * nu))) for i in range(1, nu + 1)]
print(f'step weights w_t = {np.round(WT, 6)}')
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
            S.V[t].weight[0, 0] = 1.0; Lw = S.Lw[t].weight; Lw[0, 1] = WT[t]; Lw[0, 0] = -WT[t]; Lw[0, S.C] = -WT[t]
        S.dec_lin.weight[0, 0] = 1.0
net.set_ops(*ops)

def jac(Ak, b):
    x = np.zeros_like(b); dinv = 1 / Ak.diagonal()
    for w in WT: x = x + w * dinv * (b - Ak @ x)
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
print(f"1. containment (fixed-interval Chebyshev weights, lam = 3): network vs numpy deflated + projected cycle, max rel. diff {worst:.1e}")

# 11 Oct (POLYNOMIAL_AND_FIGURES_REVIEW_ASTRA, item 1): GENERAL fixed correction polynomial by Horner's rule.
# L_t(s, q) = a_{nu-t} s_1 + q_0 in channel 0 gives x_t = B x_{t-1} + a_{nu-t} z (B = D^-1 A, z = D^-1 b), so x_nu = p(B) z with
# p(mu) = sum_j a_j mu^j of degree <= nu - 1.  Random coefficients (residual polynomial 1 - mu p(mu), possibly complex roots).
AJ = np.random.default_rng(7).normal(size=nu) * np.array([1.5, -1.0, 0.4, -0.05])
print(f'Horner coefficients a_0..a_{nu-1} = {np.round(AJ, 6)}; residual-polynomial roots {np.round(np.roots(np.r_[-AJ[::-1], 1.0]), 4)}')
with torch.no_grad():
    for S in (net.pre[0], net.post[0]):
        for t in range(nu):
            Lw = S.Lw[t].weight; Lw[0, 1] = AJ[nu - 1 - t]; Lw[0, 0] = 0.0; Lw[0, S.C] = 1.0
def horner(Ak, b):
    dinv = 1 / Ak.diagonal(); z = dinv * b; x = np.zeros_like(b)
    for t in range(1, nu + 1): x = dinv * (Ak @ x) + AJ[nu - t] * z
    return x
def powsum(Ak, b):                                                  # sum_j a_j B^j z, evaluated directly
    dinv = 1 / Ak.diagonal(); v = dinv * b; x = np.zeros_like(b)
    for j in range(nu): x = x + AJ[j] * v; v = dinv * (Ak @ v)
    return x
jac_saved = jac; jac = horner
with torch.no_grad(): out = net(torch.tensor(R)).numpy()
worst = 0.0; wh = 0.0
for i, a in enumerate(a_list):
    levs, A0 = oa_hierarchy(H, H.A_fine(a), 3, m); ref = np_vcp(levs, A0, R[i], D.Z[i].double().numpy())
    worst = max(worst, np.abs(out[i] - ref).max() / np.abs(ref).max())
    for Ak, _ in levs:
        bb = np.random.default_rng(i).normal(size=Ak.shape[0]); hx, px = horner(Ak, bb), powsum(Ak, bb)
        wh = max(wh, np.abs(hx - px).max() / np.abs(px).max())
jac = jac_saved
print(f"1b. containment (general correction polynomial, Horner construction, nu = {nu}): network vs numpy deflated + projected cycle, max rel. diff {worst:.1e}")
print(f"    numpy Horner recursion vs direct power sum on every level operator: max rel. diff {wh:.1e}")

# 11 Oct (same review, item 5): DIRECT vector comparison of the fixed-interval Chebyshev smoother, product form (as contained)
# vs the three-term recurrence of the Chebyshev semi-iteration (Saad 2003, Alg. 12.1), same interval [lam/q, lam], x = 0 start,
# on every level operator of the 12 samples; no projection or regularisation involved.
def cheb3(Ak, b):
    dinv = 1 / Ak.diagonal(); s1 = th / de; rho = 1 / s1; x = np.zeros_like(b); r = dinv * b; d = r / th
    for _ in range(nu):
        x = x + d; r = r - dinv * (Ak @ d); rho_n = 1 / (2 * s1 - rho); d = rho_n * rho * d + 2 * rho_n / de * r; rho = rho_n
    return x
wc = 0.0
for i, a in enumerate(a_list):
    levs, A0 = oa_hierarchy(H, H.A_fine(a), 3, m)
    for Ak, _ in levs:
        bb = np.random.default_rng(100 + i).normal(size=Ak.shape[0]); pf, tr = jac(Ak, bb), cheb3(Ak, bb)
        wc = max(wc, np.linalg.norm(pf - tr) / np.linalg.norm(tr))
print(f"1c. fixed-interval Chebyshev smoother vectors, product form vs three-term recurrence, all levels: max rel. diff {wc:.1e}")
