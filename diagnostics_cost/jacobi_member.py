"""11 Oct 2026 (FULL_DRAFT_REVIEW_ASTRA item 1): the classical cycle that the architecture provably contains
(Proposition containment: nu = 4 steps of weighted Jacobi, omega = 2/3, on the same operator-adapted hierarchy, Galerkin
levels, exact coarsest solve, near-kernel deflation and Galerkin output) -- the numpy reference np_vcp of
mathcheck/check_oa_vcp.py, which matches the network with Jacobi weights to 3.6e-10.  Evaluated on the test samples
300-399 of the main configurations, with the same u_c; also re-evaluates the Chebyshev reference (classical_vcycle_proj)
as a check against the stored records.  Writes diagnostics_cost/jacobi_member_<tag>.json.
usage: python diagnostics_cost/jacobi_member.py tag L k0 data_seed contrast load"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
tag, L, k0, DS, con, load = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6]
nu, om = 4, 2 / 3
S = setup(L, k0, DS, contrast=con, load=load)

def jac(Ak, b):
    x = np.zeros_like(b); dinv = 1 / Ak.diagonal()
    for _ in range(nu): x = x + om * dinv * (b - Ak @ x)
    return x
def gal(A, W, b):
    keep = np.linalg.norm(W, axis=0) > 0; W = W[:, keep]
    return W @ np.linalg.solve(W.T @ (A @ W), W.T @ b) if W.shape[1] else 0 * b
def np_vcp(levs, A0, b, Z):                                  # verbatim from mathcheck/check_oa_vcp.py
    A = levs[0][0]; x0 = gal(A, Z, b); bb = b - A @ x0
    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]; x = jac(Ak, rhs); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else np.linalg.solve(A0.toarray(), rc)
        xc = np.concatenate([ec, X @ ec]); xp = jac(Ak, rhs - Ak @ (x + xc))
        return (x, xc, xp) if parts else x + xc + xp
    return x0 + gal(A, np.column_stack(list(cyc(0, bb, True)) + [Z]), bb)

en = lambda A, x, u: float(np.sqrt(((x - u) @ (A @ (x - u))) / (u @ (A @ u))))
EJ, EC = [], []
for i in S['te']:
    A = S['A'][i]; u = S['U'][i]; uc = S['UC'][i]; r = S['Fs'][i] - A @ uc
    levs, A0 = oa_hierarchy(S['H'], A, S['nl'], 4); Z = near_kernel_basis(levs[0][0])
    EJ.append(en(A, uc + np_vcp(levs, A0, r, Z), u))
    EC.append(en(A, uc + classical_vcycle_proj(levs, A0, r, Z=Z, nu=4), u))
EJ, EC = np.array(EJ), np.array(EC)
print(f"{tag}: Jacobi member mean energy {EJ.mean():.4e}; Chebyshev reference {EC.mean():.4e}; "
      f"Chebyshev better on {np.mean(EC < EJ):.2f} of samples, G(Jacobi/Chebyshev) {np.exp(np.mean(np.log(EJ / EC))):.2f}")
json.dump(dict(tag=tag, L=L, k0=k0, data_seed=DS, contrast=con, load=load, omega=om, nu=nu, sample_ids=[int(i) for i in S['te']],
               jacobi=EJ.tolist(), chebyshev=EC.tolist()), open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'jacobi_member_{tag}.json'), 'w'))
