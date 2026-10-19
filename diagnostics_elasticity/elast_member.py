"""Elasticity (Session 4): paired comparison of OA-VCB with its classical MEMBER (one V(4,4) Chebyshev cycle on the same
operator-adapted dof-graph hierarchy + Galerkin output; no deflation, as in OA-VCB for elasticity), the plain V(4,4)
recorded by the driver, and the contained Jacobi member (11 Oct, omega 2/3), two-grid PCG (block-free weighted Jacobi(1,1) + exact level-(L-1) Galerkin) iterations to
OA-VCB's per-sample energy error.  Data rebuilt exactly as train_elasticity.py (rng 2000, bands, contrast 1e2, n 400,
L3, k0 2); the plain cycle must reproduce the driver's stored per-sample errors.  11 Oct 2026.
usage: python diagnostics_elasticity/elast_member.py s4"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
from oa_vcycle import oa_hierarchy, classical_vcycle, classical_vcycle_proj
nu, om = 4, 2 / 3
def jac(Ak, b):
    x = np.zeros_like(b); dinv = 1 / Ak.diagonal()
    for _ in range(nu): x = x + om * dinv * (b - Ak @ x)
    return x
def lam_est(Ak, its=30):
    dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam
def cheb_w(lam, q=10):
    th, de = (lam + lam / q) / 2, (lam - lam / q) / 2
    return [1 / (th - de * np.cos((2 * i - 1) * np.pi / (2 * nu))) for i in range(1, nu + 1)]
def smooth(Ak, b, W):
    x = np.zeros_like(b); dinv = 1 / Ak.diagonal()
    for w_ in W: x = x + w_ * dinv * (b - Ak @ x)
    return x
def gal(A, W, b):
    keep = np.linalg.norm(W, axis=0) > 0; W = W[:, keep]
    return W @ np.linalg.solve(W.T @ (A @ W), W.T @ b) if W.shape[1] else 0 * b
def np_vcp(levs, A0, b, WT=None):                                     # mathcheck/check_oa_vcp.py, no deflation (Z empty)
    A = levs[0][0]
    def cyc(l, rhs, parts=False):
        Ak, X = levs[l]; nc = X.shape[1]; x = jac(Ak, rhs) if WT is None else smooth(Ak, rhs, WT); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else np.linalg.solve(A0.toarray(), rc)
        xc = np.concatenate([ec, X @ ec]); xp = jac(Ak, rhs - Ak @ (x + xc)) if WT is None else smooth(Ak, rhs - Ak @ (x + xc), WT)
        return (x, xc, xp) if parts else x + xc + xp
    return gal(A, np.column_stack(list(cyc(0, b, True))), b)
D = sys.argv[1] if len(sys.argv) > 1 else 's4'
L, K0, N = 3, 2, 400
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
rng = np.random.default_rng(2000); E = [field_channels(H, 1e2, rng) for _ in range(N)]
te = np.arange(300, 400)
Pk = None
for k in range(K0 + 1, L + 1): Pk = V.P[k] if Pk is None else V.P[k] @ Pk
Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
VCL = 0; k_ = L
while k_ >= 1 and VCL < 4 and len(H.nodes[k_ - 1]) >= 16: VCL += 1; k_ -= 1
rec = {}
for f in ('results_elast_L3_n9_bands_c1e+02_400 (2).jsonl', 'results_elast_L3_n9_bands_c1e+02_400.jsonl'):
    for l in open(os.path.join(D, f)):
        r = json.loads(l); c = json.loads(r['cfg']); rec[(r['model'], c['seed'])] = r
for r in rec.values(): assert r['code'] == 'f8653612eb84' and r['result']['sample_ids'] == list(range(300, 400))
e_vcb = np.array(rec[('oa-vcb', 0)]['result']['en_s']); e_plain_rec = np.array(rec[('classical V(4,4) op.-adapted', 0)]['result']['en_s'])
en = lambda A, x, u: float(np.sqrt(((x - u) @ (A @ (x - u))) / (u @ (A @ u))))
LST = max(max(lam_est(Ak) for Ak, _ in oa_hierarchy(V, V.A_fine(E[i]).tocsr(), VCL, 4)[0]) for i in range(60))
print(f"elasticity: lam* (max per-level estimate over 60 training samples) = {LST:.3f}")
ep, em, eu, its, ej, ef3, efs = [], [], [], [], [], [], []
Pt = V.P[L].tocsr()
for j, i in enumerate(te):
    A = V.A_fine(E[i]).tocsr(); u = spsolve(A.tocsc(), V.F)
    uc = Pk @ spsolve((PkT @ A @ Pk).tocsc(), PkT @ V.F); r = V.F - A @ uc
    levs, A0 = oa_hierarchy(V, A, VCL, 4)
    ep.append(en(A, uc + classical_vcycle(levs, A0, r, fine_lmax=None), u))
    em.append(en(A, uc + classical_vcycle_proj(levs, A0, r, Z=None, nu=4, fine_lmax=None), u))
    eu.append(en(A, uc, u)); ej.append(en(A, uc + np_vcp(levs, A0, r), u)); ef3.append(en(A, uc + np_vcp(levs, A0, r, cheb_w(3.0)), u)); efs.append(en(A, uc + np_vcp(levs, A0, r, cheb_w(LST)), u))
    solc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal(); om = 2 / 3
    def M(q):
        x = om * dinv * q; x = x + Pt @ solc(Pt.T @ (q - A @ x)); return x + om * dinv * (q - A @ x)
    x = uc.copy(); rr = V.F - A @ x; z = M(rr); p = z.copy(); rz = rr @ z; k = 0
    while en(A, x, u) > e_vcb[j] and k < 60:
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; rr = rr - al * Ap; z = M(rr); rz2 = rr @ z; p = z + (rz2 / rz) * p; rz = rz2; k += 1
    its.append(k)
ep, em, eu, its, ej, ef3, efs = map(np.array, (ep, em, eu, its, ej, ef3, efs))
def G(ref, mod):
    lr = np.log(ref / mod); b = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr)))
    return np.exp(lr.mean()), np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5])), float(np.mean(lr > 0))
print(f"plain V(4,4) reproduces the driver's record: max rel diff {np.max(np.abs(ep / e_plain_rec - 1)):.1e}")
print(f"mean energy error: u_c {eu.mean():.4e}  plain V(4,4) {ep.mean():.4e}  member (V(4,4)+Galerkin output) {em.mean():.4e}  OA-VCB {e_vcb.mean():.4e}")
for lab, ref in (('member', em), ('plain V(4,4)', ep), ('Jacobi member (contained)', ej), ('fixed Chebyshev lam=3 (contained)', ef3), ('fixed Chebyshev lam* (contained)', efs)):
    g = G(ref, e_vcb); print(f"{lab}/OA-VCB: G {g[0]:.2f} [{g[1][0]:.2f}, {g[1][1]:.2f}]  W {g[2]:.2f}  worst ratio {np.min(ref / e_vcb):.2f}")
for k in ('oa-ll-op', 'gnn-op'):
    es = [np.array(rec[(k, s)]['result']['en_s']) for s in (0, 1, 2)]
    print(f"{k} (init seeds 0-2): mean energy {[round(float(e.mean()), 4) for e in es]};  member/{k} G " +
          ", ".join(f"{G(em, e)[0]:.2f}" for e in es) + ";  OA-VCB/" + k + " G " + ", ".join(f"{G(e, e_vcb)[0]:.2f}" for e in es))
print(f"two-grid PCG iterations to OA-VCB's per-sample energy error: median {np.median(its):.0f} (range {its.min()}-{its.max()}), censored >60: {(its >= 60).sum()}")
json.dump(dict(sample_ids=te.tolist(), uc=eu.tolist(), plain=ep.tolist(), member=em.tolist(), jacobi=ej.tolist(), fixed3=ef3.tolist(), fixedstar=efs.tolist(), lam_star=LST, oa_vcb=e_vcb.tolist(), pcg_its=its.tolist()),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'elast_member.json'), 'w'))
