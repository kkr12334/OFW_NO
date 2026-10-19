"""Accuracy versus time on ONE CPU (12 Oct 2026; reviews of 12 Oct: "cost-matched Pareto plot" and "stronger classical
smoother").  Everything below is measured in this one run, on this machine, with 2 threads -- OA-VCB included -- so the
points of the figure share one platform.  It does NOT replace Table cost (timing_one_platform.py, another CPU): the
figure is a separate measurement and its caption names its own platform.

Main configuration: L3, k0 2, channels 1e4, held-out data seed 1, test ids 300-399, Session-1 checkpoint 40ce3ac1f9.
Per method: mean relative energy error over the 100 test samples and mean time per sample, both from u_c (the level-k0
Galerkin solve, timed and included), set-up included, fine-matrix assembly excluded (common to all).
  * OA-VCB: operator data + inference (batch 8), as in timing_one_platform.py.
  * classical reference (deflation + one V-cycle + Galerkin output) with Chebyshev smoothing of degree nu = 4 (the paper's
    reference), 8 and 16 (stronger classical smoothers), and with symmetric Gauss-Seidel (pyamg, 2 symmetric sweeps pre and
    post = 4 sweeps each);
  * the nu = 4 reference applied k = 1..4 times (x <- x + member(F - A x); each application is energy-non-increasing);
  * PCG with the two-grid, SA-AMG (pyamg defaults) and classical operator-adapted V(4,4) preconditioners after n
    iterations (errors from an untimed run, times from a timed run with exactly n iterations);
  * sparse direct (SuperLU), exact.
Every time is the per-sample minimum over REP passes.  No error is evaluated inside a timer.
usage: python diagnostics_cost/accuracy_vs_time.py CKPT [results.jsonl] [n_test] [repeats] [out.json] [channels|barriers]
(barriers: contrast 1e-4, load no-low, per-sample loads; checkpoint 8266a233d9 for held-out seed 1)"""
import sys, os, time, json, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
from scipy.sparse.linalg import splu, factorized
from oa_vcycle import OAData, oa_hierarchy, near_kernel_basis, classical_vcycle, classical_vcycle_proj, structural_patterns
import pyamg
from pyamg.relaxation.relaxation import gauss_seidel
torch.set_num_threads(2)
CK = sys.argv[1]; RES = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != '-' else None
NT = int(sys.argv[3]) if len(sys.argv) > 3 else 100; REP = int(sys.argv[4]) if len(sys.argv) > 4 else 2
OUT = sys.argv[5] if len(sys.argv) > 5 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'accuracy_vs_time.json')
FAM = sys.argv[6] if len(sys.argv) > 6 else 'channels'
L, k0, DS = 3, 2, 1
pc = time.perf_counter
import platform, subprocess
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), platform.processor())
S = setup(L, k0, DS) if FAM == 'channels' else setup(L, k0, DS, contrast=1e-4, load='no-low'); H = S['H']
te = S['te'][:NT]; FS = [S['Fs'][i] for i in te]; As = [S['A'][i] for i in te]; U = S['U'][te]; UC = S['UC'][te]; P = S['P']; PT = P.T.tocsr()
print(f"{FAM}: L{L}, k0 {k0}, held-out data seed {DS}, {len(FS[0])} unknowns, {len(te)} test samples, CPU {cpu}, "
      f"{torch.get_num_threads()} threads, min over {REP} repeats", flush=True)
def best(fn):
    T = np.full(len(te), np.inf)
    for _ in range(REP):
        for j in range(len(te)): T[j] = min(T[j], fn(j))
    return T
def t_uc(j):
    t = pc(); P @ factorized((PT @ As[j] @ P).tocsc())(PT @ FS[j]); return pc() - t
T_uc = best(t_uc)
R = np.array([Fj - A @ uc for Fj, A, uc in zip(FS, As, UC)])
res = dict(family=FAM, cpu=cpu, threads=2, n_test=len(te), repeats=REP, uc_ms=float(T_uc.mean() * 1e3), methods={})
def put(name, e, T, **kw):
    res['methods'][name] = dict(energy_mean=float(np.mean(e)), energy_s=[float(x) for x in np.atleast_1d(e)],
                                ms_mean=float(np.mean(T) * 1e3), ms_s=[float(x * 1e3) for x in np.atleast_1d(T)], **kw)
    print(f"  {name:<52} energy {np.mean(e):.3e}   {np.mean(T) * 1e3:9.2f} ms", flush=True)

# ---- OA-VCB
t = pc(); structural_patterns(H, S['nl'], 4); T_pat = pc() - t
T_od = np.inf
for _ in range(REP):
    t = pc(); OAData(H, None, S['nl'], 4, 'cpu', H.level_coords, near_kernel=True, A_list=As); T_od = min(T_od, (pc() - t - T_pat) / len(te))
D, net, st = load_net(S, CK, te)
C8 = apply(net, D, R, S['U_scale'], bs=8)
e_net = np.array([energy(A, uc + c, u) for A, uc, c, u in zip(As, UC, C8, U)])
T_i8 = np.inf
for _ in range(REP):
    t = pc(); apply(net, D, R, S['U_scale'], bs=8); T_i8 = min(T_i8, (pc() - t) / len(te))
chk = None
if RES:
    for l in open(RES):
        r = json.loads(l)
        if r['model'] == 'oa-vcb' and json.loads(r['cfg']).get('data_seed', 0) == DS and json.loads(r['cfg'])['k0'] == k0 and json.loads(r['cfg']).get('seed', 0) == 0 and json.loads(r['cfg']).get('contrast') == (1e4 if FAM == 'channels' else 1e-4):
            chk = float(np.max(np.abs(e_net / np.array(r['result']['en_s'])[:len(te)] - 1)))
print(f"OA-VCB stored record reproduced to max rel diff {chk}", flush=True)
put('OA-VCB (one pass)', e_net, T_uc + T_od + T_i8, opdata_ms=T_od * 1e3, inference_ms=T_i8 * 1e3, record_check=chk)

# ---- classical reference with set-up / apply split (as timing_one_platform.VCycle), smoother selectable
class VCycle:
    def __init__(self, A, Z=None, q=10, nu=4, smoother='cheb'):
        self.levs, A0 = oa_hierarchy(H, A, S['nl'], 4); self.lu = splu(A0.tocsc()); self.q, self.nu, self.sm = q, nu, smoother
        self.dinv = [1 / Ak.diagonal() for Ak, _ in self.levs]
        self.lm = [3.0] + [self._lam(Ak) for Ak, _ in self.levs[1:]] if smoother == 'cheb' else None
        self.Acsr = [Ak.tocsr() for Ak, _ in self.levs] if smoother == 'sgs' else None
        self.A = self.levs[0][0]; self.Z = Z if (Z is not None and Z.shape[1]) else None
    def _lam(self, Ak, its=30):
        dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
        for _ in range(its):
            w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
        return 1.05 * lam
    def _smooth(self, l, rhs, x):
        if self.sm == 'sgs':
            x = np.array(x, dtype=float, copy=True); gauss_seidel(self.Acsr[l], x, np.asarray(rhs, float), iterations=self.nu // 2, sweep='symmetric'); return x
        Ak = self.levs[l][0]; dinv = self.dinv[l]; lo, hi = self.lm[l] / self.q, self.lm[l]; th, de = (hi + lo) / 2, (hi - lo) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(self.nu - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        return x
    def cyc(self, l, rhs, parts=False):
        Ak, X = self.levs[l]; nc = X.shape[1]
        x = self._smooth(l, rhs, np.zeros_like(rhs)); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = self.cyc(l + 1, rc) if l + 1 < len(self.levs) else self.lu.solve(rc)
        xc = np.concatenate([ec, X @ ec])
        if parts: return x, xc, self._smooth(l, rhs - Ak @ (x + xc), np.zeros_like(rhs))
        return self._smooth(l, rhs, x + xc)
    def _gal(self, W, rhs):
        G = W.T @ (self.A @ W); dg = np.diag(G); live = dg > 1e-30 * dg.max(); sc = np.where(live, 1 / np.sqrt(np.where(live, dg, 1)), 0)
        return W @ (sc * np.linalg.solve(sc[:, None] * G * sc[None, :] + np.diag((~live) + 1e-10), sc * (W.T @ rhs)))
    def member(self, b):
        x0 = self._gal(self.Z, b) if self.Z is not None else np.zeros_like(b); bb = b - self.A @ x0
        W = np.column_stack(list(self.cyc(0, bb, parts=True)) + ([self.Z] if self.Z is not None else []))
        return x0 + self._gal(W, bb)
for j in (0, 1):                                            # the nu = 4 Chebyshev member equals the paper's reference
    A = As[j]; levs, A0 = oa_hierarchy(H, A, S['nl'], 4); Z = near_kernel_basis(A)
    d2 = np.abs(VCycle(A, Z).member(R[j]) - classical_vcycle_proj(levs, A0, R[j], Z=Z, nu=4)).max() / np.abs(R[j]).max()
    assert d2 < 1e-10, d2
def member_run(nu, sm, k):
    def f(j, timed=True):
        t = pc(); V = VCycle(As[j], near_kernel_basis(As[j]), nu=nu, smoother=sm); x = UC[j].copy()
        for _ in range(k): x = x + V.member(FS[j] - As[j] @ x)
        return (pc() - t) if timed else x
    return f
for nu, sm, k, name in [(4, 'cheb', 1, 'reference, Chebyshev nu=4 (paper)'), (8, 'cheb', 1, 'reference, Chebyshev nu=8'),
                        (16, 'cheb', 1, 'reference, Chebyshev nu=16'), (4, 'sgs', 1, 'reference, symmetric Gauss-Seidel 4 sweeps'),
                        (4, 'cheb', 2, 'reference nu=4, applied 2 times'), (4, 'cheb', 3, 'reference nu=4, applied 3 times'),
                        (4, 'cheb', 4, 'reference nu=4, applied 4 times')]:
    f = member_run(nu, sm, k)
    e = np.array([energy(As[j], f(j, timed=False), U[j]) for j in range(len(te))])
    put(name, e, T_uc + best(f), nu=nu, smoother=sm, applications=k)

# ---- PCG curves
def pcg(A, x0, M, nit, u=None, F=None):
    x = x0.copy(); r = F - A @ x; z = M(r); p = z.copy(); rz = r @ z; h = [energy(A, x, u)] if u is not None else None
    for _ in range(nit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        if u is not None: h.append(energy(A, x, u))
    return x, h
def tg_setup(A):
    Pt = H.P[L].tocsr(); solc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal(); om = 2 / 3
    def M(r):
        x = om * dinv * r; x = x + Pt @ solc(Pt.T @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    return M
def amg_setup(A):
    Mm = pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle='V'); return lambda r: Mm @ r
def vc_setup(A):
    V = VCycle(A); return lambda r: V.cyc(0, r)
for name, s, NMAX in [('two-grid PCG', tg_setup, 12), ('SA-AMG PCG', amg_setup, 10), ('PCG + classical V(4,4)', vc_setup, 5)]:
    H_ = np.array([pcg(A, uc, s(A), NMAX, u, F=Fj)[1] for A, uc, u, Fj in zip(As, UC, U, FS)])   # (n_test, NMAX + 1), untimed
    curve = []
    for n in range(1, NMAX + 1):
        def run(j, n=n, s=s):
            t = pc(); pcg(As[j], UC[j], s(As[j]), n, F=FS[j]); return pc() - t
        T = T_uc + best(run); curve.append(dict(its=n, energy_mean=float(H_[:, n].mean()), ms_mean=float(T.mean() * 1e3)))
        print(f"  {name:<30} {n:2d} its: energy {H_[:, n].mean():.3e}   {T.mean() * 1e3:8.2f} ms", flush=True)
    res['methods'][name] = dict(curve=curve)
def direct(j):
    t = pc(); splu(As[j].tocsc()).solve(FS[j]); return pc() - t
T_dir = best(direct); res['methods']['sparse direct (SuperLU)'] = dict(ms_mean=float(T_dir.mean() * 1e3), energy_mean=0.0)
print(f"  sparse direct (SuperLU): {T_dir.mean() * 1e3:.2f} ms", flush=True)
json.dump(res, open(OUT, 'w'), indent=1); print('written', OUT)
