"""One-platform (this CPU, 2 threads) cost of the TRAINED oa-vcb against classical solvers, per test sample.
Version of 10 Oct 2026 (fixes requested in STATUS_2026-10-10 / RESULTS_REVIEW_2026-10-10; the 5 Oct version is kept as
timing_one_platform_pre10oct.py and produced timing_one_platform_L3_k00.log / _L4_k03.log):
  1. held-out data: coefficients from rng 1000 + data_seed (the 5 Oct version always used rng 1000 = development seed 0);
  2. the driver's inference path: float64 residual scaled by U_scale (training rms of U - U_c), float64 assembled fine
     operator (A64) for deflation + Galerkin output -- the stored per-sample energies are reproduced (printed check);
  3. no error evaluation inside any timer: for each sample the iteration count needed to reach oa-vcb's energy error on
     that sample is found first (untimed), then a run with exactly that many iterations is timed;
  4. classical V-cycles timed with an explicit setup / apply split (Chebyshev bounds and the coarsest LU computed once per
     operator, as any multigrid code does; oa_vcycle.classical_vcycle recomputes both at every call, which is fine for
     accuracy studies but not for timing).  The split implementation is checked against classical_vcycle(_proj);
  5. every timed quantity is the minimum over REPEATS passes over the test set (per sample), reported as mean (median).
Revision of 11 Oct 2026 (SECTIONS_4_5_FOLLOWUP_ASTRA_2026-10-11, point 3): the 10 Oct version timed OAData(a_list), whose
constructor re-assembles every fine matrix (H.A_fine(a)), while the classical rows start from the assembled matrices.  Now
OAData is timed with A_list = As (the same assembled matrices the classical rows use; checked to give identical operator
data), and fine-matrix assembly is timed separately and reported as a common cost excluded from every row.  The 10 Oct
version is kept as timing_one_platform_pre11oct.py with its logs timing_main_L{3,4}_*_seed1.log.
CPU only.  GPU figures (Kaggle T4) come from the driver logs and are reported separately, never mixed into this table.
usage: python diagnostics_cost/timing_one_platform.py L k0 data_seed ckpt [results.jsonl] [n_test] [repeats]"""
import sys, os, time, json, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
from scipy.sparse.linalg import splu, factorized
from oa_vcycle import OAData, oa_hierarchy, near_kernel_basis, classical_vcycle, classical_vcycle_proj, structural_patterns
import pyamg
torch.set_num_threads(2)

L, k0, DS, CK = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
RES = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] != '-' else None
NT = int(sys.argv[6]) if len(sys.argv) > 6 else 100
REP = int(sys.argv[7]) if len(sys.argv) > 7 else 3
pc = time.perf_counter

S = setup(L, k0, DS); H, F = S['H'], S['F']
te = S['te'][:NT]; As = [S['A'][i] for i in te]; U = S['U'][te]; UC = S['UC'][te]; P = S['P']
print(f"L{L}, k0 {k0}, held-out data seed {DS}, {len(F)} unknowns, {len(te)} test samples (ids {te[0]}-{te[-1]}), "
      f"CPU {torch.get_num_threads()} threads, min over {REP} repeats")


def best(fn):
    """per-sample minimum over REP passes; fn(j) returns elapsed seconds for sample j."""
    T = np.full(len(te), np.inf)
    for _ in range(REP):
        for j in range(len(te)): T[j] = min(T[j], fn(j))
    return T


fmt = lambda T: f"{T.mean() * 1e3:8.2f} ({np.median(T) * 1e3:.2f})"

# ---- u_c: Galerkin solve on the level-k0 mesh (shared input of every method below)
if P is not None:
    PT = P.T.tocsr()
    def t_uc(j):
        t = pc(); P @ factorized((PT @ As[j] @ P).tocsc())(PT @ F); return pc() - t
    T_uc = best(t_uc)
else:
    T_uc = np.zeros(len(te))

# ---- oa-vcb: operator data, then inference (batch 8 and batch 1)
t = pc(); pats = structural_patterns(H, S['nl'], 4); T_pat = pc() - t        # once per mesh hierarchy (not per sample)
T_od, T_od_asm = np.inf, np.inf
for _ in range(REP):
    t = pc(); D = OAData(H, None, S['nl'], 4, 'cpu', H.level_coords, near_kernel=True, A_list=As)
    T_od = min(T_od, (pc() - t - T_pat) / len(te))
    t = pc(); D2 = OAData(H, [S['a'][i] for i in te], S['nl'], 4, 'cpu', H.level_coords, near_kernel=True)
    T_od_asm = min(T_od_asm, (pc() - t - T_pat) / len(te))
same = max(max(float((x - y).abs().max()) for x, y in zip(D.Av, D2.Av)), float((D.A64 - D2.A64).abs().max()), float((D.Z - D2.Z).abs().max()))
del D2
def t_asm(j):
    t = pc(); H.A_fine(S['a'][te[j]]).tocsr(); return pc() - t
T_asm = best(t_asm)
D, net, st = load_net(S, CK, te)
R = np.array([F - A @ uc for A, uc in zip(As, UC)])
C8 = apply(net, D, R, S['U_scale'], bs=8)
e_net = np.array([energy(A, uc + c, u) for A, uc, c, u in zip(As, UC, C8, U)])
T_i8, T_i1 = np.inf, np.inf
for _ in range(REP):
    t = pc(); apply(net, D, R, S['U_scale'], bs=8); T_i8 = min(T_i8, (pc() - t) / len(te))
    t = pc(); apply(net, D, R, S['U_scale'], bs=1); T_i1 = min(T_i1, (pc() - t) / len(te))
chk = ''
if RES:
    for l in open(RES):
        r = json.loads(l)
        if r['model'] == 'oa-vcb' and json.loads(r['cfg']).get('data_seed', 0) == DS and json.loads(r['cfg'])['k0'] == k0:
            es = np.array(r['result']['en_s'])[:len(te)]; chk = f"; stored record reproduced to max rel diff {np.max(np.abs(e_net / es - 1)):.1e}"
print(f"\noa-vcb ({os.path.basename(CK)}, code {st.get('code')}): mean energy error {e_net.mean():.4e} (median {np.median(e_net):.4e}){chk}")


# ---- classical V-cycle with setup / apply split (same arithmetic as oa_vcycle.classical_vcycle(_proj))
class VCycle:
    def __init__(self, A, Z=None, q=10, nu=4):
        self.levs, A0 = oa_hierarchy(H, A, S['nl'], 4); self.lu = splu(A0.tocsc()); self.q, self.nu = q, nu
        self.dinv = [1 / Ak.diagonal() for Ak, _ in self.levs]
        self.lm = [3.0] + [self._lam(Ak) for Ak, _ in self.levs[1:]]
        self.A = self.levs[0][0]; self.Z = Z if (Z is not None and Z.shape[1]) else None

    def _lam(self, Ak, its=30):
        dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
        for _ in range(its):
            w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
        return 1.05 * lam

    def _cheb(self, l, rhs, x):
        Ak = self.levs[l][0]; dinv = self.dinv[l]; lo, hi = self.lm[l] / self.q, self.lm[l]; th, de = (hi + lo) / 2, (hi - lo) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(self.nu - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        return x

    def cyc(self, l, rhs, parts=False):
        Ak, X = self.levs[l]; nc = X.shape[1]
        x = self._cheb(l, rhs, np.zeros_like(rhs)); r = rhs - Ak @ x; rc = r[:nc] + X.T @ r[nc:]
        ec = self.cyc(l + 1, rc) if l + 1 < len(self.levs) else self.lu.solve(rc)
        xc = np.concatenate([ec, X @ ec])
        if parts: return x, xc, self._cheb(l, rhs - Ak @ (x + xc), np.zeros_like(rhs))
        return self._cheb(l, rhs, x + xc)

    def _gal(self, W, rhs):
        G = W.T @ (self.A @ W); dg = np.diag(G); live = dg > 1e-30 * dg.max(); sc = np.where(live, 1 / np.sqrt(np.where(live, dg, 1)), 0)
        return W @ (sc * np.linalg.solve(sc[:, None] * G * sc[None, :] + np.diag((~live) + 1e-10), sc * (W.T @ rhs)))

    def member(self, b):                                     # deflation + one V(4,4) + Galerkin output
        x0 = self._gal(self.Z, b) if self.Z is not None else np.zeros_like(b); bb = b - self.A @ x0
        W = np.column_stack(list(self.cyc(0, bb, parts=True)) + ([self.Z] if self.Z is not None else []))
        return x0 + self._gal(W, bb)


# check the split implementation against the reference functions on two samples
for j in (0, 1):
    A = As[j]; levs, A0 = oa_hierarchy(H, A, S['nl'], 4); Z = near_kernel_basis(A); V = VCycle(A, Z)
    d1 = np.abs(V.cyc(0, R[j]) - classical_vcycle(levs, A0, R[j])).max() / np.abs(R[j]).max()
    d2 = np.abs(V.member(R[j]) - classical_vcycle_proj(levs, A0, R[j], Z=Z, nu=4)).max() / np.abs(R[j]).max()
    assert d1 < 1e-10 and d2 < 1e-10, (d1, d2)


def pcg(A, x0, M, nit, u=None):
    """nit PCG iterations from x0; if u is given, also returns the energy-error history (used only in UNTIMED runs)."""
    x = x0.copy(); r = F - A @ x; z = M(r); p = z.copy(); rz = r @ z; h = [energy(A, x, u)] if u is not None else None
    for _ in range(nit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        if u is not None: h.append(energy(A, x, u))
    return x, h


def n_to_target(A, x0, M, u, target, maxit=60):
    _, h = pcg(A, x0, M, maxit, u)
    hit = np.nonzero(np.array(h) <= target)[0]
    return int(hit[0]) if len(hit) else maxit + 1


# preconditioner set-ups (returned callables); timed together with the iterations
def tg_setup(A):
    Pt = H.P[L].tocsr(); solc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal(); om = 2 / 3
    def M(r):
        x = om * dinv * r; x = x + Pt @ solc(Pt.T @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    return M
def amg_setup(A):
    Mm = pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle='V'); return lambda r: Mm @ r
def vc_setup(A):
    V = VCycle(A); return lambda r: V.cyc(0, r)

methods = {'two-grid PCG (weighted Jacobi(1,1) + level-%d Galerkin)' % (L - 1): tg_setup,
           'SA-AMG PCG (pyamg 5.x defaults, V-cycle)': amg_setup,
           'PCG + classical op.-adapted V(4,4) preconditioner': vc_setup}
NIT = {k: np.array([n_to_target(A, uc, s(A), u, tg) for A, uc, u, tg in zip(As, UC, U, e_net)]) for k, s in methods.items()}
rows = {}
for k, s in methods.items():
    def run(j, s=s, k=k):
        t = pc(); pcg(As[j], UC[j], s(As[j]), int(NIT[k][j])); return pc() - t
    rows[k] = best(run)
def direct(j):
    t = pc(); splu(As[j].tocsc()).solve(F); return pc() - t
T_dir = best(direct)
def member(j):
    t = pc(); V = VCycle(As[j], near_kernel_basis(As[j])); UC[j] + V.member(R[j]); return pc() - t
T_mem = best(member)
e_mem = np.array([energy(A, uc + VCycle(A, near_kernel_basis(A)).member(r), u) for A, uc, r, u in zip(As, UC, R, U)])

# ---- table
print(f"\nper-sample cost on this CPU, ms: mean (median).  u_c = level-{k0} Galerkin factor + solve: {fmt(T_uc)}"
      f"  [included in every row that starts from u_c]")
print(f"{'method':<58}{'energy err (mean)':>18}{'ms/sample':>22}   notes")
print(f"{'fine-matrix assembly, common to all methods (excluded)':<58}{'':>18}{fmt(T_asm):>22}   every row below starts from the assembled matrix")
print(f"{'oa-vcb, operator data (hierarchy + Galerkin levels + Z)':<58}{'':>18}{fmt(np.full(1, T_od)):>22}   per sample, from assembled A; + {T_pat * 1e3:.0f} ms once per mesh (patterns)")
print(f"{'  [10 Oct boundary: operator data incl. re-assembly]':<58}{'':>18}{fmt(np.full(1, T_od_asm)):>22}   not used; operator data identical to max abs diff {same:.1e}")
print(f"{'oa-vcb, inference batch 8 (torch CPU)':<58}{'':>18}{fmt(np.full(1, T_i8)):>22}")
print(f"{'oa-vcb, inference batch 1 (torch CPU)':<58}{'':>18}{fmt(np.full(1, T_i1)):>22}")
tot8 = T_uc + T_od + T_i8; tot1 = T_uc + T_od + T_i1
print(f"{'oa-vcb TOTAL = u_c + operator data + inference (batch 8)':<58}{e_net.mean():>18.3e}{fmt(tot8):>22}   batch 1: {fmt(tot1)}")
print(f"{'classical member, one pass (setup incl.) from u_c':<58}{e_mem.mean():>18.3e}{fmt(T_uc + T_mem):>22}   G(member/oa-vcb) {np.exp(np.mean(np.log(e_mem / e_net))):.2f}")
for k in methods:
    n = NIT[k]
    print(f"{k + ', to oa-vcb error':<58}{'(matched)':>18}{fmt(T_uc + rows[k]):>22}   its median {np.median(n):.0f} (range {n.min()}-{n.max()}), setup incl.")
print(f"{'direct (SuperLU factor + solve, exact)':<58}{'~1e-15':>18}{fmt(T_dir):>22}   does not use u_c")
print(f"\nratios (mean time): oa-vcb total batch 8 / two-grid PCG {tot8.mean() / (T_uc + rows[list(methods)[0]]).mean():.1f}, "
      f"/ SA-AMG PCG {tot8.mean() / (T_uc + rows[list(methods)[1]]).mean():.1f}, / direct {tot8.mean() / T_dir.mean():.1f}")
