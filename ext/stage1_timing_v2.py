"""Stage 1 timing harness v2 (amendment 2, B2): repairs of the frozen stage1_timing.py.  Run ALONE on the machine.
Changes against v1: (1) every classical method builds only its own smoother/set-up; (2) mesh patterns are built once per
mesh outside every timer (structural_patterns cached; cost reported separately); (3) OA-VCB batch-1 latency = per-
operator operator data (hierarchy, near-kernel basis, coarse inverse, level values) + residual + forward pass, weights
loaded once outside; (4) batch-8 throughput loads weights outside the timer; (5) the residual F - A u_c is formed inside
every timer; (6) a component profile; (7) u_c construction time (end-to-end boundary); (8) curves: time to a fixed
relative residual 1e-6 and outer termination reasons.
usage: OMP_NUM_THREADS=2 python -I ext/stage1_timing_v2.py FAMILY N_OPS OUT.json   (set C; first N_OPS test operators)"""
import os, sys
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(v, '2')
import json, time, platform, functools, numpy as np, torch, pyamg
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from scipy.sparse.linalg import splu, factorized
from s1_common import setup, energy
import oa_vcycle
from oa_vcycle import OAData, near_kernel_basis
from stage1_core import (Hier, make_cheb, make_poly, pcg4, sd4, mr4, sgs4, sgs_pcg4, one_pass, one_pass_k2, pcg_v_preconditioner,
                         lam_power, lam_lanczos, lam_gershgorin)
from stage1_manifest import manifest
import stage1_models as SM
torch.set_num_threads(2)
FAM, NOPS, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
EX = json.load(open(os.path.join(HERE, 'dev', 'stage1_fitted_methods.json')))[FAM]
REPEATS = 5; pc = time.perf_counter
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
S = setup(3, 2, 0, rng_seed={'channels': 2026101601, 'barriers': 2026101602}[FAM], **kw)
te = S['te'][:NOPS]; H = S['H']; nl = S['nl']; L = S['L']; P = S['P']; PT = P.T.tocsr()
USC = json.load(open(os.path.join(HERE, 'frozen_scales.json')))['scales'][f'{FAM}/fixed-load']
CKP = os.path.join(ROOT, {'channels': 'resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt',
                          'barriers': 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt'}[FAM])
# ---------------- (2) mesh patterns once per mesh, outside every timer
_orig_sp = oa_vcycle.structural_patterns; _cache = {}
t = pc(); _cache[(nl, 4)] = _orig_sp(H, nl, 4); T_PAT = pc() - t
def _cached_sp(H_, n_levels, m=4, A_struct=None):
    assert H_ is H and A_struct is None; return _cache[(n_levels, m)]
oa_vcycle.structural_patterns = _cached_sp

# ---------------- (1) method-specific smoothers
def build(name, hr):
    if name in ('CHEB', 'K2 (CHEB)'):
        ch = make_cheb([3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]); return ch, ch
    if name == 'CHEB-T':
        c = EX['CHEB-T']; f = lam_gershgorin if c['bound'] == 'gershgorin' else lam_lanczos; ch = make_cheb([f(Ak) for Ak, _ in hr.levs], q=c['q']); return ch, ch
    if name == 'POLY-S':
        sc = [EX['POLY-S']['beta']] * hr.nl; return make_poly(np.array(EX['POLY-S']['pre']), sc), make_poly(np.array(EX['POLY-S']['post']), sc)
    if name == 'POLY-L':
        sc = [lam_lanczos(Ak) for Ak, _ in hr.levs]; return make_poly(np.array(EX['POLY-L']['pre']), sc), make_poly(np.array(EX['POLY-L']['post']), sc)
    f = dict(PCG=pcg4, SD=sd4, MR=mr4, SGS=sgs4)[name] if name != 'SGS-PCG' else sgs_pcg4; return f, f
NAMES = ['CHEB', 'PCG', 'SD', 'MR', 'SGS', 'SGS-PCG', 'CHEB-T', 'POLY-S', 'POLY-L']
def one_pass_method(name, A, F, uc):                       # (5) residual inside
    r = F - A @ uc; hr = Hier(H, A, nl); Z = near_kernel_basis(A); pre, post = build(name, hr)
    return (one_pass_k2 if name.startswith('K2') else one_pass)(hr, r, Z, pre, post)

# ---------------- (3) OA-VCB batch 1, weights loaded once outside
D0 = OAData(H, None, nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[S['A'][te[0]]]); NET, _, _ = SM.load(CKP, D0)
def oavcb(A, F, uc, parts=None):
    t_ = pc(); D = OAData(H, None, nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[A]); t_od = pc() - t_
    r = F - A @ uc; j = torch.tensor([0])
    with torch.no_grad():
        NET.set_ops(*D.batch(j), Z=D.batch_z(j), A64=D.batch_a64(j))
        c = NET(torch.tensor(r[None] / USC, dtype=torch.float64)).double().numpy()[0] * USC
    if parts is not None: parts['operator_data'] = t_od
    return c

# ---------------- (7) u_c construction (coarse Galerkin at level k0 = 2, factorise, solve, prolong)
def make_uc(A, F):
    Ac = (PT @ A @ P).tocsc(); return P @ splu(Ac).solve(PT @ F)

# ---------------- (8) curves with termination reasons
def pcg_s(A, b, M, x0, its, rec):
    x = x0.copy(); r = b - A @ x; z = M(r); p = z.copy(); rz = r @ z; st = 'completed'
    for i in range(its):
        Ap = A @ p; pAp = p @ Ap
        if rz == 0: st = 'exact'; break
        if not pAp > 0: st = 'curvature'; break
        a = rz / pAp; x = x + a * p; r = r - a * Ap; z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2; rec(i, x)
    return st
def fcg_s(A, b, prec, its, x0, rec):
    x = x0.copy(); r = b - A @ x; dp = Adp = None; st = 'completed'
    for i in range(its):
        w = prec(r); d = w.copy()
        if dp is not None: d -= (w @ Adp) / (dp @ Adp) * dp
        Ad = A @ d; dAd = d @ Ad
        if not dAd > 0: st = 'curvature' if dAd < 0 or not np.isfinite(dAd) else 'exact'; break
        a = (d @ r) / dAd; x = x + a * d; r = r - a * Ad; dp, Adp = d, Ad; rec(i, x)
    return st
def tg_M(A):
    Pt = H.P[L].tocsr(); solc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal(); om = 2 / 3
    def M(r):
        x = om * dinv * r; x = x + Pt @ solc(Pt.T @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    return M
CURVES = {'two-grid PCG': 20, 'SA-AMG PCG': 15, 'PCG-V': 10, 'FCG-K2': 8, 'FCG-PCGsmoothed-V': 8, 'CHEB passes': 4, 'PCG passes': 4, 'SGS-PCG passes': 4}
def curve(name, A, F, uc, rec):
    n = CURVES[name]
    if name == 'two-grid PCG': return pcg_s(A, F, tg_M(A), uc, n, rec)
    if name == 'SA-AMG PCG':
        Mm = pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle='V'); return pcg_s(A, F, lambda r: Mm @ r, uc, n, rec)
    hr = Hier(H, A, nl)
    if name == 'PCG-V': return pcg_s(A, F, pcg_v_preconditioner(hr), uc, n, rec)
    if name == 'FCG-K2':
        Z = near_kernel_basis(A); ch = build('CHEB', hr)[0]; return fcg_s(A, F, lambda v: one_pass_k2(hr, v, Z, ch, ch, project=False), n, uc, rec)
    if name == 'FCG-PCGsmoothed-V': return fcg_s(A, F, lambda v: hr.cycle(0, v, pcg4, pcg4), n, uc, rec)
    sm = name.split()[0]; Z = near_kernel_basis(A); pre, post = build(sm, hr); x = uc.copy()
    for k in range(n):
        x = x + one_pass(hr, F - A @ x, Z, pre, post); rec(k, x)
    return 'completed'

rng = np.random.default_rng(20261017); recs = []
for j, i in enumerate(te):
    A = S['A'][i]; F = S['Fs'][i]; uc = S['UC'][i]; u = S['U'][i]; nF = np.linalg.norm(F)
    tasks = {f'1pass {n}': (lambda n=n: one_pass_method(n, A, F, uc)) for n in NAMES + ['K2 (CHEB)']}
    tasks['1pass OA-VCB batch1'] = lambda: oavcb(A, F, uc); tasks['direct'] = lambda: splu(A.tocsc()).solve(F); tasks['u_c construction'] = lambda: make_uc(A, F)
    for cn in CURVES: tasks[f'curve {cn}'] = (lambda cn=cn: curve(cn, A, F, uc, lambda k, x: None))
    err = {k: energy(A, uc + tasks[k](), u) for k in tasks if k.startswith('1pass')}
    err['u_c'] = energy(A, uc, u); assert np.allclose(make_uc(A, F), uc, rtol=1e-10, atol=1e-14)
    cerr, cres, cstat = {}, {}, {}
    for cn in CURVES:
        h, hr_ = [], []
        def rec(k, x): h.append(energy(A, x, u)); hr_.append(float(np.linalg.norm(F - A @ x) / nF))
        cstat[cn] = curve(cn, A, F, uc, rec); cerr[cn] = h; cres[cn] = hr_
    for f in tasks.values(): f()                                        # warm-up
    raw = {k: [] for k in tasks}; stamps = {k: [] for k in tasks if k.startswith('curve')}
    for rep in range(REPEATS):
        for k in rng.permutation(list(tasks)):
            if k.startswith('curve'):
                ts = []; t0 = pc(); curve(k[6:], A, F, uc, lambda kk, x: ts.append(pc() - t0)); raw[k].append(pc() - t0); stamps[k].append(ts)
            else:
                t0 = pc(); tasks[k](); raw[k].append(pc() - t0)
    # (6) component profile (separate, 5 repeats, median): classical one pass and OA-VCB
    prof = {}
    def tm(f):
        v = []
        for _ in range(REPEATS):
            t0 = pc(); out = f(); v.append(pc() - t0)
        return float(np.median(v)), out
    prof['hierarchy'], hr = tm(lambda: Hier(H, A, nl)); prof['near-kernel basis'], Z = tm(lambda: near_kernel_basis(A))
    prof['bounds: CHEB legacy (2 power)'], _ = tm(lambda: [lam_power(Ak) for Ak, _ in hr.levs[1:]])
    prof['bounds: Lanczos (3 levels)'], _ = tm(lambda: [lam_lanczos(Ak) for Ak, _ in hr.levs])
    prof['cycle + wrapper: PCG'], _ = tm(lambda: one_pass(hr, F - A @ uc, Z, pcg4, pcg4))
    prof['cycle + wrapper: SGS-PCG'], _ = tm(lambda: one_pass(hr, F - A @ uc, Z, sgs_pcg4, sgs_pcg4))
    pp = {}; prof['OA-VCB total'], _ = tm(lambda: oavcb(A, F, uc, pp)); prof['OA-VCB operator data (last repeat)'] = pp['operator_data']
    recs.append(dict(id=int(i), errors=err, curve_errors=cerr, curve_relres=cres, curve_status=cstat, raw_seconds=raw, curve_stamps=stamps, profile=prof))
    print(f"op {i}: " + ', '.join(f"{k} {1e3 * np.median(v):.1f} ms" for k, v in raw.items() if not k.startswith('curve')), flush=True)
# (4) throughput: weights loaded outside the timer
thr = {}; NET8, _, _ = SM.load(CKP, D0)
t0 = pc(); Dall = OAData(H, None, nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[S['A'][i] for i in te])
R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te]); SM.apply(NET8, Dall, R, USC, bs=8); thr['OA-VCB batch8 (operator data + residual + inference) per operator'] = (pc() - t0) / len(te)
for n in ('CHEB', 'PCG', 'SGS-PCG'):
    t0 = pc(); [one_pass_method(n, S['A'][i], S['Fs'][i], S['UC'][i]) for i in te]; thr[f'1pass {n} per operator'] = (pc() - t0) / len(te)
env = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, pyamg=pyamg.__version__, threads=torch.get_num_threads(),
           cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'))
json.dump(dict(version='v2 (amendment 2, B2)', set='C', family=FAM, env=env, manifest=manifest([CKP]), input_scale=USC, repeats=REPEATS,
               patterns_seconds_once_per_mesh=T_PAT, records=recs, throughput=thr, curves_max_iterations=CURVES), open(OUT, 'w'), indent=1)
print('throughput', {k: f'{1e3 * v:.1f} ms' for k, v in thr.items()})
