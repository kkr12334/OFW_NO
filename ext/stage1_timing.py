"""Stage 1 timing harness (amendment 1, A4).  Run ALONE on the machine (no concurrent jobs).
Primary workload: per-operator LATENCY.  For every operator, every method is run once untimed (warm-up), then timed in
a per-operator random order (rng 20261017), REPEATS times; raw repeats saved; median primary, minimum reported.
Boundary: conditional correction from the supplied u_c (u_c itself is given); every method includes its own set-up
(hierarchy, spectral bounds, deflation basis, factorisations, operator data).  Excluded: matrix assembly (common),
once-per-mesh patterns (structural_patterns, timed separately and reported), model weight loading (once per model).
Curves: one timed run with the maximum iteration count records a timestamp after every iteration (no error evaluated
inside the timer); errors come from a separate untimed run with identical arithmetic.
Secondary workload (labelled THROUGHPUT): OA-VCB batch 8 over the whole set and each classical method over the whole set.
usage: OMP_NUM_THREADS=2 python -I ext/stage1_timing.py SET FAMILY N_OPS OUT.json [fitted_methods.json]
       SET = D (development) or C (confirmation; only after the pre-confirmation note)"""
import os, sys
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(v, '2')
import json, time, platform, numpy as np, torch, pyamg
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from scipy.sparse.linalg import splu, factorized
from s1_common import setup, energy
from oa_vcycle import OAData, near_kernel_basis, classical_vcycle_proj, structural_patterns, oa_hierarchy
from stage1_core import (Hier, make_cheb, make_poly, pcg4, sd4, mr4, sgs4, sgs_pcg4, one_pass, one_pass_k2, fcg, pcg, pcg_v_preconditioner,
                         lam_power, lam_lanczos, lam_gershgorin)
import stage1_models as SM
torch.set_num_threads(2)
SET, FAM, NOPS, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
EX = json.load(open(sys.argv[5]))[FAM] if len(sys.argv) > 5 else {}
REPEATS = 5; pc = time.perf_counter
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
seed = dict(D=None, C={'channels': 2026101601, 'barriers': 2026101602}[FAM])[SET] if SET == 'C' else None
S = setup(3, 2, 0, rng_seed=seed, **kw) if SET == 'C' else setup(3, 2, 1, **kw)
USC = json.load(open(os.path.join(HERE, 'frozen_scales.json')))['scales'][f'{FAM}/fixed-load']   # A7.2: frozen, from set T
assert SET == 'C' or abs(S['U_scale'] / USC - 1) < 1e-12
te = S['te'][:NOPS]; H = S['H']; nl = S['nl']; L = S['L']
CKP = os.path.join(ROOT, {'channels': 'resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt',
                          'barriers': 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt'}[FAM])

# ---------------- one-pass methods: full method from u_c, set-up included
def smoother_set(hr):
    m = {}
    lam_leg = [3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]; ch = make_cheb(lam_leg); m['CHEB'] = (ch, ch)
    m['PCG'] = (pcg4, pcg4); m['SD'] = (sd4, sd4); m['MR'] = (mr4, mr4); m['SGS'] = (sgs4, sgs4); m['SGS-PCG'] = (sgs_pcg4, sgs_pcg4)
    if 'CHEB-T' in EX:
        f = lam_gershgorin if EX['CHEB-T']['bound'] == 'gershgorin' else lam_lanczos; lt = [f(Ak) for Ak, _ in hr.levs]
        m['CHEB-T'] = (make_cheb(lt, q=EX['CHEB-T']['q']),) * 2
    if 'POLY-S' in EX:
        sc = [EX['POLY-S']['beta']] * hr.nl; m['POLY-S'] = (make_poly(np.array(EX['POLY-S']['pre']), sc), make_poly(np.array(EX['POLY-S']['post']), sc))
    if 'POLY-L' in EX:
        sc = [lam_lanczos(Ak) for Ak, _ in hr.levs]; m['POLY-L'] = (make_poly(np.array(EX['POLY-L']['pre']), sc), make_poly(np.array(EX['POLY-L']['post']), sc))
    return m
NAMES = ['CHEB', 'PCG', 'SD', 'MR', 'SGS', 'SGS-PCG'] + [k for k in ('CHEB-T', 'POLY-S', 'POLY-L') if k in EX]
def one_pass_method(name, A, r, k2=False):
    hr = Hier(H, A, nl); Z = near_kernel_basis(A); pre, post = smoother_set(hr)[name]
    return (one_pass_k2 if k2 else one_pass)(hr, r, Z, pre, post)

# ---------------- OA-VCB, batch 1: operator data for this operator + set_ops + forward (net built once per mesh)
t = pc(); structural_patterns(H, nl, 4); T_pat = pc() - t
D0 = OAData(H, None, nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[S['A'][te[0]]])
NET, _, _ = SM.load(CKP, D0)
def oavcb(A, r):
    t_ = pc(); D = OAData(H, None, nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[A]); t_od = pc() - t_
    j = torch.tensor([0])
    with torch.no_grad():
        NET.set_ops(*D.batch(j), Z=D.batch_z(j), A64=D.batch_a64(j))
        c = NET(torch.tensor(r[None] / USC, dtype=torch.float64)).double().numpy()[0] * USC
    return c, t_od

# ---------------- whole-solve curves (iterations recorded with timestamps)
def tg_M(A):
    Pt = H.P[L].tocsr(); solc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal(); om = 2 / 3
    def M(r):
        x = om * dinv * r; x = x + Pt @ solc(Pt.T @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    return M
def amg_M(A):
    Mm = pyamg.smoothed_aggregation_solver(A).aspreconditioner(cycle='V'); return lambda r: Mm @ r
CURVES = {'two-grid PCG': 20, 'SA-AMG PCG': 15, 'PCG-V': 10, 'FCG-K2': 8, 'FCG-PCGsmoothed-V': 8, 'CHEB passes': 4, 'PCG passes': 4, 'SGS-PCG passes': 4}
def curve(name, A, F, uc, rec):
    if name == 'two-grid PCG': pcg(A, F, tg_M(A), uc, CURVES[name], rec)
    elif name == 'SA-AMG PCG': pcg(A, F, amg_M(A), uc, CURVES[name], rec)
    elif name == 'PCG-V': hr = Hier(H, A, nl); pcg(A, F, pcg_v_preconditioner(hr), uc, CURVES[name], rec)
    elif name == 'FCG-K2':
        hr = Hier(H, A, nl); Z = near_kernel_basis(A); ch = smoother_set(hr)['CHEB'][0]
        fcg(A, F, lambda v: one_pass_k2(hr, v, Z, ch, ch, project=False), CURVES[name], x0=uc, record=rec)
    elif name == 'FCG-PCGsmoothed-V':
        hr = Hier(H, A, nl); fcg(A, F, lambda v: hr.cycle(0, v, pcg4, pcg4), CURVES[name], x0=uc, record=rec)
    else:
        sm = name.split()[0]; hr = Hier(H, A, nl); Z = near_kernel_basis(A); pre, post = smoother_set(hr)[sm]; x = uc.copy()
        for k in range(CURVES[name]):
            x = x + one_pass(hr, F - A @ x, Z, pre, post); rec(k, x)
def direct(A, F):
    return splu(A.tocsc()).solve(F)

rng = np.random.default_rng(20261017); recs = []
for j, i in enumerate(te):
    A = S['A'][i]; F = S['Fs'][i]; uc = S['UC'][i]; u = S['U'][i]; r = F - A @ uc
    tasks = {f'1pass {n}': (lambda n=n: one_pass_method(n, A, r)) for n in NAMES}
    tasks['1pass K2 (CHEB)'] = lambda: one_pass_method('CHEB', A, r, k2=True)
    tasks['1pass OA-VCB batch1'] = lambda: oavcb(A, r)
    tasks['direct'] = lambda: direct(A, F)
    for cn in CURVES: tasks[f'curve {cn}'] = (lambda cn=cn: curve(cn, A, F, uc, lambda k, x: None))
    # errors (untimed)
    err = {}
    for n in NAMES: err[f'1pass {n}'] = energy(A, uc + one_pass_method(n, A, r), u)
    err['1pass K2 (CHEB)'] = energy(A, uc + one_pass_method('CHEB', A, r, k2=True), u)
    err['1pass OA-VCB batch1'] = energy(A, uc + oavcb(A, r)[0], u)
    for cn in CURVES:
        h = []; curve(cn, A, F, uc, lambda k, x: h.append(energy(A, x, u))); err[f'curve {cn}'] = h
    # warm-up, then timed repeats in random order
    for f in tasks.values(): f()
    raw = {k: [] for k in tasks}; stamps = {k: [] for k in tasks if k.startswith('curve')}; od = []
    for rep in range(REPEATS):
        for k in rng.permutation(list(tasks)):
            if k.startswith('curve'):
                cn = k[6:]; ts = []; t0 = pc()
                curve(cn, A, F, uc, lambda kk, x: ts.append(pc() - t0)); raw[k].append(pc() - t0); stamps[k].append(ts)
            elif k == '1pass OA-VCB batch1':
                t0 = pc(); _, t_od = tasks[k](); raw[k].append(pc() - t0); od.append(t_od)
            else:
                t0 = pc(); tasks[k](); raw[k].append(pc() - t0)
    recs.append(dict(id=int(i), errors=err, raw_seconds=raw, curve_stamps=stamps, oavcb_operator_data_seconds=od))
    print(f"op {i}: " + ', '.join(f"{k} {1e3 * np.median(v):.1f} ms" for k, v in raw.items() if k.startswith('1pass') or k == 'direct'), flush=True)

# throughput (secondary)
thr = {}
t0 = pc(); Dall = OAData(H, None, nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[S['A'][i] for i in te]); t_odall = pc() - t0 - T_pat
R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te]); t0 = pc(); SM.apply(SM.load(CKP, Dall)[0], Dall, R, USC, bs=8); t_inf = pc() - t0
thr['OA-VCB batch8 (operator data + inference) per operator'] = (t_odall + t_inf) / len(te)
for n in ('CHEB', 'PCG'):
    t0 = pc(); [one_pass_method(n, S['A'][i], S['Fs'][i] - S['A'][i] @ S['UC'][i]) for i in te]; thr[f'1pass {n} per operator'] = (pc() - t0) / len(te)
env = dict(python=platform.python_version(), numpy=np.__version__, torch=torch.__version__, pyamg=pyamg.__version__,
           threads=torch.get_num_threads(), env={v: os.environ.get(v) for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
           cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'))
from stage1_manifest import manifest
json.dump(dict(set=SET, family=FAM, env=env, manifest=manifest([CKP]), input_scale=USC, repeats=REPEATS, patterns_seconds_once_per_mesh=T_pat, records=recs, throughput=thr,
               curves_max_iterations=CURVES), open(OUT, 'w'), indent=1)
print('throughput', {k: f'{1e3 * v:.1f} ms' for k, v in thr.items()})
