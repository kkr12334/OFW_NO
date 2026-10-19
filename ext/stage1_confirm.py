"""Stage 1 confirmation runner (protocol v1 §5, amendment 1).  One family per process.
usage: OMP_NUM_THREADS=2 python -I ext/stage1_confirm.py SET FAMILY OUT.json
  SET = D  : development dry run (seed-1 test ids 300-399); gates include reproduction of ext/dev/stage1_dev_D.json
        C  : FRESH confirmation set (setup(3, 2, 0, rng_seed=2026101601 / 2026101602), ids 300-399).  Run ONCE.
        R2, R3 : retrospective data seeds 2 / 3 with their own OA-VCB checkpoints (secondary).
        S  : fresh stratified barrier diagnostic (rng 2026101611, sequential draws, first 20 of each A7.1 category,
             cap 3000 draws) -- FAMILY must be barriers (secondary, subgroup descriptives only).
        LC : fresh load challenge on the first 50 channel operators of C, 4 loads each (stage1_loads.lc_loads); fixed-load
             OA-VCB (3 inits) and the random-load OA-VCB (data seed 1, init 0); operator = bootstrap unit (secondary).
Selections (champion, spatial trigger, CHEB-T, fitted coefficients, input scales) are READ from
ext/stage1_preconfirmation.json and never re-derived here.
Confirmatory (set C only): P1 = PCG -> OA-VCB (bands), P2 = POLY-S -> OA-VCB, P3 = CHEB(legacy anchor) -> PCG, per
OA-VCB initialisation; raw (complete cases) and guarded (failure -> error of u_c).  Everything else is secondary.
Exit status non-zero if any declared gate fails (A7.3)."""
import os, sys
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(v, '2')
import json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy, load_no_low, coarse_P, vc_levels
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import OAData, near_kernel_basis, classical_vcycle_proj
from stage1_core import Hier, make_cheb, make_poly, pcg4, sd4, mr4, sgs4, sgs_pcg4, one_pass, one_pass_k2, lam_power, lam_lanczos, lam_gershgorin
from stage1_labels import label
from stage1_loads import random_model_scale, Geo, lc_loads
from stage1_manifest import manifest, Gates, fitted_methods_sha, FROZEN_FIT_SHA
import stage1_models as SM, stage1_stats as ST
torch.set_num_threads(2)
SET, FAM, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
assert SET in ('D', 'C', 'R2', 'R3', 'S', 'LC') and FAM in ('channels', 'barriers')
assert (SET != 'S' or FAM == 'barriers') and (SET != 'LC' or FAM == 'channels')
PRE = json.load(open(os.path.join(HERE, 'stage1_preconfirmation.json'))); PF = PRE['families'][FAM]
EX = json.load(open(os.path.join(HERE, PRE['fitted_methods'])))[FAM]
CK = json.load(open(os.path.join(HERE, 'checkpoint_map.json')))
G = Gates(); G.check('fitted coefficients hash = frozen', fitted_methods_sha(os.path.join(HERE, PRE['fitted_methods'])) == FROZEN_FIT_SHA == PRE['fitted_methods_sha256'])
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
DS = {'D': 1, 'C': 1, 'R2': 2, 'R3': 3, 'S': 1, 'LC': 1}[SET]       # data seed of the models evaluated
CSEED = {'channels': 2026101601, 'barriers': 2026101602}[FAM]
SMOKE = os.environ.get('STAGE1_SMOKE') == '1'      # code-path test on DEVELOPMENT data only: LC on set-D operators (5),
if SMOKE: assert SET in ('S', 'LC'); print('*** SMOKE TEST on development data; not a Stage 1 result ***', flush=True)   # S from rng 12345, 2 per category
t00 = time.time(); SAMPLES = []; S_INFO = {}
if SET in ('D', 'C', 'R2', 'R3', 'LC'):
    S = setup(3, 2, 0, rng_seed=CSEED, **kw) if (SET == 'C' or (SET == 'LC' and not SMOKE)) else setup(3, 2, DS, **kw)
    H = S['H']; nl = S['nl']
    if SET != 'LC':
        SAMPLES = [dict(sid=int(i), op=int(i), load=('f=1' if FAM == 'channels' else 'no-low'), a=S['a'][i], A=S['A'][i], F=S['Fs'][i], U=S['U'][i], UC=S['UC'][i]) for i in S['te']]
    else:
        Geo_ = Geo(H, S['L']); P = S['P']; PT = P.T.tocsr()
        for i in S['te'][:5 if SMOKE else 50]:
            A = S['A'][i]; solc = factorized((PT @ A @ P).tocsc()); sol = factorized(A.tocsc())
            for ln, F in lc_loads(Geo_, i).items():
                SAMPLES.append(dict(sid=f'{int(i)}:{ln}', op=int(i), load=ln, a=S['a'][i], A=A, F=F, U=sol(F), UC=P @ solc(PT @ F)))
else:                                                                  # set S: sequential stratified draws
    H = NestedHierarchy(3, coarse=perforated_square(n=9)); nl = vc_levels(H, 3); P = coarse_P(H, 3, 2); PT = P.T.tocsr()
    rng = np.random.default_rng(12345 if SMOKE else 2026101611); K_ = 2 if SMOKE else 20; CAP = 300 if SMOKE else 3000
    need = {c: K_ for c in ('island', 'discordant', 'EDP', 'strict non-island')}; seen = {c: 0 for c in need}; nd = 0
    while nd < CAP and any(v > 0 for v in need.values()):
        a = make_field(H, 'channels', 1e-4, rng); A = H.A_fine(a).tocsr(); lab = label(H, a, A); nd += 1; seen[lab['category']] += 1
        if need[lab['category']] > 0:
            need[lab['category']] -= 1; F = load_no_low(H, a)
            SAMPLES.append(dict(sid=nd - 1, op=nd - 1, load='no-low', a=a, A=A, F=F, U=spsolve(A.tocsc(), F), UC=P @ spsolve((PT @ A @ P).tocsc(), PT @ F), label=lab))
    S_INFO = dict(draws=nd, category_counts_among_draws=seen, kept={c: K_ - v for c, v in need.items()}, cap_reached=bool(nd >= CAP and any(v > 0 for v in need.values())))
    print('set S:', S_INFO, flush=True)
OPS = [s_['op'] for s_ in SAMPLES]; ids = [s_['sid'] for s_ in SAMPLES]
print(f"{SET} {FAM}: data generated in {time.time() - t00:.0f} s ({len(SAMPLES)} samples)", flush=True)

def ckpts():
    """(name, type, init, path, load) for every L3/k0=2/oa4/perforated checkpoint of data seed DS in this family."""
    out = []
    for r in CK:
        if (r['family'] == FAM and r['data_seed'] == DS and r['level'] == 3 and r['k0'] == 2 and r['oa_steps'] == 4
                and r['coarse'] == 'perforated' and r['model'] in ('oa-vcb', 'oa-vcbl', 'oa-vcl')):
            if SET in ('R2', 'R3', 'LC') and r['model'] != 'oa-vcb': continue
            nm = {'oa-vcb': 'OA-VCB', 'oa-vcbl': 'OA-VCBL', 'oa-vcl': 'OA-VCL'}[r['model']] + f" init{r['init_seed']}"
            if r['load'] == 'random':
                if SET in ('D', 'C', 'S'): continue                    # the random-load model is evaluated on sets LC and R
                nm += ' (random-load model)'
            out.append((nm, r['model'], r['init_seed'], os.path.join(ROOT, r['path']), r['load']))
    return sorted(out)
LEARNED = ckpts()

# ------------------------------------------------------------------ input scales (A7.2): frozen, never recomputed on C
if SET in ('D', 'C', 'S', 'LC'):
    SCALE = {'fixed': PRE['input_scales'][f'{FAM}/fixed-load']}
    if SET == 'LC': SCALE['random'] = PRE['input_scales']['channels/random-load']
    if SET == 'D': G.check('frozen fixed-load scale = training scale of seed-1 set T', abs(S['U_scale'] / SCALE['fixed'] - 1) < 1e-12, S['U_scale'])
else:
    SCALE = {'fixed': S['U_scale']}                                    # this seed's own training data (= model training data)
    if any(l == 'random' for *_, l in LEARNED): SCALE['random'] = random_model_scale(S, DS)

# ------------------------------------------------------------------ classical methods
SPATIAL = PF['spatial_trigger']['include']
def methods(hr):
    lam_leg = [3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]; ch = make_cheb(lam_leg)
    m = {'CHEB': (ch, ch), 'SD': (sd4, sd4), 'MR': (mr4, mr4), 'PCG': (pcg4, pcg4)}
    c = EX['CHEB-T']; f = lam_gershgorin if c['bound'] == 'gershgorin' else lam_lanczos; lt = [f(Ak) for Ak, _ in hr.levs]
    m['CHEB-T'] = (make_cheb(lt, q=c['q']),) * 2
    sc = [EX['POLY-S']['beta']] * hr.nl; m['POLY-S'] = (make_poly(np.array(EX['POLY-S']['pre']), sc), make_poly(np.array(EX['POLY-S']['post']), sc))
    sl = [lam_lanczos(Ak) for Ak, _ in hr.levs]; m['POLY-L'] = (make_poly(np.array(EX['POLY-L']['pre']), sl), make_poly(np.array(EX['POLY-L']['post']), sl))
    if 'SGS' in SPATIAL: m['SGS'] = (sgs4, sgs4)
    if 'SGS-PCG' in SPATIAL: m['SGS-PCG'] = (sgs_pcg4, sgs_pcg4)
    return m, ch

E = {}; WORK = {}; STATUS = {}; LAB = []; E_UC = []
t0 = time.time()
for j, smp in enumerate(SAMPLES):
    A = smp['A']; u = smp['U']; uc = smp['UC']; r = smp['F'] - A @ uc
    LAB.append(smp.get('label') or label(H, smp['a'], A)); E_UC.append(energy(A, uc, u))
    hr = Hier(H, A, nl); Z = near_kernel_basis(A)
    E.setdefault('CHEB-legacy', []).append(energy(A, uc + classical_vcycle_proj(hr.levs, hr.A0, r, Z=Z, nu=4), u))
    ms, ch = methods(hr)
    runs = [(nm, one_pass, pre, post) for nm, (pre, post) in ms.items()] + [('K2 (CHEB)', one_pass_k2, ch, ch)]
    for nm, fn, pre, post in runs:
        cnt = hr.new_counter(); st = {}
        try:
            x = fn(hr, r, Z, pre, post, cnt=cnt, stats=st); e = energy(A, uc + x, u)
            if not np.isfinite(e): st['nonfinite'] = 1
        except Exception as exc:
            e = np.nan; st['crash'] = repr(exc)
        E.setdefault(nm, []).append(float(e)); WORK.setdefault(nm, []).append({k: v for k, v in cnt.items() if k != 'status'})
        STATUS.setdefault(nm, []).append({**st, **cnt['status']})
    if j % 20 == 0: print(f"  classical {j + 1}/{len(SAMPLES)} ({time.time() - t0:.0f} s)", flush=True)
T_CLASSICAL = time.time() - t0

# ------------------------------------------------------------------ learned models (float32 cycle, as trained)
t0 = time.time()
D = OAData(H, [s_['a'] for s_ in SAMPLES], nl, 4, 'cpu', H.level_coords, near_kernel=True)
R = np.array([s_['F'] - s_['A'] @ s_['UC'] for s_ in SAMPLES])
for nm, typ, k, p, ld in LEARNED:
    net, st, _ = SM.load(p, D); sc = SCALE['random' if ld == 'random' else 'fixed']; C = SM.apply(net, D, R, sc)
    e = [float(energy(s_['A'], s_['UC'] + c, s_['U'])) for s_, c in zip(SAMPLES, C)]
    E[nm] = e; STATUS[nm] = [{'nonfinite': 1} if not np.isfinite(x) else {} for x in e]
T_LEARNED = time.time() - t0

# ------------------------------------------------------------------ gates
rel = np.abs(np.array(E['CHEB']) / np.array(E['CHEB-legacy']) - 1)
G.check('CHEB common path = legacy anchor (rel. diff <= 1e-8 on every sample)', np.nanmax(rel) <= 1e-8, float(np.nanmax(rel)))
G.check('all checkpoints match checkpoint_map SHA-256', True)          # finalised below from the manifest
OA = [nm for nm, typ, *_ in LEARNED if typ == 'oa-vcb']
if SET in ('D', 'C', 'S'): G.check('three OA-VCB initialisations present', len(OA) == 3, OA)
if SET == 'LC': G.check('three fixed-load OA-VCB + random-load OA-VCB present', len(OA) == 4, OA)
if SET == 'D':
    dev = json.load(open(os.path.join(HERE, 'dev', 'stage1_dev_D.json')))['families'][FAM]
    mp = {'SGS': 'SGS (spatial, conditional)', 'SGS-PCG': 'SGS-PCG (spatial, conditional)'}
    worst = max(float(np.max(np.abs(np.array(E[m]) / np.array(dev['errors'][mp.get(m, m)]) - 1))) for m in E)
    G.check('set D reproduces ext/dev/stage1_dev_D.json (all methods, rel. diff <= 1e-9)', worst <= 1e-9, worst)
    cnts = {c: sum(l['category'] == c for l in LAB) for c in sorted(set(l['category'] for l in LAB))}
    G.check('set-D labels = frozen counts', cnts == PF['labels_D'], cnts)

# ------------------------------------------------------------------ statistics
EG = {m: ST.guarded(e, E_UC) for m, e in E.items()}
def pair(x, y, guarded=False, k=None):
    ex, ey = (EG[x][0], EG[y][0]) if guarded else (E[x], E[y])
    if k is not None: ex, ey = np.asarray(ex)[k], np.asarray(ey)[k]
    if SET == 'LC' and k is None: return ST.paired_clustered(ex, ey, OPS)
    return ST.paired(ex, ey, None if SET == 'LC' else ([ids[j] for j in k] if k is not None else ids))
conf = {}
for nm in OA:
    P1 = pair('PCG', nm); P1['band'] = ST.band(P1); P1g = pair('PCG', nm, True); P1g['band'] = ST.band(P1g)
    conf[nm] = dict(P1=P1, P1_guarded=P1g, P2=pair('POLY-S', nm), P2_guarded=pair('POLY-S', nm, True))
conf['P3'] = pair('CHEB-legacy', 'PCG'); conf['P3_guarded'] = pair('CHEB-legacy', 'PCG', True)
OAF = [nm for nm in OA if 'random' not in nm]; bands = [conf[nm]['P1']['band'] for nm in OAF]      # family label: fixed-load inits
conf['P1_family_label'] = bands[0] if len(set(bands)) == 1 else 'initialisation-dependent: ' + '; '.join(f'{n}: {b}' for n, b in zip(OAF, bands))
champ = PF['champion']['selected']
sec = {}
for nm in OA:
    for c in ['CHEB-legacy', 'CHEB', 'CHEB-T', 'POLY-S', 'POLY-L', 'SD', 'MR', 'PCG', 'K2 (CHEB)'] + SPATIAL:
        R_ = pair(c, nm); R_['band_descriptive'] = ST.band(R_) if c in ('K2 (CHEB)', champ) else None; sec[f'{c} -> {nm}'] = R_
for m in E:
    if m != 'CHEB-legacy': sec[f'CHEB-legacy -> {m}'] = pair('CHEB-legacy', m)
phis = ({f'{c} | {nm}': ST.phi(E['CHEB-legacy'], E[c], E[nm], ids) for nm in OA
         for c in ['CHEB-T', 'POLY-S', 'POLY-L', 'SD', 'MR', 'PCG', 'K2 (CHEB)'] + SPATIAL} if SET != 'LC' else None)
STRATA = [smp['load'] for smp in SAMPLES] if SET == 'LC' else [l['category'] for l in LAB]
cats = sorted(set(STRATA))
subgroups = {}
for c in cats:
    k = [j for j, l in enumerate(STRATA) if l == c]
    subgroups[c] = dict(n=len(k), ids=[ids[j] for j in k], summary={m: ST.summary(np.array(e)[k]) for m, e in E.items()},
                        gains=({f'{x} -> {y}': pair(x, y, k=k) for x in ('CHEB-legacy', 'PCG', champ, 'POLY-S', 'K2 (CHEB)') + tuple(SPATIAL)
                                for y in OA} if len(k) >= 3 else None))
fails = {m: dict(nonfinite_or_crash=int(sum(1 for s in ss if 'nonfinite' in s or 'crash' in s)),
                 status_counts={k: int(sum(1 for s in ss if k in s)) for k in set().union(*[set(s) for s in ss])},
                 guarded_fallbacks=EG[m][1]) for m, ss in STATUS.items()}
mv = {m: float(np.mean([sum(w['matvec']) for w in ws])) for m, ws in WORK.items()}

MAN = manifest([p for *_, p, _ in LEARNED])
G.g['all checkpoints match checkpoint_map SHA-256'] = dict(passed=all(v['matches_map'] for v in MAN['checkpoints'].values()), detail=None)
print(f"GATE {'pass' if G.g['all checkpoints match checkpoint_map SHA-256']['passed'] else 'FAIL'}: all checkpoints match checkpoint_map SHA-256 (final)")
out = dict(set=SET + (' (SMOKE TEST, development data)' if SMOKE else ''), family=FAM, data=dict(rng_seed=(CSEED if SET in ('C', 'LC') else 2026101611 if SET == 'S' else 1000 + DS), ids=ids, operators=OPS,
                                         loads=[s_['load'] for s_ in SAMPLES], set_S=S_INFO or None),
           learned=[dict(name=n, type=t, init=k, path=os.path.relpath(p, ROOT), load=l) for n, t, k, p, l in LEARNED], input_scales=SCALE,
           preconfirmation=dict(champion=champ, spatial=SPATIAL, cheb_t=EX['CHEB-T']),
           labels=LAB, errors=E, error_uc=E_UC, status=STATUS, work=WORK, matvecs_per_pass=mv, failures=fails,
           confirmatory=conf if SET == 'C' else None, primary_comparisons_dry_run=conf if SET != 'C' else None,
           secondary=sec, recovered_log_gain=phis, subgroups=subgroups, summary={m: ST.summary(e) for m, e in E.items()},
           seconds=dict(total=time.time() - t00, classical=T_CLASSICAL, learned=T_LEARNED), gates=G.g, manifest=MAN)
json.dump(out, open(OUT, 'w'), indent=1)
print(f"\n== {SET} {FAM}: {len(ids)} samples, strata { {c: subgroups[c]['n'] for c in cats} }, {time.time() - t00:.0f} s")
for nm in OA:
    c = conf[nm]
    print(f"  {nm}: P1 R {c['P1']['G']:.3f} [{c['P1']['lo']:.3f}, {c['P1']['hi']:.3f}] W {c['P1']['W']:.2f} -> {c['P1']['band']};  P2 {c['P2']['G']:.3f} [{c['P2']['lo']:.3f}, {c['P2']['hi']:.3f}]")
print(f"  P3 (CHEB-legacy -> PCG) {conf['P3']['G']:.3f} [{conf['P3']['lo']:.3f}, {conf['P3']['hi']:.3f}];  P1 family label: {conf['P1_family_label']}")
print('  failures:', {m: f['nonfinite_or_crash'] for m, f in fails.items() if f['nonfinite_or_crash']} or 'none')
bad = G.failed()
if bad: print('GATES FAILED:', bad); sys.exit(1)
