"""Claims check and renderer for ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md.

Every measured number in the amendment is a {{placeholder}} in amendment_template.md and is inserted here directly from
the raw record fields (no hand transcription).  Before rendering, the script checks:
  - the smoke runs completed with every gate passed, and were produced by the CURRENT rc_fem.py / rc_fast.py / rc_pilot.py
    (manifest hashes equal the files on disk), so the amendment describes the code the pilot will run;
  - the engine self-check (engine_check.json) passed and was produced by the current rc_fast.py / rc_fem.py;
  - beta: 1/sqrt(min beta^2) is consistent, and the value agrees with ASTRA's independent audit value to 1e-12;
  - polarization rows: half-width recomputed from polar_lo/hi, every row contains J_h;
  - effectivities are recomputed from the raw eta and err fields, excluding exact and unresolved cases;
  - qualitative statements of the amendment that depend on numbers (listed as STATEMENTS below) hold.
It fails (non-zero exit) on any failed check or any placeholder left unfilled.

    python3 rc_amendment_check.py            # check + render
    python3 rc_amendment_check.py --hash     # additionally write ROUTE_C_PILOT_AMENDMENT_1_SHA256.txt
"""
import os, sys, re, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n)))
sha = lambda n: hashlib.sha256(open(os.path.join(HERE, n), 'rb').read()).hexdigest()
fails, V, checks = [], {}, []
def check(name, ok, detail=''):
    checks.append(dict(name=name, passed=bool(ok), detail=str(detail)))
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail != '' else ''))
    if not ok: fails.append(name)

f3 = lambda x: f'{x:.3f}'; f2 = lambda x: f'{x:.2f}'

# ---------------------------------------------------------------- engine and shape constant
E = J('engine_check.json')
check('engine self-check passed', E['passed'], E['max_relative_difference'])
check('engine self-check used current rc_fast.py and rc_fem.py', E['rc_fast_sha256'] == sha('rc_fast.py') and E['rc_fem_sha256'] == sha('rc_fem.py'))
def sci(x, d=1):
    m, e = f'{x:.{d}e}'.split('e'); e = int(e)
    sup = str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹')
    return f'{m}·10{str(e).translate(sup)}'
V['ENGINE_DIFF'] = sci(E['max_relative_difference'])
B = J('beta_shape.json')
check('beta: 1/sqrt(min beta^2) consistent', abs(1 / np.sqrt(B['min_beta2']) - B['inv_min_beta']) < 1e-12)
ASTRA_MIN_BETA2 = 0.11995412385678803                         # ASTRA supplement section 1 (independent assembly)
check('beta agrees with ASTRA audit to 1e-12', abs(B['min_beta2'] - ASTRA_MIN_BETA2) < 1e-12, B['min_beta2'] - ASTRA_MIN_BETA2)
check('beta constant identical on descendants of each coarse triangle (spread < 1e-12)', B['max_spread_within_coarse_triangle'] < 1e-12)
V['BETA_DIFF'] = sci(abs(B['min_beta2'] - ASTRA_MIN_BETA2))
V.update(BETA_NT=str(B['parent_triangles']), BETA_NC=str(B['coarse_triangles']), BETA_MIN2=f"{B['min_beta2']:.13f}",
         BETA_INV=f"{B['inv_min_beta']:.4f}", BETA_SPREAD=sci(B['max_spread_within_coarse_triangle']))
G = J('global_stability.json'); cg = [r['C_glob'] for r in G]
check('development C_glob below 1/beta_*', max(cg) < B['inv_min_beta'], max(cg))
V.update(CGLOB_MIN=f2(min(cg)), CGLOB_MAX=f2(max(cg)))

# ---------------------------------------------------------------- smoke runs
CODE = ['rc_fem.py', 'rc_fast.py', 'rc_pilot.py', 'rc_goal.py']
COMMON = ['coarse', 'pcg10', 'pcg30', 'pcg100', 'noise10', 'smooth10']
S = {}
for fam in ('channels', 'barriers'):
    for lev in (3, 4):
        d = J(f'runs/smoke_{fam}_L{lev}.json'); S[(fam, lev)] = d
        tag = f'smoke {fam} L{lev}'
        check(f'{tag}: complete, every gate passed', d['status'] == 'complete' and all(g['passed'] for g in d['gates']), d['status'])
        man = d['manifest']
        check(f'{tag}: produced by the current code', all(man[f'ext/routec/{n}'] == sha(n) for n in CODE))
        op = d['operators'][0]
        check(f'{tag}: no numerical warnings and nothing else unexpected', not op.get('other_warnings'), op.get('other_warnings'))
        k = fam[:2] + str(lev)
        V[f'SMK_T_{k}'] = f"{op['seconds']:.0f}"; V[f'SMK_M_{k}'] = f"{d['peak_rss_gb']:.1f}"; V[f'SMK_G_{k}'] = str(len(d['gates']))

def W(op, m, inputs=COMMON):
    """largest energy effectivity of method m over problems and inputs, recomputed from raw eta and err, resolved only."""
    vals = [op['eta'][f'{p}|{i}|{m}'] / op['err'][f'{p}|{i}'] for p in op['norms'] for i in inputs
            if op['resolved'][f'{p}|{i}'] and f'{p}|{i}|{m}' in op['eta']]
    return max(vals)

fb = {S[(f, l)]['operators'][0]['shepard_fallback_nodes']['G1SZ'] for f in ('channels', 'barriers') for l in (3, 4)}
fo = {S[(f, l)]['operators'][0]['shepard_fallback_nodes']['OAZ'] for f in ('channels', 'barriers') for l in (3, 4)}
check('fallback counts identical across smoke runs', len(fb) == 1 and len(fo) == 1, (fb, fo))
V.update(FALLBACK_G=str(fb.pop()), FALLBACK_OA=str(fo.pop()))
oc3, oc4 = S[('channels', 3)]['operators'][0], S[('channels', 4)]['operators'][0]
check('channels development operator has no near-kernel (coarse dims are plain level spaces)', oc3['near_kernel'] == 0 and oc4['near_kernel'] == 0)
V.update(L3_COARSE=str(oc3['work']['M3']['coarse_dim']), L4_COARSE=str(oc4['work']['M3']['coarse_dim']))

methods = S[('channels', 3)]['design']['methods']
lab = dict(M0='M0 (our MVY adaptation)', M0p='M0′ (PRVW variant)', M1='M1 (geometric, coarse Dirichlet hats)', M1S='M1S (matched, no Z)',
           M1SZ='M1SZ (matched control)', M2='M2 (geometric level L−2)', M3='**M3 (candidate)**', M4='M4 (M3 without Z)',
           M5='M5 (M3 with fixed lifting)', M6='M6 (global optimum)')
Wv = {(fam, m): W(S[(fam, 3)]['operators'][0], m) for fam in ('channels', 'barriers') for m in methods}
V['SMK_TABLE'] = '\n'.join(f"| {lab[m]} | {f2(Wv[('channels', m)])} | {f2(Wv[('barriers', m)])} |" for m in methods)
V['SMK_M0_ch'] = f"{Wv[('channels', 'M0')]:.0f}"
for fam in ('channels', 'barriers'):
    k = fam[:2]; q = Wv[(fam, 'M1SZ')] / Wv[(fam, 'M3')]
    V[f'SMK_{k}_Q'] = f2(q); V[f'SMK_{k}_M1SZ'] = f2(Wv[(fam, 'M1SZ')]); V[f'SMK_{k}_M3'] = f2(Wv[(fam, 'M3')]); V[f'SMK_{k}_M1'] = f2(Wv[(fam, 'M1')])
    V[f'SMK_{k}_Qpct'] = f'{100 * (q - 1):.0f}%'
check('11.3 statement: barriers Q <= 1.10 on the development operator (would read "no benefit")', Wv[('barriers', 'M1SZ')] / Wv[('barriers', 'M3')] <= 1.10)
check('11.3 statement: M1S equals M1SZ on barriers to 1% (Z not the cause)', abs(Wv[('barriers', 'M1S')] / Wv[('barriers', 'M1SZ')] - 1) < 0.01)
Js = [g['Jh'] for f in ('channels', 'barriers') for l in (3, 4) for g in S[(f, l)]['operators'][0]['goals']]
V['J_MIN'] = sci(min(abs(x) for x in Js)); V['J_MAX'] = sci(max(abs(x) for x in Js))
check('development goal values well above tau_abs = 1e-12 (> 1e-6)', min(abs(x) for x in Js) > 1e-6)
# statements in 11.3
check('11.3 barriers: M1SZ much closer to M3 than M1 (gap < 1/3 of M1 gap)',
      Wv[('barriers', 'M1SZ')] - Wv[('barriers', 'M3')] < (Wv[('barriers', 'M1')] - Wv[('barriers', 'M3')]) / 3,
      (Wv[('barriers', 'M1')], Wv[('barriers', 'M1SZ')], Wv[('barriers', 'M3')]))
check('11.3 channels: matched control remains far from M3 (ratio >= 2)', Wv[('channels', 'M1SZ')] / Wv[('channels', 'M3')] >= 2,
      Wv[('channels', 'M1SZ')] / Wv[('channels', 'M3')])
ob = S[('barriers', 3)]['operators'][0]
nflux = 3 * 7424
rows = []
for m in ('M0', 'M1', 'M1SZ', 'M2', 'M3', 'M5', 'M6'):
    w = ob['work'][m]
    ov = '—' if m == 'M0' else ('1' if m == 'M6' else f"{w['patch_elements'] / nflux:.1f}")
    rows.append(f"| {m} | {w['coarse_dim'] if m != 'M6' else '—'} | {w['patches']:,} | {w['flux_unknowns']:,} | {w['constraints']:,} | {ov} | "
                f"{w['lu_nnz']:,} | {w['max_patch_elements']:,} |")
V['WORK_TABLE'] = '\n'.join(rows)
r6 = ob['work']['M3']['lu_nnz'] / ob['work']['M6']['lu_nnz']; r1 = ob['work']['M3']['lu_nnz'] / ob['work']['M1SZ']['lu_nnz']
check('11.3: M3 not cheaper than the global optimum (LU nnz ratio > 1)', r6 > 1, r6)
V.update(W_RATIO_M6=f'{r6:.1f}', W_RATIO_M1SZ=f'{r1:.0f}')
# pilot cost estimate: L3 2 fam x 8 ops x 4 contrasts + C_glob (2 ops x 4 contrasts x 2 fam, ~global_stability timings) ; L4 2 x 2 x 2
t3 = np.mean([S[(f, 3)]['operators'][0]['seconds'] for f in ('channels', 'barriers')])
t4 = np.mean([S[(f, 4)]['operators'][0]['seconds'] for f in ('channels', 'barriers')])
ts_ = [float(x) for x in re.findall(r'\((\d+)s\)', open(os.path.join(HERE, 'logs_rc_global_stability.log')).read())]
check('eight logged C_glob timings found', len(ts_) == 8, ts_)
cg_sec = float(np.mean(ts_))                                    # mean of the eight C_glob timings in logs_rc_global_stability.log
hours = (2 * 8 * 4 * t3 + 2 * 2 * 4 * cg_sec + 2 * 2 * 2 * t4) / 3600
V['PILOT_HOURS'] = f'{hours:.1f}'

# ---------------------------------------------------------------- fidelity
Fd = J('fidelity_poisson.json')
V.update(FID_N=f"{Fd['N']:,}".replace(',', ' '), FID_NE=f"{Fd['n_fe_elements']:,}".replace(',', ' '), FID_NC=str(Fd['coarse_elements']))
for r in Fd['rows']:
    k = r['pcg_iter']
    V[f'FID_E{k}'] = f2(r['alg_error_rel']); V[f'FID_M{k}'] = f3(r['M0 (MVY)']); V[f'FID_J{k}'] = f3(r['joint optimum'])
dmax = max(abs(r['M0 (MVY)'] - r['M0p (PRVW)']) for r in Fd['rows'])
check('fidelity: M0 = M0p on Poisson to round-off (K constant on level-1 elements)', dmax < 1e-12, dmax)
V['FID_M0M0P'] = sci(dmax) if dmax > 0 else '0'
pr = [r for r in Fd['rows'] if r['pcg_iter'] in (3, 6)]
lo_, hi_ = min(r['M0 (MVY)'] for r in pr) / 1.01 - 1, max(r['M0 (MVY)'] for r in pr) / 1.01 - 1
V['FID_LOSS'] = f'{100 * lo_:.0f}–{100 * hi_:.0f}% (iterations 3 and 6, the ones PRVW report)'
d0 = max(abs(op_['eta'][k] - op_['eta'][k.replace('|M0', '|M0p')]) / op_['eta'][k]
         for op_ in (S[('channels', 3)]['operators'][0], S[('barriers', 3)]['operators'][0]) for k in op_['eta'] if k.endswith('|M0'))
check('11.1 statement: in the pilot setting M0 and M0p differ (smoke, relative > 1e-6)', d0 > 1e-6, d0)

# ---------------------------------------------------------------- polarization
P = J('polarization_dev.json'); P3 = [r for r in P if r['method'] == 'M3']
check('polarization: every development interval contains J_h', all(r['contains'] for r in P))
check('polarization: half-width = (hi - lo)/2', all(abs(r['polar_halfwidth'] - 0.5 * (r['polar_hi'] - r['polar_lo'])) <= 1e-12 * abs(r['polar_halfwidth']) for r in P))
rat = [r['polar_halfwidth'] / r['product_radius'] for r in P3]
V.update(POL_MIN=f3(min(rat)), POL_MAX=f3(max(rat)))
check('11.2 statement "a factor of 2-12": 1/ratio within [2, 12]', 1 / max(rat) >= 2 and 1 / min(rat) <= 12, (1 / max(rat), 1 / min(rat)))
pk = {'coarse/coarse': 'c', 'OA-VCB/coarse': 'n', 'PCG30/PCG30': 'p'}
for r in P3:
    k = f"POL_{r['family'][:2]}_{pk[r['pair']]}"
    V[k] = f"{r['product_radius'] / abs(r['Jh']):.3g} → {r['polar_halfwidth'] / abs(r['Jh']):.3g}"
rb = [r for r in P3 if r['family'] == 'barriers' and r['pair'] == 'OA-VCB/coarse'][0]
V['POL_ba_n_pct'] = f"{100 * rb['polar_halfwidth'] / abs(rb['Jh']):.0f}%"
eff = rb['product_radius'] / (rb['eu'] * rb['ez'])
check('11.2 statement: M3 product efficiency on the learned barrier pair is near one (< 1.3)', eff < 1.3, eff)
check('11.2 statement: the dual (coarse) input error is the larger factor of the product, e_z > e_u', rb['ez'] > rb['eu'], (rb['eu'], rb['ez']))
V['POL_ba_n_eff'] = f2(eff)

# ---------------------------------------------------------------- scoping-note defects (section 12 item 5)
T = J('scoping_tables.json')['GOAL_CHECKS']
V.update(DIV_DEF=sci(T['max_div_defect'], 2), MOM_DEF=sci(T['max_moment_defect'], 2))

V['ANALYSIS_SHA'] = sha('rc_pilot_analysis.py')
check('analysis script ran on the smoke records', os.path.exists(os.path.join(HERE, 'smoke_analysis.json')) and os.path.getmtime(os.path.join(HERE, 'smoke_analysis.json')) >= os.path.getmtime(os.path.join(HERE, 'rc_pilot_analysis.py')))
# ---------------------------------------------------------------- render
tpl = open(os.path.join(HERE, 'amendment_template.md')).read()
need = set(re.findall(r'\{\{(\w+)\}\}', tpl))
missing = need - {k for k, v in V.items() if v is not None}
check('every placeholder has a value', not missing, sorted(missing))
out = re.sub(r'\{\{(\w+)\}\}', lambda m: str(V[m.group(1)]), tpl) if not missing else tpl
check('no placeholder left in the rendered amendment', '{{' not in out)
json.dump(dict(values={k: v for k, v in V.items() if v is not None}, checks=checks), open(os.path.join(HERE, 'amendment_check.json'), 'w'), indent=1)
if fails:
    print(f'\n{len(fails)} FAILED checks; amendment not rendered'); sys.exit(1)
AM = os.path.join(HERE, 'ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md')
open(AM, 'w').write(out)
print(f'\nall {len(checks)} checks passed; rendered {os.path.basename(AM)}  sha256 {sha(os.path.basename(AM))}')
if '--hash' in sys.argv:
    open(os.path.join(HERE, 'ROUTE_C_PILOT_AMENDMENT_1_SHA256.txt'), 'w').write(f'{sha(os.path.basename(AM))}  ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md\n')
    print('hash file written')
