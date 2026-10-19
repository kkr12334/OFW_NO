"""Claims check and renderer for ROUTE_C_PILOT_REPORT_2026-10-18.md (template: report_template.md).
Every number is inserted from the raw records (pilot runs, the frozen analysis output, exploration output); every
statement in the report that depends on a number is checked here. Fails on any failed check or unfilled placeholder.
    python3 rc_report_check.py
"""
import os, sys, re, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
J = lambda n: json.load(open(os.path.join(HERE, n)))
sha = lambda n: hashlib.sha256(open(os.path.join(HERE, n), 'rb').read()).hexdigest()
fails, checks, V = [], [], {}
def check(name, ok, detail=''):
    checks.append(dict(name=name, passed=bool(ok), detail=str(detail)))
    print(('PASS ' if ok else 'FAIL ') + name + (f'  [{detail}]' if detail != '' else ''))
    if not ok: fails.append(name)
def sci(x, d=1):
    m, e = f'{x:.{d}e}'.split('e'); e = int(e)
    return f"{m}·10{str(e).translate(str.maketrans('-0123456789', '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'))}"
f2 = lambda x: f'{x:.2f}'; f3 = lambda x: f'{x:.3f}'
FAMS = ('channels', 'barriers'); CS = ('1.0', '100.0', '10000.0', '1000000.0')

# ---------------------------------------------------------------- integrity
REC = {(f, l): J(f'runs/pilot_{f}_L{l}.json') for f in FAMS for l in (3, 4)}
codes = {l.split()[1]: l.split()[0] for l in open(os.path.join(HERE, 'PILOT_CODE_SHA256_DEVIATION1.txt')).read().splitlines()}
for (f, l), d in REC.items():
    k = f[:2] + str(l)
    check(f'{f} L{l}: complete, every gate passed, no warnings', d['status'] == 'complete' and all(g['passed'] for g in d['gates'])
          and not any(o.get('other_warnings') for o in d['operators']))
    check(f'{f} L{l}: produced by the deviation-1 code', all(d['manifest'][f'ext/routec/{n}'] == codes[n] for n in ('rc_fem.py', 'rc_fast.py', 'rc_pilot.py'))
          and d['manifest']['amendment'] == codes['ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md'] and d['manifest']['deviation_1'] == codes['ROUTE_C_PILOT_DEVIATION_1_2026-10-18.md'])
    n_ops = len([o for o in d['operators'] if o['status'] == 'complete'])
    check(f'{f} L{l}: expected number of (geometry, contrast) records', n_ops == (32 if l == 3 else 4), n_ops)
    V[f'G_{k}'] = f"{len(d['gates']):,}"; V[f'REF_{k}'] = sci(max(max(o['ref_err'].values()) for o in d['operators']))
check('current files match the deviation-1 hashes', all(sha(n) == h for n, h in codes.items()))
check('analysis script unchanged since the amendment', sha('rc_pilot_analysis.py').startswith('4683bd70'))
V['PEAK'] = f"{max(d['peak_rss_gb'] for d in REC.values()):.1f}"
A = J('pilot_analysis.json')['families']
check('pilot_analysis.json newer than all run records', os.path.getmtime(os.path.join(HERE, 'pilot_analysis.json')) >=
      max(os.path.getmtime(os.path.join(HERE, f'runs/pilot_{f}_L{l}.json')) for f in FAMS for l in (3, 4)))
# reproducibility
new, old = REC[('channels', 3)], J('runs/pilot_v0_channels_L3.json')
O = {(o['id'], o['contrast']): o for o in old['operators']}
dch = max(abs(o['eta'][k] - O[(o['id'], o['contrast'])]['eta'][k]) / O[(o['id'], o['contrast'])]['eta'][k] for o in new['operators'] for k in o['eta'])
flips = sum(o['resolved'][k] != O[(o['id'], o['contrast'])]['resolved'][k] for o in new['operators'] for k in o['resolved'])
check('channels repeat reproduces v0 (no resolution flips, < 1e-10)', flips == 0 and dch < 1e-10, (dch, flips))
N = {(o['id'], o['contrast']): o for o in REC[('barriers', 3)]['operators']}
dba = 0.0
for nm in ('pilot_interrupted_barriers_L3.json', 'pilot_stopped_barriers_L3.json'):
    for o in J('runs/' + nm)['operators']:
        if o['status'] != 'complete': continue
        n = N[(o['id'], o['contrast'])]; dba = max(dba, max(abs(n['eta'][k] - o['eta'][k]) / o['eta'][k] for k in o['eta']))
check('barrier repeat reproduces earlier records (< 1e-10)', dba < 1e-10, dba)
V.update(REPRO_CH=sci(dch), REPRO_BA=sci(dba))

# ---------------------------------------------------------------- primary endpoints
def e1(f, m, c): return A[f]['E1'][m][c]
def e3(f, m, c): return A[f]['E3'][m][c]
for f in FAMS:
    rows = []
    for m, lab in (('M0', 'M0 (our MVY adaptation)'), ('M1SZ', 'M1SZ (matched control)'), ('M3', '**M3 (candidate)**'),
                   ('M5', 'M5 (M3 with fixed lifting)'), ('M6', 'M6 (global optimum)')):
        rows.append(f"| {lab} | " + ' | '.join(f"{f2(e1(f, m, c)['median'])} [{f2(e1(f, m, c)['max'])}]" for c in CS[:3]) +
                    f" | {f2(A[f]['E2'][m])} | {f3(e3(f, m, '10000.0')['median'])} / {f3(e3(f, m, '10000.0')['p90'])} [{f3(e3(f, m, '10000.0')['max'])}] | "
                    f"{f2(e1(f, m, '1000000.0')['median'])} [{f2(e1(f, m, '1000000.0')['max'])}] |")
    V[f'T1_{f}'] = '\n'.join(rows)
# verdicts
for f in FAMS:
    R = A[f]; k = f[:2].upper()
    V[f'{k}_Q'] = f2(R['adaptation']['Q']['10000.0']); V[f'{k}_LIFT'] = f"{R['lifting']['10000.0']:.0f}"
    V[f'{k}_M0_E2'] = f"{R['classical_separation']['M0']['E2']:.1f}"
    V[f'{k}_SUP_M1'] = f2(R['adaptation']['superseded_unmatched_rule']['M1_ratio']); V[f'{k}_SUP_M2'] = f2(R['adaptation']['superseded_unmatched_rule']['M2_ratio'])
check('channels candidate classification = Support', A['channels']['candidate']['classification'] == 'Support')
check('barriers candidate classification = Inconclusive (E1, E2 pass; E3 fails)', A['barriers']['candidate']['classification'] == 'Inconclusive'
      and A['barriers']['candidate']['E1_le_1p5_all'] and A['barriers']['candidate']['E2'] <= 1.25 and not A['barriers']['candidate']['E3_ok_all'])
check('barriers E3 fails at 1e4 specifically (median > 1.10 or p90 > 1.5) and passes at 1 and 1e2',
      (e3('barriers', 'M3', '10000.0')['median'] > 1.10 or e3('barriers', 'M3', '10000.0')['p90'] > 1.5) and
      all(e3('barriers', 'M3', c)['median'] <= 1.10 and e3('barriers', 'M3', c)['p90'] <= 1.5 for c in CS[:2]))
check('adaptation = benefit in both families', all(A[f]['adaptation']['classification'] == 'benefit' for f in FAMS))
check('classical separation in both families', all(A[f]['classical_separation']['M0']['separation'] for f in FAMS))
check('barrier near-kernel informative; channel near-kernel uninformative', A['barriers']['near_kernel']['informative'] and not A['channels']['near_kernel']['informative'])
check('barrier near-kernel dims at 1e4 are [0,2,1,0,...] (geometries 1 and 2)', A['barriers']['near_kernel']['dims']['10000.0'][:3] == [0, 2, 1]
      and not any(A['barriers']['near_kernel']['dims']['10000.0'][3:]))
nkd = max(abs(v - 1) for c, v in A['barriers']['near_kernel']['M4_over_M3'].items()); nkg = max(abs(v - 1) for c, v in A['barriers']['near_kernel']['M1S_over_M1SZ'].items())
check('Z changed nothing measurable (M4/M3 and M1S/M1SZ within 1e-3 of 1)', nkd < 1e-3 and nkg < 1e-3, (nkd, nkg))
V['NK_DIFF'] = sci(nkd); V['NK_DIFF_G'] = sci(nkg)
def vrow(lbl, ch, ba): return f'| {lbl} | {ch} | {ba} |'
VT = ['| Conclusion (amendment §6) | Channels | Barriers |', '|---|---|---|']
VT.append(vrow('Candidate quality (M3)', 'Support', 'Inconclusive: E1 ≤ 1.5 and E2 ≤ 1.25 pass; E3 at 10⁴ fails'))
VT.append(vrow('E1(M3) at 1, 10², 10⁴', ', '.join(f2(e1('channels', 'M3', c)['median']) for c in CS[:3]), ', '.join(f2(e1('barriers', 'M3', c)['median']) for c in CS[:3])))
VT.append(vrow('E2(M3)', f2(A['channels']['E2']['M3']), f2(A['barriers']['E2']['M3'])))
VT.append(vrow('E3(M3) at 10⁴: median / p90', f"{f3(e3('channels', 'M3', '10000.0')['median'])} / {f3(e3('channels', 'M3', '10000.0')['p90'])}",
               f"{f3(e3('barriers', 'M3', '10000.0')['median'])} / {f3(e3('barriers', 'M3', '10000.0')['p90'])} (limits 1.10 / 1.5)"))
VT.append(vrow('Adaptation Q = E1(M1SZ)/E1(M3) at 10⁴', f"{V['CH_Q']}: benefit; {A['channels']['adaptation']['n_ops_M1SZ_ge_1p5_M3']['10000.0']}/8 geometries ≥ 1.5×",
               f"{V['BA_Q']}: benefit; {A['barriers']['adaptation']['n_ops_M1SZ_ge_1p5_M3']['10000.0']}/8 geometries ≥ 1.5×"))
VT.append(vrow('Q at 1, 10², 10⁶', ', '.join(f2(A['channels']['adaptation']['Q'][c]) for c in ('1.0', '100.0', '1000000.0')),
               ', '.join(f2(A['barriers']['adaptation']['Q'][c]) for c in ('1.0', '100.0', '1000000.0'))))
VT.append(vrow('Near-kernel: E1(M4)/E1(M3) at 10⁴', 'uninformative (all dimensions 0)', f"{A['barriers']['near_kernel']['M4_over_M3']['10000.0']:.6f}"))
VT.append(vrow('Lifting: E1(M5)/E1(M3) at 10⁴', f2(A['channels']['lifting']['10000.0']), f2(A['barriers']['lifting']['10000.0'])))
VT.append(vrow('Classical separation: E2(M0), E2(M0′)', f"{A['channels']['classical_separation']['M0']['E2']:.1f}, {A['channels']['classical_separation']['M0p']['E2']:.1f}",
               f"{A['barriers']['classical_separation']['M0']['E2']:.1f}, {A['barriers']['classical_separation']['M0p']['E2']:.1f}"))
VT.append(vrow('Stress 10⁶: E1(M3) median [max]', f"{f2(e1('channels', 'M3', '1000000.0')['median'])} [{f2(e1('channels', 'M3', '1000000.0')['max'])}]",
               f"{f2(e1('barriers', 'M3', '1000000.0')['median'])} [{f2(e1('barriers', 'M3', '1000000.0')['max'])}]"))
V['T_VERDICTS'] = '\n'.join(VT)
V['CH_NOPS'] = str(A['channels']['adaptation']['n_ops_M1SZ_ge_1p5_M3']['10000.0'])
check('barriers: matched control >= 1.5x M3 on all 8 geometries at 1e4', A['barriers']['adaptation']['n_ops_M1SZ_ge_1p5_M3']['10000.0'] == 8)
V['CH_M3_MAX_PRIM'] = f2(max(e1('channels', 'M3', c)['max'] for c in CS[:3]))
check('channels M3 maxima at primary contrasts below 1.5', max(e1('channels', 'M3', c)['max'] for c in CS[:3]) < 1.5)
# summary statements
m6 = max(max(A[f]['per_operator']['M6'][c]['W']) for f in FAMS for c in CS)
check('M6 worst effectivity over all 64 L3 cases <= 1.41', m6 <= 1.41, m6); V['M6_MAX'] = f2(m6)
V['M6_MAX_BA'] = f2(max(max(A['barriers']['per_operator']['M6'][c]['W']) for c in CS))
cg = [x['C_glob'] for f in FAMS for c in CS for x in A[f]['C_glob'][c]]
check('C_glob values (8 per family) below 1/beta_* = 2.8873', len(cg) == 16 and max(cg) < 2.8873, max(cg)); V.update(CG_MIN=f2(min(cg)), CG_MAX=f2(max(cg)))
V['CH_E3_MED_MAX'] = f3(max(e3('channels', 'M3', c)['median'] for c in CS[:3])); V['CH_E3_MAX_1e4'] = f3(e3('channels', 'M3', '10000.0')['max'])
lam4 = A['barriers']['per_operator']['M3']['10000.0']['lam']; lam6 = A['barriers']['per_operator']['M3']['1000000.0']['lam']
check('barriers: exactly 3 of 8 geometries with loss > 1.5 at 1e4 and 6 of 8 at 1e6', sum(x > 1.5 for x in lam4) == 3 and sum(x > 1.5 for x in lam6) == 6)
V['BA_E3_MAX_1e4'] = f2(max(lam4)); V['BA_E3_MAX_1e6'] = f2(max(lam6))
ex = {f: {m: np.log(e1(f, m, '1000000.0')['median'] / e1(f, m, '100.0')['median']) / np.log(1e4) for m in ('M0', 'M1SZ', 'M5', 'M3')} for f in FAMS}
for m in ('M0', 'M1SZ', 'M5'):
    lo, hi = min(ex[f][m] for f in FAMS), max(ex[f][m] for f in FAMS)
    V[f'EXP_{m}'] = f'{lo:.2f}–{hi:.2f}' if abs(hi - lo) >= 0.005 else f'{lo:.2f}'
check('growth exponents of M0, M1SZ, M5 within [0.40, 0.51] (10² to 10⁶)', all(0.40 <= ex[f][m] <= 0.51 for f in FAMS for m in ('M0', 'M1SZ', 'M5')), ex)
check('channels M3 flat (exponent < 0.02)', ex['channels']['M3'] < 0.02, ex['channels']['M3'])
check('barriers M3 flat up to 1e2: E3 at 1e2 within thresholds and W max < 1.5', e3('barriers', 'M3', '100.0')['p90'] <= 1.5 and e1('barriers', 'M3', '100.0')['max'] < 1.5)

# ---------------------------------------------------------------- barrier table + exploration
EX = J('explore_localisation_loss.json')
eb = {r['id']: r for r in EX if r['family'] == 'barriers' and r['contrast'] == 1e4}
ec = {r['id']: r for r in EX if r['family'] == 'channels' and r['contrast'] == 1e6}
check('exploration covers all 8 barrier geometries (1e4) and 8 channel geometries (1e6)', sorted(eb) == list(range(8)) and sorted(ec) == list(range(8)))
P = A['barriers']['per_operator']['M3']; I4 = A['barriers']['indices']['10000.0']['OAZ']['worst']
rows = []
for g in range(8):
    rows.append(f"| {g} | {A['barriers']['near_kernel']['dims']['10000.0'][g]} | {f2(P['10000.0']['W'][g])} | {f2(P['10000.0']['lam'][g])} | "
                f"{f2(P['1000000.0']['W'][g])} | {f2(P['1000000.0']['lam'][g])} | {f2(I4[g]['index'])} | "
                f"{eb[g]['multi_component_among_index_ge_3']} of {eb[g]['n_members_index_ge_3']} |")
V['T_BARRIER_GEOM'] = '\n'.join(rows)
check('exploration index agrees with the runner diagnostic (worst OAZ index per geometry, 1e-9)',
      all(abs(eb[g]['index_max'] - I4[g]['index']) <= 1e-9 * I4[g]['index'] for g in range(8)))
nb = sum(eb[g]['n_members_index_ge_3'] for g in range(8)); ns = sum(eb[g]['multi_component_among_index_ge_3'] for g in range(8))
V.update(EXP_N_BIG=str(nb), EXP_N_STRADDLE=str(ns))
check('most high-index barrier members straddle (>= 80%)', ns >= 0.8 * nb, (ns, nb))
check('no saved worst barrier member carries an island component', not any(w['carries_island'] for g in range(8) for w in eb[g]['worst']))
isl = sorted(s for g in range(8) for s in eb[g]['island_sizes'])
V['EXP_ISLAND_SIZES'] = f"{sum(s == 1 for s in isl)} of {len(isl)} islands have one element; the largest have {', '.join(str(s) for s in isl[-2:])}"
order_idx = sorted(range(8), key=lambda g: -I4[g]['index'])[:4]; order_lam = sorted(range(8), key=lambda g: -P['1000000.0']['lam'][g])[:4]
check('top-4 geometries by index at 1e4 = top-4 by loss at 1e6 (as sets)', set(order_idx) == set(order_lam), (order_idx, order_lam))
V['TOP4_IDX'] = ', '.join(str(g) for g in sorted(order_idx)); V['TOP4_LAM'] = ', '.join(f2(P['1000000.0']['lam'][g]) for g in sorted(order_lam))
check('geometries with a detected near-kernel (1, 2) unaffected at 1e6 (loss < 1.1)', all(P['1000000.0']['lam'][g] < 1.1 for g in (1, 2)))
V['G0_1e4'] = f2(P['10000.0']['W'][0]); V['G0_1e6'] = f2(P['1000000.0']['W'][0])
aff = [g for g in range(8) if P['10000.0']['W'][g] > 1.5]
grs = [P['1000000.0']['W'][g] / P['10000.0']['W'][g] for g in aff]
check('the geometries affected at 1e4 are 0, 3, 5 and grow by 8 to 10 from 1e4 to 1e6', aff == [0, 3, 5] and all(8 <= x <= 10 for x in grs), grs)
V['AFF_GROWTH'] = f'{min(grs):.1f}–{max(grs):.1f}'
r = P['1000000.0']['W'][0] / P['10000.0']['W'][0]
check('geometry 0 growth 1e4->1e6 close to sqrt(100) = 10 (between 5 and 12)', 5 <= r <= 12, r)
Pc = A['channels']['per_operator']['M3']['1000000.0']
V['CH_G3_LAM'] = f2(Pc['lam'][3]); V['CH_G3_IDX'] = f2(ec[3]['index_max']); V['CH_G0_IDX'] = f2(ec[0]['index_max']); V['CH_G0_LAM'] = f2(Pc['lam'][0])
check('channels geometry 3 is the stress outlier and its worst member lies in one channel component',
      max(range(8), key=lambda g: Pc['lam'][g]) == 3 and ec[3]['worst'][0]['n_components'] == 1)
check('channels geometry 0: large index (> 20) but small loss (< 1.3) at 1e6', ec[0]['index_max'] > 20 and Pc['lam'][0] < 1.3)

# ---------------------------------------------------------------- OA-VCB, goals, stopping, work
ov = {m: [x for f in FAMS for x in A[f]['per_operator'][m]['10000.0']['W_oavcb']] for m in ('M0', 'M3')}
V.update(OAVCB_M3_MIN=f2(min(ov['M3'])), OAVCB_M3_MAX=f2(max(ov['M3'])), OAVCB_M0_MIN=f"{min(ov['M0']):.1f}", OAVCB_M0_MAX=f"{max(ov['M0']):.1f}")
G = {f: A[f]['goals'] for f in FAMS}
pct = lambda x: (f'{100 * x:,.0f}%' if x >= 1 else f'{100 * x:#.2g}'.rstrip('.') + '%').replace(',', ' ')
V['GOAL_CH_OAVCB'] = pct(G['channels']['M3']['10000.0']['oavcb']['rel_radius_polar_median']); V['GOAL_BA_OAVCB'] = pct(G['barriers']['M3']['10000.0']['oavcb']['rel_radius_polar_median'])
rows = ['| Family | Method | Coarse / coarse | PCG30 / PCG30 | OA-VCB / coarse |', '|---|---|---:|---:|---:|']
for f in FAMS:
    for m in ('M0', 'M3', 'M6'):
        g = G[f][m]['10000.0']
        rows.append(f"| {f} | {m} | " + ' | '.join(f"{pct(g[p]['rel_radius_polar_median'])} ({f2(g[p]['product_efficiency_median'])})" for p in ('coarse', 'pcg30', 'oavcb')) + ' |')
V['T_GOALS'] = '\n'.join(rows)
check('goal statement: coarse/coarse intervals wide at 1e4 (M3 polar median > 50% in both families)', all(G[f]['M3']['10000.0']['coarse']['rel_radius_polar_median'] > 0.5 for f in FAMS))
check('goal statement: learned/coarse intervals informative (M3 polar median < 5% in both families)', all(G[f]['M3']['10000.0']['oavcb']['rel_radius_polar_median'] < 0.05 for f in FAMS))
ST = {f: A[f]['stopping'] for f in FAMS}
rows = ['| Family | c | ε | M0 | M3 | M6 |', '|---|---:|---:|---:|---:|---:|']
for f in FAMS:
    for c, cl in (('1.0', '1'), ('10000.0', '10⁴'), ('1000000.0', '10⁶')):
        for e, el in (('0.001', '10⁻³'), ('1e-06', '10⁻⁶')):
            cell = lambda m: f"{ST[f][c][m][f'product_eps{e}']['median_first_check']:.0f} / {ST[f][c][m][f'polar_eps{e}']['median_first_check']:.0f}"
            rows.append(f'| {f} | {cl} | {el} | {cell("M0")} | {cell("M3")} | {cell("M6")} |')
V['T_STOP'] = '\n'.join(rows)
check('every stopping cell attained by all geometries, relative trigger only', all(v['fraction_attained'] == 1.0 and set(t for t in v['triggers']) == {'relative'}
      for f in FAMS for c in ST[f] for m in ST[f][c] for v in ST[f][c][m].values()))
check('summary: product eps 1e-3 at 1e4, median M3 80 vs M0 160 in both families', all(ST[f]['10000.0']['M3']['product_eps0.001']['median_first_check'] == 80
      and ST[f]['10000.0']['M0']['product_eps0.001']['median_first_check'] == 160 for f in FAMS))
m3r = [ST[f][c]['M3'][f'{k}_eps0.001']['median_first_check'] for f in FAMS for c in ('10000.0', '1000000.0') for k in ('product', 'polar')]
check('at 1e4 and 1e6, M0 needs 160 at eps 1e-3 and M3 fewer', all(ST[f][c]['M0'][f'{k}_eps0.001']['median_first_check'] == 160 for f in FAMS
      for c in ('10000.0', '1000000.0') for k in ('product', 'polar')) and max(m3r) < 160)
V['STOP_M3_RANGE'] = f'{min(m3r):.0f}–{max(m3r):.0f}'
CHK = [5, 10, 20, 40, 80, 160, 320]
def within_one(a, b):
    ia, ib = np.interp(a, CHK, range(7)), np.interp(b, CHK, range(7)); return abs(ia - ib) <= 1
check('at c = 1, the three certificates stop within one check of each other (medians)', all(within_one(ST[f]['1.0'][m][k]['median_first_check'], ST[f]['1.0']['M3'][k]['median_first_check'])
      for f in FAMS for m in ('M0', 'M6') for k in ST[f]['1.0']['M0']))
same6 = sum(ST[f][c]['M0'][f'{k}_eps1e-06']['median_first_check'] == ST[f][c]['M3'][f'{k}_eps1e-06']['median_first_check'] for f in FAMS for c in ('10000.0', '1000000.0') for k in ('product', 'polar'))
check('at eps 1e-6 the difference largely disappears (>= 6 of 8 cells equal)', same6 >= 6, same6)
W_ = {m: np.median([A[f]['work_median'][m]['lu_nnz'] for f in FAMS]) for m in ('M0', 'M3', 'M6')}
V.update(WORK_M3=f"{W_['M3'] / 1e6:.0f}·10⁶", WORK_M6=f"{W_['M6'] / 1e6:.1f}·10⁶", WORK_M0=f"{W_['M0'] / 1e6:.1f}·10⁶")
rr = W_['M3'] / W_['M6']; check('about 7x the global optimum (6.5 to 7.9)', 6.5 <= rr <= 7.9, rr)

# ---------------------------------------------------------------- L4
rows = ['| Family | Geometry | c | Method | W at L3 | W at L4 | Ratio |', '|---|---:|---:|---|---:|---:|---:|']
for f in FAMS:
    for r_ in A[f]['L4']['rows']:
        rows.append(f"| {f} | {r_['id']} | {r_['contrast']:g} | {r_['method']} | {f2(r_['W_L3'])} | {f2(r_['W_L4'])} | {f2(r_['ratio'])} |")
V['T_L4'] = '\n'.join(rows)
mr = max(A[f]['L4']['M3_max_ratio'] for f in FAMS); check('L4 reading rule met in both families', all(A[f]['L4']['M3_no_gross_deterioration'] for f in FAMS), mr)
V['L4_M3_MAX_RATIO'] = f2(mr)
rb = {(r_['id'], r_['contrast'], r_['method']): r_ for r_ in A['barriers']['L4']['rows']}
V.update(L4_M1SZ_L3=f2(rb[(0, 1e4, 'M1SZ')]['W_L3']), L4_M1SZ_L4=f2(rb[(0, 1e4, 'M1SZ')]['W_L4']), L4_M3_L3=f2(rb[(0, 1e4, 'M3')]['W_L3']), L4_M3_L4=f2(rb[(0, 1e4, 'M3')]['W_L4']))
check('L4: matched control robust at L4 in every (family, geometry) at 1e4 (W < 1.5)', all(r_['W_L4'] < 1.5 for f in FAMS for r_ in A[f]['L4']['rows'] if r_['method'] == 'M1SZ'))

# ---------------------------------------------------------------- indices
ix = {f: A[f]['indices'] for f in FAMS}
V.update(IDX_CH_1e4=f2(ix['channels']['10000.0']['OAZ']['max_median']), IDX_BA_1e4=f2(ix['barriers']['10000.0']['OAZ']['max_median']),
         IDX_CH_1e4_MAX=f2(ix['channels']['10000.0']['OAZ']['max_max']), IDX_BA_1e4_MAX=f2(ix['barriers']['10000.0']['OAZ']['max_max']),
         IDX_G1_CH=f2(ix['channels']['10000.0']['G1']['max_median']), IDX_G1_BA=f2(ix['barriers']['10000.0']['G1']['max_median']))
p95 = [ix[f][c]['OAZ']['p95_median'] for f in FAMS for c in CS]
V['IDX_P95'] = f'{min(p95):.2f}–{max(p95):.2f}'; check('OAZ p95 near one (0.9 to 1.2)', 0.9 <= min(p95) and max(p95) <= 1.2)
for f in FAMS:
    gr = ix[f]['1000000.0']['OAZ']['max_max'] / ix[f]['10000.0']['OAZ']['max_max']
    V[f'IDX_GROWTH_{f[:2].upper()}'] = f'{gr:.1f}'; check(f'{f}: largest OAZ index grows about x10 from 1e4 to 1e6 (8 to 12)', 8 <= gr <= 12, gr)

# ---------------------------------------------------------------- render
tpl = open(os.path.join(HERE, 'report_template.md')).read()
need = set(re.findall(r'\{\{(\w+)\}\}', tpl)); missing = need - set(V)
check('every placeholder has a value', not missing, sorted(missing))
out = re.sub(r'\{\{(\w+)\}\}', lambda m: str(V[m.group(1)]), tpl) if not missing else tpl
check('no placeholder left', '{{' not in out)
json.dump(dict(values={k: v for k, v in V.items() if not k.startswith('T')}, checks=checks), open(os.path.join(HERE, 'report_check.json'), 'w'), indent=1)
if fails:
    print(f'\n{len(fails)} FAILED; report not rendered'); sys.exit(1)
p = os.path.join(HERE, 'ROUTE_C_PILOT_REPORT_2026-10-18.md'); open(p, 'w').write(out)
print(f'\nall {len(checks)} checks passed; rendered {os.path.basename(p)}  sha256 {sha(os.path.basename(p))}')
