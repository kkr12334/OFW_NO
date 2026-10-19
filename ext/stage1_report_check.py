"""Claims check for STAGE1_REPORT_2026-10-17.md.
(1) Headline checks: the confirmatory P1-P3 entries are regenerated from the raw records and must appear verbatim.
(2) Provenance scan: every decimal / scientific number in the report must equal, at its printed precision, some quantity
    computed from the run records (summary leaves of every run JSON, derived ratios); unmatched numbers are listed for
    manual review (parameters, dates and seeds are expected there).
usage: python -I ext/stage1_report_check.py [report.md]  (exit 1 if a headline check fails)"""
import os, sys, re, json, math, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); EXP = os.path.join(HERE, 'explore'); sys.path.insert(0, HERE)
import stage1_stats as ST
REP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'STAGE1_REPORT_2026-10-17.md'); text = open(REP).read()
J = lambda p: json.load(open(p))
V = []
def leaves(x, skip=('errors', 'per_operator_final', 'per_operator_grad_inf', 'rho', 'ids', 'records', 'labels', 'coefficients', 'scales', 'manifest', 'raw_seconds', 'curve_stamps')):
    if isinstance(x, dict):
        for k, v in x.items():
            if k not in skip: leaves(v, skip)
    elif isinstance(x, list):
        if len(x) <= 64:
            for v in x: leaves(v, skip)
    elif isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x): V.append(float(x))
def add_pair(R):
    leaves(R)
    for k in ('G', 'lo', 'hi'):
        if R.get(k): V.append(1 / R[k])
fails = []
# ---------------- (1) headline checks
for fam in ('channels', 'barriers'):
    d = J(os.path.join(RUNS, f'confirm_C_{fam}.json')); c = d['confirmatory']
    for k in (0, 1, 2):
        P1 = c[f'OA-VCB init{k}']['P1']; s = f"{P1['G']:.2f} [{P1['lo']:.2f}, {P1['hi']:.2f}] | {P1['W']:.2f} | {P1['band']}"
        if s not in text: fails.append(('P1', fam, k, s))
        P2 = c[f'OA-VCB init{k}']['P2']; s2 = f"{P2['G']:.2f} [{P2['lo']:.2f}, {P2['hi']:.2f}] | {P2['W']:.2f}"
        if s2 not in text: fails.append(('P2', fam, k, s2))
    P3 = c['P3']; s3 = f"{P3['G']:.2f} [{P3['lo']:.2f}, {P3['hi']:.2f}], W = {P3['W']:.2f}"
    if s3 not in text: fails.append(('P3', fam, s3))
    lab = c['P1_family_label']
    if lab not in text: fails.append(('family label', fam, lab))
# ---------------- (1b) named table checks: expected cell text regenerated from the raw records
def need(label, sub):
    if sub not in text: fails.append((label, sub))
f2 = lambda x: f'{x:.2f}'
def gi(R, d=2): return f"{R['G']:.{d}f} [{R['lo']:.{d}f}, {R['hi']:.{d}f}]"
C = {f: J(os.path.join(RUNS, f'confirm_C_{f}.json')) for f in ('channels', 'barriers')}
D2 = {f: J(os.path.join(RUNS, f'd2_C_{f}.json'))['errors_D2'] for f in ('channels', 'barriers')}
def P(f, x, y):
    E = C[f]['errors']; ex = D2[f] if x == 'D2' else E[x]; ey = D2[f] if y == 'D2' else E[y]; return ST.paired(ex, ey)
# section 4.1 ladder: G(CHEB -> X) and G(X -> OA-VCB init0), both families
for row, ms in (('SD / MR', ['SD', 'MR']), ('K2-cycle (CHEB)', ['K2 (CHEB)']), ('CHEB-T', ['CHEB-T']), ('PCG (P1)', ['PCG']), ('POLY-S / POLY-L', ['POLY-S', 'POLY-L']),
                ('OA-VCL / OA-VCBL', ['OA-VCL init0', 'OA-VCBL init0']), ('SGS / SGS-PCG', ['SGS', 'SGS-PCG'])):
    line = next((l for l in text.splitlines() if l.startswith('| ' + row)), '')
    for f in ('channels', 'barriers'):
        need(f'ladder G(CHEB->X) {row} {f}', ' / '.join(f2(P(f, 'CHEB-legacy', m)['G']) for m in ms)) if ' / '.join(f2(P(f, 'CHEB-legacy', m)['G']) for m in ms) in line else fails.append(('ladder G(CHEB->X)', row, f, ' / '.join(f2(P(f, 'CHEB-legacy', m)['G']) for m in ms)))
        tgt = ' / '.join((f'{P(f, m, "OA-VCB init0")["G"]:.1f}' if P(f, m, "OA-VCB init0")["G"] >= 10 else f2(P(f, m, "OA-VCB init0")["G"])) for m in ms)
        if tgt not in line: fails.append(('ladder G(X->OA)', row, f, tgt))
for f in ('channels', 'barriers'):
    need(f'D2 ladder {f}', f"**{f2(P(f, 'CHEB-legacy', 'D2')['G'])}**"); need(f'D2->OA {f}', f"**{f2(P(f, 'D2', 'OA-VCB init0')['G'])}** [{f2(P(f, 'D2', 'OA-VCB init0')['lo'])}, {f2(P(f, 'D2', 'OA-VCB init0')['hi'])}]")
    need(f'OA-VCB ladder {f}', ' / '.join(f2(P(f, 'CHEB-legacy', f'OA-VCB init{k}')['G']) for k in (0, 1, 2)))
    # section 5.1
    for m in ('SGS', 'SGS-PCG'):
        need(f'5.1 {m} {f}', ' / '.join(f2(P(f, m, f'OA-VCB init{k}')['G']) for k in (0, 1, 2)) + f" | {f2(P(f, m, 'OA-VCB init0')['W'])}")
    # section 3 phi
    ph = C[f]['recovered_log_gain']['PCG | OA-VCB init0']; need(f'phi PCG {f}', f"φ(PCG) = {ph['phi']:.2f} [{ph['lo']:.2f}, {ph['hi']:.2f}]")
# section 4.2 D1
for f in ('channels', 'barriers'):
    d1 = J(os.path.join(RUNS, f'd1_C_{f}.json'))['summary']
    for k in ('pre', 'post'):
        v = d1[k]; need(f'D1 {f} {k}', f"| {f} | {k} | {' / '.join('%.3f' % q for q in v['rho_q'])} | {v['delta_median']:.3f} | {v['n_local_separation']}/100 |")
# section 6 subgroups (set C: PCG, K2, SGS-PCG, D2; set S: PCG, K2, SGS-PCG)
S_ = J(os.path.join(RUNS, 'confirm_S_barriers.json'))
lab = [l['category'] for l in C['barriers']['labels']]
for cat in ('island', 'strict non-island'):
    k = [j for j, l in enumerate(lab) if l == cat]; g = C['barriers']['subgroups'][cat]['gains']; gs = S_['subgroups'][cat]['gains']
    d2r = ST.paired(np.array(D2['barriers'])[k], np.array(C['barriers']['errors']['OA-VCB init0'])[k])
    cells = [gi(g['PCG -> OA-VCB init0']), gi(g['K2 (CHEB) -> OA-VCB init0']), gi(g['SGS-PCG -> OA-VCB init0']), gi(d2r),
             gi(gs['PCG -> OA-VCB init0']), gi(gs['K2 (CHEB) -> OA-VCB init0']), gi(gs['SGS-PCG -> OA-VCB init0'])]
    need(f'sec6 {cat}', ' | '.join(cells))
for cat in ('EDP', 'discordant'):
    gs = S_['subgroups'][cat]['gains']; need(f'sec6 S {cat}', ' | '.join(gi(gs[f'{x} -> OA-VCB init0']) for x in ('PCG', 'K2 (CHEB)', 'SGS-PCG')))
# section 7.1 LC
L = J(os.path.join(RUNS, 'confirm_LC_channels.json')); F = ['OA-VCB init0', 'OA-VCB init1', 'OA-VCB init2']; RL = 'OA-VCB init0 (random-load model)'
for ld, v in L['subgroups'].items():
    g = v['gains']; cell = ' / '.join(f2(g[f'PCG -> {m}']['G']) for m in F) + f" | {f2(g[f'PCG -> {RL}']['G'])} | {f2(g[f'SGS-PCG -> {RL}']['G'])}"
    need(f'LC {ld}', cell)
s = L['secondary']; need('LC pooled', ' / '.join(f2(s[f'PCG -> {m}']['G']) for m in F) + f" | {gi(s[f'PCG -> {RL}'])} | {gi(s[f'SGS-PCG -> {RL}'])}")
# section 7.2 R
for st_ in ('R2', 'R3'):
    for fam in ('channels', 'barriers'):
        d = J(os.path.join(RUNS, f'confirm_{st_}_{fam}.json')); c = d['primary_comparisons_dry_run']
        for m in [k for k in d['errors'] if k.startswith('OA-VCB')]:
            need(f'R {st_} {fam} {m}', f2(c[m]['P1']['G']))
# section 4.3 (a), (b) from the exploratory records
for f in ('channels', 'barriers'):
    x = J(os.path.join(EXP, f'x1_x3_{f}.json'))
    for k in ('pre', 'post'):
        r = x['x3'][k]; need(f'x3 {f} {k}', f"| {f} | {k} | {r['4']['q10_50_90'][1]:.3f} | {r['5']['q10_50_90'][1]:.3f} | {r['8']['q10_50_90'][1]:.3f} | {r['5']['n_below_099']}/100 | {r['8']['n_below_099']}/100 |")
    for m, v in x['x1'].items():
        R = v['to_OA'][0]; g = v['vs_CHEB_legacy']['G']; gs = f'{g:.3g}' if g >= 10 else f'{g:.2f}'
        dd = 3 if R['G'] < 0.1 else 2; cell = f"{R['G']:.{dd}f} [{R['lo']:.{dd}f}, {R['hi']:.{dd}f}]"
        if gs not in text or cell not in text: fails.append(('x1', f, m, gs, cell))
# (1c) work, timing and set-R ranges
Wk = J(os.path.join(RUNS, 'stage1_work.json'))
for meth, row in (('POLY-S', 'POLY-S / POLY-L'), ('CHEB', 'CHEB, CHEB-T'), ('PCG', 'PCG, SD'), ('SGS', 'SGS'), ('K2 (CHEB)', 'K2-cycle'), ('SGS-PCG', 'SGS-PCG')):
    w = Wk['classical']['channels'][meth]; need(f'work {meth}', f"| {row} | {w['MAC_per_pass'] / 1e6:.2f} M | {w['in_fine_matvecs']:.1f} | {w['oa_vcb_over_this']:.0f} |")
need('work OA', f"**{Wk['oa_vcb']['per_pass_MAC'] / 1e6:.0f} M** | **{Wk['oa_vcb']['per_pass_in_fine_matvecs']:,.0f}".replace(',', ' '))
lev = Wk['levels']; mac14 = sum(30 * l['nnz'] + 84 * l['n'] for l in lev)
need('work PCG14', f"| {mac14 / 1e6:.2f} M | {mac14 / lev[0]['nnz']:.1f} | {Wk['oa_vcb']['per_pass_MAC'] / mac14:.0f} |")
MC = J(os.path.join(RUNS, 'stage1_matched_cost.json'))
for fam in ('channels', 'barriers'):
    o = MC[fam]; t = o['time_to_target']
    need(f'ttt two-grid {fam}', f"{t['two-grid PCG']['median_ms']:.1f} ms"); need(f'ttt SA-AMG {fam}', f"{t['SA-AMG PCG']['median_ms']:.1f} ms")
    need(f'ttt PCG-V etc {fam}', ' / '.join(f"{t[k]['median_ms']:.0f}" for k in ('PCG-V', 'FCG-K2', 'FCG-PCGsmoothed-V')) + ' ms')
    need(f'ttt passes {fam}', ' / '.join(f"{t[k]['median_ms']:.0f}" for k in ('CHEB passes', 'PCG passes', 'SGS-PCG passes')) + ' ms')
    need(f'OA latency {fam}', f"{o['oa_vcb_latency_median_ms']:.0f} ms, error {o['one_pass']['OA-VCB batch1']['geo']:.2e}".replace('e-03', 'e-3').replace('e-02', 'e-2'))
    need(f'direct {fam}', f"{o['direct_ms']:.1f} ms")
    cl = [v['t_ms'] for k, v in o['one_pass'].items() if not k.startswith('OA')]; need(f'classical range {fam}', f"{min(cl):.0f}–{max(cl):.0f} ms")
    for k, v in t.items():
        if v['censored']: fails.append(('censored', fam, k))
sg = []
for st_ in ('R2', 'R3'):
    for fam in ('channels', 'barriers'):
        d = J(os.path.join(RUNS, f'confirm_{st_}_{fam}.json'))
        for m in [k for k in d['errors'] if k.startswith('OA-VCB') and 'random' not in k]: sg.append((fam, d['secondary'][f'SGS-PCG -> {m}']['G']))
ch = [g for f, g in sg if f == 'channels']; ba = [g for f, g in sg if f == 'barriers']
need('R SGS-PCG range', f"{min(ch):.2f}–{max(ch):.2f} for the channel fixed-load models and {min(ba):.2f}–{max(ba):.2f} for barriers")
for k, lo in zip((0, 1, 2), (1.08, 1.15, 1.18)):
    if f"{C['channels']['secondary'][f'SGS-PCG -> OA-VCB init{k}']['lo']:.2f}" != f'{lo:.2f}': fails.append(('SGS-PCG lower limit', k))
# (1d) section 4.3(c): oracle extensions on set D (exploratory records)
DEV = J(os.path.join(HERE, 'dev', 'stage1_dev_D.json'))['families']
for fam in ('channels', 'barriers'):
    x = J(os.path.join(EXP, f'x2_{fam}.json')); N = len(x['ids']); oa_ = np.array(DEV[fam]['errors']['OA-VCB init0'][:N]); ch_ = np.array(DEV[fam]['errors']['CHEB-legacy'][:N])
    for key, row in (('K4_kept', 'K₄ oracle as kept'), ('K4_continued', 'K₄ oracle, 2500 further'), ('PCG5', 'PCG(5), not an oracle'), ('K5', 'K₅ oracle')):
        e = np.exp(np.array([v['L'] if isinstance(v, dict) else v for v in x[key]])); r = ST.paired(e, oa_); c = ST.paired(ch_, e)
        line = next((l for l in text.splitlines() if l.startswith('|') and row in l), '')
        cell = f"{c['G']:.2f}" + ('**' if key == 'K5' else '') + f" | " + ('**' if key == 'K5' else '') + f"{r['G']:.2f}" + ('**' if key == 'K5' else '') + f" [{r['lo']:.2f}, {r['hi']:.2f}], W {r['W']:.2f}"
        if cell not in line: fails.append(('x2', fam, key, cell))
        V.extend([r['G'], r['lo'], r['hi'], r['W'], c['G'], 1 / r['G']])
    V.append(float(np.exp(np.mean(np.log(ch_ / oa_)))))
xb = J(os.path.join(EXP, 'x2b_channels.json'))
V.append(max(min(r[nm]['1500'] for nm in r) - min(r[nm]['4500'] for nm in r) for r in xb['runs']))
for key, row in (('K4_continued', None),):
    xx = J(os.path.join(EXP, 'x2_channels.json')); V.append(-min(v['L'] - k for v, k in zip(xx['K4_continued'], xx['K4_kept'])))
# ---------------- (2) computed quantities
for f in sorted(os.listdir(RUNS)):
    p = os.path.join(RUNS, f)
    if f.startswith('confirm_') and f.endswith('.json'):
        d = J(p)
        for key in ('confirmatory', 'primary_comparisons_dry_run', 'secondary', 'recovered_log_gain', 'summary', 'matvecs_per_pass', 'failures', 'input_scales'):
            if d.get(key): leaves(d[key])
        for k, v in d['secondary'].items(): add_pair(v)
        for cat, v in d['subgroups'].items():
            V.append(v['n']); leaves(v['summary'])
            for R in (v['gains'] or {}).values(): add_pair(R)
        if d['data'].get('set_S'): leaves(d['data']['set_S'])
        E = d['errors']
        for m, e in E.items(): V.append(float(np.exp(np.mean(np.log(e)))))
        for g in d['gates'].values():
            if isinstance(g.get('detail'), (int, float)): V.append(g['detail'])
    elif f.startswith('d1_C') and f.endswith('.json'): leaves(J(p)['summary'])
    elif f in ('stage1_work.json', 'stage1_matched_cost.json'):
        d = J(p); leaves(d)
        if f == 'stage1_matched_cost.json':
            for fam, o in d.items():
                for R in o['equal_time'].values(): add_pair(R)
for fam in ('channels', 'barriers'):                                  # D2 comparisons on set C
    d2 = J(os.path.join(RUNS, f'd2_C_{fam}.json')); d = J(os.path.join(RUNS, f'confirm_C_{fam}.json')); e = d2['errors_D2']; E = d['errors']
    V.append(float(np.exp(np.mean(np.log(e)))))
    for m in E:
        add_pair(ST.paired(E[m], e)); add_pair(ST.paired(e, E[m]))
        for m2 in E: add_pair(ST.paired(E[m], E[m2]))
    leaves(ST.phi(E['CHEB-legacy'], e, E['OA-VCB init0']))
    lab = [l['category'] for l in d['labels']]
    for cat in set(lab):
        k = [j for j, l in enumerate(lab) if l == cat]
        if len(k) >= 3: add_pair(ST.paired(np.array(e)[k], np.array(E['OA-VCB init0'])[k]))
    T = J(os.path.join(RUNS, f'timing_C_{fam}.json'))
    for k in T['records'][0]['raw_seconds']:
        V.append(float(np.median([np.median(r['raw_seconds'][k]) for r in T['records']]) * 1e3))
    V.append(float(np.median([np.median(r['oavcb_operator_data_seconds']) for r in T['records']]) * 1e3))
    for k, v in T['throughput'].items(): V.append(v * 1e3)
    d1 = J(os.path.join(RUNS, f'd1_C_{fam}.json'))
    for k in ('pre', 'post'): V.append(1 - d1['summary'][k]['rho_median'])
for f in os.listdir(EXP):
    if f.endswith('.json'):
        d = J(os.path.join(EXP, f))
        if 'x1' in d:
            for m, v in d['x1'].items():
                leaves({k: v[k] for k in ('geo', 'matvec_eq_per_pass', 'vs_CHEB_legacy')}); [add_pair(R) for R in v['to_OA']]
            leaves(d['x3'])
        if 'summary' in d: leaves(d['summary'])
# work: derived PCG(k) MACs per pass (k iterations per call, 2 calls per level, + 2 cycle residual products per level)
W = J(os.path.join(RUNS, 'stage1_work.json')); lev = W['levels']; oa = W['oa_vcb']['per_pass_MAC']
for k in (4, 5, 8, 14, 32):
    mac = sum((2 * k + 2) * l['nnz'] + 2 * k * l['n'] + 4 * k * l['n'] for l in lev)
    V += [mac, mac / 1e6, mac / lev[0]['nnz'], oa / mac]
V += [x / 1e6 for x in list(V) if x > 1e5]
# ---------------- scan
tok = re.compile(r'(?<![\w.\-/])(\d+\.\d+(?:e[−\-]?\d+)?|\d+e[−\-]?\d+)(?![\w.]*\d)')
vals = np.array(V)
def matches(t):
    s = t.replace('−', '-')
    if 'e' in s:
        x = float(s); mant = s.split('e')[0]; d = len(mant.split('.')[1]) if '.' in mant else 0
        return bool(np.any(np.abs(vals - x) <= 0.5 * 10 ** (math.floor(math.log10(abs(x))) - d) * 1.0001)) if x else True
    d = len(s.split('.')[1]); x = float(s)
    return bool(np.any(np.abs(vals - x) <= 0.5 * 10 ** (-d) * 1.0001))
un = []
for ln, line in enumerate(text.splitlines(), 1):
    for t in tok.findall(line):
        if not matches(t): un.append((ln, t, line.strip()[:110]))
print(f'headline checks: {"all passed" if not fails else fails}')
print(f'provenance scan: {len(tok.findall(text))} decimal/scientific numbers; {len(un)} without a computed match:')
for u in un: print('   line %d: %s   | %s' % u)
sys.exit(1 if fails else 0)
