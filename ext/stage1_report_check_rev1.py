"""Claims check for STAGE1_REPORT_REV1_2026-10-18.md: every table cell and quoted number is regenerated from the raw
records and must appear verbatim (named checks).  Exit 1 on any failure.  usage: python -I ext/stage1_report_check_rev1.py"""
import os, sys, json, re, numpy as np, subprocess
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); EXP = os.path.join(HERE, 'explore'); sys.path.insert(0, HERE)
import stage1_stats as ST
text = open(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else 'STAGE1_REPORT_REV1_2026-10-18.md')).read(); J = lambda p: json.load(open(p))
fails = []; n_checks = [0]
def need(label, sub):
    n_checks[0] += 1
    if sub not in text: fails.append((label, sub))
f2 = lambda x: f'{x:.2f}'
def gi(R, d=2): return f"{R['G']:.{d}f} [{R['lo']:.{d}f}, {R['hi']:.{d}f}]"
gm = lambda e: float(np.exp(np.mean(np.log(e))))
# (0) generated sections present verbatim
subprocess.run([sys.executable, '-I', os.path.join(HERE, 'stage1_rev1_snippets.py')], capture_output=True, check=True)
SN = J(os.path.join(RUNS, 'rev1_snippets.json'))
for k in ('LEDGER', 'GELU', 'TIMING', 'RESERVE', 'R4'):
    if k in SN: need(f'snippet {k}', SN[k])
    if '{{' + k + '}}' in text: fails.append(('placeholder left', k))
subprocess.run([sys.executable, '-I', os.path.join(HERE, 'stage1_rev1_summary.py')], capture_output=True, check=True)
need('snippet SUMMARY', J(os.path.join(RUNS, 'rev1_summary.json'))['SUMMARY'])
C = {f: J(os.path.join(RUNS, f'confirm_C_{f}.json')) for f in ('channels', 'barriers')}
D2 = {f: J(os.path.join(RUNS, f'd2_C_{f}.json'))['errors_D2'] for f in C}
def P(f, x, y):
    E = C[f]['errors']; return ST.paired(D2[f] if x == 'D2' else E[x], D2[f] if y == 'D2' else E[y])
# (1) section 2
for f in C:
    c = C[f]['confirmatory']
    for k in (0, 1, 2):
        p1, p2 = c[f'OA-VCB init{k}']['P1'], c[f'OA-VCB init{k}']['P2']
        need(f'P1 {f} {k}', f"| {f} | {k} | {gi(p1)} | {f2(p1['W'])} |"); need(f'P2 {f} {k}', f"| {gi(p2)} |")
    need(f'P3 {f}', f"{f2(c['P3']['G'])} [{f2(c['P3']['lo'])}, {f2(c['P3']['hi'])}] on {f}")
    ph = C[f]['recovered_log_gain']['PCG | OA-VCB init0']; need(f'phi {f}', f"{ph['phi']:.2f} [{ph['lo']:.2f}, {ph['hi']:.2f}] on {f}")
# (2) section 3 ladder
for row, ms in (('SD / MR', ['SD', 'MR']), ('K2-cycle (CHEB)', ['K2 (CHEB)']), ('CHEB-T', ['CHEB-T']), ('PCG (P1 control)', ['PCG']), ('POLY-S / POLY-L', ['POLY-S', 'POLY-L']),
                ('OA-VCL / OA-VCBL', ['OA-VCL init0', 'OA-VCBL init0']), ('SGS / SGS-PCG', ['SGS', 'SGS-PCG']), ('Best per-instance polynomial candidate found (D2)', ['D2'])):
    line = next((l for l in text.splitlines() if l.startswith('| ' + row)), '')
    cells = [' / '.join(f2(P(f, 'CHEB-legacy', m)['G']) for m in ms) for f in C]
    cells += [' / '.join((f'{P(f, m, "OA-VCB init0")["G"]:.1f}' if P(f, m, "OA-VCB init0")["G"] >= 10 else f2(P(f, m, "OA-VCB init0")["G"])) for m in ms) for f in C]
    if ms == ['D2']: cells[2:] = [gi(P(f, 'D2', 'OA-VCB init0')) for f in C]
    n_checks[0] += 1
    if not line or '| ' + ' | '.join(cells) + ' |' not in line: fails.append(('ladder', row, cells))
need('OA-VCB ladder', ' | '.join(' / '.join(f2(P(f, 'CHEB-legacy', f'OA-VCB init{k}')['G']) for k in (0, 1, 2)) for f in C))
need('D2 barriers W', f"W = {f2(P('barriers', 'D2', 'OA-VCB init0')['W'])} for OA-VCB")
# (3) section 4
for f in C:
    d1 = J(os.path.join(RUNS, f'd1_C_{f}.json'))['summary']
    for pos in ('pre', 'post'):
        v = d1[pos]; need(f'D1 {f} {pos}', f"| {f} | {pos} | {' / '.join('%.3f' % q for q in v['rho_q'])} | {v['delta_median']:.3f} | {v['n_local_separation']}/100 |")
need('D1 pcg max', f"{max(J(os.path.join(RUNS, f'd1_C_{f}.json'))['summary'][p]['pcg_vs_K_max'] for f in C for p in ('pre', 'post')):.1e}".replace('e-13', 'e-13'))
X = {f: J(os.path.join(EXP, f'x1_x3_{f}.json')) for f in C}
x3 = lambda f, pos, d: X[f]['x3'][pos][str(d)]
need('K5 ch', f"median ρ_K₅ is {x3('channels', 'pre', 5)['q10_50_90'][1]:.3f} (pre) and {x3('channels', 'post', 5)['q10_50_90'][1]:.3f} (post); ρ_K₅ ≤ 0.99 on {x3('channels', 'pre', 5)['n_below_099']}/100")
need('K5 ba', f"Pre: {x3('barriers', 'pre', 5)['n_below_099']}/100, median {x3('barriers', 'pre', 5)['q10_50_90'][1]:.3f}. Post: {x3('barriers', 'post', 5)['n_below_099']}/100, median {x3('barriers', 'post', 5)['q10_50_90'][1]:.3f}.")
need('K8 ch', f"(median ρ_K₈ {x3('channels', 'pre', 8)['q10_50_90'][1]:.3f} pre, {x3('channels', 'post', 8)['q10_50_90'][1]:.3f} post). OA-VCB still beats K₈ on {x3('channels', 'pre', 8)['n_below_099']}/100 (pre) and {x3('channels', 'post', 8)['n_below_099']}/100 (post)")
# (4) section 5
for m in ('PCG(4)', 'PCG(5)', 'PCG(8)', 'PCG(14)', 'SGS-PCG(4)'):
    cells = []
    for f in C:
        v = X[f]['x1'][m]; g = v['vs_CHEB_legacy']['G']; cells += [f'{g:.3g}' if g >= 10 else f2(g), gi(v['to_OA'][0])]
    need(f'5.1 {m}', f"| {m} | " + ' | '.join(cells) + ' |')
x2 = J(os.path.join(EXP, 'x2_channels.json')); N = len(x2['ids']); dev = J(os.path.join(HERE, 'dev', 'stage1_dev_D.json'))['families']['channels']['errors']
need('K5 candidate', f"{gi(ST.paired(np.exp(x2['K5']), dev['OA-VCB init0'][:N]))}".replace(' [', '× ['))
for f in C:
    need(f'5.3 SGS-PCG {f}', ' / '.join(f2(P(f, 'SGS-PCG', f'OA-VCB init{k}')['G']) for k in (0, 1, 2)))
    r = ST.paired(X[f]['x1']['PCG(14)']['errors'], X[f]['x1']['SGS-PCG(4)']['errors']); need(f'PCG14/SGS {f}', f"{r['G']:.3f} [{r['lo']:.3f}, {r['hi']:.3f}] on {f}")   # G(SGS-PCG4 <- PCG14) = geo e_PCG14 / e_SGS-PCG4
need('5.3 W', f"W = {f2(P('channels', 'SGS-PCG', 'OA-VCB init0')['W'])} for OA-VCB on channels and {f2(P('barriers', 'SGS-PCG', 'OA-VCB init0')['W'])} on barriers")
# (5) section 6 text
Lg = J(os.path.join(RUNS, 'stage1_ledger.json'))
need('nk mean', f"near-kernel columns is {Lg['near_kernel_columns']['barriers']['mean']:.2f}"); need('ratio change', f"at most {100 * SN['ledger_barrier_ratio_change']:.1f}%")
need('160x', f"about {round(Lg['families']['channels']['PCG(14)']['OA_VCB_over_method'], -1):.0f}× PCG(14)")
need('reductions', f"{Lg['families']['channels']['OA-VCB']['nonlinear_global_reductions_per_pass']} nonlinear global reductions")
# (6) section 8
S_ = J(os.path.join(RUNS, 'confirm_S_barriers.json')); lab = [l['category'] for l in C['barriers']['labels']]
for cat in ('island', 'strict non-island'):
    k = [j for j, l in enumerate(lab) if l == cat]; g = C['barriers']['subgroups'][cat]['gains']; gs = S_['subgroups'][cat]['gains']
    d2r = ST.paired(np.array(D2['barriers'])[k], np.array(C['barriers']['errors']['OA-VCB init0'])[k])
    need(f'sec8 {cat}', ' | '.join([gi(g['PCG -> OA-VCB init0']), gi(g['K2 (CHEB) -> OA-VCB init0']), gi(g['SGS-PCG -> OA-VCB init0']), gi(d2r),
                                    gi(gs['PCG -> OA-VCB init0']), gi(gs['K2 (CHEB) -> OA-VCB init0']), gi(gs['SGS-PCG -> OA-VCB init0'])]))
LC = J(os.path.join(RUNS, 'confirm_LC_channels.json')); need('LC RL', gi(LC['secondary']['PCG -> OA-VCB init0 (random-load model)']))
for st_ in ('R2', 'R3'):
    d = J(os.path.join(RUNS, f'confirm_{st_}_channels.json')); need(f'R {st_} random', f2(d['primary_comparisons_dry_run']['OA-VCB init0 (random-load model)']['P1']['G']))
print(f"rev1 claims check: {n_checks[0]} named checks; failures: {fails if fails else 'none'}")
sys.exit(1 if fails else 0)
