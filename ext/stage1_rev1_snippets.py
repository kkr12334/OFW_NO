"""Generates the data-driven sections of STAGE1_REPORT_REV1 (ledger, repaired timings, reserve results) from the raw
records and writes them to runs/rev1_snippets.json; with --fill, replaces the {{...}} placeholders in the report.
usage: python -I ext/stage1_rev1_snippets.py [--fill]"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); sys.path.insert(0, HERE)
import stage1_stats as ST
J = lambda p: json.load(open(os.path.join(RUNS, p)))
gm = lambda e: float(np.exp(np.mean(np.log(e))))
def gi(R, d=2): return f"{R['G']:.{d}f} [{R['lo']:.{d}f}, {R['hi']:.{d}f}]"
SN = {}
# ---------------- ledger
L = J('stage1_ledger.json'); rows = L['families']['channels']; rb = L['families']['barriers']
order = ['POLY-S', 'SGS', 'SD', 'CHEB', 'CHEB-T', 'POLY-L', 'PCG', 'MR', 'PCG(5)', 'PCG(8)', 'K2 (CHEB)', 'SGS-PCG', 'PCG(14)', 'OA-VCB']
lines = []
for m in order:
    r = rows[m]; lines.append(f"| {m} | {r['per_pass_MAC'] / 1e6:.2f} M | {r['per_pass_fine_spmv_equiv']:.1f} | {r['terms']['smoother'] / 1e6:.2f} M | "
                             f"{(r['setup_MAC'] / 1e6):.2f} M | {r['OA_VCB_over_method']:.0f} |")
SN['LEDGER'] = '\n'.join(lines)
SN['GELU'] = f"{rows['OA-VCB']['GELU_per_pass'] / 1e6:.1f} M"
SN['ledger_barrier_ratio_change'] = max(abs(rb[m]['OA_VCB_over_method'] / rows[m]['OA_VCB_over_method'] - 1) for m in order)
# ---------------- repaired timings
T = {f: J(f'timing_v2_C_{f}.json') for f in ('channels', 'barriers') if os.path.exists(os.path.join(RUNS, f'timing_v2_C_{f}.json'))}
if len(T) == 2:
    tl = ['| Quantity (median over 50 operators; ms) | Channels | Barriers |', '|---|---|---|']
    def med(f, k): return float(np.median([np.median(r['raw_seconds'][k]) for r in T[f]['records']]) * 1e3)
    def prof(f, k): return float(np.median([r['profile'][k] for r in T[f]['records']]) * 1e3)
    uc = {f: med(f, 'u_c construction') for f in T}
    cl = {f: [med(f, k) for k in T[f]['records'][0]['raw_seconds'] if k.startswith('1pass') and 'OA-VCB' not in k] for f in T}
    tl.append(f"| u_c construction (coarse Galerkin, factorise, solve, prolong) | {uc['channels']:.1f} | {uc['barriers']:.1f} |")
    tl.append(f"| Classical one pass, any smoother, conditional on u_c (range over 10 methods) | {min(cl['channels']):.0f}–{max(cl['channels']):.0f} | {min(cl['barriers']):.0f}–{max(cl['barriers']):.0f} |")
    tl.append(f"| … of which operator-adapted hierarchy (profile) | {prof('channels', 'hierarchy'):.1f} | {prof('barriers', 'hierarchy'):.1f} |")
    tl.append(f"| … near-kernel basis / legacy CHEB bounds / Lanczos bounds (profile) | {prof('channels', 'near-kernel basis'):.1f} / {prof('channels', 'bounds: CHEB legacy (2 power)'):.1f} / {prof('channels', 'bounds: Lanczos (3 levels)'):.1f} | "
              f"{prof('barriers', 'near-kernel basis'):.1f} / {prof('barriers', 'bounds: CHEB legacy (2 power)'):.1f} / {prof('barriers', 'bounds: Lanczos (3 levels)'):.1f} |")
    tl.append(f"| … cycle + wrapper only: PCG / SGS-PCG (profile) | {prof('channels', 'cycle + wrapper: PCG'):.1f} / {prof('channels', 'cycle + wrapper: SGS-PCG'):.1f} | {prof('barriers', 'cycle + wrapper: PCG'):.1f} / {prof('barriers', 'cycle + wrapper: SGS-PCG'):.1f} |")
    tl.append(f"| OA-VCB one pass, batch 1, conditional on u_c | {med('channels', '1pass OA-VCB batch1'):.0f} | {med('barriers', '1pass OA-VCB batch1'):.0f} |")
    tl.append(f"| … of which operator data incl. hierarchy (profile; median over operators of the last of five repeats) | {prof('channels', 'OA-VCB operator data (last repeat)'):.0f} | {prof('barriers', 'OA-VCB operator data (last repeat)'):.0f} |")
    tl.append(f"| Mesh patterns, once per mesh (excluded) | {T['channels']['patterns_seconds_once_per_mesh'] * 1e3:.0f} | {T['barriers']['patterns_seconds_once_per_mesh'] * 1e3:.0f} |")
    tl.append(f"| Sparse direct solve (never charged u_c) | {med('channels', 'direct'):.1f} | {med('barriers', 'direct'):.1f} |")
    for f in T:
        thr = T[f]['throughput']
    tl.append(f"| OA-VCB batch-8 throughput per operator (secondary) | {T['channels']['throughput']['OA-VCB batch8 (operator data + residual + inference) per operator'] * 1e3:.0f} | {T['barriers']['throughput']['OA-VCB batch8 (operator data + residual + inference) per operator'] * 1e3:.0f} |")
    # curves
    cv = ['| Whole solve from u_c (median ms over operators; conditional / estimated end-to-end incl. u_c) | Channels: to OA-VCB error (offline oracle) | Channels: offline-selected prefix to rel. residual 1e-6 | Barriers: to OA-VCB error (offline oracle) | Barriers: offline-selected prefix to rel. residual 1e-6 |', '|---|---|---|---|---|']
    curv = [k[6:] for k in T['channels']['records'][0]['raw_seconds'] if k.startswith('curve')]
    stat = {}
    def hit_times(f, k, crit):
        out, cens = [], 0
        for r in T[f]['records']:
            st = np.median(np.array(r['curve_stamps'][f'curve {k}']), 0)
            seq = r['curve_errors'][k] if crit == 'err' else r['curve_relres'][k]; tgt = r['errors']['1pass OA-VCB batch1'] if crit == 'err' else 1e-6
            h = next((j for j, v in enumerate(seq) if v <= tgt), None)
            u_ = np.median(r['raw_seconds']['u_c construction']) * 1e3
            if h is None: cens += 1; out.append((np.inf, np.inf))        # censored = not reached within the frozen limit
            else: out.append((st[h] * 1e3, st[h] * 1e3 + u_))             # (conditional, per-operator sum with its own u_c time)
        return out, cens
    for k in curv:
        cells = []
        for f in ('channels', 'barriers'):
            for crit in ('err', 'res'):
                t_, c_ = hit_times(f, k, crit)
                md = float(np.median([a for a, _ in t_])); me = float(np.median([b for _, b in t_]))   # medians over all 50; censored = not reached
                cells.append((f"{md:.1f} / {me:.1f}" if np.isfinite(md) else 'median not reached') + (f" (censored {c_}/50)" if c_ else ''))
            stat[(f, k)] = {s: sum(1 for r in T[f]['records'] if r['curve_status'][k] == s) for s in set(r['curve_status'][k] for r in T[f]['records'])}
        cv.append(f"| {k} | " + ' | '.join(cells) + ' |')
    def oa_e2e(f): return float(np.median([np.median(r['raw_seconds']['1pass OA-VCB batch1']) * 1e3 + np.median(r['raw_seconds']['u_c construction']) * 1e3 for r in T[f]['records']]))
    oa = ['| OA-VCB one pass (for comparison) | ' + ' | '.join(f"{med(f, '1pass OA-VCB batch1'):.0f} / {oa_e2e(f):.0f} (its own error) | one-pass relative residual not recorded in this timing experiment" for f in ('channels', 'barriers')) + ' |']
    allok = all(list(v.keys()) == ['completed'] for v in stat.values())
    errs = {f: (gm([r['errors']['1pass OA-VCB batch1'] for r in T[f]['records']]), gm([r['errors']['u_c'] for r in T[f]['records']])) for f in T}
    SN['TIMING'] = ('\n'.join(tl) + '\n\n' + '\n'.join(cv + oa) + '\n\n'
                    + f"- Medians are over all 50 operators, with censored operators (target not reached within the frozen iteration limit) counted as not reached; 'median not reached' means at least half were censored. "
                    + "End-to-end values are medians of per-operator sums (each operator's median u_c time plus its median timestamp at the first qualifying iteration): an estimate assembled from separately timed components, not a directly timed combined solve. "
                    + "Offline oracle stopping (\"to OA-VCB error\") uses each operator's true error. The relative-residual columns are offline-selected prefix costs to a deployable residual threshold, excluding convergence-check overhead: the residual was evaluated in an untimed run and the timed runs did not stop, so they are not measured online stopping times. "
                    + f"Every outer PCG/FCG loop and pass sequence terminated as 'completed' on all 50 operators per family: {allok}.\n"
                    + f"- OA-VCB's geometric-mean error on these 50 operators: {errs['channels'][0]:.2e} (channels), {errs['barriers'][0]:.2e} (barriers); u_c alone: {errs['channels'][1]:.2e}, {errs['barriers'][1]:.2e}.\n"
                    + f"- Compared with v1, classical one-pass latency drops from ~80 ms to the values above (unused bound set-up removed), and OA-VCB latency drops from ~370 ms (patterns no longer rebuilt in the timer). The qualitative conclusion is unchanged.")
    SN['timing_status_ok'] = allok
# ---------------- reserve
RS = {f: J(f'reserve_{f}.json') for f in ('channels', 'barriers') if os.path.exists(os.path.join(RUNS, f'reserve_{f}.json'))}
if len(RS) == 2:
    rl = ['| Family | Init | ' + ' | '.join(f'G({m} → OA-VCB)' for m in ('PCG(4)', 'PCG(5)', 'PCG(8)', 'PCG(14)', 'SGS-PCG(4)')) + ' | Bracket confirmed |', '|---|---|---|---|---|---|---|---|']
    for f, R in RS.items():
        for k in (0, 1, 2):
            e = R['endpoints'][f'OA-VCB init{k}']
            rl.append(f"| {f} | {k} | " + ' | '.join(gi(e[f'{m} -> OA-VCB'], 3 if e[f'{m} -> OA-VCB']['G'] < 0.1 else 2) for m in ('PCG(4)', 'PCG(5)', 'PCG(8)', 'PCG(14)', 'SGS-PCG(4)'))
                      + f" | {'yes' if e['bracket']['confirmed'] else 'no'} ({e['bracket']['lower_method']}–{e['bracket']['upper_method']}) |")
    cR = RS['channels']['endpoints']; bR = RS['barriers']['endpoints']
    SN['RESERVE'] = (f"100 fresh operators per family (rng 2026101603 channels, 2026101604 barriers), generated after the amendment-2 hash. Every recorded gate passed. The frozen-scale gate specified in B4 was not executed at run time; the scale file's current SHA-256 matches its pre-confirmation hash (8d0cc6f1…, verified retrospectively on 18 October, which does not prove its contents at run time). All recorded gates passed: "
                     f"{all(v['passed'] for R in RS.values() for v in R['gates'].values())}. Paired G with pointwise 95% intervals; W omitted for space (in the records).\n\n"
                     + '\n'.join(rl) + '\n\n'
                     + f"- **R1 (primary, channels):** bracketing PCG(8)–PCG(14) confirmed for all three initialisations: {cR['family_bracket_confirmed']}.\n"
                     + f"- **R2 (barriers):** bracketing PCG(5)–PCG(8) confirmed for all three initialisations: {bR['family_bracket_confirmed']}.\n"
                     + f"- **R3:** G(PCG(14) → SGS-PCG(4)) = {gi(cR['PCG(14) -> SGS-PCG(4)'])} (channels), {gi(bR['PCG(14) -> SGS-PCG(4)'])} (barriers).")
    SN['R1'] = cR['family_bracket_confirmed']; SN['R2'] = bR['family_bracket_confirmed']
if os.path.exists(os.path.join(RUNS, 'reserve_K_local_channels.json')):
    K = J('reserve_K_local_channels.json')['summary']
    KR = J('reserve_K_local_channels.json')['records']
    kl = ['| Init | Position | ρ_K₄ median | ρ_K₅ 10 / 50 / 90% | ρ_K₅ ≤ 0.99 | K₅ confirmed | ρ_K₈ median | ρ_K₈ ≤ 0.99 | ρ_K₈ < 1 | max ‖QᵀAQ − I‖ | max \\|E_PCG/E_K − 1\\| |', '|---|---|---|---|---|---|---|---|---|---|---|']
    allc = True
    for k in (0, 1, 2):
        for pos in ('pre', 'post'):
            a4, a5, a8 = (K[f'OA-VCB init{k} {pos} K{d}'] for d in (4, 5, 8)); allc &= bool(a5['confirmed'])
            kl.append(f"| {k} | {pos} | {a4['rho_q10_50_90'][1]:.3f} | {' / '.join('%.3f' % q for q in a5['rho_q10_50_90'])} | {a5['n_rho_le_099']}/{a5['n_full_rank']} | {'yes' if a5['confirmed'] else 'no'} | "
                      f"{a8['rho_q10_50_90'][1]:.3f} | {a8['n_rho_le_099']}/{a8['n_full_rank']} | {sum(1 for r_ in KR[f'OA-VCB init{k}'][pos] if r_['K8']['rank'] == 8 and r_['K8']['rho'] < 1)}/{a8['n_full_rank']} | {max(a4['max_orth_defect'], a5['max_orth_defect'], a8['max_orth_defect']):.1e} | "
                      f"{max(a4['max_pcg_vs_K'], a5['max_pcg_vs_K'], a8['max_pcg_vs_K']):.1e} |")
    SN['R4'] = ('Reserve channels (100 operators, all three inits, frozen criterion ρ_K₅ ≤ 0.99 on ≥ 90 full-rank operators):\n\n' + '\n'.join(kl)
                + f"\n\nLocal K₅ separation confirmed at both positions for all three initialisations: {allc}.")
    SN['R4_all'] = allc
json.dump(SN, open(os.path.join(RUNS, 'rev1_snippets.json'), 'w'), indent=1)
for k, v in SN.items(): print(f'--- {k}\n{v}\n')
if '--fill' in sys.argv:
    p = os.path.join(HERE, 'STAGE1_REPORT_REV1_2026-10-18.md'); s = open(p).read()
    for k in ('LEDGER', 'GELU', 'TIMING', 'RESERVE', 'R4'):
        if k in SN: s = s.replace('{{' + k + '}}', SN[k])
    open(p, 'w').write(s); print('filled:', [k for k in ('LEDGER', 'GELU', 'TIMING', 'RESERVE', 'R4') if k in SN])
