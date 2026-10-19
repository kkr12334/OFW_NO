"""Matched-cost summaries and the accuracy/time figure (protocol v1 §5 'matched cost', §9 'accuracy/time figure').
Reads ext/runs/timing_C_{family}.json only (first 50 set-C operators per family; latency medians of 5 repeats).
  (a) errors at equal wall-clock time: per operator, the best whole-solve curve point whose time (median over repeats of
      the iteration timestamp, set-up included) is <= OA-VCB's batch-1 latency; if no iteration finishes in time, the curve
      is credited with u_c's error (its starting point).  Reported: G(curve at OA-VCB's time -> OA-VCB) and W.
  (b) time to reach OA-VCB's per-sample error (offline oracle stopping; censored if not reached).
  (c) figure ext/runs/fig_stage1_accuracy_time.png/.pdf: geometric-mean error vs median time, both families.
usage: python -I ext/stage1_matched_cost.py"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); RUNS = os.path.join(HERE, 'runs'); sys.path.insert(0, HERE)
import stage1_stats as ST
out = {}
for fam in ('channels', 'barriers'):
    T = json.load(open(os.path.join(RUNS, f'timing_C_{fam}.json'))); recs = T['records']
    C = json.load(open(os.path.join(RUNS, f'confirm_C_{fam}.json'))); euc = dict(zip(C['data']['ids'], C['error_uc']))
    oa_t = np.array([np.median(r['raw_seconds']['1pass OA-VCB batch1']) for r in recs]); oa_e = np.array([r['errors']['1pass OA-VCB batch1'] for r in recs])
    curves = [k for k in recs[0]['raw_seconds'] if k.startswith('curve')]
    eq, ttt, fig = {}, {}, {}
    for k in curves:
        e_at, t_hit, n_none = [], [], 0
        for r, t0, e0 in zip(recs, oa_t, oa_e):
            st = np.median(np.array(r['curve_stamps'][k]), 0); h = np.array(r['errors'][k])
            ok = st <= t0
            if ok.any(): e_at.append(h[ok].min())
            else: e_at.append(euc[r['id']]); n_none += 1
            hit = np.nonzero(h <= e0)[0]; t_hit.append(st[hit[0]] if len(hit) else np.nan)
        R = ST.paired(e_at, oa_e); R['credited_u_c'] = n_none; eq[k[6:]] = R
        th = np.array(t_hit); ttt[k[6:]] = dict(median_ms=float(np.nanmedian(th) * 1e3) if np.isfinite(th).any() else None, censored=int(np.isnan(th).sum()))
        fig[k[6:]] = dict(t_ms=(np.median([np.median(np.array(r['curve_stamps'][k]), 0) for r in recs], 0) * 1e3).tolist(),
                          geo=np.exp(np.mean(np.log([r['errors'][k] for r in recs]), 0)).tolist())
    one = {k[6:]: dict(t_ms=float(np.median([np.median(r['raw_seconds'][k]) for r in recs]) * 1e3),
                       geo=float(np.exp(np.mean(np.log([r['errors'][k] for r in recs]))))) for k in recs[0]['raw_seconds'] if k.startswith('1pass')}
    out[fam] = dict(n=len(recs), oa_vcb_latency_median_ms=float(np.median(oa_t) * 1e3), equal_time=eq, time_to_target=ttt, curves=fig, one_pass=one,
                    direct_ms=float(np.median([np.median(r['raw_seconds']['direct']) for r in recs]) * 1e3))
    print(f"\n== {fam}: OA-VCB median latency {out[fam]['oa_vcb_latency_median_ms']:.0f} ms")
    for k, R in eq.items(): print(f"   at OA-VCB's time: {k:<20} G(curve -> OA-VCB) {R['G']:.2e} [{R['lo']:.2e}, {R['hi']:.2e}] W {R['W']:.2f} (credited u_c: {R['credited_u_c']})")
json.dump(out, open(os.path.join(RUNS, 'stage1_matched_cost.json'), 'w'), indent=1)

# ---------------- figure (static; light surface; categorical slots in fixed order, every series direct-labelled)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
SER = {'OA-VCB (one pass)': '#2a78d6', 'two-grid PCG': '#eb6834', 'SA-AMG PCG': '#1baf7a', 'CHEB passes': '#eda100', 'SGS-PCG passes': '#e87ba4'}
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.facecolor': SURF, 'figure.facecolor': SURF, 'savefig.facecolor': SURF})
fg, axs = plt.subplots(1, 2, figsize=(10, 4.2), sharey=False)
for ax, fam in zip(axs, ('channels', 'barriers')):
    o = out[fam]; ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True, which='major', color=GRID, lw=0.6); ax.set_axisbelow(True)
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    for nm, key in (('two-grid PCG', 'two-grid PCG'), ('SA-AMG PCG', 'SA-AMG PCG'), ('CHEB passes', 'CHEB passes'), ('SGS-PCG passes', 'SGS-PCG passes')):
        c = o['curves'][key]; ax.plot(c['t_ms'], c['geo'], '-o', color=SER[nm], lw=2, ms=4, mec=SURF, mew=1)
        if 'passes' in nm:                                   # label at the last point inside the plotted range
            j = max(i for i, g in enumerate(c['geo']) if g >= 1.5e-6); dy = 0 if nm.startswith('CHEB') else -10
            ax.annotate(nm, (c['t_ms'][j], c['geo'][j]), xytext=(6, dy), textcoords='offset points', color=INK, fontsize=8, va='center')
        else:                                                # label at the first iteration
            two = nm.startswith('two')
            ax.annotate(nm, (c['t_ms'][0], c['geo'][0]), xytext=(-4 if two else 5, 4), textcoords='offset points', color=INK, fontsize=8, va='bottom',
                        ha='right' if two else 'left')
    for k, v in o['one_pass'].items():
        if k.startswith('OA-VCB'): continue
        ax.plot(v['t_ms'], v['geo'], 'o', color='#a8a7a2', ms=5, mec=SURF, mew=1, zorder=3)
    lab = {'CHEB': 'CHEB (reference)', 'SGS-PCG': 'SGS-PCG'}
    for k, v in o['one_pass'].items():
        if k in lab: ax.annotate(lab[k], (v['t_ms'], v['geo']), xytext=(-6, 0), textcoords='offset points', ha='right', color=INK2, fontsize=7.5, va='center')
    v = o['one_pass']['OA-VCB batch1']; ax.plot(v['t_ms'], v['geo'], 'D', color=SER['OA-VCB (one pass)'], ms=8, mec=SURF, mew=1.5, zorder=4)
    ax.annotate('OA-VCB (one pass)', (v['t_ms'], v['geo']), xytext=(0, 9), textcoords='offset points', ha='center', color=INK, fontsize=8, fontweight='bold')
    ax.axvline(o['direct_ms'], color=INK2, lw=1, ls=(0, (3, 3))); ax.annotate('direct solve', (o['direct_ms'], ax.get_ylim()[0] if False else 2e-6), xytext=(3, 0), textcoords='offset points', color=INK2, fontsize=7.5, rotation=90, va='bottom')
    ax.set_xlabel('time per operator from $u_c$, set-up included [ms] (median)'); ax.set_ylabel('relative energy error (geometric mean)')
    ax.set_title(f'{fam} (set C, {o["n"]} operators)', color=INK, fontsize=10, loc='left'); ax.set_ylim(1e-6, 0.4); ax.set_xlim(0.6, 900)
fg.text(0.01, 0.01, 'Grey dots: one-pass classical methods in the same cycle (CHEB, CHEB-T, POLY-S/L, SD, MR, PCG, K2, SGS, SGS-PCG; each ~80 ms,\nthe operator-adapted hierarchy dominates). Lines: one point per iteration or pass; whole-solve curves start from $u_c$ and include their own set-up.',
        color=INK2, fontsize=7.5)
fg.tight_layout(rect=(0, 0.07, 1, 1))
for ext in ('png', 'pdf'): fg.savefig(os.path.join(RUNS, f'fig_stage1_accuracy_time.{ext}'), dpi=170)
print('figure written')
