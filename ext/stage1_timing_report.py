"""Summarise a stage1_timing.py JSON (amendment 1, A4): per method, median over operators of the per-operator median
latency (and of the per-operator minimum); errors; for whole-solve curves the offline time to reach OA-VCB's per-sample
error (labelled offline/oracle; censored if not reached within the frozen iteration limit).
usage: python -I ext/stage1_timing_report.py TIMING.json [...]"""
import sys, json, numpy as np
for path in sys.argv[1:]:
    T = json.load(open(path)); recs = T['records']
    print(f"\n== {T['family']} (set {T['set']}), {len(recs)} operators, {T['repeats']} repeats; CPU {T['env']['cpu']}, threads {T['env']['threads']}")
    oa = np.array([r['errors']['1pass OA-VCB batch1'] for r in recs])
    print(f"   {'method':<28}{'latency median [ms]':>20}{'(min-based)':>13}{'geo-mean error':>16}{'G vs CHEB':>11}")
    ch = np.array([r['errors']['1pass CHEB'] for r in recs])
    for k in recs[0]['raw_seconds']:
        if k.startswith('curve'): continue
        med = np.median([np.median(r['raw_seconds'][k]) for r in recs]) * 1e3; mn = np.median([np.min(r['raw_seconds'][k]) for r in recs]) * 1e3
        e = np.array([r['errors'].get(k, np.nan) for r in recs]) if k != 'direct' else None
        if e is not None:
            print(f"   {k:<28}{med:>20.1f}{mn:>13.1f}{np.exp(np.mean(np.log(e))):>16.3e}{np.exp(np.mean(np.log(ch / e))):>11.2f}")
        else: print(f"   {k:<28}{med:>20.1f}{mn:>13.1f}{'(solver accuracy)':>16}")
    od = np.median([np.median(r['oavcb_operator_data_seconds']) for r in recs]) * 1e3
    print(f"   OA-VCB batch1 of which operator data: {od:.1f} ms (patterns once per mesh: {T['patterns_seconds_once_per_mesh'] * 1e3:.0f} ms, excluded)")
    print("   time to reach OA-VCB's per-sample error (OFFLINE oracle stopping; median over operators; censored = not reached):")
    for k in recs[0]['raw_seconds']:
        if not k.startswith('curve'): continue
        tt, cens = [], 0
        for r, eo in zip(recs, oa):
            h = r['errors'][k]; hit = next((i for i, v in enumerate(h) if v <= eo), None)
            if hit is None: cens += 1; continue
            tt.append(np.median([s[hit] for s in r['curve_stamps'][k]]))
        print(f"     {k[6:]:<24} {('%.1f ms' % (1e3 * np.median(tt))) if tt else '-':>10}  iterations-to-target median {np.median([next((i for i, v in enumerate(r['errors'][k]) if v <= eo), np.nan) for r, eo in zip(recs, oa)]) + 1:.0f}  censored {cens}/{len(recs)}")
    print('   throughput (secondary):', {k: f'{1e3 * v:.1f} ms' for k, v in T['throughput'].items()})
