"""Print the result tables from the jsonl files train_oa_layer.py writes.

  python summarize_runs.py --out-dir $OUT
  python summarize_runs.py --out-dir $OUT --field channels

Groups by (route, level, field, epochs, width, rank) and aggregates over seeds:
mean +- sd of test rel-L2 and energy error, with the seed count.
"""
import argparse, glob, json, os
from collections import defaultdict
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--out-dir', default='.')
ap.add_argument('--field', default=None)
ap.add_argument('--markdown', action='store_true', help='emit markdown tables')
args = ap.parse_args()

rows = defaultdict(lambda: defaultdict(list))
for path in sorted(glob.glob(os.path.join(args.out_dir, 'results_train_oa_L*.jsonl'))):
    for line in open(path):
        rec = json.loads(line)
        c = json.loads(rec['cfg'])
        if args.field and c['field'] != args.field:
            continue
        key = (c.get('route', 'W'), c['level'], c['field'], c['epochs'], c['width'], c['rank'])
        rows[key][rec['model']].append(rec['result'])

if not rows:
    print(f"no results files in {args.out_dir}"); raise SystemExit

ORDER = ['oa', 'oa-hb', 'oa-frozen', 'oa-norank', 'fno', 'wno']
for key in sorted(rows):
    route, level, field, epochs, width, rank = key
    print(f"\nroute {route}, level {level}, {field}, {epochs} epochs, width {width}, rank {rank}")
    hdr = f"{'model':<11}{'params':>9}{'test rel-L2':>22}{'energy err':>22}{'seeds':>7}"
    print(hdr if not args.markdown else "| model | params | test rel-L2 | energy err | seeds |\n|---|---|---|---|---|")
    for m in ORDER:
        if m not in rows[key]:
            continue
        rs = rows[key][m]
        l2 = np.array([r['rel_l2'] for r in rs]); en = np.array([r['energy'] for r in rs])
        p = rs[0]['params']
        def fmt(v):
            return f"{v.mean():.3e}" + (f" +- {v.std(ddof=1):.1e}" if len(v) > 1 else "")
        if args.markdown:
            print(f"| {m} | {p:,} | {fmt(l2)} | {fmt(en)} | {len(rs)} |")
        else:
            print(f"{m:<11}{p:>9,}{fmt(l2):>22}{fmt(en):>22}{len(rs):>7}")
