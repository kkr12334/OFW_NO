"""Run the OFW-NO experiment suite, in value order, resumable.

Every run goes through train_oa_layer.py, which appends each finished model to
<out-dir>/results_train_oa_L{level}_{field}_route{R}.jsonl and skips models already
there for the same configuration.  So if Colab drops, rerun the SAME command and it
continues where it stopped.  Use summarize_runs.py to print the tables.

  python run_suite.py --plan seeds    --out-dir $OUT     # 1. error bars on the headline (3 seeds)
  python run_suite.py --plan fields   --out-dir $OUT     # 2. blocks8 + grf, is it channel-specific?
  python run_suite.py --plan capacity --out-dir $OUT     # 3. the ~676k budget of the old table
  python run_suite.py --plan long     --out-dir $OUT     # 4. 400 epochs (nothing had converged)
  python run_suite.py --plan data     --out-dir $OUT     # 5. 600 and 1000 samples (data is the limit)
  python run_suite.py --plan all      --out-dir $OUT     # everything, in that order
  python run_suite.py --plan all --dry-run               # just print the commands

Rough GPU times per plan (level 5, 300 samples, from your 200-epoch runs):
  seeds ~70 min, fields ~50 min, capacity ~2-3 h, long ~50 min.
"""
import argparse, itertools, subprocess, sys, time

ap = argparse.ArgumentParser()
ap.add_argument('--plan', choices=['seeds', 'fields', 'capacity', 'long', 'data', 'all'], default='all')
ap.add_argument('--out-dir', default='.')
ap.add_argument('--level', type=int, default=5)
ap.add_argument('--n', type=int, default=300)
ap.add_argument('--python', default=sys.executable)
ap.add_argument('--epochs', type=int, default=200, help='epochs for every plan except "long"')
ap.add_argument('--long-epochs', type=int, default=400)
ap.add_argument('--dry-run', action='store_true')
args = ap.parse_args()

BASE = [args.python, '-u', 'train_oa_layer.py', '--level', str(args.level), '--n', str(args.n),
        '--route', 'V', '--rank', '0', '--out-dir', args.out_dir]
MODELS = ['oa', 'oa-hb', 'oa-frozen', 'fno', 'wno']


def cmd(field, epochs, seed, width=None, models=MODELS):
    c = BASE + ['--field', field, '--epochs', str(epochs), '--seed', str(seed), '--models'] + models
    return c + (['--width', str(width)] if width else [])


plans = {
    # 1. error bars: the headline run repeated with two more seeds
    'seeds': [cmd('channels', args.epochs, s) for s in (1, 2)],
    # 2. is the ordering channel-specific?
    'fields': [cmd(f, args.epochs, 0) for f in ('blocks8', 'grf')],
    # 3. the capacity of baseline-comparison-results.md (~676k): width 96
    'capacity': [cmd('channels', args.epochs, 0, width=96)],
    # 4. convergence: nothing had flattened at 200
    'long': [cmd('channels', args.long_epochs, 0)],
    # 5. sample scaling: the capacity/length sweeps say data is the binding constraint
    'data': [BASE[:BASE.index('--n') + 1] + [str(n)] + BASE[BASE.index('--n') + 2:] +
             ['--field', 'channels', '--epochs', str(args.long_epochs), '--seed', '0',
              '--models'] + MODELS for n in (600, 1000)],
}
order = ['seeds', 'fields', 'capacity', 'long', 'data']
todo = list(itertools.chain(*(plans[p] for p in (order if args.plan == 'all' else [args.plan]))))

print(f"{len(todo)} runs queued (plan '{args.plan}'). Rerun this exact command after any "
      f"interruption; finished models are skipped.\n")
for i, c in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {' '.join(c)}", flush=True)
    if args.dry_run:
        continue
    t0 = time.time()
    r = subprocess.run(c)
    print(f"[{i}/{len(todo)}] {'ok' if r.returncode == 0 else 'FAILED rc=%d' % r.returncode} "
          f"in {(time.time() - t0) / 60:.1f} min\n", flush=True)
    if r.returncode != 0:
        print("stopping; fix the error and rerun the same command to continue"); break
