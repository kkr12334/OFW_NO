"""Rebuild the exact train_oa_layer.py command behind each configuration stored in the results files, so new
models can be added to an existing table with identical settings (the results key must match exactly).

    python show_commands.py                                   # scans /content/drive/MyDrive/ofwno_runs
    python show_commands.py <folder or results file> [--with oa-ll] [--level 3]

Prints, per configuration: the file, the models already finished, and the command (without --models /
--out-dir).  Append  --models <new models> --out-dir <folder>  to it.  Finished models are skipped automatically.
"""
import argparse, glob, json, os

FLAG = dict(level='--level', field='--field', contrast='--contrast', n='--n', epochs='--epochs', batch='--batch',
            lr='--lr', width='--width', layers='--layers', rank='--rank', rank_scale='--rank-scale',
            radius='--radius', rho='--rho', seed='--seed', route='--route', loss='--loss', k0='--k0',
            domain='--domain', bc='--bc', coarse='--coarse', coarse_n='--coarse-n', grid_ng='--grid-ng',
            data_seed='--data-seed', params='--params', coarse_rho='--coarse-rho', coarse_solve='--coarse-solve',
            ll_scaling='--ll-scaling', ll_output='--ll-output', ll_ablate='--ll-ablate')


def show(path, with_model=None, level=None):
    files = sorted(glob.glob(os.path.join(path, 'results_train_oa_L*.jsonl'))) if os.path.isdir(path) else [path]
    if not files:
        print(f"no results_train_oa_L*.jsonl in {path}; files there: {sorted(os.listdir(path))[:40]}"); return
    for f in files:
        cfgs = {}
        for line in open(f):
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            cfgs.setdefault(r['cfg'], []).append((r['model'], r['result']))
        for cfg, runs in cfgs.items():
            models = [m for m, _ in runs]
            c = json.loads(cfg)
            if with_model and with_model not in models: continue
            if level and c.get('level') != level: continue
            parts = ['python -u train_oa_layer.py']
            for k, v in c.items():
                if v is None or k not in FLAG or (k == 'grid_ng' and not v): continue
                parts.append(f"{FLAG[k]} {v}")
            print('=' * 100)
            print(os.path.basename(f))
            print('finished: ' + ', '.join(f"{m} ({r.get('rel_l2', float('nan')):.3e} / {r.get('energy', float('nan')):.3e}, "
                                           f"{r.get('params', 0):,} params)" for m, r in runs))
            print(' '.join(parts))
            unknown = [k for k in c if k not in FLAG]
            if unknown: print('  (keys without a flag, check by hand: ' + ', '.join(unknown) + ')')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('path', nargs='?', default='/content/drive/MyDrive/ofwno_runs')
    ap.add_argument('--with', dest='with_model', default=None)
    ap.add_argument('--level', type=int, default=None)
    a = ap.parse_args()
    if not os.path.exists(a.path):
        raise SystemExit(f"{a.path} does not exist -- is Google Drive mounted, and is this the --out-dir of the runs?")
    show(a.path, a.with_model, a.level)
