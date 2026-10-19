"""Command line:   python -m oavcb {train,eval,check} [options]

    python -m oavcb check                                      # verify the implementation (about a minute, CPU)
    python -m oavcb train --out-dir runs/channels              # paper configuration: channels 1e4, L = 3, 300 epochs
    python -m oavcb train --preset barriers --data-seed 1      # barriers 1e-4 with f = 1 on the conductive part
    python -m oavcb train --preset l4 --data-seed 1            # four refinements (activation checkpointing)
    python -m oavcb eval  --ckpt runs/channels/ckpt_<key>.pt   # evaluate a checkpoint (also the paper's oa-vcb checkpoints)

Every option of DataConfig / ModelConfig / TrainConfig (config.py) is a flag, e.g. --level 2 --epochs 20 --n 100.
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

from .config import DataConfig, ModelConfig, TrainConfig, to_json

PRESETS = {
    'channels': {},
    'barriers': dict(contrast=1e-4, load='no-low'),
    'polygon': dict(coarse='polygon'),
    'l4': dict(level=4, k0=3, act_ckpt=True),
    'k0': dict(k0=0),
}


def _add_flags(p, cls, over):
    for f in dataclasses.fields(cls):
        flag, default = '--' + f.name.replace('_', '-'), over.get(f.name, f.default)
        if isinstance(f.default, bool):
            p.add_argument(flag, action='store_true', default=default)
        else:
            p.add_argument(flag, type=type(f.default), default=default)


def _pick(args, cls):
    return cls(**{f.name: getattr(args, f.name) for f in dataclasses.fields(cls)})


def _parser(over=None):
    over = over or {}                                                         # a preset only changes the defaults
    ap = argparse.ArgumentParser(prog='python -m oavcb', description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('check', help='verify the implementation against the statements of the paper').add_argument(
        '--quick', action='store_true')
    for name in ('train', 'eval'):
        p = sub.add_parser(name)
        p.add_argument('--preset', choices=sorted(PRESETS), default='channels')
        p.add_argument('--out-dir', default='runs')
        p.add_argument('--cache-dir', default=None, help='cache of the generated samples (default: <out-dir>/cache)')
        p.add_argument('--classical', action='store_true', help='also evaluate the classical references and the paired G / W')
        for cls in (DataConfig, ModelConfig, TrainConfig):
            _add_flags(p, cls, over)
        if name == 'eval':
            p.add_argument('--ckpt', required=True)
    return ap


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pre, _ = _parser().parse_known_args(argv)
    if pre.cmd == 'check':
        from .checks import main as run_checks
        return run_checks(quick=pre.quick)
    args = _parser(PRESETS[pre.preset]).parse_args(argv)

    import torch
    from .evaluate import classical_reference, evaluate, report
    from .pipeline import Problem, load_weights, run_key
    from .train import fit

    dcfg, mcfg, tcfg = _pick(args, DataConfig), _pick(args, ModelConfig), _pick(args, TrainConfig)
    os.makedirs(args.out_dir, exist_ok=True)
    prob = Problem(dcfg, mcfg, tcfg.device, args.cache_dir or os.path.join(args.out_dir, 'cache'))
    model = prob.new_model(tcfg)
    from .vcycle import n_params
    print(f'{mcfg.variant}: width {model.pre[0].C}, {n_params(model):,} parameters (budget {mcfg.params:,})')
    key = run_key(dcfg, mcfg, tcfg)
    if args.cmd == 'train':
        ck = os.path.join(args.out_dir, f'ckpt_{key}.pt')
        train_s = fit(prob, model, tcfg, ck, key)
    else:
        load_weights(args.ckpt, model, prob.dev); train_s = float('nan')
    res = evaluate(prob, model, batch=tcfg.batch)
    ref = classical_reference(prob, nu=mcfg.nu) if args.classical else None
    report(res, ref)
    rec = dict(key=key, config=json.loads(to_json(dcfg, mcfg, tcfg)), params=n_params(model), train_s=train_s,
               build_ms=prob.build_ms, infer_ms=res['infer_ms'], n_worse_than_uc=res['n_worse_than_uc'],
               test_ids=res['idx'].tolist(),
               net={k: v.tolist() for k, v in res['net'].items()}, coarse={k: v.tolist() for k, v in res['coarse'].items()})
    if ref:
        rec['classical'] = {m: {k: v.tolist() for k, v in e.items()} for m, e in ref.items()}
    with open(os.path.join(args.out_dir, f'results_{key}.json'), 'w') as fh:
        json.dump(rec, fh)
    return 0


if __name__ == '__main__':
    sys.exit(main())
