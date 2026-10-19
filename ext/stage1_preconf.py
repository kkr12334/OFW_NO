"""Machine-readable part of the pre-confirmation note (amendment 1, A5.6).  Reads ONLY development outputs (set T fits,
set D runs) and writes ext/stage1_preconfirmation.json, which the confirmation runners read (they never re-derive a
selection).  usage: python -I ext/stage1_preconf.py"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from stage1_manifest import fitted_methods_sha, FROZEN_FIT_SHA
DEV = json.load(open(os.path.join(HERE, 'dev', 'stage1_dev_D.json'))); FIT = json.load(open(os.path.join(HERE, 'dev', 'stage1_fit.json')))
fm = os.path.join(HERE, 'dev', 'stage1_fitted_methods.json'); h = fitted_methods_sha(fm)
assert h == FIT['coefficients_sha256'] == FROZEN_FIT_SHA, (h, FIT['coefficients_sha256'])
CANDS = ['CHEB-T', 'POLY-S', 'POLY-L', 'SD', 'MR', 'PCG']                  # A5.1 order (ties within 1% -> earlier)
out = dict(note='frozen before set C is generated; derived from set T fits and set D runs only', fitted_methods=os.path.relpath(fm, HERE),
           fitted_methods_sha256=h, families={})
for fam, v in DEV['families'].items():
    E = v['errors']; ml = {c: float(np.mean(np.log(E[c]))) for c in CANDS}
    best = min(ml.values()); champ = next(c for c in CANDS if ml[c] <= best + np.log(1.01))
    trig = {}
    for k in (0, 1, 2):
        nm = f'OA-VCB init{k}'
        trig[nm] = dict(PCG_lo=v['comparisons'][f'PCG -> {nm}']['lo'], K2_lo=v['comparisons'][f'K2 (CHEB) -> {nm}']['lo'])
    fired = all(t['PCG_lo'] >= 1.25 and t['K2_lo'] >= 1.25 for t in trig.values())
    F = FIT[fam]
    out['families'][fam] = dict(
        cheb_t=dict(selected=F['cheb_t_selected'], grid_mean_log_error_T=F['cheb_t_grid']),
        poly_s=dict(beta=F['beta'], runs=[{k: r[k] for k in ('start', 'objective', 'grad_inf', 'evaluations')} for r in F['poly_s']['runs']],
                    kept=F['poly_s']['best']['start']),
        poly_l=dict(runs=[{k: r[k] for k in ('start', 'objective', 'grad_inf', 'evaluations')} for r in F['poly_l']['runs']],
                    kept=F['poly_l']['best']['start']),
        champion=dict(candidates_mean_log_error_D=ml, selected=champ),
        spatial_trigger=dict(lower_limits=trig, fired=fired, include=['SGS', 'SGS-PCG'] if fired else []),
        labels_D={c: sum(l['category'] == c for l in v['labels']) for c in sorted(set(l['category'] for l in v['labels']))})
    print(fam, 'champion', champ, {k: round(x, 4) for k, x in ml.items()}, '| spatial trigger', fired, '| CHEB-T', F['cheb_t_selected'])
out['input_scales'] = json.load(open(os.path.join(HERE, 'frozen_scales.json')))['scales']
json.dump(out, open(os.path.join(HERE, 'stage1_preconfirmation.json'), 'w'), indent=1)
