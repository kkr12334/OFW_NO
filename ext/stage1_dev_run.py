"""Stage 1 DEVELOPMENT run (set D: seed-1 test ids 300-399; amendment 1).  Not a confirmation run.
usage: python -I ext/stage1_dev_run.py [extra_methods.json]   -> ext/dev/stage1_dev_D.json
Methods: CHEB-legacy (classical_vcycle_proj), CHEB (common wrapper), SD, MR, PCG; optional fitted/tuned smoothers from
a JSON file (POLY-S, POLY-L, CHEB-T, written by stage1_fit.py); learned: OA-VCB init 0/1/2, OA-VCBL, OA-VCL."""
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy
from oa_vcycle import OAData, near_kernel_basis, classical_vcycle_proj
from stage1_core import Hier, make_cheb, make_poly, pcg4, sd4, mr4, sgs4, sgs_pcg4, one_pass, one_pass_k2, lam_power, lam_lanczos, lam_gershgorin
from stage1_labels import label
import stage1_models as SM, stage1_stats as ST
torch.set_num_threads(2)
CK = json.load(open(os.path.join(HERE, 'checkpoint_map.json')))
def ck(fam, model, init, ds=1):
    hits = [r['path'] for r in CK if r['family'] == fam and r['model'] == model and r['init_seed'] == init and r['data_seed'] == ds
            and r['level'] == 3 and r['k0'] == 2 and r['oa_steps'] == 4 and r['coarse'] == 'perforated' and r['load'] in ('f=1', 'no-low')]
    assert len(hits) <= 1, hits
    return os.path.join(ROOT, hits[0]) if hits else None
EXTRA = json.load(open(sys.argv[1])) if len(sys.argv) > 1 else {}

out = dict(set='D (development: seed-1 test ids 300-399)', families={})
for fam, kw in (('channels', {}), ('barriers', dict(contrast=1e-4, load='no-low'))):
    t0 = time.time(); S = setup(3, 2, 1, **kw); te = S['te']; H = S['H']; nl = S['nl']
    As = [S['A'][i] for i in te]; U = S['U'][te]; UC = S['UC'][te]
    R = np.array([S['Fs'][i] - A @ uc for i, A, uc in zip(te, As, UC)])
    E = {}; work = {}; status = {}; labs = [label(H, S['a'][i], A) for i, A in zip(te, As)]
    ex = EXTRA.get(fam, {})
    def methods(hr):
        lam_leg = [3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]
        m = {'CHEB': (make_cheb(lam_leg), make_cheb(lam_leg)), 'SD': (sd4, sd4), 'MR': (mr4, mr4), 'PCG': (pcg4, pcg4),
             'SGS (spatial, conditional)': (sgs4, sgs4), 'SGS-PCG (spatial, conditional)': (sgs_pcg4, sgs_pcg4)}
        if 'CHEB-T' in ex:
            c = ex['CHEB-T']; f = lam_gershgorin if c['bound'] == 'gershgorin' else lam_lanczos
            lt = [f(Ak) for Ak, _ in hr.levs]; m['CHEB-T'] = (make_cheb(lt, q=c['q']), make_cheb(lt, q=c['q']))
        for nm in ('POLY-S', 'POLY-L'):
            if nm in ex:
                c = ex[nm]
                sc = [c['beta']] * hr.nl if nm == 'POLY-S' else [lam_lanczos(Ak) for Ak, _ in hr.levs]
                m[nm] = (make_poly(np.array(c['pre']), sc), make_poly(np.array(c['post']), sc))
        return m
    for j, (i, A) in enumerate(zip(te, As)):
        hr = Hier(H, A, nl); Z = near_kernel_basis(A)
        e_leg = energy(A, UC[j] + classical_vcycle_proj(hr.levs, hr.A0, R[j], Z=Z, nu=4), U[j])
        E.setdefault('CHEB-legacy', []).append(e_leg)
        ch_ = make_cheb([3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]])
        E.setdefault('K2 (CHEB)', []).append(energy(A, UC[j] + one_pass_k2(hr, R[j], Z, ch_, ch_), U[j]))
        for nm, (pre, post) in methods(hr).items():
            cnt = hr.new_counter(); st = {}
            try:
                x = one_pass(hr, R[j], Z, pre, post, cnt=cnt, stats=st); e = energy(A, UC[j] + x, U[j])
                if not np.isfinite(e): st['nonfinite'] = 1
            except Exception as exc:
                e = np.nan; st['crash'] = repr(exc)
            E.setdefault(nm, []).append(e); work.setdefault(nm, []).append(cnt); status.setdefault(nm, []).append({**st, **cnt['status']})
    # learned models
    D = OAData(H, [S['a'][i] for i in te], nl, 4, 'cpu', H.level_coords, near_kernel=True)
    learned = [('OA-VCB', 'oa-vcb', k) for k in (0, 1, 2)] + [('OA-VCBL', 'oa-vcbl', k) for k in (0, 1, 2)] + [('OA-VCL', 'oa-vcl', 0)]
    for nm, t, k in learned:
        p = ck(fam, t, k)
        if p is None: continue
        net, st, _ = SM.load(p, D); C = SM.apply(net, D, R, S['U_scale'])
        E[f'{nm} init{k}'] = [energy(A, uc + c, u) for A, uc, c, u in zip(As, UC, C, U)]
    # comparisons
    comp = {}
    ref = E['CHEB']
    for nm, e in E.items():
        if nm != 'CHEB': comp[f'CHEB -> {nm}'] = ST.paired(ref, e, te)
    for k in (0, 1, 2):
        nm = f'OA-VCB init{k}'
        if nm in E:
            for c in [m for m in E if not m.startswith('OA-') and m != 'CHEB-legacy']:
                Rk = ST.paired(E[c], E[nm], te); Rk['band'] = ST.band(Rk) if c in ('PCG', 'K2 (CHEB)') else None; comp[f'{c} -> {nm}'] = Rk
    cats = sorted(set(l['category'] for l in labs))
    sub = {c: {nm: ST.summary([e for e, l in zip(E[nm], labs) if l['category'] == c]) for nm in E} for c in cats}
    subG = {c: {nm: ST.paired([e for e, l in zip(E['CHEB'], labs) if l['category'] == c], [e for e, l in zip(E[nm], labs) if l['category'] == c])
                for nm in E if nm != 'CHEB'} for c in cats if sum(l['category'] == c for l in labs) >= 3}
    mv = {nm: float(np.mean([sum(w['matvec']) for w in ws])) for nm, ws in work.items()}
    out['families'][fam] = dict(ids=[int(i) for i in te], labels=labs, errors={k: list(map(float, v)) for k, v in E.items()},
                                summary={k: ST.summary(v) for k, v in E.items()}, comparisons=comp, subgroups=sub, subgroup_gains=subG,
                                matvecs_per_pass=mv, status=status, seconds=time.time() - t0)
    print(f"\n== {fam} (D), {time.time() - t0:.0f} s; labels { {c: sum(l['category'] == c for l in labs) for c in cats} }")
    for nm in E: print(f"  {nm:<16} mean {np.mean(E[nm]):.4e}  median {np.median(E[nm]):.4e}" + (f"  matvec/pass {mv[nm]:.0f}" if nm in mv else ''))
    for k, v in comp.items():
        if k.startswith('CHEB ->') or k.startswith('PCG ->') or k.startswith('POLY-S ->') or k.startswith('K2') or 'SGS' in k.split(' -> ')[0]:
            print(f"  {k:<30} G {v['G']:.3f} [{v['lo']:.3f}, {v['hi']:.3f}] W {v['W']:.2f}" + (f"  band: {v['band']}" if v.get('band') else ''))
json.dump(out, open(os.path.join(HERE, 'dev', 'stage1_dev_D.json'), 'w'), indent=1)
