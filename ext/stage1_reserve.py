"""Reserve experiment, amendment 2 (B4).  Run ONCE per family, after the amendment hash.
usage: OMP_NUM_THREADS=2 python -I ext/stage1_reserve.py FAMILY
  channels: setup(3, 2, 0, rng_seed=2026101603); barriers: setup(3, 2, 0, contrast=1e-4, load='no-low', rng_seed=2026101604)
Outputs runs/reserve_{family}.json (R1-R3) and, for channels, runs/reserve_K_local_channels.json (R4)."""
import os, sys
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(v, '2')
import json, glob, time, functools, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from scipy.sparse.linalg import spsolve
from s1_common import setup, energy
from oa_vcycle import OAData, near_kernel_basis, classical_vcycle_proj
from stage1_core import Hier, make_cheb, one_pass, pcg4, sgs_pcg4, lam_power, krylov_basis
from stage1_manifest import manifest, Gates
import stage1_models as SM, stage1_stats as ST
torch.set_num_threads(2)
FAM = sys.argv[1]; assert FAM in ('channels', 'barriers')
SEED = {'channels': 2026101603, 'barriers': 2026101604}[FAM]
OUT = os.path.join(HERE, 'runs', f'reserve_{FAM}.json'); OUTK = os.path.join(HERE, 'runs', 'reserve_K_local_channels.json')
SMOKE = os.environ.get('STAGE1_SMOKE') == '1'       # code-path test on DEVELOPMENT set D (5 operators), outputs to the scratch dir
if SMOKE:
    SD = os.environ['STAGE1_SMOKE_DIR']; OUT = os.path.join(SD, f'smoke_reserve_{FAM}.json'); OUTK = os.path.join(SD, 'smoke_reserve_K.json')
    print('*** SMOKE TEST on development data; not a reserve result ***')
G = Gates()
# gate: the reserve seed appears in no record before generation, and the outputs do not exist yet
hits = [p for p in glob.glob(os.path.join(HERE, '**', '*'), recursive=True) if p.endswith(('.json', '.log')) and str(SEED) in open(p, errors='ignore').read()]
G.check(f'reserve seed {SEED} absent from every record before generation', not hits and not os.path.exists(OUT), hits[:3])
if G.failed(): sys.exit(1)
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
t00 = time.time(); S = setup(3, 2, 1, **kw) if SMOKE else setup(3, 2, 0, rng_seed=SEED, **kw); te = S['te'][:5] if SMOKE else S['te']; H = S['H']; nl = S['nl']
print(f'reserve {FAM}: data generated ({len(te)} operators) in {time.time() - t00:.0f} s', flush=True)
CK = json.load(open(os.path.join(HERE, 'checkpoint_map.json')))
LEARNED = sorted((r['init_seed'], os.path.join(ROOT, r['path'])) for r in CK if r['family'] == FAM and r['model'] == 'oa-vcb' and r['data_seed'] == 1
                 and r['level'] == 3 and r['k0'] == 2 and r['oa_steps'] == 4 and r['coarse'] == 'perforated' and r['load'] in ('f=1', 'no-low'))
G.check('three fixed-load OA-VCB initialisations', [k for k, _ in LEARNED] == [0, 1, 2], [k for k, _ in LEARNED])
USC = json.load(open(os.path.join(HERE, 'frozen_scales.json')))['scales'][f'{FAM}/fixed-load']
M = {f'PCG({k})': functools.partial(pcg4, its=k) for k in (4, 5, 8, 14)}; M['SGS-PCG(4)'] = sgs_pcg4
E = {m: [] for m in ['CHEB-legacy', 'CHEB', 'PCG4-direct'] + list(M)}; E_UC = []
for i in te:
    A = S['A'][i]; u = S['U'][i]; uc = S['UC'][i]; r = S['Fs'][i] - A @ uc; hr = Hier(H, A, nl); Z = near_kernel_basis(A); E_UC.append(energy(A, uc, u))
    E['CHEB-legacy'].append(energy(A, uc + classical_vcycle_proj(hr.levs, hr.A0, r, Z=Z, nu=4), u))
    ch = make_cheb([3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]); E['CHEB'].append(energy(A, uc + one_pass(hr, r, Z, ch, ch), u))
    E['PCG4-direct'].append(energy(A, uc + one_pass(hr, r, Z, pcg4, pcg4), u))
    for m, f in M.items():
        try: E[m].append(float(energy(A, uc + one_pass(hr, r, Z, f, f), u)))
        except Exception as exc: E[m].append(float('nan')); print('failure', m, i, repr(exc))
G.check('CHEB common path = legacy anchor (<= 1e-8)', np.max(np.abs(np.array(E['CHEB']) / np.array(E['CHEB-legacy']) - 1)) <= 1e-8)
G.check('PCG(4) row = stage1_core.pcg4 through the wrapper', np.max(np.abs(np.array(E['PCG(4)']) / np.array(E['PCG4-direct']) - 1)) == 0)
D = OAData(H, [S['a'][i] for i in te], nl, 4, 'cpu', H.level_coords, near_kernel=True)
R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te]); NETS = {}
for k, p in LEARNED:
    net, _, _ = SM.load(p, D); NETS[k] = net
    C = SM.apply(net, D, R, USC); E[f'OA-VCB init{k}'] = [float(energy(S['A'][i], S['UC'][i] + c, S['U'][i])) for i, c in zip(te, C)]
ids = [int(i) for i in te]; res = dict(family=FAM, rng_seed=SEED, ids=ids, errors=E, error_uc=E_UC, endpoints={})
pk = ('PCG(8)', 'PCG(14)') if FAM == 'channels' else ('PCG(5)', 'PCG(8)')
for k in (0, 1, 2):
    nm = f'OA-VCB init{k}'; lo_ = ST.paired(E[pk[0]], E[nm], ids); hi_ = ST.paired(E[pk[1]], E[nm], ids)
    res['endpoints'][nm] = dict(**{f'{m} -> OA-VCB': ST.paired(E[m], E[nm], ids) for m in M},
                                bracket=dict(lower_method=pk[0], upper_method=pk[1], lower_limit_of_lower=lo_['lo'], upper_limit_of_upper=hi_['hi'],
                                             confirmed=bool(lo_['lo'] > 1 and hi_['hi'] < 1)))
res['endpoints']['PCG(14) -> SGS-PCG(4)'] = ST.paired(E['PCG(14)'], E['SGS-PCG(4)'], ids)
res['endpoints']['family_bracket_confirmed'] = all(res['endpoints'][f'OA-VCB init{k}']['bracket']['confirmed'] for k in (0, 1, 2))
res['endpoints']['CHEB-legacy -> X'] = {m: ST.paired(E['CHEB-legacy'], E[m], ids)['G'] for m in E if m not in ('CHEB-legacy', 'PCG4-direct')}
print(f"\n== reserve {FAM} ({'R1' if FAM == 'channels' else 'R2'}: bracket {pk[0]} .. {pk[1]})")
for k in (0, 1, 2):
    e = res['endpoints'][f'OA-VCB init{k}']
    print(f"  init{k}: " + '; '.join(f"{m} {e[f'{m} -> OA-VCB']['G']:.3f} [{e[f'{m} -> OA-VCB']['lo']:.3f}, {e[f'{m} -> OA-VCB']['hi']:.3f}]" for m in M) + f"  -> bracket confirmed: {e['bracket']['confirmed']}")
print('  family bracket confirmed:', res['endpoints']['family_bracket_confirmed'], '; G(PCG14 -> SGS-PCG4):', round(res['endpoints']['PCG(14) -> SGS-PCG(4)']['G'], 3))
# ---------------- R4: local K_d test (channels)
if FAM == 'channels':
    loc = {}
    for k, net in NETS.items():
        calls = []; orig = net._smooth
        def spy(Sm, b, l, net=net, orig=orig):
            x = orig(Sm, b, l)
            if l == 0: calls.append(('pre' if Sm is net.pre[0] else 'post', b.detach().double().numpy()[0].copy(), x.detach().double().numpy()[0].copy()))
            return x
        net._smooth = spy; rec = {'pre': [], 'post': []}
        for j, i in enumerate(te):
            calls.clear(); A = S['A'][i]; r = S['Fs'][i] - A @ S['UC'][i]; jt = torch.tensor([j])
            with torch.no_grad():
                net.set_ops(*D.batch(jt), Z=D.batch_z(jt), A64=D.batch_a64(jt)); net(torch.tensor(r[None] / USC, dtype=torch.float64))
            for kind, b, x in calls:
                uu = spsolve(A.tocsc(), b); En = float(np.sqrt((uu - x) @ (A @ (uu - x)))); d_ = dict(id=int(i))
                for d in (4, 5, 8):
                    Q, rk = krylov_basis(A, b, d); xK = Q @ (Q.T @ b); EK = float(np.sqrt((uu - xK) @ (A @ (uu - xK))))
                    xp = pcg4(A, 1 / A.diagonal(), b, 0, its=d); Ep = float(np.sqrt((uu - xp) @ (A @ (uu - xp))))
                    d_[f'K{d}'] = dict(rank=rk, rho=En / EK if EK > 0 else None, orth_defect=float(np.max(np.abs(Q.T @ (A @ Q) - np.eye(rk)))) if rk else 0.0,
                                       pcg_vs_K=Ep / EK - 1 if EK > 0 else None)
                rec[kind].append(d_)
        net._smooth = orig; loc[f'OA-VCB init{k}'] = rec
    summ = {}
    for nm, rec in loc.items():
        for kind, rows in rec.items():
            for d in (4, 5, 8):
                ok = [r_[f'K{d}'] for r_ in rows if r_[f'K{d}']['rank'] == d and r_[f'K{d}']['rho'] is not None]
                rho = np.array([o['rho'] for o in ok])
                summ[f'{nm} {kind} K{d}'] = dict(n_full_rank=len(ok), excluded=[r_['id'] for r_ in rows if r_[f'K{d}']['rank'] != d],
                                                 rho_q10_50_90=[float(q) for q in np.quantile(rho, [.1, .5, .9])], n_rho_le_099=int((rho <= 0.99).sum()),
                                                 max_orth_defect=float(max(o['orth_defect'] for o in ok)), max_pcg_vs_K=float(max(abs(o['pcg_vs_K']) for o in ok)),
                                                 confirmed=(bool((rho <= 0.99).sum() >= (4 if SMOKE else 90)) if d == 5 else None))
                s_ = summ[f'{nm} {kind} K{d}']
                print(f"  R4 {nm} {kind} K{d}: rho q10/50/90 {['%.3f' % q for q in s_['rho_q10_50_90']]}, rho<=0.99 on {s_['n_rho_le_099']}/{s_['n_full_rank']}"
                      + (f", K5 separation confirmed: {s_['confirmed']}" if d == 5 else '') + f"; orth {s_['max_orth_defect']:.1e}, |PCG/K-1| {s_['max_pcg_vs_K']:.1e}")
    json.dump(dict(family=FAM, rng_seed=SEED, ids=ids, summary=summ, records=loc, gates=G.g), open(OUTK, 'w'), indent=1)
res['gates'] = G.g; res['manifest'] = manifest([p for _, p in LEARNED]); res['seconds'] = time.time() - t00
G.g['checkpoints match map'] = dict(passed=all(v['matches_map'] for v in res['manifest']['checkpoints'].values()), detail=None)
json.dump(res, open(OUT, 'w'), indent=1)
if G.failed(): print('GATES FAILED', G.failed()); sys.exit(1)
