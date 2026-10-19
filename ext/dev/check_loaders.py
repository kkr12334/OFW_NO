"""Amendment 1 / A1: the architecture-specific loaders reproduce the stored seed-1 test errors of OA-VCB, OA-VCBL and
OA-VCL (both families).  Tolerances as Stage 0: mean 1e-5, non-island samples 1e-4 (relative).  Exit 1 on failure."""
import os, sys, json, glob, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
sys.path[:0] = [EXT, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy
from oa_vcycle import OAData, near_kernel_basis
import stage1_models as SM
torch.set_num_threads(2)
CK = {('channels', 'oa-vcb'): 'resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt',
      ('channels', 'oa-vcbl'): 'Results_kaggle_session 2/ckpt_40ce3ac1f9_oa-vcbl.pt',
      ('channels', 'oa-vcl'): 'Results_kaggle_session 2/ckpt_40ce3ac1f9_oa-vcl.pt',
      ('barriers', 'oa-vcb'): 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt',
      ('barriers', 'oa-vcbl'): 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcbl.pt',
      ('barriers', 'oa-vcl'): 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcl.pt'}
REC = {'channels': ['resultskaggle_2026-10-06session_1/results/darcy_ours/*.jsonl', 'Results_kaggle_session 2/linear_controls*.jsonl'],
       'barriers': ['Results_kaggle_session 3-3b/results_*.jsonl']}
fails = []; out = {}
for fam, kw, con in (('channels', {}, 1e4), ('barriers', dict(contrast=1e-4, load='no-low'), 1e-4)):
    S = setup(3, 2, 1, **kw); te = S['te']; H = S['H']
    D = OAData(H, [S['a'][i] for i in te], S['nl'], 4, 'cpu', H.level_coords, near_kernel=True)
    As = [S['A'][i] for i in te]; R = np.array([S['Fs'][i] - A @ S['UC'][i] for i, A in zip(te, As)])
    isl = np.array([near_kernel_basis(A).shape[1] > 0 for A in As])
    for t in ('oa-vcb', 'oa-vcbl', 'oa-vcl'):
        stored = None
        for pat in REC[fam]:
            for f in glob.glob(os.path.join(ROOT, pat)):
                for l in open(f):
                    r = json.loads(l); c = json.loads(r['cfg'])
                    if r['model'] == t and c.get('data_seed') == 1 and c.get('contrast') == con and c.get('seed', 0) == 0 and c.get('k0') == 2:
                        assert stored is None, 'ambiguous'; stored = np.array(r['result']['en_s'])
        net, st, tt = SM.load(os.path.join(ROOT, CK[(fam, t)]), D)
        C = SM.apply(net, D, R, S['U_scale'])
        e = np.array([energy(A, S['UC'][i] + c, S['U'][i]) for i, A, c in zip(te, As, C)])
        d = np.abs(e / stored - 1); dm = abs(e.mean() / stored.mean() - 1); gate = d[~isl].max() if fam == 'barriers' else d.max()
        ok = dm <= 1e-5 and gate <= 1e-4
        out[f'{fam}.{t}'] = dict(mean_rel=float(dm), gated_max_rel=float(gate), island_max_rel=float(d[isl].max()) if isl.any() else None,
                                 mean_new=float(e.mean()), mean_stored=float(stored.mean()), ok=bool(ok), flags=SM.FLAGS[t])
        print(('PASS ' if ok else 'FAIL ') + f'{fam}.{t}', out[f'{fam}.{t}'])
        if not ok: fails.append(f'{fam}.{t}')
json.dump(dict(results=out, failed=fails), open(os.path.join(HERE, 'check_loaders.json'), 'w'), indent=1)
sys.exit(1 if fails else 0)
