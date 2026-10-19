"""10 Oct: the Session 5 k0_0 results file was overwritten when the job folders were merged (both Session 5 jobs write
results_train_oa_L3_perforatedd9_channels_routeV.jsonl; the copy in the folder is geo_transfers').  This re-evaluates
the k0 = 0 OA-VCB checkpoints on CPU through the driver's exact data and inference path (s1_common) and writes driver-
format records (cfg = the checkpoint's own cfg, per-sample energies, sample_ids) to k0_0_oa-vcb_reeval.jsonl.  The means
are checked against the GPU log (k0_0.log).   usage: python diagnostics_cost/reeval_k0_0.py ckpt_dir"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
torch.set_num_threads(2)
LOG = {1: 7.619e-03, 2: 7.576e-03, 3: 8.133e-03}          # k0_0.log, oa-vcb TEST energy (GPU)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'k0_0_oa-vcb_reeval.jsonl'); fh = open(out, 'w')
for ds, key in ((1, '74711e0cb3'), (2, '962c1c2e4f'), (3, 'e86afa9d01')):
    S = setup(3, 0, ds); te = S['te']
    D, net, st = load_net(S, os.path.join(sys.argv[1], f'ckpt_{key}_oa-vcb.pt'), te)
    cfg = json.loads(st['cfg']); assert cfg['k0'] == 0 and cfg['data_seed'] == ds, cfg
    R = np.array([S['F'] for _ in te])                     # u_c = 0: r = F
    C = apply(net, D, R, S['U_scale']); Cp = apply(net, D, R, S['U_scale'], plain=True)
    e = np.array([energy(S['A'][i], c, S['U'][i]) for i, c in zip(te, C)])
    ep = np.array([energy(S['A'][i], c, S['U'][i]) for i, c in zip(te, Cp)])
    print(f"seed {ds} ({key}, code {st['code']}): energy {e.mean():.4e} (GPU log {LOG[ds]:.4e}, rel diff {e.mean() / LOG[ds] - 1:+.1e}); plain {ep.mean():.4e}; worse than u=0: {(e > 1).sum()}")
    fh.write(json.dumps(dict(cfg=st['cfg'], model='oa-vcb', code=st['code'], reeval='CPU 10 Oct, s1_common', sample_ids=[int(i) for i in te],
                             result=dict(energy=float(e.mean()), en_s=e.tolist(), energy_plain=float(ep.mean())))) + '\n')
print('->', out)
