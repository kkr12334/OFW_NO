"""D1 local K4 diagnostic (protocol v1 §3.4, amendment 1 A2.4).  For OA-VCB's FINE-level pre-smoother (input = the
deflated residual) and post-smoother (input = OA-VCB's own post-smoothing right-hand side), captured inside the network:
  delta = A-distance of the network's output from K4 (rank-aware), rho = E_NN / E_K, PCG cross-check.
The network computes in float32 with float32 level values; the diagnostic evaluates in float64 with the assembled
operator.  A local separation is DESCRIBED only if rank = 4, E_K > 0 and rho <= 0.99.
usage: python -I ext/stage1_d1.py SET FAMILY CKPT N OUT.json   (SET = D, or C after the pre-confirmation note)"""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup
from oa_vcycle import OAData
from stage1_core import k4_diagnostic, pcg4
import stage1_models as SM
torch.set_num_threads(2)
SET, FAM, CKP, N, OUT = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
S = setup(3, 2, 1, **kw) if SET == 'D' else setup(3, 2, 0, rng_seed={'channels': 2026101601, 'barriers': 2026101602}[FAM], **kw)
USC = json.load(open(os.path.join(HERE, 'frozen_scales.json')))['scales'][f'{FAM}/fixed-load']   # A7.2: frozen, from set T
te = S['te'][:N]; H = S['H']
D = OAData(H, [S['a'][i] for i in te], S['nl'], 4, 'cpu', H.level_coords, near_kernel=True)
net, _, _ = SM.load(os.path.join(ROOT, CKP), D)
calls = []
orig = net._smooth
def spy(Sm, b, l):
    x = orig(Sm, b, l)
    if l == 0: calls.append(('pre' if Sm is net.pre[0] else 'post', b.detach().double().numpy().copy(), x.detach().double().numpy().copy()))
    return x
net._smooth = spy
out = []
for j, i in enumerate(te):
    calls.clear(); A = S['A'][i]; r = S['Fs'][i] - A @ S['UC'][i]
    jt = torch.tensor([j])
    with torch.no_grad():
        net.set_ops(*D.batch(jt), Z=D.batch_z(jt), A64=D.batch_a64(jt)); net(torch.tensor(r[None] / USC, dtype=torch.float64))
    rec = dict(id=int(i))
    for kind, b, x in calls:
        b = b[0]; x = x[0]; di = 1 / A.diagonal()
        rec[kind] = k4_diagnostic(A, b, x, x_pcg=pcg4(A, di, b, 0))
    out.append(rec)
def col(k, f): return np.array([r[k][f] for r in out if r[k].get(f) is not None])
summ = {}
for k in ('pre', 'post'):
    rho, dl, rk = col(k, 'rho'), col(k, 'delta'), col(k, 'rank')
    sep = [r['id'] for r in out if r[k]['rank'] == 4 and r[k]['E_K'] > 0 and r[k]['rho'] is not None and r[k]['rho'] <= 0.99]
    summ[k] = dict(rho_median=float(np.median(rho)), rho_q=[float(q) for q in np.quantile(rho, [.1, .5, .9])], delta_median=float(np.median(dl)),
                   delta_q=[float(q) for q in np.quantile(dl, [.1, .5, .9])], ranks=sorted(set(map(int, rk))), n_local_separation=len(sep),
                   pcg_vs_K_max=float(np.max(np.abs(col(k, 'pcg_vs_K')))))
    print(FAM, k, summ[k])
from stage1_manifest import manifest
json.dump(dict(set=SET, family=FAM, ckpt=CKP, input_scale=USC, summary=summ, records=out, manifest=manifest([os.path.join(ROOT, CKP)])), open(OUT, 'w'), indent=1)
