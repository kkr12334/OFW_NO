"""Work accounting per one-pass method (protocol v1 §6; reported, not a gate).  Structure only: the level sparsity
patterns are identical for every operator on the L3 mesh, so one set-C operator (channels id 300) is used.
  * classical: mean per-level counters from ext/runs/confirm_C_{family}.json (A_k products incl. Gauss-Seidel sweeps, D^-1
    applications, dot products) weighted by each level's nnz / n -> multiply-adds (MACs) per pass, plus the common cycle
    work (residuals, transfers, coarsest solve) which every method shares and which is listed separately;
  * OA-VCB: analytic MACs of one OpSmoother call (width C, nu steps) on each level from its node count n and off-diagonal
    entry count m, cross-checked against torch's FlopCounterMode for the dense layers; activations (GELU), gathers and the
    O(n) normalisation are not counted (they would only increase the learned count).
usage: python -I ext/stage1_work.py -> ext/runs/stage1_work.json"""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup
from oa_vcycle import OAData
from stage1_core import Hier
import stage1_models as SM
from torch.utils.flop_counter import FlopCounterMode
torch.set_num_threads(2)
S = setup(3, 2, 0, rng_seed=2026101601); i = int(S['te'][0]); A = S['A'][i]; H = S['H']; nl = S['nl']
hr = Hier(H, A, nl)
lev = [dict(n=int(Ak.shape[0]), nnz=int(Ak.nnz), m=int(Ak.nnz - Ak.shape[0])) for Ak, _ in hr.levs]
X = [dict(nc=int(Xl.shape[1]), nnzX=int(Xl.nnz)) for _, Xl in hr.levs]
def mac_smoother(n, m, C=15, nu=4):
    enc = n * (2 * 2 * C + 2 * C * C + 2 * C)              # enc MLP (2 -> 2C -> C) + enc_lin (2 -> C)
    step = n * (C * C + 2 * C * C + 4 * C * C + C) + m * (C + (2 * C + 4) * C + C * C + C)   # V, Lw, upd | opa, msg MLP, aggregation
    dec = n * (C + C * C + C)                               # dec_lin + dec MLP (C -> C -> 1)
    return dict(dense_nodes=n * (2 * 2 * C + 2 * C * C + 2 * C) + nu * n * 7 * C * C + n * (C + C * C + C),
                dense_edges=nu * m * (2 * C + 4) * C + nu * m * C * C, sparse=nu * m * 2 * C, total=enc + nu * step + dec)
mac = [mac_smoother(l['n'], l['m']) for l in lev]
# cross-check of the dense-layer MACs with torch's FlopCounterMode (flops = 2 x MACs for mm/addmm/bmm)
D = OAData(H, [S['a'][i]], nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[A])
ck = os.path.join(ROOT, 'resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt'); net, _, _ = SM.load(ck, D)
j = torch.tensor([0]); net.set_ops(*D.batch(j), Z=D.batch_z(j), A64=D.batch_a64(j))
fc = []
for l in range(nl):
    b = torch.randn(1, lev[l]['n'], dtype=next(net.parameters()).dtype)
    with FlopCounterMode(display=False) as F, torch.no_grad(): net.pre[0](b, net.lv[l], net._A[l])
    fc.append(F.get_total_flops() / 2)
calls_per_level = 2                                         # one pre + one post call per level in a V-cycle pass
oa_total = calls_per_level * sum(x['total'] for x in mac)
mv = sum(l['nnz'] for l in lev)                             # one product with every level matrix
# classical: per-level counters from the set-C run
out = dict(levels=lev, transfers=X, oa_vcb=dict(per_call_MAC=mac, flopcounter_dense_MAC=fc, per_pass_MAC=oa_total,
           per_pass_in_fine_matvecs=oa_total / lev[0]['nnz']), classical={}, fine_matvec_MAC=lev[0]['nnz'])
for fam in ('channels', 'barriers'):
    d = json.load(open(os.path.join(HERE, 'runs', f'confirm_C_{fam}.json'))); W = d['work']; out['classical'][fam] = {}
    for meth, ws in W.items():
        mvl = np.mean([w['matvec'] for w in ws], 0); dil = np.mean([w['dinv'] for w in ws], 0); dot = np.mean([w['dot'] for w in ws], 0)
        tot = float(sum(mvl[l] * lev[l]['nnz'] + dil[l] * lev[l]['n'] + dot[l] * lev[l]['n'] for l in range(nl)))
        out['classical'][fam][meth] = dict(matvec_per_level=mvl.tolist(), MAC_per_pass=tot, in_fine_matvecs=tot / lev[0]['nnz'],
                                           oa_vcb_over_this=oa_total / tot)
json.dump(out, open(os.path.join(HERE, 'runs', 'stage1_work.json'), 'w'), indent=1)
print('levels', lev); print('analytic dense MACs per call', [x['dense_nodes'] + x['dense_edges'] for x in mac], 'flopcounter', fc)
print(f"OA-VCB per pass: {oa_total / 1e6:.1f} M MACs = {oa_total / lev[0]['nnz']:.0f} fine matvecs")
for fam, v in out['classical'].items():
    for meth, w in v.items(): print(f"  {fam:<9}{meth:<11} {w['MAC_per_pass'] / 1e6:7.3f} M MACs = {w['in_fine_matvecs']:6.1f} fine matvecs; OA-VCB/this = {w['oa_vcb_over_this']:.0f}")
