"""EXPLORATORY (post-hoc, after set C was opened; development set D only; not a Stage 1 confirmation result).
X1  Work sweep of residual-adaptive smoothers in the common one-pass cycle (same wrapper as stage1_confirm.py):
    Jacobi-PCG(k), k in {4, 5, 8, 14, 32}, and SGS-preconditioned PCG(k), k in {4, 8}; errors and matvec-equivalents.
    Question: is SGS-PCG's gain just more work (it uses 14 matvec-equivalents per call against PCG(4)'s 4)?
X3  Reach-matched local test: OA-VCB's fine-level smoother has a 4-hop reach (4 message-passing steps), one more than K4
    (outputs p(B) D^-1 b, deg p <= 3, reach 3).  For OA-VCB init 0's captured fine-level pre/post inputs, report
    rho_d = E_NN / E_Kd for d = 4, 5, 8 (rank-aware basis, same as D1).
usage: python -I ext/explore/x1_x3_ksweep_reach.py FAMILY OUT.json"""
import os, sys, json, functools, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
sys.path[:0] = [EXT, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy
from oa_vcycle import OAData, near_kernel_basis
from scipy.sparse.linalg import spsolve
from stage1_core import Hier, one_pass, pcg4, sgs_pcg4, krylov_basis
import stage1_models as SM, stage1_stats as ST
torch.set_num_threads(2)
FAM, OUT = sys.argv[1], sys.argv[2]
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
S = setup(3, 2, 1, **kw); te = S['te']; H = S['H']; nl = S['nl']
dev = json.load(open(os.path.join(EXT, 'dev', 'stage1_dev_D.json')))['families'][FAM]
assert dev['ids'] == [int(i) for i in te]
M = {f'PCG({k})': functools.partial(pcg4, its=k) for k in (4, 5, 8, 14, 32)}
M.update({f'SGS-PCG({k})': functools.partial(sgs_pcg4, its=k) for k in (4, 8)})
E = {m: [] for m in M}; MV = {m: [] for m in M}
for i in te:
    A = S['A'][i]; r = S['Fs'][i] - A @ S['UC'][i]; hr = Hier(H, A, nl); Z = near_kernel_basis(A)
    for m, f in M.items():
        cnt = hr.new_counter(); x = one_pass(hr, r, Z, f, f, cnt=cnt)
        E[m].append(float(energy(A, S['UC'][i] + x, S['U'][i]))); MV[m].append(sum(cnt['matvec']))
chk = float(np.max(np.abs(np.array(E['PCG(4)']) / np.array(dev['errors']['PCG']) - 1)))
chk2 = float(np.max(np.abs(np.array(E['SGS-PCG(4)']) / np.array(dev['errors']['SGS-PCG (spatial, conditional)']) - 1)))
res = dict(note='EXPLORATORY, development set D', family=FAM, check_reproduces_dev=dict(PCG4=chk, SGS_PCG4=chk2), x1={})
for m in M:
    res['x1'][m] = dict(geo=float(np.exp(np.mean(np.log(E[m])))), matvec_eq_per_pass=float(np.mean(MV[m])),
                        vs_CHEB_legacy=ST.paired(dev['errors']['CHEB-legacy'], E[m]),
                        to_OA=[ST.paired(E[m], dev['errors'][f'OA-VCB init{k}']) for k in (0, 1, 2)], errors=E[m])
    print(f"{FAM} {m:<12} matvec-eq/pass {np.mean(MV[m]):6.0f}  G(CHEB->X) {res['x1'][m]['vs_CHEB_legacy']['G']:.2f}  "
          f"G(X->OA-VCB init0) {res['x1'][m]['to_OA'][0]['G']:.3f} [{res['x1'][m]['to_OA'][0]['lo']:.3f}, {res['x1'][m]['to_OA'][0]['hi']:.3f}] W {res['x1'][m]['to_OA'][0]['W']:.2f}", flush=True)
print('reproduction check vs dev JSON (PCG(4), SGS-PCG(4)):', chk, chk2)
# ---------------- X3: reach-matched local test (OA-VCB init 0, fine level)
CK = json.load(open(os.path.join(EXT, 'checkpoint_map.json')))
p = [r['path'] for r in CK if r['family'] == FAM and r['model'] == 'oa-vcb' and r['init_seed'] == 0 and r['data_seed'] == 1 and r['level'] == 3
     and r['k0'] == 2 and r['oa_steps'] == 4 and r['coarse'] == 'perforated' and r['load'] in ('f=1', 'no-low')][0]
USC = json.load(open(os.path.join(EXT, 'frozen_scales.json')))['scales'][f'{FAM}/fixed-load']
D = OAData(H, [S['a'][i] for i in te], nl, 4, 'cpu', H.level_coords, near_kernel=True)
net, _, _ = SM.load(os.path.join(ROOT, p), D); calls = []; orig = net._smooth
def spy(Sm, b, l):
    x = orig(Sm, b, l)
    if l == 0: calls.append(('pre' if Sm is net.pre[0] else 'post', b.detach().double().numpy()[0].copy(), x.detach().double().numpy()[0].copy()))
    return x
net._smooth = spy; x3 = {'pre': {4: [], 5: [], 8: []}, 'post': {4: [], 5: [], 8: []}}; ranks = set()
for j, i in enumerate(te):
    calls.clear(); A = S['A'][i]; r = S['Fs'][i] - A @ S['UC'][i]; jt = torch.tensor([j])
    with torch.no_grad():
        net.set_ops(*D.batch(jt), Z=D.batch_z(jt), A64=D.batch_a64(jt)); net(torch.tensor(r[None] / USC, dtype=torch.float64))
    for kind, b, x in calls:
        u = spsolve(A.tocsc(), b); En = np.sqrt((u - x) @ (A @ (u - x)))
        for d in (4, 5, 8):
            Q, rk = krylov_basis(A, b, d); ranks.add((d, rk)); xK = Q @ (Q.T @ b); EK = np.sqrt((u - xK) @ (A @ (u - xK)))
            x3[kind][d].append(float(En / EK))
res['x3'] = {k: {str(d): dict(q10_50_90=[float(q) for q in np.quantile(v, [.1, .5, .9])], n_below_099=int(np.sum(np.array(v) <= 0.99)), n=len(v), rho=v)
                 for d, v in dd.items()} for k, dd in x3.items()}
res['x3_ranks'] = sorted(ranks)
for k, dd in res['x3'].items():
    for d, v in dd.items(): print(f"{FAM} X3 {k} K{d}: rho quantiles {['%.3f' % q for q in v['q10_50_90']]}, rho <= 0.99 on {v['n_below_099']}/{v['n']}")
json.dump(res, open(OUT, 'w'), indent=1)
