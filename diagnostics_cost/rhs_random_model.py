"""usage: python diagnostics_cost/rhs_random_model.py [data_seed ckpt_path records_glob]   (default: seed 1, Session 2 checkpoint)
11 Oct (RANDOM_LOADS_PLAN_2026-10-11.md, Test B): the random-load OA-VCB checkpoint (Session 2, run key 7cae2898f7,
trained on independent samples of the random-load distribution, data seed 1) applied to the loads of
rhs_generalisation_s1.py on the same 100 held-out operators (ids 300-399 of rng 1001).  Same metrics; side by side with
the fixed-load checkpoint (40ce3ac1f9) from rhs_generalisation_s1_seed1.json.  The network's U_scale is the rms of
U - U_c over its own training samples (random loads)."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
from scipy.sparse.linalg import spsolve, factorized
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
import glob
DS = int(sys.argv[1]) if len(sys.argv) > 1 else 1
CKP = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, 'ckb', 'ckpt_7cae2898f7_oa-vcb.pt')
RECS = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, 's2', 'random_results_train_oa_L3_channels.jsonl')
S = setup(3, 2, DS); H, L, P = S['H'], S['L'], S['P']; PT = P.T.tocsr(); te = S['te']
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
cx, cy = xy.mean(1).T
def assemble(fe):
    Fv = np.zeros(len(H.coords)); m = H.keep
    for j in range(3): np.add.at(Fv, H.elem[m, j], (fe * area)[m] / 3.0)
    return Fv[H.nodes[L]]
def random_load(i):                                   # = train_oa_layer.load_vec(--load random), sample i
    g = np.random.default_rng(777_000 + 100_000 * DS + i); K = g.normal(size=(4, 4))
    fe = g.uniform(-1, 1) + sum(K[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) / (1 + p + q) for p in range(4) for q in range(4))
    if g.uniform() < 0.5:
        x0 = g.uniform(0.1, 0.9, 2); w = g.uniform(0.02, 0.08)
        fe = fe + g.uniform(2, 10) * np.exp(-((cx - x0[0]) ** 2 + (cy - x0[1]) ** 2) / (2 * w ** 2))
    return assemble(fe)
def loads(i):
    h = np.random.default_rng(5_000 + 100_000 * DS + i); K2 = h.normal(size=(4, 4))
    sm = sum(K2[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) for p in range(4) for q in range(4))
    c0 = h.uniform(0.15, 0.85, 2); pt = np.exp(-((cx - c0[0]) ** 2 + (cy - c0[1]) ** 2) / (2 * 0.03 ** 2))
    return {'f = 1 (training)': H.F, 'f = -1': -H.F, 'f = 3': 3 * H.F, 'session-2 random': random_load(i), 'smooth +-': assemble(sm), 'point source': assemble(pt)}
# the random model's own U_scale: training samples 0-299 with their random loads
Ut = []
for i in S['tr']:
    A = S['A'][i]; F = random_load(int(i)); u = spsolve(A.tocsc(), F); uc = P @ spsolve((PT @ A @ P).tocsc(), PT @ F); Ut.append(u - uc)
U_scale = float(np.sqrt(np.mean(np.array(Ut) ** 2)))
D, net, st = load_net(S, CKP, te)
cfg = json.loads(st['cfg']); assert cfg.get('load') == 'random' and cfg.get('data_seed') == DS, cfg
names = list(loads(int(te[0]))); E = {n: {'net': [], 'mem': [], 'uc': []} for n in names}; R = {n: [] for n in names}; UU = {n: [] for n in names}; UC = {n: [] for n in names}
for j, i in enumerate(te):
    A = S['A'][i]; levs, A0 = oa_hierarchy(H, A, S['nl'], 4); Z = near_kernel_basis(A); solc = factorized((PT @ A @ P).tocsc())
    for n, F in loads(int(i)).items():
        u = spsolve(A.tocsc(), F); uc = P @ solc(PT @ F); r = F - A @ uc
        R[n].append(r); UU[n].append(u); UC[n].append(uc)
        E[n]['mem'].append(energy(A, uc + classical_vcycle_proj(levs, A0, r, Z=Z, nu=4), u)); E[n]['uc'].append(energy(A, uc, u))
for n in names:
    C = apply(net, D, np.array(R[n]), U_scale)
    E[n]['net'] = [energy(S['A'][i], uc + c, u) for i, uc, c, u in zip(te, UC[n], C, UU[n])]
fixed = json.load(open(os.path.join(HERE, f'rhs_generalisation_s1_seed{DS}.json')))['loads']
stored = [json.loads(l) for f in glob.glob(RECS) for l in open(f) if json.loads(l)['model'] == 'oa-vcb' and json.loads(json.loads(l)['cfg']).get('data_seed') == DS][0]['result']['en_s']
print(f"random-load checkpoint {st.get('code')} (cfg load {cfg['load']}, data seed {cfg['data_seed']}), U_scale {U_scale:.4e}; in-distribution anchor reproduces the stored record to "
      f"{np.max(np.abs(np.array(E['session-2 random']['net']) / np.array(stored) - 1)):.1e}")
print(f"{'load':<20}{'u_c':>8}{'random-load OA-VCB':>20}{'fixed-load OA-VCB':>19}{'member':>10}   member/random-model G [95% CI]  W   >u_c   random/fixed model: G, W")
out = {}
for n in names:
    e = np.array(E[n]['net']); m = np.array(E[n]['mem']); f = np.array(fixed[n]['net']); c = np.array(E[n]['uc'])
    lr = np.log(m / e); b = np.random.default_rng(0).integers(0, 100, (5000, 100)); ci = np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5]))
    lf = np.log(f / e)
    print(f"{n:<20}{c.mean():>8.3f}{e.mean():>20.3e}{f.mean():>19.3e}{m.mean():>10.3e}   {np.exp(lr.mean()):6.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  {np.mean(lr > 0):.2f}  {int(np.sum(e > c * (1 + 1e-6))):>3}   {np.exp(lf.mean()):.2f}, {np.mean(lf > 0):.2f}")
    out[n] = dict(net=e.tolist(), mem=m.tolist(), uc=c.tolist(), fixed_net=f.tolist())
json.dump(dict(data_seed=DS, ckpt=os.path.basename(CKP), code=st.get('code'), sample_ids=[int(i) for i in te], loads=out), open(os.path.join(HERE, f'rhs_random_model_seed{DS}.json'), 'w'))
