"""10 Oct: CPU reproduction check of the Session 1 OA-VCB checkpoints against their stored per-sample energy errors, and of
the classical member against its stored per-sample record.  usage: python diagnostics_cost/repro_s1.py data_seed ckpt res classical"""
import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
from oa_vcycle import oa_hierarchy, classical_vcycle_proj, near_kernel_basis
torch.set_num_threads(2)
ds, ck, resf, clf = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
t = time.time(); S = setup(3, 2, ds); te = S['te']
print(f"seed {ds}: setup {time.time() - t:.1f}s, U_scale {S['U_scale']:.6e}")
rec = [json.loads(l) for l in open(resf)]
rec = [r for r in rec if json.loads(r['cfg']).get('data_seed') == ds and r['model'] == 'oa-vcb'][0]['result']
cl = [json.loads(l) for l in open(clf)]
cl = [r for r in cl if json.loads(r['cfg']).get('data_seed') == ds and r['model'] == 'V(4,4) op.-adapted + defl. + proj.'][0]
assert cl['sample_ids'] == list(map(int, te))
D, net, st = load_net(S, ck, te)
print("ckpt code", st.get('code'), "finished", st.get('finished'))
R = np.array([S['F'] - S['A'][i] @ S['UC'][i] for i in te])
C = apply(net, D, R, S['U_scale'])
e = np.array([energy(S['A'][i], S['UC'][i] + c, S['U'][i]) for i, c in zip(te, C)])
es = np.array(rec['en_s'])
print(f"oa-vcb energy: recomputed mean {e.mean():.6e}  stored {es.mean():.6e}  max rel diff per sample {np.max(np.abs(e / es - 1)):.2e}")
em = []
for i in te[:20]:
    A = S['A'][i]; levs, A0 = oa_hierarchy(S['H'], A, S['nl'], 4)
    x = S['UC'][i] + classical_vcycle_proj(levs, A0, S['F'] - A @ S['UC'][i], Z=near_kernel_basis(A), nu=4)
    em.append(energy(A, x, S['U'][i]))
em = np.array(em); cs = np.array(cl['en_s'][:20])
print(f"member (first 20): max rel diff per sample {np.max(np.abs(em / cs - 1)):.2e}")
