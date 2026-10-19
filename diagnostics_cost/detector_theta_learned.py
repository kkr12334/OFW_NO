"""11 Oct 2026 (independent reviews): does the detector threshold theta change the BARRIER comparison?  The trained OA-VCB
checkpoints (theta = 0.1 during training) and the classical reference are re-evaluated on the test samples of held-out data
seeds 1-3 with Z from theta in {0.1, 0.05} (tau = 0.05).  Z enters only the deflation and the Galerkin output; the learned
smoothers are unchanged.  theta = 0.1 must reproduce the recorded errors (sanity check).
usage: python diagnostics_cost/detector_theta_learned.py"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup as _s  # noqa: puts the code folder on sys.path
import oa_vcycle
from s1_common import setup, load_net, apply
from oa_vcycle import oa_hierarchy, classical_vcycle_proj
BASE = '/mnt/user-data/uploads/OA_NO/Working codes/OFW_NO/'
CK = {1: BASE + 'Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt', 2: BASE + 'Results_kaggle_session 7/ckpt_8063a3449e_oa-vcb.pt',
      3: BASE + 'Results_kaggle_session 7/ckpt_62400ed25c_oa-vcb.pt'}
ORIG = oa_vcycle.near_kernel_basis
en = lambda A, x, u: float(np.sqrt(((x - u) @ (A @ (x - u))) / (u @ (A @ u))))
out = {}
for ds, ck in CK.items():
    S = setup(3, 2, ds, contrast=1e-4, load='no-low'); te = S['te']
    R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te])
    for th in (0.1, 0.05):
        oa_vcycle.near_kernel_basis = lambda A, theta=th, tau=0.05: ORIG(A, theta, tau)      # used inside OAData
        D, net, st = load_net(S, ck, te); C = apply(net, D, R, S['U_scale'])
        ev = np.array([en(S['A'][i], S['UC'][i] + C[j], S['U'][i]) for j, i in enumerate(te)])
        er = []
        for j, i in enumerate(te):
            A = S['A'][i]; levs, A0 = oa_hierarchy(S['H'], A, S['nl'], 4)
            er.append(en(A, S['UC'][i] + classical_vcycle_proj(levs, A0, R[j], Z=ORIG(A, th, 0.05), nu=4), S['U'][i]))
        er = np.array(er); G = float(np.exp(np.mean(np.log(er / ev)))); W = float(np.mean(ev < er))
        out[f"ds{ds}_theta{th}"] = dict(oa_vcb=float(ev.mean()), ref=float(er.mean()), G=G, W=W, en_vcb=ev.tolist(), en_ref=er.tolist())
        print(f"barriers ds{ds} theta {th}: OA-VCB {ev.mean():.4e}  reference {er.mean():.4e}  G {G:.3f}  W {W:.2f}", flush=True)
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'detector_theta_learned.json'), 'w'))
