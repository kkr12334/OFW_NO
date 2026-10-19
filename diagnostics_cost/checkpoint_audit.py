"""5 Oct (independent review, follow-up): BEFORE / AFTER audit of saved oa-vc* checkpoints.
The same trained weights are evaluated on the same test samples with
  before = legacy/oa_vcycle_2026-10-05_pre_review.py (float32 residual and Gram, as when the results were produced)
  after  = oa_vcycle.py (float64 assembled operator + residual in deflation / Galerkin output, acceptance step, safe gradients)
and reported per sample (paired): energy, nodal rel-l2 and FE L2 errors, prediction difference, energy-safeguard
violations (worse than u_c), projection statistics of the repaired code (line-search backstops, rejected corrections,
numerical rank deficit of the scaled Gram matrix) and CPU inference time.  The BEFORE numbers must reproduce the logged
test errors -- that validates the reconstruction of the evaluation (pool rng 1000, samples 300-399 = test set,
U_scale from the 300 training targets, as train_oa_layer.py does).  Original results are left untouched; this writes a
separate JSON next to the checkpoint.
usage: python diagnostics_cost/checkpoint_audit.py --level L --k0 K --ckpt PATH [--logged-energy E --logged-rl2 R]
       [--n 400] [--test-frac 0.25] [--field channels --contrast 1e4] [--project both] [--deflate 1] [--device cpu]"""
import sys, os, argparse, json, time, importlib.util, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from fe_metrics import fe_mass, rel_norm
import oa_vcycle as NEW
spec = importlib.util.spec_from_file_location('oa_vcycle_old', os.path.join(ROOT, 'legacy', 'oa_vcycle_2026-10-05_pre_review.py'))
OLD = importlib.util.module_from_spec(spec); spec.loader.exec_module(OLD)

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, required=True); ap.add_argument('--k0', type=int, required=True)
ap.add_argument('--ckpt', required=True); ap.add_argument('--n', type=int, default=400); ap.add_argument('--test-frac', type=float, default=0.25)
ap.add_argument('--field', default='channels'); ap.add_argument('--contrast', type=float, default=1e4)
ap.add_argument('--project', default='both'); ap.add_argument('--deflate', type=int, default=1)
ap.add_argument('--logged-energy', type=float, default=None); ap.add_argument('--logged-rl2', type=float, default=None)
ap.add_argument('--batch', type=int, default=8); ap.add_argument('--device', default='cpu'); ap.add_argument('--threads', type=int, default=2)
args = ap.parse_args(); torch.set_num_threads(args.threads); dev = torch.device(args.device)
L, k0 = args.level, args.k0
H = NestedHierarchy(L, coarse=perforated_square(n=9))
rng = np.random.default_rng(1000); t0 = time.time()
a_all = [make_field(H, args.field, args.contrast, rng) for _ in range(args.n)]
U = np.stack([spsolve(H.A_fine(a).tocsc(), H.F) for a in a_all])
UC = np.zeros_like(U)
if k0 > 0:
    Pk = None
    for k in range(k0 + 1, L + 1): Pk = H.P[k] if Pk is None else H.P[k] @ Pk
    Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
    UC = np.stack([Pk @ spsolve((PkT @ H.A_fine(a) @ Pk).tocsc(), PkT @ H.F) for a in a_all])
n_te = int(round(args.test_frac * args.n)); n_tr = args.n - n_te; te = np.arange(n_tr, args.n)
U_scale = float(np.sqrt(np.mean((U - UC)[:n_tr] ** 2)))
print(f"L{L} k0 {k0}: {len(H.F)} unknowns, pool + coarse solves {time.time() - t0:.0f}s; U_scale {U_scale:.6e}", flush=True)
nl = 0; k_ = L
while k_ >= 1 and nl < 4 and len(H.nodes[k_ - 1]) >= 16: nl += 1; k_ -= 1
t0 = time.time()
D = NEW.OAData(H, [a_all[i] for i in te], nl, 4, dev, H.level_coords, near_kernel=bool(args.deflate))
print(f"operator data for {n_te} test samples, {nl} levels: {time.time() - t0:.0f}s", flush=True)
R = np.array([H.F - H.A_fine(a_all[i]).tocsr() @ UC[i] for i in te]) / U_scale
SD = torch.load(args.ckpt, map_location='cpu', weights_only=False)['model']; W = SD['pre.0.enc_lin.weight'].shape[0]
pj = {'both': 'both', 'true': True, 'eval': 'eval', 'false': False}[args.project.lower()]
mass = fe_mass(H, L); out = {}
AF = [H.A_fine(a_all[i]).tocsr() for i in te]
for tag, mod in (('before', OLD), ('after', NEW)):
    net = mod.OAVCycle(D, width=W, nu=4, project=pj, deflate=bool(args.deflate)).to(dev); net.load_state_dict(SD); net.eval()
    net.collect_stats = True; P = []; t1 = time.time()
    with torch.no_grad():
        for idx in np.array_split(np.arange(n_te), int(np.ceil(n_te / args.batch))):
            if tag == 'before':
                net.set_ops(*D.batch(idx), Z=D.batch_z(idx))
                o = net(torch.tensor(R[idx], dtype=torch.float32, device=dev))
            else:
                net.set_ops(*D.batch(idx), Z=D.batch_z(idx), A64=D.batch_a64(idx))
                o = net(torch.tensor(R[idx], dtype=torch.float64, device=dev))
            P.append(o.double().cpu().numpy())
    ms = (time.time() - t1) / n_te * 1e3
    P = np.concatenate(P) * U_scale + UC[te]
    E = lambda j, x: float(np.sqrt((U[te[j]] - x) @ (AF[j] @ (U[te[j]] - x)) / (U[te[j]] @ (AF[j] @ U[te[j]]))))
    en = np.array([E(j, p) for j, p in enumerate(P)]); en_c = np.array([E(j, UC[i]) for j, i in enumerate(te)])
    rl2 = np.linalg.norm(P - U[te], axis=1) / np.linalg.norm(U[te], axis=1); l2 = rel_norm(mass, P - U[te], U[te])
    out[tag] = dict(P=P, en=en, rl2=rl2, l2=l2, ms=ms, worse=int(np.sum(en > en_c * (1 + 1e-12))),
                    worse_ratio=float(np.max(en / en_c)), stats=dict(getattr(net, 'stats', {})))
    print(f"{tag:<7} energy {en.mean():.4e}  nodal rel-l2 {rl2.mean():.4e}  FE L2 {l2.mean():.4e}  "
          f"worse than u_c {out[tag]['worse']}/{n_te} (max ratio to u_c {out[tag]['worse_ratio']:.6f})  CPU {ms:.0f} ms/sample", flush=True)
b, a = out['before'], out['after']
if args.logged_energy:
    print(f"logged (Kaggle GPU, pre-review code): energy {args.logged_energy:.4e}  rel-l2 {args.logged_rl2:.4e}  -> "
          f"BEFORE reproduces it to {abs(b['en'].mean() / args.logged_energy - 1) * 100:.2f}% / {abs(b['rl2'].mean() / args.logged_rl2 - 1) * 100:.2f}%")
dP = np.linalg.norm(a['P'] - b['P'], axis=1) / np.linalg.norm(b['P'] - UC[te] + 1e-300, axis=1)
den = (a['en'] - b['en']) / b['en']
print(f"PAIRED after vs before: mean energy change {(a['en'].mean() / b['en'].mean() - 1) * 100:+.4f}%; per sample relative "
      f"energy change median {np.median(den) * 100:+.4f}%, range [{den.min() * 100:+.4f}%, {den.max() * 100:+.4f}%]; "
      f"after lower on {np.mean(a['en'] < b['en']) * 100:.0f}% of samples")
print(f"       relative change of the correction ||c_after - c_before|| / ||c_before||: median {np.median(dP):.2e}, max {dP.max():.2e}")
s = a['stats']
print(f"REPAIRED projection statistics: {s.get('samples', 0)} projections (deflation + output), line-search backstop "
      f"{s.get('backstop', 0)}, rejected {s.get('rejected', 0)}, rank deficit {s.get('rank_deficit', 0)} of {s.get('cols', 0)} live columns")
jp = os.path.splitext(args.ckpt)[0] + '_audit_2026-10-05.json'
json.dump(dict(args=vars(args), U_scale=U_scale, **{t: dict(energy=v['en'].tolist(), rl2=v['rl2'].tolist(), fe_l2=v['l2'].tolist(),
          ms=v['ms'], worse=v['worse'], worse_ratio=v['worse_ratio'], stats=v['stats']) for t, v in out.items()}), open(jp, 'w'))
print(f"per-sample results: {jp}")
