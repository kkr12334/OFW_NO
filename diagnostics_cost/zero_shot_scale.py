"""11 Oct 2026 ("toy problem" concern): apply a TRAINED OA-VCB checkpoint, unchanged, to finer meshes of the same problem.
The coefficient draw is mesh-independent (field_channels draws a fixed number of random numbers per field and evaluates the
bands at element centroids), so sample i of data seed s is the same physical coefficient at every level; only the mesh is
refined.  The smoothers are shared by all levels and see only scale-free features (z / sigma, log(d / median d), edge vectors
in units of h_k, W_ij), so the network applies to any hierarchy without retraining.  The classical reference is evaluated on
the same hierarchy.  Only the test samples are solved (rng sequence of all 400 kept).
usage: python diagnostics_cost/zero_shot_scale.py CKPT L k0 data_seed contrast load n_test [tag]   (env ZS_NL: cycle levels)"""
import sys, os, time, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import coarse_P, vc_levels, load_no_low
from scipy.sparse.linalg import spsolve, splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, near_kernel_basis, classical_vcycle_proj
torch.set_num_threads(int(os.environ.get('ZS_THREADS', 4)))
CK, L, k0, DS, con, load, NT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), sys.argv[6], int(sys.argv[7])
TAG = sys.argv[8] if len(sys.argv) > 8 else f"L{L}_k{k0}_ds{DS}"
t0 = time.time()
H = NestedHierarchy(L, coarse=perforated_square(n=9)); nl = vc_levels(H, L)
if os.environ.get('ZS_NL'): nl = int(os.environ['ZS_NL'])          # override the number of cycle levels (depth test)
rng = np.random.default_rng(1000 + DS); a_all = [make_field(H, 'channels', con, rng) for _ in range(400)]
ids = list(range(300, 300 + NT))
P = coarse_P(H, L, k0)
A, F, U, UC = {}, {}, {}, {}
for i in ids:
    A[i] = H.A_fine(a_all[i]).tocsr(); F[i] = H.F if load == 'all' else load_no_low(H, a_all[i])
    U[i] = spsolve(A[i].tocsc(), F[i]); UC[i] = P @ spsolve((P.T @ A[i] @ P).tocsc(), P.T @ F[i])
print(f"L{L}: {A[ids[0]].shape[0]} unknowns, {nl} cycle levels, k0 {k0}; data ready ({time.time() - t0:.0f}s)", flush=True)
en = lambda Ai, x, u: float(np.sqrt(((x - u) @ (Ai @ (x - u))) / (u @ (Ai @ u))))
# learned pass (U_scale cancels by positive homogeneity; use the rms of the corrections of the evaluated samples)
st = torch.load(CK, map_location='cpu', weights_only=False); SD = st['model']
ev, er, eu, tl = [], [], [], []
BS = int(os.environ.get('ZS_BS', 4)); import gc
for j0 in range(0, NT, BS):
    blk = ids[j0:j0 + BS]
    tt = time.perf_counter()
    D = OAData(H, [a_all[i] for i in blk], nl, 4, 'cpu', H.level_coords, near_kernel=True)
    net = OAVCycle(D, width=SD['pre.0.enc_lin.weight'].shape[0], nu=4, project='both', deflate=True)
    net.load_state_dict(SD); net.eval()
    R = np.array([F[i] - A[i] @ UC[i] for i in blk]); s = float(np.sqrt(np.mean(R ** 2))) or 1.0
    jt = torch.arange(len(blk))
    with torch.no_grad():
        net.set_ops(*D.batch(jt), Z=D.batch_z(jt), A64=D.batch_a64(jt))
        C = net(torch.tensor(R / s, dtype=torch.float64)).double().numpy() * s
    tl.append((time.perf_counter() - tt) / len(blk))
    for k, i in enumerate(blk):
        levs, A0 = oa_hierarchy(H, A[i], nl, 4)
        xr = UC[i] + classical_vcycle_proj(levs, A0, F[i] - A[i] @ UC[i], Z=near_kernel_basis(A[i]), nu=4)
        ev.append(en(A[i], UC[i] + C[k], U[i])); er.append(en(A[i], xr, U[i])); eu.append(en(A[i], UC[i], U[i]))
    print(f"  samples {blk[0]}-{blk[-1]}: OA-VCB {np.mean(ev[-len(blk):]):.3e}  reference {np.mean(er[-len(blk):]):.3e}  "
          f"({tl[-1]:.1f} s/sample, operator data + inference)", flush=True)
    del D, net, C; gc.collect()
ev, er, eu = map(np.array, (ev, er, eu)); lr = np.log(er / ev)
b = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr))); ci = np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5]))
res = dict(ckpt=os.path.basename(CK), trained_cfg=st.get('cfg'), L=L, k0=k0, data_seed=DS, contrast=con, load=load, ids=ids,
           unknowns=int(A[ids[0]].shape[0]), cycle_levels=nl, oa_vcb=float(ev.mean()), ref=float(er.mean()), uc=float(eu.mean()),
           G=float(np.exp(lr.mean())), G_ci=ci.tolist(), W=float(np.mean(lr > 0)), worse_than_uc=int(np.sum(ev > eu * (1 + 1e-9))),
           s_per_sample=float(np.median(tl)), en_vcb=ev.tolist(), en_ref=er.tolist(), en_uc=eu.tolist())
print(f"RESULT {TAG}: {res['unknowns']} unknowns; OA-VCB {ev.mean():.4e}  reference {er.mean():.4e}  u_c {eu.mean():.4e}  "
      f"G {res['G']:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  W {res['W']:.2f}  worse than u_c {res['worse_than_uc']}", flush=True)
out = os.environ.get('ZS_OUT') or os.path.join(os.path.dirname(os.path.abspath(__file__)), 'zero_shot_scale.json')   # Kaggle: input is read-only
J = json.load(open(out)) if os.path.exists(out) else {}
J[TAG] = res; json.dump(J, open(out, 'w'))
