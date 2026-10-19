"""Scalar elliptic problems with a tensor coefficient (anisotropic heat conduction in a tow-reinforced
composite), same protocol as train_oa_layer.py / train_elasticity.py: capacity-matched models, same data,
epochs and objective, the exact Galerkin coarse solve on the level-k0 mesh supplied to every model, errors on
held-out samples in relative L2 (temperature) and in the energy norm (heat flux).

    python -u train_scalar.py --pde aniso --bc insulated --level 3 --contrast 1e2 --k0 2 --loss mix \
        --models oa-ll oa-ll-s gnn transolver lino wno fno --out-dir <dir>

--pde darcy reproduces the isotropic Darcy set-up (for cross-checking against train_oa_layer.py).

Models: oa-ll (learnable lifting, per-level networks), oa-ll-s (same, ONE predict/update shared by all levels:
runs on a finer hierarchy without retraining), gnn (MeshGraphNet-style), transolver (mesh-native), lino, wno
(our light WNO layer), wno-off (the published WNO, needs >= ~75k parameters), fno (grid models through the bridge).

--transfer-level Lt: after training at --level, evaluate every model zero-shot on the level-Lt hierarchy (same
coefficient functions, same test indices, same coarse level k0).  Mesh models run on the finer mesh directly,
grid models on a (2 ng + 1)^2 grid.  Trained weights are saved (model_*.pt) so a transfer can be run later.
"""
import argparse, hashlib, json, os, pickle, time
import numpy as np, torch, torch.nn as nn
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from scalar_problems import TensorProblem, iso, field_tows, tensor_features
from mesh_grid_bridge import MeshGridBridge
from run_identity import CODE_HASH, reusable
from fe_metrics import fe_mass, rel_norm, atomic_pickle

ap = argparse.ArgumentParser()
ap.add_argument('--pde', default='aniso', choices=['aniso', 'darcy'])
ap.add_argument('--bc', default='insulated', choices=['insulated', 'dirichlet'], help='pore boundaries')
ap.add_argument('--level', type=int, default=3)
ap.add_argument('--coarse-n', type=int, default=9)
ap.add_argument('--contrast', type=float, default=1e2, help='aniso: axial tow conductivity; darcy: channel contrast')
ap.add_argument('--k-t', type=float, default=5.0, help='aniso: transverse tow conductivity (matrix = 1)')
ap.add_argument('--strict-code', type=int, default=0,
                help='1: reuse finished results / checkpoints only if produced by the same code version (run_identity.py); '
                     'otherwise retrain.  The paper runs use 1.')
ap.add_argument('--n', type=int, default=400)
ap.add_argument('--test-frac', type=float, default=0.25)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--loss', default='mix', choices=['rel_l2', 'energy', 'mix'])
ap.add_argument('--k0', type=int, default=2)
ap.add_argument('--params', type=int, default=22000)
ap.add_argument('--models', nargs='+', default=['oa-ll', 'transolver', 'lino', 'wno', 'fno'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data-seed', type=int, default=0)
ap.add_argument('--grid-ext', default='nearest', choices=['nearest', 'zero'],
                help="grid baselines: extension of node fields into the pores. 'zero' was used before 28 Sep 2026; "
                     "with insulated / traction-free pores it puts a jump at every pore edge (round-trip floor of the "
                     "exact solution ~15%% rel-L2 on the anisotropic L3 case, against 0.75%% with 'nearest')")
ap.add_argument('--transfer-level', type=int, default=0,
                help='evaluate the trained models zero-shot on this (finer) level; 0 = off')
ap.add_argument('--vc-ckpt', type=int, default=0, help='oa-vc*: activation checkpointing of the smoothers (memory)')
ap.add_argument('--classical', type=int, default=0, help='1: classical one-pass members of oa-vcb\'s class on the test set')
ap.add_argument('--out-dir', default='.')
ap.add_argument('--rerun', action='store_true')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
if args.k0 >= args.level: ap.error(f'--k0 {args.k0} must be below --level {args.level} (the coarse solve is on a coarser mesh)')
if args.transfer_level and args.transfer_level <= args.level: ap.error('--transfer-level must be finer than --level')
dev = args.device
L = args.level
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
os.makedirs(args.out_dir, exist_ok=True)
n_te = int(round(args.test_frac * args.n)); n_tr = args.n - n_te
tr, te = np.arange(n_tr), np.arange(n_tr, args.n)


def build(Lx):
    """Hierarchy, samples (cached), Galerkin coarse solve and raw features on level Lx.  The coefficient
    functions depend only on the rng draws, so sample i is the same function on every level."""
    D = {}
    H = D['H'] = (NestedHierarchy(Lx, coarse=perforated_square(n=args.coarse_n), bc='mixed', neumann=outer)
                  if args.bc == 'insulated' else NestedHierarchy(Lx, coarse=perforated_square(n=args.coarse_n)))
    TP = D['TP'] = TensorProblem(H)
    tag = (f"{args.pde}_{args.bc}_L{Lx}_n{args.coarse_n}_c{args.contrast:.0e}" + ('' if float(f"{args.contrast:.0e}") == args.contrast else f"x{args.contrast:.8g}")
           + (f"_kT{args.k_t:g}" if args.pde == 'aniso' else '')
           + f"_{args.n}" + (f"_ds{args.data_seed}" if args.data_seed else ''))
    D['tag'] = tag
    cache = os.path.join(args.out_dir, f"pool_scalar_{tag}.pkl")
    pool = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    if pool and not reusable(pool.get('code'), args.strict_code, f"data pool {os.path.basename(cache)}"):
        pool = {}                                         # 6 Oct (Astra): cached numerical data are versioned like results
    if 'K' not in pool:
        rng = np.random.default_rng(3000 + args.data_seed); t0 = time.time()
        pool['K'] = [field_tows(H, args.contrast, rng, k_T=args.k_t) if args.pde == 'aniso'
                     else iso(field_channels(H, args.contrast, rng)) for _ in range(args.n)]
        pool['u'] = [spsolve(TP.A_fine(K), TP.F) for K in pool['K']]
        pool['t_direct'] = (time.time() - t0) / args.n; pool['code'] = CODE_HASH
        atomic_pickle(pool, cache)
    ck = f"uc_k{args.k0}"
    if args.k0 and ck not in pool:
        Pk = None
        for k in range(args.k0 + 1, Lx + 1):
            Pk = H.P[k] if Pk is None else H.P[k] @ Pk
        Pk = Pk.tocsr(); PkT = Pk.T.tocsr(); t0 = time.time()
        pool[ck] = np.stack([Pk @ spsolve((PkT @ TP.A_fine(K) @ Pk).tocsc(), PkT @ TP.F) for K in pool['K']])
        pool[ck + '_t'] = (time.time() - t0) / args.n
        atomic_pickle(pool, cache)
    D['pool'] = pool
    D['U'] = U = np.stack(pool['u'])
    D['UC'] = pool[ck] if args.k0 else np.zeros_like(U)
    D['nI'] = len(H.nodes[Lx]); D['n_all'] = n_all = len(H.coords)
    kp = H.keep
    def nodal(v_elem):                                    # (ne, c) -> (n_all, c): mean over adjacent elements
        s_ = np.zeros((n_all, v_elem.shape[1])); c_ = np.zeros(n_all)
        for j in range(3):
            np.add.at(s_, H.elem[kp, j], v_elem[kp]); np.add.at(c_, H.elem[kp, j], 1)
        return s_ / np.maximum(c_, 1)[:, None]
    D['KF_raw'] = np.stack([nodal(tensor_features(K)) for K in pool['K']])    # (n, n_all, 3)
    D['dom'] = dom = np.nonzero(H.in_domain)[0]
    pos_dom = {g: i for i, g in enumerate(dom)}
    D['take_dom'] = torch.tensor([pos_dom[g] for g in H.nodes[Lx]], device=dev)
    return D


def finish(D, mu_f, sd_f, scale):
    """Normalised features and torch tensors of a level (training-level statistics)."""
    H = D['H']; n_all = D['n_all']
    KF = (D['KF_raw'] - mu_f) / sd_f
    if args.pde == 'darcy': KF = KF[..., :1]                                      # isotropic: log a only
    xy = (H.coords - H.coords.min(0)) / (H.coords.max(0) - H.coords.min(0))
    flag = np.zeros(n_all); flag[H.boundary] = 1.0; flag[H.neumann_nodes] = -1.0
    ucn = np.zeros((args.n, n_all)); ucn[:, H.nodes[H.L]] = D['UC'] / scale
    FEAT = D['FEAT'] = np.concatenate([KF, np.broadcast_to(xy, (args.n, n_all, 2)),
                                       np.broadcast_to(flag[None, :, None], (args.n, n_all, 1))]
                                      + ([ucn[..., None]] if args.k0 else []), -1)
    D['X_dom'] = torch.tensor(FEAT[:, D['dom']], dtype=torch.float32, device=dev)
    D['X_int'] = torch.tensor(FEAT[:, H.nodes[H.L]], dtype=torch.float32, device=dev)
    D['ELEM'] = torch.tensor(H.elem, device=dev)
    D['MU'] = torch.tensor(D['TP'].M, dtype=torch.float32, device=dev)          # (ne, 3, 3, 3)
    D['KK'] = torch.tensor(np.stack([np.where(H.keep[:, None], K, 0.0) for K in D['pool']['K']]),
                           dtype=torch.float32, device=dev)
    D['IDX'] = torch.tensor(H.nodes[H.L], device=dev)


def energy_sq(D, v, idx):                                  # v: (B, nI);  sum_e sum_c K_c e^T M_c e
    full = v.new_zeros(v.shape[0], D['n_all']); full[:, D['IDX']] = v
    ge = full[:, D['ELEM']]                                 # (B, ne, 3)
    q = torch.einsum('bei,ecij,bej->bec', ge, D['MU'], ge)
    return (q * D['KK'][idx]).sum((1, 2))


D0 = build(L)
H, U, UC, nI = D0['H'], D0['U'], D0['UC'], D0['nI']
tag = D0['tag']
print(f"{args.pde}, pores {args.bc}, level {L}: {nI} unknowns ({len(H.neumann_nodes)} insulated pore nodes), "
      f"contrast {args.contrast:.0e}" + (f", k_T {args.k_t:g}" if args.pde == 'aniso' else '') +
      f", {args.n} samples, device {dev}", flush=True)
print(f"  direct solve {D0['pool']['t_direct'] * 1e3:.0f} ms/sample", flush=True)
TGT = U - UC
scale = float(np.sqrt(np.mean(TGT[tr] ** 2)))
Y = torch.tensor(TGT / scale, dtype=torch.float32, device=dev)
KFt = D0['KF_raw'][tr]
mu_f, sd_f = KFt.mean((0, 1)), KFt.std((0, 1)) + 1e-8
finish(D0, mu_f, sd_f, scale)
cin = D0['FEAT'].shape[-1]


def report(D, pred, idx):
    """rel-L2 and energy error of full predictions (numpy, (len(idx), nI))."""
    Ut = D['U'][idx]; e = pred - Ut
    rl = float(np.mean(np.linalg.norm(e, axis=1) / np.linalg.norm(Ut, axis=1)))
    en = float(np.mean([np.sqrt(e[j] @ D['TP'].A_fine(D['pool']['K'][i]) @ e[j] /
                                (Ut[j] @ D['TP'].A_fine(D['pool']['K'][i]) @ Ut[j])) for j, i in enumerate(idx)]))
    return rl, en


if args.k0:
    r_ = report(D0, UC[te], te)
    print(f"  coarse Galerkin solve to level {args.k0}: {D0['pool']['uc_k%d_t' % args.k0] * 1e3:.0f} ms/sample; alone it gives "
          f"rel-L2 {r_[0]:.3e}, energy {r_[1]:.3e}", flush=True)


MASS = fe_mass(D0['H'], L)


def report_extra(D, pred, idx):
    """5 Oct (review): FE L2 (mass-matrix) relative error, and how many predictions are worse in energy than u_c alone."""
    s = per_sample(D, pred, idx)
    return float(np.mean(s['l2fe_s'])), s['n_worse']


def per_sample(D, pred, idx):
    """6 Oct (Astra): per-sample nodal rel-l2, energy and FE L2 errors with the sample ids (paired comparisons, worst cases)."""
    Ut = D['U'][idx]; e = pred - Ut; out = dict(sample_ids=[int(i) for i in idx], rl2_s=[], en_s=[], l2fe_s=[])
    l2 = rel_norm(MASS, e, Ut); worse = 0
    for j, i in enumerate(idx):
        Af = D['TP'].A_fine(D['pool']['K'][i]); ec = UC[i] - Ut[j]
        out['rl2_s'].append(float(np.linalg.norm(e[j]) / np.linalg.norm(Ut[j])))
        out['en_s'].append(float(np.sqrt((e[j] @ Af @ e[j]) / (Ut[j] @ Af @ Ut[j]))))
        out['l2fe_s'].append(float(l2[j])); worse += int((e[j] @ Af @ e[j]) > (1 + 1e-6) * (ec @ Af @ ec))
    out['n_worse'] = worse
    return out


def loss_fn(p, t, idx):
    p, t = p.reshape(len(idx), -1), t.reshape(len(idx), -1)
    rl = (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1)).mean()
    if args.loss == 'rel_l2': return rl
    en = (energy_sq(D0, p - t, idx).clamp_min(0).sqrt() / energy_sq(D0, t, idx).clamp_min(1e-30).sqrt()).mean()
    return en if args.loss == 'energy' else 0.5 * (en + rl)


from model_zoo import Zoo, GRID_KINDS, OP_KINDS, n_params
ZOO = Zoo(cin, dev, cout=1, grid_ext=args.grid_ext, batch=args.batch, scale=scale)
D0['k0'] = args.k0
if any(k in OP_KINDS for k in args.models):    # error-equation inputs (README 2 Oct)
    if not args.k0: raise SystemExit('oa-ll-op / gnn-op need --k0 > 0')
    t0 = time.time(); ZOO.op_setup(D0, [D0['TP'].A_fine(K) for K in D0['pool']['K']], D0['TP'].F, tr)
    print(f"  residual input + operator weights a_ij/a_ii built in {time.time() - t0:.0f}s", flush=True)

# ------------------------------------------------------------------------------------ oa-vcb (5 Oct)
# The learned operator-adapted V-cycle of train_oa_layer.py on the tensor operator: same model (oa_vcycle.py), the
# per-sample stiffness matrices passed directly, the structural pattern from a random tensor coefficient (nonzero on
# edges where every isotropic stiffness cancels), residual input r = F - A u_c, output in target units.
VC_KINDS = ('oa-vc', 'oa-vcb')
if any(k in VC_KINDS for k in args.models) or args.classical:
    from oa_vcycle import OAData, OAVCycle
    t0 = time.time(); VC_LEVELS = 0; k_ = L
    while k_ >= 1 and VC_LEVELS < 4 and len(H.nodes[k_ - 1]) >= 16:
        VC_LEVELS += 1; k_ -= 1
    _g = np.random.default_rng(54321)
    A_struct = D0['TP'].A_fine(np.stack([_g.uniform(1, 2, H.n_elem), _g.uniform(-0.5, 0.5, H.n_elem),
                                         _g.uniform(1, 2, H.n_elem)], 1))
    A_ALL = [D0['TP'].A_fine(K).tocsr() for K in D0['pool']['K']]
    OAD = OAData(H, None, VC_LEVELS, 4, dev, H.level_coords, near_kernel=True, A_list=A_ALL, A_struct=A_struct)
    RV = torch.tensor(np.array([D0['TP'].F - A_ALL[i] @ UC[i] for i in range(args.n)]) / scale, dtype=torch.float64, device=dev)
    print(f"  V-cycle operator data ({VC_LEVELS} levels, coarsest {OAD.n0} nodes) built in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / args.n * 1e3:.0f} ms/sample); near-kernel vectors per sample: max {max(OAD.k_nk)}, "
          f"{np.mean(np.array(OAD.k_nk) > 0) * 100:.0f}% of samples have one", flush=True)
def vc_make(kind, w):
    return OAVCycle(OAD, width=w, nu=4, project='both' if kind == 'oa-vcb' else False,
                    deflate=kind == 'oa-vcb', ckpt=bool(args.vc_ckpt))
def vc_forward(model, it):
    idx = it.cpu().numpy() if torch.is_tensor(it) else it
    model.set_ops(*OAD.batch(idx), Z=OAD.batch_z(idx), A64=OAD.batch_a64(idx)); return model(RV[idx]).float()
def vc_predict(model, idx):
    model.eval(); out = []
    with torch.no_grad():
        for b in np.array_split(np.asarray(idx), int(np.ceil(len(idx) / args.batch))):
            out.append(vc_forward(model, b).cpu().numpy())
    return np.concatenate(out) * scale + UC[np.asarray(idx)]

# ------------------------------------------------------------------------------------ training
res_path = os.path.join(args.out_dir, f"results_scalar_{tag}.jsonl")
cfg = json.dumps(dict(pde=args.pde, bc=args.bc, level=L, contrast=args.contrast, k_t=args.k_t, n=args.n, epochs=args.epochs,
                      loss=args.loss, k0=args.k0, params=args.params, seed=args.seed, lr=args.lr, batch=args.batch,
                      **({'grid_ext': args.grid_ext} if args.grid_ext != 'zero' else {}),
                      data_seed=args.data_seed, **({'test_frac': args.test_frac} if args.test_frac != 0.25 else {})),
                 sort_keys=True)
chash = hashlib.md5(cfg.encode()).hexdigest()[:10]
done = {}
if os.path.exists(res_path):
    for line in open(res_path):
        r = json.loads(line)
        if r['cfg'] == cfg and reusable(r.get('code'), args.strict_code, f"recorded result of {r['model']}"):
            done[r['model']] = r['result']
ng0 = ZOO.grid_data(D0)[0].ng if any(k in GRID_KINDS for k in args.models) else 0
if ng0:
    for ext, (_, f_) in ZOO.grid_floor(D0, U[te]).items():   # interpolation floor of the exact solution
        print(f"  grid baselines: mesh->grid->mesh round trip of the exact solution, {ext}-extended into the pores, "
              f"costs rel-L2 {f_:.3e}" + ('   <- used' if ext == args.grid_ext else ''), flush=True)
results, trained = {}, {}
for kind in args.models:
    mpath = os.path.join(args.out_dir, f"model_scalar_{chash}_{kind}.pt")
    if kind in done and not args.rerun:
        print(f"\n[{kind}] already done -- skipped"); results[kind] = done[kind]
        if args.transfer_level and os.path.exists(mpath) and reusable(
                torch.load(mpath, map_location='cpu', weights_only=False).get('code'), args.strict_code, f"model {os.path.basename(mpath)}"):
            st = torch.load(mpath, map_location=dev, weights_only=False)
            model = ZOO.make(kind, st['width'], D0, ng0).to(dev); model.load_state_dict(st['state'], strict=False)
            trained[kind] = model
        continue
    if kind in VC_KINDS:                                # widest oa-vc within the parameter budget
        lo = max(w for w in range(4, 33) if n_params(vc_make(kind, w)) <= args.params)
    else:
        lo = ZOO.fit_width(kind, args.params, D0, ng0)
    if lo is None:
        print(f"\n[{kind}] cannot be built within {args.params:,} parameters -- skipped (the published WNO needs ~75k)")
        continue
    torch.manual_seed(args.seed)
    model = (vc_make(kind, lo) if kind in VC_KINDS else ZOO.make(kind, lo, D0, ng0)).to(dev)
    note = (f"learned operator-adapted V-cycle, {VC_LEVELS} levels" + (", deflation + Galerkin output, loss on projected + plain"
            if kind == 'oa-vcb' else '')) if kind in VC_KINDS else ZOO.note(kind, D0, ng0)
    print(f"\n[{kind}] width {lo}, params {n_params(model):,}  ({note})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    nb = int(np.ceil(n_tr / args.batch))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * nb)
    g = np.random.default_rng(args.seed); t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for idx in np.array_split(g.permutation(tr), nb):
            it = torch.tensor(idx, device=dev)
            if kind in VC_KINDS:
                out = vc_forward(model, it); loss = loss_fn(out, Y[it], it)
                if getattr(model, 'aux_out', None) is not None: loss = loss + loss_fn(model.aux_out.float(), Y[it], it)
            else:
                loss = loss_fn(ZOO.forward(kind, model, D0, it), Y[it], it)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == 0:
            print(f"    {kind:<10} ep {ep + 1:>4}/{args.epochs}  train {args.loss} {tot / nb:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    train_s = time.time() - t0
    torch.save(dict(width=lo, state=model.state_dict(), cfg=cfg, code=CODE_HASH), mpath)
    PRED = (lambda idx: vc_predict(model, idx)) if kind in VC_KINDS else (lambda idx: ZOO.predict(kind, model, D0, idx))
    t1 = time.time(); pred = PRED(te); infer_ms = (time.time() - t1) / n_te * 1e3
    rl2, en = report(D0, pred, te)
    trs = tr[:len(te)]; rl2_tr, en_tr = report(D0, PRED(trs), trs)   # generalisation gap
    print(f"    {kind:<10} TRAIN rel-L2 {rl2_tr:.3e}  energy {en_tr:.3e}  (first {len(trs)} training samples)", flush=True)
    ps = per_sample(D0, pred, te); l2fe, n_worse = float(np.mean(ps['l2fe_s'])), ps['n_worse']
    r = dict(rel_l2=rl2, energy=en, params=n_params(model), train_s=train_s, infer_ms=infer_ms,
             train_rel_l2=rl2_tr, train_energy=en_tr, l2fe=l2fe, n_worse_than_uc=n_worse,
             sample_ids=ps['sample_ids'], rl2_s=ps['rl2_s'], en_s=ps['en_s'], l2fe_s=ps['l2fe_s'])
    print(f"    {kind:<10} TEST  nodal rel-l2 {rl2:.3e}  FE rel-L2 {l2fe:.3e}  energy {en:.3e}  "
          f"(worse than u_c in energy on {n_worse}/{n_te})", flush=True)
    results[kind] = r; trained[kind] = model
    with open(res_path, 'a') as fh: fh.write(json.dumps(dict(cfg=cfg, model=kind, result=r, code=CODE_HASH, n_train=int(n_tr), n_test=int(n_te))) + '\n')

if args.classical:                                       # classical one-pass members of oa-vcb's class, same u_c
    from oa_vcycle import oa_hierarchy, classical_vcycle, classical_vcycle_proj, near_kernel_basis
    t0 = time.time(); cv, cm = [], []
    for i in te:
        A = A_ALL[i]; levs, A0 = oa_hierarchy(H, A, VC_LEVELS, 4); r = D0['TP'].F - A @ UC[i]
        cv.append(UC[i] + classical_vcycle(levs, A0, r, fine_lmax=None))
        cm.append(UC[i] + classical_vcycle_proj(levs, A0, r, Z=near_kernel_basis(A), fine_lmax=None))
    for name, P_ in (('classical V(4,4), op.-adapted', cv), ('classical member (+ defl. + proj.)', cm)):
        rl2, en = report(D0, np.array(P_), te); ps = per_sample(D0, np.array(P_), te); l2fe = float(np.mean(ps['l2fe_s']))
        print(f"  {name:<36} nodal rel-l2 {rl2:.3e}  FE rel-L2 {l2fe:.3e}  energy {en:.3e}", flush=True)
        results[name] = dict(rel_l2=rl2, energy=en, l2fe=l2fe, params=0, train_s=0.0, infer_ms=(time.time() - t0) / n_te * 1e3,
                             sample_ids=ps['sample_ids'], rl2_s=ps['rl2_s'], en_s=ps['en_s'], l2fe_s=ps['l2fe_s'], classical=True)
        with open(res_path, 'a') as fh:                  # 6 Oct: classical comparators recorded with per-sample errors
            fh.write(json.dumps(dict(cfg=cfg, model=name, result=results[name], code=CODE_HASH, n_train=int(n_tr), n_test=int(n_te))) + '\n')

print(f"\n\nRESULTS  {args.pde}, pores {args.bc}, level {L}, contrast {args.contrast:.0e}, {n_tr}/{n_te}, {args.epochs} epochs, "
      f"loss {args.loss}, k0 {args.k0}\nmodel          params  rel-L2 (T)  energy (flux)  train s  infer ms")
for k, r in results.items():
    print(f"{k:<12}{r['params']:>9,}   {r['rel_l2']:.3e}      {r['energy']:.3e}   {r['train_s']:>6.0f}   {r['infer_ms']:>6.1f}")

# ------------------------------------------------------------------------- zero-shot transfer to a finer level
if args.transfer_level and trained:
    Lt = args.transfer_level
    Dt = build(Lt); finish(Dt, mu_f, sd_f, scale)
    ngt = 2 * ng0 + 1 if ng0 else None
    print(f"\nZERO-SHOT TRANSFER  level {L} -> {Lt} ({Dt['nI']} unknowns); grid models on their training grid {ng0} "
          f"and on a {ngt} grid (the published WNO only on its training grid: its resolution change needs a dyadic size "
          f"that db6 boundary extension does not give)", flush=True)
    if args.k0:
        r_ = report(Dt, Dt['UC'][te], te)
        print(f"  coarse solve alone on level {Lt}: rel-L2 {r_[0]:.3e}, energy {r_[1]:.3e}")
    # reference: the level-L exact solution prolonged to level Lt (what a perfect level-L model would give)
    Pup = None
    for k in range(L + 1, Lt + 1):
        Pup = Dt['H'].P[k] if Pup is None else Dt['H'].P[k] @ Pup
    r_ = report(Dt, (Pup @ U[te].T).T, te)
    print(f"  exact level-{L} solution prolonged to level {Lt}: rel-L2 {r_[0]:.3e}, energy {r_[1]:.3e}  "
          f"(the floor for any model that only reproduces level {L})")
    tpath = os.path.join(args.out_dir, f"results_scalar_{tag}_transfer_L{Lt}.jsonl")
    for kind, model in trained.items():
        if kind in VC_KINDS: print(f"  {kind}: zero-shot transfer not implemented (hierarchy is per level)"); continue
        grids = ([ng0] + ([ngt] if kind != 'wno-off' else [])) if kind in GRID_KINDS else [None]
        for ngx in grids:
            name = kind + (f'@{ngx}' if ngx else '')
            try:
                ZOO.rebind(kind, model, Dt)
                pred = ZOO.predict(kind, model, Dt, te, ngx)
            except Exception as ex:                         # e.g. per-level oa-ll on a deeper hierarchy
                print(f"  {name:<14} cannot run on level {Lt}: {type(ex).__name__}: {ex}"); continue
            rl2, en = report(Dt, pred, te)
            print(f"  {name:<14} level {L} -> {Lt}: rel-L2 {rl2:.3e}  energy {en:.3e}   (native {results[kind]['rel_l2']:.3e} / "
                  f"{results[kind]['energy']:.3e})", flush=True)
            with open(tpath, 'a') as fh:
                fh.write(json.dumps(dict(cfg=cfg, model=name, transfer_level=Lt, result=dict(rel_l2=rl2, energy=en), code=CODE_HASH)) + '\n')
