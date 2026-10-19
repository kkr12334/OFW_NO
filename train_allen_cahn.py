"""Allen-Cahn u0 -> u(T) on the perforated unstructured mesh (the third problem class), same protocol as
train_scalar.py / train_oa_layer.py: capacity-matched models, the same data, epochs, batch / lr and objective,
the coarse solve on the level-k0 mesh (the same Allen-Cahn scheme, prolonged) supplied to every model as an
input, the network learning u(T) - u_c.  Errors on held-out samples: relative L2 of the nodal values (as in
the other classes), relative H1 seminorm (the interfaces; no energy norm for a nonlinear problem) and the
mass-weighted relative L2.

    python -u train_allen_cahn.py --level 3 --k0 2 --loss mix --models oa-ll oa-ll-s gnn transolver lino wno fno \
        --transfer-level 4 --out-dir <dir>

Models and --transfer-level: see model_zoo.py and train_scalar.py.
Solver and its verification: allen_cahn.py, diagnostics_allen_cahn/verify_allen_cahn.py.
"""
import argparse, hashlib, json, os, pickle, time
import numpy as np, torch
from allen_cahn import neumann_hierarchy, ACOperators, grf_initial
from model_zoo import Zoo, GRID_KINDS, n_params

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, default=3)
ap.add_argument('--coarse-n', type=int, default=9)
ap.add_argument('--eps', type=float, default=0.015, help='interface parameter: u_t = eps^2 Lap u - (u^3 - u)')
ap.add_argument('--T', type=float, default=5.0, help='final time')
ap.add_argument('--dt', type=float, default=0.01)
ap.add_argument('--S', type=float, default=2.0, help='stabilisation constant (>= 1 for energy stability)')
ap.add_argument('--tau', type=float, default=15.0, help='GRF initial condition: inverse length scale')
ap.add_argument('--alpha', type=float, default=2.0, help='GRF initial condition: smoothness exponent')
ap.add_argument('--u0-std', type=float, default=0.6, help='GRF initial condition: std before clipping to [-1, 1]')
ap.add_argument('--n', type=int, default=400)
ap.add_argument('--test-frac', type=float, default=0.25)
ap.add_argument('--epochs', type=int, default=300)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--loss', default='mix', choices=['rel_l2', 'h1', 'mix'])
ap.add_argument('--k0', type=int, default=2)
ap.add_argument('--params', type=int, default=22000)
ap.add_argument('--models', nargs='+', default=['oa-ll', 'transolver', 'lino', 'wno', 'fno'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--data-seed', type=int, default=0)
ap.add_argument('--grid-ext', default='nearest', choices=['nearest', 'zero'],
                help='grid baselines: how node fields are extended into the pores (Neumann pores: nearest is fair)')
ap.add_argument('--grid-ng', type=int, default=0, help='grid side for FNO/WNO/LiNO (0 = matched to the node count)')
ap.add_argument('--transfer-level', type=int, default=0,
                help='evaluate the trained models zero-shot on this (finer) level; 0 = off')
ap.add_argument('--out-dir', default='.')
ap.add_argument('--rerun', action='store_true')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
if args.k0 >= args.level: ap.error(f'--k0 {args.k0} must be below --level {args.level} (the coarse solve is on a coarser mesh)')
if args.transfer_level and args.transfer_level <= args.level: ap.error('--transfer-level must be finer than --level')
dev = args.device
L = args.level
os.makedirs(args.out_dir, exist_ok=True)
n_te = int(round(args.test_frac * args.n)); n_tr = args.n - n_te
tr, te = np.arange(n_tr), np.arange(n_tr, args.n)


def build(Lx):
    """Hierarchy, samples (cached), coarse solve and raw features on level Lx.  u0 is the same function on every
    level (FFT field on a 256^2 grid interpolated to the nodes), so sample i is the same problem on every level."""
    D = {}
    H = D['H'] = neumann_hierarchy(Lx, args.coarse_n)
    OP = D['OP'] = ACOperators(H)
    tag = (f"L{Lx}_n{args.coarse_n}_eps{args.eps:g}_T{args.T:g}_dt{args.dt:g}_S{args.S:g}_tau{args.tau:g}_a{args.alpha:g}"
           f"_sd{args.u0_std:g}_{args.n}" + (f"_ds{args.data_seed}" if args.data_seed else ''))
    D['tag'] = tag
    cache = os.path.join(args.out_dir, f"pool_ac_{tag}.pkl")
    pool = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
    if 'u' not in pool:
        rng = np.random.default_rng(5000 + args.data_seed)
        pool['u0'] = grf_initial(H, args.n, rng, tau=args.tau, alpha=args.alpha, std=args.u0_std)
        t0 = time.time(); OP.solve(pool['u0'][0], Lx, args.eps, args.T, args.dt, args.S); t_one = time.time() - t0
        t0 = time.time(); OP.solve(pool['u0'][1], Lx, args.eps, args.T, args.dt, args.S); pool['t_direct'] = time.time() - t0
        pool['t_factor'] = max(t_one - pool['t_direct'], 0.0)
        t0 = time.time()
        pool['u'] = OP.solve(pool['u0'], Lx, args.eps, args.T, args.dt, args.S)
        pool['t_batched'] = (time.time() - t0) / args.n
        pickle.dump(pool, open(cache, 'wb'))
    ck = f"uc_k{args.k0}"
    if args.k0 and ck not in pool:
        OP.coarse_solve(pool['u0'][0], args.k0, args.eps, args.T, args.dt, args.S)          # factorisation
        t0 = time.time(); OP.coarse_solve(pool['u0'][1], args.k0, args.eps, args.T, args.dt, args.S)
        pool[ck + '_t'] = time.time() - t0
        pool[ck] = OP.coarse_solve(pool['u0'], args.k0, args.eps, args.T, args.dt, args.S)
        pickle.dump(pool, open(cache, 'wb'))
    D['pool'] = pool
    D['U'], D['U0'] = pool['u'], pool['u0']
    D['UC'] = pool[ck] if args.k0 else np.zeros_like(pool['u'])
    D['nI'] = len(H.nodes[Lx]); D['n_all'] = n_all = len(H.coords)
    xy = (H.coords - H.coords.min(0)) / (H.coords.max(0) - H.coords.min(0))
    flag = np.zeros(n_all); flag[H.neumann_nodes] = -1.0          # boundary (all Neumann) nodes
    u0n = np.zeros((args.n, n_all)); u0n[:, H.nodes[Lx]] = D['U0']
    ucn = np.zeros((args.n, n_all)); ucn[:, H.nodes[Lx]] = D['UC']  # u0, u_c are O(1): fed unscaled
    D['FEAT'] = FEAT = np.concatenate([u0n[..., None], np.broadcast_to(xy, (args.n, n_all, 2)),
                                       np.broadcast_to(flag[None, :, None], (args.n, n_all, 1))]
                                      + ([ucn[..., None]] if args.k0 else []), -1)
    D['dom'] = dom = np.nonzero(H.in_domain)[0]
    pos_dom = {g: i for i, g in enumerate(dom)}
    D['take_dom'] = torch.tensor([pos_dom[g] for g in H.nodes[Lx]], device=dev)
    D['X_dom'] = torch.tensor(FEAT[:, dom], dtype=torch.float32, device=dev)
    D['X_int'] = torch.tensor(FEAT[:, H.nodes[Lx]], dtype=torch.float32, device=dev)
    Kc = OP.K[Lx].tocoo()
    D['KT'] = torch.sparse_coo_tensor(np.vstack([Kc.row, Kc.col]), Kc.data.astype('float32'), Kc.shape, device=dev).coalesce()
    return D


def report(D, pred, true):
    OP = D['OP']; e = pred - true
    return (float(np.mean(np.linalg.norm(e, axis=1) / np.linalg.norm(true, axis=1))),
            float(np.mean(np.sqrt(OP.h1_sq(e) / OP.h1_sq(true)))),
            float(np.mean(np.sqrt(OP.l2_sq(e) / OP.l2_sq(true)))))


D0 = build(L)
H, U, U0, UC, nI, tag, pool = D0['H'], D0['U'], D0['U0'], D0['UC'], D0['nI'], D0['tag'], D0['pool']
print(f"Allen-Cahn, Neumann on all boundaries, level {L}: {nI} unknowns, eps {args.eps:g}, T {args.T:g}, dt {args.dt:g}, "
      f"S {args.S:g}, u0 GRF (tau {args.tau:g}, alpha {args.alpha:g}, std {args.u0_std:g}), {args.n} samples, device {dev}", flush=True)
print(f"  fine solve: {pool['t_direct'] * 1e3:.0f} ms/sample ({int(round(args.T / args.dt))} steps, one factorisation shared "
      f"by all samples, {pool['t_factor'] * 1e3:.0f} ms); batched {pool['t_batched'] * 1e3:.1f} ms/sample", flush=True)
if args.k0:
    r_ = report(D0, UC[te], U[te])
    print(f"  coarse Allen-Cahn solve on level {args.k0} ({len(H.nodes[args.k0])} nodes): {pool['uc_k%d_t' % args.k0] * 1e3:.0f} ms/sample; "
          f"alone it gives rel-L2 {r_[0]:.3e}, H1-semi {r_[1]:.3e}", flush=True)
r_ = report(D0, U0[te], U[te])
print(f"  reference: u0 itself as the prediction gives rel-L2 {r_[0]:.3e}, H1-semi {r_[1]:.3e}", flush=True)
TGT = U - UC
scale = float(np.sqrt(np.mean(TGT[tr] ** 2)))
Y = torch.tensor(TGT / scale, dtype=torch.float32, device=dev)


def h1_sq(v):                                              # v: (B, nI), H1 seminorm with the stiffness matrix
    return (v * torch.sparse.mm(D0['KT'], v.t()).t()).sum(1)


def loss_fn(p, t, idx):
    p, t = p.reshape(len(idx), -1), t.reshape(len(idx), -1)
    rl = (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1)).mean()
    if args.loss == 'rel_l2': return rl
    h1 = (h1_sq(p - t).clamp_min(0).sqrt() / h1_sq(t).clamp_min(1e-30).sqrt()).mean()
    return h1 if args.loss == 'h1' else 0.5 * (h1 + rl)


cin = D0['FEAT'].shape[-1]
ZOO = Zoo(cin, dev, cout=1, grid_ext=args.grid_ext, batch=args.batch, scale=scale)
NG = args.grid_ng or None

# ------------------------------------------------------------------------------------ training
res_path = os.path.join(args.out_dir, f"results_ac_{tag}.jsonl")
cfg = json.dumps(dict(level=L, coarse_n=args.coarse_n, eps=args.eps, T=args.T, dt=args.dt, S=args.S, tau=args.tau,
                      alpha=args.alpha, u0_std=args.u0_std, n=args.n, epochs=args.epochs, loss=args.loss, k0=args.k0,
                      params=args.params, seed=args.seed, lr=args.lr, batch=args.batch, data_seed=args.data_seed,
                      grid_ext=args.grid_ext, grid_ng=args.grid_ng),
                 sort_keys=True)
chash = hashlib.md5(cfg.encode()).hexdigest()[:10]
done = {}
if os.path.exists(res_path):
    for line in open(res_path):
        r = json.loads(line)
        if r['cfg'] == cfg: done[r['model']] = r['result']
ng0 = ZOO.grid_data(D0, NG)[0].ng if any(k in GRID_KINDS for k in args.models) else 0
if ng0:
    for ext, (rt, f_) in ZOO.grid_floor(D0, U[te], NG).items():   # interpolation floor of the exact u(T)
        r_ = report(D0, rt, U[te])
        print(f"  grid baselines: mesh->grid->mesh round trip of the exact u(T), {ext}-extended into the pores, "
              f"costs rel-L2 {r_[0]:.3e}, H1-semi {r_[1]:.3e} (interpolation floor)"
              + ('   <- used' if ext == args.grid_ext else ''), flush=True)
results, trained = {}, {}
for kind in args.models:
    mpath = os.path.join(args.out_dir, f"model_ac_{chash}_{kind}.pt")
    if kind in done and not args.rerun:
        print(f"\n[{kind}] already done -- skipped"); results[kind] = done[kind]
        if args.transfer_level and os.path.exists(mpath):
            st = torch.load(mpath, map_location=dev, weights_only=False)
            model = ZOO.make(kind, st['width'], D0, ng0).to(dev); model.load_state_dict(st['state'], strict=False)
            trained[kind] = model
        continue
    lo = ZOO.fit_width(kind, args.params, D0, ng0)
    if lo is None:
        print(f"\n[{kind}] cannot be built within {args.params:,} parameters -- skipped (the published WNO needs ~75k)")
        continue
    torch.manual_seed(args.seed); model = ZOO.make(kind, lo, D0, ng0).to(dev)
    fwd = lambda m, it: ZOO.forward(kind, m, D0, it, NG)
    print(f"\n[{kind}] width {lo}, params {n_params(model):,}  ({ZOO.note(kind, D0, ng0)})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    nb = int(np.ceil(n_tr / args.batch))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * nb)
    g = np.random.default_rng(args.seed); t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for idx in np.array_split(g.permutation(tr), nb):
            it = torch.tensor(idx, device=dev)
            loss = loss_fn(fwd(model, it), Y[it], it)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == 0:
            print(f"    {kind:<10} ep {ep + 1:>4}/{args.epochs}  train {args.loss} {tot / nb:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    train_s = time.time() - t0
    torch.save(dict(width=lo, state=model.state_dict()), mpath)
    t1 = time.time(); pred = ZOO.predict(kind, model, D0, te, NG); infer_ms = (time.time() - t1) / n_te * 1e3
    rl2, h1, l2m = report(D0, pred, U[te])
    r = dict(rel_l2=rl2, h1=h1, l2_mass=l2m, params=n_params(model), train_s=train_s, infer_ms=infer_ms)
    print(f"    {kind:<10} TEST  rel-L2 {rl2:.3e}  H1-semi {h1:.3e}  (mass-L2 {l2m:.3e})", flush=True)
    results[kind] = r; trained[kind] = model
    with open(res_path, 'a') as fh: fh.write(json.dumps(dict(cfg=cfg, model=kind, result=r)) + '\n')

print(f"\n\nRESULTS  Allen-Cahn, level {L}, eps {args.eps:g}, T {args.T:g}, {n_tr}/{n_te}, {args.epochs} epochs, "
      f"loss {args.loss}, k0 {args.k0}\nmodel          params  rel-L2      H1-semi     mass-L2    train s  infer ms")
for k, r in results.items():
    print(f"{k:<12}{r['params']:>9,}   {r['rel_l2']:.3e}   {r['h1']:.3e}   {r['l2_mass']:.3e}   {r['train_s']:>6.0f}   {r['infer_ms']:>6.1f}")

# ------------------------------------------------------------------------- zero-shot transfer to a finer level
if args.transfer_level and trained:
    Lt = args.transfer_level
    Dt = build(Lt)
    ngt = 2 * ng0 + 1 if ng0 else None
    print(f"\nZERO-SHOT TRANSFER  level {L} -> {Lt} ({Dt['nI']} unknowns); grid models on their training grid {ng0} "
          f"and on a {ngt} grid (the published WNO only on its training grid)", flush=True)
    if args.k0:
        r_ = report(Dt, Dt['UC'][te], Dt['U'][te])
        print(f"  coarse solve alone on level {Lt}: rel-L2 {r_[0]:.3e}, H1-semi {r_[1]:.3e}")
    Pup = Dt['OP'].Pc[L]                                     # level L -> level Lt prolongation
    r_ = report(Dt, (Pup @ U[te].T).T, Dt['U'][te])
    print(f"  exact level-{L} solution prolonged to level {Lt}: rel-L2 {r_[0]:.3e}, H1-semi {r_[1]:.3e}  "
          f"(the floor for any model that only reproduces level {L})")
    tpath = os.path.join(args.out_dir, f"results_ac_{tag}_transfer_L{Lt}.jsonl")
    for kind, model in trained.items():
        grids = ([ng0] + ([ngt] if kind != 'wno-off' else [])) if kind in GRID_KINDS else [None]
        for ngx in grids:
            name = kind + (f'@{ngx}' if ngx else '')
            try:
                ZOO.rebind(kind, model, Dt)
                pred = ZOO.predict(kind, model, Dt, te, ngx)
            except Exception as ex:
                print(f"  {name:<14} cannot run on level {Lt}: {type(ex).__name__}: {ex}"); continue
            rl2, h1, l2m = report(Dt, pred, Dt['U'][te])
            print(f"  {name:<14} level {L} -> {Lt}: rel-L2 {rl2:.3e}  H1-semi {h1:.3e}   (native {results[kind]['rel_l2']:.3e} / "
                  f"{results[kind]['h1']:.3e})", flush=True)
            with open(tpath, 'a') as fh:
                fh.write(json.dumps(dict(cfg=cfg, model=name, transfer_level=Lt, result=dict(rel_l2=rl2, h1=h1, l2_mass=l2m))) + '\n')
