"""Second problem class: plane-strain elasticity of a porous fibre composite on an unstructured mesh.

Same protocol as train_oa_layer.py for Darcy: capacity-matched models, same data/epochs/objective,
exact coarse solve (Galerkin on the level-k0 mesh) supplied to every model, errors on held-out samples
in the relative L2 norm of the displacement and in the energy (strain-energy) norm, which measures the
stress error.  Models: oa-ll (learnable lifting on the mesh hierarchy), transolver (mesh-native),
lino / wno / fno (grid, through the mesh<->grid bridge); oa-ll-op and gnn-op (3 Oct) in their 2x2-block form:
residual input D^-1 r and operator blocks D_i^-1 A_ij, D_i the nodal 2x2 diagonal block (block Jacobi).

    python -u train_elasticity.py --level 3 --n 400 --epochs 300 --loss mix --k0 2 --contrast 1e2 \
        --models oa-ll-op gnn-op --out-dir <dir>

    python -u train_elasticity.py --level 3 --n 400 --epochs 300 --loss mix --k0 2 \
        --models oa-ll transolver lino wno fno --out-dir <dir>
"""
import argparse, hashlib, json, os, pickle, time
import numpy as np, scipy.sparse as sp, torch, torch.nn as nn
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, fibre_composite
from oa_lifting import field_channels
from mesh_grid_bridge import MeshGridBridge
from run_identity import CODE_HASH, reusable
from fe_metrics import fe_mass, fe_mass_vector, rel_norm, atomic_pickle

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, default=3)
ap.add_argument('--coarse-n', type=int, default=9)
ap.add_argument('--contrast', type=float, default=1e3)
ap.add_argument('--vf', type=float, default=0.3, help='fibre volume fraction (short fibres)')
ap.add_argument('--field', default='bands', choices=['bands', 'fibres'],
                help="'bands': continuous stiff fibre bands crossing the domain (4 meandering bands, "
                     "horizontal or vertical); 'fibres': random short circular fibres")
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
ap.add_argument('--params', type=int, default=22000, help='parameter budget for every model')
ap.add_argument('--models', nargs='+', default=['oa-ll', 'transolver', 'lino', 'wno', 'fno'])
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--ll-output', default='latent', choices=['latent', 'synth'],
                help="oa-ll output: 'latent' (inverse lifting + pointwise decoder) or 'synth' (network gives "
                     "hierarchical-basis coefficients, synthesised through the geometric interpolation P)")
ap.add_argument('--resid-feat', default='none', choices=['none', 'resid', 'jres'],
                help="extra input: residual of the coarse solution r = F - A u_c ('resid') or its Jacobi "
                     "step diag(A)^-1 r ('jres'); one sparse mat-vec per sample, given to ALL models")
ap.add_argument('--coarse-rho', type=float, default=0, help='with --coarse-space adapted: 0 = exact, >0 = local patches of this radius (coarse widths)')
ap.add_argument('--coarse-space', default='galerkin', choices=['galerkin', 'adapted'],
                help="coarse solve: 'galerkin' (geometric P) or 'adapted' (exact operator-adapted, Route V "
                     "Phi = [I; -A_FF^-1 A_FC], one level, needs k0 = level-1; a reference, expensive)")
ap.add_argument('--comp-scale', action='store_true',
                help='normalise u_x and u_y separately (network output parameterisation only; the loss is unchanged)')
ap.add_argument('--grid-ext', default='nearest', choices=['nearest', 'zero'],
                help="grid baselines: extension of node fields into the pores. 'zero' was used before 28 Sep 2026; "
                     "with insulated / traction-free pores it puts a jump at every pore edge (round-trip floor of the "
                     "exact solution ~15%% rel-L2 on the anisotropic L3 case, against 0.75%% with 'nearest')")
ap.add_argument('--vc-ckpt', type=int, default=0, help='oa-vc*: activation checkpointing of the smoothers (memory)')
ap.add_argument('--classical', type=int, default=0, help='1: classical one-pass members of oa-vcb\'s class on the test set')
ap.add_argument('--out-dir', default='.')
ap.add_argument('--rerun', action='store_true')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
if args.k0 >= args.level: ap.error(f'--k0 {args.k0} must be below --level {args.level} (the coarse solve is on a coarser mesh)')
dev = torch.device(args.device); torch.manual_seed(args.seed); np.random.seed(args.seed)
L = args.level
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
H = NestedHierarchy(L, coarse=perforated_square(n=args.coarse_n), bc='mixed', neumann=outer)
V = VectorHierarchy(H)
nI = len(H.nodes[L])
print(f"elasticity, porous {args.field} composite, level {L}: {2 * nI} dofs ({len(H.neumann_nodes)} traction-free "
      f"pore nodes), E_f/E_m {args.contrast:.0e}, vf {args.vf}, {args.n} samples, device {dev}", flush=True)

# ------------------------------------------------------------------------------------------- data
os.makedirs(args.out_dir, exist_ok=True)
tag = (f"L{L}_n{args.coarse_n}_{args.field}_c{args.contrast:.0e}" + ('' if float(f"{args.contrast:.0e}") == args.contrast else f"x{args.contrast:.8g}"))
tag = tag + (f"_vf{args.vf}" if args.field == 'fibres' else '') + f"_{args.n}"
cache = os.path.join(args.out_dir, f"pool_elast_{tag}.pkl")
pool = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {}
if pool and not reusable(pool.get('code'), args.strict_code, f"data pool {os.path.basename(cache)}"):
    pool = {}                                             # 6 Oct (Astra): cached numerical data are versioned like results
if 'E' not in pool:
    rng = np.random.default_rng(2000); t0 = time.time()
    pool['E'] = [fibre_composite(H, args.contrast, rng, vf=args.vf) if args.field == 'fibres'
                 else field_channels(H, args.contrast, rng) for _ in range(args.n)]
    pool['u'] = [spsolve(V.A_fine(E), V.F) for E in pool['E']]
    pool['t_direct'] = (time.time() - t0) / args.n; pool['code'] = CODE_HASH
    atomic_pickle(pool, cache)
print(f"  direct solve {pool['t_direct'] * 1e3:.0f} ms/sample", flush=True)
U = np.stack(pool['u'])
ck = f"uc_k{args.k0}" + (('_adapted' + (f'_rho{args.coarse_rho:g}' if args.coarse_rho else '')) if args.coarse_space == 'adapted' else '')
if args.k0 and args.coarse_space == 'adapted' and ck not in pool:
    if args.k0 != L - 1: raise SystemExit('--coarse-space adapted: only k0 = level-1 is implemented')
    from scipy.sparse.linalg import splu
    Pg = V.P[L].tocsr(); ncd = Pg.shape[1]; Ek = V.E[L]; t0 = time.time(); UCa = []
    for E in pool['E']:
        A_ = V.A_fine(E).tocsr()
        A_FF = (Ek.T @ A_ @ Ek).tocsc(); A_FC = (Ek.T @ A_)[:, :ncd].tocsc()
        if args.coarse_rho:
            from oa_lifting import _local_C
            Ph = _local_C(A_FF, A_FC, V.new_coords(L), V.level_coords(L - 1), args.coarse_rho * H.h[L - 1])
        else:
            Ph = -splu(A_FF).solve(A_FC.toarray())
        Phi = sp.vstack([sp.identity(ncd, format='csr'), sp.csr_matrix(Ph)]).tocsr()
        UCa.append(Phi @ spsolve((Phi.T @ A_ @ Phi).tocsc(), Phi.T @ V.F))
    pool[ck] = np.stack(UCa); pool[ck + '_t'] = (time.time() - t0) / args.n
    atomic_pickle(pool, cache)
if args.k0 and ck not in pool:
    Pk = None
    for k in range(args.k0 + 1, L + 1):
        Pk = V.P[k] if Pk is None else V.P[k] @ Pk
    Pk = Pk.tocsr(); PkT = Pk.T.tocsr(); t0 = time.time()
    pool[ck] = np.stack([Pk @ spsolve((PkT @ V.A_fine(E) @ Pk).tocsc(), PkT @ V.F) for E in pool['E']])
    pool[ck + '_t'] = (time.time() - t0) / args.n
    atomic_pickle(pool, cache)
UC = pool[ck] if args.k0 else np.zeros_like(U)
n_te = int(round(args.test_frac * args.n)); n_tr = args.n - n_te
tr, te = np.arange(n_tr), np.arange(n_tr, args.n)
TGT = U - UC
if args.comp_scale:
    scale = np.array([np.sqrt(np.mean(TGT[tr][:, c::2] ** 2)) for c in (0, 1)])
    scale_dof = np.tile(scale, nI)                                  # interleaved (x, y) per node
else:
    scale = float(np.sqrt(np.mean(TGT[tr] ** 2))); scale_dof = np.full(2 * nI, scale)
SD = torch.tensor(scale_dof, dtype=torch.float32, device=dev)
Y = torch.tensor(TGT / scale_dof, dtype=torch.float32, device=dev).reshape(args.n, nI, 2)
if args.k0:
    e = U - UC
    print(f"  coarse {args.coarse_space} solve to level {args.k0}: {pool[ck + '_t'] * 1e3:.0f} ms/sample; alone it gives "
          f"rel-L2 {np.mean(np.linalg.norm(e[te], axis=1) / np.linalg.norm(U[te], axis=1)):.3e}", flush=True)

# energy (strain-energy) norm, element by element: e^T A e = sum_e E_e ge^T K1_e ge
n_all2 = 2 * len(H.coords)
EDOF = torch.tensor(V.edofs, device=dev)
K1 = torch.tensor(V.K1, dtype=torch.float32, device=dev)
Emat = torch.tensor(np.stack([np.where(H.keep, E, 0.0) for E in pool['E']]), dtype=torch.float32, device=dev)
IDX = torch.tensor(V.nodes[L], device=dev)
def energy_sq(v, idx):                                     # v: (B, 2 nI) dof vectors
    full = v.new_zeros(v.shape[0], n_all2); full[:, IDX] = v
    ge = full[:, EDOF]                                     # (B, ne, 6)
    return (Emat[idx] * torch.einsum('bei,eij,bej->be', ge, K1, ge)).sum(1)

def loss_fn(p, t, idx):
    p, t = p.reshape(len(idx), -1) * SD, t.reshape(len(idx), -1) * SD     # physical units (scale-free ratios)
    rl = (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1)).mean()
    if args.loss == 'rel_l2': return rl
    en = (energy_sq(p - t, idx).clamp_min(0).sqrt() / energy_sq(t, idx).clamp_min(1e-30).sqrt()).mean()
    return en if args.loss == 'energy' else 0.5 * (en + rl)

# ------------------------------------------------------------------------------------- features
n_all = len(H.coords); kp = H.keep
def nodal(v_elem):
    s_ = np.zeros(n_all); c_ = np.zeros(n_all)
    for j in range(3):
        np.add.at(s_, H.elem[kp, j], v_elem[kp]); np.add.at(c_, H.elem[kp, j], 1)
    return np.where(c_ > 0, s_ / np.maximum(c_, 1), 0.0)
LE = np.stack([nodal(np.log(E)) for E in pool['E']])
LE = (LE - LE[tr].mean()) / (LE[tr].std() + 1e-8)
xy = (H.coords - H.coords.min(0)) / (H.coords.max(0) - H.coords.min(0))
flag = np.zeros(n_all); flag[H.boundary] = 1.0; flag[H.neumann_nodes] = -1.0
ucn = np.zeros((args.n, n_all, 2)); ucn[:, H.nodes[L]] = (UC / scale_dof).reshape(args.n, nI, 2)
FEAT = np.concatenate([LE[..., None], np.broadcast_to(xy, (args.n, n_all, 2)),
                       np.broadcast_to(flag[None, :, None], (args.n, n_all, 1))]
                      + ([ucn] if args.k0 else []), -1)            # (n, n_all, c)
if args.resid_feat != 'none':
    if not args.k0: raise SystemExit('--resid-feat needs --k0 > 0')
    rk = f"res_{args.resid_feat}_k{args.k0}"
    if rk not in pool:
        R = []
        for E, uc in zip(pool['E'], UC):
            A_ = V.A_fine(E); r = V.F - A_ @ uc
            R.append(r / A_.diagonal() if args.resid_feat == 'jres' else r)
        pool[rk] = np.stack(R); atomic_pickle(pool, cache)
    RS = pool[rk]; rs = float(np.sqrt(np.mean(RS[tr] ** 2)))
    rn = np.zeros((args.n, n_all, 2)); rn[:, H.nodes[L]] = (RS / rs).reshape(args.n, nI, 2)
    FEAT = np.concatenate([FEAT, rn], -1)
dom = np.nonzero(H.in_domain)[0]
pos_dom = {g: i for i, g in enumerate(dom)}
take_dom = torch.tensor([pos_dom[g] for g in H.nodes[L]], device=dev)
X_dom = torch.tensor(FEAT[:, dom], dtype=torch.float32, device=dev)
X_int = torch.tensor(FEAT[:, H.nodes[L]], dtype=torch.float32, device=dev)
cin = FEAT.shape[-1]

def n_params(m): return sum(p.numel() for p in m.parameters())

from model_zoo import OP_KINDS                     # every error-equation kind (model_zoo)
if any(k in OP_KINDS for k in args.models):
    if not args.k0: raise SystemExit('oa-ll-op / gnn-op need --k0 > 0 (the residual of the coarse solve is their input)')
    if args.resid_feat != 'none': raise SystemExit('oa-ll-op / gnn-op build their own residual input; drop --resid-feat')

VC_KINDS = ('oa-vc', 'oa-vcb')
def vc_setup():
    """oa-vcb for elasticity (5 Oct): the learned operator-adapted V-cycle of train_oa_layer.py on the dof graph
    (unknown approach): prolongations P_k (x) I_2 with coarse dofs first (VectorHierarchy), route-V predict by 4 Jacobi
    steps of prolongator smoothing, Galerkin levels, exact coarsest solve.  Rigid-body modes are kept by every Jacobi
    step (P1 interpolation reproduces linear fields, A annihilates them in the interior).  No near-kernel deflation:
    the scalar island detector does not apply to vector problems, and the stiff bands reach the clamped boundary;
    the Galerkin output is kept.  Input r = F - A u_c (scaled), output the correction (same scale)."""
    if getattr(vc_setup, 'done', False): return
    if args.comp_scale: raise SystemExit('oa-vc* needs one scale for both components (drop --comp-scale)')
    from oa_vcycle import OAData
    global OAD, RV, VC_LEVELS, A_ALL
    t0 = time.time(); VC_LEVELS = 0; k_ = L
    while k_ >= 1 and VC_LEVELS < 4 and len(H.nodes[k_ - 1]) >= 16:
        VC_LEVELS += 1; k_ -= 1
    A_ALL = [V.A_fine(E).tocsr() for E in pool['E']]
    OAD = OAData(V, None, VC_LEVELS, 4, dev, V.level_coords, near_kernel=False, A_list=A_ALL)
    RV = torch.tensor(np.array([V.F - A_ALL[i] @ UC[i] for i in range(args.n)]) / scale, dtype=torch.float64, device=dev)
    print(f"  V-cycle operator data ({VC_LEVELS} levels, coarsest {OAD.n0} dofs) built in {time.time() - t0:.0f}s "
          f"({(time.time() - t0) / args.n * 1e3:.0f} ms/sample)", flush=True)
    vc_setup.done = True

def builder(kind):
    if kind in VC_KINDS:
        vc_setup(); from oa_vcycle import OAVCycle
        mk = lambda w: OAVCycle(OAD, width=w, nu=4, project='both' if kind == 'oa-vcb' else False,
                                deflate=False, ckpt=bool(args.vc_ckpt))
        def fwd(m, it):
            idx = it.cpu().numpy(); m.set_ops(*OAD.batch(idx), A64=OAD.batch_a64(idx)); return m(RV[idx]).float().reshape(len(idx), nI, 2)
        return mk, fwd, f"learned operator-adapted V-cycle on the dof graph, {VC_LEVELS} levels" + (
            ", Galerkin output, loss on projected + plain" if kind == 'oa-vcb' else '')
    if kind in ('gnn', 'gnn30', 'wno-off') or kind in OP_KINDS or (kind.startswith('oa-ll') and kind != 'oa-ll'):   # via model_zoo
        from model_zoo import Zoo
        D0 = builder.D0 = getattr(builder, 'D0', None) or dict(H=H, FEAT=FEAT, X_int=X_int, X_dom=X_dom, dom=dom,
                                                                take_dom=take_dom, UC=UC, k0=args.k0)
        Z = Zoo(cin, dev, cout=2, grid_ext=args.grid_ext)
        if kind in OP_KINDS and 'OPW' not in D0:
            # error equation A e = r, r = F - A u_c, with 2x2 nodal blocks (README 3 Oct): inputs log E, xy, flag and
            # D^-1 r (block Jacobi, replaces u_c), operator blocks W_ij = D_i^-1 A_ij on the edges, output x sigma
            t0 = time.time(); Z.op_setup_block(D0, [V.A_fine(E) for E in pool['E']], V.F, tr, b=2)
            sg = D0['SIG'].cpu().numpy()
            print(f"  residual input D^-1 r + operator blocks D_i^-1 A_ij built in {time.time() - t0:.0f}s; "
                  f"rel. spread of |D^-1 r| over samples {sg.std() / sg.mean():.2f}", flush=True)
        ng0 = Z.grid_data(D0)[0].ng if kind == 'wno-off' else None
        return ((lambda w: Z.make(kind, w, D0, ng0, ll_output=args.ll_output)),
                (lambda m, idx: Z.forward(kind, m, D0, idx).reshape(len(idx), nI, 2)), Z.note(kind, D0, ng0))
    if kind == 'oa-ll':
        from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO
        MG = MeshHierarchyGeometry(H, dev)
        return (lambda w: MeshLiftingNO(MG, cin, width=w, out_channels=2, output=args.ll_output)), (lambda m, idx: m(X_int[idx])), MG.describe()
    if kind == 'transolver':
        from baselines_extra import Transolver
        return (lambda w: Transolver(cin, hidden=w, out_channels=2)), \
               (lambda m, idx: m(X_dom[idx])[:, take_dom]), f"{len(dom)} mesh nodes, native"
    # grid models through the bridge
    BR = builder.BR = getattr(builder, 'BR', None) or MeshGridBridge(H)
    Xg = builder.Xg = getattr(builder, 'Xg', None)
    if Xg is None:
        full = FEAT.copy(); full[:, ~H.in_domain] = 0
        chans = [BR.node_field_to_grid(full[..., c], args.grid_ext) for c in range(cin)]
        g = np.stack(chans + [np.broadcast_to(BR.mask.astype(float), chans[0].shape)], -1)
        builder.Xg = Xg = torch.tensor(g, dtype=torch.float32, device=dev)
        co = BR.to_nodes.tocoo()
        builder.TN = torch.sparse_coo_tensor(np.vstack([co.row, co.col]), co.data.astype('float32'),
                                             co.shape, device=dev).coalesce()
    TN, ng = builder.TN, BR.ng
    def to_nodes(out):                                   # (B, ng, ng, 2) -> (B, nI, 2)
        B = out.shape[0]
        flat = out.reshape(B, ng * ng, 2).permute(1, 0, 2).reshape(ng * ng, B * 2)
        return torch.sparse.mm(TN, flat).reshape(nI, B, 2).permute(1, 0, 2)
    c2 = Xg.shape[-1]
    if kind == 'lino':
        from baselines_extra import LiNO2D
        mk = lambda w: LiNO2D(c2, hidden=w, levels=3, out_channels=2)
        return mk, (lambda m, idx: to_nodes(m(Xg[idx]))), BR.report()
    from ofwno_nets import RealFNO2D, RealWNO2D
    def mk(w):
        m = RealFNO2D(c2, hidden=w, modes=min(12, ng // 2), n_layers=4) if kind == 'fno' else RealWNO2D(c2, hidden=w, n_layers=4, J=2)
        m.output_proj[2] = nn.Conv2d(w, 2, 1)
        return m
    return mk, (lambda m, idx: to_nodes(m(Xg[idx]).permute(0, 2, 3, 1))), BR.report()   # (B,2,n,n) -> nodes

# ------------------------------------------------------------------------------------ training
res_path = os.path.join(args.out_dir, f"results_elast_{tag}.jsonl")
cfg = json.dumps(dict(level=L, field=args.field, contrast=args.contrast, vf=args.vf, n=args.n, epochs=args.epochs, loss=args.loss,
                      k0=args.k0, params=args.params, seed=args.seed, lr=args.lr, batch=args.batch,
                      **({'grid_ext': args.grid_ext} if args.grid_ext != 'zero' else {}),
                      **({'ll_output': args.ll_output} if args.ll_output != 'latent' else {}),
                      **({'comp_scale': True} if args.comp_scale else {}),
                      **({'coarse_space': args.coarse_space} if args.coarse_space != 'galerkin' else {}),
                      **({'coarse_rho': args.coarse_rho} if args.coarse_rho else {}),
                      **({'resid_feat': args.resid_feat} if args.resid_feat != 'none' else {}),
                      **({'test_frac': args.test_frac} if args.test_frac != 0.25 else {})), sort_keys=True)
MASS2 = fe_mass_vector(fe_mass(H, L))


def per_sample(Pn):
    """6 Oct (Astra): per-sample nodal rel-l2, energy (float64, assembled operator) and FE L2 errors with the sample ids."""
    out = dict(sample_ids=[int(i) for i in te], rl2_s=[], en_s=[], l2fe_s=[float(v) for v in rel_norm(MASS2, Pn - U[te], U[te])])
    worse = 0
    for j, i in enumerate(te):
        Ai = V.A_fine(pool['E'][i]); e = Pn[j] - U[i]; ec = UC[i] - U[i]
        out['rl2_s'].append(float(np.linalg.norm(e) / np.linalg.norm(U[i])))
        out['en_s'].append(float(np.sqrt((e @ (Ai @ e)) / (U[i] @ (Ai @ U[i])))))
        worse += int((e @ (Ai @ e)) > (1 + 1e-6) * (ec @ (Ai @ ec)))
    out['n_worse'] = worse
    return out                    # 5 Oct (review H): FE L2 of the displacement, (x, y) interleaved
done = {}
if os.path.exists(res_path):
    for line in open(res_path):
        r = json.loads(line)
        if r['cfg'] == cfg and reusable(r.get('code'), args.strict_code, f"recorded result of {r['model']}"):
            done[r['model']] = r['result']
results = {}
for kind in args.models:
    if kind in done and not args.rerun:
        print(f"\n[{kind}] already done -- skipped"); results[kind] = done[kind]; continue
    mk, fwd, note = builder(kind)
    lo, hi = 2, 256
    while hi - lo > 1:
        mid = (lo + hi) // 2
        try: ok = n_params(mk(mid)) <= args.params
        except Exception: ok = False
        if ok: lo = mid
        else: hi = mid
    torch.manual_seed(args.seed); model = mk(lo).to(dev)
    if n_params(model) > args.params:
        print(f"\n[{kind}] cannot be built within {args.params:,} parameters -- skipped"); continue
    print(f"\n[{kind}] width {lo}, params {n_params(model):,}  ({note})", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    nb = int(np.ceil(n_tr / args.batch))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * nb)
    g = np.random.default_rng(args.seed); t0 = time.time()
    for ep in range(args.epochs):
        model.train(); tot = 0.0
        for idx in np.array_split(g.permutation(tr), nb):
            it = torch.tensor(idx, device=dev)
            loss = loss_fn(fwd(model, it), Y[it], it)
            if getattr(model, 'aux_out', None) is not None and model.training:   # oa-vcb: + loss of the plain cycle
                loss = loss + loss_fn(model.aux_out.float().reshape(len(it), nI, 2), Y[it], it)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item()
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == 0:
            print(f"    {kind:<10} ep {ep + 1:>4}/{args.epochs}  train {args.loss} {tot / nb:.3e}  ({time.time() - t0:.0f}s)", flush=True)
    train_s = time.time() - t0
    model.eval(); P = []; t1 = time.time()
    with torch.no_grad():
        for idx in np.array_split(te, int(np.ceil(n_te / args.batch))):
            P.append(fwd(model, torch.tensor(idx, device=dev)).reshape(len(idx), -1))
        P = torch.cat(P)
        infer_ms = (time.time() - t1) / n_te * 1e3
        Ut = torch.tensor(U[te], dtype=torch.float32, device=dev)
        pred = P * SD + torch.tensor(UC[te], dtype=torch.float32, device=dev)
        ite = torch.tensor(te, device=dev)
        rl2 = float((torch.linalg.norm(pred - Ut, dim=1) / torch.linalg.norm(Ut, dim=1)).mean())
        en = float((energy_sq(pred - Ut, ite).sqrt() / energy_sq(Ut, ite).sqrt()).mean())
        UCt = torch.tensor(UC[te], dtype=torch.float32, device=dev)
        n_worse = int((energy_sq(pred - Ut, ite) > (1 + 1e-6) * energy_sq(UCt - Ut, ite)).sum())
    pn = pred.double().cpu().numpy(); ps = per_sample(pn); l2fe = float(np.mean(ps['l2fe_s']))
    # 6 Oct (Astra): the reported means come from the saved float64 per-sample errors (the float32 values above were only
    # the GPU-side estimate)
    rl2, en, n_worse = float(np.mean(ps['rl2_s'])), float(np.mean(ps['en_s'])), ps['n_worse']
    r = dict(rel_l2=rl2, energy=en, params=n_params(model), train_s=train_s, infer_ms=infer_ms, l2fe=l2fe, n_worse_than_uc=n_worse,
             sample_ids=ps['sample_ids'], rl2_s=ps['rl2_s'], en_s=ps['en_s'], l2fe_s=ps['l2fe_s'])
    torch.save(dict(width=lo, state=model.state_dict(), cfg=cfg, code=CODE_HASH),     # 6 Oct: weights kept for re-evaluation
               os.path.join(args.out_dir, f"model_elast_{hashlib.md5(cfg.encode()).hexdigest()[:10]}_{kind}.pt"))
    print(f"    {kind:<10} TEST  nodal rel-l2 {rl2:.3e}  FE rel-L2 {l2fe:.3e}  energy {en:.3e}  "
          f"(worse than u_c in energy on {n_worse}/{n_te})", flush=True)
    results[kind] = r
    with open(res_path, 'a') as fh: fh.write(json.dumps(dict(cfg=cfg, model=kind, result=r, code=CODE_HASH, n_train=int(n_tr), n_test=int(n_te))) + '\n')

if args.classical:                                       # classical one-pass members of oa-vcb's class, same u_c
    vc_setup()
    from oa_vcycle import oa_hierarchy, classical_vcycle
    t0 = time.time(); Pc = []
    for i in te:
        A = A_ALL[i]; levs, A0 = oa_hierarchy(V, A, VC_LEVELS, 4)
        Pc.append(UC[i] + classical_vcycle(levs, A0, V.F - A @ UC[i], fine_lmax=None))
    Pc = torch.tensor(np.array(Pc), dtype=torch.float32, device=dev); Ut = torch.tensor(U[te], dtype=torch.float32, device=dev)
    ite = torch.tensor(te, device=dev)
    c_ms = (time.time() - t0) / n_te * 1e3                # measured: hierarchy build + one cycle, CPU, per sample
    ps = per_sample(Pc.double().cpu().numpy())
    rl2, en = float(np.mean(ps['rl2_s'])), float(np.mean(ps['en_s']))
    print(f"  classical V(4,4), op.-adapted (dof graph)   nodal rel-l2 {rl2:.3e}  FE rel-L2 {np.mean(ps['l2fe_s']):.3e}  "
          f"energy {en:.3e}  ({c_ms:.0f} ms/sample incl. build)", flush=True)
    results['classical V(4,4) op.-adapted'] = dict(rel_l2=rl2, energy=en, params=0, train_s=0.0, infer_ms=c_ms,
                                                   infer_ms_note='CPU, hierarchy build + one cycle', classical=True,
                                                   n_worse_than_uc=ps['n_worse'],
                                                   l2fe=float(np.mean(ps['l2fe_s'])), sample_ids=ps['sample_ids'],
                                                   rl2_s=ps['rl2_s'], en_s=ps['en_s'], l2fe_s=ps['l2fe_s'])
    with open(res_path, 'a') as fh:                      # 6 Oct: classical comparator recorded with per-sample errors
        fh.write(json.dumps(dict(cfg=cfg, model='classical V(4,4) op.-adapted', result=results['classical V(4,4) op.-adapted'],
                                 code=CODE_HASH, n_train=int(n_tr), n_test=int(n_te))) + '\n')

print(f"\n\nRESULTS  elasticity, level {L}, E_f/E_m {args.contrast:.0e}, {n_tr}/{n_te}, {args.epochs} epochs, "
      f"loss {args.loss}, k0 {args.k0}\nmodel          params  rel-L2 (u)  energy (stress)  train s  infer ms")
for k, r in results.items():
    print(f"{k:<11}{r['params']:>10,}{r['rel_l2']:>12.3e}{r['energy']:>17.3e}{r['train_s']:>9.0f}{r['infer_ms']:>10.1f}")
