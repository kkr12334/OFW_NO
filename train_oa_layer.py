"""Step 4 of the formulation's next-experiments list: train the OA-lifting layer.

Models, all on the SAME samples, same loss (relative L2 of the interior nodal
solution), same optimiser, epochs, batch and seed:

  oa         OA-lifting NO, per-sample LOCAL adapted transform (rho), Jacobi-scaled,
             shared local graph kernel + low-rank global term      <- the proposal
  oa-hb      same network, transform = plain hierarchical basis (C = 0)
             -> isolates the value of operator adaptation
  oa-frozen  same network, transform built once at a = 1 (localised, rho)
  oa-norank  'oa' with the low-rank term switched off (rank 0)
             -> isolates the value of the low-rank component (formulation section 7)
  fno, wno   your RealFNO2D / RealWNO2D on the exact node grid (build_grid_bijection),
             hidden width bisected to match the 'oa' parameter count

Also reported: the per-sample transform build time the OA models pay at
inference (the honest total cost), and the energy-norm error.

usage:
  python train_oa_layer.py                                   # level 4 smoke test, CPU, ~minutes
  python train_oa_layer.py --level 5 --field channels --n 300 --epochs 200
  python train_oa_layer.py --level 6 --field blocks8 --n 300 --epochs 200 --models oa oa-hb fno wno
The pool is cached to <out-dir>/pool_oa_L{level}_{field}_{contrast}_{n}_rho{rho}.pkl, and every
finished model is appended to <out-dir>/results_train_oa_L{level}_{field}.jsonl, so an
interrupted run resumes where it stopped (same arguments). On Colab use
  --out-dir /content/drive/MyDrive/ofwno_runs
One model at a time also works:  --models oa   then  --models oa-hb   ...
"""
import argparse, hashlib, json, os, pickle, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_identity import CODE_HASH, reusable
import numpy as np
import torch
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp
sys.path.insert(0, '.')
from oa_lifting import NestedHierarchy, make_field, OATransform, pcg_kappa, ritz_extremes
from oa_layer_torch import (SharedGeometry, OABatch, OALiftingNO, sample_operator_data,
                            frozen_C, n_params, nodal_log_a)

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, default=4)
ap.add_argument('--field', default='channels', choices=['channels', 'blocks8', 'blocks16', 'blocks4', 'grf'])
ap.add_argument('--contrast', type=float, default=1e4)
ap.add_argument('--classical', type=int, default=0,
                help='1: after the table, evaluate classical methods from the SAME u_c on the SAME test set (needs --k0 > 0 '
                     'and --coarse-solve galerkin): coarse alone, the best one-pass Chebyshev post-smoother (4 and 16 steps), '
                     'and two-grid PCG, with the iterations it needs to match each model.  CPU, a few ms per sample.')
ap.add_argument('--eval-partial', action='store_true',
                help='evaluate an UNFINISHED training checkpoint as it is (no further training) and print the result; nothing '
                     'is written to the results file or the checkpoint, so a later normal run still resumes and finishes')
ap.add_argument('--vc-ckpt', type=int, default=0,
                help='oa-vc*: 1 = activation checkpointing of the smoothers during training (same gradients, ~30%% slower, '
                     'far less GPU memory; needed for L4 on a 16 GB GPU)')
ap.add_argument('--oa-steps', type=int, default=4,
                help='oa-ll-op-oa: Jacobi steps (omega 2/3) of prolongator smoothing from the geometric predict towards the '
                     'route-V predict -A_FF^-1 A_FC at every lifting level (diagnostics_cost/cheap_oa_predict.log)')
ap.add_argument('--load', default='all', choices=['all', 'no-low', 'random'],
                help="'all': f = 1 on the whole domain (every earlier run).  'no-low': f = 1 only on elements with a >= 0.5, "
                     "i.e. no source inside the low-conductivity barriers of --contrast < 1 (3 Oct, diagnostics_cost/"
                     "barrier_anatomy.log: with f = 1 inside an insulator, 96%% of ||u||^2 sits in the barriers).  "
                     "For channels with contrast >= 1 the two are identical.")
ap.add_argument('--strict-code', type=int, default=0,
                help='1: reuse finished results / checkpoints only if produced by the same code version (run_identity.py); '
                     'otherwise retrain.  The paper runs use 1.')
ap.add_argument('--n', type=int, default=120)
ap.add_argument('--test-frac', type=float, default=0.25)
ap.add_argument('--epochs', type=int, default=60)
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--lr', type=float, default=2e-3)
ap.add_argument('--width', type=int, default=32)
ap.add_argument('--params', type=int, default=0,
                help='parameter budget for the capacity-matched models (0 = that of the oa model at --width, as before)')
ap.add_argument('--layers', type=int, default=4)
ap.add_argument('--rank', type=int, default=16,
                help='low-rank size per level (or its cap when --rank-scale is set)')
ap.add_argument('--rank-scale', type=float, default=None,
                help='level-dependent rank r_k = min(rank, ceil(s*sqrt(m_k))). Channel spectra '
                     '(3 samples, L7 run) need s ~ 1.0 at level 4, 1.3 at level 5, 1.2-1.7 at '
                     'level 6: use s = 1.7 with --rank 256')
ap.add_argument('--radius', type=float, default=2.0, help='graph-kernel radius in units of h_k')
ap.add_argument('--rho', type=float, default=2.0, help='patch radius of the local adapted transform')
ap.add_argument('--coarse-solve', default='adapted', choices=['adapted', 'galerkin', 'oa'],
                help="'oa' (3 Oct): Galerkin solve on the OPERATOR-ADAPTED coarse space range(Phi), Phi = prod_k [I; Phat_k], "
                     "Phat_k = route-V a-harmonic predict -A_FF^-1 A_FC localised to --coarse-rho coarse widths, A_{k-1} = Phi_k^T A_k Phi_k "
                     "(diagnostics_cost/operator_dependent_coarse.log: fixes the barrier field for one-pass methods; L3 rho 3 ~190 ms/sample).  "
                     "'galerkin': standard FE solve on the level-k0 mesh (P^T A P), prolonged to the fine mesh. "
                     "Level 4 perforated, k0 3: 13 ms/sample and coarse-only rel-L2 7.8e-2, against 1650 ms "
                     "and 2.5e-1 for the rho=2 adapted coarse block. 'adapted' kept for reproducing old runs.")
ap.add_argument('--coarse-rho', type=float, default=0,
                help='patch radius of the transform used for the exact coarse solve (0 = same as --rho). '
                     'The coarse space is only as good as its localisation: at level 4 of the perforated '
                     'mesh the coarse-only rel-L2 (k0 3) is 2.5e-1 at rho 2, 3.2e-2 at rho 3, 1.4e-2 at rho 4.')
ap.add_argument('--k0', type=int, default=0,
                help='EXACT COARSE SOLVE up to this level: solve K_k0 c = g_k0 directly per sample, '
                     'prolong it to the fine mesh, feed it to the network as an input channel and '
                     'have the network learn only the DETAIL correction u - u_k0. k0=0 (default) is '
                     'the current behaviour, where V_0 = {0} and the network learns everything. '
                     'Uses the scale-orthogonality of Theorem 3.4: the coarse block decouples from '
                     'every detail block, so solving it exactly costs one small direct solve.')
ap.add_argument('--certify', type=int, default=1,
                help='1 (default): for every test prediction of every model, compute the a-posteriori '
                     'certificate eta^2 = r^T B r with B the Route V preconditioner, report its '
                     'effectivity eta/||e||_A, and the warm-started PCG iterations needed to reach '
                     'the exact discrete solution. 0: skip (saves ~0.2 s per test sample).')
ap.add_argument('--cert-n', type=int, default=0,
                help='certify only the first N test samples (0 = all). The certificate builds a transform '
                     'per sample and radius; at level 4 of the perforated mesh that is ~7 s/sample.')
ap.add_argument('--ckpt-every', type=int, default=0,
                help='save a resumable training checkpoint every N epochs (0 = at every progress line)')
ap.add_argument('--cert-rhos', default='1,2,4',
                help='patch radii of the transforms used for the certificate. Small rho = cheap B and '
                     'a looser eta; large rho = B close to A^-1 and eta -> ||e||_A exactly '
                     '(Corollary 6.2). The sweep IS the result: effectivity vs cost of B.')
ap.add_argument('--loss', choices=['rel_l2', 'energy', 'mix'], default='rel_l2',
                help="rel_l2: relative L2 of the nodal solution (default). energy: relative "
                     "ENERGY norm ||u-uhat||_A/||u||_A -- the norm the certificate bounds, and the "
                     "one rel_l2 training never optimises. mix: their mean.")
ap.add_argument('--route', choices=['V', 'W'], default='V',
                help="V: operator in the predict (well-conditioned level blocks, no low-rank term "
                     "needed); W: operator in the update (the original construction)")
ap.add_argument('--models', nargs='+', default=['oa', 'oa-hb', 'oa-frozen', 'oa-norank', 'fno', 'wno'])
ap.add_argument('--coarse', default='none', choices=['none', 'perforated', 'lshape-u', 'polygon'],
                help='unstructured COARSE MESH: the hierarchy is built by refining it --level times. '
                     'FNO/WNO then need interpolation to and from a uniform grid (mesh_grid_bridge).')
ap.add_argument('--grid-ng', type=int, default=0,
                help='grid side for the FNO/WNO bridge (0 = matched to the mesh node count)')
ap.add_argument('--coarse-n', type=int, default=9, help='coarse mesh density parameter')
ap.add_argument('--ll-scaling', default='none', choices=['none', 'jacobi'],
                help="oa-ll: 'jacobi' = details in energy-normalised coordinates (diag(A_FF)^-1/2 per level)")
ap.add_argument('--ll-output', default='latent', choices=['latent', 'synth'],
                help="oa-ll: 'latent' = inverse lifting + pointwise decoder (LiNO-style); "
                     "'synth' = linear synthesis in the hierarchical FE basis")
ap.add_argument('--ll-ablate', default='none', choices=['none', 'fixed-lift', 'no-attn', 'no-stiffness'],
                help='oa-ll ablation for the paper')
ap.add_argument('--domain', default='square', choices=['square', 'L'],
                help="'L': unit square minus [1/2,1]^2 (re-entrant corner at (1/2,1/2))")
ap.add_argument('--bc', default='dirichlet', choices=['dirichlet', 'mixed'],
                help="'mixed': homogeneous Neumann on the outer top and right edges, Dirichlet elsewhere")
ap.add_argument('--seed', type=int, default=0, help='network initialisation seed')
ap.add_argument('--data-seed', type=int, default=0,
                help='seed for the coefficient samples (and so the train/test split). The pool cache is '
                     'keyed on it; 0 reuses the existing caches. Before this option the pool was always '
                     'drawn once, so earlier "seeds" varied only the network initialisation.')
ap.add_argument('--out-dir', default='.',
                help='where the pool cache and the results file go (on Colab: a Google Drive folder, '
                     'so an interrupted runtime loses nothing)')
ap.add_argument('--reeval', action='store_true',
                help='re-evaluate finished models from their checkpoints (no training) instead of skipping them; needed to get '
                     'the per-sample errors (island split of --classical) for runs made before 4 Oct')
ap.add_argument('--rerun', action='store_true',
                help='retrain models already recorded in the results file (default: skip them)')
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
args = ap.parse_args()
dev = torch.device(args.device)
torch.manual_seed(args.seed); np.random.seed(args.seed)

if args.coarse != 'none':
    import coarse_meshes as cm
    _cm = {'perforated': lambda: cm.perforated_square(n=args.coarse_n),
           'lshape-u': lambda: cm.l_shape_coarse(n=max(5, args.coarse_n // 2)),
           'polygon': lambda: cm.polygon_domain([(0, 0), (1, 0), (1, .4), (.6, .6), (1, .8), (1, 1), (0, 1)],
                                                n=args.coarse_n)}[args.coarse]()
    _nm = (lambda x, y: np.isclose(y, 1.0)) if args.bc == 'mixed' else None
    H = NestedHierarchy(args.level, coarse=_cm, bc=args.bc, neumann=_nm)
    print(f"coarse mesh '{args.coarse}': {len(_cm[0])} nodes, {len(_cm[1])} elements, "
          f"{cm.mesh_quality(*_cm)[2]:.1f} deg min angle -> refined {args.level}x")
else:
    H = NestedHierarchy(args.level, domain=args.domain, bc=args.bc)
L = args.level
N_int = len(H.nodes[L])
_lxy = H.coords[H.elem]
_larea = 0.5 * np.abs((_lxy[:, 1, 0] - _lxy[:, 0, 0]) * (_lxy[:, 2, 1] - _lxy[:, 0, 1])
                      - (_lxy[:, 2, 0] - _lxy[:, 0, 0]) * (_lxy[:, 1, 1] - _lxy[:, 0, 1]))
_lcen = _lxy.mean(1)                                    # element centroids (random loads)


def load_vec(a, i=0):
    """per-sample load vector on H.nodes[L]; 'all' returns H.F unchanged (bit-for-bit as before).
    'random' (5 Oct): a different load for every sample i, element-wise constant, drawn from its own seed so the pool is
    reproducible:  f = c + sum_{p,q<=4} K_pq sin((p+1) pi x) sin((q+1) pi y) / (1 + p + q)  [K_pq ~ N(0,1), c ~ U(-1, 1)]
    plus, with probability 1/2, a Gaussian source of random width 0.02-0.08 and amplitude 2-10 at a random place.
    Both signs, smooth and localised parts: the right-hand-side family the one-pass corrector must handle."""
    if args.load == 'all':
        return H.F
    if args.load == 'random':
        g = np.random.default_rng(777_000 + 100_000 * args.data_seed + i)
        cx, cy = _lcen[:, 0], _lcen[:, 1]; K = g.normal(size=(4, 4))
        fe = g.uniform(-1, 1) + sum(K[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) / (1 + p + q)
                                    for p in range(4) for q in range(4))
        if g.uniform() < 0.5:
            x0 = g.uniform(0.1, 0.9, 2); w = g.uniform(0.02, 0.08)
            fe = fe + g.uniform(2, 10) * np.exp(-((cx - x0[0]) ** 2 + (cy - x0[1]) ** 2) / (2 * w ** 2))
        Fv = np.zeros(len(H.coords)); m = H.keep
        for j in range(3):
            np.add.at(Fv, H.elem[m, j], (fe * _larea)[m] / 3.0)
        return Fv[H.nodes[L]]
    m = H.keep & (np.asarray(a).ravel() >= 0.5)
    Fv = np.zeros(len(H.coords))
    for j in range(3):
        np.add.at(Fv, H.elem[m, j], _larea[m] / 3.0)
    return Fv[H.nodes[L]]


def oa_prolongation(A, k0, rho):
    """Operator-adapted prolongation from level k0 to L (route V, local predict of radius rho coarse widths)."""
    from oa_lifting import _local_C
    Ak = A.tocsr(); Ptot = None
    for k in range(L, k0, -1):
        nc = H.P[k].shape[1]
        assert (H.P[k][:nc] - sp.identity(nc)).nnz == 0, 'level ordering must list the coarse nodes first'
        A_FF = Ak[nc:, nc:].tocsc(); A_FC = Ak[nc:, :nc].tocsr()
        Ph = _local_C(A_FF, A_FC, H.new_coords(k), H.level_coords(k - 1), rho * H.h[k - 1])
        Phi = sp.vstack([sp.identity(nc, format='csr'), sp.csr_matrix(Ph)]).tocsr()
        Ak = (Phi.T @ Ak @ Phi).tocsr()
        Ptot = Phi if Ptot is None else (Ptot @ Phi).tocsr()
    return Ptot


print((f"coarse mesh {args.coarse}" if args.coarse != 'none' else f"domain {args.domain}") + f", bc {args.bc}" + (f" ({len(H.neumann_nodes)} Neumann nodes)" if args.bc == 'mixed' else ''))
print(f"level {L}: {N_int} interior nodes, field {args.field}, contrast {args.contrast:.0e}, "
      f"{args.n} samples, device {dev}" + ('' if args.load == 'all' else f", load {args.load}"))
print(f"ROUTE {args.route}  ({'operator in the PREDICT, D_k = A_FF' if args.route == 'V' else 'operator in the UPDATE, D_k = Schur complement'})"
      f"   rho={args.rho:g}  rank={args.rank}  rank_scale={args.rank_scale}")

# ---------------------------------------------------------------------------
# pool
# ---------------------------------------------------------------------------
need = {'oa': 'local', 'oa-norank': 'local', 'oa-hb': 'hb', 'oa-frozen': 'frozen'}
modes = sorted({need[m] for m in args.models if m in need})
os.makedirs(args.out_dir, exist_ok=True)
DOM = '' if (args.domain, args.bc) == ('square', 'dirichlet') else f'_{args.domain}{args.bc[0]}'
if args.coarse != 'none': DOM = f'_{args.coarse}{args.bc[0]}{args.coarse_n}'   # old file names unchanged
_ctag = f"{args.contrast:.0e}" if float(f"{args.contrast:.0e}") == args.contrast else f"{args.contrast:.8g}"   # 5 Oct: no rounding clash
cache = os.path.join(args.out_dir,
                     f"pool_oa_L{L}{DOM}_{args.field}_{_ctag}_{args.n}_rho{args.rho:g}_route{args.route}{f'_ds{args.data_seed}' if args.data_seed else ''}{'' if args.load == 'all' else '_load' + args.load}.pkl")


def save_pool():
    # 5 Oct (review E): atomic replacement -- a reader never sees a partly written file.  Concurrent jobs must still use
    # separate --out-dir (kaggle cells): two writers would each replace the other's operator entries.
    tmp = f"{cache}.{os.getpid()}.tmp"
    with open(tmp, 'wb') as fh:
        pickle.dump(pool, fh)
    os.replace(tmp, cache)


POOL_SPEC = dict(level=L, coarse=args.coarse, coarse_n=args.coarse_n, domain=args.domain, bc=args.bc, field=args.field,
                 contrast=args.contrast, n=args.n, data_seed=args.data_seed, load=args.load)
pool = pickle.load(open(cache, 'rb')) if os.path.exists(cache) else {'a': [], 'u': [], 'op': {}}
if pool['a'] and not reusable(pool.get('code'), args.strict_code, f"data pool {os.path.basename(cache)} (coefficients, solutions, coarse solves, operator data)"):
    pool = {'a': [], 'u': [], 'op': {}}                 # 6 Oct (Astra): cached numerical data are versioned like results
if pool.get('spec') not in (None, POOL_SPEC):        # 5 Oct (review D): the cached coefficients must be the ones asked for
    raise SystemExit(f"pool cache {cache} was generated with {pool['spec']}, not {POOL_SPEC}: use another --out-dir")
rng = np.random.default_rng(1000 + args.data_seed)
if not pool['a']:
    t0 = time.time()
    for i in range(args.n):
        a = make_field(H, args.field, args.contrast, rng)
        pool['a'].append(a)
        pool['u'].append(spsolve(H.A_fine(a).tocsc(), load_vec(a, i)))
    print(f"  solved {args.n} samples in {time.time() - t0:.0f}s")
    pool['code'] = CODE_HASH                            # the code version that generated this pool
pool['spec'] = POOL_SPEC
FS = [load_vec(a, i) for i, a in enumerate(pool['a'])]   # per-sample load (all = H.F; random: one load per sample)
A_ref = H.A_fine(np.ones(H.n_elem))
build_time = {}
pool['op'] = {m: v for m, v in pool['op'].items() if v.get('format') == 3}   # drop old-format caches
for mode in modes:
    if mode in pool['op']:
        build_time[mode] = pool['op'][mode]['time']; continue
    t0 = time.time(); ops = []
    Cf = frozen_C(H, A_ref, args.rho, route=args.route) if mode == 'frozen' else None
    for i, a in enumerate(pool['a']):
        ops.append(sample_operator_data(H, a, mode=mode, rho=args.rho, C_frozen=Cf, route=args.route))
        if (i + 1) % 100 == 0: print(f"  transform '{mode}': {i + 1}/{args.n}", flush=True)
    pool['op'][mode] = {'ops': ops, 'time': (time.time() - t0) / args.n, 'format': 3}
    build_time[mode] = pool['op'][mode]['time']
    print(f"  transform '{mode}': {build_time[mode] * 1e3:.0f} ms/sample")
save_pool()

# results file: one JSON line per finished model; reruns skip what is already there
cfg = dict(level=L, field=args.field, contrast=args.contrast, n=args.n, epochs=args.epochs,
           batch=args.batch, lr=args.lr, width=args.width, layers=args.layers, rank=args.rank,
           rank_scale=args.rank_scale, radius=args.radius, rho=args.rho, seed=args.seed,
           route=args.route, loss=args.loss, k0=args.k0)
if DOM: cfg.update(domain=args.domain, bc=args.bc)
if args.coarse != 'none': cfg.update(coarse=args.coarse, coarse_n=args.coarse_n, grid_ng=args.grid_ng)          # square/dirichlet keys stay as before
if args.data_seed: cfg.update(data_seed=args.data_seed)
if args.load != 'all': cfg.update(load=args.load)
if ('oa-ll-op-oa' in args.models or any(m.startswith('oa-vc') for m in args.models)) and args.oa_steps != 4: cfg.update(oa_steps=args.oa_steps)
if args.params: cfg.update(params=args.params)
if args.test_frac != 0.25: cfg.update(test_frac=args.test_frac)        # 5 Oct (review D): the split is part of the run
if args.coarse_rho and args.coarse_rho != args.rho: cfg.update(coarse_rho=args.coarse_rho)
if args.coarse_solve != 'adapted': cfg.update(coarse_solve=args.coarse_solve)
if 'oa-ll' in args.models and (args.ll_scaling, args.ll_output) != ('none', 'latent'):
    cfg.update(ll_scaling=args.ll_scaling, ll_output=args.ll_output)
if 'oa-ll' in args.models and args.ll_ablate != 'none':
    cfg.update(ll_ablate=args.ll_ablate)
cfg_key = json.dumps(cfg, sort_keys=True)
print(f"code version (md5 of the *.py files) {CODE_HASH}; run key {hashlib.md5(cfg_key.encode()).hexdigest()[:10]}")
res_path = os.path.join(args.out_dir, f"results_train_oa_L{L}{DOM}_{args.field}_route{args.route}.jsonl")
done = {}
if os.path.exists(res_path):
    for line in open(res_path):
        rec = json.loads(line)
        if rec.get('cfg') == cfg_key and reusable(rec.get('code'), args.strict_code, f"recorded result of {rec['model']}"):
            done[rec['model']] = rec['result']


def record(name, r):
    if args.eval_partial:                              # partial evaluations are never recorded
        print(f"    [{name}] --eval-partial: result NOT written to {res_path}"); return
    with open(res_path, 'a') as fh:
        fh.write(json.dumps(dict(cfg=cfg_key, model=name, result=r, code=CODE_HASH, n_train=int(n_tr), n_test=int(n_te))) + '\n')
    done[name] = r

U = np.stack(pool['u'])

# ---------------------------------------------------------------------------
# exact coarse solve (scale-orthogonality: the coarse block is independent of all details)
# ---------------------------------------------------------------------------
UC = np.zeros_like(U); COARSE_MS = 0.0
if args.k0 > 0:
    t0 = time.time()
    CR = args.coarse_rho or args.rho
    GAL = args.coarse_solve == 'galerkin'
    ck = (f"coarse_k{args.k0}_galerkin" if GAL else f"coarse_k{args.k0}_oa_rho{CR:g}" if args.coarse_solve == 'oa'
          else f"coarse_k{args.k0}_rho{CR:g}_route{args.route}")
    if ck in pool:
        UC = pool[ck]; COARSE_MS = pool.get(ck + '_ms', float('nan'))
    elif args.coarse_solve == 'oa':
        for i, a in enumerate(pool['a']):
            A_i = H.A_fine(a).tocsr(); Po = oa_prolongation(A_i, args.k0, CR)
            UC[i] = Po @ spsolve((Po.T @ A_i @ Po).tocsc(), Po.T @ FS[i])
        pool[ck] = UC
        save_pool()
    elif GAL:                                             # plain coarse FE solve, prolonged
        Pk = None
        for k in range(args.k0 + 1, L + 1):
            Pk = H.P[k] if Pk is None else H.P[k] @ Pk
        Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
        for i, a in enumerate(pool['a']):
            A_i = H.A_fine(a)
            UC[i] = Pk @ spsolve((PkT @ A_i @ Pk).tocsc(), PkT @ FS[i])
        pool[ck] = UC
        save_pool()
    else:
        for i, a in enumerate(pool['a']):
            T = OATransform(H, H.A_fine(a), 'local', rho=CR, route=args.route, dense_D_max=0)
            g = T.dual(FS[i])                                   # load in the adapted basis
            co = {k: np.zeros(T.size(k)) for k in T.levels}
            for k in range(args.k0 + 1):                      # exact solve on the coarse block only
                if T.size(k):
                    co[k] = T.block_solve(k, g[k])
            UC[i] = T.synth(co)
        pool[ck] = UC
        save_pool()
    if ck + '_ms' not in pool:
        pool[ck + '_ms'] = COARSE_MS = (time.time() - t0) / args.n * 1e3; save_pool()
    err = np.mean(np.linalg.norm(U - UC, axis=1) / np.linalg.norm(U, axis=1))
    print(f"  exact coarse solve to level {args.k0} "
          f"({'Galerkin on the level-%d mesh' % args.k0 if GAL else ('operator-adapted space, Galerkin, rho %g' % CR) if args.coarse_solve == 'oa' else 'adapted, rho %g' % CR}): "
          f"{(time.time() - t0) / args.n * 1e3:.0f} ms/sample, on its own it already gives "
          f"rel-L2 {err:.3e} -- the network learns the rest", flush=True)
n_te = int(round(args.test_frac * args.n)); n_tr = args.n - n_te
tr, te = np.arange(n_tr), np.arange(n_tr, args.n)
TARGET = U - UC                                    # what the network must predict
U_scale = float(np.sqrt(np.mean(TARGET[tr] ** 2)))
U_t = torch.tensor(TARGET / U_scale, dtype=torch.float32, device=dev)
UC_t = torch.tensor(UC / U_scale, dtype=torch.float32, device=dev)   # extra input channel
loga_all = np.stack([nodal_log_a(H, a) for a in pool['a']])
la_mean = torch.tensor(loga_all[tr].mean(), device=dev)
la_std = torch.tensor(loga_all[tr].std() + 1e-8, device=dev)
xy = torch.tensor(H.level_coords(L), dtype=torch.float32, device=dev)


def rel_l2_t(p, t):
    return (torch.linalg.norm(p - t, dim=1) / torch.linalg.norm(t, dim=1)).mean()


def rel_energy_t(p, t, batch):
    # note: p, t are the DETAIL parts when --k0 > 0; the energy norm of the detail error is the
    # energy norm of the total error, because the coarse part is exact (Theorem 3.4).
    """mean over the batch of ||p-t||_A / ||t||_A, with A the sample's own stiffness."""
    e = (p - t).reshape(-1, 1)
    num = batch.energy_sq(e).clamp_min(0).sqrt()
    den = batch.energy_sq(t.reshape(-1, 1)).clamp_min(1e-30).sqrt()
    return (num / den).mean()


def loss_fn(pred, true, batch):
    if args.loss == 'rel_l2' or batch is None or batch.A is None:
        return rel_l2_t(pred, true)
    en = rel_energy_t(pred, true, batch)
    return en if args.loss == 'energy' else 0.5 * (en + rel_l2_t(pred, true))


# 5 Oct (review H): "rel-L2" in every earlier table is the NODAL Euclidean norm.  The FE L2 norm uses the consistent P1
# mass matrix of the finest mesh (restricted to the unknowns; the Dirichlet values are zero).  Both are reported.
_me = H.elem[H.keep]; _ma = _larea[H.keep]
_Mloc = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 12.0
MASS = sp.csr_matrix((np.einsum('e,ij->eij', _ma, _Mloc).ravel(),
                      (np.repeat(_me, 3, axis=1).ravel(), np.tile(_me, (1, 3)).ravel())),
                     shape=(len(H.coords), len(H.coords)))[H.nodes[L]][:, H.nodes[L]].tocsr()


def l2fe_err(i, pred):
    u = U[i]; e = u - pred
    return float(np.sqrt(e @ (MASS @ e) / (u @ (MASS @ u))))


def energy_err(i, pred):
    A = H.A_fine(pool['a'][i]); u = U[i]; e = u - pred
    return float(np.sqrt(e @ (A @ e) / (u @ (A @ u))))


_cert_T, CERT_RHOS = {}, [float(x) for x in args.cert_rhos.split(',')]
_cert_lam = {}                                   # (sample, rho) -> Ritz (lmin, lmax) of B A


def cert_transform(i, rho):
    """Route V localised transform of sample i at patch radius rho, cached across models."""
    if (i, rho) not in _cert_T:
        _cert_T[(i, rho)] = OATransform(H, H.A_fine(pool['a'][i]), 'local', rho=rho,
                                        route='V', dense_D_max=0)
    return _cert_T[(i, rho)]


def pcg_warm(A, b, M, x0, tol=1e-8, maxiter=500):
    """PCG warm-started at x0; returns the iterations needed to reach tol (Algorithm 1 step 4)."""
    x = x0.copy(); r = b - A @ x; nb = np.linalg.norm(b)
    if np.linalg.norm(r) <= tol * nb:
        return 0
    z = M(r); p = z.copy(); rz = r @ z
    for it in range(1, maxiter + 1):
        Ap = A @ p
        al = rz / (p @ Ap)
        x += al * p; r -= al * Ap
        if np.linalg.norm(r) <= tol * nb:
            return it
        z = M(r); rz_new = r @ z
        p = z + (rz_new / rz) * p; rz = rz_new
    return maxiter


def certify(P, tag=''):
    """Certificate effectivity per patch radius, and the cost of correcting to the exact solution."""
    out = {}
    nc = len(te) if args.cert_n <= 0 else min(args.cert_n, len(te))
    te_c, P_c = te[:nc], P[:nc]
    step = max(1, nc // 4)
    for rho in CERT_RHOS:
        eff, warm, cold, lmn, lmx = [], [], [], [], []
        t_r = time.time()
        for j, (i, p) in enumerate(zip(te_c, P_c)):
            if j and j % step == 0:
                print(f"    {tag:<10} certifying rho{rho:g}: {j}/{nc} samples ({time.time() - t_r:.0f}s)",
                      flush=True)
            A = H.A_fine(pool['a'][i]).tocsc()
            T = cert_transform(i, rho)
            r = FS[i] - A @ p
            eta = float(np.sqrt(max(r @ T.precond(r), 0.0)))
            e = U[i] - p
            ea = float(np.sqrt(max(e @ (A @ e), 1e-300)))
            eff.append(eta / ea)
            warm.append(pcg_warm(A, FS[i], T.precond, p.copy()))
            cold.append(pcg_warm(A, FS[i], T.precond, np.zeros_like(p)))
            if (i, rho) not in _cert_lam:          # per-sample spectrum of B A (model-independent, cached)
                _cert_lam[(i, rho)] = ritz_extremes(A, T.precond, n_iter=40)
            lmn.append(_cert_lam[(i, rho)][0]); lmx.append(_cert_lam[(i, rho)][1])
        eff = np.array(eff); lmn = np.array(lmn); lmx = np.array(lmx)
        # Theorem 6.1 per sample: sqrt(lmin_i) <= eff_i <= sqrt(lmax_i), lmin/lmax of B A.  5 Oct (review A): lmn here is the
        # smallest RITZ value of 40 Lanczos steps, which over-estimates lmin, so eta_i / sqrt(lmn_i) is an ESTIMATED upper
        # bound, not a guaranteed one.  A guarantee needs a certified lower bound on lmin (e.g. a Cholesky of
        # T^T A T - sigma D at sigma slightly below the Ritz value with a rounding-error margin, mathcheck/
        # check_certificate_lmin.py), which this driver does not compute.
        ub = eff / np.sqrt(lmn)
        inside = np.mean((eff >= np.sqrt(lmn) * (1 - 1e-6)) & (eff <= np.sqrt(lmx) * (1 + 1e-6)))
        out[f'rho{rho:g}'] = dict(eff_med=float(np.median(eff)), eff_min=float(eff.min()),
                                  eff_max=float(eff.max()), pcg_warm=float(np.mean(warm)),
                                  pcg_cold=float(np.mean(cold)), kappa=float(np.median(lmx / lmn)),
                                  kappa_max=float(np.max(lmx / lmn)), lam_min=float(lmn.min()),
                                  lam_max=float(lmx.max()), ub_med=float(np.median(ub)),
                                  ub_max=float(ub.max()), inside=float(inside))
    return dict(cert=out)


def _ckpt_path(tag):
    return os.path.join(args.out_dir, f"ckpt_{hashlib.md5(cfg_key.encode()).hexdigest()[:10]}_{tag}.pt")


def train(model, forward, tag):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs *
                                              int(np.ceil(n_tr / args.batch)))
    g = np.random.default_rng(args.seed)
    ck = _ckpt_path(tag); start, t_prev, finished = 0, 0.0, None
    if os.path.exists(ck) and not args.rerun and reusable(torch.load(ck, map_location='cpu', weights_only=False).get('code'),
                                                          args.strict_code, f"checkpoint {os.path.basename(ck)}"):
        st = torch.load(ck, map_location=dev, weights_only=False)
        if st.get('cfg') not in (None, cfg_key):        # 5 Oct (review D): never reuse weights of another experiment
            raise SystemExit(f"checkpoint {ck} belongs to another configuration:\n  {st['cfg']}\n  vs {cfg_key}")
        if st.get('finished'):
            finished = st
            print(f"    {tag:<10} training already finished (checkpoint) -- going straight to evaluation", flush=True)
        else:
            model.load_state_dict(st['model']); opt.load_state_dict(st['opt']); sch.load_state_dict(st['sch'])
            g.bit_generator.state = st['rng']; start, t_prev = st['epoch'], st['elapsed']
            print(f"    {tag:<10} resuming from checkpoint at epoch {start}", flush=True)
            if args.eval_partial:
                finished = dict(model=st['model'], train_s=t_prev, partial=True)
                print(f"    {tag:<10} --eval-partial: evaluating the epoch-{start} model of {args.epochs} without training", flush=True)
    if args.eval_partial and finished is None and start == 0:
        raise SystemExit(f'--eval-partial: no unfinished checkpoint {ck}')
    every = args.ckpt_every or max(1, args.epochs // 6)
    t0 = time.time() - t_prev
    for ep in range(start, args.epochs if finished is None else start):
        model.train(); tot = 0.0; nb = 0
        for idx in np.array_split(g.permutation(tr), int(np.ceil(n_tr / args.batch))):
            opt.zero_grad()
            out = forward(model, idx)
            loss = loss_fn(out, U_t[idx], getattr(forward, 'last_batch', None))
            if getattr(model, 'aux_out', None) is not None:          # oa-vcb: + loss of the plain cycle output
                loss = loss + loss_fn(model.aux_out.float(), U_t[idx], getattr(forward, 'last_batch', None))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sch.step(); tot += loss.item(); nb += 1
        if (ep + 1) % max(1, args.epochs // 6) == 0 or ep == 0:
            print(f"    {tag:<10} ep {ep + 1:>4}/{args.epochs}  train {args.loss} {tot / nb:.3e}  "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if (ep + 1) % every == 0 and ep + 1 < args.epochs:
            torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sch=sch.state_dict(),
                            rng=g.bit_generator.state, epoch=ep + 1, elapsed=time.time() - t0,
                            cfg=cfg_key, code=CODE_HASH, n_train=int(n_tr)), ck + '.tmp')
            os.replace(ck + '.tmp', ck)                    # atomic: a disconnect never leaves a broken file
    if finished is not None:
        model.load_state_dict(finished['model'])
        train_s = finished['train_s']
    else:
        train_s = time.time() - t0
    model.eval(); preds = []
    t1 = time.time()
    with torch.no_grad():
        for idx in np.array_split(te, int(np.ceil(n_te / args.batch))):
            preds.append(forward(model, idx).cpu().numpy())
    infer_ms = (time.time() - t1) / n_te * 1e3
    P = np.concatenate(preds) * U_scale + UC[te]   # add the exact coarse part back
    rl2_s = np.linalg.norm(P - U[te], axis=1) / np.linalg.norm(U[te], axis=1)
    en_s = np.array([energy_err(i, p) for i, p in zip(te, P)])
    l2_s = np.array([l2fe_err(i, p) for i, p in zip(te, P)])
    rl2, en = float(rl2_s.mean()), float(en_s.mean())
    # 5 Oct (review B): per-sample check of the energy safeguard in deployment precision -- how many test predictions are
    # worse in energy than the input u_c alone (a guarantee for the Galerkin-output models, informative for the others)
    en_c = np.array([energy_err(i, UC[i]) for i in te])
    n_worse = int(np.sum(en_s > en_c * (1 + 1e-6)))
    # 5 Oct (review, loss-matched 2 x 2 control): models trained on loss(projected) + loss(plain) are also evaluated on
    # their PLAIN cycle output, which separates the projection at inference from the change of the training objective
    # Astra (mathematical-direction review, 5 Oct): plain vs projected output of ONE model isolates the inference-time
    # projection, not the training objective.  Models trained WITHOUT the projected loss (oa-vc, oa-vcz, oa-vcl) are
    # therefore also evaluated WITH the Galerkin output: with oa-vcb / oa-vcq (plain and projected) this completes the
    # 2 x 2 factorial  training objective {plain, plain + projected} x inference {plain, projected}  at fixed deflation,
    # without extra training (up to initialisation noise between the two trained models).
    plain = None
    _pj0 = getattr(model, 'project', None)
    if _pj0 in ('both', False) and hasattr(model, '_galerkin'):
        model.project = (False if _pj0 == 'both' else True); pp = []
        with torch.no_grad():
            for idx in np.array_split(te, int(np.ceil(n_te / args.batch))):
                pp.append(forward(model, idx).cpu().numpy())
        model.project = _pj0
        Pp = np.concatenate(pp) * U_scale + UC[te]
        _k = 'plain' if _pj0 == 'both' else 'proj'
        plain = {f'rel_l2_{_k}': float(np.mean(np.linalg.norm(Pp - U[te], axis=1) / np.linalg.norm(U[te], axis=1))),
                 f'energy_{_k}': float(np.mean([energy_err(i, p) for i, p in zip(te, Pp)]))}
    # energy line search along the network's correction c = P - u_c (4 Oct): alpha = c^T r / c^T A c with r = F - A u_c,
    # the Galerkin projection of the error onto span{c}.  Computable without u, one mat-vec; the energy error can never
    # exceed that of u_c alone (alpha = 0 is admissible), and alpha = 1 recovers the plain prediction.
    PL, alph = [], []
    for i, p in zip(te, P):
        A_ = H.A_fine(pool['a'][i]).tocsr(); c_ = p - UC[i]; Ac = A_ @ c_; den = float(c_ @ Ac)
        al = float(c_ @ (FS[i] - A_ @ UC[i])) / den if den > 0 else 0.0
        PL.append(UC[i] + al * c_); alph.append(al)
    PL = np.array(PL)
    rl2_ls = np.linalg.norm(PL - U[te], axis=1) / np.linalg.norm(U[te], axis=1)
    en_ls = np.array([energy_err(i, p) for i, p in zip(te, PL)])
    out = dict(rel_l2=rl2, energy=en, train_s=train_s, infer_ms=infer_ms,
               l2fe=float(l2_s.mean()), l2fe_s=[float(v) for v in l2_s], n_worse_than_uc=n_worse, coarse_ms=COARSE_MS,
               rl2_s=[float(v) for v in rl2_s], en_s=[float(v) for v in en_s],     # per test sample (4 Oct)
               rel_l2_ls=float(rl2_ls.mean()), energy_ls=float(en_ls.mean()),
               rl2_ls_s=[float(v) for v in rl2_ls], en_ls_s=[float(v) for v in en_ls], alpha_s=[float(v) for v in alph])
    if plain is not None: out.update(plain)
    if finished is not None and finished.get('partial'): out['partial_epoch'] = start
    if finished is None:                               # training is safe from here on
        torch.save(dict(finished=True, model=model.state_dict(), train_s=train_s, cfg=cfg_key, code=CODE_HASH,
                        n_train=int(n_tr)), ck + '.tmp')
        os.replace(ck + '.tmp', ck)
    print(f"    {tag:<10} TEST  nodal rel-l2 {rl2:.3e}  FE rel-L2 {out['l2fe']:.3e}  energy {en:.3e}  (train {train_s:.0f}s; model saved)"
          f"\n    {'':<10} energy worse than u_c alone on {n_worse}/{n_te} test samples"
          + ((f"; plain cycle output (no projection): nodal rel-l2 {plain['rel_l2_plain']:.3e}  energy {plain['energy_plain']:.3e}"
              if 'energy_plain' in plain else
              f"; with Galerkin output at inference (not trained for it): nodal rel-l2 {plain['rel_l2_proj']:.3e}  energy {plain['energy_proj']:.3e}")
             if plain is not None else '') +
          f"\n    {'':<10} + energy line search: rel-L2 {out['rel_l2_ls']:.3e}  energy {out['energy_ls']:.3e}  "
          f"(alpha median {np.median(alph):.3f}, range {np.min(alph):.2f}-{np.max(alph):.2f})"
          + (f" -- certifying {args.cert_n if args.cert_n > 0 else n_te} samples x rho {args.cert_rhos}"
             if args.certify else ''), flush=True)
    if args.certify:
        t2 = time.time()
        out.update(certify(P, tag))
        out['cert_ms'] = (time.time() - t2) / (args.cert_n if 0 < args.cert_n < n_te else n_te) * 1e3
        for name, c in out['cert'].items():
            print(f"    {tag:<10} certificate {name:<6} effectivity {c['eff_med']:.3f} "
                  f"[{c['eff_min']:.3f}, {c['eff_max']:.3f}]  per-sample sqrt(lam) in "
                  f"[{c['lam_min']**0.5:.3f}, {c['lam_max']**0.5:.3f}] (inside {100 * c['inside']:.0f}%)"
                  f"  | estimated bound eta/sqrt(Ritz lmin), not certified: effectivity {c['ub_med']:.2f} (max {c['ub_max']:.2f})"
                  f"  | PCG {c['pcg_warm']:.1f}/{c['pcg_cold']:.1f}", flush=True)
    return out


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------
G = SharedGeometry(H, radius=args.radius, device=dev)


def oa_forward_factory(mode):
    ops = pool['op'][mode]['ops']
    def fwd(model, idx):
        batch = OABatch(G, [ops[i] for i in idx], device=dev)
        fwd.last_batch = batch
        extra = UC_t[idx][:, :, None] if args.k0 else None
        return model(batch, xy, la_mean, la_std, extra)
    fwd.last_batch = None
    return fwd


results = {}
oa_params = args.params or n_params(OALiftingNO(width=args.width, n_layers=args.layers, rank=args.rank,
                                 n_levels=L, rank_scale=args.rank_scale,
                                 extra_in=1 if args.k0 else 0))
for name in [m for m in args.models if m.startswith('oa') and not m.startswith('oa-ll') and not m.startswith('oa-vc')]:
    if name in done and not args.rerun and not args.reeval:
        print(f"\n[{name}] already in {res_path} -- skipped (use --rerun to retrain)")
        results[name] = done[name]; continue
    rank = 0 if name == 'oa-norank' else args.rank
    torch.manual_seed(args.seed)
    model = OALiftingNO(width=args.width, n_layers=args.layers, rank=rank, n_levels=L,
                        rank_scale=args.rank_scale, extra_in=1 if args.k0 else 0).to(dev)
    np_ = n_params(model)
    print(f"\n[{name}] params {np_:,}   transform build {build_time[need[name]] * 1e3:.0f} ms/sample")
    r = train(model, oa_forward_factory(need[name]), name)
    r['params'] = np_; r['build_ms'] = build_time[need[name]] * 1e3
    results[name] = r; record(name, r)

def op_for_energy():
    """the energy loss only needs each sample's stiffness A, which every cached op set carries; use
    whichever is in the pool, and build the cheap 'hb' set if none is (baseline-only runs)"""
    op = modes[0] if modes else next(iter(pool['op']), None)
    if args.loss != 'rel_l2' and op is None:
        t0 = time.time()
        pool['op']['hb'] = {'ops': [sample_operator_data(H, a, mode='hb', rho=args.rho, route=args.route)
                                    for a in pool['a']], 'time': 0.0, 'format': 3}
        save_pool(); op = 'hb'
        print(f"  operator data for the energy loss built in {time.time() - t0:.0f}s", flush=True)
    return op


# kinds that take the error-equation inputs (residual D^-1 r / sigma instead of u_c, output x sigma); the oa-ll ones
# also take the operator weights a_ij / a_ii in their smoother.  fno-op / wno-op / lino-op / transolver-op (3 Oct, second
# review): the standard neural operators given the SAME residual input, so the input effect is separated from the model.
OP_W_KINDS = ('oa-ll-op', 'oa-ll-opl', 'oa-ll-op-lin', 'oa-ll-op-nomlp', 'oa-ll-op-oa')
OP_RES_KINDS = OP_W_KINDS + ('oa-ll-epz',)
if any(m.endswith('-op') and not m.startswith('oa-ll') and m != 'gnn-op' for m in args.models) or 'gnn-op' in args.models:
    _Z = np.array([(FS[i] - H.A_fine(pool['a'][i]).tocsr() @ UC[i]) / H.A_fine(pool['a'][i]).diagonal() for i in range(args.n)])
    _sig = np.sqrt(np.mean(_Z ** 2, 1)) + 1e-30
    ZN = _Z / _sig[:, None]                                         # (n, nI) normalised D^-1 r, H.nodes[L] order
    SIG_ALL = torch.tensor(_sig / _sig[tr].mean(), dtype=torch.float32, device=dev)
grid_models = [m for m in args.models if m in ('fno', 'wno', 'lino', 'wno-off', 'fno-op', 'wno-op', 'lino-op',
                                               'mgno', 'mgno-op', 'mgno-off')]
if grid_models:
    from ofwno_nets import RealFNO2D, RealWNO2D, build_grid_bijection
    n_all = len(H.coords)
    MESH_GRID = args.coarse != 'none'
    if MESH_GRID:                       # unstructured mesh: interpolate mesh -> grid -> mesh
        from mesh_grid_bridge import MeshGridBridge
        BR = MeshGridBridge(H, ng=(args.grid_ng or None)); ng = BR.ng
        print('  baselines: ' + BR.report())
        gxy = (BR.gxy - BR.gxy.min((0, 1))) / (BR.gxy.max((0, 1)) - BR.gxy.min((0, 1)))
        dom_g = BR.mask
        g2n = None
    else:
        g2n, n2g, ng, gxy = build_grid_bijection(H.mesh, L)
    if not MESH_GRID:
        fine_pos = {g: i for i, g in enumerate(H.nodes[L])}
        int_mask = np.array([g in fine_pos for g in g2n.ravel()]).reshape(ng, ng)
        gi, gj = np.nonzero(int_mask)
        int_order = np.array([fine_pos[g] for g in g2n[gi, gj]])
    # full-node log a (boundary nodes included) on the grid
    def full_loga(a):
        s = np.zeros(n_all); c = np.zeros(n_all); kp = H.keep; la = np.log(np.asarray(a)[kp])
        for j in range(3):
            np.add.at(s, H.elem[kp, j], la); np.add.at(c, H.elem[kp, j], 1)
        return np.where(c > 0, s / np.maximum(c, 1), 0.0)
    LAn = np.stack([full_loga(a) for a in pool['a']])
    if MESH_GRID:
        LA = BR.node_field_to_grid(LAn)                                  # (n, ng, ng), 0 outside
    else:
        LA = LAn[:, g2n]
        dom_g = H.in_domain[g2n]                                         # (ng, ng) node-in-domain mask
    mu_, sd_ = LA[tr][:, dom_g].mean(), LA[tr][:, dom_g].std() + 1e-8
    LA = np.where(dom_g, (LA - mu_) / sd_, 0.0)                          # zero outside the domain
    chans = [LA[..., None], np.broadcast_to(gxy, LA.shape + (2,))]
    if args.domain != 'square' or MESH_GRID:       # masked-grid input: the baselines see the geometry
        chans.append(np.broadcast_to(dom_g.astype(float)[None, :, :, None], LA.shape + (1,)))
    if args.bc != 'dirichlet':                     # ...and where the Neumann edges are
        nm = np.zeros(n_all); nm[H.neumann_nodes] = 1.0
        nmg = BR.node_field_to_grid(nm) if MESH_GRID else nm[g2n]
        chans.append(np.broadcast_to(nmg[None, :, :, None], LA.shape + (1,)))
    if args.k0:                                    # the baselines get the exact coarse solve too
        if MESH_GRID:
            ucg = BR.interior_to_grid(UC / U_scale)
        else:
            ucg = np.zeros((len(UC), ng, ng)); ucg[:, gi, gj] = (UC / U_scale)[:, int_order]
        chans.append(ucg[..., None])
    X = np.concatenate(chans, -1)
    X_t = torch.tensor(X, dtype=torch.float32, device=dev)
    if any(k.endswith('-op') for k in grid_models):                # residual channel in place of u_c (k0 0: appended)
        zg = BR.interior_to_grid(ZN) if MESH_GRID else np.zeros((len(ZN), ng, ng))
        if not MESH_GRID: zg[:, gi, gj] = ZN[:, int_order]
        X_z = torch.tensor(np.concatenate((chans[:-1] if args.k0 else chans) + [zg[..., None]], -1),
                           dtype=torch.float32, device=dev)
    if MESH_GRID:                       # sparse bilinear grid -> mesh-node map (fixed, differentiable)
        co = BR.to_nodes.tocoo()
        TN = torch.sparse_coo_tensor(np.vstack([co.row, co.col]), co.data.astype('float32'),
                                     co.shape, device=dev).coalesce()
    else:
        gi_t = torch.tensor(gi, device=dev); gj_t = torch.tensor(gj, device=dev)
        inv = np.empty_like(int_order); inv[int_order] = np.arange(len(int_order))
        inv_t = torch.tensor(inv, device=dev)

    OP_FOR_A = op_for_energy()

    def grid_fwd(model, idx):
        if grid_fwd.op:                             # error-equation inputs, output x sigma (positively homogeneous)
            out = model(X_z[idx]) * SIG_ALL[idx][:, None, None]
        else:
            out = model(X_t[idx])                   # (B, ng, ng)
        if args.loss != 'rel_l2':                   # baselines get the same objective
            grid_fwd.last_batch = OABatch(G, [pool['op'][OP_FOR_A]['ops'][i] for i in idx], device=dev)
        if MESH_GRID:                               # bilinear interpolation back to the mesh nodes
            return torch.sparse.mm(TN, out.reshape(out.shape[0], -1).t()).t()
        return out[:, gi_t, gj_t][:, inv_t]         # interior nodes, oa ordering
    grid_fwd.last_batch = None; grid_fwd.op = False

    def build(kind, h):
        cin = X.shape[-1]; kind = kind.replace('-op', '')
        if kind == 'lino':                          # Pandey et al. 2026, re-implemented (baselines_extra.py)
            from baselines_extra import LiNO2D
            return LiNO2D(cin, hidden=h, levels=3)
        if kind in ('mgno', 'mgno-off'):           # MgNO (He, Liu, Xu, ICLR 2024), official model code (baselines_extra.py)
            from baselines_extra import MgNO2D
            return MgNO2D(cin, hidden=24 if kind == 'mgno-off' else h)   # mgno-off: their published Darcy size
        if kind == 'wno-off':                       # the published WNO (Tripura & Chakraborty), exact port
            from baselines_extra import WNO2dOfficial
            return WNO2dOfficial(cin, width=h, size=(ng, ng))
        return RealFNO2D(cin, hidden=h, modes=min(12, ng // 2), n_layers=4) if kind == 'fno' \
            else RealWNO2D(cin, hidden=h, n_layers=4, J=2)

    for kind in grid_models:
        if kind in done and not args.rerun and not args.reeval:
            print(f"\n[{kind}] already in {res_path} -- skipped"); results[kind] = done[kind]; continue
        target = oa_params if kind != 'mgno-off' else float('inf')        # mgno-off: published size, no budget
        lo, hi = 2, 256
        while hi - lo > 1:
            mid = (lo + hi) // 2
            try: ok = n_params(build(kind, mid)) <= target
            except Exception: ok = False
            if ok: lo = mid
            else: hi = mid
        if kind == 'mgno-off': lo = 24                                     # published width (build ignores h)
        torch.manual_seed(args.seed)
        model = build(kind, lo).to(dev)
        if n_params(model) > target:
            print(f"\n[{kind}] cannot be built within {target:,} parameters -- skipped (the published WNO needs ~75k)"); continue
        print(f"\n[{kind}] hidden {lo}, params {n_params(model):,} (target {target:,})"
              + ('  -- error-equation inputs (D^-1 r / sigma instead of u_c, output x sigma)' if kind.endswith('-op') else ''))
        grid_fwd.op = kind.endswith('-op')
        r = train(model, grid_fwd, kind)
        r['params'] = n_params(model); r['build_ms'] = 0.0
        results[kind] = r; record(kind, r)

# ---------------------------------------------------------------------------------------------------
# mesh-native baseline: Transolver on the mesh nodes (no grid, no interpolation)
# ---------------------------------------------------------------------------------------------------
mesh_models = [m for m in args.models if m in ('transolver', 'transolver-op', 'gnn', 'gnn30', 'gnn-op', 'oa-vc', 'oa-vcp', 'oa-vcd', 'oa-vcb', 'oa-vcz', 'oa-vcq', 'oa-vcl', 'oa-vcbl') or m.startswith('oa-ll')]
if mesh_models:
    n_all = len(H.coords)
    dom_nodes = np.nonzero(H.in_domain)[0]                 # every node of the domain, boundary included
    pos_in = {g: i for i, g in enumerate(dom_nodes)}
    take = torch.tensor([pos_in[g] for g in H.nodes[L]], device=dev)   # interior unknowns, oa order
    kp = H.keep
    def nodal_loga_all(a):
        s_ = np.zeros(n_all); c_ = np.zeros(n_all); la = np.log(np.asarray(a)[kp])
        for j in range(3):
            np.add.at(s_, H.elem[kp, j], la); np.add.at(c_, H.elem[kp, j], 1)
        return np.where(c_ > 0, s_ / np.maximum(c_, 1), 0.0)
    LAm = np.stack([nodal_loga_all(a) for a in pool['a']])[:, dom_nodes]
    LAm = (LAm - LAm[tr].mean()) / (LAm[tr].std() + 1e-8)
    xy_ = H.coords[dom_nodes]; xy_ = (xy_ - xy_.min(0)) / (xy_.max(0) - xy_.min(0))
    bflag = np.zeros(n_all); bflag[H.boundary] = 1.0
    if len(getattr(H, 'neumann_nodes', [])): bflag[H.neumann_nodes] = -1.0
    chans = [LAm[..., None], np.broadcast_to(xy_, LAm.shape + (2,)),
             np.broadcast_to(bflag[dom_nodes][None, :, None], LAm.shape + (1,))]
    if args.k0:
        ucm = np.zeros((len(UC), n_all)); ucm[:, H.nodes[L]] = UC / U_scale
        chans.append(ucm[:, dom_nodes][..., None])
    XM = torch.tensor(np.concatenate(chans, -1), dtype=torch.float32, device=dev)
    XI = XM[:, take].contiguous()                            # interior unknowns only (oa-ll)
    OPA = op_for_energy()
    OPH = None
    if any(m.startswith('oa-ll') or m.startswith('oa-vc') for m in mesh_models):     # oa-ll needs the (cheap) hb operator data
        if 'hb' not in pool['op']:
            t0 = time.time()
            pool['op']['hb'] = {'ops': [sample_operator_data(H, a, mode='hb', rho=args.rho, route=args.route)
                                        for a in pool['a']], 'time': 0.0, 'format': 3}
            save_pool()
            print(f"  hb operator data built in {time.time() - t0:.0f}s", flush=True)
        OPH = 'hb'
    def mesh_fwd_factory(kind):
        def mesh_fwd(model, idx):
            if kind == 'gnn-op':
                model.set_op(OPWG[idx])
                out = model(XMZ[idx])[:, take] * SIG[idx][:, None]
                if args.loss != 'rel_l2':
                    mesh_fwd.last_batch = OABatch(G, [pool['op'][OPA]['ops'][i] for i in idx], device=dev)
                return out
            if kind.startswith('oa-vc'):                    # learned V-cycle: residual in, correction out (target units)
                model.set_ops(*OAD.batch(idx), Z=OAD.batch_z(idx), A64=OAD.batch_a64(idx))
                out = model(RV[idx]).float()           # float64 residual in (exact safeguard), float32 for the loss
                if args.loss != 'rel_l2':
                    mesh_fwd.last_batch = OABatch(G, [pool['op'][OPA]['ops'][i] for i in idx], device=dev)
                return out
            if kind == 'transolver-op':
                out = model(XMZ_T[idx])[:, take] * SIG_ALL[idx][:, None]
                if args.loss != 'rel_l2':
                    mesh_fwd.last_batch = OABatch(G, [pool['op'][OPA]['ops'][i] for i in idx], device=dev)
                return out
            if kind in ('transolver', 'gnn', 'gnn30'):
                out = model(XM[idx])[:, take]
                if args.loss != 'rel_l2':
                    mesh_fwd.last_batch = OABatch(G, [pool['op'][OPA]['ops'][i] for i in idx], device=dev)
                return out
            bt = OABatch(G, [pool['op'][OPH]['ops'][i] for i in idx], device=dev)   # hb data: A + scaling
            scal = {k: v.reshape(len(idx), -1) for k, v in bt.d.items() if k > 0}
            mesh_fwd.last_batch = bt if args.loss != 'rel_l2' else None
            if kind in OP_RES_KINDS:   # residual input; output positively homogeneous in the residual's scale
                if kind in OP_W_KINDS: model.set_op(OPW[idx])     # + operator weights in the smoother
                if kind == 'oa-ll-op-oa': model.set_oa([v[idx] for v in OAV], OAPATS)   # + operator-adapted predict
                return model(XZ[idx], scal) * SIG[idx][:, None]
            return model(XI[idx], scal)
        mesh_fwd.last_batch = None
        return mesh_fwd
    for kind in mesh_models:
        if kind in done and not args.rerun and not args.reeval:
            print(f"\n[{kind}] already in {res_path} -- skipped"); results[kind] = done[kind]; continue
        if kind in ('transolver', 'transolver-op'):
            from baselines_extra import Transolver
            if kind == 'transolver-op':                 # the same residual input as oa-ll-op / gnn-op
                zm_ = np.zeros((args.n, n_all)); zm_[:, H.nodes[L]] = ZN
                zt_ = torch.tensor(zm_[:, dom_nodes], dtype=torch.float32, device=dev)[..., None]
                XMZ_T = torch.cat([XM[..., :-1], zt_], -1) if args.k0 else torch.cat([XM, zt_], -1)
            mk = lambda w: Transolver(XM.shape[-1] + (0 if (kind == 'transolver' or args.k0) else 1), hidden=w)
            note = f"{len(dom_nodes)} mesh nodes, native (no grid)" + (', error-equation inputs' if kind == 'transolver-op' else '')
        elif kind == 'gnn-op':
            # the GNN with EXACTLY the inputs of oa-ll-op: residual D^-1 r instead of u_c, operator weights
            # a_ij / a_ii as a per-sample edge feature (0 on edges touching a boundary node), output x sigma
            from baselines_extra import MeshGNN, mesh_edges
            _ed, _ef = mesh_edges(H.coords, H.elem[H.keep], dom_nodes)
            posL = -np.ones(len(H.coords), dtype=np.int64); posL[H.nodes[L]] = np.arange(len(H.nodes[L]))
            gs, gd = posL[dom_nodes[_ed[0]]], posL[dom_nodes[_ed[1]]]; ok_ = (gs >= 0) & (gd >= 0)
            Zs, Ws = [], []
            for i in range(args.n):
                A_i = H.A_fine(pool['a'][i]).tocsr(); dg = A_i.diagonal()
                Zs.append((FS[i] - A_i @ UC[i]) / dg)
                wv = np.zeros(len(gs)); wv[ok_] = np.asarray(A_i[np.array(gd[ok_]), np.array(gs[ok_])]).ravel() / dg[gd[ok_]]
                Ws.append(wv)
            Zs = np.array(Zs); sig = np.sqrt(np.mean(Zs ** 2, 1)) + 1e-30
            SIG = torch.tensor(sig / sig[tr].mean(), dtype=torch.float32, device=dev)
            zm = np.zeros((args.n, n_all)); zm[:, H.nodes[L]] = Zs / sig[:, None]
            zt = torch.tensor(zm[:, dom_nodes], dtype=torch.float32, device=dev)[..., None]
            XMZ = torch.cat([XM[..., :-1], zt], -1) if args.k0 else torch.cat([XM, zt], -1)   # k0 0: u_c = 0, r = F
            OPWG = torch.tensor(np.array(Ws), dtype=torch.float32, device=dev)
            mk = lambda w: MeshGNN(XMZ.shape[-1], _ed, _ef, width=w, steps=10, op_edge=True)
            note = f"MeshGraphNet-style GNN, 10 steps, + residual input and operator edge weights (= oa-ll-op inputs)"
        elif kind in ('oa-vc', 'oa-vcp', 'oa-vcd', 'oa-vcb', 'oa-vcz', 'oa-vcq', 'oa-vcl', 'oa-vcbl'):
            # one learned V-cycle: operator-adapted transfers (route V, --oa-steps Jacobi steps of prolongator smoothing),
            # Galerkin level operators, exact coarsest solve; learned operator-consistent pre/post smoothers (oa_vcycle.py)
            from oa_vcycle import OAData, OAVCycle
            t0 = time.time(); VC_LEVELS = 0; k_ = L
            while k_ >= 1 and VC_LEVELS < 4 and len(H.nodes[k_ - 1]) >= 16:
                VC_LEVELS += 1; k_ -= 1
            OAD = OAData(H, pool['a'], VC_LEVELS, args.oa_steps, dev, H.level_coords, near_kernel=(kind not in ('oa-vc', 'oa-vcq')))
            VC_BUILD_MS = (time.time() - t0) / args.n * 1e3      # hierarchy + Galerkin levels + near-kernel, per sample (review F)
            RV = torch.tensor(np.array([FS[i] - H.A_fine(pool['a'][i]).tocsr() @ UC[i] for i in range(args.n)]) / U_scale,
                              dtype=torch.float64, device=dev)       # float64 since 5 Oct (review point 3)
            print(f"  V-cycle operator data ({VC_LEVELS} levels, {args.oa_steps} smoothing steps of the predict, coarsest "
                  f"{OAD.n0} nodes) built in {time.time() - t0:.0f}s ({(time.time() - t0) / args.n * 1e3:.0f} ms/sample)"
                  + (f"; near-kernel vectors per sample: max {max(OAD.k_nk)}, mean {np.mean(OAD.k_nk):.2f}, "
                     f"{np.mean(np.array(OAD.k_nk) > 0) * 100:.0f}% of samples have one" if OAD.Z is not None else ''), flush=True)
            # ablation matrix (5 Oct, review): oa-vcz = deflation only (plain cycle output), oa-vcq = Galerkin output and
            # two-term loss without deflation; with oa-vc (neither) and oa-vcb (both) the 2 x 2 table is complete
            # residual-linear controls (5 Oct, review point 5): oa-vcl = oa-vcz with LINEAR smoothers (the whole map r -> c is
            # then linear in r for each A); oa-vcbl = oa-vcb with linear smoothers (only the Galerkin output is nonlinear)
            _pj = {'oa-vc': False, 'oa-vcp': True, 'oa-vcd': 'eval', 'oa-vcb': 'both', 'oa-vcz': False, 'oa-vcq': 'both',
                   'oa-vcl': False, 'oa-vcbl': 'both'}[kind]
            _df = kind in ('oa-vcp', 'oa-vcd', 'oa-vcb', 'oa-vcz', 'oa-vcl', 'oa-vcbl')
            _lin = kind in ('oa-vcl', 'oa-vcbl')
            mk = lambda w: OAVCycle(OAD, width=w, nu=4, project=_pj, deflate=_df, ckpt=bool(args.vc_ckpt), linear=_lin)
            note = (f"learned V-cycle, {VC_LEVELS} levels, operator-adapted transfers, Galerkin levels, exact coarsest solve, "
                    f"operator-consistent smoothers (nu 4, shared)"
                    + {'oa-vc': '', 'oa-vcp': ', near-kernel deflation + Galerkin output projection',
                       'oa-vcd': ', near-kernel deflation; Galerkin output projection at evaluation',
                       'oa-vcb': ', near-kernel deflation + Galerkin output projection; loss on projected + plain output',
                       'oa-vcz': ', near-kernel deflation only (plain cycle output)',
                       'oa-vcq': ', Galerkin output projection, loss on projected + plain output, no deflation',
                       'oa-vcl': ', near-kernel deflation, RESIDUAL-LINEAR smoothers (map linear in r)',
                       'oa-vcbl': ', near-kernel deflation + Galerkin output, RESIDUAL-LINEAR smoothers'}[kind])
        elif kind in ('gnn', 'gnn30'):
            from baselines_extra import MeshGNN, mesh_edges
            _ed, _ef = mesh_edges(H.coords, H.elem[H.keep], dom_nodes)
            _steps = 30 if kind == 'gnn30' else 10
            mk = lambda w: MeshGNN(XM.shape[-1], _ed, _ef, width=w, steps=_steps)
            note = f"MeshGraphNet-style GNN, {_steps} message-passing steps, on {len(dom_nodes)} mesh nodes"
        else:
            from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant
            MG = (MeshHierarchyGeometry(H, dev, max_levels=max(1, L - args.k0)) if kind == 'oa-ll-opl' and args.k0
                  else MeshHierarchyGeometry(H, dev))   # opl: lifting only on levels k0+1..L, complementary to u_c
            if kind in OP_RES_KINDS:
                # error equation A e = r, r = F - A u_c: the exact information the correction depends on.
                # inputs (log a, xy, boundary flag, D^-1 r / sigma), u_c itself is NOT an input (e depends on a, r only);
                # per-sample sigma = rms(D^-1 r) and the output is multiplied back -> exact linearity in the scale of r
                t0 = time.time(); src_, dst_ = np.array(MG.fine['src'].cpu().numpy()), np.array(MG.fine['dst'].cpu().numpy())
                Zs, Ws = [], []
                for i in range(args.n):
                    A_i = H.A_fine(pool['a'][i]).tocsr(); dg = A_i.diagonal()
                    Zs.append((FS[i] - A_i @ UC[i]) / dg)
                    Ws.append(np.asarray(A_i[dst_, src_]).ravel() / dg[dst_])
                Zs = np.array(Zs); sig = np.sqrt(np.mean(Zs ** 2, 1)) + 1e-30
                SIG = torch.tensor(sig / sig[tr].mean(), dtype=torch.float32, device=dev)
                zt = torch.tensor(Zs / sig[:, None], dtype=torch.float32, device=dev)[..., None]
                XZ = torch.cat([XI[..., :-1], zt], -1) if args.k0 else torch.cat([XI, zt], -1)   # k0 0: u_c = 0, r = F
                OPW = torch.tensor(np.array(Ws), dtype=torch.float32, device=dev)
                print(f"  operator weights a_ij/a_ii ({OPW.shape[1]} edges) + residual input built in {time.time() - t0:.0f}s; "
                      f"rel. spread of |D^-1 r| over samples {sig.std() / sig.mean():.2f}", flush=True)
            if kind == 'oa-ll-op-oa':
                # per sample and lifting level k: Phat_k = m Jacobi steps (omega 2/3) on A_FF X = -A_FC from the geometric
                # predict, A_{k-1} = Phi_k^T A_k Phi_k (route V).  Values on a FIXED structural pattern (boolean products
                # of the sparsity patterns, so no sample can have an entry outside it).
                t0 = time.time(); OAPATS, OAV, pat_keys = [], [], []
                one = lambda M: (abs(M) > 0).astype(np.float64)
                As = one(H.A_fine(np.random.default_rng(12345).uniform(1, 2, H.n_elem)).tocsr())   # no special-a cancellations
                for Lv in MG.levels:
                    k, nc = Lv['k'], Lv['nc']
                    Xs = one(H.P[k][nc:].tocsr()); BFF, BFC = As[nc:, nc:].tocsr(), As[nc:, :nc].tocsr()
                    for _ in range(args.oa_steps):
                        Xs = one((Xs + BFC + BFF @ Xs).tocsr())
                    Xs = Xs.tocoo(); kk = Xs.row.astype(np.int64) * nc + Xs.col; o = np.argsort(kk)
                    pat_keys.append((kk[o], nc)); OAPATS.append((torch.tensor(Xs.row[o], dtype=torch.long, device=dev),
                                                                  torch.tensor(Xs.col[o], dtype=torch.long, device=dev)))
                    Phs = sp.vstack([sp.identity(nc, format='csr'), Xs.tocsr()]).tocsr()
                    As = one((Phs.T @ As @ Phs).tocsr())
                vals = [np.zeros((args.n, len(pk[0]))) for pk in pat_keys]
                for i in range(args.n):
                    Ak = H.A_fine(pool['a'][i]).tocsr()
                    for l, Lv in enumerate(MG.levels):
                        k, nc = Lv['k'], Lv['nc']
                        A_FF, A_FC = Ak[nc:, nc:].tocsr(), Ak[nc:, :nc].tocsr(); Dinv = sp.diags(1 / A_FF.diagonal())
                        X = H.P[k][nc:].tocsr()
                        for _ in range(args.oa_steps):
                            X = (X + (2 / 3) * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
                        Xc_ = X.tocoo(); nz_ = Xc_.data != 0
                        Xc_ = sp.coo_matrix((Xc_.data[nz_], (Xc_.row[nz_], Xc_.col[nz_])), Xc_.shape)
                        kk = Xc_.row.astype(np.int64) * nc + Xc_.col
                        pos_ = np.searchsorted(pat_keys[l][0], kk)
                        assert np.all(pat_keys[l][0][np.minimum(pos_, len(pat_keys[l][0]) - 1)] == kk), 'entry outside pattern'
                        vals[l][i, pos_] = Xc_.data
                        Phi = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr(); Ak = (Phi.T @ Ak @ Phi).tocsr()
                OAV = [torch.tensor(v, dtype=torch.float32, device=dev) for v in vals]
                print(f"  operator-adapted predicts ({args.oa_steps} Jacobi steps, {len(MG.levels)} levels, nnz "
                      f"{', '.join(str(v.shape[1]) for v in vals)}) built in {time.time() - t0:.0f}s "
                      f"({(time.time() - t0) / args.n * 1e3:.0f} ms/sample)", flush=True)
            _kw = dict(scaling=args.ll_scaling, output=args.ll_output, ablate=args.ll_ablate); _kw.update(variant(kind))
            mk = lambda w: MeshLiftingNO(MG, (XZ if kind in OP_RES_KINDS else XI).shape[-1], width=w, **_kw)
            note = "learnable lifting on the mesh hierarchy: " + MG.describe()
        lo, hi = 4, 256
        while hi - lo > 1:
            mid = (lo + hi) // 2
            try: ok = n_params(mk(mid)) <= oa_params
            except Exception: ok = False
            if ok: lo = mid
            else: hi = mid
        torch.manual_seed(args.seed)
        model = mk(lo).to(dev)
        print(f"\n[{kind}] width {max(4, (lo // 4) * 4)}, params {n_params(model):,} (target {oa_params:,}), {note}")
        r = train(model, mesh_fwd_factory(kind), kind)
        r['params'] = n_params(model); r['build_ms'] = VC_BUILD_MS if kind.startswith('oa-vc') else 0.0
        results[kind] = r; record(kind, r)

DOMTXT = f"coarse mesh {args.coarse} (n={args.coarse_n})" if args.coarse != 'none' else f"domain {args.domain}"
print(f"\n\nRESULTS  route {args.route}, level {L}, {DOMTXT}, bc {args.bc}, {args.field}, contrast {args.contrast:.0e}, "
      f"{n_tr} train / {n_te} test, {args.epochs} epochs, rank {args.rank}, loss {args.loss}"
      f"{', exact coarse solve to level %d (%s)' % (args.k0, 'galerkin' if args.coarse_solve == 'galerkin' else ('oa rho %g' if args.coarse_solve == 'oa' else 'adapted rho %g') % (args.coarse_rho or args.rho)) if args.k0 else ''}"
      f"{' x sqrt(m_k) scaling %g' % args.rank_scale if args.rank_scale else ''}")
order = ['oa', 'oa-hb', 'oa-ll', 'oa-ll-s', 'oa-ll-e', 'oa-ll-e4', 'oa-ll-se', 'oa-ll-ep', 'oa-ll-p', 'oa-ll-p8', 'oa-ll-ep8', 'oa-ll-sep', 'oa-ll-epr', 'oa-ll-epm', 'oa-ll-op', 'oa-ll-opl', 'oa-ll-epz', 'oa-ll-op-lin', 'oa-ll-op-nomlp', 'oa-ll-op-oa', 'oa-vc', 'oa-vcp', 'oa-vcd', 'oa-vcb', 'oa-vcz', 'oa-vcq', 'oa-vcl', 'oa-vcbl', 'oa-frozen', 'oa-norank', 'fno', 'wno', 'wno-off', 'lino', 'fno-op', 'wno-op', 'lino-op', 'mgno', 'mgno-op', 'mgno-off', 'transolver', 'transolver-op', 'gnn', 'gnn30', 'gnn-op']
for k, r in done.items():                       # models finished in an earlier run, same config
    results.setdefault(k, r)
results = {k: results[k] for k in order if k in results}
cert = args.certify and any('cert' in r for r in results.values())
rho_last = f"rho{CERT_RHOS[-1]:g}"; rho_first = f"rho{CERT_RHOS[0]:g}"
print("(rel-l2 = nodal Euclidean; FE-L2 = mass-matrix norm; +setup = per-sample operator data the model needs (transform /"
      " hierarchy), CPU, excluding the coarse solve" + (f" ({COARSE_MS:.0f} ms/sample)" if args.k0 else '') + ")")
print(f"{'model':<11}{'params':>10}{'nodal rel-l2':>13}{'FE-L2':>11}{'energy err':>12}{'train s':>9}"
      f"{'infer ms':>10}{'+setup ms':>11}"
      + (f"{'effectivity ' + rho_first:>22}{'effectivity ' + rho_last:>22}{'PCG warm/cold':>16}" if cert else ""))
for k, r in results.items():
    line = (f"{k:<11}{r['params']:>10,}{r['rel_l2']:>13.3e}" + (f"{r['l2fe']:>11.3e}" if 'l2fe' in r else f"{'--':>11}")
            + f"{r['energy']:>12.3e}{r['train_s']:>9.0f}{r['infer_ms']:>10.1f}{r['build_ms']:>11.0f}")
    if cert and 'cert' in r:
        for rk in (rho_first, rho_last):
            c = r['cert'][rk]
            line += f"{c['eff_med']:.3f} ({c['eff_min']:.2f}-{c['eff_max']:.2f})".rjust(22)
        line += f"{r['cert'][rho_last]['pcg_warm']:.1f} / {r['cert'][rho_last]['pcg_cold']:.1f}".rjust(16)
    print(line)


# ---------------------------------------------------------------------------
# classical methods on the networks' own test set (3 Oct): same u_c, same samples, same error definitions
# ---------------------------------------------------------------------------
if args.classical and not (args.k0 and args.coarse_solve in ('galerkin', 'oa')):
    print("\n--classical needs --k0 > 0 and --coarse-solve galerkin or oa: skipped")
elif args.classical:
    from scipy.sparse.linalg import factorized as _fact
    _P = None
    for k in range(args.k0 + 1, L + 1):
        _P = H.P[k] if _P is None else H.P[k] @ _P
    _P = _P.tocsr(); _PT = _P.T.tocsr()

    def _errs(x, i, A):
        u = U[i]; e = x - u
        return (np.linalg.norm(e) / np.linalg.norm(u), float(np.sqrt(e @ (A @ e) / (u @ (A @ u)))),
                float(np.sqrt(e @ (MASS @ e) / (u @ (MASS @ u)))))

    def _cheb(A, dinv, q, x, b, n, lmax=3.0):          # Chebyshev on D^-1 A over [lmax/q, lmax], lmax = 3 (proved bound)
        lo_, hi_ = lmax / q, lmax; th, de = (hi_ + lo_) / 2, (hi_ - lo_) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (b - A @ x) / th; x = x + d
        for _ in range(n - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (b - A @ x); x = x + d; r0 = r1
        return x

    QS, NIT = (4, 10, 30, 100), 16
    cheb = {(nu, q): [] for nu in (4, 16) for q in QS}; base, tgh, t_c, t_it = [], [], [], []
    for i in te:
        A = H.A_fine(pool['a'][i]).tocsr(); b = FS[i]; dinv = 1 / A.diagonal()
        t0 = time.perf_counter()
        if args.coarse_solve == 'oa':                     # the networks' coarse space, rebuilt per sample (timed)
            _P = oa_prolongation(A, args.k0, args.coarse_rho or args.rho); _PT = _P.T.tocsr()
        solc = _fact((_PT @ A @ _P).tocsc()); uc = _P @ solc(_PT @ b); t_c.append(time.perf_counter() - t0)
        base.append(_errs(uc, i, A))
        for nu in (4, 16):
            for q in QS:
                cheb[(nu, q)].append(_errs(_cheb(A, dinv, q, uc, b, nu), i, A))
        def tg(r):
            y = (2 / 3) * dinv * r; y = y + _P @ solc(_PT @ (r - A @ y)); return y + (2 / 3) * dinv * (r - A @ y)
        x = uc.copy(); r = b - A @ x; z = tg(r); pdir = z.copy(); rz = r @ z; h = [_errs(x, i, A)]
        t0 = time.perf_counter()
        for _ in range(NIT):
            Ap = A @ pdir; al = rz / (pdir @ Ap); x = x + al * pdir; r = r - al * Ap
            z = tg(r); rz2 = r @ z; pdir = z + (rz2 / rz) * pdir; rz = rz2; h.append(_errs(x, i, A))
        t_it.append((time.perf_counter() - t0) / NIT); tgh.append(h)
    tgh = np.array(tgh); mt = tgh.mean(0)
    print(f"\nCLASSICAL, same u_c ({args.coarse_solve} coarse space), same {n_te} test samples (CPU; coarse {'build+' if args.coarse_solve == 'oa' else ''}factor+solve {1e3 * np.median(t_c):.1f} ms, "
          f"one TG-PCG iteration {1e3 * np.median(t_it):.2f} ms per sample)")
    print(f"{'method':<34}{'nodal rel-l2':>13}{'energy err':>12}{'FE-L2':>11}")
    m = np.mean(base, 0); print(f"{'coarse solve alone':<34}{m[0]:>13.3e}{m[1]:>12.3e}{m[2]:>11.3e}")
    for nu in (4, 16):
        qb = min(QS, key=lambda q: np.mean(cheb[(nu, q)], 0)[1]); m = np.mean(cheb[(nu, qb)], 0)
        print(f"{'Chebyshev post(%d), best q=%d' % (nu, qb):<34}{m[0]:>13.3e}{m[1]:>12.3e}{m[2]:>11.3e}")
    for j in (1, 2, 3, 4, 6, 8):
        print(f"{'two-grid PCG, %d its' % j:<34}{mt[j, 0]:>13.3e}{mt[j, 1]:>12.3e}{mt[j, 2]:>11.3e}")
    VC_ROWS = {}
    if any(k.startswith('oa-vc') for k in results):                         # the classical members of oa-vc's class, from the same u_c
        from oa_vcycle import oa_hierarchy, classical_vcycle
        nl_ = 0; k_ = L
        while k_ >= 1 and nl_ < 4 and len(H.nodes[k_ - 1]) >= 16:
            nl_ += 1; k_ -= 1
        for mm, lab in ((0, 'geometric'), (args.oa_steps, 'operator-adapted')):
            E_, t_ = [], []
            for i in te:
                A = H.A_fine(pool['a'][i]).tocsr(); t0 = time.perf_counter()
                levs, A0 = oa_hierarchy(H, A, nl_, mm); x = UC[i] + classical_vcycle(levs, A0, FS[i] - A @ UC[i], nu=4)
                t_.append(time.perf_counter() - t0); E_.append(_errs(x, i, A))
            m_ = np.mean(E_, 0); VC_ROWS[lab] = np.array(E_)
            print(f"{'V(4,4) Chebyshev, %s' % lab:<34}{m_[0]:>13.3e}{m_[1]:>12.3e}{m_[2]:>11.3e}   ({1e3 * np.median(t_):.1f} ms incl. build)")
        if any(k in results for k in ('oa-vcp', 'oa-vcd', 'oa-vcb', 'oa-vcz', 'oa-vcq', 'oa-vcl', 'oa-vcbl')):
            # their classical members: the same cycle with deflation and / or Galerkin projection (the 2 x 2 ablation)
            from oa_vcycle import classical_vcycle_proj, near_kernel_basis
            for lab, use_z, proj in (('op.-adapted + defl. + proj.', True, True), ('op.-adapted + defl. only', True, False),
                                     ('op.-adapted + proj. only', False, True)):
                if lab != 'op.-adapted + defl. + proj.' and not any(k in results for k in ('oa-vcz', 'oa-vcq', 'oa-vcl')): continue
                E_, t_ = [], []
                for i in te:
                    A = H.A_fine(pool['a'][i]).tocsr(); t0 = time.perf_counter()
                    levs, A0 = oa_hierarchy(H, A, nl_, args.oa_steps); Zi = near_kernel_basis(A) if use_z else None
                    x = UC[i] + classical_vcycle_proj(levs, A0, FS[i] - A @ UC[i], Z=Zi, nu=4, project=proj)
                    t_.append(time.perf_counter() - t0); E_.append(_errs(x, i, A))
                m_ = np.mean(E_, 0); VC_ROWS[lab] = np.array(E_)
                print(f"{'V(4,4) ' + lab:<34}{m_[0]:>13.3e}{m_[1]:>12.3e}{m_[2]:>11.3e}   ({1e3 * np.median(t_):.1f} ms incl. build)")
    # 6 Oct (Astra): the classical comparators' PER-SAMPLE errors are recorded too (paired comparisons, worst cases)
    _crow = {'coarse solve alone': np.array(base)}
    for nu in (4, 16):
        for q in QS: _crow[f'Chebyshev post({nu}), q={q}'] = np.array(cheb[(nu, q)])
    for j in range(1, NIT + 1): _crow[f'two-grid PCG, {j} its'] = tgh[:, j]
    _crow.update({f'V(4,4) {k}': v for k, v in VC_ROWS.items()})
    with open(os.path.join(args.out_dir, f"classical_{os.path.basename(res_path)}"), 'a') as fh:
        for name_, E in _crow.items():
            E = np.asarray(E)
            fh.write(json.dumps(dict(cfg=cfg_key, model=name_, code=CODE_HASH, sample_ids=[int(i) for i in te],
                                     rl2_s=E[:, 0].tolist(), en_s=E[:, 1].tolist(), l2fe_s=E[:, 2].tolist())) + '\n')
    for k, r in results.items():
        hit = [np.nonzero((hh[:, 0] <= r['rel_l2']) & (hh[:, 1] <= r['energy']))[0] for hh in tgh]
        hit = [int(t[0]) if len(t) else NIT + 1 for t in hit]
        print(f"  two-grid PCG iterations to match {k:<16}: median {np.median(hit):.0f}  (min {min(hit)}, max {max(hit)}"
              f"{'; >%d means not reached' % NIT if max(hit) > NIT else ''})")

    # ---- split by FLOATING ISLANDS (4 Oct): conductive regions (a >= 0.5) without a Dirichlet node.  With barriers
    # (contrast < 1) four domain-crossing strips can enclose such a region; it carries a near-singular mode (eigenvalue
    # ~ contrast) and a solution ~100x larger, and dominates the means (diagnostics_cost/barrier_islands.log).
    if args.contrast < 1:
        import scipy.sparse as _sp
        from scipy.sparse.csgraph import connected_components as _cc
        _dir = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L])
        def _has_island(a):
            cond = H.keep & (np.asarray(a).ravel() >= 0.5); el = H.elem[cond]
            M = _sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())),
                               shape=(len(el), len(H.coords)))
            G = (M.T @ M).tocsr(); used = np.unique(el.ravel())
            n_, lab = _cc(G[used][:, used], directed=False)
            touch = np.zeros(n_, bool); touch[lab[np.isin(used, _dir)]] = True
            return not touch.all()
        isl = np.array([_has_island(pool['a'][i]) for i in te])
        print(f"\nSPLIT by floating conductive islands: {isl.sum()} of {n_te} test samples have one "
              f"(rel-L2 / energy; mean | median)")
        rows = [('coarse solve alone', np.array(base))]
        qb = min(QS, key=lambda q: np.mean(cheb[(4, q)], 0)[1]); rows.append(('Chebyshev post(4)', np.array(cheb[(4, qb)])))
        rows += [(f'two-grid PCG, {j} its', tgh[:, j]) for j in (2, 4, 8)]
        rows += [(f'V(4,4) Chebyshev, {k}', v) for k, v in VC_ROWS.items()]
        rows += [(k, np.column_stack([r['rl2_s'], r['en_s']])) for k, r in results.items() if 'en_s' in r]
        rows += [(k + ' + line search', np.column_stack([r['rl2_ls_s'], r['en_ls_s']])) for k, r in results.items() if 'en_ls_s' in r]
        print(f"{'':<34}{'with island':>34}{'without island':>34}")
        for name_, E in rows:
            a_, b_ = E[isl], E[~isl]
            f = lambda Z: (f"{Z[:, 0].mean():.2e}/{Z[:, 1].mean():.3f} | {np.median(Z[:, 0]):.2e}/{np.median(Z[:, 1]):.3f}"
                           if len(Z) else '--')
            print(f"{name_:<34}{f(a_):>34}{f(b_):>34}")
        miss = [k for k, r in results.items() if 'en_s' not in r]
        if miss:
            print(f"  (no per-sample errors for {', '.join(miss)}: rerun with --reeval to evaluate them from their checkpoints)")

