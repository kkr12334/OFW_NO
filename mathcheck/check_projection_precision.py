"""5 Oct (independent review, finding B, and its follow-up): the energy safeguard of the Galerkin output AS DEPLOYED, through
the actual PyTorch code (OAVCycle.forward / OAVCycle._galerkin), on the device given by --device (cpu or cuda; the paper
runs use cuda, float32 network).  The energy error is measured in float64 against the exact discrete error e = A^-1 r of
the ASSEMBLED operator (scipy).

Paths:  deployed = what the drivers do since 5 Oct: float32 operator data for the cycle, the float64 assembled fine
        operator (OAData.A64) and the float64 residual for deflation and projection;
        legacy   = float32 residual and no A64 (the safeguard then holds in the norm of the float32-rounded operator only).
  1. real samples (L3 perforated, barriers 1e-4 and channels 1e4), Jacobi weights and 3 random weight sets:
     ratio ||e - x||_A / ||e||_A <= 1 (never worse than u_c);  deployed tolerance 1e-12, legacy 1e-6;
  2. stress through _galerkin on a real operator: near-duplicate / duplicate directions (eps 1e-2 .. 0), as projection
     (b = r) and as deflation (Z with duplicate and zero padding columns);
  3. residual scale: 0, 1e-30 .. 1e8: finite, ratio <= 1, positive homogeneity (enforced, tolerance 1e-5);
  4. gradients: backward through _galerkin at near-duplicate / duplicate columns, through the whole network in training
     mode (loss on projected + plain output) at zero and tiny residuals; finiteness and size; torch.autograd.gradcheck of
     _galerkin (float64, well-conditioned case).
Statistics: backstop activations (acceptance failed -> line search), rejected corrections (alpha = 0), numerical rank
deficit of the Jacobi-scaled Gram matrix.
Run:  python mathcheck/check_projection_precision.py [--device cuda]"""
import sys, os, argparse, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle

ap = argparse.ArgumentParser(); ap.add_argument('--device', default='cpu'); args = ap.parse_args()
dev = torch.device(args.device)
print(f"device {dev}" + (f" ({torch.cuda.get_device_name(dev)})" if dev.type == 'cuda' else '') + f", torch {torch.__version__}")
L, m, nu, om = 3, 4, 4, 2 / 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
P2 = H.P[3].tocsr()
TOL = {'deployed': 1e-12, 'legacy': 1e-6}


def energy(A, v):
    return float(np.sqrt(max(v @ (A @ v), 0.0)))


def jacobi_weights(net):
    with torch.no_grad():
        for S in (net.pre[0], net.post[0]):
            for p in S.parameters(): p.zero_()
            S.enc_lin.weight[1, 0] = 1.0
            for t in range(nu):
                S.V[t].weight[0, 0] = 1.0; Lw = S.Lw[t].weight; Lw[0, 1] = om; Lw[0, 0] = -om; Lw[0, S.C] = -om
            S.dec_lin.weight[0, 0] = 1.0


def stats_line(net):
    s = getattr(net, 'stats', {})
    return (f"backstop {s.get('backstop', 0)}, rejected {s.get('rejected', 0)} of {s.get('samples', 0)} projections; "
            f"rank deficit {s.get('rank_deficit', 0)} of {s.get('cols', 0)} live columns")


worst = {'deployed': 0.0, 'legacy': 0.0}; viol = {'deployed': 0, 'legacy': 0}
FAILS = []                     # required checks that failed (6 Oct: the pre-flight must fail loudly)
for name, contrast in (('barriers 1e-4', 1e-4), ('channels 1e4', 1e4)):
    rng = np.random.default_rng(1000); a_all = [field_channels(H, contrast, rng) for _ in range(400)]
    a_list = [a_all[i] for i in range(300, 324)]
    D = OAData(H, a_list, 3, m, dev, H.level_coords, near_kernel=True)
    rows = []
    for i, a in enumerate(a_list):
        A = H.A_fine(a).tocsr(); F = H.F
        uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); r = F - A @ uc
        rows.append((A, r, spsolve(A.tocsc(), r)))
    R64 = torch.tensor(np.array([r for _, r, _ in rows]), dtype=torch.float64, device=dev)
    for wname in ('jacobi', 'seed0', 'seed1', 'seed2'):
        torch.manual_seed(int(wname[-1]) if wname != 'jacobi' else 0)
        net = OAVCycle(D, width=8 if wname != 'jacobi' else 4, nu=nu, project='both', deflate=True).to(dev)
        if wname == 'jacobi': jacobi_weights(net)
        net.eval(); net.collect_stats = True
        for path in ('deployed', 'legacy'):
            net.set_ops(D.Av, D.Xv, D.A0inv, D.Z, A64=D.A64 if path == 'deployed' else None)
            with torch.no_grad():
                X = net(R64 if path == 'deployed' else R64.float()).double().cpu().numpy()
            rat = np.array([energy(A, e - X[i]) / energy(A, e) for i, (A, _, e) in enumerate(rows)])
            if path == 'deployed' and not (np.all(np.isfinite(X)) and np.all(np.isfinite(rat))):   # NaN > tol is False
                FAILS.append(f"non-finite deployed output ({name}, {wname})")
            worst[path] = max(worst[path], rat.max()); viol[path] += int(np.sum(rat > 1 + TOL[path]))
            if path == 'deployed':
                print(f"1. {name:<14} {wname:<7} deployed: ratio max {rat.max():.12f}, median {np.median(rat):.4f}, "
                      f"violations {int(np.sum(rat > 1 + TOL[path]))}/{len(rat)}", flush=True)
            else:
                print(f"   {'':<14} {'':<7} legacy  : ratio max {rat.max():.12f}, violations (tol 1e-6) "
                      f"{int(np.sum(rat > 1 + TOL[path]))}/{len(rat)}", flush=True)
        print(f"   {'':<14} {'':<7} {stats_line(net)}", flush=True)

# 2. stress through _galerkin on a real operator (channels data, sample 0), deployed path
one = lambda i: ([v[i:i + 1] for v in D.Av], [x[i:i + 1] for x in D.Xv], D.A0inv[i:i + 1], D.Z[i:i + 1])
A, r0, e0 = rows[0]; n = A.shape[0]
net = OAVCycle(D, width=8, nu=nu, project='both', deflate=True).to(dev)
Av, Xv, A0i, Z1 = one(0); net.set_ops(Av, Xv, A0i, Z1, A64=D.A64[0:1]); net.collect_stats = True
b = torch.tensor(r0[None], dtype=torch.float64, device=dev)
g = np.random.default_rng(7); w2 = {'projection': 0.0, 'deflation': 0.0}; bad2 = {'projection': 0, 'deflation': 0}; cases = 0
for eps in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 0.0):
    for t in range(30):
        v = g.standard_normal(n); w = v + eps * g.standard_normal(n); u = g.standard_normal(n)
        for kind, Wn in (('projection', np.column_stack([v, w, u, A @ v])),
                         ('deflation', np.column_stack([np.abs(v), np.abs(w), np.zeros(n), np.zeros(n)]))):
            Wt = torch.tensor(Wn[None], dtype=torch.float32, device=dev)
            with torch.no_grad(): x = net._galerkin(Wt, b)[0].cpu().numpy()
            q = energy(A, e0 - x) / energy(A, e0)
            if not (np.all(np.isfinite(x)) and np.isfinite(q)): bad2[kind] += 1; continue      # non-finite counts as a violation
            w2[kind] = max(w2[kind], q); bad2[kind] += int(q > 1 + 1e-12)
        cases += 1
print(f"2. near-duplicate / duplicate directions (eps 1e-2 .. 0), {cases} cases each: projection worst ratio "
      f"{w2['projection']:.12f}, violations {bad2['projection']}; deflation (+ zero padding) worst {w2['deflation']:.12f}, "
      f"violations {bad2['deflation']};  {stats_line(net)}")

# 3. residual scale, deployed path, sample 1
A, r1, e1 = rows[1]; Av, Xv, A0i, Z1 = one(1); net.set_ops(Av, Xv, A0i, Z1, A64=D.A64[1:2]); net.eval()
base = None; ok3 = True
for s in (0.0, 1e-30, 1e-20, 1e-8, 1.0, 1e8):
    with torch.no_grad(): x = net(torch.tensor((s * r1)[None], dtype=torch.float64, device=dev))[0].cpu().numpy()
    fin = bool(np.all(np.isfinite(x)))
    q = energy(A, s * e1 - x) / energy(A, s * e1) if s > 0 else float(np.abs(x).max())
    if s == 1.0: base = x
    ok3 &= fin and (q <= 1 + 1e-12 if s > 0 else q == 0)
    hom = f"  homogeneity vs scale 1: {np.linalg.norm(x / s - base) / np.linalg.norm(base):.1e}" if (s > 1 and base is not None) else ''
    print(f"3. r scaled by {s:<7g} finite {fin}  " + (f"ratio {q:.12f}" if s > 0 else f"max|x| {q:.1e}") + hom)
# 7 Oct (first Kaggle pre-flight, T4): homogeneity is enforced at 1e-5 for the scales at which the smallest input (r) and
# output (x) components stay in the NORMAL float32 range (here 1e-20, 1e-8, 1e8).  At 1e-30 the smallest residual components fall below the smallest normal float32 (1.2e-38);
# CUDA kernels flush such subnormals to zero, which costs relative accuracy there (T4: 2.5e-5; reproduced on CPU with
# torch.set_flush_denormal(True): 2.7e-5, all other scales unchanged at 4-7e-7).  Finiteness and the energy ratio are still
# enforced at 1e-30 (above), and the deviation there must stay below 1e-3 -- a real break of homogeneity is O(1).
SCALES = (1e-30, 1e-20, 1e-8, 1e8)
with torch.no_grad():
    xs = {s_: net(torch.tensor((s_ * r1)[None], dtype=torch.float64, device=dev))[0].cpu().numpy() / s_ for s_ in SCALES}
devs = {s_: np.linalg.norm(x_ - base) / np.linalg.norm(base) for s_, x_ in xs.items()}
smin = float(min(np.min(np.abs(r1[r1 != 0])), np.min(np.abs(base[base != 0]))))    # smallest input or output component at scale 1
HOM_TOL, HOM_TOL_SUB = 1e-5, 1e-3              # 6 Oct (Astra): enforced; float32 network round-off measured 4-7e-7
hom_max = max(devs[s_] for s_ in SCALES if s_ * smin >= 1.2e-38)
hom_sub = max([devs[s_] for s_ in SCALES if s_ * smin < 1.2e-38] or [0.0])
print("   positive homogeneity, relative deviation vs scale 1: " + ", ".join(f"{s_:g}: {devs[s_]:.1e}" for s_ in SCALES))
print(f"   enforced: scales with float32-normal data max {hom_max:.1e} (tolerance {HOM_TOL:g}); scales reaching the subnormal "
      f"range (smallest |s r_i| or |s x_i| < 1.2e-38) max {hom_sub:.1e} (tolerance {HOM_TOL_SUB:g}; CUDA flushes subnormals to zero)")
if not (np.isfinite(hom_max) and hom_max <= HOM_TOL and np.isfinite(hom_sub) and hom_sub <= HOM_TOL_SUB): ok3 = False

# 4. gradients
gmax = {}; nonfinite = 0
for eps in (1e-2, 1e-4, 1e-6, 0.0):
    v = g.standard_normal(n); u = g.standard_normal(n)
    Wn = torch.tensor(np.column_stack([v, v + eps * g.standard_normal(n), u])[None], dtype=torch.float32, device=dev,
                      requires_grad=True)
    tgt = torch.tensor(g.standard_normal((1, n)), dtype=torch.float64, device=dev)
    d = net._galerkin(Wn, b); ((d - tgt) ** 2).sum().backward()
    gr = Wn.grad.double(); nonfinite += int(~torch.isfinite(gr).all()); gmax[eps] = float(gr.abs().max() / (2 * (d - tgt).abs().max()))
print("4a. backward through _galerkin, |dL/dW|_max / |dL/dd|_max at column distance eps: "
      + ', '.join(f"eps {k:g}: {v_:.2e}" for k, v_ in gmax.items()) + f";  non-finite gradients: {nonfinite}")
net.train(); nonfin_b = []
for s in (0.0, 1e-30, 1e-20, 1.0):
    net.zero_grad()
    out = net(torch.tensor((s * r1)[None], dtype=torch.float64, device=dev))
    loss = (out.float() ** 2).sum() + (net.aux_out.float() ** 2).sum() + 0.0 * sum(p.sum() for p in net.parameters())
    loss.backward()
    gn = torch.sqrt(sum((p.grad.double() ** 2).sum() for p in net.parameters() if p.grad is not None))
    nonfin_b.append((s, bool(torch.isfinite(gn)), float(gn)))
print("4b. whole network, training mode (projected + plain output), parameter-gradient norm at residual scale: "
      + ', '.join(f"{s:g}: {'finite' if f else 'NON-FINITE'} {v_:.2e}" for s, f, v_ in nonfin_b))
net.eval()
Ws = torch.tensor(g.standard_normal((1, n, 3)), dtype=torch.float64, device=dev, requires_grad=True)
bs = torch.tensor(r1[None], dtype=torch.float64, device=dev, requires_grad=True)
try:
    gc = torch.autograd.gradcheck(lambda W_, b_: net._galerkin(W_, b_), (Ws, bs), eps=1e-6, atol=1e-5, rtol=1e-4, fast_mode=True)
except Exception as ex:                                        # gradcheck raises on mismatch
    print(f"    gradcheck raised: {str(ex)[:300]}"); gc = False
print(f"4c. torch.autograd.gradcheck of _galerkin (float64, 3 random directions): {'pass' if gc else 'FAIL'}")
ok4 = nonfinite == 0 and all(f for _, f, _ in nonfin_b) and gc
print(f"SUMMARY: deployed path worst ratio {worst['deployed']:.12f} ({viol['deployed']} violations, tol 1e-12); legacy path "
      f"worst {worst['legacy']:.9f} ({viol['legacy']} violations, tol 1e-6); stress violations "
      f"{bad2['projection'] + bad2['deflation']}; scale tests {'pass' if ok3 else 'FAIL'}; gradient tests {'pass' if ok4 else 'FAIL'}")
if viol['deployed']: FAILS.append(f"{viol['deployed']} deployed energy-safeguard violations")
if bad2['projection'] + bad2['deflation']: FAILS.append(f"{bad2['projection'] + bad2['deflation']} stress violations (incl. non-finite)")
if not ok3: FAILS.append("residual-scale tests (finiteness, energy ratio, positive homogeneity)")
if not ok4: FAILS.append("gradient tests")
# legacy-path violations are diagnostic only (the repaired path is what runs)
if FAILS:
    print("PREFLIGHT FAIL (projection precision): " + "; ".join(FAILS)); sys.exit(1)
print("PREFLIGHT PASS (projection precision): deployed path, stress, scale (incl. homogeneity) and gradient checks all passed")
