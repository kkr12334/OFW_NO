"""Step 1 of the formulation's next-experiments list (section 10).

For each coefficient regime, at the project's resolution, measure
  (a) how much cross-scale coupling each TRANSFORM regime leaves
      -> PCG iterations and kappa(M A) with M = T diag(D)^-1 T^T
         for T = exact / local(rho) / frozen(a=1) / plain HB
  (b) the spectral structure of the exact level blocks D_k
      -> kappa raw, kappa after Jacobi scaling, number of eigenvalues below
         lambda_max/20 (the 'low-rank defect'), kappa after deflating them
  (c) decay of the Jacobi-scaled D_L^{-1} with distance (is a local R justified?)

No training. Level 6 takes a few minutes per field; level 7 is feasible for
(a) but (b) needs dense eigensolves of blocks up to ~12k -- use --spectra-max.

usage:
  python exp_level_spectra.py                       # level 6, all fields, contrast 1e4
  python exp_level_spectra.py --level 7 --samples 1 --spectra-max 4000 --frozen-rho 4 --maxiter 1500
  python exp_level_spectra.py --fields channels --contrasts 1e2 1e4 1e6
"""
import argparse, time, sys
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
sys.path.insert(0, '.')
from oa_lifting import (NestedHierarchy, OATransform, make_field, pcg_kappa,
                        block_spectrum, jacobi_scaled, rel_l2)

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, default=6)
ap.add_argument('--fields', nargs='+', default=['blocks8', 'blocks16', 'grf', 'channels'])
ap.add_argument('--contrasts', nargs='+', type=float, default=[1e4])
ap.add_argument('--samples', type=int, default=3)
ap.add_argument('--rhos', nargs='+', type=float, default=[1, 2, 3, 4, 6])
ap.add_argument('--spectra-max', type=int, default=6000, help='largest block to eigendecompose')
ap.add_argument('--no-exact', action='store_true', help='skip exact mode (memory at level 7)')
ap.add_argument('--channel-width', type=float, default=1 / 16)
ap.add_argument('--route', choices=['V','W'], default='V',
                help="V: operator in the predict (default); W: operator in the update")
ap.add_argument('--seed', type=int, default=2026)
ap.add_argument('--modes', nargs='+', default=['exact', 'local', 'frozen', 'hb'],
                help='transform regimes to run; frozen and hb are the slow ones at level 7 '
                     '(thousands of PCG iterations, each with a dense solve)')
ap.add_argument('--maxiter', type=int, default=5000)
ap.add_argument('--frozen-rho', type=float, default=None,
                help='localise the frozen (a=1) transform to this patch radius; keeps its '
                     'level blocks sparse. Strongly recommended at level 7 (e.g. 4)')
args = ap.parse_args()

H = NestedHierarchy(args.level)
A1 = H.A_fine(np.ones(H.n_elem))
print(f"level {args.level}: {len(H.nodes[args.level])} interior nodes; "
      f"block sizes {[len(H.nodes[0])] + [len(H.new[k]) for k in range(1, args.level + 1)]}")
rng = np.random.default_rng(args.seed)


def decay_profile(Dk, xy, h, dists=(1, 2, 4, 8, 16)):
    Dj, _ = jacobi_scaled(Dk)
    X = np.linalg.inv(Dj)
    c = int(np.argmin(((xy - 0.5) ** 2).sum(1)))
    d = np.sqrt(((xy - xy[c]) ** 2).sum(1)) / h
    out = []
    for r in dists:
        m = (d >= r) & (d < 2 * r if r > 0 else 1)
        out.append(np.abs(X[c, m]).max() / abs(X[c, c]) if m.any() else np.nan)
    return out


summary = []
for field in args.fields:
    for contrast in args.contrasts:
        for s in range(args.samples):
            kw = dict(width=args.channel_width) if field == 'channels' else {}
            a = make_field(H, field, contrast, rng, **kw)
            A = H.A_fine(a)
            u = spsolve(A, H.F)
            tag = f"{field} c={contrast:.0e} #{s}"
            print(f"\n=== {tag}   a in [{a.min():.2g}, {a.max():.2g}]", flush=True)
            print(f"  {'transform':<14}{'build s':>9}{'PCG it':>8}{'PCG s':>8}{'kappa(MA)':>12}{'err':>10}", flush=True)
            modes = ([('exact', {})] if ('exact' in args.modes and not args.no_exact) else []) + \
                    ([(f'local rho={r:g}', dict(rho=r)) for r in args.rhos] if 'local' in args.modes else []) + \
                    ([('frozen a=1', dict(A_ref=A1, frozen_local_rho=args.frozen_rho))] if 'frozen' in args.modes else []) + \
                    ([('hb', {})] if 'hb' in args.modes else [])
            row = dict(tag=tag)
            Tex = None
            for name, kw2 in modes:
                mode = name.split()[0]
                T = OATransform(H, A, mode, route=args.route, dense_D_max=args.spectra_max, **kw2)
                t0 = time.time()
                x, it, kap = pcg_kappa(A, H.F, T.precond, maxiter=args.maxiter)
                flag = '' if it < args.maxiter else '  (hit --maxiter)'
                print(f"  {name:<14}{T.build_time:>9.1f}{it:>8d}{time.time() - t0:>8.1f}{kap:>12.1f}"
                      f"{rel_l2(u, x):>10.1e}{flag}", flush=True)
                row[name] = (it, kap)
                if mode == 'exact': Tex = T
            if Tex is not None:
                print(f"  exact level blocks (gap = lambda_max/20):")
                print(f"  {'k':>3}{'size':>7}{'kappa raw':>12}{'kappa Jac':>11}{'#small/50':>10}"
                      f"{'#small/20':>10}{'#small/10':>10}{'kappa defl':>11}")
                for k in range(1, args.level + 1):
                    if Tex.size(k) > args.spectra_max:
                        print(f"  {k:>3}{Tex.size(k):>7}   (skipped: > --spectra-max)"); continue
                    t0 = time.time()
                    b = block_spectrum(Tex.D[k])
                    print(f"  {k:>3}{b['n']:>7}{b['kappa_raw']:>12.3g}{b['kappa_jac']:>11.3g}"
                          f"{b['n_small_50']:>10}{b['n_small']:>10}{b['n_small_10']:>10}{b['kappa_deflated']:>11.3g}   ({time.time() - t0:.0f}s)", flush=True)
                    row[f'D{k}'] = b
                kL = max(k for k in range(1, args.level + 1) if Tex.size(k) <= args.spectra_max)
                prof = decay_profile(Tex.D[kL], H.new_coords(kL), H.h[kL])
                print(f"  decay of Jacobi-scaled D_{kL}^-1 (max |row| in [r,2r) h / diagonal): "
                      + "  ".join(f"r={r}:{p:.1e}" for r, p in zip((1, 2, 4, 8, 16), prof)))
                del Tex
            summary.append(row)

print("\n\nSUMMARY  PCG iterations (kappa) per transform")
if not summary or len(summary[0]) <= 1: sys.exit(0)
names = [k for k in summary[0] if k != 'tag' and not k.startswith('D')]
print(f"{'sample':<26}" + "".join(f"{n:>18}" for n in names))
for row in summary:
    print(f"{row['tag']:<26}" + "".join(f"{row[n][0]:>7d} ({row[n][1]:>8.1f})" for n in names))
