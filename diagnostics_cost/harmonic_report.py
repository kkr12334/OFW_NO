"""Decision statistics of protocol v2/v3 section 5 from harmonic_<mode>.json: per seed and pooled, R_cap and G24 with
5,000-resample paired bootstrap 95% intervals, kappa quartiles, the verdict, and the secondary learned-vs-classical
comparison after the same enrichment.  usage: harmonic_report.py mode rho"""
import json, sys, os, numpy as np
mode, rho = sys.argv[1], int(sys.argv[2]); R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'harmonic_{mode}.json')))
B = np.random.default_rng(0)
def stats(recs, who):
    isl = [r for r in recs if r['n_islands'] > 0]; n = len(isl)
    e0 = np.array([r['sets'][who]['E0']['err'] for r in isl]); e2 = np.array([r['sets'][who][f'E2_{rho}']['err'] for r in isl])
    e4 = np.array([r['sets'][who][f'E4_{rho}']['err'] for r in isl]); kap = np.array([r['sets'][who][f'E2_{rho}']['kappa'] for r in isl])
    b = B.integers(0, n, (5000, n))
    Rc = 1 - e2.mean() / e0.mean(); Rb = 1 - e2[b].mean(1) / e0[b].mean(1)
    lg = np.log(e4 / e2); G = np.exp(lg.mean()); Gb = np.exp(lg[b].mean(1))
    verdict = 'STRONG' if (Rc >= 0.5 and G >= 1.25) else ('PARTIAL' if Rc >= 0.2 else 'LITTLE SUPPORT')
    return n, e0.mean(), e2.mean(), Rc, np.percentile(Rb, [2.5, 97.5]), G, np.percentile(Gb, [2.5, 97.5]), np.percentile(kap, [25, 50, 75]), verdict
def lc(recs, s, island_only):
    rs = [r for r in recs if (r['n_islands'] > 0 or not island_only)]
    lr = np.log(np.array([r['sets']['member'][s]['err'] for r in rs]) / np.array([r['sets']['oa-vcb'][s]['err'] for r in rs]))
    b = B.integers(0, len(lr), (5000, len(lr))); return np.exp(lr.mean()), np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5])), np.mean(lr > 0), len(lr)
groups = sorted({r['seed'] for r in R})
for who in ('oa-vcb', 'member'):
    print(f"== {who}, rho {rho}, set {mode}")
    for g in groups + ['pooled']:
        recs = R if g == 'pooled' else [r for r in R if r['seed'] == g]
        n, e0, e2, Rc, Rci, G, Gci, kq, v = stats(recs, who)
        print(f"  {str(g):>6}: n_island {n:3d}  E0 {e0:.4f} -> E2 {e2:.4f}  R_cap {Rc:+.3f} [{Rci[0]:+.3f}, {Rci[1]:+.3f}]  G24 {G:.3f} [{Gci[0]:.3f}, {Gci[1]:.3f}]  "
              f"kappa q1/med/q3 {kq[0]:.3f}/{kq[1]:.3f}/{kq[2]:.3f}  -> {v}")
print("== learned vs classical, paired G(member / OA-VCB) [95% CI], W")
for s in ('E0', f'E2_{rho}'):
    for io in (True, False):
        G, ci, w, n = lc(R, s, io); print(f"  {s:<6} {'island samples' if io else 'all samples':<15} n {n:3d}: G {G:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]  W {w:.2f}")
