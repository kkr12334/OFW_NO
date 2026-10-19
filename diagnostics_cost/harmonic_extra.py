"""E3 vs E2 on the fixed E3 subset, label disagreements, timings -- from harmonic_<mode>.json.  usage: harmonic_extra.py mode rho"""
import json, sys, os, numpy as np
mode, rho = sys.argv[1], int(sys.argv[2]); R = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'harmonic_{mode}.json')))
for who in ('oa-vcb', 'member'):
    sub = [r for r in R if 'E3' in r['sets'][who]]
    if sub:
        e0 = np.mean([r['sets'][who]['E0']['err'] for r in sub]); e2 = np.mean([r['sets'][who][f'E2_{rho}']['err'] for r in sub]); e3 = np.mean([r['sets'][who]['E3']['err'] for r in sub])
        print(f"{who}: E3 subset {len(sub)} samples: E0 {e0:.4f}  E2(rho {rho}) {e2:.4f} (R_cap {1 - e2 / e0:+.3f})  E3 global {e3:.4f} (R_cap {1 - e3 / e0:+.3f})")
det = np.array([r['n_islands'] > 0 for r in R]); geo = np.array([r['geometric_island'] for r in R])
print(f"labels: detector {det.sum()}, geometric {geo.sum()}, both {(det & geo).sum()}, detector only {(det & ~geo).sum()}, geometric only {(~det & geo).sum()}")
isl = [r for r in R if r['n_islands'] > 0]
print(f"island samples: islands per sample {np.mean([r['n_islands'] for r in isl]):.2f}; patch size (rho {rho}) median {np.median(sum([r[f'patch_sizes_{rho}'] for r in isl], [])):.0f}; "
      f"build E2 median {1e3 * np.median([r['t_build'][f'E2_{rho}'] for r in isl]):.1f} ms, apply median {1e3 * np.median([r['sets']['oa-vcb'][f'E2_{rho}']['t_apply'] for r in isl]):.1f} ms, detect median {1e3 * np.median([r['t_detect'] for r in R]):.1f} ms")
print(f"retained/nominal E2 columns: {sum(r['sets']['oa-vcb'][f'E2_{rho}']['retained'] for r in isl)}/{sum(r['sets']['oa-vcb'][f'E2_{rho}']['nominal'] for r in isl)};  E4: {sum(r['sets']['oa-vcb'][f'E4_{rho}']['retained'] for r in isl)}/{sum(r['sets']['oa-vcb'][f'E4_{rho}']['nominal'] for r in isl)}; "
      f"max scaled cond E2 {max(r['sets']['oa-vcb'][f'E2_{rho}']['cond_scaled'] for r in isl):.1e}; max |gain - gain check| / ||e||_A^2 "
      f"{max(abs(r['sets'][w][s]['gain'] - r['sets'][w][s]['gain_check']) * abs(r['sets'][w][s]['kappa']) / abs(r['sets'][w][s]['gain']) for r in isl for w in ('oa-vcb', 'member') for s in r['sets'][w] if s != 'E0' and r['sets'][w][s]['gain'] != 0):.1e}")
non = [r for r in R if r['n_islands'] == 0]
print(f"non-island samples {len(non)}: max |E2 change| {max([abs(r['sets']['oa-vcb'][f'E2_{rho}']['err'] - r['sets']['oa-vcb']['E0']['err']) for r in non] or [0]):.1e} (empty by construction)")
