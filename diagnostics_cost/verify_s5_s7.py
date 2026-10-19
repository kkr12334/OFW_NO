"""10 Oct: verification of the Session 5 (geometric transfers) and Session 7 (barriers seeds 2-3, baselines) summaries
from the raw jsonl records.  Paired per-sample energy statistics, test ids 300-399."""
import json, glob, numpy as np, sys
def recs(files, **want):
    out = {}
    for f in files:
        for l in open(f):
            r = json.loads(l); c = json.loads(r['cfg'])
            if all(c.get(k, d) == v for k, (v, d) in want.items()):
                res = r.get('result', r); out[r['model']] = dict(en=np.array(res['en_s']), ids=r.get('sample_ids'), res=res, code=r.get('code'))
    return out
def G(a, b):
    lr = np.log(a / b); bs = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr)))
    ci = np.exp(np.percentile(lr[bs].mean(1), [2.5, 97.5])); return np.exp(lr.mean()), ci, np.mean(lr > 0), np.exp(lr.min())
MEM = 'V(4,4) op.-adapted + defl. + proj.'
S1 = glob.glob('s1/*.jsonl'); S3 = glob.glob('s3/*.jsonl'); S5 = glob.glob('s5/*.jsonl'); S7 = glob.glob('s7/*.jsonl')
print("== Session 5: geometric vs operator-adapted transfers (OA-VCB, each with its own classical member)")
for lab, ds, contrast, load, adapt_files in (('channels', 1, 1e4, 'all', S1), ('channels', 2, 1e4, 'all', S1), ('channels', 3, 1e4, 'all', S1),
                                             ('barriers', 1, 1e-4, 'no-low', S3)):
    ad = recs(adapt_files, data_seed=(ds, 0), contrast=(contrast, None), load=(load, 'all'), oa_steps=(4, 4))
    ge = recs(S5, data_seed=(ds, 0), contrast=(contrast, None), load=(load, 'all'), oa_steps=(0, 4))
    assert ge[MEM]['ids'] == ad[MEM]['ids']
    same_uc = np.max(np.abs(ge['coarse solve alone']['en'] / ad['coarse solve alone']['en'] - 1))
    g, ci, w, _ = G(ge['oa-vcb']['en'], ad['oa-vcb']['en']); gm = G(ge[MEM]['en'], ad[MEM]['en'])[0]
    la = G(ad[MEM]['en'], ad['oa-vcb']['en'])[0]; lg = G(ge[MEM]['en'], ge['oa-vcb']['en'])
    print(f"  {lab} s{ds}: OA-VCB adapted {ad['oa-vcb']['en'].mean():.3e} geometric {ge['oa-vcb']['en'].mean():.3e}; geo/adapted G {g:.2f} [{ci[0]:.2f},{ci[1]:.2f}] W {w:.2f};"
          f" member geo/adapted G {gm:.2f} (means {ge[MEM]['en'].mean():.2e}/{ad[MEM]['en'].mean():.2e}); learned vs own member: adapted {la:.2f}, geometric {lg[0]:.2f} W {lg[2]:.2f}; u_c identical to {same_uc:.0e}; codes {ge['oa-vcb']['code']}")
print("\n== Session 7 (+ Session 3 seed 1): barriers, held-out seeds 1-3")
pool_lr = []
for ds in (1, 2, 3):
    R = recs(S3 + S7, data_seed=(ds, 0), contrast=(1e-4, None), load=('no-low', 'all'), oa_steps=(4, 4))
    e = R['oa-vcb']['en']; m = R[MEM]['en']; cs = R['coarse solve alone']['en']
    g, ci, w, worst = G(m, e); pool_lr.append(np.log(m / e))
    print(f"  seed {ds}: OA-VCB {e.mean():.4f}  member {m.mean():.4f}  ratio of means {m.mean() / e.mean():.2f}  G {g:.2f} [{ci[0]:.2f},{ci[1]:.2f}]  W {w:.2f}  worst {worst:.2f}  coarse {cs.mean():.4f}")
    print(f"     OA-VCB worse than u_c: {(e > cs * (1 + 1e-6)).sum()}; n_worse field {R['oa-vcb']['res'].get('n_worse_than_uc')}")
    for k in ['oa-vc', 'oa-ll-op', 'gnn-op', 'gnn', 'transolver-op', 'fno-op', 'wno-op', 'lino-op', 'mgno', 'mgno-op']:
        if k not in R: print(f"     {k:<14} MISSING"); continue
        x = R[k]['en']; gx = G(x, e)
        print(f"     {k:<14} mean {x.mean():.4f}  {'BELOW' if x.mean() < cs.mean() else 'at/above'} coarse ({cs.mean():.4f}); worse than u_c on {(x > cs * (1 + 1e-6)).sum():>3}; "
              f"model/OA-VCB G {gx[0]:6.2f} W {gx[2]:.2f}; vs member G {G(m, x)[0]:.2f}")
    PC = np.array([R[f'two-grid PCG, {j} its']['en'] for j in range(1, 17)]).T
    hit = [int(np.nonzero(h <= t)[0][0]) + 1 if (h <= t).any() else 17 for h, t in zip(PC, e)]
    print(f"     two-grid PCG its to OA-VCB (energy only): median {np.median(hit):.0f} ({min(hit)}-{max(hit)})")
lr = np.concatenate(pool_lr); print(f"  pooled seeds 1-3: G {np.exp(lr.mean()):.2f}  W {np.mean(lr > 0):.2f}")
