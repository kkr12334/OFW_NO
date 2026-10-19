"""Stage 0 reproduction gate (16 Oct 2026, extension programme; see CLAUDE_ROADMAP_REVIEW_RESPONSE_2026-10-14.md).

Rules fixed BEFORE running (do not edit after the first run; amendments go in a dated note):
  Configuration: L3, k0 2, held-out data seed 1, test ids 300-399, oa_steps 4, CPU, float64 residuals (s1_common path).
  Checkpoints:   channels  resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt
                 barriers  Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt   (contrast 1e-4, load no-low)
  Stored records: OA-VCB 'en_s' of the matching oa-vcb record (data_seed 1, k0 2, seed 0, contrast);
                 reference 'en_s' of 'V(4,4) op.-adapted + defl. + proj.' (same data seed / contrast).
  Labels (A-only detector = oa_vcycle.near_kernel_basis; geometric = harmonic_study.geometric_island):
     island      : default detector (theta .1, tau .05) returns >= 1 vector
     near-island : default detector returns none, relaxed detector (theta .1, tau .2) returns >= 1
     non-island  : neither detector returns a vector AND geometric label negative
  Four representative cases = the FIRST test id (in order 300..399) of: channels (any); barrier island with geometric
     label positive; barrier near-island; barrier non-island.  If a category is empty it is reported empty, not redefined.
  Tolerances (relative, |new/stored - 1|):
     reference (classical, float64)       : every sample <= 1e-8
     OA-VCB mean over 100 samples         : <= 1e-5
     OA-VCB per sample, non-island        : <= 1e-4
     OA-VCB per sample, island/near-island: reported only (documented single-precision sensitivity, harmonic-study
                                            amendment v3)
  The full 100-sample sets are re-evaluated; the four cases are reported in detail.
usage: python -I ext/stage0_repro.py  (from OFW_NO/)"""
import os, sys, json, glob, time, hashlib, platform
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [ROOT, os.path.join(ROOT, 'diagnostics_cost')]
import numpy as np, scipy, torch
from s1_common import setup, load_net, apply, energy
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
from harmonic_study import geometric_island
from run_identity import CODE_HASH
torch.set_num_threads(2)

FAM = {
    'channels': dict(ckpt='resultskaggle_2026-10-06session_1/results/darcy_ours/ckpt_40ce3ac1f9_oa-vcb.pt',
                     recs='resultskaggle_2026-10-06session_1/results/darcy_ours', contrast=1e4, load='all'),
    'barriers': dict(ckpt='Results_kaggle_session 3-3b/ckpt_8266a233d9_oa-vcb.pt',
                     recs='Results_kaggle_session 3-3b', contrast=1e-4, load='no-low'),
}
REF = 'V(4,4) op.-adapted + defl. + proj.'


def sha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()


def stored(fam):
    c0 = FAM[fam]; learned = ref = None
    for f in sorted(glob.glob(os.path.join(ROOT, c0['recs'], '*.jsonl'))):
        for l in open(f):
            r = json.loads(l); c = json.loads(r['cfg'])
            if c.get('data_seed', 0) != 1 or c.get('k0') != 2 or c.get('contrast') != c0['contrast']: continue
            if r['model'] == 'oa-vcb' and c.get('seed', 0) == 0 and c.get('oa_steps', 4) == 4:
                assert learned is None, 'ambiguous oa-vcb record'; learned = (np.array(r['result']['en_s']), r['code'], os.path.basename(f))
            if r['model'] == REF:
                assert ref is None, 'ambiguous reference record'; assert r['sample_ids'] == list(range(300, 400))
                ref = (np.array(r['en_s']), r['code'], os.path.basename(f))
    assert learned is not None and ref is not None, 'missing record'
    return learned, ref


def run(fam):
    c0 = FAM[fam]; t0 = time.time()
    S = setup(3, 2, 1, contrast=c0['contrast'], load=c0['load']); te = S['te']; H = S['H']
    As = [S['A'][i] for i in te]; FS = [S['Fs'][i] for i in te]; U = S['U'][te]; UC = S['UC'][te]
    R = np.array([F - A @ uc for F, A, uc in zip(FS, As, UC)])
    D, net, st = load_net(S, os.path.join(ROOT, c0['ckpt']), te)
    C = apply(net, D, R, S['U_scale'], bs=8)
    e_net = np.array([energy(A, uc + c, u) for A, uc, c, u in zip(As, UC, C, U)])
    e_ref = []; lab = []; ngeo = []
    for j, A in enumerate(As):
        levs, A0 = oa_hierarchy(H, A, S['nl'], 4); Z = near_kernel_basis(A)
        e_ref.append(energy(A, UC[j] + classical_vcycle_proj(levs, A0, R[j], Z=Z, nu=4), U[j]))
        zr = near_kernel_basis(A, theta=0.1, tau=0.2).shape[1]; g = geometric_island(H, S['a'][te[j]])
        lab.append('island' if Z.shape[1] else ('near-island' if zr else ('non-island' if not g else 'other')))
        ngeo.append(bool(g))
    (s_net, code_net, f_net), (s_ref, code_ref, f_ref) = stored(fam)
    return dict(family=fam, ids=[int(i) for i in te], e_net=e_net.tolist(), e_ref=list(map(float, e_ref)),
                stored_net=s_net.tolist(), stored_ref=s_ref.tolist(), labels=lab, geometric=ngeo,
                ckpt=c0['ckpt'], ckpt_sha256=sha(os.path.join(ROOT, c0['ckpt'])), records=dict(net=f_net, ref=f_ref),
                record_code=dict(net=code_net, ref=code_ref), seconds=time.time() - t0)


if __name__ == '__main__':
    env = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, torch=torch.__version__,
               cpu=next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '?'),
               threads=torch.get_num_threads(), code_hash=CODE_HASH, script_sha256=sha(__file__))
    out = dict(env=env, runs=[run(f) for f in ('channels', 'barriers')])
    json.dump(out, open(os.path.join(HERE, 'stage0_repro.json'), 'w'), indent=1)
    print(json.dumps(env))
    for r in out['runs']:
        en, sn, er, sr = (np.array(r[k]) for k in ('e_net', 'stored_net', 'e_ref', 'stored_ref'))
        lab = np.array(r['labels']); dn = np.abs(en / sn - 1); dr = np.abs(er / sr - 1)
        print(f"\n{r['family']}: {r['seconds']:.0f} s; labels {dict(zip(*np.unique(lab, return_counts=True)))}")
        print(f"  reference  max rel diff {dr.max():.2e}   (gate 1e-8: {'PASS' if dr.max() <= 1e-8 else 'FAIL'})")
        print(f"  OA-VCB mean new {en.mean():.7e} stored {sn.mean():.7e} rel {abs(en.mean() / sn.mean() - 1):.2e} "
              f"(gate 1e-5: {'PASS' if abs(en.mean() / sn.mean() - 1) <= 1e-5 else 'FAIL'})")
        ni = lab == 'non-island' if r['family'] == 'barriers' else np.ones(len(lab), bool)
        print(f"  OA-VCB per-sample max rel diff, gated subset ({ni.sum()}): {dn[ni].max():.2e} "
              f"(gate 1e-4: {'PASS' if dn[ni].max() <= 1e-4 else 'FAIL'}); island/near-island max {dn[~ni].max() if (~ni).any() else float('nan'):.2e}")
        G = np.exp(np.mean(np.log(er / en))); print(f"  paired G (reference/OA-VCB) {G:.4f}; W {np.mean(er > en):.2f}")
        cats = ['any'] if r['family'] == 'channels' else ['island', 'near-island', 'non-island']
        for c in cats:
            idx = [j for j in range(len(lab)) if c == 'any' or (lab[j] == c and (c != 'island' or r['geometric'][j]))]
            if not idx: print(f"  case {c}: none in 300-399"); continue
            j = idx[0]
            print(f"  case {c:<11} id {r['ids'][j]}: OA-VCB {en[j]:.6e} (stored {sn[j]:.6e}, rel {dn[j]:.1e}); "
                  f"reference {er[j]:.6e} (stored {sr[j]:.6e}, rel {dr[j]:.1e})")
