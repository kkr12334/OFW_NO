"""10 Oct: classical comparators for Session 5's k0 = 0 runs (no externally supplied coarse solution: every method starts
from u = 0), Darcy L3 perforated, channels 1e4, HELD-OUT data seeds (rng 1000 + seed), test ids 300-399.  Replaces the
4 Oct k0_0_classical.py, which used the development pool (rng 1000) and model-mean thresholds.
Same information as oa-vcb at k0 = 0 (the operator, its operator-adapted hierarchy with 3 levels, the exact 28-node
coarsest solve -- the network also contains that internal coarsest solve):
  * one classical op.-adapted V(4,4)-cycle from zero;
  * the classical member of oa-vcb's class (the same cycle + near-kernel deflation + Galerkin output projection);
  * PCG from zero preconditioned by that V-cycle, and two-grid PCG from zero (energy-error history, 0-12 its).
Per-sample records are written to k0_0_classical_seed{s}.jsonl (sample_ids stored) so that the Session 5 OA-VCB records
(run keys 74711e0cb3 / 962c1c2e4f / e86afa9d01) can be paired as soon as they are available:
  python diagnostics_cost/k0_0_classical_heldout.py seed [session5_results.jsonl]
With a results file, also prints the paired member / oa-vcb statistics and PCG iterations to oa-vcb's error."""
import sys, os, json, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, energy
from scipy.sparse.linalg import factorized
from oa_vcycle import oa_hierarchy, classical_vcycle, classical_vcycle_proj, near_kernel_basis
DS = int(sys.argv[1]); RES = sys.argv[2] if len(sys.argv) > 2 else None
S = setup(3, 0, DS); H, F, L = S['H'], S['F'], S['L']; te = S['te']
rows = {'V(4,4) op.-adapted, from 0': [], 'V(4,4) op.-adapted + defl. + proj., from 0': []}
hv, ht = [], []
for i in te:
    A = S['A'][i]; u = S['U'][i]; levs, A0 = oa_hierarchy(H, A, S['nl'], 4)
    rows['V(4,4) op.-adapted, from 0'].append(energy(A, classical_vcycle(levs, A0, F), u))
    rows['V(4,4) op.-adapted + defl. + proj., from 0'].append(energy(A, classical_vcycle_proj(levs, A0, F, Z=near_kernel_basis(A), nu=4), u))
    Pt = H.P[L].tocsr(); sc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal()
    def tg(r):
        y = (2 / 3) * dinv * r; y = y + Pt @ sc(Pt.T @ (r - A @ y)); return y + (2 / 3) * dinv * (r - A @ y)
    for M, hist in ((lambda r: classical_vcycle(levs, A0, r), hv), (tg, ht)):
        x = np.zeros_like(F); r = F.copy(); z = M(r); p = z.copy(); rz = r @ z; h = [1.0]
        for _ in range(12):
            Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
            z = M(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2; h.append(energy(A, x, u))
        hist.append(h)
hv, ht = np.array(hv), np.array(ht)
for j in range(1, 13):
    rows[f'PCG + op.-adapted V(4,4), {j} its, from 0'] = hv[:, j]
    rows[f'two-grid PCG, {j} its, from 0'] = ht[:, j]
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'k0_0_classical_seed{DS}.jsonl')
with open(out, 'w') as fh:
    for k, v in rows.items():
        fh.write(json.dumps(dict(data_seed=DS, k0=0, level=3, model=k, sample_ids=[int(i) for i in te], en_s=list(map(float, v)))) + '\n')
print(f"held-out data seed {DS}, k0 0, {len(te)} test samples (ids {te[0]}-{te[-1]}); per-sample records -> {os.path.basename(out)}")
for k in list(rows)[:2]: print(f"  {k:<48} mean energy {np.mean(rows[k]):.4e}")
for j in (1, 2, 3, 4, 6):
    print(f"  PCG + V(4,4) from 0, {j} its: {hv[:, j].mean():.4e}    two-grid PCG from 0, {j} its: {ht[:, j].mean():.4e}")
if RES:
    for l in open(RES):
        r = json.loads(l); c = json.loads(r['cfg'])
        if c.get('k0') != 0 or c.get('data_seed', 0) != DS or r['model'] not in ('oa-vcb', 'oa-ll-op', 'gnn-op'): continue
        e = np.array(r['result']['en_s']); m = np.array(rows['V(4,4) op.-adapted + defl. + proj., from 0'])
        lr = np.log(m / e); b = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr)))
        ci = np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5]))
        hit_v = [int(np.nonzero(h <= t)[0][0]) if (h <= t).any() else 13 for h, t in zip(hv, e)]
        hit_t = [int(np.nonzero(h <= t)[0][0]) if (h <= t).any() else 13 for h, t in zip(ht, e)]
        print(f"  {r['model']:<9} mean {e.mean():.4e}: member / model G {np.exp(lr.mean()):.2f} [{ci[0]:.2f}, {ci[1]:.2f}], W {np.mean(m > e):.2f}; "
              f"PCG+V(4,4) its to match median {np.median(hit_v):.0f} ({min(hit_v)}-{max(hit_v)}), two-grid PCG {np.median(hit_t):.0f} ({min(hit_t)}-{max(hit_t)})")
