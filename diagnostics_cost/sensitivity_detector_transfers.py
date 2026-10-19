"""11 Oct 2026 (independent reviews DeepSeek / Gemini; open TODO of review I): sensitivity of two fixed algorithmic choices,
evaluated with the CLASSICAL REFERENCE (no retraining; the learned models are not re-evaluated here).
  A. near-kernel detector thresholds theta (strength of connection) and tau (Rayleigh-quotient cut-off), on the barrier test
     sets (contrast 1e-4, load no-low, k0 = 2, L3), data seeds 0-3: recall / precision against the coefficient-geometry
     ground truth (as mathcheck/check_oa_vcp.py, part 3b) and the reference's mean relative energy error (all test samples
     and the island samples) with that Z.
  B. number m of Jacobi steps in the transfers (m = 0: geometric), channels and barriers, data seed 1: reference energy
     error, paired geometric-mean ratio to m = 4, operator complexity sum_k nnz(A_k) / nnz(A_L), median hierarchy build time.
usage: python diagnostics_cost/sensitivity_detector_transfers.py [A|B|AB]"""
import sys, os, time, json, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.csgraph import connected_components
from s1_common import setup
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
WHICH = sys.argv[1] if len(sys.argv) > 1 else 'AB'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sensitivity_detector_transfers.json')
en = lambda A, x, u: float(np.sqrt(((x - u) @ (A @ (x - u))) / (u @ (A @ u))))
res = json.load(open(OUT)) if os.path.exists(OUT) else {}

def coef_islands(H, a):
    L = H.L; dirn = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L])
    pos = -np.ones(len(H.coords), int); pos[H.nodes[L]] = np.arange(len(H.nodes[L]))
    cond = H.keep & (np.asarray(a).ravel() >= 0.5); el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M.T @ M).tocsr(); used = np.unique(el.ravel()); n_, lab = connected_components(G[used][:, used], directed=False)
    touch = np.zeros(n_, bool); touch[lab[np.isin(used, dirn)]] = True
    return [set(pos[used[lab == c]]) for c in np.nonzero(~touch)[0]]

if 'A' in WHICH:
    TH, TAU = (0.05, 0.1, 0.2), (0.02, 0.05, 0.1)
    for ds in (0, 1, 2, 3):
        S = setup(3, 2, ds, contrast=1e-4, load='no-low'); H = S['H']
        cache = {}
        for i in S['te']:
            A = S['A'][i]; cache[i] = (oa_hierarchy(H, A, S['nl'], 4), coef_islands(H, S['a'][i]), S['Fs'][i] - A @ S['UC'][i])
        isl_ids = [i for i in S['te'] if cache[i][1]]
        for th in TH:
            for tau in TAU:
                tp = fp = cov = tot = nz = 0; E = {}
                for i in S['te']:
                    A = S['A'][i]; (levs, A0), ci, r = cache[i]; Z = near_kernel_basis(A, theta=th, tau=tau); nz += Z.shape[1]
                    for s_ in ci:
                        z = np.zeros(A.shape[0]); z[list(s_)] = 1.0; tot += 1
                        rr = z - Z @ np.linalg.lstsq(Z, z, rcond=None)[0] if Z.shape[1] else z
                        cov += np.linalg.norm(rr) / np.linalg.norm(z) < 1e-8
                    for j in range(Z.shape[1]):
                        Sj = set(np.nonzero(Z[:, j])[0])
                        if max((len(Sj & s_) / len(Sj) for s_ in ci), default=0.0) >= 0.9: tp += 1
                        else: fp += 1
                    x = S['UC'][i] + classical_vcycle_proj(levs, A0, r, Z=Z, nu=4)
                    E[int(i)] = en(A, x, S['U'][i])
                row = dict(recall=f"{cov}/{tot}", precision=f"{tp}/{tp + fp}", detected=nz,
                           ref_all=float(np.mean(list(E.values()))), ref_islands=float(np.mean([E[int(i)] for i in isl_ids])),
                           n_island_samples=len(isl_ids), en_s=[E[int(i)] for i in S['te']])
                res[f"A_ds{ds}_th{th}_tau{tau}"] = row
                print(f"A ds{ds} theta {th:<5} tau {tau:<5} recall {row['recall']:>6} precision {row['precision']:>6} "
                      f"detected {nz:3d}  reference energy: all {row['ref_all']:.4e}  island samples ({len(isl_ids)}) "
                      f"{row['ref_islands']:.4e}", flush=True)
        json.dump(res, open(OUT, 'w'))

if 'B' in WHICH:
    for tag, con, load in (('channels', 1e4, 'all'), ('barriers', 1e-4, 'no-low')):
        S = setup(3, 2, 1, contrast=con, load=load); H = S['H']; E = {}
        for m in (0, 1, 2, 4, 8):
            e_, t_, oc = [], [], []
            for i in S['te']:
                A = S['A'][i]; t0 = time.perf_counter(); levs, A0 = oa_hierarchy(H, A, S['nl'], m); t_.append(time.perf_counter() - t0)
                oc.append((sum(Ak.nnz for Ak, _ in levs) + A0.nnz) / A.nnz)
                x = S['UC'][i] + classical_vcycle_proj(levs, A0, S['Fs'][i] - A @ S['UC'][i], Z=near_kernel_basis(A), nu=4)
                e_.append(en(A, x, S['U'][i]))
            E[m] = np.array(e_)
            res[f"B_{tag}_m{m}"] = dict(ref_mean=float(E[m].mean()), op_complexity=float(np.median(oc)),
                                        build_ms=1e3 * float(np.median(t_)), en_s=E[m].tolist())
        for m in (0, 1, 2, 4, 8):
            g = float(np.exp(np.mean(np.log(E[m] / E[4])))); res[f"B_{tag}_m{m}"]['G_vs_m4'] = g
            r_ = res[f"B_{tag}_m{m}"]
            print(f"B {tag:<8} m {m}: reference energy {r_['ref_mean']:.4e}  G(m / m=4) {g:.3f}  operator complexity "
                  f"{r_['op_complexity']:.2f}  build {r_['build_ms']:.1f} ms", flush=True)
        json.dump(res, open(OUT, 'w'))
