"""Harmonic enrichment of the frozen OA-VCB output -- implementation of HARMONIC_STUDY_DESIGN_2026-10-10.md (protocol v2).
Barriers (contrast 1e-4, --load no-low), L3, k0 2.  Modes:
  gate             reproduce the stored per-sample OA-VCB energies of seeds 1-3 (test ids 300-399)
  dev              set D (seed-1 test set): E0-E4, rho in {1, 2, 4, 8}; prints the rho* rule
  eval  RHO        set R (seeds 2-3 test sets), frozen rho*
  conf  RHO        set C (100 fresh operators per checkpoint: rng 9002 / 9003), frozen rho*
Per-sample records go to harmonic_<mode>.json next to this file."""
import sys, os, json, time, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
from scipy.sparse.linalg import splu
from scipy.sparse.csgraph import connected_components
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
torch.set_num_threads(2)
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
CK = {1: 'ckb/ckpt_8266a233d9_oa-vcb.pt', 2: 'ckb/ckpt_8063a3449e_oa-vcb.pt', 3: 'ckb/ckpt_62400ed25c_oa-vcb.pt'}
REC = {1: 's3', 2: 's7', 3: 's7'}
RHOS = (1, 2, 4, 8)
pc = time.perf_counter


def stored_energies(ds):
    import glob
    for f in glob.glob(os.path.join(ROOT, REC[ds], '*.jsonl')):
        for l in open(f):
            r = json.loads(l); c = json.loads(r['cfg'])
            if r['model'] == 'oa-vcb' and c['contrast'] < 1 and c.get('data_seed', 0) == ds and c.get('oa_steps', 4) == 4:
                return np.array(r['result']['en_s'])
    raise SystemExit(f'no stored record for seed {ds}')


def geometric_island(H, a):
    """independent label: a conductive component (a >= 0.5) touching no Dirichlet node (barrier_islands.py)."""
    cond = H.keep & (np.asarray(a) >= 0.5); el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M.T @ M).tocsr(); used = np.unique(el.ravel()); n_, lab = connected_components(G[used][:, used], directed=False)
    dir_nodes = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[H.L])
    touch = np.zeros(n_, bool); touch[lab[np.isin(used, dir_nodes)]] = True
    return bool((~touch).any())


def rings(Adj, I, rho):
    m = np.zeros(Adj.shape[0], bool); m[I] = True
    for _ in range(rho):
        m = m | (Adj @ m.astype(float) > 0)
    return m


def harmonic_col(A, I, patch):
    """z = 1 on I, A-harmonic on patch \\ I with zero values outside the patch."""
    z = np.zeros(A.shape[0]); z[I] = 1.0
    F = np.nonzero(patch & (z == 0))[0]
    if len(F):
        z[F] = splu(A[F][:, F].tocsc()).solve(-(A[F][:, I] @ np.ones(len(I))))
    return z


def project(A, Z, r, e):
    """sequential A-orthogonal correction onto span Z with the frozen numerics; returns d and diagnostics."""
    if Z is None or Z.shape[1] == 0:
        return np.zeros_like(r), dict(nominal=0, retained=0, gain=0.0, gain_check=0.0, cond_raw=1.0, cond_scaled=1.0)
    AZ = A @ Z; nrm = np.sqrt(np.einsum('ij,ij->j', Z, AZ)); keep = nrm > 0
    Zn = Z[:, keep] / nrm[keep]; AZn = AZ[:, keep] / nrm[keep]
    Gs = Zn.T @ AZn; Gs = 0.5 * (Gs + Gs.T)
    # greedy pivoted Cholesky on the scaled Gram (ties by column order), stop at pivot < 1e-10
    m = Gs.shape[0]; diag = np.diag(Gs).copy(); Lc = np.zeros((m, m)); piv = []
    for k in range(m):
        j = int(np.argmax(np.where(np.isin(np.arange(m), piv), -np.inf, diag)))
        if diag[j] < 1e-10: break
        piv.append(j); l = (Gs[:, j] - Lc[:, :k] @ Lc[j, :k]) / np.sqrt(diag[j]); Lc[:, k] = l; diag = diag - l ** 2
    piv = sorted(piv); Zr = Zn[:, piv]; AZr = AZn[:, piv]; G = Zr.T @ AZr
    y = np.linalg.solve(G, Zr.T @ r); d = Zr @ y
    gain = float(2 * d @ r - d @ (A @ d)); gc = float(e @ (A @ e) - (e - d) @ (A @ (e - d)))
    Graw = (Z[:, keep][:, piv]).T @ (AZ[:, keep][:, piv])
    return d, dict(nominal=int(Z.shape[1]), retained=len(piv), gain=gain, gain_check=gc,
                   cond_raw=float(np.linalg.cond(Graw)), cond_scaled=float(np.linalg.cond(G)))


def run_set(ds, S, idx, rhos, e3_ids, label):
    H = S['H']; D, net, st = load_net(S, os.path.join(ROOT, CK[ds]), idx)
    R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in idx])
    C = apply(net, D, R, S['U_scale'])
    out = []
    for j, i in enumerate(idx):
        A = S['A'][i]; u = S['U'][i]; uc = S['UC'][i]; r0 = R[j]; nu2 = float(u @ (A @ u))
        levs, A0 = oa_hierarchy(H, A, S['nl'], 4); t0 = pc(); Zi = near_kernel_basis(A); t_det = pc() - t0
        xs = {'oa-vcb': uc + C[j], 'member': uc + classical_vcycle_proj(levs, A0, r0, Z=Zi, nu=4), 'u_c': uc.copy()}
        Adj = (abs(A) > 0).astype(float).tocsr(); Adj.setdiag(0); Adj.eliminate_zeros()
        isl = [np.nonzero(Zi[:, k])[0] for k in range(Zi.shape[1])]
        rec = dict(id=int(i), seed=ds, set=label, n_islands=len(isl), island_sizes=[len(I) for I in isl],
                   geometric_island=geometric_island(H, S['a'][i]), sets={})
        spaces = {'E0': None, 'E1': Zi}; tbuild = {}
        for rho in rhos:
            t0 = pc(); cols = []; pcols = []; psz = []
            for I in isl:
                patch = rings(Adj, I, rho); psz.append(int(patch.sum()))
                cols.append(harmonic_col(A, I, patch)); pcols.append(patch.astype(float))
            tbuild[f'E2_{rho}'] = pc() - t0
            spaces[f'E2_{rho}'] = np.column_stack([Zi] + cols) if isl else Zi
            spaces[f'E4_{rho}'] = np.column_stack([Zi] + pcols) if isl else Zi
            rec[f'patch_sizes_{rho}'] = psz
        if int(i) in e3_ids and isl:
            t0 = pc(); allm = np.ones(A.shape[0], bool)
            spaces['E3'] = np.column_stack([Zi] + [harmonic_col(A, I, allm) for I in isl]); tbuild['E3'] = pc() - t0
        for pname, x in xs.items():
            e = u - x; r = S['Fs'][i] - A @ x; e2 = float(e @ (A @ e)); res = {}
            for sname, Z in spaces.items():
                t0 = pc(); d, info = project(A, Z, r, e); info['t_apply'] = pc() - t0
                ea = e - d; info['err'] = float(np.sqrt(max(ea @ (A @ ea), 0) / nu2)); info['err0'] = float(np.sqrt(e2 / nu2))
                info['kappa'] = info['gain'] / e2 if e2 > 0 else 0.0
                res[sname] = info
            rec['sets'][pname] = res
        rec['t_detect'] = t_det; rec['t_build'] = tbuild
        out.append(rec)
    return out, st


def summarize(recs, rho_list, who='oa-vcb'):
    isl = [r for r in recs if r['n_islands'] > 0]
    def col(s, rs=isl, p=who): return np.array([r['sets'][p][s]['err'] for r in rs if s in r['sets'][p]])
    e0 = col('E0'); print(f"  {who}: island samples {len(isl)}, E0 mean {e0.mean():.4f}; E1 mean {col('E1').mean():.4f} "
                          f"(max rel gain {max(abs(r['sets'][who]['E1']['kappa']) for r in isl):.1e})")
    for rho in rho_list:
        e2, e4 = col(f'E2_{rho}'), col(f'E4_{rho}')
        Rc = 1 - e2.mean() / e0.mean(); G24 = np.exp(np.mean(np.log(e4 / e2)))
        kap = np.array([r['sets'][who][f'E2_{rho}']['kappa'] for r in isl])
        print(f"    rho {rho}: E2 mean {e2.mean():.4f} (R_cap {Rc:+.3f})  E4 mean {e4.mean():.4f}  G24 {G24:.3f}  kappa quartiles "
              f"{np.percentile(kap, 25):.3f} / {np.median(kap):.3f} / {np.percentile(kap, 75):.3f}")


if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'gate':
        for ds in (1, 2, 3):
            S = setup(3, 2, ds, contrast=1e-4, load='no-low'); te = S['te']
            D, net, st = load_net(S, os.path.join(ROOT, CK[ds]), te)
            R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te]); C = apply(net, D, R, S['U_scale'])
            e = np.array([energy(S['A'][i], S['UC'][i] + c, S['U'][i]) for i, c in zip(te, C)]); es = stored_energies(ds)
            print(f"seed {ds} ({os.path.basename(CK[ds])}, code {st['code']}, cfg data_seed {json.loads(st['cfg']).get('data_seed')}): "
                  f"mean {e.mean():.6f} stored {es.mean():.6f}; max per-sample rel diff {np.max(np.abs(e / es - 1)):.2e}  "
                  f"-> {'PASS' if np.max(np.abs(e / es - 1)) < 1e-4 else 'FAIL'}", flush=True)
    elif mode == 'dev':
        S = setup(3, 2, 1, contrast=1e-4, load='no-low'); te = list(S['te'])
        isl_ids = [i for i in te if near_kernel_basis(S['A'][i]).shape[1] > 0][:20]
        recs, st = run_set(1, S, te, RHOS, set(isl_ids), 'D')
        json.dump(recs, open(os.path.join(HERE, 'harmonic_dev.json'), 'w'))
        summarize(recs, RHOS); summarize(recs, RHOS, 'member')
        isl = [r for r in recs if r['n_islands'] > 0]
        m = {rho: np.mean([r['sets']['oa-vcb'][f'E2_{rho}']['err'] for r in isl]) for rho in RHOS}
        best = min(m.values()); rs = min(rho for rho in RHOS if m[rho] <= 1.1 * best)
        print(f"rho* rule (smallest rho within 10% of the best mean island error): {rs}   (means {m})")
    elif mode in ('eval', 'conf'):
        rho = int(sys.argv[2]); allrecs = []
        for ds in (2, 3):
            if mode == 'eval':
                S = setup(3, 2, ds, contrast=1e-4, load='no-low'); idx = list(S['te']); lab = 'R'
            else:
                Sd = setup(3, 2, ds, contrast=1e-4, load='no-low')             # the checkpoint's training U_scale
                S = setup(3, 2, ds, n=100, contrast=1e-4, load='no-low', rng_seed=9000 + ds); S['U_scale'] = Sd['U_scale']
                idx = list(range(100)); lab = 'C'
            isl_ids = [i for i in idx if near_kernel_basis(S['A'][i]).shape[1] > 0][:20]
            recs, st = run_set(ds, S, idx, (rho,), set(isl_ids), lab); allrecs += recs
            print(f"seed {ds} ({lab}):"); summarize(recs, (rho,)); summarize(recs, (rho,), 'member')
        json.dump(allrecs, open(os.path.join(HERE, f'harmonic_{mode}.json'), 'w'))
        print("pooled:"); summarize(allrecs, (rho,)); summarize(allrecs, (rho,), 'member')
