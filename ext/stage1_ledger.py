"""Operation ledger per one pass (amendment 2, B3).  Structure only; output runs/stage1_ledger.json.
Conventions: MAC = one multiply + one add; a diagonal scaling, division or vector update of length n counts n; a dot
product counts n and one scalar global reduction; GELU evaluations, nonlinear global reductions (max, rms, median) and
bias additions are listed separately (bias additions are also included in MACs); stored sparse entries (including
structural zeros) are what a scipy SpMV executes, while the network uses its own pattern (fine level: 19 926 off-diagonal
entries).  Terms per pass: initial residual, wrapper (deflation x0 with k near-kernel columns, b = r - A x0, Galerkin output
over W = [x, xc, xp, Z]), smoothers, cycle residuals, transfers, coarsest solve; method-specific set-up separately; common
set-up (operator-adapted hierarchy, near-kernel basis, coarse factorisation) is reported as measured time only (timing v2).
Totals are also given as 'fine-SpMV-equivalent arithmetic' = MACs / fine nnz (an arithmetic unit, not SpMV calls or time).
Cross-check: SpMV-equivalents per level of the cycle part equal the counters recorded in runs/confirm_C_*.json."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup
from oa_vcycle import OAData, near_kernel_basis
from stage1_core import Hier
C_, NU = 15, 4
S = setup(3, 2, 0, rng_seed=2026101601); i0 = int(S['te'][0]); A = S['A'][i0]; H = S['H']; nl = S['nl']; hr = Hier(H, A, nl)
D = OAData(H, [S['a'][i0]], nl, 4, 'cpu', H.level_coords, near_kernel=True, A_list=[A])
lev = [dict(n=int(Ak.shape[0]), nnz=int(Ak.nnz), m_net=int(len(D.lv[l]['off'])), nnzX=int(X.nnz), nc=int(X.shape[1])) for l, (Ak, X) in enumerate(hr.levs)]
n0f, nnz0 = lev[0]['n'], lev[0]['nnz']; LU = int(hr.lu.L.nnz + hr.lu.U.nnz); NA0 = hr.A0.shape[0]

# ---------------- smoother costs per call: (MACs, SpMV-equivalents, dots)
def sm_cost(name, n, nnz):
    if name in ('CHEB', 'CHEB-T'): return 4 * nnz + 22 * n, 4, 0
    if name == 'POLY': return 3 * nnz + 19 * n, 3, 0
    if name.startswith('PCG'):
        k = int(name[4:-1]) if '(' in name else 4; return k * (nnz + 6 * n) + 2 * n, k, 2 * k + 1
    if name == 'SD': return 4 * (nnz + 5 * n), 4, 8
    if name == 'MR': return 4 * (nnz + 7 * n), 4, 8
    if name == 'SGS': return 4 * (nnz + n), 4, 0
    if name == 'SGS-PCG': return 4 * (3 * nnz + 7 * n) + 2 * nnz + 3 * n, 14, 9
    raise KeyError(name)
def learned_call(l):
    n, m = lev[l]['n'], lev[l]['m_net']; C = C_
    enc = n * (2 * 2 * C + 2 * C * C + 2 * C) + n * (2 * C + C)              # enc MLP + enc_lin; bias adds (2C + C)
    step = n * (C * C + 2 * C * C + 4 * C * C) + m * ((2 * C + 4) * C + C * C) + m * C * 2 + n * C * 3 + m * 2 * C + n * 2 * C
    #      V, Lw, upd MLP (nodes)                 | msg MLP (edges)          | opa w*V gather+add | /deg, s+ds, concat | msg bias | upd bias
    dec = n * (C + C * C + C) + n * (C + 1)                                    # dec_lin + dec MLP; bias adds
    norm = 8 * n + m                                                           # diagonal gather, z = b/dg, max, (z/mz)^2 mean, lsd, y*sig; W_ij = A_ij/A_ii
    gelu = n * 2 * C + NU * (m * C + n * C) + n * C
    return dict(MAC=enc + NU * step + dec + norm, GELU=gelu, nonlinear_reductions=3, dense_nodes=n * (2 * 2 * C + 2 * C * C + 2 * C) + NU * n * 7 * C * C + n * (C + C * C + C),
                dense_edges=NU * m * ((2 * C + 4) * C + C * C))

# ---------------- cycle (V or K2) costs per pass, mirroring stage1_core.Hier.cycle / kcycle
def vcycle(l, sm, learned=False):
    n, nnz, nnzX, nc = lev[l]['n'], lev[l]['nnz'], lev[l]['nnzX'], lev[l]['nc']
    t = dict(smoother=0, residual=0, transfer=0, coarsest=0, vector=0, spmv=[0] * nl, dots=0)
    if learned: c = learned_call(l); t['smoother'] += 2 * c['MAC']
    else:
        mac, sp_, dt = sm_cost(sm, n, nnz); t['smoother'] += 2 * mac; t['spmv'][l] += 2 * sp_; t['dots'] += 2 * dt
    t['residual'] += 2 * nnz; t['spmv'][l] += 2; t['vector'] += 3 * n + (2 * n if l > 0 else 0)   # r, x+xc, rhs-.., (x+xc+xp)
    t['transfer'] += 2 * nnzX + nc
    if l + 1 < nl:
        sub = vcycle(l + 1, sm, learned)
        for k in t: t[k] = [a + b for a, b in zip(t[k], sub[k])] if k == 'spmv' else t[k] + sub[k]
    else: t['coarsest'] += NA0 * NA0 if learned else LU
    return t
def kcycle(l, sm, gamma=2):
    n, nnz, nnzX, nc = lev[l]['n'], lev[l]['nnz'], lev[l]['nnzX'], lev[l]['nc']
    mac, sp_, dt = sm_cost(sm, n, nnz)
    t = dict(smoother=2 * mac, residual=2 * nnz, transfer=2 * nnzX + nc, coarsest=0, vector=3 * n + (2 * n if l > 0 else 0), spmv=[0] * nl, dots=2 * dt)
    t['spmv'][l] += 2 * sp_ + 2
    if l + 1 < nl:
        n1, nnz1 = lev[l + 1]['n'], lev[l + 1]['nnz']
        t['residual'] += gamma * nnz1; t['spmv'][l + 1] += gamma; t['vector'] += gamma * 4 * n1 + (gamma - 1) * 3 * n1; t['dots'] += gamma * 2 + (gamma - 1) * 2
        for _ in range(gamma):
            sub = kcycle(l + 1, sm, gamma)
            for k in t: t[k] = [a + b for a, b in zip(t[k], sub[k])] if k == 'spmv' else t[k] + sub[k]
    else: t['coarsest'] += LU
    return t
def wrapper(k):
    defl = (k * nnz0 + k * k * n0f + 2 * k * n0f + 2 * n0f + nnz0) if k else 0
    m = 3 + k; out = m * nnz0 + m * m * n0f + 2 * m * n0f + 2 * n0f + nnz0 + n0f
    return dict(initial_residual=nnz0 + n0f, deflation=defl, b_minus_Ax0=nnz0 + n0f, galerkin_output=out,
                dots=(k * (k + 1) // 2 + k + 2 if k else 0) + m * (m + 1) // 2 + m + 2)
def setup_cost(name):
    pw = lambda l: 30 * (lev[l]['nnz'] + 4 * lev[l]['n'])
    lz = lambda l: 2 * lev[l]['nnz'] + 10 * lev[l]['nnz'] + 50 * lev[l]['n'] + 2 * lev[l]['n'] * 55
    if name in ('CHEB', 'K2 (CHEB)'): return pw(1) + pw(2)
    if name in ('CHEB-T', 'POLY-L'): return sum(lz(l) for l in range(nl))
    return 0

# near-kernel column counts on set C (deflation and output width depend on k)
KS = {}
for fam, kw in (('channels', {}), ('barriers', dict(contrast=1e-4, load='no-low'))):
    Sf = S if fam == 'channels' else setup(3, 2, 0, rng_seed=2026101602, **kw)
    KS[fam] = [int(near_kernel_basis(Sf['A'][i]).shape[1]) for i in Sf['te']]
METH = {'CHEB': 'CHEB', 'CHEB-T': 'CHEB-T', 'POLY-S': 'POLY', 'POLY-L': 'POLY', 'SD': 'SD', 'MR': 'MR', 'PCG': 'PCG', 'SGS': 'SGS', 'SGS-PCG': 'SGS-PCG',
        'PCG(5)': 'PCG(5)', 'PCG(8)': 'PCG(8)', 'PCG(14)': 'PCG(14)', 'K2 (CHEB)': 'CHEB', 'OA-VCB': None}
out = dict(levels=lev, coarse=dict(n0=NA0, lu_nnz=LU), conventions=__doc__, near_kernel_columns={f: dict(mean=float(np.mean(k)), counts={str(v): k.count(v) for v in sorted(set(k))}) for f, k in KS.items()},
           learned_call=[learned_call(l) for l in range(nl)], families={})
cnt_check = {}
for fam, ks in KS.items():
    rows = {}
    for nm, sm in METH.items():
        cyc = kcycle(0, sm) if nm.startswith('K2') else vcycle(0, sm, learned=(nm == 'OA-VCB'))
        wr = [wrapper(k) for k in ks]; wmean = {key: float(np.mean([w[key] for w in wr])) for key in wr[0]}
        terms = dict(initial_residual=wmean['initial_residual'], wrapper=wmean['deflation'] + wmean['b_minus_Ax0'] + wmean['galerkin_output'],
                     smoother=cyc['smoother'], cycle_residuals=cyc['residual'], transfers=cyc['transfer'], coarsest=cyc['coarsest'], cycle_vector_ops=cyc['vector'])
        tot = float(sum(terms.values())); st = setup_cost(nm)
        rows[nm] = dict(terms=terms, per_pass_MAC=tot, per_pass_fine_spmv_equiv=tot / nnz0, setup_MAC=st, setup_fine_spmv_equiv=st / nnz0,
                        cycle_spmv_per_level=cyc['spmv'], scalar_dots=cyc['dots'] + wmean['dots'])
    oa = rows['OA-VCB']['per_pass_MAC']
    for nm, r in rows.items():
        r['OA_VCB_over_method'] = oa / r['per_pass_MAC']; r['OA_VCB_over_method_incl_setup'] = oa / (r['per_pass_MAC'] + r['setup_MAC'])
    rows['OA-VCB']['GELU_per_pass'] = 2 * sum(c['GELU'] for c in out['learned_call']); rows['OA-VCB']['nonlinear_global_reductions_per_pass'] = 2 * nl * 3
    out['families'][fam] = rows
    rec = json.load(open(os.path.join(HERE, 'runs', f'confirm_C_{fam}.json')))['work']
    for nm in ('CHEB', 'PCG', 'SD', 'MR', 'POLY-S', 'SGS', 'SGS-PCG', 'K2 (CHEB)'):
        recorded = np.mean([w['matvec'] for w in rec[nm]], 0).tolist(); cnt_check[f'{fam} {nm}'] = dict(ledger=rows[nm]['cycle_spmv_per_level'], counters=recorded,
                                                                                                       equal=bool(np.allclose(rows[nm]['cycle_spmv_per_level'], recorded)))
out['counter_cross_check'] = cnt_check
json.dump(out, open(os.path.join(HERE, 'runs', 'stage1_ledger.json'), 'w'), indent=1)
print('levels', lev, 'LU nnz', LU, 'near-kernel k:', {f: v['counts'] for f, v in out['near_kernel_columns'].items()})
print('counter cross-check (cycle SpMV-equivalents per level):', all(v['equal'] for v in cnt_check.values()), [k for k, v in cnt_check.items() if not v['equal']])
for fam, rows in out['families'].items():
    print(f'\n== {fam}')
    for nm, r in rows.items():
        t = r['terms']
        print(f"  {nm:<10} total {r['per_pass_MAC'] / 1e6:8.3f} M ({r['per_pass_fine_spmv_equiv']:7.1f} fine-SpMV-eq)  smoother {t['smoother'] / 1e6:7.3f}  wrapper {t['wrapper'] / 1e6:.3f}  "
              f"residuals {t['cycle_residuals'] / 1e6:.3f}  transfers {t['transfers'] / 1e6:.3f}  setup {r['setup_MAC'] / 1e6:.3f}  OA/this {r['OA_VCB_over_method']:6.0f}")
