"""Tables of the Stage 1 evidence (v2 manuscript), generated from the stored Stage 1 records; no number is typed by hand.
Run from the OFW_NO folder:   python -I manuscript/make_tables_stage1.py [root]      (root defaults to OFW_NO)
Reads   <root>/ext/runs/{confirm_C_*,d2_C_*,reserve_*,reserve_K_local_channels,stage1_ledger,timing_v2_C_*,confirm_LC_channels}.json
        and (for the original-test-set panel only) manuscript/tables/numbers.txt written by make_tables.py (13 Oct run).
Writes  manuscript/tables/tab_s1_main.tex, tab_s1_local.tex, tab_s1_brackets.tex, tab_s1_cost.tex, tab_s1_loads.tex,
        numbers_stage1.txt and MANIFEST_stage1.txt (every input file used, with SHA-256).
Every printed statistic that is also stored in a record (paired G with its pointwise bootstrap interval, K-local counts) is
recomputed from the per-operator arrays with the Stage 1 statistics (ext/stage1_stats.py: 10,000 paired bootstrap resamples,
rng 20261016) and must agree with the stored value to 1e-9; the script stops otherwise.  Original and Stage 1 test sets are
never pooled."""
import json, os, re, sys, hashlib, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE)
RUNS = os.path.join(ROOT, 'ext', 'runs')
OUT = os.path.join(HERE, 'tables'); os.makedirs(OUT, exist_ok=True)
NUM = open(os.path.join(OUT, 'numbers_stage1.txt'), 'w'); USED = {}
INITS = (0, 1, 2); FAMS = ('channels', 'barriers')


def note(s): print(s); NUM.write(s + '\n')


def J(name):
    p = os.path.join(RUNS, name)
    USED[os.path.relpath(p, ROOT).replace(os.sep, '/')] = hashlib.sha256(open(p, 'rb').read()).hexdigest()
    return json.load(open(p))


# ---- statistics (identical to ext/stage1_stats.py.paired)
B_RESAMPLES, B_SEED = 10_000, 20261016


def paired(e_x, e_y):
    e_x, e_y = np.asarray(e_x, float), np.asarray(e_y, float)
    ok = np.isfinite(e_x) & np.isfinite(e_y) & (e_x > 0) & (e_y > 0)
    lr = np.log(e_x[ok] / e_y[ok]); n = len(lr)
    idx = np.random.default_rng(B_SEED).integers(0, n, size=(B_RESAMPLES, n)); bs = np.exp(lr[idx].mean(1))
    return dict(G=float(np.exp(lr.mean())), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)),
                W=float(np.mean(e_y[ok] < e_x[ok])), n=int(n))


def agree(mine, stored, what, tol=1e-9):
    for k in ('G', 'lo', 'hi', 'W'):
        if abs(mine[k] - stored[k]) > tol: raise AssertionError(f'{what}: {k} recomputed {mine[k]!r} != stored {stored[k]!r}')


def gci(r, d=2): return f"{r['G']:.{d}f} [{r['lo']:.{d}f}, {r['hi']:.{d}f}]"


# ---- Table: set C one-pass comparisons -------------------------------------------------------------------------------
C = {f: J(f'confirm_C_{f}.json') for f in FAMS}
D2 = {f: J(f'd2_C_{f}.json') for f in FAMS}
rows = {}
for f in FAMS:
    E = C[f]['errors']
    for k in INITS:
        oa = E[f'OA-VCB init{k}']
        for m in ('CHEB', 'PCG', 'POLY-S', 'SGS-PCG'):
            r = paired(E[m], oa)
            st = C[f]['confirmatory'][f'OA-VCB init{k}']['P1'] if m == 'PCG' else C[f]['secondary'][f'{m} -> OA-VCB init{k}']
            agree(r, st, f'set C {f} init{k} {m}'); rows[(f, k, m)] = r
            note(f'[set C] {f} init{k}: G({m} -> OA-VCB) {gci(r)} W {r["W"]:.2f} n {r["n"]}')
    r = paired(D2[f]['errors_D2'], C[f]['errors']['OA-VCB init0']); rows[(f, 0, 'D2')] = r
    note(f'[set C] {f} init0: G(best per-instance polynomial candidate, D2 -> OA-VCB) {gci(r)} W {r["W"]:.2f}')
p3 = {f: paired(C[f]['errors']['CHEB'], C[f]['errors']['PCG']) for f in FAMS}
for f in FAMS:
    note(f'[set C] {f}: G(Chebyshev nu=4 -> Jacobi-PCG(4)) {gci(p3[f])}')

# original test sets (13 Oct run): per-seed Chebyshev panel, parsed from numbers.txt written by make_tables.py
orig = {}
numtxt = os.path.join(OUT, 'numbers.txt'); USED['manuscript/tables/numbers.txt'] = hashlib.sha256(open(numtxt, 'rb').read()).hexdigest()
for line in open(numtxt, encoding='utf-8'):
    m = re.match(r'\[tab_(channels|barriers)\] seed (\d): .* G ([\d.]+) \[([\d.]+), ([\d.]+)\] W ([\d.]+)', line)
    if m: orig[(m.group(1), int(m.group(2)))] = dict(G=float(m.group(3)), lo=float(m.group(4)), hi=float(m.group(5)), W=float(m.group(6)))
assert len(orig) == 6, 'original per-seed rows not found in numbers.txt'
for (f, s), r in sorted(orig.items()): note(f'[original test sets, 13 Oct] {f} data seed {s}: G(Chebyshev nu=4 -> OA-VCB) {gci(r)} W {r["W"]:.2f}')


def cell(r): return gci(r)


def tab_main():
    L = []
    L.append(r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}')
    L.append(r'\caption{One-pass energy error of the classical comparator divided by that of OA-VCB (paired geometric-mean $G$, pointwise 95\% bootstrap interval over operators; $G>1$ favours OA-VCB). '
             r'\emph{Panel A} (Stage 1 set C): 100 operators generated for the confirmation, one fixed-load network per initialisation seed; every comparator is the same cycle with only the smoothers replaced. '
             r'``Jacobi-PCG(4)\textquoteright\textquoteright{} is a four-step Jacobi-preconditioned conjugate-gradient smoother \emph{inside every smoothing call}, not the whole-system two-grid PCG solver of Section~\ref{sec:cost}. '
             r'``Polynomial candidate\textquoteright\textquoteright{} is the best per-instance fit of separate degree-three polynomials for the six smoothing calls, obtained with the exact solution and five starts; it is an upper bound on the class optimum, not a certified one. '
             r'\emph{Panel B} (original test sets, 13 Oct run, initialisation 0): 100 test operators per data seed, Chebyshev reference only. The two panels use different operators and checkpoints and are never pooled.}\label{tab:s1main}')
    L.append(r'\begin{tabular}{p{5.0cm}lll}\toprule')
    labs = (('Chebyshev $\\nu=4$ (reference of the original study)', 'CHEB'), ('Jacobi-PCG(4) smoothing calls (primary, pre-declared)', 'PCG'), ('fitted fixed polynomial, degree 3 (contained class)', 'POLY-S'))
    LABS = [(l, m) for l, m in labs]
    for f, ttl in (('channels', 'channels (contrast $10^4$)'), ('barriers', 'barriers (contrast $10^{-4}$)')):
        L.append(r'\multicolumn{4}{l}{\emph{A. Stage 1 set C, ' + ttl + r'; columns: OA-VCB initialisation seed}} \\')
        L.append(r'comparator & 0 & 1 & 2 \\\midrule')
        for lab, m in LABS:
            L.append(lab + ' & ' + ' & '.join(cell(rows[(f, k, m)]) for k in INITS) + r' \\')
        L.append(r'polynomial candidate (per instance, upper bound) & ' + cell(rows[(f, 0, 'D2')]) + r' & -- & -- \\\midrule')
    L.append(r'\multicolumn{4}{l}{\emph{B. Original test sets (13 Oct), Chebyshev $\nu=4$ only; columns: data seed}} \\')
    L.append(r'family & 1 & 2 & 3 \\\midrule')
    for f in FAMS:
        L.append(f + ' & ' + ' & '.join(cell(orig[(f, s_)]) for s_ in (1, 2, 3)) + r' \\')
    L.append(r'\bottomrule\end{tabular}\end{table}')
    open(os.path.join(OUT, 'tab_s1_main.tex'), 'w').write('\n'.join(L) + '\n')


tab_main()

# ---- Table: local Krylov-space diagnostic (reserve) -------------------------------------------------------------------
K = J('reserve_K_local_channels.json')
KS = K['summary']; KR = K['records']
LOC = {}
for k in INITS:
    for p in ('pre', 'post'):
        rec = KR[f'OA-VCB init{k}'][p]; row = {}
        for d in ('K4', 'K5', 'K8'):
            rho = np.array([r_[d]['rho'] for r_ in rec if r_[d]['rank'] == int(d[1:])]); s = KS[f'OA-VCB init{k} {p} {d}']
            assert len(rho) == s['n_full_rank'], (k, p, d)
            n99 = int((rho <= 0.99).sum()); assert n99 == s['n_rho_le_099'], (k, p, d, n99, s['n_rho_le_099'])
            q = np.quantile(rho, [.1, .5, .9]); assert np.allclose(q, s['rho_q10_50_90'], atol=1e-9, rtol=0)
            row[d] = dict(q=q, n99=n99, n=len(rho), lt1=int((rho < 1).sum()))
        LOC[(k, p)] = row
        note(f'[reserve K-local, channels] init{k} {p}: rho_K4 median {row["K4"]["q"][1]:.3f}; rho_K5 10/50/90 {row["K5"]["q"][0]:.3f}/{row["K5"]["q"][1]:.3f}/{row["K5"]["q"][2]:.3f}, '
             f'<=0.99 on {row["K5"]["n99"]}/{row["K5"]["n"]}; rho_K8 median {row["K8"]["q"][1]:.3f}, <=0.99 on {row["K8"]["n99"]}/{row["K8"]["n"]}, <1 on {row["K8"]["lt1"]}')
allk5 = [LOC[x]['K5']['n99'] for x in LOC]; allk8 = [LOC[x]['K8']['n99'] for x in LOC]
note(f'[reserve K-local, channels] K5: rho<=0.99 on {min(allk5)}-{max(allk5)} of 100 inputs per init/position; K8: {min(allk8)}-{max(allk8)} of 100')
note(f'[reserve K-local, channels] median rho_K5 pre {min(LOC[(k,"pre")]["K5"]["q"][1] for k in INITS):.3f}-{max(LOC[(k,"pre")]["K5"]["q"][1] for k in INITS):.3f}; '
     f'post {min(LOC[(k,"post")]["K5"]["q"][1] for k in INITS):.3f}-{max(LOC[(k,"post")]["K5"]["q"][1] for k in INITS):.3f}')
note(f'[reserve K-local, channels] max orthonormality defect {max(KS[x]["max_orth_defect"] for x in KS):.1e}; max |E_PCG/E_K - 1| {max(KS[x]["max_pcg_vs_K"] for x in KS):.1e}')


def tab_local():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
         r'\caption{Local Krylov-space diagnostic on 100 fresh channel operators (Stage 1 reserve set). For the input $b$ of one fine-level smoothing call, $\rho_{K_d}=E_{\mathrm{NN}}/E_{K_d}$ is the energy error of the network smoother divided by the exact energy-norm minimum over the $d$-dimensional Krylov space $K_d=\operatorname{span}\{b,D^{-1}Ab,\dots\}$ of the \emph{same} fixed input (rank-aware $A$-orthonormal basis; Jacobi-preconditioned). Values below one favour the network. '
         r'Pre- and post-smoother inputs of the same complete cycle are listed separately; $\rho_{K_5}$ is a degree control (matrix-propagation degree four plus one), not an equal-information or equal-cost control, because the network normalises by a global residual scale and uses nonlinear features. The diagnostic concerns one fixed input and does not describe complete cycles, whose later inputs differ.}\label{tab:s1local}',
         r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llllllll}\toprule',
         r'init & input & $\rho_{K_4}$ median & $\rho_{K_5}$ 10/50/90\% & $\rho_{K_5}\le0.99$ & $\rho_{K_8}$ median & $\rho_{K_8}\le0.99$ & $\rho_{K_8}<1$ \\\midrule']
    for k in INITS:
        for p in ('pre', 'post'):
            r_ = LOC[(k, p)]
            L.append(f'{k} & {p} & {r_["K4"]["q"][1]:.3f} & {r_["K5"]["q"][0]:.3f} / {r_["K5"]["q"][1]:.3f} / {r_["K5"]["q"][2]:.3f} & {r_["K5"]["n99"]}/{r_["K5"]["n"]} & '
                     f'{r_["K8"]["q"][1]:.3f} & {r_["K8"]["n99"]}/{r_["K8"]["n"]} & {r_["K8"]["lt1"]}/{r_["K8"]["n"]} \\\\')
    L.append(r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    open(os.path.join(OUT, 'tab_s1_local.tex'), 'w').write('\n'.join(L) + '\n')


tab_local()

# ---- D1 on set C (local K4 test; pre-declared diagnostic)
D1 = {f: J(f'd1_C_{f}.json')['summary'] for f in FAMS}
for f in FAMS:
    for p_ in ('pre', 'post'):
        d = D1[f][p_]; note(f'[set C D1, {f}] {p_}: rho_K4 10/50/90 {d["rho_q"][0]:.3f}/{d["rho_q"][1]:.3f}/{d["rho_q"][2]:.3f}; rho_K4 <= 0.99 on {d["n_local_separation"]}/100; rank {d["ranks"]}; max |PCG/K4 - 1| {d["pcg_vs_K_max"]:.1e}')

# ---- Table: larger Jacobi-PCG budgets (reserve) -----------------------------------------------------------------------
RES = {f: J(f'reserve_{f}.json') for f in FAMS}
BR = {}
for f in FAMS:
    E = RES[f]['errors']
    for k in INITS:
        for m in ('PCG(4)', 'PCG(5)', 'PCG(8)', 'PCG(14)', 'SGS-PCG(4)'):
            r = paired(E[m], E[f'OA-VCB init{k}']); agree(r, RES[f]['endpoints'][f'OA-VCB init{k}'][f'{m} -> OA-VCB'], f'reserve {f} init{k} {m}'); BR[(f, k, m)] = r
            note(f'[reserve] {f} init{k}: G({m} -> OA-VCB) {gci(r)} W {r["W"]:.2f}')
        b = RES[f]['endpoints'][f'OA-VCB init{k}']['bracket']; note(f'[reserve] {f} init{k}: bracket {b["lower_method"]}-{b["upper_method"]} confirmed {b["confirmed"]}')
    r = paired(E['PCG(14)'], E['SGS-PCG(4)']); note(f'[reserve] {f}: G(PCG(14) -> SGS-PCG(4)) {gci(r)}')
for f, ms in (('channels', ('PCG(8)', 'PCG(14)')), ('barriers', ('PCG(5)', 'PCG(8)'))):
    for m in ms:
        g = [BR[(f, k, m)]['G'] for k in INITS]; note(f'[reserve] {f}: G({m} -> OA-VCB) over the three initialisations {min(g):.2f}-{max(g):.2f}; every lower limit >= {min(BR[(f,k,m)]["lo"] for k in INITS):.3f}, every upper limit <= {max(BR[(f,k,m)]["hi"] for k in INITS):.3f}')
for f in FAMS:
    g = [1 / BR[(f, k, 'PCG(14)')]['G'] for k in INITS]; note(f'[reserve] {f}: PCG(14) is {min(g):.1f}-{max(g):.1f} times more accurate than OA-VCB (paired geometric mean)')


def tab_brackets():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
         r'\caption{Larger classical smoothing budgets in the complete cycle (Stage 1 reserve set; 100 fresh operators per family; generated after the amendment that fixed these endpoints). $G$ is the paired geometric-mean ratio of the energy error of the cycle with Jacobi-PCG($s$) smoothing calls ($s$ steps in each pre- and post-smoothing call, all other ingredients unchanged) to that of OA-VCB, with the pointwise 95\% bootstrap interval; values below one mean that the classical cycle is more accurate. The bracket concerns paired geometric-mean error over operators, not each operator and not an exact crossover step count. Rows within a family are different trained networks evaluated on the same 100 operators.}\label{tab:s1brackets}',
         r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{llllllll}\toprule',
         r'init & input & $\rho_{K_4}$ median & $\rho_{K_5}$ 10/50/90\% & $\rho_{K_5}\le0.99$ & $\rho_{K_8}$ median & $\rho_{K_8}\le0.99$ & $\rho_{K_8}<1$ \\\midrule']
    for k in INITS:
        for p in ('pre', 'post'):
            r_ = LOC[(k, p)]
            L.append(f'{k} & {p} & {r_["K4"]["q"][1]:.3f} & {r_["K5"]["q"][0]:.3f} / {r_["K5"]["q"][1]:.3f} / {r_["K5"]["q"][2]:.3f} & {r_["K5"]["n99"]}/{r_["K5"]["n"]} & '
                     f'{r_["K8"]["q"][1]:.3f} & {r_["K8"]["n99"]}/{r_["K8"]["n"]} & {r_["K8"]["lt1"]}/{r_["K8"]["n"]} \\\\')
    L.append(r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    open(os.path.join(OUT, 'tab_s1_local.tex'), 'w').write('\n'.join(L) + '\n')


tab_local()

# ---- D1 on set C (local K4 test; pre-declared diagnostic)
D1 = {f: J(f'd1_C_{f}.json')['summary'] for f in FAMS}
for f in FAMS:
    for p_ in ('pre', 'post'):
        d = D1[f][p_]; note(f'[set C D1, {f}] {p_}: rho_K4 10/50/90 {d["rho_q"][0]:.3f}/{d["rho_q"][1]:.3f}/{d["rho_q"][2]:.3f}; rho_K4 <= 0.99 on {d["n_local_separation"]}/100; rank {d["ranks"]}; max |PCG/K4 - 1| {d["pcg_vs_K_max"]:.1e}')

# ---- Table: larger Jacobi-PCG budgets (reserve) -----------------------------------------------------------------------
RES = {f: J(f'reserve_{f}.json') for f in FAMS}
BR = {}
for f in FAMS:
    E = RES[f]['errors']
    for k in INITS:
        for m in ('PCG(4)', 'PCG(5)', 'PCG(8)', 'PCG(14)', 'SGS-PCG(4)'):
            r = paired(E[m], E[f'OA-VCB init{k}']); agree(r, RES[f]['endpoints'][f'OA-VCB init{k}'][f'{m} -> OA-VCB'], f'reserve {f} init{k} {m}'); BR[(f, k, m)] = r
            note(f'[reserve] {f} init{k}: G({m} -> OA-VCB) {gci(r)} W {r["W"]:.2f}')
        b = RES[f]['endpoints'][f'OA-VCB init{k}']['bracket']; note(f'[reserve] {f} init{k}: bracket {b["lower_method"]}-{b["upper_method"]} confirmed {b["confirmed"]}')
    r = paired(E['PCG(14)'], E['SGS-PCG(4)']); note(f'[reserve] {f}: G(PCG(14) -> SGS-PCG(4)) {gci(r)}')
for f, ms in (('channels', ('PCG(8)', 'PCG(14)')), ('barriers', ('PCG(5)', 'PCG(8)'))):
    for m in ms:
        g = [BR[(f, k, m)]['G'] for k in INITS]; note(f'[reserve] {f}: G({m} -> OA-VCB) over the three initialisations {min(g):.2f}-{max(g):.2f}; every lower limit >= {min(BR[(f,k,m)]["lo"] for k in INITS):.3f}, every upper limit <= {max(BR[(f,k,m)]["hi"] for k in INITS):.3f}')
for f in FAMS:
    g = [1 / BR[(f, k, 'PCG(14)')]['G'] for k in INITS]; note(f'[reserve] {f}: PCG(14) is {min(g):.1f}-{max(g):.1f} times more accurate than OA-VCB (paired geometric mean)')


def tab_brackets():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
         r'\caption{Larger classical smoothing budgets in the complete cycle (Stage 1 reserve set; 100 fresh operators per family; generated after the amendment that fixed these endpoints). $G$ is the paired geometric-mean ratio of the energy error of the cycle with Jacobi-PCG($s$) smoothing calls ($s$ steps in each pre- and post-smoothing call, all other ingredients unchanged) to that of OA-VCB, with the pointwise 95\% bootstrap interval; values below one mean that the classical cycle is more accurate. The bracket concerns paired geometric-mean error over operators, not each operator and not an exact crossover step count. Rows within a family are different trained networks evaluated on the same 100 operators.}\label{tab:s1brackets}',
         r'\begin{tabular}{p{4.2cm}lll}\toprule']
    for f in FAMS:
        L.append(r'\multicolumn{4}{l}{\emph{' + f + r'; columns: OA-VCB initialisation seed}} \\')
        L.append(r'comparator & 0 & 1 & 2 \\\midrule')
        for lab, m in (('Jacobi-PCG(4)', 'PCG(4)'), ('Jacobi-PCG(5)', 'PCG(5)'), ('Jacobi-PCG(8)', 'PCG(8)'), ('Jacobi-PCG(14)', 'PCG(14)'), ('SGS-PCG(4)', 'SGS-PCG(4)')):
            L.append(lab + ' & ' + ' & '.join(gci(BR[(f, k, m)]) for k in INITS) + r' \\')
        L.append(r'bracket confirmed between & ' + ' & '.join(RES[f]['endpoints'][f'OA-VCB init{k}']['bracket']['lower_method'] + '--' + RES[f]['endpoints'][f'OA-VCB init{k}']['bracket']['upper_method'] for k in INITS) + r' \\')
        if f == 'channels': L.append(r'\midrule')
    L.append(r'\bottomrule\end{tabular}\end{table}')
    open(os.path.join(OUT, 'tab_s1_brackets.tex'), 'w').write('\n'.join(L) + '\n')


tab_brackets()

# ---- Table: cost (operation ledger and repaired timings; one platform, one run) ----------------------------------------
LED = J('stage1_ledger.json')['families']
TIM = {f: J(f'timing_v2_C_{f}.json') for f in FAMS}


def med(f, key): return float(np.median([np.median(r['raw_seconds'][key]) for r in TIM[f]['records']]) * 1e3)


def time_to_target(f, curve):
    """median over operators of the time (ms) at the first iteration whose error reaches OA-VCB's one-pass error on that
    operator; (conditional on u_c, estimated end-to-end = per-operator sum with u_c's own median).  Censored -> inf."""
    c, e = [], []
    for r in TIM[f]['records']:
        st = np.median(np.array(r['curve_stamps'][f'curve {curve}']), 0); tgt = r['errors']['1pass OA-VCB batch1']
        h = next((j for j, v in enumerate(r['curve_errors'][curve]) if v <= tgt), None)
        u = np.median(r['raw_seconds']['u_c construction']) * 1e3
        c.append(st[h] * 1e3 if h is not None else np.inf); e.append(st[h] * 1e3 + u if h is not None else np.inf)
    return float(np.median(c)), float(np.median(e)), int(np.isinf(c).sum())


T = {}
for f in FAMS:
    T[f] = dict(oa=med(f, '1pass OA-VCB batch1'), uc=med(f, 'u_c construction'), direct=med(f, 'direct'),
                one=[med(f, k) for k in TIM[f]['records'][0]['raw_seconds'] if k.startswith('1pass ') and 'OA-VCB' not in k])
    for nm, cv in (('two-grid PCG', 'two-grid PCG'), ('SA-AMG PCG', 'SA-AMG PCG')):
        T[f][nm] = time_to_target(f, cv)
    note(f'[timing v2, {f}] OA-VCB one pass (batch 1, conditional on u_c) {T[f]["oa"]:.0f} ms; classical one pass {min(T[f]["one"]):.0f}-{max(T[f]["one"]):.0f} ms; u_c {T[f]["uc"]:.1f} ms; direct {T[f]["direct"]:.1f} ms')
    for nm in ('two-grid PCG', 'SA-AMG PCG'):
        c, e, ncen = T[f][nm]; note(f'[timing v2, {f}] {nm} to OA-VCB error (offline oracle): {c:.1f} ms conditional on u_c, {e:.1f} ms estimated end-to-end; censored {ncen}/50')
note(f"[ledger] OA-VCB per pass {LED['channels']['OA-VCB']['per_pass_MAC'] / 1e6:.0f} M MAC-equivalents ({LED['channels']['OA-VCB']['per_pass_fine_spmv_equiv']:.0f} fine-SpMV-equivalents); "
     f"ratio to PCG {LED['channels']['PCG']['OA_VCB_over_method']:.0f}, to PCG(8) {LED['channels']['PCG(8)']['OA_VCB_over_method']:.0f}, to PCG(14) {LED['channels']['PCG(14)']['OA_VCB_over_method']:.0f}")
note(f"[ledger] MAC-equivalents per pass: PCG {LED['channels']['PCG']['per_pass_MAC'] / 1e6:.2f} M, PCG(8) {LED['channels']['PCG(8)']['per_pass_MAC'] / 1e6:.2f} M, PCG(14) {LED['channels']['PCG(14)']['per_pass_MAC'] / 1e6:.2f} M; OA-VCB activation evaluations {LED['channels']['OA-VCB']['GELU_per_pass'] / 1e6:.1f} M, nonlinear global reductions {LED['channels']['OA-VCB']['nonlinear_global_reductions_per_pass']}")


def tab_cost():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
         r'\caption{Complete cost of the Stage 1 comparison (one CPU, one run, Python implementations; channels unless stated; the platform differs from that of Table~\ref{tab:cost}, and the two are not combined). '
         r'\emph{Panel A}: approximate MAC-equivalent arithmetic of one pass of the complete cycle, from an operation ledger whose dense-layer and sparse-product counts agree with independent counters; it excludes the small dense solves of the Galerkin wrapper and the operator-adapted hierarchy (common to all rows) and is neither a time nor a communication volume. '
         r'\emph{Panel B}: median over 50 operators (ms) of separately timed components; ``to OA-VCB error\textquoteright{}\textquoteright{} selects, offline and per operator, the first iteration whose energy error reaches the one-pass OA-VCB error (an oracle stopping rule, not a deployable one); ``estimated end-to-end\textquoteright{}\textquoteright{} adds the median time of constructing $u_c$ per operator. The direct solve is never charged $u_c$.}\label{tab:s1cost}',
         r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lrrr}\toprule',
         r'\multicolumn{4}{l}{\emph{A. Arithmetic per pass (channels)}} \\ method & MMACs & fine-SpMV-equiv. & OA-VCB $\div$ method \\\midrule']
    for lab, key in (('Jacobi-PCG(4) smoothing', 'PCG'), ('Jacobi-PCG(8)', 'PCG(8)'), ('Jacobi-PCG(14)', 'PCG(14)'), ('SGS-PCG(4)', 'SGS-PCG'), ('Chebyshev $\\nu=4$ (with bounds set-up)', 'CHEB'), ('OA-VCB (four learned stages)', 'OA-VCB')):
        e = LED['channels'][key]
        L.append(f'{lab} & {e["per_pass_MAC"] / 1e6:.1f} & {e["per_pass_fine_spmv_equiv"]:.0f} & {e["OA_VCB_over_method"]:.0f} \\\\')
    L.append(r'\midrule\multicolumn{4}{l}{\emph{B. Time per operator, ms (median over 50)}} \\ quantity & channels & barriers & \\\midrule')
    L.append(f'OA-VCB one pass, batch 1, given $u_c$ & {T["channels"]["oa"]:.0f} & {T["barriers"]["oa"]:.0f} & \\\\')
    L.append(f'classical one pass, any smoother tested, given $u_c$ (range) & {min(T["channels"]["one"]):.0f}--{max(T["channels"]["one"]):.0f} & {min(T["barriers"]["one"]):.0f}--{max(T["barriers"]["one"]):.0f} & \\\\')
    for nm in ('two-grid PCG', 'SA-AMG PCG'):
        a, b = T['channels'][nm], T['barriers'][nm]
        L.append(f'{nm} to OA-VCB error, given $u_c$ / estimated end-to-end & {a[0]:.1f} / {a[1]:.1f} & {b[0]:.1f} / {b[1]:.1f} & \\\\')
    L.append(f'sparse direct solve (to solver accuracy) & {T["channels"]["direct"]:.1f} & {T["barriers"]["direct"]:.1f} & \\\\')
    L.append(r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    open(os.path.join(OUT, 'tab_s1_cost.tex'), 'w').write('\n'.join(L) + '\n')


tab_cost()

# ---- Table: load challenge against stronger smoothers (set LC) ---------------------------------------------------------
LC = J('confirm_LC_channels.json'); E = LC['errors']; cl = np.asarray(LC['data']['operators'])
assert len(E['PCG']) == 200 and len(set(cl.tolist())) == 50


def paired_clustered(e_x, e_y, clusters):
    e_x, e_y, c = np.asarray(e_x, float), np.asarray(e_y, float), np.asarray(clusters)
    ok = np.isfinite(e_x) & np.isfinite(e_y) & (e_x > 0) & (e_y > 0); lr = np.log(e_x[ok] / e_y[ok]); c = c[ok]; u = np.unique(c)
    s = np.array([lr[c == k].sum() for k in u]); m = np.array([(c == k).sum() for k in u])
    idx = np.random.default_rng(B_SEED).integers(0, len(u), size=(B_RESAMPLES, len(u))); bs = np.exp(s[idx].sum(1) / m[idx].sum(1))
    return dict(G=float(np.exp(lr.mean())), lo=float(np.quantile(bs, .025)), hi=float(np.quantile(bs, .975)), W=float(np.mean(e_y[ok] < e_x[ok])), n=int(ok.sum()))


LD = {}
for model, tag in (('OA-VCB init0', ''), ('OA-VCB init0 (random-load model)', ' (random-load model)')):
    for m in ('CHEB', 'PCG', 'POLY-S', 'SGS-PCG'):
        r = paired_clustered(E[m], E[model], cl); agree(r, LC['secondary'][f'{m} -> OA-VCB init0{tag}'], f'LC {m}{tag}'); LD[(model, m)] = r
        note(f'[set LC, channels, 200 load--operator pairs from 50 operators] G({m} -> {model}) {gci(r)} W {r["W"]:.2f}')


def tab_loads():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
         r'\caption{Loads other than the training load, against stronger classical smoothers (Stage 1 set LC: 50 fresh channel operators, four off-training load families---$f=-1$, smooth loads of both signs, a localised source and a random-load draw---200 load--operator pairs). Paired geometric-mean ratio of the comparator cycle\textquoteright{}s energy error to the network\textquoteright{}s, with a 95\% bootstrap interval that resamples operators; values below one favour the classical comparator. The fixed-load network was trained with $f=1$ only (initialisation 0); the random-load network was trained on a load distribution. The Chebyshev entry for the random-load network is not comparable with the larger gains of Section~\ref{sec:rhs}, which use other loads and the original test operators.}\label{tab:s1loads}',
         r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lll}\toprule',
         r'comparator & fixed-load network & random-load network \\\midrule']
    for lab, m in (('Chebyshev $\\nu=4$', 'CHEB'), ('Jacobi-PCG(4) smoothing calls', 'PCG'), ('SGS-PCG(4) smoothing calls', 'SGS-PCG')):
        a, b = LD[('OA-VCB init0', m)], LD[('OA-VCB init0 (random-load model)', m)]
        L.append(f'{lab} & {gci(a)}, $W={a["W"]:.2f}$ & {gci(b)}, $W={b["W"]:.2f}$ \\\\')
    L.append(r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    open(os.path.join(OUT, 'tab_s1_loads.tex'), 'w').write('\n'.join(L) + '\n')


tab_loads()

# ---- Appendix tables: full ladder on set C, barrier subgroups, loads by family, anisotropy rows ------------------------------
LADDER = (('steepest descent, 4 steps (Jacobi-preconditioned)', 'SD'), ('minimal residual, 4 steps (Jacobi-preconditioned)', 'MR'), ('K2-cycle (recursive flexible CG on the coarse levels, Chebyshev smoothers)', 'K2 (CHEB)'),
          ('Chebyshev, Lanczos bounds on every level (interval ratio chosen on the training operators)', 'CHEB-T'),
          ('fitted fixed polynomial, shared by all levels (contained class)', 'POLY-S'), ('fitted fixed polynomials, one set per level (not contained)', 'POLY-L'),
          ('Jacobi-PCG(4) smoothing calls (primary)', 'PCG'), ('symmetric Gauss--Seidel, two symmetric sweeps', 'SGS'), ('SGS-PCG(4) smoothing calls', 'SGS-PCG'))


def tab_ladder():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
         r'\caption{All one-pass comparators of Stage 1 on set C (100 operators per family; OA-VCB initialisation 0), in the same cycle with only the smoothers replaced. $G(\mathrm{Cheb}\!\to\!X)$: Chebyshev reference error divided by the comparator\textquoteright{}s; $G(X\!\to\!\mathrm{OA\text{-}VCB})$: comparator error divided by OA-VCB\textquoteright{}s, with the pointwise 95\% bootstrap interval; $W$: fraction of operators on which OA-VCB is more accurate. The per-instance candidate is an upper bound on the best polynomial class member, not a certified optimum. OA-VCL and OA-VCBL are learned residual-linear controls trained in the original study.}\label{tab:s1ladder}',
         r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lllllll}\toprule',
         r' & \multicolumn{3}{c}{channels} & \multicolumn{3}{c}{barriers} \\ \cmidrule(lr){2-4}\cmidrule(lr){5-7}',
         r'comparator & $G(\mathrm{Cheb}\!\to\!X)$ & $G(X\!\to\!\mathrm{OA\text{-}VCB})$ & $W$ & $G(\mathrm{Cheb}\!\to\!X)$ & $G(X\!\to\!\mathrm{OA\text{-}VCB})$ & $W$ \\\midrule']
    items = []
    for lab, key in [(l[0], l[-1]) for l in LADDER]:
        items.append((lab, key))
    items.insert(7, ('per-instance polynomial candidate (upper bound)', 'D2'))
    items += [('OA-VCL (learned, residual-linear)', 'OA-VCL init0'), ('OA-VCBL (learned, residual-linear, Galerkin output)', 'OA-VCBL init0')]
    for lab, key in items:
        cells = []
        for f in FAMS:
            E = C[f]['errors']; ex = D2[f]['errors_D2'] if key == 'D2' else E[key]
            g1 = paired(E['CHEB'], ex); g2 = paired(ex, E['OA-VCB init0'])
            if key != 'D2' and ('init' not in key):
                st = C[f]['secondary'].get(f'{key} -> OA-VCB init0') or (C[f]['confirmatory']['OA-VCB init0']['P1'] if key == 'PCG' else None)
                if st is not None: agree(g2, st, f'ladder {f} {key}')
            note(f'[set C ladder] {f} {lab}: G(Cheb -> X) {gci(g1)}; G(X -> OA-VCB init0) {gci(g2)} W {g2["W"]:.2f}')
            cells += [f'{g1["G"]:.2f}', gci(g2), f'{g2["W"]:.2f}']
        L.append(lab + ' & ' + ' & '.join(cells) + r' \\')
    L.append(r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    open(os.path.join(OUT, 'tab_s1_ladder.tex'), 'w').write('\n'.join(L) + '\n')


tab_ladder()

# barrier subgroups (set C), geometric-label groups as stored
cb = C['barriers']; ids_all = list(cb['data']['ids']); pos = {i: n for n, i in enumerate(ids_all)}
SG = {}
for grp in ('island', 'strict non-island'):
    ix = [pos[i] for i in cb['subgroups'][grp]['ids']]
    for m in ('PCG', 'K2 (CHEB)', 'SGS-PCG'):
        r = paired(np.asarray(cb['errors'][m])[ix], np.asarray(cb['errors']['OA-VCB init0'])[ix])
        agree(r, cb['subgroups'][grp]['gains'][f'{m} -> OA-VCB init0'], f'subgroup {grp} {m}'); SG[(grp, m)] = r
        note(f'[set C barriers, {grp}, n={len(ix)}] G({m} -> OA-VCB init0) {gci(r)} W {r["W"]:.2f}')
    r = paired(np.asarray(D2['barriers']['errors_D2'])[ix], np.asarray(cb['errors']['OA-VCB init0'])[ix]); SG[(grp, 'D2')] = r
    note(f'[set C barriers, {grp}, n={len(ix)}] G(per-instance polynomial candidate -> OA-VCB init0) {gci(r)} W {r["W"]:.2f}')


def tab_subgroups():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
         r'\caption{Barrier operators of set C split by floating islands (geometric label; 42 island operators and 52 strict non-island operators of 100; the remaining six are excluded here as ambiguous or too few). $G(X\!\to\!\mathrm{OA\text{-}VCB})$ for OA-VCB initialisation 0 with pointwise 95\% bootstrap intervals; values below one favour the comparator. Statements about intervals are not tests of equivalence.}\label{tab:s1sub}',
         r'\begin{tabular}{lll}\toprule', r'comparator & island & strict non-island \\\midrule']
    for lab, m in (('Jacobi-PCG(4)', 'PCG'), ('K2-cycle (Chebyshev)', 'K2 (CHEB)'), ('SGS-PCG(4)', 'SGS-PCG'), ('per-instance polynomial candidate', 'D2')):
        L.append(f'{lab} & {gci(SG[("island", m)])} & {gci(SG[("strict non-island", m)])} \\\\')
    L.append(r'\bottomrule\end{tabular}\end{table}')
    open(os.path.join(OUT, 'tab_s1_subgroups.tex'), 'w').write('\n'.join(L) + '\n')


tab_subgroups()

# loads by family (set LC): fixed-load initialisations 0-2 against Jacobi-PCG(4)
idsLC = list(LC['data']['ids']); posLC = {i: n for n, i in enumerate(idsLC)}
LF = {}
FAMLAB = (('f = -1', '$f=-1$'), ('smooth +-', 'smooth, both signs'), ('point source (w 0.03)', 'localised source'), ('random-load draw', 'random-load draw'))
for fam, lab in FAMLAB:
    ix = [posLC[i] for i in LC['subgroups'][fam]['ids']]
    for model in ('OA-VCB init0', 'OA-VCB init1', 'OA-VCB init2', 'OA-VCB init0 (random-load model)'):
        r = paired(np.asarray(E['PCG'])[ix], np.asarray(E[model])[ix]); tag = model.replace('OA-VCB ', '')
        agree(r, LC['subgroups'][fam]['gains'][f'PCG -> {model}'], f'LC {fam} {model}'); LF[(fam, model)] = r
        note(f'[set LC, {fam}, 50 operators] G(Jacobi-PCG(4) -> {model}) {gci(r)} W {r["W"]:.2f}')


def tab_loadfam():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{3.5pt}',
         r'\caption{Stage 1 set LC by load family (50 channel operators each): $G(\text{Jacobi-PCG(4)}\!\to\!\text{network})$ with the pointwise 95\% bootstrap interval, for the three fixed-load initialisations and the random-load network. Values below one mean that the one-pass Jacobi-PCG(4) cycle is more accurate.}\label{tab:s1loadfam}',
         r'\begin{adjustbox}{max width=\linewidth}\begin{tabular}{lllll}\toprule',
         r'load family & fixed-load init 0 & fixed-load init 1 & fixed-load init 2 & random-load network \\\midrule']
    for fam, lab in FAMLAB:
        L.append(lab + ' & ' + ' & '.join(gci(LF[(fam, m)]) for m in ('OA-VCB init0', 'OA-VCB init1', 'OA-VCB init2', 'OA-VCB init0 (random-load model)')) + r' \\')
    L.append(r'\bottomrule\end{tabular}\end{adjustbox}\end{table}')
    open(os.path.join(OUT, 'tab_s1_loadfam.tex'), 'w').write('\n'.join(L) + '\n')


tab_loadfam()

# anisotropy: the three rows of the original settings table, parsed from numbers.txt (13 Oct run); no new computation
ANI = {}
for line in open(numtxt, encoding='utf-8'):
    m = re.match(r'\s*anisotropic tensor coefficient, seed (\d)( \(dev\.\))?: OA-VCB ([\d.e+-]+) member ([\d.e+-]+) G ([\d.]+) \[([\d.]+), ([\d.]+)\] W ([\d.]+)', line)
    if m: ANI[int(m.group(1))] = dict(oa=float(m.group(3)), ref=float(m.group(4)), G=float(m.group(5)), lo=float(m.group(6)), hi=float(m.group(7)), W=float(m.group(8)))
assert sorted(ANI) == [0, 1, 2], ANI
for s_, r in sorted(ANI.items()): note(f'[anisotropic tensor coefficient, original protocol] data seed {s_}: OA-VCB {r["oa"]:.2e}, reference {r["ref"]:.2e}, G(Chebyshev nu=4 -> OA-VCB) {gci(r)} W {r["W"]:.2f}')


def tab_aniso():
    L = [r'\begin{table}[tbp]\centering\footnotesize\setlength{\tabcolsep}{4pt}',
         r'\caption{Supplementary scope check: anisotropic tensor-coefficient family (meandering tows, $k_L=100$, $k_T=5$ against a unit background, insulated holes), one separately trained OA-VCB per data seed with the original protocol and initialisation seed $0$; $100$ test operators per seed. Mean relative energy errors, paired $G$ of the degree-four Chebyshev reference to OA-VCB with the 95\% bootstrap interval, and $W$. Data seed $0$ is the development seed, so the three rows are three configurations including the development one, not three independent held-out replications. Stronger classical smoothing budgets, Krylov-space controls and matched-accuracy solver costs were not evaluated for this family.}\label{tab:aniso}',
         r'\begin{tabular}{llllll}\toprule', r'data seed & status & OA-VCB & reference & $G$ [95\% CI] & $W$ \\\midrule']
    for s_, st in ((0, 'development'), (1, 'subsequent evaluation'), (2, 'subsequent evaluation')):
        r = ANI[s_]; L.append(f'{s_} & {st} & {r["oa"]:.2e} & {r["ref"]:.2e} & {gci(r)} & {r["W"]:.2f} \\\\')
    L.append(r'\bottomrule\end{tabular}\end{table}')
    open(os.path.join(OUT, 'tab_aniso.tex'), 'w').write('\n'.join(L) + '\n')


tab_aniso()


# original main table without the learned-baseline rows (those stay in the archive); a deterministic filter of tables/tab_core.tex
def tab_core_trim():
    src = os.path.join(OUT, 'tab_core.tex')
    txt = open(src, encoding='utf-8').read()
    USED['manuscript/tables/tab_core.tex'] = hashlib.sha256(txt.encode('utf-8')).hexdigest()
    keep = [l for l in txt.split('\n') if not l.startswith(('OA-Lift', 'GNN-op', 'Transolver-op'))]
    out = '\n'.join(keep)
    i = out.index(r'\caption{'); j = out.index(r'\label{tab:core}')
    cap = (r'\caption{Main comparison of the original study, held-out data seeds 1--3 (100 test operators each; L3, $k_0=2$; Chebyshev reference only). '
           r'Energy: mean of the three per-seed arithmetic mean relative energy errors. $G$: paired geometric-mean ratio of the classical reference' + "'" + r's error to the model' + "'" + r's over the pooled 300 samples [95\% CI]; '
           r'$W$: fraction of samples on which the model is more accurate than the reference. Rows for the other learned baselines of the original study are omitted here; they are kept in the archive and are not part of the comparisons of this paper.}')
    out = out[:i] + cap + out[j:]
    open(os.path.join(OUT, 'tab_core_trim.tex'), 'w').write(out)


tab_core_trim()

NUM.close()
open(os.path.join(OUT, 'MANIFEST_stage1.txt'), 'w').write(''.join(f'{h}  {p}\n' for p, h in sorted(USED.items())))
print('Stage 1 tables written to', OUT, f'({len(USED)} input files, MANIFEST_stage1.txt)')
