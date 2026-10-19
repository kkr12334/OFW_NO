"""Writes the summary section of STAGE1_REPORT_REV1 from the raw records (reserve, ledger, timing v2, set C) and the
placeholders' replacements into runs/rev1_summary.json.  usage: python -I ext/stage1_rev1_summary.py [--fill]"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); J = lambda p: json.load(open(os.path.join(HERE, 'runs', p)))
R = {f: J(f'reserve_{f}.json')['endpoints'] for f in ('channels', 'barriers')}; K = J('reserve_K_local_channels.json')['summary']
def rng(f, m, key='G', inv=False):
    v = [R[f][f'OA-VCB init{k}'][f'{m} -> OA-VCB'][key] for k in (0, 1, 2)]; v = [1 / x for x in v] if inv else v; return min(v), max(v)
c8 = rng('channels', 'PCG(8)'); c8lo = rng('channels', 'PCG(8)', 'lo'); c14 = rng('channels', 'PCG(14)'); c14hi = rng('channels', 'PCG(14)', 'hi')
b5 = rng('barriers', 'PCG(5)'); b8 = rng('barriers', 'PCG(8)'); i14c = rng('channels', 'PCG(14)', inv=True); i14b = rng('barriers', 'PCG(14)', inv=True)
k5pre = [K[f'OA-VCB init{k} pre K5']['rho_q10_50_90'][1] for k in (0, 1, 2)]; k5post = [K[f'OA-VCB init{k} post K5']['rho_q10_50_90'][1] for k in (0, 1, 2)]
k5n = [K[f'OA-VCB init{k} {p} K5']['n_rho_le_099'] for k in (0, 1, 2) for p in ('pre', 'post')]; k8n = [K[f'OA-VCB init{k} {p} K8']['n_rho_le_099'] for k in (0, 1, 2) for p in ('pre', 'post')]
L = J('stage1_ledger.json')['families']['channels']
T = {f: J(f'timing_v2_C_{f}.json') for f in ('channels', 'barriers')}
med = lambda f, k: float(np.median([np.median(r['raw_seconds'][k]) for r in T[f]['records']]) * 1e3)
uc = {f: med(f, 'u_c construction') for f in T}
def ttt(f):
    t = []
    for r in T[f]['records']:
        st = np.median(np.array(r['curve_stamps']['curve two-grid PCG']), 0); h = next((j for j, v in enumerate(r['curve_errors']['two-grid PCG']) if v <= r['errors']['1pass OA-VCB batch1']), None)
        u_ = np.median(r['raw_seconds']['u_c construction']) * 1e3
        t.append((st[h] * 1e3, st[h] * 1e3 + u_) if h is not None else (np.inf, np.inf))
    return float(np.median([a for a, _ in t])), float(np.median([b for _, b in t]))   # conditional; median of per-operator sums
(tc, ec), (tb, eb) = ttt('channels'), ttt('barriers')
KR = J('reserve_K_local_channels.json')['records']
k8lt1 = [sum(1 for r_ in KR[f'OA-VCB init{k}'][p] if r_['K8']['rank'] == 8 and r_['K8']['rho'] < 1) for k in (0, 1, 2) for p in ('pre', 'post')]; oa = [med(f, '1pass OA-VCB batch1') for f in T]
C = {f: J(f'confirm_C_{f}.json')['confirmatory'] for f in ('channels', 'barriers')}
p1 = {f: [C[f][f'OA-VCB init{k}']['P1']['G'] for k in (0, 1, 2)] for f in C}
fmt = lambda x: f'{x:,.0f}'.replace(',', ' ')
S = f"""**Confirmatory (set C, unchanged):**
- OA-VCB reduces one-pass energy error relative to the four-step Jacobi-PCG smoother by geometric factors of **{min(p1['channels']):.2f}–{max(p1['channels']):.2f} on channels** and **{min(p1['barriers']):.2f}–{max(p1['barriers']):.2f} on barriers**.
- Frozen labels: "not accounted for by the tested four-step Jacobi-PCG control" on channels; "partly accounted for / inconclusive" on barriers.

**Local mechanism (channels):**
- On set C, OA-VCB's fine-level smoother improves on the exact K₄ energy minimum on every tested pre- and post-smoother input.
- On fresh reserve operators (amendment 2, R4), it also improves on the exact K₅ minimum. K₅ is the polynomial class of matching matrix-propagation degree.
  - ρ_K₅ ≤ 0.99 on {min(k5n)}–{max(k5n)} of 100 inputs at both positions for all three initialisations.
  - Median ρ_K₅ is {min(k5pre):.3f}–{max(k5pre):.3f} (pre) and {min(k5post):.3f}–{max(k5post):.3f} (post).
- Against the K₈ minimum, OA-VCB's error is at least 1% lower on only {min(k8n)}–{max(k8n)} of 100 inputs, and lower at all on {min(k8lt1)}–{max(k8lt1)}.
- This is a local, fixed-input separation. It is not a separation between complete cycles.

**Full cycle:**
- OA-VCB outperforms the best full-cycle polynomial candidates found by our per-instance optimisation (4.14× on channels, set C). Those candidates do not certify a separation from the complete polynomial class.
- On barriers the candidate is more accurate than OA-VCB.

**Larger Krylov budgets (reserve-confirmed, R1–R2):**
- Jacobi-PCG with 8 to 14 steps per call brackets OA-VCB on channels, for all three initialisations:
  - G(PCG(8) → OA-VCB) {c8[0]:.2f}–{c8[1]:.2f}, every lower limit ≥ {c8lo[0]:.3f};
  - G(PCG(14) → OA-VCB) {c14[0]:.2f}–{c14[1]:.2f}, every upper limit ≤ {c14hi[1]:.3f}.
- On barriers the bracket is 5 to 8 steps: G(PCG(5) → OA-VCB) {b5[0]:.2f}–{b5[1]:.2f}; G(PCG(8) → OA-VCB) {b8[0]:.2f}–{b8[1]:.2f}.

**Cost (operation ledger, §6; repaired timings, §7):**
- One OA-VCB pass costs roughly {L['OA-VCB']['per_pass_MAC'] / 1e6:.0f} million MACs, about {fmt(round(L['OA-VCB']['per_pass_fine_spmv_equiv'], -2))} fine-SpMV-equivalents. These are approximate MAC-equivalent counts under the ledger's convention; §6 lists the exclusions.
- That is roughly {round(L['PCG(14)']['OA_VCB_over_method'], -1):.0f}× the PCG(14) pass. PCG(14) is {i14c[0]:.1f}–{i14c[1]:.1f}× more accurate on channels and {i14b[0]:.1f}–{i14b[1]:.1f}× on barriers (reserve).
- Repaired timings, OA-VCB: batch-1 latency is {oa[0]:.0f} / {oa[1]:.0f} ms (channels / barriers), conditional on u_c.
- Repaired timings, classical: two-grid PCG reaches OA-VCB's per-sample error in a median {tc:.1f} / {tb:.1f} ms conditional on u_c, or an estimated {ec:.1f} / {eb:.1f} ms including u_c's construction (median of per-operator sums of separately timed components).

**These accuracy gains therefore do not establish a computational advantage.** The repairs correct the cost and timing numbers but leave the earlier qualitative conclusion unchanged."""
json.dump(dict(SUMMARY=S), open(os.path.join(HERE, 'runs', 'rev1_summary.json'), 'w'), indent=1); print(S)
if '--fill' in sys.argv:
    p = os.path.join(HERE, 'STAGE1_REPORT_REV1_2026-10-18.md'); s = open(p).read(); s = s.replace('{{SUMMARY}}', S); open(p, 'w').write(s); print('filled SUMMARY')
