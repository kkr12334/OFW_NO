# Stage 1 report, revision 1: what learned smoothing adds beyond classical smoothing in the same cycle

18 October 2026. Claude, for Kumar and ASTRA.

**What this revision replaces.** It supersedes the interpretation, cost and timing statements of report v1 (`STAGE1_REPORT_2026-10-17.md`, SHA-256 `9ec033f3…`). That file stays frozen as a historical record.

**What drove the changes:**
- ASTRA's review (`ASTRA_STAGE1_REPORT_REVIEW_2026-10-17.md`), answered point by point in `CLAUDE_RESPONSE_TO_ASTRA_STAGE1_REPORT_REVIEW_2026-10-18.md`;
- protocol amendment 2 (`STAGE1_PROTOCOL_AMENDMENT_2_2026-10-18.md`, SHA-256 `5480b6a9…`), hashed before the reserve operators existed.

**Unchanged:** the confirmatory records and P1–P3.

**New:**
- repaired timings (`runs/timing_v2_C_*.json`);
- a complete operation ledger (`runs/stage1_ledger.json`);
- the frozen reserve experiment (`runs/reserve_*.json`).

**Status labels:**
- **confirmatory:** P1–P3 on set C;
- **reserve-confirmed:** amendment 2 endpoints R1–R4, on operators generated after the amendment hash;
- **secondary:** pre-declared, on set C;
- **exploratory:** development set D, analyses chosen after set C was opened.

## 1 Summary

**Confirmatory (set C, unchanged):**
- OA-VCB reduces one-pass energy error relative to the four-step Jacobi-PCG smoother by geometric factors of **5.70–6.12 on channels** and **1.38–1.45 on barriers**.
- Frozen labels: "not accounted for by the tested four-step Jacobi-PCG control" on channels; "partly accounted for / inconclusive" on barriers.

**Local mechanism (channels):**
- On set C, OA-VCB's fine-level smoother improves on the exact K₄ energy minimum on every tested pre- and post-smoother input.
- On fresh reserve operators (amendment 2, R4), it also improves on the exact K₅ minimum. K₅ is the polynomial class of matching matrix-propagation degree.
  - ρ_K₅ ≤ 0.99 on 97–100 of 100 inputs at both positions for all three initialisations.
  - Median ρ_K₅ is 0.689–0.709 (pre) and 0.595–0.614 (post).
- Against the K₈ minimum, OA-VCB is better on only 19–43 of 100 inputs.
- This is a local, fixed-input separation. It is not a separation between complete cycles.

**Full cycle:**
- OA-VCB outperforms the best full-cycle polynomial candidates found by our per-instance optimisation (4.14× on channels, set C). Those candidates do not certify a separation from the complete polynomial class.
- On barriers the candidate is more accurate than OA-VCB.

**Larger Krylov budgets (reserve-confirmed, R1–R2):**
- Jacobi-PCG with 8 to 14 steps per call brackets OA-VCB on channels, for all three initialisations:
  - G(PCG(8) → OA-VCB) 1.40–1.51, every lower limit ≥ 1.279;
  - G(PCG(14) → OA-VCB) 0.37–0.40, every upper limit ≤ 0.426.
- On barriers the bracket is 5 to 8 steps: G(PCG(5) → OA-VCB) 1.21–1.26; G(PCG(8) → OA-VCB) 0.72–0.76.

**Cost (operation ledger, §6; repaired timings, §7):**
- One OA-VCB pass costs 713 million MACs, about 30 330 fine-SpMV-equivalents of arithmetic.
- That is 161× the PCG(14) pass. PCG(14) is 2.5–2.7× more accurate on channels and 2.7–2.8× on barriers (reserve).
- Repaired timings, OA-VCB: batch-1 latency is 284 / 282 ms (channels / barriers), conditional on u_c.
- Repaired timings, classical: two-grid PCG reaches OA-VCB's per-sample error in a median 4.3 / 4.0 ms conditional on u_c, or 6.9 / 6.6 ms end-to-end including u_c's construction.

**These accuracy gains therefore do not establish a computational advantage.** The repairs correct the cost and timing numbers but leave the earlier qualitative conclusion unchanged.

## 2 Confirmatory results (set C; unchanged)

Here R = G(PCG → OA-VCB): the geometric mean over operators of e_PCG / e_OA-VCB, with PCG the four-step Jacobi-preconditioned CG smoother. Values above 1 favour OA-VCB. Intervals are pointwise 95% bootstrap over operators.

| Family | Init | R [95%] | W | Frozen P1 label | P2: G(POLY-S → OA-VCB) [95%] |
|---|---|---|---|---|---|
| channels | 0 | 5.70 [5.25, 6.13] | 0.99 | not accounted for by Krylov smoothing | 5.14 [4.79, 5.51] |
| channels | 1 | 5.94 [5.56, 6.32] | 1.00 | same | 5.37 [4.96, 5.81] |
| channels | 2 | 6.12 [5.71, 6.54] | 1.00 | same | 5.53 [5.13, 5.95] |
| barriers | 0 | 1.45 [1.33, 1.59] | 0.82 | partly accounted for / inconclusive | 1.76 [1.60, 1.93] |
| barriers | 1 | 1.42 [1.28, 1.58] | 0.77 | same | 1.73 [1.55, 1.91] |
| barriers | 2 | 1.38 [1.24, 1.54] | 0.76 | same | 1.67 [1.51, 1.85] |

- **P3** = G(CHEB → PCG): 1.61 [1.58, 1.64] on channels, 1.37 [1.32, 1.43] on barriers.
- **ASTRA's independent recomputation** from the per-operator errors, without our statistics code, agrees to 8.9e-16.
- **How to read the frozen label in prose:** "not accounted for by the tested four-step Jacobi-PCG control". It concerns one algorithm, not every Krylov method.
- **φ(PCG):** the recovered log-gain share is 0.22 [0.21, 0.22] on channels and 0.46 [0.39, 0.54] on barriers (OA-VCB init 0, descriptive).

## 3 Polynomial smoothing in the full cycle: best candidates found, not class bounds

### 3.1 The ladder on set C (OA-VCB init 0)

| Method (one pass, same cycle) | Channels G(CHEB → X) | Barriers G(CHEB → X) | Channels G(X → OA-VCB) | Barriers G(X → OA-VCB) |
|---|---|---|---|---|
| SD / MR | 0.82 / 0.83 | 0.82 / 0.79 | 11.3 / 11.1 | 2.42 / 2.52 |
| K2-cycle (CHEB) | 0.97 | 1.40 | 9.48 | 1.42 |
| CHEB-T\* | 1.31 | 1.12 | 6.99 | 1.79 |
| PCG (P1 control) | 1.61 | 1.37 | 5.70 | 1.45 |
| POLY-S / POLY-L | 1.79 / 1.86 | 1.13 / 1.30 | 5.14 / 4.93 | 1.76 / 1.54 |
| Best per-instance polynomial candidate found (D2) | 2.22 | 2.65 | 4.14 [3.82, 4.45] | 0.75 [0.68, 0.83] |
| OA-VCL / OA-VCBL (learned, residual-linear) | 2.48 / 2.60 | 0.67 / 1.10 | 3.70 / 3.53 | 2.99 / 1.81 |
| SGS / SGS-PCG | 1.27 / 7.94 | 1.01 / 3.44 | 7.23 / 1.16 | 1.97 / 0.58 |
| OA-VCB init 0 / 1 / 2 | 9.18 / 9.58 / 9.87 | 2.00 / 1.96 / 1.90 | — | — |

**What the D2 candidate is.** It is the best full-cycle polynomial smoothing found by our per-instance optimisation: separate degree-3 polynomials for each of the six smoothing calls, fitted with the exact solution, from five starts.

**What its feasible set contains.** In exact arithmetic it contains the trajectory of every smoother whose output lies in K₄ of its own call's input, under this wrapper. That includes PCG, SD, MR and the fixed polynomials. ASTRA confirms this containment.

**What it does not show.** The fitted value only bounds the class optimum from above. So:
- **Channels:** an observed candidate ratio of 4.14 does not establish that the class optimum loses to OA-VCB. An undiscovered polynomial schedule could do better.
- **Barriers:** the candidate beats OA-VCB on most operators (W = 0.34 for OA-VCB). That does establish that OA-VCB is not better than the whole class on those instances.

## 4 Local Krylov-space test (D1, set C; and the reserve confirmation)

ρ_d = E_NN / E_{K_d}: the energy-norm error of the network's output for a given fine-level input, divided by that of the exact minimiser over the d-dimensional Krylov space of the same input. The minimiser is computed from a rank-aware A-orthonormal basis, so this local comparison is exact.

| Family | Position | ρ_K₄ 10 / 50 / 90% | Median δ | ρ_K₄ ≤ 0.99 |
|---|---|---|---|---|
| channels | pre | 0.566 / 0.614 / 0.673 | 0.249 | 100/100 |
| channels | post | 0.420 / 0.491 / 0.748 | 0.145 | 100/100 |
| barriers | pre | 0.929 / 0.998 / 5.845 | 0.207 | 44/100 |
| barriers | post | 0.697 / 0.841 / 0.977 | 0.088 | 94/100 |

**Checks:** the rank is 4 throughout, and PCG(4) equals the K₄ minimiser to 3.8e-13.

**What this establishes:** a local, fixed-input separation at the tested positions on channels. It is not a separation between complete cycles, whose later inputs differ.

**Degree control K₅** (development set D, OA-VCB init 0; exploratory):
- **Channels:** median ρ_K₅ is 0.715 (pre) and 0.620 (post); ρ_K₅ ≤ 0.99 on 100/100 at both positions.
- **Barriers:** the result depends on position. Pre: 6/100, median 1.054. Post: 63/100, median 0.976.
- **K₈:** the median advantage reverses on channels (median ρ_K₈ 1.024 pre, 1.079 post). OA-VCB still beats K₈ on 23/100 (pre) and 33/100 (post) inputs, so K₈ does not dominate on every input.

**Reserve confirmation of the channel K₅ result (R4).** Reserve channels (100 operators, all three inits, frozen criterion ρ_K₅ ≤ 0.99 on ≥ 90 full-rank operators):

| Init | Position | ρ_K₄ median | ρ_K₅ 10 / 50 / 90% | ρ_K₅ ≤ 0.99 | K₅ confirmed | ρ_K₈ median | ρ_K₈ ≤ 0.99 | max ‖QᵀAQ − I‖ | max \|E_PCG/E_K − 1\| |
|---|---|---|---|---|---|---|---|---|---|
| 0 | pre | 0.610 | 0.662 / 0.709 / 0.768 | 100/100 | yes | 1.025 | 19/100 | 5.2e-15 | 4.4e-16 |
| 0 | post | 0.488 | 0.518 / 0.607 / 0.767 | 97/100 | yes | 1.069 | 23/100 | 5.3e-15 | 2.9e-15 |
| 1 | pre | 0.601 | 0.649 / 0.695 / 0.763 | 100/100 | yes | 1.011 | 36/100 | 5.2e-15 | 4.4e-16 |
| 1 | post | 0.488 | 0.528 / 0.614 / 0.779 | 99/100 | yes | 1.061 | 21/100 | 4.4e-15 | 1.3e-15 |
| 2 | pre | 0.595 | 0.640 / 0.689 / 0.755 | 100/100 | yes | 1.002 | 43/100 | 5.2e-15 | 4.4e-16 |
| 2 | post | 0.471 | 0.475 / 0.595 / 0.835 | 98/100 | yes | 1.049 | 35/100 | 6.0e-15 | 3.1e-15 |

Local K₅ separation confirmed at both positions for all three initialisations: True.

**Why K₅ is only a degree control.** It is the polynomial class whose matrix-propagation degree matches four message-passing layers. It is not a matched-dependence class. OA-VCB normalises its input by σ, a global max/RMS of the residual, before a nonlinear encoder. Its output therefore depends on distant residual entries outside any 4-hop neighbourhood.

## 5 Larger Krylov budgets in the cycle (exploratory on D; reserve-confirmed in §5.2)

### 5.1 Development set D (OA-VCB init 0)

| Smoother per call | Channels G(CHEB → X) | Channels G(X → OA-VCB) [95%] | Barriers G(CHEB → X) | Barriers G(X → OA-VCB) [95%] |
|---|---|---|---|---|
| PCG(4) | 1.62 | 6.31 [6.02, 6.58] | 1.36 | 1.57 [1.41, 1.74] |
| PCG(5) | 2.62 | 3.91 [3.73, 4.08] | 1.76 | 1.21 [1.08, 1.35] |
| PCG(8) | 6.86 | 1.49 [1.43, 1.56] | 2.90 | 0.74 [0.64, 0.85] |
| PCG(14) | 24.9 | 0.41 [0.39, 0.44] | 6.21 | 0.34 [0.28, 0.42] |
| SGS-PCG(4) | 8.22 | 1.24 [1.19, 1.30] | 3.33 | 0.64 [0.55, 0.75] |

Also on D: the best K₅ polynomial candidate found stays 2.36× [2.15, 2.57] below OA-VCB on channels (20 operators). It is stable under the tested runs three times longer (10 operators), but this is not a class bound.

### 5.2 Reserve experiment (frozen in amendment 2; R1–R3)

100 fresh operators per family (rng 2026101603 channels, 2026101604 barriers), generated after the amendment-2 hash; all gates passed: True. Paired G with pointwise 95% intervals; W omitted for space (in the records).

| Family | Init | G(PCG(4) → OA-VCB) | G(PCG(5) → OA-VCB) | G(PCG(8) → OA-VCB) | G(PCG(14) → OA-VCB) | G(SGS-PCG(4) → OA-VCB) | Bracket confirmed |
|---|---|---|---|---|---|---|---|
| channels | 0 | 5.96 [5.41, 6.41] | 3.65 [3.32, 3.92] | 1.40 [1.28, 1.50] | 0.37 [0.34, 0.40] | 1.16 [1.06, 1.24] | yes (PCG(8)–PCG(14)) |
| channels | 1 | 6.20 [5.80, 6.57] | 3.80 [3.56, 4.02] | 1.45 [1.37, 1.53] | 0.39 [0.36, 0.41] | 1.21 [1.14, 1.27] | yes (PCG(8)–PCG(14)) |
| channels | 2 | 6.43 [6.05, 6.78] | 3.94 [3.72, 4.15] | 1.51 [1.43, 1.58] | 0.40 [0.38, 0.43] | 1.25 [1.19, 1.31] | yes (PCG(8)–PCG(14)) |
| barriers | 0 | 1.54 [1.39, 1.70] | 1.21 [1.08, 1.34] | 0.72 [0.62, 0.84] | 0.35 [0.29, 0.43] | 0.64 [0.55, 0.75] | yes (PCG(5)–PCG(8)) |
| barriers | 1 | 1.60 [1.45, 1.77] | 1.26 [1.13, 1.40] | 0.76 [0.66, 0.87] | 0.37 [0.30, 0.45] | 0.67 [0.57, 0.78] | yes (PCG(5)–PCG(8)) |
| barriers | 2 | 1.55 [1.39, 1.72] | 1.21 [1.09, 1.36] | 0.73 [0.63, 0.84] | 0.36 [0.29, 0.44] | 0.64 [0.55, 0.75] | yes (PCG(5)–PCG(8)) |

- **R1 (primary, channels):** bracketing PCG(8)–PCG(14) confirmed for all three initialisations: True.
- **R2 (barriers):** bracketing PCG(5)–PCG(8) confirmed for all three initialisations: True.
- **R3:** G(PCG(14) → SGS-PCG(4)) = 0.32 [0.31, 0.34] (channels), 0.55 [0.51, 0.60] (barriers).

### 5.3 SGS-PCG

- **On set C:** G(SGS-PCG → OA-VCB) is 1.16 / 1.21 / 1.24 on channels and 0.58 / 0.57 / 0.55 on barriers. W = 0.79 for OA-VCB on channels and 0.29 on barriers (init 0).
- **On set D:** PCG(14) beats SGS-PCG(4), with geometric e_PCG14 / e_SGS-PCG4 of 0.331 [0.315, 0.348] on channels and 0.535 [0.490, 0.582] on barriers.
- **Interpretation, wording from ASTRA:** the earlier comparison does not isolate a spatial mechanism; increasing Jacobi-PCG's budget suffices to match or exceed this SGS-PCG configuration (natural ordering, one symmetric sweep per application). Neither result is a verdict on optimally ordered, block or Schwarz smoothers.

## 6 Operation ledger (`runs/stage1_ledger.json`; amendment 2, B3)

Arithmetic per one pass, in MACs (multiply–add counts). Per-term conventions are stated in the ledger file.

**What is counted:**
- every term: initial residual, wrapper (deflation, b = r − Ax₀, Galerkin output), smoothers, cycle residuals and vector updates, transfers and the coarsest solve;
- method-specific set-up, listed separately;
- the learned smoother on its own sparsity pattern.

**Checks:**
- The learned dense-layer counts equal torch's FlopCounterMode on all three levels.
- The cycle's SpMV-equivalent counts per level equal the counters recorded in the confirmation runs, for all 16 method–family pairs.

**Units.** "Fine-SpMV-equivalent" means MACs ÷ fine nnz. It is an arithmetic unit, not a count of SpMV calls, a time or a communication volume.

| Method | Channels MACs / pass | Fine-SpMV-equivalent | Of which smoother | Method set-up | OA-VCB ÷ method |
|---|---|---|---|---|---|
| POLY-S | 1.48 M | 63.1 | 0.81 M | 0.00 M | 481 |
| SGS | 1.56 M | 66.4 | 0.88 M | 0.00 M | 457 |
| SD | 1.70 M | 72.5 | 1.03 M | 0.00 M | 418 |
| CHEB | 1.72 M | 73.3 | 1.04 M | 2.59 M | 414 |
| CHEB-T | 1.72 M | 73.3 | 1.04 M | 1.99 M | 414 |
| POLY-L | 1.48 M | 63.1 | 0.81 M | 1.99 M | 481 |
| PCG | 1.76 M | 74.8 | 1.08 M | 0.00 M | 406 |
| MR | 1.78 M | 75.5 | 1.10 M | 0.00 M | 402 |
| PCG(5) | 2.02 M | 86.1 | 1.35 M | 0.00 M | 352 |
| PCG(8) | 2.82 M | 120.0 | 2.14 M | 0.00 M | 253 |
| K2 (CHEB) | 3.62 M | 153.9 | 2.25 M | 2.59 M | 197 |
| SGS-PCG | 3.92 M | 166.9 | 3.25 M | 0.00 M | 182 |
| PCG(14) | 4.41 M | 187.8 | 3.74 M | 0.00 M | 161 |
| OA-VCB | 712.81 M | 30329.9 | 712.14 M | 0.00 M | 1 |

**Common set-up is not counted here.** The operator-adapted hierarchy, the near-kernel basis and the coarse factorisation are timed in §7. On barriers the wrapper costs a little more, because the mean number of near-kernel columns is 0.72; the ratios change by at most 5.3%.

**OA-VCB beyond MACs, per pass:**
- 13.1 M GELU evaluations;
- 18 nonlinear global reductions (max, RMS and median, three per call).

**Reading.** At four stages, OA-VCB's smoothing arithmetic is two to three orders of magnitude above every classical smoother tested, including the larger-budget ones in §5: about 160× PCG(14), which is more accurate.

**Caveat.** These are operation counts in one convention, not time or energy measurements, and not a claim about communication.

## 7 Repaired timings (`runs/timing_v2_C_*.json`; amendment 2, B2)

The repairs:
- each method builds only its own set-up;
- mesh patterns are built once per mesh, outside the timers;
- weights are loaded outside the timers;
- residuals are formed inside the timers;
- end-to-end times include constructing u_c.

The v1 timings are superseded.

| Quantity (median over 50 operators; ms) | Channels | Barriers |
|---|---|---|
| u_c construction (coarse Galerkin, factorise, solve, prolong) | 2.6 | 2.6 |
| Classical one pass, any smoother, conditional on u_c (range over 10 methods) | 66–72 | 66–71 |
| … of which operator-adapted hierarchy (profile) | 62.6 | 63.1 |
| … near-kernel basis / legacy CHEB bounds / Lanczos bounds (profile) | 1.3 / 2.3 / 4.7 | 1.5 / 2.3 / 4.8 |
| … cycle + wrapper only: PCG / SGS-PCG (profile) | 1.9 / 4.5 | 2.0 / 4.6 |
| OA-VCB one pass, batch 1, conditional on u_c | 284 | 282 |
| … of which operator data incl. hierarchy (profile) | 87 | 86 |
| Mesh patterns, once per mesh (excluded) | 77 | 95 |
| Sparse direct solve (never charged u_c) | 10.1 | 10.0 |
| OA-VCB batch-8 throughput per operator (secondary) | 206 | 211 |

| Whole solve from u_c (median ms; conditional / end-to-end incl. u_c) | Channels: to OA-VCB error | Channels: to rel. residual 1e-6 | Barriers: to OA-VCB error | Barriers: to rel. residual 1e-6 |
|---|---|---|---|---|
| two-grid PCG | 4.3 / 6.9 | 6.5 / 9.1 (censored 3/50) | 4.0 / 6.6 | 6.1 / 8.7 (censored 12/50) |
| SA-AMG PCG | 14.2 / 16.8 | 20.2 / 22.8 (censored 9/50) | 14.5 / 17.0 | median not reached (censored 25/50) |
| PCG-V | 71.6 / 74.2 | 78.9 / 81.5 | 71.3 / 73.9 | 77.5 / 80.1 (censored 2/50) |
| FCG-K2 | 76.5 / 79.1 | 92.4 / 95.0 | 73.4 / 76.0 | 87.6 / 90.2 (censored 7/50) |
| FCG-PCGsmoothed-V | 66.1 / 68.7 | 74.3 / 76.9 (censored 3/50) | 66.7 / 69.3 | 73.5 / 76.1 (censored 14/50) |
| CHEB passes | 71.7 / 74.3 | median not reached (censored 50/50) | 70.4 / 73.0 | median not reached (censored 50/50) |
| PCG passes | 68.2 / 70.8 | median not reached (censored 50/50) | 68.5 / 71.1 | median not reached (censored 40/50) |
| SGS-PCG passes | 72.9 / 75.5 | 88.2 / 90.8 (censored 23/50) | 69.4 / 72.0 | 88.1 / 90.7 (censored 24/50) |
| OA-VCB one pass (for comparison) | 284 / 287 (its own error) | (not reached by one pass) | 282 / 285 (its own error) | (not reached by one pass) |

- Medians are over all 50 operators, with censored operators (target not reached within the frozen iteration limit) counted as not reached; 'median not reached' means at least half were censored. Offline oracle stopping ("to OA-VCB error") uses each operator's true error; the relative-residual rule is deployable. Every outer PCG/FCG loop and pass sequence terminated as 'completed' on all 50 operators per family: True.
- OA-VCB's geometric-mean error on these 50 operators: 2.56e-03 (channels), 1.47e-02 (barriers); u_c alone: 3.60e-01, 4.00e-01.
- Compared with v1, classical one-pass latency drops from ~80 ms to the values above (unused bound set-up removed), and OA-VCB latency drops from ~370 ms (patterns no longer rebuilt in the timer). The qualitative conclusion is unchanged.

## 8 Barrier structure, loads and seeds (set C, set S, set LC, set R; unchanged, wording corrected)

**Barrier subgroups** (OA-VCB init 0; G(X → OA-VCB) with 95% interval):

| Category (n in C / S) | PCG (C) | K2 (C) | SGS-PCG (C) | D2 candidate (C) | PCG (S) | K2 (S) | SGS-PCG (S) |
|---|---|---|---|---|---|---|---|
| island (42 / 20) | 1.45 [1.20, 1.72] | 0.95 [0.80, 1.13] | 1.10 [0.90, 1.33] | 0.57 [0.47, 0.69] | 1.30 [0.93, 1.77] | 0.85 [0.60, 1.19] | 1.01 [0.73, 1.38] |
| strict non-island (52 / 20) | 1.49 [1.37, 1.63] | 1.99 [1.82, 2.17] | 0.37 [0.32, 0.43] | 0.96 [0.89, 1.04] | 1.54 [1.38, 1.71] | 2.02 [1.81, 2.24] | 0.39 [0.33, 0.46] |
| EDP (4 / 20) | 1.04 | 1.01 | 0.29 | 0.52 | 1.53 [1.21, 1.91] | 1.36 [0.87, 2.05] | 0.44 [0.31, 0.61] |
| discordant (2 / 20) | n < 3 | | | | 1.16 [0.83, 1.58] | 1.28 [0.83, 1.92] | 0.30 [0.21, 0.41] |

- **Island operators:** the K-cycle's interval and SGS-PCG's interval both include parity with OA-VCB. The polynomial candidate is better than OA-VCB. These are statements about intervals, not tests of equivalence.
- **Strict non-island operators:** the polynomial candidate's interval includes parity, and SGS-PCG is better than OA-VCB.

**Loads (set LC):**
- The fixed-load models are less accurate than one-pass PCG on every new load. The one exception is init 2 on the random-load draw, where the interval includes parity.
- The random-load model gives R = 2.79 [2.52, 3.08] (pooled, operator-cluster bootstrap). SGS-PCG beats it on every load.

**Seeds (set R):**
- Seeds 2 and 3 give the set C bands for every model.
- The channel init-0 models at seeds 2 and 3 are random-load models (R = 4.14 and 4.33). They are not replications of the fixed-load models.

## 9 Interpretation

**Supported:**
1. **On channels, OA-VCB is far more accurate per pass than every tested four-stage Jacobi-polynomial or Krylov smoother in the same cycle.** This includes the best full-cycle polynomial candidates we found. The gain is reproducible across three trainings, fresh operators and two further data seeds.
2. **On channels, OA-VCB's fine-level smoother beats the exact local K₄ minimiser on every tested input**, at both positions, on set C. Locally it also beats the K₅ minimiser. This is reserve-confirmed (R4) for all three initialisations at both positions, on 97–100 of 100 inputs.
3. **Larger Krylov budgets close the per-pass gap.** On channels, PCG with 8 to 14 steps per call brackets OA-VCB; on barriers, 5 to 8 steps. Both brackets are reserve-confirmed (R1, R2) for all three initialisations.
4. **No computational advantage is demonstrated.** OA-VCB's smoothing arithmetic is 10²–10³× that of the classical smoothers, which are more accurate at larger but still far smaller budgets. In the repaired timings, classical whole solves reach OA-VCB's accuracy much faster (§7).
5. **Barriers:** OA-VCB is not better than the polynomial class. The best candidate found beats it, mostly on islands.

**Not supported:**
- any separation from the complete class of polynomial-smoothed cycles;
- any statement about communication rounds, communication efficiency or efficiency "per stage";
- a spatial mechanism;
- that nonlinearity is required: a spatially varying linear operator can also leave the scalar-polynomial class;
- anything beyond L3, 2D, two coefficient families and this architecture.

**Threats to validity:**
- **Precision:** the learned cycle runs in float32 inside float64 deflation and output, while the classical cycles are float64. No precision ablation was run.
- **Optimisation is not certification:** the polynomial candidates come from L-BFGS without a certificate.
- **Configurations:** SGS-PCG and the PCG ladder were each tested in one configuration.
- **Timings:** Python implementation, one CPU.
- **Intervals** are pointwise.
- **"No failures"** covers the recorded checks only. The inner FCG of the K-cycle does not record its termination reason.

**Relevance to the submitted manuscript.** Nothing contradicts it. If a reviewer asks for Krylov smoothers or a K-cycle, §§2–5 answer the question at four stages and at larger budgets.

## 10 Route implications (for discussion; no route chosen)

| Route | Reading after the review |
|---|---|
| **B (mechanism)** | Passes the local-diagnostic part of the trigger: local separation from K₄ (exact, set C) and from K₅ (exact, reserve-confirmed). It does **not** establish a full-cycle class separation or any resource-based advantage. To become substantial it needs three things: a meaningful resource restriction, a provable or numerically certified separation under that restriction, and persistence beyond one load regime. |
| **B′ (evaluation)** | Only with a contribution more specific than "stronger baselines matter". That message is already a published contribution: McGreivy and Hakim, Nature Machine Intelligence, 2024. The candidate is a validated method that separates three things, with independent examples: local representational gain, complete-cycle gain and deployable solver benefit. Next step: a bounded external-reproduction pilot of one independently implemented learned smoother on its native benchmark. If it only repeats the known overhead lesson, stop. |
| **A (classical operator-adapted solver)** | Unchanged. The repaired profile attributes most of the classical one-pass time to building the operator-adapted hierarchy (§7). |
| **C (certified goal-oriented outputs)** | Unchanged and not reviewed in this round. Learned passes are unlikely to win on cost, so its viable form is a certification result, not a speed result. |

## 11 Provenance and checks

- **Records:** confirmation records and the v1 report are unchanged.
- **Amendment 2** was hashed before the reserve operators were generated. The reserve runner's gate confirms that the seed was absent from every record before generation. Chain log: `runs/chain_amendment2.log`.
- **New scripts:** `stage1_timing_v2.py`, `stage1_ledger.py`, `stage1_reserve.py`.
- **Claims check:** `stage1_report_check_rev1.py`. Named checks regenerate every table cell in this revision from raw records. The numeric-token scan is kept only as a typo check, as ASTRA notes; it does not establish that a claim is supported. Result on 18 Oct: 60 named checks, no failures. This includes the generated summary, ledger, timing, reserve and K-local sections, which are regenerated from the records and compared verbatim.

### Sources

- N. McGreivy, A. Hakim, *Weak baselines and reporting biases lead to overoptimism in machine learning for fluid-related partial differential equations*, Nature Machine Intelligence (2024); arXiv:2407.07218.
- R. Huang, R. Li, Y. Xi, *Learning optimal multigrid smoothers via neural networks*, arXiv:2102.12071. Baselines: ω-Jacobi, and Gauss–Seidel in the spectral-radius tables.
- L. Li, Y. K. Chu, R. Chan, J. Wan, *A level-wise training scheme for learning neural multigrid smoothers with application to integral equations*, arXiv:2603.01064. §4.2 / Table 4 compare whole-solve iterations and times against classical multigrid and CG; CG is faster for large α, the learned method for small α.
- E. Chillón, A. K. Lidtke, N. A. K. Doan, B. Font, *Acceleration of an algebraic multigrid pressure solver using graph neural networks*, arXiv:2606.19251. The smoother is a precomputed, residual-linear polynomial per operator. The baseline is relaxed Jacobi. Set-up, solve time and amortisation are reported.
- Z. Zhang, R. Li, Y. Saad, *Graph Neural Multilevel Preconditioners for Iterative Solvers*, arXiv:2607.28456. Abstract only: over 800 matrices, against AMG, ILUT and GNN preconditioners, including overhead regimes.
- J. Lottes, *Optimal polynomial smoothers for multigrid V-cycles*, arXiv:2202.08830. The roughly 18% figure concerns an asymptotic contraction-factor bound, not measured runtime.
- M. Adams, M. Brezina, J. Hu, R. Tuminaro, *Parallel multigrid smoothing: polynomial versus Gauss–Seidel*, J. Comput. Phys. 188 (2003) 593–610.
- Y. Notay, P. S. Vassilevski, *Recursive Krylov-based multigrid cycles*, Numer. Linear Algebra Appl. 15 (2008) 473–487.
