# Stage 1 report: what learned smoothing adds beyond classical smoothing in the same cycle

17 October 2026. Written by Claude in extra-effort mode, for the author and ASTRA.

**Basis:**
- Protocol v1 (`STAGE1_PROTOCOL_2026-10-16.md`), amendment 1, and the pre-confirmation note (SHA-256 `66be1dc3…`, hashed before set C existed).
- The confirmation chain `runs/run_confirmation.sh` (`runs/chain.log`): every run exited 0 with all gates passed.
- Every number below is recomputed from the raw run records by `stage1_tables.py`, `stage1_work.py`, `stage1_matched_cost.py` and the claims check `stage1_report_check.py` (§10).

**Labels used throughout:**
- **Confirmatory:** P1–P3 on the fresh set C (amendment 1, A5.4).
- **Secondary:** pre-declared comparisons, on set C unless stated.
- **Exploratory:** analyses I added after set C was opened. They were run on development set D only and are labelled at each use. They are not Stage 1 results; §9 proposes how to confirm the ones that matter.
- **Name clash:** "diagnostic D1/D2" below means the local K₄ test and the oracle-fitted polynomial of protocol §3.4. They are not the roadmap's learned set-up directions, which share the names.

## 1 Answer in brief

**Confirmatory (fresh set C, three OA-VCB trainings each).** Same operator-adapted V-cycle and hierarchy, same deflation and Galerkin output, four update stages per smoother; only the smoother changes.

| | Channels | Barriers |
|---|---|---|
| P1: R = G(PCG → OA-VCB), inits 0 / 1 / 2 | 5.70 / 5.94 / 6.12 | 1.45 / 1.42 / 1.38 |
| P1 lower 95% limits | 5.25 / 5.56 / 5.71 | 1.33 / 1.28 / 1.24 |
| P1 band (all three inits) | **not accounted for by Krylov smoothing** | **partly accounted for / inconclusive** |
| P2: G(POLY-S → OA-VCB) | 5.14 / 5.37 / 5.53 | 1.76 / 1.73 / 1.67 |
| P3: G(CHEB → PCG) | 1.61 | 1.37 |

**What these numbers mean, combining the secondary and exploratory evidence:**

1. **At a fixed number of update stages, OA-VCB's channel gain is not reproduced by any Jacobi-preconditioned polynomial or Krylov smoother we could find.**
   - This includes the per-operator oracle-fitted polynomial over the whole Krylov class (diagnostic D2): OA-VCB is 4.1–4.5× more accurate, W ≥ 0.99.
   - Locally, OA-VCB's fine-level smoother beats the best element of K₄ on every channel operator.
   - The learned smoother reaches one hop further than a K₄ smoother, so the fair class is K₅. Against the oracle over K₅ the channel gap is smaller but persists: about 2.4×, W = 1.00 (exploratory, 20 set-D operators; a run three times longer on 10 of them changed it by less than 0.5%, but the optimum is not certified).
   - On barriers there is no such separation. The oracle polynomial beats OA-VCB (G 0.72–0.75), mostly on island operators.
2. **What the learned smoother achieves per stage is roughly what Jacobi-Krylov smoothing achieves with two to three times as many steps.** Exploratory, set D:
   - locally, it beats the K₅ minimiser on every channel operator, but in median not the K₈ minimiser (median ρ_K₈ = 1.02 pre-smoother, 1.08 post-smoother);
   - in the cycle, OA-VCB sits between Jacobi-PCG(8) and PCG(14) per smoothing call on channels, and between PCG(5) and PCG(8) on barriers.
3. **It is not an advantage at matched arithmetic.**
   - **Arithmetic:** one OA-VCB pass costs 685 million multiply-adds (MACs), the equivalent of about 29 000 fine-level matrix–vector products. That is about 590× the PCG-smoothed pass and about 210× the SGS-PCG pass.
   - **Jacobi-PCG(14):** about 190× fewer MACs than OA-VCB, and 2.4× (channels) and 2.9× (barriers) more accurate. Exploratory, set D.
   - **Whole solve:** two-grid PCG reaches OA-VCB's per-sample error in a median 4 ms, against OA-VCB's 370 ms latency.
4. **The spatial control (SGS-PCG) is close to OA-VCB on channels and better on barriers.**
   - Channels: G(SGS-PCG → OA-VCB) = 1.16–1.24. Barriers: 0.55–0.58, so SGS-PCG is the more accurate.
   - **Correction to my earlier development-phase reading:** I told the author that "the learned smoother's advantage appears to be spatial". The exploratory work-matched comparison does not support that. Jacobi-PCG(14) has the same matrix–vector-equivalent count as SGS-PCG(4) (90 per pass; 3.56 M against 3.25 M MACs), and it is more accurate than SGS-PCG(4) on both families. SGS-PCG's result is explained mainly by its larger smoothing work, not by the Gauss–Seidel ordering.
5. **Robustness of the learned gains:**
   - The fixed-load models lose to plain PCG on every new load (set LC), apart from one tie (init 2 on the random-load draw).
   - The random-load model keeps R = 2.8 (pooled) on new loads but is less accurate than SGS-PCG there.
   - Seeds 2–3 reproduce the set C bands for every model (set R).

**So learned smoothing adds a real, reproducible gain per update stage (per neighbour-exchange round) on channels, beyond the whole Jacobi-Krylov class at that stage count. It adds nothing on barriers that classical smoothing cannot match, and it adds nothing at matched arithmetic or time on either family.**

## 2 What was run

| Set | Content | Status | Records |
|---|---|---|---|
| C | 100 fresh operators per family (ids 300–399 of rng 2026101601 / 2026101602) | confirmatory P1–P3; everything else secondary | `runs/confirm_C_*.json` |
| C (timing) | First 50 operators per family, run alone, 5 repeats | diagnostic | `runs/timing_C_*.json` |
| C (D1, D2) | 100 operators per family | diagnostic | `runs/d1_C_*.json`, `runs/d2_C_*.json` |
| S | Fresh stratified barriers: 20 per category; 1208 draws (cap 3000 not reached) | secondary, descriptive | `runs/confirm_S_barriers.json` |
| LC | 50 channel operators of C × 4 new loads | secondary | `runs/confirm_LC_channels.json` |
| R | Seeds 2 and 3, both families, previously inspected | secondary, retrospective | `runs/confirm_R{2,3}_*.json` |
| D (exploratory) | Seed-1 test operators (development) | exploratory, post-hoc | `explore/*.json` |

- **Failures:** none. No crash, non-finite output, breakdown or backstop activation in any method, on any set.
- **Guarded variants** (a failure takes u_c's error) are therefore identical to the raw ones.
- **Gates** (non-zero exit on failure), every one of which passed:
  - fitted-coefficient hash;
  - CHEB common path equal to the legacy anchor, maximum relative difference 7.8e-16 (channels) and 8.4e-13 (barriers) on set C;
  - checkpoint hashes against the checkpoint map;
  - three initialisations present.

## 3 Confirmatory results (set C)

### Channels (n = 100)

| Model | P1: R = G(PCG → OA-VCB) [95%] | W | Band | P2: G(POLY-S → OA-VCB) [95%] | W |
|---|---|---|---|---|---|
| OA-VCB init 0 | 5.70 [5.25, 6.13] | 0.99 | not accounted for by Krylov smoothing | 5.14 [4.79, 5.51] | 1.00 |
| OA-VCB init 1 | 5.94 [5.56, 6.32] | 1.00 | not accounted for by Krylov smoothing | 5.37 [4.96, 5.81] | 1.00 |
| OA-VCB init 2 | 6.12 [5.71, 6.54] | 1.00 | not accounted for by Krylov smoothing | 5.53 [5.13, 5.95] | 1.00 |

P3 = G(CHEB → PCG) = 1.61 [1.58, 1.64], W = 1.00. **Family label: not accounted for by Krylov smoothing.**

### Barriers (n = 100)

| Model | P1: R [95%] | W | Band | P2 [95%] | W |
|---|---|---|---|---|---|
| OA-VCB init 0 | 1.45 [1.33, 1.59] | 0.82 | partly accounted for / inconclusive | 1.76 [1.60, 1.93] | 0.91 |
| OA-VCB init 1 | 1.42 [1.28, 1.58] | 0.77 | partly accounted for / inconclusive | 1.73 [1.55, 1.91] | 0.90 |
| OA-VCB init 2 | 1.38 [1.24, 1.54] | 0.76 | partly accounted for / inconclusive | 1.67 [1.51, 1.85] | 0.86 |

P3 = 1.37 [1.32, 1.43], W = 0.99. **Family label: partly accounted for / inconclusive.**

**Notes:**
- All intervals are pointwise. The bands are descriptive labels, applied in the frozen order.
- The confirmatory values are a little below the development values on set D: channels R 6.3–6.6 and barriers 1.50–1.57 on D. They are in the same band.
- **Recovered log-gain** φ_Y = log G(CHEB → Y) / log G(CHEB → OA-VCB), descriptive, OA-VCB init 0:
  - channels: φ(PCG) = 0.22 [0.21, 0.22]; φ(POLY-L) = 0.28;
  - barriers: φ(PCG) = 0.46 [0.39, 0.54].

  The Jacobi-Krylov smoother at four stages therefore recovers about a fifth of the channel log-gain and about half of the barrier log-gain.

## 4 The polynomial/Krylov class: how far is OA-VCB from it?

### 4.1 The ladder on set C (geometric-mean gain over CHEB; init-0 comparisons)

| Method (one pass, same cycle) | Class | Channels G(CHEB → X) | Barriers G(CHEB → X) | Channels G(X → OA-VCB) | Barriers G(X → OA-VCB) |
|---|---|---|---|---|---|
| SD / MR | Jacobi-Krylov, adaptive | 0.82 / 0.83 | 0.82 / 0.79 | 11.3 / 11.1 | 2.42 / 2.52 |
| K2-cycle (CHEB) | recursive Krylov cycle | 0.97 | 1.40 | 9.48 | 1.42 |
| CHEB-T\* (Lanczos, q = 10) | fixed polynomial | 1.31 | 1.12 | 6.99 | 1.79 |
| PCG (P1) | Jacobi-Krylov, K₄ minimiser per call | 1.61 | 1.37 | 5.70 | 1.45 |
| POLY-S / POLY-L | fitted fixed polynomial | 1.79 / 1.86 | 1.13 / 1.30 | 5.14 / 4.93 | 1.76 / 1.54 |
| **D2: oracle polynomial** | per-call oracle over K₄ (candidate) | **2.22** | **2.65** | **4.14** [3.82, 4.45] | **0.75** [0.68, 0.83] |
| OA-VCL / OA-VCBL (learned, residual-linear) | learned linear | 2.48 / 2.60 | 0.67 / 1.10 | 3.70 / 3.53 | 2.99 / 1.81 |
| SGS / SGS-PCG | spatial (pre-declared, triggered) | 1.27 / 7.94 | 1.01 / 3.44 | 7.23 / 1.16 | 1.97 / 0.58 |
| OA-VCB init 0 / 1 / 2 | learned nonlinear | 9.18 / 9.58 / 9.87 | 2.00 / 1.96 / 1.90 | — | — |

**Why diagnostic D2 bounds the whole Jacobi-Krylov class at this stage count:**
- In one V-cycle pass each level's pre- and post-smoother is called exactly once. D2 chooses an independent polynomial for each of these six calls, with the exact solution available.
- So its feasible set contains every smoother whose output lies in K₄(input) of its own call: CHEB, CHEB-T, POLY, SD, MR and PCG.
- **Caveat:** D2 is a numerical optimum from five starts and is not certified. The true class optimum can only be more accurate. §4.3 tests how much this matters.

**Reading:**
- **Channels:** the whole ladder of Jacobi-Krylov smoothing at four stages reaches at most G ≈ 2.2 over CHEB, against OA-VCB's 9.2–9.9. The gap to OA-VCB stays ≥ 4.1 even for the oracle (W = 0.99 for init 0; 1.00 for inits 1 and 2).
- **Channels, learned linear models:** they reach 2.5–2.6. That is slightly above the oracle K₄ polynomial (G(D2 → OA-VCBL init 0) = 1.17), possibly because of their extra hop of reach (§4.3).
- **Barriers:** the oracle polynomial (2.65) is above OA-VCB (2.0). The PCG → OA-VCB gap of 1.4 is a gap between OA-VCB and one particular member of the class, not the class itself.

### 4.2 Local test D1 (set C, OA-VCB init 0, fine level)

ρ = E_NN / E_K₄ for the same input; a local separation is described when the rank is 4 and ρ ≤ 0.99.

| Family | Position | ρ 10 / 50 / 90% | δ (median A-distance from K₄) | Local separation |
|---|---|---|---|---|
| channels | pre | 0.566 / 0.614 / 0.673 | 0.249 | 100/100 |
| channels | post | 0.420 / 0.491 / 0.748 | 0.145 | 100/100 |
| barriers | pre | 0.929 / 0.998 / 5.845 | 0.207 | 44/100 |
| barriers | post | 0.697 / 0.841 / 0.977 | 0.088 | 94/100 |

- **Rank and cross-check:** the rank is 4 everywhere, and PCG(4) equals the K₄ minimiser to 3.8e-13.
- **Channels:** for its own input, the learned fine-level smoother produces an output whose A-norm lies about a quarter outside K₄ (median δ 0.25 pre, 0.14 post). Its energy-norm error is 39–51% smaller (median) than that of the best element of K₄.
- **Barriers:** the pre-smoother is locally indistinguishable from the K₄ optimum in median, and much worse on a tail (90th percentile ρ = 5.8). The post-smoother does beat K₄ locally (94/100), but not K₅ in median (§4.3). The barrier gain therefore does not come from the fine-level pre-smoother.

### 4.3 Exploratory: reach and degree (set D only; post-hoc)

The learned smoother runs four message-passing steps, so its output at a node depends on the 4-hop neighbourhood. A K₄ output p(B)D⁻¹b with deg p ≤ 3 reaches 3 hops. The matched-reach Krylov class is therefore K₅. Two checks address this and the D2 optimisation cap.

**(a) Local ρ against larger Krylov spaces** (`explore/x1_x3_*.json`; set D, OA-VCB init 0, 100 operators per family):

| Family | Position | median ρ_K₄ | median ρ_K₅ | median ρ_K₈ | ρ_K₅ ≤ 0.99 | ρ_K₈ ≤ 0.99 |
|---|---|---|---|---|---|---|
| channels | pre | 0.624 | 0.715 | 1.024 | 100/100 | 23/100 |
| channels | post | 0.498 | 0.620 | 1.079 | 100/100 | 33/100 |
| barriers | pre | 1.005 | 1.054 | 1.232 | 6/100 | 0/100 |
| barriers | post | 0.848 | 0.976 | 1.159 | 63/100 | 1/100 |

**(b) Smoothing degree in the cycle.** Jacobi-PCG(k) per call, same wrapper, set D, against OA-VCB init 0:

| k | matvec-equivalents / pass | Channels G(CHEB → X) | Channels G(X → OA-VCB) [95%] | Barriers G(CHEB → X) | Barriers G(X → OA-VCB) [95%] |
|---|---|---|---|---|---|
| 4 (= P1's PCG) | 30 | 1.62 | 6.31 [6.02, 6.58] | 1.36 | 1.57 [1.41, 1.74] |
| 5 (reach-matched) | 36 | 2.62 | 3.91 [3.73, 4.08] | 1.76 | 1.21 [1.08, 1.35] |
| 8 | 54 | 6.86 | 1.49 [1.43, 1.56] | 2.90 | 0.74 [0.64, 0.85] |
| 14 | 90 | 24.9 | 0.41 [0.39, 0.44] | 6.21 | 0.34 [0.28, 0.42] |
| 32 | 198 | 669 | 0.015 [0.012, 0.019] | 25.4 | 0.084 [0.059, 0.121] |
| SGS-PCG(4) | 90 | 8.22 | 1.24 [1.19, 1.30] | 3.33 | 0.64 [0.55, 0.75] |
| SGS-PCG(8) | 162 | 54.8 | 0.19 [0.17, 0.21] | 8.48 | 0.25 [0.20, 0.32] |

The PCG(4) and SGS-PCG(4) rows reproduce the development records exactly.

**(c) Oracle polynomial: continued optimisation and the reach-matched K₅ oracle** (`explore/x2_*.json`; first 20 operators of set D per family; G(X → OA-VCB init 0) [95%], W).

| Variant | Channels G(CHEB → X) | Channels G(X → OA-VCB) | Barriers G(CHEB → X) | Barriers G(X → OA-VCB) |
|---|---|---|---|---|
| K₄ oracle as kept (D2, 500 iterations) | 2.31 | 4.26 [3.83, 4.69], W 1.00 | 3.23 | 0.77 [0.61, 0.95], W 0.30 |
| K₄ oracle, 2500 further iterations | 2.31 | 4.25 [3.82, 4.68], W 1.00 | 3.45 | 0.72 [0.55, 0.91], W 0.30 |
| PCG(5), not an oracle | 2.63 | 3.74 [3.36, 4.11], W 1.00 | 1.88 | 1.32 [1.09, 1.61], W 0.70 |
| **K₅ oracle** (reach-matched; 2 starts × 1500 iterations) | **4.17** | **2.36** [2.15, 2.57], W 1.00 | **6.24** | **0.40** [0.31, 0.50], W 0.00 |
| OA-VCB init 0 (these 20 operators) | 9.84 | — | 2.48 | — |

- **K₄ oracle:** converged in practice on channels. Continuing it changed no operator's log error by more than 0.012, and 15 of 20 runs stopped before the cap. On barriers it improved by 7%.
- **K₅ oracle:** most runs stopped at the iteration cap (19/20 channels, 20/20 barriers), so its gaps are upper bounds on the true K₅ gap. A convergence check on the first 10 channel operators (`explore/x2b_channels.json`; both starts run to 4500 iterations) moved G(K₅ oracle → OA-VCB) from 2.392 to 2.386; the largest per-operator change in log error was 0.012. The channel gap of about 2.4 is therefore not an artefact of the cap, though it remains a numerical, not a certified, optimum.

**Reading:**
- **Channels:** the learned smoother separates from the matched-reach Jacobi-Krylov class (K₅), both locally (every operator) and through the cycle. Against the reach-matched oracle the gap shrinks from about 4.3 to 2.4 (or less), with W = 1.00. In the cycle, PCG closes the gap between 8 and 14 steps per call.
- **Barriers:** there is no separation from K₅ locally. In the cycle, PCG(5) is within 1.2× of OA-VCB (1.21 on all 100 operators, 1.32 on these 20), and the reach-matched oracle polynomial is more accurate than OA-VCB on all 20 operators (G 0.40).
- **Efficient per stage, not beyond Krylov smoothing.** The local K₈ comparison shows that the learned output is not beyond what eight Krylov steps achieve. The learned smoother is efficient per stage, but not outside the reach of longer Krylov smoothing.

## 5 The spatial control and matched work

### 5.1 SGS and SGS-PCG on set C

The spatial-control trigger fired on set D before confirmation, so these are pre-declared secondary comparisons.

| | Channels G(X → OA-VCB) init 0 / 1 / 2 | W (init 0) | Barriers G(X → OA-VCB) init 0 / 1 / 2 | W (init 0) |
|---|---|---|---|---|
| SGS (2 symmetric sweeps) | 7.23 / 7.54 / 7.77 | 0.99 | 1.97 / 1.93 / 1.87 | 0.91 |
| SGS-PCG (4 PCG steps, SGS-preconditioned) | 1.16 / 1.21 / 1.24 | 0.79 | 0.58 / 0.57 / 0.55 | 0.29 |

- On channels OA-VCB remains significantly more accurate than SGS-PCG; the lower limits are 1.08, 1.15 and 1.18.
- On barriers SGS-PCG is significantly more accurate than OA-VCB for all three inits.

### 5.2 Work per pass (`runs/stage1_work.json`)

Work is counted in multiply-adds (MACs) from the level sizes (n, nnz) = (3486, 23 502), (814, 52 224), (174, 30 276). The operator-adapted coarse levels are dense: 64 and 174 entries per row.

| Method | MACs per pass | in fine-level matvecs | OA-VCB / method |
|---|---|---|---|
| POLY-S / POLY-L | 0.88 M | 37.6 | 775 |
| CHEB, CHEB-T | 1.10 M | 46.6 | 625 |
| PCG, SD | 1.17 M | 49.7 | 587 |
| SGS | 1.06 M | 45.1 | 647 |
| K2-cycle | 2.77 M | 117.9 | 247 |
| SGS-PCG | 3.25 M | 138.4 | 211 |
| Jacobi-PCG(14), exploratory | 3.56 M | 151.3 | 193 |
| **OA-VCB** (analytic; dense layers cross-checked against torch's FlopCounterMode, see below) | **685 M** | **29 162** | 1 |

- **What the OA-VCB count leaves out:** GELU activations, gathers and the O(n) normalisation are not counted, so the learned count is a lower bound.
- **Cross-check:** the analytic dense-layer counts equal the FlopCounterMode counts on the two coarse levels. On the fine level they differ by 0.3%, because the assembled fine matrix stores 90 explicit zeros that the network's pattern omits.
- **What "four update stages" matches:** the same number of neighbour exchanges per smoothing call. It does not match arithmetic.

**Exploratory results at matched work (set D):**
- Jacobi-PCG(14) has the same matvec-equivalent count as SGS-PCG(4) and is more accurate than it on both families: G(CHEB → X) 24.9 against 8.2 (channels) and 6.2 against 3.3 (barriers). It is also more accurate than OA-VCB on both families at about 1/190 of its arithmetic.
- **Consequence:** I withdraw my earlier development-phase reading that the remaining channel gain is "spatial". SGS-PCG's advantage over PCG(4) is mostly its extra work.

### 5.3 Wall-clock (set C, first 50 operators; latency per operator, median of 5 repeats; 2 threads, Xeon 2.1 GHz)

| Method | Channels | Barriers |
|---|---|---|
| OA-VCB, one pass (batch 1; operator data 163 / 164 ms) | 370 ms, error 2.56e-3 | 371 ms, error 1.47e-2 |
| Classical one pass (any smoother) | 79–83 ms | 80–83 ms |
| Sparse direct solve | 10.0 ms | 10.1 ms |
| Two-grid PCG, time to OA-VCB's per-sample error (offline oracle stopping, median) | 4.3 ms (6 it.) | 4.0 ms (4 it.) |
| SA-AMG PCG, same | 14.0 ms | 14.6 ms |
| PCG-V / FCG-K2 / FCG with PCG-smoothed V, same | 72 / 86 / 67 ms | 73 / 85 / 69 ms |
| 1–4 passes of CHEB / PCG / SGS-PCG cycles, same | 82 / 81 / 86 ms | 83 / 83 / 86 ms |

- **Censoring:** no curve was censored.
- **Equal time:** at OA-VCB's latency, every classical whole-solve curve had already reached its frozen iteration limit. Its error was smaller than OA-VCB's by factors of 40 (CHEB passes, channels) up to more than 10⁸ (PCG-V); see `runs/stage1_matched_cost.json`.
- **Figure:** `runs/fig_stage1_accuracy_time.png` / `.pdf`.
- **Throughput (secondary):** OA-VCB at batch 8 takes 206 / 220 ms per operator.

These timings measure this Python implementation at 3486 unknowns, not optimised implementations. They agree with the manuscript's §5.3 conclusion: at equal cost the learned pass is not the more accurate one.

## 6 Where the barrier result comes from (sets C and S)

Results are for OA-VCB init 0; G(X → OA-VCB) with 95% intervals.

| Category (n in C / S) | PCG (C) | K2 (C) | SGS-PCG (C) | D2 oracle (C) | PCG (S) | K2 (S) | SGS-PCG (S) |
|---|---|---|---|---|---|---|---|
| island (42 / 20) | 1.45 [1.20, 1.72] | 0.95 [0.80, 1.13] | 1.10 [0.90, 1.33] | 0.57 [0.47, 0.69] | 1.30 [0.93, 1.77] | 0.85 [0.60, 1.19] | 1.01 [0.73, 1.38] |
| strict non-island (52 / 20) | 1.49 [1.37, 1.63] | 1.99 [1.82, 2.17] | 0.37 [0.32, 0.43] | 0.96 [0.89, 1.04] | 1.54 [1.38, 1.71] | 2.02 [1.81, 2.24] | 0.39 [0.33, 0.46] |
| EDP (4 / 20) | 1.04 | 1.01 | 0.29 | 0.52 | 1.53 [1.21, 1.91] | 1.36 [0.87, 2.05] | 0.44 [0.31, 0.61] |
| discordant (2 / 20) | n < 3 | | | | 1.16 [0.83, 1.58] | 1.28 [0.83, 1.92] | 0.30 [0.21, 0.41] |

**On island operators:**
- The K-cycle, with its better coarse-level solves, matches OA-VCB.
- The per-call oracle polynomial beats OA-VCB by 1.8×.
- SGS-PCG ties with OA-VCB.

This points to the coarse and near-kernel part of the cycle, rather than fine-level smoothing, as where the island difficulty lies. It agrees with D1, where the barrier pre-smoother does not beat K₄ locally.

**On strict non-island operators:**
- OA-VCB ties with the oracle polynomial.
- SGS-PCG beats OA-VCB 2.7×.

Set S, drawn fresh and stratified, reproduces the set C pattern in every category with enough samples.

## 7 Loads and seeds

### 7.1 Load challenge (set LC; 50 channel operators of C × 4 loads)

Entries are G(X → model); the pooled row uses an operator-cluster bootstrap. Column F0, F1, F2 = fixed-load OA-VCB init 0, 1, 2; RL = the random-load model.

| Load | PCG → F0 / F1 / F2 | PCG → RL | SGS-PCG → RL |
|---|---|---|---|
| f = −1 | 0.15 / 0.09 / 0.48 | 3.49 | 0.72 |
| smooth ± | 0.17 / 0.09 / 0.54 | 3.64 | 0.70 |
| point source (width 0.03) | 0.27 / 0.61 / 0.56 | 1.38 | 0.24 |
| random-load draw | 0.35 / 0.26 / 0.97 | 3.47 | 0.71 |
| pooled | 0.23 / 0.19 / 0.61 | 2.79 [2.52, 3.08] | 0.54 [0.49, 0.59] |

- **Fixed-load models:** less accurate than one-pass PCG on all four loads, except init 2 on the random draw (0.97, interval includes 1).
- **Random-load model:** keeps a 1.4–3.6× advantage over PCG, but SGS-PCG beats it on every load.
- **Classical methods:** load-agnostic. The PCG geometric-mean error stays within 1.0–1.4e-2 across loads.

### 7.2 Retrospective seeds (set R)

| Model | R = G(PCG → model), seed 2 | R, seed 3 | Band |
|---|---|---|---|
| channels, fixed-load inits 1–2 | 6.10, 6.19 | 6.18, 6.39 | not accounted for (all) |
| channels, random-load init 0 | 4.14 | 4.33 | not accounted for |
| barriers, inits 0–2 | 1.47, 1.47, 1.50 | 1.40, 1.39, 1.30 | partly accounted for / inconclusive (all) |

- G(SGS-PCG → model) is 1.17–1.30 for the channel fixed-load models and 0.51–0.66 for barriers. The set C picture holds across data seeds.

## 8 What learned smoothing adds: interpretation

### 8.1 Supported by the evidence

1. **A per-stage advantage on channels beyond Jacobi-Krylov smoothing at matched stage count.** It holds:
   - confirmatory against PCG;
   - against the fitted polynomials;
   - against the per-call oracle polynomial (a candidate, from five starts);
   - locally against the K₄ and, exploratory, K₅ minimisers;
   - exploratory, against the reach-matched K₅ oracle (about 2.4×, W = 1.00 on 20 operators; stable under three times longer optimisation).

   It is reproduced over three trainings, fresh operators and two further data seeds.
2. **Size of that advantage:** OA-VCB is about as good as PCG with 8–14 steps per call, or SGS-PCG with 4. Exploratory, set D.
3. **Barriers:** no separation from classical smoothing. The PCG gap (1.4) is within the reach of the oracle polynomial, PCG(5–8), the K-cycle on islands and SGS-PCG on non-islands.

### 8.2 Not supported

- **"Learned smoothing is more efficient":** at matched arithmetic or matched time, classical methods are more accurate on both families. For example, Jacobi-PCG(14) is more accurate at about 1/190 of OA-VCB's MACs (exploratory).
- **"The gain is spatial":** withdrawn (§5.2).
- **"The gain requires nonlinearity":** neither shown nor refuted.
  - The classical smoothers that match or beat OA-VCB at more work (PCG(k), SGS-PCG) are themselves residual-adaptive, that is, nonlinear in the input.
  - The manuscript's sixteen-step fixed (linear) Chebyshev narrows the channel gap to 1.8.
  - The manuscript's OA-VCB vs OA-VCBL comparison (3.5× on channels here) compares two trained models. It does not establish a requirement for nonlinearity.
- **Any statement beyond L3, 2D, two coefficient families and this architecture.**

### 8.3 Threats to validity

- **Scope:** one architecture and one mesh size (3486 unknowns), 2D, two coefficient families, operator-adapted hierarchy only.
- **D2 is not certified.** The class optimum may be more accurate; §4.3(c) bounds this for 20 operators per family.
- **Exploratory analyses are post-hoc,** run on development set D and chosen after seeing set C. Claims that rely on them need the reserve set (§9).
- **Precision:** the learned cycle runs in float32 inside float64 deflation and output, while classical cycles are float64. This can only disadvantage the learned model, slightly.
- **SGS-PCG** was run in one configuration: natural ordering, one symmetric sweep per application.
- **Timings:** one CPU, Python, latency dominated by the operator-adapted hierarchy set-up. GPU inference was not tested.
- **Bands and intervals:** bands are labels and intervals are pointwise. With 3 inits × 2 families × 3 comparisons, a few intervals would miss by chance; no claim rests on a single interval.

### 8.4 Relevance to the submitted manuscript

Nothing in Stage 1 contradicts the manuscript. It already states that at equal cost the learned pass is not the more accurate one, and that stronger classical budgets narrow or reverse the gain. If a reviewer asks "why not Krylov smoothers or a K-cycle?", Stage 1 gives a ready answer:
- at four stages, PCG, fitted and oracle polynomials and the K-cycle do not reach the channel gain;
- with more classical work, they do.

The SGS-PCG and PCG(k) results would be natural additions to the manuscript's smoothing-budget section (§5.3) in a revision.

## 9 Consequences for the route choice

These are for discussion with ASTRA; no route is chosen here, as the protocol requires.

| Route | Stage 1 evidence | My reading |
|---|---|---|
| **B (restricted-class separation)**: ASTRA's trigger needs "a reproducible gap [that] survives the stronger controls, and a precisely defined restricted-class separation" | **Channels:** a reproducible gap survives every Jacobi-Krylov control at matched stage count, including the oracle polynomial (4.1–4.5×) and, exploratory, the reach-matched K₅ oracle (about 2.4×). The class is precise: per-call K_d smoothing, with d tied to the number of neighbour-exchange rounds. **It does not survive** larger stage counts or matched work. **Barriers:** none. | The separation is real but narrow: it concerns the number of neighbour-exchange rounds, not arithmetic or accuracy per cost. A theory paper ("what can d-hop nonlinear local operators achieve per neighbour-exchange round that K_{d+1} cannot?") would need a setting where rounds, not flops, are the bottleneck. The learned pass here costs about 200–775× the arithmetic of the classical one-pass smoothers. **Weak practical motivation; high risk.** |
| **B′ (evaluation)** | The central lesson is quantitative and transferable: a comparison at matched update stages can hide a two-to-three-orders-of-magnitude difference in arithmetic, and the per-stage gain closes within a modest Krylov ladder. The tools are reusable: oracle polynomial, local K_d test, PCG(k) ladder, MAC accounting. | **Strongest candidate**, if breadth can be added: the same protocol applied to one or two independently implemented learned smoothers on their own benchmarks. Two learned-smoother papers I checked compare only against Jacobi-type relaxation, so their per-stage gains against a Krylov ladder are untested. Huang, Li and Xi's CNN smoothers are compared against ω-Jacobi, with Gauss–Seidel in the spectral-radius tables only. Chillón et al.'s GNN-predicted polynomial smoother for AMG (2026) is compared against ω-Jacobi. Needs ASTRA's view on novelty against the evaluation literature. |
| **A (classical OA solver)** | Untouched. Stage 1 re-confirms that the operator-adapted hierarchy, not smoothing, dominates cost (about 80 ms against 4 ms for the full two-grid PCG solve). | Unchanged: home ground, but needs truncation and scale. |
| **C (certified QoI)** | Untouched. | Unchanged: best fit to the author's background. |

**Before any claim relying on §4.3 or §5.2 is used in a paper,** confirm the three exploratory findings on the unopened reserve operators (rng 2026101603 channels / 2026101604 barriers), per amendment 1, A5.5:
- (i) PCG(k) brackets OA-VCB between k = 8 and 14 on channels;
- (ii) the K₅ local separation on channels;
- (iii) the reach-matched K₅ oracle gap on channels (about 2.4×).

That is about 1–2 hours of CPU, with a short protocol amendment first.

## 10 Provenance and claims check

- **Pre-confirmation note:** SHA-256 `66be1dc3bbadf920…`. `runs/chain.log` records the hash check passing at 2026-10-16T19:48Z, before set C was generated.
- **Executed-source manifests:** written by every run. The confirmation runs used the code hashed in `stage1_preconf_manifest.json`. All 16 `ext/*.py` files listed there still have the frozen hashes (rechecked 17 Oct).
- **Added after confirmation:** four reporting-only scripts (`stage1_tables.py`, `stage1_work.py`, `stage1_matched_cost.py`, `stage1_report_check.py`) and the exploratory scripts in `explore/`. None changes a confirmation record.
- **Claims check:** `stage1_report_check.py` recomputes each headline number from the raw records and checks every other number in this report against the computed set at its printed precision. Result on 17 Oct: all headline and table checks passed. Of the 521 decimal or scientific numbers in the report, every one matches a computed quantity except the four arXiv identifiers. Integer counts were checked in the named table checks.

### Sources consulted for §9

- R. Huang, R. Li, Y. Xi, *Learning optimal multigrid smoothers via neural networks*, arXiv:2102.12071. Learned CNN smoothers; experiments compare against ω-Jacobi, with Gauss–Seidel only in the spectral-radius tables.
- L. Li, Y. K. Chu, R. Chan, J. Wan, *A level-wise training scheme for learning neural multigrid smoothers with application to integral equations*, arXiv:2603.01064. The abstract names no specific classical smoother baseline.
- J. Lottes, *Optimal polynomial smoothers for multigrid V-cycles*, arXiv:2202.08830. V-cycle-optimal polynomial smoothers, about 18% better than fourth-kind Chebyshev in the many-step limit; a stronger fixed-polynomial control than CHEB-T for any follow-up.
- E. Chillón, A. K. Lidtke, N. A. K. Doan, B. Font, *Acceleration of an algebraic multigrid pressure solver using graph neural networks*, arXiv:2606.19251. A GNN predicts per-level polynomial coefficients of a Jacobi-type smoother. The baseline is relaxed Jacobi (ω = 2/3); Gauss–Seidel and SOR are discussed but not compared.
- M. Adams, M. Brezina, J. Hu, R. Tuminaro, *Parallel multigrid smoothing: polynomial versus Gauss–Seidel*, J. Comput. Phys. 188 (2003) 593–610: the classical per-stage versus parallel-cost trade-off that §8–9 extends to learned smoothers.
- Y. Notay, P. S. Vassilevski, *Recursive Krylov-based multigrid cycles*, Numer. Linear Algebra Appl. 15 (2008) 473–487 (K-cycle).
