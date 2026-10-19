# Stage 1 pre-confirmation note

17 October 2026. Required by protocol v1 §7 and amendment 1 A5.6.

**Status when this note was written and hashed:**
- **No set C, S or LC data had been generated or inspected.** The only fresh-seed code-path tests used development data: LC on five set-D operators, and S from rng 12345 with 2 per category. Both outputs are labelled "SMOKE TEST" and were discarded.
- Everything below comes from set T (fits), set D (development) or small checks.
- The machine-readable selections are in `ext/stage1_preconfirmation.json`. The confirmation runners **read** them and never re-derive a selection.

## 1 Frozen selections (A5)

### 1.1 CHEB-T\* and the bound policy

**CHEB-T\*: Lanczos × 1.1, q = 10, both families.** Selected on set T by mean log relative energy error of the final projected output (300 operators per family).

| bound, q | channels | barriers |
|---|---|---|
| Gershgorin, 4 | −3.408 | −3.061 |
| Gershgorin, 10 | −3.867 | −3.359 |
| Gershgorin, 30 | −3.569 | −2.978 |
| Lanczos × 1.1, 4 | −3.508 | −3.138 |
| **Lanczos × 1.1, 10** | **−3.983** | **−3.427** |
| Lanczos × 1.1, 30 | −3.614 | −2.987 |

**Bound policy, by method:**

| Method | Interval / bound | Status |
|---|---|---|
| CHEB (legacy anchor) | Fine level λ = 3; coarse levels 1.05 × 30-step power iteration | Unchanged. Below the exact λ_max on coarse levels on 0–10% of operators, by up to 7% (A3.1). |
| CHEB-T\* | Lanczos (10 steps) × 1.1 on every level | Not guaranteed: below exact by < 1% on at least one operator. Used only as a smoother interval, never for a PCG preconditioner. |
| POLY-L, D2 | Per-level scales s_l = Lanczos × 1.1 | — |
| POLY-S | One scale β = max over levels and T operators of Lanczos × 1.1 | channels β = 3.5563; barriers β = 4.2569 |
| PCG-V (curves only) | Gershgorin | Guaranteed |

### 1.2 Fitted coefficients (POLY-S, POLY-L)

- **Coefficients:** `ext/dev/stage1_fitted_methods.json`.
- **SHA-256** of the canonical JSON: `9dffe18716911af29afd32103c772ae30978b4e6adbc2bd3cc511a7c446fa2fd`. Every runner checks this hash as a gate.
- **Fit settings:** separate fit per family on set T (300 operators). L-BFGS with strong Wolfe line search, float64 autograd, rounds of 20 iterations, at most 25 rounds (500 iterations). It stops when the gradient ∞-norm < 1e-6 or the objective changes by < 1e-12 between rounds.

**Fit diagnostics** (objective = mean log error on T; ✔ = start kept):

| Family | Method | Start | Objective | ‖grad‖∞ | Evaluations | Outcome |
|---|---|---|---|---|---|---|
| channels | POLY-S | CHEB-equivalent ✔ | −4.348700 | 6.2e-8 | 76 | converged |
| channels | POLY-S | zero | −4.348700 | 8.2e-8 | 99 | converged |
| channels | POLY-L | CHEB-equivalent ✔ | −4.356112 | 9.6e-7 | 528 | converged |
| channels | POLY-L | zero | −1.924483 | 13.3 | 386 | not converged |
| barriers | POLY-S | CHEB-equivalent ✔ | −3.459650 | 3.9e-7 | 66 | converged |
| barriers | POLY-S | zero | −3.459650 | 8.9e-7 | 84 | converged |
| barriers | POLY-L | CHEB-equivalent ✔ | −3.569163 | 9.9e-7 | 374 | converged |
| barriers | POLY-L | zero | −2.559094 | 7.9e-5 | 601 | not converged |

**Notes on the fits:**
- Both POLY-S starts reach the same optimum, to within 5e-12.
- The POLY-L zero starts stall at poor points. The kept POLY-L optima are from the CHEB-equivalent start. No global optimum is claimed.
- **Check:** with the CHEB-equivalent coefficients, the batched torch objective (no backstop) reproduces CHEB-T(Lanczos, q = 10) through the numpy path: difference 0 (channels) and 1.7e-14 (barriers) in mean log error.

### 1.3 Secondary classical champion (A5.1)

Rule: smallest mean log relative energy error on set D among {CHEB-T\*, POLY-S, POLY-L, SD, MR, PCG}. Ties within 1% go to the earlier candidate.

| Candidate | channels | barriers |
|---|---|---|
| CHEB-T\* | −3.960 | −3.689 |
| POLY-S | −4.295 | −3.687 |
| POLY-L | **−4.328** | −3.831 |
| SD | −3.489 | −3.378 |
| MR | −3.506 | −3.341 |
| PCG | −4.174 | **−3.881** |

**Champions: channels POLY-L; barriers PCG.** Neither choice is within 1% of the runner-up: the margins are 0.034 and 0.050 in mean log error, against the threshold log 1.01 = 0.00995.

### 1.4 Spatial-control trigger (A5.2)

The trigger requires lower 95% limits ≥ 1.25 for **both** PCG → OA-VCB and K2 → OA-VCB, for **all three** OA-VCB initialisations. Results on set D:

| Family | init | PCG → OA-VCB lower limit | K2 → OA-VCB lower limit |
|---|---|---|---|
| channels | 0 / 1 / 2 | 6.02 / 6.14 / 6.33 | 10.06 / 10.21 / 10.58 |
| barriers | 0 / 1 / 2 | 1.409 / 1.379 / 1.330 | 1.319 / 1.284 / 1.253 |

**Fired for both families.** SGS (2 symmetric sweeps) and SGS-PCG (4 PCG iterations preconditioned by one symmetric Gauss–Seidel sweep) are included in the confirmation runs as **secondary** comparisons. They are not P1–P3.

The barrier trigger passed narrowly: init 2's K2 lower limit is 1.253.

### 1.5 Input scales (A7.2)

Computed once from each model's own training data and stored in `ext/frozen_scales.json`. They are never recomputed on C/S/LC.

| Model | Scale |
|---|---|
| channels fixed-load (set T) | 4.684449e-4 |
| barriers fixed-load (set T) | 1.181380 |
| channels random-load (training loads 777 000 + 100 000·1 + i, i < 300) | 4.678465e-4 |

- The set-D dry run checks that the frozen fixed-load scale equals the scale used in development, to 1e-12.
- Set R uses each seed's own training scale, which is that model's training data. For the random-load seed-2/3 models, R computes the scale from their training loads.

## 2 What is confirmatory (A5.4)

- **Confirmatory, set C only:** P1–P3 per family and per OA-VCB initialisation, each with its pointwise 95% bootstrap interval; the P1 bands; and the family label.
  - **P1** = G(PCG → OA-VCB);
  - **P2** = G(POLY-S → OA-VCB);
  - **P3** = G(CHEB-legacy → PCG). The legacy anchor is the reference (A1.1).
  - Each is reported **raw** (complete cases) and **guarded** (a failure takes u_c's error).
- **Secondary or diagnostic:** everything else, including:
  - the champion, K2, SGS, SGS-PCG, CHEB-T\*, POLY-L, SD and MR against OA-VCB;
  - φ (recovered log-gain, reference CHEB-legacy), with the unstable flag;
  - OA-VCBL and OA-VCL;
  - label subgroups;
  - sets S, LC and R;
  - D1, D2, timing, and the whole-solve curves.

## 3 Run plan for the confirmation computations

Each computation runs **once**. A rerun is allowed only after a crash or non-finite output; it is logged, and the failed output is kept (v1 §7). Every runner exits non-zero on a failed gate and writes its manifest.

| # | Command (`OMP_NUM_THREADS=2 python -I …`) | Output (`ext/runs/`) | Status |
|---|---|---|---|
| 1 | `ext/stage1_confirm.py C channels` | `confirm_C_channels.json` | confirmatory |
| 2 | `ext/stage1_confirm.py C barriers` | `confirm_C_barriers.json` | confirmatory |
| 3 | `ext/stage1_timing.py C {family} 50 … ext/dev/stage1_fitted_methods.json` | `timing_C_{family}.json` | diagnostic |
| 4 | `ext/stage1_d1.py C {family} <OA-VCB init-0 checkpoint> 100` | `d1_C_{family}.json` | diagnostic |
| 5 | `ext/stage1_d2.py C {family} 100` | `d2_C_{family}.json` | diagnostic |
| 6 | `ext/stage1_confirm.py S barriers` | `confirm_S_barriers.json` | secondary |
| 7 | `ext/stage1_confirm.py LC channels` | `confirm_LC_channels.json` | secondary |
| 8 | `ext/stage1_confirm.py R2 / R3 {family}` | `confirm_R{2,3}_{family}.json` | secondary, retrospective |

**Timing (row 3):**
- Runs **alone on the machine**. A development test showed that a concurrent 2-thread job slows the OA-VCB batch-1 path roughly 13×, through torch thread contention, while the classical timings barely move.
- Uses the first 50 test operators (ids 300–349) of set C per family and the init-0 OA-VCB checkpoint.
- Conventions as in A4, unchanged from the development harness.

## 4 Implementation decisions settled in development

These are implementation or bookkeeping decisions; none is a choice informed by set C, S or LC outcomes.

1. **Set S categories.** Set S uses the four A7.1 categories: island, discordant, EDP and strict non-island.
   - The first 20 draws of each category are kept from rng 2026101611. Fields come from `make_field(H, 'channels', 1e-4, rng)` with the no-low load. The cap is 3000 draws; counts are reported if it is reached.
   - v1 named three categories; the discordant stratum follows the A7.1 labeller. Subgroup descriptives only.
2. **Set LC design.**
   - **Operators:** ids 300–349 of set C (channels). Loads from `stage1_loads.lc_loads`, a verbatim copy of the `rhs_random_model.py` arithmetic with the fresh seeds of v1 §4.
   - **Models:** fixed-load OA-VCB init 0/1/2 and the random-load OA-VCB (data seed 1, init 0, scale 4.678465e-4).
   - **Tables:** one per load (50 operators), plus a pooled table with an operator-cluster bootstrap (`stage1_stats.paired_clustered`).
   - **Family label:** formed from the fixed-load initialisations only.
3. **Set R models.** For channels data seeds 2 and 3, the init-0 checkpoints are **random-load** models (`checkpoint_map.json`; this is the mapping v1 §4 refers to). Set R therefore evaluates, per seed:
   - channels: fixed-load init 1 and init 2, plus the random-load init 0, labelled as such;
   - barriers: init 0, 1 and 2.

   This is a superset of v1's "init 0". The fitted classical coefficients from seed-1 set T are applied unchanged.
4. **D2, oracle polynomial candidate.**
   - **Fit:** per operator, 24 coefficients at the CHEB-T\* per-level scales, fitted by L-BFGS against the reference solution with the same stopping rule as §1.2.
   - **Starts:** CHEB-T\*-equivalent; POLY-S re-expressed exactly in the per-level basis; POLY-L; PCG's first-call coefficients (A-norm least squares at each level and position, captured in the PCG cycle; in a V-cycle every call is a first call, so this start reproduces the PCG cycle to 1e-12); and zero. The best of the five is kept per operator.
   - **Reported error:** from the numpy path (common Galerkin with backstop). Gate: equal to the batched optimum within 1e-6 in log error.
   - **Joint fit rejected.** A joint L-BFGS over all operators was tried first. It converged poorly (10 barrier operators: −4.87 against −5.15 per operator), so each operator is now fitted separately.
   - Most per-operator runs stop at the 500-iteration cap. D2 is a **candidate** and no optimum is certified.
   - Development on set D: this is a diagnostic and **not confirmatory**. Outputs: `ext/dev/stage1_d2_D_{channels,barriers}.json`, 100 operators each, gates passed.

     | Family | D2 geometric-mean error | D2 → OA-VCB, inits 0 / 1 / 2 | PCG → D2 | Kept start (CHEB-T\* / POLY-S / POLY-L / PCG / zero) |
     |---|---|---|---|---|
     | channels | 1.111e-2 | 4.56 / 4.62 / 4.78 | 1.39 | 29 / 23 / 21 / 27 / 0 |
     | barriers | 9.776e-3 | 0.74 / 0.73 / 0.71 | 2.11 | 14 / 24 / 28 / 34 / 0 |

     - On channels, OA-VCB beats the per-operator oracle polynomial candidate on every operator (W = 1.00).
     - On barriers the candidate beats OA-VCB. By label (init 0): island 0.55 (n = 39); strict non-island 0.98 [0.87, 1.08] (n = 54).
     - Run time about 35–50 min per family at 2 threads.
5. **Work counts.** The SGS-PCG counter now includes the initial symmetric sweep: 14 matvec-equivalents per call and 90 per pass. The development JSON (`stage1_dev_D.json`) still shows the earlier 78 per pass; its errors are unaffected. SGS-PCG also records breakdown status now, which does not change its arithmetic.
6. **Gates** (non-zero exit on failure):
   - fitted-coefficient hash;
   - CHEB common path equals the legacy anchor (relative difference ≤ 1e-8 on every sample);
   - checkpoint SHA-256 values equal `checkpoint_map.json`;
   - three OA-VCB initialisations present;
   - on set D: frozen scale, frozen label counts, and exact reproduction of `stage1_dev_D.json`.

## 5 Development dry run of the confirmation runner on set D

The dry run reproduces `ext/dev/stage1_dev_D.json` exactly, with a maximum relative difference of 0.0 across all methods. These are **development numbers, not confirmatory**:

| Family | P1: R = G(PCG → OA-VCB), inits 0 / 1 / 2 | Band | P2: POLY-S → OA-VCB, init 0 | P3: CHEB-legacy → PCG |
|---|---|---|---|---|
| channels | 6.31 / 6.40 / 6.62 | not accounted for by Krylov smoothing | 5.59 | 1.62 |
| barriers | 1.57 / 1.54 / 1.50 | partly accounted for / inconclusive | 1.91 | 1.36 |

**Secondary results on set D, init 0:**
- **Channels:** SGS-PCG → OA-VCB = 1.25 [1.19, 1.30]. φ(SGS-PCG) = 0.91 and φ(PCG) = 0.21.
- **Barriers:** SGS-PCG → OA-VCB = 0.64 [0.55, 0.75], so SGS-PCG is more accurate than OA-VCB on set D.
- **D1, OA-VCB init 0:**
  - channels: pre-smoother ρ median 0.62; local separation described on 100 of 100 operators;
  - barriers: pre-smoother ρ median 1.005, so mostly not distinguishable; post-smoother ρ median 0.85.

## 6 Executed-source manifest (A7.3)

`ext/stage1_preconf_manifest.json` holds:
- the SHA-256 of every `ext/*.py`;
- every imported top-level or `diagnostics_cost` module;
- the frozen inputs (protocol, amendment, preconfirmation JSON, scales, coefficients, checkpoint map);
- every checkpoint the runners load, each matching `checkpoint_map.json`;
- package versions and thread settings.

| File | SHA-256 (first 16 hex) |
|---|---|
| `ext/stage0_repro.py` | `4a5dafca1b37454e` |
| `ext/stage1_confirm.py` | `1bc252e3db9cc29d` |
| `ext/stage1_core.py` | `fd6cdfcd3e83bed4` |
| `ext/stage1_d1.py` | `3d9d6e1b08269b6b` |
| `ext/stage1_d2.py` | `3772e71cfe39c28c` |
| `ext/stage1_dev_run.py` | `d546d9f2040a96d0` |
| `ext/stage1_fit.py` | `a98055cbbfe55a4b` |
| `ext/stage1_freeze.py` | `02f31499ea548171` |
| `ext/stage1_labels.py` | `5450b4280e07416d` |
| `ext/stage1_loads.py` | `ac154828b7bf417b` |
| `ext/stage1_manifest.py` | `1f61668ae21fd084` |
| `ext/stage1_models.py` | `f21ea14fc6b8ed50` |
| `ext/stage1_preconf.py` | `0b3c89925c1ef782` |
| `ext/stage1_stats.py` | `827345d0d1291ec5` |
| `ext/stage1_timing.py` | `58edf6331e234b2c` |
| `ext/stage1_timing_report.py` | `289c1c5a3c45ecc2` |
| `coarse_meshes.py` | `0d76304d2b86eb69` |
| `diagnostics_cost/s1_common.py` | `0a93d102e8934f24` |
| `frozen_basis.py` | `3c13f7433e71f028` |
| `grf_field.py` | `595897985cc4d407` |
| `oa_lifting.py` | `58d0af2b9d01b1bf` |
| `oa_vcycle.py` | `f5407fc5c60c14aa` |
| `ofwno_core.py` | `87f06ac5d734925d` |
| `run_identity.py` | `bf581bcc0759873a` |
| `ext/stage1_preconfirmation.json` (input) | `9504aa86b5d568ef` |
| `ext/frozen_scales.json` (input) | `8d0cc6f1d29553c9` |
| `ext/checkpoint_map.json` (input) | `da6caaa472230e3d` |
| `ext/dev/stage1_fitted_methods.json` (input) | `9bb4ed8f19c464e7` |
| `ext/dev/stage1_fit.json` (input) | `6f7294895ee37131` |
| `ext/STAGE1_PROTOCOL_2026-10-16.md` (input) | `324897bd94018407` |
| `ext/STAGE1_PROTOCOL_AMENDMENT_1_2026-10-16.md` (input) | `406c5bd0890eff89` |

25 checkpoints (seed-1 OA-VCB/OA-VCBL/OA-VCL including the random-load model; seeds 2–3 OA-VCB), all equal to `checkpoint_map.json`. The four seed-2/3 init-0 checkpoints (Sessions 7 and 8) were copied from the user's folder for this purpose and match the map. Versions: Python 3.13.16, NumPy 2.5.3, SciPy 1.18.1, PyTorch 2.14.1+cpu, PyAMG 5.3.0, Matplotlib 3.11.2; CPU Intel(R) Xeon(R) Processor @ 2.10GHz; runs use 2 threads (`OMP/MKL/OPENBLAS_NUM_THREADS=2`, `torch.set_num_threads(2)`).

**Top-level code hash:** f8653612eb84, unchanged.

Hash of this note: `ext/STAGE1_PRECONFIRMATION_SHA256.txt`.
