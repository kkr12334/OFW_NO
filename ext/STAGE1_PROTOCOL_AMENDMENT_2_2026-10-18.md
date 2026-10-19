# Stage 1 protocol, amendment 2

18 October 2026. Responds to `ASTRA_STAGE1_REPORT_REVIEW_2026-10-17.md`.

**Status when this amendment was written and hashed:**
- **No reserve data had been generated.** The reserve rng seeds 2026101603 and 2026101604 appear in no code, record or log; they appear only in protocol and report text. This was checked by searching the workspace on 18 October.
- The Stage 1 report v1 (SHA-256 `9ec033f3…`) and every confirmation record stay frozen as historical records.
- **Corrections** go into:
  - a dated report revision, `STAGE1_REPORT_REV1_2026-10-18.md`;
  - a timing and work correction, run under this amendment.
- P1–P3 are not rerun.

## B1 Errata

1. **Amendment 1, A3.2, PCG-V argument.**
   - "0 < q(λ) < 1 on (0, λ̄]" is wrong: a Chebyshev residual polynomial changes sign on [λ̄/q, λ̄].
   - The correct sufficient condition is |q(λ)| < 1 on the positive spectrum (0, λ̄].
   - With the same polynomial before and after smoothing (S A-self-adjoint, ‖S‖_A < 1) and an exact coarsest solve, the V-cycle error propagator is E = S(I − P(I − E_c)A_c⁻¹PᵀA)S. By induction, 0 ≤ E_c ≤ I in the A_c inner product. The middle factor then lies between I − Π and I, where Π is the A-orthogonal coarse projection, so 0 ≤ E ≤ ‖S‖²_A·I < I in the A inner product. Hence the spectrum of MA lies in (0, 1] and M is SPD.
   - The conclusion stands; the argument is corrected. The numerical SPD checks of A3.2 are unchanged.
2. **Timing harness (`stage1_timing.py`, frozen).** It breaks the A4 boundaries in four places, which ASTRA found and I confirmed in the code:
   - (a) `smoother_set(hr)` builds every classical smoother, including two power-iteration and six Lanczos bound computations, before one is selected. Every classical one-pass method, the repeated-pass curves and FCG-K2 are therefore charged set-up they do not use.
   - (b) `OAData(...)` calls `structural_patterns` on every construction, so OA-VCB's batch-1 latency includes mesh-pattern construction that A4 excludes.
   - (c) The batch-8 throughput timer includes model weight loading.
   - (d) One-pass methods receive their residual from outside the timer, while iterative solvers form it inside.
   
   The end-to-end boundary A4.4(ii), which includes computing u_c, was not delivered. The v1 timing numbers are therefore superseded for every quantitative use.
3. **Work accounting (`stage1_work.py`).** It counted only the learned smoother calls for OA-VCB, and only counter-recorded matrix, diagonal and dot products for the classical methods. It omitted:
   - transfers and the coarsest solve;
   - the deflation and Galerkin-output wrapper;
   - PCG's initial products and all vector updates.
   
   It also miscounted the minimal-residual diagonal products, and its fine-level learned count included 90 structural zeros. Its ratios are superseded by the ledger of B3.

## B2 Timing repair (v2): specification

**Workload.** Same as A4:
- the first 50 set-C operators per family, the same saved inputs, the init-0 checkpoint;
- 2 threads, 5 repeats, per-operator random order (rng 20261017), one warm-up run per task;
- run alone on the machine;
- outputs `runs/timing_v2_C_{family}.json`. The v1 files are kept.

**Fixes:**
1. **Method-specific set-up only.** Each classical method builds only its own smoother and the set-up it needs:
   - CHEB: the legacy bounds;
   - CHEB-T\* and POLY-L: Lanczos bounds;
   - POLY-S: none beyond β;
   - PCG, SD, MR, SGS, SGS-PCG: none;
   - K2 and the CHEB passes: the CHEB bounds.
2. **Mesh patterns** are built once per mesh outside every timer. A cache of `structural_patterns` is installed in the timing script; no release file changes. Their cost is reported separately.
3. **OA-VCB batch-1 latency** counts, inside the timer:
   - per-operator operator data: hierarchy, near-kernel basis, coarse inverse and level values;
   - the residual;
   - the forward pass.
   
   Weights are loaded once, outside the timer.
4. **Batch-8 throughput** counts operator data plus inference, with weights loaded outside the timer.
5. **The residual r = F − A u_c is formed inside every timer.**
6. **Decomposition.** Recorded by a separate untimed-order profile, 5 repeats, median:
   - the classical one-pass time: hierarchy, deflation basis, method-specific bounds, cycle application and Galerkin wrapper;
   - OA-VCB: operator data and forward pass.
7. **End-to-end boundary (A4.4(ii)).** The time to construct u_c is measured per operator: coarse Galerkin operator, factorisation, solve and prolongation. Times are reported both conditional on u_c and end-to-end (u_c plus method) for every method that starts from u_c. The direct solve is never charged u_c.
8. **Whole-solve curves.**
   - Unchanged: the solvers, the iteration limits and the offline time to reach OA-VCB's per-sample error, labelled as offline oracle stopping.
   - Added, for every curve: the time to reach a fixed relative residual ‖F − Ax‖/‖F‖ ≤ 10⁻⁶. The residual is evaluated offline at each recorded iterate, with censoring counts.
   - Outer PCG/FCG loops record their termination reason: completed, exact (rᵀz = 0) or curvature. The inner FCG of the K-cycle is unchanged and does not record one; this is stated.
9. **Equal-time factors:** ratios at or beyond the reference-solve floor are not quoted; the report states only that the curve reached its iteration limit before OA-VCB's time.

## B3 Operation ledger: specification

**Script:** `stage1_ledger.py`, structure only (level sizes and patterns), output `runs/stage1_ledger.json`.

**Conventions:**
- A MAC is one multiply and one add.
- A diagonal scaling, a division or a vector update counts n operations, listed as MACs.
- A dot product is n MACs.
- Global reductions (dots, max, rms, median) are also counted separately as reductions.
- Activations (GELU) are counted separately as evaluations, not MACs.
- Fine-level learned counts use the network's own pattern: 19 926 off-diagonal entries, against 20 016 stored in the assembled matrix, 90 of which are structural zeros.

**Terms per one pass,** each listed separately:
1. smoother applications;
2. cycle residuals;
3. transfers (restriction and prolongation);
4. coarsest solve (dense 28 × 28 for OA-VCB, triangular LU solves for classical);
5. wrapper: deflation x₀ with k near-kernel vectors, b = r − Ax₀, and the Galerkin output (AW products, Gram matrix, small solve, update);
6. method-specific set-up: spectral bounds, and OA-VCB's per-operator feature computation.

**Not counted, reported as measured time only:** common set-up (operator-adapted hierarchy, near-kernel basis, coarse factorisation).

**Units:** totals are "fine-SpMV-equivalent arithmetic" (MACs ÷ fine nnz), named explicitly as such. They are never presented as actual SpMV calls, time or communication.

## B4 Reserve experiment (frozen here; run once)

**Purpose.** Confirm the selected exploratory findings of report v1, §4.3 and §5.2, on unopened operators.

**Data:**
- **channels:** `setup(3, 2, 0, rng_seed=2026101603)`;
- **barriers:** `setup(3, 2, 0, contrast=1e-4, load='no-low', rng_seed=2026101604)`;
- test ids 300–399, i.e. 100 operators per family.

**Models:** fixed-load OA-VCB, data seed 1, inits 0, 1 and 2 (checkpoint SHA-256 values as in the pre-confirmation manifest); frozen input scales.

**Methods:**
- one pass of the common wrapper (`stage1_core.one_pass`) with Jacobi-PCG(k), k ∈ {4, 5, 8, 14}, and with SGS-PCG(4);
- CHEB-legacy, as the gate anchor only.

**Statistics:** paired geometric gain G(X → OA-VCB) = exp(mean log(e_X / e_NN)), W, pointwise 95% bootstrap over operators (10 000 resamples, rng 20261016), complete cases with exclusions listed.

**Endpoints:**
- **R1 (primary; channels).** For each init: bracketing is confirmed iff the lower 95% limit of G(PCG(8) → OA-VCB) > 1 **and** the upper 95% limit of G(PCG(14) → OA-VCB) < 1.
  - The family statement requires all three inits.
  - "Bracketing" concerns paired geometric errors, not individual operators.
- **R2 (secondary; barriers).** The same test with PCG(5) and PCG(8).
- **R3 (secondary; both families).** G(SGS-PCG(4) → OA-VCB) and G(PCG(14) → SGS-PCG(4)), descriptive.
- **R4 (secondary; channels; local).** For each init, at OA-VCB's fine-level pre- and post-smoother inputs, ρ_d = E_NN / E_{K_d} for d = 4, 5 and 8, with the rank-aware basis of `stage1_core.krylov_basis`.
  - Recorded per input: rank, A-orthogonality defect ‖QᵀAQ − I‖_max, and |E(PCG(d)) / E_{K_d} − 1|.
  - Local K₅ separation is **confirmed** for an init and position iff ρ_K₅ ≤ 0.99 on at least 90 of the operators with rank 5.
  - Full distributions are reported. K₈ is descriptive.

**Gates** (non-zero exit on failure):
- CHEB common path equals the legacy anchor to 1e-8;
- checkpoint hashes;
- frozen scales;
- the reserve seeds are absent from every record before generation;
- the PCG(4) row equals `stage1_core.pcg4` through the wrapper.

**Failures:** recorded, never imputed. Rank-deficient inputs are excluded from R4 counts and listed.

**Not run:** the longer K₅ oracle fit (ASTRA §6.4: optional; it cannot certify a class bound).

**Outputs:** `runs/reserve_{family}.json`, `runs/reserve_K_local_channels.json`. Script: `stage1_reserve.py`. Each computation runs once.

## B5 Wording corrections adopted for report revision 1

All of ASTRA §2–§5 and §8 are adopted:
- the full-cycle polynomial result is "the best candidate found by our per-instance polynomial optimisation", never a class bound;
- no "per neighbour-exchange round", "same number of neighbour exchanges" or "efficient per stage" (OA-VCB's σ normalisation is a global, nonlinear dependence);
- the SGS sentence uses ASTRA's replacement;
- "matches" and "ties" mean "the interval includes parity";
- the float32 sentence is deleted;
- the barrier K₅ local wording is position-dependent;
- K₈ does not dominate on every input;
- the 4 ms two-grid time is conditional on u_c;
- the literature entries are corrected (Li et al. compare against CG; McGreivy and Hakim 2024, and Zhang, Li and Saad 2026, are added);
- the numeric-token scan is described as a typo check only.
