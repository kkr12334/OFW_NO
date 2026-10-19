# Stage 1 protocol: what does learned smoothing add beyond classical residual-adaptive smoothing?

**Version 1, frozen 16 October 2026.** Author of the protocol: Claude, for the author and Astra.

**Basis:**
- `CLAUDE_ROADMAP_REVIEW_RESPONSE_2026-10-14.md` (plan v2);
- `ASTRA_REVIEW_OF_CLAUDE_ROADMAP_2026-10-14.md`;
- the Stage 0 gate (`ext/STAGE0_GATE_2026-10-16.md`), which passed;
- the audit rules (`ext/REVIEW_AUDIT_2026-10-16.md`).

**Freeze rules:**
- Changes after this point are recorded as dated amendments **before** any confirmation number they affect is computed.
- Development work on the data marked *development* below may continue, but only for the items listed in §7.
- No numerical threshold in this document selects the next paper (Astra §4).

## 1 Questions

**Q1, matched update count.** In the same operator-adapted V-cycle, with the same transfers, Galerkin levels, exact coarsest solve, near-kernel deflation and Galerkin output, with four update stages per pre- and post-smoother on every level, how much of OA-VCB's paired energy-error advantage over the degree-four Chebyshev reference is obtained by classical smoothers that:
- (a) use a fixed polynomial fitted to the training distribution;
- (b) adapt their coefficients to each residual (steepest descent, minimal residual, diagonal-PCG)?

**Q2, stronger classical cycles and equal cost.** Does any remaining gap survive:
- a K-cycle (recursive inner flexible CG);
- a tuned smoothing interval;
- comparison at equal wall-clock time and at matched accuracy?

**Q3, mechanism diagnostic.** Is the learned smoother's output, for a given input residual, representable in the same four-dimensional Krylov space K₄ that every classical four-stage smoother here uses? Does it beat the best element of that space locally?

## 2 Facts the design relies on (checked numerically, `ext/dev/check_protocol_claims.py`, 16 Oct)

On the level matrices of development operators (barrier id 300, channel id 300, seed 1; levels k = 3, 2, 1), with D = diag(A_k), B = D⁻¹A_k and K₄(b) = span{D⁻¹b, B D⁻¹b, B² D⁻¹b, B³ D⁻¹b}. The right-hand side b was the actual fine-level residual r = F − Au_c on k = 3, and a random Gaussian vector on k = 2 and 1.

1. **PCG(4) is the K₄ energy minimiser.** Four Jacobi-preconditioned CG iterations from zero return the minimiser of ‖A_k⁻¹b − x‖_{A_k} over K₄(b); relative difference ≤ 6e-14. This is the standard Krylov optimality property; the check confirms the implementation.
2. **The other four-stage smoothers stay in K₄.** The outputs of the reference Chebyshev(4) smoother, of 4-step steepest descent and of 4-step minimal residual (both Jacobi-preconditioned) lie in K₄(b); relative A-distance ≤ 9e-13. Hence, **for the same input b**, the level-energy error of PCG(4) is no larger than theirs; this held at every checked level.
3. **The adaptive smoothers are odd but nonlinear.** PCG, steepest descent and minimal-residual smoothers satisfy S(λb) = λS(b) for every real λ, including λ < 0 (defect ≤ 1.5e-15). They are **not** additive (relative additivity defect 0.9–5.2% on the checked fine levels). Chebyshev(4) is linear (defect ~1e-16). A cycle using the adaptive smoothers is therefore odd in r but nonlinear. It is not a fixed SPD preconditioner, so ordinary PCG theory does not apply to it.
4. **The output projection never increases energy, whatever the directions.** The exact Galerkin output never increased the energy error relative to the deflated start (200 random direction sets per family). The guarantee does not depend on how the cycle directions are produced.

**Consequences:**
- All classical four-stage smoothers in Q1 choose coefficients within the **same** space K₄. They differ only in whether the coefficients are fixed or chosen per residual, and PCG(4) chooses the locally optimal ones.
- The learned smoother (OpSmoother, width C = 15, four message-passing steps, log-diagonal coefficient features, MLP edge messages) is not restricted to K₄.
- Local K₄-optimality does **not** imply that the PCG-smoothed cycle has the smallest final error. Pre-smoothing that is energy-optimal locally need not leave the error best suited to the coarse correction. Q1 is therefore an empirical question.

## 3 Methods

### 3.1 Common cycle (all of Stage 1a)

- **Configuration:** L3; supplied coarse solution u_c from the level-k0 = 2 Galerkin solve. Cycle levels k = 3, 2, 1 have smoothers, with an exact dense solve on level 0 (28 nodes). Hierarchy: `oa_hierarchy` with m = 4 Jacobi transfer steps, ω = 2/3.
- **Deflation:** x₀ = Z(ZᵀAZ)⁻¹Zᵀr, with Z from `near_kernel_basis` (θ = 0.1, τ = 0.05). The cycle acts on b = r − Ax₀.
- **Smoothing calls:** every smoothing call starts from zero on its own right-hand side. Pre-smoothing acts on b; post-smoothing acts on b − A(x + x_c).
- **Output:** u_c + x₀ + Wy, with W = [x, x_c, x_p, Z] at the fine level. The Gram system is solved with the Jacobi-scaled 1e-10 ridge of `classical_vcycle_proj`, in float64.
- **Variation:** only the smoother changes between methods.

### 3.2 Smoothers substituted on every level (pre and post), four update stages each

| Id | Smoother | Definition | Class |
|---|---|---|---|
| CHEB | Reference (anchor) | The `classical_vcycle_proj` Chebyshev recurrence, residual degree 4, interval [λ/10, λ]. λ = 3 on the finest level (a heuristic fixed value, not a certified bound); on coarse levels, 1.05 × a 30-step power-iteration estimate of λ_max(D⁻¹A_k). | fixed polynomial, linear |
| CHEB-T | Interval-tuned Chebyshev | As CHEB, but λ_max(D⁻¹A_k) is estimated on **every** level by 10 Lanczos steps × 1.1 (Adams et al. 2003, §5.1). Ratio q ∈ {4, 10, 30} is selected on training operators (§7). | fixed polynomial, linear |
| POLY-S | Fitted shared polynomial | x = p(B)D⁻¹b with p of degree ≤ 3. There is one coefficient set for pre-smoothing and one for post-smoothing, each shared by all levels, as the learned pre and post networks are (8 numbers). Written in a scaled Chebyshev basis on [0, β] with one fixed β (the maximum, over all levels and all training operators, of the CHEB-T Lanczos λ estimates). It is therefore a fixed polynomial in B, inside the class the four-stage architecture is proved to contain (paper Prop. 2). Fitted on training operators (§7). | fixed polynomial, linear |
| POLY-L | Fitted per-level polynomial | As POLY-S, but separate pre and post coefficient sets per level, each in a Chebyshev basis scaled by that level's CHEB-T λ estimate. Not in the contained class; a stronger linear classical control. | fixed polynomial per level, linear |
| SD | Steepest descent | x ← x + αp, p = D⁻¹r, α = pᵀr / pᵀA_k p (minimises level energy along p). 4 steps. | residual-adaptive, odd, nonlinear |
| MR | Minimal residual | p = D⁻¹r, α = (A_k p)ᵀD⁻¹r / (A_k p)ᵀD⁻¹(A_k p) (minimises ‖r‖ in the D⁻¹ norm along p). 4 steps. | residual-adaptive, odd, nonlinear |
| PCG | Diagonal-PCG | 4 Jacobi-preconditioned CG iterations from zero, restarted at every smoothing call (§2.1: the K₄ energy minimiser). If rᵀD⁻¹r = 0, return the current iterate. | residual-adaptive, odd, nonlinear |
| OA-VCL, OA-VCBL | Learned residual-linear controls | Existing checkpoints. OA-VCBL's Galerkin output is nonlinear. | learned |
| OA-VCB | Learned | Existing checkpoints, inference path of `s1_common.apply` (float32 network, float64 residual, deflation and output). | learned, positively homogeneous |

### 3.3 Stage 1b: stronger classical cycles and the cost frontier

**K2-cycle** (Notay & Vassilevski 2008):
- **Structure:**
  - On the level-k = 2 and k = 1 coarse systems, γ = 2 inner flexible-CG iterations (their Algorithm 3.1 with m_i = 1), preconditioned recursively.
  - Two-grid with the exact level-0 solve at the bottom (k = 1).
  - k₀ = 1.
  - Smoother: CHEB, the reference, pre with M and post with the same polynomial.
- **Variants:** with the same deflation and Galerkin output (projected), and plain.
- **Assumptions to state:** their analysis assumes a convergent smoother, ‖I − M⁻¹A‖_A < 1. The CHEB interval rests on heuristic λ estimates, so this assumption is not certified here.

**Whole-solve curves** on one platform, from u_c, set-up included, fine-matrix assembly excluded (common):
- 1–4 passes of each Stage 1a classical cycle (x ← x + cycle(F − Ax));
- PCG with the plain linear CHEB V(4,4) cycle, which is symmetric and fixed;
- flexible CG (Notay 2000, m_i = 1) with the K2-cycle and with the PCG-smoothed V-cycle, both nonlinear;
- two-grid PCG and SA-AMG PCG, as in `accuracy_vs_time.py`;
- a sparse direct solve.

**Flexible-CG statements** will cite Notay's sufficient condition only: ‖B(r) − B⁻¹r‖_B ≤ ε < 1 for one fixed SPD B (Notay 2000, Thm 3.1). It is not treated as necessary, and it is not claimed to hold unless checked.

**Conditional spatial control.** Run this only if, after 1a and the K-cycle, a family's direct ratio (§5) has a lower 95% limit ≥ 1.25:
- symmetric Gauss–Seidel (2 symmetric sweeps = 4 stages) in the same cycle;
- 4-step PCG preconditioned by symmetric Gauss–Seidel.

### 3.4 Diagnostics (labelled as such; never an online method)

- **D1, local K₄ test.** For OA-VCB's fine-level pre-smoother on its actual input b (identical for every method, since it precedes any smoothing), and for its fine-level post-smoother on OA-VCB's own post-smoothing input, report:
  - δ = ‖x_NN − Π_K₄ x_NN‖_A / ‖x_NN‖_A, where Π_K₄ is the A_k-orthogonal projection, computed with an orthonormal basis and a least-squares Gram solve;
  - ρ = E(x_NN) / E(x_PCG), with E(x) = ‖A_k⁻¹b − x‖_{A_k}.

  ρ < 1 means the network's output for **that** input beats every K₄ correction locally, a local separation for that residual and degree only. No full-cycle or universal claim follows.
- **D2, oracle-fitted polynomial candidate.** Per operator, the CHEB-T-scaled per-level pre/post coefficients (≤ 24 numbers) are optimised by L-BFGS against the reference solution, minimising that operator's final relative energy error. Five starts: CHEB, POLY-S, POLY-L, PCG's first-call coefficients, zero. This uses the exact solution, does not certify a global optimum, and is called a *candidate*, not a ceiling.

## 4 Data

| Set | Label | Content | Use |
|---|---|---|---|
| T | development (training) | Seed-1 training operators, ids 0–299 (rng 1001), f = 1; barriers with contrast 1e-4 and load no-low | Fitting POLY-S/POLY-L and selecting CHEB-T's q. The learned checkpoints were trained on these operators too. |
| D | development | Seed-1 test operators, ids 300–399, both families (inspected in the paper and in Stage 0) | Implementation checks, debugging, development numbers. **Not** used for confirmation claims. |
| R | retrospective | Seeds 2–3 test sets with their own OA-VCB checkpoints, both families, init 0. Channel seed-2/3 init-0 checkpoints are mapped from their stored config before use, as in `ext/checkpoint_map.json`. | Secondary robustness across data seeds (previously inspected, so not fresh). |
| C | **fresh confirmation** | Ids 300–399 of `setup(3, 2, 0, rng_seed=2026101601)` (channels) and `setup(3, 2, 0, contrast=1e-4, load='no-low', rng_seed=2026101602)` (barriers). The positional data seed is ignored when `rng_seed` is given. | **Primary results.** Never generated before this protocol. |
| S | fresh stratified diagnostic | Barrier fields drawn sequentially from rng 2026101611, labelled by structure only (§4.1). The first 20 of each label (island, near-island, non-island) are kept; cap 3000 draws, with counts reported if the cap is reached. | Subgroup descriptives only. Never pooled into population averages. |
| LC | fresh load challenge | The first 50 channel operators of C. Loads: f = −1, smooth ±, pure localised Gaussian (width 0.03), and a draw from the random-load training distribution, generated exactly as in `rhs_random_model.py` with fresh seeds: smooth/point rng 2026101621·10³ + i, random-load rng 2026101622·10³ + i | Load robustness of the classical adaptive smoothers against the fixed-load and random-load OA-VCB (channels, data seed 1). There is no barrier random-load model, so no barrier load challenge. |

The seeds 1000–1003, 9002/9003 (harmonic-study confirmation) and the paper's load seeds (5 000/777 000 + 100 000·DS + i) are avoided.

### 4.1 Labels (structure only, fixed now; no solver error is used)

- **Island:** `near_kernel_basis(A)` (θ = 0.1, τ = 0.05) returns ≥ 1 vector. The geometric label `geometric_island` is reported alongside, with disagreements counted.
- **Near-island:** detector-negative **and** geometric-island-negative, **and** a one-layer erosion of the conductive set leaves a conductive component of ≥ 30 elements that touches no Dirichlet node. One-layer erosion removes every conductive element sharing a node with an insulating element. The rule therefore marks a conductive region attached to anchored material only through a bridge at most about two elements wide. On development set D it marks 5/100 barrier samples and 0/100 channel samples.
- **Non-island:** neither of the above.
- **Superseded rule:** the provisional Stage 0 rule (relaxed τ = 0.2 detector) is superseded. It also fired on 14/100 channel samples. The Stage 0 results stand as run.

## 5 Metrics and statistics

- **Per-sample metric:** relative energy error e = ‖u − û‖_A / ‖u‖_A of the final output, with u from a sparse direct solve.
- **Paired gain** of method Y over method X: G = exp(mean log(e_X / e_Y)). Win fraction: W = mean(e_Y < e_X). Also reported: arithmetic means, the ratio of means, medians, 95th percentile, maximum, and the full per-sample records.
- **Uncertainty:** paired percentile bootstrap over **operators**, 10 000 resamples, rng 20261016. When several loads share an operator, the operator is the resampling unit.
- **Training variability:** OA-VCB has three initialisations per family and OA-VCBL three on channels. They are reported per initialisation, with the range, and never treated as additional test samples.
- **Primary comparisons, fixed now** (set C, per family, per OA-VCB initialisation):
  - **P1:** OA-VCB against the PCG-smoothed cycle. The direct ratio is R = G(PCG → OA-VCB) = exp(mean log(e_PCG / e_OA-VCB)).
  - **P2:** OA-VCB against the POLY-S cycle.
  - **P3:** the PCG-smoothed cycle against CHEB, i.e. what classical residual adaptivity buys.
- **Everything else is secondary.** "Best classical" is chosen on set D and then evaluated on C, never chosen on C.
- **Descriptive bands for P1**, per family and per OA-VCB initialisation. These are labels for describing the result, not routing gates. They are applied **in this order**, and the first that holds is used, so they are disjoint and exhaustive:
  1. *"Classical adaptive smoothing more accurate":* upper 95% limit of R < 1.
  2. *"Largely accounted for at matched stages":* upper 95% limit of R ≤ 1.25.
  3. *"Not accounted for by Krylov smoothing":* lower 95% limit of R ≥ 2.
  4. *"Partly accounted for / inconclusive":* every remaining case.
  - **Family label:** the band shared by all three initialisations; otherwise "initialisation-dependent", with the three bands listed.
  - Families are labelled separately; a family-specific difference is reported, never pooled away.
- **Recovered log-gain:** φ_Y = log G(CHEB → Y) / log G(CHEB → OA-VCB), descriptive only, with a bootstrap interval. It is reported together with R. It is not interpreted as the share of a mechanism and is flagged as unstable if the interval of G(CHEB → OA-VCB) includes values below 1.25.
- **Matched cost**, reported side by side with matched stages:
  - errors at equal wall-clock time per sample: the best whole-solve curve point with time ≤ OA-VCB's time;
  - time to reach OA-VCB's per-sample error. This is an offline diagnostic using reference-solution iteration counts, not a stopping rule.

## 6 Work and timing records

- **Work counts per sample and method** (counters, verified against analytic counts):
  - per level: A_k products, D⁻¹ applications, dot products / reductions, restrictions / prolongations, coarsest solves;
  - set-up operations: λ estimates, deflation, Gram solves.
- **OA-VCB work:** four message-passing steps × width 15 per smoothing call, plus edge/node MLP evaluations, with an analytic multiply-add count.
- **Timing:**
  - one platform: this container, Intel Xeon @ 2.10 GHz, 2 threads, PyTorch 2.14.1+cpu;
  - every method in one run; per-sample minimum over 2 repetitions; no error evaluated inside a timer;
  - OA-VCB timed as operator data (A_list passed, no reassembly) plus batch-8 inference;
  - hierarchy set-up shared by all OA cycles and timed once.
- **Scope:** these timings are not merged with the paper's tables, which came from other machines. Python timings measure this implementation, not an optimised one.

## 7 What development may still change (before the pre-confirmation note)

1. **Implementation bugs**, at any time before confirmation.
2. **CHEB-T:** choose q ∈ {4, 10, 30} by the smallest mean log error on set T.
3. **POLY-S / POLY-L:**
   - Fit by minimising the mean over set T of log relative energy error of the final projected output. This is the evaluation metric, so the fit favours the classical control.
   - Use L-BFGS on float64 autograd; optimiser settings may be iterated until converged.
   - Freeze the coefficients and record their hash.
4. **Nothing else.** K2 uses γ = 2 and k₀ = 1 as fixed above, and the labels, sets, metrics, primary comparisons and bands are fixed.

Before set C is generated, a short **pre-confirmation note** records:
- q*;
- the fitted coefficients and their hash;
- the SHA-256 of every `ext/` script;
- the top-level code hash (f8653612eb84, unchanged);
- the checkpoint SHA-256s.

If Astra reviews this protocol, every point raised is answered by amendment before set C is generated. Each confirmation computation then runs once. A rerun is allowed only after a crash or non-finite output; it is logged, and the failure is kept in the records.

## 8 Failures and missing data

- Non-finite outputs, smoother breakdowns (pᵀAp ≤ 0), Gram-solve rank deficiencies, safeguard back-stops and crashes are counted per method and kept.
- Paired statistics are reported both with failures excluded and with each failure assigned the error of u_c. The projected cycles cannot be worse than u_c in exact arithmetic.
- No sample is dropped silently.

## 9 Reporting and what follows

**The Stage 1 report contains:**
- per-operator JSON records (ids, rng seeds, labels, every method's error, work counts, timings, hashes);
- P1–P3 tables per family and initialisation;
- secondary tables;
- subgroup (set S) and load-challenge (set LC) tables;
- the accuracy/time figure;
- D1/D2 diagnostics;
- a short comparison with the closest prior art (learned-smoother papers' baselines; K-cycle; polynomial smoothers).

**What follows (for orientation only, per Astra §3–4):**
- A largely-accounted-for outcome supports an evaluation-type finding, which still needs breadth beyond this architecture before it becomes a paper.
- A not-accounted-for outcome motivates the restricted-class separation question (Route B, Astra's trigger).
- Either way, Routes A and C and the D1/D2 learned-set-up directions stay open.
- The paper choice is made only after the report, a closest-prior-art comparison and Astra's review.

## References used in this protocol

- Y. Notay, P. S. Vassilevski, *Recursive Krylov-based multigrid cycles*, Numer. Linear Algebra Appl. 15 (2008) 473–487 (K-cycle definition, Alg. 2.1, Alg. 3.1, Thm 5.1).
- M. Adams, M. Brezina, J. Hu, R. Tuminaro, *Parallel multigrid smoothing: polynomial versus Gauss–Seidel*, J. Comput. Phys. 188 (2003) 593–610 (λ_max by 10 CG/Lanczos steps × 1.1; lower end λ_max/30; insensitivity to the lower end, §3.3 and §5.1).
- Y. Notay, *Flexible conjugate gradients*, SIAM J. Sci. Comput. 22 (2000) 1444–1460 (Thm 3.1, sufficient condition (3.4)).
