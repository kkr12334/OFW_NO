# Stage 1 protocol, amendment 1

16 October 2026. Responds to `ASTRA_STAGE1_PROTOCOL_REVIEW_2026-10-16.md`.

**Status of this amendment:**
- The frozen v1 protocol (`STAGE1_PROTOCOL_2026-10-16.md`, SHA-256 `324897bd…99fc`) is preserved unchanged. This amendment supersedes it where they differ.
- **No set C, S or LC data had been generated or inspected when this amendment was written.**
- New evidence, all on development data and small matrices:
  - `ext/dev/checks_v2.py` / `.json`: exit 0, all checks pass;
  - `ext/dev/check_loaders.py` / `.json`: exit 0;
  - `ext/stage1_core.py`, `ext/stage1_models.py`.

**All of Astra's findings are accepted.** I re-ran its small-case script and reproduced every point:
- a rank-1 Krylov space with four spurious QR columns, giving distance 0 against a true distance of 1;
- a symmetric Chebyshev two-level cycle with eigenvalue −3678.27 when λ_max is underestimated;
- a one-sided deflation symmetry defect of 0.707.

I also confirmed in my copy of Notay (2000) that Theorem 3.1, condition (3.4), is the **relative** form. My v1 text dropped the denominator.

## A1 Implemented comparison and loaders

1. **Legacy anchor.**
   - CHEB evaluated by `oa_vcycle.classical_vcycle_proj`, unchanged; it reproduces the stored records to ≤ 5.3e-12 (Stage 0).
   - It stays as the reproduction anchor and the reported reference in P3.
2. **Common classical output.**
   - Every classical substitution (CHEB through the new path, CHEB-T, POLY-S, POLY-L, SD, MR, PCG) uses `stage1_core.one_pass`. Its deflation and final output both call `stage1_core.galerkin`, which mirrors `OAVCycle._galerkin` in float64: symmetrised Gram matrix, Jacobi scaling with dead-column masking, 1e-10 ridge, and the energy-acceptance backstop 2dᵀb − dᵀAd ≥ 0, falling back to the exact line-search step otherwise.
   - Backstop activations and rejections are counted per method.
   - **Agreement with the anchor on all of set D:** CHEB through the common path against `classical_vcycle_proj` differs by at most 8.4e-13 (barriers) and 7.8e-16 (channels) in relative energy error, with zero backstop activations. Declared tolerance: 1e-8.
3. **Numerical differences remaining between the learned and classical paths.** These are stated, not removed:
   - the learned cycle runs in float32 inside a float64 deflation and output, while classical cycles are float64 throughout;
   - the learned output uses the torch implementation of the same Gram procedure.
   - So "only the smoother changes" holds **within the classical family**. Learned against classical additionally differs in cycle arithmetic precision.
4. **Learned models.** `stage1_models.load` builds the constructor flags of the training driver (`train_oa_layer.py` l.915–919) from the checkpoint type and cross-checks them against the state dict (residual-linear ⇔ no `pre.0.enc.*` MLP; width from `enc_lin`):
   - OA-VCB: deflate, projected output, nonlinear;
   - OA-VCBL: deflate, projected output, linear smoothers;
   - **OA-VCL: deflate, plain output, linear smoothers.** This is the historical full model.
   - **Loader reproduction on set D:** all six (three types × two families) match the stored errors. Mean relative difference ≤ 4.3e-6; non-island per-sample difference ≤ 2.3e-5; island samples ≤ 1.4e-3, as documented in Stage 0.
   - OA-VCL is reported as the historical plain-output model. If OA-VCL's smoothers are evaluated with the projected output, that is a **new variant**, named "OA-VCL + Galerkin output (new evaluation)".

## A2 Rank-aware Krylov diagnostic; corrected checks

1. **Basis construction.** `stage1_core.krylov_basis`:
   - Arnoldi in the A_k inner product, with two reorthogonalisation passes;
   - a vector is dropped, and the iteration stops (happy breakdown), when its A-norm after orthogonalisation is ≤ 1e-10 × its A-norm before;
   - **K₄ means dimension at most 4**, and the retained rank is reported.
2. **Edge cases, all passing:**
   - Astra's case A = I, b = e₁: rank 1, true distance of e₂ = 1;
   - b = 0: rank 0;
   - diagonal A, where B = I: rank 1;
   - an eigenvector input: rank 1;
   - three eigencomponents: rank 3;
   - repeated eigenvalues: rank 2;
   - two clusters split by 1e-13: rank 2;
   - six distinct eigenvalues: capped at 4;
   - A-orthonormality to 1e-12.
3. **Zero cases** are flagged ("zero output", "exact convergence in K") and never divided.
4. **D1 decision rule.** D1 reports the reliable retained-space minimum E_K = E(QQᵀb) and cross-checks PCG against it, requiring |E_PCG/E_K − 1| < 1e-8 (observed ≤ 5.5e-14). For a given input, a local separation is **described** only if all of the following hold:
   - the retained rank is 4;
   - E_K > 0;
   - ρ = E_NN/E_K ≤ 0.99.

   A ratio within 1% of one is reported as "not distinguishable". No full-cycle or universal claim follows (v1 §3.4 is unchanged in that respect).
5. **Corrected descriptions of the v1 §2 checks.**
   - v1 §2.2 used random right-hand sides and a hard-coded λ = 2.0 on coarse levels. `checks_v2` repeats it on the **actual level inputs captured from the reference cycle**, with the legacy production estimators. Results: rank 4 at every level; PCG vs K-min ≤ 5.5e-14; A-distance of CHEB/SD/MR/PCG outputs from K ≤ 5.4e-15.
   - v1 §2.4 compared against the undeflated solution. `checks_v2` measures energy relative to the **deflated start x₀ through the actual wrapper** (ridge + backstop). Result: never increased in 200 random direction sets per family, zero backstops.
   - The homogeneity and non-additivity results are implementation checks on two operators, saved as records. The adaptive smoothers are "nonlinear in general": they can act linearly on special matrices or inputs.

## A3 Ordinary-PCG preconditioner, flexible CG, K-cycle

1. **Spectral bounds compared with exact λ_max(D⁻¹A_k).** Computed on 20 operators per family on set D (`eigsh`):

   | Estimator | min estimator / exact over operators and levels | below exact |
   |---|---|---|
   | Legacy CHEB (fine 3.0; coarse 1.05 × 30-step power iteration) | 0.930 (channels, level k = 1) | coarse levels on 0–10% of operators, by up to 7% |
   | Lanczos (10 steps) × 1.1 | 0.993 (channels, level k = 2) | on at least one operator, by < 1% |
   | Gershgorin row-sum of H = D^-1/2 A D^-1/2 | 1.009 | never; this bound is guaranteed |

   The paper's reference smoother therefore occasionally uses an interval slightly below λ_max on a coarse level. This does not change its definition as the anchor. It is a reason for CHEB-T (below).
2. **Ordinary-PCG cycle (PCG-V).** Plain symmetric V(4,4) cycle, **without deflation**:
   - Chebyshev smoothers with the **Gershgorin** upper bound on every level and q = 10;
   - the same polynomial pre and post (A_k-self-adjoint);
   - exact coarsest solve.

   **Sufficient condition (argument).**
   - With λ̄ ≥ λ_max, the Chebyshev residual polynomial normalised at 0 satisfies 0 < q(λ) < 1 on (0, λ̄], so each smoother is A_k-convergent.
   - The symmetric V-cycle with exact coarsest solve then has an A-self-adjoint, A-positive-semidefinite error propagator E with ‖E‖_A < 1, by induction from the coarsest level.
   - Hence MA has spectrum in (0, 1] and M is SPD.

   **Checks** on the barrier island operator and the channel operator (id 300, assembled M, n = 3486):
   - symmetry defect ≤ 7e-16;
   - spectrum of MA in [0.068, 1.000] (barriers) and [0.756, 1.000] (channels).

   The Gershgorin bounds cost one pass over each level matrix, which is counted as set-up.
   The legacy-interval cycle was also SPD on those two operators. That is descriptive only; it is not used for PCG.
3. **Secondary deflated variant** (balanced): 𝓜 = C + (I − CA)M(I − AC), C = Z(ZᵀAZ)⁻¹Zᵀ. Checked SPD on the island operator (minimum eigenvalue 0.083). Rank handling: Z columns pass the same scaled-ridge solve; cost counted. One-sided deflation followed by a cycle is not used as a PCG preconditioner.
4. **CHEB-T candidates.** Upper bound ∈ {Gershgorin, Lanczos × 1.1} × q ∈ {4, 10, 30}, selected on set T (§A5).
5. **Flexible-CG condition** (corrects v1 §3.3; Notay 2000, Thm 3.1, condition (3.4)): ‖𝓑(r) − B⁻¹r‖_B / ‖B⁻¹r‖_B ≤ ε < 1 for all r ≠ 0 encountered, for one fixed SPD B. This is a sufficient condition. It is not claimed to hold unless checked, and a few probes do not establish it.
6. **K2-cycle, frozen pseudocode.**
   - Naming: k0 = 2 remains the supplied-coarse-solution level. The K-cycle spacing parameter is renamed **s_K = 1** (inner iterations at every eligible level). γ = 2.

```
K(l, r):                          # l = 0 finest (k = 3), l = 1 (k = 2), l = 2 (k = 1); exact A_0 below l = 2
    x  = S(l, r)                  # CHEB smoother of the variant (legacy or CHEB-T* intervals), from zero
    rc = R_l (r - A_l x)
    if l == 2:  ec = A_0^{-1} rc  # LU factor computed once per operator, reused in every call
    else:       ec = FCG(A_{l+1}, rc, prec = K(l+1, .), iterations = 2)
    xc = P_l ec
    xp = S(l, r - A_l (x + xc))
    return x + xc + xp            # at l = 0 in projected mode: parts (x, xc, xp) for the common Galerkin output
FCG(A, b, prec, iterations):      # Notay & Vassilevski Alg. 3.1, x_0 = 0, m_0 = 0, m_1 = 1
    r = b; x = 0
    for i in 0, 1:
        w = prec(r); d = w - (if i == 1: (w^T A d_0 / d_0^T A d_0) d_0)
        x = x + (d^T r / d^T A d) d;  r = r - (d^T r / d^T A d) A d
    return x
```

## A4 Timing workloads and boundaries

1. **Primary workload: per-operator latency.**
   - Every method, OA-VCB included, is timed on the same operator, OA-VCB with **batch size 1**.
   - One warm-up run per method per operator.
   - Method order randomised per operator (rng 20261017).
   - **5 repeats; the median is primary**, the minimum is reported too, and all raw repeats are saved.
2. **Secondary workload: throughput.** OA-VCB batch-8 inference over the whole set, and every classical method over the whole set, each reported as total/N **labelled throughput**. No per-operator time is derived from it.
3. **Threads.** `torch.set_num_threads(2)`, with OMP/MKL/OPENBLAS_NUM_THREADS set to 2 before import. The actual BLAS configuration is recorded (`numpy.show_config`, `torch.__config__`).
4. **Boundaries.**
   - (i) **Conditional correction cost given u_c** is the main matched-time boundary.
   - (ii) **End-to-end**, including computing u_c, is reported for methods that need it. The direct solve and the whole-solve PCG curves start from u_c only if that is the stated variant; a direct solve is never charged an unused u_c.
   - Every method's standalone total includes its own required set-up once: hierarchy, spectral bounds, factorisations, deflation basis, conversions, operator data. No method receives set-up for free because another benchmark built an identical object. Reference-error evaluation stays outside timers.
5. **Whole-solve curves, frozen limits.** Every iteration's error is recorded offline; if the target is not reached, the result is labelled **censored**.

   | Solver | Limit |
   |---|---|
   | Two-grid PCG | ≤ 20 iterations |
   | SA-AMG PCG | ≤ 15 |
   | PCG-V (A3.2) | ≤ 10 |
   | FCG with K2 | ≤ 8 |
   | FCG with the PCG-smoothed V-cycle | ≤ 8 |
   | Repeated passes of each 1a cycle | 1–4 |

   The per-sample best curve point is an offline envelope. It is labelled as such, separately from the oracle stopping-time diagnostic.

## A5 Selection, fitting and confirmation rules

1. **Secondary classical champion.**
   - Candidates: {CHEB-T\*, POLY-S, POLY-L, SD, MR, PCG}.
   - Rule: the smallest mean log relative energy error on set D, **per family** (the champion may differ by family).
   - Ties within 1% go to the earlier candidate in that list.
   - Raw errors are used, and failed samples count with their raw error, or with u_c's error under the guarded variant only (A6).
2. **Spatial-control trigger** (v1 §3.3), evaluated on set D **before** confirmation. For a family, include SGS(2 sweeps) and SGS-preconditioned PCG(4) in the confirmation if **both** of the following hold for **all three** OA-VCB initialisations:
   - R = G(PCG → OA-VCB) has a lower 95% limit ≥ 1.25;
   - the same holds against the projected K2 cycle.
3. **Fitting** (POLY-S, POLY-L; CHEB-T selection).
   - Separate fits per family: channels on channel set T, barriers on barrier set T.
   - Objective: mean log relative energy error of the final projected output on set T.
   - Starts: the CHEB-equivalent coefficients and the zero polynomial (two starts). The better optimum is kept, and both are reported.
   - L-BFGS (float64 autograd), stopped at gradient ∞-norm < 1e-6 or 500 iterations, whichever comes first.
   - Recorded: final objective, gradient norm, iterations, coefficients and their SHA-256.
4. **Confirmatory and diagnostic status.**
   - **Confirmatory:** P1–P3 on set C, plus their per-initialisation bands.
   - **Secondary or diagnostic:** everything else, including set S, set LC, D1, D2, K2 and the whole-solve frontier.
   - All intervals are **pointwise** 95% bootstrap intervals; no simultaneous coverage is claimed.
5. **Corrections after confirmation** (replaces v1 §7's rerun rule).
   - Original outputs are always preserved.
   - A defect found after confirmation is documented with its correction, and the affected rerun is labelled **corrected/retrospective**.
   - If outcome inspection influenced a method or selection choice, a **new unopened confirmation set** is generated (rng 2026101603 / 2026101604) for the affected claims.
   - Status is never restored silently.
6. **The pre-confirmation note** records all A5 outputs: q\*, bound policy, coefficients and hashes, fit diagnostics, champions, trigger outcome. It also records the executed-source manifest (A7.3) before set C is generated.

## A6 Failure policy

- **Raw records.** Each sample records a raw status and the raw error when available. Status categories:
  - ok;
  - rank loss handled by the ridge, which is **not** a failure;
  - safeguard backstop activated;
  - exact convergence;
  - invalid curvature (pᵀAp ≤ 0);
  - non-finite output;
  - crash or time-out.
- **Guarded variants.** For any method, a variant "-G" may return u_c on non-finite output or a crash. Its fallback count, its error and the attempted work and time are reported **separately** from the unguarded method.
- **No imputation.** Errors are never imputed after the fact.
- **Paired summaries.** For each pair, they are computed on the complete-case set of identical operator ids, with the excluded ids listed.

## A7 Further corrections

1. **Near-island label.**
   - Renamed the **erosion-disconnected proxy (EDP)**. Its algorithmic definition is kept: a one-layer erosion leaves a conductive component of ≥ 30 elements touching no Dirichlet node. It does not establish a unique bridge or a width.
   - Categories, in this order:
     - **island**: detector-positive and geometric-positive;
     - **discordant**: exactly one of detector or geometric positive;
     - **EDP**: both negative, and the erosion criterion holds;
     - **strict non-island**: everything else.
   - The single final labeller (`ext/stage1_labels.py`) gives on set D:
     - barriers: island 39, discordant 3 (ids 315, 332, 358; all detector-positive, geometric-negative), EDP 4 (ids 301, 308, 312, 392), strict non-island 54;
     - channels: strict non-island 100.
   - Detector-positive 42 and geometric-positive 39 reproduce the Stage 0 and paper counts. The exploratory script's "5 erosion candidates" included id 332, which is detector-positive and is therefore discordant under the final ordering.
   - These counts are frozen as the labeller's set-D check.
2. **Input scale.**
   - Each model's input scale is computed **once from its own training data** (set T for the fixed-load models; the random-load training loads for the random-load model) and stored.
   - It is never recomputed on C/S/LC. Reference solutions on C/S/LC are computed separately, for evaluation only.
   - Positive homogeneity makes the scale cancel in exact arithmetic; this is provenance, not a performance correction.
3. **Provenance and executable gates.**
   - Every Stage 1 runner exits non-zero when a declared gate fails.
   - Each run writes a manifest with:
     - the SHA-256 of every executed source: `ext/*.py`, the imported `diagnostics_cost` helpers and the top-level modules;
     - package versions: NumPy, SciPy, PyTorch, PyAMG, Matplotlib;
     - thread and BLAS settings;
     - checkpoint SHA-256s.
   - The Stage 0 record is preserved as it was. It omitted the PyAMG/Matplotlib versions and helper hashes, and is not backfilled.
   - The original learned records carry no explicit sample ids (ordering inherited from the driver split), unlike the classical records. This is noted in the Stage 0 gate.
4. **Review-history audit.** Its proportions and its statements about what a higher-effort setting "would" have caught are **judgements**, not measurements. No further work on it is planned.

## Hashes

- v1 protocol: `324897bd94018407e396f19df7a9344eab74380d30e15d7107fb3cf59fd599fc` (unchanged).
- Amendment 1 and its evidence files: `STAGE1_AMENDMENT_1_SHA256.txt`.
