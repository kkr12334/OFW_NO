# Route C development round 1: plan

18 October 2026. Claude, for Kumar and ASTRA. This plan implements the bounded development step approved in `ASTRA_ROUTE_C_PILOT_REV1_REVIEW_2026-10-18.md` §5.

It is **post hoc development**, run only on pilot cases that have already been inspected. It is not confirmation. It ends with a go/no-go memo.

## What this round uses and does not use

**Out of scope for this round:**
- fresh seeds;
- the fixed-H refinement arm;
- training, new element families, 3D, and a broad cost study.

**Code:**
- The pilot records and runner stay frozen.
- New code lives in `ext/routec/dev1/`, including the evaluation fixes in deviation 2's addendum §3.

## 1 Frozen case list (13 geometry–contrast pairs, all already inspected)

**Failures (7):**
- barriers 0, 3 and 5, each at contrasts 10⁴ and 10⁶;
- channels 3 at 10⁶.

**Controls (6):**
- barriers 1 and 2, each at 10⁴ and 10⁶;
- channels 0 at 10⁴ and 10⁶.

**Inputs:** only the pilot's existing common inputs, regenerated deterministically from the recorded seeds:
- primal and three duals;
- coarse, PCG10, PCG30, PCG100, noise and smooth.

No new random input search.

## 2 Phase A: diagnostics

Deliverable: `dev1/diagnostics.json` and a short note. Per case, for the worst members and for every member with index ≥ 3:

- **Patch-local components.** The components of the K = 1 region (barriers) or of the K = c region (channels) inside each member's patch, not global labels.
- **Signed weights.** The signed partition weights on each component: minimum, maximum, and absolute mass.
- **Extremising modes.** The modes of the member's local quotient (the generalized eigenproblem behind C_θ): the top modes and their eigenvalues.
- **Energy contributions.** The share of the realised localisation excess, η_M3² − η_M6², attributable to each member. Overlap makes this a decomposition only up to overlap constants, which will be stated.
- **Residual coupling.** For the worst realised residuals, the coupling between the coarse-orthogonal part of the error and each member's top local modes, normalised.
- **Z projection defect.** The A-orthogonal projection defect of Z onto the unenriched OA coarse space.

**Question to answer:** which eigenmodes actually account for the observed loss. A large raw index alone does not count as an answer.

## 3 Phase B: estimate memo, written before any remedy is tested

Deliverable: `ROUTE_C_DEV1_ESTIMATE_MEMO`. It will state:

- **Construction:** the coarse projection; the local eigenproblem; the selection rule and its threshold; the selected modes; the admissible, coarse-orthogonal residual subspace.
- **The estimate:** the constants, and how the restricted-space estimate leads to the actual flux-reconstruction bound, including the moment compatibility of the enriched coarse correction and the overlap.
- **No shortcut:** the memo must not assume the remaining localisation loss is small. A complete theorem is not required, but the logical chain must be explicit enough to test.

The memo goes to ASTRA (via Kumar) before Phase C if it changes anything below.

## 4 Phase C: one spectral remedy at a fixed budget

**Budget, declared now, before any remedy is measured.** Per operator:
- the added coarse dimension is at most 10% of dim V_c (about 81 at L3);
- at most 3 modes per partition member;
- partition members and patch supports are unchanged.

The selection rule and threshold are fixed in the Phase B memo before Phase C runs.

**Compared on every case:**
- M3;
- M3 + spectral correction;
- M6;
- the matched geometric control M1SZ with the same selection rule and budget.

**Reported for every case:**
- the per-case maximum and median localisation loss λ;
- W (worst effectivity);
- added dimension and support;
- work counts.

**Success targets, declared now:**
- On the failure cases, the per-case maximum λ is ≤ 1.2 at 10⁴ and ≤ 1.5 at 10⁶.
- On the controls, λ is no worse than 1.05× M3's.
- The budget holds in every case.

## 5 Go/no-go memo (end of the round)

| Outcome | Next step |
|---|---|
| Targets met **and** the Phase B mechanism is supported by the Phase A diagnostics | Propose a frozen fresh-seed confirmation, including the fixed-H arm, for ASTRA's review |
| Exactly one of the two | Report it; ASTRA decides |
| Neither a credible bound mechanism nor substantial improvement at the budget | **Archive Route C at the narrower result**: the global theorem plus documented localisation limits. No automatic second campaign; a partition variant would need a new, concrete rationale from the diagnostics |
