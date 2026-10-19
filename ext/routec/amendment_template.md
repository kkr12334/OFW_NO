# Route C pilot: amendment 1

18 October 2026. Claude, for Kumar and ASTRA. This responds to `ASTRA_ROUTE_C_SCOPING_REVIEW_2026-10-18.md` and its mathematical supplement. It is a protocol amendment, written before any pilot geometry is generated.

**What stays frozen:**
- The scoping note `ROUTE_C_SCOPING_2026-10-18.md` is unchanged (SHA-256 `d9486e263a5dd0bd7f9b502e9ad252ee4b875ed87d7c074786a6c7acc8d577aa`). Where this amendment changes one of its rules, §2 lists the change.
- The seed allocations are unchanged: 2026101801 (channels) and 2026101802 (barriers).
- No pilot geometry has been generated. The runner refuses to start unless this file matches its declared hash, and unless both seeds are absent from every `.json` and `.log` record under `ext/`. The pilot's own records (`runs/pilot_*`) are exempt, so the L4 arm can follow the L3 arm.
- This amendment replaces the unreviewed local `ROUTE_C_SCOPING_ADDENDUM_1_2026-10-18.md` (the L4 arm), which is folded into §9.

**Data used for this amendment.** Every measured number comes from development data: seed-1 operator 300, or the published Poisson test in §11.1. None is confirmatory. All measured numbers are inserted directly from the raw record fields by `rc_amendment_check.py`, which also checks the conditions behind each numerical statement (§13).

## 1 Dispositions of the review

Every point of the review is accepted. Nothing is disputed.

| ASTRA § | Point | Disposition |
|---|---|---|
| 1 | Energy efficiency is not goal usefulness | **Accepted; headline changed (§3).** The pilot leads with the guaranteed algebraic **energy-error** certificate. Goal intervals are a secondary application, and goal usefulness is judged only through attainment of a declared tolerance by the deployable stopping rule (§7). "Goal-radius effectivity" (note §0 item 7) is renamed **product efficiency**. Polarization was tried first on development data (§11.2), as ASTRA suggested. |
| 2 | Lemma 1 correct; Proposition 3 needs a constructive feasibility argument and componentwise treatment | **Accepted.** ASTRA's three-case construction replaces the sentence "Dirichlet moments absorb the mean". Patches are unions of complete parent triangles with all three centroid sub-triangles, which is how the code builds them. A floating patch with more than one edge-connected component, or a boundary patch with a component that has no free Dirichlet edge, now **stops the run** (§8). |
| 2 | T11 can be proved | **Accepted, after an independent check.** ASTRA's Alfeld-split argument gives β_*²A ≤ GM⁻¹Gᵀ ≤ A, so C_glob ≤ 1/β_*, independent of scalar contrast and coefficient topology. I re-derived each step: per-element surjectivity of the zero-normal space onto constant vectors; the same β_T in the K⁻¹-weighted norm, because K is constant on T; and the direct-sum inclusion in Q_h. I also recomputed β_T with my own assembly, using rc_fem's Alfeld split, on all {{BETA_NT}} L3 triangles. Result: min β_T² = {{BETA_MIN2}}, so 1/β_* = {{BETA_INV}}, which matches ASTRA's independent value from the {{BETA_NC}} coarse shapes to {{BETA_DIFF}}. The constant is the same across the descendants of each coarse triangle (largest spread {{BETA_SPREAD}}), as red refinement predicts. The development values C_glob = {{CGLOB_MIN}}–{{CGLOB_MAX}} are below the bound. **T11 is no longer open in this setting.** It is ASTRA's argument; novelty is not claimed. Adaptation is not used, so it is not evidence that adaptation is needed. |
| 2 | T12/T13 open; flat p95 does not bound the maximum; "nearly quasi-monotone" is not a hypothesis | **Accepted.** The index maximum, p95 and the three worst members with their centroids are recorded per operator. Index-guided enrichment is **not** part of this pilot, and nothing is tuned on pilot outputs. |
| 2 | Narrow the (C) conclusion | **Accepted.** New wording: "the present (C) construction is not useful on these cases; sharpening the goal bound and/or refining the mesh remains outside this pilot." |
| 3 | M0 is an adaptation; check fidelity | **Accepted.** M0/M0′ are named "our RT0/Alfeld adaptation of MVY App. 8 / PRVW Def. 4.3". A fidelity check on PRVW's own Poisson test is in §11.1. Claims about classical methods are scoped to these implementations. |
| 3 | Matched controls for near-kernel and boundary completion | **Accepted (§5).** M1S and M1SZ use the same Shepard normalisation and fine Dirichlet-hat completion as M3, without and with Z. The adaptation criterion now uses M1SZ (§6). |
| 3 | Record actual support and cost; strong classical solvers before cost claims | **Accepted.** Work counts are recorded per construction (§5). This pilot makes **no cost or speed claim**, so the comparators ASTRA lists for a cost study are deferred to such a study: improving v, w with ρ first, a direct solve, a strong preconditioned solver, and a Gauss–Radau bound with a valid spectral bound. |
| 4A | J_h-normalised stopping is not deployable; charge all certificate work | **Accepted (§7).** The new rule is r ≤ ε/(1+ε)·\|ĉ\|, with a fixed absolute tolerance. The conclusion is limited to iteration and work counts. The pilot calls the cached engine `rc_fast.py`, never `rc_goal.energy_bounds`. Its lifting-free construction computes no lifting; the fixed-lifting constructions (M0, M0′, M5) need the lifting and compute it inside their own `batch`. |
| 4B | Barrier forcing must not depend on contrast | **Accepted.** f = 1 off the geometric mask, and 0 on it, at every contrast. The mask is stored before coefficients are assigned. Development scripts that used (K ≥ 0.5) are marked as such where cited (§11.2) and are not reused. |
| 4C | Aggregation, near-exact floor, inconclusive outcome, separate conclusions | **Accepted (§6).** |
| 4D | Executable gates, saved before outcomes, with code and input hashes | **Accepted (§8).** |
| 4D | Containment wording | **Accepted.** New wording: "contains J_h within the declared floating-point allowance 10³·eps·(\|J_h\| + \|ĉ\|)". The allowance is saved with each interval. Exact-input rows are numerical-zero checks. Divergence defects (absolute) and moment defects (relative) are reported separately. |
| 5 | Keep P1 triangles; L4 subset with inherited data; state the coarse scaling | **Accepted (§9).** |
| 6 | Literature; Li's scope | **Accepted.** Added: Ern–Guzmán–Potu–Vohralík (Alfeld splits) and Vacek–Papež–Strakoš. EGPV is to be read before any novelty claim about the centroid-split mechanism. Grappein–Weder (2607.11297) is marked screened. Li's result concerns finite-range estimators of the continuous error; it is not a no-go for target (D). |
| 7 | Do not force a goal-oriented headline | **Accepted.** If only the energy certificate succeeds, the narrower result is reported as such. |

## 2 Changes to the pre-declared pilot plan (note §9–§10)

| Note §9–10 says | Amendment | Reason |
|---|---|---|
| Methods M0–M6 with M1 = G2 and M2 = G1, lifting-free, coarse Dirichlet hats | Kept. **Added M1S, M1SZ** (matched controls) | ASTRA §3 |
| E1: median over operators of the worst effectivity over inputs, primal and dual | Kept. The input set is now fixed: the six non-exact inputs common to all contrasts. OA-VCB is reported separately, so E2 compares like with like. Unresolved cases are excluded | ASTRA §4C |
| E3: median and 90th percentile of η_method/η_joint | Kept, evaluated at each primary contrast. The maximum is reported next to them | ASTRA §4C |
| Supported requires "M0 has E2 ≥ 3 in at least one family" | **Removed from the candidate rule.** Reported as a separate classical-separation conclusion with the same threshold | ASTRA §4C: not a logical prerequisite |
| Supported requires "M1 and M2 both have E1(10⁴) ≥ 1.5 × M3's in at least one family" | **Replaced** by the matched adaptation conclusion, E1(M1SZ)/E1(M3) at 10⁴ (§6). The original comparison is still computed and reported, labelled superseded | ASTRA §3/§4C: M1/M2 change the near-kernel and the boundary completion together with adaptation |
| "Reduced contribution" if M3 is within 10% of M1 or M2 everywhere | Same 10% threshold, applied to the matched control | Same |
| No inconclusive outcome | **Inconclusive** class added | ASTRA §4C |
| Stopping: radius ≤ 10⁻³\|J_h\| and ≤ 10⁻⁶\|J_h\| | Same two tolerances (plus 10⁻² and 10⁻⁴ for resolution), with the **deployable rule** r ≤ ε/(1+ε)\|ĉ\| and a fixed absolute tolerance. Product and polarization intervals | ASTRA §4A, §1 |
| (C) certificate at 10⁴ on 4 operators (informative only) | **Dropped** | ASTRA §2: the present (C) construction is not useful at L3; the pilot is scoped to (D) |
| Work: local problems, coarse dimension, approximate MACs relative to one V-cycle | Exact counts instead: patches, flux unknowns, constraints, summed patch elements (overlap), LU non-zeros, largest patch, coarse dimension, PCG iterations. **No MAC estimate** and no speed claim | ASTRA §3/§4A |
| Containment must be 100%; any failure stops | Kept, "within the declared allowance". Any gate failure stops the run (§8) | ASTRA §4D |
| Barrier load not specified as contrast-independent | Fixed mask load | ASTRA §4B |
| L3 only | **L4 arm added (§9)** | Kumar's question; ASTRA §5 |
| Go/no-go criterion (iii): fewer iterations to a certified tolerance against M0 | Replaced by ASTRA's §7 criterion (§6, end) | ASTRA §4A, §7 |
| Expected cost 4–6 CPU hours | About 3 CPU hours (§10), with the cached engine | Implementation |

**Naming.** In the runner's records, geometric partitions are named by how many levels they sit below the fine level. Runner `G1` is level L−1, the note's "G2" at L3. Runner `G2` is level L−2, the note's "G1". Runner `OAZ` is the note's "OA2+Z". Method labels M0–M6 keep the note's meaning.

## 3 Headline object of the pilot

**Primary:** the guaranteed upper bound η on the algebraic energy error |||u_h − v||| of the assembled FE system, for arbitrary v ∈ V_h. Lemma 1 guarantees it for every construction tested. The questions are efficiency, robustness in contrast, and what localisation loses against the global optimum.

**Secondary:** the discrete goal interval for J(u_h), in product and polarization form, judged by attainment of declared tolerances (§7).

**Not in this pilot:**
- the continuous target (C);
- cost or speed claims;
- index-guided enrichment;
- new element families;
- training.

## 4 Metrics (recorded per operator, contrast, problem, input and method)

**Primary metrics:**
- **Energy effectivity:** η / |||e|||, for the primal and each of the three duals.
- **Localisation loss:** λ = η_method / η_M6, on the same residual.

**Goal metrics** (secondary), with ĉ = J(v) + R_v(w) and rem = J_h − ĉ:

| Metric | Definition | Notes |
|---|---|---|
| Product efficiency | η_u η_z / (\|\|\|e_u\|\|\| \|\|\|e_z\|\|\|) | |
| Corrected-goal effectivity | radius / \|rem\| | Only when \|rem\| > 10⁻¹² (\|J_h\| + \|ĉ\|) |
| Relative radius | radius / \|J_h\| | |
| Absolute radius | radius | |

Each goal metric is given for both the product and the polarization intervals.

**Polarization interval.** With s² = η_z/η_u:

J_h ∈ ĉ + [ (L₊² − U₋²)/4 , (U₊² − L₋²)/4 ]

- **Upper parts:** U± are the norms of s σ_u ± σ_z/s. Every construction is linear in the residual, so no new solves are needed.
- **Lower parts:** L± = |||s ρ_u ± ρ_z/s|||, where ρ is the Galerkin projection of the error onto the OA+Z coarse space. Since a(e, ρ) = |||ρ|||², the bound |||e||| ≥ |||ρ||| holds. The lower parts are the same for all methods.

**Near-exact floor:** a case with |||e||| ≤ 10⁻¹⁰ |||u_h||| is excluded from every ratio endpoint and kept as a numerical-zero check. Exact inputs are always numerical-zero checks.

**Inputs** (unchanged from the note), primal and dual alike:
- exact;
- coarse Galerkin on level L−1;
- Jacobi-PCG with 10, 30 and 100 steps;
- 10% white noise;
- 10% smooth perturbation (A⁻¹ applied to a random load);
- OA-VCB one pass: primal only, L3, c = 10⁴.

The two perturbations are normalised to 10% of |||u_h||| in energy. The perturbation seed [seed, id, round(10·log₁₀ c) + 100] is saved.

## 5 Methods

All methods use the cached engine `rc_fast.py`. It equals the reference implementation `rc_fem.py` to {{ENGINE_DIFF}} relative (`engine_check.json`).

**Matched Shepard construction** (`rc_fem.pu_shepard`), used identically for the geometric and the operator-adapted functions:
- At unknown nodes where the coarse functions sum to at least 10⁻³, the members are the normalised functions, so every normalising denominator is at least 10⁻³.
- At unknown nodes below that threshold, the node's own fine hat becomes a member and is added to the coarse space. On the development operator, the geometric hats need this at {{FALLBACK_G}} nodes and the operator-adapted functions at {{FALLBACK_OA}}.
- Dirichlet nodes get fine hats, with flux free on Dirichlet edges.

| ID | Construction | Coarse space | Role |
|---|---|---|---|
| M0 | Our RT0/Alfeld adaptation of MVY App. 8, fixed lifting | P1(T_0) | Classical reference |
| M0′ | Same, with PRVW Def. 4.3 data (ρ₀ term only on level 1) | P1(T_0) | Fidelity variant |
| M1 | Lifting-free; geometric level-(L−1) hats; coarse Dirichlet hats | P1(T_{L−1}) | As in the note |
| **M1S** | Lifting-free; matched Shepard construction on geometric level-(L−1) interior hats | Hats + fallback | **Matched control without Z** |
| **M1SZ** | As M1S, plus near-kernel Z | + Z | **Matched control for the adaptation effect** |
| M2 | Lifting-free; geometric level-(L−2) hats | P1(T_{L−2}) | Alternative; changes coarse dimension and patch size together |
| **M3** | Lifting-free; matched Shepard construction on the one-level OA functions, plus Z | + Z | **Candidate** |
| M4 | As M3, without Z | — | Near-kernel effect |
| M5 | As M3, with the fixed lifting (Prop. 2) | + Z | Lifting effect |
| M6 | Global moment-constrained optimum on RT0(T_b) | — | Localisation reference; not a deployable cheap baseline |

**M0 and M0′ coincide when K is constant on every level-1 element**, for example on Poisson. They differ only in the term K∇ρ₀·∇ψ on levels j ≥ 2. That term is then piecewise constant on T_{j−1}, so the operator (Id − Π_{j−1}) removes it.

**Work counts**, recorded per construction call:
- number of patches;
- flux unknowns;
- constraints;
- summed patch elements, which gives the overlap;
- LU non-zeros;
- largest patch;
- coarse dimension;
- build-and-evaluate time.

The PCG iterations of the stopping test are recorded separately. These are counts. Timings are recorded but support no conclusion.

## 6 Endpoints, aggregation and decision rules

**Unit of analysis.** The geometry (operator) is the independent unit. Its four problems (primal and three duals), its inputs and its goals are not independent samples. Eight geometries per family make a **screening pilot**, not a population-level robustness claim.

**Per-operator summaries**, for family f, method m and contrast c:
- **W(op):** the largest energy effectivity over the four problems and the six common inputs (coarse, PCG10, PCG30, PCG100, noise, smooth), excluding unresolved cases.
- **λ(op):** the largest localisation loss over the same cases.

All per-operator values are saved. A per-input-type breakdown is reported, and the OA-VCB effectivities are reported separately.

**Endpoints:**
- **E1(f, m, c):** median of W over the eight operators. The maximum is reported next to it.
- **E2(f, m):** E1(f, m, 10⁴) / E1(f, m, 1).
- **E3(f, m, c):** median and 90th percentile (numpy linear interpolation) of λ over operators. The maximum is reported.

Primary contrasts are 1, 10² and 10⁴. The 10⁶ arm is a reported stress test, outside the decision rules.

**Candidate quality, M3, per family.** The thresholds are the note's engineering screening targets, not theorems or statistical bounds.

- **Support:** all three conditions hold:
  - E1 ≤ 1.5 at each primary contrast;
  - E2 ≤ 1.25;
  - E3 median ≤ 1.10 and 90th percentile ≤ 1.5 at each primary contrast.
- **Stop:** any one of the following:
  - E2 > 2;
  - E3 median > 1.5 at a primary contrast;
  - a gate failure that is not a defect of the check itself (§8).
- **Inconclusive:** anything else.

"Candidate supported" in the note's sense requires Support in **both** families.

**Separate conclusions.** None is a prerequisite for another.

- **Adaptation effect, per family.** Q = E1(M1SZ, 10⁴) / E1(M3, 10⁴).
  - Q ≥ 1.5 is a benefit.
  - Q ≤ 1.10 is no benefit. In both families, that is the note's "reduced contribution": lifting-free localisation only.
  - Anything between is inconclusive.
  - Also reported: Q at every contrast, and the number of operators where W(M1SZ) ≥ 1.5·W(M3).
  - The superseded unmatched rule (M1 and M2 against M3) is reported as well.
- **Near-kernel effect.** E1(M4)/E1(M3) and E1(M1S)/E1(M1SZ), with the near-kernel dimension of every barrier operator. If no pilot operator has dimension ≥ 1, this comparison is labelled uninformative.
- **Lifting effect.** E1(M5) / E1(M3).
- **Classical separation**, scoped to our adaptations: E2 for M0 and M0′. E2 ≥ 3 in a family counts as separation (the note's threshold).
- **Localisation.** E3 for every method.
- **Indices.** C_θ maximum and p95 for runner G1, G1SZ and OAZ, with the three worst members' centroids; C_glob for the first two operators per family at L3.

**Proceed beyond the pilot (replaces note §10)**, following ASTRA §7. Both conditions must hold:

1. The matched comparison shows a localisation benefit: "Support" and "benefit" in at least one family, reproduced across its operators (maxima reported, not just medians).
2. Either a credible localisation theorem (T12) or useful certification at a justified total cost. The second needs a separately declared cost study.

**Insufficient on their own:**
- fewer iterations with a more expensive test;
- product efficiency near one with a huge goal interval;
- favourable medians that hide uncontrolled maxima.

If only the energy certificate succeeds, that narrower result is reported as such.

## 7 Goal intervals and deployable stopping (secondary)

**Intervals.** Every (goal, input pair, method) gets both interval forms. The OA-VCB primal is paired with the coarse dual; every other input is paired with the same input type. Each interval must contain J_h within the allowance, or the run stops.

**Stopping test.** Goal 0; methods M0, M3 and M6 (M6 as reference).
- Jacobi-PCG runs on the primal and on the dual from zero.
- The certificate is evaluated at iterations 5, 10, 20, 40, 80, 160 and 320 on both iterates.
- At each check, each interval form gives a centre ĉ and a radius r.
- **Stop at the first check where r ≤ ε/(1+ε)·|ĉ|**, which implies r ≤ ε|J_h| for every value in the interval, **or where r ≤ τ_abs = 10⁻¹²**.
- J is the mean of u over the goal disc, so τ_abs is in units of u. The development goal values are {{J_MIN}} to {{J_MAX}}, so τ_abs only matters for near-zero goals.
- ε ∈ {10⁻³, 10⁻⁶} as declared in the note, plus 10⁻² and 10⁻⁴.

**Recorded:**
- the first check satisfying the rule, and which criterion triggered;
- non-attainment by iteration 320;
- the number of certificate evaluations;
- the construction work counts;
- the PCG iterations on both problems;
- all rows, so any other ε can be read off without a new run.

**Allowed conclusion:** an iteration and work comparison between certificates on the same iterates. It is not a cost conclusion: at N = 3486, Jacobi-PCG reaches round-off within 320 iterations, so attainment itself is expected and the comparison is the information.

**Goal usefulness:** only for (family, contrast, ε, method) cells where the rule attains ε within 320 iterations, reported as the fraction of operators.

## 8 Executable gates

Gate results are saved before the outcome records of each operator and contrast are written. Any failure stops the run and records the reason. A failure caused by a defect of the check itself (as in §11.4) may be fixed only through a dated note; the run is then repeated from its start with the same seed, and the failed record is kept under `runs/` with a `pilot_` prefix.

**Before any data are generated (pilot mode):**
- the amendment hash matches;
- both seeds are absent from every non-pilot record;
- the manifest records the SHA-256 of the code files, the OA-VCB checkpoint, `checkpoint_map.json`, `frozen_scales.json` and this amendment.

**Per operator and contrast:**
- **Exact solves:** finite, with residual ≤ 10⁻⁹ relative.
- **Partitions of unity:** each sums to one at every node (defect ≤ 10⁻¹²) and has finite values. The Shepard fallback count is saved.
- **Inputs:** finite.
- **Coarse spaces:** the Cholesky factorisation of CᵀAC succeeds after Z is added, which certifies full rank. A failure stops the run as an error.
- **Galerkin residual:** ≤ 10⁻⁸, scaled by |C|ᵀ|R|.
- **Patch topology** (checked at construction): a floating patch must be one edge-connected component, and every component of a boundary patch must have a free Dirichlet edge.
- **Patch solutions:** finite; full constraint defect, including any row removed as redundant, ≤ 10⁻⁸ relative to the largest constraint target of the construction. Non-finite defects raise an error instead of passing through a max.
- **Per (problem, input, method):**
  - the bound is finite;
  - global moment defect ≤ 10⁻⁸ relative (this also covers M6);
  - the guarantee η ≥ |||e|||(1 − 10⁻⁹) holds.
- **Per (goal, input, method):** the product and polarization intervals contain J_h within the allowance.
- **Warnings:** no RuntimeWarning, LinAlgWarning or MatrixRankWarning. Other warnings are saved.
- **Errors:** any other exception stops the run with its message recorded.

## 9 Mesh-size arm (L4; replaces the addendum)

**Data:**
- The first two pilot geometries per family, at contrasts 1 and 10⁴.
- The mask (hence K and f) and the goal regions are **inherited** from the L3 ancestor elements. Goals are unions of L3 elements, so neither coefficient interfaces nor goal regions move.

**Methods:** M0, M1SZ, M3, M6. **Inputs:** as at L3, without OA-VCB. No stopping test.

**Coarse scaling.** One-level coarsening is kept, so H/h = 2 as at L3. The coarse dimension grows with the mesh: the OA and geometric level-3 spaces at L4 have {{L4_COARSE}} functions, against {{L3_COARSE}} at L3 (development operator, no near-kernel). This is the regime tested; a fixed coarse space would be a different experiment.

**Reading rule:** for M3, a ratio of W at L4 to W at L3 ≤ 1.25 for each of the four (geometry, contrast) pairs means no gross deterioration. With two operators, h-robustness is not established.

## 10 Run plan and cost

**Smoke runs** (development operator 300, contrast 10⁴, final code) completed with every gate passing:

| Run | Seconds per (operator, contrast) | Peak memory | Gates passed |
|---|---:|---:|---:|
| L3 channels | {{SMK_T_ch3}} | {{SMK_M_ch3}} GB | {{SMK_G_ch3}} |
| L3 barriers | {{SMK_T_ba3}} | {{SMK_M_ba3}} GB | {{SMK_G_ba3}} |
| L4 channels | {{SMK_T_ch4}} | {{SMK_M_ch4}} GB | {{SMK_G_ch4}} |
| L4 barriers | {{SMK_T_ba4}} | {{SMK_M_ba4}} GB | {{SMK_G_ba4}} |

**Pilot:**
- L3: 2 families × 8 operators × 4 contrasts, plus C_glob for two operators per family.
- L4: 2 × 2 × 2.

The runs are sequential (memory): channels L3, barriers L3, channels L4, barriers L4. Total about {{PILOT_HOURS}} CPU hours. No GPU and no training.

## 11 Development evidence behind this amendment (not confirmatory)

### 11.1 Fidelity of the classical baseline (`rc_fidelity_poisson.py` → `fidelity_poisson.json`)

**Test:** PRVW's sinus problem.
- Ω = (−1,1)², u = sin 2πx sin 2πy.
- Our own unstructured initial mesh ({{FID_NC}} triangles), refined 4 times: {{FID_NE}} triangles, N = {{FID_N}}.
- PCG with an ILU(10⁻⁴) preconditioner, zero start.

**Effectivity of the algebraic upper bound:**

| PCG iteration | Relative algebraic error | M0 (= M0′ here) | Joint optimum M6 |
|---:|---:|---:|---:|
| 1 | {{FID_E1}} | {{FID_M1}} | {{FID_J1}} |
| 2 | {{FID_E2}} | {{FID_M2}} | {{FID_J2}} |
| 3 | {{FID_E3}} | {{FID_M3}} | {{FID_J3}} |
| 6 | {{FID_E6}} | {{FID_M6}} | {{FID_J6}} |

M0 and M0′ differ by at most {{FID_M0M0P}} here (K = 1, §5). PRVW report 1.01 at p = 1 with RTN₁ spaces on every level.

**Reading.** Our lowest-order adaptation is {{FID_LOSS}} less sharp than the published value on Poisson. The cause cannot be separated here between the space (RT0 against RTN₁), the lifting, the preconditioner and the mesh. Either way, a factor of this size cannot by itself explain M0's effectivity of {{SMK_M0_ch}} on the channels development operator at contrast 10⁴ (§11.3). The test cannot distinguish M0 from M0′; in the pilot they differ (§11.3).

### 11.2 Polarization on the development goal pairs (`rc_polarization_dev.py` → `polarization_dev.json`)

These rows used the development barrier load (K ≥ 0.5), not the fixed mask load of §1. They are not reused.

Polarization half-width divided by the product radius, method M3: {{POL_MIN}}–{{POL_MAX}} across the six input pairs. All intervals contain J_h.

**Radii relative to |J_h|** (M3; product → polarization half-width):

| Family | Coarse/coarse | OA-VCB/coarse | PCG30/PCG30 |
|---|---:|---:|---:|
| Channels | {{POL_ch_c}} | {{POL_ch_n}} | {{POL_ch_p}} |
| Barriers | {{POL_ba_c}} | {{POL_ba_n}} | {{POL_ba_p}} |

**Reading.** Polarization helps (a factor of 2–12), but the learned barrier interval is still {{POL_ba_n_pct}} of |J_h|. Its width comes from the size of the input errors, chiefly the coarse dual, not from the flux reconstruction: on this pair M3's product efficiency is {{POL_ba_n_eff}}. Hence the energy-first headline (§3).

### 11.3 Matched controls and work in the smoke runs (`runs/smoke_*_L3.json`; contrast 10⁴)

Largest energy effectivity over the four problems and the six common inputs:

| Method | Channels | Barriers |
|---|---:|---:|
{{SMK_TABLE}}

**Reading:**
- On **barriers**, the matched control M1SZ ({{SMK_ba_M1SZ}}) is within {{SMK_ba_Qpct}} of M3 ({{SMK_ba_M3}}), against {{SMK_ba_M1}} for M1. On this operator, the whole geometric deficit came from the **boundary completion**, not from operator adaptation (Q = {{SMK_ba_Q}}). M1SZ is also far cheaper (table below).
- On **channels**, the matched control remains far from M3 (Q = {{SMK_ch_Q}}).
- **Expectation stated before the pilot:** if the barrier result reproduces, the barrier adaptation conclusion will be "no benefit", and any adaptation case rests on channels. One development operator decides nothing; the pilot's matched comparison does.

**Work per construction (barriers smoke run):**

| Method | Coarse dim | Patches | Flux unknowns | Constraints | Overlap | LU non-zeros | Largest patch (flux elements) |
|---|---:|---:|---:|---:|---:|---:|---:|
{{WORK_TABLE}}

**Reading.** The operator-adapted supports are large. At L3, M3's summed LU non-zeros are {{W_RATIO_M6}} times the global optimum's and {{W_RATIO_M1SZ}} times the matched control's. **At this size the candidate is not cheaper than the global optimum.** This reinforces that the pilot makes no cost claim, and any later cost case must come from scaling, not from L3.

### 11.4 Gate defects fixed during development (no pilot data involved)

Three checks were themselves wrong and were corrected before this amendment:

| Gate | What happened | Fix |
|---|---|---|
| Geometric Shepard denominators | Zero at some nodes | Fallback member (§5) |
| Patch constraint defect | Normalised by each patch's own data, which can be at round-off | Normalised by the largest target of the construction |
| Galerkin residual | Normalised by CᵀR, which vanishes when the input is already Galerkin in that space | Normalised by \|C\|ᵀ\|R\| |

**Accounting fix:** work counts accumulated across calls; they are now reset per call. The earlier smoke records are kept in `runs/superseded_smoke/`.

**After all fixes:**
- the engine was re-verified against the reference;
- all four smoke runs were repeated with the final code (§10).

## 12 Corrections to the scoping note (for any later revision; the frozen note is unchanged)

1. §0 item 7: "goal-radius effectivity" → "product efficiency η_uη_z/(|||e_u||| |||e_z|||)". Add the corrected-goal and relative radii (ASTRA §1 table).
2. §0 item 2 and §6: narrow the (C) conclusion as in §1.
3. §2 T11: proved for this setting (ASTRA supplement §1), with the independent check of the constant in §1.
4. §4 Proposition 3: replace "(i)" by ASTRA's three-case constructive feasibility proof, with the componentwise rule.
5. §3/§6:
   - "every interval contained its target" → "within the declared floating-point allowance";
   - give the divergence defects (absolute, maximum {{DIV_DEF}}) and moment defects (relative, maximum {{MOM_DEF}}) separately.
6. §1/§8: Li (2026) concerns finite-range estimators of the continuous error. It is not a no-go for (D), and a small number of non-quasi-monotone vertices does not neutralise it for (C).
7. Element scope: Lemma 1 is general. Propositions 2–3 as stated use P1 interpolation and the centroid-split moment map.
8. §8: add Ern–Guzmán–Potu–Vohralík (to read before any novelty claim about the centroid-split construction) and Vacek–Papež–Strakoš; mark 2607.11297 screened.

## 13 Records

**Code:**
- `rc_pilot.py`: runner.
- `rc_fast.py`: engine; its self-check writes `engine_check.json`.
- `rc_fem.py`: reference implementation, with `pu_shepard`.

**Development checks:**
- `rc_beta_shape.py` → `beta_shape.json`.
- `rc_fidelity_poisson.py`, `rc_polarization_dev.py`.
- `runs/smoke_*`: development smoke runs.

**Analysis, fixed before generation:**
- `rc_pilot_analysis.py` (SHA-256 `{{ANALYSIS_SHA}}`) implements §6–§9 exactly and was exercised on the smoke records. Any later change to it is a dated, labelled deviation.

**This amendment:**
- `amendment_template.md` + `rc_amendment_check.py` → this file and `amendment_check.json`. The script inserts every measured number directly from the raw record fields, checks the conditions behind each numerical statement (listed in its output), and fails if a check fails or a placeholder remains.
- `ROUTE_C_PILOT_AMENDMENT_1_SHA256.txt`: the hash the runner checks.
