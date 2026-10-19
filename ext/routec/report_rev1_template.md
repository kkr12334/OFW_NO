# Route C pilot: report, revision 1

18 October 2026. Claude, for Kumar and ASTRA. This revision responds to `ASTRA_ROUTE_C_PILOT_REVIEW_2026-10-18.md`. The first report (`ROUTE_C_PILOT_REPORT_2026-10-18.md`) is kept unchanged.

**What the revision rests on:**
- **Protocol:** `ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md` and its deviations 1 (reference solves) and 2 (goal-interval evaluation and stopping gates).
- **Data:** a complete repeat of the pilot made after deviation 2.
- **Checks:** every number below is inserted from the raw records by `rc_report_check_rev1.py`, which also checks each statement that depends on a number (§12). The point-by-point response is in `CLAUDE_RESPONSE_TO_ASTRA_ROUTE_C_PILOT_REVIEW_2026-10-18.md`.

## Summary

**Gate status.** The first report's runs did not gate the stopping test. ASTRA found four stopping intervals that missed the reference value (barriers, geometry 1, 10⁶, iteration 320).
- **Cause:** the interval centre was evaluated in double precision. Its rounding error, 6.7·10⁻¹⁰, exceeded the radius near convergence.
- **Disposition:** a genuine implementation defect in the goal-interval evaluation, shared by all methods. Under amendment §6 the pilot carried a Stop status until repaired.
- **Repair (deviation 2):**
  - residuals and centres in extended precision;
  - explicit evaluation terms in every interval and in the stopping rule;
  - the same gates on the stopping path as on the ordinary path.
- **Result:** all four runs were repeated from the start. **Every gate now passes**, {{G_TOTAL}} in all, of which {{G_STOP}} are on the stopping path. The four intervals that failed are now contained (§6).

| Verdict under the declared rules (amendment §6) | Channels | Barriers |
|---|---|---|
| Candidate (M3) | **Support** | **Inconclusive.** E1 and E2 pass; localisation loss E3 fails at 10⁴ |
| Operator adaptation, matched control | **Benefit** (Q = {{CH_Q}}) | **Benefit** (Q = {{BA_Q}}) |
| Near-kernel enrichment Z | Uninformative (no near-kernel) | Negligible on the declared endpoint; does not remove the failures |
| Fixed lifting against lifting-free | Lifting-free lower by {{CH_LIFT}}× at 10⁴ | {{BA_LIFT}}× |
| Classical separation (our M0, M0′) | Yes (E2 = {{CH_M0_E2}}) | Yes (E2 = {{BA_M0_E2}}) |

**The repeat changed none of the verdicts:**
- **Endpoints:** every E1 and E3 value agrees with the deviation-1 records to {{REPRO_ENDPOINTS}}.
- **Overall:** "candidate supported" (Support in both families) is not met. **No full programme**: neither a localisation theorem nor a cost case exists. ASTRA agrees.

**What the pilot shows** (screening pilot, 8 fresh geometries per family at L3):
1. **The global certificate (M6) behaved contrast-robustly** on every sampled case: worst energy effectivity {{M6_MAX}} over all 64 L3 (geometry, contrast) cases.
   - The contrast-independent result is ASTRA's proof for this scalar P0/P1/Alfeld setting.
   - These effectivities, and the computed constants C_glob = {{CG_MIN}}–{{CG_MAX}} (bound 2.8873), are supporting diagnostics, not universal or machine-verified bounds.
2. **The localised candidate M3 matches M6 on channels at the primary contrasts (1, 10², 10⁴)**: median localisation loss ≤ {{CH_E3_MED_MAX}}, maximum {{CH_E3_MAX_1e4}} at 10⁴.
   - Even on channels, robustness is not unrestricted: one geometry loses {{CH_G3_LAM}} at the 10⁶ stress contrast.
   - **On barriers M3 does not match M6.** Three of 8 geometries lose up to {{BA_E3_MAX_1e4}} at 10⁴, and six of 8 lose up to {{BA_E3_MAX_1e6}} at 10⁶.
3. **In the tested ablations over the sampled contrasts 10²–10⁶**, the constructions lacking one ingredient have finite-range log–log slopes p, with effectivity ∝ c^p:

   | Construction | Lacks | p |
   |---|---|---|
   | M0 (classical) | Both | {{EXP_M0}} |
   | M1SZ (geometric partition, lifting-free) | Operator adaptation | {{EXP_M1SZ}} |
   | M5 (operator-adapted partition, fixed lifting) | Lifting-free construction | {{EXP_M5}} |

   M3 stays flat wherever it does not lose to localisation. These slopes describe the implemented constructions over this range. They are not asymptotic rates, and not a necessity result for all constructions.
4. **The barrier failures are associated with partition members that straddle a barrier** (exploratory, §5). This is an association, not a failure criterion, and not yet a sufficient hypothesis for T12.
5. **The L4 arm is a valid but limited refinement check** (§7). It passes its rule under the declared one-level scaling, H/h = 2. It does not test the regime where the coarse mesh leaves the coefficient unresolved; that needs the fixed-H arm proposed in §10.
6. **Goal intervals (secondary) are useful on many, not all, input–goal pairs** (§6). With the learned primal and a coarse dual at 10⁴, the median half-width is {{GOAL_CH_OAVCB}} of |J_h| on channels and {{GOAL_BA_OAVCB}} on barriers, but the barrier maximum is {{GOAL_BA_OAVCB_MAX}}.
   - The repaired stopping test reaches every tolerance within 320 iterations.
   - M3 needs fewer iterations than M0 at 10⁴, but its summed LU non-zeros per construction are much larger. **No cost advantage is demonstrated.**

## 1 What was run, and integrity

| Run | Records | Gates | All passed | Largest reference-error estimate |
|---|---|---:|---|---:|
| Channels L3 | 8 geometries × 4 contrasts | {{G_ch3}} | yes | {{REF_ch3}} |
| Barriers L3 | 8 × 4 | {{G_ba3}} | yes | {{REF_ba3}} |
| Channels L4 | 2 × 2 | {{G_ch4}} | yes | {{REF_ch4}} |
| Barriers L4 | 2 × 2 | {{G_ba4}} | yes | {{REF_ba4}} |

- **Code and analysis.** All four records were produced by one code version (`PILOT_CODE_SHA256_DEVIATION2.txt`). The extended format is recorded in each record's design block ({{EXT_DESC}}). The analysis is the frozen `rc_pilot_analysis.py`, unchanged (SHA-256 `4683bd70…`). No numerical warning was raised. Peak memory was {{PEAK}} GB.
- **Earlier records**, all kept in `runs/` with a `pilot_` prefix:
  - the aborted launch and the container-interrupted run (`runs/pilot_aborted_launch_NOTE.md`);
  - the deviation-1 gate stop;
  - the first complete repeat (`pilot_v0_channels_L3`);
  - the deviation-1 runs on which the first report was based (`pilot_dev1_*`).
- **The deviation-2 repeat against the deviation-1 records:**
  - every bound agrees to within {{REPRO_ETA}}, relative;
  - the changes come from the extended-precision residuals;
  - no case changes resolution status;
  - the stopping first checks differ in {{STOP_CHANGED}} of {{STOP_TOTAL}} (case, method, tolerance) cells.
- **The development expectation was wrong.** The amendment (§11.3) expected "no adaptation benefit" on barriers from one development operator. On every pilot barrier geometry the matched control is at least 1.5× worse at 10⁴.

## 2 Primary endpoints

W is the worst energy effectivity of an operator over the four problems and six common inputs, unresolved cases excluded. E1 is the median of W over the 8 geometries, with the maximum in brackets. E2 = E1(10⁴)/E1(1). E3 is the median / 90th percentile [maximum] over geometries of the per-geometry worst η_m/η_M6. Full tables for all ten methods are in `pilot_tables.md`.

**Channels**

| Method | E1, c = 1 | E1, c = 10² | E1, c = 10⁴ | E2 | E3 at 10⁴ | E1, c = 10⁶ (stress) |
|---|---:|---:|---:|---:|---:|---:|
{{T1_channels}}

**Barriers**

| Method | E1, c = 1 | E1, c = 10² | E1, c = 10⁴ | E2 | E3 at 10⁴ | E1, c = 10⁶ (stress) |
|---|---:|---:|---:|---:|---:|---:|
{{T1_barriers}}

## 3 Verdicts under the declared rules

{{T_VERDICTS}}

**Near-kernel (corrected).** The tested Z enrichment has a negligible effect on the declared worst-case median endpoints, and it does not remove the observed failures. Some individual residuals improve modestly. Across all barrier records:
- the largest per-residual ratio η_M4/η_M3 is {{NK_M4M3_MAX}}, at geometry {{NK_M4M3_WHERE}};
- the largest η_M1S/η_M1SZ is {{NK_M1S_MAX}}, at geometry {{NK_M1S_WHERE}}.

The first report suggested that the one-level OA functions already carry the near-kernel modes. That was untested and is withdrawn. The development step will measure the A-orthogonal projection defect of Z onto the unenriched coarse space.

**Other points:**
- **Superseded unmatched rule (note §9).** At 10⁴, E1(M1)/E1(M3) and E1(M2)/E1(M3) are {{CH_SUP_M1}} and {{CH_SUP_M2}} on channels and {{BA_SUP_M1}} and {{BA_SUP_M2}} on barriers.
- **OA-VCB inputs** (learned one-pass primal, 10⁴, reported separately). M3 has W between {{OAVCB_M3_MIN}} and {{OAVCB_M3_MAX}}. M0 has {{OAVCB_M0_MIN}}–{{OAVCB_M0_MAX}}.

## 4 Barrier localisation failure

| Geometry | Near-kernel dim (10⁴) | W(M3) at 10⁴ | λ(M3) at 10⁴ | W(M3) at 10⁶ | λ(M3) at 10⁶ | Worst OAZ index at 10⁴ | High-index members that straddle (exploratory) |
|---:|---:|---:|---:|---:|---:|---:|---|
{{T_BARRIER_GEOM}}

- **This is a localisation loss.** M6 stays at most {{M6_MAX_BA}} on every barrier case.
- **Growth.** The three geometries affected at 10⁴ (0, 3, 5) grow by a factor of {{AFF_GROWTH}} from 10⁴ to 10⁶, a finite-range observation.
- **The index tracks the loss.** The four geometries with the largest OAZ robustness index at 10⁴ ({{TOP4_IDX}}) have the four largest losses at 10⁶ ({{TOP4_LAM}}).
- **The two geometries with a detected near-kernel are not affected.**

## 5 The straddling association (exploratory; no verdict depends on it)

`rc_explore_localisation_loss.py` labels the edge-connected components of the K = 1 region globally. It counts a member as carrying a component if max θ ≥ 0.05 there. At barriers, 10⁴:

- **High-index members:** {{EXP_N_STRADDLE}} of the {{EXP_N_BIG}} members with index ≥ 3 carry weight on two or more K = 1 components separated by a barrier. One does not.
- **The association is not specific.** {{EXP_N_ALL_STRADDLE}} members straddle in total, so {{EXP_N_LOW_STRADDLE}} straddling members have index < 3.
- **The inventory is partial.** The script saves only the five worst members per geometry, and none of those carries an enclosed island.

**Why this is an association, not a criterion** (following ASTRA §5):
- The labels are **global**. A region that is connected globally can still split into several components inside one patch, which is what the local quotient sees.
- The carrying threshold is diagnostic. With signed OA functions it can miss negative weight.
- A large index is a supremum over test functions. It does not show that the realised residual excites the bad direction.
- Component counts do not control variation along a component, bottlenecks, overlap, or shape constants.
- The single channel stress outlier (λ = {{CH_G3_LAM}}) has its worst member, index {{CH_G3_IDX}}, inside one channel component.

**"No straddling" is checkable, so it is not circular. It is not sufficient as formulated.** Two remedy mechanisms must also be kept apart:
- **Enlarging the coarse space** with the partition fixed cannot reduce the raw index C_θ. It can only remove dangerous modes from the admissible residual space, so a proof needs a bound on a restricted, coarse-orthogonal space that the selected modes control.
- **Changing the partition** changes the quotient itself.

## 6 Goal intervals and stopping (secondary)

**Learned primal with coarse dual, and PCG30 on both sides, at 10⁴.** Polarization half-width / |J_h|, M3; 24 geometry–goal pairs per family (not 24 independent geometries):

{{T_GOAL_TAILS}}

**Medians by method and input pair** (median product efficiency in brackets):

{{T_GOALS}}

**Reading:**
- The intervals are useful on many of these input–goal pairs. A favourable median is not uniform usefulness.
- These width statistics are descriptive. They are separate from amendment §7's tolerance-attainment definition.

**Repaired evaluation** (deviation 2). In the barrier stress case that failed before (geometry 1, 10⁶, iteration 320):

{{T_FAILCASE}}

- **Evaluation terms.** On the ordinary goal intervals, the rigorous centre-evaluation bound τ_eval is at most {{TAU_GOAL_MAX_REL}} of the certified radius.
- **Converged stopping iterates.** On the stopping rows τ_eval exceeds the certified radius in {{TAU_DOM_ROWS}} of {{TAU_ROWS}} rows, all at iterations ≥ {{TAU_DOM_MINIT}}, where the certified radius has fallen to round-off level. There the total radius is set by the evaluation bound; its largest value is {{TAU_EVAL_MAX_ABS}}.
- **The bound is pessimistic.** In the case above, τ_eval = 1.8·10⁻¹⁰, while the actual discrepancy is 1.3·10⁻¹².
- **The empirical allowance.** The old allowance test, still recorded but no longer a gate, would now fail on {{LEGACY_FAILS}} stopping rows and {{LEGACY_GOAL_FAILS}} ordinary goal intervals.
- **What is estimated.** The second-order terms a_u and a_z, and the reference uncertainty τ_ref, are estimates (deviation 2). No rigorous floating-point enclosure of the whole computation is claimed.

**Deployable stopping test** (goal 0, Jacobi-PCG). Median first check reaching the tolerance, product / polarization, using total radii:

{{T_STOP}}

**Reading:**
- Every cell attains every tolerance within 320 iterations, and every stop was triggered by the relative criterion.
- At c = 1 the three certificates stop within one check of each other.
- At 10⁴ and 10⁶, M3 and M6 reach ε = 10⁻³ earlier than M0 (medians {{STOP_M3_RANGE}} against 160 iterations).
- **This is an iteration comparison only.** Median summed LU non-zeros per construction are M3 {{WORK_M3}}, M6 {{WORK_M6}} and M0 {{WORK_M0}}. That measures factor storage, a work proxy; it is not a FLOP or wall-time ratio.

## 7 Mesh-size arm (L4)

{{T_L4}}

**Reading rule (declared):** W(M3, L4)/W(M3, L3) ≤ 1.25 for each (geometry, contrast). It is **met**: the largest ratio is {{L4_M3_MAX_RATIO}}.

**What this shows.** The arm is a valid, limited check: it refines the same inherited PDE data under the declared one-level coarse scaling (H/h = 2, with a growing coarse space), and M3 shows no gross deterioration on two geometries.

**What it does not show:**
- **Robustness when the coarse mesh leaves the coefficient unresolved.** At L4 the coarse mesh (L3) is the mesh on which the inherited coefficient is defined. The matched control M1SZ drops from {{L4_M1SZ_L3}} to {{L4_M1SZ_L4}}, and M3's barrier loss from {{L4_M3_L3}} to {{L4_M3_L4}} (barrier geometry 0, 10⁴). This is consistent with coefficient alignment, but the comparison cannot separate alignment from changes in resolution, support or coarse dimension.
- **General h-robustness.** Two geometries cannot establish it.

The first report called the arm unable to test what it was meant to. That overstated the limitation, and is corrected here.

## 8 Indices and the global constant

- **OAZ robustness index at 10⁴** (median over geometries of the per-geometry maximum): {{IDX_CH_1e4}} on channels and {{IDX_BA_1e4}} on barriers. The maxima are {{IDX_CH_1e4_MAX}} and {{IDX_BA_1e4_MAX}}.
- **Runner G1:** medians {{IDX_G1_CH}} and {{IDX_G1_BA}}.
- **OAZ p95:** {{IDX_P95}}. The largest values grow by ×{{IDX_GROWTH_CH}} (channels) and ×{{IDX_GROWTH_BA}} (barriers) from 10⁴ to 10⁶.
- **C_glob** (geometries 0–1 per family): {{CG_MIN}}–{{CG_MAX}}.

## 9 What the pilot does and does not establish

**It establishes, as screening evidence:**
- **The global moment-constrained certificate behaves contrast-robustly** on all sampled cases, consistent with ASTRA's theorem.
- **The lifting-free, operator-adapted localisation realises it on channels at the primary contrasts, and on barriers up to 10².**
- **In the tested ablations**, constructions lacking either ingredient deteriorate over the sampled range.

**It does not establish:**
- contrast-robust localisation on barriers, or unrestricted robustness on channels;
- robustness when the coarse mesh leaves the coefficient unresolved, or general h-robustness;
- any cost advantage (summed LU non-zeros for M3 are about 7× M6's);
- a rigorous floating-point enclosure;
- any population-level claim.

## 10 Decision and next steps

**Proceed decision** (amendment §6, ASTRA §7):

| Condition | Status |
|---|---|
| (1) Matched benefit with Support | Met for channels at the primary contrasts: Q = {{CH_Q}}, M1SZ ≥ 1.5× M3 on {{CH_NOPS}}/8 geometries, M3 maxima ≤ {{CH_M3_MAX_PRIM}}. Not met for barriers |
| (2) Localisation theorem, or justified total cost | Not met |

**Decision: no full programme.**

**Next** (ASTRA §6; no fresh seeds until step 3):
1. **Development study on the inspected pilot failures**, labelled post hoc development. A fixed set: barrier geometries 0, 3 and 5 at 10⁴ and 10⁶; the channel stress outlier (geometry 3, 10⁶); and unaffected controls (barrier geometries 1 and 2, channel geometry 0).
   - Save local component and mode diagnostics: patch-local components, signed weights, energy contributions, and the extremising modes of the local quotient.
   - Save how strongly the realised residuals couple to those modes.
   - Measure the A-orthogonal projection defect of Z onto the unenriched coarse space.
2. **One remedy, chosen with its estimate first.** The priority is a spectrally guided coarse correction. First, derive the residual-space (coarse-orthogonal) estimate it must satisfy. Then test, at a recorded dimension and overlap budget, whether the selected modes reduce the realised localisation loss. A straddle-aware partition is the secondary option, which needs partition of unity, feasible connected patches and controlled multiplier energy.
3. **Only then**, freeze the algorithm, threshold, budget, endpoints and failure rules, and confirm on fresh seeds. The comparison: M3, the remedy, M6, and a geometric control with comparable enrichment and boundary treatment. Include the fixed-H arm: coarse level fixed at L2, fine levels L3 and L4, identical inherited data, and the OA basis on the fixed coarse nodes specified as prolonged or re-optimised.
4. **Stop expanding the route** if step 2 yields neither a credible stability argument nor a practical remedy. In that case, keep the narrower result: the global theorem, plus documented localisation limits.

## 11 Deviations and records

**Deviations:**
- Deviation 1 (reference solves).
- Deviation 2 (goal-interval evaluation, stopping gates).
- The aborted launch and the container restart.
- §5 is exploratory.
- The interim one-family analyses used unmodified copies of the frozen script.

**Records:**
- Runs: `runs/pilot_{channels,barriers}_L{3,4}` (deviation-2 repeat), plus earlier `pilot_*` records.
- Analysis: `pilot_analysis.json`, `pilot_tables.md`.
- Diagnosis: `rc_stop_diag.py` → `stop_diag_barriers_1_1e+06.json`.
- Exploration: `explore_localisation_loss.json`.

## 12 Claims check

`rc_report_check_rev1.py` inserts every number in this revision from the records above. It checks each statement that depends on a number, and fails on any failed check or unfilled placeholder. Its output is in `report_check_rev1.json`.
