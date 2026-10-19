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
- **Result:** all four runs were repeated from the start. **Every gate now passes**, 68,648 in all, of which 8,320 are on the stopping path. The four intervals that failed are now contained (§6).

| Verdict under the declared rules (amendment §6) | Channels | Barriers |
|---|---|---|
| Candidate (M3) | **Support** | **Inconclusive.** E1 and E2 pass; localisation loss E3 fails at 10⁴ |
| Operator adaptation, matched control | **Benefit** (Q = 7.15) | **Benefit** (Q = 15.71) |
| Near-kernel enrichment Z | Uninformative (no near-kernel) | Negligible on the declared endpoint; does not remove the failures |
| Fixed lifting against lifting-free | Lifting-free lower by 24× at 10⁴ | 24× |
| Classical separation (our M0, M0′) | Yes (E2 = 24.1) | Yes (E2 = 24.2) |

**The repeat changed none of the verdicts:**
- **Endpoints:** every E1 and E3 value agrees with the deviation-1 records to 4.5·10⁻⁹.
- **Overall:** "candidate supported" (Support in both families) is not met. **No full programme**: neither a localisation theorem nor a cost case exists. ASTRA agrees.

**What the pilot shows** (screening pilot, 8 fresh geometries per family at L3):
1. **The global certificate (M6) behaved contrast-robustly** on every sampled case: worst energy effectivity 1.41 over all 64 L3 (geometry, contrast) cases.
   - The contrast-independent result is ASTRA's proof for this scalar P0/P1/Alfeld setting.
   - These effectivities, and the computed constants C_glob = 2.07–2.26 (bound 2.8873), are supporting diagnostics, not universal or machine-verified bounds.
2. **The localised candidate M3 matches M6 on channels at the primary contrasts (1, 10², 10⁴)**: median localisation loss ≤ 1.023, maximum 1.177 at 10⁴.
   - Even on channels, robustness is not unrestricted: one geometry loses 4.10 at the 10⁶ stress contrast.
   - **On barriers M3 does not match M6.** Three of 8 geometries lose up to 2.34 at 10⁴, and six of 8 lose up to 20.90 at 10⁶.
3. **In the tested ablations over the sampled contrasts 10²–10⁶**, the constructions lacking one ingredient have finite-range log–log slopes p, with effectivity ∝ c^p:

   | Construction | Lacks | p |
   |---|---|---|
   | M0 (classical) | Both | 0.49–0.50 |
   | M1SZ (geometric partition, lifting-free) | Operator adaptation | 0.41–0.49 |
   | M5 (operator-adapted partition, fixed lifting) | Lifting-free construction | 0.49–0.50 |

   M3 stays flat wherever it does not lose to localisation. These slopes describe the implemented constructions over this range. They are not asymptotic rates, and not a necessity result for all constructions.
4. **The barrier failures are associated with partition members that straddle a barrier** (exploratory, §5). This is an association, not a failure criterion, and not yet a sufficient hypothesis for T12.
5. **The L4 arm is a valid but limited refinement check** (§7). It passes its rule under the declared one-level scaling, H/h = 2. It does not test the regime where the coarse mesh leaves the coefficient unresolved; that needs the fixed-H arm proposed in §10.
6. **Goal intervals (secondary) are useful on many, not all, input–goal pairs** (§6). With the learned primal and a coarse dual at 10⁴, the median half-width is 0.26% of |J_h| on channels and 3.0% on barriers, but the barrier maximum is 31%.
   - The repaired stopping test reaches every tolerance within 320 iterations.
   - M3 needs fewer iterations than M0 at 10⁴, but its summed LU non-zeros per construction are much larger. **No cost advantage is demonstrated.**

## 1 What was run, and integrity

| Run | Records | Gates | All passed | Largest reference-error estimate |
|---|---|---:|---|---:|
| Channels L3 | 8 geometries × 4 contrasts | 32,870 | yes | 9.1·10⁻¹⁶ |
| Barriers L3 | 8 × 4 | 32,870 | yes | 1.2·10⁻¹³ |
| Channels L4 | 2 × 2 | 1,454 | yes | 1.4·10⁻¹⁵ |
| Barriers L4 | 2 × 2 | 1,454 | yes | 2.8·10⁻¹⁴ |

- **Code and analysis.** All four records were produced by one code version (`PILOT_CODE_SHA256_DEVIATION2.txt`). The extended format is recorded in each record's design block (numpy float128, 64-bit mantissa, unit roundoff 5.4·10⁻²⁰, linux). The analysis is the frozen `rc_pilot_analysis.py`, unchanged (SHA-256 `4683bd70…`). No numerical warning was raised. Peak memory was 3.2 GB.
- **Earlier records**, all kept in `runs/` with a `pilot_` prefix:
  - the aborted launch and the container-interrupted run (`runs/pilot_aborted_launch_NOTE.md`);
  - the deviation-1 gate stop;
  - the first complete repeat (`pilot_v0_channels_L3`);
  - the deviation-1 runs on which the first report was based (`pilot_dev1_*`).
- **The deviation-2 repeat against the deviation-1 records:**
  - every bound agrees to within 4.3·10⁻⁷ on resolved cases (2.6·10⁻⁶ including unresolved near-floor cases, which no endpoint uses), relative;
  - the changes come from the extended-precision residuals;
  - no case changes resolution status;
  - the stopping first checks differ in 0 of 1536 (case, method, tolerance) cells.
- **The development expectation was wrong.** The amendment (§11.3) expected "no adaptation benefit" on barriers from one development operator. On every pilot barrier geometry the matched control is at least 1.5× worse at 10⁴.

## 2 Primary endpoints

W is the worst energy effectivity of an operator over the four problems and six common inputs, unresolved cases excluded. E1 is the median of W over the 8 geometries, with the maximum in brackets. E2 = E1(10⁴)/E1(1). E3 is the median / 90th percentile [maximum] over geometries of the per-geometry worst η_m/η_M6. Full tables for all ten methods are in `pilot_tables.md`.

**Channels**

| Method | E1, c = 1 | E1, c = 10² | E1, c = 10⁴ | E2 | E3 at 10⁴ | E1, c = 10⁶ (stress) |
|---|---:|---:|---:|---:|---:|---:|
| M0 (our MVY adaptation) | 1.51 [1.52] | 3.89 [4.05] | 36.57 [38.30] | 24.15 | 27.628 / 31.293 [31.471] | 369.88 [396.09] |
| M1SZ (matched control) | 1.38 [1.38] | 1.49 [1.90] | 9.85 [17.20] | 7.13 | 9.454 / 12.439 [16.552] | 64.98 [108.95] |
| **M3 (candidate)** | 1.34 [1.35] | 1.38 [1.40] | 1.38 [1.41] | 1.03 | 1.023 / 1.092 [1.177] | 1.40 [4.23] |
| M5 (M3 with fixed lifting) | 1.48 [1.48] | 3.58 [3.72] | 32.81 [34.58] | 22.22 | 24.169 / 25.051 [25.104] | 335.21 [357.28] |
| M6 (global optimum) | 1.34 [1.34] | 1.37 [1.40] | 1.38 [1.41] | 1.03 | 1.000 / 1.000 [1.000] | 1.38 [1.40] |

**Barriers**

| Method | E1, c = 1 | E1, c = 10² | E1, c = 10⁴ | E2 | E3 at 10⁴ | E1, c = 10⁶ (stress) |
|---|---:|---:|---:|---:|---:|---:|
| M0 (our MVY adaptation) | 1.52 [1.53] | 3.63 [4.65] | 36.71 [45.66] | 24.16 | 30.472 / 35.720 [39.039] | 367.36 [456.70] |
| M1SZ (matched control) | 1.39 [1.39] | 2.44 [3.60] | 21.61 [30.12] | 15.57 | 17.937 / 23.138 [25.751] | 216.11 [301.20] |
| **M3 (candidate)** | 1.35 [1.35] | 1.36 [1.36] | 1.38 [3.07] | 1.02 | 1.112 / 2.240 [2.343] | 4.53 [27.39] |
| M5 (M3 with fixed lifting) | 1.48 [1.49] | 3.29 [3.89] | 33.01 [35.52] | 22.27 | 26.752 / 29.167 [30.165] | 330.33 [355.66] |
| M6 (global optimum) | 1.34 [1.35] | 1.35 [1.35] | 1.35 [1.36] | 1.00 | 1.000 / 1.000 [1.000] | 1.35 [1.36] |

## 3 Verdicts under the declared rules

| Conclusion (amendment §6) | Channels | Barriers |
|---|---|---|
| Candidate quality (M3) | Support | Inconclusive: E1 ≤ 1.5 and E2 ≤ 1.25 pass; E3 at 10⁴ fails |
| E1(M3) at 1, 10², 10⁴ | 1.34, 1.38, 1.38 | 1.35, 1.36, 1.38 |
| E2(M3) | 1.03 | 1.02 |
| E3(M3) at 10⁴: median / p90 | 1.023 / 1.092 | 1.112 / 2.240 (limits 1.10 / 1.5) |
| Adaptation Q = E1(M1SZ)/E1(M3) at 10⁴ | 7.15: benefit; 7/8 geometries ≥ 1.5× | 15.71: benefit; 8/8 geometries ≥ 1.5× |
| Q at 1, 10², 10⁶ | 1.03, 1.08, 46.37 | 1.03, 1.80, 47.72 |
| Near-kernel: E1(M4)/E1(M3) at 10⁴ | uninformative (all dimensions 0) | 1.000000 |
| Lifting: E1(M5)/E1(M3) at 10⁴ | 23.80 | 24.00 |
| Classical separation: E2(M0), E2(M0′) | 24.1, 24.2 | 24.2, 24.2 |
| Stress 10⁶: E1(M3) median [max] | 1.40 [4.23] | 4.53 [27.39] |

**Near-kernel (corrected).** The tested Z enrichment has a negligible effect on the declared worst-case median endpoints, and it does not remove the observed failures. Some individual residuals improve modestly. Across all barrier records:
- the largest per-residual ratio η_M4/η_M3 is 1.0139, at geometry 1, contrast 10⁶, primal|coarse;
- the largest η_M1S/η_M1SZ is 1.0631, at geometry 2, contrast 10⁴, dual1|pcg30.

The first report suggested that the one-level OA functions already carry the near-kernel modes. That was untested and is withdrawn. The development step will measure the A-orthogonal projection defect of Z onto the unenriched coarse space.

**Other points:**
- **Superseded unmatched rule (note §9).** At 10⁴, E1(M1)/E1(M3) and E1(M2)/E1(M3) are 11.37 and 13.42 on channels and 16.39 and 14.41 on barriers.
- **OA-VCB inputs** (learned one-pass primal, 10⁴, reported separately). M3 has W between 1.05 and 1.89. M0 has 9.2–33.5.

## 4 Barrier localisation failure

| Geometry | Near-kernel dim (10⁴) | W(M3) at 10⁴ | λ(M3) at 10⁴ | W(M3) at 10⁶ | λ(M3) at 10⁶ | Worst OAZ index at 10⁴ | High-index members that straddle (exploratory) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3.07 | 2.34 | 27.39 | 20.90 | 14.34 | 2 of 2 |
| 1 | 2 | 1.35 | 1.06 | 1.36 | 1.05 | 3.33 | 1 of 1 |
| 2 | 1 | 1.36 | 1.01 | 1.36 | 1.01 | 1.39 | 0 of 0 |
| 3 | 0 | 2.21 | 1.86 | 18.37 | 15.41 | 10.13 | 8 of 9 |
| 4 | 0 | 1.35 | 1.03 | 2.79 | 2.30 | 4.90 | 1 of 1 |
| 5 | 0 | 2.57 | 2.20 | 22.05 | 18.85 | 9.98 | 4 of 4 |
| 6 | 0 | 1.39 | 1.16 | 6.27 | 5.24 | 10.23 | 2 of 2 |
| 7 | 0 | 1.35 | 1.02 | 2.59 | 2.18 | 2.82 | 0 of 0 |

- **This is a localisation loss.** M6 stays at most 1.36 on every barrier case.
- **Growth.** The three geometries affected at 10⁴ (0, 3, 5) grow by a factor of 8.3–8.9 from 10⁴ to 10⁶, a finite-range observation.
- **The index tracks the loss.** The four geometries with the largest OAZ robustness index at 10⁴ (0, 3, 5, 6) have the four largest losses at 10⁶ (20.90, 15.41, 18.85, 5.24).
- **The two geometries with a detected near-kernel are not affected.**

## 5 The straddling association (exploratory; no verdict depends on it)

`rc_explore_localisation_loss.py` labels the edge-connected components of the K = 1 region globally. It counts a member as carrying a component if max θ ≥ 0.05 there. At barriers, 10⁴:

- **High-index members:** 18 of the 19 members with index ≥ 3 carry weight on two or more K = 1 components separated by a barrier. One does not.
- **The association is not specific.** 40 members straddle in total, so 22 straddling members have index < 3.
- **The inventory is partial.** The script saves only the five worst members per geometry, and none of those carries an enclosed island.

**Why this is an association, not a criterion** (following ASTRA §5):
- The labels are **global**. A region that is connected globally can still split into several components inside one patch, which is what the local quotient sees.
- The carrying threshold is diagnostic. With signed OA functions it can miss negative weight.
- A large index is a supremum over test functions. It does not show that the realised residual excites the bad direction.
- Component counts do not control variation along a component, bottlenecks, overlap, or shape constants.
- The single channel stress outlier (λ = 4.10) has its worst member, index 54.64, inside one channel component.

**"No straddling" is checkable, so it is not circular. It is not sufficient as formulated.** Two remedy mechanisms must also be kept apart:
- **Enlarging the coarse space** with the partition fixed cannot reduce the raw index C_θ. It can only remove dangerous modes from the admissible residual space, so a proof needs a bound on a restricted, coarse-orthogonal space that the selected modes control.
- **Changing the partition** changes the quotient itself.

## 6 Goal intervals and stopping (secondary)

**Learned primal with coarse dual, and PCG30 on both sides, at 10⁴.** Polarization half-width / |J_h|, M3; 24 geometry–goal pairs per family (not 24 independent geometries):

| Family | Input pair | Median | p90 | Maximum | ≤ 1% | ≤ 5% |
|---|---|---:|---:|---:|---:|---:|
| channels | learned / coarse | 0.26% | 1.1% | 2.2% | 20/24 | 24/24 |
| channels | PCG30 / PCG30 | 0.049% | 0.34% | 1.1% | 23/24 | 24/24 |
| barriers | learned / coarse | 3.0% | 8.6% | 31% | 5/24 | 16/24 |
| barriers | PCG30 / PCG30 | 0.87% | 2.0% | 22% | 14/24 | 22/24 |

**Medians by method and input pair** (median product efficiency in brackets):

| Family | Method | Coarse / coarse | PCG30 / PCG30 | OA-VCB / coarse |
|---|---|---:|---:|---:|
| channels | M0 | 215% (4.90) | 29% (347.33) | 11% (38.05) |
| channels | M3 | 54% (1.32) | 0.049% (1.10) | 0.26% (1.29) |
| channels | M6 | 52% (1.28) | 0.048% (1.09) | 0.26% (1.26) |
| barriers | M0 | 26 390% (99.03) | 925% (429.90) | 221% (67.66) |
| barriers | M3 | 279% (1.55) | 0.87% (1.10) | 3.0% (1.29) |
| barriers | M6 | 213% (1.35) | 0.67% (1.07) | 2.8% (1.25) |

**Reading:**
- The intervals are useful on many of these input–goal pairs. A favourable median is not uniform usefulness.
- These width statistics are descriptive. They are separate from amendment §7's tolerance-attainment definition.

**Repaired evaluation** (deviation 2). In the barrier stress case that failed before (geometry 1, 10⁶, iteration 320):

| Method | Old centre miss (double) | New rem = J_ref − ĉ | Certified radius | τ_eval | Total radius | τ_ref | Product contained | Polarization contained |
|---|---:|---:|---:|---:|---:|---:|---|---|
| M0 | 6.6·10⁻¹⁰ | -1.3·10⁻¹² | 5.1·10⁻⁸ | 1.8·10⁻¹⁰ | 5.1·10⁻⁸ | 1.8·10⁻¹⁰ | yes | yes |
| M3 | 6.6·10⁻¹⁰ | -1.3·10⁻¹² | 1.5·10⁻¹² | 1.8·10⁻¹⁰ | 1.9·10⁻¹⁰ | 1.8·10⁻¹⁰ | yes | yes |
| M6 | 6.6·10⁻¹⁰ | -1.3·10⁻¹² | 1.4·10⁻¹² | 1.8·10⁻¹⁰ | 1.9·10⁻¹⁰ | 1.8·10⁻¹⁰ | yes | yes |

- **Evaluation terms.** On the ordinary goal intervals, the rigorous centre-evaluation bound τ_eval is at most 3.0·10⁻⁴ of the certified radius.
- **Converged stopping iterates.** On the stopping rows τ_eval exceeds the certified radius in 191 of 1344 rows, all at iterations ≥ 320, where the certified radius has fallen to round-off level. There the total radius is set by the evaluation bound; its largest value is 2.9·10⁻¹⁰.
- **The bound is pessimistic.** In the case above, τ_eval = 1.8·10⁻¹⁰, while the actual discrepancy is 1.3·10⁻¹².
- **The empirical allowance.** The old allowance test, still recorded but no longer a gate, would now fail on 0 stopping rows and 0 ordinary goal intervals.
- **What is estimated.** The second-order terms a_u and a_z, and the reference uncertainty τ_ref, are estimates (deviation 2). No rigorous floating-point enclosure of the whole computation is claimed.

**Deployable stopping test** (goal 0, Jacobi-PCG). Median first check reaching the tolerance, product / polarization, using total radii:

| Family | c | ε | M0 | M3 | M6 |
|---|---:|---:|---:|---:|---:|
| channels | 1 | 10⁻³ | 80 / 40 | 80 / 40 | 80 / 40 |
| channels | 1 | 10⁻⁶ | 80 / 80 | 80 / 80 | 80 / 80 |
| channels | 10⁴ | 10⁻³ | 160 / 160 | 80 / 40 | 80 / 40 |
| channels | 10⁴ | 10⁻⁶ | 160 / 160 | 160 / 160 | 160 / 160 |
| channels | 10⁶ | 10⁻³ | 160 / 160 | 40 / 30 | 40 / 30 |
| channels | 10⁶ | 10⁻⁶ | 160 / 160 | 160 / 80 | 160 / 80 |
| barriers | 1 | 10⁻³ | 80 / 60 | 80 / 40 | 80 / 40 |
| barriers | 1 | 10⁻⁶ | 80 / 80 | 80 / 80 | 80 / 80 |
| barriers | 10⁴ | 10⁻³ | 160 / 160 | 80 / 80 | 80 / 80 |
| barriers | 10⁴ | 10⁻⁶ | 160 / 160 | 160 / 160 | 160 / 160 |
| barriers | 10⁶ | 10⁻³ | 160 / 160 | 80 / 80 | 80 / 80 |
| barriers | 10⁶ | 10⁻⁶ | 160 / 160 | 160 / 160 | 160 / 160 |

**Reading:**
- Every cell attains every tolerance within 320 iterations, and every stop was triggered by the relative criterion.
- At c = 1 the three certificates stop within one check of each other.
- At 10⁴ and 10⁶, M3 and M6 reach ε = 10⁻³ earlier than M0 (medians 30–80 against 160 iterations).
- **This is an iteration comparison only.** Median summed LU non-zeros per construction are M3 61·10⁶, M6 8.2·10⁶ and M0 2.5·10⁶. That measures factor storage, a work proxy; it is not a FLOP or wall-time ratio.

## 7 Mesh-size arm (L4)

| Family | Geometry | c | Method | W at L3 | W at L4 | Ratio |
|---|---:|---:|---|---:|---:|---:|
| channels | 0 | 1 | M0 | 1.52 | 1.51 | 1.00 |
| channels | 0 | 1 | M1SZ | 1.38 | 1.38 | 1.00 |
| channels | 0 | 1 | M3 | 1.35 | 1.34 | 1.00 |
| channels | 0 | 1 | M6 | 1.34 | 1.34 | 1.00 |
| channels | 1 | 1 | M0 | 1.51 | 1.51 | 1.00 |
| channels | 1 | 1 | M1SZ | 1.38 | 1.38 | 1.00 |
| channels | 1 | 1 | M3 | 1.35 | 1.34 | 1.00 |
| channels | 1 | 1 | M6 | 1.34 | 1.34 | 1.00 |
| channels | 0 | 10000 | M0 | 37.00 | 32.41 | 0.88 |
| channels | 0 | 10000 | M1SZ | 10.09 | 1.39 | 0.14 |
| channels | 0 | 10000 | M3 | 1.38 | 1.36 | 0.98 |
| channels | 0 | 10000 | M6 | 1.38 | 1.36 | 0.98 |
| channels | 1 | 10000 | M0 | 37.79 | 29.68 | 0.79 |
| channels | 1 | 10000 | M1SZ | 17.20 | 1.39 | 0.08 |
| channels | 1 | 10000 | M3 | 1.37 | 1.36 | 0.99 |
| channels | 1 | 10000 | M6 | 1.37 | 1.36 | 0.99 |
| barriers | 0 | 1 | M0 | 1.51 | 1.52 | 1.00 |
| barriers | 0 | 1 | M1SZ | 1.38 | 1.38 | 1.00 |
| barriers | 0 | 1 | M3 | 1.34 | 1.34 | 1.00 |
| barriers | 0 | 1 | M6 | 1.34 | 1.34 | 1.00 |
| barriers | 1 | 1 | M0 | 1.52 | 1.51 | 0.99 |
| barriers | 1 | 1 | M1SZ | 1.39 | 1.37 | 0.99 |
| barriers | 1 | 1 | M3 | 1.35 | 1.34 | 0.99 |
| barriers | 1 | 1 | M6 | 1.35 | 1.34 | 1.00 |
| barriers | 0 | 10000 | M0 | 39.54 | 34.88 | 0.88 |
| barriers | 0 | 10000 | M1SZ | 28.83 | 1.37 | 0.05 |
| barriers | 0 | 10000 | M3 | 3.07 | 1.34 | 0.44 |
| barriers | 0 | 10000 | M6 | 1.35 | 1.34 | 1.00 |
| barriers | 1 | 10000 | M0 | 35.60 | 38.07 | 1.07 |
| barriers | 1 | 10000 | M1SZ | 13.01 | 1.38 | 0.11 |
| barriers | 1 | 10000 | M3 | 1.35 | 1.34 | 0.99 |
| barriers | 1 | 10000 | M6 | 1.35 | 1.34 | 0.99 |

**Reading rule (declared):** W(M3, L4)/W(M3, L3) ≤ 1.25 for each (geometry, contrast). It is **met**: the largest ratio is 1.00.

**What this shows.** The arm is a valid, limited check: it refines the same inherited PDE data under the declared one-level coarse scaling (H/h = 2, with a growing coarse space), and M3 shows no gross deterioration on two geometries.

**What it does not show:**
- **Robustness when the coarse mesh leaves the coefficient unresolved.** At L4 the coarse mesh (L3) is the mesh on which the inherited coefficient is defined. The matched control M1SZ drops from 28.83 to 1.37, and M3's barrier loss from 3.07 to 1.34 (barrier geometry 0, 10⁴). This is consistent with coefficient alignment, but the comparison cannot separate alignment from changes in resolution, support or coarse dimension.
- **General h-robustness.** Two geometries cannot establish it.

The first report called the arm unable to test what it was meant to. That overstated the limitation, and is corrected here.

## 8 Indices and the global constant

- **OAZ robustness index at 10⁴** (median over geometries of the per-geometry maximum): 1.65 on channels and 7.44 on barriers. The maxima are 5.47 and 14.34.
- **Runner G1:** medians 40.64 and 45.36.
- **OAZ p95:** 0.96–1.12. The largest values grow by ×10.0 (channels) and ×10.0 (barriers) from 10⁴ to 10⁶.
- **C_glob** (geometries 0–1 per family): 2.07–2.26.

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
| (1) Matched benefit with Support | Met for channels at the primary contrasts: Q = 7.15, M1SZ ≥ 1.5× M3 on 7/8 geometries, M3 maxima ≤ 1.41. Not met for barriers |
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
