# Route C pilot: report

18 October 2026. Claude, for Kumar and ASTRA. The pilot follows `ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md` (SHA-256 `e83c0b06…`) and its `ROUTE_C_PILOT_DEVIATION_1_2026-10-18.md`. Every number below is inserted from the raw run records by `rc_report_check.py`, which also checks the statements that depend on them (§12).

## Summary

| Verdict under the declared rules (amendment §6) | Channels | Barriers |
|---|---|---|
| Candidate (M3) | **Support** | **Inconclusive.** E1 and E2 pass; localisation loss E3 fails at 10⁴ |
| Operator adaptation, matched control | **Benefit** (Q = 7.15) | **Benefit** (Q = 15.71) |
| Near-kernel enrichment | Uninformative (no near-kernel) | No effect detected |
| Fixed lifting against lifting-free | Lifting-free better by 24× at 10⁴ | 24× |
| Classical separation (our M0) | Yes (E2 = 24.1) | Yes (E2 = 24.2) |

**"Candidate supported" in the note's sense (Support in both families) is not met.** Following ASTRA §7, the pilot does not justify a full programme yet. It does justify the separately declared follow-up that the amendment reserved for this outcome (§10).

**What the pilot shows:**
1. **The global certificate (M6) is contrast-robust.** Its worst energy effectivity over all 64 L3 (geometry, contrast) cases is 1.41. The computed global constants (2 geometries per family) are 2.07–2.26, below the proved bound 1/β_* = 2.8873.
2. **The localised candidate M3 matches M6 on channels at every primary contrast**: median localisation loss ≤ 1.023, maximum 1.177 at 10⁴. **On barriers it does not.** Three of 8 geometries lose up to 2.34 at 10⁴, and six of 8 lose up to 20.90 at the 10⁶ stress contrast.
3. **Both ingredients are needed in this regime.** Every construction that lacks one of them grows like c^p:

   | Construction | Lacks | p |
   |---|---|---|
   | M0 (classical) | Both | 0.49–0.50 |
   | M1SZ (geometric partition, lifting-free) | Operator adaptation | 0.41–0.49 |
   | M5 (operator-adapted partition, fixed lifting) | Lifting-free construction | 0.49–0.50 |

   M3, which has both, stays flat wherever it does not lose to localisation.
4. **The barrier failures sit on partition members that straddle a barrier** (exploratory, §5). This is a concrete target for the follow-up and for the open localisation theorem (T12).
5. **The L4 arm passes its reading rule but cannot test what it was meant to** (§7). With one-level coarsening, the L4 coarse mesh is the mesh on which the coefficient is defined, so even the geometric control becomes robust there.
6. **Goal intervals (secondary) are useful when the inputs are good.** For the learned primal with a coarse dual at 10⁴, the median polarization half-width is 0.26% of |J_h| on channels and 3.0% on barriers. With coarse inputs on both sides they remain uninformative. The stopping test reaches every tolerance within 320 iterations. At 10⁴, M3 reaches ε = 10⁻³ earlier than M0 (product interval: median 80 against 160 iterations in both families), but each M3 evaluation costs far more (§6). **No cost advantage is claimed.**

## 1 What was run, and integrity

| Run | Records | Gates | All passed | Largest reference error |
|---|---|---:|---|---:|
| Channels L3 | 8 geometries × 4 contrasts | 28,644 | yes | 9.1·10⁻¹⁶ |
| Barriers L3 | 8 × 4 | 28,644 | yes | 1.2·10⁻¹³ |
| Channels L4 | 2 × 2 | 1,444 | yes | 1.4·10⁻¹⁵ |
| Barriers L4 | 2 × 2 | 1,444 | yes | 2.8·10⁻¹⁴ |

- **Code.** All four records were produced by one code version, with hashes in `PILOT_CODE_SHA256_DEVIATION1.txt`. The analysis is the frozen `rc_pilot_analysis.py`, unchanged (SHA-256 `4683bd70…`). No numerical warning was raised. Peak memory was 3.2 GB.
- **Three interruptions, all documented.** Each record is kept in `runs/` with a `pilot_` prefix:
  1. A launch was stopped after about a minute, before any outcome, because of a scheduling limit.
  2. A container restart interrupted barriers L3.
  3. Barriers L3 stopped at the reference-solve gate at 10⁶. This led to **deviation 1**: the reference solves are refined in extended precision, and the gate tests the reference's own energy error, which must be ≤ 10⁻¹² against the 10⁻¹⁰ near-exact floor.
- **Reproducibility.** All runs were repeated from the start after deviation 1.
  - The repeated channels run reproduces the first one to 5.5·10⁻¹³ relative in every bound. No case changes resolution status.
  - The barrier results completed before the stop reproduce to 3.7·10⁻¹².
- **The development expectation was wrong.** The amendment (§11.3) stated, before the pilot, that barriers would probably show "no adaptation benefit", because the matched control nearly tied with M3 on the development operator. That did not reproduce: on every pilot barrier geometry the matched control is at least 1.5× worse at 10⁴. The development operator was not representative.

## 2 Primary endpoints

W is the worst energy effectivity of an operator over the four problems and six common inputs, unresolved cases excluded. E1 is the median of W over the 8 geometries, with the maximum in brackets. E2 = E1(10⁴)/E1(1). E3 is the median / 90th percentile [maximum] over geometries of the per-geometry worst localisation loss η_m/η_M6. Full tables for all ten methods are in `pilot_tables.md`.

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

**Further points:**
- **Near-kernel.** Barrier geometries 1 and 2 have near-kernel dimensions 2 and 1 at 10⁴, so the comparison is informative. Yet M3 and M4 agree to 3.4·10⁻⁸, and M1S and M1SZ agree to 2.3·10⁻⁴. In this pilot Z changed nothing measurable, presumably because the one-level OA functions already carry those modes. That is a reading, not a check.
- **Superseded unmatched rule (note §9).** At 10⁴, E1(M1)/E1(M3) and E1(M2)/E1(M3) are 11.37 and 13.42 on channels and 16.39 and 14.41 on barriers, so the old rule would also have read "adaptation matters".
- **OA-VCB inputs** (learned one-pass primal, 10⁴, reported separately). M3 has W between 1.05 and 1.89 across both families. M0 has 9.2–33.5.

## 4 Barrier localisation failure

| Geometry | Near-kernel dim (10⁴) | W(M3) at 10⁴ | λ(M3) at 10⁴ | W(M3) at 10⁶ | λ(M3) at 10⁶ | Worst OAZ index at 10⁴ | Members with index ≥ 3 that straddle a barrier (exploratory) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 0 | 3.07 | 2.34 | 27.39 | 20.90 | 14.34 | 2 of 2 |
| 1 | 2 | 1.35 | 1.06 | 1.36 | 1.05 | 3.33 | 1 of 1 |
| 2 | 1 | 1.36 | 1.01 | 1.36 | 1.01 | 1.39 | 0 of 0 |
| 3 | 0 | 2.21 | 1.86 | 18.37 | 15.41 | 10.13 | 8 of 9 |
| 4 | 0 | 1.35 | 1.03 | 2.79 | 2.30 | 4.90 | 1 of 1 |
| 5 | 0 | 2.57 | 2.20 | 22.05 | 18.85 | 9.98 | 4 of 4 |
| 6 | 0 | 1.39 | 1.16 | 6.27 | 5.24 | 10.23 | 2 of 2 |
| 7 | 0 | 1.35 | 1.02 | 2.59 | 2.18 | 2.82 | 0 of 0 |

- **This is a localisation loss.** M6 stays at most 1.36 on every barrier case, so the certificate itself is not the limit.
- **Growth on the affected geometries is close to √c.** The three geometries affected at 10⁴ (0, 3, 5) grow by a factor of 8.3–8.9 from 10⁴ to 10⁶, against √100 = 10. Geometry 0 goes from 3.07 to 27.39.
- **The index tracks the loss.** In the recorded diagnostics, the largest OAZ robustness index per geometry orders the geometries much as the loss does. The four geometries with the largest index at 10⁴ (0, 3, 5, 6) have the four largest losses at 10⁶ (20.90, 15.41, 18.85, 5.24).
- **The two geometries with a detected near-kernel are not affected.**

## 5 Where the loss comes from (exploratory; after the pilot; no verdict depends on it)

`rc_explore_localisation_loss.py` rebuilds the OAZ partition exactly as the runner does. It labels the edge-connected components of the normal-conductivity (K = 1) region, and records, for each member, how many of them it carries weight on (max θ ≥ 0.05).

**Barriers at 10⁴:**
- **The high-index members straddle barriers.** Of the 19 members with index ≥ 3 across the 8 geometries, 18 carry weight on two or more K = 1 components separated by a barrier.
- **None of those components is an enclosed island**, at least for the five worst members of every geometry, which are all the script saves. Each reaches the Dirichlet boundary somewhere, so no near-kernel mode exists for Z to capture.
- **The enclosed islands present are tiny** (9 of 12 islands have one element; the largest have 18, 45 elements).

So the mechanism is local. My interpretation: a member that is not constant on each component it touches lets the PU-weighted Poincaré quotient pick up a factor of order √c. The function q may take different constants on two components at an energy cost of only O(1/c) in the barrier, while θ_i(q − const) still has O(1) energy on the components. That is the Graham–Lechner–Scheichl mechanism, transposed from channels to the components that barriers separate. It is also the situation a T12 proof would have to exclude or treat.

**Channels.** The single stress outlier (geometry 3 at 10⁶, λ = 4.10) is not explained by this statistic: its worst member (index 54.64) lies within one channel component. That fits the channel version of the same mechanism, a member that varies along a channel, but I did not examine it further. On channels the index is a weaker predictor. For example, geometry 0 has index 41.68 at 10⁶ but λ = 1.17, because the index is a supremum over all functions, while λ depends on the residuals that actually occur.

**What this suggests for the follow-up:**
- a **straddle-aware partition**, splitting a member along the barrier it straddles, with the cut placed inside the low-conductivity layer; or
- **index-guided local enrichment** of those members (GenEO-type).

Either should be compared with the matched control and with M6 on fresh geometries.

## 6 Goal intervals and stopping (secondary)

**Median polarization half-width / |J_h|** over geometries × goals at 10⁴, with the median product efficiency in brackets:

| Family | Method | Coarse / coarse | PCG30 / PCG30 | OA-VCB / coarse |
|---|---|---:|---:|---:|
| channels | M0 | 215% (4.90) | 29% (347.33) | 11% (38.05) |
| channels | M3 | 54% (1.32) | 0.049% (1.10) | 0.26% (1.29) |
| channels | M6 | 52% (1.28) | 0.048% (1.09) | 0.26% (1.26) |
| barriers | M0 | 26 390% (99.03) | 925% (429.90) | 221% (67.66) |
| barriers | M3 | 279% (1.55) | 0.87% (1.10) | 3.0% (1.29) |
| barriers | M6 | 213% (1.35) | 0.67% (1.07) | 2.8% (1.25) |

**Reading:**
- With good inputs the intervals are informative. With coarse inputs on both sides they remain wide, especially on barriers. This is consistent with ASTRA §1: the input quality, not the certificate, sets the width.
- The development polarization numbers (amendment §11.2) used a contrast-dependent barrier load, so they do not carry over.

**Deployable stopping test** (goal 0, Jacobi-PCG). Median first check reaching the tolerance, product / polarization:

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
- At 10⁴ and 10⁶, M3 and M6 reach ε = 10⁻³ earlier than M0 (medians 30–80 against 160 iterations). At ε = 10⁻⁶ the difference largely disappears.
- **Each M3 evaluation costs more than M0's.** Median summed LU non-zeros per construction: M3 61·10⁶, M6 8.2·10⁶, M0 2.5·10⁶.
- So this is an iteration comparison only, and per ASTRA §7 it is not evidence of a cost advantage.

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

**But the arm cannot test what it was meant to.** With one-level coarsening, the L4 coarse space lives on the L3 mesh, which is exactly the mesh on which the inherited coefficient is defined.

- **Even the geometric control is robust at L4.** The matched control M1SZ drops from 28.83 at L3 to 1.37 at L4 (barrier geometry 0, 10⁴).
- **M3's barrier loss vanishes too.** It drops from 3.07 to 1.34 on the same case.

So the L4 arm probes a coefficient-resolving coarse space, not h-dependence in the regime that matters, where the coarse space does not resolve the coefficient. The amendment documented the coarse growth, but it did not foresee this. A proper refinement arm must keep the coefficient's length scale below the coarse mesh size at both levels: for example, define the geometry at the finer level, or fix H while h decreases.

## 8 Indices and the global constant

- **Robustness index of the OAZ partition at 10⁴ (median over geometries of the per-geometry maximum):** 1.65 on channels and 7.44 on barriers. The maxima are 5.47 and 14.34.
- **The geometric partitions are much larger:** runner G1 has medians 40.64 and 45.36.
- **The p95 of the OAZ index stays near one** (0.96–1.12), as in the scoping note. The largest values grow like √c: ×10.0 on channels and ×10.0 on barriers from 10⁴ to 10⁶.
- **The global constant C_glob** (geometries 0–1 per family) ranges over 2.07–2.26 across contrasts. This is consistent with the proved contrast-independent bound.

## 9 What the pilot does and does not establish

**It establishes**, on 8 fresh geometries per family at L3 (a screening pilot):
- **The global moment-constrained certificate is contrast-robust in practice**, as ASTRA's theorem predicts.
- **The lifting-free, operator-adapted localisation realises it on channels at every primary contrast, and on barriers up to 10².**
- **In this coarse-space regime, both ingredients are needed:** constructions missing either one grow like c^{0.4–0.5}.

**It does not establish:**
- **contrast-robust localisation on barriers.** Barrier-straddling members break it, so robust localisation cannot be claimed in general;
- **h-robustness**, which the L4 arm could not test (§7);
- **any cost advantage.** At L3, M3's local factorisations are about 7× the global optimum's;
- **any population-level claim.**

## 10 Decision and proposed next steps

**Proceed decision** (amendment §6, following ASTRA §7). Two conditions must both hold:

| Condition | Status |
|---|---|
| (1) Matched localisation benefit with Support, reproduced across a family's geometries | Met for channels: Q = 7.15, with M1SZ ≥ 1.5× M3 on 7/8 geometries and M3's maxima ≤ 1.41 at primary contrasts. Not met for barriers (Inconclusive) |
| (2) A credible localisation theorem, or useful certification at justified total cost | Not met. T12 is open, and §5 shows which hypothesis it needs. No cost case exists at L3 |

**Decision: do not start a full programme yet.**

**Proposed** (for ASTRA's review before anything is run):
1. **A declared follow-up on fresh seeds**, frozen before generation. Compare M3, a straddle-aware variant (split members) and an index-guided enriched variant against M1SZ and M6, with the same endpoints and rules.
2. **T12 under an explicit hypothesis**: no member carries weight on two components separated by a low-conductivity layer, or every such member is handled by enrichment. Try to prove the localisation bound under that hypothesis, using ASTRA's global proof as the outer layer.
3. **A redesigned refinement arm** in which the coarse mesh does not resolve the coefficient at either level.
4. **Cost and scaling** only after 1–2 succeed, with the comparators ASTRA listed.

**Questions for ASTRA:**
1. Do you accept the classifications above (channels Support; barriers Inconclusive; overall not supported)?
2. Is the follow-up in 1–3 the right next step, or should the route stop here with the narrower result (global certificate robust; localisation robust on channels only)?
3. Is the straddle hypothesis a reasonable theorem hypothesis, or too close to restating the conclusion?

## 11 Deviations and records

**Deviations:**
- Deviation 1 (reference solves).
- The aborted launch and the container restart (`runs/pilot_aborted_launch_NOTE.md`).
- §5 is exploratory and was added after the pilot.
- The interim channels-only and barriers-only analyses were run with unmodified copies of the frozen script restricted to one family. This was needed because the script fails on a record that has no completed operator yet. The final analysis ran the script itself.

**Records:**

| Kind | Files |
|---|---|
| Runs | `runs/pilot_{channels,barriers}_L{3,4}.json` / `.log`, plus earlier `pilot_*` records |
| Analysis | `pilot_analysis.json`, `pilot_tables.md` |
| Exploration | `explore_localisation_loss.json`, `logs_explore_localisation_*.log` |
| Deviation 1 diagnostics | `exact_gate_diag.json`, `logs_exact_refine_proto.log` |

## 12 Claims check

`rc_report_check.py` inserts every number in this report directly from the records named above. It also checks each statement that depends on a number, for example:
- M6 bounded by 1.41 on every case;
- C_glob below 2.8873;
- the 3/8 and 6/8 counts;
- the 18-of-19 straddle count;
- the L4 drops.

It fails if any check fails or a placeholder remains. Its output is in `report_check.json`.
