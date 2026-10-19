# Route C pilot: deviation 2 (goal-interval evaluation and stopping gates)

18 October 2026. Claude, for Kumar and ASTRA. A dated note under amendment 1, §8, responding to `ASTRA_ROUTE_C_PILOT_REVIEW_2026-10-18.md` §2. Unchanged: the amendment, the endpoints, the decision rules, the frozen analysis script, the seeds, and the flux constructions M0–M6.

## What was wrong

**Missing gates.** The runner gated the goal intervals of the ordinary inputs, but not the deployable stopping test. That test calls the flux constructions again and records intervals and first stops with no moment, defect, guarantee or containment check.

**Containment failures.** ASTRA's audit of the 2,688 saved stopping intervals found four that miss the reference value by more than the declared allowance. All four are at barriers, geometry 1, contrast 10⁶ (stress), iteration 320: M3 and M6, product and polarization. They miss by 6.6·10⁻¹⁰, with radii of 1.4·10⁻¹² to 2.5·10⁻¹³.

## Diagnosis (`rc_stop_diag.py` → `stop_diag_barriers_1_1e+06.json`)

This was a targeted reproduction of that one case: same geometry, coefficients, loads, reference solves and Jacobi-PCG iterates. No new geometry was generated, and no outcome other than this case was computed.

| Candidate cause | Finding at iteration 320 |
|---|---|
| Reference value J_h | Not the cause. The runner's value agrees to 1.3·10⁻¹² with the corrected value J = G·u + r(u)·z accumulated in extended precision. That value comes from two independent LU factorisations (different column orderings), each refined, which agree with each other to 2.8·10⁻¹³ |
| Certificate radius | Not the cause. The true error product is \|\|\|e_u\|\|\| \|\|\|e_z\|\|\| = 1.24·10⁻¹². The M6 radius is 1.42·10⁻¹² and the M3 radius 1.47·10⁻¹², whether built from the double-precision or the extended-precision residual |
| **Centre ĉ = G·v + (F − A v)·w in double precision** | **The cause.** It differs from the extended-precision value by −6.66·10⁻¹⁰, which matches the observed miss. The residual F − A v carries a double-precision rounding error of up to 3.4·10⁻¹³ per component (against \|r\| ≈ 4·10⁻⁷), and the dual iterate w is large here (\|\|\|z_h\|\|\| = 98, inside barrier-enclosed regions). With the extended-precision centre, \|J − ĉ\| = 1.34·10⁻¹², inside the certificate radius |

**Disposition.** This is a genuine implementation defect in how the goal intervals were evaluated: the evaluation error of the centre was not accounted for. It affects all methods alike (M6 as well as M3). It is not a defect of the check or of the reference, and not a defect of the energy certificate. The energy endpoints never use the centre.

Under amendment §6 the pilot's overall status is therefore **Stop until repaired**. The numerical endpoint classifications are kept as numerical findings, pending the repaired repeat.

## Change (`rc_pilot.py`; pre-change copy kept as `rc_pilot_pre_deviation2.py`)

1. **Extended-precision evaluation.**
   - Every residual b − A v, for ordinary inputs and stopping iterates, is accumulated in extended precision. It is then rounded to double for the flux reconstruction.
   - Every goal centre is accumulated in extended precision.
   - The format is recorded in the design block, and a run-level gate requires a mantissa of at least 64 bits. NumPy `longdouble` is platform-dependent: here it is x86-64 Linux 80-bit, unit roundoff 5.4·10⁻²⁰. On Windows it equals double, and there the runner refuses to run.
2. **Explicit evaluation terms**, recorded separately from the certified radius:
   - **τ_eval:** an a priori bound on the evaluation error of the centre, with Higham's γ_n at the recorded precision. It covers the residual evaluation (componentwise γ_{k+1}(\|b\| + \|A\|\|v\|)) times \|w\|, and both dot products. It is **rigorous for IEEE extended arithmetic, given the stored vectors**.
   - **a_u, a_z (estimates):** the energy norm of A⁻¹(r_exact − r_double). The known rounding part is obtained exactly with one solve. The remaining part is bounded via an eigsh estimate of λ_min(A), which is not a verified bound. These terms are second order in the goal interval.
   - **Product interval:** ĉ ± [(η_u + a_u)(η_z + a_z)(1 + 10⁻⁹) + τ_eval].
   - **Polarization interval:** U± gains s·a_u + a_z/s, both ends are widened by τ_eval, and the 10⁻⁹ slack is now applied outward on both ends. The amendment's form had narrowed the lower end when it was positive.
3. **Polarization lower bound in Cauchy–Schwarz form.** L = max(0, R(ρ) − τ_R)/\|\|\|ρ\|\|\|, with R(ρ) evaluated in extended precision and its rigorous evaluation bound τ_R subtracted. This is valid for any ρ, not only an exact Galerkin projection, and equals \|\|\|ρ\|\|\| when ρ is the exact projection. The projection is still validated by a Galerkin-residual gate (≤ 10⁻⁸, scaled by \|C\|ᵀ\|R\|).
4. **Reference for the gates.** J_ref = G·u + r(u)·z, accumulated in extended precision from the refined references. It carries τ_ref: a rigorous evaluation bound plus the second-order estimate (ref_err_u \|\|\|u\|\|\|)(ref_err_z \|\|\|z\|\|\|). The empirical allowance 10³·eps·(\|J\| + \|ĉ\|) is no longer a gate; its check is still recorded (`legacy_check`) for comparison.
5. **The stopping path gets the same gates as the ordinary path:**
   - finite, non-negative bounds;
   - full moment constraints;
   - coarse Galerkin and patch defects;
   - the guarantee (η + a)(1 + 10⁻⁹) ≥ \|\|\|e\|\|\| − (reference-error estimate), for primal and dual at every check;
   - the Galerkin residual of the lower-bound projection;
   - product and polarization containment of J_ref within τ_ref.

   All terms are saved with each row.
6. **Stopping rule:** it uses the total radius, certified plus evaluation terms, in both r ≤ ε/(1+ε)\|ĉ\| and r ≤ τ_abs.
7. **The ordinary-input guarantee gate** takes the same form as in 5.

**What is rigorous and what is estimated** (to be stated in the report):
- **Rigorous** in exact arithmetic on the flux side, given the computed residual: the certified radius. The flux solves themselves are gated by their moment and defect checks, not enclosed.
- **Rigorous a priori rounding bounds** given IEEE extended arithmetic: τ_eval and τ_R.
- **Estimates:** a_u, a_z, the second-order part of τ_ref, and λ_min.
- **Not claimed:** a rigorous floating-point enclosure of the whole computation.

## Checks before relaunch

**Smoke run on the development operator** (`runs/deviation2_test/`; the amendment's smoke record is restored unchanged):
- **Gates:** all 1,088 passed.
- **Bounds:** η changed by at most 5.8·10⁻¹⁰ relative against the amendment smoke run. The cause is the extended-precision residuals.
- **Size of the new terms:** the evaluation terms are at most 6.6·10⁻¹⁵ on the goal intervals, at most 1.7·10⁻⁹ of the certified radius.
- **Stopping:** the first stops are unchanged for M0, M3 and M6 at every tolerance.

## Runs

As amendment §8 prescribes, all four pilot runs are repeated from the start with the changed runner. The previous records are kept as `runs/pilot_dev1_*`. The report revision will:
- compare the repeated energy endpoints with them;
- state the failing case's new intervals;
- report the stopping results with the evaluation terms.

New hashes are in `PILOT_CODE_SHA256_DEVIATION2.txt`. `rc_fast.py`, `rc_fem.py` and `rc_pilot_analysis.py` are unchanged.
