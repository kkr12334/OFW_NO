# Route C pilot: addendum to deviation 2

18 October 2026. Claude, for Kumar and ASTRA. This addendum responds to `ASTRA_ROUTE_C_PILOT_REV1_REVIEW_2026-10-18.md` §2–§3.

`ROUTE_C_PILOT_DEVIATION_2_2026-10-18.md` itself is unchanged, because its SHA-256 is recorded in every pilot run manifest. No pilot record, code or decision changes. No repeat is needed: the points below change no formula used in the saved records. They qualify what the records show and fix the code that will be used from now on.

## 1 Disclosure: the repaired stopping test depends on a global direct solver

`rc_pilot.py` computes three things with global machinery:

- **λ_min:** a shift-invert `eigsh` estimate of λ_min(A), one per operator (line 168).
- **a_u, a_z:** in every `res_full` call (line 177), each is computed as the energy of `lu.solve(lo)` plus ‖bnd‖₂/√λ_min, where `lu` is the global factorisation also used for the reference solves.
- **Discarded work:** `centre_eval` calls `res_full` and so pays for that solve even though it discards the estimate.

Because a_u and a_z enter the total product and polarization radii, they also enter the stopping decisions.

They are computable from A and the stored vectors; they do not use an unknown exact solution. But the repaired stopping test is therefore **not independent of a direct solver**. It is a diagnostic of certificate behaviour, not a deployable or cost-competitive stopping procedure. It must not be presented as evidence of practical savings over solving the system. "No cost advantage is claimed" remains true, and now carries this additional reason.

The disclosure is made in report revision 1.1, §6.

## 2 Qualifications of the arithmetic statements

**Wording.** The wording from now on is "extended-precision evaluation with explicit rounding-error budgets and labelled numerical estimates". Deviation 2's terms "rigorous" and "exactly with one solve" are withdrawn for the implemented quantities:

- **The solve.** The known residual difference is represented accurately, but its inverse-norm is evaluated with a numerical solve and norm.
- **Final conversions.** The budgets τ_eval and τ_R cover the extended-precision operations. They do not yet cover the final conversion of each result to binary64 (`float(c)` in `centre_eval`; the accumulated value in `polar_lower`).
- **Reciprocal.** The reciprocal 1/s is formed in double before conversion (`LD(sign / s_)`).
- **Energy norm.** The lower bound divides by an energy norm computed in floating point.

ASTRA's audit finds no failure in the saved data from these gaps. They are gaps in the stated derivation, not evidence that an interval misses its target.

## 3 Code changes for the development round (not applied to the pilot)

The development round uses a separate module. The pilot runner stays as recorded. The module will:

1. **Separate the two kinds of evaluation.** `res_full` will only evaluate the residual in extended precision and return the rounding bound. The solve-based estimate (a_u, a_z) becomes a separate, explicitly named diagnostic. It will never be called by the centre evaluation, nor by any construction that is meant to be deployable.
2. **Fix the precision handling:**
   - form `LD(sign) / LD(s_)` in extended precision;
   - add the final binary64 conversion terms (one unit roundoff of the returned value) to the budgets;
   - keep the polarization lower-bound numerator in extended precision until the final division.
3. **Gate complete intervals.** Add a direct gate that the centre, τ_eval, a_u, a_z, τ_ref, both total radii and the polarization endpoints are finite, non-negative where required, and correctly ordered (lo ≤ hi). Containment is checked only after that gate.
4. **Keep reference solves and M6 diagnostic.** They are kept as diagnostic comparisons, outside any candidate remedy's construction and outside any intended production error budget.
5. **Name a bound route for any later deployability claim.** Any later deployability claim must replace the solve-based estimate with an independently justified computable bound. One example: ‖d‖₂/√λ_lower, with a verified λ_lower ≤ λ_min(A). Alternatively, an explicitly costed construction may be used.

**Testing.** If any formula used for a recorded quantity changes, the change will be checked on the formerly failing case and against the saved records before any decision about a repeat.
