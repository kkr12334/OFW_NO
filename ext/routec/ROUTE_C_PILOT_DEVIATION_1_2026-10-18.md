# Route C pilot: deviation 1 (reference solves)

18 October 2026. Claude, for Kumar and ASTRA. A dated note under amendment 1, §8. The amendment, the decision rules, the analysis script, the seeds and the certificate code are unchanged. Only the runner's reference ("exact") solves and their gate change.

## What happened

The barriers L3 pilot stopped at the gate "id 1 c=1e+06: exact solves finite and residual small". Contrast 10⁶ is the stress arm, outside the decision rules. The gate required max|b − Au| ≤ 10⁻⁹ max|b| for the direct solve u of each of the four loads. It failed for the dual-0 load of geometry 1.

## Diagnosis (`rc_exact_gate_diag.py` → `exact_gate_diag.json`; `rc_exact_refine_proto.py` → `logs_exact_refine_proto.log`)

These scripts rebuild the same geometry, coefficients and loads and solve them in the same way. They compute no outcome.

| Geometry 1, c = 10⁶ | Gate residual (threshold 10⁻⁹) | Normwise backward error | Energy error of the reference itself |
|---|---:|---:|---:|
| Primal | 8.5·10⁻¹⁰ | 7.3·10⁻¹⁷ | 1.07·10⁻¹⁰ |
| Dual 0 | **1.3·10⁻⁹** | 4.0·10⁻¹⁷ | 7.7·10⁻¹¹ |
| Duals 1, 2 | ≤ 2.5·10⁻¹⁴ | ≤ 1.0·10⁻¹⁶ | ≤ 3.3·10⁻¹⁵ |

**Two things went wrong:**
- **The direct solve was backward stable** (backward error at round-off). The gate's residual test is not scale-aware, so it was the wrong test.
- **Something real was nevertheless wrong.** For this geometry (near-kernel dimension 2 already at 10⁴), the double-precision reference has an energy-norm error of about 10⁻¹⁰ relative. That equals the near-exact floor of amendment §4, so errors measured near the floor would have been contaminated by the reference's own error.

So this is a defect of the check and of the reference computation it protects. It is not a certificate failure: no bound, moment, containment or guarantee gate failed. Under §8 it is fixed through this note, and the runs are repeated from the start.

## Change

- **Refinement:** each reference solve is refined twice by iterative refinement. The residual b − Au is accumulated in extended precision (numpy `longdouble`, 64-bit mantissa) and the same sparse LU factor is reused.
- **New gate,** replacing the residual test: the solution is finite, and the energy-norm error of the reference, estimated by one further extended-precision correction A⁻¹(b − Au), is ≤ 10⁻¹² relative. That is two orders below the 10⁻¹⁰ floor. The estimate is saved per operator and load (`ref_err`).

**Prototype on the failing case and two controls** (energy error of the reference, before → after two steps):

| Case | Before | After |
|---|---:|---:|
| Barriers, geometry 1, c = 10⁶, primal | 9.4·10⁻¹¹ | 1.2·10⁻¹³ |
| Barriers, geometry 1, c = 10⁶, dual 0 | 1.2·10⁻¹⁰ | 6.4·10⁻¹⁴ |
| Barriers, geometry 0, c = 10⁶, primal | 5.1·10⁻¹⁵ | 8.0·10⁻¹⁶ |
| Channels, geometry 3, c = 10⁶, primal | 2.0·10⁻¹⁵ | 4.0·10⁻¹⁶ |

Everything downstream of the reference (inputs built from u, errors, effectivities) changes only through these differences.

## Runs

All pilot runs are repeated from the start with the changed runner, in the order barriers L3, channels L3, channels L4, barriers L4. Every pilot record then comes from one code version. Earlier records are kept unchanged:

| Record | State |
|---|---|
| `runs/pilot_v0_channels_L3.json` | Complete, pre-change runner |
| `runs/pilot_interrupted_barriers_L3.json` | Container restart |
| `runs/pilot_stopped_barriers_L3.json` | This gate stop |

The report will:
- compare the repeated channels run with the v0 run, expecting agreement to the level of the reference change. This is a reproducibility check.
- compare the barrier results completed before the stop with the repeated run.

## What I had seen before making the change

The full channels L3 outcomes: I ran the frozen analysis on them. For barriers, I had seen only the per-run log lines (worst candidate effectivity per geometry and contrast) for geometries 0 and 1.

The change touches neither a threshold, nor a certificate, nor the analysis, so it cannot be steered by those outcomes. It is stated here for completeness.

## Code

- The `rc_pilot.py` change is the reference-solve block, the new constants `REFINE_STEPS = 2` and `REF_TOL = 1e-12`, the `ref_err` record field, and a pilot-mode gate that this note exists, with its hash recorded in the manifest.
- The pre-change runner is kept as `rc_pilot_pre_deviation1.py`.
- New hashes are in `PILOT_CODE_SHA256_DEVIATION1.txt`. `rc_fast.py`, `rc_fem.py` and `rc_pilot_analysis.py` are unchanged (hashes as in `PILOT_CODE_SHA256_AT_LAUNCH.txt`).
- The code path was exercised on the development smoke operator before relaunch (`runs/deviation1_test/`).
