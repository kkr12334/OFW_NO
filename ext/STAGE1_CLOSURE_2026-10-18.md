# Stage 1 closure

18 October 2026. Claude, for Kumar and ASTRA. Stage 1 is closed on ASTRA's decision in `ASTRA_STAGE1_REV1_REVIEW_2026-10-18.md`.

## Approved conclusions

These are worded as in the final report, revision 1.1, at the levels ASTRA approved.

1. **Confirmatory (set C).** On channels, OA-VCB reduces one-pass energy error relative to the four-step Jacobi-PCG smoother by geometric factors of 5.70–6.12. On barriers the factors are 1.38–1.45. The frozen labels are "not accounted for by the tested four-step Jacobi-PCG control" (channels) and "partly accounted for / inconclusive" (barriers).
2. **Reserve brackets (R1, R2).** The paired geometric errors place OA-VCB between Jacobi-PCG(8) and PCG(14) on channels, and between PCG(5) and PCG(8) on barriers, for all three initialisations. This is a distribution-level statement. It is neither a bracket for each operator nor an exact crossover budget.
3. **Local K₅ result (R4).** On channels, OA-VCB's fine-level smoother improves on the exact K₅ energy minimum (ρ ≤ 0.99) on 97–100 of 100 inputs, at both positions, for all three initialisations. It is a local, fixed-input separation, not a full-cycle class separation.
4. **No computational advantage.** No computational advantage is demonstrated for the tested implementation. Larger Krylov budgets and the measured classical solvers remain decisive controls.

## How ASTRA's review of revision 1 was handled

All four corrections were accepted without dispute and are applied in revision 1.1. No result, interval or conclusion changed.

- **A (timing aggregates).** End-to-end times are now medians of per-operator sums: 6.94 / 6.58 ms, previously 6.93 / 6.61. The relative-residual times are labelled offline-selected prefix costs. The hard-coded OA-VCB cell is replaced. The profile is labelled "last of five repeats".
- **B (ledger).** Now called approximate MAC-equivalent arithmetic. The report lists the exclusions: the dense Galerkin solve; Gram symmetrisation, scaling and masking; and the padded near-kernel width of the neural wrapper. Ratios are quoted as roughly 400× and 160×.
- **C (K₈ counts).** "At least 1% lower on 19–43; lower at all on 22–49", with a ρ < 1 column added.
- **D (gates).** The missing run-time frozen-scale gate is disclosed, along with the retrospective hash match. That match does not prove the file's contents at run time.
- **Route C wording.** Revision 1's "cannot be a speed result" is withdrawn in §10.

The revision 1 generator outputs are kept in `runs/rev1_snippets_as_published.json` and `runs/rev1_summary_as_published.json`. The revision 1 generator and check scripts that ASTRA reviewed are kept unchanged in `superseded_rev1/`, with their hashes. The revision 1.1 claims check passes 60 named checks with no failures (`stage1_report_check_rev1.py`).

## Not claimed

- Separation from the complete class of polynomial-smoothed cycles. The best candidate found by optimisation is not a class bound.
- Communication efficiency, or efficiency "per stage".
- A spatial mechanism.
- That nonlinearity is necessary.
- Exact operation ratios. The ledger is approximate.
- Measured online stopping times.
- Anything beyond L3 (3486 unknowns), 2D, two coefficient families and this architecture.

## Final documents

| Document | Role | SHA-256 |
|---|---|---|
| `STAGE1_REPORT_REV1_1_2026-10-18.md` | **Final Stage 1 report** | `5c49f87e016add6c…` |
| `STAGE1_REPORT_REV1_2026-10-18.md` | Revision 1, frozen | `b4f51048…` |
| `STAGE1_REPORT_2026-10-17.md` | v1, frozen | `9ec033f3…` |
| `STAGE1_PROTOCOL_2026-10-16.md` | Protocol v1 | `324897bd…` |
| `STAGE1_PROTOCOL_AMENDMENT_1_2026-10-16.md` | Amendment 1 | `406c5bd0…` |
| `STAGE1_PROTOCOL_AMENDMENT_2_2026-10-18.md` | Amendment 2 (reserve plan, repairs, errata) | `5480b6a9…` |
| `STAGE1_PRECONFIRMATION_NOTE_2026-10-17.md` | Freeze before set C | `66be1dc3…` |
| `CLAUDE_RESPONSE_TO_ASTRA_STAGE1_REPORT_REVIEW_2026-10-18.md` | Response to ASTRA's first report review | `8f06505d…` |

All records are under `ext/runs/`, `ext/dev/` and `ext/explore/`. The chain logs are `runs/chain.log` and `runs/chain_amendment2.log`.

## Data status

| Set | Source (rng) | Status |
|---|---|---|
| T, D (development) | Seed-1 data: training ids 0–299 (rng 1001), test ids 300–399 | Used; inspected in the paper and Stage 0 |
| R (retrospective) | Seed-2 and seed-3 test sets | Used; previously inspected |
| C (confirmation) | 2026101601 channels / 2026101602 barriers | Used |
| S (stratified barriers) | 2026101611 | Used |
| LC (load challenge) | Loads 2026101621·10³ + i and 2026101622·10³ + i on the first 50 operators of C | Used |
| Reserve | 2026101603 channels / 2026101604 barriers | **Used** (amendment 2) |

Any future confirmatory claim needs newly allocated seeds, documented before use.

## Lessons carried forward

ASTRA found three process errors in Stage 1. Each becomes a standing rule:

- **Do not read an optimiser's best candidate as a class bound.**
- **Validate timing boundaries by profiling before quoting them.**
- **Write every specified gate into the runner and save the inputs it checks** (for example, the frozen scale and its hash).

## Handoff: Route C scoping (go for scoping, ASTRA)

The question to test, from ASTRA: can operator-adapted coarse spaces and localised equilibrium repair produce useful, rigorously reliable goal-error intervals for high-contrast media, at substantially lower certification cost and with controlled dependence on contrast and topology?

**Deliverables, in order:**
1. **Precise target and theorem map.** Start with a bounded volume-average goal. Separate continuous from discrete guarantees, state the assumptions, and explain how patch compatibility is restored for arbitrary inexact inputs.
2. **Classical reference certificate.** Implement an established equilibrated or functional construction, first with accurate and then with deliberately inexact primal and dual inputs. A learned approximation may be one input source; it never decides validity.
3. **Small discriminating pilot.** Compare standard reconstruction with operator-adapted repair on channels and barriers over a declared contrast range. Measure width, effectivity, equilibrium defects and costs.
4. **Go/no-go decision.** Proceed only with a credible robustness or localisation advance, or a meaningful reduction in cost to a prescribed certified tolerance against the strongest classical baseline.

**Closest prior art to clear first:**
- Mallik, Vohralík and Yousef (JCAM 366, 2020): guaranteed goal-oriented bounds with inexact solvers.
- Fanaskov, Rudikov and Oseledets (2024): neural functional a posteriori estimates.
- Ernst, Rekatsinas and Urban: certification of neural-network approximations.
- Multilevel algebraic-error flux reconstructions, to be identified and checked in the scoping.

**Effort mode:** theory work, so extra effort.
