# Response to ASTRA's review of the Stage 1 report

18 October 2026. Claude, for Kumar and ASTRA. Responds to `ASTRA_STAGE1_REPORT_REVIEW_2026-10-17.md`.

**Outcome.** I re-checked every code-level and literature finding before accepting it. All are accepted. No point is disputed in substance. Two points get a factual qualification, marked "qualified".

**Delivered:**
- **Amendment 2** (`STAGE1_PROTOCOL_AMENDMENT_2_2026-10-18.md`, SHA-256 `5480b6a9…`), hashed before any reserve operator was generated. It contains:
  - errata;
  - the timing-repair and ledger specifications;
  - the frozen reserve experiment;
  - the adopted wording corrections.
- **The dated report revision** `STAGE1_REPORT_REV1_2026-10-18.md`. The v1 report stays frozen.
- **Repaired timings** `runs/timing_v2_C_*.json`.
- **The operation ledger** `runs/stage1_ledger.json`.
- **The reserve results** `runs/reserve_*.json`.

## Point by point

| ASTRA § | Finding | Check | Response |
|---|---|---|---|
| 1 | Primary results reproduce; read the P1 label as "not accounted for by the tested four-step Jacobi-PCG control" | — | **Accepted.** The prose translation is used throughout revision 1. |
| 2 | D2 contains the K₄-valued trajectories, but its value only bounds the class optimum from above. "Not an artefact of the cap" and similar phrases overclaim. | The logic is right. I inverted the direction of the inference: a candidate's error bounds E\* from above, so E\_hat/E\_NN > 1 says nothing about E\*/E\_NN. | **Accepted.** All class-wide wording is removed. The candidate is called "best per-instance polynomial candidate found". The barrier asymmetry is kept: there the candidate beats the network. |
| 3 | K₅ is a sensible degree control, but OA-VCB is not strictly 4-hop. σ is global and nonlinear, PCG uses reductions, and SGS propagates sequentially. Communication language is unsupported. Barrier K₅ wording is wrong; K₈ does not dominate on every input. | Confirmed in `OpSmoother.forward`: σ = max\|z\|·rms(z/max\|z\|) enters z/σ before the GELU encoder. | **Accepted.** "Per neighbour-exchange round" and "efficient per stage" are deleted. K₅ is described as a degree control only. Barrier K₅ is described as position-dependent (pre 6/100, post 63/100). The 23/100 and 33/100 K₈ counts are reported. |
| 4.1 | The ledger is incomplete: transfers, coarsest solve and wrapper are omitted; PCG's initial products and MR's third diagonal product are not counted; the fine-level count includes 90 structural zeros, so it is not a strict lower bound. | All confirmed. Transfers come to 236 822 MACs, matching ASTRA's figure. The MR and PCG counter omissions are as stated. The fine-level overcount is 264 600 MACs per call. | **Accepted.** New `stage1_ledger.py` lists every term. Checks: the learned dense counts now equal FlopCounterMode on all three levels, and the cycle's SpMV-equivalents per level equal the recorded counters for all 16 method–family pairs. The OA-VCB/PCG ratio falls from 587 to about 406. |
| 4.2 | Timing bugs: (a) unused spectral set-up charged to every classical method; (b) mesh patterns rebuilt inside the OA-VCB timer; (c) weights loaded in the batch-8 timer; (d) residual formed outside the one-pass timers. No end-to-end boundary. | All four confirmed in the code. (b) holds because `OAData.__init__` calls `structural_patterns` uncached. | **Accepted.** `stage1_timing_v2.py` fixes all four on the same saved inputs, run alone. It adds the u_c-construction and end-to-end times, a component profile, time to a fixed relative residual of 10⁻⁶, and outer termination reasons. **Qualified:** the new profile confirms that the operator-adapted hierarchy is most of the classical one-pass time (revision §7). The v1 statement was right in substance but had not been measured. |
| 4.2 | Do not emphasise factors above 10⁸ near reference precision | Agreed | **Accepted.** Factors at or beyond the reference floor are no longer quoted. |
| 5 | Withdrawing the spatial reading is right; replacing it with a new causal story is not | Recomputed: e_PCG14 / e_SGS-PCG4 = 0.331 [0.315, 0.348] (channels) and 0.535 [0.490, 0.582] (barriers) | **Accepted.** ASTRA's replacement sentence is used verbatim. |
| 5 | "Matches/ties"; delete the float32 sentence; keep the random-load distinction; Chebyshev proof; scope of "no failures"; numeric scan is a typo check | Chebyshev: correct, the residual polynomial changes sign | **Accepted, all.** The A3.2 argument is corrected in amendment 2 (B1.1): \|q\| < 1 with symmetric pre/post smoothing gives 0 ≤ E ≤ ‖S‖²_A·I. The conclusion stands. |
| 6 | Steps 1–4 | — | Steps 1–3 done in this order: revision and repair, then a frozen reserve plan, then the reserve run. Step 4 (longer K₅ fit) was **not run**: it cannot certify a class bound. The x2 runs saved losses but not coefficients, as ASTRA notes. Any future fit will save coefficients and round-trip checks. |
| 7 | Route B passes the local part only. B′ needs a more specific contribution than "stronger baselines matter". Corrections to the literature notes. | Verified: Li et al. §4.2 / Table 4 compare against classical MG and CG (CG faster for large α). Zhang–Li–Saad (arXiv:2607.28456) abstract confirmed. McGreivy & Hakim (2024) is a published contribution on weak baselines. | **Accepted.** Revision 1 §10 adopts ASTRA's framing: separate local representational gain, complete-cycle gain and deployable benefit. B′ proceeds only through a bounded external-reproduction pilot, and stops if the pilot repeats the overhead lesson. |
| 8 | Suggested central conclusion | — | **Adopted**, extended with the reserve and repaired-timing results; see revision 1, §1. |

## Reserve and repair results

All endpoints frozen in amendment 2 were met. Full tables are in revision 1, §§4–7.

- **R1 (primary, channels): confirmed for all three initialisations.**
  - Jacobi-PCG(8) is less accurate than OA-VCB: G = 1.40–1.51, lower limits ≥ 1.279.
  - Jacobi-PCG(14) is more accurate: G = 0.37–0.40, upper limits ≤ 0.426.
- **R2 (barriers): confirmed for all three initialisations,** with PCG(5) and PCG(8) bracketing OA-VCB.
- **R3:** G(PCG(14) → SGS-PCG(4)) = 0.32 (channels) and 0.55 (barriers). The set D finding reproduces.
- **R4 (channels, local): K₅ separation confirmed** at both positions for all three initialisations, with ρ_K₅ ≤ 0.99 on 97–100 of 100 inputs. Against K₈, OA-VCB is better on only 19–43 of 100 inputs.
- **Ledger.** OA-VCB costs 713 M MACs per pass. The ratio to PCG is 406× (v1 said 587×) and to PCG(14) 161× (v1 said 193×). Classical denominators now include the transfers, the coarsest solve and the wrapper.
- **Repaired timings.**
  - Classical one pass: 66–72 ms (v1: about 80 ms), of which the operator-adapted hierarchy is about 63 ms.
  - OA-VCB batch-1: about 283 ms (v1: about 370 ms).
  - u_c construction: 2.6 ms.
  - Two-grid PCG reaches OA-VCB's per-sample error in 4.3 / 4.0 ms conditional on u_c, or 6.9 / 6.6 ms end-to-end.
  - Every outer loop terminated as "completed".
  - Against the deployable 10⁻⁶ relative-residual rule, some curves are censored at their frozen iteration limits. Revision 1, §7 reports these counts.

## For the next review

- Revision 1, amendment 2 and the files listed above.
- **Open question for ASTRA:** whether Route C, in its certification form, deserves a scoping note now. This review did not address it. Kumar raised the goal-oriented direction, which is closest to his own work.
