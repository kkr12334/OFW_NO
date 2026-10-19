# Route C scoping: target, theorem map, reference certificate and pilot plan

18 October 2026. Claude, for Kumar and ASTRA. This responds to the "go for scoping" in `ASTRA_STAGE1_REV1_REVIEW_2026-10-18.md`, §3.

**Status of the four requested items:**
- **1. Target and theorem map:** done (§§1–5).
- **2. Classical reference certificate:** implemented and checked on development operators (§6).
- **3. Discriminating pilot:** designed (§9) but **not run**. It needs fresh, newly allocated operators, and I would like ASTRA's review of the design first.
- **4. Go/no-go:** the criteria are fixed in advance (§10). The decision waits for the pilot.

**Data used so far.** Only development data (seed-1 data, mostly operator 300), plus contrast sweeps on that one geometry. Every number in §§6–7 is a development number. None is confirmatory.

## Summary

1. **Target.** The goal is the bounded volume average J(u) = |ω|⁻¹ ∫_ω u, with ω a union of mesh elements. There are two certified targets:
   - **(D) discrete:** J(u_h), the exact finite-element output. This certifies an inexact solve.
   - **(C) continuous:** J(u), the PDE output.

   Both use the standard identity J(·) − J(v) − R_v(w) = a(e_u, e_z), with radius η_u·η_z. All data (K, f, the goal weight, the residual lifting) are piecewise constant on the flux mesh, so equilibrium is exact: there is no oscillation term and no unknown constant.

2. **The continuous certificate is uninformative on the paper's L3 mesh, even with exact solves.**
   - The radius is 2.6 |J| on channels and 18 |J| on barriers (development operator 300).
   - The L3 discretisation error is large: for the exact FE solution the total estimator is about 29% of |||u_h||| on channels and 39% on barriers. Even the L5 reference has relative energy bounds of 15–17%.
   - Making (C) useful would need goal-oriented mesh adaptivity, which is classical (MVY 2020 already does it).
   - **Recommendation:** Route C should target (D), certified outputs of inexact solves. (C) is still reported, for honesty.

3. **The classical algebraic certificate loses efficiency as contrast grows.**
   - The classical certificate is the multilevel residual lifting of Mallik–Vohralík–Yousef (MVY 2020, App. 8) and Papež–Rüde–Vohralík–Wohlmuth (PRVW 2020).
   - At contrast 1 my lowest-order implementation gives effectivities of 1.13 (30 Jacobi-PCG steps) and 1.40 (u_c). PRVW report 1.00–1.01 for PCG iterates on a smooth Poisson problem with degree-p spaces (their Table 3). That is the same order, though the problems differ.
   - At contrast 10⁴ they rise to 22.05 (channels, PCG-30) and 4.78 (barriers, u_c).
   - This matches the literature. Papež–Vohralík (2022) state that their efficiency constants depend on the ratio of the largest to smallest diffusion eigenvalue. The mixed-FE work of Miraçi–Papež–Vohralík–Yotov has the same dependence.

4. **Two sources of overestimation, and two corresponding changes.**
   - **(a) The fixed element-local residual lifting.** On the same operators, the best possible RT0 flux with that lifting overestimates 10% white noise 15–30-fold. If the lifting is optimised jointly with the flux, the overestimation is only 1.34–1.38-fold.
   - **(b) The geometric coarse space and partition of unity.**

   The changes, in development:
   - **Lifting-free local problems:** constrain only the hat moments of the divergence, splitting the residual vector nodally.
   - **An operator-adapted partition of unity:** Shepard-normalised one-level operator-adapted functions plus near-kernel vectors.

   Together they come within 0–3% of the global joint optimum, per energy bound. This holds in every development case at contrasts 1–10⁴, with effectivity 1.02–1.25 for all tested inputs except white noise. White noise is at about 1.35–1.38 for the global joint optimum and the operator-adapted construction alike. At 10⁶ the gap is at most 21% (PCG-100 on channels).

   The worst case over **all** inputs of the global optimum itself, C_glob, is a generalised eigenvalue. It is 2.06–2.14 and flat in contrast from 1 to 10⁶ (§7.4).

5. **A computable, input-independent robustness index is consistent with the difference.**
   - For geometric partitions of unity the index grows like c^{1/2}: about 34 at 10⁴ and 340 at 10⁶.
   - For the operator-adapted partition it is flat on barriers (maximum 1.27 at every contrast from 1 to 10⁶). On channels it is flat at the 95th percentile but grows for a few members (3.6 at 10⁴, 36 at 10⁶). These are presumably functions whose support cuts a channel, the mechanism of Graham–Lechner–Scheichl. They will be identified in the pilot.

6. **Candidate contribution, for testing and not yet claimed.** Contrast-robust, guaranteed algebraic-error and goal certificates for arbitrary inexact inputs (classical, learned or perturbed), built from:
   - operator-adapted partitions of unity;
   - lifting-free localised equilibration;
   - a computable index that predicts where efficiency is lost (a proof that it bounds the efficiency is still open, §5).

   **Novelty status:** I did not find this in the ten closest papers read in full (§8), but that does not establish novelty. The known ingredients are listed in §8.

7. **The learned output is only one input.** The certificate never depends on it. On development operator 300 the OA-VCB output was certified at a goal-radius effectivity of 1.30 (channels) and 1.20 (barriers).

## 1 Problem, goal and assumptions

**Problem.** Find u ∈ H¹₀(Ω) with a(u, φ) := (K∇u, ∇φ) = (f, φ) for all φ.
- **Discretisation:** P1 conforming FE, V_h ⊂ H¹₀(Ω), stiffness A, residual R_v := F − Av.
- **Dual problem:** a(φ, z) = J(φ), with g := 1_ω/|ω|.

**Assumptions** (facts checked by `rc_facts.py` and `rc_feasibility.py`):

| # | Assumption | Our data |
|---|---|---|
| A1 | Ω polygonal | Unit square minus three polygonal holes. Refinement uses plain midpoints (checked in `ofwno_core.MeshHierarchy`), so Ω is the same at every level. |
| A2 | Γ_D = ∂Ω, u = 0 | Outer and hole boundaries; 448 Dirichlet nodes at L3. |
| A3 | K ∈ P0(T_h), K > 0 | Set per L3 element from its centroid, so represented exactly. Two values {1, c}: channels c = 10⁴; barriers 1/c, c = 10⁴. |
| A4 | f ∈ P0(T_h) | f = 1 (channels); f = 1 on {K ≥ 0.5} (barriers). Loads are exact. |
| A5 | g ∈ P0(T_h) | ω is a union of L3 elements, so J is bounded on L² ⊂ H⁻¹. |
| A6 | v, w ∈ V_h | Any nodal vector on the 3486 unknowns. The OA-VCB output, u_c and solver iterates all qualify; Dirichlet conditions hold exactly. |
| A7 | Conforming shape-regular mesh | 7424 triangles; angles 21.2°–126.5°; 2048 obtuse elements (no M-matrix property is needed). |
| A8 | Exact arithmetic for the guarantees | Floating-point equilibrium and moment defects are reported (≤ 10⁻¹¹ relative in all development runs). This is not interval arithmetic. |

**Topology facts** (development operators 300–309 of seed-1 data unless stated):
- Near-kernel ("island") dimension on barriers, operators 300–399: 0 for 58 operators, 1 for 31, 2 for 4, 3 for 7. Channels: 0 for all.
- Interior vertices that are not quasi-monotone: mean 1.9 (0–7) of 3486 per operator. The count is identical for both families, as expected for a two-valued coefficient.
- So the local-estimator obstruction of Li (2026), which is built on a checkerboard vertex, could bite at only a handful of vertices per operator here.

## 2 Theorem map

Status codes:
- **K:** known, with source.
- **R:** routine adaptation; proof given here.
- **N:** proved here, novelty not cleared.
- **C:** conjecture or open; evidence in §7.

| # | Statement | Status | Source / proof | Assumptions | Development check |
|---|---|---|---|---|---|
| T1 | J(u) − J(v) − R_v(w) = a(u − v, z − w); the same holds in V_h for (u_h, z_h) | K | Standard; MVY (2020) §4 | A1–A6 | Exact to round-off |
| T2 | Prager–Synge: if div σ = f, then \|\|\|u − v\|\|\| ≤ ‖K^{-1/2}(σ + K∇v)‖, constant 1, any K > 0 | K | Prager–Synge; Braess–Schöberl; Vohralík (2011) | v ∈ H¹₀ | All intervals contained their targets |
| T3 | Discrete version: if (div σ, φ) = R_v(φ) for all φ ∈ V_h, then \|\|\|u_h − v\|\|\| ≤ ‖K^{-1/2}σ‖ | K | PRVW (7.1a) uses it with a lifting r_h; proof in §4 | — | 100% containment |
| T4 | Exact equilibrium without data oscillation | R | f, g, r_h ∈ P0(T_b) and div RT0(T_b) = P0(T_b) | A3–A5 | Defects ≤ 10⁻¹¹ |
| T5 | Element-local exact lifting r_h ∈ P0(T_b) with (r_h, ψ_a) = R_a | R | MVY (5.4) / PRVW (6.7) moments realised on the barycentric refinement (3×3 local systems; for equal sub-element areas the matrix has eigenvalues |K|/3 and −|K|/9, so it is invertible) | — | Defect ≤ 3·10⁻¹⁶ (relative) |
| T6 | Discretisation patches are compatible for any v | K | MVY (5.7), PRVW (4.1) | T5 | Patch divergence defects ≤ 10⁻¹² |
| T7 | Multilevel algebraic flux with div σ_alg = r_h for any v | K | PRVW Def. 4.3 / MVY App. 8 | Nested hierarchy | Exact |
| T8 | Efficiency of T7 with constants depending on contrast; proved for Poisson | K | PRVW Thm 7.4 (Poisson); Papež–Vohralík (2022) Thm 11.1–11.2 (depends on λ_max/λ_min of K); MPVY (mixed FE): β depends on Λ_max/Λ_min | — | Contrast sweep, §7.2 |
| T9 | Two-level flux with **any** partition of unity whose floating members lie in V_c, exact divergence r_h | R | §4, Prop. 2 | PU sums to 1 at the nodes; floating supports connected | Exact |
| T10 | **Lifting-free** two-level flux: moments (div σ, ψ_a) = R_a for any v, from the nodal residual split | N | §4, Prop. 3 | As T9 | Moment defect ≤ 4·10⁻¹² (relative) |
| T11 | Factorisation: effectivity = (localisation loss) × (global stability of the moment-constrained RT0(T_b) problem). The second factor is computable as a generalised eigenvalue, C_glob | R / C | §5. The identity is elementary; contrast-independence of C_glob is **C** (proof idea in §5) | — | §7.4: flat in contrast |
| T12 | Localisation loss is bounded by the partition-of-unity index C_θ (input-independent) and overlap, under a local stability hypothesis | C | §5 sketch; robustness indices in the spirit of Graham–Lechner–Scheichl (2007) π(α), γ(α) | Hypothesis S | Index sweep, §7.3 |
| T13 | For operator-adapted partitions, C_θ is contrast-independent when each support's high-K components are flattened and floating components are in V_c (near-kernel Z); geometric partitions give C_θ ≳ c^{1/2} | C | Mechanism as in GLS (2007) | Topology hypotheses | Barriers yes; channels: p95 yes, max no (§7.3) |
| T14 | Efficiency of the **discretisation** estimator: robust under local quasi-monotonicity for suitable modified estimators; not robust for any finite-range local estimator in general | K | Cai–Zhang (2012); He (2026); Li (2026, impossibility via checkerboard vertex); Vohralík (2011): robust energy-norm efficiency needs K constant on dual cells, not our setting | — | Our σ_disc is the unmodified K⁻¹-weighted patch minimisation; its efficiency is not claimed |
| T15 | Sharper goal intervals (MVY averaged ½ form; polarisation) | K | MVY Thm 4.8; Sauer-Budge et al. (2004) | — | Not implemented yet |

**Continuous versus discrete.**
- **(D) certifies J(u_h)** for the assembled system. It needs T1, T3 and an algebraic flux. It is a statement about the discrete problem and must be labelled so.
- **(C) certifies J(u)** for the PDE with the stated K, f, Ω and boundary conditions. It needs T1, T2 and the total flux σ_disc + σ_alg with div = f. No coefficient, data or geometry approximation enters, because of A1–A5.

## 3 Reference certificate (deliverable 2)

**Code:** `ext/routec/rc_fem.py`, `rc_goal.py`.

**Spaces.** RT0 on the barycentric refinement T_b of the L3 mesh: 22 272 elements and 33 632 edges.

**Algebraic fluxes implemented.** The first is the classical reference; the others are development variants.

| Name | Construction | Role |
|---|---|---|
| MVY | MVY 2020 App. 8, multilevel on T_0, …, T_3, T_b; fixed lifting r_h; coarse Riesz solve on T_0 (28 unknowns) | **Classical reference** |
| Two-level, fixed lifting | Patches on T_b, partition of unity G2/G1/OA2, divergence = local data from r_h | Isolates the lifting effect |
| Relaxed | As above, but only the hat moments of the local data are imposed | Development step |
| **Lifting-free** | Moments from the nodal residual split (§4 Prop. 3); no r_h | Candidate |
| Joint optimum | Global min ‖K^{-1/2}σ‖ over RT0(T_b) subject to (div σ, ψ_a) = R_a | Best achievable RT0(T_b) bound; a global mixed solve, so a reference only |

**Partitions of unity.**
- **G2 / G1:** geometric level-2 / level-1 hats. Coarse space: geometric P1, 814 / 174 unknowns.
- **OA2:** the one-level operator-adapted functions [I; P̂₃] of the paper's hierarchy (4 Jacobi steps of prolongator smoothing). They are Shepard-normalised at the nodes and completed by the fine hats of the Dirichlet nodes. Coarse space: the 814 normalised functions plus near-kernel Z.

**Why only one level?** The composite operator-adapted functions of levels 1 and 0 are nearly global and far from a partition of unity:
- mean supports of 2188 (level 1) and 3459 (level 0) of 3486 nodes;
- nodal sums from −1.23 to 1.74.

The one-level functions have mean support 59 nodes (maximum 89), mean overlap 13.8, and nodal sums in [0.014, 1.42] > 0, so Shepard normalisation is well defined.

**Discretisation flux:** vertex patches on T_b with the Braess–Schöberl target ψ_a K∇v (MVY Def. 5.1).

**Implementation checks:**
- The assembled stiffness equals the paper's `H.A_fine` to 5·10⁻¹⁵ relative.
- At contrast 1, MVY gives 1.13 / 1.40 (PCG-30 / u_c), the same order as the published values (PRVW Table 3, 1.00–1.01 on a smooth Poisson problem).
- Every interval in the development runs contained its target: 40 (input, method) cases over both families. Maximum moment defect 3.2e-12 (relative); maximum equilibrium defect 5.3e-11.
- For (C), the L3 interval had to contain the **whole** guaranteed L5 bracket: J(u_5) ± η_u5·η_z5, with exact L5 solves and K, f, g inherited from L3.

## 4 Constructions and proofs

Let e := u_h − v.

**Lemma 1 (discrete guarantee, T3).** Let σ ∈ H(div, Ω) satisfy (div σ, ψ_a) = R_a for every unknown node a. Then |||e||| ≤ ‖K^{-1/2}σ‖.

*Proof.* Since e ∈ V_h ⊂ H¹₀,

|||e|||² = R_v(e) = (div σ, e) = −(σ, ∇e) ≤ ‖K^{-1/2}σ‖ |||e|||. ∎

**Proposition 2 (two-level construction, any partition of unity; T9).**

*Setting:*
- {θ_i} are P1 functions with Σ_i θ_i ≡ 1 at every node.
- Each "floating" member i ∈ I_f satisfies θ_i ∈ V_c ⊂ V_h and has a connected patch ω_i (checked in the code; no disconnected patch occurred).
- Every other member has, in each connected component of its patch ω_i, at least one Dirichlet boundary edge.
- ρ ∈ V_c is the Galerkin solution a(ρ, φ) = R_v(φ), φ ∈ V_c.

*Construction.* With d_i := Π₀(r_h θ_i) − K∇ρ·∇θ_i, let σ_i ∈ RT0(T_b|ω_i) satisfy div σ_i = d_i, with zero normal flux on ∂ω_i (on ∂ω_i \ Γ_D for non-floating members), minimising ‖K^{-1/2}(σ_i + θ_i K∇ρ)‖.

*Claim.* Σ σ_i ∈ H(div) and div Σ σ_i = r_h.

*Proof.*
- Compatibility for i ∈ I_f: ∫ d_i = (r_h, θ_i) − a(ρ, θ_i) = R_v(θ_i) − R_v(θ_i) = 0. This uses θ_i ∈ V_h (lifting property) and θ_i ∈ V_c (Galerkin property).
- Summing: Σ d_i = Π₀ r_h − K∇ρ·∇1 = r_h, because r_h ∈ P0(T_b).
- Zero extension preserves H(div) conformity. ∎

**Proposition 3 (lifting-free construction; T10).** Same setting. For each member i, impose only the hat moments

(div σ_i, ψ_a)_{ω_i} = m_a^{(i)} := θ_i(a) (R_v − Aρ)_a + (θ_i K∇ρ, ∇ψ_a)   for every unknown node a touching ω_i,

and minimise ‖K^{-1/2}(σ_i + θ_i K∇ρ)‖. Then:
- (i) each local problem is consistent;
- (ii) σ := Σ σ_i satisfies (div σ, ψ_a) = R_a for every unknown a, so Lemma 1 applies **for every v ∈ V_h**;
- (iii) no residual lifting is used.

*Proof.*
- **(ii)** Members whose patch does not touch a contribute nothing on either side. Hence

  Σ_i m_a^{(i)} = (R_v − Aρ)_a Σ_i θ_i(a) + (K∇ρ, ∇ψ_a) = R_a − (Aρ)_a + (Aρ)_a = R_a.

- **(i)** For a floating member whose touching nodes are all unknown, the hats sum to 1 on ω_i, and zero normal flux gives Σ_a (div σ_i, ψ_a) = ∫ div σ_i = 0. The targets are consistent because

  Σ_a m_a^{(i)} = R_v(θ_i) − a(ρ, θ_i) + (θ_i K∇ρ, ∇1) = 0.

  One constraint is redundant and is dropped. If the patch touches Dirichlet nodes, their moments are unconstrained and absorb the mean.

  Surjectivity onto the consistent targets: element by element, the 3×3 moment map of P0(T_b) is invertible, so any distribution of moments is realised by some P0(T_b) function (mean zero when consistent); RT0 divergence maps onto mean-zero P0 on a connected patch.
- **(iii)** By construction, only R_v and local stiffness integrals of ρ enter. ∎

**Remarks.**
- The nodal split θ_i(a)(R − Aρ)_a is the partition-of-unity weighting familiar from restricted additive Schwarz.
- The fixed-lifting split (r_h θ_i, ψ_a) depends on how r_h is distributed inside each hat's support. This dependence is what inflates the noise effectivity by a factor of 15–30 (§7.1).
- The construction is linear in R_v, so its worst case over all inputs is a generalised eigenvalue. That is computable in principle; it was not computed here.

## 5 Efficiency: what is proved and what is open

**Factorisation.** For any construction (exact):

η_loc / |||e||| = (η_loc / η_joint) · (η_joint / |||e|||),

where η_joint is the global moment-constrained minimum. Since η_loc ≥ η_joint ≥ |||e|||, both factors are at least 1.

**Global factor (T11).** The worst case over **all** inputs is C_glob = λ_min(S, A)^{-1/2}, where:
- S = G M⁻¹ Gᵀ;
- M is the K⁻¹-weighted RT0(T_b) mass matrix;
- (Gσ)_a = (div σ, ψ_a).

It is input-independent and computable offline. Development values are in §7.4.

An equivalent form: integrating by parts with φ_λ = Σ λ_a ψ_a ∈ H¹₀,

λᵀSλ = ‖K^{-1/2} Π(K∇φ_λ)‖²,

where Π is the K⁻¹-weighted L² projection onto RT0(T_b). So C_glob⁻¹ is the cosine of the largest angle, in the K⁻¹ inner product, between the fluxes {K∇φ : φ ∈ V_h} and RT0(T_b).

**Proof idea (open).** K∇φ is constant on each L3 element and fails only normal continuity across element edges. The barycentric refinement has exactly one sub-element adjacent to each edge. A normal jump δ = K₁∇φ₁·n − K₂∇φ₂·n can therefore be absorbed on the **higher-coefficient** side at cost δ²/K_max, which is bounded by the local energy independently of the contrast. Turning this into a lower bound on ‖Πw‖/‖w‖ that does not depend on K is the lemma to prove.

**Localisation factor (T12), sketch.** Write τ_i := σ_i + θ_i K∇ρ. Then σ + K∇ρ = Σ τ_i, so

η_loc ≤ |||ρ||| + N_ov^{1/2} (Σ_i ‖K^{-1/2}τ_i‖²)^{1/2},   with |||ρ||| ≤ |||e|||.

The local functional seen by member i is

ℓ_i(q) = Σ_a θ_i(a) q(a) (R − Aρ)_a = a(e − ρ, I_h(θ_i q)),

and it vanishes on constants for floating members. Hence |ℓ_i(q)| ≤ |||e − ρ|||_{ω_i} C_θ,i |||q|||_{ω_i}, where

C_θ,i² := sup_q inf_c |||I_h(θ_i(q − c))|||² / |||q|||²_{ω_i}

(for boundary members, without the infimum and with q = 0 on Dirichlet nodes). This is the **partition-of-unity robustness index**: input-independent and computed by small local eigenproblems (`pu_robustness_index`).

**Hypothesis S (open).** ‖K^{-1/2}τ_i‖ ≤ C_S (sup_q ℓ_i(q)/|||q|||_{ω_i} + conformity term of θ_i K∇ρ). Under S,

η_loc ≲ (1 + N_ov C_S max_i C_θ,i) |||e||| + (conformity terms).

**Status of this route.** It identifies **what** must be robust: the index C_θ and the constant of S. It is not yet a proof: S and the conformity term need a real lemma. A clean theorem would replace S by a discrete weighted stability result, presumably under quasi-monotonicity inside each patch, which our data nearly satisfy.

## 6 Reference-certificate results (development; seed-1 operator 300; goal disc centred at (0.5, 0.45), radius 0.08)

Columns:
- **D radius / (e_u e_z):** the certified radius divided by the product of the true discrete energy errors, i.e. the efficiency of the product bound. It equals 1 at best.
- **C radius / |J|:** relative width of the continuous interval.

**Reference brackets:**
- channels: J(u_5) = 3.0281e-04 ± 2.27e-04;
- barriers: J(u_5) = 2.1092e-02 ± 5.88e-02.

| Family | Input pair | Method | D: radius / (e_u e_z) | D: contains J(u_h) | C: radius / abs(J) | C: contains reference bracket |
|---|---|---|---:|---|---:|---|
| channels | exact (u_h, z_h) | MVY multilevel, fixed lifting | n/a (exact) | yes | 2.6 | yes |
| channels | exact (u_h, z_h) | G2, lifting-free | n/a (exact) | yes | 2.6 | yes |
| channels | exact (u_h, z_h) | OA2+Z, lifting-free | n/a (exact) | yes | 2.6 | yes |
| channels | exact (u_h, z_h) | joint optimum (reference) | n/a (exact) | yes | 2.6 | yes |
| channels | coarse (u_c, z_c) | MVY multilevel, fixed lifting | 5.30 | yes | 9.7 | yes |
| channels | coarse (u_c, z_c) | G2, lifting-free | 2.04 | yes | 7.8 | yes |
| channels | coarse (u_c, z_c) | OA2+Z, lifting-free | 1.30 | yes | 5.3 | yes |
| channels | coarse (u_c, z_c) | joint optimum (reference) | 1.26 | yes | 5.0 | yes |
| channels | OA-VCB primal, coarse dual | MVY multilevel, fixed lifting | 52.66 | yes | 5.4 | yes |
| channels | OA-VCB primal, coarse dual | G2, lifting-free | 6.31 | yes | 4.5 | yes |
| channels | OA-VCB primal, coarse dual | OA2+Z, lifting-free | 1.30 | yes | 3.7 | yes |
| channels | OA-VCB primal, coarse dual | joint optimum (reference) | 1.26 | yes | 3.6 | yes |
| channels | Jacobi-PCG 30 (both) | MVY multilevel, fixed lifting | 587.47 | yes | 9.6 | yes |
| channels | Jacobi-PCG 30 (both) | G2, lifting-free | 25.78 | yes | 2.9 | yes |
| channels | Jacobi-PCG 30 (both) | OA2+Z, lifting-free | 1.08 | yes | 2.5 | yes |
| channels | Jacobi-PCG 30 (both) | joint optimum (reference) | 1.07 | yes | 2.3 | yes |
| channels | 10% noise (both) | MVY multilevel, fixed lifting | 1110.06 | yes | 28.2 | yes |
| channels | 10% noise (both) | G2, lifting-free | 5.78 | yes | 3.3 | yes |
| channels | 10% noise (both) | OA2+Z, lifting-free | 1.90 | yes | 2.6 | yes |
| channels | 10% noise (both) | joint optimum (reference) | 1.88 | yes | 2.7 | yes |
| barriers | exact (u_h, z_h) | MVY multilevel, fixed lifting | n/a (exact) | yes | 17.5 | yes |
| barriers | exact (u_h, z_h) | G2, lifting-free | n/a (exact) | yes | 17.5 | yes |
| barriers | exact (u_h, z_h) | OA2+Z, lifting-free | n/a (exact) | yes | 17.5 | yes |
| barriers | exact (u_h, z_h) | joint optimum (reference) | n/a (exact) | yes | 17.5 | yes |
| barriers | coarse (u_c, z_c) | MVY multilevel, fixed lifting | 9.42 | yes | 145.8 | yes |
| barriers | coarse (u_c, z_c) | G2, lifting-free | 2.96 | yes | 111.3 | yes |
| barriers | coarse (u_c, z_c) | OA2+Z, lifting-free | 1.26 | yes | 50.0 | yes |
| barriers | coarse (u_c, z_c) | joint optimum (reference) | 1.24 | yes | 49.5 | yes |
| barriers | OA-VCB primal, coarse dual | MVY multilevel, fixed lifting | 46.85 | yes | 132.7 | yes |
| barriers | OA-VCB primal, coarse dual | G2, lifting-free | 1.92 | yes | 33.3 | yes |
| barriers | OA-VCB primal, coarse dual | OA2+Z, lifting-free | 1.20 | yes | 30.9 | yes |
| barriers | OA-VCB primal, coarse dual | joint optimum (reference) | 1.19 | yes | 30.7 | yes |
| barriers | Jacobi-PCG 30 (both) | MVY multilevel, fixed lifting | 62.52 | yes | 160.7 | yes |
| barriers | Jacobi-PCG 30 (both) | G2, lifting-free | 2.92 | yes | 48.5 | yes |
| barriers | Jacobi-PCG 30 (both) | OA2+Z, lifting-free | 1.05 | yes | 44.5 | yes |
| barriers | Jacobi-PCG 30 (both) | joint optimum (reference) | 1.04 | yes | 44.4 | yes |
| barriers | 10% noise (both) | MVY multilevel, fixed lifting | 332.45 | yes | 68.9 | yes |
| barriers | 10% noise (both) | G2, lifting-free | 2.04 | yes | 18.6 | yes |
| barriers | 10% noise (both) | OA2+Z, lifting-free | 1.82 | yes | 18.4 | yes |
| barriers | 10% noise (both) | joint optimum (reference) | 1.80 | yes | 18.4 | yes |

**Reading.**
- **(D)** The lifting-free operator-adapted certificate tracks the global joint optimum on every input pair. The classical multilevel certificate is guaranteed but overestimates the product by up to three orders of magnitude for oscillatory or smooth iterative errors.
- **(C)** Every interval is guaranteed, and every one is too wide to be useful at L3 (radius ≥ 2.6 |J| even for exact solves). This is the discretisation error, not the algebraic part.
- On barriers, 30 Jacobi-PCG steps barely reduce the error (relative error 0.92, §7.1). Those rows test the certificate on near-initial errors, not on converged iterates.

## 7 Development diagnostics (operator-300 geometry only)

### 7.1 How much is the lifting? (contrast 10⁴; global optima)

| Family | Input | Rel. algebraic error | Best flux, fixed lifting | Best flux, lifting optimised (joint) |
|---|---|---:|---:|---:|
| channels | u_c | 0.3 | 1.94 | 1.15 |
| channels | 30 Jacobi-PCG steps | 0.031 | 1.27 | 1.03 |
| channels | u_h + 10% noise | 0.1 | 29.93 | 1.38 |
| barriers | u_c | 0.52 | 4.37 | 1.12 |
| barriers | 30 Jacobi-PCG steps | 0.92 | 1.19 | 1.02 |
| barriers | u_h + 10% noise | 0.1 | 15.61 | 1.34 |

### 7.2 Contrast sweep (same geometry; channels K ∈ {1, c}, barriers K ∈ {1, 1/c}; algebraic effectivity η_alg / |||u_h − v|||)

| Family | Contrast | Input | MVY multilevel (fixed lifting) | Best flux, fixed lifting | G2 lifting-free | OA2+Z lifting-free | Joint optimum |
|---|---:|---|---:|---:|---:|---:|---:|
| channels | 1e+00 | PCG30 | 1.13 | 1.06 | 1.04 | 1.04 | 1.04 |
| channels | 1e+00 | u_c | 1.40 | 1.36 | 1.29 | 1.25 | 1.25 |
| channels | 1e+01 | PCG30 | 1.25 | 1.07 | 1.04 | 1.03 | 1.03 |
| channels | 1e+01 | u_c | 1.60 | 1.50 | 1.31 | 1.22 | 1.21 |
| channels | 1e+02 | PCG30 | 2.74 | 1.09 | 1.09 | 1.03 | 1.02 |
| channels | 1e+02 | u_c | 2.02 | 1.85 | 1.40 | 1.17 | 1.17 |
| channels | 1e+03 | PCG30 | 8.77 | 1.11 | 1.72 | 1.02 | 1.02 |
| channels | 1e+03 | u_c | 2.12 | 1.93 | 1.42 | 1.16 | 1.15 |
| channels | 1e+04 | PCG30 | 22.05 | 1.27 | 3.68 | 1.03 | 1.03 |
| channels | 1e+04 | u_c | 2.13 | 1.94 | 1.42 | 1.17 | 1.15 |
| barriers | 1e+00 | PCG30 | 1.13 | 1.06 | 1.04 | 1.04 | 1.04 |
| barriers | 1e+00 | u_c | 1.40 | 1.36 | 1.29 | 1.25 | 1.25 |
| barriers | 1e+01 | PCG30 | 1.22 | 1.06 | 1.05 | 1.04 | 1.03 |
| barriers | 1e+01 | u_c | 1.59 | 1.49 | 1.31 | 1.23 | 1.22 |
| barriers | 1e+02 | PCG30 | 2.05 | 1.10 | 1.06 | 1.03 | 1.03 |
| barriers | 1e+02 | u_c | 2.64 | 2.40 | 1.63 | 1.20 | 1.19 |
| barriers | 1e+03 | PCG30 | 2.91 | 1.17 | 1.09 | 1.02 | 1.02 |
| barriers | 1e+03 | u_c | 4.35 | 3.98 | 2.28 | 1.15 | 1.14 |
| barriers | 1e+04 | PCG30 | 3.09 | 1.19 | 1.10 | 1.02 | 1.02 |
| barriers | 1e+04 | u_c | 4.78 | 4.37 | 2.45 | 1.12 | 1.12 |

### 7.3 Robustness index C_θ (input-independent) against contrast

"Floating" members are those in V_c. G2/G1 boundary members are the coarse Dirichlet hats. OA2 boundary members are fine Dirichlet hats, with index 0 by construction.

| Family | Contrast | G2 floating max (p95) | G2 boundary max | G1 floating max (p95) | G1 boundary max | OA2 floating max (p95) |
|---|---:|---:|---:|---:|---:|---:|
| channels | 1e+00 | 1.17 (0.98) | 1.12 | 1.13 (1.01) | 1.20 | 1.27 (0.96) |
| channels | 1e+01 | 1.35 (1.08) | 1.30 | 1.45 (1.16) | 1.55 | 1.27 (0.96) |
| channels | 1e+02 | 3.50 (1.16) | 3.66 | 4.18 (2.18) | 4.51 | 1.29 (0.97) |
| channels | 1e+03 | 10.80 (1.17) | 11.42 | 13.09 (6.70) | 14.16 | 2.91 (1.02) |
| channels | 1e+04 | 34.06 (1.17) | 36.06 | 41.36 (21.14) | 44.74 | 3.64 (1.07) |
| channels | 1e+06 | 340.50 (1.17) | 360.52 | 413.55 (211.30) | 447.39 | 36.31 (1.09) |
| barriers | 1e+00 | 1.17 (0.98) | 1.12 | 1.13 (1.01) | 1.20 | 1.27 (0.96) |
| barriers | 1e+01 | 1.27 (1.07) | 1.56 | 1.55 (1.31) | 1.21 | 1.27 (0.96) |
| barriers | 1e+02 | 1.57 (1.18) | 4.40 | 4.43 (3.60) | 1.66 | 1.27 (0.97) |
| barriers | 1e+03 | 1.84 (1.20) | 13.74 | 13.84 (11.19) | 5.09 | 1.27 (1.02) |
| barriers | 1e+04 | 1.87 (1.20) | 43.40 | 43.70 (35.34) | 16.05 | 1.27 (1.07) |
| barriers | 1e+06 | 1.88 (1.21) | 433.93 | 436.99 (353.37) | 160.49 | 1.27 (1.11) |

### 7.4 Global factor C_glob (worst case over all inputs)

| Family | Contrast | C_glob (worst case over all inputs) |
|---|---:|---:|
| channels | 1e+00 | 2.070 |
| channels | 1e+02 | 2.133 |
| channels | 1e+04 | 2.137 |
| channels | 1e+06 | 2.137 |
| barriers | 1e+00 | 2.070 |
| barriers | 1e+02 | 2.059 |
| barriers | 1e+04 | 2.058 |
| barriers | 1e+06 | 2.058 |

### 7.5 Lifting-free against moment-relaxed (contrast 10⁴)

| Family | Input | Joint optimum | G2 relaxed | G2 lifting-free | G1 relaxed | G1 lifting-free | OA2+Z relaxed | OA2+Z lifting-free |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| channels | u_c | 1.151 | 1.420 | 1.422 | 1.193 | 1.190 | 1.179 | 1.165 |
| channels | 30 Jacobi-PCG | 1.029 | 3.684 | 3.681 | 6.249 | 6.248 | 1.034 | 1.033 |
| channels | u_h + 10% noise | 1.375 | 4.968 | 1.816 | 3.404 | 2.937 | 2.362 | 1.378 |
| barriers | u_c | 1.117 | 2.601 | 2.452 | 1.184 | 1.166 | 1.412 | 1.124 |
| barriers | 30 Jacobi-PCG | 1.020 | 1.109 | 1.104 | 2.546 | 2.545 | 1.024 | 1.024 |
| barriers | u_h + 10% noise | 1.344 | 2.754 | 1.508 | 2.131 | 1.928 | 1.846 | 1.349 |

### 7.6 Stress at contrast 10⁶

| Family | Input | Rel. algebraic error | G2 lifting-free | OA2+Z lifting-free | Joint optimum |
|---|---|---:|---:|---:|---:|
| channels | PCG10 | 0.3 | 1.21 | 1.061 | 1.058 |
| channels | PCG30 | 0.02 | 5.52 | 1.050 | 1.043 |
| channels | PCG100 | 0.0002 | 71.53 | 1.247 | 1.028 |
| channels | u_c | 0.3 | 1.42 | 1.167 | 1.151 |
| channels | noise 10% | 0.1 | 11.60 | 1.406 | 1.375 |
| barriers | PCG10 | 1 | 1.32 | 1.025 | 1.020 |
| barriers | PCG30 | 1 | 1.10 | 1.025 | 1.020 |
| barriers | PCG100 | 1 | 4.23 | 1.025 | 1.020 |
| barriers | u_c | 0.6 | 2.47 | 1.126 | 1.114 |
| barriers | noise 10% | 0.1 | 6.32 | 1.349 | 1.344 |

## 8 Prior art: what is known and what may be new

Read in full: PRVW (CMAME 2020), MVY (JCAM 2020, appendix via the primary manuscript), Papež–Vohralík (Numer. Algorithms 2022), MPV (SINUM 2020), Bastidas Olivares–Beni Hamad–Vohralík–Yotov (CMAME 2025), Li (arXiv 2609.27817v2), Graham–Lechner–Scheichl (Numer. Math. 2007), Vohralík (JSC 2011), Cai–Zhang (SINUM 2012), Sudarshan–Amaratunga–Grätsch (IJNME 2006). The others were checked at abstract level.

| Work | What it covers | Relation to the candidate |
|---|---|---|
| MVY 2020 | Guaranteed goal bounds with inexact solvers; multilevel algebraic flux (App. 8); SPE10 test; no robustness claim | **Closest overall.** Our identity, Prager–Synge use and multilevel reference come from here |
| PRVW 2020 | Hierarchical H(div) lifting of r_h; mass balance "in any situation"; efficiency for Poisson; effectivity about 1.0 for PCG iterates | The classical algebraic flux; lifting r_h (6.7); Poisson theory |
| Papež–Vohralík 2022 | Cheaper liftings (large/small patch, single element, explicit RT0); efficiency constants **depend on λ_max/λ_min of K** | Confirms the gap: contrast-robust efficiency is not provided |
| MPV 2020; MPVY (mixed, 2024/26) | p-robust multilevel estimator and solver; MPVY's β depends on Λ_max/Λ_min | Same gap, for p rather than contrast |
| Bastidas Olivares et al. 2025 | Mixed FE nonoverlapping DD, guaranteed algebraic bounds, coarse balancing; robust iteration counts for jumps **aligned with the coarse mesh** (Table 1, 10¹–10⁸) | Our coefficients are **not** coarse-aligned (channels cross coarse elements): the GLS regime |
| Graham–Lechner–Scheichl 2007 | Robustness indicators π(α), γ(α) for energy-minimising partitions of unity in additive Schwarz preconditioners | Origin of the mechanism; our index is an estimator-side analogue (sharp, eigenvalue-based, not L∞) |
| Li 2026 | No finite-range local estimator can be robust (checkerboard cross point) | Limits (C); does not cover (D), whose estimators are non-local by construction |
| Vohralík 2011; Cai–Zhang 2012; He 2026 | Robust discretisation estimators under stated conditions | Background for T14 |
| Sudarshan–Amaratunga–Grätsch 2006 | Operator-customised wavelets for goal-oriented **estimation and adaptivity** (two-level estimates, not guaranteed bounds) | The operator-adapted lineage this project continues; Route C would add guarantees |
| Fanaskov–Rudikov–Oseledets 2024; Ernst–Rekatsinas–Urban (2026 rev.); RBNO/FOSLS 2025; Roth–Schröder–Wick 2022 | Learned or certified approximations: functional majorants, domain extension, FOSLS losses, neural DWR | Learning is not the contribution here; none treats contrast-robust algebraic certificates |
| arXiv 2607.11297 | Interface coarsening with an equilibrated estimator | **Abstract not read** (rate-limited) |

**Not found in the papers read:**
- **(a)** lifting-free, moment-constrained local flux problems for algebraic error;
- **(b)** operator-adapted (energy-minimising) partitions of unity inside equilibrated algebraic estimators;
- **(c)** a computable efficiency index for such estimators;
- **(d)** contrast-robust efficiency of guaranteed algebraic-error bounds for coefficients not resolved by the coarse mesh.

**Known ingredients:**
- partition-of-unity flux localisation (Braess–Schöberl, PRVW);
- Shepard normalisation;
- energy-minimising coarse spaces and their robustness indicators (GLS);
- spectral enrichment (GenEO) as the likely remedy for the channel maximum.

**Contribution threshold.** A publishable result needs at least (d) proved under stated topology hypotheses, or (d) demonstrated on fresh data together with T11/T12 made rigorous.

## 9 Pilot plan (deliverable 3) — draft for review, not run

**Data:**
- Fresh geometries, to be allocated and recorded before generation: rng 2026101801 (channels) and 2026101802 (barriers), ids 0–7.
- The draw does not depend on the contrast value, so each geometry is reused at every contrast (paired design).
- No development operator is reused.

**Contrasts:**
- Primary: c ∈ {1, 10², 10⁴}.
- Stress (secondary): 10⁶.

**Goals:** three discs of radius 0.08 centred at (0.5, 0.45), (0.15, 0.15) and (0.85, 0.85). All are clear of the holes, checked against the hole geometry.

**Inputs** (primal and dual alike, unless noted):
- exact;
- coarse Galerkin u_c / z_c;
- Jacobi-PCG with 10, 30 and 100 steps;
- 10% white noise;
- 10% smooth perturbation: A⁻¹ applied to a random load, normalised;
- OA-VCB one pass (primal, c = 10⁴ only, the trained contrast; secondary).

**Methods:**

| ID | Method | Purpose |
|---|---|---|
| M0 | MVY multilevel, fixed lifting | Classical reference |
| M0′ | PRVW Def. 4.3 variant (no ρ₀ term above level 1) | Fidelity check of M0 |
| M1 | G2, lifting-free | Lifting-free control without operator adaptation |
| M2 | G1, lifting-free | Same, larger patches (patch-size control) |
| M3 | OA2 + Z, lifting-free | Candidate |
| M4 | OA2 without Z, lifting-free | Isolates the near-kernel |
| M5 | OA2 + Z, fixed lifting | Isolates the lifting |
| M6 | Joint optimum | Accuracy ceiling (global) |

**Primary endpoints** (per family, contrast and method, over operators):
- **E1:** median of the worst algebraic effectivity over inputs, primal and dual.
- **E2:** E1 at 10⁴ divided by E1 at 1.
- **E3:** median and 90th percentile of η_method / η_joint.

**Secondary endpoints:**
- D-radius / (e_u e_z), and interval containment (must be 100%; any failure stops the pilot for debugging);
- index distributions (max, p95) for G2 and OA2, and the members achieving the maximum, with their geometry;
- C_glob;
- the (C) certificate at 10⁴ on 4 operators per family (informative only);
- **certified-stopping experiment:** Jacobi-PCG iterations needed until the D radius is ≤ 10⁻³ |J_h| and ≤ 10⁻⁶ |J_h|, with M0 against M3 as the stopping test. This is ASTRA's "cost to a certified tolerance". The estimator is evaluated at iterations 5, 10, 20, 40, 80, 160 and 320;
- work counts: number and total size of local problems, coarse dimension, and approximate MACs relative to one V-cycle. No wall-clock speed claim at N = 3486.

**Pre-declared decision rules:**
- **Candidate supported** if all of the following hold:
  - M3 has E1 ≤ 1.5 at every primary contrast in both families;
  - E2 ≤ 1.25;
  - E3 median ≤ 1.10 and 90th percentile ≤ 1.5;
  - M0 has E2 ≥ 3 in at least one family;
  - M1 and M2 both have E1(10⁴) ≥ 1.5 × M3's in at least one family. Without this, operator adaptation is not what matters.
- **Reduced contribution** if M3 is within 10% of M1 or M2 everywhere. The finding is then lifting-free localisation only.
- **Stop or reframe** if:
  - any unexplained containment failure;
  - M3 has E2 > 2;
  - M3's E3 median > 1.5;
  - the development advantage does not reproduce on fresh operators.

**Expected cost:** about 4–6 CPU hours on this two-core workspace, run in the background. No GPU and no training.

## 10 Go/no-go criteria (deliverable 4; decided after the pilot)

**Proceed to a full programme only if:**
- the pilot supports the candidate, and
- at least one of the following holds:
  - (i) a proof of T11 and T12 (or of a variant) appears feasible under stated topology hypotheses;
  - (ii) an index-guided enrichment removes the channel maximum without large coarse growth;
  - (iii) the certified-stopping experiment shows a material reduction in iterations to a certified tolerance against M0.

**Stop or reframe if:**
- the advantage is lifting-free localisation alone and already implicit in the literature on a closer reading; or
- (C) is the only route to a useful result. That would need goal-oriented adaptivity, where MVY and classical work dominate.

## 11 Questions for ASTRA

1. Is targeting (D), certified outputs of the discrete system, acceptable as Route C's core, with (C) reported but uninformative at L3?
2. Is M0/M0′ plus the global joint optimum M6 the right set of classical baselines? Should a Gauss–Radau CG bound be added? It needs a guaranteed λ_min bound, which is itself contrast-sensitive.
3. Are the endpoints and thresholds in §9 adequate, and should the pilot use more operators (8 per family is the present budget)?
4. Should the index-guided enrichment (GenEO-type) be part of this pilot or a follow-up?

## Records

- `ext/routec/rc_fem.py`, `rc_goal.py`: certificate code.
- `ext/routec/rc_facts.py`, `rc_feasibility.py`: discretisation facts.
- `ext/routec/rc_dev_goal.py` → `dev_goal_{channels,barriers}.json/.log`: goal certificates.
- `ext/routec/rc_joint_oracle.py`, `rc_mvy_contrast.py`, `rc_test_liftfree.py`, `rc_stress_1e6.py`, `rc_index_sweep.py`, `rc_global_stability.py` → `logs_*.log`, `global_stability.json`.
- `ext/routec/rc_scoping_tables.py` → `scoping_tables.json`: every table above is generated from these records.
- `ext/routec/rc_fem_v1_p0lift.py`: first version (global P0 lifting), kept for the record of §3.
