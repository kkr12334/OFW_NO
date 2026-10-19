# Addendum 1 to the Route C scoping note: mesh-size arm for the pilot

18 October 2026. Claude, for Kumar and ASTRA. Applies to `ROUTE_C_SCOPING_2026-10-18.md` (SHA-256 `d9486e26…`), which is unchanged. Written after that note went to ASTRA and before any pilot run. No fresh operator has been generated.

Proposed addition to §9 for ASTRA to accept, modify or reject with the rest of the pilot design:

**Mesh-size arm (secondary; added 18 October before any review or run):**
- **Why:** all evidence so far is on L3 (3486 unknowns). Robustness in h matters as much as robustness in contrast.
- **Data:** L4 meshes (the same coarse mesh refined once more), first 2 geometries per family from the pilot draws, K inherited per element from the L3 draw.
- **Contrasts:** 1 and 10⁴.
- **Methods:** M0, M1, M3, M6.
- **Inputs:** coarse Galerkin, Jacobi-PCG 30 and 100, 10% white noise.
- **Reported:** the same effectivities, the index (max and p95), and C_glob if affordable.
- **Reading rule:** a ratio of M3's E1 at L4 to its E1 at L3 ≤ 1.25 is read as h-robust in this range. Nothing is confirmed from 2 operators.

**Cost:** to be measured by a smoke test on development data before launch; reported with the launch request.
