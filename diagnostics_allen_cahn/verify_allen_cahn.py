"""Verification of the Allen-Cahn solver (allen_cahn.py) before any training.

1. operators: K 1 = 0 (Neumann), sum of lumped mass = domain area, Galerkin K_k = K assembled on the level-k mesh
2. spatially constant data: the scheme reduces to an ODE scheme for u' = u - u^3 -> first order in dt vs the exact solution
3. convergence in dt (fixed mesh) against a dt/32 reference: first order
4. convergence in h (fixed dt) on the nested meshes against level 5 (prolonged exactly by P)
5. energy decay: E(u^{n+1}) <= E(u^n) at every step for every sample, also at dt = 0.2 (unconditional stability)
"""
import sys, time, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from allen_cahn import neumann_hierarchy, ACOperators, grf_initial

EPS, T = 0.015, 5.0
t00 = time.time()
H = neumann_hierarchy(5); op = ACOperators(H)
print(f"levels 0..5 unknowns {[len(H.nodes[k]) for k in range(6)]}, eps {EPS}, T {T}")

print("\n1. operators")
import numpy.linalg as la
from coarse_meshes import perforated_square
holes = ((0.30, 0.32, 0.12), (0.68, 0.62, 0.15), (0.30, 0.72, 0.10))
print(f"   max |K 1| = {abs(op.K[5] @ np.ones(op.K[5].shape[0])).max():.1e}")
# domain area: shoelace over the elements of the finest mesh (polygonal holes)
xy = H.coords[H.elem]
area = 0.5 * abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1])).sum()
print(f"   sum lumped mass: level 5 {op.m[5].sum():.12f}, level 0 {op.m[0].sum():.12f}, element area {area:.12f}")
for k in (1, 2, 3):
    Hk = neumann_hierarchy(k); ok = ACOperators(Hk)
    same = np.array_equal(Hk.nodes[k], H.nodes[k])
    print(f"   level {k}: same node ordering {same}; |K_k(Galerkin) - K_k(assembled)| = {abs(op.K[k] - ok.K[k]).max():.1e}; "
          f"|m_k - m_k(assembled)| = {abs(op.m[k] - ok.m[k]).max():.1e}")

print("\n2. constant initial data (reduces to u' = u - u^3; exact u = c e^t / sqrt(1 + c^2 (e^{2t} - 1)))")
c = 0.1; ex = c * np.exp(T) / np.sqrt(1 + c ** 2 * (np.exp(2 * T) - 1))
errs = []
for dt in (0.1, 0.05, 0.025, 0.0125):
    u = op.solve(np.full(len(H.nodes[2]), c), 2, EPS, T, dt)
    errs.append(abs(u.mean() - ex)); spread = u.max() - u.min()
    print(f"   dt {dt:<7} u(T) {u.mean():.6f} (exact {ex:.6f}), error {errs[-1]:.2e}, spatial spread {spread:.1e}")
print(f"   observed orders {np.round(np.log2(np.array(errs[:-1]) / np.array(errs[1:])), 2)}")

rng = np.random.default_rng(0)
U0 = grf_initial(H, 4, rng)
print("\n3. convergence in dt (level 3 mesh, 4 GRF samples), reference dt = 0.3125e-3")
k = 3; u0 = U0[:, :len(H.nodes[k])]
ref = op.solve(u0, k, EPS, T, 0.0003125)
errs = []
for dt in (0.02, 0.01, 0.005, 0.0025):
    e = op.solve(u0, k, EPS, T, dt) - ref
    errs.append(np.mean(np.sqrt(op.l2_sq(e, k) / op.l2_sq(ref, k))))
    print(f"   dt {dt:<7} rel L2 error {errs[-1]:.3e}")
print(f"   observed orders {np.round(np.log2(np.array(errs[:-1]) / np.array(errs[1:])), 2)}")

print("\n4. convergence in h (dt 0.005, against level 5 prolonged), GRF samples")
for eps in (0.05, EPS):
    ref = op.solve(U0, 5, eps, T, 0.005)
    l2, h1 = [], []
    for k in (1, 2, 3, 4):
        uk = op.solve(U0[:, :len(H.nodes[k])], k, eps, T, 0.005)
        e = (op.Pc[k] @ uk.T).T - ref
        l2.append(np.mean(np.sqrt(op.l2_sq(e) / op.l2_sq(ref)))); h1.append(np.mean(np.sqrt(op.h1_sq(e) / op.h1_sq(ref))))
    print(f"   eps {eps}: level 1-4 rel L2 {['%.2e' % x for x in l2]}  rel H1-semi {['%.2e' % x for x in h1]}")
    print(f"            orders L2 {np.round(np.log2(np.array(l2[:-1]) / np.array(l2[1:])), 2)}  H1 {np.round(np.log2(np.array(h1[:-1]) / np.array(h1[1:])), 2)}")

print("\n5. energy decay (level 3, 4 samples)")
for dt in (0.01, 0.2):
    u, E = op.solve(u0, 3, EPS, T, dt, record_energy=True)
    dE = np.diff(E, axis=0)
    print(f"   dt {dt}: E(0) {E[0].mean():.4e} -> E(T) {E[-1].mean():.4e}; max step increase {dE.max():.1e} "
          f"(relative {dE.max() / E[0].min():.1e}); max |u| {abs(u).max():.4f}")
print(f"\ntotal {time.time() - t00:.0f}s")
