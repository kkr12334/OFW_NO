"""Stock-take (3 Oct): the cheapest CLASSICAL route to the Allen-Cahn surrogate's accuracy.  L3, the GPU setting
(eps 0.015, T 5, S 2), 16 test samples of the default pool.  Reference: the training target, fine mesh, dt 0.01.
Classical alternatives: the same fine scheme with a larger time step (fewer steps, one factorisation), and the coarse
(level-2) scheme with a larger step (the network's input could be made cheaper the same way).  Batched timings
(all 16 samples as columns, per sample), as in the data generation.
Trained-model reference (GPU, seed 0): gnn 3.46e-2 / H1 0.099, oa-ll-epm 3.65e-2 / 0.104; coarse alone 0.155 / 0.549."""
import sys, os, time, warnings, numpy as np
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from allen_cahn import neumann_hierarchy, ACOperators, grf_initial
L, k0, eps, T, S = 3, 2, 0.015, 5.0, 2.0
H = neumann_hierarchy(L); OP = ACOperators(H); K = OP.K[L]
U0 = grf_initial(H, 400, np.random.default_rng(5000))[300:316]
def errs(p, t):
    e = p - t
    return (np.mean(np.linalg.norm(e, axis=1) / np.linalg.norm(t, axis=1)),
            np.mean(np.sqrt(np.einsum('bi,bi->b', e, (K @ e.T).T) / np.einsum('bi,bi->b', t, (K @ t.T).T))))
def timed(fn):
    fn(); t0 = time.perf_counter(); out = fn(); return out, (time.perf_counter() - t0) / len(U0) * 1e3
ref, t_ref = timed(lambda: OP.solve(U0, L, eps, T, 0.01, S))
print(f"fine reference dt 0.01: {t_ref:.1f} ms/sample (batched)")
for dt in (0.02, 0.05, 0.1, 0.2, 0.5):
    u, t = timed(lambda: OP.solve(U0, L, eps, T, dt, S)); r = errs(u, ref)
    print(f"  fine mesh, dt {dt:<4}: rel-L2 {r[0]:.3e}  H1 {r[1]:.3e}   {t:6.1f} ms/sample")
for dt in (0.01, 0.05, 0.1):
    u, t = timed(lambda: OP.coarse_solve(U0, k0, eps, T, dt, S)); r = errs(u, ref)
    print(f"  coarse (level {k0}), dt {dt:<4}: rel-L2 {r[0]:.3e}  H1 {r[1]:.3e}   {t:6.1f} ms/sample")

# a stronger classical baseline: Strang splitting, EXACT reaction flow u' = u - u^3, u(t) = u / sqrt(u^2 + (1-u^2) e^{-2t}),
# diffusion by Crank-Nicolson with the lumped mass (second order overall), one factorisation per dt
import scipy.sparse as sp
from scipy.sparse.linalg import splu
def strang(U, dt, k=L):
    m = OP.m[k][:, None]; Kk = OP.K[k]
    lu = splu((sp.diags(OP.m[k]) + 0.5 * dt * eps ** 2 * Kk).tocsc()); Bm = sp.diags(OP.m[k]) - 0.5 * dt * eps ** 2 * Kk
    R = lambda V, t: V / np.sqrt(V ** 2 + (1 - V ** 2) * np.exp(-2 * t))
    V = U.T.copy()
    for _ in range(int(round(T / dt))):
        V = R(V, dt / 2); V = lu.solve(Bm @ V); V = R(V, dt / 2)
    return V.T
for dt in (0.05, 0.1, 0.2, 0.5):
    u, t = timed(lambda: strang(U0, dt)); r = errs(u, ref)
    print(f"  fine mesh, Strang (exact reaction + CN), dt {dt:<4}: rel-L2 {r[0]:.3e}  H1 {r[1]:.3e}   {t:6.1f} ms/sample")
