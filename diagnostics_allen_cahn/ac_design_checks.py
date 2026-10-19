"""Numerical checks behind the Allen-Cahn design of oa-ll-op (3 Oct).  Level 3, the GPU setting (eps 0.015, T 5,
dt 0.01, S 2, GRF u0, data stream of train_allen_cahn.py with data seed 0), test samples 300.. of the pool.

  1. Symmetry: the fine and coarse maps are odd, S(-u0) = -S(u0) (f~ odd, scheme linear otherwise) -> exact.
  2. One fine time step is LOCAL: D^-1 A_h = I + O(small) for A_h = (1 + S dt) M + dt eps^2 K, so a few Jacobi sweeps
     replace the solve; measured over the whole trajectory (s sweeps per step instead of the LU solve).
  3. Where the correction sits: share of ||u_h(T) - u_c(T)||^2 in the interface band |u_c(T)| < 0.9.
  4. Two-level-in-time decomposition of the coarse error: coarse solve to T - tau, then the FINE scheme for the last
     tau (cost tau / T of a fine solve).  tau = T leaves only the initial-data interpolation error.
  5. The ceiling of what the network class can represent with nu = 4 smoother steps: coarse to T - tau, then 4 fine
     steps of size tau / 4, each ONE semi-implicit step with s Jacobi sweeps (a stencil + a pointwise cubic).
  6. The same relaxation started from the coarse FINAL state u_c(T) (no extra input; time shifts to T + tau_r).
  7. How nonlinear the response to the initial-data defect e0 = u0 - P R u0 is (no residual scaling if it is).
  8. Spread of the per-node factor q_i = m_i / (A_h)_ii of a Jacobi-swept step (must be operator data if it varies).
  9. The ceiling at the step fixed by theory (profile relaxation rate 3/2 -> nu * dt_net = 2/3).
README 3 Oct (Allen-Cahn design).
"""
import sys, os, time, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import splu
from allen_cahn import neumann_hierarchy, ACOperators, grf_initial, f_tilde

L, k0, eps, T, dt, S = 3, 2, 0.015, 5.0, 0.01, 2.0
H = neumann_hierarchy(L); OP = ACOperators(H); nI = len(H.nodes[L])
rng = np.random.default_rng(5000)
U0 = grf_initial(H, 400, rng)[300:316]                       # 16 test samples of the default pool
m, K = OP.m[L], OP.K[L]; P = OP.Pc[k0]; n0 = len(H.nodes[k0])


def stepper(k, dts):
    A = sp.diags((1 + S * dts) * OP.m[k]) + dts * eps ** 2 * OP.K[k]
    return splu(A.tocsc()), A.tocsr()


def run(U, k, t, dts, sweeps=None):
    """advance columns U (n_k, B) on level k for time t with step dts; exact solve or `sweeps` Jacobi sweeps"""
    lu, A = stepper(k, dts); mk = OP.m[k][:, None]; dg = A.diagonal()[:, None]
    for _ in range(int(round(t / dts))):
        b = mk * ((1 + S * dts) * U - dts * f_tilde(U))
        if sweeps is None: U = lu.solve(b)
        else:
            x = U.copy()
            for _ in range(sweeps): x = x + (b - A @ x) / dg
            U = x
    return U


def errs(pred, true):
    e = pred - true
    rl = np.mean(np.linalg.norm(e, axis=0) / np.linalg.norm(true, axis=0))
    h1 = np.mean(np.sqrt(np.einsum('ib,ib->b', e, K @ e) / np.einsum('ib,ib->b', true, K @ true)))
    return rl, h1


t0 = time.time()
Uf = run(U0.T, L, T, dt)                                    # fine reference u_h(T), (nI, B)
print(f"level {L}: {nI} unknowns, coarse level {k0} ({n0} nodes), {U0.shape[0]} test samples; fine solve {time.time() - t0:.1f}s")

# 1. symmetry
Ufm = run(-U0[:4].T, L, T, dt); Ucp = run(U0[:4, :n0].T, k0, T, dt); Ucm = run(-U0[:4, :n0].T, k0, T, dt)
print(f"1. odd map: fine |S(-u0) + S(u0)| = {np.abs(Ufm + Uf[:, :4]).max():.1e}, coarse {np.abs(Ucm + Ucp).max():.1e}")

# 2. locality of one time step
lu, A = stepper(L, dt); dg = A.diagonal()
offd = np.abs(A - sp.diags(dg)).sum(1).A1 / dg
print(f"2. dt {dt}: row sums of |D^-1 A_h - I|: max {offd.max():.3f}, median {np.median(offd):.3f}  (Jacobi contraction <= max)")
for s in (1, 2, 3):
    print(f"   whole trajectory with {s} Jacobi sweep(s) per step instead of the solve: error vs u_h(T) "
          "rel-L2 %.1e, H1 %.1e" % errs(run(U0.T, L, T, dt, sweeps=s), Uf))

# coarse trajectory, recorded at the times needed
taus = [0.04, 0.2, 0.4, 0.8, 1.6, 5.0]
Vc = {}; Uc = U0[:, :n0].T.copy(); lu_c, _ = stepper(k0, dt); mc = OP.m[k0][:, None]; N = int(round(T / dt))
need = {int(round((T - tau) / dt)): tau for tau in taus}
for n in range(N + 1):
    if n in need: Vc[need[n]] = P @ Uc
    if n == N: Vc[0.0] = P @ Uc
    if n < N: Uc = lu_c.solve(mc * ((1 + S * dt) * Uc - dt * f_tilde(Uc)))
v = Vc[0.0]
print("   coarse alone: rel-L2 %.3f, H1 %.3f" % errs(v, Uf))

# 3. where the correction sits
e = Uf - v; band = np.abs(v) < 0.9
share = np.sum(m[:, None] * e ** 2 * band, 0) / np.sum(m[:, None] * e ** 2, 0)
area = np.sum(m[:, None] * band, 0) / m.sum()
print(f"3. interface band |u_c(T)| < 0.9: {area.mean():.1%} of the area carries {share.mean():.1%} of ||e||^2 (mass-weighted)")

# 4. two-level-in-time decomposition
print("4. coarse to T - tau, then the fine scheme (dt 0.01) for tau:")
for tau in taus:
    r_ = errs(run(Vc[tau], L, tau, dt), Uf)
    print(f"   tau {tau:>4}: rel-L2 {r_[0]:.3f}  H1 {r_[1]:.3f}   (fine cost {tau / T:.0%} of a fine solve)")

# 5. what nu = 4 network steps can represent
print("5. coarse to T - tau, then 4 fine steps of tau/4 (one semi-implicit step, s Jacobi sweeps each):")
for tau in (0.04, 0.2, 0.4, 0.8):
    row = [errs(run(Vc[tau], L, tau, tau / 4, sweeps=s), Uf) for s in (1, 2)]
    ex = errs(run(Vc[tau], L, tau, tau / 4), Uf)
    print(f"   tau {tau:>4}: 1 sweep rel-L2 {row[0][0]:.3f} H1 {row[0][1]:.3f} | 2 sweeps {row[1][0]:.3f} / {row[1][1]:.3f}"
          f" | exact step solve {ex[0]:.3f} / {ex[1]:.3f}")

# 6. relaxation of the coarse FINAL state on the fine mesh (moves time to T + tau_r; interface speed ~ eps^2 kappa)
print("6. fine scheme from u_c(T) for tau_r (no extra input; time shifts to T + tau_r):")
Ufp = Uf.copy(); tprev = 0.0
for tr_ in (0.2, 0.4, 0.8, 1.6):
    Ufp = run(Ufp, L, tr_ - tprev, dt); tprev = tr_
    r1 = errs(run(v, L, tr_, dt), Uf); r2 = errs(Ufp, Uf)
    r3 = errs(run(v, L, tr_, tr_ / 4, sweeps=1), Uf)
    print(f"   tau_r {tr_}: exact fine relaxation rel-L2 {r1[0]:.3f} H1 {r1[1]:.3f} | 4 steps of tau_r/4, 1 sweep "
          f"{r3[0]:.3f} / {r3[1]:.3f} | drift of the fine solution itself u_h(T+tau_r) vs u_h(T): {r2[0]:.3f} / {r2[1]:.3f}")

# 7. how nonlinear is the initial-data part?  D(alpha) = S_h(PRu0 + alpha e0) - S_h(PRu0)
U0T = U0.T; PRu0 = P @ U0T[:n0]; e0 = U0T - PRu0
print(f"7. initial-data defect e0 = u0 - P R u0: ||e0|| / ||u0|| = {np.mean(np.linalg.norm(e0, axis=0) / np.linalg.norm(U0T, axis=0)):.3f}")
base = run(PRu0, L, T, dt); Dl = {}
for al in (1e-3, 0.1, 0.5, 1.0):
    Dl[al] = (run(PRu0 + al * e0, L, T, dt) - base) / al
for al in (0.1, 0.5, 1.0):
    print(f"   alpha {al}: ||D(alpha)/alpha - linearised|| / ||linearised|| = "
          f"{np.mean(np.linalg.norm(Dl[al] - Dl[1e-3], axis=0) / np.linalg.norm(Dl[1e-3], axis=0)):.2f}")

# 8. per-node factor of one Jacobi-swept step: q_i = m_i / (A_h)_ii at the step sizes considered
for dts in (dt, 0.2):
    _, A_ = stepper(L, dts); q = m / A_.diagonal() * (1 + S * dts)
    print(f"8. dt {dts}: q_i (1 + S dt) = (1 + S dt) m_i / (A_h)_ii in [{q.min():.3f}, {q.max():.3f}], median {np.median(q):.3f}")

# 9. the step size fixed by theory, not by the table above: the linearisation of u_t = eps^2 u_xx + u - u^3 about the
#    1-D interface profile has eigenvalues 0 (translation), -3/2, then a continuum <= -2 (Poeschl-Teller, l = 2), so
#    profile errors relax on the time scale 2/3 while the interface itself moves with speed ~ eps^2 * curvature.
tr_ = 2 / 3
r1 = errs(run(v, L, tr_, tr_ / 4, sweeps=1), Uf); r2 = errs(run(v, L, tr_, tr_ / 4), Uf)
print(f"9. tau_r = 2/3 (4 steps of 1/6): 1 sweep rel-L2 {r1[0]:.3f} H1 {r1[1]:.3f} | exact step solve {r2[0]:.3f} / {r2[1]:.3f}")
print(f"total {time.time() - t0:.0f}s")
