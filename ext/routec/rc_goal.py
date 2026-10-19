"""Route C: goal-oriented certificates for J(u) = (g, u), g = 1_omega / |omega|, omega a union of FE elements.

For any primal / dual approximations v, w in V_h:
    J(u)   - J(v) - R_v(w) = a(u - v, z - w)          (continuous target; u, z exact PDE solutions)
    J(u_h) - J(v) - R_v(w) = a(u_h - v, z_h - w)      (discrete target; u_h, z_h exact FE solutions)
so with guaranteed energy bounds the target lies in  centre +- eta_u * eta_z,  centre = J(v) + R_v(w).
  discrete (D):  eta = ||K^{-1/2} sigma_alg||, (div sigma_alg, psi_a) = R_a                  -> certifies J(u_h)
  continuous (C): eta = ||K^{-1/2}(sigma_disc + sigma_alg + K grad v)||, div(...) = f exactly -> certifies J(u)
The algebraic flux is built by one of: 'mvy' (MVY 2020 App. 8, fixed lifting), 'pu' (two-level, fixed lifting),
'pu-relaxed' (two-level, moment-relaxed patches), 'pu-liftfree' (two-level, lifting-free moment patches), 'joint' (global optimum over RT0(T_b) with moment constraints; reference)."""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import rc_fem as R


def joint_flux(M, K, Rv, Mb):
    fl = M.fl; nb, nE = len(fl.E), fl.n_edge
    I = np.repeat(fl.edge, 3, 1).ravel(); J = np.tile(fl.edge, (1, 3)).ravel()
    Mg = sp.csr_matrix((Mb.ravel(), (I, J)), shape=(nE, nE))
    D = sp.csr_matrix((fl.div_int.ravel(), (np.repeat(np.arange(nb), 3), fl.edge.ravel())), shape=(nb, nE))
    G = (M.Bb @ sp.diags(1 / fl.area) @ D).tocsr()
    sol = spsolve(sp.bmat([[Mg, -G.T], [G, None]]).tocsc(), np.concatenate([np.zeros(nE), Rv]))
    return R.Field(M).add(M.L + 1, np.arange(nE), sol[:nE])


def energy_bounds(M, K, A, dataK, x, method, pu=None, Mb=None, total=True):
    """guaranteed bounds for the problem A x* = (data, psi): returns dict(alg=..., tot=..., defects)."""
    Mb = R.local_mass(M, K, M.L + 1) if Mb is None else Mb
    F = M.load(dataK); Rv = F - A @ x; rb = R.lift_residual(M, Rv)
    if method == 'mvy':
        S, info = R.flux_alg_mvy(M, K, A, rb, Rv)
    elif method == 'pu':
        S, info = R.flux_alg_pu(M, K, A, rb, Rv, *pu, Mloc_b=Mb)
    elif method == 'pu-relaxed':
        S, info = R.flux_alg_pu_relaxed(M, K, A, rb, Rv, *pu, Mloc_b=Mb)
    elif method == 'pu-liftfree':
        S, info = R.flux_alg_pu_liftfree(M, K, A, Rv, *pu, Mloc_b=Mb)
    elif method == 'joint':
        S, info = joint_flux(M, K, Rv, Mb), {}
    else: raise ValueError(method)
    lift = S.div()                                             # the lifting actually carried by sigma_alg
    mom = float(np.max(np.abs(M.Bb @ lift - Rv)) / max(np.max(np.abs(Rv)), 1e-300))
    out = dict(alg=R.ps_norm(M, K, S), moment_defect=mom)
    if total:
        Sd, di = R.flux_disc(M, K, dataK, lift, M.full(x), Mb)
        St = R.Field(M); St += Sd; St += S
        out['tot'] = R.ps_norm(M, K, St, M.full(x))
        out['div_defect'] = float(np.max(np.abs(St.div() - dataK[M.par])))
    return out


def goal_weight(M, centre, radius):
    """g = 1_omega / |omega| with omega = FE elements whose centroid lies within `radius` of `centre`."""
    c = M.fe.centroid; om = np.hypot(c[:, 0] - centre[0], c[:, 1] - centre[1]) < radius
    if not om.any(): raise ValueError('empty goal region')
    return om / np.sum(M.fe.area[om]), om
