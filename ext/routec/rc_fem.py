"""Route C, deliverable 2: classical reference certificate (equilibrated fluxes, Prager--Synge, goal interval).

Problem.  -div(K grad u) = f in Omega (unit square minus three polygonal holes), u = 0 on the whole boundary
(outer and hole boundaries), K and f piecewise constant on the FE mesh T_h (level 3).  P1 conforming FE, V_h.

Everything is written for ARBITRARY primal / dual approximations v, w in V_h (any nodal vector on the unknown nodes:
exact solve, coarse solve, a few cycles, a neural output, a perturbed field).  No property of v other than v in V_h is
used; the certificate's validity never depends on how v was produced.

Flux space.  Lowest-order Raviart--Thomas RT0 on the BARYCENTRIC refinement T_b of T_h (each element split into three
at its centroid).  Reason (development check, 18 Oct): the residual lifting r_h must satisfy (r_h, psi_a) = R_a for every
unknown node a.  An element-local exact lifting in the spirit of Mallik--Vohralik--Yousef (JCAM 2020, eq. (5.4)),
(r_h, psi_a)_K = R_a / N_a, needs three degrees of freedom per element; P0(T_b) supplies exactly three, with a
well-conditioned 3x3 local system.  A global minimum-norm P0(T_h) lifting is exact but unstable for oscillatory
residuals (the best RT0 flux then overestimated a 10% white-noise error 186-fold; rc_fem_v1_p0lift.py).
Because f, the goal weight g and r_h are piecewise constant on T_b, every equilibrium condition holds EXACTLY (no data
oscillation); only floating-point round-off remains, and it is reported.

Main objects
  Mesh(H, coarse)                 levels T_0..T_L (geometric, nested) and the flux level T_b = bary(T_L)
  lift_residual(M, R)             r_h in P0(T_b), element-local, (r_h, psi_a) = R_a exactly
  flux_disc(...)                  vertex-patch flux on T_b (Braess--Schoeberl with the MVY lifting): div = f - r_h
  flux_alg_oracle(...)            global RT0(T_b) mixed solve, div = r_h (best RT0 algebraic flux for this lifting)
  flux_alg_mvy(...)               MVY 2020 Appendix 8 multilevel construction on T_0..T_L, T_b: div = r_h
  flux_alg_pu(...)                two-level construction on T_b with ANY partition of unity whose floating members lie in
                                  the coarse space (geometric hats or operator-adapted functions): div = r_h
  ps_norm(...)                    || K^{-1/2} (sigma + K grad v) ||  (Prager--Synge; guaranteed when div sigma = f)
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve


def _signed_area(X, E):
    return 0.5 * ((X[E[:, 1], 0] - X[E[:, 0], 0]) * (X[E[:, 2], 1] - X[E[:, 0], 1])
                  - (X[E[:, 2], 0] - X[E[:, 0], 0]) * (X[E[:, 1], 1] - X[E[:, 0], 1]))


class Level:
    """one triangulation: counter-clockwise elements, edges with a global orientation, RT0 data, P1 gradients.
    Local edge i is opposite local vertex i."""
    def __init__(self, X, E, dirichlet_mask):
        E = np.array(E, dtype=np.int64)
        s = _signed_area(X, E); flip = s < 0
        E[flip] = E[flip][:, [0, 2, 1]]
        self.X, self.E, self.area = X, E, np.abs(s)
        ne = len(E)
        a = np.stack([E[:, 1], E[:, 2], E[:, 0]], 1); b = np.stack([E[:, 2], E[:, 0], E[:, 1]], 1)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        uk, inv = np.unique((lo * len(X) + hi).ravel(), return_inverse=True)
        self.edge = inv.reshape(ne, 3); self.n_edge = len(uk)
        self.edge_nodes = np.stack([uk // len(X), uk % len(X)], 1)
        cnt = np.bincount(inv, minlength=self.n_edge)
        self.boundary_edge = cnt == 1
        # global normal = outward normal of the lowest-id element containing the edge
        first = np.full(self.n_edge, np.iinfo(np.int64).max)
        np.minimum.at(first, inv, np.arange(ne * 3))
        sign = -np.ones(ne * 3); sign[first] = 1.0
        self.sign = sign.reshape(ne, 3)
        self.len = np.linalg.norm(X[a] - X[b], axis=2)
        self.dirichlet_edge = self.boundary_edge & dirichlet_mask[self.edge_nodes[:, 0]] & dirichlet_mask[self.edge_nodes[:, 1]]
        x0, x1, x2 = X[E[:, 0]], X[E[:, 1]], X[E[:, 2]]
        det = 2 * self.area; g = np.zeros((ne, 3, 2))
        for i, (p, q) in enumerate(((x1, x2), (x2, x0), (x0, x1))):
            g[:, i, 0] = (p[:, 1] - q[:, 1]) / det; g[:, i, 1] = (q[:, 0] - p[:, 0]) / det
        self.grad = g
        self.centroid = (x0 + x1 + x2) / 3
        self.mid = 0.5 * (X[a] + X[b])                               # (ne, 3, 2) edge midpoints (quadrature points)
        self.rt_c = self.sign * self.len / (2 * self.area[:, None])  # phi_i(x) = rt_c_i (x - p_i), normal component = sign_i
        self.div_int = self.sign * self.len                          # (div phi_i, 1)_K


class Mesh:
    """levels T_0..T_L of the nested hierarchy H (NestedHierarchy) and the flux level T_b = barycentric refinement of T_L.
    levels[j], j = 0..L, are geometric; levels[L+1] = T_b.  anc[j][t] = level-j ancestor of T_b element t."""
    def __init__(self, H, coarse_mesh):
        from ofwno_core import MeshHierarchy
        self.H = H; L = H.L; self.L = L
        X = np.asarray(H.coords, float)
        dmask = np.zeros(len(X), bool); dmask[H.boundary] = True
        mh = MeshHierarchy(np.asarray(coarse_mesh[0]), np.asarray(coarse_mesh[1])); Ej = []
        for j in range(L + 1):
            if j > 0: mh.refine_one_level()
            Ej.append(np.asarray(mh.elements, np.int64))
        assert np.allclose(np.asarray(mh.coords), X), 'level rebuild does not reproduce the fine coordinates'
        self.levels = [Level(X, Ej[j], dmask) for j in range(L + 1)]
        fe = self.levels[L]; nf = len(fe.E)
        # barycentric refinement: sub-element 3k+i = (c_k, v_{i+1}, v_{i+2}), opposite parent vertex i
        Xb = np.vstack([X, fe.centroid]); c = len(X) + np.arange(nf)
        Eb = np.empty((3 * nf, 3), np.int64)
        for i in range(3):
            Eb[i::3] = np.column_stack([c, fe.E[:, (i + 1) % 3], fe.E[:, (i + 2) % 3]])
        self.dmask_b = np.concatenate([dmask, np.zeros(nf, bool)])
        self.levels.append(Level(Xb, Eb, self.dmask_b))
        assert np.array_equal(self.levels[L + 1].E[:, 0], np.repeat(c, 3)), 'barycentric sub-elements must keep the centroid first'
        self.fe, self.fl = fe, self.levels[L + 1]
        nb = len(Eb)
        self.par = np.repeat(np.arange(nf), 3)
        self.opp = np.tile(np.arange(3), nf)
        self.anc = {L + 1: np.arange(nb), L: self.par}
        for j in range(L - 1, -1, -1): self.anc[j] = (np.arange(nf) // 4 ** (L - j))[self.par]
        self.dirichlet = dmask
        self.unk = np.asarray(H.nodes[L])
        self.pos = -np.ones(len(X), np.int64); self.pos[self.unk] = np.arange(len(self.unk)); self.N = len(self.unk)
        # T_b vertices in parent barycentric coordinates (the orientation flip in Level may reorder the last two)
        fl = self.fl
        vb = np.zeros((nb, 3, 3))
        for q in range(3):
            node = fl.E[:, q]
            isc = node >= len(X)
            vb[isc, q, :] = 1 / 3
            pe = fe.E[self.par]
            for i in range(3): vb[~isc & (pe[:, i] == node), q, i] = 1.0
        assert np.allclose(vb.sum(2), 1)
        self.qbary = np.stack([0.5 * (vb[:, (q + 1) % 3] + vb[:, (q + 2) % 3]) for q in range(3)], 1)   # (nb, 3pts, 3)
        self.sub_mean = vb.mean(1)                                   # parent-barycentric means over each sub-element
        p = fl.X[fl.E]
        self.phi_q = fl.rt_c[:, :, None, None] * (fl.mid[:, None, :, :] - p[:, :, None, :])  # (nb, 3, 3pts, 2)
        self.wq = fl.area / 3
        self.el_of = [[] for _ in range(len(X))]
        for k, t in enumerate(fe.E):
            for a in t: self.el_of[a].append(k)
        self.Nl = np.array([len(e) for e in self.el_of], float)
        r, cc, v = [], [], []
        for i in range(3):
            node = fe.E[self.par, i]; m = self.pos[node] >= 0
            r.append(self.pos[node[m]]); cc.append(np.nonzero(m)[0]); v.append((fl.area * self.sub_mean[:, i])[m])
        self.Bb = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(cc))), shape=(self.N, nb))

    def full(self, v):
        x = np.zeros(len(self.fe.X)); x[self.unk] = v; return x

    def grad(self, vfull):
        return np.einsum('kid,ki->kd', self.fe.grad, vfull[self.fe.E])

    def stiffness(self, K):
        f = self.fe
        Ke = np.einsum('kid,kjd->kij', f.grad, f.grad) * (K * f.area)[:, None, None]
        I = np.repeat(f.E, 3, 1).ravel(); J = np.tile(f.E, (1, 3)).ravel()
        A = sp.csr_matrix((Ke.ravel(), (I, J)), shape=(len(f.X), len(f.X)))
        return A[self.unk][:, self.unk].tocsc()

    def load(self, fK):
        """(f, psi_a) on the unknowns for f piecewise constant on T_L."""
        return self.Bb @ fK[self.par]

    def energy(self, K, e_full):
        g = self.grad(e_full)
        return float(np.sqrt(np.sum(K * self.fe.area * np.sum(g * g, 1))))


# --------------------------------------------------------------------------------------------------------------------
_W = np.array([[1., 4., 4.], [4., 1., 4.], [4., 4., 1.]]) / 27.0   # (chi_{T_i}, psi_l)_K / |K|, T_i opposite vertex i
_Winv = np.linalg.inv(_W)


def lift_residual(M, R):
    """r_h in P0(T_b): on each T_L element K, (r_h, psi_l)_K = R_l / N_l for its unknown vertices l and 0 for its Dirichlet
    vertices.  Summing over the elements around l gives (r_h, psi_l) = R_l exactly.  Values per T_b element."""
    fe = M.fe; nf = len(fe.E); b = np.zeros((nf, 3))
    for i in range(3):
        node = fe.E[:, i]; m = M.pos[node] >= 0
        b[m, i] = R[M.pos[node[m]]] / M.Nl[node[m]]
    # exact local moments from the actual sub-element geometry: (chi_t, psi_l)_K = |t| * sub_mean[t, l]
    Wk = (M.fl.area * M.sub_mean.T).T.reshape(nf, 3, 3).transpose(0, 2, 1)   # Wk[k, l, i] for sub-element i of K
    r = np.linalg.solve(Wk, b[:, :, None])[:, :, 0]
    return r.ravel()


# --------------------------------------------------------------------------------------------------------------------
class Field:
    """sum over levels j of RT0(T_j) fields (coefficient = normal component on each level-j edge, global orientation)."""
    def __init__(self, M):
        self.M = M; self.c = {}

    def add(self, j, edges, coef):
        if j not in self.c: self.c[j] = np.zeros(self.M.levels[j].n_edge)
        np.add.at(self.c[j], edges, coef); return self

    def __iadd__(self, other):
        for j, c in other.c.items(): self.c[j] = self.c.get(j, 0) + c
        return self

    def div(self):
        M = self.M; d = np.zeros(len(M.fl.E))
        for j, c in self.c.items():
            lev = M.levels[j]
            d += (np.einsum('ki,ki->k', c[lev.edge], lev.div_int) / lev.area)[M.anc[j]]
        return d

    def at_quad(self):
        M = self.M; fl = M.fl; out = np.zeros((len(fl.E), 3, 2))
        for j, c in self.c.items():
            lev = M.levels[j]; par = M.anc[j]
            p = lev.X[lev.E[par]]
            coef = c[lev.edge[par]] * lev.rt_c[par]
            out += np.einsum('ki,kpid->kpd', coef, fl.mid[:, :, None, :] - p[:, None, :, :])
        return out


def ps_norm(M, K, sigma, vfull=None):
    """|| K^{-1/2} (sigma + K grad v) || with exact quadrature on T_b."""
    Kb = K[M.par]; s = sigma.at_quad()
    if vfull is not None: s = s + (Kb[:, None] * M.grad(vfull)[M.par])[:, None, :]
    return float(np.sqrt(np.sum(np.sum(s * s, 2) * M.wq[:, None] / Kb[:, None])))


# --------------------------------------------------------------------------------------------------------------------
def local_mass(M, K, j):
    """(n_j, 3, 3): (K^{-1} phi_i, phi_k) on each level-j element, integrated exactly over its T_b descendants."""
    lev = M.levels[j]; fl = M.fl; par = M.anc[j]; Kb = K[M.par]
    p = lev.X[lev.E[par]]
    ph = lev.rt_c[par][:, :, None, None] * (fl.mid[:, None, :, :] - p[:, :, None, :])
    contrib = np.einsum('kipd,kjpd,k->kij', ph, ph, M.wq / Kb)
    out = np.zeros((len(lev.E), 3, 3)); np.add.at(out, par, contrib)
    return out


def solve_patch(lev, elems, data, Mloc, rhs_loc=None, free_dirichlet=False):
    """min || K^{-1/2}(sigma + w) ||_patch over RT0(patch) with div sigma = data (per element) and zero normal flux on the
    patch boundary, except on Dirichlet boundary edges when free_dirichlet.  rhs_loc (m, 3) = (K^{-1} w, phi_i)_k.
    A patch with no free edge is floating: zero-mean pressure, data must be compatible (checked by the caller).
    Returns (global edge ids, coefficients, max |div sigma - data|)."""
    elems = np.asarray(elems); m = len(elems)
    ue, loc = np.unique(lev.edge[elems].ravel(), return_inverse=True); loc = loc.reshape(m, 3)
    cnt = np.bincount(loc.ravel(), minlength=len(ue))
    active = cnt == 2
    if free_dirichlet: active |= (cnt == 1) & lev.dirichlet_edge[ue]
    floating = not np.any(active & (cnt == 1))
    na = int(active.sum()); idx = -np.ones(len(ue), np.int64); idx[active] = np.arange(na)
    li = idx[loc]; mask = li >= 0
    e3, i3, j3 = np.nonzero(mask[:, :, None] & mask[:, None, :])
    Ml = Mloc[elems]
    Mm = sp.csr_matrix((Ml[e3, i3, j3], (li[e3, i3], li[e3, j3])), shape=(na, na))
    r_, c_ = np.nonzero(mask)
    Bd = sp.csr_matrix((lev.div_int[elems][r_, c_], (r_, li[r_, c_])), shape=(m, na))
    rhs = np.zeros(na)
    if rhs_loc is not None: np.add.at(rhs, li[r_, c_], -rhs_loc[r_, c_])
    g = data * lev.area[elems]
    if floating:
        ar = sp.csr_matrix(lev.area[elems][:, None])
        S = sp.bmat([[Mm, -Bd.T, None], [Bd, None, ar], [None, ar.T, None]]).tocsc(); b = np.concatenate([rhs, g, [0.0]])
    else:
        S = sp.bmat([[Mm, -Bd.T], [Bd, None]]).tocsc(); b = np.concatenate([rhs, g])
    coef = spsolve(S, b)[:na]
    return ue[active], coef, float(np.max(np.abs(Bd @ coef - g) / lev.area[elems]))


def _target_rhs(M, ts, theta_full, gfe):
    """(K^{-1} w, phi_i) on T_b elements ts for w = theta K g: theta P1 (FE nodal values), g constant per FE element."""
    lam = np.einsum('kpi,ki->kp', M.qbary[ts], theta_full[M.fe.E[M.par[ts]]])
    return np.einsum('kipd,kp,kd->ki', M.phi_q[ts], lam * M.wq[ts][:, None], gfe[M.par[ts]])


# --------------------------------------------------------------------------------------------------------------------
def flux_disc(M, K, fK, rb, vfull, Mloc_b=None):
    """sigma_disc in RT0(T_b): sum over ALL FE vertices a of argmin ||K^{-1/2}(sigma_a + psi_a K grad v)||_{omega_a} with
    div sigma_a = Pi_0((f - r_h) psi_a) - K grad v . grad psi_a on the T_b elements of omega_a; zero normal flux on the
    patch boundary for unknown vertices (compatible because (r_h, psi_a) = R_a), free on Dirichlet edges for Dirichlet
    vertices.  div sigma_disc = f - r_h exactly."""
    fe, fl = M.fe, M.fl
    Mloc_b = local_mass(M, K, M.L + 1) if Mloc_b is None else Mloc_b
    gv = M.grad(vfull); F = Field(M); worst = 0.0; compat = 0.0; fb = fK[M.par]; Kb = K[M.par]
    for a in range(len(fe.X)):
        ks = np.asarray(M.el_of[a])
        if len(ks) == 0: continue
        ts = (3 * ks[:, None] + np.arange(3)).ravel()
        li = np.argmax(fe.E[M.par[ts]] == a, axis=1)
        data = (fb[ts] - rb[ts]) * M.sub_mean[ts, li] - Kb[ts] * np.sum(gv[M.par[ts]] * fe.grad[M.par[ts], li], 1)
        th = np.zeros(len(fe.X)); th[a] = 1.0
        e, c, d = solve_patch(fl, ts, data, Mloc_b, _target_rhs(M, ts, th, gv), free_dirichlet=bool(M.dirichlet[a]))
        F.add(M.L + 1, e, c); worst = max(worst, d)
        if not M.dirichlet[a]: compat = max(compat, abs(np.sum(data * fl.area[ts])))
    return F, dict(defect=worst, compat=compat)


def flux_alg_oracle(M, K, rb, Mloc_b=None):
    """argmin ||K^{-1/2} sigma|| over RT0(T_b) with div sigma = r_h (all boundary edges free: whole boundary Dirichlet).
    The best RT0(T_b) algebraic flux for this lifting -- a reference, not a cheap method."""
    fl = M.fl; nb, nE = len(fl.E), fl.n_edge
    Mloc_b = local_mass(M, K, M.L + 1) if Mloc_b is None else Mloc_b
    I = np.repeat(fl.edge, 3, 1).ravel(); J = np.tile(fl.edge, (1, 3)).ravel()
    Mg = sp.csr_matrix((Mloc_b.ravel(), (I, J)), shape=(nE, nE))
    Bg = sp.csr_matrix((fl.div_int.ravel(), (np.repeat(np.arange(nb), 3), fl.edge.ravel())), shape=(nb, nE))
    sol = spsolve(sp.bmat([[Mg, -Bg.T], [Bg, None]]).tocsc(), np.concatenate([np.zeros(nE), rb * fl.area]))
    return Field(M).add(M.L + 1, np.arange(nE), sol[:nE])


def geometric_P(M, j):
    """composite geometric prolongation, level-j unknowns -> FE unknowns."""
    H = M.H; P = None
    for k in range(j + 1, H.L + 1): P = H.P[k] if P is None else H.P[k] @ P
    return sp.identity(M.N, format='csr') if P is None else P.tocsr()


def _hat_tables(M, j):
    """for every T_b element and each local vertex of its level-j ancestor: hat mean over the T_b element and gradient."""
    lev, fl = M.levels[j], M.fl; anc = M.anc[j]
    Pc = lev.X[lev.E[anc]]
    Tm = np.stack([Pc[:, 0] - Pc[:, 2], Pc[:, 1] - Pc[:, 2]], 2)
    l01 = np.linalg.solve(Tm, (fl.centroid - Pc[:, 2])[:, :, None])[:, :, 0]
    return np.column_stack([l01, 1 - l01.sum(1)]), lev.grad[anc]


def flux_alg_mvy(M, K, A, rb, R, variant='mvy'):
    """Mallik--Vohralik--Yousef (JCAM 2020) Appendix 8 multilevel algebraic flux, RT0 on T_0, ..., T_L, T_b:
      rho_0 in P1(T_0) cap H^1_0 with (K grad rho_0, grad v_0) = (r_h, v_0) = R(v_0)          [coarse solve]
      level j = 1..L+1, every vertex a of T_{j-1}: RT0(T_j) on omega^a_{j-1},
         div sigma = Pi_j (Id - Pi_{j-1}) (r_h psi^a_{j-1} - K grad rho_0 . grad psi^a_{j-1}),  Pi_0 := 0,
         zero normal flux on the patch boundary (interior a) / except on Dirichlet edges (Dirichlet a),
         minimising ||K^{-1/2} sigma|| (MVY (8.3a), no target).
    Telescoping gives div sigma_alg = Pi_{L+1} r_h = r_h.
    variant='prvw': Papez--Rude--Vohralik--Wohlmuth (CMAME 2020) Def. 4.3, where the rho_0 term appears only on level 1
    ((4.7b) vs (4.11b)); both telescope to r_h because the level-(j-1) hats form a partition of unity."""
    fl, L = M.fl, M.L
    P0 = geometric_P(M, 0)
    rho0 = P0 @ np.linalg.solve((P0.T @ A @ P0).toarray(), P0.T @ R)
    grho_b = M.grad(M.full(rho0))[M.par]; Kb = K[M.par]
    F = Field(M); worst = 0.0; compat = 0.0
    for j in range(1, L + 2):
        levc, lev = M.levels[j - 1], M.levels[j]
        Mloc = local_mass(M, K, j)
        ancc, ancj = M.anc[j - 1], M.anc[j]
        mean_c, grad_c = _hat_tables(M, j - 1)
        Vc = levc.E[ancc]                                             # (nb, 3) level-(j-1) vertices of each T_b element
        order = np.argsort(Vc.ravel(), kind='stable'); sv = Vc.ravel()[order]
        starts = np.searchsorted(sv, np.unique(sv)); ends = np.append(starts[1:], len(sv))
        for a, s0, s1 in zip(np.unique(sv), starts, ends):
            flat = order[s0:s1]; ts, li = flat // 3, flat % 3
            use_rho = (variant == 'mvy') or j == 1        # PRVW Def. 4.3 (4.11b): no rho_0 term above level 1
            ig = (rb[ts] * mean_c[ts, li] - (Kb[ts] * np.sum(grho_b[ts] * grad_c[ts, li], 1) if use_rho else 0.0)) * fl.area[ts]
            uj, invj = np.unique(ancj[ts], return_inverse=True)
            sj = np.bincount(invj, ig) / lev.area[uj]
            if j > 1:
                uc, invc = np.unique(ancc[ts], return_inverse=True)
                sc = np.bincount(invc, ig) / levc.area[uc]
                cj = np.zeros(len(uj), np.int64); cj[invj] = ancc[ts]
                sj = sj - sc[np.searchsorted(uc, cj)]
            e, c, d = solve_patch(lev, uj, sj, Mloc, None, free_dirichlet=bool(M.dmask_b[a]))
            F.add(j, e, c); worst = max(worst, d)
            if not M.dmask_b[a]: compat = max(compat, abs(np.sum(sj * lev.area[uj])))
    return F, dict(defect=worst, compat=compat, rho=rho0)


def flux_alg_pu(M, K, A, rb, R, Theta, in_Vc, C, target=True, Mloc_b=None):
    """two-level algebraic flux on T_b with an arbitrary partition of unity.
    Theta: (n_FE_nodes, m) nodal values of P1 functions, Theta.sum(1) == 1 at every FE node.
    in_Vc[i]: member i vanishes on the Dirichlet boundary and lies in span(C): floating patch (zero normal flux on its
              whole patch boundary), compatible because rho is the Galerkin solution in span(C).  Otherwise the member
              must touch the Dirichlet boundary and its flux is free on Dirichlet edges.
    rho = C (C^T A C)^{-1} C^T R.  Patch i = T_b elements of supp(theta_i):
       div sigma_i = Pi_0(r_h theta_i) - K grad rho . grad theta_i,   minimise ||K^{-1/2}(sigma_i + theta_i K grad rho)||
    (target=False: minimise ||K^{-1/2} sigma_i||).  Sum: div sigma_alg = r_h - K grad rho . grad(sum theta) = r_h."""
    fe, fl = M.fe, M.fl
    Mloc_b = local_mass(M, K, M.L + 1) if Mloc_b is None else Mloc_b
    CA = C.T @ (A @ C); CA = CA.toarray() if sp.issparse(CA) else np.asarray(CA)
    rho = np.asarray(C @ np.linalg.solve(CA, np.asarray(C.T @ R).ravel())).ravel()
    grho = M.grad(M.full(rho)); Kb = K[M.par]
    Th = Theta.tocsc(); F = Field(M); worst = 0.0; compat = 0.0; sizes = []
    for i in range(Th.shape[1]):
        lo, hi = Th.indptr[i], Th.indptr[i + 1]
        nodes, vals = Th.indices[lo:hi], Th.data[lo:hi]
        if len(nodes) == 0: continue
        th = np.zeros(len(fe.X)); th[nodes] = vals
        ks = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
        ts = (3 * ks[:, None] + np.arange(3)).ravel()
        tv = th[fe.E[M.par[ts]]]
        gth = np.einsum('kid,ki->kd', fe.grad[M.par[ts]], tv)
        data = rb[ts] * np.sum(M.sub_mean[ts] * tv, 1) - Kb[ts] * np.sum(grho[M.par[ts]] * gth, 1)
        rhs = _target_rhs(M, ts, th, grho) if target else None
        e, c, d = solve_patch(fl, ts, data, Mloc_b, rhs, free_dirichlet=not bool(in_Vc[i]))
        F.add(M.L + 1, e, c); worst = max(worst, d); sizes.append(len(ks))
        if in_Vc[i]: compat = max(compat, abs(np.sum(data * fl.area[ts])))
    return F, dict(defect=worst, compat=compat, rho=rho, patch_sizes=np.array(sizes))


def hats_full(M, j):
    """nodal values on ALL FE nodes of the level-j hats of all level-j vertices: (n_FE_nodes, n_vertices), vertex ids."""
    lev = M.levels[j]; verts = np.unique(lev.E); X = M.fe.X; n = len(X)
    owner = -np.ones(n, np.int64); par = np.arange(len(M.fe.E)) // 4 ** (M.L - j)
    for i in range(3): owner[M.fe.E[:, i]] = par
    P = lev.X[lev.E[owner]]
    Tm = np.stack([P[:, 0] - P[:, 2], P[:, 1] - P[:, 2]], 2)
    l01 = np.linalg.solve(Tm, (X - P[:, 2])[:, :, None])[:, :, 0]
    lam = np.column_stack([l01, 1 - l01.sum(1)])
    col = -np.ones(n, np.int64); col[verts] = np.arange(len(verts))
    rows = np.repeat(np.arange(n), 3); cols = col[lev.E[owner]].ravel(); vals = lam.ravel()
    keep = np.abs(vals) > 1e-14
    return sp.csr_matrix((vals[keep], (rows[keep], cols[keep])), shape=(n, len(verts))), verts


def pu_geometric(M, j, Z=None):
    """partition of unity: level-j hats of all level-j vertices; interior ones lie in V_c = P1(T_j) (+ Z)."""
    Th, verts = hats_full(M, j)
    C = geometric_P(M, j)
    if Z is not None and Z.shape[1]: C = sp.hstack([C, sp.csr_matrix(Z)]).tocsr()
    return Th.tocsr(), ~M.dirichlet[verts], C


def pu_operator_adapted(M, Psi, Z=None):
    """partition of unity from operator-adapted coarse functions (columns of Psi, on the unknowns), Shepard-normalised at
    the nodes and completed by the FE hats of the Dirichlet nodes.  V_c = span(normalised functions) (+ Z).
    Requires s = Psi 1 > 0 at every unknown node."""
    s = np.asarray(Psi @ np.ones(Psi.shape[1])).ravel()
    if s.min() <= 0: raise ValueError(f'operator-adapted functions do not have a positive sum (min {s.min():.3e})')
    Tn = (sp.diags(1 / s) @ Psi.tocsr()).tocoo()
    n = len(M.fe.X); dn = np.nonzero(M.dirichlet)[0]
    full = sp.csr_matrix((Tn.data, (M.unk[Tn.row], Tn.col)), shape=(n, Tn.shape[1]))
    Dh = sp.csr_matrix((np.ones(len(dn)), (dn, np.arange(len(dn)))), shape=(n, len(dn)))
    Theta = sp.hstack([full, Dh]).tocsr()
    in_Vc = np.concatenate([np.ones(Tn.shape[1], bool), np.zeros(len(dn), bool)])
    Tn = Tn.tocsr()
    C = Tn if Z is None or Z.shape[1] == 0 else sp.hstack([Tn, sp.csr_matrix(Z)]).tocsr()
    return Theta, in_Vc, C


# --------------------------------------------------------------------------------------------------------------------
# moment-relaxed patch problems (development, 18 Oct): constrain only the hat moments of the divergence
# --------------------------------------------------------------------------------------------------------------------
def solve_patch_moments(M, ts, data, Mloc_b, rhs_loc=None, free_dirichlet=False):
    """as solve_patch on T_b elements ts, but the divergence is constrained only through its moments against the FE hats
    of the UNKNOWN nodes touching the patch: (div sigma - data, psi_a)_patch = 0.  The divergence itself is free within the
    kernel of those moments, so the residual lifting is optimised locally.  Validity: summed over a partition of unity,
    (div sigma_alg, psi_a) = sum_i (data_i, psi_a) = (r_h, psi_a) = R_a, which is all the discrete certificate needs.
    Returns (edge ids, coefficients, max moment defect, divergence per element)."""
    fl, fe = M.fl, M.fe; ts = np.asarray(ts); m = len(ts)
    ue, loc = np.unique(fl.edge[ts].ravel(), return_inverse=True); loc = loc.reshape(m, 3)
    cnt = np.bincount(loc.ravel(), minlength=len(ue))
    active = cnt == 2
    if free_dirichlet: active |= (cnt == 1) & fl.dirichlet_edge[ue]
    floating = not np.any(active & (cnt == 1))
    na = int(active.sum()); idx = -np.ones(len(ue), np.int64); idx[active] = np.arange(na)
    li = idx[loc]; mask = li >= 0
    e3, i3, j3 = np.nonzero(mask[:, :, None] & mask[:, None, :])
    Ml = Mloc_b[ts]
    Mm = sp.csr_matrix((Ml[e3, i3, j3], (li[e3, i3], li[e3, j3])), shape=(na, na))
    r_, c_ = np.nonzero(mask)
    Bd = sp.csr_matrix((fl.div_int[ts][r_, c_], (r_, li[r_, c_])), shape=(m, na))     # (div phi, 1)_t
    # moment rows: unknown FE nodes of the parents; (chi_t, psi_a) = |t| sub_mean[t, i]
    pn = fe.E[M.par[ts]]                                                              # (m, 3)
    unk = M.pos[pn] >= 0
    nodes = np.unique(pn[unk]); q = len(nodes); ni = np.searchsorted(nodes, pn)
    rr, cc = np.nonzero(unk)
    Mo = sp.csr_matrix((M.sub_mean[ts][rr, cc], (ni[rr, cc], rr)), shape=(q, m))       # times (div sigma)_t * ... see below
    # (div sigma, psi_a) = sum_t div_t |t| sub_mean = Mo @ (Bd sigma)   since (Bd sigma)_t = div_t |t|
    G = (Mo @ Bd).tocsr()
    target = Mo @ (data * fl.area[ts])
    if floating and not np.any(~unk):                     # hats sum to one on the patch: one redundant constraint
        G = G[:-1]; target = target[:-1]
    rhs = np.zeros(na)
    if rhs_loc is not None: np.add.at(rhs, li[r_, c_], -rhs_loc[r_, c_])
    S = sp.bmat([[Mm, -G.T], [G, None]]).tocsc()
    coef = spsolve(S, np.concatenate([rhs, target]))[:na]
    dv = (Bd @ coef) / fl.area[ts]
    mom = Mo @ (dv * fl.area[ts]) - Mo @ (data * fl.area[ts])
    return ue[active], coef, float(np.max(np.abs(mom))) if q else 0.0, dv


def flux_alg_pu_relaxed(M, K, A, rb, R, Theta, in_Vc, C, target=True, Mloc_b=None):
    """flux_alg_pu with moment-relaxed patch problems (solve_patch_moments).  div sigma_alg is a valid lifting of R
    ((div sigma_alg, psi_a) = R_a), generally different from r_h.  Use for the discrete (algebraic) certificate; for the
    total certificate pass div(sigma_alg) as the lifting to flux_disc."""
    fe, fl = M.fe, M.fl
    Mloc_b = local_mass(M, K, M.L + 1) if Mloc_b is None else Mloc_b
    CA = C.T @ (A @ C); CA = CA.toarray() if sp.issparse(CA) else np.asarray(CA)
    rho = np.asarray(C @ np.linalg.solve(CA, np.asarray(C.T @ R).ravel())).ravel()
    grho = M.grad(M.full(rho)); Kb = K[M.par]
    Th = Theta.tocsc(); F = Field(M); worst = 0.0
    for i in range(Th.shape[1]):
        lo, hi = Th.indptr[i], Th.indptr[i + 1]
        nodes, vals = Th.indices[lo:hi], Th.data[lo:hi]
        if len(nodes) == 0: continue
        th = np.zeros(len(fe.X)); th[nodes] = vals
        ks = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
        ts = (3 * ks[:, None] + np.arange(3)).ravel()
        tv = th[fe.E[M.par[ts]]]
        gth = np.einsum('kid,ki->kd', fe.grad[M.par[ts]], tv)
        data = rb[ts] * np.sum(M.sub_mean[ts] * tv, 1) - Kb[ts] * np.sum(grho[M.par[ts]] * gth, 1)
        rhs = _target_rhs(M, ts, th, grho) if target else None
        e, c, d, _ = solve_patch_moments(M, ts, data, Mloc_b, rhs, free_dirichlet=not bool(in_Vc[i]))
        F.add(M.L + 1, e, c); worst = max(worst, d)
    return F, dict(defect=worst, rho=rho)


def flux_alg_pu_liftfree(M, K, A, R, Theta, in_Vc, C, Mloc_b=None, check_connected=True):
    """lifting-free two-level algebraic flux (development, 18 Oct).  For PU member theta_i the patch problem is
         min || K^{-1/2} (sigma_i + theta_i K grad rho) ||   over RT0(T_b) on supp theta_i (zero normal flux on the patch
         boundary; free on Dirichlet edges for boundary members), subject to the hat moments
         (div sigma_i, psi_a) = m_a^(i) := theta_i(a) (R - A rho)_a + (theta_i K grad rho, grad psi_a)
       for every unknown node a touching the patch.
    No residual lifting is used.  sum_i m^(i) = R (sum_i theta_i = 1 at the nodes), and for a floating member
    sum_a m_a^(i) = R(theta_i) - a(rho, theta_i) = 0 because theta_i is in span(C) and rho is the Galerkin solution, so the
    zero-flux patch problem is consistent.  Hence (div sigma_alg, psi_a) = R_a for every unknown a: valid for the discrete
    certificate, and div sigma_alg is a valid lifting for the total one."""
    fe, fl = M.fe, M.fl
    Mloc_b = local_mass(M, K, M.L + 1) if Mloc_b is None else Mloc_b
    CA = C.T @ (A @ C); CA = CA.toarray() if sp.issparse(CA) else np.asarray(CA)
    rho = np.asarray(C @ np.linalg.solve(CA, np.asarray(C.T @ R).ravel())).ravel()
    rfull = M.full(R - A @ rho); grho = M.grad(M.full(rho))
    Th = Theta.tocsc(); F = Field(M); worst = 0.0; disconnected = 0
    for i in range(Th.shape[1]):
        lo, hi = Th.indptr[i], Th.indptr[i + 1]
        nodes, vals = Th.indices[lo:hi], Th.data[lo:hi]
        if len(nodes) == 0: continue
        th = np.zeros(len(fe.X)); th[nodes] = vals
        ks = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
        if check_connected and in_Vc[i] and _n_components(M, ks) > 1: disconnected += 1
        ts = (3 * ks[:, None] + np.arange(3)).ravel()
        # local stiffness moments (theta K grad rho, grad psi_a) over the FE elements ks (theta linear: mean at centroid)
        tbar = th[fe.E[ks]].mean(1)
        loc = np.einsum('kd,kid->ki', (K[ks] * fe.area[ks] * tbar)[:, None] * grho[ks], fe.grad[ks])   # (m, 3)
        mom_full = np.zeros(len(fe.X)); np.add.at(mom_full, fe.E[ks].ravel(), loc.ravel())
        mom_full += th * rfull
        e, c, d = _solve_moment_targets(M, ts, mom_full, Mloc_b, _target_rhs(M, ts, th, grho), not bool(in_Vc[i]))
        F.add(M.L + 1, e, c); worst = max(worst, d)
    return F, dict(defect=worst, rho=rho, disconnected=disconnected)


def _n_components(M, ks):
    """edge-connected components of a set of FE elements."""
    fe = M.fe; ks = np.asarray(ks); eid = fe.edge[ks]
    m = len(ks); parent = np.arange(m)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    first = {}
    for r in range(m):
        for e in eid[r]:
            if e in first: a, b = find(r), find(first[e]); parent[a] = b
            else: first[e] = r
    return len({find(r) for r in range(m)})


def _solve_moment_targets(M, ts, mom_full, Mloc_b, rhs_loc, free_dirichlet):
    """solve_patch_moments with the hat-moment targets given directly (mom_full[a] for FE node a)."""
    fl, fe = M.fl, M.fe; ts = np.asarray(ts); m = len(ts)
    ue, loc = np.unique(fl.edge[ts].ravel(), return_inverse=True); loc = loc.reshape(m, 3)
    cnt = np.bincount(loc.ravel(), minlength=len(ue))
    active = cnt == 2
    if free_dirichlet: active |= (cnt == 1) & fl.dirichlet_edge[ue]
    floating = not np.any(active & (cnt == 1))
    na = int(active.sum()); idx = -np.ones(len(ue), np.int64); idx[active] = np.arange(na)
    li = idx[loc]; mask = li >= 0
    e3, i3, j3 = np.nonzero(mask[:, :, None] & mask[:, None, :])
    Ml = Mloc_b[ts]
    Mm = sp.csr_matrix((Ml[e3, i3, j3], (li[e3, i3], li[e3, j3])), shape=(na, na))
    r_, c_ = np.nonzero(mask)
    Bd = sp.csr_matrix((fl.div_int[ts][r_, c_], (r_, li[r_, c_])), shape=(m, na))
    pn = fe.E[M.par[ts]]; unk = M.pos[pn] >= 0
    nodes = np.unique(pn[unk]); q = len(nodes); ni = np.searchsorted(nodes, pn)
    rr, cc = np.nonzero(unk)
    Mo = sp.csr_matrix((M.sub_mean[ts][rr, cc], (ni[rr, cc], rr)), shape=(q, m))
    Gm = (Mo @ Bd).tocsr(); target = mom_full[nodes].copy()
    if floating and not np.any(~unk):
        Gm = Gm[:-1]; tq = target[:-1]
    else: tq = target
    rhs = np.zeros(na)
    if rhs_loc is not None: np.add.at(rhs, li[r_, c_], -rhs_loc[r_, c_])
    coef = spsolve(sp.bmat([[Mm, -Gm.T], [Gm, None]]).tocsc(), np.concatenate([rhs, tq]))[:na]
    mom = Mo @ (Bd @ coef) - target
    return ue[active], coef, float(np.max(np.abs(mom))) if q else 0.0


def pu_robustness_index(M, K, Theta, in_Vc):
    """per PU member i the discrete PU-weighted Poincare constant (input-independent; development diagnostic)
         floating i:  C_i^2 = sup_q inf_c |||I_h(theta_i (q - c))|||^2 / |||q|||^2_{omega_i}
         boundary i:  C_i^2 = sup_{q = 0 on Dirichlet nodes} |||I_h(theta_i q)|||^2 / |||q|||^2_{omega_i}
    with q a P1 function on the FE nodes of omega_i = elements touching supp theta_i and I_h the nodal interpolant.
    It enters the efficiency estimate eta_alg <= [1 + N_ov max_i C_stab,i (1 + C_i)] |||u_h - v|||."""
    fe = M.fe; Th = Theta.tocsc(); out = np.zeros(Th.shape[1])
    Ke = np.einsum('kid,kjd->kij', fe.grad, fe.grad) * (K * fe.area)[:, None, None]
    for i in range(Th.shape[1]):
        lo, hi = Th.indptr[i], Th.indptr[i + 1]
        nodes, vals = Th.indices[lo:hi], Th.data[lo:hi]
        if len(nodes) == 0: continue
        ks = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
        loc_nodes = np.unique(fe.E[ks]); n = len(loc_nodes); li = np.searchsorted(loc_nodes, fe.E[ks])
        Aw = np.zeros((n, n)); np.add.at(Aw, (li[:, :, None], li[:, None, :]), Ke[ks])
        th = np.zeros(len(fe.X)); th[nodes] = vals; t = th[loc_nodes]
        B = (t[:, None] * Aw) * t[None, :]
        if in_Vc[i]:
            Q = np.linalg.qr(np.column_stack([np.ones(n), np.eye(n)[:, :-1]]))[0][:, 1:]   # complement of constants
            b1 = B @ np.ones(n); B = B - np.outer(b1, b1) / (np.ones(n) @ b1)
        else:
            keep = ~M.dirichlet[loc_nodes]; Q = np.eye(n)[:, keep]
        As, Bs = Q.T @ Aw @ Q, Q.T @ B @ Q
        if As.shape[0] == 0: continue
        L = np.linalg.cholesky(As + 1e-14 * np.trace(As) / len(As) * np.eye(len(As)))
        Li = np.linalg.inv(L)
        out[i] = np.sqrt(max(np.linalg.eigvalsh(Li @ Bs @ Li.T)[-1], 0.0))
    return out


def pu_shepard(M, Psi, Z=None, smin=1e-3):
    """matched partition-of-unity construction for ANY coarse functions Psi (columns on the unknowns) -- used identically
    for operator-adapted functions and for geometric interior hats (pilot amendment 1, matched controls):
      - unknown nodes a with s(a) = sum_c Psi[a, c] >= smin: members theta_c(a) = Psi[a, c] / s(a);
      - unknown nodes with s(a) < smin (fallback): the node's own fine hat is a member and is added to the coarse space;
      - Dirichlet nodes: their fine hats (boundary members, flux free on Dirichlet edges).
    Sum of members = 1 at every node.  Coarse space V_c = span(normalised members, fallback hats) (+ Z).
    Returns (Theta, in_Vc, C, n_fallback)."""
    Psi = Psi.tocsr(); s = np.asarray(Psi @ np.ones(Psi.shape[1])).ravel()
    ok = s >= smin; scale = np.where(ok, 1 / np.where(ok, s, 1), 0.0)
    Tn = (sp.diags(scale) @ Psi).tocsr(); Tn.eliminate_zeros()
    fb = np.nonzero(~ok)[0]
    Fb = sp.csr_matrix((np.ones(len(fb)), (fb, np.arange(len(fb)))), shape=(M.N, len(fb)))
    members = sp.hstack([Tn, Fb]).tocsr()
    keep = np.asarray(abs(members).sum(0)).ravel() > 0
    members = members[:, np.nonzero(keep)[0]].tocoo()
    n = len(M.fe.X); dn = np.nonzero(M.dirichlet)[0]
    full = sp.csr_matrix((members.data, (M.unk[members.row], members.col)), shape=(n, members.shape[1]))
    Dh = sp.csr_matrix((np.ones(len(dn)), (dn, np.arange(len(dn)))), shape=(n, len(dn)))
    Theta = sp.hstack([full, Dh]).tocsr()
    in_Vc = np.concatenate([np.ones(members.shape[1], bool), np.zeros(len(dn), bool)])
    C = members.tocsr()
    if Z is not None and Z.shape[1]: C = sp.hstack([C, sp.csr_matrix(Z)]).tocsr()
    return Theta, in_Vc, C, int(len(fb))
