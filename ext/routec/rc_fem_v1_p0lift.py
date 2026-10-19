"""Route C, deliverable 2: classical reference certificate (equilibrated fluxes, Prager--Synge, goal interval).

Problem.  -div(K grad u) = f in Omega (unit square minus three polygonal holes), u = 0 on the whole boundary
(outer and hole boundaries), K and f piecewise constant on the fine (level-3) mesh T_h.  P1 conforming FE.

Everything here is written for ARBITRARY primal / dual approximations v, w in V_h (any nodal vector on the unknown
nodes: exact solve, coarse solve, a few cycles, a neural output, a perturbed field).  No property of v other than
v in V_h is used; validity never depends on how v was produced.

Flux spaces: lowest-order Raviart--Thomas RT0 (divergence piecewise constant).  Because f, the goal weight g and the
residual lifting r_h are piecewise constant on T_h, every equilibrium condition below holds EXACTLY (no data
oscillation term); the only error left is floating-point round-off, which is reported.

Main objects
  Mesh(H)                         fine mesh, edges, RT0 data, P1 gradients, coarse geometric meshes T_0..T_3
  lift_residual(R)                r_h in P0(T_h) with (r_h, psi_a) = R_a for every unknown node a (exact, global SPD solve)
  flux_disc(v, f, r_h)            Braess--Schoeberl / MVY Def. 5.1 vertex-patch flux, div = f - r_h
  flux_alg_oracle(r_h)            global RT0 mixed solve, div = r_h (best RT0 algebraic flux; reference only)
  flux_alg_mvy(r_h, rho0)         Mallik--Vohralik--Yousef 2020, Appendix 8, geometric multilevel construction, div = r_h
  flux_alg_pu(r_h, PU, Vc)        two-level construction on the FINE mesh with any partition of unity whose floating
                                  members lie in the coarse space Vc (geometric hats or operator-adapted functions)
  pager_synge(sigma, v)           || K^{-1/2} (sigma + K grad v) ||   (guaranteed energy bound when div sigma = f)
"""
import numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, spsolve


def _areas(X, E):
    return 0.5 * ((X[E[:, 1], 0] - X[E[:, 0], 0]) * (X[E[:, 2], 1] - X[E[:, 0], 1])
                  - (X[E[:, 2], 0] - X[E[:, 0], 0]) * (X[E[:, 1], 1] - X[E[:, 0], 1]))


class Level:
    """one triangulation (coords shared with the fine mesh): edges, orientation, RT0 geometry, P1 gradients."""
    def __init__(self, X, E, dirichlet_mask):
        E = np.array(E, dtype=np.int64)
        s = _areas(X, E)
        flip = s < 0                                   # make every element counter-clockwise
        E[flip] = E[flip][:, [0, 2, 1]]
        self.X, self.E, self.area = X, E, np.abs(s)
        ne = len(E)
        # local edge i is opposite local vertex i: edge (E[:, (i+1)%3], E[:, (i+2)%3])
        a = np.stack([E[:, 1], E[:, 2], E[:, 0]], 1); b = np.stack([E[:, 2], E[:, 0], E[:, 1]], 1)
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * len(X) + hi
        uk, inv = np.unique(key.ravel(), return_inverse=True)
        self.edge = inv.reshape(ne, 3)                 # global edge id of local edge i
        self.n_edge = len(uk)
        self.edge_nodes = np.stack([uk // len(X), uk % len(X)], 1)
        # global orientation: normal points from the element with the smaller id to the larger; boundary edges outward
        cnt = np.bincount(inv, minlength=self.n_edge)
        self.boundary_edge = cnt == 1
        first = np.full(self.n_edge, -1); sign = np.ones((ne, 3))
        for k in range(ne):
            for i in range(3):
                e = self.edge[k, i]
                if first[e] < 0: first[e] = k
                else: sign[k, i] = -1.0
        self.sign = sign                                # +1: outward normal of this element is the global normal
        self.len = np.linalg.norm(X[a] - X[b], axis=2)  # (ne, 3)
        self.dirichlet_edge = self.boundary_edge & dirichlet_mask[self.edge_nodes[:, 0]] & dirichlet_mask[self.edge_nodes[:, 1]]
        # P1 gradients: grad lambda_i constant on each element
        x0, x1, x2 = X[E[:, 0]], X[E[:, 1]], X[E[:, 2]]
        det = 2 * self.area
        g = np.zeros((ne, 3, 2))
        for i, (p, q) in enumerate(((x1, x2), (x2, x0), (x0, x1))):
            g[:, i, 0] = (p[:, 1] - q[:, 1]) / det
            g[:, i, 1] = (q[:, 0] - p[:, 0]) / det
        self.grad = g
        self.centroid = (x0 + x1 + x2) / 3
        self.mid = 0.5 * (X[a] + X[b])                  # (ne, 3, 2) midpoint of local edge i

    def rt0(self, k, x):
        """values (3, npts, 2) of the three local RT0 basis functions of element k at points x (npts, 2), global orientation.
        phi_i(x) = sign_i |e_i| / (2|K|) (x - p_i), p_i the vertex opposite edge i; div phi_i = sign_i |e_i| / |K|."""
        p = self.X[self.E[k]]
        c = self.sign[k] * self.len[k] / (2 * self.area[k])
        return c[:, None, None] * (x[None, :, :] - p[:, None, :])

    def rt0_div(self, k):
        return self.sign[k] * self.len[k] / self.area[k]


class Mesh:
    """fine level T_3 plus geometric coarse levels T_0..T_2 of the nested hierarchy H (NestedHierarchy)."""
    def __init__(self, H):
        from ofwno_core import MeshHierarchy
        self.H = H
        L = H.L; self.L = L
        X = np.asarray(H.coords, float)
        dmask = np.zeros(len(X), bool); dmask[H.boundary] = True
        self.dirichlet = dmask
        self.levels = []
        Ej = []                                           # rebuild levels 0..L deterministically from the coarse mesh
        mh = MeshHierarchy(np.asarray(self._coarse0(H)[0]), np.asarray(self._coarse0(H)[1]))
        for j in range(L + 1):
            if j > 0: mh.refine_one_level()
            Ej.append(np.asarray(mh.elements, np.int64))
        assert np.allclose(np.asarray(mh.coords), X), 'coarse rebuild does not reproduce the fine coordinates'
        for j in range(L + 1):
            self.levels.append(Level(X, Ej[j], dmask))
        self.fine = self.levels[L]
        # ancestors: element i of level j has parent i // 4 at level j-1 (MeshHierarchy appends 4 children per parent)
        nf = len(self.fine.E)
        self.anc = {L: np.arange(nf)}
        for j in range(L - 1, -1, -1):
            self.anc[j] = self.anc[j + 1] // 4
        for j in range(L):                                # check: fine element centroid lies in its ancestor
            assert len(self.levels[j].E) * 4 ** (L - j) == nf
        self.unk = np.asarray(H.nodes[L])
        self.pos = -np.ones(len(X), np.int64); self.pos[self.unk] = np.arange(len(self.unk))
        self.N = len(self.unk)
        # P0 hat-moment map B: (N, n_elem), B[a, K] = |K| / 3 for unknown vertex a of K
        f = self.fine; r, c, v = [], [], []
        for i in range(3):
            m = self.pos[f.E[:, i]] >= 0
            r.append(self.pos[f.E[m, i]]); c.append(np.nonzero(m)[0]); v.append(f.area[m] / 3)
        self.B = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(self.N, len(f.E)))
        self._BBt = splu((self.B @ self.B.T).tocsc())
        # vertex -> elements (fine)
        self.el_of = [[] for _ in range(len(X))]
        for k, t in enumerate(f.E):
            for a in t: self.el_of[a].append(k)

    @staticmethod
    def _coarse0(H):
        from coarse_meshes import perforated_square
        return perforated_square(n=9)

    # ---------------------------------------------------------------- P1 helpers on the fine mesh
    def full(self, v):
        """nodal vector on all fine nodes (zero on Dirichlet nodes) from a vector on the unknowns."""
        x = np.zeros(len(self.fine.X)); x[self.unk] = v; return x

    def grad(self, vfull):
        """(n_elem, 2) constant gradient of a P1 function given by all nodal values."""
        f = self.fine
        return np.einsum('kid,ki->kd', f.grad, vfull[f.E])

    def stiffness(self, K):
        f = self.fine
        Ke = np.einsum('kid,kjd->kij', f.grad, f.grad) * (K * f.area)[:, None, None]
        I = np.repeat(f.E, 3, 1).ravel(); J = np.tile(f.E, (1, 3)).ravel()
        A = sp.csr_matrix((Ke.ravel(), (I, J)), shape=(len(f.X), len(f.X)))
        return A[self.unk][:, self.unk].tocsc()

    def load(self, fK):
        """load vector (fK piecewise constant on fine elements) on the unknowns: (f, psi_a)."""
        return self.B @ fK

    def energy(self, K, e_full):
        g = self.grad(e_full)
        return float(np.sqrt(np.sum(K * self.fine.area * np.sum(g * g, 1))))


def lift_residual(M, R):
    """r_h in P0(T_h) with (r_h, psi_a) = sum_K r_K |K|/3 [a in K] = R_a.  Minimum-norm solution of B r = R in the
    Euclidean norm of the coefficient vector (r_K): r = B^T (B B^T)^{-1} R.  Global but contrast-free (no K)."""
    lam = M._BBt.solve(R)
    r = M.B.T @ lam
    return r


# --------------------------------------------------------------------------------------------------------------------
# local mixed RT0 problems
# --------------------------------------------------------------------------------------------------------------------
def _quad(lev, k):
    """edge-midpoint rule on element k: exact for quadratics.  points (3, 2), weights (3,)."""
    return lev.mid[k], np.full(3, lev.area[k] / 3)


class FineK:
    """element data for RT0 mass matrices on a level-j element when K is piecewise constant on the FINE mesh:
    integrate over the fine descendants of each level-j element (exact: RT0 basis is linear on the parent)."""
    def __init__(self, M, j, K):
        self.M, self.j, self.K = M, j, K
        lev = M.levels[j]; fine = M.fine
        self.children = [[] for _ in range(len(lev.E))]
        for t, p in enumerate(M.anc[j]): self.children[p].append(t)

    def mass(self, k):
        """3x3 matrix (K^{-1} phi_i, phi_j) on level-j element k."""
        lev = self.M.levels[self.j]; fine = self.M.fine; Mk = np.zeros((3, 3))
        for t in self.children[k]:
            x, w = _quad(fine, t)
            ph = lev.rt0(k, x)                         # (3, 3pts, 2)
            Mk += np.einsum('ipd,jpd,p->ij', ph, ph, w) / self.K[t]
        return Mk

    def rhs_target(self, k, target_fn):
        """(K^{-1} w, phi_i) on level-j element k for a target field w given per fine element as w(t, x) -> (npts, 2)."""
        lev = self.M.levels[self.j]; fine = self.M.fine; b = np.zeros(3)
        for t in self.children[k]:
            x, w = _quad(fine, t)
            ph = lev.rt0(k, x)
            b += np.einsum('ipd,pd,p->i', ph, target_fn(t, x), w) / self.K[t]
        return b


def solve_patch(lev, elems, data, mass_fn, rhs_fn=None, free_dirichlet=False):
    """min || K^{-1/2} (sigma + w) ||_patch over sigma in RT0(patch) with div sigma = data (P0, per element of elems),
    sigma . n = 0 on the patch boundary except on Dirichlet boundary edges when free_dirichlet is True.
    w enters through rhs_fn(k) = (K^{-1} w, phi_i)_k (None: w = 0).
    Returns (edge ids, edge coefficients) of the global RT0 field and the max divergence defect."""
    elems = np.asarray(elems)
    eids = lev.edge[elems]                                    # (m, 3)
    ue, loc = np.unique(eids.ravel(), return_inverse=True); loc = loc.reshape(-1, 3)
    # an edge is an unknown iff interior to the patch (shared by two patch elements) or a free Dirichlet edge
    cnt = np.bincount(loc.ravel(), minlength=len(ue))
    active = cnt == 2
    if free_dirichlet: active |= (cnt == 1) & lev.dirichlet_edge[ue]
    idx = -np.ones(len(ue), np.int64); idx[active] = np.arange(active.sum()); na = int(active.sum())
    m = len(elems)
    Mm = np.zeros((na, na)); Bd = np.zeros((m, na)); rhs = np.zeros(na)
    for r, k in enumerate(elems):
        Mk = mass_fn(k); dv = lev.rt0_div(k) * lev.area[k]       # (div phi_i, 1)_k
        bk = rhs_fn(k) if rhs_fn is not None else np.zeros(3)
        li = idx[loc[r]]
        for i in range(3):
            if li[i] < 0: continue
            rhs[li[i]] -= bk[i]
            Bd[r, li[i]] += dv[i]
            for jj in range(3):
                if li[jj] >= 0: Mm[li[i], li[jj]] += Mk[i, jj]
    g = data * lev.area[elems]                                  # (data, 1)_k
    floating = not (free_dirichlet and np.any(lev.dirichlet_edge[ue][cnt == 1]))
    # saddle point [[M, -B^T, 0], [B, 0, c], [0, c^T, 0]] with the mean constraint on gamma when floating
    nb = m + (1 if floating else 0)
    S = np.zeros((na + nb, na + nb)); S[:na, :na] = Mm; S[:na, na:na + m] = -Bd.T; S[na:na + m, :na] = Bd
    b = np.concatenate([rhs, g, [0.0] if floating else []])
    if floating:
        ar = lev.area[elems]; S[na:na + m, na + m] = ar; S[na + m, na:na + m] = ar
    sol = np.linalg.lstsq(S, b, rcond=None)[0] if na == 0 else np.linalg.solve(S, b)
    coef = sol[:na]
    defect = Bd @ coef - g
    return ue[active], coef, float(np.max(np.abs(defect / lev.area[elems]))) if m else 0.0


# --------------------------------------------------------------------------------------------------------------------
# RT0 fields as sums of level-wise coefficient vectors; evaluation, divergence, Prager--Synge norms
# --------------------------------------------------------------------------------------------------------------------
class Field:
    """sum over levels j of RT0(T_j) fields, each given by one coefficient per level-j edge (its normal component)."""
    def __init__(self, M):
        self.M = M; self.c = {}

    def add(self, j, edges, coef):
        if j not in self.c: self.c[j] = np.zeros(self.M.levels[j].n_edge)
        np.add.at(self.c[j], edges, coef)
        return self

    def __iadd__(self, other):
        for j, c in other.c.items():
            self.c[j] = self.c.get(j, 0) + c
        return self

    def scaled(self, s):
        F = Field(self.M); F.c = {j: s * c for j, c in self.c.items()}; return F

    def div(self):
        """divergence on the fine elements (piecewise constant)."""
        M = self.M; d = np.zeros(len(M.fine.E))
        for j, c in self.c.items():
            lev = M.levels[j]
            dj = np.einsum('ki,ki->k', c[lev.edge], lev.sign * lev.len) / lev.area
            d += dj[M.anc[j]]
        return d

    def at_quad(self):
        """values (n_fine, 3, 2) at the fine edge-midpoint quadrature points."""
        M = self.M; fine = M.fine; out = np.zeros((len(fine.E), 3, 2))
        for j, c in self.c.items():
            lev = M.levels[j]; par = M.anc[j]
            p = lev.X[lev.E[par]]                                    # (nf, 3, 2) parent vertices
            coef = c[lev.edge[par]] * lev.sign[par] * lev.len[par] / (2 * lev.area[par])[:, None]   # (nf, 3)
            x = fine.mid                                             # (nf, 3pts, 2)
            out += np.einsum('ki,kpid->kpd', coef, x[:, :, None, :] - p[:, None, :, :])
        return out


def ps_norm(M, K, sigma, vfull=None):
    """|| K^{-1/2} (sigma + K grad v) || (exact quadrature).  With div sigma = f this is >= |||u - v|||."""
    fine = M.fine; s = sigma.at_quad()
    if vfull is not None: s = s + (K[:, None] * M.grad(vfull))[:, None, :]
    return float(np.sqrt(np.sum(np.sum(s * s, 2) * (fine.area / 3)[:, None] / K[:, None])))


# --------------------------------------------------------------------------------------------------------------------
# discretisation flux: fine vertex patches (Braess--Schoeberl with the MVY residual lifting)
# --------------------------------------------------------------------------------------------------------------------
def flux_disc(M, K, fK, rK, vfull):
    """sigma_disc in RT0(T_h): sum over ALL fine vertices a of argmin ||K^{-1/2}(sigma_a + psi_a K grad v)||_{omega_a}
    with div sigma_a = Pi_0((f - r_h) psi_a) - K grad v . grad psi_a; zero normal flux on the patch boundary for
    interior vertices (compatible because (r_h, psi_a) = R_a), free on Dirichlet edges for boundary vertices.
    Result: div sigma_disc = f - r_h exactly."""
    fine = M.fine; gv = M.grad(vfull); F = Field(M); worst = 0.0
    lev = fine
    mass = lambda k: _mass_fine(lev, k, K)
    for a in range(len(fine.X)):
        el = np.asarray(M.el_of[a])
        if len(el) == 0: continue
        li = np.array([int(np.nonzero(fine.E[k] == a)[0][0]) for k in el])
        gpsi = fine.grad[el, li]                                    # (m, 2)
        data = (fK[el] - rK[el]) / 3 - K[el] * np.sum(gv[el] * gpsi, 1)
        def rhs(k, a=a):
            x, w = _quad(fine, k); i = int(np.nonzero(fine.E[k] == a)[0][0])
            lam = _bary(fine, k, x)[:, i]                            # psi_a at the points
            ph = fine.rt0(k, x)
            return np.einsum('ipd,pd,p->i', ph, lam[:, None] * gv[k][None, :], w)      # (K^{-1} psi K grad v, phi)
        e, c, d = solve_patch(fine, el, data, mass, rhs, free_dirichlet=bool(M.dirichlet[a]))
        F.add(M.L, e, c); worst = max(worst, d)
    return F, worst


def _mass_fine(lev, k, K):
    x, w = _quad(lev, k); ph = lev.rt0(k, x)
    return np.einsum('ipd,jpd,p->ij', ph, ph, w) / K[k]


def _bary(lev, k, x):
    """barycentric coordinates (npts, 3) of points x in element k."""
    p = lev.X[lev.E[k]]
    T = np.array([[p[0, 0] - p[2, 0], p[1, 0] - p[2, 0]], [p[0, 1] - p[2, 1], p[1, 1] - p[2, 1]]])
    l01 = np.linalg.solve(T, (x - p[2]).T).T
    return np.column_stack([l01, 1 - l01.sum(1)])


# --------------------------------------------------------------------------------------------------------------------
# algebraic fluxes: div sigma_alg = r_h
# --------------------------------------------------------------------------------------------------------------------
def flux_alg_oracle(M, K, rK):
    """global RT0 mixed problem: the minimum of ||K^{-1/2} sigma|| over RT0(T_h) with div sigma = r_h (all boundary
    edges free: the whole boundary is Dirichlet).  Reference only (cost of a global mixed solve)."""
    fine = M.fine; ne, nE = len(fine.E), fine.n_edge
    rows, cols, vals = [], [], []; br, bc, bv = [], [], []
    for k in range(ne):
        Mk = _mass_fine(fine, k, K); ed = fine.edge[k]
        rows.append(np.repeat(ed, 3)); cols.append(np.tile(ed, 3)); vals.append(Mk.ravel())
        br.append(np.full(3, k)); bc.append(ed); bv.append(fine.rt0_div(k) * fine.area[k])
    Mg = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nE, nE))
    Bg = sp.csr_matrix((np.concatenate(bv), (np.concatenate(br), np.concatenate(bc))), shape=(ne, nE))
    S = sp.bmat([[Mg, -Bg.T], [Bg, None]]).tocsc()
    sol = spsolve(S, np.concatenate([np.zeros(nE), rK * fine.area]))
    return Field(M).add(len(M.levels) - 1, np.arange(nE), sol[:nE])


def geometric_P(M, j):
    """composite geometric prolongation from level-j unknowns to fine unknowns (H.P)."""
    H = M.H; P = None
    for k in range(j + 1, H.L + 1): P = H.P[k] if P is None else H.P[k] @ P
    return sp.identity(M.N, format='csr') if P is None else P.tocsr()


def hats_full(M, j):
    """nodal values on ALL fine nodes of the level-j hat functions of ALL level-j vertices (incl. Dirichlet):
    sparse (n_nodes, n_level_j_vertices) and the list of level-j vertex ids.  Built by interpolating the coarse hats."""
    lev = M.levels[j]; verts = np.unique(lev.E)
    X = M.fine.X; n = len(X)
    # every fine node lies in some level-j element; evaluate barycentrics there
    rows, cols, vals = [], [], []
    owner = -np.ones(n, np.int64)
    par = M.anc[j]
    for t, k in enumerate(par):
        for a in M.fine.E[t]:
            if owner[a] < 0: owner[a] = k
    col = -np.ones(n, np.int64); col[verts] = np.arange(len(verts))
    for a in range(n):
        k = owner[a]
        if k < 0: continue
        lam = _bary(lev, k, X[a][None, :])[0]
        for i in range(3):
            if abs(lam[i]) > 1e-14:
                rows.append(a); cols.append(col[lev.E[k, i]]); vals.append(lam[i])
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, len(verts))), verts


def _hat_on_fine(M, j, a):
    """for level-j vertex a: fine elements of its level-j patch, the hat's gradient there and its value at their
    centroids (the hat is linear on each level-j element, so the centroid value is its fine-element mean)."""
    lev = M.levels[j]; fine = M.fine
    pk = np.nonzero(np.any(lev.E == a, 1))[0]                       # level-j patch elements
    ts = np.nonzero(np.isin(M.anc[j], pk))[0]                        # fine elements inside
    par = M.anc[j][ts]
    li = np.array([int(np.nonzero(lev.E[k] == a)[0][0]) for k in par])
    grad = lev.grad[par, li]
    val = np.array([_bary(lev, k, fine.centroid[t][None, :])[0, i] for t, k, i in zip(ts, par, li)])
    return pk, ts, grad, val


def flux_alg_mvy(M, K, A, rK, R, target=False):
    """Mallik--Vohralik--Yousef (JCAM 2020) Appendix 8 / Papez--Rude--Vohralik--Wohlmuth (CMAME 2020) multilevel
    algebraic flux with RT0 on the GEOMETRIC hierarchy T_0 ... T_L:
      rho_0 in P1(T_0) cap H^1_0: (K grad rho_0, grad v_0) = (r_h, v_0)                    [coarse solve]
      level j = 1..L, every vertex a of T_{j-1}: RT0(T_j) on omega^a_{j-1},
          div sigma = Pi_j (Id - Pi_{j-1}) (r_h psi^a_{j-1} - K grad rho_0 . grad psi^a_{j-1}),  Pi_0 := 0,
          zero normal flux on the patch boundary (interior a) / except on Dirichlet edges (boundary a),
          minimising ||K^{-1/2} sigma|| (target=False, as written in MVY (8.3a)).
    Telescoping gives div sigma_alg = Pi_L r_h = r_h."""
    fine = M.fine; L = M.L
    P0 = geometric_P(M, 0); A0 = (P0.T @ A @ P0).toarray()
    rho0 = P0 @ np.linalg.solve(A0, P0.T @ R)
    grho = M.grad(M.full(rho0))
    F = Field(M); worst = 0.0; nfloat_bad = 0.0
    for j in range(1, L + 1):
        levc, lev = M.levels[j - 1], M.levels[j]
        FK = FineK(M, j, K)
        verts = np.unique(levc.E)
        for a in verts:
            pk, ts, gpsi, vpsi = _hat_on_fine(M, j - 1, a)
            ig = (rK[ts] * vpsi - K[ts] * np.sum(grho[ts] * gpsi, 1)) * fine.area[ts]     # int_t g_a
            elj = M.anc[j][ts]; elc = M.anc[j - 1][ts]
            uj, invj = np.unique(elj, return_inverse=True)
            sj = np.bincount(invj, ig) / lev.area[uj]                      # Pi_j g on the level-j elements
            if j > 1:
                uc, invc = np.unique(elc, return_inverse=True)
                sc = np.bincount(invc, ig) / levc.area[uc]                 # Pi_{j-1} g on the level-(j-1) elements
                cpar = M.anc[j - 1][np.array([np.nonzero(M.anc[j] == k)[0][0] for k in uj])]
                sj = sj - sc[np.searchsorted(uc, cpar)]
            rhs = None
            if target:
                def rhs(k, a=a, j=j):   # (K^{-1} psi K grad rho0, phi) = (psi grad rho0, phi) over fine children
                    b = np.zeros(3); levc_ = M.levels[j - 1]
                    for t in FK.children[k]:
                        x, w = _quad(fine, t); kc = M.anc[j - 1][t]; i = int(np.nonzero(levc_.E[kc] == a)[0][0])
                        lam = _bary(levc_, kc, x)[:, i]
                        b += np.einsum('ipd,pd,p->i', lev.rt0(k, x), lam[:, None] * grho[t][None, :], w)
                    return b
            e, c, d = solve_patch(lev, uj, sj, FK.mass, rhs, free_dirichlet=bool(M.dirichlet[a]))
            F.add(j, e, c); worst = max(worst, d)
            if not M.dirichlet[a]: nfloat_bad = max(nfloat_bad, abs(np.sum(sj * lev.area[uj])))
    return F, dict(defect=worst, compat=nfloat_bad, rho=rho0)


def flux_alg_pu(M, K, A, rK, R, Theta, in_Vc, C, target=True):
    """two-level algebraic flux on the FINE mesh with an arbitrary partition of unity.
    Theta: (n_nodes, m) nodal values (P1 functions) with Theta.sum(1) == 1 at every fine node.
    in_Vc[i]: member i vanishes on the Dirichlet boundary and lies in span(C) -> floating patch (zero normal flux on
              its whole patch boundary), compatible because rho is the Galerkin solution in span(C);
              otherwise the member must touch the Dirichlet boundary and its patch flux is free on Dirichlet edges.
    C: (N, n_c) coarse basis on the unknowns.  rho = C (C^T A C)^{-1} C^T R.
    Patch problem i: div sigma_i = Pi_0(r_h theta_i) - K grad rho . grad theta_i on supp theta_i, minimising
    ||K^{-1/2}(sigma_i + theta_i K grad rho)|| (target=True, Braess--Schoeberl form) or ||K^{-1/2} sigma_i||.
    Sum: div sigma_alg = Pi_0(r_h) - K grad rho . grad 1 = r_h."""
    fine = M.fine
    Ac = (C.T @ (A @ C)); Ac = Ac.toarray() if sp.issparse(Ac) else Ac
    rho = C @ np.linalg.solve(Ac, C.T @ R)
    rho = np.asarray(rho).ravel()
    grho = M.grad(M.full(rho))
    Th = Theta.tocsc(); F = Field(M); worst = 0.0; compat = 0.0
    E = fine.E
    mass = lambda k: _mass_fine(fine, k, K)
    sizes = []
    for i in range(Th.shape[1]):
        lo, hi = Th.indptr[i], Th.indptr[i + 1]
        nodes, vals = Th.indices[lo:hi], Th.data[lo:hi]
        if len(nodes) == 0: continue
        th = np.zeros(len(fine.X)); th[nodes] = vals
        ts = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
        tv = th[E[ts]]                                               # (m, 3) nodal values on the elements
        gth = np.einsum('kid,ki->kd', fine.grad[ts], tv)
        data = rK[ts] * tv.mean(1) - K[ts] * np.sum(grho[ts] * gth, 1)
        rhs = None
        if target:
            def rhs(k, th=th):
                x, w = _quad(fine, k); lam = _bary(fine, k, x) @ th[E[k]]
                return np.einsum('ipd,pd,p->i', fine.rt0(k, x), lam[:, None] * grho[k][None, :], w)
        e, c, d = solve_patch(fine, ts, data, mass, rhs, free_dirichlet=not bool(in_Vc[i]))
        F.add(M.L, e, c); worst = max(worst, d); sizes.append(len(ts))
        if in_Vc[i]: compat = max(compat, abs(np.sum(data * fine.area[ts])))
    return F, dict(defect=worst, compat=compat, rho=rho, patch_sizes=np.array(sizes))


def pu_geometric(M, j):
    """partition of unity: level-j hats of all level-j vertices; members at interior vertices are in V_c = P1(T_j)."""
    Th, verts = hats_full(M, j)
    in_Vc = ~M.dirichlet[verts]
    return Th.tocsr(), in_Vc, geometric_P(M, j)


def pu_operator_adapted(M, Psi, Z=None):
    """partition of unity from operator-adapted coarse functions (columns of Psi, on the unknowns), Shepard-normalised at
    the nodes, completed by the fine hats of the Dirichlet nodes.  V_c = span(normalised functions) (+ Z).
    Requires s = Psi 1 > 0 at every unknown node (checked)."""
    s = np.asarray(Psi @ np.ones(Psi.shape[1])).ravel()
    if s.min() <= 0: raise ValueError(f'operator-adapted functions do not have a positive sum (min {s.min():.3e})')
    Tn = sp.diags(1 / s) @ Psi.tocsr()                                # normalised, on the unknowns
    n = len(M.fine.X); dnodes = np.nonzero(M.dirichlet)[0]
    full = sp.csr_matrix((Tn.tocoo().data, (M.unk[Tn.tocoo().row], Tn.tocoo().col)), shape=(n, Tn.shape[1]))
    Dh = sp.csr_matrix((np.ones(len(dnodes)), (dnodes, np.arange(len(dnodes)))), shape=(n, len(dnodes)))
    Theta = sp.hstack([full, Dh]).tocsr()
    in_Vc = np.concatenate([np.ones(Tn.shape[1], bool), np.zeros(len(dnodes), bool)])
    C = Tn if Z is None or Z.shape[1] == 0 else sp.hstack([Tn, sp.csr_matrix(Z)]).tocsr()
    return Theta, in_Vc, C
