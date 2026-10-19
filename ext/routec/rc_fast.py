"""Cached versions of the Route C algebraic fluxes for the pilot (implementation only; same mathematics as rc_fem.py).

Every algebraic flux is linear in the residual R, and every local saddle-point matrix depends only on (operator, partition
of unity) -- not on the input.  `batch(Rs)` therefore factorises each patch system once, solves it for all residuals,
and frees it (memory stays small at L4); `__call__(R)` is batch([R]).  `check_against_reference()` verifies equality with rc_fem.py to round-off.

Classes (call(R) returns an rc_fem.Field):
  MVYCache(M, K, A, variant)          MVY 2020 App. 8 ('mvy') or PRVW 2020 Def. 4.3 ('prvw'), fixed lifting r_h
  LiftFreeCache(M, K, A, Theta, in_Vc, C)   lifting-free two-level construction (Prop. 3)
  FixedPUCache(M, K, A, Theta, in_Vc, C)    two-level construction with the fixed lifting (Prop. 2)
  JointCache(M, K)                    global moment-constrained optimum (reference)
"""
import numpy as np, scipy.sparse as sp, scipy.linalg as sl
from scipy.sparse.linalg import splu
import rc_fem as R


class PatchSolver:
    """RT0 mixed problem on `elems` of level `lev`: min ||K^{-1/2}(sigma + w)|| with zero normal flux on the patch boundary
    (free on Dirichlet edges when free_dirichlet), constrained either pointwise (div sigma = data on each element) or, if
    `moments` = (M, ts) is given (flux level only), through the hat moments of the unknown FE nodes touching the patch."""
    def __init__(self, lev, elems, Mloc, free_dirichlet, moments=None):
        elems = np.asarray(elems); m = len(elems); self.m = m; self.elems = elems; self.lev = lev
        ue, loc = np.unique(lev.edge[elems].ravel(), return_inverse=True); loc = loc.reshape(m, 3)
        cnt = np.bincount(loc.ravel(), minlength=len(ue))
        active = cnt == 2
        if free_dirichlet: active |= (cnt == 1) & lev.dirichlet_edge[ue]
        self.floating = not np.any(active & (cnt == 1))
        na = int(active.sum()); idx = -np.ones(len(ue), np.int64); idx[active] = np.arange(na)
        li = idx[loc]; mask = li >= 0
        e3, i3, j3 = np.nonzero(mask[:, :, None] & mask[:, None, :])
        Ml = Mloc[elems]
        Mm = sp.csr_matrix((Ml[e3, i3, j3], (li[e3, i3], li[e3, j3])), shape=(na, na))
        r_, c_ = np.nonzero(mask); self.r_, self.c_, self.li = r_, c_, li
        Bd = sp.csr_matrix((lev.div_int[elems][r_, c_], (r_, li[r_, c_])), shape=(m, na)); self.Bd = Bd
        self.edges, self.na = ue[active], na
        # topology gate (ASTRA review, section 2 / 4D): connectivity is checked component by component
        comp = _components(elems, loc, cnt)
        ncomp = int(comp.max()) + 1
        if self.floating and ncomp != 1:
            raise PatchTopologyError(f'floating patch with {ncomp} edge-connected components')
        if not self.floating:
            free_e = np.nonzero(active & (cnt == 1))[0]
            has_free = np.zeros(ncomp, bool)
            for e in free_e:
                has_free[comp[np.nonzero((loc == e).any(1))[0]]] = True
            if not has_free.all():
                raise PatchTopologyError('a component of a boundary patch has no free Dirichlet edge')
        self.ncomp = ncomp
        if moments is None:
            if self.floating:
                ar = sp.csr_matrix(lev.area[elems][:, None])
                S = sp.bmat([[Mm, -Bd.T, None], [Bd, None, ar], [None, ar.T, None]])
            else:
                S = sp.bmat([[Mm, -Bd.T], [Bd, None]])
            self.kind = 'div'
        else:
            M, ts = moments; fe = M.fe
            pn = fe.E[M.par[ts]]; unk = M.pos[pn] >= 0
            nodes = np.unique(pn[unk]); q = len(nodes); ni = np.searchsorted(nodes, pn)
            rr, cc = np.nonzero(unk)
            Mo = sp.csr_matrix((M.sub_mean[ts][rr, cc], (ni[rr, cc], rr)), shape=(q, m))
            G = (Mo @ Bd).tocsr()
            self.drop = bool(self.floating and not np.any(~unk))
            if self.drop: G = G[:-1]
            S = sp.bmat([[Mm, -G.T], [G, None]])
            self.nodes, self.Mo, self.kind = nodes, Mo, 'mom'
        self.lu = splu(S.tocsc())
        self.ncons = S.shape[0] - na
        self.stats = dict(patches=1, flux_unknowns=na, constraints=self.ncons, lu_nnz=int(self.lu.L.nnz + self.lu.U.nnz),
                          max_patch_elements=m, patch_elements=m)          # patch_elements summed: overlap = sum / #elements
        self.max_defect = 0.0; self.max_target = 0.0

    def solve(self, target, rhs_loc=None):
        """target: per-element data (kind 'div') or per-node moment targets on self.nodes (kind 'mom')."""
        rhs = np.zeros(self.na)
        if rhs_loc is not None: np.add.at(rhs, self.li[self.r_, self.c_], -rhs_loc[self.r_, self.c_])
        if self.kind == 'div':
            g = target * self.lev.area[self.elems]
            b = np.concatenate([rhs, g, [0.0]]) if self.floating else np.concatenate([rhs, g])
        else:
            t = target[:-1] if self.drop else target
            b = np.concatenate([rhs, t])
        coef = self.lu.solve(b)[:self.na]
        if not np.all(np.isfinite(coef)): raise FloatingPointError('non-finite patch solution')
        # full constraint defect, including a row removed as redundant (gate 4D)
        if self.kind == 'div':
            d = np.abs(self.Bd @ coef - g); scale = np.abs(g).max(initial=0.0)
        else:
            d = np.abs(self.Mo @ (self.Bd @ coef) - target); scale = np.abs(target).max(initial=0.0)
        if not (np.all(np.isfinite(d)) and np.isfinite(scale)): raise FloatingPointError('non-finite constraint defect')
        self.max_defect = max(self.max_defect, float(d.max(initial=0.0)))       # absolute; normalised over all patches
        self.max_target = max(self.max_target, float(scale))
        return coef


class PatchTopologyError(RuntimeError):
    pass


def _components(elems, loc, cnt):
    """edge-connected components of the patch elements (two elements sharing an interior patch edge are adjacent)."""
    m = len(elems); parent = np.arange(m)
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    first = {}
    for r in range(m):
        for e in loc[r]:
            if cnt[e] != 2: continue
            if e in first:
                a, b = find(r), find(first[e])
                if a != b: parent[a] = b
            else: first[e] = r
    roots = np.array([find(r) for r in range(m)])
    _, comp = np.unique(roots, return_inverse=True)
    return comp


def _acc(work, solver):
    for k, v in solver.stats.items():
        work[k] = max(work.get(k, 0), v) if k.startswith('max') else work.get(k, 0) + v
    work['max_abs_constraint_defect'] = max(work.get('max_abs_constraint_defect', 0.0), solver.max_defect)
    work['max_abs_constraint_target'] = max(work.get('max_abs_constraint_target', 0.0), solver.max_target)
    work['max_constraint_defect'] = work['max_abs_constraint_defect'] / max(work['max_abs_constraint_target'], 1e-300)


class MVYCache:
    def __init__(self, M, K, A, variant='mvy'):
        self.M, self.K, self.A, self.variant = M, K, A, variant
        fl, L = M.fl, M.L
        self.P0 = R.geometric_P(M, 0); self.A0 = sl.cho_factor((self.P0.T @ A @ self.P0).toarray())
        self.Kb = K[M.par]; self.levels = []; self.work = {}; self.coarse_dim = self.P0.shape[1]
        for j in range(1, L + 2):
            levc, lev = M.levels[j - 1], M.levels[j]
            Mloc = R.local_mass(M, K, j)
            ancc, ancj = M.anc[j - 1], M.anc[j]
            mean_c, grad_c = R._hat_tables(M, j - 1)
            Vc = levc.E[ancc]
            order = np.argsort(Vc.ravel(), kind='stable'); sv = Vc.ravel()[order]
            ua = np.unique(sv); starts = np.searchsorted(sv, ua); ends = np.append(starts[1:], len(sv))
            patches = []
            for a, s0, s1 in zip(ua, starts, ends):
                flat = order[s0:s1]; ts, li = flat // 3, flat % 3
                uj, invj = np.unique(ancj[ts], return_inverse=True)
                rec = dict(ts=ts, li=li, uj=uj, invj=invj, free=bool(M.dmask_b[a]))
                if j > 1:
                    uc, invc = np.unique(ancc[ts], return_inverse=True)
                    cj = np.zeros(len(uj), np.int64); cj[invj] = ancc[ts]
                    rec.update(uc=uc, invc=invc, cpos=np.searchsorted(uc, cj))
                patches.append(rec)
            self.levels.append((j, levc, lev, mean_c, grad_c, patches, Mloc))

    def __call__(self, Rv):
        return self.batch([Rv])[0]

    def batch(self, Rs):
        """fluxes for several residuals at once: each patch is factorised once and freed."""
        self.work = {}                                  # work counts of THIS call
        M, fl = self.M, self.M.fl
        rbs = [R.lift_residual(M, Rv) for Rv in Rs]
        grhos = [M.grad(M.full(self.P0 @ sl.cho_solve(self.A0, self.P0.T @ Rv)))[M.par] for Rv in Rs]
        Fs = [R.Field(M) for _ in Rs]
        for j, levc, lev, mean_c, grad_c, patches, Mloc in self.levels:
            use_rho = (self.variant == 'mvy') or j == 1
            for p in patches:
                ts, li = p['ts'], p['li']
                solver = PatchSolver(lev, p['uj'], Mloc, p['free'])
                for F, rb, grho_b in zip(Fs, rbs, grhos):
                    ig = (rb[ts] * mean_c[ts, li] - (self.Kb[ts] * np.sum(grho_b[ts] * grad_c[ts, li], 1) if use_rho else 0.0)) * fl.area[ts]
                    sj = np.bincount(p['invj'], ig) / lev.area[p['uj']]
                    if j > 1:
                        sc = np.bincount(p['invc'], ig) / levc.area[p['uc']]
                        sj = sj - sc[p['cpos']]
                    F.add(j, solver.edges, solver.solve(sj))
                _acc(self.work, solver)
        return Fs


class _PUBase:
    def __init__(self, M, K, A, Theta, in_Vc, C, moments):
        self.M, self.K, self.A = M, K, A
        fe = M.fe
        self.C = C.tocsr() if sp.issparse(C) else np.asarray(C)
        CA = C.T @ (A @ C); CA = CA.toarray() if sp.issparse(CA) else np.asarray(CA)
        self.CA = sl.cho_factor(CA)
        Mloc_b = R.local_mass(M, K, M.L + 1); self.Mloc_b, self.moments = Mloc_b, moments
        self.work = {}; self.coarse_dim = int(C.shape[1]); self.galerkin_defect = 0.0
        Th = Theta.tocsc(); self.members = []
        for i in range(Th.shape[1]):
            lo, hi = Th.indptr[i], Th.indptr[i + 1]
            nodes, vals = Th.indices[lo:hi], Th.data[lo:hi]
            if len(nodes) == 0: continue
            th = np.zeros(len(fe.X)); th[nodes] = vals
            ks = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
            ts = (3 * ks[:, None] + np.arange(3)).ravel()
            lam = np.einsum('kpi,ki->kp', M.qbary[ts], th[fe.E[M.par[ts]]])
            PW = np.einsum('kipd,kp->kid', M.phi_q[ts], lam * M.wq[ts][:, None])        # target rhs = PW . grad rho
            rec = dict(nodes=nodes, vals=vals, ks=ks, ts=ts, PW=PW, free=not bool(in_Vc[i]))
            tv = th[fe.E[M.par[ts]]]
            rec['th_mean'] = np.sum(M.sub_mean[ts] * tv, 1)
            rec['gth'] = np.einsum('kid,ki->kd', fe.grad[M.par[ts]], tv)
            tbar = th[fe.E[ks]].mean(1)
            rec['Gk'] = (K[ks] * fe.area[ks] * tbar)[:, None, None] * fe.grad[ks]           # moments = sum_k Gk_k(i) . grad rho_k
            self.members.append(rec)

    def solver(self, p):
        return PatchSolver(self.M.fl, p['ts'], self.Mloc_b, p['free'], moments=(self.M, p['ts']) if self.moments else None)

    def __call__(self, Rv):
        return self.batch([Rv])[0]

    def coarse(self, Rv):
        rhs = np.asarray(self.C.T @ Rv).ravel()
        rho = np.asarray(self.C @ sl.cho_solve(self.CA, rhs)).ravel()
        res = np.asarray(self.C.T @ (Rv - self.A @ rho)).ravel()             # Galerkin residual gate, scale |C|^T |R|
        absC = abs(self.C) if sp.issparse(self.C) else np.abs(self.C)
        scale = float(np.asarray(absC.T @ np.abs(Rv)).max())
        gd = float(np.abs(res).max() / max(scale, 1e-300))
        if not (np.isfinite(gd) and np.all(np.isfinite(rho))): raise FloatingPointError('non-finite coarse correction')
        self.galerkin_defect = max(self.galerkin_defect, gd)
        return rho


class LiftFreeCache(_PUBase):
    def __init__(self, M, K, A, Theta, in_Vc, C):
        super().__init__(M, K, A, Theta, in_Vc, C, moments=True)

    def batch(self, Rs):
        self.work = {}                                  # work counts of THIS call
        M, fe = self.M, self.M.fe; n = len(fe.X)
        pre = []
        for Rv in Rs:
            rho = self.coarse(Rv); pre.append((M.grad(M.full(rho)), M.full(Rv - self.A @ rho)))
        Fs = [R.Field(M) for _ in Rs]
        for p in self.members:
            s = self.solver(p); ks = p['ks']
            for F, (grho, rfull) in zip(Fs, pre):
                loc = np.einsum('kid,kd->ki', p['Gk'], grho[ks])
                mom = np.zeros(n); np.add.at(mom, fe.E[ks].ravel(), loc.ravel())
                mom[p['nodes']] += p['vals'] * rfull[p['nodes']]
                rhs = np.einsum('kid,kd->ki', p['PW'], grho[M.par[p['ts']]])
                F.add(M.L + 1, s.edges, s.solve(mom[s.nodes], rhs))
            _acc(self.work, s)
        return Fs


class FixedPUCache(_PUBase):
    def __init__(self, M, K, A, Theta, in_Vc, C):
        super().__init__(M, K, A, Theta, in_Vc, C, moments=False)

    def batch(self, Rs):
        self.work = {}                                  # work counts of THIS call
        M = self.M; Kb = self.K[M.par]; pre = []
        for Rv in Rs:
            rho = self.coarse(Rv); pre.append((R.lift_residual(M, Rv), M.grad(M.full(rho))))
        Fs = [R.Field(M) for _ in Rs]
        for p in self.members:
            s = self.solver(p); ts = p['ts']
            for F, (rb, grho) in zip(Fs, pre):
                data = rb[ts] * p['th_mean'] - Kb[ts] * np.sum(grho[M.par[ts]] * p['gth'], 1)
                rhs = np.einsum('kid,kd->ki', p['PW'], grho[M.par[ts]])
                F.add(M.L + 1, s.edges, s.solve(data, rhs))
            _acc(self.work, s)
        return Fs


class JointCache:
    def __init__(self, M, K):
        self.M = M; fl = M.fl; nb, nE = len(fl.E), fl.n_edge
        Mb = R.local_mass(M, K, M.L + 1)
        I = np.repeat(fl.edge, 3, 1).ravel(); J = np.tile(fl.edge, (1, 3)).ravel()
        Mg = sp.csr_matrix((Mb.ravel(), (I, J)), shape=(nE, nE))
        D = sp.csr_matrix((fl.div_int.ravel(), (np.repeat(np.arange(nb), 3), fl.edge.ravel())), shape=(nb, nE))
        G = (M.Bb @ sp.diags(1 / fl.area) @ D).tocsr()
        self.lu = splu(sp.bmat([[Mg, -G.T], [G, None]]).tocsc()); self.nE = nE
        self.work = dict(patches=1, flux_unknowns=nE, constraints=G.shape[0], lu_nnz=int(self.lu.L.nnz + self.lu.U.nnz),
                         max_patch_elements=nb, max_constraint_defect=0.0); self.coarse_dim = 0

    def __call__(self, Rv):
        sol = self.lu.solve(np.concatenate([np.zeros(self.nE), Rv]))
        return R.Field(self.M).add(self.M.L + 1, np.arange(self.nE), sol[:self.nE])

    def batch(self, Rs):
        return [self(Rv) for Rv in Rs]


def check_against_reference():
    """equality with rc_fem.py on development operator 300 (both families), inputs u_c and 10% noise."""
    import os, sys, time
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, ROOT)
    from scipy.sparse.linalg import spsolve
    from oa_lifting import NestedHierarchy, make_field
    from coarse_meshes import perforated_square
    from oa_vcycle import oa_hierarchy, near_kernel_basis
    from rc_goal import joint_flux
    H = NestedHierarchy(3, coarse=perforated_square(n=9)); M = R.Mesh(H, perforated_square(n=9)); worst = 0.0
    for fam, con in (('channels', 1e4), ('barriers', 1e-4)):
        rng = np.random.default_rng(1001); K = [make_field(H, 'channels', con, rng) for _ in range(301)][300]
        fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
        A = M.stiffness(K); F = M.load(fK); uh = spsolve(A, F); Mb = R.local_mass(M, K, M.L + 1)
        P2 = R.geometric_P(M, 2); uc = P2 @ np.linalg.solve((P2.T @ A @ P2).toarray(), P2.T @ F)
        n = np.random.default_rng(7).standard_normal(M.N); n *= 0.1 * M.energy(K, M.full(uh)) / M.energy(K, M.full(n))
        levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
        Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
        OA = R.pu_operator_adapted(M, Phi3, Z); G2 = R.pu_geometric(M, 2)
        t0 = time.time()
        caches = dict(mvy=MVYCache(M, K, A, 'mvy'), oa=LiftFreeCache(M, K, A, *OA), g2=LiftFreeCache(M, K, A, *G2),
                      oafix=FixedPUCache(M, K, A, *OA), joint=JointCache(M, K))
        tb = time.time() - t0
        for v in (uc, uh + n):
            Rv = F - A @ v; rb = R.lift_residual(M, Rv)
            ref = dict(mvy=R.flux_alg_mvy(M, K, A, rb, Rv)[0], oa=R.flux_alg_pu_liftfree(M, K, A, Rv, *OA, Mloc_b=Mb)[0],
                       g2=R.flux_alg_pu_liftfree(M, K, A, Rv, *G2, Mloc_b=Mb)[0], oafix=R.flux_alg_pu(M, K, A, rb, Rv, *OA, Mloc_b=Mb)[0],
                       joint=joint_flux(M, K, Rv, Mb))
            t0 = time.time(); fast = {k: c.batch([Rv, 2.0 * Rv])[0] for k, c in caches.items()}; tc = time.time() - t0
            for k in ref:
                a, b = R.ps_norm(M, K, ref[k]), R.ps_norm(M, K, fast[k]); worst = max(worst, abs(a - b) / a)
            print(f'{fam}: build {tb:.1f}s, all five fluxes per input {tc:.1f}s; max relative difference so far {worst:.1e}', flush=True)
    return worst


if __name__ == '__main__':
    import json, os, hashlib
    w = check_against_reference()
    print('PASS' if w < 1e-9 else 'FAIL', f'max relative difference {w:.2e}')
    here = os.path.dirname(os.path.abspath(__file__))
    json.dump(dict(max_relative_difference=w, passed=bool(w < 1e-9),
                   rc_fast_sha256=hashlib.sha256(open(os.path.join(here, 'rc_fast.py'), 'rb').read()).hexdigest(),
                   rc_fem_sha256=hashlib.sha256(open(os.path.join(here, 'rc_fem.py'), 'rb').read()).hexdigest()),
              open(os.path.join(here, 'engine_check.json'), 'w'), indent=1)
