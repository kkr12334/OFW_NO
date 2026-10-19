"""Route C pilot runner, as amended by ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md (ASTRA review of the scoping note).

    python3 rc_pilot.py FAMILY MODE LEVEL
      FAMILY: channels | barriers
      MODE:   smoke  (development seed-1 operator 300, contrast 1e4 only; never touches the pilot seeds)
              pilot  (fresh seeds 2026101801 channels / 2026101802 barriers; refuses to run if the seed already appears in
                      any .json/.log record under ext/, or if the amendment hash file is missing or does not match)
      LEVEL:  3 (main arms) | 4 (mesh-size arm)

Design (frozen in the amendment):
  geometry  make_field(H3, 'channels', base, rng), base 1e4 / 1e-4; mask = (a != 1); same geometry at every contrast.
            L4 inherits mask, load and goal regions from the L3 ancestor elements.
  K(c)      channels: c on the mask, 1 elsewhere; barriers: 1/c on the mask, 1 elsewhere.
  f         channels 1; barriers 1 off the mask (independent of c).
  goals     unions of L3 elements whose centroid lies within 0.08 of (0.5,0.45), (0.15,0.15), (0.85,0.85).
  inputs    exact; coarse Galerkin (level L-1); Jacobi-PCG 10/30/100; exact + 10% white noise; exact + 10% smooth
            perturbation; OA-VCB one pass (primal, L3, c = 1e4).
  methods   L3: M0 M0p M1 M1S M1SZ M2 M3 M4 M5 M6;  L4: M0 M1SZ M3 M6 (see METHODS below).
Gates are evaluated and saved before outcomes; any gate failure stops the run with the reason recorded.
"""
import os, sys, time, json, hashlib, glob, resource, warnings, numpy as np, scipy.sparse as sp, scipy.linalg as sl
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE, os.path.join(ROOT, 'diagnostics_cost')): sys.path.insert(0, p)
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis, OAData
import rc_fem as R, rc_fast as FST

FAM, MODE, LEVEL = sys.argv[1], sys.argv[2], int(sys.argv[3])
assert FAM in ('channels', 'barriers') and MODE in ('smoke', 'pilot') and LEVEL in (3, 4)
SEED = {'channels': 2026101801, 'barriers': 2026101802}[FAM] if MODE == 'pilot' else 1001
IDS = (list(range(8)) if LEVEL == 3 else [0, 1]) if MODE == 'pilot' else [300]
CONTRASTS = ([1.0, 1e2, 1e4, 1e6] if LEVEL == 3 else [1.0, 1e4]) if MODE == 'pilot' else [1e4]
GOALS = [(0.5, 0.45), (0.15, 0.15), (0.85, 0.85)]
METHODS = ['M0', 'M0p', 'M1', 'M1S', 'M1SZ', 'M2', 'M3', 'M4', 'M5', 'M6'] if LEVEL == 3 else ['M0', 'M1SZ', 'M3', 'M6']
STOP_METHODS = ['M0', 'M3', 'M6'] if LEVEL == 3 else []
CHECKS = [5, 10, 20, 40, 80, 160, 320]
EPS = [1e-3, 1e-6, 1e-2, 1e-4]                        # deployable stopping (amendment section 6); 1e-3, 1e-6 declared in the note
ABS_TOL = 1e-12                                       # fixed absolute radius tolerance for near-zero goals (units of u)
FLOOR = 1e-10                                         # near-exact floor for ratio endpoints (relative energy error)
SHEPARD_MIN = 1e-3                                    # Shepard denominators below this use the fallback member
REFINE_STEPS = 2; REF_TOL = 1e-12                     # deviation 1: reference solves (ROUTE_C_PILOT_DEVIATION_1_2026-10-18.md)
DEV1 = os.path.join(HERE, 'ROUTE_C_PILOT_DEVIATION_1_2026-10-18.md')
OUT = os.path.join(HERE, 'runs', f'{MODE}_{FAM}_L{LEVEL}.json'); os.makedirs(os.path.dirname(OUT), exist_ok=True)
AMEND = os.path.join(HERE, 'ROUTE_C_PILOT_AMENDMENT_1_2026-10-18.md')
AMEND_SHA = os.path.join(HERE, 'ROUTE_C_PILOT_AMENDMENT_1_SHA256.txt')


def sha(p):
    with open(p, 'rb') as fh: return hashlib.sha256(fh.read()).hexdigest()


def read_text(p):
    with open(p, errors='ignore') as fh: return fh.read()


class GateFailure(RuntimeError):
    pass


gate_log = []
def gate(name, ok, detail=None):
    gate_log.append(dict(name=name, passed=bool(ok), detail=detail))
    if not ok: raise GateFailure(f'{name}: {detail}')


records = dict(design=dict(family=FAM, mode=MODE, level=LEVEL, seed=SEED, ids=IDS, contrasts=CONTRASTS, goals=GOALS, methods=METHODS,
                           stop_methods=STOP_METHODS, checks=CHECKS, eps=EPS, abs_tol=ABS_TOL, floor=FLOOR, shepard_min=SHEPARD_MIN,
                           refine_steps=REFINE_STEPS, ref_tol=REF_TOL),
               gates=gate_log, operators=[], status='running')

def peak_gb(): return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
def save():
    records['peak_rss_gb'] = peak_gb()
    with open(OUT, 'w') as fh: json.dump(records, fh)

try:
    # ------------------------------------------------------------------ run-level gates (before any data are generated)
    if MODE == 'pilot':
        gate('amendment hash file present', os.path.exists(AMEND_SHA) and os.path.exists(AMEND))
        declared = read_text(AMEND_SHA).split()[0]
        gate('amendment matches its declared SHA-256', sha(AMEND) == declared, declared)
        # records of this pilot itself (runs/pilot_*, e.g. the L3 record when the L4 arm starts) are not development records
        own = lambda f: os.path.dirname(os.path.abspath(f)).startswith(os.path.join(HERE, 'runs')) and os.path.basename(f).startswith('pilot_')
        hits = [f for f in glob.glob(os.path.join(EXT, '**', '*.json'), recursive=True) + glob.glob(os.path.join(EXT, '**', '*.log'), recursive=True)
                if not own(f) and str(SEED) in read_text(f)]
        gate(f'seed {SEED} absent from every record', not hits, hits)
    records['manifest'] = {os.path.relpath(p, ROOT): sha(p) for p in
                           [os.path.join(HERE, n) for n in ('rc_fem.py', 'rc_fast.py', 'rc_goal.py', 'rc_pilot.py')] +
                           [os.path.join(ROOT, n) for n in ('oa_vcycle.py', 'oa_lifting.py', 'coarse_meshes.py', 'ofwno_core.py')]}
    if MODE == 'pilot': records['manifest']['amendment'] = sha(AMEND)
    if MODE == 'pilot':
        gate('deviation 1 note present', os.path.exists(DEV1))
        records['manifest']['deviation_1'] = sha(DEV1)
    save()

    cm = perforated_square(n=9)
    H3 = NestedHierarchy(3, coarse=cm); M3m = R.Mesh(H3, cm) if LEVEL == 4 else None
    H = H3 if LEVEL == 3 else NestedHierarchy(4, coarse=cm)
    M = R.Mesh(H, cm); L = H.L
    base = 1e4 if FAM == 'channels' else 1e-4
    rng = np.random.default_rng(SEED)
    draws = [make_field(H3, 'channels', base, rng) for _ in range(max(IDS) + 1)]
    child = np.arange(len(M.fe.E)) // 4 ** (L - 3)
    c3 = (R.Mesh(H3, cm) if LEVEL == 3 else M3m).fe.centroid
    goal_masks = [np.hypot(c3[:, 0] - x, c3[:, 1] - y)[child] < 0.08 for x, y in GOALS]
    goal_w = [om / M.fe.area[om].sum() for om in goal_masks]
    CK = json.loads(read_text(os.path.join(EXT, 'checkpoint_map.json')))
    SCALE = json.loads(read_text(os.path.join(EXT, 'frozen_scales.json')))['scales'][f'{FAM}/fixed-load']
    CKP = [r for r in CK if r['family'] == FAM and r['data_seed'] == 1 and r['level'] == 3 and r['k0'] == 2 and r['oa_steps'] == 4
           and r['coarse'] == 'perforated' and r['model'] == 'oa-vcb' and r['init_seed'] == 0 and r['load'] != 'random'][0]
    for p in (os.path.join(ROOT, CKP['path']), os.path.join(EXT, 'checkpoint_map.json'), os.path.join(EXT, 'frozen_scales.json')):
        records['manifest'][os.path.relpath(p, ROOT)] = sha(p)
    save()

    def jacobi_pcg(A, b, its, record=None):
        d = A.diagonal(); x = np.zeros(len(b)); r = b.copy(); z = r / d; p = z.copy(); rz = r @ z; out = {}
        for k in range(1, its + 1):
            Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
            if record is not None and k in record: out[k] = x.copy()
        return (x, out) if record is not None else x

    def comb(Fa, Fb, a, b):
        F_ = R.Field(M)
        for j in set(Fa.c) | set(Fb.c): F_.c[j] = a * Fa.c.get(j, 0) + b * Fb.c.get(j, 0)
        return F_

    t_start = time.time()
    for idx in IDS:
        mask = (draws[idx] != 1.0)[child]
        for c in CONTRASTS:
            t0 = time.time(); g0 = len(gate_log)
            tag = f'id {idx} c={c:.0e}'
            wctx = warnings.catch_warnings(record=True); caught = wctx.__enter__(); warnings.simplefilter('always')
            K = np.where(mask, c if FAM == 'channels' else 1.0 / c, 1.0)
            fK = np.ones(len(K)) if FAM == 'channels' else (~mask).astype(float)
            A = M.stiffness(K); lu = splu(A); F = M.load(fK)
            loads = [('primal', F)] + [(f'dual{g}', M.load(gw)) for g, gw in enumerate(goal_w)]
            # deviation 1: reference solves refined twice with the residual accumulated in extended precision; the gate checks the
            # energy-norm error of the reference itself (estimated with one further extended-precision residual)
            Acsr = A.tocsr(); Ald = Acsr.data.astype(np.longdouble)
            def res_ld(b, u):
                p = Ald * u.astype(np.longdouble)[Acsr.indices]
                Au = np.add.reduceat(p, Acsr.indptr[:-1]); Au[np.diff(Acsr.indptr) == 0] = 0
                return (b.astype(np.longdouble) - Au).astype(np.float64)
            exact, ref_err = {}, {}
            for nm, b in loads:
                u = lu.solve(b)
                for _ in range(REFINE_STEPS): u = u + lu.solve(res_ld(b, u))
                d = lu.solve(res_ld(b, u)); exact[nm] = u
                ref_err[nm] = float(M.energy(K, M.full(d)) / max(M.energy(K, M.full(u)), 1e-300))
            gate(f'{tag}: exact solves finite; reference energy error <= {REF_TOL:g} (deviation 1)',
                 all(np.all(np.isfinite(u)) for u in exact.values()) and max(ref_err.values()) <= REF_TOL, ref_err)
            # ---------------------------------------------------------------- partitions of unity and coarse spaces
            levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
            Phi = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr()
            Z = near_kernel_basis(levs[0][0])
            Pg = R.geometric_P(M, L - 1)
            shep = dict(G1S=R.pu_shepard(M, Pg, None, SHEPARD_MIN), G1SZ=R.pu_shepard(M, Pg, Z, SHEPARD_MIN),
                        OAZ=R.pu_shepard(M, Phi, Z, SHEPARD_MIN), OA=R.pu_shepard(M, Phi, None, SHEPARD_MIN))
            fallback = {k: v[3] for k, v in shep.items()}
            PU = dict(G1=R.pu_geometric(M, L - 1), G2=R.pu_geometric(M, L - 2), **{k: v[:3] for k, v in shep.items()})
            for nm, (Th, inv, C) in PU.items():
                sdef = float(np.abs(np.asarray(Th.sum(1)).ravel() - 1).max())
                gate(f'{tag}: PU {nm} sums to one at every node', sdef <= 1e-12, sdef)
                gate(f'{tag}: PU {nm} finite', np.all(np.isfinite(Th.data)) and (not sp.issparse(C) or np.all(np.isfinite(C.data))))
            build = dict(M0=lambda: FST.MVYCache(M, K, A, 'mvy'), M0p=lambda: FST.MVYCache(M, K, A, 'prvw'),
                         M1=lambda: FST.LiftFreeCache(M, K, A, *PU['G1']), M1S=lambda: FST.LiftFreeCache(M, K, A, *PU['G1S']),
                         M1SZ=lambda: FST.LiftFreeCache(M, K, A, *PU['G1SZ']), M2=lambda: FST.LiftFreeCache(M, K, A, *PU['G2']),
                         M3=lambda: FST.LiftFreeCache(M, K, A, *PU['OAZ']), M4=lambda: FST.LiftFreeCache(M, K, A, *PU['OA']),
                         M5=lambda: FST.FixedPUCache(M, K, A, *PU['OAZ']), M6=lambda: FST.JointCache(M, K))
            # ---------------------------------------------------------------- inputs (seeds and normalisations recorded)
            Ac = sl.cho_factor((Pg.T @ A @ Pg).toarray())
            pert_seed = [SEED, idx, int(round(np.log10(c) * 10)) + 100]
            rn = np.random.default_rng(pert_seed)
            inputs = {}
            for nm, b in loads:
                u = exact[nm]; nrm = M.energy(K, M.full(u))
                wn = rn.standard_normal(M.N); wn *= 0.1 * nrm / M.energy(K, M.full(wn))
                sm = lu.solve(M.Bb @ rn.standard_normal(len(M.fl.E))); sm *= 0.1 * nrm / M.energy(K, M.full(sm))
                inp = dict(exact=u, coarse=Pg @ sl.cho_solve(Ac, Pg.T @ b), pcg10=jacobi_pcg(A, b, 10), pcg30=jacobi_pcg(A, b, 30),
                           pcg100=jacobi_pcg(A, b, 100), noise10=u + wn, smooth10=u + sm)
                if nm == 'primal' and LEVEL == 3 and c == 1e4:
                    import stage1_models as SM
                    D = OAData(H, [K], 3, 4, 'cpu', H.level_coords, near_kernel=True)
                    net, _, _ = SM.load(os.path.join(ROOT, CKP['path']), D)
                    inp['oavcb'] = inp['coarse'] + SM.apply(net, D, (b - A @ inp['coarse'])[None, :], SCALE)[0]
                gate(f'{tag}: inputs for {nm} finite', all(np.all(np.isfinite(x)) for x in inp.values()))
                inputs[nm] = inp
            norms = {nm: M.energy(K, M.full(exact[nm])) for nm, _ in loads}
            err = {(nm, i): M.energy(K, M.full(exact[nm] - v)) for nm, _ in loads for i, v in inputs[nm].items()}
            resolved = {k: (v > FLOOR * norms[k[0]]) for k, v in err.items()}
            # stopping iterates (primal and goal-0 dual)
            if STOP_METHODS:
                _, xs = jacobi_pcg(A, F, CHECKS[-1], record=set(CHECKS)); _, zs = jacobi_pcg(A, loads[1][1], CHECKS[-1], record=set(CHECKS))
            # lower bounds for polarization: Galerkin projection onto the OA+Z coarse space (method-independent)
            Cl = PU['OAZ'][2]; CAl = sl.cho_factor((Cl.T @ (A @ Cl)).toarray())
            def rho_lb(Rv): return np.asarray(Cl @ sl.cho_solve(CAl, np.asarray(Cl.T @ Rv).ravel())).ravel()
            records['operators'].append(dict(id=idx, contrast=c, gates_passed=len(gate_log) - g0, status='gates passed; outcomes pending')); save()
            # ---------------------------------------------------------------- outcomes, one method in memory at a time
            eta, goals, stop, work = {}, [], {}, {}
            pairs = [(pin, 'coarse' if pin == 'oavcb' else pin) for pin in inputs['primal'] if pin != 'exact']
            for m in METHODS:
                tb = time.time(); op = build[m]()
                keys = [(nm, i) for nm, _ in loads for i in inputs[nm] if i != 'exact']
                Rs = [dict(loads)[nm] - A @ inputs[nm][i] for nm, i in keys]
                Fs = dict(zip(keys, op.batch(Rs)))
                for (nm, i), Rv in zip(keys, Rs):
                    S = Fs[(nm, i)]
                    mom = float(np.max(np.abs(M.Bb @ S.div() - Rv)) / max(np.max(np.abs(Rv)), 1e-300))
                    val = R.ps_norm(M, K, S)
                    gate(f'{tag} {m} {nm} {i}: finite bound and full moment constraints', np.isfinite(val) and mom <= 1e-8, mom)
                    gate(f'{tag} {m} {nm} {i}: guarantee eta >= |||e||| (floating-point allowance 1e-9 rel)', val >= err[(nm, i)] * (1 - 1e-9),
                         dict(eta=val, err=err[(nm, i)]))
                    eta[(nm, i, m)] = val
                if hasattr(op, 'galerkin_defect'):
                    gate(f'{tag} {m}: Galerkin residual of the coarse correction', op.galerkin_defect <= 1e-8, op.galerkin_defect)
                gate(f'{tag} {m}: patch constraint defects incl. removed rows', op.work.get('max_constraint_defect', 0.0) <= 1e-8,
                     op.work.get('max_constraint_defect'))
                # goal intervals: product and polarization
                for g in range(len(GOALS)):
                    dn = f'dual{g}'; Gv = loads[g + 1][1]; Jh = float(Gv @ exact['primal'])
                    for pin, din in pairs:
                        v, w = inputs['primal'][pin], inputs[dn][din]
                        centre = float(Gv @ v + (F - A @ v) @ w); rem = Jh - centre
                        nu_, nz_ = eta[('primal', pin, m)], eta[(dn, din, m)]
                        s_ = np.sqrt(nz_ / nu_)
                        Up = R.ps_norm(M, K, comb(Fs[('primal', pin)], Fs[(dn, din)], s_, 1 / s_))
                        Um = R.ps_norm(M, K, comb(Fs[('primal', pin)], Fs[(dn, din)], s_, -1 / s_))
                        ru, rz = rho_lb(F - A @ v), rho_lb(Gv - A @ w)
                        Lp, Lm = M.energy(K, M.full(s_ * ru + rz / s_)), M.energy(K, M.full(s_ * ru - rz / s_))
                        lo, hi = 0.25 * (Lp ** 2 - Um ** 2), 0.25 * (Up ** 2 - Lm ** 2)
                        allow = 1e3 * np.finfo(float).eps * (abs(Jh) + abs(centre))
                        okp, okq = abs(rem) <= nu_ * nz_ * (1 + 1e-9) + allow, lo - allow <= rem <= hi + allow
                        gate(f'{tag} {m} goal {g} {pin}: product interval contains J_h (allowance {allow:.1e})', okp)
                        gate(f'{tag} {m} goal {g} {pin}: polarization interval contains J_h', okq)
                        goals.append(dict(method=m, goal=g, primal=pin, dual=din, Jh=Jh, centre=centre, rem=rem, allowance=allow,
                                          product_radius=nu_ * nz_, polar_lo=lo, polar_hi=hi, eu=err[('primal', pin)], ez=err[(dn, din)]))
                # deployable certified stopping (goal 0)
                if m in STOP_METHODS:
                    Gv = loads[1][1]; rows = []
                    SF = op.batch([F - A @ xs[k] for k in CHECKS] + [Gv - A @ zs[k] for k in CHECKS])
                    r0 = None
                    for t, k in enumerate(CHECKS):
                        v, w = xs[k], zs[k]; Su, Sz = SF[t], SF[len(CHECKS) + t]
                        nu_, nz_ = R.ps_norm(M, K, Su), R.ps_norm(M, K, Sz); s_ = np.sqrt(nz_ / nu_)
                        Up = R.ps_norm(M, K, comb(Su, Sz, s_, 1 / s_)); Um = R.ps_norm(M, K, comb(Su, Sz, s_, -1 / s_))
                        ru, rz = rho_lb(F - A @ v), rho_lb(Gv - A @ w)
                        Lp, Lm = M.energy(K, M.full(s_ * ru + rz / s_)), M.energy(K, M.full(s_ * ru - rz / s_))
                        lo, hi = 0.25 * (Lp ** 2 - Um ** 2), 0.25 * (Up ** 2 - Lm ** 2)
                        cp = float(Gv @ v + (F - A @ v) @ w)
                        rows.append(dict(it=k, centre=cp, product_radius=nu_ * nz_, polar_centre=cp + 0.5 * (lo + hi), polar_radius=0.5 * (hi - lo)))
                    res = {}
                    for eps in EPS:
                        for kind, rk, ck in (('product', 'product_radius', 'centre'), ('polar', 'polar_radius', 'polar_centre')):
                            hit = next((r['it'] for r in rows if r[rk] <= eps / (1 + eps) * abs(r[ck]) or r[rk] <= ABS_TOL), None)
                            trig = None
                            if hit is not None:
                                r = [r for r in rows if r['it'] == hit][0]
                                trig = 'relative' if r[rk] <= eps / (1 + eps) * abs(r[ck]) else 'absolute'
                            res[f'{kind}_eps{eps:g}'] = dict(first_check=hit, trigger=trig)
                    stop[m] = dict(rows=rows, attainment=res, checks_evaluated=len(CHECKS), work=dict(op.work))
                work[m] = dict(op.work, coarse_dim=int(getattr(op, 'coarse_dim', 0)), build_and_eval_seconds=time.time() - tb)
                del op, Fs
            # ---------------------------------------------------------------- input-independent diagnostics
            diag = {}
            for nm_pu in ('G1', 'G1SZ', 'OAZ'):
                ci = R.pu_robustness_index(M, K, PU[nm_pu][0], PU[nm_pu][1]); fl_ = ci[PU[nm_pu][1]]
                th = PU[nm_pu][0].tocsc(); order = np.argsort(np.where(PU[nm_pu][1], ci, -1))[::-1][:3]
                diag[nm_pu] = dict(floating_max=float(fl_.max()), floating_p95=float(np.quantile(fl_, 0.95)),
                                   worst3=[dict(index=float(ci[i]), centroid=M.fe.X[th.indices[th.indptr[i]:th.indptr[i + 1]]].mean(0).tolist()) for i in order])
            if LEVEL == 3 and IDS.index(idx) < 2:
                fl = M.fl; nb, nE = len(fl.E), fl.n_edge
                Mb = R.local_mass(M, K, M.L + 1)
                I = np.repeat(fl.edge, 3, 1).ravel(); J = np.tile(fl.edge, (1, 3)).ravel()
                Mg = sp.csr_matrix((Mb.ravel(), (I, J)), shape=(nE, nE)).tocsc()
                Dg = sp.csr_matrix((fl.div_int.ravel(), (np.repeat(np.arange(nb), 3), fl.edge.ravel())), shape=(nb, nE))
                G = (M.Bb @ sp.diags(1 / fl.area) @ Dg).tocsr(); Sm = G @ splu(Mg).solve(G.T.toarray()); Sm = 0.5 * (Sm + Sm.T)
                diag['C_glob'] = float(1 / np.sqrt(sl.eigh(Sm, A.toarray(), eigvals_only=True, subset_by_index=[0, 0])[0]))
            wctx.__exit__(None, None, None)
            from scipy.sparse.linalg import MatrixRankWarning
            numw = [w for w in caught if issubclass(w.category, (RuntimeWarning, sl.LinAlgWarning, MatrixRankWarning))]
            other_warnings = sorted({f'{w.category.__name__}: {str(w.message)[:120]}' for w in caught if w not in numw})
            gate(f'{tag}: no numerical or solver warnings', not numw, [f'{w.category.__name__}: {w.message}' for w in numw][:5])
            rec = records['operators'][-1]
            rec.update(status='complete', ref_err=ref_err, peak_rss_gb=peak_gb(), other_warnings=other_warnings, near_kernel=int(Z.shape[1]), shepard_fallback_nodes=fallback, perturbation_seed=pert_seed, seconds=time.time() - t0,
                       norms=norms, err={f'{a}|{b}': v for (a, b), v in err.items()}, resolved={f'{a}|{b}': bool(v) for (a, b), v in resolved.items()},
                       eta={f'{a}|{b}|{m}': v for (a, b, m), v in eta.items()}, goals=goals, stop=stop, work=work, diag=diag)
            save()
            w3 = max((eta[(nm, i, 'M3')] / err[(nm, i)] for (nm, i, m) in eta if m == 'M3' and resolved[(nm, i)]), default=float('nan')) if 'M3' in METHODS else float('nan')
            print(f'{FAM} L{L} {tag} nk={Z.shape[1]}: {time.time() - t0:.0f}s; worst M3 effectivity {w3:.3f}; gates passed {len(gate_log) - g0}', flush=True)
    records['status'] = 'complete'
except GateFailure as e:
    records['status'] = f'STOPPED at gate: {e}'
except FST.PatchTopologyError as e:
    gate_log.append(dict(name='patch topology', passed=False, detail=str(e))); records['status'] = f'STOPPED at gate: patch topology: {e}'
except Exception as e:                                # any other failure (singular coarse space, non-finite values, ...) also stops
    gate_log.append(dict(name='unexpected error', passed=False, detail=f'{type(e).__name__}: {e}'))
    records['status'] = f'STOPPED by error: {type(e).__name__}: {e}'
finally:
    save()
print(records['status'], '->', OUT)
