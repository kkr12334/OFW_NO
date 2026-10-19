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
from scipy.sparse.linalg import splu, eigsh
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
DEV2 = os.path.join(HERE, 'ROUTE_C_PILOT_DEVIATION_2_2026-10-18.md')
# deviation 2: residuals, goal centres and polarization lower bounds are accumulated in extended precision and carry explicit
# evaluation terms; the stopping path has the same gates as the ordinary-input path (ASTRA pilot review, section 2)
LD = np.longdouble; LD_INFO = np.finfo(LD); UX = float(LD_INFO.eps) / 2          # unit roundoff of the extended format
def gam(m): return m * UX / (1 - m * UX)                                          # Higham gamma_m
SLACK = 1 + 1e-9                                                                  # relative slack on certified radii (reconstruction defects)
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
                           refine_steps=REFINE_STEPS, ref_tol=REF_TOL,
                           extended=dict(dtype=str(np.dtype(LD)), mantissa_bits=int(LD_INFO.nmant) + 1, unit_roundoff=UX, platform=sys.platform)),
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
        gate('deviation 2 note present', os.path.exists(DEV2))
        records['manifest']['deviation_2'] = sha(DEV2)
    gate('extended precision available (>= 64-bit mantissa; numpy longdouble is platform-dependent)', LD_INFO.nmant + 1 >= 64,
         dict(mantissa_bits=int(LD_INFO.nmant) + 1, platform=sys.platform))
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
            # ---- deviation 2: extended-precision evaluation with explicit error terms
            absA = abs(Acsr); KROW = int(np.diff(Acsr.indptr).max()); NN = M.N
            lam_min = float(eigsh(A.tocsc(), k=1, sigma=0, which='LM', return_eigenvectors=False)[0])
            gate(f'{tag}: smallest eigenvalue estimate positive', lam_min > 0, lam_min)
            def res_full(b, v):
                """residual b - A v accumulated in extended precision: (r_ext, r rounded to double, componentwise a priori bound on
                |r_ext - r_exact| (Higham gamma_{KROW+1}), estimated energy norm of A^{-1}(r_exact - r_double))."""
                p = Ald * v.astype(LD)[Acsr.indices]; Av = np.add.reduceat(p, Acsr.indptr[:-1]); Av[np.diff(Acsr.indptr) == 0] = 0
                r = b.astype(LD) - Av; rd = r.astype(np.float64)
                bnd = gam(KROW + 1) * (np.abs(b) + absA @ np.abs(v)) * (1 + 1e-6)
                lo = (r - rd.astype(LD)).astype(np.float64)                         # the part lost by rounding to double (known)
                a_est = M.energy(K, M.full(lu.solve(lo))) + float(np.linalg.norm(bnd)) / np.sqrt(lam_min)
                return r, rd, bnd, a_est
            def dot_ld(x, y): return (x.astype(LD) * y.astype(LD)).sum()
            def centre_eval(Gv, v, b, w):
                """centre c = G.v + r(v).w in extended precision and a priori bound on its evaluation error (rigorous for IEEE
                extended arithmetic, given the stored vectors; includes the residual-evaluation bound times |w|)."""
                r, rd, bnd, _ = res_full(b, v)
                c = dot_ld(Gv, v) + dot_ld(r, w)
                tau = (gam(NN) * float(np.abs(Gv) @ np.abs(v)) + gam(NN) * float(np.abs(rd) @ np.abs(w)) * (1 + 1e-6)
                       + float(bnd @ np.abs(w)) + gam(2) * abs(float(c))) * (1 + 1e-6)
                return float(c), tau
            J_ref, tau_ref = {}, {}
            for g in range(len(GOALS)):
                dn = f'dual{g}'; Gv_ = loads[g + 1][1]
                c_, t_ = centre_eval(Gv_, exact['primal'], F, exact[dn])          # corrected reference J = G.u + r(u).z
                J_ref[g] = c_
                tau_ref[g] = t_ + (ref_err['primal'] * M.energy(K, M.full(exact['primal']))) * (ref_err[dn] * M.energy(K, M.full(exact[dn])))
            def polar_lower(rs_u, rs_z, s_, sign):
                """Cauchy-Schwarz lower bound |||s e_u +- e_z/s||| >= R(rho)/|||rho||| for ANY rho (here the OA+Z Galerkin
                projection), with R evaluated in extended precision and its evaluation bound subtracted."""
                (ru, rud, bu, _), (rz, rzd, bz, _) = rs_u, rs_z
                rcd = s_ * rud + sign * rzd / s_
                rho = rho_lb(rcd); nr = M.energy(K, M.full(rho))
                if nr == 0: return 0.0, 0.0
                Rr = float(((LD(s_) * ru + LD(sign / s_) * rz) * rho.astype(LD)).sum())
                tR = (gam(NN + 3) * float((s_ * np.abs(rud) + np.abs(rzd) / s_) @ np.abs(rho))
                      + float((s_ * bu + bz / s_) @ np.abs(rho))) * (1 + 1e-6)
                return max(Rr - tR, 0.0) / nr, rho
            def interval(v, w, Gv_, b_dual, Su, Sz, rs_u=None, rs_z=None):
                """product and polarization intervals with all terms recorded separately."""
                rs_u = rs_u or res_full(F, v); rs_z = rs_z or res_full(b_dual, w)
                c_, tev = centre_eval(Gv_, v, F, w)
                nu_, nz_ = R.ps_norm(M, K, Su), R.ps_norm(M, K, Sz); au, az = rs_u[3], rs_z[3]
                s_ = np.sqrt(nz_ / nu_) if nu_ > 0 and nz_ > 0 else 1.0
                Up = R.ps_norm(M, K, comb(Su, Sz, s_, 1 / s_)) + s_ * au + az / s_
                Um = R.ps_norm(M, K, comb(Su, Sz, s_, -1 / s_)) + s_ * au + az / s_
                Lp, rho_p = polar_lower(rs_u, rs_z, s_, +1); Lm, rho_m = polar_lower(rs_u, rs_z, s_, -1)
                lo = 0.25 * (Lp ** 2 - Um ** 2) - (SLACK - 1) * 0.25 * (Lp ** 2 + Um ** 2) - tev
                hi = 0.25 * (Up ** 2 - Lm ** 2) + (SLACK - 1) * 0.25 * (Up ** 2 + Lm ** 2) + tev
                prod_cert = nu_ * nz_; prod_tot = (nu_ + au) * (nz_ + az) * SLACK + tev
                return dict(centre=c_, tau_eval=tev, a_u=au, a_z=az, eta_u=nu_, eta_z=nz_, product_radius=prod_cert, product_radius_total=prod_tot,
                            polar_lo=lo, polar_hi=hi, polar_centre=c_ + 0.5 * (lo + hi), polar_radius=0.5 * (hi - lo), s=s_)
            def galerkin_defect_lb(Rv):
                rho = rho_lb(Rv); res = np.asarray(Cl.T @ (Rv - A @ rho)).ravel()
                return float(np.abs(res).max() / max(float(np.asarray(abs(Cl).T @ np.abs(Rv)).max()), 1e-300))
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
            Cl = PU['OAZ'][2]; CAl = sl.cho_factor((Cl.T @ (A @ Cl)).toarray())
            def rho_lb(Rv): return np.asarray(Cl @ sl.cho_solve(CAl, np.asarray(Cl.T @ Rv).ravel())).ravel()
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
            # lower bounds for polarization: projection onto the OA+Z coarse space (Cl, rho_lb above; method-independent; validated below)
            refabs = {nm: ref_err[nm] * norms[nm] for nm in norms}
            records['operators'].append(dict(id=idx, contrast=c, gates_passed=len(gate_log) - g0, status='gates passed; outcomes pending')); save()
            # ---------------------------------------------------------------- outcomes, one method in memory at a time
            eta, eta_fp, goals, stop, work = {}, {}, [], {}, {}
            pairs = [(pin, 'coarse' if pin == 'oavcb' else pin) for pin in inputs['primal'] if pin != 'exact']
            keys = [(nm, i) for nm, _ in loads for i in inputs[nm] if i != 'exact']
            RSF = {(nm, i): res_full(dict(loads)[nm], inputs[nm][i]) for nm, i in keys}
            Rs = [RSF[k][1] for k in keys]
            for m in METHODS:
                tb = time.time(); op = build[m]()
                Fs = dict(zip(keys, op.batch(Rs)))
                for (nm, i), Rv in zip(keys, Rs):
                    S = Fs[(nm, i)]
                    mom = float(np.max(np.abs(M.Bb @ S.div() - Rv)) / max(np.max(np.abs(Rv)), 1e-300))
                    val = R.ps_norm(M, K, S); afp = RSF[(nm, i)][3]
                    gate(f'{tag} {m} {nm} {i}: finite bound and full moment constraints', np.isfinite(val) and mom <= 1e-8, mom)
                    gate(f'{tag} {m} {nm} {i}: guarantee eta + a >= |||e||| - reference error (deviation 2)',
                         (val + afp) * SLACK >= err[(nm, i)] - refabs[nm], dict(eta=val, a=afp, err=err[(nm, i)], ref=refabs[nm]))
                    eta[(nm, i, m)] = val; eta_fp[(nm, i, m)] = afp
                if hasattr(op, 'galerkin_defect'):
                    gate(f'{tag} {m}: Galerkin residual of the coarse correction', op.galerkin_defect <= 1e-8, op.galerkin_defect)
                gate(f'{tag} {m}: patch constraint defects incl. removed rows', op.work.get('max_constraint_defect', 0.0) <= 1e-8,
                     op.work.get('max_constraint_defect'))
                if m == METHODS[0]:
                    gdl = max(galerkin_defect_lb(RSF[k_][1]) for k_ in keys)
                    gate(f'{tag}: lower-bound projection Galerkin residual (ordinary inputs)', gdl <= 1e-8, gdl)
                # goal intervals: product and polarization
                for g in range(len(GOALS)):
                    dn = f'dual{g}'; Gv = loads[g + 1][1]; Jh = J_ref[g]
                    for pin, din in pairs:
                        v, w = inputs['primal'][pin], inputs[dn][din]
                        I_ = interval(v, w, Gv, dict(loads)[dn], Fs[('primal', pin)], Fs[(dn, din)], RSF[('primal', pin)], RSF[(dn, din)])
                        rem = Jh - I_['centre']
                        okp = abs(rem) <= I_['product_radius_total'] + tau_ref[g]
                        okq = I_['polar_lo'] - tau_ref[g] <= rem <= I_['polar_hi'] + tau_ref[g]
                        legacy = 1e3 * np.finfo(float).eps * (abs(Jh) + abs(I_['centre']))
                        gate(f'{tag} {m} goal {g} {pin}: product interval (with evaluation terms) contains J_ref (deviation 2)', okp,
                             dict(rem=rem, radius=I_['product_radius_total'], tau_ref=tau_ref[g]))
                        gate(f'{tag} {m} goal {g} {pin}: polarization interval (with evaluation terms) contains J_ref (deviation 2)', okq,
                             dict(rem=rem, lo=I_['polar_lo'], hi=I_['polar_hi'], tau_ref=tau_ref[g]))
                        goals.append(dict(method=m, goal=g, primal=pin, dual=din, Jh=Jh, tau_ref=tau_ref[g], rem=rem, legacy_allowance=legacy,
                                          legacy_check=bool(abs(rem) <= I_['product_radius'] * (1 + 1e-9) + legacy),
                                          eu=err[('primal', pin)], ez=err[(dn, din)], **I_))
                # deployable certified stopping (goal 0)
                if m in STOP_METHODS:
                    Gv = loads[1][1]; b1 = loads[1][1]; rows = []
                    RSu = [res_full(F, xs[k]) for k in CHECKS]; RSz = [res_full(b1, zs[k]) for k in CHECKS]
                    SR = [x[1] for x in RSu] + [x[1] for x in RSz]
                    SF = op.batch(SR)
                    # deviation 2: the stopping evaluations get the same gates as the ordinary-input path
                    for t, Rv in enumerate(SR):
                        S = SF[t]; mom = float(np.max(np.abs(M.Bb @ S.div() - Rv)) / max(np.max(np.abs(Rv)), 1e-300)); val = R.ps_norm(M, K, S)
                        gate(f'{tag} {m} stopping {t}: finite bound and full moment constraints', np.isfinite(val) and val >= 0 and mom <= 1e-8, mom)
                    if hasattr(op, 'galerkin_defect'):
                        gate(f'{tag} {m} stopping: Galerkin residual of the coarse correction', op.galerkin_defect <= 1e-8, op.galerkin_defect)
                    gate(f'{tag} {m} stopping: patch constraint defects incl. removed rows', op.work.get('max_constraint_defect', 0.0) <= 1e-8,
                         op.work.get('max_constraint_defect'))
                    for t, k in enumerate(CHECKS):
                        v, w = xs[k], zs[k]; Su, Sz = SF[t], SF[len(CHECKS) + t]
                        I_ = interval(v, w, Gv, b1, Su, Sz, RSu[t], RSz[t])
                        eu_t, ez_t = M.energy(K, M.full(exact['primal'] - v)), M.energy(K, M.full(exact['dual0'] - w))
                        gate(f'{tag} {m} stopping it {k}: guarantee eta + a >= |||e||| - reference error, primal and dual',
                             (I_['eta_u'] + I_['a_u']) * SLACK >= eu_t - refabs['primal'] and (I_['eta_z'] + I_['a_z']) * SLACK >= ez_t - refabs['dual0'],
                             dict(eta_u=I_['eta_u'], eu=eu_t, eta_z=I_['eta_z'], ez=ez_t))
                        gdl = max(galerkin_defect_lb(RSu[t][1]), galerkin_defect_lb(RSz[t][1]))
                        gate(f'{tag} {m} stopping it {k}: lower-bound projection Galerkin residual', gdl <= 1e-8, gdl)
                        rem = J_ref[0] - I_['centre']
                        okp = abs(rem) <= I_['product_radius_total'] + tau_ref[0]
                        okq = I_['polar_lo'] - tau_ref[0] <= rem <= I_['polar_hi'] + tau_ref[0]
                        gate(f'{tag} {m} stopping it {k}: product interval (with evaluation terms) contains J_ref', okp,
                             dict(rem=rem, radius=I_['product_radius_total'], tau_ref=tau_ref[0]))
                        gate(f'{tag} {m} stopping it {k}: polarization interval (with evaluation terms) contains J_ref', okq,
                             dict(rem=rem, lo=I_['polar_lo'], hi=I_['polar_hi'], tau_ref=tau_ref[0]))
                        legacy = 1e3 * np.finfo(float).eps * (abs(J_ref[0]) + abs(I_['centre']))
                        rows.append(dict(it=k, J_ref=J_ref[0], tau_ref=tau_ref[0], rem=rem, eu=eu_t, ez=ez_t, galerkin_defect_lb=gdl,
                                         legacy_check=bool(abs(rem) <= I_['product_radius'] * (1 + 1e-9) + legacy), **I_))
                    res = {}
                    for eps in EPS:
                        for kind, rk, ck in (('product', 'product_radius_total', 'centre'), ('polar', 'polar_radius', 'polar_centre')):
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
            rec.update(status='complete', ref_err=ref_err, lam_min_est=lam_min, J_ref={str(g): J_ref[g] for g in J_ref},
                       tau_ref={str(g): tau_ref[g] for g in tau_ref}, eta_fp={f'{a}|{b}|{m}': v for (a, b, m), v in eta_fp.items()}, peak_rss_gb=peak_gb(), other_warnings=other_warnings, near_kernel=int(Z.shape[1]), shepard_fallback_nodes=fallback, perturbation_seed=pert_seed, seconds=time.time() - t0,
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
