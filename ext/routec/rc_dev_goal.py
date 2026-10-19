"""development run (seed-1 development operators only; NOT the pilot): goal certificates for exact, coarse, learned,
iterative and perturbed inputs, discrete and continuous targets, against level-5 references."""
import os, sys, time, json, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE, os.path.join(ROOT, 'diagnostics_cost')): sys.path.insert(0, p)
from scipy.sparse.linalg import spsolve, splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis, OAData
import rc_fem as R, rc_goal as G
import stage1_models as SM

fam = sys.argv[1]; ids = [int(s) for s in sys.argv[2].split(',')]
con = {'channels': 1e4, 'barriers': 1e-4}[fam]
cm = perforated_square(n=9)
H = NestedHierarchy(3, coarse=cm); M = R.Mesh(H, cm)
H5 = NestedHierarchy(5, coarse=cm); M5 = R.Mesh(H5, cm)
assert np.array_equal(H5.nodes[3], H.nodes[3])
P35 = (H5.P[5] @ H5.P[4]).tocsr()
CK = json.load(open(os.path.join(EXT, 'checkpoint_map.json')))
ck = [r for r in CK if r['family'] == fam and r['data_seed'] == 1 and r['level'] == 3 and r['k0'] == 2 and r['oa_steps'] == 4
      and r['coarse'] == 'perforated' and r['model'] == 'oa-vcb' and r['init_seed'] == 0 and r['load'] != 'random'][0]
scale = json.load(open(os.path.join(EXT, 'frozen_scales.json')))['scales'][f'{fam}/fixed-load']
gK, om = G.goal_weight(M, (0.5, 0.45), 0.08)
rng = np.random.default_rng(1001); aa = [make_field(H, 'channels', con, rng) for _ in range(max(ids) + 1)]
out = []
for idx in ids:
    K = aa[idx]; fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)
    A = M.stiffness(K); F = M.load(fK); Gv = M.load(gK); Mb = R.local_mass(M, K, M.L + 1)
    lu = splu(A); uh = lu.solve(F); zh = lu.solve(Gv)
    nu_, nz_ = M.energy(K, M.full(uh)), M.energy(K, M.full(zh))
    # reference (level 5, K f g inherited)
    K5 = K[np.arange(len(M5.fe.E)) // 16]; f5 = fK[np.arange(len(M5.fe.E)) // 16]; g5 = gK[np.arange(len(M5.fe.E)) // 16]
    A5 = M5.stiffness(K5); lu5 = splu(A5); u5 = lu5.solve(M5.load(f5)); z5 = lu5.solve(M5.load(g5))
    t0 = time.time()
    cache = os.path.join(HERE, 'cache', f'ref5_{fam}_{idx}.npz'); os.makedirs(os.path.dirname(cache), exist_ok=True)
    if os.path.exists(cache):
        c_ = np.load(cache); e5u, e5z = float(c_['e5u']), float(c_['e5z'])
    else:   # exact solves: R = 0, so only the discretisation flux (vertex patches) is needed
        Mb5 = R.local_mass(M5, K5, M5.L + 1); zero = np.zeros(len(M5.fl.E))
        Su, _ = R.flux_disc(M5, K5, f5, zero, M5.full(u5), Mb5); e5u = R.ps_norm(M5, K5, Su, M5.full(u5))
        Sz, _ = R.flux_disc(M5, K5, g5, zero, M5.full(z5), Mb5); e5z = R.ps_norm(M5, K5, Sz, M5.full(z5))
        np.savez(cache, e5u=e5u, e5z=e5z, u5=u5, z5=z5)
    J5 = float(M5.load(g5) @ u5); r5 = e5u * e5z
    print(f'{fam} id {idx}: reference L5 built {time.time() - t0:.0f}s; J(u_5) = {J5:.6e} +- {r5:.2e}; J(u_h) = {Gv @ uh:.6e}; '
          f'eta5_u/|||u5||| {e5u / M5.energy(K5, M5.full(u5)):.3e}', flush=True)
    # inputs
    P2 = R.geometric_P(M, 2); A2 = (P2.T @ A @ P2).toarray()
    uc = P2 @ np.linalg.solve(A2, P2.T @ F); zc = P2 @ np.linalg.solve(A2, P2.T @ Gv)
    D = OAData(H, [K], 3, 4, 'cpu', H.level_coords, near_kernel=True)
    net, _, _ = SM.load(os.path.join(ROOT, ck['path']), D)
    u_nn = uc + SM.apply(net, D, (F - A @ uc)[None, :], scale)[0]
    def pcg(b, its):
        d = A.diagonal(); x = np.zeros(M.N); r = b.copy(); z = r / d; p = z.copy(); rz = r @ z
        for _ in range(its):
            Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        return x
    rn = np.random.default_rng(100 + idx)
    def noisy(x, ref, nrm):
        n = rn.standard_normal(M.N); return ref + n * (0.1 * nrm / M.energy(K, M.full(n)))
    pairs = {'exact (u_h, z_h)': (uh, zh), 'coarse (u_c, z_c)': (uc, zc), 'OA-VCB primal, coarse dual': (u_nn, zc),
             'Jacobi-PCG 30 (both)': (pcg(F, 30), pcg(Gv, 30)), '10% noise (both)': (noisy(None, uh, nu_), noisy(None, zh, nz_))}
    levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
    Phi3 = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
    PU = {'OA2+Z': R.pu_operator_adapted(M, Phi3, Z), 'G2': R.pu_geometric(M, 2)}
    methods = [('MVY multilevel, fixed lifting', 'mvy', None), ('G2, lifting-free', 'pu-liftfree', PU['G2']),
               ('OA2+Z, lifting-free', 'pu-liftfree', PU['OA2+Z']), ('joint optimum (reference)', 'joint', None)]
    for pname, (v, w) in pairs.items():
        eu, ez = M.energy(K, M.full(uh - v)), M.energy(K, M.full(zh - w))
        centre = float(Gv @ v + (F - A @ v) @ w)
        rem_h = float(Gv @ uh) - centre
        v5, w5 = P35 @ v, P35 @ w
        eu5, ez5 = M5.energy(K5, M5.full(u5 - v5)), M5.energy(K5, M5.full(z5 - w5))
        for mname, meth, pu in methods:
            t0 = time.time()
            bu = G.energy_bounds(M, K, A, fK, v, meth, pu, Mb); bz = G.energy_bounds(M, K, A, gK, w, meth, pu, Mb); tt = time.time() - t0
            radD, radC = bu['alg'] * bz['alg'], bu['tot'] * bz['tot']
            okD = abs(rem_h) <= radD * (1 + 1e-9) + 1e3 * np.finfo(float).eps * (abs(float(Gv @ uh)) + abs(centre))   # round-off allowance
            loC, hiC = centre - radC, centre + radC
            okC = (loC <= J5 - r5) and (J5 + r5 <= hiC)             # the whole reference bracket inside the interval
            rec = dict(family=fam, id=idx, pair=pname, method=mname, J_h=float(Gv @ uh), J5=J5, r5=r5, centre=centre,
                       rem_h=rem_h, radD=radD, radC=radC, eu=eu, ez=ez, eta_u_alg=bu['alg'], eta_z_alg=bz['alg'],
                       eta_u_tot=bu['tot'], eta_z_tot=bz['tot'], eu_cont_lo=eu5, ez_cont_lo=ez5,
                       eu_cont_hi=float(np.hypot(eu5, e5u)), ez_cont_hi=float(np.hypot(ez5, e5z)),
                       okD=bool(okD), okC=bool(okC), moment_defect=max(bu['moment_defect'], bz['moment_defect']),
                       div_defect=max(bu['div_defect'], bz['div_defect']), seconds=tt)
            out.append(rec)
            effD = radD / (eu * ez) if eu * ez > 0 else float('nan')
            print(f'  {pname:28s} {mname:30s} D: |rem| {abs(rem_h):.2e} rad {radD:.2e} (rad/(e_u e_z) {effD:6.2f}) {"ok" if okD else "FAIL"} | '
                  f'C: rad {radC:.2e} rel {radC / abs(J5):.1e} {"ok" if okC else "FAIL"}  eta_u_tot/|||u-v|||<= {bu["tot"] / rec["eu_cont_lo"] if rec["eu_cont_lo"] > 0 else float("nan"):.2f} | {tt:.0f}s', flush=True)
json.dump(out, open(os.path.join(HERE, f'dev_goal_{fam}.json'), 'w'), indent=1)
