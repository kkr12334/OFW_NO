"""Development check (seed-1 operator 300 only): does the classical polarization bound sharpen the discrete goal interval?
For s > 0,  a(e_u, e_z) = 1/4 ( |||s e_u + e_z/s|||^2 - |||s e_u - e_z/s|||^2 ).
Upper bounds U_+- = ||K^{-1/2}(s sigma_u +- sigma_z / s)|| (the fluxes are linear in the residual, so no new solves);
lower bounds L_+- = |||rho_+-||| with rho the coarse Galerkin correction (a(e, rho) = R(rho) = |||rho|||^2, so |||e||| >= |||rho|||).
Interval for J(u_h):  centre + [ (L_+^2 - U_-^2)/4 , (U_+^2 - L_-^2)/4 ],  s^2 = eta_z / eta_u.
Compared with the product interval centre +- eta_u eta_z (same fluxes, method M3 and the joint optimum M6)."""
import os, sys, json, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE, os.path.join(ROOT, 'diagnostics_cost')): sys.path.insert(0, p)
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis, OAData
import rc_fem as R, rc_fast as FST, rc_goal as RG, stage1_models as SM

cm = perforated_square(n=9); H = NestedHierarchy(3, coarse=cm); M = R.Mesh(H, cm)
gK, _ = RG.goal_weight(M, (0.5, 0.45), 0.08)
out = []
for fam, con in (('channels', 1e4), ('barriers', 1e-4)):
    rng = np.random.default_rng(1001); K = [make_field(H, 'channels', con, rng) for _ in range(301)][300]
    fK = np.ones(len(K)) if fam == 'channels' else (K >= 0.5).astype(float)      # as in the scoping-note development runs
    A = M.stiffness(K); lu = splu(A); F = M.load(fK); Gv = M.load(gK); uh = lu.solve(F); zh = lu.solve(Gv); Jh = float(Gv @ uh)
    P2 = R.geometric_P(M, 2); A2 = (P2.T @ A @ P2).toarray()
    uc = P2 @ np.linalg.solve(A2, P2.T @ F); zc = P2 @ np.linalg.solve(A2, P2.T @ Gv)
    ck = [r for r in json.load(open(os.path.join(EXT, 'checkpoint_map.json'))) if r['family'] == fam and r['data_seed'] == 1 and r['level'] == 3
          and r['k0'] == 2 and r['oa_steps'] == 4 and r['coarse'] == 'perforated' and r['model'] == 'oa-vcb' and r['init_seed'] == 0 and r['load'] != 'random'][0]
    D = OAData(H, [K], 3, 4, 'cpu', H.level_coords, near_kernel=True); net, _, _ = SM.load(os.path.join(ROOT, ck['path']), D)
    scale = json.load(open(os.path.join(EXT, 'frozen_scales.json')))['scales'][f'{fam}/fixed-load']
    unn = uc + SM.apply(net, D, (F - A @ uc)[None, :], scale)[0]
    def pcg(b, its):
        d = A.diagonal(); x = np.zeros(M.N); r = b.copy(); z = r / d; p = z.copy(); rz = r @ z
        for _ in range(its):
            Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        return x
    levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
    Phi = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
    m3 = FST.LiftFreeCache(M, K, A, *R.pu_operator_adapted(M, Phi, Z)); m6 = FST.JointCache(M, K)
    for pname, (v, w) in {'coarse/coarse': (uc, zc), 'OA-VCB/coarse': (unn, zc), 'PCG30/PCG30': (pcg(F, 30), pcg(Gv, 30))}.items():
        Ru, Rz = F - A @ v, Gv - A @ w
        centre = float(Gv @ v + Ru @ w); rem = Jh - centre
        eu, ez = M.energy(K, M.full(uh - v)), M.energy(K, M.full(zh - w))
        rho_u, rho_z = m3.coarse(Ru), m3.coarse(Rz)                            # Galerkin corrections in the OA coarse space
        for mname, op in (('M3', m3), ('M6', m6)):
            Su, Sz = op.batch([Ru, Rz])
            nu_, nz_ = R.ps_norm(M, K, Su), R.ps_norm(M, K, Sz)
            s = np.sqrt(nz_ / nu_)
            Sp = R.Field(M); Sp += Su.scaled(s) if hasattr(Su, 'scaled') else Su;
            # linear combinations of fields
            def comb(a, b):
                F_ = R.Field(M)
                for j in set(Su.c) | set(Sz.c): F_.c[j] = a * Su.c.get(j, 0) + b * Sz.c.get(j, 0)
                return F_
            Up, Um = R.ps_norm(M, K, comb(s, 1 / s)), R.ps_norm(M, K, comb(s, -1 / s))
            Lp = M.energy(K, M.full(s * rho_u + rho_z / s)); Lm = M.energy(K, M.full(s * rho_u - rho_z / s))
            lo, hi = 0.25 * (Lp ** 2 - Um ** 2), 0.25 * (Up ** 2 - Lm ** 2)
            ok = (lo - 1e-12 * abs(Jh) <= rem <= hi + 1e-12 * abs(Jh))
            rec = dict(family=fam, pair=pname, method=mname, Jh=Jh, rem=rem, product_radius=nu_ * nz_, polar_lo=lo, polar_hi=hi,
                       polar_halfwidth=0.5 * (hi - lo), contains=bool(ok), eu=eu, ez=ez)
            out.append(rec)
            print(f"{fam:8s} {pname:14s} {mname}: |J_h - centre| = {abs(rem):.2e};  product radius {nu_ * nz_:.2e} "
                  f"({nu_ * nz_ / abs(Jh):.2e} |J_h|);  polarization half-width {0.5 * (hi - lo):.2e} ({0.5 * (hi - lo) / abs(Jh):.2e} |J_h|), "
                  f"ratio {0.5 * (hi - lo) / (nu_ * nz_):.3f}; contains {ok}", flush=True)
json.dump(out, open(os.path.join(HERE, 'polarization_dev.json'), 'w'), indent=1)
