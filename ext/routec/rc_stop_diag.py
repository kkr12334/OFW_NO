"""Diagnosis for ASTRA's pilot review §2 (18 Oct 2026): stopping intervals at barriers geometry 1, contrast 1e6, iteration
320 miss the saved J_h by 6.6e-10 (radius ~1e-12).  Targeted reproduction of that case only (no new geometry):
rebuilds exactly what rc_pilot.py builds (same seed, mask, K, f, goal 0, refined reference solves, Jacobi-PCG iterates)
and separates the candidate causes:
  (a) the reference J_h = G.u_h (double dot product) against an independent reference: a second LU factorisation with a
      different column ordering, both refined, and the corrected value J = G.u + r(u).z accumulated in extended precision;
  (b) the centre c = G.v + (F - A v).w evaluated in double (as the runner) against extended-precision accumulation;
  (c) the computed residual F - A v in double against extended precision (the certificate is built from the former);
  (d) the true algebraic errors of the iterates and the radii of M6 and M3 built from either residual.
Extended precision = numpy longdouble; its actual precision is printed and saved.  Writes stop_diag.json."""
import os, sys, json, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE): sys.path.insert(0, p)
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis
import rc_fem as R, rc_fast as FST

IDX, C, CHECKS = int(sys.argv[1]) if len(sys.argv) > 1 else 1, float(sys.argv[2]) if len(sys.argv) > 2 else 1e6, [5, 10, 20, 40, 80, 160, 320]
FAM = sys.argv[3] if len(sys.argv) > 3 else 'barriers'
LD = np.longdouble; fi = np.finfo(LD)
prec = dict(dtype=str(np.dtype(LD)), mantissa_bits=int(fi.nmant) + 1, eps=float(fi.eps), platform=sys.platform)
print('extended precision:', prec)
cm = perforated_square(n=9); H = NestedHierarchy(3, coarse=cm); M = R.Mesh(H, cm)
rng = np.random.default_rng({'channels': 2026101801, 'barriers': 2026101802}[FAM])
draws = [make_field(H, 'channels', 1e4 if FAM == 'channels' else 1e-4, rng) for _ in range(IDX + 1)]
mask = draws[IDX] != 1.0
K = np.where(mask, C if FAM == 'channels' else 1.0 / C, 1.0); fK = np.ones(len(K)) if FAM == 'channels' else (~mask).astype(float)
A = M.stiffness(K); lu = splu(A); F = M.load(fK)
c3 = M.fe.centroid; om = np.hypot(c3[:, 0] - 0.5, c3[:, 1] - 0.45) < 0.08; Gv = M.load(om / M.fe.area[om].sum())
Acsr = A.tocsr(); Ald = Acsr.data.astype(LD); absA = abs(Acsr)
def Av_ld(u):
    p = Ald * u.astype(LD)[Acsr.indices]; Au = np.add.reduceat(p, Acsr.indptr[:-1]); Au[np.diff(Acsr.indptr) == 0] = 0; return Au
def res_ld(b, u): return b.astype(LD) - Av_ld(u)
def refine(solver, b, steps=2):
    u = solver.solve(b)
    for _ in range(steps): u = u + solver.solve(res_ld(b, u).astype(np.float64))
    return u
en = lambda x: float(M.energy(K, M.full(x)))
# references: the runner's (default ordering) and an independent factorisation (different column ordering)
u1, z1 = refine(lu, F), refine(lu, Gv)
lu2 = splu(A.tocsc(), permc_spec='MMD_AT_PLUS_A'); u2, z2 = refine(lu2, F), refine(lu2, Gv)
J_runner = float(Gv @ u1)                                          # exactly as the runner (double dot product)
J_corr1 = float((Gv.astype(LD) @ u1.astype(LD)) + res_ld(F, u1) @ z1.astype(LD))
J_corr2 = float((Gv.astype(LD) @ u2.astype(LD)) + res_ld(F, u2) @ z2.astype(LD))
print(f'J_h runner {J_runner!r}; corrected (ext) ordering 1 {J_corr1!r}; ordering 2 {J_corr2!r}')
print(f'  |J_runner - J_corr1| = {abs(J_runner - J_corr1):.3e}; |J_corr1 - J_corr2| = {abs(J_corr1 - J_corr2):.3e}; '
      f'|||u1-u2|||/|||u||| = {en(u1 - u2) / en(u1):.2e}, |||z1-z2|||/|||z||| = {en(z1 - z2) / en(z1):.2e}; |||u||| {en(u1):.3e}, |||z||| {en(z1):.3e}')
def jacobi_pcg(b, its, record):
    d = A.diagonal(); x = np.zeros(len(b)); r = b.copy(); z = r / d; p = z.copy(); rz = r @ z; out = {}
    for k in range(1, its + 1):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap; z = r / d; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        if k in record: out[k] = x.copy()
    return out
xs, zs = jacobi_pcg(F, 320, set(CHECKS)), jacobi_pcg(Gv, 320, set(CHECKS))
ops = dict(M6=FST.JointCache(M, K))
if '--m3' in sys.argv:
    levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
    Phi = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
    ops['M3'] = FST.LiftFreeCache(M, K, A, *R.pu_shepard(M, Phi, Z, 1e-3)[:3])
rows = []
for k in CHECKS:
    v, w = xs[k], zs[k]
    rd_u, rd_z = F - A @ v, Gv - A @ w                                    # runner residuals (double)
    re_u, re_z = res_ld(F, v), res_ld(Gv, w)                              # extended
    c_run = float(Gv @ v + rd_u @ w)                                      # runner centre
    c_ext = float(Gv.astype(LD) @ v.astype(LD) + re_u @ w.astype(LD))
    # a priori evaluation bound for the runner's double centre (first order, Higham gamma_n with unit roundoff 2^-53)
    ud = 2.0 ** -53; n = len(v)
    bound_c_double = float(n * ud * (np.abs(Gv) @ np.abs(v)) + (absA @ np.abs(v) + np.abs(F)) @ np.abs(w) * 8 * ud + n * ud * np.abs(rd_u) @ np.abs(w))
    eu, ez = en(u1 - v), en(z1 - w)
    row = dict(it=k, centre_runner=c_run, centre_ext=c_ext, J_corr=J_corr1, J_runner=J_runner,
               rem_ext=J_corr1 - c_ext, rem_runner_vs_saved=J_runner - c_run, centre_diff=c_run - c_ext, bound_centre_double=bound_c_double,
               res_diff_u=float(np.abs(rd_u - re_u.astype(np.float64)).max()), res_ext_u=float(np.abs(re_u).max()),
               res_diff_z=float(np.abs(rd_z - re_z.astype(np.float64)).max()), res_ext_z=float(np.abs(re_z).max()), eu=eu, ez=ez, true_product=eu * ez)
    for nm, op in ops.items():
        Sd = op.batch([rd_u, rd_z]); Se = op.batch([re_u.astype(np.float64), re_z.astype(np.float64)])
        row[f'{nm}_radius_double_res'] = R.ps_norm(M, K, Sd[0]) * R.ps_norm(M, K, Sd[1])
        row[f'{nm}_radius_ext_res'] = R.ps_norm(M, K, Se[0]) * R.ps_norm(M, K, Se[1])
        row[f'{nm}_eta_u_ext'] = R.ps_norm(M, K, Se[0]); row[f'{nm}_eta_z_ext'] = R.ps_norm(M, K, Se[1])
    rows.append(row)
    print(f"it {k:3d}: |||e_u||| {eu:.2e} |||e_z||| {ez:.2e} product {eu * ez:.2e} | centre runner-ext {c_run - c_ext:+.2e} (a priori double bound {bound_c_double:.1e}) | "
          f"J_corr - centre_ext {J_corr1 - c_ext:+.2e} | max|res diff| u {row['res_diff_u']:.1e} (|res| {row['res_ext_u']:.1e}) | "
          + ' '.join(f"{nm} radius dbl-res {row[f'{nm}_radius_double_res']:.2e} ext-res {row[f'{nm}_radius_ext_res']:.2e}" for nm in ops), flush=True)
json.dump(dict(case=dict(family=FAM, id=IDX, contrast=C), precision=prec, J_runner=J_runner, J_corr_ordering1=J_corr1, J_corr_ordering2=J_corr2,
               energy_u=en(u1), energy_z=en(z1), rows=rows), open(os.path.join(HERE, f'stop_diag_{FAM}_{IDX}_{C:g}.json'), 'w'), indent=1)
