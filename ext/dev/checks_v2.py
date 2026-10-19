"""Repaired protocol checks (amendment 1, 16 Oct 2026; Astra review A1-A3).  Development data only (set D, seed 1, and
small synthetic matrices).  Writes ext/dev/checks_v2.json.  Exit status 1 if any declared check fails.
  K  rank-aware Krylov basis: edge cases (b = 0, A = I probe, eigenvector input, repeated / clustered eigenvalues)
  L  level checks on ACTUAL level inputs captured from the reference CHEB cycle, production (legacy) estimators
  W  common classical wrapper vs the legacy anchor classical_vcycle_proj on all of D (energy errors)
  P  Galerkin output never increases energy relative to the DEFLATED start x0, through the actual wrapper
  S  ordinary-PCG cycle: guaranteed Gershgorin intervals, same polynomial pre/post, no deflation -> symmetric, MA
     spectrum in (0, 1]; balanced deflation C + (I - CA) M (I - AC) on an island operator; legacy-interval cycle reported
  H  oddness / homogeneity / non-additivity of the adaptive smoothers, saved as records (implementation checks)"""
import os, sys, json, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
sys.path[:0] = [EXT, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from scipy.sparse.linalg import spsolve
from s1_common import setup, energy
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle_proj
from stage1_core import (Hier, make_cheb, pcg4, sd4, mr4, galerkin, one_pass, krylov_basis, k4_diagnostic,
                         lam_power, lam_gershgorin, lam_lanczos)
R = {}; FAIL = []
def check(name, ok, **kw):
    R[name] = dict(ok=bool(ok), **kw); print(('PASS ' if ok else 'FAIL ') + name, {k: v for k, v in kw.items() if not isinstance(v, list)})
    if not ok: FAIL.append(name)

# ---------------- K: rank-aware Krylov basis on small matrices
I6 = sp.identity(6, format='csr'); e = np.eye(6)
Q, rk = krylov_basis(I6, e[:, 0]); dist = np.linalg.norm(e[:, 1] - Q @ (Q.T @ e[:, 1]))
check('K.astra_A=I_probe', rk == 1 and abs(dist - 1) < 1e-14, rank=rk, distance=float(dist))
Q, rk = krylov_basis(I6, np.zeros(6)); check('K.b=0', rk == 0, rank=rk)
Ad = sp.diags([1., 2, 3, 4, 5, 6]).tocsr()          # diagonal: D^-1 A = I, so every Krylov space is 1-dimensional
Q, rk = krylov_basis(Ad, np.ones(6)); check('K.diagonal_A_gives_B=I', rk == 1, rank=rk)
def blocks(*s):                                      # unit-diagonal 2x2 blocks [[1,s],[s,1]]: B = A, eigenvalues 1 -+ s
    return sp.block_diag([np.array([[1., t], [t, 1.]]) for t in s]).tocsr()
Ae = blocks(.1, .3, .5); w, V = np.linalg.eigh(Ae.toarray())
Q, rk = krylov_basis(Ae, V[:, 2]); check('K.eigenvector_input', rk == 1, rank=rk)
Q, rk = krylov_basis(Ae, V[:, 0] + V[:, 3] + V[:, 5]); check('K.three_eigencomponents', rk == 3, rank=rk)
Q, rk = krylov_basis(blocks(.5, .5, .5), np.arange(1., 7)); check('K.repeated_eigenvalues(2 distinct)', rk == 2, rank=rk)
Q, rk = krylov_basis(blocks(.5, .5 + 1e-13, .5), np.arange(1., 7)); check('K.clustered(2 groups, splits 1e-13)', rk == 2, rank=rk)
Q, rk = krylov_basis(Ae, np.arange(1., 7)); check('K.six_distinct_capped_at_4', rk == 4, rank=rk)
Ab = blocks(.1, .3, .5); xs = V[:, 0] + V[:, 3] + V[:, 5]; Qx, _ = krylov_basis(Ab, xs)
check('K.A_orthonormal', np.abs(Qx.T @ (Ab @ Qx) - np.eye(Qx.shape[1])).max() < 1e-12)
# zero-case flags of the diagnostic
d0 = k4_diagnostic(Ad, e[:, 2], np.zeros(6)); check('K.diag_zero_output_flag', d0.get('flag') in ('zero output', 'exact convergence in K'), out=str(d0))
d1 = k4_diagnostic(Ad, e[:, 2], e[:, 2] / 3); check('K.diag_exact_convergence_flag', d1['rho'] is None and d1.get('flag') == 'exact convergence in K', out=str(d1))

# ---------------- L, W, P, H on development data
recs = {}
for fam, kw in (('barriers', dict(contrast=1e-4, load='no-low')), ('channels', {})):
    S = setup(3, 2, 1, **kw); te = S['te']; Hh = S['H']; nl = S['nl']
    # W: common wrapper vs legacy anchor over all of D
    dev_rel = []; st = {}
    for i in te:
        A = S['A'][i]; r = S['Fs'][i] - A @ S['UC'][i]; hr = Hier(Hh, A, nl); Z = near_kernel_basis(A)
        lam = [3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]; ch = make_cheb(lam)
        e_new = energy(A, S['UC'][i] + one_pass(hr, r, Z, ch, ch, stats=st), S['U'][i])
        e_old = energy(A, S['UC'][i] + classical_vcycle_proj(hr.levs, hr.A0, r, Z=Z, nu=4), S['U'][i])
        dev_rel.append(abs(e_new / e_old - 1))
    check(f'W.{fam}.wrapper_vs_legacy', max(dev_rel) <= 1e-8, max_rel=float(max(dev_rel)), backstop=st.get('backstop', 0),
          tolerance=1e-8)
    # L: level checks on the actual level inputs of the reference cycle (operator id 300)
    i = te[0]; A = S['A'][i]; r = S['Fs'][i] - A @ S['UC'][i]; hr = Hier(Hh, A, nl); Z = near_kernel_basis(A)
    lam = [3.0] + [lam_power(Ak) for Ak, _ in hr.levs[1:]]; ch = make_cheb(lam)
    captured = []
    def cap_pre(Ak, di, rhs, l, cnt=None): captured.append((l, rhs.copy())); return ch(Ak, di, rhs, l, cnt)
    x0 = galerkin(hr.A, Z, r) if Z.shape[1] else np.zeros_like(r)
    hr.cycle(0, r - hr.A @ x0, cap_pre, ch, parts=True)
    for l, b in captured:
        Ak = hr.levs[l][0]; di = hr.dinv[l]
        dg = k4_diagnostic(Ak, b, ch(Ak, di, b, l), x_pcg=pcg4(Ak, di, b, l))
        Qk, rk = krylov_basis(Ak, b)
        dist = {}
        for nm, f in (('cheb', ch), ('sd', sd4), ('mr', mr4), ('pcg', pcg4)):
            x = f(Ak, di, b, l); px = Qk @ (Qk.T @ (Ak @ x)); dd = x - px
            dist[nm] = float(np.sqrt(max(dd @ (Ak @ dd), 0)) / np.sqrt(x @ (Ak @ x)))
        check(f'L.{fam}.level{l}', rk == 4 and abs(dg['pcg_vs_K']) < 1e-10 and max(dist.values()) < 1e-9,
              rank=rk, pcg_vs_Kmin=dg['pcg_vs_K'], max_dist=max(dist.values()), lam=lam[l], dist=list(dist.items()))
    # P: energy non-increase relative to the deflated start, through the actual wrapper
    u = spsolve(A.tocsc(), r); rng = np.random.default_rng(1); worst = -np.inf; st = {}
    eA = lambda v: np.sqrt(max(v @ (A @ v), 0))
    for t in range(200):
        W = np.column_stack([rng.standard_normal(A.shape[0]) for _ in range(3)] + ([Z] if Z.shape[1] else []))
        bb = r - A @ x0; c = galerkin(A, W, bb, st)
        worst = max(worst, eA(u - x0 - c) / eA(u - x0) - 1)
    check(f'P.{fam}.deflated_start_nonincrease', worst <= 1e-12, worst=float(worst), n_island_vectors=int(Z.shape[1]),
          backstop=st.get('backstop', 0))
    # H: oddness / homogeneity / additivity, structured
    Ak = hr.levs[0][0]; di = hr.dinv[0]; b1, b2 = rng.standard_normal(Ak.shape[0]), rng.standard_normal(Ak.shape[0])
    for nm, f in (('pcg', pcg4), ('sd', sd4), ('mr', mr4), ('cheb', ch)):
        fb = f(Ak, di, b1, 0); nb = np.abs(fb).max()
        recs[f'{fam}.{nm}'] = dict(odd=float(np.abs(f(Ak, di, -b1, 0) + fb).max() / nb),
                                   hom_neg=float(np.abs(f(Ak, di, -2.5 * b1, 0) + 2.5 * fb).max() / nb),
                                   additivity=float(np.linalg.norm(f(Ak, di, b1 + b2, 0) - fb - f(Ak, di, b2, 0)) / np.linalg.norm(f(Ak, di, b1 + b2, 0))))
    # S: ordinary-PCG cycle with guaranteed intervals (plain, no deflation, same polynomial pre/post)
    gb = [lam_gershgorin(Ak) for Ak, _ in hr.levs]; lz = [lam_lanczos(Ak) for Ak, _ in hr.levs]
    chg = make_cheb(gb); n = A.shape[0]
    M = np.column_stack([hr.cycle(0, np.eye(n)[:, j], chg, chg) for j in range(n)])
    sym = np.abs(M - M.T).max() / np.abs(M).max(); ev = np.linalg.eigvals(M @ A.toarray())
    check(f'S.{fam}.pcg_cycle_gershgorin', sym < 1e-10 and np.abs(ev.imag).max() < 1e-8 and ev.real.min() > 0 and ev.real.max() <= 1 + 1e-10,
          symmetry_defect=float(sym), MA_min=float(ev.real.min()), MA_max=float(ev.real.max()), gershgorin=gb, lanczos_x1_1=lz,
          legacy=lam)
    if Z.shape[1]:                                                   # balanced deflation on an island operator
        C = Z @ np.linalg.solve(Z.T @ (A @ Z), Z.T); IA = np.eye(n) - C @ A.toarray()
        Mb = C + IA @ M @ IA.T; symb = np.abs(Mb - Mb.T).max() / np.abs(Mb).max(); evb = np.linalg.eigvalsh(0.5 * (Mb + Mb.T))
        check(f'S.{fam}.balanced_deflation', symb < 1e-10 and evb.min() > 0, symmetry_defect=float(symb), min_eig_M=float(evb.min()))
    Ml = np.column_stack([hr.cycle(0, np.eye(n)[:, j], ch, ch) for j in range(n)]); evl = np.linalg.eigvals(Ml @ A.toarray())
    R[f'S.{fam}.legacy_interval_cycle_info'] = dict(MA_min=float(evl.real.min()), MA_max=float(evl.real.max()),
                                                    symmetry_defect=float(np.abs(Ml - Ml.T).max() / np.abs(Ml).max()))
    print('info legacy-interval plain cycle', R[f'S.{fam}.legacy_interval_cycle_info'])
R['H.records'] = recs; print('H', json.dumps(recs, indent=0))
R['failed'] = FAIL
json.dump(R, open(os.path.join(HERE, 'checks_v2.json'), 'w'), indent=1)
sys.exit(1 if FAIL else 0)
