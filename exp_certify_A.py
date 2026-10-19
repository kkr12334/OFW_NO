"""exp_certify_A.py -- Experiment A: does the energy-norm certificate stay sharp
when M is not the per-sample adapted basis?

THE QUESTION
------------
C2 (certified neural operators) rests on eta^2 = r^T M^-1 r having effectivity
near 1.  Effectivity is governed by how spectrally equivalent M is to A, and PCG
iteration count is the square root of that same quantity.  Measured iteration
counts for the four candidate M's span 15 -> ~1700.  If effectivity degrades the
way conditioning does, then a cheap M gives a useless certificate, and the only
sharp M costs ~85 s/sample at level 7 -- more than a direct solve of the same
system, which inverts the whole value proposition.

Nobody has varied M and measured effectivity.  This script does.

WHAT IS MEASURED
----------------
N channelized samples at one level and contrast.  Six error structures, each
built per sample and then scored by all four M's, so every preconditioner is
graded on identical predictions.  For each (M, structure, sample):

    eta         = sqrt(r^T M^-1 r)        the certificate
    true        = sqrt(e^T Kg e)          energy-norm error, e = x* - x_hat
    effectivity = eta / true              1.0 is exact

The four M's differ ONLY in which S_filters dict is installed:

    adapted   per-sample operator-adapted basis at this a
    bank      nearest of K=8 k-means centres in log-a space
    frozen    adapted basis built once at a = 1
    hb        S = {} -- plain hierarchical block preconditioner, which is what
              paper1_energy_adaptive.py and paper_matched_wavelet_adaptivity.py
              actually use (K_alpha_HB_j = K_blocks[j], uncustomised)

ERROR STRUCTURES ARE BASIS-NEUTRAL
----------------------------------
The v1 run defined 'structured truncation' as one application of the ADAPTED
preconditioner and 'partial PCG' as 3 adapted-preconditioned iterations.  Both
are adversarial for the adapted arm specifically: the residual left by an
M-based iteration is nearly orthogonal to that same M's error modes, which is a
known estimator pathology and not a property of the basis.  It cost the adapted
arm an effectivity of 0.44 on that row and inflated its spread from 1.20 to
2.25.  Both are replaced here with operator-based, basis-free constructions:

    damped Jacobi (20 it)   diag(Kg) only -- aligned with no multiresolution basis
    plain CG (30 it)        unpreconditioned -- aligned with no basis

'finest level omitted' is aligned with the level partition, but all four arms
share that partition, so it is fair to all of them.

RUN
---
    PYTHONPATH=/path/to/OFW_NO python exp_certify_A.py [level] [contrast] [n_samples]

Defaults: level 6, contrast 1e4, 10 samples.  Level 6 keeps an adapted build at
~12 s; the bank arm costs 8 builds once, shared across samples.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg, factorized, spsolve

from ofwno_core import (ChannelizedOperatorAssembly, HierarchicalTransform,
                        MeshHierarchy, PoissonAssembler, WaveletBlockExtractor,
                        precompute_fixed_load_blocks)
from ofwno_benchmark_v2 import AdaptiveBasisSolver
from frozen_basis import sample_a_lognormal

ARMS = ['adapted', 'bank', 'frozen', 'hb']
STRUCTS = ['smooth low-freq', 'white noise', 'localized blob',
           'damped Jacobi 20it', 'plain CG 30it', 'finest level omitted']


# --------------------------------------------------------------------------
def make_Minv(S, Kb, levels, off, ndof):
    """M^-1 = W D^-1 W^T for the basis given by S.

    Deliberately NOT FrozenBasisSolver.precond: that reads self.S inside the
    matvec, so the operator silently changes meaning if self.S is reassigned.
    Here S is captured by value, which is what comparing four bases on one
    problem requires.  S = {} degenerates to block-Jacobi-by-level on the raw
    hierarchical blocks.
    """
    fac = {l: (factorized(Kb[l].tocsc()) if Kb[l].shape[0] else None)
           for l in levels}

    def matvec(r):
        d = {l: r[a:b] for l, (a, b) in off.items()}
        for j in levels:
            for i in levels:
                if i < j and (i, j) in S:
                    d[j] = d[j] - S[(i, j)].T @ d[i]
        w = {l: (fac[l](d[l]) if fac[l] is not None else d[l]) for l in levels}
        for j in sorted(levels, reverse=True):
            for i in levels:
                if i < j and (i, j) in S:
                    w[i] = w[i] - S[(i, j)] @ w[j]
        return np.concatenate([w[l] for l in levels])

    return LinearOperator((ndof, ndof), matvec=matvec)


def diag_scaled_cg(A, rhs, steps=5):
    """Verbatim _diag_scaled_cg from paper1_energy_adaptive.py: fixed iteration
    budget, info discarded."""
    if len(rhs) == 0:
        return np.zeros_like(rhs)
    D = A.diagonal().copy()
    D[D == 0.0] = 1.0
    invsqrt = 1.0 / np.sqrt(D)
    Sc = sp.diags(invsqrt)
    xhat, _ = cg((Sc @ A @ Sc).tocsr(), invsqrt * rhs, x0=np.zeros_like(rhs),
                 atol=0.0, rtol=1e-14, maxiter=steps)
    return invsqrt * xhat


def eta_2006(r, Kb, levels, off, steps=5):
    """Amaratunga & Sudarshan (2006) Eq. (27) applied to an external residual:
    per-level diagonally-scaled CG, weighted by diag(K_j), UNCUSTOMISED blocks."""
    tot = 0.0
    for l in levels:
        a, b = off[l]
        if b > a:
            rt = diag_scaled_cg(Kb[l], r[a:b], steps=steps)
            tot += float(np.sum((rt ** 2) * Kb[l].diagonal()))
    return np.sqrt(max(tot, 0.0))


def eta_exact_diag(r, Kb, levels, off):
    """Same diag(K_j) weighting, exact block solve.  Isolates the cost of the
    truncated inner solve."""
    tot = 0.0
    for l in levels:
        a, b = off[l]
        if b > a:
            rt = factorized(Kb[l].tocsc())(r[a:b])
            tot += float(np.sum((rt ** 2) * Kb[l].diagonal()))
    return np.sqrt(max(tot, 0.0))


# --------------------------------------------------------------------------
def build_predictions(x_star, Kg, Fg, T, mesh, perm, is_bnd, bnd, off, level, rng):
    """Six error structures.  All basis-neutral except the last, which is
    neutral across the four arms because they share the level partition."""
    coords = np.asarray(mesh.coords)
    xs, ys = coords[:, 0], coords[:, 1]
    scale = np.abs(x_star).max()

    def from_nodal(p):
        u = np.zeros(len(mesh.coords)); u[perm] = x_star
        g = spsolve(T.tocsc(), (T @ u) + p)
        xh = g[perm].copy(); xh[is_bnd] = 0.0
        return xh

    p_smooth = 0.05 * scale * np.sin(np.pi * xs) * np.sin(np.pi * ys)
    p_noise = 0.05 * scale * rng.standard_normal(len(coords)); p_noise[bnd] = 0.0
    p_blob = 0.30 * scale * np.exp(-((xs - 0.3) ** 2 + (ys - 0.7) ** 2) / 0.01)
    p_blob[bnd] = 0.0

    # damped Jacobi on the permuted HB system: diag(Kg) only, no basis
    d = Kg.diagonal().copy(); d[d == 0.0] = 1.0
    xj = np.zeros_like(Fg)
    for _ in range(20):
        xj = xj + (2.0 / 3.0) * (Fg - Kg @ xj) / d
    xj[is_bnd] = 0.0

    # unpreconditioned CG, 30 iterations
    xc, _ = cg(Kg, Fg, rtol=1e-14, maxiter=30)

    x_drop = x_star.copy()
    a_, b_ = off[level]
    x_drop[a_:b_] = 0.0

    return [from_nodal(p_smooth), from_nodal(p_noise), from_nodal(p_blob),
            xj, xc, x_drop]


def main(level=6, contrast=1e4, n_samples=10, n_bank=8, bank_pool=40, seed=12345):
    t_start = time.time()
    print("Experiment A -- certificate sharpness vs preconditioner quality")
    print(f"level {level}, channelized 4x4, contrast {contrast:.0e}, "
          f"{n_samples} samples, seed {seed}\n")

    mesh = MeshHierarchy(); mesh.refine_to_level(level)
    T = HierarchicalTransform.build(mesh)
    bnd = mesh.get_boundary_nodes()
    asm = ChannelizedOperatorAssembly(mesh, PoissonAssembler, 4, 4, 1.0)
    Fb = precompute_fixed_load_blocks(mesh, T, bnd, lambda x, y: 1.0,
                                      PoissonAssembler, HierarchicalTransform,
                                      WaveletBlockExtractor)
    print(f"mesh: {len(mesh.coords)} nodes, {len(bnd)} boundary")

    S = AdaptiveBasisSolver(mesh, T, bnd, asm, Fb, level, mode='per_sample',
                            n_layers=6, drop_tol=1e-6, verbose=True)
    S_frozen = S.S                      # built at a = 1 by __init__, free

    rng = np.random.default_rng(seed)
    pool = sample_a_lognormal(bank_pool, asm.n_channels, contrast, rng=rng)

    S.mode, S.n_bank = 'bank', n_bank
    t0 = time.time(); S.fit_bank(pool, verbose=False); t_bank = time.time() - t0
    print(f"bank: K={n_bank} over a {bank_pool}-sample pool, {t_bank:.1f}s "
          f"({t_bank / n_bank:.1f}s per centre)\n")

    levels, off, ndof, perm = S.levels, S.off, S.ndof, S.perm
    is_bnd = np.isin(perm, bnd)
    Fg = S.Fg

    eff = {arm: {st: [] for st in STRUCTS} for arm in ARMS}
    eff_2006 = {st: [] for st in STRUCTS}
    eff_hbex = {st: [] for st in STRUCTS}
    its = {arm: [] for arm in ARMS}
    t_build = []

    for n in range(n_samples):
        a = pool[n]
        Kb, Bb, Khb = S._blocks(a)
        Kg = S._global(Khb)

        t0 = time.time(); S_adapted = S._build_S(a); t_build.append(time.time() - t0)
        S_bank = S._bank[int(S._km.predict(np.log(np.atleast_2d(a)))[0])]
        bases = {'adapted': S_adapted, 'bank': S_bank,
                 'frozen': S_frozen, 'hb': {}}
        Minv = {k: make_Minv(v, Kb, levels, off, ndof) for k, v in bases.items()}

        x_star = spsolve(Kg.tocsc(), Fg)
        for arm in ARMS:
            it = [0]
            _, info = cg(Kg, Fg, rtol=1e-10, maxiter=20000, M=Minv[arm],
                         callback=lambda z: it.__setitem__(0, it[0] + 1))
            its[arm].append(it[0] if info == 0 else np.nan)

        preds = build_predictions(x_star, Kg, Fg, T, mesh, perm, is_bnd, bnd,
                                  off, level, rng)
        for st, xh in zip(STRUCTS, preds):
            e = x_star - xh
            true = np.sqrt(max(float(e @ (Kg @ e)), 0.0))
            if true <= 0:
                continue
            r = Fg - Kg @ xh
            for arm in ARMS:
                eff[arm][st].append(
                    np.sqrt(max(float(r @ (Minv[arm] @ r)), 0.0)) / true)
            eff_hbex[st].append(eta_exact_diag(r, Kb, levels, off) / true)
            eff_2006[st].append(eta_2006(r, Kb, levels, off, steps=5) / true)

        print(f"  sample {n + 1}/{n_samples}  ratio {a.max() / a.min():.2e}  "
              f"build {t_build[-1]:.1f}s  PCG "
              + " ".join(f"{arm}={its[arm][-1]:.0f}" for arm in ARMS))

    # ---------------- Table 1 ----------------
    print("\n" + "=" * 86)
    print(f"TABLE 1  effectivity = eta/||e||_a,  mean +/- sd over {n_samples} samples")
    print("=" * 86)
    print(f"{'error structure':<22}" + "".join(f"{a:>16}" for a in ARMS))
    print("-" * 86)
    for st in STRUCTS:
        row = ""
        for arm in ARMS:
            v = np.array(eff[arm][st])
            row += f"{v.mean():>10.3f}+-{v.std():<5.3f}"
        print(f"{st:<22}{row}")
    print("-" * 86)
    print(f"{'PCG iterations':<22}" + "".join(
        f"{np.nanmean(its[a]):>10.0f}+-{np.nanstd(its[a]):<5.0f}" for a in ARMS))
    print("-" * 86)
    for arm in ARMS:
        allv = np.array([v for st in STRUCTS for v in eff[arm][st]])
        print(f"{arm:<9} over all {len(allv)} (sample,structure) pairs: "
              f"min {allv.min():.3f}  max {allv.max():.3f}  "
              f"max/min {allv.max() / allv.min():.3f}  "
              f"understates on {100 * np.mean(allv < 1.0):.0f}%")

    # ---------------- Table 2 ----------------
    print("\n" + "=" * 86)
    print("TABLE 2  formula error vs basis error, same residuals (means)")
    print("  exact-M   r^T M^-1 r, ADAPTED basis      hb-exact  exact block solve, diag weight")
    print("  hb-2006   5-step diag-scaled CG, diag weight  <- the uploaded 2006 code")
    print("=" * 86)
    print(f"{'error structure':<22}{'exact-M':>12}{'hb-exact':>12}{'hb-2006':>12}"
          f"{'2006/exact':>13}")
    print("-" * 86)
    for st in STRUCTS:
        m = np.mean(eff['adapted'][st]); he = np.mean(eff_hbex[st])
        h6 = np.mean(eff_2006[st])
        print(f"{st:<22}{m:>12.4f}{he:>12.4f}{h6:>12.4f}{h6 / he:>13.5f}")

    # ---------------- Table 3 ----------------
    print("\n" + "=" * 86)
    print("TABLE 3  cost of one certificate")
    print("=" * 86)
    r = np.random.default_rng(0).standard_normal(ndof)
    Kb, _, Khb = S._blocks(pool[0])
    Minv0 = {'adapted': make_Minv(S._build_S(pool[0]), Kb, levels, off, ndof),
             'hb': make_Minv({}, Kb, levels, off, ndof)}
    print(f"{'basis':<12}{'build s':>12}{'eta eval ms':>14}{'PCG it':>12}")
    print("-" * 86)
    for arm, bs in (('adapted', np.mean(t_build)), ('bank', t_bank / n_bank),
                    ('frozen', S.build_time), ('hb', 0.0)):
        key = 'adapted' if arm != 'hb' else 'hb'
        t0 = time.time()
        for _ in range(5):
            Minv0[key] @ r
        print(f"{arm:<12}{bs:>12.1f}{(time.time() - t0) / 5 * 1000:>14.1f}"
              f"{np.nanmean(its[arm]):>12.0f}")
    t0 = time.time()
    for _ in range(5):
        eta_2006(r, Kb, levels, off, steps=5)
    print(f"{'hb-2006':<12}{0.0:>12.1f}{(time.time() - t0) / 5 * 1000:>14.1f}"
          f"{'-':>12}")

    print(f"\ntotal wall clock {time.time() - t_start:.1f}s")


if __name__ == '__main__':
    lvl = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    ctr = float(sys.argv[2]) if len(sys.argv) > 2 else 1e4
    ns = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    main(level=lvl, contrast=ctr, n_samples=ns)
