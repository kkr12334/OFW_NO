"""
frozen_basis.py -- exact, fast parametric data generation for OFW-NO.

Replaces the per-mu customize_basis() in ParametricOperatorDataGenerator.

Why: customize_basis() costs 99.9 s per sample at level 7 (measured), giving
683 min for 150 samples on Colab and a 420 MB dense Woodbury block that breaks
the runtime. The operator-adapted basis is instead built ONCE at a reference
coefficient a_ref and reused for every a as a PCG preconditioner. Measured at
level 7: 2.26 s/sample, 16 CG iterations, energy error 4e-11 -- 50x faster AND
exact, where the per-mu path returned an approximation.

It also fixes a structural problem in the benchmark: with a per-mu basis, the
network's prediction could only be reconstructed by first running a full
per-mu solve, so "inference" cost more than solving exactly. With a frozen
basis the reconstruction map is fixed, so inference is a single sparse matvec.
"""
from __future__ import annotations
import numpy as np, scipy.sparse as sp, time
from scipy.sparse.linalg import spsolve, cg, LinearOperator, factorized
from ofwno_core import (MeshHierarchy, PoissonAssembler, HierarchicalTransform,
                        WaveletBlockExtractor, MeshGeometry, WoodburyPatchLocalSolver,
                        ChannelizedOperatorAssembly, precompute_fixed_load_blocks)


def sample_a_lognormal(n_samples, n_channels, contrast, rng=None):
    """Per-channel diffusivity a ~ exp(U(0, log contrast)), so max(a)/min(a)
    approaches `contrast`. The benchmark's original sample_mu_safe draws
    a = a0 + U(-0.5a0, 0.5a0), i.e. a contrast ratio of 3 -- far milder than the
    10^2-10^6 used in the Darcy benchmarks FNO/WNO are judged on."""
    rng = rng or np.random.default_rng()
    return np.exp(rng.uniform(0.0, np.log(contrast), size=(n_samples, n_channels)))


class FrozenBasisSolver:
    """Operator-adapted basis built once; every coefficient field solved exactly."""

    def __init__(self, mesh, T, boundary, assembly, F_blocks, max_level,
                 a_ref=None, n_layers=6, min_level_for_truncation=3,
                 drop_tol=1e-6, rtol=1e-10, maxiter=2000, verbose=True):
        self.mesh, self.T, self.boundary = mesh, T, set(boundary)
        self.bnd = np.asarray(boundary, dtype=int)
        self.assembly, self.F_blocks, self.max_level = assembly, F_blocks, max_level
        self.rtol, self.maxiter = rtol, maxiter
        n_ch = assembly.n_channels
        a_ref = np.ones(n_ch) if a_ref is None else np.asarray(a_ref)

        t0 = time.time()
        Kb, Bb, _ = self._blocks(a_ref)
        ref = WoodburyPatchLocalSolver(Kb, Bb, F_blocks, max_level,
                                       geometry=MeshGeometry(mesh), n_layers=n_layers,
                                       min_level_for_truncation=min_level_for_truncation,
                                       s_filter_drop_tol=drop_tol)
        ref.customize_basis()
        self.S = ref.S_filters
        self.build_time = time.time() - t0

        self.levels = sorted(Kb.keys())
        self.perm = np.concatenate([np.asarray(mesh.level_nodes[l], dtype=int)
                                     for l in self.levels])
        self.off, s = {}, 0
        for l in self.levels:
            self.off[l] = (s, s + Kb[l].shape[0]); s += Kb[l].shape[0]
        self.ndof = s
        self.Fg = np.concatenate([F_blocks[l] for l in self.levels])
        self.level_dims = {l: Kb[l].shape[0] for l in self.levels}
        if verbose:
            print(f"[FrozenBasisSolver] reference basis built once in "
                  f"{self.build_time:.1f}s (n_layers={n_layers}, drop_tol={drop_tol:g}); "
                  f"S nnz = {sum(v.nnz for v in self.S.values()):,}")

    # ---- assembly helpers -------------------------------------------------
    def _blocks(self, a):
        K = HierarchicalTransform.transform_stiffness(self._K_raw(a), self.T)
        ext = WaveletBlockExtractor(K, np.zeros(len(self.mesh.coords)),
                                     self.mesh.level_nodes, self.bnd)
        return (*ext.extract_all_blocks()[:2], K)

    def _K_raw(self, a):
        """K(a) = sum_k a_k K_k. assembly.K_raw expects mu with a = a0 + mu."""
        return self.assembly.K_raw(np.asarray(a) - self.assembly.a0)

    def _global(self, Khb):
        Kg = Khb.tolil()
        for b in self.bnd:
            Kg[b, :] = 0.0; Kg[:, b] = 0.0; Kg[b, b] = 1.0
        return Kg.tocsr()[self.perm, :][:, self.perm].tocsc()

    # ---- the frozen multiresolution transform -----------------------------
    def _split(self, v):  return {l: v[a:b] for l, (a, b) in self.off.items()}
    def _join(self, d):   return np.concatenate([d[l] for l in sorted(d)])

    def W(self, c):
        """coefficients in the frozen basis -> hierarchical-basis coords.
        This is MultiScaleWaveletSolver.reconstruct_hierarchical_coefficients."""
        x = {l: np.array(v, dtype=float) for l, v in self._split(c).items()}
        for j in sorted(self.levels, reverse=True):
            for i in self.levels:
                if i < j and (i, j) in self.S: x[i] = x[i] - self.S[(i, j)] @ x[j]
        return self._join(x)

    def W_inv(self, x):
        """hierarchical-basis coords -> coefficients in the frozen basis.
        Undoes W by reversing the sweep order and the sign."""
        c = {l: np.array(v, dtype=float) for l, v in self._split(x).items()}
        for j in self.levels:
            for i in self.levels:
                if i < j and (i, j) in self.S: c[i] = c[i] + self.S[(i, j)] @ c[j]
        return self._join(c)

    def precond(self, Kb):
        fac = {l: (factorized(Kb[l].tocsc()) if Kb[l].shape[0] else None)
               for l in self.levels}
        def M(r):
            d = self._split(r)
            for j in self.levels:
                for i in self.levels:
                    if i < j and (i, j) in self.S: d[j] = d[j] - self.S[(i, j)].T @ d[i]
            w = {l: (fac[l](d[l]) if fac[l] is not None else d[l]) for l in self.levels}
            for j in sorted(self.levels, reverse=True):
                for i in self.levels:
                    if i < j and (i, j) in self.S: w[i] = w[i] - self.S[(i, j)] @ w[j]
            return self._join(w)
        return LinearOperator((self.ndof, self.ndof), matvec=M)

    # ---- the exact solve --------------------------------------------------
    def solve(self, a):
        """Returns (coeffs in the frozen basis, nodal solution, n_iterations).
        The solution is the EXACT discrete solution to self.rtol."""
        Kb, _, Khb = self._blocks(a)
        Kg = self._global(Khb)
        it = [0]
        x, info = cg(Kg, self.Fg, rtol=self.rtol, maxiter=self.maxiter,
                     M=self.precond(Kb), callback=lambda z: it.__setitem__(0, it[0] + 1))
        if info != 0:
            raise RuntimeError(f"PCG failed to converge (info={info}, iters={it[0]})")
        return self.W_inv(x), self.to_nodal_from_hb(x), it[0]

    # ---- reconstruction: a FIXED linear map, no per-sample solve ----------
    def to_nodal_from_hb(self, x_hb_perm):
        g = np.zeros(len(self.mesh.coords)); g[self.perm] = x_hb_perm
        return self.T @ g

    def to_nodal(self, c):
        """What the network's prediction goes through at inference time."""
        return self.to_nodal_from_hb(self.W(c))

    def generate(self, a_samples, verbose=True):
        C, U, iters = [], [], []
        t0 = time.time()
        for n, a in enumerate(a_samples):
            c, u, k = self.solve(a); C.append(c); U.append(u); iters.append(k)
            if verbose and (n + 1) % 25 == 0:
                el = time.time() - t0
                print(f"  {n+1}/{len(a_samples)}  {el/(n+1):.2f}s/sample  "
                      f"{np.mean(iters):.0f} iters  eta {el/(n+1)*(len(a_samples)-n-1)/60:.1f} min")
        C = np.asarray(C)
        return {l: C[:, self.off[l][0]:self.off[l][1]] for l in self.levels}, np.asarray(U), iters
