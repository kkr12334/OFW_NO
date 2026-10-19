"""
ofwno_benchmark_v2.py -- capacity-matched, high-contrast OFW-NO benchmark.

Three changes from the existing benchmark, each forced by a measurement:

1. CONTRAST. sample_mu_safe gives a in [0.5,1.5] -- a contrast ratio of 3. The
   Darcy benchmarks FNO/WNO are judged on use 1e2-1e6. Contrast 3 is also the
   regime where operator-adaptation does not matter: measured PCG iterations
   with a FROZEN reference basis were 16 at contrast 3 but 916 at 1e4, while the
   per-sample adapted basis stayed at 12-15 across that whole range. The method
   only has a story at high contrast, so the benchmark must run there.

2. BASIS MODE. 'per_sample' rebuilds the adapted basis for each coefficient
   field: contrast-robust (14 iters at contrast 5e3) but pays the build.
   'frozen' builds once and reuses: 50x faster at contrast 3 (2.26 s/sample vs
   113 s/sample at level 7, measured) but collapses at high contrast.
   Default is per_sample; use frozen only if a contrast sweep shows it holds.

3. CAPACITY. The existing defaults give FNO 4.74M parameters against OFW-NO's
   0.51M and WNO's 0.08M -- a 9.3x and 56x mismatch. match_capacity() sizes the
   baselines to a common budget so the comparison tests architecture, not width.

Reported per run: relative L2 error (as before), AND the PCG iteration count of
the adapted basis, which is the cleanest single number showing whether
operator-adaptation is doing anything on this problem.
"""
from __future__ import annotations
import numpy as np, time
from frozen_basis import FrozenBasisSolver, sample_a_lognormal
from ofwno_core import (MeshHierarchy, PoissonAssembler, HierarchicalTransform,
                        WaveletBlockExtractor, MeshGeometry, WoodburyPatchLocalSolver,
                        ChannelizedOperatorAssembly, precompute_fixed_load_blocks)
from fast_woodbury import FastWoodbury


class AdaptiveBasisSolver(FrozenBasisSolver):
    """mode='per_sample' rebuilds the adapted basis for every a (contrast-robust);
    mode='frozen' inherits the parent's single reference basis."""

    """mode='per_sample' rebuilds the adapted basis for every a (contrast-robust);
    mode='bank' builds K bases at k-means centres in log-a space and uses the
    nearest one; mode='frozen' inherits the parent's single reference basis.

    Measured, level 6, 40 samples, contrast 1e4 (wall clock, build + solve):
        per-sample   376.6s   14.5 PCG iterations (max 17)
        bank K=8     112.8s  352.6 PCG iterations (max 2908)   3.3x faster
        frozen      ~360s   1579.6 PCG iterations (max 3440)
    The bank does NOT recover per-sample contrast robustness -- 8 centres in a
    16-dimensional log-a space with 10^4 dynamic range still leave ratios of
    10^2-10^3 on several channels -- but basis construction dominates the wall
    clock, so trading 24x more iterations for 1/5 the builds still wins, and
    wins harder as the level rises (at level 7 a build is ~85s against ~7ms per
    iteration, so a K=8 bank projects to ~12x).  A budget-triggered per-sample
    fallback is NOT worth it: at level 6 with a 150-iteration budget it fired on
    35 of 40 samples and cut the speedup to 1.3x.
    """

    def __init__(self, *a, mode='per_sample', n_bank=8, bank_seed=0,
                 b_tilde_drop_tol=0.0, **k):
        self.mode, self.n_bank, self.bank_seed = mode, n_bank, bank_seed
        self.b_tilde_drop_tol = b_tilde_drop_tol
        self._bank, self._km = None, None
        super().__init__(*a, **k)
        self._geo = MeshGeometry(self.mesh)
        self._nl = k.get('n_layers', 6)
        self._mlt = k.get('min_level_for_truncation', 3)
        self._dt = k.get('drop_tol', 1e-6)

    def _build_S(self, a):
        Kb, Bb, Khb = self._blocks(a)
        w = FastWoodbury(Kb, Bb, self.F_blocks, self.max_level,
                         geometry=self._geo, n_layers=self._nl,
                         min_level_for_truncation=self._mlt,
                         s_filter_drop_tol=self._dt,
                         b_tilde_drop_tol=self.b_tilde_drop_tol)
        w.customize_basis()
        return w.S_filters

    def fit_bank(self, a_samples, verbose=True):
        """Build the k-means basis bank. Call once before generate() in
        mode='bank'; a_samples must be the full pool the bank will serve."""
        from sklearn.cluster import KMeans
        import time as _t
        A = np.asarray(a_samples)
        K = min(self.n_bank, len(A))
        self._km = KMeans(K, n_init=10, random_state=self.bank_seed).fit(np.log(A))
        t0 = _t.time()
        self._bank = [self._build_S(np.exp(c)) for c in self._km.cluster_centers_]
        if verbose:
            print(f"[bank] {K} adapted bases built in {_t.time()-t0:.1f}s")
        return self

    def solve(self, a):
        if self.mode == 'frozen':
            return super().solve(a)
        from scipy.sparse.linalg import cg
        Kb, Bb, Khb = self._blocks(a)
        if self.mode == 'bank':
            if self._bank is None:
                raise RuntimeError("call fit_bank(a_samples) before solving in mode='bank'")
            S_new = self._bank[int(self._km.predict(np.log(np.atleast_2d(a)))[0])]
        else:
            S_new = self._build_S(a)
        # SEPARATION OF CONCERNS -- the two bases do different jobs:
        #   * the PER-SAMPLE adapted basis is used only as a PRECONDITIONER, to
        #     get the exact solution x in few iterations. It is contrast-robust
        #     (14 iterations at contrast 5e3 vs 2907 for a frozen basis), which
        #     is the whole reason operator-adaptation is worth its cost.
        #   * the FROZEN basis is used to EXPRESS the answer. The network's
        #     target and its reconstruction map must not depend on mu, or
        #     inference would need a per-mu basis build and cost more than
        #     solving exactly.
        # Getting this backwards silently corrupts the targets: an earlier
        # revision computed W_inv with the per-sample basis still installed and
        # reconstructed with the frozen one, giving 2.4e-3 round-trip error
        # instead of 1e-16.
        S_keep, self.S = self.S, S_new
        try:
            Kg = self._global(Khb); it = [0]
            x, info = cg(Kg, self.Fg, rtol=self.rtol, maxiter=self.maxiter,
                         M=self.precond(Kb), callback=lambda z: it.__setitem__(0, it[0] + 1))
            if info != 0:
                raise RuntimeError(f"PCG failed (info={info})")
        finally:
            self.S = S_keep                       # restore the SHARED basis first
        return self.W_inv(x), self.to_nodal_from_hb(x), it[0]


def flatten(c_by_level, i, levels):
    """Per-level dict of (N, n_l) arrays -> one sample's flat coefficient vector,
    in the ordering AdaptiveBasisSolver.to_nodal expects."""
    return np.concatenate([c_by_level[l][i] for l in levels])


def stack(c_by_level, levels):
    """-> (N, ndof) array of all samples' flat coefficient vectors."""
    return np.concatenate([c_by_level[l] for l in levels], axis=1)


def _as_dim_list(level_dims):
    """Accept either a list/array of per-level widths or the {level: width}
    dict that build_pool() returns, and return widths in ascending level order.

    build_pool returns a dict; iterating a dict yields KEYS, so passing it
    straight into a `for out in level_dims` loop silently counts 0,1,2,...
    instead of the widths.  Normalise here so both spellings work.
    """
    if isinstance(level_dims, dict):
        return [level_dims[k] for k in sorted(level_dims)]
    return list(level_dims)


def ofwno_param_count(level_dims, hidden=64, proj=16, mu=16):
    """Trainable-parameter count for LinearSkipOFWNeuralOperator.

    Mirrors the module tree exactly:
      mu_encoder        Linear(mu,hidden) + Linear(hidden,hidden)
      per level l:
        couplings[l]    Linear(cum, proj)         -- ONLY when cum > 0
                        (CrossLevelCoupling sets proj_dim = 0 at level 0,
                         so the block input is `hidden`, not `hidden+proj`)
        blocks[l].net   Linear(in,hidden) + Linear(hidden,hidden)
                                                 + Linear(hidden,out)
        blocks[l].skip  Linear(mu, out)           -- has bias
        linear_heads[l] Linear(mu, out, bias=False)

    NOTE: the last term is the LinearSkip head.  If you train the plain
    OFWNeuralOperator instead, drop `mu*out` (pass linear_skip=False) or the
    capacity budget is ~mu*sum(level_dims) too generous.
    """
    dims = _as_dim_list(level_dims)
    tot = mu*hidden+hidden + hidden*hidden+hidden          # mu_encoder
    cum = 0
    for out in dims:
        p = proj if cum > 0 else 0                         # level 0 has no coupling
        if cum > 0:
            tot += cum*proj + proj                         # couplings[l]
        in_dim = hidden + p
        tot += in_dim*hidden+hidden + hidden*hidden+hidden + hidden*out+out
        tot += mu*out + out                                # blocks[l].skip (bias)
        tot += mu*out                                      # linear_heads[l] (no bias)
        cum += out
    return tot


def fno_params(h, modes, layers, cin=3):
    """RealFNO2D real-valued DOF count.

    The spectral weight is a complex Parameter of shape (h,h,modes,modes);
    it carries 2*h*h*modes*modes REAL degrees of freedom.  torch's
    `sum(p.numel() for p in model.parameters())` reports the COMPLEX count
    (half this), so do not compare that printout against this number --
    use 2*numel for complex parameters when auditing.
    DWT/IDWT filters in the WNO are buffers, not parameters: excluded.
    """
    return cin*h+h + layers*(2*h*h*modes*modes + h*h+h) + h*h+h + h+1


def wno_params(h, layers, J=2, cin=3):
    return cin*h+h + layers*(h*h+h + J*((3*h)*(3*h)+3*h) + h*h+h) + h*h+h + h+1


def mlp_params(mu, out, dims):
    p, d = 0, mu
    for x in dims: p += d*x+x; d = x
    return p + d*out+out

def match_capacity(level_dims, ofw_hidden=64, modes=12, layers=4, verbose=True,
                   mu_dim=16, target=None, fno_hi=4096, wno_hi=4096, mlp_hi=65536):
    """Size FNO/WNO/MLP to OFW-NO's parameter budget.

    mu_dim is the baseline MLP's input width: 16 for the channelized problem,
    n*n for a GRF (there is no finite mu, so the MLP takes the whole field).
    Getting this wrong silently decouples the MLP's real parameter count from
    the printed budget.

    target overrides the budget, for when the reference model is SpatialOFWNO
    rather than the dense-mu OFW-NO.

    fno_hi / wno_hi / mlp_hi cap the bisection search.  A bisection that hits
    its cap returns an UNDERSIZED model and used to do so silently -- the
    caller then printed a "capacity-matched" table in which one baseline was
    at a fraction of budget.  Saturation is now reported in the result
    (cfg['saturated']) and shouted about when verbose.  The caps here are
    deliberately generous; raise them rather than accepting a saturated match.
    """
    dims = _as_dim_list(level_dims)
    if target is None:
        target = ofwno_param_count(dims, hidden=ofw_hidden)
    n_out = sum(dims)
    saturated, over_floor = [], []

    def bisect(name, f, lo, hi):
        if f(hi) < target:                       # cannot reach the target at all
            saturated.append(name)
            return hi
        if f(lo) > target:
            # Mirror problem: the model's minimum size already exceeds budget.
            # For a GRF the MLP's first layer alone is mu_dim x 256, so it has
            # a hard floor well above a small target and CANNOT be matched
            # downward -- the table is then biased in the MLP's favour.
            over_floor.append(name)
            return lo
        while lo < hi:
            m = (lo + hi) // 2
            if f(m) < target: lo = m + 1
            else: hi = m
        return lo

    h_fno = bisect('FNO', lambda h: fno_params(h, modes, layers), 4, fno_hi)
    h_wno = bisect('WNO', lambda h: wno_params(h, layers), 4, wno_hi)
    h_mlp = bisect('MLP', lambda h: mlp_params(mu_dim, n_out, (256, 256, h)), 4, mlp_hi)
    cfg = dict(target=target, fno_hidden=h_fno, fno_modes=modes, fno_layers=layers,
               wno_hidden=h_wno, wno_layers=layers, mlp_hidden=(256, 256, h_mlp),
               mu_dim=mu_dim, saturated=saturated, over_floor=over_floor)
    if verbose:
        print(f"capacity target: {target:,} params   (MLP input width {mu_dim})")
        for nm, p in (("FNO", fno_params(h_fno, modes, layers)),
                      ("WNO", wno_params(h_wno, layers)),
                      ("MLP", mlp_params(mu_dim, n_out, (256, 256, h_mlp)))):
            flag = ("   <-- BISECTION CAP HIT, TARGET NOT REACHED" if nm in saturated
                    else "   <-- MINIMUM SIZE ALREADY OVER BUDGET" if nm in over_floor
                    else "")
            print(f"  {nm}: {p:>9,}  ({p/target:.2f}x target){flag}")
        print(f"  -> fno_hidden={h_fno}, wno_hidden={h_wno}, mlp_hidden=(256,256,{h_mlp})")
        if saturated:
            print(f"  !! {', '.join(saturated)} undersized -- raise "
                  f"fno_hi/wno_hi/mlp_hi; this table is NOT capacity-matched")
        if over_floor:
            print(f"  !! {', '.join(over_floor)} cannot be shrunk to budget "
                  f"(narrow the hidden_dims prefix, or raise the target); "
                  f"this table is NOT capacity-matched")
    return cfg


def build_pool(MAX_LEVEL=6, N=150, N_CX=4, N_CY=4, contrast=1e4, mode='per_sample',
               n_layers=6, drop_tol=1e-6, seed=12345, verbose=True):
    """Exact solutions for N coefficient fields, plus the fixed reconstruction map."""
    mesh = MeshHierarchy(); mesh.refine_to_level(MAX_LEVEL)
    T = HierarchicalTransform.build(mesh); bnd = mesh.get_boundary_nodes()
    asm = ChannelizedOperatorAssembly(mesh, PoissonAssembler, N_CX, N_CY, 1.0)
    Fb = precompute_fixed_load_blocks(mesh, T, bnd, lambda x, y: 1.0,
                                      PoissonAssembler, HierarchicalTransform,
                                      WaveletBlockExtractor)
    S = AdaptiveBasisSolver(mesh, T, bnd, asm, Fb, MAX_LEVEL, mode=mode,
                            n_layers=n_layers, drop_tol=drop_tol, verbose=verbose)
    rng = np.random.default_rng(seed)
    a = sample_a_lognormal(N, N_CX*N_CY, contrast, rng=rng)
    t0 = time.time()
    c_by_level, u_nodal, iters = S.generate(a, verbose=verbose)
    if verbose:
        print(f"pool: {N} exact samples, {time.time()-t0:.1f}s "
              f"({(time.time()-t0)/N:.2f}s/sample), PCG iters "
              f"{np.mean(iters):.0f} +/- {np.std(iters):.0f}, contrast {contrast:.0e}")
    return dict(mesh=mesh, T=T, boundary=bnd, solver=S, a=a, c=c_by_level,
                u=u_nodal, iters=iters, level_dims={l: v.shape[1] for l, v in c_by_level.items()})


def build_pool_grf(MAX_LEVEL=5, N=200, contrast=1e4, corr_length=0.05, nu=0.5,
                   mode='per_sample', n_layers=6, drop_tol=1e-6, seed=12345,
                   verbose=True):
    """Exact solutions for N Gaussian-random-field coefficients.

    Same contract as build_pool (returns mesh/solver/a/c/u/level_dims) so
    train_compare can consume either, but `a` here is the per-ELEMENT field and
    `a_grid` carries the same field on the n x n node grid for FNO/WNO input.
    There is no finite mu: the dense-mu OFWNeuralOperator is not defined on
    this problem, only SpatialOFWNO.
    """
    from grf_field import sample_grf, grf_at_elements, ElementwiseOperatorAssembly
    mesh = MeshHierarchy(); mesh.refine_to_level(MAX_LEVEL)
    T = HierarchicalTransform.build(mesh); bnd = mesh.get_boundary_nodes()
    n = 2 ** MAX_LEVEL + 1
    asm = ElementwiseOperatorAssembly(mesh, PoissonAssembler)
    Fb = precompute_fixed_load_blocks(mesh, T, bnd, lambda x, y: 1.0,
                                      PoissonAssembler, HierarchicalTransform,
                                      WaveletBlockExtractor)
    S = AdaptiveBasisSolver(mesh, T, bnd, asm, Fb, MAX_LEVEL, mode=mode,
                            n_layers=n_layers, drop_tol=drop_tol, verbose=verbose)
    rng = np.random.default_rng(seed)
    a_grid = sample_grf(N, n, corr_length=corr_length, nu=nu, contrast=contrast, rng=rng)
    a_elem = grf_at_elements(a_grid, mesh, n)
    if mode == 'bank':
        S.fit_bank(a_elem, verbose=verbose)
    t0 = time.time()
    c_by_level, u_nodal, iters = S.generate(a_elem, verbose=verbose)
    if verbose:
        print(f"pool(GRF): {N} exact samples, {time.time()-t0:.1f}s "
              f"({(time.time()-t0)/N:.2f}s/sample), PCG iters "
              f"{np.mean(iters):.0f} +/- {np.std(iters):.0f}, contrast {contrast:.0e}, "
              f"corr_length {corr_length} ({corr_length*(n-1):.1f} cells), nu {nu}")
    return dict(mesh=mesh, T=T, boundary=bnd, solver=S, a=a_elem, a_grid=a_grid,
                c=c_by_level, u=u_nodal, iters=iters, grf=True,
                level_dims={l: v.shape[1] for l, v in c_by_level.items()})
