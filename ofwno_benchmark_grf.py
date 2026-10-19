"""GRF-coefficient benchmark pool -- the piece that coefficient-regime-and-grf-
result.md describes as `build_pool_grf` but that was never actually saved into
this folder (ofwno_benchmark_v2.py has no GRF path at all: verified by grep).

Mirrors build_pool() in ofwno_benchmark_v2.py exactly, swapping
ChannelizedOperatorAssembly + sample_a_lognormal for ElementwiseOperatorAssembly
+ sample_grf. AdaptiveBasisSolver / FrozenBasisSolver need no change -- they
only assume `assembly` exposes .a0, .n_channels, .K_raw(), .a_min_max(), which
ElementwiseOperatorAssembly already provides (grf_field.py's own docstring
makes this claim; this file is what actually exercises it end to end).
"""
from __future__ import annotations
import numpy as np, time
from frozen_basis import FrozenBasisSolver
from ofwno_core import (MeshHierarchy, PoissonAssembler, HierarchicalTransform,
                        WaveletBlockExtractor, MeshGeometry, WoodburyPatchLocalSolver,
                        precompute_fixed_load_blocks)
from ofwno_benchmark_v2 import AdaptiveBasisSolver
from grf_field import sample_grf, grf_at_elements, ElementwiseOperatorAssembly


def build_pool_grf(MAX_LEVEL=4, N=60, corr_length=0.05, nu=0.5, contrast=1e4,
                    mode='per_sample', n_layers=6, drop_tol=1e-6, seed=12345,
                    verbose=True):
    """Exact solutions for N GRF coefficient fields, plus the fixed
    reconstruction map. Same shape of return as build_pool(), plus 'grid_a'
    holding the raw (N, n, n) field the grid models (FNO/WNO/OFW-NO-sp) and
    the MLP baseline both see directly -- there is no finite mu for a GRF, so
    unlike build_pool() there is no `a` (channel amplitudes) to report; `a`
    here IS the (N, n_elements) per-element diffusivity used by the solver.
    """
    mesh = MeshHierarchy(); mesh.refine_to_level(MAX_LEVEL)
    T = HierarchicalTransform.build(mesh); bnd = mesh.get_boundary_nodes()
    asm = ElementwiseOperatorAssembly(mesh, PoissonAssembler)
    Fb = precompute_fixed_load_blocks(mesh, T, bnd, lambda x, y: 1.0,
                                      PoissonAssembler, HierarchicalTransform,
                                      WaveletBlockExtractor)
    S = AdaptiveBasisSolver(mesh, T, bnd, asm, Fb, MAX_LEVEL, mode=mode,
                            n_layers=n_layers, drop_tol=drop_tol, verbose=verbose)

    n_grid = 2 ** MAX_LEVEL + 1
    rng = np.random.default_rng(seed)
    G = sample_grf(N, n_grid, corr_length=corr_length, nu=nu, contrast=contrast, rng=rng)
    a_elem = grf_at_elements(G, mesh, n_grid)      # (N, n_elements), what the solver consumes

    t0 = time.time()
    c_by_level, u_nodal, iters = S.generate(a_elem, verbose=verbose)
    if verbose:
        print(f"pool (GRF): {N} exact samples, {time.time()-t0:.1f}s "
              f"({(time.time()-t0)/N:.2f}s/sample), PCG iters "
              f"{np.mean(iters):.0f} +/- {np.std(iters):.0f}, contrast {contrast:.0e}, "
              f"corr_length {corr_length}, nu {nu}")
    return dict(mesh=mesh, T=T, boundary=bnd, solver=S, a=a_elem, grid_a=G, n_grid=n_grid,
                c=c_by_level, u=u_nodal, iters=iters,
                level_dims={l: v.shape[1] for l, v in c_by_level.items()})
