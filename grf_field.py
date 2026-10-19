"""Gaussian-random-field coefficients, and an operator assembly that can take
them.

Why this replaces the channelized field
---------------------------------------
a(x) piecewise constant on a 4x4 channel grid is 16-dimensional and spatially
smooth.  That is exactly the regime where FNO's low-mode truncation costs
nothing, and where a dense mu-to-coefficient map is even definable.  A GRF has
no finite mu, so SpatialOFWNO becomes the architecture rather than a variant,
and the coefficient genuinely carries energy at every scale -- the regime where
spectral bias in operator learning is documented.
"""
import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
def sample_grf(n_samples, n, corr_length=0.05, nu=0.5, contrast=1e4, rng=None):
    """Matern-spectrum Gaussian field on an n x n periodic grid over [0,1]^2,
    exponentiated to a diffusivity.

    corr_length : correlation length in domain units.  Short (<= 4-5 grid
                  spacings) is what makes the field multiscale rather than a
                  smooth blob.
    nu          : Matern smoothness.  0.5 = exponential covariance, continuous
                  but nowhere differentiable -- rough, and the usual choice for
                  subsurface benchmarks.  Larger nu is smoother.
    contrast    : each realisation is rescaled so max(a)/min(a) is EXACTLY
                  this, matching sample_a_lognormal's convention (a in
                  [1, contrast]) so the two benchmarks are comparable.

    Returns (n_samples, n, n) of a-values at grid nodes.
    """
    rng = rng or np.random.default_rng()
    k = np.fft.fftfreq(n, d=1.0 / n)                     # cycles per domain
    KX, KY = np.meshgrid(k, k, indexing='ij')
    k2 = (2 * np.pi) ** 2 * (KX ** 2 + KY ** 2)
    # Matern spectral density in 2d, up to a constant
    S = (1.0 + corr_length ** 2 * k2) ** (-(nu + 1.0))
    amp = np.sqrt(S)
    amp[0, 0] = 0.0                                      # zero mean

    out = np.empty((n_samples, n, n))
    for s in range(n_samples):
        w = rng.standard_normal((n, n))
        u = np.real(np.fft.ifft2(np.fft.fft2(w) * amp))
        u = (u - u.min()) / (u.max() - u.min() + 1e-300)  # -> [0, 1]
        out[s] = np.exp(np.log(contrast) * u)             # -> [1, contrast]
    return out


def grf_at_elements(a_grid, mesh, n):
    """Sample a field given on the n x n dyadic grid at element centroids.

    The mesh nodes ARE the grid points, so the centroid of a triangle is the
    mean of three grid samples -- no interpolation onto a foreign mesh, just
    the natural P1 element average.
    """
    coords = np.asarray(mesh.coords)
    elems = np.asarray([np.asarray(e, dtype=int) for e in mesh.elements])
    xc = coords[elems, 0].mean(1)
    yc = coords[elems, 1].mean(1)
    i = np.clip(np.rint(xc * (n - 1)).astype(int), 0, n - 1)
    j = np.clip(np.rint(yc * (n - 1)).astype(int), 0, n - 1)
    return a_grid[..., i, j]


# ---------------------------------------------------------------------------
class ElementwiseOperatorAssembly:
    """K(a) = sum_e a_e K_e for an arbitrary per-element diffusivity.

    Same interface as ChannelizedOperatorAssembly (a0, n_channels, K_raw,
    a_min_max) so FrozenBasisSolver / AdaptiveBasisSolver need no change, but
    with n_channels = n_elements and a0 = 0, i.e. K_raw(a) = sum_e a_e K_e.

    The per-element 3x3 stiffness entries are precomputed once; a sample is
    then one vectorised COO build, not an element loop.
    """

    def __init__(self, mesh, PoissonAssembler_cls):
        self.mesh = mesh
        self.PoissonAssembler_cls = PoissonAssembler_cls
        self.a0 = 0.0
        self.n_elements = len(mesh.elements)
        self.n_channels = self.n_elements
        self._build()

    def _build(self):
        coords = np.asarray(self.mesh.coords)
        self.n = len(coords)
        I, J, V, E = [], [], [], []
        for e, elem in enumerate(self.mesh.elements):
            ids = np.asarray(elem, dtype=int)
            Ke, _ = self.PoissonAssembler_cls._element_matrix(coords[ids], lambda x, y: 0.0)
            for a_local, ia in enumerate(ids):
                for b_local, ib in enumerate(ids):
                    I.append(ia); J.append(ib)
                    V.append(Ke[a_local, b_local]); E.append(e)
        self.I = np.asarray(I); self.J = np.asarray(J)
        self.V = np.asarray(V, dtype=float); self.E = np.asarray(E)

    def a_min_max(self, a):
        a = np.asarray(a).ravel()
        return float(a.min()), float(a.max())

    def K_raw(self, a, min_diffusivity=1e-6):
        a = np.asarray(a).ravel()
        if a.size != self.n_elements:
            raise ValueError(f"a has {a.size} entries, expected {self.n_elements}")
        if a.min() <= min_diffusivity:
            raise ValueError(f"Non-elliptic sample: min a = {a.min():.4g}")
        return sp.coo_matrix((self.V * a[self.E], (self.I, self.J)),
                             shape=(self.n, self.n)).tocsr()

    def validate_fast_assembly(self, a, tol=1e-13):
        """Against a plain element loop. Relative, because entries scale with a."""
        coords = np.asarray(self.mesh.coords)
        K = sp.lil_matrix((self.n, self.n))
        for e, elem in enumerate(self.mesh.elements):
            ids = np.asarray(elem, dtype=int)
            Ke, _ = self.PoissonAssembler_cls._element_matrix(coords[ids], lambda x, y: 0.0)
            for p, ia in enumerate(ids):
                for q, ib in enumerate(ids):
                    K[ia, ib] += a[e] * Ke[p, q]
        Kf = self.K_raw(a)
        d = abs(Kf - K.tocsr()).max() / max(abs(Kf).max(), 1e-300)
        assert d < tol, f"assembly mismatch (relative) {d:.3e}"
        return float(d)

    def __repr__(self):
        return f"ElementwiseOperatorAssembly(n_elements={self.n_elements}, n_nodes={self.n})"
