"""Coefficient fields a(x) (element-wise constant) and right-hand sides.

channels   contrast > 1: four meandering high-conductivity bands (a = contrast) on a background a = 1
barriers   contrast < 1: the same geometry with a = contrast, i.e. insulating bands; four domain-crossing barriers can
           enclose a floating conductive island (the case the near-kernel deflation is for)
The random stream is consumed in a fixed order, so (data_seed, sample index) determines the sample.
"""
import numpy as np


def field_channels(centroids, contrast, rng, n_channels=4, width=1 / 16, amp=0.08, background=1.0):
    xc, yc = centroids[:, 0], centroids[:, 1]
    a = np.full(len(centroids), background)
    for _ in range(n_channels):
        horiz = rng.random() < 0.5
        s, t = (xc, yc) if horiz else (yc, xc)
        t0 = rng.uniform(0.15, 0.85); f = rng.uniform(0.5, 2.0); ph = rng.uniform(0, 2 * np.pi)
        centre = t0 + amp * np.sin(2 * np.pi * f * s + ph)
        a[np.abs(t - centre) < width / 2] = background * contrast
    return a


def make_field(H, kind, contrast, rng):
    if kind == 'channels':
        return field_channels(H.centroids, contrast, rng)
    if kind == 'const':
        return np.ones(H.n_elem)
    raise ValueError(f"unknown field '{kind}'")


def load_vector(H, a, kind='all', i=0, data_seed=0):
    """Load on the unknowns of level L for sample i.
    all      f = 1 everywhere.
    no-low   f = 1 only where a >= 0.5, i.e. no source inside insulating barriers (contrast < 1).
    random   element-wise f = c + sum_{p,q<4} K_pq sin((p+1) pi x) sin((q+1) pi y) / (1 + p + q), K_pq ~ N(0,1), c ~ U(-1,1),
             plus with probability 1/2 a Gaussian source of random width 0.02-0.08 and amplitude 2-10 at a random place;
             one load per sample, drawn from its own seed."""
    if kind == 'all':
        return H.F
    if kind == 'no-low':
        return H.load((np.asarray(a).ravel() >= 0.5).astype(float))
    if kind == 'random':
        g = np.random.default_rng(777_000 + 100_000 * data_seed + i)
        cx, cy = H.centroids[:, 0], H.centroids[:, 1]; K = g.normal(size=(4, 4))
        fe = g.uniform(-1, 1) + sum(K[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) / (1 + p + q)
                                    for p in range(4) for q in range(4))
        if g.uniform() < 0.5:
            x0 = g.uniform(0.1, 0.9, 2); w = g.uniform(0.02, 0.08)
            fe = fe + g.uniform(2, 10) * np.exp(-((cx - x0[0]) ** 2 + (cy - x0[1]) ** 2) / (2 * w ** 2))
        return H.load(fe)
    raise ValueError(f"unknown load '{kind}'")
