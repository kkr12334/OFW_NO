"""Data generation: coefficient samples, reference solutions, the external Galerkin coarse solve u_c and the residual the
network corrects.  Everything is reproducible from DataConfig (rng 1000 + data_seed for the coefficients)."""
import hashlib
import json
import os
from dataclasses import asdict

import numpy as np
from scipy.sparse.linalg import spsolve

from .config import DataConfig
from .fields import make_field, load_vector
from .mesh import Hierarchy, coarse_mesh


def make_hierarchy(cfg: DataConfig):
    return Hierarchy(coarse_mesh(cfg.coarse, cfg.coarse_n), cfg.level, h0=1.0 if cfg.coarse == 'square' else None)


class Dataset:
    """n samples of  -div(a grad u) = f.

    a, F, U      coefficient (n, n_elem), load (n, N), exact discrete solution (n, N)
    UC           Galerkin solution on the level-k0 mesh, prolonged (zeros when k0 = 0)
    R            float64 residual F - A UC;  e = U - UC = A^-1 R is what the network predicts
    scale        rms of the training targets; the network works in units of `scale`
    tr, te       training / test indices (the last test_frac of the samples are the test set)
    """

    def __init__(self, cfg: DataConfig, cache_dir=None, verbose=True):
        self.cfg = cfg
        self.H = H = make_hierarchy(cfg)
        spec = hashlib.md5(json.dumps(asdict(cfg), sort_keys=True).encode()).hexdigest()[:10]
        path = os.path.join(cache_dir, f'data_{spec}.npz') if cache_dir else None
        if path and os.path.exists(path):
            z = np.load(path); self.a, self.U, self.UC = z['a'], z['U'], z['UC']
            if verbose: print(f'data: loaded {path}')
        else:
            rng = np.random.default_rng(1000 + cfg.data_seed)
            self.a = np.stack([make_field(H, cfg.field, cfg.contrast, rng) for _ in range(cfg.n)])
            F = np.stack([load_vector(H, a, cfg.load, i, cfg.data_seed) for i, a in enumerate(self.a)])
            self.U = np.stack([spsolve(H.A_fine(a).tocsc(), f) for a, f in zip(self.a, F)])
            self.UC = np.zeros_like(self.U)
            if cfg.k0 > 0:                      # external coarse solve: Galerkin P^T A P on the level-k0 mesh
                P = H.prolongation(cfg.k0, H.L); PT = P.T.tocsr()
                for i, (a, f) in enumerate(zip(self.a, F)):
                    A = H.A_fine(a)
                    self.UC[i] = P @ spsolve((PT @ A @ P).tocsc(), PT @ f)
            if path:
                os.makedirs(cache_dir, exist_ok=True); np.savez(path, a=self.a, U=self.U, UC=self.UC)
        self.F = np.stack([load_vector(H, a, cfg.load, i, cfg.data_seed) for i, a in enumerate(self.a)])
        self.n_te = int(round(cfg.test_frac * cfg.n)); self.n_tr = cfg.n - self.n_te
        self.tr, self.te = np.arange(self.n_tr), np.arange(self.n_tr, cfg.n)
        self.scale = float(np.sqrt(np.mean((self.U - self.UC)[self.tr] ** 2)))

    def operators(self, idx=None):
        """Generator of the fine stiffness matrices (csc), all samples or those in idx."""
        for i in (range(len(self.a)) if idx is None else idx):
            yield self.H.A_fine(self.a[i])

    def residuals(self):
        """float64 residuals F - A u_c, shape (n, N)."""
        return np.stack([self.F[i] - A.tocsr() @ self.UC[i] for i, A in enumerate(self.operators())])
