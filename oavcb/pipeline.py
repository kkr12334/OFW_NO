"""Glue: from configs to a ready-to-train problem (`Problem`), checkpoint I/O."""
import hashlib
import os
import time

import numpy as np
import torch

from .config import DataConfig, ModelConfig, TrainConfig, to_json
from .data import Dataset
from .operator_data import OAData
from .transfers import choose_levels
from .vcycle import build_model, n_params


def resolve_device(name='auto'):
    return torch.device('cuda' if (name == 'auto' and torch.cuda.is_available()) else ('cpu' if name == 'auto' else name))


class Problem:
    """Dataset + operator data + device tensors.

    ds        Dataset (numpy);   od  OAData (all samples);   levels  number of V-cycle levels
    R         float64 residuals / scale (device);   T  targets (U - UC) / scale (float32, device)
    """

    def __init__(self, dcfg: DataConfig, mcfg: ModelConfig, device='auto', cache_dir=None, verbose=True):
        self.dcfg, self.mcfg, self.dev = dcfg, mcfg, resolve_device(device)
        t0 = time.time()
        self.ds = ds = Dataset(dcfg, cache_dir, verbose)
        H = ds.H
        self.levels = choose_levels(H, mcfg.max_levels, mcfg.min_coarsest)
        self.od = OAData(H, ds.operators(), self.levels, mcfg.oa_steps, self.dev, near_kernel=True,
                         theta=mcfg.theta, tau=mcfg.tau)
        self.build_ms = (time.time() - t0) / dcfg.n * 1e3
        self.R = torch.tensor(ds.residuals() / ds.scale, dtype=torch.float64, device=self.dev)
        self.T = torch.tensor((ds.U - ds.UC) / ds.scale, dtype=torch.float32, device=self.dev)
        if verbose:
            print(f'problem: level {H.L} ({len(H.nodes[H.L])} unknowns), {self.levels} V-cycle levels, coarsest '
                  f'{self.od.n0} nodes, {dcfg.n} samples ({ds.n_tr} train / {ds.n_te} test), device {self.dev}; operator '
                  f'data {self.build_ms:.0f} ms/sample; near-kernel vectors per sample: max {max(self.od.k_nk)}, '
                  f'{100 * np.mean(np.array(self.od.k_nk) > 0):.0f}% of samples have one')

    def new_model(self, tcfg: TrainConfig = TrainConfig()):
        torch.manual_seed(tcfg.seed)
        m = build_model(self.od, self.mcfg.variant, self.mcfg.nu, self.mcfg.params, self.mcfg.width, tcfg.act_ckpt)
        return m.to(self.dev)

    def forward(self, model, idx):
        """Network correction (target units, float32) for samples idx."""
        self.od.set_batch(model, idx)
        return model(self.R[idx]).float()


def run_key(dcfg, mcfg, tcfg):
    return hashlib.md5(to_json(dcfg, mcfg, tcfg).encode()).hexdigest()[:10]


def save_checkpoint(path, model, key, **extra):
    tmp = path + '.tmp'
    torch.save(dict(model=model.state_dict(), key=key, **extra), tmp)
    os.replace(tmp, path)                                       # atomic: a disconnect never leaves a broken file


def load_weights(path, model, device='cpu'):
    """Load weights from an OAVCycle checkpoint of this package or of the paper's runs (`finished` checkpoints)."""
    st = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(st['model'])
    return st
