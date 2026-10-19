"""Configuration of one OA-VCB experiment.  Defaults are the paper's main configuration (Darcy channels, perforated square,
L = 3 refinements, k0 = 2 Galerkin coarse solve, 22k-parameter budget, 300 epochs, mixed loss)."""
from dataclasses import dataclass, asdict
import json


@dataclass
class DataConfig:
    coarse: str = 'perforated'      # perforated | polygon | lshape | square   (unstructured coarse triangulation)
    coarse_n: int = 9               # coarse mesh density parameter
    level: int = 3                  # L: uniform refinements of the coarse mesh
    k0: int = 2                     # external Galerkin coarse solve on the level-k0 mesh (0: none, u_c = 0)
    field: str = 'channels'         # channels (contrast > 1) / barriers (contrast < 1) | const
    contrast: float = 1e4
    load: str = 'all'               # all (f = 1) | no-low (f = 1 on a >= 0.5 only; used for barriers) | random
    n: int = 400                    # samples; the LAST test_frac of them are the test set
    data_seed: int = 0
    test_frac: float = 0.25


@dataclass
class ModelConfig:
    variant: str = 'oa-vcb'         # see vcycle.VARIANTS
    nu: int = 4                     # learned update stages per pre-/post-smoother
    oa_steps: int = 4               # m: Jacobi steps of prolongator smoothing (eq. 4 of the paper)
    params: int = 22000             # parameter budget; the width is the largest one that fits
    width: int = 0                  # >0: fixed latent width, ignores the budget
    max_levels: int = 4             # V-cycle levels (coarsest exact solve below them)
    min_coarsest: int = 16          # a level is added only while the next coarser one has >= this many nodes
    theta: float = 0.1              # near-kernel detector: strength threshold
    tau: float = 0.05               # near-kernel detector: Rayleigh-quotient threshold


@dataclass
class TrainConfig:
    epochs: int = 300
    batch: int = 8
    lr: float = 2e-3
    weight_decay: float = 1e-5
    loss: str = 'mix'               # mix (0.5 (energy + rel-L2)) | energy | rel_l2
    seed: int = 0                   # network initialisation / batch order
    ckpt_every: int = 0             # 0: every epochs // 6
    act_ckpt: bool = False          # activation checkpointing of the smoothers (needed for L4 on a 16 GB GPU)
    device: str = 'auto'            # auto | cpu | cuda


def to_json(*cfgs):
    return json.dumps({type(c).__name__: asdict(c) for c in cfgs}, sort_keys=True)
