"""OA-VCB: operator-adapted learned V-cycle with near-kernel deflation and Galerkin output (single, self-contained package).

    config      DataConfig / ModelConfig / TrainConfig
    mesh        coarse triangulations, nested refinement, P1 assembly
    fields      coefficient fields and loads
    data        Dataset: samples, reference solutions, Galerkin coarse solve u_c, residuals
    transfers   per-sample operator data from A alone: operator-adapted transfers, Galerkin levels, near-kernel vectors
    operator_data  OAData: those data as batched tensors on fixed sparsity patterns
    smoother    the learned operator-consistent smoother (the only trainable part)
    vcycle      OAVCycle (cycle + deflation + Galerkin output), model variants
    classical   matched classical reference and contained members (numpy)
    pipeline    Problem (everything on the device), checkpoints
    train       loss and training loop
    evaluate    errors, classical references, paired G / W
    checks      verification of the implementation against Section 3
    cli         python -m oavcb {train,eval,check}
"""
from .config import DataConfig, ModelConfig, TrainConfig
from .pipeline import Problem
from .vcycle import OAVCycle, VARIANTS, build_model

__all__ = ['DataConfig', 'ModelConfig', 'TrainConfig', 'Problem', 'OAVCycle', 'VARIANTS', 'build_model']
