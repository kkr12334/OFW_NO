# oavcb — the OA-VCB method as one modular package

A self-contained re-organisation of the method of the paper (*Operator-adapted learned multigrid cycles for high-contrast
elliptic problems*): one learned V-cycle with operator-adapted transfers, near-kernel deflation and a Galerkin output.
It needs only NumPy, SciPy and PyTorch (`../requirements.txt`) and imports nothing from the surrounding scripts.

```
python -m oavcb check                                   # verify the implementation (≈ 1 min, CPU)
python -m oavcb train --out-dir runs/channels           # paper main configuration (channels 1e4, L=3, k0=2, 300 epochs)
python -m oavcb train --preset barriers --data-seed 1 --classical
python -m oavcb eval  --ckpt runs/channels/ckpt_<key>.pt
```

Run from the folder that contains `oavcb/` (`OFW_NO/`). Every field of `config.py` is a command-line flag
(`--level 2 --epochs 20 --n 100 --params 8000 --variant oa-vcb ...`); presets: `channels`, `barriers`, `polygon`, `l4`, `k0`.

## Modules and the paper

| module | content | paper |
|---|---|---|
| `config.py` | `DataConfig`, `ModelConfig`, `TrainConfig` (defaults = main configuration) | §4.1 |
| `mesh.py` | coarse triangulations, red refinement, nested node sets, P1 stiffness / load / mass | §3.1 |
| `fields.py` | channel / barrier coefficients, loads (`all`, `no-low`, `random`) | §4.1 |
| `data.py` | samples, exact solutions, Galerkin coarse solve `u_c`, residuals | §3.1, §4.1 |
| `transfers.py` | operator-adapted transfers (eq. 4), Galerkin levels, fixed patterns, near-kernel detector | §3.2, §3.4, Alg. 1 |
| `operator_data.py` | `OAData`: the above as batched tensors | Alg. 1 |
| `smoother.py` | `OpSmoother`, the only learned component (eq. 6) | §3.3 |
| `vcycle.py` | `OAVCycle`: cycle, deflation, Galerkin output; `VARIANTS` (oa-vc … oa-vcbl); width from the parameter budget | §3.3–3.5, Alg. 2 |
| `classical.py` | classical reference (adaptive Chebyshev), contained Jacobi / fixed-interval Chebyshev members, two-grid PCG | §3.6, Prop. 2 |
| `pipeline.py` | `Problem` (data + operator data + device tensors), checkpoints | |
| `train.py` | mixed rel-L2 / energy loss (+ plain-cycle term for OA-VCB), AdamW, one-cycle, resumable | §3.5 |
| `evaluate.py` | errors, classical references on the same `u_c`, paired `G` (with bootstrap interval) and `W` | §4.1 |
| `checks.py` | Lemma 1, Propositions 2–4, detector, training smoke test | §3 |
| `cli.py` | `python -m oavcb {train,eval,check}` | |

Library use:

```python
from oavcb import DataConfig, ModelConfig, TrainConfig, Problem
from oavcb.train import fit
from oavcb.evaluate import evaluate, classical_reference, report

prob = Problem(DataConfig(contrast=1e4), ModelConfig(), device='auto')
model = prob.new_model(TrainConfig())
fit(prob, model, TrainConfig())
report(evaluate(prob, model), classical_reference(prob))
```

A different operator (tensor coefficient, elasticity) plugs in through `OAData(H, A_list, ..., A_struct=...)`: the network
sees only the matrices. The detector is for scalar diffusion; pass `near_kernel=False` otherwise.

## Relation to the released scripts

* Module and parameter names of `OpSmoother` / `OAVCycle` are those of the released `oa_vcycle.py`, so the released
  `ckpt_*_oa-vcb.pt` files load with `eval --ckpt` when the configuration matches (same mesh, level, `--params`).
* The sample stream is the same (`default_rng(1000 + data_seed)`, same channel generator, same loads, last 25 % are the test
  set), the training recipe is the same (AdamW 2e-3, wd 1e-5, one-cycle, clip 1, batch 8, loss on projected + plain output).
  Results are not guaranteed bit-identical: stiffness assembly is vectorised, and `h_0` is the mean coarse edge length
  also for the plain unit square (the released code used 1 there; the paper's meshes are unaffected).
* This directory adds no top-level `.py` file, so the code identity `f8653612eb84` of `run_identity.py` is unchanged.
  `RELEASE_SHA256.txt` must be regenerated (`tools/release_manifest.py`) if this directory is part of the release.

## Not included

Baselines (FNO, WNO, MgNO, LiNO, Transolver, GNN, oa-ll-op), the anisotropic-tensor and elasticity assemblers,
mixed boundary conditions, the L-shaped grid domain, the Gaussian-random-field and block coefficients, the unit-test-style
`mathcheck/` scripts beyond those re-stated in `checks.py`, and the cost/sensitivity diagnostics. They stay in the
top-level scripts.
