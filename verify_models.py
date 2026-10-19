"""Smoke test every exported model: instantiate, RUN forward, RUN backward,
and check the real parameter count against the analytic formula.

This exists because the previous audit reported a shape-OK column for
OFWNeuralOperator that was hardcoded, not measured -- and the class had in
fact been truncated mid-loop during a copy, so calling it returned None.
Nothing here is allowed to assert a result it did not compute.
"""
import sys, numpy as np, torch, torch.nn as nn
sys.path.insert(0, '.')
from ofwno_nets import (OFWNeuralOperator, LinearSkipOFWNeuralOperator, RealFNO2D,
                        RealWNO2D, MLPBaseline, SpatialOFWNO, build_calibrated_model,
                        spatial_ofwno_params, _WAVELET_BACKEND)
from ofwno_benchmark_v2 import ofwno_param_count, fno_params, wno_params, mlp_params

DIMS = [3, 5, 21, 105, 465, 1953, 1673]
LD = {i: v for i, v in enumerate(DIMS)}
MU, N, B = 16, 65, 3
torch.manual_seed(0)

def real_params(m):
    return sum(p.numel() * (2 if p.is_complex() else 1)
               for p in m.parameters() if p.requires_grad)

def exercise(name, model, x, want, formula=None):
    """Run forward AND backward; every column below is measured."""
    out = model(x)
    d = out if isinstance(out, dict) else {None: out}
    shapes = {k: tuple(v.shape) for k, v in d.items()}
    ok_shape = all(shapes[k] == want[k] for k in want) and set(shapes) == set(want)
    loss = sum(v.float().pow(2).mean() for v in d.values())
    loss.backward()
    n_par = sum(1 for _ in model.parameters())
    n_grad = sum(p.grad is not None for p in model.parameters())
    p = real_params(model)
    ok_par = (formula is None) or (p == formula)
    print(f"{name:<28}{p:>10,}{'' if formula is None else format(formula, '>10,')}"
          f"{'  OK' if ok_shape else '  BAD SHAPE':>13}"
          f"{f'  {n_grad}/{n_par} grads':>16}"
          f"{'' if ok_par else '   PARAM MISMATCH'}")
    return ok_shape and ok_par and n_grad == n_par

print(f"wavelet backend: {_WAVELET_BACKEND}\n")
print(f"{'model':<28}{'actual':>10}{'formula':>10}{'forward':>13}{'backward':>16}")
mu = torch.randn(B, MU)
xg = torch.randn(B, N, N, 3)
want_lv = {l: (B, LD[l]) for l in LD}
ij = {l: torch.stack([torch.arange(LD[l]) % N, torch.arange(LD[l]) // N], 1) for l in LD}

ok = []
ok.append(exercise("OFWNeuralOperator", OFWNeuralOperator(MU, LD), mu, want_lv,
                   ofwno_param_count(LD) - MU * sum(DIMS)))
ok.append(exercise("LinearSkipOFWNeuralOperator", LinearSkipOFWNeuralOperator(MU, LD),
                   mu, want_lv, ofwno_param_count(LD)))
ok.append(exercise("build_calibrated_model",
                   build_calibrated_model(MU, LD, np.random.randn(20, MU).astype('float32'),
                                          {l: np.random.randn(20, d).astype('float32')
                                           for l, d in LD.items()}),
                   mu, want_lv, ofwno_param_count(LD)))
ok.append(exercise("SpatialOFWNO", SpatialOFWNO(ij, 3, 16, 12, 4, 64), xg, want_lv,
                   spatial_ofwno_params(16, 12, 4, 64, len(DIMS))))
ok.append(exercise("RealFNO2D", RealFNO2D(3, 21, 12, 4), xg, {None: (B, N, N)},
                   fno_params(21, 12, 4)))
ok.append(exercise("RealWNO2D", RealWNO2D(3, 80, 4), xg, {None: (B, N, N)},
                   wno_params(80, 4)))
ok.append(exercise("MLPBaseline", MLPBaseline(MU, sum(DIMS), (256, 256, 97)), mu,
                   {None: (B, sum(DIMS))}, mlp_params(MU, sum(DIMS), (256, 256, 97))))

print(f"\n{sum(ok)}/{len(ok)} models forward, backward and count correctly")
sys.exit(0 if all(ok) else 1)
