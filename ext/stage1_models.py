"""Architecture-specific loaders for the learned Stage 1 models (amendment 1, A1).  The flags reproduce the training
driver's constructor (train_oa_layer.py, lines 915-919):
    oa-vcb : deflate, Galerkin output (project='both' -> projected at evaluation), nonlinear smoothers
    oa-vcbl: deflate, Galerkin output, residual-linear smoothers (output still nonlinear)
    oa-vcl : deflate, PLAIN cycle output, residual-linear smoothers (historical full model; map linear in r for each A)
The model type comes from the checkpoint file name and is cross-checked against the state dict: residual-linear
smoothers have no encoder MLP ('pre.0.enc.*'); the width is read from 'pre.0.enc_lin.weight'."""
import os, sys, json, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'diagnostics_cost')):
    if p not in sys.path: sys.path.insert(0, p)
from oa_vcycle import OAData, OAVCycle   # noqa: E402

FLAGS = {'oa-vcb': dict(project='both', deflate=True, linear=False),
         'oa-vcbl': dict(project='both', deflate=True, linear=True),
         'oa-vcl': dict(project=False, deflate=True, linear=True)}


def model_type(path):
    t = os.path.basename(path).rsplit('_', 1)[1][:-3]
    if t not in FLAGS: raise ValueError(f'unsupported model type {t}')
    return t


def load(path, D):
    """network for checkpoint `path` on operator data D (OAData built with near_kernel=True)."""
    st = torch.load(path, map_location='cpu', weights_only=False); SD = st['model']; t = model_type(path)
    lin = not any(k.startswith('pre.0.enc.') for k in SD)
    if lin != FLAGS[t]['linear']: raise ValueError(f'{path}: state dict linear={lin} contradicts type {t}')
    net = OAVCycle(D, width=SD['pre.0.enc_lin.weight'].shape[0], nu=4, **FLAGS[t])
    net.load_state_dict(SD); net.eval()
    return net, st, t


def apply(net, D, R, scale=1.0, bs=8):
    """correction for residual rows R (float64): scale * net(R / scale).  The networks are positively homogeneous, so
    `scale` cancels in exact arithmetic; it is passed only to keep the training-time input magnitudes (frozen per model,
    amendment 1) and is never recomputed from reference solutions of evaluation operators."""
    out = []
    with torch.no_grad():
        for j in np.array_split(np.arange(len(R)), int(np.ceil(len(R) / bs))):
            jt = torch.tensor(j)
            net.set_ops(*D.batch(jt), Z=D.batch_z(jt), A64=D.batch_a64(jt))
            out.append(net(torch.tensor(R[j] / scale, dtype=torch.float64)).double().numpy())
    return np.concatenate(out) * scale
