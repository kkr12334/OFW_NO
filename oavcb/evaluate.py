"""Evaluation: errors of the network and of the matched classical references on the test split, paired comparison (G, W)."""
import time

import numpy as np
import torch

from .classical import classical_cycle
from .pipeline import Problem
from .transfers import oa_hierarchy, near_kernel_basis


@torch.no_grad()
def predict(prob: Problem, model, idx, batch=8, project=None):
    """Full solution estimates u_c + c for samples idx (numpy, float64).  project overrides model.project."""
    old = model.project
    if project is not None:
        model.project = project
    model.eval(); out = []
    for ch in np.array_split(np.asarray(idx), int(np.ceil(len(idx) / batch))):
        out.append(prob.forward(model, ch).cpu().numpy())
    model.project = old
    return np.concatenate(out).astype(np.float64) * prob.ds.scale + prob.ds.UC[idx]


def errors(prob: Problem, P, idx):
    """Per-sample errors of estimates P (rows aligned with idx): nodal relative L2, relative energy, relative FE-L2."""
    ds = prob.ds; M = ds.H.mass_matrix()
    rl2, en, l2fe = [], [], []
    for p, i in zip(P, idx):
        A = ds.H.A_fine(ds.a[i]).tocsr(); u = ds.U[i]; e = p - u
        rl2.append(np.linalg.norm(e) / np.linalg.norm(u))
        en.append(float(np.sqrt(e @ (A @ e) / (u @ (A @ u)))))
        l2fe.append(float(np.sqrt(e @ (M @ e) / (u @ (M @ u)))))
    return dict(rel_l2=np.array(rl2), energy=np.array(en), l2fe=np.array(l2fe))


def evaluate(prob: Problem, model, idx=None, batch=8):
    """Errors of the trained network on idx (default: the test split), with the coarse solve u_c alone, the number of samples
    on which the output is worse in energy than u_c, and (for project='both') the plain-cycle output."""
    ds = prob.ds; idx = ds.te if idx is None else np.asarray(idx)
    t0 = time.time(); P = predict(prob, model, idx, batch); infer_ms = (time.time() - t0) / len(idx) * 1e3
    out = dict(idx=idx, net=errors(prob, P, idx), coarse=errors(prob, ds.UC[idx], idx), infer_ms=infer_ms)
    out['n_worse_than_uc'] = int(np.sum(out['net']['energy'] > out['coarse']['energy'] * (1 + 1e-6)))
    if model.project == 'both':
        out['plain'] = errors(prob, predict(prob, model, idx, batch, project=False), idx)
    return out


def classical_reference(prob: Problem, idx=None, nu=4, members=('reference', 'fixed_cheb', 'jacobi')):
    """Per-sample errors of the classical counterparts of OA-VCB from the same u_c, same transfers and Galerkin levels, same
    deflation vectors and Galerkin output (energy errors of the whole estimate u_c + correction).
    reference   degree-nu Chebyshev, per-level adaptive interval (the paper's classical reference)
    fixed_cheb  contained fixed-interval Chebyshev member (lambda = 3 on every level)
    jacobi      contained weighted-Jacobi member"""
    ds, mc = prob.ds, prob.mcfg; H = ds.H
    idx = ds.te if idx is None else np.asarray(idx)
    P = {m: [] for m in members}
    for i in idx:
        A = H.A_fine(ds.a[i]).tocsr(); r = ds.F[i] - A @ ds.UC[i]
        levs, A0 = oa_hierarchy(H, A, prob.levels, mc.oa_steps); Z = near_kernel_basis(A, mc.theta, mc.tau)
        for m in members:
            kw = dict(reference={}, fixed_cheb=dict(lmax=3.0), jacobi=dict(smoother='jacobi'))[m]
            P[m].append(ds.UC[i] + classical_cycle(levs, A0, r, Z=Z, nu=nu, project=True, **kw))
    return {m: errors(prob, np.array(P[m]), idx) for m in members}


def paired(ref, net, n_boot=5000, seed=0):
    """Paired comparison of per-sample errors: G = geometric mean of ref / net (> 1: the network is more accurate), with a
    95% bootstrap interval over samples, and W = fraction of samples on which the network is the more accurate."""
    ref, net = np.asarray(ref), np.asarray(net)
    lr = np.log(ref / net)
    b = np.random.default_rng(seed).integers(0, len(lr), (n_boot, len(lr)))
    lo, hi = np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5]))
    return dict(G=float(np.exp(lr.mean())), G_lo=float(lo), G_hi=float(hi), W=float(np.mean(lr > 0)))


def report(res, ref=None, log=print):
    """Print the table of one evaluation (res from evaluate, ref from classical_reference)."""
    n = len(res['idx'])
    log(f"{'':<26}{'nodal rel-L2':>14}{'FE rel-L2':>12}{'energy':>12}")
    rows = [('coarse solve u_c', res['coarse']), ('OA-VCB (network)', res['net'])]
    if 'plain' in res:
        rows.append(('  plain cycle output', res['plain']))
    if ref:
        names = dict(reference='classical reference', fixed_cheb='contained Chebyshev', jacobi='contained Jacobi')
        rows += [(names.get(k, k), v) for k, v in ref.items()]
    for name, e in rows:
        log(f"{name:<26}{e['rel_l2'].mean():>14.3e}{e['l2fe'].mean():>12.3e}{e['energy'].mean():>12.3e}")
    log(f"energy worse than u_c on {res['n_worse_than_uc']}/{n} test samples; inference {res['infer_ms']:.1f} ms/sample")
    if ref:
        for k, e in ref.items():
            p = paired(e['energy'], res['net']['energy'])
            log(f"paired vs {k:<11}: G = {p['G']:.2f} [{p['G_lo']:.2f}, {p['G_hi']:.2f}], W = {p['W']:.2f}")
