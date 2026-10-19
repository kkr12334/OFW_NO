"""Self-contained verification of the implementation against the statements of Section 3 (CPU, about a minute).

    python -m oavcb check

 1. mesh/FE        stiffness matrix symmetric, P1 solve reproduces A u = F; nested prolongations
 2. Lemma 1        exact route-V transfers on a two-level dense problem give A^-1 = Phi S^-1 Phi^T + E_F A_FF^-1 E_F^T
 3. Proposition 2  smoothers set to nu-step weighted Jacobi: the network equals the classical deflated, projected cycle
 4. Proposition 3  positive homogeneity  N(lambda r) = lambda N(r)
 5. Proposition 4  energy safeguard for random (untrained) weights: never worse than u_c, never worse than deflation alone
 6. detector       near-kernel vectors vs the floating conductive islands of the coefficient (barrier fields)
 7. training       a short run trains, its output is never worse than u_c in energy, a checkpoint round-trips
"""
import os
import sys
import tempfile
import time

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve

from .classical import classical_cycle, galerkin_projection
from .config import DataConfig, ModelConfig, TrainConfig
from .evaluate import evaluate
from .fields import field_channels
from .mesh import Hierarchy, coarse_mesh
from .operator_data import OAData
from .pipeline import Problem, load_weights, save_checkpoint
from .train import fit
from .transfers import oa_hierarchy, near_kernel_basis, choose_levels
from .vcycle import OAVCycle

RESULTS = []


def _ok(name, cond, detail=''):
    RESULTS.append((name, bool(cond)))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)


def check_mesh():
    H = Hierarchy(coarse_mesh('perforated', 9), 2)
    a = np.random.default_rng(0).uniform(1, 100, H.n_elem)
    A = H.A_fine(a)
    sym = abs(A - A.T).max() / abs(A).max()
    u = spsolve(A.tocsc(), H.F); res = np.linalg.norm(A @ u - H.F) / np.linalg.norm(H.F)
    nested = all(np.array_equal(H.nodes[k][:len(H.nodes[k - 1])], H.nodes[k - 1]) for k in range(1, H.L + 1))
    _ok('1. mesh / FE', sym < 1e-13 and res < 1e-10 and u @ H.F > 0 and nested,
        f'symmetry {sym:.1e}, solve residual {res:.1e}, {len(H.nodes[H.L])} unknowns, nested node sets: {nested}')


def check_lemma1():
    H = Hierarchy(coarse_mesh('perforated', 9), 2)
    A = H.A_fine(np.random.default_rng(1).uniform(1, 100, H.n_elem)).toarray()
    nc = H.P[2].shape[1]; nF = A.shape[0] - nc
    AFF, AFC = A[nc:, nc:], A[nc:, :nc]
    Phi = np.vstack([np.eye(nc), -np.linalg.solve(AFF, AFC)])
    EF = np.vstack([np.zeros((nc, nF)), np.eye(nF)])
    S = Phi.T @ A @ Phi
    Ainv = Phi @ np.linalg.solve(S, Phi.T) + EF @ np.linalg.solve(AFF, EF.T)
    err = np.abs(Ainv @ A - np.eye(A.shape[0])).max(); off = np.abs(EF.T @ A @ Phi).max() / np.abs(A).max()
    _ok('2. Lemma 1 (route-V block diagonalisation)', err < 1e-8 and off < 1e-10,
        f'|A^-1 A - I|max {err:.1e}, E_F^T A Phi {off:.1e}')


def _barrier_problem(n=8, level=3):
    d = DataConfig(level=level, k0=level - 1, field='channels', contrast=1e-4, load='no-low', n=n, data_seed=1,
                   test_frac=0.25)
    return Problem(d, ModelConfig(), device='cpu', verbose=False)


def check_containment(prob):
    mc, ds = prob.mcfg, prob.ds
    od = OAData(ds.H, ds.operators(), prob.levels, mc.oa_steps, 'cpu', near_kernel=True, dtype=torch.float64)
    nu, om = mc.nu, 2 / 3
    net = OAVCycle(od, width=4, nu=nu, project=True, deflate=True).double()
    with torch.no_grad():
        for S in (net.pre[0], net.post[0]):
            for p in S.parameters():
                p.zero_()
            S.enc_lin.weight[1, 0] = 1.0                      # channel 1 carries z / sigma
            for t in range(nu):
                S.V[t].weight[0, 0] = 1.0
                Lw = S.Lw[t].weight; Lw[0, 1] = om; Lw[0, 0] = -om; Lw[0, S.C] = -om
            S.dec_lin.weight[0, 0] = 1.0
    idx = np.arange(len(od)); od.set_batch(net, idx)
    R = torch.tensor(ds.residuals(), dtype=torch.float64)
    with torch.no_grad():
        out = net(R).numpy()
    worst = 0.0
    for i, A in enumerate(ds.operators()):
        A = A.tocsr(); levs, A0 = oa_hierarchy(ds.H, A, prob.levels, mc.oa_steps)
        ref = classical_cycle(levs, A0, R[i].numpy(), Z=near_kernel_basis(A, mc.theta, mc.tau), nu=nu, project=True,
                              smoother='jacobi')
        worst = max(worst, np.abs(out[i] - ref).max() / np.abs(ref).max())
    _ok('3. Proposition 2 (containment of the weighted-Jacobi cycle)', worst < 1e-8,
        f'network vs classical deflated + projected V({nu},{nu}), max rel. difference {worst:.1e} '
        f'(near-kernel vectors per sample: {od.k_nk})')


def check_safeguard_homogeneity(prob):
    ds, od = prob.ds, prob.od
    viol, ratios, worst_h = 0, [], 0.0
    for seed in range(3):
        torch.manual_seed(seed)
        net = prob.new_model(TrainConfig(seed=seed)); net.eval()
        for i in range(len(ds.a)):
            with torch.no_grad():
                c = prob.forward(net, np.array([i]))[0].double().numpy() * ds.scale
            A = ds.H.A_fine(ds.a[i]).tocsr(); e = ds.U[i] - ds.UC[i]; r = A @ e
            Zi = od.Z[i].double().numpy(); x0 = galerkin_projection(A, Zi[:, np.linalg.norm(Zi, axis=0) > 0], r)
            en = lambda v: np.sqrt(v @ (A @ v))
            if en(e - c) > en(e) * (1 + 1e-5) or en(e - c) > en(e - x0) * (1 + 1e-5):
                viol += 1
            ratios.append(en(e - c) / en(e))
        idx = np.arange(len(ds.a)); od.set_batch(net, idx)
        with torch.no_grad():
            f1 = net(prob.R[idx]); f3 = net(2.5 * prob.R[idx])
        worst_h = max(worst_h, float((f3 - 2.5 * f1).abs().max() / (2.5 * f1).abs().max()))
    _ok('4. Proposition 3 (positive homogeneity)', worst_h < 1e-5, f'max rel. deviation {worst_h:.1e} (single precision)')
    _ok('5. Proposition 4 (energy safeguard, random weights)', viol == 0,
        f'{viol} violations in {3 * len(ds.a)} cases; ||e - c||_A / ||e||_A in [{min(ratios):.3f}, {max(ratios):.3f}]')


def check_detector(n=40):
    H = Hierarchy(coarse_mesh('perforated', 9), 3)
    rng = np.random.default_rng(1000); pos = -np.ones(len(H.coords), int); pos[H.nodes[3]] = np.arange(len(H.nodes[3]))
    n_isl = n_det = n_cov = 0; rq_max = 0.0; fp = 0
    for _ in range(n):
        a = field_channels(H.centroids, 1e-4, rng); A = H.A_fine(a); d = A.diagonal()
        Z = near_kernel_basis(A); n_det += Z.shape[1]
        for j in range(Z.shape[1]):
            rq_max = max(rq_max, Z[:, j] @ (A @ Z[:, j]) / (Z[:, j] @ (d * Z[:, j])))
        el = H.elem[a >= 0.5]                                # coefficient islands: conductive components without a Dirichlet node
        M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
        used = np.unique(el.ravel()); n_, lab = connected_components((M.T @ M).tocsr()[used][:, used], directed=False)
        touch = np.zeros(n_, bool); touch[lab[np.isin(used, H.boundary)]] = True
        for c in np.nonzero(~touch)[0]:
            z = np.zeros(A.shape[0]); z[pos[used[lab == c]]] = 1.0; n_isl += 1
            res = z - Z @ np.linalg.lstsq(Z, z, rcond=None)[0] if Z.shape[1] else z
            n_cov += int(np.linalg.norm(res) / np.linalg.norm(z) < 1e-8)
    _ok('6. near-kernel detector', n_isl == 0 or n_cov >= 0.9 * n_isl,
        f'{n} barrier samples: {n_isl} coefficient islands, {n_cov} exactly in span(Z); {n_det} vectors detected, '
        f'max Rayleigh quotient {rq_max:.1e}')


def check_training():
    d = DataConfig(level=2, k0=1, field='channels', contrast=1e4, n=16, data_seed=0, test_frac=0.25)
    prob = Problem(d, ModelConfig(params=3000), device='cpu', verbose=False)
    tcfg = TrainConfig(epochs=6, batch=4, device='cpu')
    model = prob.new_model(tcfg)
    with tempfile.TemporaryDirectory() as tmp:
        ck = os.path.join(tmp, 'ckpt.pt')
        t0 = time.time(); fit(prob, model, tcfg, ck, key='check', log=lambda *_: None)
        res = evaluate(prob, model)
        m2 = prob.new_model(tcfg); load_weights(ck, m2)
        same = all(torch.equal(p, q) for p, q in zip(model.state_dict().values(), m2.state_dict().values()))
    en_net, en_c = res['net']['energy'].mean(), res['coarse']['energy'].mean()
    _ok('7. training / evaluation / checkpoint', np.isfinite(en_net) and res['n_worse_than_uc'] == 0 and same,
        f'6 epochs in {time.time() - t0:.0f}s, test energy error {en_net:.3e} (u_c alone {en_c:.3e}), '
        f'worse than u_c on {res["n_worse_than_uc"]} samples, checkpoint round-trip: {same}')


def main(quick=False):
    torch.manual_seed(0)
    check_mesh(); check_lemma1()
    prob = _barrier_problem()
    check_containment(prob); check_safeguard_homogeneity(prob)
    check_detector(n=12 if quick else 40)
    check_training()
    bad = [n for n, ok in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed" + (f"; FAILED: {bad}" if bad else ''))
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
