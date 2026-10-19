"""10 Oct: shared set-up for CPU re-evaluation of the paper checkpoints (Darcy, perforated coarse mesh n 9, channels 1e4,
--coarse-solve galerkin), reproducing train_oa_layer.py's data and inference path exactly:
  * coefficients: rng 1000 + data_seed, n = 400, make_field(H, 'channels', 1e4, rng); test ids 300-399;
  * u_c: Galerkin solve on the level-k0 mesh, prolonged (k0 = 0: u_c = 0);
  * U_scale = rms of (U - U_c) over the 300 training samples (the network is positively homogeneous, but the driver feeds
    r / U_scale, so the same scaling is used here and the output is multiplied back);
  * residual in float64, fine operator passed in float64 (A64) for deflation + Galerkin output -- the driver's path.
This file lives outside the top-level code folder, so it does not change the code hash (run_identity.py)."""
import os, sys, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle


def vc_levels(H, L):
    nl, k_ = 0, L
    while k_ >= 1 and nl < 4 and len(H.nodes[k_ - 1]) >= 16:
        nl += 1; k_ -= 1
    return nl


def coarse_P(H, L, k0):
    if k0 == 0: return None
    P = None
    for k in range(k0 + 1, L + 1): P = H.P[k] if P is None else H.P[k] @ P
    return P.tocsr()


def load_no_low(H, a):
    """train_oa_layer.load_vec with --load no-low: f = 1 only on elements with a >= 0.5."""
    xy = H.coords[H.elem]
    area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
    m = H.keep & (np.asarray(a).ravel() >= 0.5); Fv = np.zeros(len(H.coords))
    for j in range(3): np.add.at(Fv, H.elem[m, j], area[m] / 3.0)
    return Fv[H.nodes[H.L]]


def setup(L, k0, data_seed, n=400, contrast=1e4, field='channels', load='all', rng_seed=None):
    """load 'all' (f = 1) or 'no-low' (barriers).  rng_seed overrides 1000 + data_seed (fresh confirmatory operators)."""
    H = NestedHierarchy(L, coarse=perforated_square(n=9))
    rng = np.random.default_rng(1000 + data_seed if rng_seed is None else rng_seed)
    a_all = [make_field(H, field, contrast, rng) for _ in range(n)]
    Fs = [H.F if load == 'all' else load_no_low(H, a) for a in a_all]
    F = H.F
    As = [H.A_fine(a).tocsr() for a in a_all]
    U = np.array([spsolve(A.tocsc(), Fi) for A, Fi in zip(As, Fs)])
    P = coarse_P(H, L, k0)
    UC = (np.array([P @ spsolve((P.T @ A @ P).tocsc(), P.T @ Fi) for A, Fi in zip(As, Fs)]) if P is not None else np.zeros_like(U))
    n_te = int(round(0.25 * n)); tr = np.arange(n - n_te); te = np.arange(n - n_te, n)
    U_scale = float(np.sqrt(np.mean((U - UC)[tr] ** 2)))
    return dict(H=H, L=L, k0=k0, a=a_all, A=As, F=F, Fs=Fs, U=U, UC=UC, P=P, te=te, tr=tr, U_scale=U_scale, nl=vc_levels(H, L))


def load_net(S, ckpt, idx, oa_steps=4):
    """OAData for samples idx (order kept), and the oa-vcb network with the checkpoint's weights."""
    H = S['H']
    D = OAData(H, [S['a'][i] for i in idx], S['nl'], oa_steps, 'cpu', H.level_coords, near_kernel=True)
    st = torch.load(ckpt, map_location='cpu', weights_only=False)
    SD = st['model']
    net = OAVCycle(D, width=SD['pre.0.enc_lin.weight'].shape[0], nu=4, project='both', deflate=True)
    net.load_state_dict(SD); net.eval()
    return D, net, st


def apply(net, D, R, U_scale, bs=8, plain=False):
    """correction c = U_scale * net(R / U_scale) for residual rows R (float64, rows in D's sample order)."""
    out = []; pj = net.project
    if plain: net.project = False
    with torch.no_grad():
        for j in np.array_split(np.arange(len(R)), int(np.ceil(len(R) / bs))):
            jt = torch.tensor(j)
            net.set_ops(*D.batch(jt), Z=D.batch_z(jt), A64=D.batch_a64(jt))
            out.append(net(torch.tensor(R[j] / U_scale, dtype=torch.float64)).double().numpy())
    net.project = pj
    return np.concatenate(out) * U_scale


def energy(A, x, u):
    e = x - u
    return float(np.sqrt(e @ (A @ e) / (u @ (A @ u))))
