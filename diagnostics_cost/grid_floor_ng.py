"""10 Oct: grid floor vs grid resolution (training samples 0-19 of data seed 1, channels 1e4, L3, k0 2).  Best energy-norm
approximation of the detail target from range(T_ng) via sparse normal equations (T^T A T + eps I) g = T^T A t."""
import sys, os, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup
from mesh_grid_bridge import MeshGridBridge
S = setup(3, 2, 1); H = S['H']
en = lambda A, e, t: np.sqrt(e @ (A @ e) / (t @ (A @ t)))
for ng in (63, 90, 126, 180):
    BR = MeshGridBridge(H, ng=ng); T = BR.to_nodes.tocsr(); rt_e, rt_l, b_e, b_l = [], [], [], []
    for i in S['tr'][:20]:
        A = S['A'][i]; t = S['U'][i] - S['UC'][i]
        rt = T @ BR.interior_to_grid(t[None])[0].ravel(); rt_e.append(en(A, rt - t, t)); rt_l.append(np.linalg.norm(rt - t) / np.linalg.norm(t))
        M = (T.T @ A @ T).tocsc(); M = M + 1e-12 * M.diagonal().max() * sp.identity(M.shape[0], format='csc')
        b = T @ spsolve(M, T.T @ (A @ t)); b_e.append(en(A, b - t, t)); b_l.append(np.linalg.norm(b - t) / np.linalg.norm(t))
    print(f"ng {ng:>3} ({BR.ng ** 2:>5} grid pts): round trip energy {np.mean(rt_e):7.3f}  L2 {np.mean(rt_l):.3f} | best in range(T) energy {np.mean(b_e):.3f}  L2 {np.mean(b_l):.3f}", flush=True)
