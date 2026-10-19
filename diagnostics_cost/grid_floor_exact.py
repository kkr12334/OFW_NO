"""10 Oct (after HARMONIC_DESIGN_REVIEW point 3 on the grid note): robust oracle for the grid bridge.  range(T) is
computed as the eigenvectors of T T^T (interior-node space, 3486 x 3486) with eigenvalue > 1e-12 x max (numerical rank
reported); the best energy-norm approximation of t from range(T) is then the exact Galerkin solve with Q^T A Q (Q
orthonormal basis).  Training samples ids 0-39 of data seed 1 (channels 1e4, L3, k0 2).  Errors relative to ||t||_A (the
detail target) AND converted to the full-solution metric (x ||t||_A / ||u||_A), with the residual orthogonality check."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup
from mesh_grid_bridge import MeshGridBridge
S = setup(3, 2, 1); H = S['H']
for ng in (63, 90, 126):
    BR = MeshGridBridge(H, ng=ng); T = BR.to_nodes.tocsr()
    TT = (T @ T.T).toarray(); w, V = np.linalg.eigh(TT); keep = w > 1e-12 * w.max(); Q = V[:, keep]
    rt_d, rt_f, b_d, b_f, b_l, orth = [], [], [], [], [], []
    for i in S['tr'][:40]:
        A = S['A'][i]; u = S['U'][i]; t = u - S['UC'][i]
        nt, nu = np.sqrt(t @ (A @ t)), np.sqrt(u @ (A @ u))
        rt = T @ BR.interior_to_grid(t[None])[0].ravel(); e = rt - t
        rt_d.append(np.sqrt(e @ (A @ e)) / nt); rt_f.append(np.sqrt(e @ (A @ e)) / nu)
        AQ = (A @ Q); y = np.linalg.solve(Q.T @ AQ, AQ.T @ t); b = Q @ y; e = b - t
        b_d.append(np.sqrt(e @ (A @ e)) / nt); b_f.append(np.sqrt(e @ (A @ e)) / nu); b_l.append(np.linalg.norm(e) / np.linalg.norm(t))
        orth.append(np.abs(AQ.T @ e).max() / np.abs(AQ.T @ t).max())
    f = lambda v: f"{np.mean(v):.3e} [{np.min(v):.2e}, {np.max(v):.2e}]"
    print(f"ng {ng}: rank(T) {keep.sum()} of {T.shape[0]}\n  round trip   energy/||t|| {f(rt_d)}   in the full-solution metric {f(rt_f)}"
          f"\n  best range   energy/||t|| {f(b_d)}   full-solution metric {f(b_f)}   nodal rel-L2 of t {f(b_l)}   max rel orthogonality residual {max(orth):.1e}", flush=True)
