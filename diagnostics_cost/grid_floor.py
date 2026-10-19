"""10 Oct: representation floor of the grid baselines (FNO / WNO / LiNO / MgNO + the mesh-grid bridge), TRAINING samples
only (ids 0-299 of held-out data seed 1; channels 1e4, L3, k0 2).  A grid model's prediction reaches the mesh only
through the fixed bilinear map T (grid -> interior nodes).  Two oracle quantities for the detail target t = u - u_c:
  round trip   t -> grid (barycentric, zero outside the domain) -> T: the field a grid model would output if it
               predicted the exact grid samples of t;
  best in range(T)  min_g ||t - T g||_A / ||t||_A  (exact energy-norm projection onto range(T)), with rank(T).
Also the same for the full solution u.  Energy and nodal-L2 relative errors."""
import sys, os, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup
from mesh_grid_bridge import MeshGridBridge
S = setup(3, 2, 1); H = S['H']; BR = MeshGridBridge(H); T = BR.to_nodes.tocsr()
print(BR.report(), f"; T: {T.shape}")
idx = S['tr'][:40]
en = lambda A, e, t: np.sqrt(e @ (A @ e) / (t @ (A @ t)))
rows = {'detail t, round trip': [], 'u, round trip': [], 'detail t, best in range(T)': []}
l2 = {k: [] for k in rows}
# rank of T (dense SVD of the interior-node x grid map is fine at L3)
sv = np.linalg.svd(T.toarray(), compute_uv=False); rk = int((sv > 1e-10 * sv[0]).sum())
print(f"rank(T) = {rk} of {T.shape[0]} interior nodes ({'full row rank: any nodal field is representable' if rk == T.shape[0] else 'rank deficient'})")
for i in idx:
    A = S['A'][i]; u = S['U'][i]; t = u - S['UC'][i]
    for name, f in (('detail t, round trip', t), ('u, round trip', u)):
        rt = T @ BR.interior_to_grid(f[None])[0].ravel()
        rows[name].append(en(A, rt - f, f)); l2[name].append(np.linalg.norm(rt - f) / np.linalg.norm(f))
    if rk < T.shape[0]:
        M = (T.T @ A @ T).toarray(); g = np.linalg.lstsq(M, T.T @ (A @ t), rcond=1e-12)[0]; b = T @ g
    else:
        b = t
    rows['detail t, best in range(T)'].append(en(A, b - t, t)); l2['detail t, best in range(T)'].append(np.linalg.norm(b - t) / np.linalg.norm(t))
for k in rows:
    print(f"  {k:<30} energy {np.mean(rows[k]):.3f} (min {np.min(rows[k]):.3f}, max {np.max(rows[k]):.3f})   nodal rel-L2 {np.mean(l2[k]):.3f}")
