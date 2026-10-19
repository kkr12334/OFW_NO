"""EXPLORATORY (not part of the protocol's decision; development set D only, seed 1): where does OA-VCB's remaining
island-sample error live?  Energy of e = u - x (x = OA-VCB, and OA-VCB + E2(rho*=2)) split by element region:
floating conductive islands (a >= 0.5, component touching no Dirichlet node), barrier elements (a < 0.5), connected
conductive elements.  e^T A_region e with A_region the stiffness assembled on that region's elements only."""
import sys, os, json, numpy as np, scipy.sparse as sp, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply
from harmonic_study import CK, ROOT, rings, harmonic_col, project
from scipy.sparse.csgraph import connected_components
from oa_vcycle import near_kernel_basis
torch.set_num_threads(2)
S = setup(3, 2, 1, contrast=1e-4, load='no-low'); H = S['H']; te = list(S['te'])
D, net, st = load_net(S, os.path.join(ROOT, CK[1]), te)
R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te]); C = apply(net, D, R, S['U_scale'])
dir_nodes = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[H.L])
def regions(a):
    a = np.asarray(a); cond = H.keep & (a >= 0.5); el_idx = np.nonzero(cond)[0]; el = H.elem[el_idx]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M @ M.T).tocsr(); n_, lab = connected_components(G, directed=False)
    touch = np.zeros(n_, bool)
    for k, e in enumerate(el):
        if np.isin(e, dir_nodes).any(): touch[lab[k]] = True
    isl = np.zeros(H.n_elem, bool); isl[el_idx[~touch[lab]]] = True
    bar = H.keep & (a < 0.5); conn = cond & ~isl
    return isl, bar, conn
out = {'oa-vcb': [], 'oa-vcb+E2': []}
for j, i in enumerate(te):
    A = S['A'][i]; Zi = near_kernel_basis(A)
    if Zi.shape[1] == 0: continue
    a = np.asarray(S['a'][i]); u = S['U'][i]; x = S['UC'][i] + C[j]
    Adj = (abs(A) > 0).astype(float).tocsr(); Adj.setdiag(0); Adj.eliminate_zeros()
    Z = np.column_stack([Zi] + [harmonic_col(A, np.nonzero(Zi[:, k])[0], rings(Adj, np.nonzero(Zi[:, k])[0], 2)) for k in range(Zi.shape[1])])
    e = u - x; d, _ = project(A, Z, S['Fs'][i] - A @ x, e)
    isl, bar, conn = regions(a); parts = [H.A_fine(np.where(m, a, 0.0)).tocsr() for m in (isl, bar, conn)]
    for name, ee in (('oa-vcb', e), ('oa-vcb+E2', e - d)):
        tot = ee @ (A @ ee); out[name].append([ee @ (P @ ee) / tot for P in parts])
for name, v in out.items():
    v = np.array(v); print(f"{name:<10} ({len(v)} island samples): share of error energy -- islands {v[:, 0].mean():.3f} (median {np.median(v[:, 0]):.3f}), "
                          f"barriers {v[:, 1].mean():.3f}, connected conductive {v[:, 2].mean():.3f}  (sum check {v.sum(1).mean():.3f})")
