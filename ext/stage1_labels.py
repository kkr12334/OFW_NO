"""Final structural labeller (amendment 1, A7.1): island / discordant / EDP / strict non-island.  Structure only."""
import os, sys, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, 'diagnostics_cost')):
    if p not in sys.path: sys.path.insert(0, p)
from scipy.sparse.csgraph import connected_components
from oa_vcycle import near_kernel_basis

def _components(H, mask):
    el = H.elem[mask]
    if len(el) == 0: return []
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    n_, lab = connected_components((M @ M.T).tocsr(), directed=False)
    dirn = np.zeros(len(H.coords), bool); dirn[np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[H.L])] = True
    return [(int((lab == c).sum()), bool(dirn[el[lab == c].ravel()].any())) for c in range(n_)]

def label(H, a, A, smin=30):
    a = np.asarray(a).ravel(); cond = H.keep & (a >= 0.5); ins = H.keep & (a < 0.5)
    det = near_kernel_basis(A).shape[1] > 0
    geo = any(not t for _, t in _components(H, cond))
    insn = np.zeros(len(H.coords), bool); insn[H.elem[ins].ravel()] = True
    edp = any((not t) and n >= smin for n, t in _components(H, cond & ~insn[H.elem].any(1)))
    cat = 'island' if (det and geo) else ('discordant' if det != geo else ('EDP' if edp else 'strict non-island'))
    return dict(category=cat, detector=bool(det), geometric=bool(geo), erosion=bool(edp))
