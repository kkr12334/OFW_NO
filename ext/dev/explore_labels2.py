"""Development-only (16 Oct): geometric near-island candidate via one-layer erosion. Structure only; no solver errors."""
import os, sys, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square

def floating_components(H, cond_mask):
    """components (vertex adjacency) of the element set cond_mask; returns list of (n_elements, touches_dirichlet)."""
    el = H.elem[cond_mask]
    if len(el) == 0: return []
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M @ M.T).tocsr()                                   # element-element via shared node
    n_, lab = connected_components(G, directed=False)
    dirn = np.zeros(len(H.coords), bool); dirn[np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[H.L])] = True
    touch = np.zeros(n_, bool)
    for c in range(n_):
        touch[c] = dirn[el[lab == c].ravel()].any()
    return [(int((lab == c).sum()), bool(touch[c])) for c in range(n_)]

def labels(H, a, smin=30):
    a = np.asarray(a).ravel(); cond = H.keep & (a >= 0.5); ins = H.keep & (a < 0.5)
    isl = [c for c in floating_components(H, cond) if not c[1] and c[0] >= 1]
    # one-layer erosion: drop conductive elements sharing a node with an insulating element
    ins_nodes = np.zeros(len(H.coords), bool); ins_nodes[H.elem[ins].ravel()] = True
    er = cond & ~ins_nodes[H.elem].any(1)
    er_float = [c for c in floating_components(H, er) if not c[1] and c[0] >= smin]
    return len(isl), len(er_float), sorted([c[0] for c in er_float], reverse=True)[:3]

H = NestedHierarchy(3, coarse=perforated_square(n=9))
print('elements', H.n_elem, 'kept', int(H.keep.sum()))
for fam, contrast in (('barriers', 1e-4), ('channels', 1e4)):
    rng = np.random.default_rng(1001); a_all = [make_field(H, 'channels', contrast, rng) for _ in range(400)]
    res = [labels(H, a_all[i]) for i in range(300, 400)]
    geo_isl = sum(r[0] > 0 for r in res)
    near = [i for i, r in zip(range(300, 400), res) if r[0] == 0 and r[1] > 0]
    both = sum(r[0] > 0 and r[1] > 0 for r in res)
    print(f"{fam}: geometric-island samples {geo_isl}; eroded-floating (>=30 el) among NON-island samples {len(near)}; "
          f"among island samples {both}")
    print('   near-island ids', near[:20], ' sizes', [res[i-300][2] for i in near[:8]])
