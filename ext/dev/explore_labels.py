"""Development-only exploration (16 Oct): structural near-island candidates, A-only. Uses coefficient/operator structure
only -- NO solver errors are read, so the label definition is not tuned on outcomes."""
import os, sys, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square

def comps(A, theta=0.1):
    A = A.tocsr(); d = A.diagonal(); C = A.tocoo(); off = C.row != C.col
    strong = off & (-C.data >= theta * np.sqrt(d[C.row] * d[C.col]))
    S = sp.csr_matrix((np.ones(strong.sum()), (C.row[strong], C.col[strong])), shape=A.shape)
    n_, lab = connected_components(S, directed=False)
    rs = np.asarray(A.sum(1)).ravel(); anch = np.zeros(n_, bool); anch[lab[rs >= theta * d]] = True
    out = []
    for c in range(n_):
        z = (lab == c).astype(float); size = int(z.sum())
        rq = (z @ (A @ z)) / (z @ (d * z)); out.append((c, size, bool(anch[c]), rq))
    return out

H = NestedHierarchy(3, coarse=perforated_square(n=9))
for fam, contrast in (('barriers', 1e-4), ('channels', 1e4)):
    rng = np.random.default_rng(1001); a_all = [make_field(H, 'channels', contrast, rng) for _ in range(400)]
    rows = []
    for i in range(300, 400):
        A = H.A_fine(a_all[i]).tocsr(); cs = comps(A)
        unanch_nk = [c for c in cs if not c[2] and c[3] < 0.05]
        anch_small = sorted([(round(c[3], 4), c[1]) for c in cs if c[2] and c[1] >= 5])[:3]
        rows.append((i, len(unanch_nk), anch_small))
    n_isl = sum(r[1] > 0 for r in rows)
    print(f"\n{fam}: island (detector) samples {n_isl}/100")
    for thr in (0.005, 0.01, 0.02, 0.05):
        cand = [r for r in rows if r[1] == 0 and r[2] and r[2][0][0] < thr]
        print(f"  non-island samples with an ANCHORED strong component (size>=5) of Rayleigh quotient < {thr}: {len(cand)}")
    rq = np.array([r[2][0][0] for r in rows if r[1] == 0 and r[2]])
    print('  min anchored-component RQ over non-island samples, quantiles 0/10/25/50/75/90%:', np.round(np.quantile(rq, [0, .1, .25, .5, .75, .9]), 4))
