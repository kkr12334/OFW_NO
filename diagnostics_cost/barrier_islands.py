"""4 Oct: the barrier field (channels, contrast 1e-4, --load no-low) mixes two regimes.  Four domain-crossing
low-conductivity strips can enclose a conductive region with no Dirichlet node: a FLOATING ISLAND.  Its indicator is a
near-kernel vector of A (Rayleigh quotient ~ contrast), the solution there is ~100x larger, and every one-pass method --
and few-iteration Krylov -- fails on those samples, which then dominate the test means.

Classical remedy (deflation / coarse-space enrichment, Nicolaides; GenEO-type spectral coarse spaces): add the island
indicators Z (one column per island, 1 on the island's nodes) to the coarse space,
    x = Z E^-1 Z^T b;  x += V(b - A x);  x += Z E^-1 Z^T (b - A x),   E = Z^T A Z  (k x k, k = number of islands).
Islands are found from the coefficient alone (connected components of the conductive elements, a >= 0.5, that contain
no Dirichlet node) -- per sample, a graph search, negligible cost.  Darcy L3, the GPU run's test set (samples 300-399,
rng 1000), geometric u_c (k0 2), energy errors.  Also: the Rayleigh quotient of the island indicator."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, classical_vcycle

L = 3; H = NestedHierarchy(L, coarse=perforated_square(n=9))
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]
P2 = H.P[3].tocsr()
dir_nodes = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L]); pos = -np.ones(len(H.coords), int); pos[H.nodes[L]] = np.arange(len(H.nodes[L]))

def island_basis(a):
    cond = H.keep & (a >= 0.5); el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M.T @ M).tocsr(); used = np.unique(el.ravel())
    n_, lab = connected_components(G[used][:, used], directed=False)
    touch = np.zeros(n_, bool); touch[lab[np.isin(used, dir_nodes)]] = True
    cols = []
    for c in np.nonzero(~touch)[0]:
        z = np.zeros(len(H.nodes[L])); z[pos[used[lab == c]]] = 1.0; cols.append(z)
    return np.array(cols).T if cols else None

def errs(x, u, A):
    e = x - u; return np.sqrt(e @ (A @ e) / (u @ (A @ u)))

rng = np.random.default_rng(1000); A_all = [field_channels(H, 1e-4, rng) for _ in range(400)]
R = {k: [] for k in ('island', 'k', 'rq', 'coarse', 'vc', 'vc_defl', 'tg4', 'tg4_defl')}
for i in range(300, 400):
    a = A_all[i]; A = H.A_fine(a).tocsr(); F = load(a >= 0.5); u = spsolve(A.tocsc(), F); d = A.diagonal()
    uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); r = F - A @ uc
    levs, A0 = oa_hierarchy(H, A, 3, 4)
    Z = island_basis(a); R['island'].append(Z is not None); R['k'].append(0 if Z is None else Z.shape[1])
    if Z is not None:
        E = Z.T @ (A @ Z); defl = lambda v: Z @ np.linalg.solve(E, Z.T @ v)
        R['rq'].append(np.min(np.diag(E) / (Z * Z * d[:, None]).sum(0)))       # Rayleigh quotient of D^-1 A for the indicator
    else:
        defl = lambda v: 0 * v
    R['coarse'].append(errs(uc, u, A))
    R['vc'].append(errs(uc + classical_vcycle(levs, A0, r), u, A))
    x = defl(r); x = x + classical_vcycle(levs, A0, r - A @ x); x = x + defl(r - A @ x)
    R['vc_defl'].append(errs(uc + x, u, A))
    # two-grid PCG, 4 its, plain and with the islands added to the coarse space (deflated preconditioner)
    for tag, use in (('tg4', False), ('tg4_defl', True)):
        Pg = P2 if not (use and Z is not None) else sp.hstack([P2, sp.csr_matrix(Z)]).tocsr()
        solc = spsolve; Ac = (Pg.T @ A @ Pg).tocsc()
        from scipy.sparse.linalg import factorized; fc = factorized(Ac); dinv = 1 / d
        def tg(rr):
            y = (2 / 3) * dinv * rr; y = y + Pg @ fc(Pg.T @ (rr - A @ y)); return y + (2 / 3) * dinv * (rr - A @ y)
        xx = uc.copy(); rr = F - A @ xx; z = tg(rr); p = z.copy(); rz = rr @ z
        for _ in range(4):
            Ap = A @ p; al = rz / (p @ Ap); xx = xx + al * p; rr = rr - al * Ap; z = tg(rr); rz2 = rr @ z; p = z + (rz2 / rz) * p; rz = rz2
        R[tag].append(errs(xx, u, A))
isl = np.array(R['island'])
print(f"L3 barrier test set (samples 300-399): {isl.sum()} of 100 with a floating island (islands per such sample: "
      f"{np.bincount(np.array(R['k'])[isl])[1:]} for 1, 2, ...); min Rayleigh quotient of D^-1 A for an island indicator: "
      f"{np.min(R['rq']):.1e} (median {np.median(R['rq']):.1e})")
print(f"{'energy error, mean | median':<42}{'with island':>24}{'without island':>24}")
for k, lab in (('coarse', 'geometric coarse solve alone'), ('vc', 'classical V(4,4), operator-adapted'),
               ('vc_defl', '  + island deflation'), ('tg4', 'two-grid PCG, 4 its'), ('tg4_defl', '  + islands in the coarse space')):
    v = np.array(R[k])
    print(f"{lab:<42}{f'{v[isl].mean():.3f} | {np.median(v[isl]):.3f}':>24}{f'{v[~isl].mean():.3f} | {np.median(v[~isl]):.3f}':>24}")
