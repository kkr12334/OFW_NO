"""Fidelity check of the classical baseline M0 (our RT0/Alfeld adaptation of MVY App. 8 / PRVW Def. 4.3) on the PRVW 2020
"sinus" Poisson test (their Sec. 9, Table 3, p = 1): Omega = (-1, 1)^2, u = sin(2 pi x) sin(2 pi y), unstructured initial
Delaunay mesh refined J = 4 times, conforming P1, PCG with an incomplete-factorisation preconditioner (drop tolerance 1e-4),
zero initial guess.  PRVW report algebraic upper-bound effectivity 1.01 after 3 and 6 PCG iterations (p = 1).
Differences from PRVW, recorded rather than hidden: RT0 on each level plus the Alfeld split at the finest level (theirs:
RTN_p with p = 1 on every level); our element-local P0 lifting; scipy's ILU (spilu) as preconditioner (theirs: incomplete
Cholesky); our own initial mesh (theirs is not published).  Development check only; no pilot data."""
import os, sys, json, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse.linalg import splu, spilu
from oa_lifting import NestedHierarchy
from coarse_meshes import polygon_domain
import rc_fem as R, rc_fast as FST

cm = polygon_domain([(-1, -1), (1, -1), (1, 1), (-1, 1)], n=9, jitter=0.25, seed=0)
H = NestedHierarchy(4, coarse=cm); M = R.Mesh(H, cm)
K = np.ones(len(M.fe.E))
A = M.stiffness(K)
# load (f, psi_a) with f = 8 pi^2 sin(2 pi x) sin(2 pi y), 6-point quadrature on each FE triangle (exact for degree 4)
X, E = M.fe.X, M.fe.E
f = lambda p: 8 * np.pi ** 2 * np.sin(2 * np.pi * p[..., 0]) * np.sin(2 * np.pi * p[..., 1])
qa, qb, w1 = 0.445948490915965, 0.091576213509771, 0.223381589678011; w2 = 0.109951743655322
bary = np.array([[qa, qa, 1 - 2 * qa], [qa, 1 - 2 * qa, qa], [1 - 2 * qa, qa, qa], [qb, qb, 1 - 2 * qb], [qb, 1 - 2 * qb, qb], [1 - 2 * qb, qb, qb]])
wts = np.array([w1] * 3 + [w2] * 3)
pts = np.einsum('qi,kid->kqd', bary, X[E])
loc = np.einsum('kq,qi,q->ki', f(pts), bary, wts) * M.fe.area[:, None]
Ffull = np.zeros(len(X)); np.add.at(Ffull, E.ravel(), loc.ravel()); F = Ffull[M.unk]
uh = splu(A).solve(F)
ilu = spilu(A.tocsc(), drop_tol=1e-4)
def pcg(its):
    x = np.zeros(M.N); r = F.copy(); z = ilu.solve(r); p = z.copy(); rz = r @ z; out = {}
    for k in range(1, its + 1):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap; z = ilu.solve(r); rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
        out[k] = x.copy()
    return out
its = pcg(6)
mvy = FST.MVYCache(M, K, A, 'mvy'); prvw = FST.MVYCache(M, K, A, 'prvw'); joint = FST.JointCache(M, K)
res = dict(N=int(M.N), n_fe_elements=int(len(E)), coarse_elements=int(len(M.levels[0].E)), rows=[])
for k in (1, 2, 3, 6):
    v = its[k]; Rv = F - A @ v; e = M.energy(K, M.full(uh - v))
    row = dict(pcg_iter=k, alg_error=e, alg_error_rel=e / M.energy(K, M.full(uh)))
    for nm, op in (('M0 (MVY)', mvy), ('M0p (PRVW)', prvw), ('joint optimum', joint)):
        row[nm] = R.ps_norm(M, K, op(Rv)) / e
    res['rows'].append(row)
    print(f"PCG(ILU) iter {k}: algebraic error {e:.2e} (rel {row['alg_error_rel']:.1e}); effectivity M0 {row['M0 (MVY)']:.3f}, "
          f"M0p {row['M0p (PRVW)']:.3f}, joint {row['joint optimum']:.3f}", flush=True)
print(f"mesh: {res['coarse_elements']} coarse triangles, {res['n_fe_elements']} FE triangles, N = {res['N']} (PRVW p = 1: N_h = 3.5e4)")
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fidelity_poisson.json'), 'w'), indent=1)
