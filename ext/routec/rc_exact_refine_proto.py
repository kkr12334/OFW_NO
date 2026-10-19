"""Prototype for deviation 1 (18 Oct 2026): iterative refinement of the reference solves with the residual accumulated in
extended precision (numpy longdouble). Prints the energy-norm reference error estimate after each step. No outcome computed."""
import os, sys, numpy as np
HERE = '/home/claude/ofw/ext/routec'; EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE): sys.path.insert(0, p)
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
import rc_fem as R
FAM, idx, c = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
SEED = {'channels': 2026101801, 'barriers': 2026101802}[FAM]
cm = perforated_square(n=9); H = NestedHierarchy(3, coarse=cm); M = R.Mesh(H, cm)
rng = np.random.default_rng(SEED); draws = [make_field(H, 'channels', 1e4 if FAM == 'channels' else 1e-4, rng) for _ in range(idx + 1)]
mask = draws[idx] != 1.0
K = np.where(mask, c if FAM == 'channels' else 1.0 / c, 1.0); fK = np.ones(len(K)) if FAM == 'channels' else (~mask).astype(float)
A = M.stiffness(K).tocsr(); lu = splu(A.tocsc())
c3 = M.fe.centroid; om = np.hypot(c3[:, 0] - 0.5, c3[:, 1] - 0.45) < 0.08
loads = dict(primal=M.load(fK), dual0=M.load(om / M.fe.area[om].sum()))
Ad = A.data.astype(np.longdouble); ip = A.indptr; ix = A.indices
def res_ld(b, u):
    p = Ad * u.astype(np.longdouble)[ix]
    Au = np.add.reduceat(p, ip[:-1]); Au[np.diff(ip) == 0] = 0
    return b.astype(np.longdouble) - Au
en = lambda x: float(np.sqrt(max(x @ (A @ x), 0.0)))
for nm, b in loads.items():
    u = lu.solve(b)
    for k in range(4):
        r = res_ld(b, u); d = lu.solve(r.astype(np.float64))
        print(f'{FAM} id {idx} c={c:.0e} {nm} step {k}: est reference energy error {en(d) / en(u):.2e}  (double-residual estimate {en(lu.solve(b - A @ u)) / en(u):.2e})')
        u = u + d
