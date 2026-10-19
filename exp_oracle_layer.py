"""Step 3 of the formulation's next-experiments list (section 10): the ORACLE layer.

One application of the layer  u_hat = T R T^T F  with R built from the EXACT
level blocks, then deliberately restricted to what a trained network could
represent.  This upper-bounds any learned R of that form -- if the oracle
fails, training cannot succeed; if it succeeds, it tells you the radius and
rank the network needs.  Same spirit as exp_pod_amortization.py.

Variants of R_k (the coarsest block R_0 = A_0^-1 is always exact -- it is tiny):
  exact            D_k^-1                                    -> must give u exactly
  diag             Jacobi only
  local r          Lambda^-1/2 [ (Jacobi-scaled D_k)^-1 truncated to |x_i-x_j| <= r h_k ] Lambda^-1/2
  local r + rank m the m lowest Jacobi-scaled eigenmodes kept EXACTLY, the rest
                   of the inverse truncated to radius r
                   (m = 'auto' uses the number of eigenvalues below lambda_max/20)
and, for the transform itself,
  T frozen / T local(rho) / T hb  with THEIR OWN exact level blocks
  (i.e. the best a block-diagonal layer can do when T is not adapted exactly).

Reported: relative energy error and relative L2 error against the direct solve,
for the fixed load f = 1 AND for random smooth loads (so the result is about
the operator, not one right-hand side).

usage:
  python exp_oracle_layer.py                      # level 5, all fields, 3 samples
  python exp_oracle_layer.py --level 6 --samples 2
"""
import argparse, sys
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.spatial import cKDTree
import scipy.sparse as sp
sys.path.insert(0, '.')
from oa_lifting import (NestedHierarchy, OATransform, make_field, jacobi_scaled,
                        energy_err, rel_l2)

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, default=5)
ap.add_argument('--fields', nargs='+', default=['blocks8', 'grf', 'channels'])
ap.add_argument('--contrast', type=float, default=1e4)
ap.add_argument('--samples', type=int, default=3)
ap.add_argument('--radii', nargs='+', type=float, default=[1, 2, 4, 8])
ap.add_argument('--ranks', nargs='+', default=['auto', '8', '32'])
ap.add_argument('--rhos', nargs='+', type=float, default=[2, 4])
ap.add_argument('--n-loads', type=int, default=4, help='extra random smooth loads')
ap.add_argument('--route', choices=['V','W'], default='V',
                help="V: operator in the predict (default); W: operator in the update")
ap.add_argument('--seed', type=int, default=7)
args = ap.parse_args()

H = NestedHierarchy(args.level)
L = args.level
A1 = H.A_fine(np.ones(H.n_elem))
rng = np.random.default_rng(args.seed)
xy = H.level_coords(L)


def smooth_loads(n):
    """random low-frequency loads: sums of sin(pi p x) sin(pi q y), p,q <= 6."""
    out = []
    for _ in range(n):
        f = np.zeros(len(xy))
        for _ in range(6):
            p, q = rng.integers(1, 7, 2)
            f += rng.standard_normal() * np.sin(np.pi * p * xy[:, 0]) * np.sin(np.pi * q * xy[:, 1])
        out.append(f * (1.0 / 2 ** L) ** 2)       # lumped-mass scaling, P1 on right triangles
    return out


def trunc_mask(k, r):
    pts = H.new_coords(k)
    tree = cKDTree(pts)
    M = tree.sparse_distance_matrix(tree, r * H.h[k] + 1e-12, output_type='coo_matrix')
    M = sp.coo_matrix((np.ones(M.nnz), (M.row, M.col)), shape=(len(pts), len(pts))).tocsr()
    M = (M + sp.eye(len(pts), format='csr')).tocsr()
    M.data[:] = 1.0          # self-distance may or may not be stored; never count it twice
    return M


_mask_cache = {}


def _eig(T, k):
    # cached ON the transform object (id() is reused after garbage collection)
    cache = T.__dict__.setdefault('_eig_cache', {})
    if k not in cache:
        Dj, d = jacobi_scaled(T.D[k])
        ev, V = np.linalg.eigh(Dj)
        cache[k] = (d, ev, V, (V / ev) @ V.T)
    return cache[k]


def build_R(T, r=None, m=0, diag=False):
    R = {0: np.linalg.inv(T.D[0]) if T.D[0].size else T.D[0]}
    info = []
    for k in range(1, L + 1):
        d, ev, V, Xinv = _eig(T, k)
        if diag:
            R[k] = sp.diags(d * d); continue
        if r is None and m == 0:
            R[k] = (d[:, None] * Xinv) * d[None, :]; continue
        mk = int((ev < ev[-1] / 20).sum()) if m == 'auto' else min(int(m), len(ev) - 1)
        info.append(mk)
        low = (V[:, :mk] / ev[:mk]) @ V[:, :mk].T if mk else 0.0
        bulk = Xinv - low if mk else Xinv
        if (k, r) not in _mask_cache: _mask_cache[(k, r)] = trunc_mask(k, r)
        mask = _mask_cache[(k, r)]
        bulk_t = mask.multiply(bulk).tocsr()
        X = bulk_t + (sp.csr_matrix(low) if mk else 0)
        R[k] = sp.diags(d) @ X @ sp.diags(d)
    return R, info


def evaluate(T, R, A, loads, sols):
    ee, ll = [], []
    for f, u in zip(loads, sols):
        uh = T.apply_R(f, R)
        ee.append(energy_err(A, u, uh)); ll.append(rel_l2(u, uh))
    return ee[0], ll[0], float(np.mean(ee[1:])) if len(ee) > 1 else np.nan


rows = []
for field in args.fields:
    for s in range(args.samples):
        a = make_field(H, field, args.contrast, rng)
        A = H.A_fine(a).tocsc()
        loads = [H.F] + smooth_loads(args.n_loads)
        sols = [spsolve(A, f) for f in loads]
        tag = f"{field} #{s}"
        T = OATransform(H, A, 'exact', route=args.route, dense_D_max=10 ** 9)
        res = []
        R, _ = build_R(T);             res.append(('exact R', *evaluate(T, R, A, loads, sols), ''))
        R, _ = build_R(T, diag=True);  res.append(('diag', *evaluate(T, R, A, loads, sols), ''))
        for r in args.radii:
            R, _ = build_R(T, r=r);    res.append((f'local r={r:g}', *evaluate(T, R, A, loads, sols), ''))
            for m in args.ranks:
                R, info = build_R(T, r=r, m=m)
                res.append((f'local r={r:g} + rank {m}', *evaluate(T, R, A, loads, sols),
                            f"ranks/level {info}"))
        for rho in args.rhos:
            Tl = OATransform(H, A, 'local', route=args.route, rho=rho, dense_D_max=10 ** 9)
            R, _ = build_R(Tl); res.append((f'T local rho={rho:g}, exact blocks', *evaluate(Tl, R, A, loads, sols), ''))
        for name, Tx in [('T frozen a=1, exact blocks', OATransform(H, A, 'frozen', route=args.route, A_ref=A1, dense_D_max=10 ** 9)),
                         ('T hb, exact blocks', OATransform(H, A, 'hb', route=args.route, dense_D_max=10 ** 9))]:
            R, _ = build_R(Tx); res.append((name, *evaluate(Tx, R, A, loads, sols), ''))

        print(f"\n=== {tag}  (level {L}, contrast {args.contrast:.0e})")
        print(f"  {'layer R':<38}{'energy err f=1':>15}{'L2 err f=1':>12}{'energy err, smooth f':>22}")
        for name, e, l2, es, note in res:
            print(f"  {name:<38}{e:>15.2e}{l2:>12.2e}{es:>22.2e}  {note}")
        rows.append((tag, res))

print("\n\nSUMMARY  energy error for f = 1 (mean over samples)")
names = [r[0] for r in rows[0][1]]
fields = sorted(set(t.split(' #')[0] for t, _ in rows), key=args.fields.index)
print(f"{'layer R':<38}" + "".join(f"{f:>12}" for f in fields))
for i, n in enumerate(names):
    vals = [np.mean([res[i][1] for t, res in rows if t.startswith(f + ' ')]) for f in fields]
    print(f"{n:<38}" + "".join(f"{v:>12.2e}" for v in vals))
