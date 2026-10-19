"""Does S_ij(mu) lie near a low-dimensional manifold?

If S(mu) ~ S0 + sum_r alpha_r(mu) S_r with small R, the expensive OA build can
be amortised: build R+1 bases offline, predict alpha(mu) online.

The test below uses an ORACLE: each sample's S is projected onto the POD basis
built from ALL samples including itself.  That is the best any predictor of
alpha(mu) could ever do, so it upper-bounds the idea.  If the oracle fails,
the idea fails.
"""
import numpy as np, scipy.sparse as sp, sys, time
sys.path.insert(0, '.')
import cluster_lib as C   # level 6, 40 samples, contrast 1e4

M = 30
A = C.A[:M]
keys = None
print(f"building {M} per-sample adapted bases at level {C.L} ...")
t0 = time.time(); banks = []
for n in range(M):
    S = C.build_basis(A[n]); banks.append(S)
    if keys is None: keys = sorted(S.keys())
print(f"  {time.time()-t0:.0f}s\n")

# union sparsity pattern per block, then one flat vector per sample
pat, off, tot = {}, {}, 0
for k in keys:
    U = banks[0][k].tocsr().copy(); U.data[:] = 1.0
    for b in banks[1:]:
        V = b[k].tocsr().copy(); V.data[:] = 1.0
        U = (U + V)
    U = U.tocoo()
    pat[k] = (U.row, U.col, U.shape)
    off[k] = (tot, tot + len(U.row)); tot += len(U.row)
print(f"union support across samples: {tot:,} entries "
      f"(single sample nnz {sum(v.nnz for v in banks[0].values()):,})")

X = np.zeros((M, tot))
for n, b in enumerate(banks):
    for k in keys:
        r, c, shp = pat[k]
        X[n, off[k][0]:off[k][1]] = np.asarray(b[k].tocsr()[r, c]).ravel()

Xm = X.mean(0); Xc = X - Xm
U_, s_, Vt = np.linalg.svd(Xc, full_matrices=False)
energy = np.cumsum(s_**2) / np.sum(s_**2)
print(f"\nPOD spectrum (mean removed), {M} snapshots:")
print("  rank :", "  ".join(f"{r:>7d}" for r in [1,2,3,5,8,12,20,M-1]))
print("  energy:", "  ".join(f"{energy[r-1]*100:>6.2f}%" for r in [1,2,3,5,8,12,20,M-1]))
for tgt in (0.99, 0.999, 0.9999):
    print(f"  rank for {tgt*100:g}% energy: {int(np.searchsorted(energy, tgt))+1}")

def S_from_vec(v):
    out = {}
    for k in keys:
        r, c, shp = pat[k]
        d = v[off[k][0]:off[k][1]]
        out[k] = sp.csr_matrix((d, (r, c)), shape=shp)
    return out

print(f"\nORACLE rank-R reconstruction used as preconditioner "
      f"(best any alpha(mu) predictor could do):")
print(f"{'R':>4}{'rel err in S':>15}{'PCG it mean':>13}{'max':>7}{'fail':>6}")
exact = [C.solve(A[n], banks[n])[0] for n in range(6)]
print(f"{'exact':>4}{0.0:>15.2e}{np.mean(exact):>13.1f}{max(exact):>7}{0:>6}")
for R in [1, 2, 4, 8, 16]:
    if R > M-1: break
    P = Vt[:R]
    Xr = Xm + (Xc @ P.T) @ P
    err = np.linalg.norm(Xr - X) / np.linalg.norm(X)
    its, fails = [], 0
    for n in range(6):
        k, info = C.solve(A[n], S_from_vec(Xr[n]), maxiter=3000)
        its.append(k); fails += (info != 0)
    print(f"{R:>4}{err:>15.2e}{np.mean(its):>13.1f}{max(its):>7}{fails:>6}")
