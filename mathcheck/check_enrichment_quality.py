"""Enrichment quality on island / near-island samples (5 Oct night; revised 6 Oct after Astra's follow-up review).
L3 perforated, barriers 1e-4.  M = classical symmetric operator-adapted V(4,4) Chebyshev; B = C + Q M Q^T (balanced).
Pilot on 5 selected matrices -- evidence for these operators only.

Part 1 -- the slowest mode of B balanced with the detector's INDICATOR vectors, described correctly (Astra: the earlier
"alignment" (sum_I x)^2 / (|I| |x|^2) is a global squared cosine with the indicator; it mixes variation inside the island with
mass outside it).  Reported separately: the fraction of ||x||^2 on the island, on its 4-ring neighbourhood outside it, and
elsewhere; and inside the island the coefficient of variation std / |mean|.

Part 2 -- enrichment variants, delta = 1 - lambda_min(A^1/2 B A^1/2) to 4 decimals and kappa:
  indicator (detector, A only) | geometric indicator |
  A-ONLY extension: supports taken from the detector's vectors (no coefficient geometry), A-harmonic extension
    (value 1 on the support, A z = 0 elsewhere) restricted to r graph rings (r = 2, 4, 8) and global;
  geometry-selected extension (r = 4 and global) for comparison (sample 309: the detector returns nothing; the two weakly
    anchored components are geometry-selected).
For every extension: patch size (unknowns solved for), setup time of the local solves, ||z_r - z_global||_A / ||z_global||_A,
and the condition number of Z^T A Z (rank / conditioning of the enrichment basis).
The evaluated operator is the CLASSICAL balanced cycle, not the learned cycle with its residual-dependent output."""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.linalg import cholesky, eigh, solve_triangular
from scipy.sparse.linalg import spsolve
from oa_lifting import field_channels
from oa_vcycle import oa_hierarchy, near_kernel_basis
import check_slow_modes as CS

H, nl = CS.H, CS.nl


def spectrum(B, Lc):
    S = Lc.T @ B @ Lc; lam, W = eigh(0.5 * (S + S.T)); return lam, W


def balanced(Z, M, Ad):
    C = Z @ np.linalg.solve(Z.T @ Ad @ Z, Z.T); Q = np.eye(len(Ad)) - C @ Ad; return C + Q @ M @ Q.T


def rings(A, idx, r):
    G = (abs(A) > 0).astype(float).tocsr(); m = np.zeros(A.shape[0]); m[idx] = 1
    for _ in range(r): m = np.minimum(1, G @ m)
    return np.nonzero(m)[0]


def extension(A, idx, r=None):
    """A-harmonic extension of the constant 1 on idx; on r graph rings around idx (None: all unknowns).  Returns z, patch size, s."""
    n = A.shape[0]; z = np.zeros(n); z[idx] = 1.0
    R = np.setdiff1d(np.arange(n) if r is None else rings(A, idx, r), idx)
    t0 = time.perf_counter()
    if len(R): z[R] = spsolve(A[R][:, R].tocsc(), -(A[R][:, idx] @ np.ones(len(idx))))
    return z, len(R), time.perf_counter() - t0


rng = np.random.default_rng(1000); a_all = [field_channels(H, 1e-4, rng) for _ in range(400)]
print("Part 1: slowest mode of B balanced with the detector's indicator vectors")
rows2 = []
for i, kind in ((301, 'island'), (304, 'island'), (306, 'island'), (308, 'island'), (309, 'near-island')):
    a = a_all[i]; A = H.A_fine(a).tocsr(); Ad = A.toarray(); Lc = cholesky(Ad, lower=True); En = lambda v: np.sqrt(v @ (A @ v))
    M = CS.vcycle_matrix(*oa_hierarchy(H, A, nl, 4))
    comps = CS.components(a)
    geo_sel = [idx for idx, nd in comps if (nd == 0 if kind == 'island' else 0 < nd <= 4) and len(idx)]
    Zd = near_kernel_basis(A); det_sel = [np.nonzero(Zd[:, j] > 0.5)[0] for j in range(Zd.shape[1])]
    if Zd.shape[1]:
        lam, W = spectrum(balanced(Zd, M, Ad), Lc)
        x = solve_triangular(Lc.T, W[:, 0], lower=False); x2 = x @ x
        for idx in det_sel:
            nb = np.setdiff1d(rings(A, idx, 4), idx)
            f_in, f_nb = (x[idx] @ x[idx]) / x2, (x[nb] @ x[nb]) / x2
            cv = np.std(x[idx]) / max(abs(np.mean(x[idx])), 1e-300)
            print(f"  {i} ({kind}): lambda_min {lam[0]:.4f}; ||x||^2 on the island ({len(idx)} unknowns) {f_in:.3f}, on its 4-ring "
                  f"neighbourhood {f_nb:.3f}, elsewhere {1 - f_in - f_nb:.3f}; inside the island std/|mean| {cv:.3f}", flush=True)
    variants = [('none (M alone)', None, None)]
    if Zd.shape[1]: variants.append(('indicator (detector, A only)', Zd, None))
    variants.append(('indicator (geometric)', np.column_stack([np.isin(np.arange(A.shape[0]), idx).astype(float) for idx in geo_sel]), None))
    for lab, sel in ((('A-only ext', det_sel) if det_sel else (None, None)), ('geometric ext', geo_sel)):
        if lab is None: continue
        glob_ = [extension(A, idx) for idx in sel]
        for r in ((2, 4, 8) if lab == 'A-only ext' else (4,)) + (None,):
            ext = glob_ if r is None else [extension(A, idx, r) for idx in sel]
            Z = np.column_stack([e[0] for e in ext])
            info = dict(patch=sum(e[1] for e in ext), t_ms=1e3 * sum(e[2] for e in ext),
                        dz=max(En(e[0] - g[0]) / En(g[0]) for e, g in zip(ext, glob_)), condZ=np.linalg.cond(Z.T @ Ad @ Z))
            variants.append((f"{lab}, {'global' if r is None else f'{r} rings'}", Z, info))
    for name, Z, info in variants:
        lam, _ = spectrum(M if Z is None else balanced(Z, M, Ad), Lc)
        extra = '' if info is None else (f"  patch {info['patch']:>5} unknowns, local solves {info['t_ms']:6.1f} ms, "
                                         f"||z - z_global||_A/||z_global||_A {info['dz']:.1e}, cond(Z^T A Z) {info['condZ']:.1e}")
        rows2.append((i, kind, name, 1 - lam[0], lam[-1] / lam[0]))
        print(f"    {i} {kind:<11} {name:<30} #vec {0 if Z is None else Z.shape[1]}  delta {1 - lam[0]:.4f}  kappa {lam[-1] / lam[0]:7.3f}{extra}", flush=True)
