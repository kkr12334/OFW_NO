"""5 Oct (night, Astra's review of the direction assessment, point 1): a larger PILOT of the classical bar, with
  * islands classified from the COEFFICIENT GEOMETRY (conductive elements a >= 0.5, connected components without a Dirichlet
    node), independently of the near-kernel detector (whose count is reported next to it);
  * the SLOW MODES of the balanced operator-adapted cycle inspected instead of assumed: for the 3 lowest eigenvalues of
    A^1/2 B A^1/2 the error mode x = L^-T w is compared with conductive-component indicators -- "alignment" =
    (x . 1_S)^2 / (|S| |x|^2), the squared Euclidean cosine between x and the indicator of S, maximised over components S
    (6 Oct, Astra: it mixes variation inside S with mass outside S; it is NOT a measure of constancy inside S -- see
    check_enrichment_quality.py part 1 for the norm fraction on S and the within-S variation), whether that component
    touches the Dirichlet boundary and with how many Dirichlet nodes, and the mode's Rayleigh quotient x^T A x / x^T D x;
  * PCG iterations for BOTH loads, f = 1 and the no-low load of the planned barrier experiments (labelled).
L3 perforated, barriers 1e-4 samples 300-311 and channels 1e4 samples 300-305 (rng 1000).  Pilot evidence on 18 operators,
not a regime-wide bound."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.linalg import cholesky, eigh, solve, solve_triangular
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis

L, NU, QC = 3, 4, 10
H = NestedHierarchy(L, coarse=perforated_square(n=9))
nl = 0; k_ = L
while k_ >= 1 and nl < 4 and len(H.nodes[k_ - 1]) >= 16: nl += 1; k_ -= 1
DIR = np.setdiff1d(np.unique(H.elem[H.keep]), H.nodes[L])            # Dirichlet nodes (global ids)
posL = -np.ones(len(H.coords), dtype=np.int64); posL[H.nodes[L]] = np.arange(len(H.nodes[L]))
xy = H.coords[H.elem]; area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))


def load_nolow(a):
    m = H.keep & (np.asarray(a).ravel() >= 0.5); F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[m, j], area[m] / 3.0)
    return F[H.nodes[L]]


def components(a):
    """conductive components: list of (unknown indices, number of Dirichlet nodes touched)."""
    cond = H.keep & (np.asarray(a).ravel() >= 0.5); el = H.elem[cond]
    M = sp.csr_matrix((np.ones(3 * len(el)), (np.repeat(np.arange(len(el)), 3), el.ravel())), shape=(len(el), len(H.coords)))
    G = (M.T @ M).tocsr(); used = np.unique(el.ravel())
    n_, lab = connected_components(G[used][:, used], directed=False)
    out = []
    for c in range(n_):
        nodes = used[lab == c]; idx = posL[nodes]; idx = idx[idx >= 0]
        out.append((idx, int(np.isin(nodes, DIR).sum())))
    return out


def lam_max(Ak, its=50):
    dd = Ak.diagonal(); v = np.random.default_rng(0).standard_normal(Ak.shape[0]); lam = 1.0
    for _ in range(its):
        w = (Ak @ v) / dd; lam = np.linalg.norm(w) / np.linalg.norm(v); v = w / np.linalg.norm(w)
    return 1.05 * lam


def vcycle_matrix(levs, A0):
    lm = [lam_max(Ak) for Ak, _ in levs]
    def cheb(Ak, rhs, x, lmax):
        dinv = (1 / Ak.diagonal())[:, None]; lo, hi = lmax / QC, lmax; th, de = (hi + lo) / 2, (hi - lo) / 2
        s1 = th / de; r0 = 1 / s1; d = dinv * (rhs - Ak @ x) / th; x = x + d
        for _ in range(NU - 1):
            r1 = 1 / (2 * s1 - r0); d = r1 * r0 * d + 2 * r1 / de * dinv * (rhs - Ak @ x); x = x + d; r0 = r1
        return x
    A0d = A0.toarray()
    def cyc(l, rhs):
        Ak, X = levs[l]; nc = X.shape[1]
        x = cheb(Ak, rhs, np.zeros_like(rhs), lm[l]); r = rhs - Ak @ x
        rc = r[:nc] + X.T @ r[nc:]
        ec = cyc(l + 1, rc) if l + 1 < len(levs) else solve(A0d, rc, assume_a='pos')
        return cheb(Ak, rhs, x + np.vstack([ec, X @ ec]), lm[l])
    return cyc(0, np.eye(levs[0][0].shape[0]))


def pcg_its(A, B, b, Lc, tol=1e-8, maxit=300):
    u = solve_triangular(Lc.T, solve_triangular(Lc, b, lower=True), lower=False); nu_ = np.sqrt(u @ (A @ u))
    x = np.zeros_like(b); r = b.copy(); z = B @ r; p = z.copy(); rz = r @ z
    for it in range(1, maxit + 1):
        Ap = A @ p; al = rz / (p @ Ap); x += al * p; r -= al * Ap; e = u - x
        if np.sqrt(e @ (A @ e)) <= tol * nu_: return it
        z = B @ r; rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
    return maxit


if __name__ == '__main__':
    rows = []
    print(f"L{L}: {len(H.F)} unknowns, {nl} levels; B = symmetric V({NU},{NU}) Chebyshev, operator-adapted transfers; "
          f"balanced with the detected Z where the detector returns vectors")
    for cname, contrast, ids in (('barriers 1e-4', 1e-4, range(300, 312)), ('channels 1e4', 1e4, range(300, 306))):
        rng = np.random.default_rng(1000); a_all = [field_channels(H, contrast, rng) for _ in range(400)]
        for i in ids:
            a = a_all[i]; A = H.A_fine(a).tocsr(); Ad = A.toarray(); d = A.diagonal(); Lc = cholesky(Ad, lower=True)
            comps = components(a); geo_isl = sum(1 for _, nd in comps if nd == 0)
            Z = near_kernel_basis(A); M = vcycle_matrix(*oa_hierarchy(H, A, nl, 4))
            if Z.shape[1]:
                C = Z @ np.linalg.solve(Z.T @ Ad @ Z, Z.T); Q = np.eye(len(d)) - C @ Ad; B = C + Q @ M @ Q.T
            else:
                B = M
            S = Lc.T @ B @ Lc; lam, W = eigh(0.5 * (S + S.T))
            lam_m = np.linalg.eigvalsh(0.5 * (Lc.T @ M @ Lc + (Lc.T @ M @ Lc).T))
            its1 = pcg_its(A, B, H.F, Lc); its2 = pcg_its(A, B, load_nolow(a), Lc) if contrast < 1 else its1
            modes = []
            for j in range(3):
                x = solve_triangular(Lc.T, W[:, j], lower=False); x2 = x @ x
                al = [((x[idx].sum()) ** 2 / (len(idx) * x2), nd, len(idx)) for idx, nd in comps if len(idx)]
                best = max(al) if al else (0.0, -1, 0)
                modes.append((lam[j], best[0], best[1], best[2], (x @ (A @ x)) / (x @ (d * x))))
            rows.append(dict(cls=cname, i=i, geo=geo_isl, det=Z.shape[1], delta=1 - lam[0], kappa=lam[-1] / lam[0],
                             delta_M=1 - lam_m[0], its1=its1, its2=its2, modes=modes))
            print(f"  {cname} {i}: geometric islands {geo_isl}, detector vectors {Z.shape[1]} | M_oa delta {1 - lam_m[0]:.3f} -> "
                  f"B delta {1 - lam[0]:.3f} kappa {lam[-1] / lam[0]:.2f} | PCG its f=1 {its1}" + (f", no-low {its2}" if contrast < 1 else ''), flush=True)
            for (lj, alg, nd, sz, rq) in modes:
                print(f"      slow mode lambda {lj:.3f}: max squared cosine with a component indicator {alg:.2f} "
                      f"(component of {sz} unknowns, touching {nd} Dirichlet nodes); Rayleigh x^T A x / x^T D x {rq:.2e}", flush=True)

    print("\nSUMMARY (pilot, per class): delta of B = 1 - lambda_min (median / worst), PCG its (median / worst)")
    for key, sel in (('channels', lambda r: r['cls'].startswith('channels')),
                     ('barriers, geometric island', lambda r: r['cls'].startswith('barriers') and r['geo'] > 0),
                     ('barriers, no geometric island', lambda r: r['cls'].startswith('barriers') and r['geo'] == 0)):
        R = [r for r in rows if sel(r)]
        if not R: continue
        dl = np.array([r['delta'] for r in R]); dm = np.array([r['delta_M'] for r in R])
        i1 = np.array([r['its1'] for r in R]); i2 = np.array([r['its2'] for r in R])
        al = np.array([r['modes'][0][1] for r in R]); nd = [r['modes'][0][2] for r in R]
        print(f"  {key:<30} n={len(R):2d}  M_oa delta median {np.median(dm):.3f} worst {dm.max():.3f} | balanced delta median "
              f"{np.median(dl):.3f} worst {dl.max():.3f} | PCG f=1 {np.median(i1):.0f}/{i1.max()}, no-low {np.median(i2):.0f}/{i2.max()} | "
              f"slowest mode: alignment median {np.median(al):.2f}, Dirichlet nodes of its component {nd}")
    print("  detector vs geometry (barriers): " + ', '.join(f"{r['i']}: geo {r['geo']} / det {r['det']}" for r in rows if r['cls'].startswith('barriers')))
