"""4 Oct (user's question): in the energy-norm operator-adapted basis, how many basis functions really matter for the
correction e = u - u_c, and can they be found BEFORE solving?

Exact route-V basis on L3 -> L0 (Phat_k = -A_FF^-1 A_FC, levels block-diagonal): e = W c, and by scale-orthogonality
    ||e||_A^2 = c_0^T A_0 c_0 + sum_k d_k^T A_FF^(k) d_k.
(a) ORACLE: keep the coarse block and the N detail coefficients with the largest d_i^2 (A_FF)_ii; exact energy error of
    the truncation (dropped coefficients, per level d_drop^T A_FF d_drop).
(b) A PRIORI INDICATOR from the residual only: the dual analysis g = W^T r gives d_k = A_FF^-1 g_F; since D^-1 A_FF is
    well conditioned (route V, kappa ~ 9-16 for P1) the indicator iota_i = g_i^2 / (A_FF)_ii predicts d_i^2 (A_FF)_ii.
    Select top-N by iota, same exact error.
Darcy perforated L3, geometric u_c (k0 2), 10 samples each: channels 1e4, barriers 1e-4 (--load no-low)."""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, splu
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square

L = 3; H = NestedHierarchy(L, coarse=perforated_square(n=9))
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]
P2 = H.P[3].tocsr()

def exact_levels(A):
    Ak = A.toarray(); out = []
    for k in range(L, 0, -1):
        nc = H.P[k].shape[1]; AFF = Ak[nc:, nc:]; AFC = Ak[nc:, :nc]
        X = -np.linalg.solve(AFF, AFC); out.append((nc, X, AFF))
        Phi = np.vstack([np.eye(nc), X]); Ak = Phi.T @ Ak @ Phi
    return out, Ak

FR = (0.01, 0.03, 0.1, 0.2)
for title, contrast, nolow in (('channels 1e4', 1e4, False), ('barriers 1e-4, --load no-low', 1e-4, True)):
    rng = np.random.default_rng(1000); res = {('oracle', f): [] for f in FR}; res.update({('indicator', f): [] for f in FR})
    hit = []
    for s in range(10):
        a = field_channels(H, contrast, rng); A = H.A_fine(a).tocsr(); F = load(a >= 0.5) if nolow else H.F
        u = spsolve(A.tocsc(), F); uc = P2 @ spsolve((P2.T @ A @ P2).tocsc(), P2.T @ F); e = u - uc; r = F - A @ uc
        levs, A0 = exact_levels(A)
        # primal analysis of e (coefficients) and dual analysis of r (indicator)
        x = e.copy(); g = r.copy(); D, IOTA, IMP, BLK = [], [], [], []
        for nc, X, AFF in levs:
            d = x[nc:] - X @ x[:nc]; x = x[:nc]
            gF = g[nc:]; g = g[:nc] + X.T @ gF
            dd = np.diag(AFF); D.append(d); IMP.append(d * d * dd); IOTA.append(gF * gF / dd); BLK.append(AFF)
        Etot = e @ (A @ e); E0 = x @ (A0 @ x)
        imp = np.concatenate(IMP); iota = np.concatenate(IOTA); n_det = len(imp)
        offs = np.cumsum([0] + [len(v) for v in D])
        def trunc_err(keep_mask):
            err = 0.0
            for j, (d, AFF) in enumerate(zip(D, BLK)):
                drop = ~keep_mask[offs[j]:offs[j + 1]]; dz = np.where(drop, d, 0.0); err += dz @ (AFF @ dz)
            return np.sqrt(err / Etot)
        for f in FR:
            N = int(round(f * n_det))
            for name, score in (('oracle', imp), ('indicator', iota)):
                m = np.zeros(n_det, bool); m[np.argsort(-score)[:N]] = True; res[(name, f)].append(trunc_err(m))
        top = int(0.1 * n_det); hit.append(len(np.intersect1d(np.argsort(-imp)[:top], np.argsort(-iota)[:top])) / top)
    print(f"\n=== {title}: {n_det} detail coefficients (L3 -> L0, exact route-V basis), coarse block kept; "
          f"energy error of the truncated correction, relative to ||e||_A, mean of 10 samples ===")
    print(f"{'kept fraction':<16}" + ''.join(f"{f:>10.0%}" for f in FR))
    for name in ('oracle', 'indicator'):
        print(f"{name + ' top-N':<16}" + ''.join(f"{np.mean(res[(name, f)]):>10.3f}" for f in FR))
    print(f"overlap of the top-10% sets (indicator vs oracle): {np.mean(hit):.2f}")
