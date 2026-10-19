"""Math check: is the certificate's lambda_min a rigorous lower bound, or only a Lanczos estimate?

The guaranteed bound ||e||_A <= eta / sqrt(lambda_min(B A)) needs a LOWER bound on lambda_min. Lanczos (Ritz)
values lie inside the spectrum, so lmin_ritz >= lambda_min: on its own it is an estimate from ABOVE -- the
wrong side for a guarantee.  A rigorous check: with the transform matrix T (B = T D^-1 T^T),

    lambda_min(B A) >= sigma   <=>   T^T A T - sigma D  is positive semidefinite,

decided by one Cholesky factorisation (Sylvester).  Taking sigma = (1 - delta) lmin_ritz, success certifies the
bound with a (1 - delta)^-1/2 safety factor; failure at sigma = (1 + delta) lmin_ritz shows the Ritz value is sharp.
"""
import sys, time, numpy as np, scipy.linalg as sla
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from oa_lifting import NestedHierarchy, make_field, OATransform, ritz_extremes
from coarse_meshes import perforated_square


def transform_matrix(T):
    cols = []
    for k in T.levels:
        m = T.size(k)
        if m == 0: continue
        co = {j: np.zeros((T.size(j), m)) for j in T.levels}; co[k] = np.eye(m)
        cols.append(T.synth(co))
    return np.hstack(cols)


def lmin_is_at_least(A, T, sigma):
    Tm = transform_matrix(T)
    K = Tm.T @ (A @ Tm)
    D = sla.block_diag(*[(T.D[k].toarray() if hasattr(T.D[k], 'toarray') else T.D[k]) for k in T.levels if T.size(k)])
    try:
        sla.cholesky(0.5 * (K + K.T) - sigma * D, lower=True); return True
    except np.linalg.LinAlgError:
        return False


if __name__ == '__main__':
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    H = NestedHierarchy(L, coarse=perforated_square(n=9))
    rng = np.random.default_rng(0)
    print(f"perforated level {L}: {len(H.nodes[L])} unknowns, channels 1e4, Route V local")
    for rho in (2, 4):
        for s in range(3):
            a = make_field(H, 'channels', 1e4, rng); A = H.A_fine(a).tocsc()
            t0 = time.time()
            T = OATransform(H, A, 'local', rho=rho, route='V', dense_D_max=0)
            lmn, lmx = ritz_extremes(A, T.precond, n_iter=40)
            lo = lmin_is_at_least(A, T, (1 - 1e-3) * lmn)
            hi = lmin_is_at_least(A, T, (1 + 1e-3) * lmn)
            print(f"  rho {rho} sample {s}: Ritz lmin {lmn:.6f}; PSD at 0.999*lmin: {lo}; at 1.001*lmin: {hi}  "
                  f"-> {'certified, Ritz value sharp' if lo and not hi else 'CHECK'}  ({time.time() - t0:.1f}s)", flush=True)
