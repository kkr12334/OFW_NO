"""Why does a learned one-pass corrector beat every classical one-pass smoother (stock-take §9)?  Hypothesis: the
corrections e = u - u_c over the coefficient DISTRIBUTION lie close to a low-dimensional space (small Kolmogorov
n-width), which a classical, distribution-agnostic smoother cannot exploit but a learned map can.  Measured: POD of the
training corrections (300 samples), then the ORACLE projection error of TEST corrections onto the top-m modes, in each
test sample's own energy norm relative to ||u||_A (the metric of the tables).  Darcy L3, k0 2, channels 1e4 and the
barrier family 1e-4; a GRF field as a smooth-coefficient control.
Result: for channels / barriers the oracle top-300 POD space removes NOTHING in the energy norm (a fixed mode costs
enormous energy inside each sample's own stiff channels, so its best coefficient is ~0) -- the correction set has no
useful linear structure in the norm that matters; the learned advantage is a nonlinear, coefficient-adaptive map,
not a reduced basis.  (L2 projection does reduce, which is why L2 metrics can hide this.)"""
import sys, os, warnings, numpy as np, scipy.linalg as sla
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels, make_field
from coarse_meshes import perforated_square
L, k0 = 3, 2
H = NestedHierarchy(L, coarse=perforated_square(n=9)); F = H.F
Pk = None
for k in range(k0 + 1, L + 1):
    Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr(); PkT = Pk.T.tocsr()
for name, gen in (('channels 1e4', lambda r: field_channels(H, 1e4, r)), ('barriers 1e-4', lambda r: field_channels(H, 1e-4, r)),
                  ('grf (make_field)', lambda r: make_field(H, 'grf', 1e4, r))):
    rng = np.random.default_rng(1000); E, As, Us = [], [], []
    for s in range(400):
        A = H.A_fine(gen(rng)).tocsr(); u = spsolve(A.tocsc(), F)
        uc = Pk @ factorized((PkT @ A @ Pk).tocsc())(PkT @ F); E.append(u - uc); As.append(A); Us.append(u)
    E = np.array(E); tr, te = np.arange(300), np.arange(300, 400)
    U, S, Vt = sla.svd(E[tr], full_matrices=False)               # L2 POD of the training corrections
    en = lambda A, v: np.sqrt(v @ (A @ v))
    base = np.mean([en(As[i], E[i]) / en(As[i], Us[i]) for i in te]); bl2 = np.mean([np.linalg.norm(E[i]) / np.linalg.norm(Us[i]) for i in te])
    row = []
    for m in (10, 50, 100, 200, 300):
        Q = Vt[:m].T
        errs = []
        for i in te:                                              # best approximation in the sample's OWN energy norm
            A = As[i]; AQ = A @ Q; c = np.linalg.solve(Q.T @ AQ, AQ.T @ E[i])
            errs.append(en(A, E[i] - Q @ c) / en(A, Us[i]))
        l2 = np.mean([np.linalg.norm(E[i] - Q @ (Q.T @ E[i])) / np.linalg.norm(Us[i]) for i in te])
        row.append(f"m {m}: {np.mean(errs):.3f} (L2 {l2:.3f})")
    print(f"{name:<18} coarse alone energy {base:.3f} (L2 {bl2:.3f});  oracle top-m POD (energy, rel. to ||u||_A): " + ', '.join(row), flush=True)
