"""Stock-take checks (3 Oct): two statements the paper leans on, re-checked on OUR discretisation.

1. Poeschl-Teller: the linearisation of u_t = eps^2 u_xx + u - u^3 about the interface profile tanh(x / (sqrt 2 eps))
   has eigenvalues 0, 3/2 and a continuum from 2 (used to fix the Allen-Cahn smoother step).
2. Route V finest block: kappa(diag(A_FF)^-1 A_FF) on the P1 perforated mesh.  The foundations draft (v0.1) proved
   'kappa <= 3 + sqrt 3 independent of the coefficient' on Q1 dyadic grids.  On P1 the element-by-element argument
   gives a bound independent of the coefficient only if the coefficient is constant on each level-(L-1) element
   (the 4 children of a coarse triangle share it); otherwise the middle child's 3x3 midpoint matrix has the constants
   in its kernel and a stiff middle child over soft corner children can drive lambda_min to 0.  Measured here.
"""
import sys, os, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy

# 1 ---------------------------------------------------------------------------------------------------------------
eps, n = 0.015, 4000; x = np.linspace(-0.6, 0.6, n); h = x[1] - x[0]
phi = np.tanh(x / (np.sqrt(2) * eps))
T = sp.diags([np.full(n - 1, -1), np.full(n, 2), np.full(n - 1, -1)], [-1, 0, 1]) / h ** 2   # Dirichlet ends
Lop = (eps ** 2 * T + sp.diags(3 * phi ** 2 - 1)).tocsc()
ev = np.sort(eigsh(Lop, k=4, sigma=-0.1, which='LM', return_eigenvectors=False))
print(f"1. linearisation about the interface: lowest eigenvalues {np.round(ev, 4)}  (theory 0, 1.5, then >= 2)")

# 2 ---------------------------------------------------------------------------------------------------------------
def kappa_ff(A, nc, block=1):
    A = A.tocsr(); AFF = A[nc:, nc:]
    if block == 1: Dg = sp.diags(AFF.diagonal())
    else:
        m = AFF.shape[0] // 2; i = np.arange(m)
        r = np.concatenate([2 * i, 2 * i, 2 * i + 1, 2 * i + 1]); c = np.concatenate([2 * i, 2 * i + 1, 2 * i, 2 * i + 1])
        Dg = sp.csr_matrix((np.asarray(AFF[r, c]).ravel(), (r, c)), shape=AFF.shape)
    w = eigh(AFF.toarray(), Dg.toarray(), eigvals_only=True)
    return w[-1] / w[0], w[0], w[-1]

rng = np.random.default_rng(0)
for L in (2, 3):
    H = NestedHierarchy(L, coarse=perforated_square(n=9)); ne = len(H.elem); g = ne // 4
    assert ne % 4 == 0
    nc = len(H.nodes[L - 1])
    mid = np.zeros(ne, bool); mid[3::4] = True                    # child 4p+3 is the middle child of parent p
    fields = {
        'homogeneous': np.ones(ne),
        'constant on coarse elements, log-uniform 1e-4..1e4': np.repeat(10 ** rng.uniform(-4, 4, g), 4),
        'channels 1e4 (fine centroids; the benchmark)': field_channels(H, 1e4, np.random.default_rng(1)),
        'adversarial: middle child 1e4, corner children 1': np.where(mid, 1e4, 1.0),
        'adversarial: middle child 1, corner children 1e4': np.where(mid, 1.0, 1e4),
    }
    print(f"\n2. L{L} perforated (Dirichlet), finest Route V block A_FF ({len(H.nodes[L]) - nc} new nodes): "
          "kappa(diag^-1 A_FF), lambda_min, lambda_max")
    for name, a in fields.items():
        k, lo, hi = kappa_ff(H.A_fine(a), nc)
        print(f"   {name:<52} {k:9.2f}   {lo:.3e}  {hi:.3f}")
    if L == 2:
        outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
        He = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(He)
        nce = 2 * len(He.nodes[L - 1])
        print(f"   elasticity L{L}, 2x2 block diagonal of A_FF:")
        for name in ('homogeneous', 'constant on coarse elements, log-uniform 1e-4..1e4',
                     'adversarial: middle child 1e4, corner children 1'):
            k, lo, hi = kappa_ff(V.A_fine(fields[name]), nce, block=2)
            print(f"   {name:<52} {k:9.2f}   {lo:.3e}  {hi:.3f}")
