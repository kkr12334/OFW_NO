"""Representability check for the 2x2-BLOCK oa-ll-op (plane elasticity), 3 Oct.

The vector version replaces the scalar weights a_ij / a_ii by the nodal blocks W_ij = D_i^-1 A_ij, D_i = A_ii (2 x 2),
and the input D^-1 r by the block-Jacobi residual z_i = D_i^-1 r_i; every node's latent is read as C/2 vectors.

Checked here (porous plate, clamped outer edge, traction-free pores, body force (0, -1), stiff bands; Galerkin and
exact operator-adapted coarse solves):
  1. the error equation A e = r holds for the coarse solve (Galerkin projection, so Pi_H e = 0);
  2. Proposition (block form): with the MLPs at zero the network output is EXACTLY the nu-step weighted BLOCK-Jacobi
     iterate x_{k+1} = x_k + omega D^-1 (r - A x_k), x_0 = 0, with ONE parameter set for every coefficient field;
  3. why block and not point Jacobi:
     (a) lambda_max(D^-1 A) <= 3 for every coefficient (P1 triangles: A = sum_e A_e and each SPSD A_e <= 3 x its own
         nodal block diagonal, by Cauchy-Schwarz over its 3 nodes), so omega = 2/3 is a coefficient-independent
         convergent damping; the point-Jacobi bound is 6 (6 dofs per element), i.e. omega <= 1/3;
     (b) frame invariance: rotating the displacement components by Q maps the block iterate to Q^T x exactly; the
         point-Jacobi iterate is not covariant (it depends on the choice of axes);
  4. the ceiling of the class on actual samples: ||e - x_nu||_A / ||e||_A, block (omega 2/3) vs point (omega 1/3).
"""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore', category=UserWarning)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, splu, eigsh
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, fibre_composite
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant

torch.set_default_dtype(torch.float64)
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))


def setup(L):
    H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer)
    return H, VectorHierarchy(H)


def galerkin_coarse(V, A, k0):
    L = V.L; Pk = None
    for k in range(k0 + 1, L + 1):
        Pk = V.P[k] if Pk is None else V.P[k] @ Pk
    Pk = Pk.tocsr()
    return Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ V.F)


def adapted_coarse(V, A, rho=0):                              # Route V coarse space, one level (k0 = L - 1)
    L = V.L; A = A.tocsr(); Ek = V.E[L]; ncd = V.P[L].shape[1]   # rho = 0: exact; rho > 0: local patches (driver's)
    A_FF = (Ek.T @ A @ Ek).tocsc(); A_FC = (Ek.T @ A)[:, :ncd].tocsc()
    if rho:
        from oa_lifting import _local_C
        Ph = _local_C(A_FF, A_FC, V.new_coords(L), V.level_coords(L - 1), rho * V.h[L - 1])
        Ph = Ph.toarray() if sp.issparse(Ph) else Ph
    else:
        Ph = -splu(A_FF).solve(A_FC.toarray())
    Phi = sp.vstack([sp.identity(ncd, format='csr'), sp.csr_matrix(Ph)]).tocsr()
    return Phi @ spsolve((Phi.T @ A @ Phi).tocsc(), Phi.T @ V.F)


def block_diag_inv(A, nI):
    A = A.tocsr(); i = np.arange(nI)
    Db = np.stack([np.stack([np.asarray(A[2 * i + p, 2 * i + q]).ravel() for q in (0, 1)], -1) for p in (0, 1)], 1)
    return np.linalg.inv(Db), Db


def bd_matrix(Bk):                                            # (n, 2, 2) -> sparse block-diagonal
    n = len(Bk); i = np.arange(n)
    rows = (2 * i[:, None, None] + np.arange(2)[None, :, None]) + 0 * np.arange(2)[None, None, :]
    cols = (2 * i[:, None, None] + np.arange(2)[None, None, :]) + 0 * np.arange(2)[None, :, None]
    return sp.csr_matrix((np.asarray(Bk).ravel(), (rows.ravel(), cols.ravel())), shape=(2 * n, 2 * n))


def jacobi(A, r, Minv, omega, nu):
    x = np.zeros_like(r); out = {}
    for k in range(1, nu + 1):
        x = x + omega * (Minv @ (r - A @ x))
        out[k] = x.copy()
    return out


en = lambda A, v: np.sqrt(v @ (A @ v))
rng = np.random.default_rng(0)

# ------------------------------------------------------------------------------- 1, 2: representability at L2
L, k0, nu, omega = 2, 1, 4, 2 / 3
H, V = setup(L); nI = len(H.nodes[L])
E = field_channels(H, 1e2, rng)
A = V.A_fine(E).tocsr(); F = V.F
u = spsolve(A.tocsc(), F); uc = galerkin_coarse(V, A, k0)
e = u - uc; r = F - A @ uc
print(f"L{L}: {2 * nI} dofs, bands E_f/E_m 1e2, Galerkin coarse solve to level {k0}")
print(f"  ||A e - r|| / ||r|| = {np.linalg.norm(A @ e - r) / np.linalg.norm(r):.1e}   (error equation holds)")
Dinv, Db = block_diag_inv(A, nI)
z = np.einsum('ipq,iq->ip', Dinv, r.reshape(nI, 2))

MG = MeshHierarchyGeometry(H, 'cpu')
for kk, v in list(MG.fine.items()):
    if torch.is_tensor(v) and v.is_floating_point(): MG.fine[kk] = v.double()
for Lv in MG.levels:
    for kk, v in list(Lv.items()):
        if torch.is_tensor(v) and v.is_floating_point(): Lv[kk] = v.double()
C, cin = 8, 6                                                 # inputs: log E, x, y, flag, z_x, z_y
m = MeshLiftingNO(MG, cin, width=C, out_channels=2, op_block=2, **variant('oa-ll-op')).double()
src, dst = MG.fine['src'].numpy(), MG.fine['dst'].numpy()
Ablk = np.stack([np.stack([np.asarray(A[2 * dst + p, 2 * src + q]).ravel() for q in (0, 1)], -1) for p in (0, 1)], 1)
W = torch.tensor(np.einsum('epq,eqr->epr', Dinv[dst], Ablk))[None]
m.set_op(W)
with torch.no_grad():
    zero = lambda lin: (lin.weight.zero_(), lin.bias.zero_() if lin.bias is not None else None)
    zero(m.enc[2]); zero(m.att.out_proj); zero(m.cm[2]); zero(m.dec[2])
    for f in m.evo: zero(f[2])
    for f in m.post_upd: zero(f[2])
    # latent vector 0 = (ch 0, ch 1) = iterate x (x_0 = 0); vector 1 = (ch 2, ch 3) = z = D^-1 r
    m.enc_lin.weight.zero_(); m.enc_lin.weight[2, 4] = 1.0; m.enc_lin.weight[3, 5] = 1.0
    m.dec_lin.weight.zero_(); m.dec_lin.weight[0, 0] = 1.0; m.dec_lin.weight[1, 1] = 1.0
    for Vl, Lin in zip(m.post_V, m.post_L):
        Vl.weight.zero_(); Vl.weight[0, 0] = 1.0; Vl.weight[1, 1] = 1.0          # vector 0 of V s = x
        Lin.weight.zero_()
        for c in (0, 1):                                      # x_c += omega (z_c - x_c - [sum_j W_ij x_j]_c)
            Lin.weight[c, 2 + c] = omega; Lin.weight[c, c] = -omega; Lin.weight[c, C + c] = -omega
x_in = torch.zeros(1, nI, cin); x_in[0, :, 4:6] = torch.tensor(z)
x_in[0, :, :4] = torch.tensor(rng.standard_normal((nI, 4)))    # other inputs arbitrary: must not matter
with torch.no_grad():
    out = m(x_in)[0].numpy().ravel()                          # (nI, 2) -> interleaved dofs
Mb = bd_matrix(Dinv)
xk = jacobi(A, r, Mb, omega, nu)[nu]
print(f"  network (op_block 2) vs {nu}-step block-Jacobi iterate: max |diff| = {np.abs(out - xk).max():.1e}  "
      f"(|x| ~ {np.abs(xk).max():.1e})")

# ------------------------------------------------------------------------------- 3b: frame invariance
th = 0.7; Q = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
R = bd_matrix(np.broadcast_to(Q, (nI, 2, 2)))                        # u = R u'  (components in the rotated frame)
Ar = (R.T @ A @ R).tocsr(); rr = R.T @ r
Dinv_r, _ = block_diag_inv(Ar, nI)
xb_r = jacobi(Ar, rr, bd_matrix(Dinv_r), omega, nu)[nu]
xp = jacobi(A, r, sp.diags(1 / A.diagonal()), 1 / 3, nu)[nu]
xp_r = jacobi(Ar, rr, sp.diags(1 / Ar.diagonal()), 1 / 3, nu)[nu]
print(f"  frame rotation by {th} rad: block iterate  |R^T x - x'| / |x| = "
      f"{np.linalg.norm(R.T @ xk - xb_r) / np.linalg.norm(xk):.1e};  point iterate {np.linalg.norm(R.T @ xp - xp_r) / np.linalg.norm(xp):.1e}")

# ------------------------------------------------------------------------------- 3a: lambda_max bounds
print("\n  lambda_max of D^-1 A (generalised eigenproblem), block bound 3, point bound 6:")
for name, Ef in [('bands 1e2', field_channels(H, 1e2, rng)), ('bands 1e4', field_channels(H, 1e4, rng)),
                 ('fibres 1e3', fibre_composite(H, 1e3, rng)), ('homogeneous', np.ones(H.n_elem))]:
    A_ = V.A_fine(Ef).tocsr(); Di, Db_ = block_diag_inv(A_, nI)
    lb = eigsh(A_, k=1, M=bd_matrix(Db_).tocsc(), which='LA', return_eigenvectors=False)[0]
    lp = eigsh(A_, k=1, M=sp.diags(A_.diagonal()).tocsc(), which='LA', return_eigenvectors=False)[0]
    print(f"    {name:<12} block {lb:.3f}   point {lp:.3f}")

# ------------------------------------------------------------------------------- 4: ceilings on samples
def ceilings(L, k0, n_s, contrast, coarse, rho=0):
    H, V = setup(L); nI = len(H.nodes[L]); res = {'block': [], 'point': []}; t0 = time.time()
    rg = np.random.default_rng(2000)                            # the driver's data stream (bands)
    for s in range(n_s):
        E_ = field_channels(H, contrast, rg)
        A_ = V.A_fine(E_).tocsr(); u_ = spsolve(A_.tocsc(), V.F)
        uc_ = galerkin_coarse(V, A_, k0) if coarse == 'galerkin' else adapted_coarse(V, A_, rho)
        e_ = u_ - uc_; r_ = V.F - A_ @ uc_
        Di, _ = block_diag_inv(A_, nI)
        for kind, Mi, om in (('block', bd_matrix(Di), 2 / 3), ('point', sp.diags(1 / A_.diagonal()), 1 / 3)):
            it = jacobi(A_, r_, Mi, om, 50)
            res[kind].append([en(A_, e_ - it[n]) / en(A_, e_) for n in (1, 4, 10, 50)])
        res.setdefault('coarse', []).append(np.linalg.norm(e_) / np.linalg.norm(u_))
    print(f"    L{L} k0 {k0} {coarse + (f' rho {rho}' if rho else ''):<14} bands {contrast:.0e} ({n_s} samples, coarse alone rel-L2 "
          f"{np.mean(res['coarse']):.3f}, {time.time() - t0:.0f}s)")
    for kind in ('block', 'point'):
        v = np.mean(res[kind], 0)
        print(f"      {kind:<5} Jacobi (omega {'2/3' if kind == 'block' else '1/3'}): nu 1 {v[0]:.3f}  nu 4 {v[1]:.3f}  "
              f"nu 10 {v[2]:.3f}  nu 50 {v[3]:.3f}")

print("\n  ceiling of the class ||e - x_nu||_A / ||e||_A (mean over samples):")
ceilings(2, 1, 10, 1e2, 'galerkin')
ceilings(2, 1, 10, 1e2, 'adapted')
ceilings(3, 2, 4, 1e2, 'galerkin')
ceilings(3, 2, 2, 1e2, 'adapted')
ceilings(3, 2, 4, 1e2, 'adapted', rho=3)
