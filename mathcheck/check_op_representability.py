"""Representability check for oa-ll-op (operator-consistent smoother), 2 Oct.

Claim (Proposition, README 2 Oct): for every coefficient a and every nu, the oa-ll-op class with nu smoother steps
contains a parameter setting whose output is EXACTLY the nu-step weighted-Jacobi iterate on the error equation
    A e = r,   r = F - A u_c,   e = u_h - u_c,   x_0 = 0,   x_{k+1} = x_k + omega D^-1 (r - A x_k),
so  inf_theta ||e - N_theta||_A  <=  ||(I - omega D^-1 A)^nu e||_A   for every sample simultaneously.
Checked here with a random lognormal coefficient on the perforated level-2 mesh: network output == Jacobi iterate
(to float32 round-off), and the Jacobi energy-error ratio is printed (the guaranteed ceiling of the class).
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant

torch.set_default_dtype(torch.float64)
L, k0, nu, omega = 2, 1, 4, 2 / 3
H = NestedHierarchy(L, coarse=perforated_square(n=9))
rng = np.random.default_rng(0)
a = np.exp(2.0 * rng.standard_normal(len(H.elem)))                    # contrast ~ e^8 over the elements
A = H.A_fine(a).tocsr(); F = H.F
u = spsolve(A.tocsc(), F)
Pk = None
for k in range(k0 + 1, L + 1):
    Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr()
uc = Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F)
e = u - uc; r = F - A @ uc; dg = A.diagonal(); z = r / dg
print(f"||A e - r|| / ||r|| = {np.linalg.norm(A @ e - r) / np.linalg.norm(r):.1e}   (error equation holds)")

MG = MeshHierarchyGeometry(H, 'cpu')
for kk, v in list(MG.fine.items()):
    if torch.is_tensor(v) and v.is_floating_point(): MG.fine[kk] = v.double()
for Lv in MG.levels:
    for kk, v in list(Lv.items()):
        if torch.is_tensor(v) and v.is_floating_point(): Lv[kk] = v.double()
C, cin = 8, 5
m = MeshLiftingNO(MG, cin, width=C, **variant('oa-ll-op')).double()
src, dst = np.array(MG.fine["src"].numpy()), np.array(MG.fine["dst"].numpy())
w = torch.tensor(np.asarray(A[dst, src]).ravel() / dg[dst])[None]
m.set_op(w)
with torch.no_grad():
    zero = lambda lin: (lin.weight.zero_(), lin.bias.zero_() if lin.bias is not None else None)
    zero(m.enc[2]); zero(m.att.out_proj); zero(m.cm[2]); zero(m.dec[2])
    for f in m.evo: zero(f[2])
    for f in m.post_upd: zero(f[2])
    m.enc_lin.weight.zero_(); m.enc_lin.weight[1, 4] = 1.0           # latent ch.1 <- D^-1 r ; ch.0 = iterate x_0 = 0
    m.dec_lin.weight.zero_(); m.dec_lin.weight[0, 0] = 1.0           # output <- iterate
    for V, Lin in zip(m.post_V, m.post_L):
        V.weight.zero_(); V.weight[0, 0] = 1.0                       # sum_j w_ij x_j
        Lin.weight.zero_()
        Lin.weight[0, 1] = omega; Lin.weight[0, 0] = -omega; Lin.weight[0, C] = -omega   # x += w (z - x - sum w x)
x_in = torch.zeros(1, len(z), cin); x_in[0, :, 4] = torch.tensor(z)
x_in[0, :, :4] = torch.tensor(rng.standard_normal((len(z), 4)))     # other inputs arbitrary: must not matter
with torch.no_grad():
    out = m(x_in)[0].numpy()
xk = np.zeros_like(z)
for _ in range(nu):
    xk = xk + omega * (r - A @ xk) / dg
en = lambda v: np.sqrt(v @ (A @ v))
print(f"network vs Jacobi iterate: max |diff| = {np.abs(out - xk).max():.1e}  (|x| ~ {np.abs(xk).max():.1e})")
for n_ in (1, 4, 10, 50):
    y = np.zeros_like(z)
    for _ in range(n_): y = y + omega * (r - A @ y) / dg
    print(f"  ceiling of the class, nu={n_:>2}: ||e - x_nu||_A / ||e||_A = {en(e - y) / en(e):.3f}")
