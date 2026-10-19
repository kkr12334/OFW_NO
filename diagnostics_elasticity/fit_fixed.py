"""Unit test: can the network represent ONE fixed correction field (homogeneous material)?
Full-batch on a single sample; reports the relative L2 / energy error of the CORRECTION u - u_c,
and where the unfitted part lives (HB level shares)."""
import numpy as np, torch, sys, argparse, time
sys.path.insert(0, '.')
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO
ap = argparse.ArgumentParser()
ap.add_argument('--problem', default='elast'); ap.add_argument('--L', type=int, default=2); ap.add_argument('--k0', type=int, default=1)
ap.add_argument('--steps', type=int, default=1500); ap.add_argument('--lr', type=float, default=1e-3)
ap.add_argument('--width', type=int, default=24); ap.add_argument('--output', default='latent')
ap.add_argument('--loss', default='mix'); ap.add_argument('--feat', default='base')
args = ap.parse_args(); torch.manual_seed(0)
L = args.L
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
if args.problem == 'elast':
    H = NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H); nc_out = 2
    A = V.A_fine(np.ones(H.n_elem)); F = V.F; P = V.P
else:
    H = NestedHierarchy(L, coarse=perforated_square(n=9)); nc_out = 1
    A = H.A_fine(np.ones(H.n_elem)); F = H.F; P = H.P
u = spsolve(A, F); Pk = None
for k in range(args.k0 + 1, L + 1): Pk = P[k] if Pk is None else P[k] @ Pk
Pk = Pk.tocsr(); uc = Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F); t = u - uc
nI = len(H.nodes[L]); s = np.sqrt(np.mean(t ** 2))
xy = H.coords[H.nodes[L]]; xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
flag = np.zeros(len(H.coords)); flag[H.neumann_nodes] = -1.0
feats = [np.zeros((nI, 1)), xy, flag[H.nodes[L]][:, None], (uc / s).reshape(nI, nc_out)]
if args.feat in ('resid', 'jres'):   # operator information: residual of the coarse solution (one matvec)
    r = F - A @ uc
    if args.feat == 'jres': r = r / A.diagonal()      # one Jacobi step: diag(A)^-1 r
    feats += [(r / np.sqrt(np.mean(r ** 2))).reshape(nI, nc_out)]
if args.feat == 'fourier':          # random Fourier features of position (tests spectral bias)
    B = np.random.default_rng(0).normal(0, 8, (2, 16)); feats += [np.sin(xy @ B), np.cos(xy @ B)]
X = torch.tensor(np.concatenate(feats, 1)[None], dtype=torch.float32)
T = torch.tensor((t / s).reshape(1, nI, nc_out), dtype=torch.float32)
At = torch.tensor(A.toarray(), dtype=torch.float32)
G = MeshHierarchyGeometry(H, 'cpu')
m = MeshLiftingNO(G, X.shape[-1], width=args.width, out_channels=nc_out, output=args.output)
opt = torch.optim.Adam(m.parameters(), lr=args.lr)
def errs(p):
    d = (p - T).reshape(-1); tt = T.reshape(-1)
    return torch.linalg.norm(d) / torch.linalg.norm(tt), torch.sqrt(d @ At @ d) / torch.sqrt(tt @ At @ tt)
t0 = time.time()
for it in range(args.steps + 1):
    p = m(X).reshape(1, nI, nc_out); rl, en = errs(p)
    loss = {'mix': 0.5 * (rl + en), 'rel_l2': rl, 'energy': en}[args.loss]
    opt.zero_grad(); loss.backward(); opt.step()
    if it % (args.steps // 5) == 0:
        print(f"  step {it:5d}  correction rel-L2 {rl.item():.3f}  energy {en.item():.3f}  ({time.time()-t0:.0f}s)", flush=True)
# where is the unfitted part?  HB level shares of the residual (L2)
r = (T - m(X).reshape(1, nI, nc_out)).detach().numpy().reshape(-1)
out = {}; w = r
for k in range(L, 0, -1):
    Pk_ = P[k].tocsr(); nc = Pk_.shape[1]; c = w[:nc]; d = w[nc:] - (Pk_ @ c)[nc:]; out[k] = np.linalg.norm(d) ** 2; w = c
out[0] = np.linalg.norm(w) ** 2; tot = sum(out.values())
print("  residual HB share by level:", {k: round(v / tot, 3) for k, v in sorted(out.items())})
