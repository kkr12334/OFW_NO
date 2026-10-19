"""5 Oct: does the trained oa-vcb generalise in the RIGHT-HAND SIDE?  Every model was trained with the single load f = 1
(random coefficients only).  On the same test operators (samples 300-.. of rng 1000, channels 1e4) we feed residuals of
other loads and compare one learned pass with the classical member of its class (same hierarchy, V(4,4) Chebyshev +
deflation + Galerkin projection) and with the plain classical V(4,4).  Energy error of the correction relative to the
error it corrects, ||e - c||_A / ||e||_A (the network is positively homogeneous, so the load's scale is irrelevant).
Loads: f = 1 (training); a smooth random load (low Fourier modes, both signs); a Gaussian point source at a random
place; nodal white noise (worst case-like, out of any physical distribution).
usage: python diagnostics_cost/rhs_generalisation.py L k0 ckpt n_test"""
import sys, os, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, near_kernel_basis, classical_vcycle, classical_vcycle_proj
torch.set_num_threads(2)
L, k0, CK, NT = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
H = NestedHierarchy(L, coarse=perforated_square(n=9)); XY = H.level_coords(L)
rng = np.random.default_rng(1000); A_all = [field_channels(H, 1e4, rng) for _ in range(400)]
te = list(range(300, 300 + NT)); a_te = [A_all[i] for i in te]
nl = 0; k_ = L
while k_ >= 1 and nl < 4 and len(H.nodes[k_ - 1]) >= 16: nl += 1; k_ -= 1
Pk = None
for k in range(k0 + 1, L + 1): Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr() if (Pk is not None and k0 > 0) else None
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load_elem(fe):                                         # element-wise constant load -> nodal vector
    F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[H.keep, j], (fe * area)[H.keep] / 3.0)
    return F[H.nodes[L]]
cen = xy.mean(1); g = np.random.default_rng(7)
def loads(a, i):
    gi = np.random.default_rng(100 + i)
    K = gi.normal(size=(4, 4)); sm = sum(K[p, q] * np.sin((p + 1) * np.pi * cen[:, 0]) * np.sin((q + 1) * np.pi * cen[:, 1])
                                         for p in range(4) for q in range(4))
    x0 = gi.uniform(0.15, 0.85, 2); pt = np.exp(-((cen - x0) ** 2).sum(1) / (2 * 0.03 ** 2))
    return {'f = 1 (training load)': H.F, 'smooth random load': load_elem(sm), 'point source': load_elem(pt),
            'nodal white noise': gi.normal(size=len(H.F)) * H.F.mean()}
D = OAData(H, a_te, nl, 4, 'cpu', H.level_coords, near_kernel=True)
SD = torch.load(CK, map_location='cpu', weights_only=False)['model']
net = OAVCycle(D, width=SD['pre.0.enc_lin.weight'].shape[0], nu=4, project='both', deflate=True); net.load_state_dict(SD); net.eval()
res = {}
for j, (i, a) in enumerate(zip(te, a_te)):
    A = H.A_fine(a).tocsr(); levs, A0 = oa_hierarchy(H, A, nl, 4); Z = near_kernel_basis(A)
    net.set_ops(*D.batch([j]), Z=D.batch_z([j]))
    for name, F in loads(a, i).items():
        u = spsolve(A.tocsc(), F)
        uc = Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F) if Pk is not None else np.zeros_like(F)
        r = F - A @ uc; e = u - uc; nE = np.sqrt(e @ (A @ e))
        with torch.no_grad(): c = net(torch.tensor(r[None], dtype=torch.float32)).numpy()[0].astype(float)
        cm = classical_vcycle_proj(levs, A0, r, Z=Z, nu=4); cv = classical_vcycle(levs, A0, r)
        E = lambda x: np.sqrt((e - x) @ (A @ (e - x))) / nE
        res.setdefault(name, []).append((E(c), E(cm), E(cv)))
print(f"L{L}, k0 {k0}, {NT} test operators, model {os.path.basename(CK)}; energy error of the correction relative to the error it corrects")
print(f"{'load':<30} {'oa-vcb':>9} {'classical member':>17} {'classical V(4,4)':>17}  learned better on")
for name, v in res.items():
    v = np.array(v)
    print(f"{name:<30} {v[:, 0].mean():9.4f} {v[:, 1].mean():17.4f} {v[:, 2].mean():17.4f}  {np.mean(v[:, 0] < v[:, 1]) * 100:5.0f}% of samples")
