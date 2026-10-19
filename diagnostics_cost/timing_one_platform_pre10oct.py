"""5 Oct: one-platform (this CPU, 2 threads) cost of the TRAINED oa-vcb against classical solvers, per test sample, on the
driver's test set (samples 300-399 of rng 1000, channels 1e4).  The trained weights come from the Kaggle checkpoints
(results/ckpt/); the network is positively homogeneous, so the driver's target scale cancels.
For oa-vcb: operator data (operator-adapted hierarchy + near-kernel vectors, scipy) and inference (torch, batch 8 and 1).
Classical, from the same u_c: sparse direct solve (SuperLU factor + solve); two-grid PCG and smoothed-aggregation AMG PCG
(pyamg) run until each sample reaches oa-vcb's energy error on that sample (time and iterations); the classical
operator-adapted V(4,4)-cycle as a PCG preconditioner (same hierarchy as oa-vcb, build included).
usage: python diagnostics_cost/timing_one_platform.py L k0 ckpt [n_test]"""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, splu, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle, oa_hierarchy, near_kernel_basis, classical_vcycle
import pyamg
torch.set_num_threads(2)

L, k0, CK = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]; NT = int(sys.argv[4]) if len(sys.argv) > 4 else 100
H = NestedHierarchy(L, coarse=perforated_square(n=9)); F = H.F
rng = np.random.default_rng(1000); A_all = [field_channels(H, 1e4, rng) for _ in range(400)]
te = list(range(300, 300 + NT)); a_te = [A_all[i] for i in te]
nl = 0; k_ = L
while k_ >= 1 and nl < 4 and len(H.nodes[k_ - 1]) >= 16: nl += 1; k_ -= 1
Pk = None
for k in range(k0 + 1, L + 1): Pk = H.P[k] if Pk is None else H.P[k] @ Pk
Pk = Pk.tocsr() if (Pk is not None and k0 > 0) else None          # k0 = 0: no external coarse solve, u_c = 0

As = [H.A_fine(a).tocsr() for a in a_te]
U = np.array([spsolve(A.tocsc(), F) for A in As])
t = time.time()
UC = np.array([Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F) if Pk is not None else np.zeros_like(F) for A in As])
t_uc = (time.time() - t) / NT
R = np.array([F - A @ uc for A, uc in zip(As, UC)])
en = lambda A, e, u: float(np.sqrt(e @ (A @ e) / (u @ (A @ u))))

# ---- oa-vcb: operator data (timed per sample, scipy) + inference
t = time.time()
for A in As[:10]: oa_hierarchy(H, A, nl, 4); near_kernel_basis(A)
t_build = (time.time() - t) / 10
D = OAData(H, a_te, nl, 4, 'cpu', H.level_coords, near_kernel=True)
SD = torch.load(CK, map_location='cpu', weights_only=False)['model']
net = OAVCycle(D, width=SD['pre.0.enc_lin.weight'].shape[0], nu=4, project='both', deflate=True)   # width from the weights
net.load_state_dict(SD); net.eval()
Rt = torch.tensor(R, dtype=torch.float32)
def infer(bs):
    out = []; t = time.time()
    with torch.no_grad():
        for idx in np.array_split(np.arange(NT), int(np.ceil(NT / bs))):
            net.set_ops(*D.batch(idx), Z=D.batch_z(idx)); out.append(net(Rt[idx]).numpy())
    return np.concatenate(out), (time.time() - t) / NT
X8, t_inf8 = infer(8); _, t_inf1 = infer(1)
P = UC + X8.astype(np.float64)
e_net = np.array([en(A, p - u, u) for A, p, u in zip(As, P, U)])
r_net = np.array([np.linalg.norm(p - u) / np.linalg.norm(u) for p, u in zip(P, U)])
print(f"L{L}, k0 {k0}, {len(F)} unknowns, {NT} test samples, CPU 2 threads")
print(f"oa-vcb (trained, {os.path.basename(CK)}): rel-L2 {r_net.mean():.3e}  energy {e_net.mean():.3e}")
print(f"  cost per sample: u_c {t_uc * 1e3:.1f} ms + operator data {t_build * 1e3:.0f} ms (scipy) + inference "
      f"{t_inf8 * 1e3:.0f} ms (batch 8) / {t_inf1 * 1e3:.0f} ms (batch 1)")

# ---- classical
def pcg_to(A, x0, Minv, u, target, maxit=60):
    x = x0.copy(); r = F - A @ x; z = Minv(r); p = z.copy(); rz = r @ z; it = 0
    while en(A, x - u, u) > target and it < maxit:
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = Minv(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2; it += 1
    return x, it
rows = {'direct (SuperLU factor + solve)': [], 'two-grid PCG (Jacobi(1) + level-%d Galerkin)' % max(k0, L - 1): [],
        'SA-AMG PCG (pyamg, V(1,1))': [], 'PCG, classical op.-adapted V(4,4) preconditioner': []}
its = {k: [] for k in rows}
Pt = H.P[L].tocsr(); om = 2 / 3
for j, (A, uc, u) in enumerate(zip(As, UC, U)):
    tg = e_net[j]
    t = time.time(); splu(A.tocsc()).solve(F); rows['direct (SuperLU factor + solve)'].append(time.time() - t)
    t = time.time(); solc = factorized((Pt.T @ A @ Pt).tocsc()); dinv = 1 / A.diagonal()
    def M(r):
        x = om * dinv * r; x = x + Pt @ solc(Pt.T @ (r - A @ x)); return x + om * dinv * (r - A @ x)
    x0 = uc if Pk is not None else np.zeros_like(F)
    _, n_ = pcg_to(A, x0, M, u, tg); k_ = list(rows)[1]; rows[k_].append(time.time() - t + (t_uc if Pk is not None else 0)); its[k_].append(n_)
    t = time.time(); ml = pyamg.smoothed_aggregation_solver(A); Mm = ml.aspreconditioner(cycle='V')
    _, n_ = pcg_to(A, x0, lambda r: Mm @ r, u, tg); k_ = list(rows)[2]; rows[k_].append(time.time() - t + (t_uc if Pk is not None else 0)); its[k_].append(n_)
    if j < 20:
        t = time.time(); levs, A0 = oa_hierarchy(H, A, nl, 4)
        _, n_ = pcg_to(A, x0, lambda r: classical_vcycle(levs, A0, r), u, tg); k_ = list(rows)[3]
        rows[k_].append(time.time() - t + (t_uc if Pk is not None else 0)); its[k_].append(n_)
print("classical, from the same u_c, each sample run to oa-vcb's energy error on that sample:")
for k, v in rows.items():
    it = f"  its median {np.median(its[k]):.0f} (max {max(its[k])})" if its[k] else ""
    print(f"  {k:<52} {np.mean(v) * 1e3:7.1f} ms/sample{it}{'  [first 20 samples]' if k.startswith('PCG, classical') else ''}")
