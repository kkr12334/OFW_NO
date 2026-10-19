"""5 Oct: operator complexity of the operator-adapted hierarchy, and the price of localising it.
The current build (m = 4 Jacobi steps of prolongator smoothing on each level's Galerkin operator) fills in level by level:
at L4 the level-2 operator is nearly dense (profile: 98k / 270k / 447k nonzeros on 14398 / ~3500 / ~850 nodes), so the
work per sample is dominated by the coarse levels and grows faster than N.  Localised variant (oa_vcycle.oa_hierarchy_loc):
each Jacobi step restricted to the level-k patch of radius r (graph distance on the level-k mesh), row sums projected.
Patch-localised (r = 1, 3), FILTERED prolongator smoothing (oa_hierarchy_filt, m = 4 and 2) and the current build: nonzeros per row by level, operator complexity sum_k nnz(A_k) / nnz(A_fine),
build time per sample, and the CLASSICAL one-pass accuracy that the learned cycle builds on:
  channels 1e4: V(4,4) from the Galerkin u_c (k0 = L-1, as in the GPU runs) and from zero;
  barriers 1e-4 (--load no-low), L3 only: the classical member (V(4,4) + near-kernel deflation + Galerkin projection).
usage: python diagnostics_cost/oa_localised.py L n_samples"""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, oa_hierarchy_loc, oa_hierarchy_filt, level_graphs, loc_patterns, classical_vcycle, classical_vcycle_proj, near_kernel_basis

L = int(sys.argv[1]); NS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
nl = 0; k_ = L
while k_ >= 1 and nl < 4 and True:
    nl += 1; k_ -= 1
H = NestedHierarchy(L, coarse=perforated_square(n=9)); F1 = H.F
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
def load(mask):
    F = np.zeros(len(H.coords))
    for j in range(3): np.add.at(F, H.elem[H.keep & mask, j], area[H.keep & mask] / 3.0)
    return F[H.nodes[L]]
Pk = H.P[L].tocsr()                                   # Galerkin u_c on level L-1 (k0 = L-1)
def gal_uc(A, F): return Pk @ spsolve((Pk.T @ A @ Pk).tocsc(), Pk.T @ F)

classes = [('channels', 1e4)] + ([('barriers', 1e-4)] if L <= 3 else [])
variants = [('current (unlocalised)', None), ('patch r = 1', 1), ('patch r = 3', 3), ('filtered, m = 4', 'f4'),
            ('filtered, m = 2', 'f2'), ('strength 0.01, m = 4', 's4:0.01'), ('strength 0.05, m = 4', 's4:0.05'),
            ('strength 0.2, m = 4', 's4:0.2')] if L <= 3 else [('current (unlocalised)', None), ('filtered, m = 4', 'f4'), ('filtered, m = 2', 'f2'),
     ('strength 0.01, m = 4', 's4:0.01'), ('strength 0.05, m = 4', 's4:0.05'), ('strength 0.2, m = 4', 's4:0.2')]
masks = {r: loc_patterns(H, nl, r) for _, r in variants if isinstance(r, int)}; GR = level_graphs(H, nl)
for cname, con in classes:
    rng = np.random.default_rng(1000); A_all = [field_channels(H, con, rng) for _ in range(400)]
    print(f"\n=== L{L} {cname} ({con:g}), {nl} levels, test samples 300-{299 + NS} ===")
    for vname, r in variants:
        nnz, tb, e1, e2, e3 = None, [], [], [], []
        for i in range(300, 300 + NS):
            a = A_all[i]; A = H.A_fine(a).tocsr(); F = F1 if cname == 'channels' else load(a >= 0.5)
            u = spsolve(A.tocsc(), F); en = lambda v: np.sqrt(v @ (A @ v)) / np.sqrt(u @ (A @ u))
            t = time.time()
            levs, A0 = (oa_hierarchy(H, A, nl, 4) if r is None else (oa_hierarchy_filt(H, A, nl, int(r[1]), graphs=GR) if r[0] == 'f' else
                         oa_hierarchy_filt(H, A, nl, int(r[1]), theta=float(r.split(':')[1])))
                        if isinstance(r, str) else oa_hierarchy_loc(H, A, nl, 4, r=r, masks=masks[r]))
            tb.append(time.time() - t)
            if nnz is None: nnz = [(Ak.shape[0], Ak.nnz) for Ak, _ in levs] + [(A0.shape[0], A0.nnz)]
            if cname == 'channels':
                uc = gal_uc(A, F)
                e1.append(en(uc + classical_vcycle(levs, A0, F - A @ uc) - u))
                e2.append(en(classical_vcycle(levs, A0, F) - u))
            else:
                uc = gal_uc(A, F)
                e3.append(en(uc + classical_vcycle_proj(levs, A0, F - A @ uc, Z=near_kernel_basis(A), nu=4) - u))
        per_row = " / ".join(f"{z / n:.0f}" for n, z in nnz)
        cop = sum(z for _, z in nnz) / nnz[0][1]
        acc = (f"V(4,4) from u_c {np.mean(e1):.4f}, from zero {np.mean(e2):.4f}" if cname == 'channels'
               else f"classical member from u_c {np.mean(e3):.4f}")
        print(f"  {vname:<22} nnz/row by level {per_row:<26} op. complexity {cop:5.2f}  build {np.mean(tb) * 1e3:6.0f} ms  | {acc}", flush=True)
