"""Verification of the elasticity side: element, mesh/BCs, solver convergence, Galerkin coarse
operator, vector HB / adapted transform, rigid-body modes, local-transform conditioning."""
import numpy as np, scipy.sparse as sp, sys, time
from scipy.sparse.linalg import spsolve, eigsh, splu
sys.path.insert(0, '.')
from oa_lifting import NestedHierarchy, OATransform, field_channels, pcg_kappa
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy, unit_element_matrices
outer = lambda x, y: not (np.isclose(x, 0) or np.isclose(x, 1) or np.isclose(y, 0) or np.isclose(y, 1))
mk = lambda L: NestedHierarchy(L, coarse=perforated_square(n=9), bc='mixed', neumann=outer)

print("1. ELEMENT")
H = mk(1); V = VectorHierarchy(H)
ev = np.linalg.eigvalsh(V.K1)                                   # (ne, 6)
print(f"   zero eigenvalues per element (|ev| < 1e-10 max ev): {np.unique((np.abs(ev) < 1e-10 * ev.max(1, keepdims=True)).sum(1))}  (expect 3 rigid modes)")
print(f"   min nonzero / max eigenvalue over elements: {ev[:, 3:].min():.2e} / {ev.max():.2e};  symmetric: {np.abs(V.K1 - V.K1.transpose(0,2,1)).max():.1e}")
# patch test: global K (no BCs) times linear field = 0 in the interior and rigid modes in the kernel
I, J, nd = V._IJ; K = sp.coo_matrix((V.K1.ravel(), (I, J)), shape=(nd, nd)).tocsr()
x, y = H.coords[:, 0], H.coords[:, 1]
rbm = [np.stack([np.ones_like(x), 0*x], 1).ravel(), np.stack([0*x, np.ones_like(x)], 1).ravel(), np.stack([-y, x], 1).ravel()]
print(f"   ||K r|| / ||K|| ||r|| for the 3 rigid modes: {[f'{np.linalg.norm(K@r)/(abs(K).max()*np.linalg.norm(r)):.1e}' for r in rbm]}")
lin = np.stack([0.3*x + 0.2*y, -0.1*x + 0.5*y], 1).ravel()        # constant strain field
bnd = np.zeros(len(x), bool); bnd[H.mesh.get_boundary_nodes()] = True
res = (K @ lin).reshape(-1, 2)
print(f"   patch test: max |K u_lin| at interior nodes {np.abs(res[~bnd]).max():.1e} (boundary nodes carry the traction)")

print("\n2. MESH / BOUNDARY CONDITIONS")
H = mk(3); V = VectorHierarchy(H)
print(f"   level 3: {len(H.nodes[3])} unknown nodes, {len(H.neumann_nodes)} traction-free pore nodes, "
      f"{len(H.boundary)} clamped nodes; clamped nodes all on the outer square: "
      f"{bool(np.all([not outer(*H.coords[g]) for g in H.boundary]))}")
A1 = V.A_fine(np.ones(H.n_elem))
print(f"   A symmetric: {abs(A1 - A1.T).max():.1e};  smallest eigenvalue > 0: {eigsh(A1, 1, sigma=0, which='LM')[0][0]:.2e}")
print(f"   load: sum F_x {V.F[0::2].sum():.2e}, sum F_y {V.F[1::2].sum():.4f} (area of the free domain incl. clamped-node share)")

print("\n3. SOLVER CONVERGENCE (E = 1 and bands 1e2), energy error vs level-5 reference")
Hs = {L: mk(L) for L in range(1, 6)}; Vs = {L: VectorHierarchy(Hs[L]) for L in Hs}
rng = np.random.default_rng(0)
Eref = np.asarray(field_channels(Hs[5], 1e2, rng)).ravel()
# element coefficient is defined on the fine mesh; for coarser levels use the same function at centroids
def restrict_field(Lc):  # coarse element -> value of the finest element containing its centroid
    from scipy.spatial import cKDTree
    return Eref[cKDTree(Hs[5].centroids).query(Hs[Lc].centroids)[1]]
for name, Ef in [('E = 1', lambda L: np.ones(Hs[L].n_elem)), ('bands 1e2', restrict_field)]:
    A5 = Vs[5].A_fine(Ef(5)); u5 = spsolve(A5, Vs[5].F); e5 = np.sqrt(u5 @ A5 @ u5)
    errs = []
    for L in range(1, 5):
        uL = spsolve(Vs[L].A_fine(Ef(L)), Vs[L].F)
        Pp = sp.identity(len(uL), format='csr')
        for k in range(L + 1, 6): Pp = Vs[5].P[k] @ Pp if k > L + 1 else Vs[5].P[k]
        # prolong through the level-5 hierarchy's own P's (nested meshes, same node ids)
        Pp = None
        for k in range(L + 1, 6): Pp = Vs[5].P[k] if Pp is None else Vs[5].P[k] @ Pp
        d = u5 - Pp @ uL
        errs.append(np.sqrt(d @ A5 @ d) / e5)
    rates = [np.log2(errs[i] / errs[i + 1]) for i in range(3)]
    print(f"   {name:10s}: energy error L1..L4 {', '.join(f'{e:.3f}' for e in errs)};  rates {', '.join(f'{r:.2f}' for r in rates)}")

print("\n4. GALERKIN COARSE OPERATOR = ASSEMBLED COARSE OPERATOR (E = 1)")
for L in (2, 3):
    A_L = Vs[L].A_fine(np.ones(Hs[L].n_elem)); A_c = Vs[L - 1].A_fine(np.ones(Hs[L - 1].n_elem))
    G = Vs[L].P[L].T @ A_L @ Vs[L].P[L]
    print(f"   level {L}->{L-1}: max |P^T A P - A_coarse| / max|A| = {abs(G - A_c).max() / abs(A_c).max():.1e}")

print("\n5. VECTOR TRANSFORMS on the level-3 bands 1e2 operator")
H = Hs[3]; V = Vs[3]
E3 = restrict_field(3); A = V.A_fine(E3)
for mode, rho in [('hb', 0), ('exact', 0), ('local', 2), ('local', 3), ('local', 4)]:
    t0 = time.time(); T = OATransform(V, A, mode=mode, rho=rho, route='V'); tb = time.time() - t0
    b = np.random.default_rng(1).standard_normal(A.shape[0]); _, its, kap = pcg_kappa(A, b, T.precond)
    print(f"   {mode:5s} rho {rho}: build {tb*1e3:6.0f} ms, kappa(BA) {kap:10.2f}, PCG its {its}")

print("\n6. RIGID-BODY MODES through the level-3 predict (interior new nodes, away from the clamp)")
Hc = Hs[3]; k = 3; nc = Vs[3].P[k].shape[1]
Pg = Vs[3].P[k].toarray()[nc:, :]
xyc = Hc.level_coords(k - 1); xyn = Hc.new_coords(k)
mode_c = lambda xy: [np.stack([np.ones(len(xy)), np.zeros(len(xy))], 1).ravel(), np.stack([-xy[:, 1], xy[:, 0]], 1).ravel()]
far = np.repeat(np.min(np.minimum.reduce([xyn[:, 0], 1 - xyn[:, 0], xyn[:, 1], 1 - xyn[:, 1]]) [None], 0) > 0.25, 2) if False else None
dist = np.minimum.reduce([xyn[:, 0], 1 - xyn[:, 0], xyn[:, 1], 1 - xyn[:, 1]]); interior = np.repeat(dist > 0.2, 2)
A_FF = (Vs[3].E[k].T @ A @ Vs[3].E[k]).tocsc(); A_FC = (Vs[3].E[k].T @ A)[:, :nc].tocsc()
from oa_lifting import _local_C
P_ex = -splu(A_FF).solve(A_FC.toarray())
P_l2 = _local_C(A_FF, A_FC, Vs[3].new_coords(k), Vs[3].level_coords(k - 1), 2 * Hc.h[k - 1]).toarray()
P_l4 = _local_C(A_FF, A_FC, Vs[3].new_coords(k), Vs[3].level_coords(k - 1), 4 * Hc.h[k - 1]).toarray()
for nm, rc, rf in zip(['translation x', 'rotation'], mode_c(xyc), mode_c(xyn)):
    out = []
    for lab, Pm in [('HB', Pg), ('exact', P_ex), ('local rho2', P_l2), ('local rho4', P_l4)]:
        out.append(f"{lab} {np.linalg.norm((Pm @ rc - rf)[interior]) / np.linalg.norm(rf[interior]):.1e}")
    print(f"   {nm:13s}: reproduction error  " + ",  ".join(out))
