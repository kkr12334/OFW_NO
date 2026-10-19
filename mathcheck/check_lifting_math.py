"""Math check: does 'oa-ll with an HB-based predict' reproduce the operator-adapted scheme?
One refinement level of the perforated mesh (level L split into coarse C = level L-1 nodes, new F)."""
import numpy as np, scipy.sparse as sp, scipy.sparse.linalg as spla, sys
sys.path.insert(0, '.')
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square

L = 2
H = NestedHierarchy(L, coarse=perforated_square(n=9))
rng = np.random.default_rng(0)
a = np.asarray(make_field(H, 'channels', 1e4, rng)).ravel()
A = H.A_fine(a).toarray(); f = H.F
nc = len(H.nodes[L - 1]); n = A.shape[0]; C, F = slice(0, nc), slice(nc, n)
print(f"level {L}: {n} unknowns, coarse {nc}, new {n-nc}, contrast 1e4 channels")
AFF, AFC, ACC = A[F, F], A[F, C], A[C, C]
P_hb = H.P[L].toarray()[F, :]                          # geometric interpolation rows (0.5 / 0.5)
P_oa = -np.linalg.solve(AFF, AFC)                      # Route V exact predict

def T_of(Pr):  # u = T w,  w = (w_C, d_F):  u_C = w_C, u_F = Pr w_C + d_F
    T = np.eye(n); T[F, C] = Pr; return T
off = lambda M: np.linalg.norm(M[F, C]) / np.linalg.norm(M)

print("\n1. Congruence T^T A T (what operator-adapted wavelets do)")
for name, Pr in [('HB', P_hb), ('adapted', P_oa)]:
    T = T_of(Pr); K = T.T @ A @ T
    print(f"   {name:8s} off-diagonal block / ||K|| = {off(K):.2e}")
Phb_full = H.P[L].toarray()
print(f"   HB coarse block == Galerkin P^T A P ?  diff {np.abs((T_of(P_hb).T@A@T_of(P_hb))[C,C] - Phb_full.T@A@Phb_full).max():.1e}")

print("\n2. Similarity T^-1 A T (what an exactly-invertible lifting network represents)")
for name, Pr in [('HB', P_hb), ('adapted', P_oa)]:
    T = T_of(Pr); S = np.linalg.solve(T, A @ T)
    print(f"   {name:8s} off-diagonal block / ||S|| = {off(S):.2e}")
print("   -> no local lifting can block-diagonalise A by similarity: that needs range(T_F) to be an")
print("      invariant subspace of A (eigenvectors, global).  Congruence needs only A-orthogonality (local).")

print("\n3. The classical solve u = T D^-1 T^T f uses TWO different liftings")
T = T_of(P_oa); D = T.T @ A @ T
u_ex = np.linalg.solve(A, f)
g = T.T @ f                                            # analysis of the LOAD: update-only lifting with P^T
print(f"   (T^T f)_C - (f_C + P^T f_F) = {np.abs(g[C] - (f[C] + P_oa.T @ f[F])).max():.1e}   (update step, predict 0)")
w = np.concatenate([np.linalg.solve(D[C, C], g[C]), np.linalg.solve(D[F, F], g[F])])
print(f"   T D^-1 T^T f vs direct solve: rel err {np.linalg.norm(T@w-u_ex)/np.linalg.norm(u_ex):.1e}")
w_bad = np.linalg.solve(T, f)                          # analysis with T^-1 instead (inverse-pair design)
w_bad = np.concatenate([np.linalg.solve(D[C, C], w_bad[C]), np.linalg.solve(D[F, F], w_bad[F])])
print(f"   T D^-1 T^-1 f (inverse pair)   : rel err {np.linalg.norm(T@w_bad-u_ex)/np.linalg.norm(u_ex):.1e}")

print("\n4. How far is the adapted predict from HB at contrast 1e4?")
print(f"   ||P_oa - P_hb||_F / ||P_hb||_F = {np.linalg.norm(P_oa-P_hb)/np.linalg.norm(P_hb):.2f}")
nz = np.abs(P_oa) > 1e-3 * np.abs(P_oa).max()
print(f"   support: HB {(P_hb!=0).sum(1).mean():.1f} coarse nodes per new node, adapted "
      f"{nz.sum(1).mean():.1f} (|w| > 1e-3 max); row sums HB {P_hb.sum(1).mean():.2f}, adapted {P_oa.sum(1).mean():.2f}")

print("\n5. Can a predict that sees only COARSE-node features represent P_oa?")
# elements whose 3 vertices are all new nodes (centre child triangles) touch no coarse node
lvl = np.full(len(H.coords), -1)
for k in range(L, -1, -1): lvl[H.mesh.level_nodes[k]] = k
is_new = np.zeros(len(H.coords), bool); is_new[H.new[L]] = True
centre = np.all(is_new[H.elem], axis=1)
print(f"   {centre.sum()} of {len(H.elem)} elements touch no coarse node")
a2 = a.copy(); a2[centre] *= 1e3
A2 = H.A_fine(a2).toarray(); P2 = -np.linalg.solve(A2[F, F], A2[F, C])
coarse_ids = H.nodes[L - 1]
touch = np.zeros(len(H.coords), bool); touch[H.elem[~centre].ravel()] = True
print(f"   coefficient x1e3 on those elements only: coarse-node coefficient data changed? "
      f"{'no' if not np.any(np.isin(coarse_ids, H.elem[centre].ravel())) else 'yes'}")
print(f"   change of the adapted predict ||dP||/||P|| = {np.linalg.norm(P2-P_oa)/np.linalg.norm(P_oa):.2f}")
print("   -> the adapted weights depend on fine-scale coefficient data that a lifting predict acting on")
print("      the transformed coarse features cannot see; it must be CONDITIONED on the operator separately.")
