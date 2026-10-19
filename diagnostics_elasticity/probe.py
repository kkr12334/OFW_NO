import pickle, numpy as np, glob, sys; sys.path.insert(0, '.')
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from elasticity import VectorHierarchy
pool = pickle.load(open(glob.glob('/tmp/claude-0/ov/pool_elast_*.pkl')[0], 'rb'))
outer = lambda x, y: not (np.isclose(x,0) or np.isclose(x,1) or np.isclose(y,0) or np.isclose(y,1))
H = NestedHierarchy(3, coarse=perforated_square(n=9), bc='mixed', neumann=outer); V = VectorHierarchy(H)
U = np.stack(pool['u']); UC = pool['uc_k2']; T = U - UC
A = [V.A_fine(E) for E in pool['E']]
eT = np.array([np.sqrt(T[i] @ (A[i] @ T[i])) for i in range(40)]); eU = np.array([np.sqrt(U[i] @ (A[i] @ U[i])) for i in range(40)])
print("correction energy / solution energy: min %.3f median %.3f max %.3f" % tuple(np.percentile(eT/eU, [0, 50, 100])))
print("correction rel-L2 norm: min %.3f max %.3f" % ((np.linalg.norm(T,axis=1)/np.linalg.norm(U,axis=1)).min(), (np.linalg.norm(T,axis=1)/np.linalg.norm(U,axis=1)).max()))
Tx, Ty = T[:, 0::2], T[:, 1::2]
print("rms ux / uy of correction: %.3e / %.3e" % (np.sqrt(np.mean(Tx**2)), np.sqrt(np.mean(Ty**2))))
m = T[:32].mean(0); R = T - m
print("sample-specific part: rms ux / uy %.3e / %.3e" % (np.sqrt(np.mean(R[:, 0::2]**2)), np.sqrt(np.mean(R[:, 1::2]**2))))
# where is the sample-specific correction energy?  pore-boundary nodes vs rest (nodal L2 share)
nodes = H.nodes[3]; pore = np.isin(nodes, H.neumann_nodes)
r2 = (R[:, 0::2]**2 + R[:, 1::2]**2).mean(0)
print(f"pore-boundary nodes: {pore.mean()*100:.1f}% of nodes carry {r2[pore].sum()/r2.sum()*100:.1f}% of the sample-specific correction (L2)")
# is the correction concentrated in soft or stiff material?
E0 = np.asarray(pool['E'][0])
