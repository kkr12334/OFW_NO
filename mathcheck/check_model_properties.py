"""Checks on the network models (29 Sep 2026).
1. oa-ll and oa-ll-s: the lifting is exactly invertible for ANY parameters (forward lifting, then the inverse
   lifting with the same details, returns the encoded features to round-off).
2. oa-ll-s trained on a level-L hierarchy runs unchanged on level L+1 (shapes), oa-ll refuses.
3. MeshGNN is permutation-equivariant (relabelling the nodes relabels the output).
"""
import sys, torch, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, '..')
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO
from baselines_extra import MeshGNN, mesh_edges

torch.manual_seed(0)
H3 = NestedHierarchy(3, coarse=perforated_square(n=9)); H4 = NestedHierarchy(4, coarse=perforated_square(n=9))
G3, G4 = MeshHierarchyGeometry(H3, 'cpu'), MeshHierarchyGeometry(H4, 'cpu')
x = torch.randn(3, len(H3.nodes[3]), 5, dtype=torch.float64)
for share in (False, True):
    m = MeshLiftingNO(G3, 5, width=24, share_levels=share).double()
    with torch.no_grad():
        X0 = m.enc(x); X = X0; dets = []
        for l, Lv in enumerate(m.g.levels):
            Xc, Xf = X[:, :Lv['nc']], X[:, Lv['nc']:]
            d = Xf - m._predict(l, Xc); X = Xc + m._update(l, d); dets.append(d)
        s = X
        for l in reversed(range(len(m.g.levels))):
            Xc = s - m._update(l, dets[l]); s = torch.cat([Xc, dets[l] + m._predict(l, Xc)], 1)
    print(f"1. {'oa-ll-s' if share else 'oa-ll  '}: inverse(forward(X)) - X = {(s - X0).abs().max().item():.1e} (float64)")
m = MeshLiftingNO(G3, 5, width=24, share_levels=True); m.set_geometry(G4)
print(f"2. oa-ll-s on level 4: output {tuple(m(torch.randn(2, len(H4.nodes[4]), 5)).shape)}")
try:
    MeshLiftingNO(G3, 5, width=24).set_geometry(G4); print("2. oa-ll accepted a deeper hierarchy -- WRONG")
except ValueError:
    print("2. oa-ll (per-level networks) refuses a deeper hierarchy, as it should")
dom = np.nonzero(H3.in_domain)[0]; ed, ef = mesh_edges(H3.coords, H3.elem, dom)
g = MeshGNN(5, ed, ef, width=16, steps=4)
xg = torch.randn(2, len(dom), 5); perm = torch.randperm(len(dom)); inv = torch.argsort(perm)
y = g(xg)
ed2 = inv.numpy()[ed]                                   # node i -> new label inv[i]
g.set_graph(ed2, ef); y2 = g(xg[:, perm])
print(f"3. MeshGNN permutation equivariance: max |f(Px) - P f(x)| = {(y2 - y[:, perm]).abs().max().item():.1e}")
