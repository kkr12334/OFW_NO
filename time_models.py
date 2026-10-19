"""Inference cost against mesh size, at the protocol budget (~20k parameters), random weights (cost does not depend
on the weights).  Evidence for the scaling argument: the lifting network reaches the whole domain through O(log N)
levels at O(N) cost, a flat message-passing network needs O(sqrt N) steps for the same reach, i.e. O(N^1.5).

    python time_models.py --levels 3 4 5 --models oa-ll oa-ll-ep oa-ll-epm gnn gnn30 --batch 8
"""
import argparse, time, numpy as np, torch
from oa_lifting import NestedHierarchy
from coarse_meshes import perforated_square
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant
from baselines_extra import MeshGNN, mesh_edges

ap = argparse.ArgumentParser()
ap.add_argument('--levels', type=int, nargs='+', default=[3, 4, 5])
ap.add_argument('--models', nargs='+', default=['oa-ll', 'oa-ll-ep', 'oa-ll-epm', 'gnn', 'gnn30'])
ap.add_argument('--batch', type=int, default=8)
ap.add_argument('--budget', type=int, default=22049)
ap.add_argument('--reps', type=int, default=10)
ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
a = ap.parse_args()
dev, cin = a.device, 5
n_params = lambda m: sum(p.numel() for p in m.parameters())


def fit(mk):
    lo, hi = 2, 256
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if n_params(mk(mid)) <= a.budget: lo = mid
        else: hi = mid
    return mk(lo)


print(f"device {dev}, batch {a.batch}, budget {a.budget:,}")
print(f"{'level':>5} {'nodes':>7} " + ''.join(f"{m:>14}" for m in a.models) + "   (ms per sample)")
for L in a.levels:
    H = NestedHierarchy(L, coarse=perforated_square(n=9))
    MG = MeshHierarchyGeometry(H, dev)
    dom = np.nonzero(H.in_domain)[0]
    ed, ef = mesh_edges(H.coords, H.elem[H.keep], dom)
    row = []
    for kind in a.models:
        if kind.startswith('oa-ll'):
            m = fit(lambda w: MeshLiftingNO(MG, cin, width=w, **variant(kind))).to(dev).eval()
            x = torch.randn(a.batch, len(H.nodes[L]), cin, device=dev)
        else:
            steps = 30 if kind == 'gnn30' else 10
            m = fit(lambda w: MeshGNN(cin, ed, ef, width=w, steps=steps)).to(dev).eval()
            x = torch.randn(a.batch, len(dom), cin, device=dev)
        with torch.no_grad():
            for _ in range(2): m(x)                                  # warm-up
            if dev == 'cuda': torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(a.reps): m(x)
            if dev == 'cuda': torch.cuda.synchronize()
        row.append((time.time() - t0) / a.reps / a.batch * 1e3)
    print(f"{L:>5} {len(H.nodes[L]):>7} " + ''.join(f"{t:>14.2f}" for t in row), flush=True)
