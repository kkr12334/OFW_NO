"""Route C scoping, task 1: discretisation facts the theorem map depends on (development data only, seed 1)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'diagnostics_cost'))
from s1_common import setup
from oa_vcycle import oa_hierarchy, near_kernel_basis
for fam, kw in (('channels', dict(contrast=1e4, field='channels', load='all')), ('barriers', dict(contrast=1e-4, field='channels', load='no-low'))):
    S = setup(3, 2, 0, n=400, **kw) if False else None
    from oa_lifting import NestedHierarchy, make_field
    from coarse_meshes import perforated_square
    H = NestedHierarchy(3, coarse=perforated_square(n=9))
    rng = np.random.default_rng(1001)
    a = [make_field(H, 'channels', kw['contrast'], rng) for _ in range(400)]
    xy = H.coords[H.elem]
    e = np.stack([xy[:, 1] - xy[:, 0], xy[:, 2] - xy[:, 1], xy[:, 0] - xy[:, 2]], 1)
    L = np.linalg.norm(e, axis=2)
    area = 0.5 * np.abs(e[:, 0, 0] * (-e[:, 2, 1]) - e[:, 0, 1] * (-e[:, 2, 0]))
    ang = []
    for i in range(3):
        u_, v_ = -e[:, i - 1], e[:, i]
        ang.append(np.degrees(np.arccos(np.clip((u_ * v_).sum(1) / (np.linalg.norm(u_, axis=1) * np.linalg.norm(v_, axis=1)), -1, 1))))
    ang = np.array(ang)
    print(fam, 'elements', H.n_elem, 'all nodes', len(H.coords), 'unknowns', len(H.nodes[3]), 'Dirichlet nodes', len(H.boundary),
          'h max/min %.4f/%.4f' % (L.max(), L.min()), 'min angle %.1f max angle %.1f' % (ang.min(), ang.max()), 'area', area.sum())
    vals = np.unique(np.concatenate(a))
    print('  coefficient values', vals, ' unknowns per level', [len(H.nodes[k]) for k in range(4)])
    ks = [near_kernel_basis(H.A_fine(ai).tocsr()).shape[1] for ai in a[300:400]]
    print('  near-kernel dims on dev ids 300-399:', np.bincount(ks))
    obtuse = (ang > 90 + 1e-9).any(0).sum(); print('  obtuse elements', obtuse)
