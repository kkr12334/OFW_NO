"""EXPLORATORY, post hoc (after the barriers L3 pilot; affects no verdict). Where does M3's localisation loss on barriers
come from?  For each pilot barrier geometry at one contrast:
  - K = 1 regions: edge-connected components of the non-barrier elements; which touch the Dirichlet boundary (outer
    boundary or holes).  Components that do not are enclosed islands (candidates for the near-kernel Z).
  - OAZ partition members (exactly as in the runner): robustness index C_theta; how many distinct K = 1 components the
    member carries non-negligible weight on (max theta >= 0.05 on that component), and the theta range there.
Writes explore_localisation_loss.json; prints a per-geometry summary and the worst members.
    python3 rc_explore_barrier_loss.py CONTRAST [IDS] [FAMILY]
For FAMILY = channels the components are those of the high-conductivity (K = c) channel elements instead."""
import os, sys, json, numpy as np, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE): sys.path.insert(0, p)
from scipy.sparse.csgraph import connected_components
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis
import rc_fem as R

c = float(sys.argv[1]); IDS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else list(range(8))
FAM = sys.argv[3] if len(sys.argv) > 3 else 'barriers'
cm = perforated_square(n=9); H = NestedHierarchy(3, coarse=cm); M = R.Mesh(H, cm); fe = M.fe
rng = np.random.default_rng({'channels': 2026101801, 'barriers': 2026101802}[FAM])
draws = [make_field(H, 'channels', 1e4 if FAM == 'channels' else 1e-4, rng) for _ in range(max(IDS) + 1)]
# element adjacency through shared edges
E = fe.E; ne = len(E)
edges = np.sort(np.concatenate([E[:, [0, 1]], E[:, [1, 2]], E[:, [0, 2]]]), 1)
key = edges[:, 0] * len(fe.X) + edges[:, 1]; el = np.tile(np.arange(ne), 3)
order = np.argsort(key); ks, els = key[order], el[order]
same = np.nonzero(ks[1:] == ks[:-1])[0]
adj_i, adj_j = els[same], els[same + 1]
out = []
for g in IDS:
    mask = draws[g] != 1.0
    K = np.where(mask, c if FAM == 'channels' else 1.0 / c, 1.0)
    comp_el = mask if FAM == 'channels' else ~mask             # elements whose components matter
    A = M.stiffness(K)
    levs, _ = oa_hierarchy(H, H.A_fine(K), 3, 4)
    Phi = sp.vstack([sp.identity(levs[0][1].shape[1], format='csr'), levs[0][1]]).tocsr(); Z = near_kernel_basis(levs[0][0])
    Th, inVc, C, nfb = R.pu_shepard(M, Phi, Z, 1e-3)
    idx = R.pu_robustness_index(M, K, Th, inVc)
    # K = 1 components
    keep = comp_el[adj_i] & comp_el[adj_j]
    G = sp.csr_matrix((np.ones(keep.sum()), (adj_i[keep], adj_j[keep])), shape=(ne, ne))
    ncomp, lab = connected_components(G, directed=False)
    lab = np.where(comp_el, lab, -1)
    comps = [k for k in np.unique(lab) if k >= 0]
    touches = {k: bool(M.dirichlet[np.unique(E[lab == k])].any()) for k in comps}
    islands = [k for k in comps if not touches[k]]
    # per member: components carrying weight
    Thc = Th.tocsc(); rows = []
    for i in np.nonzero(inVc)[0]:
        lo, hi = Thc.indptr[i], Thc.indptr[i + 1]; nodes, vals = Thc.indices[lo:hi], Thc.data[lo:hi]
        th = np.zeros(len(fe.X)); th[nodes] = vals
        kel = np.unique(np.concatenate([M.el_of[a] for a in nodes]))
        per = {}
        for k in kel:
            if lab[k] < 0: continue
            per.setdefault(int(lab[k]), []).append(th[E[k]].max())
        carried = {k: float(max(v)) for k, v in per.items() if max(v) >= 0.05}
        rows.append(dict(member=int(i), index=float(idx[i]), n_components=len(carried),
                         carries_island=any(not touches[k] for k in carried), centroid=fe.X[nodes].mean(0).tolist(),
                         barrier_fraction=float(mask[kel].mean())))
    rows.sort(key=lambda r: -r['index'])
    big = [r for r in rows if r['index'] >= 3]
    rec = dict(family=FAM, id=g, contrast=c, near_kernel=int(Z.shape[1]), n_K1_components=len(comps), n_islands=len(islands),
               island_sizes=[int((lab == k).sum()) for k in islands], index_max=rows[0]['index'],
               n_members_index_ge_3=len(big), multi_component_among_index_ge_3=sum(r['n_components'] >= 2 for r in big),
               multi_component_all=sum(r['n_components'] >= 2 for r in rows), worst=rows[:5])
    out.append(rec)
    print(f"{FAM} id {g} c={c:.0e}: near-kernel {rec['near_kernel']}; K=1 components {len(comps)}, enclosed islands {len(islands)} "
          f"(sizes {rec['island_sizes'][:5]}); index max {rec['index_max']:.2f}; members with index >= 3: {len(big)}, "
          f"of which straddle >= 2 K=1 components: {rec['multi_component_among_index_ge_3']} (all members straddling: {rec['multi_component_all']})", flush=True)
    for r in rows[:3]:
        print(f"     member {r['member']}: index {r['index']:.2f}, components {r['n_components']}, island {r['carries_island']}, "
              f"barrier fraction {r['barrier_fraction']:.2f}, centroid ({r['centroid'][0]:.3f}, {r['centroid'][1]:.3f})", flush=True)
p = os.path.join(HERE, 'explore_localisation_loss.json')
prev = json.load(open(p)) if os.path.exists(p) else []
json.dump(prev + out, open(p, 'w'), indent=1)
