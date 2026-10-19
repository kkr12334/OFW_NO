"""Route C scoping: quick feasibility facts on development operators (seed-1 data, ids 300-309).  Diagnostic only.
(1) is the P0 hat-moment map B (P0(T_h) -> R^N) of full row rank (exact P0 residual lifting exists)?
(2) quasi-monotonicity census of interior vertices (two-valued K: non-QM iff >= 2 cyclic runs of max-K elements);
(3) operator-adapted partition of unity: sum of composite OA basis functions vs 1, supports and overlap per level."""
import os, sys, numpy as np, scipy.sparse as sp
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
from oa_vcycle import oa_hierarchy, near_kernel_basis

H = NestedHierarchy(3, coarse=perforated_square(n=9))
E, X = H.elem, H.coords
nn = len(X); ne = len(E)
area = 0.5 * np.abs((X[E[:, 1], 0] - X[E[:, 0], 0]) * (X[E[:, 2], 1] - X[E[:, 0], 1]) - (X[E[:, 2], 0] - X[E[:, 0], 0]) * (X[E[:, 1], 1] - X[E[:, 0], 1]))
unk = H.nodes[3]; pos = -np.ones(nn, int); pos[unk] = np.arange(len(unk))
# (1) B: rows = unknown nodes, cols = elements, B[a,K] = |K|/3 if a in K
r, c, v = [], [], []
for j in range(3):
    m = pos[E[:, j]] >= 0
    r.append(pos[E[m, j]]); c.append(np.nonzero(m)[0]); v.append(area[m] / 3)
B = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(len(unk), ne))
G = (B @ B.T).toarray(); ev = np.linalg.eigvalsh(G)
print(f'(1) B is {B.shape}; eig(B B^T) min/max = {ev[0]:.3e} / {ev[-1]:.3e}  (cond {ev[-1]/ev[0]:.1f})')
# vertex patches in cyclic order are not needed: count edge-connected components of max-K elements in each patch
el_of = [[] for _ in range(nn)]
for k, t in enumerate(E):
    for a in t: el_of[a].append(k)
edge_el = {}
for k, t in enumerate(E):
    for i in range(3):
        e = tuple(sorted((t[i], t[(i + 1) % 3]))); edge_el.setdefault(e, []).append(k)
def ncomp(elems, mask):
    S = [k for k in elems if mask[k]]
    if not S: return 0
    Sset = set(S); seen = set(); n = 0
    for s in S:
        if s in seen: continue
        n += 1; stack = [s]; seen.add(s)
        while stack:
            k = stack.pop(); t = E[k]
            for i in range(3):
                e = tuple(sorted((t[i], t[(i + 1) % 3])))
                for k2 in edge_el[e]:
                    if k2 in Sset and k2 not in seen: seen.add(k2); stack.append(k2)
    return n
rng = np.random.default_rng(1001)
fields = {'channels': [], 'barriers': []}
a_all = [make_field(H, 'channels', 1e4, rng) for _ in range(400)]
rng = np.random.default_rng(1001)
b_all = [make_field(H, 'channels', 1e-4, rng) for _ in range(400)]
for fam, aa in (('channels', a_all), ('barriers', b_all)):
    nq = []; nmix = []
    for a in aa[300:310]:
        mx = a == a.max()
        bad = 0; mixed = 0
        for p in unk:
            el = el_of[p]; vals = a[el]
            if vals.min() == vals.max(): continue
            mixed += 1
            if ncomp(el, a == vals.max()) >= 2: bad += 1
        nq.append(bad); nmix.append(mixed)
    print(f'(2) {fam}: interior vertices with mixed K per operator {np.mean(nmix):.0f}; non-quasi-monotone {np.mean(nq):.1f} (min {min(nq)}, max {max(nq)}) of {len(unk)}')
# (3) OA partition of unity at each coarse level, channels and barriers, ids 300-302
for fam, aa in (('channels', a_all), ('barriers', b_all)):
    for i in (300, 301, 302):
        A = H.A_fine(aa[i]).tocsr()
        levs, A0 = oa_hierarchy(H, A, 3, 4)
        Phis = []
        for l, (Ak, Xk) in enumerate(levs):
            nc = Xk.shape[1]; Phis.append(sp.vstack([sp.identity(nc, format='csr'), Xk]).tocsr())
        Psi = sp.identity(len(unk), format='csr'); out = []
        for l in range(3):                       # composite to level k = 2, 1, 0
            Psi = (Psi @ Phis[l]).tocsr(); k = 2 - l
            s = Psi @ np.ones(Psi.shape[1]); dev = np.abs(s - 1)
            supp = [np.count_nonzero(Psi[:, j].toarray()) for j in range(Psi.shape[1])] if Psi.shape[1] < 900 else None
            nz_rows = np.diff(Psi.tocsr().indptr)
            out.append(f'k={k}: dim {Psi.shape[1]}, |sum-1| max {dev.max():.2e}, nodes with |sum-1|>1e-12: {np.count_nonzero(dev > 1e-12)}, '
                       f'min(sum) {s.min():.3f}, max(sum) {s.max():.3f}, overlap max {nz_rows.max()} mean {nz_rows.mean():.1f}, '
                       f'support nodes mean {np.mean(supp):.0f} max {np.max(supp)}, neg entries {np.count_nonzero(Psi.data < -1e-14)}')
        Z = near_kernel_basis(A)
        print(f'(3) {fam} id {i}: near-kernel dim {Z.shape[1]}'); [print('     ', o) for o in out]
