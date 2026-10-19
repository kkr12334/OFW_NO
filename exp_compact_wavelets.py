"""Step 2 of the formulation's next-experiments list (section 10):
D'Heedene (2005, sections 5.2.2-5.3) compact, EXACTLY operator-orthogonal P1
wavelets via lifting + stable completion, on the project's triangle hierarchy.

For every interior new node m (= midpoint of the coarse edge (k1, k2)):
  support      Omega = supp(phi_k1) U supp(phi_k2)       (coarse hats)
  unknowns     g: primitive wavelets (fine hats of new nodes) inside Omega
               s: coarse hats inside Omega
  constraints  a(psi, phi_l) = 0 for every coarse hat overlapping Omega
  psi_m        the null-space vector closest to 'the primitive wavelet at m',
               normalised so g_m = 1
Then Psi = E G + P S with G square (stable-completion matrix).

Questions answered, per level and per coefficient field:
  * does every m get a wavelet?                (null-space dimension, failures)
  * exact decoupling?                          max |P^T A Psi| / max |A|
  * is the family a basis?                     cond(G)  (T invertible iff G is)
  * sparsity                                   nnz per wavelet vs exact lifting
  * conditioning of the level block            kappa(D), Jacobi-scaled kappa
  * does the exact identity A^-1 = T D^-1 T^T hold with this T? (full check)

usage:
  python exp_compact_wavelets.py                  # level 4, const + grf + channels
  python exp_compact_wavelets.py --level 5
"""
import argparse, sys, time
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from scipy.sparse.linalg import splu, spsolve
sys.path.insert(0, '.')
from ofwno_core import MeshHierarchy
from oa_lifting import NestedHierarchy, make_field, jacobi_scaled, energy_err

ap = argparse.ArgumentParser()
ap.add_argument('--level', type=int, default=4)
ap.add_argument('--fields', nargs='+', default=['const', 'grf', 'blocks8', 'channels'])
ap.add_argument('--contrast', type=float, default=1e4)
ap.add_argument('--seed', type=int, default=11)
ap.add_argument('--support', choices=['edge', 'edge+ring'], default='edge',
                help="edge: supp(phi_k1) U supp(phi_k2) (D'Heedene); edge+ring: also the hats of all coarse neighbours of k1, k2")
args = ap.parse_args()

H = NestedHierarchy(args.level)
L = args.level
rng = np.random.default_rng(args.seed)
bset = set(int(b) for b in H.boundary)

# element lists per level (same deterministic refinement => same node ids)
elems = {}
for k in range(L + 1):
    m = MeshHierarchy(); m.refine_to_level(k)
    elems[k] = np.asarray([np.asarray(e, int) for e in m.elements])
    assert len(m.coords) <= len(H.coords) and np.allclose(np.asarray(m.coords), H.coords[:len(m.coords)])


def level_structures(k):
    """topology for the pair (k-1 coarse, k fine); fine element e has parent e // 4."""
    ce, fe = elems[k - 1], elems[k]
    node_ce = {}                                     # coarse node -> coarse elements
    for i, el in enumerate(ce):
        for v in el: node_ce.setdefault(int(v), set()).add(i)
    node_fe = {}                                     # fine node -> fine elements
    for i, el in enumerate(fe):
        for v in el: node_fe.setdefault(int(v), set()).add(i)
    return node_ce, node_fe


def compact_level(k, Ak, max_dense=6000):
    """Returns Ac, G, S, Psi, null-space dims, number of missing wavelets."""
    P, E = H.P[k], H.E[k]
    Ac = (P.T @ Ak @ P).tocsr()
    Bk = (P.T @ Ak @ E).tocsr()
    coarse = H.nodes[k - 1]; new = H.new[k]
    mk = len(new)
    cpos = {int(g): i for i, g in enumerate(coarse)}
    npos = {int(g): i for i, g in enumerate(new)}
    node_ce, node_fe = level_structures(k)
    cand_G, cand_S, nulldims = [], [], []          # candidate wavelets (sparse columns)
    for t, mg in enumerate(new):
        k1, k2 = H.mesh.node_parents[int(mg)]
        cset = node_ce[k1] | node_ce[k2]
        if args.support == 'edge+ring':
            ring = {int(v) for c in cset for v in elems[k - 1][c]}
            cset = set().union(*[node_ce[v] for v in ring])
        fset = {4 * c + j for c in cset for j in range(4)}
        m_set = sorted({npos[int(v)] for e in fset for v in elems[k][e]
                        if int(v) in npos and node_fe[int(v)] <= fset})
        k_set = sorted({cpos[int(v)] for c in cset for v in elems[k - 1][c]
                        if int(v) in cpos and node_ce[int(v)] <= cset})
        l_set = sorted({cpos[int(v)] for c in cset for v in elems[k - 1][c] if int(v) in cpos})
        nm = len(m_set)
        if not l_set:                                  # no coarse interior dofs touch Omega
            N = np.eye(nm + len(k_set))
        else:
            Mm = np.hstack([Bk[l_set][:, m_set].toarray(), Ac[l_set][:, k_set].toarray()])
            _, sv, Vt = np.linalg.svd(Mm, full_matrices=True)
            rank = int((sv > 1e-10 * sv.max()).sum()) if sv.size else 0
            N = Vt[rank:].T
        nulldims.append(N.shape[1])
        for q in range(N.shape[1]):
            g, sv_ = N[:nm, q], N[nm:, q]
            ng = np.linalg.norm(g)
            if ng < 1e-10: continue                   # pure-coarse combination: not a wavelet
            cand_G.append((np.asarray(m_set), g / ng))
            cand_S.append((np.asarray(k_set, int), sv_ / ng))
    nc = len(cand_G)
    if mk > max_dense:
        raise MemoryError(f"level {k}: m_k={mk} > max_dense; selection uses a dense QR")
    Gall = np.zeros((mk, nc))
    for j, (idx, g) in enumerate(cand_G): Gall[idx, j] = g
    # greedy selection of a well-conditioned complete family
    _, R, piv = sla.qr(Gall, mode='economic', pivoting=True)
    dR = np.abs(np.diag(R))
    rank = int((dR > 1e-8 * dR[0]).sum())
    chosen = piv[:min(rank, mk)]
    fails = mk - len(chosen)
    rows_G, cols_G, vals_G, rows_S, cols_S, vals_S = [], [], [], [], [], []
    for t, j in enumerate(chosen):
        idx, g = cand_G[j]; kidx, s = cand_S[j]
        rows_G += list(idx); cols_G += [t] * len(idx); vals_G += list(g)
        rows_S += list(kidx); cols_S += [t] * len(kidx); vals_S += list(s)
    if fails:
        # complete with EXACT (global, non-compact) lifted primitive wavelets in the
        # directions the compact family misses: psi = E e + P C e, C = -Ac^-1 B
        Gc = Gall[:, chosen]
        Q, _ = np.linalg.qr(Gc, mode='complete')
        miss = Q[:, len(chosen):]                     # orthonormal complement in G-space
        if Ac.shape[0]:
            Cm = -splu(Ac.tocsc()).solve(Bk @ miss)
        else:
            Cm = np.zeros((0, fails))
        for q in range(fails):
            t = len(chosen) + q
            nz = np.flatnonzero(np.abs(miss[:, q]) > 1e-14)
            rows_G += list(nz); cols_G += [t] * len(nz); vals_G += list(miss[nz, q])
            nz = np.flatnonzero(np.abs(Cm[:, q]) > 1e-14)
            rows_S += list(nz); cols_S += [t] * len(nz); vals_S += list(Cm[nz, q])
    G = sp.csr_matrix((vals_G, (rows_G, cols_G)), shape=(mk, mk))
    S = sp.csr_matrix((vals_S, (rows_S, cols_S)), shape=(len(coarse), mk))
    S.data[np.abs(S.data) < 1e-14] = 0; S.eliminate_zeros()
    G.data[np.abs(G.data) < 1e-14] = 0; G.eliminate_zeros()
    Psi = (E @ G + P @ S).tocsc()
    return Ac, G, S, Psi, np.array(nulldims), fails


for field in args.fields:
    a = make_field(H, field, args.contrast, rng)
    A = H.A_fine(a).tocsc()
    print(f"\n=== {field}  (level {L}, contrast {args.contrast:.0e})")
    print(f"  {'k':>3}{'m_k':>6}{'null dim min/med/max':>22}{'missing*':>9}{'|P^T A Psi|':>13}"
          f"{'cond G':>10}{'nnz/psi':>9}{'(exact)':>9}{'kappa D':>10}{'Jac kappa':>11}{'(exact lift)':>13}")
    Ak = A
    Ts = {}
    for k in range(L, 0, -1):
        t0 = time.time()
        Ac, G, S, Psi, nd, fails = compact_level(k, Ak)
        P = H.P[k]
        PAP = (P.T @ Ak @ Psi)
        orth = abs(PAP).max() / abs(Ak).max() if PAP.nnz else 0.0
        Gd = G.toarray()
        condG = np.linalg.cond(Gd)
        D = (Psi.T @ Ak @ Psi).toarray()
        try:
            ev = np.linalg.eigvalsh(D); kapD = ev[-1] / ev[0]
            Dj, _ = jacobi_scaled(D); evj = np.linalg.eigvalsh(Dj); kapJ = evj[-1] / evj[0]
        except Exception:
            kapD = kapJ = np.nan
        # exact-lifting comparison
        Cex = (-splu(Ac.tocsc()).solve((P.T @ Ak @ H.E[k]).toarray()) if Ac.shape[0]
               else np.zeros((0, len(H.new[k]))))
        Pex = H.E[k].toarray() + P @ Cex
        nnz_ex = (np.abs(Pex) > 1e-8 * np.abs(Pex).max()).sum() / Pex.shape[1]
        Dex = Pex.T @ (Ak @ Pex); Dex = 0.5 * (Dex + Dex.T)
        evx = np.linalg.eigvalsh(jacobi_scaled(Dex)[0]); kapX = evx[-1] / evx[0]
        print(f"  {k:>3}{len(H.new[k]):>6}{nd.min():>8}{int(np.median(nd)):>7}{nd.max():>7}{fails:>9}"
              f"{orth:>13.1e}{condG:>10.2e}{Psi.nnz / Psi.shape[1]:>9.1f}{nnz_ex:>9.1f}{kapD:>10.2e}{kapJ:>11.2e}{kapX:>13.2e}")
        Ts[k] = (G, S, Psi, D)
        Ak = Ac
    print("  * missing = directions the compact family does not span; filled with exact non-compact lifted wavelets")
    # full identity check with the compact transform (only if every level succeeded)
    if all(np.linalg.cond(Ts[k][0].toarray()) < 1e12 for k in Ts):
        n = A.shape[0]
        cols = []
        up = sp.eye(n, format='csr')
        blocks = []
        for k in range(L, 0, -1):
            cols.append((up @ Ts[k][2]).toarray()); blocks.append(Ts[k][3])
            up = up @ H.P[k]
        cols.append(up.toarray()); blocks.append(Ak.toarray())
        Tm = np.hstack(cols[::-1]); Dl = blocks[::-1]
        u = spsolve(A, H.F)
        Dinv = sla.block_diag(*[np.linalg.inv(D) for D in Dl])
        uh = Tm @ (Dinv @ (Tm.T @ H.F))
        off = Tm.T @ A.toarray() @ Tm
        sizes = [D.shape[0] for D in Dl]; o = np.cumsum([0] + sizes)
        for i in range(len(sizes)): off[o[i]:o[i + 1], o[i]:o[i + 1]] = 0
        print(f"  full check: cond(T) = {np.linalg.cond(Tm):.2e}   cross-level ||.||/||T^T A T|| = "
              f"{np.linalg.norm(off) / np.linalg.norm(Tm.T @ A.toarray() @ Tm):.1e}   "
              f"energy err of T D^-1 T^T F = {energy_err(A, u, uh):.1e}")
    else:
        print("  full check skipped: some level is not a basis (G singular / wavelets missing)")
