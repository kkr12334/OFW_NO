"""10 Oct: right-hand-side generalisation of the f = 1-trained Session 1 checkpoints (Darcy channels 1e4, L3, k0 2,
--coarse-solve galerkin; held-out data seeds 1-3, test ids 300-399).  STATUS_2026-10-10 item 5 (f = -1) and the
review's request for other loads.  Every model was trained with the single load f = 1.  For each test operator and load:
u = A^-1 F, u_c = level-k0 Galerkin solve of the SAME load, r = F - A u_c, and the one-pass corrections of
  * oa-vcb (trained on f = 1, driver inference path: float64 r / U_scale, A64),
  * its classical member (op.-adapted V(4,4) Chebyshev + deflation + Galerkin output) and the plain op.-adapted V(4,4).
Errors: relative energy error of u_c + correction, ||u - x||_A / ||u||_A (the paper's metric), paired G / W against the
member, and the two-grid PCG iterations from u_c needed to reach oa-vcb's error (energy only, per sample).
Loads (element-wise constant, assembled like the driver's load_vec):
  f = 1          training load (sanity: reproduces the stored record)
  f = -1         sign flip.  Linear solvers and the member are odd maps; the network is positively homogeneous only
  f = 3          positive scaling (sanity: identical to f = 1 by positive homogeneity)
  session-2 random   the --load random family of Session 2 (seeded exactly as the driver: 777000 + 100000 ds + i), i.e.
                 the loads the random-load OA-VCB (G 6.27 on seed 1) was trained / tested on
  smooth +-      sum_{p,q<=4} K_pq sin((p+1) pi x) sin((q+1) pi y), K ~ N(0,1)  (both signs, no constant)
  point source   Gaussian, width 0.03, random centre in [0.15, 0.85]^2
usage: python diagnostics_cost/rhs_generalisation_s1.py data_seed ckpt [results.jsonl] [n_test]"""
import sys, os, json, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup, load_net, apply, energy
from scipy.sparse.linalg import spsolve, factorized
from oa_vcycle import oa_hierarchy, near_kernel_basis, classical_vcycle, classical_vcycle_proj
torch.set_num_threads(2)
DS, CK = int(sys.argv[1]), sys.argv[2]
RES = sys.argv[3] if len(sys.argv) > 3 and sys.argv[3] != '-' else None
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 100
S = setup(3, 2, DS); H, L, P = S['H'], S['L'], S['P']; PT = P.T.tocsr()
te = S['te'][:NT]
xy = H.coords[H.elem]
area = 0.5 * np.abs((xy[:, 1, 0] - xy[:, 0, 0]) * (xy[:, 2, 1] - xy[:, 0, 1]) - (xy[:, 2, 0] - xy[:, 0, 0]) * (xy[:, 1, 1] - xy[:, 0, 1]))
cx, cy = xy.mean(1).T


def assemble(fe):
    Fv = np.zeros(len(H.coords)); m = H.keep
    for j in range(3): np.add.at(Fv, H.elem[m, j], (fe * area)[m] / 3.0)
    return Fv[H.nodes[L]]


assert np.allclose(assemble(np.ones(H.n_elem)), H.F)


def loads(i):
    g = np.random.default_rng(777_000 + 100_000 * DS + i)          # = train_oa_layer.load_vec(..., i) with --load random
    K = g.normal(size=(4, 4))
    fe = g.uniform(-1, 1) + sum(K[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) / (1 + p + q)
                                for p in range(4) for q in range(4))
    if g.uniform() < 0.5:
        x0 = g.uniform(0.1, 0.9, 2); w = g.uniform(0.02, 0.08)
        fe = fe + g.uniform(2, 10) * np.exp(-((cx - x0[0]) ** 2 + (cy - x0[1]) ** 2) / (2 * w ** 2))
    h = np.random.default_rng(5_000 + 100_000 * DS + i)
    K2 = h.normal(size=(4, 4))
    sm = sum(K2[p, q] * np.sin((p + 1) * np.pi * cx) * np.sin((q + 1) * np.pi * cy) for p in range(4) for q in range(4))
    c0 = h.uniform(0.15, 0.85, 2); pt = np.exp(-((cx - c0[0]) ** 2 + (cy - c0[1]) ** 2) / (2 * 0.03 ** 2))
    return {'f = 1 (training)': H.F, 'f = -1': -H.F, 'f = 3': 3 * H.F, 'session-2 random': assemble(fe),
            'smooth +-': assemble(sm), 'point source': assemble(pt)}


D, net, st = load_net(S, CK, te)
names = list(loads(int(te[0])))
E = {n: {'net': [], 'mem': [], 'v44': [], 'uc': [], 'pcg': []} for n in names}
Rall = {n: [] for n in names}; Uall = {n: [] for n in names}; UCall = {n: [] for n in names}
t0 = time.time()
for j, i in enumerate(te):
    A = S['A'][i]; levs, A0 = oa_hierarchy(H, A, S['nl'], 4); Z = near_kernel_basis(A); solc = factorized((PT @ A @ P).tocsc())
    dinv = 1 / A.diagonal(); Pt = H.P[L].tocsr(); soltg = factorized((Pt.T @ A @ Pt).tocsc())
    def tg(r):
        y = (2 / 3) * dinv * r; y = y + Pt @ soltg(Pt.T @ (r - A @ y)); return y + (2 / 3) * dinv * (r - A @ y)
    for n, F in loads(int(i)).items():
        u = spsolve(A.tocsc(), F); uc = P @ solc(PT @ F); r = F - A @ uc
        Rall[n].append(r); Uall[n].append(u); UCall[n].append(uc)
        E[n]['mem'].append(energy(A, uc + classical_vcycle_proj(levs, A0, r, Z=Z, nu=4), u))
        E[n]['v44'].append(energy(A, uc + classical_vcycle(levs, A0, r, nu=4), u))
        E[n]['uc'].append(energy(A, uc, u))
        x = uc.copy(); rr = r.copy(); z = tg(rr); p = z.copy(); rz = rr @ z; h = [E[n]['uc'][-1]]
        for _ in range(16):
            Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; rr = rr - al * Ap
            z = tg(rr); rz2 = rr @ z; p = z + (rz2 / rz) * p; rz = rz2; h.append(energy(A, x, u))
        E[n]['pcg'].append(h)
for n in names:
    C = apply(net, D, np.array(Rall[n]), S['U_scale'])
    E[n]['net'] = [energy(S['A'][i], uc + c, u) for i, uc, c, u in zip(te, UCall[n], C, Uall[n])]
    # plain cycle output of the same checkpoint (no Galerkin output at inference), for the factorial reading
    Cp = apply(net, D, np.array(Rall[n]), S['U_scale'], plain=True)
    E[n]['plain'] = [energy(S['A'][i], uc + c, u) for i, uc, c, u in zip(te, UCall[n], Cp, Uall[n])]
chk = ''
if RES:
    for l in open(RES):
        r = json.loads(l)
        if r['model'] == 'oa-vcb' and json.loads(r['cfg']).get('data_seed', 0) == DS:
            es = np.array(r['result']['en_s'])[:len(te)]; chk = f"  (f = 1 reproduces the stored record to max rel diff {np.max(np.abs(np.array(E[names[0]]['net']) / es - 1)):.1e})"
print(f"data seed {DS}, {len(te)} test operators, checkpoint {os.path.basename(CK)} (code {st.get('code')}, trained on f = 1){chk}; {time.time() - t0:.0f}s")
print(f"{'load':<20}{'u_c alone':>10}{'oa-vcb':>10}{'(plain)':>10}{'member':>10}{'V(4,4)':>10}   member/oa-vcb: G [95% CI]  W   worst   PCG its to oa-vcb: median (range)")
out = {}
for n in names:
    e = {k: np.array(v) for k, v in E[n].items()}
    lr = np.log(e['mem'] / e['net']); G = np.exp(lr.mean())
    b = np.random.default_rng(0).integers(0, len(lr), (5000, len(lr))); ci = np.exp(np.percentile(lr[b].mean(1), [2.5, 97.5]))
    hit = [int(np.nonzero(np.array(h) <= t)[0][0]) if (np.array(h) <= t).any() else 17 for h, t in zip(e['pcg'], e['net'])]
    nw = int(np.sum(e['net'] > e['uc'] * (1 + 1e-6)))
    print(f"{n:<20}{e['uc'].mean():>10.3e}{e['net'].mean():>10.3e}{e['plain'].mean():>10.3e}{e['mem'].mean():>10.3e}{e['v44'].mean():>10.3e}"
          f"   {G:6.2f} [{ci[0]:.2f}, {ci[1]:.2f}]  {np.mean(e['mem'] > e['net']):.2f}  {np.exp(lr.min()):.2f}"
          f"   {np.median(hit):.0f} ({min(hit)}-{max(hit)})" + (f"   oa-vcb worse than u_c on {nw}" if nw else ''))
    out[n] = {k: (v.tolist() if k != 'pcg' else None) for k, v in e.items()}
    out[n]['pcg_its_to_net'] = hit
json.dump(dict(data_seed=DS, ckpt=os.path.basename(CK), code=st.get('code'), sample_ids=[int(i) for i in te], loads=out),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'rhs_generalisation_s1_seed{DS}.json'), 'w'))
