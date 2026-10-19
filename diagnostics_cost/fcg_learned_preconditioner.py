"""Feasibility (3 Oct): oa-ll-op as the smoother of a two-level PRECONDITIONER inside flexible CG.
Darcy, perforated L2, channels 1e4, k0 1 (Galerkin).  A small oa-ll-op is trained on CPU exactly on the error
equation (input D^-1 r / sigma, operator weights a_ij / a_ii, output x sigma; stiffness feature off, which made no
difference in the 28 Sep ablation), then used as
    M(r) = x_c + sigma(r') N(a, r'/sigma),   x_c = P A_c^-1 P^T r,   r' = r - A x_c          (one coarse solve)
in flexible CG (full orthogonalisation; M is nonlinear).  Compared with CG preconditioned by the symmetric two-grid cycle
(omega 0.6 Jacobi) and by AMG (pyamg SA).  Two training sets: residuals of the coarse solve only (= the surrogate
setting), and augmented with residuals from two-grid PCG iterates (the residuals a preconditioner actually sees)."""
import sys, os, time, warnings, numpy as np, torch
warnings.filterwarnings('ignore'); torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', 2)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant
import pyamg
L, k0, N, NTR, EP = 2, 1, 300, 225, int(os.environ.get('EPOCHS', 100))
H = NestedHierarchy(L, coarse=perforated_square(n=9)); F = H.F; nI = len(H.nodes[L])
P = H.P[L].tocsr(); PT = P.T.tocsr(); rng = np.random.default_rng(1000)
MG = MeshHierarchyGeometry(H, 'cpu'); src, dst = np.array(MG.fine['src'].numpy()), np.array(MG.fine['dst'].numpy())
kp = H.keep
def nodal_loga(a):
    s_ = np.zeros(len(H.coords)); c_ = np.zeros(len(H.coords)); la = np.log(a[kp])
    for j in range(3): np.add.at(s_, H.elem[kp, j], la); np.add.at(c_, H.elem[kp, j], 1)
    return (s_ / np.maximum(c_, 1))[H.nodes[L]]
xy = H.coords[H.nodes[L]]; xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
S = []
t0 = time.time()
for i in range(N):
    a = field_channels(H, 1e4, rng); A = H.A_fine(a).tocsr(); u = spsolve(A.tocsc(), F)
    solc = factorized((PT @ A @ P).tocsc()); dg = A.diagonal()
    S.append(dict(A=A, u=u, solc=solc, dg=dg, la=nodal_loga(a), w=np.asarray(A[dst, src]).ravel() / dg[dst]))
LA = np.array([s['la'] for s in S]); mu, sd = LA[:NTR].mean(), LA[:NTR].std()
print(f"data: {N} samples, {nI} unknowns, {time.time() - t0:.0f}s", flush=True)

def coarse_correct(s, r):
    xc = P @ s['solc'](PT @ r); return xc, r - s['A'] @ xc
def tg_apply(s, r, om=0.6):
    A, dinv = s['A'], 1 / s['dg']
    x = om * dinv * r; x = x + P @ s['solc'](PT @ (r - A @ x)); return x + om * dinv * (r - A @ x)

def residual_pairs(augment):
    """(sample index, r') with r' a post-coarse-correction residual and target e' = A^-1 r'."""
    pairs = []
    for i in range(NTR):
        s = S[i]; _, r1 = coarse_correct(s, F); pairs.append((i, r1))
        if augment:                                  # residuals seen inside two-grid PCG from 0
            x = np.zeros(nI); r = F.copy(); z = tg_apply(s, r); p = z.copy(); rz = r @ z
            for it in range(6):
                Ap = s['A'] @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
                _, rr = coarse_correct(s, r); pairs.append((i, rr))
                z = tg_apply(s, r); rz2 = r @ z; p = z + rz2 / rz * p; rz = rz2
    return pairs

def feats(i, r):
    z = r / S[i]['dg']; sig = np.sqrt(np.mean(z ** 2)) + 1e-300
    X = np.column_stack([(S[i]['la'] - mu) / sd, xy, np.zeros(nI), z / sig])
    return X, sig

def train(augment, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    pairs = residual_pairs(augment)
    Xs, Ys, Ws, As = [], [], [], []
    for i, r in pairs:
        X, sig = feats(i, r); e = spsolve(S[i]['A'].tocsc(), r)
        Xs.append(X); Ys.append(e / sig); Ws.append(S[i]['w']); As.append(i)
    Xs = torch.tensor(np.array(Xs), dtype=torch.float32); Ys = torch.tensor(np.array(Ys), dtype=torch.float32)
    Ws = torch.tensor(np.array(Ws), dtype=torch.float32)
    At = [torch.sparse_csr_tensor(torch.tensor(S[i]['A'].indptr), torch.tensor(S[i]['A'].indices),
                                  torch.tensor(S[i]['A'].data, dtype=torch.float32), (nI, nI)) for i in range(NTR)]
    w = 2
    while sum(p.numel() for p in MeshLiftingNO(MG, 5, width=w + 1, **variant('oa-ll-op')).parameters()) <= 22000: w += 1
    m = MeshLiftingNO(MG, 5, width=w, **variant('oa-ll-op'))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-5); nb = int(np.ceil(len(pairs) / 8))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=EP * nb); g = np.random.default_rng(seed)
    t0 = time.time()
    for ep in range(EP):
        tot = 0.0
        for idx in np.array_split(g.permutation(len(pairs)), nb):
            m.set_op(Ws[idx]); p = m(Xs[idx]); t = Ys[idx]; d = p - t
            en = torch.stack([torch.sqrt((d[k] * (At[As[j]] @ d[k])).sum() / (t[k] * (At[As[j]] @ t[k])).sum()) for k, j in enumerate(idx)])
            loss = 0.5 * (en.mean() + (torch.linalg.norm(d, dim=1) / torch.linalg.norm(t, dim=1)).mean())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); sch.step(); tot += loss.item()
        if ep % 20 == 0 or ep == EP - 1: print(f"  [{'augmented' if augment else 'coarse-residual'}] ep {ep + 1}/{EP} loss {tot / nb:.3e} ({time.time() - t0:.0f}s)", flush=True)
    m.eval(); return m, sum(p.numel() for p in m.parameters())

def net_apply(m, i, r):
    xc, r1 = coarse_correct(S[i], r); X, sig = feats(i, r1)
    with torch.no_grad():
        m.set_op(torch.tensor(S[i]['w'][None], dtype=torch.float32))
        out = m(torch.tensor(X[None], dtype=torch.float32))[0].numpy().astype(float)
    return xc + sig * out

def fcg(i, Minv, maxit=40, tol=1e-6):
    s = S[i]; A, u = s['A'], s['u']; en = lambda v: np.sqrt(v @ (A @ v)); eu = en(u)
    x = np.zeros(nI); r = F.copy(); Ps, APs = [], []; hist = [1.0]; first = None
    for it in range(maxit):
        z = Minv(r); p = z.copy()
        for pj, apj in zip(Ps, APs): p -= (apj @ z) / (apj @ pj) * pj      # full A-orthogonalisation (flexible)
        Ap = A @ p; al = (p @ r) / (p @ Ap); x = x + al * p; r = r - al * Ap; Ps.append(p); APs.append(Ap)
        hist.append(en(x - u) / eu)
        if hist[-1] < tol: break
    return np.array(hist)

te = np.arange(NTR, N)
res = {}
for name, mk in [('two-grid', lambda i: (lambda r: tg_apply(S[i], r))),
                 ('AMG (SA)', lambda i, ML={}: (lambda r: (ML.setdefault(i, pyamg.smoothed_aggregation_solver(S[i]['A']).aspreconditioner()) @ r)))]:
    res[name] = [fcg(i, mk(i)) for i in te]
for aug in (False, True):
    m, npar = train(aug)
    one = []
    for i in te:                                    # one-pass accuracy (the surrogate number) from u_c
        xc, r1 = coarse_correct(S[i], F); X, sig = feats(i, r1)
        x = net_apply(m, i, F); A = S[i]['A']; u = S[i]['u']; e = x - u
        one.append((np.linalg.norm(e) / np.linalg.norm(u), np.sqrt(e @ (A @ e) / (u @ (A @ u)))))
    print(f"  one-pass (surrogate) test error, {'augmented' if aug else 'coarse-residual'} training, {npar:,} params: "
          f"rel-L2 {np.mean(one, 0)[0]:.3e}  energy {np.mean(one, 0)[1]:.3f}", flush=True)
    res[f"learned ({'augmented' if aug else 'coarse-residual'})"] = [fcg(i, (lambda r, i=i: net_apply(m, i, r))) for i in te]
print("\nflexible CG from 0 (energy error relative to ||u||_A), 75 test samples:")
for name, hs in res.items():
    its = [len(h) - 1 if h[-1] < 1e-6 else np.nan for h in hs]
    j = lambda k: np.mean([h[min(k, len(h) - 1)] for h in hs])
    print(f"  {name:<28} its to 1e-6: median {np.nanmedian(its):.0f} [max {np.nanmax(its):.0f}], reached {np.sum(~np.isnan(its))}/75;  "
          f"error after 1 / 2 / 4 its: {j(1):.2e} / {j(2):.2e} / {j(4):.2e}", flush=True)
