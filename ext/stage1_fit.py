"""Stage 1 fitting on set T (seed-1 training operators 0-299), amendment 1 A5.3; development only.
  CHEB-T: bound in {gershgorin, lanczos x1.1} x q in {4, 10, 30}, smallest mean log relative energy error on T.
  POLY-S: pre and post degree-3 correction polynomials (Chebyshev basis on [0, beta], 1/beta normalisation), shared by
          levels; beta = max over levels and T operators of the Lanczos x1.1 estimate.
  POLY-L: pre and post coefficients per level, basis scaled per level and operator by its Lanczos x1.1 estimate.
  Objective: mean over T of log relative energy error of the final projected output (common Galerkin without the
  acceptance backstop, which never fired on D); L-BFGS float64 autograd, starts: CHEB-equivalent and zero polynomial;
  stop at gradient inf-norm < 1e-6 or 500 iterations.
usage: python -I ext/stage1_fit.py [n_train]  ->  ext/dev/stage1_fit.json (+ coefficients for stage1_dev_run.py)"""
import os, sys, json, time, hashlib, numpy as np, torch, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy
from oa_vcycle import near_kernel_basis
from stage1_core import Hier, galerkin, make_cheb, one_pass, lam_lanczos, lam_gershgorin
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
NT = int(sys.argv[1]) if len(sys.argv) > 1 else 300
FAMS = sys.argv[2].split(',') if len(sys.argv) > 2 else ['channels', 'barriers']
class _SpMV(torch.autograd.Function):
    """y = M x with scipy CSR (float64); backward: M^T g.  M, MT are scipy matrices captured by the wrapper."""
    @staticmethod
    def forward(ctx, x, M, MT):
        ctx.MT = MT; return torch.from_numpy(M @ x.detach().numpy())
    @staticmethod
    def backward(ctx, g):
        return torch.from_numpy(ctx.MT @ g.detach().numpy()), None, None


class SpOp:
    def __init__(self, M):
        self.M = M.tocsr(); self.MT = M.T.tocsr()
    def __matmul__(self, x):
        return _SpMV.apply(x, self.M, self.MT)


T_ = SpOp


def cheb_coeffs(q=10):
    """Chebyshev basis coefficients (with 1/s normalisation) of the degree-3 correction polynomial of the CHEB smoother
    on [s/q, s]: x = p(B) D^-1 b with 1 - t p(t) = T4((th - t)/de) / T4(th/de); scale-free (s = 1)."""
    lo, hi = 1 / q, 1.0; th, de = (hi + lo) / 2, (hi - lo) / 2
    t = np.cos(np.pi * (np.arange(4) + 0.5) / 4) * 0.5 + 0.5 + 1e-3        # 4 nodes in (0, 1)
    T4 = lambda z: np.cos(4 * np.arccos(np.clip(z, -1, 1))) if np.all(np.abs(z) <= 1) else np.polynomial.chebyshev.chebval(z, [0, 0, 0, 0, 1])
    q_ = np.polynomial.chebyshev.chebval((th - t) / de, [0, 0, 0, 0, 1]) / np.polynomial.chebyshev.chebval(th / de, [0, 0, 0, 0, 1])
    p = (1 - q_) / t                                                        # p(t) at the nodes (s = 1)
    V = np.polynomial.chebyshev.chebvander(2 * t - 1, 3)                   # basis T_j(2t - 1)
    return np.linalg.solve(V, p)


class Batch:
    """all training operators of one family, batched: block-diagonal sparse level matrices and transfers (torch, float64),
    vectors as (B, n_l) arrays flattened for the sparse products.  Same arithmetic as the per-operator numpy path."""
    def __init__(self, hrs, Rs, Zs, E_true, unorms, scales):
        self.B = B = len(hrs); self.nl = hrs[0].nl
        self.n = [Ak.shape[0] for Ak, _ in hrs[0].levs]; self.nc = [X.shape[1] for _, X in hrs[0].levs]
        bd = lambda Ms: T_(sp.block_diag(Ms, format='csr'))
        self.A = [bd([h.levs[l][0] for h in hrs]) for l in range(self.nl)]
        self.Xs = [bd([h.levs[l][1] for h in hrs]) for l in range(self.nl)]
        self.XT = [bd([h.levs[l][1].T for h in hrs]) for l in range(self.nl)]
        self.dinv = [torch.tensor(np.stack([h.dinv[l] for h in hrs])) for l in range(self.nl)]
        self.A0inv = torch.tensor(np.stack([np.linalg.inv(h.A0.toarray()) for h in hrs]))
        x0s = [galerkin(h.A, Z, r) if Z.shape[1] else np.zeros_like(r) for h, Z, r in zip(hrs, Zs, Rs)]
        self.bb = torch.tensor(np.stack([r - h.A @ x0 for h, r, x0 in zip(hrs, Rs, x0s)]))
        self.target = torch.tensor(np.stack([e - x0 for e, x0 in zip(E_true, x0s)]))
        km = max(1, max(Z.shape[1] for Z in Zs)); Zp = np.zeros((B, self.n[0], km))
        for k, Z in enumerate(Zs): Zp[k, :, :Z.shape[1]] = Z
        self.Z = torch.tensor(Zp); self.unorm = torch.tensor(unorms); self.scl = torch.tensor(np.array(scales))   # (B, nl)

    def mv(self, l, x):                                                       # x: (B, n_l)
        return (self.A[l] @ x.reshape(-1)).reshape(self.B, -1)

    def smooth(self, c, l, rhs, s):                                           # s: (B,) scale per operator
        di = self.dinv[l]; s = s[:, None]; Bv = lambda y: di * self.mv(l, y)
        v = di * rhs; t0 = v; t1 = 2 / s * Bv(v) - v; x = c[0] * t0 + c[1] * t1
        for j in (2, 3):
            t2 = 2 * (2 / s * Bv(t1) - t1) - t0; x = x + c[j] * t2; t0, t1 = t1, t2
        return x / s

    def cycle(self, l, rhs, cpre, cpost, sc, parts=False):
        nc = self.nc[l]
        x = self.smooth(cpre[l], l, rhs, sc[:, l]); r = rhs - self.mv(l, x)
        rc = r[:, :nc] + (self.XT[l] @ r[:, nc:].reshape(-1)).reshape(self.B, nc)
        ec = self.cycle(l + 1, rc, cpre, cpost, sc) if l + 1 < self.nl else torch.bmm(self.A0inv, rc[..., None])[..., 0]
        xc = torch.cat([ec, (self.Xs[l] @ ec.reshape(-1)).reshape(self.B, -1)], 1)
        xp = self.smooth(cpost[l], l, rhs - self.mv(l, x + xc), sc[:, l])
        return (x, xc, xp) if parts else x + xc + xp

    def logerr(self, cpre, cpost, sc):
        """mean over operators of log relative energy error of the final projected output (common Galerkin, no backstop)."""
        parts = self.cycle(0, self.bb, cpre, cpost, sc, parts=True)
        W = torch.cat([torch.stack(parts, -1), self.Z], -1)                   # (B, n, m); zero Z columns masked below
        AW = torch.stack([self.mv(0, W[..., k]) for k in range(W.shape[-1])], -1)
        G = W.transpose(1, 2) @ AW; G = 0.5 * (G + G.transpose(1, 2)); g = (W.transpose(1, 2) @ self.bb[..., None])[..., 0]
        dg = torch.diagonal(G, dim1=1, dim2=2); live = dg > 1e-30 * dg.amax(1, keepdim=True)
        sc_ = torch.where(live, torch.where(live, dg, torch.ones_like(dg)).rsqrt(), torch.zeros_like(dg))
        Gs = sc_[:, :, None] * G * sc_[:, None, :] + torch.diag_embed((~live).double() + 1e-10)
        y = sc_ * torch.linalg.solve(Gs, sc_ * g)
        e = self.target - (W @ y[..., None])[..., 0]
        return (0.5 * torch.log((e * self.mv(0, e)).sum(1) / self.unorm ** 2)).mean()


def fit(ops, kind, beta, starts):
    best = None; diag = []
    for nm, c0 in starts:
        if kind == 'S': th = torch.tensor(np.concatenate([c0, c0]), requires_grad=True)
        else: th = torch.tensor(np.concatenate([np.tile(c0, 3), np.tile(c0, 3)]), requires_grad=True)
        def unpack(th):
            if kind == 'S': return [th[:4]] * 3, [th[4:]] * 3
            return [th[4 * l:4 * l + 4] for l in range(3)], [th[12 + 4 * l:12 + 4 * l + 4] for l in range(3)]
        opt = torch.optim.LBFGS([th], lr=1, max_iter=20, tolerance_grad=1e-6, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')
        its = [0]
        def closure():
            opt.zero_grad(); cp, cq = unpack(th)
            L = ops.logerr(cp, cq, ops.scl * 0 + beta if kind == 'S' else ops.scl); L.backward(); its[0] += 1; return L
        t0 = time.time(); last = None
        for outer in range(25):                                         # <= 500 inner iterations
            L = opt.step(closure); gn = float(th.grad.abs().max())
            if gn < 1e-6 or (last is not None and abs(last - float(L)) < 1e-12): break
            last = float(L)
        with torch.no_grad(): cp, cq = unpack(th); Lf = float(ops.logerr(cp, cq, ops.scl * 0 + beta if kind == 'S' else ops.scl))
        d = dict(start=nm, objective=Lf, grad_inf=gn, evaluations=its[0], seconds=time.time() - t0,
                 pre=[c.detach().numpy().tolist() for c in cp] if kind == 'L' else cp[0].detach().numpy().tolist(),
                 post=[c.detach().numpy().tolist() for c in cq] if kind == 'L' else cq[0].detach().numpy().tolist())
        diag.append(d); print(f"    POLY-{kind} start {nm}: objective {Lf:.5f} (geo-mean error {np.exp(Lf):.4e}), |grad| {gn:.1e}, {its[0]} evals, {d['seconds']:.0f} s", flush=True)
        if best is None or Lf < best['objective']: best = d
    return best, diag


out = {}; extra = {}
for fam, kw in [(f, {} if f == 'channels' else dict(contrast=1e-4, load='no-low')) for f in FAMS]:
    t0 = time.time(); S = setup(3, 2, 1, **kw); tr = S['tr'][:NT]; H = S['H']
    hrs, Zs, Rs, lz, gz = [], [], [], [], []
    for i in tr:
        A = S['A'][i]; hr = Hier(H, A, S['nl']); hrs.append(hr); Zs.append(near_kernel_basis(A)); Rs.append(S['Fs'][i] - A @ S['UC'][i])
        lz.append([lam_lanczos(Ak) for Ak, _ in hr.levs]); gz.append([lam_gershgorin(Ak) for Ak, _ in hr.levs])
    print(f"\n== {fam}: set T, {len(tr)} operators, set-up {time.time() - t0:.0f} s", flush=True)
    # CHEB-T grid
    grid = {}
    for bound, LS in (('gershgorin', gz), ('lanczos', lz)):
        for q in (4, 10, 30):
            le = [np.log(energy(S['A'][i], S['UC'][i] + one_pass(hr, r, Z, make_cheb(ls, q=q), make_cheb(ls, q=q)), S['U'][i]))
                  for i, hr, r, Z, ls in zip(tr, hrs, Rs, Zs, LS)]
            grid[f'{bound},q={q}'] = float(np.mean(le)); print(f"    CHEB-T {bound:<10} q={q:<3} mean log error {np.mean(le):.5f}", flush=True)
    bestT = min(grid, key=grid.get); b_, q_ = bestT.split(',q=')
    # POLY fits
    beta = float(np.max(lz)); c0 = cheb_coeffs(10)
    ops = Batch(hrs, Rs, Zs, [S['U'][i] - S['UC'][i] for i in tr], [float(np.sqrt(S['U'][i] @ (S['A'][i] @ S['U'][i]))) for i in tr], lz)
    with torch.no_grad():                                               # sanity: CHEB-equivalent start reproduces CHEB on [s/10, s]
        chk = float(ops.logerr([torch.tensor(c0)] * 3, [torch.tensor(c0)] * 3, ops.scl))
    le_cheb_l = float(np.mean([np.log(energy(S['A'][i], S['UC'][i] + one_pass(hr, r, Z, make_cheb(ls, q=10), make_cheb(ls, q=10)), S['U'][i]))
                               for i, hr, r, Z, ls in zip(tr, hrs, Rs, Zs, lz)]))
    print(f"    check: CHEB-equivalent coefficients (torch, no backstop) {chk:.6f} vs CHEB lanczos q=10 (numpy) {le_cheb_l:.6f}", flush=True)
    starts = [('CHEB-equivalent', c0), ('zero', np.zeros(4))]
    bS, dS = fit(ops, 'S', beta, starts); bL, dL = fit(ops, 'L', beta, starts)
    out[fam] = dict(n_train=len(tr), cheb_t_grid=grid, cheb_t_selected=bestT, beta=beta, cheb_equivalent_check=[chk, le_cheb_l],
                    poly_s=dict(best=bS, runs=dS), poly_l=dict(best=bL, runs=dL), seconds=time.time() - t0)
    extra[fam] = {'CHEB-T': dict(bound=b_, q=int(q_)), 'POLY-S': dict(beta=beta, pre=bS['pre'], post=bS['post']),
                  'POLY-L': dict(pre=bL['pre'], post=bL['post'])}
    json.dump(dict(fit=out[fam], methods=extra[fam]), open(os.path.join(HERE, 'dev', f'stage1_fit_{fam}.json'), 'w'), indent=1)
    del ops, hrs, Zs, Rs, S
# merge whatever per-family files exist
allf = {f: json.load(open(os.path.join(HERE, 'dev', f'stage1_fit_{f}.json'))) for f in ('channels', 'barriers')
        if os.path.exists(os.path.join(HERE, 'dev', f'stage1_fit_{f}.json'))}
if len(allf) == 2:
    extra = {f: v['methods'] for f, v in allf.items()}; out = {f: v['fit'] for f, v in allf.items()}
    out['coefficients_sha256'] = hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()
    json.dump(out, open(os.path.join(HERE, 'dev', 'stage1_fit.json'), 'w'), indent=1)
    json.dump(extra, open(os.path.join(HERE, 'dev', 'stage1_fitted_methods.json'), 'w'), indent=1, sort_keys=True)
    print('coefficients sha256', out['coefficients_sha256'])
