"""D2: oracle-fitted polynomial CANDIDATE (protocol v1 §3.4; diagnostic, amendment 1 A5.4).
Per operator: pre and post coefficients per level (3 levels x 2 x 4 = 24 numbers) of the degree-3 correction
polynomial in the Chebyshev basis on [0, s_l] (1/s normalisation, s_l = the CHEB-T* per-level Lanczos x1.1 bound),
optimised against the REFERENCE solution to minimise that operator's log relative energy error of the final projected
output.  Five starts: CHEB-T*-equivalent, POLY-S (re-expressed in the per-level basis), POLY-L, PCG's first-call
coefficients (least-squares fit of PCG(4)'s output at each level/position in the 4-dim polynomial basis, captured in
the PCG cycle), zero.  Each operator is optimised separately (L-BFGS float64 autograd, strong Wolfe; stop at gradient
inf-norm < 1e-6, a change < 1e-12 between 20-iteration rounds, or 500 iterations), and per operator the best of the five
starts is kept.  Reported error: the numpy path (stage1_core.one_pass with make_poly,
common Galerkin with backstop).  Uses the exact solution; no global optimum is certified; called a candidate.
usage: OMP_NUM_THREADS=2 python -I ext/stage1_d2.py SET FAMILY N OUT.json   (SET = D, or C after the pre-confirmation note)"""
import os, sys
for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ.setdefault(v, '2')
import json, time, numpy as np, torch, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path[:0] = [HERE, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy
from oa_vcycle import near_kernel_basis
from stage1_core import Hier, galerkin, one_pass, make_poly, make_cheb, pcg4, lam_lanczos
from stage1_manifest import manifest, Gates
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)


class _SpMV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, M, MT):
        ctx.MT = MT; return torch.from_numpy(M @ x.detach().numpy())
    @staticmethod
    def backward(ctx, g):
        return torch.from_numpy(ctx.MT @ g.detach().numpy()), None, None


class SpOp:
    def __init__(self, M): self.M = M.tocsr(); self.MT = M.T.tocsr()
    def __matmul__(self, x): return _SpMV.apply(x, self.M, self.MT)


def cheb_coeffs(q=10):
    """= stage1_fit.cheb_coeffs (CHEB smoother on [s/q, s] in the scaled Chebyshev basis)."""
    lo, hi = 1 / q, 1.0; th, de = (hi + lo) / 2, (hi - lo) / 2
    t = np.cos(np.pi * (np.arange(4) + 0.5) / 4) * 0.5 + 0.5 + 1e-3
    q_ = np.polynomial.chebyshev.chebval((th - t) / de, [0, 0, 0, 0, 1]) / np.polynomial.chebyshev.chebval(th / de, [0, 0, 0, 0, 1])
    return np.linalg.solve(np.polynomial.chebyshev.chebvander(2 * t - 1, 3), (1 - q_) / t)


def rescale(c, s_from, s_to):
    """coefficients of the same polynomial p(t) = (1/s_from) sum c_j T_j(2t/s_from - 1) in the basis on [0, s_to]."""
    t = (np.cos(np.pi * (np.arange(4) + 0.5) / 4) * 0.5 + 0.5) * s_to
    p = np.polynomial.chebyshev.chebval(2 * t / s_from - 1, c) / s_from
    return np.linalg.solve(np.polynomial.chebyshev.chebvander(2 * t / s_to - 1, 3) / s_to, p)


class Batch:
    """batched cycle with PER-OPERATOR coefficients c[l]: (B, 4); same arithmetic as stage1_fit.Batch."""
    def __init__(self, hrs, Rs, Zs, E_true, unorms, scales):
        self.B = B = len(hrs); self.nl = hrs[0].nl
        self.n = [Ak.shape[0] for Ak, _ in hrs[0].levs]; self.nc = [X.shape[1] for _, X in hrs[0].levs]
        bd = lambda Ms: SpOp(sp.block_diag(Ms, format='csr'))
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
        self.Z = torch.tensor(Zp); self.unorm = torch.tensor(unorms); self.scl = torch.tensor(np.array(scales))

    def mv(self, l, x): return (self.A[l] @ x.reshape(-1)).reshape(self.B, -1)

    def smooth(self, c, l, rhs, s):
        di = self.dinv[l]; s = s[:, None]; Bv = lambda y: di * self.mv(l, y)
        v = di * rhs; t0 = v; t1 = 2 / s * Bv(v) - v; x = c[:, 0:1] * t0 + c[:, 1:2] * t1
        for j in (2, 3):
            t2 = 2 * (2 / s * Bv(t1) - t1) - t0; x = x + c[:, j:j + 1] * t2; t0, t1 = t1, t2
        return x / s

    def cycle(self, l, rhs, cpre, cpost, parts=False):
        nc = self.nc[l]; sc = self.scl[:, l]
        x = self.smooth(cpre[l], l, rhs, sc); r = rhs - self.mv(l, x)
        rc = r[:, :nc] + (self.XT[l] @ r[:, nc:].reshape(-1)).reshape(self.B, nc)
        ec = self.cycle(l + 1, rc, cpre, cpost) if l + 1 < self.nl else torch.bmm(self.A0inv, rc[..., None])[..., 0]
        xc = torch.cat([ec, (self.Xs[l] @ ec.reshape(-1)).reshape(self.B, -1)], 1)
        xp = self.smooth(cpost[l], l, rhs - self.mv(l, x + xc), sc)
        return (x, xc, xp) if parts else x + xc + xp

    def logerr(self, cpre, cpost):
        """per-operator log relative energy error of the final projected output (common Galerkin, no backstop): (B,)"""
        parts = self.cycle(0, self.bb, cpre, cpost, parts=True)
        W = torch.cat([torch.stack(parts, -1), self.Z], -1)
        AW = torch.stack([self.mv(0, W[..., k]) for k in range(W.shape[-1])], -1)
        G = W.transpose(1, 2) @ AW; G = 0.5 * (G + G.transpose(1, 2)); g = (W.transpose(1, 2) @ self.bb[..., None])[..., 0]
        dg = torch.diagonal(G, dim1=1, dim2=2); live = dg > 1e-30 * dg.amax(1, keepdim=True)
        sc_ = torch.where(live, torch.where(live, dg, torch.ones_like(dg)).rsqrt(), torch.zeros_like(dg))
        Gs = sc_[:, :, None] * G * sc_[:, None, :] + torch.diag_embed((~live).double() + 1e-10)
        y = sc_ * torch.linalg.solve(Gs, sc_ * g)
        e = self.target - (W @ y[..., None])[..., 0]
        return 0.5 * torch.log((e * self.mv(0, e)).sum(1) / self.unorm ** 2)


def unpack(th, B):                                  # th: (B, 24) -> pre[l], post[l]: (B, 4)
    return [th[:, 4 * l:4 * l + 4] for l in range(3)], [th[:, 12 + 4 * l:12 + 4 * l + 4] for l in range(3)]


def optimise(op, th0, max_outer=25):
    """one operator (op: Batch of size 1); th0: (24,)."""
    th = torch.tensor(th0[None].copy(), requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=1, max_iter=20, tolerance_grad=1e-6, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')
    nev = [0]
    def closure():
        opt.zero_grad(); L = op.logerr(*unpack(th, 1)).sum(); L.backward(); nev[0] += 1; return L
    last = None; stop = 'iterations'
    for _ in range(max_outer):
        L = float(opt.step(closure)); gmax = float(th.grad.abs().max())
        if not np.isfinite(L): stop = 'non-finite'; break
        if gmax < 1e-6: stop = 'gradient'; break
        if last is not None and abs(last - L) < 1e-12: stop = 'stalled'; break
        last = L
    th.grad = None; L = op.logerr(*unpack(th, 1)); L.sum().backward()
    return th.detach().numpy()[0], float(L.detach()[0]), float(th.grad.abs().max()), nev[0], stop


if __name__ == '__main__':
    SET, FAM, N, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    PRE = json.load(open(os.path.join(HERE, 'stage1_preconfirmation.json'))); EX = json.load(open(os.path.join(HERE, PRE['fitted_methods'])))[FAM]
    kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
    S = setup(3, 2, 1, **kw) if SET == 'D' else setup(3, 2, 0, rng_seed={'channels': 2026101601, 'barriers': 2026101602}[FAM], **kw)
    te = S['te'][:N]; H = S['H']; nl = S['nl']; G = Gates(); t00 = time.time()
    hrs, Zs, Rs, LZ, X0 = [], [], [], [], []
    for i in te:
        A = S['A'][i]; hr = Hier(H, A, nl); hrs.append(hr); Zs.append(near_kernel_basis(A)); r = S['Fs'][i] - A @ S['UC'][i]; Rs.append(r)
        LZ.append([lam_lanczos(Ak) for Ak, _ in hr.levs])
    ops = Batch(hrs, Rs, Zs, [S['U'][i] - S['UC'][i] for i in te], [float(np.sqrt(S['U'][i] @ (S['A'][i] @ S['U'][i]))) for i in te], LZ)
    B = len(te); c0 = cheb_coeffs(10)
    # starts (B, 24)
    st = {'CHEB-T*-equivalent': np.tile(np.concatenate([np.tile(c0, 3), np.tile(c0, 3)]), (B, 1)),
          'POLY-S': np.array([np.concatenate([rescale(np.array(EX['POLY-S'][w]), EX['POLY-S']['beta'], LZ[b][l]) for w in ('pre', 'post') for l in range(3)]) for b in range(B)]),
          'POLY-L': np.tile(np.concatenate([np.ravel(EX['POLY-L']['pre']), np.ravel(EX['POLY-L']['post'])]), (B, 1))}
    pc = np.zeros((B, 24))                                       # PCG first-call coefficients, captured in the PCG cycle
    for b, (hr, Z, r) in enumerate(zip(hrs, Zs, Rs)):
        calls = []
        def spy(pos):
            def f(Ak, dinv, rhs, l, cnt=None):
                x = pcg4(Ak, dinv, rhs, l); calls.append((pos, l, rhs.copy(), x)); return x
            return f
        one_pass(hr, r, Z, spy('pre'), spy('post'))
        for pos, l, rhs, x in calls:
            Ak, _ = hr.levs[l]; di = hr.dinv[l]; s = LZ[b][l]; Bv = lambda y: di * (Ak @ y)
            v = di * rhs; T = [v, 2 / s * Bv(v) - v]
            for j in (2, 3): T.append(2 * (2 / s * Bv(T[-1]) - T[-1]) - T[-2])
            V = np.column_stack(T) / s; AV = Ak @ V                       # least squares in the A_l inner product
            cf = np.linalg.lstsq(V.T @ AV, V.T @ (Ak @ x), rcond=None)[0]
            pc[b, (0 if pos == 'pre' else 12) + 4 * l:(0 if pos == 'pre' else 12) + 4 * l + 4] = cf
    st["PCG first-call"] = pc; st['zero'] = np.zeros((B, 24))
    # check: the batched objective reproduces the numpy path for the CHEB-T* start and POLY-L (no backstop fired on D)
    with torch.no_grad():
        lb = ops.logerr(*unpack(torch.tensor(st['POLY-L']), B)).numpy()
    ln = np.array([np.log(energy(S['A'][i], S['UC'][i] + one_pass(hr, r, Z, make_poly(np.array(EX['POLY-L']['pre']), lz), make_poly(np.array(EX['POLY-L']['post']), lz)), S['U'][i]))
                   for i, hr, r, Z, lz in zip(te, hrs, Rs, Zs, LZ)])
    G.check('batched objective = numpy path (POLY-L start), max |diff| of log error <= 1e-8', np.max(np.abs(lb - ln)) <= 1e-8, float(np.max(np.abs(lb - ln))))
    with torch.no_grad():
        lp = ops.logerr(*unpack(torch.tensor(pc), B)).numpy()
    lpcg = np.array([np.log(energy(S['A'][i], S['UC'][i] + one_pass(hr, r, Z, pcg4, pcg4), S['U'][i])) for i, hr, r, Z in zip(te, hrs, Rs, Zs)])
    print(f"  PCG first-call start vs PCG cycle: mean log error {lp.mean():.5f} vs {lpcg.mean():.5f} (equal only if every call is the first call, which holds in a V-cycle)", flush=True)
    single = [Batch([hr], [r], [Z], [S['U'][i] - S['UC'][i]], [float(np.sqrt(S['U'][i] @ (S['A'][i] @ S['U'][i])))], [lz])
              for i, hr, r, Z, lz in zip(te, hrs, Rs, Zs, LZ)]
    runs = {}
    for nm, th0 in st.items():
        t0 = time.time()
        with torch.no_grad(): L0 = ops.logerr(*unpack(torch.tensor(th0), B)).numpy()
        res = [optimise(op, th0[b]) for b, op in enumerate(single)]
        th = np.stack([r_[0] for r_ in res]); L = np.array([r_[1] for r_ in res]); gop = np.array([r_[2] for r_ in res])
        L = np.where(np.isfinite(L), L, np.inf)
        runs[nm] = dict(theta=th, L=L, L0=L0, grad=gop, evals=[r_[3] for r_ in res], stop=[r_[4] for r_ in res], seconds=time.time() - t0)
        stops = {k: runs[nm]['stop'].count(k) for k in sorted(set(runs[nm]['stop']))}
        print(f"  start {nm:<20} mean log error {L0.mean():.5f} -> {np.mean(L[np.isfinite(L)]):.5f}; stops {stops}; "
              f"median |grad| {np.median(gop):.1e}; {sum(runs[nm]['evals'])} evals, {time.time() - t0:.0f} s", flush=True)
    names = list(runs); Ls = np.stack([runs[n]['L'] for n in names]); kb = Ls.argmin(0)
    best = np.stack([runs[names[k]]['theta'][b] for b, k in enumerate(kb)])
    e_d2 = []
    for b, (i, hr, r, Z, lz) in enumerate(zip(te, hrs, Rs, Zs, LZ)):
        cp, cq = best[b, :12].reshape(3, 4), best[b, 12:].reshape(3, 4)
        e_d2.append(float(energy(S['A'][i], S['UC'][i] + one_pass(hr, r, Z, make_poly(cp, lz), make_poly(cq, lz)), S['U'][i])))
    gap = np.max(np.abs(np.log(e_d2) - Ls.min(0)))
    G.check('numpy error of the kept D2 coefficients = batched optimum (|log diff| <= 1e-6)', gap <= 1e-6, float(gap))
    out = dict(set=SET, family=FAM, ids=[int(i) for i in te], errors_D2=e_d2, kept_start=[names[k] for k in kb],
               starts={n: dict(mean_log_initial=float(r['L0'].mean()), mean_log_final=float(np.mean(r['L'][np.isfinite(r['L'])])), per_operator_final=[float(x) for x in r['L']],
                               per_operator_grad_inf=r['grad'].tolist(), evaluations=r['evals'], stop=r['stop'], seconds=r['seconds']) for n, r in runs.items()},
               coefficients=best.tolist(), scales=LZ, gates=G.g, seconds=time.time() - t00, manifest=manifest())
    json.dump(out, open(OUT, 'w'), indent=1)
    print(f"== D2 {SET} {FAM}: geo-mean error {np.exp(np.mean(np.log(e_d2))):.4e}; kept starts { {n: int((kb == k).sum()) for k, n in enumerate(names)} }; {time.time() - t00:.0f} s")
    if G.failed(): sys.exit(1)
