"""EXPLORATORY (post-hoc; development set D only).  Robustness of the oracle-polynomial diagnostic D2 (channels gap ~4x).
For the first N operators of set D:
 (a) K4 continued: restart L-BFGS from the kept D2 optimum (ext/dev/stage1_d2_D_{family}.json) for 2500 more iterations;
 (b) K5 oracle (reach-matched to OA-VCB's 4 message-passing steps: output p(B) D^-1 b with deg p <= 4, 5 coefficients per
     call, 30 per operator), starts: PCG(5) first-call coefficients and the kept K4 optimum embedded (5th coefficient 0);
     1500 iterations each.  Same objective, scaling and stopping rule as stage1_d2.py otherwise.
usage: python -I ext/explore/x2_d2_extended.py FAMILY N OUT.json"""
import os, sys, json, functools, time, numpy as np, torch, scipy.sparse as sp
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
sys.path[:0] = [EXT, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
from s1_common import setup, energy
from oa_vcycle import near_kernel_basis
from stage1_core import Hier, galerkin, one_pass, pcg4, lam_lanczos
import stage1_d2 as D2
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
FAM, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
S = setup(3, 2, 1, **kw); te = S['te'][:N]; H = S['H']; nl = S['nl']
prev = json.load(open(os.path.join(EXT, 'dev', f'stage1_d2_D_{FAM}.json'))); assert prev['ids'][:N] == [int(i) for i in te]


class BatchD(D2.Batch):
    """degree-general smoother: x = (1/s) sum_{j<=deg} c_j T_j(2B/s - I) D^-1 b."""
    deg = 3
    def smooth(self, c, l, rhs, s):
        di = self.dinv[l]; s = s[:, None]; Bv = lambda y: di * self.mv(l, y)
        v = di * rhs; t0 = v; t1 = 2 / s * Bv(v) - v; x = c[:, 0:1] * t0 + c[:, 1:2] * t1
        for j in range(2, self.deg + 1):
            t2 = 2 * (2 / s * Bv(t1) - t1) - t0; x = x + c[:, j:j + 1] * t2; t0, t1 = t1, t2
        return x / s


def unpack(th, k):
    return [th[:, k * l:k * l + k] for l in range(3)], [th[:, 3 * k + k * l:3 * k + k * l + k] for l in range(3)]


def optimise(op, th0, k, rounds):
    th = torch.tensor(th0[None].copy(), requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=1, max_iter=20, tolerance_grad=1e-6, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); L = op.logerr(*unpack(th, k)).sum(); L.backward(); return L
    last = None; stop = 'iterations'
    for _ in range(rounds):
        L = float(opt.step(closure).detach()); g = float(th.grad.abs().max())
        if not np.isfinite(L): stop = 'non-finite'; break
        if g < 1e-6: stop = 'gradient'; break
        if last is not None and abs(last - L) < 1e-12: stop = 'stalled'; break
        last = L
    with torch.no_grad(): Lf = float(op.logerr(*unpack(th, k))[0])
    return th.detach().numpy()[0], Lf, stop


res = dict(note='EXPLORATORY, development set D', family=FAM, ids=[int(i) for i in te], K4_kept=[], K4_continued=[], K5=[], K5_starts=[], PCG5=[])
t0 = time.time()
for b, i in enumerate(te):
    A = S['A'][i]; hr = Hier(H, A, nl); Z = near_kernel_basis(A); r = S['Fs'][i] - A @ S['UC'][i]; lz = [lam_lanczos(Ak) for Ak, _ in hr.levs]
    args = ([hr], [r], [Z], [S['U'][i] - S['UC'][i]], [float(np.sqrt(S['U'][i] @ (A @ S['U'][i])))], [lz])
    op4 = BatchD(*args); op5 = BatchD(*args); op5.deg = 4
    th4 = np.array(prev['coefficients'][b]); L4 = float(op4.logerr(*unpack(torch.tensor(th4[None]), 4))[0])
    res['K4_kept'].append(L4)
    thc, Lc, st = optimise(op4, th4, 4, 125); res['K4_continued'].append(dict(L=Lc, stop=st))
    # PCG(5) first-call coefficients in the degree-4 basis
    calls = []
    def spy(pos):
        def f(Ak, dinv, rhs, l, cnt=None):
            x = pcg4(Ak, dinv, rhs, l, its=5); calls.append((pos, l, rhs.copy(), x)); return x
        return f
    xp = one_pass(hr, r, Z, spy('pre'), spy('post')); res['PCG5'].append(float(np.log(energy(A, S['UC'][i] + xp, S['U'][i]))))
    pc = np.zeros(30)
    for pos, l, rhs, x in calls:
        Ak, _ = hr.levs[l]; di = hr.dinv[l]; s = lz[l]; Bv = lambda y: di * (Ak @ y); v = di * rhs; T = [v, 2 / s * Bv(v) - v]
        for j in (2, 3, 4): T.append(2 * (2 / s * Bv(T[-1]) - T[-1]) - T[-2])
        V = np.column_stack(T) / s; AV = Ak @ V; cf = np.linalg.lstsq(V.T @ AV, V.T @ (Ak @ x), rcond=None)[0]
        o = (0 if pos == 'pre' else 15) + 5 * l; pc[o:o + 5] = cf
    emb = np.zeros(30)
    for l in range(3): emb[5 * l:5 * l + 4] = th4[4 * l:4 * l + 4]; emb[15 + 5 * l:15 + 5 * l + 4] = th4[12 + 4 * l:12 + 4 * l + 4]
    with torch.no_grad(): Lp0 = float(op5.logerr(*unpack(torch.tensor(pc[None]), 5))[0])
    best = None; ss = {}
    for nm, s0 in (('PCG(5) first-call', pc), ('K4 optimum embedded', emb)):
        th, L, st = optimise(op5, s0, 5, 75); ss[nm] = dict(L=L, stop=st)
        if best is None or L < best: best = L
    res['K5'].append(best); res['K5_starts'].append(dict(PCG5_start_value=Lp0, **ss))
    print(f"op {i}: K4 kept {L4:.4f} -> continued {Lc:.4f} ({st}); PCG(5) cycle {res['PCG5'][-1]:.4f} (start check {Lp0:.4f}); K5 oracle {best:.4f}; {time.time() - t0:.0f} s", flush=True)
oa = json.load(open(os.path.join(EXT, 'dev', 'stage1_dev_D.json')))['families'][FAM]['errors']
loa = np.log(oa['OA-VCB init0'][:N]); lch = np.log(oa['CHEB-legacy'][:N])
summ = {k: dict(G_over_CHEB=float(np.exp(np.mean(lch - np.array(v)))), G_to_OA_VCB_init0=float(np.exp(np.mean(np.array(v) - loa))))
        for k, v in (('K4 kept (D2)', res['K4_kept']), ('K4 continued', [x['L'] for x in res['K4_continued']]), ('PCG(5) cycle', res['PCG5']), ('K5 oracle', res['K5']))}
summ['OA-VCB init0'] = dict(G_over_CHEB=float(np.exp(np.mean(lch - loa))))
res['summary'] = summ; json.dump(res, open(OUT, 'w'), indent=1)
for k, v in summ.items(): print(FAM, k, {a: round(b, 3) for a, b in v.items()})
