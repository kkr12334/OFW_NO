"""EXPLORATORY (post-hoc; development set D only).  Convergence check of the reach-matched K5 oracle of x2_d2_extended.py,
whose 1500-iteration runs mostly stopped at the iteration cap.  For the first N operators: both K5 starts (PCG(5)
first-call coefficients; kept K4 optimum embedded), 4500 iterations each, objective recorded after 1500, 3000 and 4500.
usage: python -I ext/explore/x2b_k5_longer.py FAMILY N OUT.json"""
import os, sys, json, time, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
sys.path[:0] = [HERE, EXT, ROOT, os.path.join(ROOT, 'diagnostics_cost')]
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
from s1_common import setup
from oa_vcycle import near_kernel_basis
from stage1_core import Hier, one_pass, pcg4, lam_lanczos
import importlib.util
spec = importlib.util.spec_from_file_location('x2', os.path.join(HERE, 'x2_d2_extended.py'))
FAM, N, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# reuse the class and helpers without running x2's main loop: exec only its definitions
src = open(os.path.join(HERE, 'x2_d2_extended.py')).read(); defs = src[src.index('class BatchD'):src.index('res = dict(')]
import stage1_d2 as D2; g = dict(D2=D2, np=np, torch=torch); exec(defs, g); BatchD, unpack = g['BatchD'], g['unpack']
kw = {} if FAM == 'channels' else dict(contrast=1e-4, load='no-low')
S = setup(3, 2, 1, **kw); te = S['te'][:N]; H = S['H']; nl = S['nl']
prev = json.load(open(os.path.join(EXT, 'dev', f'stage1_d2_D_{FAM}.json')))
def run(op, th0, total, marks):
    th = torch.tensor(th0[None].copy(), requires_grad=True)
    opt = torch.optim.LBFGS([th], lr=1, max_iter=20, tolerance_grad=1e-6, tolerance_change=1e-12, history_size=20, line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad(); L = op.logerr(*unpack(th, 5)).sum(); L.backward(); return L
    out = {}
    for rd in range(1, total // 20 + 1):
        opt.step(closure)
        if rd * 20 in marks:
            with torch.no_grad(): out[rd * 20] = float(op.logerr(*unpack(th, 5))[0])
    return out
res = dict(note='EXPLORATORY, development set D', family=FAM, ids=[int(i) for i in te], runs=[]); t0 = time.time()
for b, i in enumerate(te):
    A = S['A'][i]; hr = Hier(H, A, nl); Z = near_kernel_basis(A); r = S['Fs'][i] - A @ S['UC'][i]; lz = [lam_lanczos(Ak) for Ak, _ in hr.levs]
    op = BatchD([hr], [r], [Z], [S['U'][i] - S['UC'][i]], [float(np.sqrt(S['U'][i] @ (A @ S['U'][i])))], [lz]); op.deg = 4
    calls = []
    def spy(pos):
        def f(Ak, dinv, rhs, l, cnt=None):
            x = pcg4(Ak, dinv, rhs, l, its=5); calls.append((pos, l, rhs.copy(), x)); return x
        return f
    one_pass(hr, r, Z, spy('pre'), spy('post')); pc = np.zeros(30)
    for pos, l, rhs, x in calls:
        Ak, _ = hr.levs[l]; di = hr.dinv[l]; s = lz[l]; Bv = lambda y: di * (Ak @ y); v = di * rhs; T = [v, 2 / s * Bv(v) - v]
        for j in (2, 3, 4): T.append(2 * (2 / s * Bv(T[-1]) - T[-1]) - T[-2])
        V = np.column_stack(T) / s; AV = Ak @ V; cf = np.linalg.lstsq(V.T @ AV, V.T @ (Ak @ x), rcond=None)[0]
        o = (0 if pos == 'pre' else 15) + 5 * l; pc[o:o + 5] = cf
    th4 = np.array(prev['coefficients'][b]); emb = np.zeros(30)
    for l in range(3): emb[5 * l:5 * l + 4] = th4[4 * l:4 * l + 4]; emb[15 + 5 * l:15 + 5 * l + 4] = th4[12 + 4 * l:12 + 4 * l + 4]
    rr = {nm: run(op, s0, 4500, (1500, 3000, 4500)) for nm, s0 in (('PCG(5)', pc), ('K4 embedded', emb))}
    res['runs'].append(rr); best = {m: min(rr[nm][m] for nm in rr) for m in (1500, 3000, 4500)}
    print(f"op {i}: best K5 log error after 1500/3000/4500 iterations: {best[1500]:.4f} / {best[3000]:.4f} / {best[4500]:.4f}; {time.time() - t0:.0f} s", flush=True)
oa = np.log(json.load(open(os.path.join(EXT, 'dev', 'stage1_dev_D.json')))['families'][FAM]['errors']['OA-VCB init0'][:N])
res['summary'] = {m: float(np.exp(np.mean(np.array([min(r[nm][m] for nm in r) for r in res['runs']]) - oa))) for m in (1500, 3000, 4500)}
print('G(K5 oracle -> OA-VCB init0) after 1500 / 3000 / 4500 iterations:', res['summary'])
json.dump(res, open(OUT, 'w'), indent=1)
