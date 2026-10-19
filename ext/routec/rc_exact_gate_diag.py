"""Diagnostic for the barriers L3 pilot stop at 'id 1 c=1e+06: exact solves finite and residual small' (18 Oct 2026).

Rebuilds exactly what rc_pilot.py builds for the exact solves (same seed, geometry, K, loads, splu) and reports, per load:
  rel_res   = max|b - A u| / max|b|                       (the runner's gate, threshold 1e-9)
  omega     = ||b - A u||_inf / (||A||_inf ||u||_inf + ||b||_inf)   (normwise backward error of the solve)
  ref_err   = |||A^{-1}(b - A u)||| / |||u|||               (energy-norm error of the reference itself, one refinement solve)
No outcome is computed.
    python3 rc_exact_gate_diag.py FAMILY ID[,ID..] CONTRAST[,..]   -> prints and appends to exact_gate_diag.json
"""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.dirname(HERE); ROOT = os.path.dirname(EXT)
for p in (ROOT, EXT, HERE): sys.path.insert(0, p)
from scipy.sparse.linalg import splu
from oa_lifting import NestedHierarchy, make_field
from coarse_meshes import perforated_square
import rc_fem as R

FAM = sys.argv[1]; IDS = [int(x) for x in sys.argv[2].split(',')]; CS = [float(x) for x in sys.argv[3].split(',')]
SEED = {'channels': 2026101801, 'barriers': 2026101802}[FAM]
GOALS = [(0.5, 0.45), (0.15, 0.15), (0.85, 0.85)]
cm = perforated_square(n=9); H = NestedHierarchy(3, coarse=cm); M = R.Mesh(H, cm)
base = 1e4 if FAM == 'channels' else 1e-4
rng = np.random.default_rng(SEED); draws = [make_field(H, 'channels', base, rng) for _ in range(max(IDS) + 1)]
c3 = M.fe.centroid
goal_masks = [np.hypot(c3[:, 0] - x, c3[:, 1] - y) < 0.08 for x, y in GOALS]
goal_w = [om / M.fe.area[om].sum() for om in goal_masks]
out_p = os.path.join(HERE, 'exact_gate_diag.json')
rows = json.load(open(out_p)) if os.path.exists(out_p) else []
for idx in IDS:
    mask = draws[idx] != 1.0
    for c in CS:
        K = np.where(mask, c if FAM == 'channels' else 1.0 / c, 1.0)
        fK = np.ones(len(K)) if FAM == 'channels' else (~mask).astype(float)
        A = M.stiffness(K).tocsr(); lu = splu(A.tocsc()); F = M.load(fK)
        loads = [('primal', F)] + [(f'dual{g}', M.load(gw)) for g, gw in enumerate(goal_w)]
        Ainf = float(abs(A).sum(1).max())
        for nm, b in loads:
            u = lu.solve(b); r = b - A @ u
            d = lu.solve(r)
            en = lambda x: float(np.sqrt(max(x @ (A @ x), 0.0)))
            row = dict(family=FAM, id=idx, contrast=c, load=nm, finite=bool(np.all(np.isfinite(u))),
                       rel_res=float(np.abs(r).max() / np.abs(b).max()),
                       omega=float(np.abs(r).max() / (Ainf * np.abs(u).max() + np.abs(b).max())),
                       ref_err=en(d) / en(u), gate_passed=bool(np.abs(r).max() <= 1e-9 * np.abs(b).max()))
            rows.append(row)
            print(f"{FAM} id {idx} c={c:.0e} {nm:6s}: rel_res {row['rel_res']:.2e} (gate {'pass' if row['gate_passed'] else 'FAIL'}); "
                  f"backward error {row['omega']:.2e}; reference energy error {row['ref_err']:.2e}", flush=True)
json.dump(rows, open(out_p, 'w'), indent=1)
