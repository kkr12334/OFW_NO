"""Breakdown of OA-VCB's per-sample operator data on this CPU (2 threads), from the assembled fine matrices (11 Oct 2026,
same boundary as timing_one_platform.py v11oct): operator-adapted hierarchy (oa_hierarchy), near-kernel detection
(near_kernel_basis), coarsest dense inverse; per-sample minimum over REP passes, mean over the samples.
usage: python diagnostics_cost/opdata_breakdown.py L k0 data_seed [n_test] [repeats]"""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s1_common import setup
from oa_vcycle import oa_hierarchy, near_kernel_basis
torch.set_num_threads(2)
L, k0, DS = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
NT = int(sys.argv[4]) if len(sys.argv) > 4 else 100; REP = int(sys.argv[5]) if len(sys.argv) > 5 else 3
S = setup(L, k0, DS); te = S['te'][:NT]; As = [S['A'][i] for i in te]; pc = time.perf_counter
Th, Tz, Ti = (np.full(len(te), np.inf) for _ in range(3))
for _ in range(REP):
    for j, A in enumerate(As):
        t = pc(); levs, A0 = oa_hierarchy(S['H'], A, S['nl'], 4); Th[j] = min(Th[j], pc() - t)
        t = pc(); near_kernel_basis(levs[0][0]); Tz[j] = min(Tz[j], pc() - t)
        t = pc(); np.linalg.inv(A0.toarray()); Ti[j] = min(Ti[j], pc() - t)
f = lambda T: f"{T.mean() * 1e3:.2f} ({np.median(T) * 1e3:.2f})"
print(f"L{L} k0 {k0} seed {DS}, {len(te)} samples, min over {REP}: hierarchy {f(Th)} ms; detection {f(Tz)} ms; coarsest inverse {f(Ti)} ms")
