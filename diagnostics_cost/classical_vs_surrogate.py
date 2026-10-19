"""Stock-take (3 Oct): does a few iterations of a CLASSICAL method, started from the same coarse solve, already reach
oa-ll-op's accuracy, and at what cost on the same CPU?  Darcy, perforated mesh, channels 1e4 (the paper's setting),
L3 (k0 2) and L4 (k0 3), Galerkin coarse solve, 10 samples.  Everything timed here on one CPU (2 cores), including
the oa-ll-op forward pass (untrained weights: the cost does not depend on them).

Methods, all from u_c (the input every model gets):
  TG-PCG  CG preconditioned by the symmetric two-grid cycle (omega-Jacobi, exact Galerkin coarse solve, omega-Jacobi)
          -- i.e. the classical method whose one-pass, nu-step version is the Proposition's ceiling;
  J-PCG   Jacobi-preconditioned CG;
  AMG     pyamg smoothed aggregation as a PCG preconditioner, started from u_c (setup timed separately).
Targets: oa-ll-op's 3-seed test errors, L3 9.99e-3 / 0.024 and L4 8.09e-3 / 0.026 (rel-L2 / energy).
"""
import sys, os, time, warnings, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.sparse.linalg import spsolve, splu, factorized
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
import pyamg

TARGET = {3: (9.99e-3, 0.024), 4: (8.09e-3, 0.026)}
torch.set_num_threads(2)


def pcg(A, b, x0, Minv, u, maxit=30):
    x = x0.copy(); r = b - A @ x; z = Minv(r); p = z.copy(); rz = r @ z; hist = []
    en = lambda v: np.sqrt(v @ (A @ v)); nu_ = en(u); nl = np.linalg.norm(u)
    hist.append((np.linalg.norm(x - u) / nl, en(x - u) / nu_))
    for _ in range(maxit):
        Ap = A @ p; al = rz / (p @ Ap); x = x + al * p; r = r - al * Ap
        z = Minv(r); rz2 = r @ z; p = z + (rz2 / rz) * p; rz = rz2
        hist.append((np.linalg.norm(x - u) / nl, en(x - u) / nu_))
    return np.array(hist)


def first_hit(h, tgt):
    ok = np.nonzero((h[:, 0] <= tgt[0]) & (h[:, 1] <= tgt[1]))[0]
    return int(ok[0]) if len(ok) else None


for L, k0 in ((3, 2), (4, 3)):
    H = NestedHierarchy(L, coarse=perforated_square(n=9)); rng = np.random.default_rng(1000)
    Pk = None
    for k in range(k0 + 1, L + 1):
        Pk = H.P[k] if Pk is None else H.P[k] @ Pk
    Pk = Pk.tocsr(); PkT = Pk.T.tocsr(); F = H.F; om = 2 / 3
    rows = {'TG-PCG': [], 'J-PCG': [], 'AMG-PCG': []}; tm = {k: [] for k in ('direct', 'coarse', 'tg_it', 'j_it', 'amg_setup', 'amg_it')}
    for s in range(10):
        a = field_channels(H, 1e4, rng); A = H.A_fine(a).tocsr()
        t0 = time.perf_counter(); u = spsolve(A.tocsc(), F); tm['direct'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); Ac = (PkT @ A @ Pk).tocsc(); solc = factorized(Ac); uc = Pk @ solc(PkT @ F)
        tm['coarse'].append(time.perf_counter() - t0)
        dinv = 1 / A.diagonal()
        def tg(r):
            x = om * dinv * r; x = x + Pk @ solc(PkT @ (r - A @ x)); return x + om * dinv * (r - A @ x)
        t0 = time.perf_counter(); h = pcg(A, F, uc, tg, u); tm['tg_it'].append((time.perf_counter() - t0) / 30); rows['TG-PCG'].append(h)
        t0 = time.perf_counter(); h = pcg(A, F, uc, lambda r: dinv * r, u); tm['j_it'].append((time.perf_counter() - t0) / 30); rows['J-PCG'].append(h)
        t0 = time.perf_counter(); ml = pyamg.smoothed_aggregation_solver(A); Mamg = ml.aspreconditioner(cycle='V')
        tm['amg_setup'].append(time.perf_counter() - t0)
        t0 = time.perf_counter(); h = pcg(A, F, uc, lambda r: Mamg @ r, u); tm['amg_it'].append((time.perf_counter() - t0) / 30); rows['AMG-PCG'].append(h)
    ms = {k: 1e3 * np.median(v) for k, v in tm.items()}
    print(f"\nL{L} (k0 {k0}, {len(H.nodes[L])} unknowns), 10 samples, CPU medians: direct {ms['direct']:.1f} ms, Galerkin coarse "
          f"solve {ms['coarse']:.1f} ms; per iteration: TG-PCG {ms['tg_it']:.2f}, J-PCG {ms['j_it']:.2f}, AMG-PCG "
          f"{ms['amg_it']:.2f} ms (+ AMG setup {ms['amg_setup']:.1f} ms)")
    tgt = TARGET[L]
    print(f"  oa-ll-op target {tgt[0]:.2e} / {tgt[1]:.3f}")
    for name, hs in rows.items():
        hs = np.array(hs); m = hs.mean(0)
        hits = [first_hit(h, tgt) for h in hs]
        its = ' '.join(f"{j}:{m[j, 0]:.1e}/{m[j, 1]:.3f}" for j in (0, 1, 2, 3, 5, 8))
        hv = [x for x in hits if x is not None]
        per = {'TG-PCG': ms['tg_it'], 'J-PCG': ms['j_it'], 'AMG-PCG': ms['amg_it']}[name]
        extra = ms['amg_setup'] if name == 'AMG-PCG' else 0.0
        it_med = np.median(hv) if hv else np.nan
        print(f"  {name:<8} mean err after j its (rel-L2/energy): {its}")
        print(f"           iterations to reach the target: median {it_med:.0f} (max {max(hv) if hv else 'n/a'}, "
              f"{len(hv)}/10 reached in 30); cost from u_c ~ {extra + it_med * per:.1f} ms + coarse {ms['coarse']:.1f} ms")
    # oa-ll-op forward on the same CPU (batch 1), untrained weights
    from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant
    MG = MeshHierarchyGeometry(H, 'cpu'); w0 = 20 if L == 3 else 12
    m_ = MeshLiftingNO(MG, 5, width=w0, **variant('oa-ll-op')).eval()
    m_.set_op(torch.randn(1, len(MG.fine['src']))); x = torch.randn(1, MG.fine['n'], 5)
    with torch.no_grad():
        for _ in range(3): m_(x)
        t0 = time.perf_counter()
        for _ in range(20): m_(x)
    print(f"  oa-ll-op forward, batch 1, same CPU: {(time.perf_counter() - t0) / 20 * 1e3:.1f} ms "
          f"({sum(p.numel() for p in m_.parameters()):,} params)  + residual/weights (one mat-vec) + coarse {ms['coarse']:.1f} ms")
