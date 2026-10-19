"""Which coefficient regime actually produces a multiscale SOLUTION?

No training here: for each field regime, solve directly and ask how much of u a
plain linear map of the field already explains.  If linear regression is
competitive, no architecture can distinguish itself on that regime and the
benchmark is uninformative regardless of which model wins.
"""
import numpy as np, sys
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
sys.path.insert(0, '.')
from ofwno_core import MeshHierarchy, PoissonAssembler
from grf_field import sample_grf, grf_at_elements, ElementwiseOperatorAssembly
from frozen_basis import sample_a_lognormal
from ofwno_core import ChannelizedOperatorAssembly, build_channel_assignment

L, N = 5, 150
n = 2 ** L + 1
mesh = MeshHierarchy(); mesh.refine_to_level(L)
bnd = np.asarray(mesh.get_boundary_nodes(), int)
asm = ElementwiseOperatorAssembly(mesh, PoissonAssembler)
_, F0 = PoissonAssembler.assemble(mesh, lambda x, y: 1.0)
coords = np.asarray(mesh.coords)
elems = np.asarray([np.asarray(e, int) for e in mesh.elements])

def solve_all(a_elem):
    U = np.zeros((len(a_elem), len(coords)))
    for i, ae in enumerate(a_elem):
        K = asm.K_raw(ae).tolil(); F = F0.copy()
        for b in bnd: K[b, :] = 0; K[:, b] = 0; K[b, b] = 1.0; F[b] = 0.0
        U[i] = spsolve(K.tocsr(), F)
    return U

def rel(p, t):
    return float(np.mean(np.linalg.norm(p-t, axis=1)/(np.linalg.norm(t, axis=1)+1e-30)))

def report(tag, X, U, extra=""):
    tr, te = np.arange(0, int(.75*N)), np.arange(int(.75*N), N)
    Xm = X[tr].mean(0); Xc = X - Xm
    _,_,Vt = np.linalg.svd(Xc[tr], full_matrices=False)
    best, bestr = 9e9, 0
    for r in (10, 20, 40, 80, 120):
        if r >= len(tr): break
        P = Vt[:r]
        A = np.c_[Xc[tr]@P.T, np.ones(len(tr))]; B = np.c_[Xc[te]@P.T, np.ones(len(te))]
        W,*_ = np.linalg.lstsq(A, U[tr], rcond=None)
        e = rel(B@W, U[te])
        if e < best: best, bestr = e, r
    mean_e = rel(np.tile(U[tr].mean(0), (len(te),1)), U[te])
    g = U[:, g2n]
    Ff = np.abs(np.fft.rfft2(g, axes=(1,2)))**2; tot = Ff.sum(axis=(1,2))
    lo = np.mean(Ff[:, :4, :4].sum(axis=(1,2))/tot)*100
    print(f"{tag:<34}{mean_e:>9.3f}{best:>10.3f}{best/mean_e:>9.2f}{lo:>11.1f}%  {extra}")

from ofwno_nets import build_grid_bijection
g2n, n2g, _, gxy = build_grid_bijection(mesh, L)

print(f"level {L}, {N} samples, contrast 1e4\n")
print(f"{'field regime':<34}{'mean':>9}{'linear':>10}{'ratio':>9}{'u in 4x4':>12}")
rng = np.random.default_rng(3)
for cl in (0.05, 0.1, 0.2, 0.4):
    G = sample_grf(N, n, corr_length=cl, nu=0.5, contrast=1e4, rng=rng)
    ae = grf_at_elements(G, mesh, n)
    report(f"GRF nu=0.5  corr={cl}", np.log(G.reshape(N,-1)), solve_all(ae),
           f"({cl*(n-1):.1f} cells)")
for cl in (0.2, 0.4):
    G = sample_grf(N, n, corr_length=cl, nu=1.5, contrast=1e4, rng=rng)
    ae = grf_at_elements(G, mesh, n)
    report(f"GRF nu=1.5  corr={cl}", np.log(G.reshape(N,-1)), solve_all(ae),
           f"({cl*(n-1):.1f} cells)")
# channelized reference (the benchmark we already have a table for)
for ncx in (4, 8):
    ch = build_channel_assignment(mesh, ncx, ncx)
    mu = sample_a_lognormal(N, ncx*ncx, 1e4, rng)
    ae = 1.0 + mu[:, ch]
    report(f"channelized {ncx}x{ncx}", np.log(mu), solve_all(ae),
           f"({(n-1)//ncx} cells/block)")
print("\nratio = linear / predict-the-mean.  Near 1.0 means even a linear map is")
print("useless; near 0 means a linear map already solves it and no architecture")
print("can distinguish itself.  The informative regime is in between.")
