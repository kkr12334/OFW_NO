"""Is the GRF table discriminating, or is every model at the same floor?"""
import numpy as np, sys
sys.path.insert(0, '.')
from ofwno_benchmark_v2 import build_pool_grf
from train_compare import prepare, rel_l2

P = build_pool_grf(MAX_LEVEL=5, N=250, contrast=1e4, corr_length=0.05, nu=0.5,
                   mode='per_sample', n_layers=6, seed=12345, verbose=False)
D = prepare(P); tr, te = D['tr'], D['te']
mu, u = D['mu'], D['u']
print(f"train {len(tr)}  test {len(te)}  input dim {mu.shape[1]}  nodes {u.shape[1]}\n")
print(f"{'reference':<36}{'rel-L2':>10}")
print(f"{'predict the training mean':<36}{rel_l2(np.tile(u[tr].mean(0),(len(te),1)), u[te]):>10.4f}")
A = np.c_[mu[tr], np.ones(len(tr))]; B = np.c_[mu[te], np.ones(len(te))]
W,*_ = np.linalg.lstsq(A, u[tr], rcond=None)
print(f"{'linear lstsq(log a-field -> u)':<36}{rel_l2(B@W, u[te]):>10.4f}")
Ua,Sa,Vt = np.linalg.svd(mu[tr]-mu[tr].mean(0), full_matrices=False)
for r in (20, 50, 100, 150):
    Pr = Vt[:r]
    Atr = np.c_[(mu[tr]-mu[tr].mean(0))@Pr.T, np.ones(len(tr))]
    Ate = np.c_[(mu[te]-mu[tr].mean(0))@Pr.T, np.ones(len(te))]
    Wr,*_ = np.linalg.lstsq(Atr, u[tr], rcond=None)
    print(f"{f'  linear on {r} leading PCs of the field':<36}{rel_l2(Ate@Wr, u[te]):>10.4f}")
from scipy.spatial import cKDTree
_, idx = cKDTree(mu[tr]).query(mu[te])
print(f"{'nearest neighbour in field space':<36}{rel_l2(u[tr][idx], u[te]):>10.4f}")
g = u[:, D['g2n']]
F = np.abs(np.fft.rfft2(g, axes=(1,2)))**2; tot = F.sum(axis=(1,2))
print()
for m in (4, 8, 12, 16):
    print(f"  energy of u in lowest {m:2d}x{m:2d} Fourier modes: "
          f"{np.mean(F[:, :m, :m].sum(axis=(1,2))/tot)*100:5.2f}%")
