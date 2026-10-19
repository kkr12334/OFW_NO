import sys, os, json, numpy as np, torch, copy
sys.path.insert(0, '/home/claude/ofw/diagnostics_cost'); sys.path.insert(0, '/home/claude/ofw')
from s1_common import setup, load_net, apply, energy
from harmonic_study import stored_energies, CK, ROOT
from oa_vcycle import near_kernel_basis
torch.set_num_threads(2)
ds = int(sys.argv[1])
S = setup(3, 2, ds, contrast=1e-4, load='no-low'); te = S['te']
D, net, st = load_net(S, os.path.join(ROOT, CK[ds]), te)
R = np.array([S['Fs'][i] - S['A'][i] @ S['UC'][i] for i in te])
E = lambda C: np.array([energy(S['A'][i], S['UC'][i] + c, S['U'][i]) for i, c in zip(te, C)])
e32 = E(apply(net, D, R, S['U_scale']))
e32b1 = E(apply(net, D, R, S['U_scale'], bs=1))
es = stored_energies(ds)
# float64 network (same weights): reference for float32 round-off
net64 = copy.deepcopy(net).double()
D.Av = [v.double() for v in D.Av]; D.Xv = [v.double() for v in D.Xv]; D.A0inv = D.A0inv.double()
if D.Z is not None: D.Z = D.Z.double()
e64 = E(apply(net64, D, R, S['U_scale']))
isl = np.array([near_kernel_basis(S['A'][i]).shape[1] > 0 for i in te])
rd = lambda a, b: np.abs(a / b - 1)
print(f"seed {ds}: CPU32 vs stored(GPU32): max {rd(e32, es).max():.2e} median {np.median(rd(e32, es)):.1e} | islands max {rd(e32, es)[isl].max():.2e}, no-island max {rd(e32, es)[~isl].max():.2e}")
print(f"  CPU32 batch 8 vs batch 1: max {rd(e32, e32b1).max():.2e}")
print(f"  float64 net vs CPU32: max {rd(e64, e32).max():.2e} median {np.median(rd(e64, e32)):.1e};  float64 vs stored: max {rd(e64, es).max():.2e}")
print(f"  means: stored {es.mean():.6f} cpu32 {e32.mean():.6f} f64 {e64.mean():.6f}")
k = np.argsort(-rd(e32, es))[:5]; print("  worst samples (id, island, stored, cpu32, f64):", [(int(te[j]), bool(isl[j]), round(es[j], 6), round(e32[j], 6), round(e64[j], 6)) for j in k])
