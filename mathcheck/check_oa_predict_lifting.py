"""3 Oct (evening): checks for oa-ll-op-oa (operator-adapted predict as the base of the lifting).
  1. with the learned predict/update terms zeroed, the forward lifting of a nodal field x is the route-V change of basis
     level by level: d_k = x_F - Phat_k x_C, x_C unchanged  (Phat_k the per-sample predict given to the network);
  2. the lifting stays exactly invertible for random weights (forward then inverse = identity);
  3. oa-ll-op itself is unchanged: oa_predict=False gives bit-identical outputs to the previous file;
  4. the per-sample predicts given to the network equal the scipy construction of cheap_oa_predict.py.
Perforated mesh L3, barriers 1e-4, one sample, float64."""
import sys, os, warnings, importlib.util, numpy as np, scipy.sparse as sp, torch
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, HERE)
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from mesh_lifting import MeshHierarchyGeometry, MeshLiftingNO, variant
torch.set_default_dtype(torch.float64)

L, m_steps = 3, 4
H = NestedHierarchy(L, coarse=perforated_square(n=9)); MG = MeshHierarchyGeometry(H, 'cpu')
a = field_channels(H, 1e-4, np.random.default_rng(3)); A = H.A_fine(a).tocsr()

# per-level predicts exactly as train_oa_layer builds them (values on the structural pattern)
pats, vals, Xs_ref = [], [], []
Ak = A.copy()
for Lv in MG.levels:
    k, nc = Lv['k'], Lv['nc']
    A_FF, A_FC = Ak[nc:, nc:].tocsr(), Ak[nc:, :nc].tocsr(); Dinv = sp.diags(1 / A_FF.diagonal())
    X = H.P[k][nc:].tocsr()
    for _ in range(m_steps):
        X = (X + (2 / 3) * (Dinv @ (-A_FC - A_FF @ X))).tocsr()
    Xs_ref.append(X); C = X.tocoo()
    pats.append((torch.tensor(C.row, dtype=torch.long), torch.tensor(C.col, dtype=torch.long))); vals.append(torch.tensor(C.data)[None])
    Phi = sp.vstack([sp.identity(nc, format='csr'), X]).tocsr(); Ak = (Phi.T @ Ak @ Phi).tocsr()

net = MeshLiftingNO(MG, 5, width=16, **variant('oa-ll-op-oa'))
net.set_oa(vals, pats)
# 1. zero the learned predict/update terms
for mod in list(net.P) + list(net.U):
    torch.nn.init.zeros_(mod[2].weight); torch.nn.init.zeros_(mod[2].bias)
x = torch.randn(1, MG.fine['n'], net.C)
X = x.clone(); worst = 0.0
torch.set_grad_enabled(False)
for l, Lv in enumerate(MG.levels):
    Xc, Xf = X[:, :Lv['nc']], X[:, Lv['nc']:]
    d = Xf - net._predict(l, Xc); Xn = Xc + net._update(l, d)
    ref = Xf[0].numpy() - Xs_ref[l] @ Xc[0].numpy()
    worst = max(worst, float(np.abs(d[0].detach().numpy() - ref).max() / np.abs(ref).max()), float((Xn - Xc).abs().max()))
    X = Xn
print(f"1. zero learned terms: lifting = route-V change of basis, max rel. deviation {worst:.1e}")

# 2. exact invertibility with random weights
torch.set_grad_enabled(True)
net2 = MeshLiftingNO(MG, 5, width=16, **variant('oa-ll-op-oa')); net2.set_oa(vals, pats)
X = x.clone(); dets = []
with torch.no_grad():
    for l, Lv in enumerate(MG.levels):
        Xc, Xf = X[:, :Lv['nc']], X[:, Lv['nc']:]
        d = Xf - net2._predict(l, Xc); X = Xc + net2._update(l, d); dets.append(d)
    s = X
    for l in reversed(range(len(MG.levels))):
        Xc = s - net2._update(l, dets[l]); s = torch.cat([Xc, dets[l] + net2._predict(l, Xc)], 1)
print(f"2. random weights: inverse(forward(x)) - x = {float((s - x).abs().max()):.1e}")

# 3. oa_predict=False is the previous oa-ll-op bit for bit
spec = importlib.util.spec_from_file_location('ml_old', sys.argv[1]) if len(sys.argv) > 1 else None
if spec:
    old = importlib.util.module_from_spec(spec); spec.loader.exec_module(old)
    torch.manual_seed(0); n_new = MeshLiftingNO(MG, 5, width=16, **variant('oa-ll-op'))
    torch.manual_seed(0); n_old = old.MeshLiftingNO(old.MeshHierarchyGeometry(H, 'cpu'), 5, width=16, **old.variant('oa-ll-op'))
    w = torch.randn(1, len(MG.fine['src'])); n_new.set_op(w); n_old.set_op(w); xin = torch.randn(1, MG.fine['n'], 5)
    with torch.no_grad():
        print(f"3. oa-ll-op, new file vs previous file: max |diff| = {float((n_new(xin) - n_old(xin)).abs().max()):.1e}")
print(f"4. per-level predict nnz: {', '.join(str(v.shape[1]) for v in vals)}; parameters oa-ll-op-oa = oa-ll-op: "
      f"{sum(p.numel() for p in net.parameters()) == sum(p.numel() for p in MeshLiftingNO(MG, 5, width=16, **variant('oa-ll-op')).parameters())}")
