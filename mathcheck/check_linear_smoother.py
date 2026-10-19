"""5 Oct (review point 5): the residual-linear control.  For OAVCycle(linear=True) with deflation and the plain cycle output
(oa-vcl): (1) the map r -> c is LINEAR for each A (random weights): ||f(a r1 + b r2) - a f(r1) - b f(r2)|| / ||.|| ~ round-off,
also for negative coefficients; (2) it still contains the classical member (Jacobi weights): equal to the non-linear class's
Jacobi member; (3) parameter count vs the non-linear smoother at the same width.  For comparison the non-linear model's
deviation from linearity.  float64, L3 barriers samples 300-305."""
import sys, os, warnings, numpy as np, torch
warnings.filterwarnings('ignore'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oa_lifting import NestedHierarchy, field_channels
from coarse_meshes import perforated_square
from oa_vcycle import OAData, OAVCycle
torch.set_default_dtype(torch.float64)
H = NestedHierarchy(3, coarse=perforated_square(n=9)); rng = np.random.default_rng(1000)
A_all = [field_channels(H, 1e-4, rng) for _ in range(306)]; a_list = A_all[300:306]
D = OAData(H, a_list, 3, 4, 'cpu', H.level_coords, near_kernel=True)
ops = dict(A64=D.A64)
g = torch.Generator().manual_seed(0); n = D.A64.shape[0]
r1, r2 = torch.randn(6, len(H.F), generator=g), torch.randn(6, len(H.F), generator=g)
def setw(net, om=2 / 3):
    with torch.no_grad():
        for S in (net.pre[0], net.post[0]):
            for p in S.parameters(): p.zero_()
            S.enc_lin.weight[1, 0] = 1.0
            for t in range(4):
                S.V[t].weight[0, 0] = 1.0; Lw = S.Lw[t].weight; Lw[0, 1] = om; Lw[0, 0] = -om; Lw[0, S.C] = -om
            S.dec_lin.weight[0, 0] = 1.0
devs = {}
for lin in (True, False):
    torch.manual_seed(1); net = OAVCycle(D, width=8, project=False, deflate=True, linear=lin)
    net.set_ops([v.double() for v in D.Av], [x.double() for x in D.Xv], D.A0inv.double(), D.Z.double(), A64=D.A64); net.eval()
    with torch.no_grad():
        f = lambda r: net(r)
        dev_ = max(float((f(a * r1 + b * r2) - a * f(r1) - b * f(r2)).norm() / f(a * r1 + b * r2).norm()) for a, b in ((1, 1), (2.5, -0.7), (-1, 0.3)))
    devs[lin] = dev_
    print(f"1. {'linear' if lin else 'non-linear'} smoothers, random weights: max relative deviation from linearity {dev_:.1e}; "
          f"params at width 8: {sum(p.numel() for p in net.parameters()):,}")
outs = []
for lin in (True, False):
    net = OAVCycle(D, width=4, project=False, deflate=True, linear=lin); setw(net)
    net.set_ops([v.double() for v in D.Av], [x.double() for x in D.Xv], D.A0inv.double(), D.Z.double(), A64=D.A64); net.eval()
    with torch.no_grad(): outs.append(net(r1))
print(f"2. Jacobi member: linear vs non-linear class, max rel. diff {float((outs[0] - outs[1]).abs().max() / outs[1].abs().max()):.1e}")
jdiff = float((outs[0] - outs[1]).abs().max() / outs[1].abs().max())
FAILS = []
if not (np.isfinite(devs[True]) and devs[True] < 1e-10): FAILS.append(f"linear class deviates from linearity by {devs[True]:.1e} (tol 1e-10)")
if not (np.isfinite(devs[False]) and devs[False] > 1e-3): FAILS.append(f"non-linear class deviation {devs[False]:.1e} <= 1e-3: the test cannot detect non-linearity")
if not (np.isfinite(jdiff) and jdiff < 1e-12): FAILS.append(f"Jacobi member differs between the classes by {jdiff:.1e} (tol 1e-12)")
if FAILS:
    print("PREFLIGHT FAIL (linear control): " + "; ".join(FAILS)); sys.exit(1)
print("PREFLIGHT PASS (linear control): linearity 1e-10, non-linearity detectable, Jacobi member identical")
