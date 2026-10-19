"""Session 9 (11 Oct 2026): baseline sanity check on the PUBLISHED Darcy benchmark, to show that our baseline implementations
reproduce published accuracy on the problem they were designed for.
Data: Li et al. (FNO) Darcy, piecewise-constant coefficients, files piececonst_r421_N1024_smooth1.mat (train) and
piececonst_r421_N1024_smooth2.mat (test), downsampled by 5 to 85 x 85; 1000 training samples (first 1000 of smooth1), 100 test
samples (first 100 of smooth2), as in the FNO and WNO papers.  Metric: mean relative L2 error over the test set.
Published reference values on this benchmark (85 x 85): FNO 1.08 % (Li et al. 2021; WNO paper Table 3), WNO 0.84 % (Tripura &
Chakraborty, arXiv 2205.02191 Table 3: db4, level 4, width 64, 500 epochs).
Models (our code, unchanged): fno = ofwno_nets.RealFNO2D (4 layers, width 32, 12 modes; NOTE: keeps only the [:m, :m]
Fourier corner, no padding, GELU after every layer -- the implementation behind the paper's FNO-op rows);
fno-official = faithful port (below) of the original FNO-2D of Li et al. (two Fourier corners, padding 9, no activation after
the last layer, fc 128) -- the reference implementation, to measure the effect of the simplification;
wno-light = ofwno_nets.RealWNO2D (our light WNO layer behind the paper's WNO-op rows; db4, J 2, width 64);
wno-db4 / wno-db6 = baselines_extra.WNO2dOfficial (port of WNO v2.0.0; level 4, width 64; db4 per the paper's Table 1, db6 per
the repository's Darcy script); mgno = baselines_extra.MgNO2D (official MgNO model classes, width 24); transolver =
baselines_extra.Transolver (our re-implementation; 8 layers, width 128, 8 heads, 64 slices, the published Darcy size) on the
85 x 85 grid points as a point cloud.
Training protocols follow the respective papers as far as our code allows: FNO / MgNO: Adam lr 1e-3 wd 1e-4, StepLR(100, 0.5),
batch 20, 500 epochs; WNO: Adam lr 1e-3 wd 1e-6, StepLR(50, 0.75), batch 20, 500 epochs; Transolver: AdamW lr 1e-3 wd 1e-5,
OneCycle, batch 4, 500 epochs.  Pointwise Gaussian normalisation of input and output (FNO code); relative L2 loss.
This file lives in kaggle_runs/, outside the hashed top-level code (code hash unchanged).
usage: python kaggle_runs/published_darcy.py prepare OUT_DIR   (once), then
       python kaggle_runs/published_darcy.py MODEL OUT_DIR [epochs]"""
import sys, os, glob, json, time, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL, OUT = sys.argv[1], sys.argv[2]; EPOCHS = int(sys.argv[3]) if len(sys.argv) > 3 else 500
os.makedirs(OUT, exist_ok=True); dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); torch.manual_seed(0); np.random.seed(0)

def find(name):
    hits = sorted(glob.glob(f'/kaggle/input/**/{name}', recursive=True)) + sorted(glob.glob(f'**/{name}', recursive=True))
    if not hits: raise SystemExit(f'{name} not found under /kaggle/input: attach a Kaggle dataset containing the FNO Darcy files '
                                  '(e.g. Add Input -> Datasets -> scaomath/pde-dataset)')
    return hits[0]
def load(path, n, first=True):
    try:
        import scipy.io; d = scipy.io.loadmat(path); a, u = d['coeff'], d['sol']
    except (NotImplementedError, ValueError):                     # MATLAB v7.3 (full axis reversal, as in FNO's MatReader)
        import h5py; f = h5py.File(path, 'r'); a = np.array(f['coeff']).T; u = np.array(f['sol']).T
    a, u = a[:n] if first else a[-n:], u[:n] if first else u[-n:]
    r = 5
    return (torch.tensor(a[:, ::r, ::r][:, :85, :85], dtype=torch.float32), torch.tensor(u[:, ::r, ::r][:, :85, :85], dtype=torch.float32))
# 11 Oct (Session 9 rerun): the .mat files are 1.7 GB each; MODEL 'prepare' loads them once and caches the 85 x 85 subsample
# (float32, ~64 MB) at CACHE; every training run then reads the cache.  'prepare' also fails within seconds if data are missing.
CACHE = os.environ.get('PD_CACHE', os.path.join(os.path.dirname(os.path.abspath(OUT)), 'pd_data_85.npz'))
NTR = 1000
if os.environ.get('PD_FAKE'):                                     # test-only switch: random data, 40 training samples
    NTR = 40; xtr, ytr, xte, yte = torch.rand(NTR, 85, 85) + .5, torch.rand(NTR, 85, 85), torch.rand(100, 85, 85) + .5, torch.rand(100, 85, 85)
elif MODEL == 'prepare' or not os.path.exists(CACHE):
    p1, p2 = find('piececonst_r421_N1024_smooth1.mat'), find('piececonst_r421_N1024_smooth2.mat'); print('data:', p1, p2, flush=True)
    xtr, ytr = load(p1, 1000); xte, yte = load(p2, 100)
    np.savez(CACHE, xtr=xtr.numpy(), ytr=ytr.numpy(), xte=xte.numpy(), yte=yte.numpy(), src=np.array([p1, p2]))
    print(f'cached {CACHE}: train {tuple(xtr.shape)}, test {tuple(xte.shape)}; a in [{xtr.min():.3g}, {xtr.max():.3g}], '
          f'|u|max {ytr.abs().max():.3g}', flush=True)
else:
    z = np.load(CACHE); xtr, ytr, xte, yte = (torch.tensor(z[k]) for k in ('xtr', 'ytr', 'xte', 'yte')); print('data from cache', CACHE)
if MODEL == 'prepare': raise SystemExit(0)
NTR = len(xtr)
print(f'{MODEL}: train {tuple(xtr.shape)}, test {tuple(xte.shape)}, device {dev}', flush=True)

class Gauss:                                                       # FNO's UnitGaussianNormalizer (pointwise)
    def __init__(self, x, eps=1e-5): self.m, self.s, self.eps = x.mean(0), x.std(0), eps
    def enc(self, x): return (x - self.m.to(x.device)) / (self.s.to(x.device) + self.eps)
    def dec(self, x): return x * (self.s.to(x.device) + self.eps) + self.m.to(x.device)
na, nu_ = Gauss(xtr), Gauss(ytr)
Xtr, Xte, Ytr = na.enc(xtr), na.enc(xte), nu_.enc(ytr)
g = torch.linspace(0, 1, 85); GX, GY = torch.meshgrid(g, g, indexing='ij')
def feats(X, grid):                                                # (B, 85, 85, c)
    f = [X[..., None]]
    if grid: f += [GX.expand(X.shape[0], 85, 85)[..., None], GY.expand(X.shape[0], 85, 85)[..., None]]
    return torch.cat(f, -1)
class _SpectralConv2d(nn.Module):                                 # original FNO code (Li et al., fourier_2d.py)
    def __init__(self, cin, cout, m1, m2):
        super().__init__(); self.m1, self.m2 = m1, m2; sc = 1 / (cin * cout)
        self.w1 = nn.Parameter(sc * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
        self.w2 = nn.Parameter(sc * torch.rand(cin, cout, m1, m2, dtype=torch.cfloat))
    def forward(self, x):
        B = x.shape[0]; xf = torch.fft.rfft2(x)
        out = torch.zeros(B, self.w1.shape[1], x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, :self.m1, :self.m2], self.w1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum('bixy,ioxy->boxy', xf[:, :, -self.m1:, :self.m2], self.w2)
        return torch.fft.irfft2(out, s=(x.size(-2), x.size(-1)))
class FNO2dOriginal(nn.Module):
    def __init__(self, modes=12, width=32, padding=9):
        super().__init__(); self.padding = padding; self.fc0 = nn.Linear(3, width)
        self.convs = nn.ModuleList([_SpectralConv2d(width, width, modes, modes) for _ in range(4)])
        self.ws = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(4)])
        self.fc1 = nn.Linear(width, 128); self.fc2 = nn.Linear(128, 1)
    def forward(self, x):                                            # (B, H, W, 3) -> (B, H, W)
        x = self.fc0(x).permute(0, 3, 1, 2); x = nn.functional.pad(x, [0, self.padding, 0, self.padding])
        for i, (c, w) in enumerate(zip(self.convs, self.ws)):
            x = c(x) + w(x)
            if i < 3: x = nn.functional.gelu(x)
        x = x[..., :-self.padding, :-self.padding].permute(0, 2, 3, 1)
        return self.fc2(nn.functional.gelu(self.fc1(x)))[..., 0]
rel = lambda p, y: (torch.linalg.norm((p - y).reshape(len(y), -1), dim=1) / torch.linalg.norm(y.reshape(len(y), -1), dim=1))

if MODEL == 'fno':
    from ofwno_nets import RealFNO2D
    net = RealFNO2D(3, hidden=32, modes=12, n_layers=4); grid = True; fwd = lambda x: net(x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5); bs = 20
elif MODEL == 'fno-official':
    net = FNO2dOriginal(); grid = True; fwd = lambda x: net(x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5); bs = 20
elif MODEL in ('wno-db4', 'wno-db6'):
    from baselines_extra import WNO2dOfficial
    net = WNO2dOfficial(1, width=64, size=(85, 85), level=4, wavelet=MODEL[-3:]); grid = False; fwd = lambda x: net(x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6); sch = torch.optim.lr_scheduler.StepLR(opt, 50, 0.75); bs = 20
elif MODEL == 'wno-light':                                        # our light WNO layer (paper rows "WNO-op"), width 64
    from ofwno_nets import RealWNO2D
    net = RealWNO2D(3, hidden=64, n_layers=4, J=2); grid = True; fwd = lambda x: net(x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-6); sch = torch.optim.lr_scheduler.StepLR(opt, 50, 0.75); bs = 20
elif MODEL == 'mgno':
    from baselines_extra import MgNO2D
    net = MgNO2D(1, hidden=24); grid = False; fwd = lambda x: net(x)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.StepLR(opt, 100, 0.5); bs = 20
elif MODEL == 'transolver':
    from baselines_extra import Transolver
    net = Transolver(3, hidden=128, layers=8, heads=8, slices=64); grid = True
    fwd = lambda x: net(x.reshape(x.shape[0], -1, x.shape[-1])).reshape(x.shape[0], 85, 85)
    bs = 4; opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=EPOCHS * (NTR // bs))
else:
    raise SystemExit(MODEL)
net = net.to(dev); npar = sum(p.numel() for p in net.parameters())
print(f'{MODEL}: {npar:,} parameters, batch {bs}, {EPOCHS} epochs', flush=True)
Ftr, Fte = feats(Xtr, grid), feats(Xte, grid)
t0 = time.time()
for ep in range(EPOCHS):
    net.train(); perm = torch.randperm(NTR); tl = 0.0
    for i in range(0, NTR, bs):
        idx = perm[i:i + bs]; x, y = Ftr[idx].to(dev), ytr[idx].to(dev)
        p = nu_.dec(fwd(x)); loss = rel(p, y).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if MODEL == 'transolver': sch.step()
        tl += loss.item() * len(idx)
    if MODEL != 'transolver': sch.step()
    if ep % 50 == 0 or ep == EPOCHS - 1:
        net.eval()
        with torch.no_grad(): te = torch.cat([rel(nu_.dec(fwd(Fte[j:j + 10].to(dev))), yte[j:j + 10].to(dev)) for j in range(0, len(Fte), 10)])
        print(f'{MODEL} ep {ep + 1:4d}  train {tl / NTR:.4e}  test rel L2 {te.mean().item():.4e}  ({time.time() - t0:.0f}s)', flush=True)
res = dict(model=MODEL, params=npar, epochs=EPOCHS, batch=bs, test_rel_l2=float(te.mean()), test_rel_l2_s=te.cpu().tolist(),
           train_s=time.time() - t0, published={'fno': 0.0108, 'fno-official': 0.0108, 'wno-db4': 0.0084, 'wno-db6': 0.0084, 'wno-light': None}.get(MODEL))
json.dump(res, open(os.path.join(OUT, f'published_darcy_{MODEL}.json'), 'w'))
print(f'RESULT {MODEL}: test mean relative L2 {res["test_rel_l2"]:.4%} (published {res["published"]})', flush=True)
