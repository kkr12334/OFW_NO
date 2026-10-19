"""Extra baselines for the OFW-NO comparison.

LiNO   -- Pandey, Patel, Behera, "LiNO: Lifting based multiresolution neural operator", arXiv 2607.02715.
          Re-implemented from the paper (Algorithm 1, Sec. 2.3): conv encoder, L learnable lifting levels
          (predict P_omega / update U_phi conv nets with coordinate channels, per-level parameters,
          fixed anti-aliasing pre-filter), coarse + directional (h, v, d) residual conv operators shared
          across levels, exact inverse lifting, conv decoder.  Grids of any size are zero-padded to a
          multiple of 2^L and cropped back (the paper's own suggestion for non-dyadic sizes).
Transolver -- Wu et al., "Transolver: a fast transformer solver for PDEs on general geometries",
          ICML 2024.  Physics-Attention: points are softly assigned to M learnable slices, attention
          runs among slice tokens, results are broadcast back.  Works directly on mesh nodes.
Both follow the published designs; widths are chosen by the driver to match the parameter budget."""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------------------------------------------------------------------------- LiNO
def _coords(b, h, w, device):
    y = torch.linspace(0, 1, h, device=device); x = torch.linspace(0, 1, w, device=device)
    g = torch.stack(torch.meshgrid(y, x, indexing='ij'), 0)
    return g[None].expand(b, -1, -1, -1)


class _ConvNet(nn.Module):                      # conv - GELU - conv, plus a 1x1 projection bypass
    def __init__(self, cin, cout, width):
        super().__init__()
        self.c1 = nn.Conv2d(cin, width, 3, padding=1); self.c2 = nn.Conv2d(width, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1)
    def forward(self, x):
        return self.c2(F.gelu(self.c1(x))) + self.skip(x)


class _KOp(nn.Module):                           # Y' = W3 * s(W2 * s(W1 * Y)) + Wskip Y   (eq. 18)
    def __init__(self, c):
        super().__init__()
        self.w1 = nn.Conv2d(c, c, 3, padding=1); self.w2 = nn.Conv2d(c, c, 3, padding=1)
        self.w3 = nn.Conv2d(c, c, 3, padding=1); self.skip = nn.Conv2d(c, c, 1)
    def forward(self, y):
        return self.w3(F.gelu(self.w2(F.gelu(self.w1(y))))) + self.skip(y)


class LiNO2D(nn.Module):
    def __init__(self, in_channels, hidden=16, levels=3, out_channels=1):
        super().__init__()
        C = hidden; self.L = levels; self.cout = out_channels
        self.enc = nn.Sequential(nn.Conv2d(in_channels, C, 3, padding=1), nn.GELU(), nn.Conv2d(C, C, 1))
        self.P = nn.ModuleList([_ConvNet(C + 2, 3 * C, C) for _ in range(levels)])
        self.U = nn.ModuleList([_ConvNet(3 * (C + 2), C, C) for _ in range(levels)])
        self.Kc, self.Kh, self.Kv, self.Kd = _KOp(C), _KOp(C), _KOp(C), _KOp(C)
        self.dec = nn.Sequential(nn.Conv2d(C, C, 1), nn.GELU(), nn.Conv2d(C, out_channels, 1))
        k = torch.tensor([1., 2., 1.]); k = (k[:, None] * k[None, :]) / 16
        self.register_buffer('blur', k[None, None])

    def _prefilter(self, x):
        c = x.shape[1]
        return F.conv2d(F.pad(x, (1, 1, 1, 1), mode='replicate'), self.blur.expand(c, 1, 3, 3), groups=c)

    def forward(self, x):                        # x: (B, H, W, cin) -> (B, H, W)
        B, H0, W0, _ = x.shape
        m = 2 ** self.L
        Hp, Wp = math.ceil(H0 / m) * m, math.ceil(W0 / m) * m
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, (0, Wp - W0, 0, Hp - H0))
        X = self.enc(x)
        details = []
        for l in range(self.L):
            X = self._prefilter(X)
            X00, X01 = X[:, :, 0::2, 0::2], X[:, :, 0::2, 1::2]
            X10, X11 = X[:, :, 1::2, 0::2], X[:, :, 1::2, 1::2]
            g = _coords(B, X00.shape[2], X00.shape[3], x.device)
            P01, P10, P11 = self.P[l](torch.cat([X00, g], 1)).chunk(3, 1)
            d = (X01 - P01, X10 - P10, X11 - P11)
            U = self.U[l](torch.cat([d[0], g, d[1], g, d[2], g], 1))
            details.append(d); X = X00 + U
        s = self.Kc(X)
        details = [(self.Kh(a), self.Kv(b), self.Kd(c)) for a, b, c in details]
        for l in reversed(range(self.L)):
            d01, d10, d11 = details[l]
            g = _coords(B, s.shape[2], s.shape[3], x.device)
            X00 = s - self.U[l](torch.cat([d01, g, d10, g, d11, g], 1))
            P01, P10, P11 = self.P[l](torch.cat([X00, g], 1)).chunk(3, 1)
            Hh, Ww = X00.shape[2] * 2, X00.shape[3] * 2
            s = X00.new_zeros(B, X00.shape[1], Hh, Ww)
            s[:, :, 0::2, 0::2] = X00; s[:, :, 0::2, 1::2] = d01 + P01
            s[:, :, 1::2, 0::2] = d10 + P10; s[:, :, 1::2, 1::2] = d11 + P11
        out = self.dec(s)[:, :, :H0, :W0]
        return out[:, 0] if self.cout == 1 else out.permute(0, 2, 3, 1)     # (B,H,W) or (B,H,W,c)


# ------------------------------------------------------------------------------------------ Transolver
class _PhysicsAttention(nn.Module):
    def __init__(self, dim, heads=4, slices=16):
        super().__init__()
        self.h, self.d = heads, dim // heads
        self.inx = nn.Linear(dim, dim); self.infx = nn.Linear(dim, dim)
        self.slice = nn.Linear(self.d, slices)
        self.temperature = nn.Parameter(torch.full((1, heads, 1, 1), 0.5))
        self.q = nn.Linear(self.d, self.d, bias=False); self.k = nn.Linear(self.d, self.d, bias=False)
        self.v = nn.Linear(self.d, self.d, bias=False); self.out = nn.Linear(dim, dim)

    def forward(self, x):                                        # x: (B, N, dim)
        B, N, _ = x.shape
        fx = self.infx(x).view(B, N, self.h, self.d).transpose(1, 2)        # (B,h,N,d)
        xm = self.inx(x).view(B, N, self.h, self.d).transpose(1, 2)
        w = torch.softmax(self.slice(xm) / self.temperature.clamp_min(0.05), -1)   # (B,h,N,M)
        norm = w.sum(2)[..., None] + 1e-5                                    # (B,h,M,1)
        z = torch.einsum('bhnm,bhnd->bhmd', w, fx) / norm                   # slice tokens
        a = torch.softmax(self.q(z) @ self.k(z).transpose(-1, -2) / math.sqrt(self.d), -1)
        z = a @ self.v(z)
        y = torch.einsum('bhmd,bhnm->bhnd', z, w)                           # deslice
        return self.out(y.transpose(1, 2).reshape(B, N, -1))


class Transolver(nn.Module):
    def __init__(self, in_channels, hidden=32, layers=4, heads=4, slices=16, out_channels=1):
        super().__init__()
        self.cout = out_channels
        hidden = max(heads, (hidden // heads) * heads)
        self.pre = nn.Sequential(nn.Linear(in_channels, 2 * hidden), nn.GELU(), nn.Linear(2 * hidden, hidden))
        self.blocks = nn.ModuleList()
        for _ in range(layers):
            self.blocks.append(nn.ModuleDict(dict(
                n1=nn.LayerNorm(hidden), att=_PhysicsAttention(hidden, heads, slices),
                n2=nn.LayerNorm(hidden), mlp=nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.GELU(),
                                                           nn.Linear(2 * hidden, hidden)))))
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, out_channels))

    def forward(self, x):                                        # (B, N, cin) -> (B, N)
        h = self.pre(x)
        for b in self.blocks:
            h = h + b['att'](b['n1'](h))
            h = h + b['mlp'](b['n2'](h))
        out = self.head(h)
        return out[..., 0] if self.cout == 1 else out


# ------------------------------------------------------------------------------------ WNO (official)
# Faithful port of the published WNO, Version 2.0.0 of https://github.com/TapasTripura/WNO
# (wavelet_convolution.WaveConv2d + wno2d_*_dwt.WNO2d): fc0 lifting of (input, grid), `layers` blocks
# x -> WaveConv2d(x) + Conv1x1(x) with mish between blocks (none after the last), fc1 (128) -> gelu -> fc2.
# WaveConv2d: DWT to `level` (pytorch_wavelets, mode 'symmetric'), SPATIALLY VARYING weights
# (in, out, m1, m2) on the approximation and on the three detail bands of the COARSEST level only; all finer
# detail bands are zeroed; IDWT.  Official settings: db6, width 64, 4 layers, level 4 (Darcy 85^2), 3 (notch
# 51^2), 2 (Allen-Cahn 43^2), padding 1.  Differences from the published code, none affecting the model:
# the input already carries its grid channels (we still append the official grid, as they do), and the
# output is cropped back to the input size when the IDWT returns one extra row/column (odd sizes).
def _wno_level_for(n, wavelet='db6', target=16):
    """largest level whose approximation grid is still >= `target` points per side (official configs
    give 14-20 there); at least 1."""
    import pywt
    m = n; L_ = 0
    flen = pywt.Wavelet(wavelet).dec_len
    while True:
        m_next = pywt.dwt_coeff_len(m, flen, 'symmetric')
        if m_next < target: break
        m = m_next; L_ += 1
    return max(L_, 1)


class WaveConv2dOfficial(nn.Module):
    def __init__(self, in_channels, out_channels, level, size, wavelet='db6', mode='symmetric'):
        super().__init__()
        from pytorch_wavelets import DWT, IDWT
        self.level, self.size, self.wavelet, self.mode = level, list(size), wavelet, mode
        self.dwt = DWT(J=level, mode=mode, wave=wavelet)
        self.idwt = IDWT(mode=mode, wave=wavelet)
        with torch.no_grad():
            yl, _ = self.dwt(torch.zeros(1, 1, *self.size))
        m1, m2 = yl.shape[-2:]
        s = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(s * torch.rand(in_channels, out_channels, m1, m2))
        self.weights2 = nn.Parameter(s * torch.rand(in_channels, out_channels, m1, m2))
        self.weights3 = nn.Parameter(s * torch.rand(in_channels, out_channels, m1, m2))
        self.weights4 = nn.Parameter(s * torch.rand(in_channels, out_channels, m1, m2))

    @staticmethod
    def mul2d(x, w):
        return torch.einsum("bixy,ioxy->boxy", x, w)

    def forward(self, x):
        from pytorch_wavelets import DWT
        if x.shape[-1] != self.size[-1]:                  # resolution change, as in the official code
            f = int(np.log2(max(x.shape[-1], self.size[-1]) // min(x.shape[-1], self.size[-1])))
            J = self.level + f if x.shape[-1] > self.size[-1] else self.level - f
            x_ft, x_coeff = DWT(J=J, mode=self.mode, wave=self.wavelet).to(x.device)(x)
        else:
            x_ft, x_coeff = self.dwt(x)
        out_ft = self.mul2d(x_ft, self.weights1)
        out_coeff = [torch.zeros_like(c) for c in x_coeff]
        out_coeff[-1] = torch.stack([self.mul2d(x_coeff[-1][:, :, 0], self.weights2),
                                     self.mul2d(x_coeff[-1][:, :, 1], self.weights3),
                                     self.mul2d(x_coeff[-1][:, :, 2], self.weights4)], 2)
        y = self.idwt((out_ft, out_coeff))
        return y[..., :x.shape[-2], :x.shape[-1]]


class WNO2dOfficial(nn.Module):
    """(B, H, W, cin) -> (B, H, W) [or (B, H, W, cout)].  `width` is the official lifting width."""
    def __init__(self, in_channels, width=64, size=(64, 64), level=None, layers=4, wavelet='db6', padding=1,
                 out_channels=1):
        super().__init__()
        import numpy as _np
        self.padding, self.cout = padding, out_channels
        sz = [size[0] + padding, size[1] + padding]
        self.level = level or _wno_level_for(min(sz), wavelet)
        self.fc0 = nn.Linear(in_channels + 2, width)
        self.conv = nn.ModuleList([WaveConv2dOfficial(width, width, self.level, sz, wavelet) for _ in range(layers)])
        self.w = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(layers)])
        self.fc1 = nn.Linear(width, 128); self.fc2 = nn.Linear(128, out_channels)

    def forward(self, x):
        B, H, W, _ = x.shape
        gx = torch.linspace(0, 1, H, device=x.device).reshape(1, H, 1, 1).expand(B, H, W, 1)
        gy = torch.linspace(0, 1, W, device=x.device).reshape(1, 1, W, 1).expand(B, H, W, 1)
        x = self.fc0(torch.cat([x, gx, gy], -1)).permute(0, 3, 1, 2)
        if self.padding: x = F.pad(x, [0, self.padding, 0, self.padding])
        for i, (c, w) in enumerate(zip(self.conv, self.w)):
            x = c(x) + w(x)
            if i != len(self.conv) - 1: x = F.mish(x)
        if self.padding: x = x[..., :-self.padding, :-self.padding]
        x = self.fc2(F.gelu(self.fc1(x.permute(0, 2, 3, 1))))
        return x[..., 0] if self.cout == 1 else x


# ------------------------------------------------------------------------------------ graph baseline
class MeshGNN(nn.Module):
    """MeshGraphNet-style encode-process-decode GNN (Pfaff et al., ICLR 2021) on the fine mesh edges.
    Edge features: relative position and length (in units of the mean edge length).  `steps` unshared
    processor blocks, each an edge update e <- e + MLP([e, h_src, h_dst]) and a node update
    h <- h + MLP([h, mean_in(e)]), MLPs with LayerNorm as in MGN.  Receptive field = `steps` hops."""
    def __init__(self, in_channels, edges, rel, width=32, steps=10, out_channels=1, op_edge=False, op_dim=1):
        super().__init__()
        self.set_graph(edges, rel)
        # op_edge (2 Oct, 'gnn-op'): one extra PER-SAMPLE edge feature, the operator weight a_ij / a_ii (set_op),
        # so the GNN gets exactly the operator information oa-ll-op gets (fair comparison).  op_dim (3 Oct): number
        # of such features per edge -- 4 for elasticity (the 2x2 block D_i^-1 A_ij, flattened); 1 reproduces gnn-op.
        self.op_edge, self._w = op_edge, None
        self.cout = out_channels
        mlp = lambda i, o: nn.Sequential(nn.Linear(i, width), nn.GELU(), nn.Linear(width, o), nn.LayerNorm(o))
        self.enc_n = mlp(in_channels, width); self.enc_e = mlp(3 + op_dim if op_edge else 3, width)
        self.edge = nn.ModuleList([mlp(3 * width, width) for _ in range(steps)])
        self.node = nn.ModuleList([mlp(2 * width, width) for _ in range(steps)])
        self.dec = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, out_channels))

    def set_op(self, w):
        self._w = w

    def set_graph(self, edges, rel):
        """(re)bind the mesh graph -- the weights do not depend on it, so a trained model runs on another mesh.
        Graph tensors are plain attributes (not buffers): they are not part of the saved state."""
        dev = next(self.parameters()).device if len(list(self.parameters())) else 'cpu'
        self.src = torch.as_tensor(edges[0], dtype=torch.long, device=dev)
        self.dst = torch.as_tensor(edges[1], dtype=torch.long, device=dev)
        self.ef = torch.as_tensor(rel, dtype=torch.float32, device=dev)
        n = int(max(edges[0].max(), edges[1].max())) + 1
        self.deg = torch.bincount(self.dst, minlength=n).clamp(min=1).float()

    def _apply(self, fn, *a, **k):                              # keep the graph on the model's device
        out = super()._apply(fn, *a, **k)
        for nm in ('src', 'dst', 'ef', 'deg'):
            if hasattr(self, nm): setattr(self, nm, fn(getattr(self, nm)))
        return out

    def forward(self, x):                                       # x: (B, N, cin) -> (B, N)
        B, N, _ = x.shape
        if self.op_edge:
            w = self._w[..., None] if self._w.dim() == 2 else self._w.reshape(B, self._w.shape[1], -1)
            e = self.enc_e(torch.cat([self.ef[None].expand(B, -1, -1), w], -1))
        else:
            e = self.enc_e(self.ef)[None].expand(B, -1, -1)
        h = self.enc_n(x)
        for fe, fn in zip(self.edge, self.node):
            e = e + fe(torch.cat([e, h[:, self.src], h[:, self.dst]], -1))
            agg = h.new_zeros(B, N, e.shape[-1]).index_add_(1, self.dst, e) / self.deg[None, :, None]
            h = h + fn(torch.cat([h, agg], -1))
        out = self.dec(h)
        return out[..., 0] if self.cout == 1 else out


def mesh_edges(coords, elem, nodes):
    """Directed edges (both ways) of the mesh restricted to `nodes` (global ids, in the model's node order),
    with features (dx, dy, |d|) / mean edge length."""
    import numpy as _np
    pos = -_np.ones(len(coords), dtype=_np.int64); pos[_np.asarray(nodes)] = _np.arange(len(nodes))
    e = _np.concatenate([elem[:, [0, 1]], elem[:, [1, 2]], elem[:, [2, 0]]])
    e = _np.unique(_np.sort(e, 1), axis=0)
    e = e[(pos[e[:, 0]] >= 0) & (pos[e[:, 1]] >= 0)]
    s = _np.concatenate([e[:, 0], e[:, 1]]); d = _np.concatenate([e[:, 1], e[:, 0]])
    rel = coords[d] - coords[s]; L = _np.linalg.norm(rel, axis=1); h = L.mean()
    feat = _np.column_stack([rel / h, L / h])
    return _np.stack([pos[s], pos[d]]), feat


# ---------------------------------------------------------------------------------------------------------------
# MgNO (He, Liu, Xu, ICLR 2024, "MgNO: Efficient Parameterization of Linear Operators via Multigrid").  The model
# classes below are taken from the official code, github.com/xlliu2017/MgNO (models.py: MgIte, MgIte_init, Restrict,
# MgConv_DC, MgNO_DC; MIT licence, (c) 2024 Xinliang Liu), unchanged except for the removed torchinfo import.
# MgNO2D wraps MgNO_DC for our channel-last grid interface: (B, ng, ng, cin) -> (B, ng, ng), padding the grid to a
# multiple of 2^(levels-1) (stride-2 convolutions) and cropping back.  Default V-cycle of their Darcy script.
# ---------------------------------------------------------------------------------------------------------------
class _MgIte(nn.Module):
    def __init__(self, A, S):
        super().__init__(); self.A = A; self.S = S
    def forward(self, out):
        if isinstance(out, tuple):
            u, f = out; u = u + self.S(f - self.A(u))
        else:
            f = out; u = self.S(f)
        return (u, f)


class _MgIte_init(nn.Module):
    def __init__(self, S):
        super().__init__(); self.S = S
    def forward(self, f):
        return (self.S(f), f)


class _Restrict(nn.Module):
    def __init__(self, Pi=None, R=None, A=None):
        super().__init__(); self.Pi = Pi; self.R = R; self.A = A
    def forward(self, out):
        u, f = out
        f = self.R(f - self.A(u)) if self.A is not None else self.R(f)
        return (self.Pi(u), f)


class _MgConv_DC(nn.Module):
    def __init__(self, num_iteration, num_channel_u, num_channel_f, padding_mode='zeros', bias=False, use_res=False):
        super().__init__()
        self.num_iteration = num_iteration; self.num_channel_u = num_channel_u; self.padding_mode = padding_mode
        self.RTlayers = nn.ModuleList([nn.ConvTranspose2d(num_channel_u, num_channel_u, kernel_size=4, stride=2,
                                                          padding=1, bias=False) for _ in range(len(num_iteration) - 1)])
        layers = []
        for l, num_iteration_l in enumerate(num_iteration):
            post_smooth_layers = []
            for i in range(num_iteration_l[0]):
                S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                if l == 0 and i == 0:
                    layers.append(_MgIte_init(S))
                else:
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    layers.append(_MgIte(A, S))
            if num_iteration_l[1] != 0:
                for _ in range(num_iteration_l[1]):
                    S = nn.Conv2d(num_channel_f, num_channel_u, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                    post_smooth_layers.append(_MgIte(A, S))
            else:
                post_smooth_layers.append(nn.Identity())
            setattr(self, 'layer' + str(l), nn.Sequential(*layers))
            setattr(self, 'post_smooth_layer' + str(l), nn.Sequential(*post_smooth_layers))
            if l < len(num_iteration) - 1:
                A = nn.Conv2d(num_channel_u, num_channel_f, kernel_size=3, stride=1, padding=1, bias=bias, padding_mode=padding_mode)
                Pi = nn.Conv2d(num_channel_u, num_channel_u, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                R = nn.Conv2d(num_channel_f, num_channel_f, kernel_size=3, stride=2, padding=1, bias=False, padding_mode=padding_mode)
                layers = [_Restrict(Pi, R, A) if use_res else _Restrict(Pi=Pi, R=R)]

    def forward(self, f):
        out_list = [0] * len(self.num_iteration); out = f
        for l in range(len(self.num_iteration)):
            out = getattr(self, 'layer' + str(l))(out); out_list[l] = out
        for j in range(len(self.num_iteration) - 2, -1, -1):
            u, f = out_list[j][0], out_list[j][1]
            u_post = u + self.RTlayers[j](out_list[j + 1][0])
            out_list[j] = getattr(self, 'post_smooth_layer' + str(j))((u_post, f))
        return out_list[0][0]


class _MgNO_DC(nn.Module):
    def __init__(self, num_layer, num_channel_u, num_channel_f, num_iteration, padding_mode='zeros'):
        super().__init__()
        self.num_layer = num_layer
        self.conv_list = nn.ModuleList([_MgConv_DC(num_iteration, num_channel_u, num_channel_f, padding_mode=padding_mode)])
        self.linear_list = nn.ModuleList([nn.Conv2d(num_channel_f, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True)])
        for _ in range(num_layer - 1):
            self.conv_list.append(_MgConv_DC(num_iteration, num_channel_u, num_channel_u, padding_mode=padding_mode))
            self.linear_list.append(nn.Conv2d(num_channel_u, num_channel_u, kernel_size=1, stride=1, padding=0, bias=True))
        self.linear = nn.Conv2d(num_channel_u, 1, kernel_size=1, bias=False)
        self.act = nn.GELU()

    def forward(self, u):
        for i in range(self.num_layer):
            u = self.act(self.conv_list[i](u) + self.linear_list[i](u))
        return self.linear(u)


class MgNO2D(nn.Module):
    """MgNO_DC on our grid interface; num_iteration = their Darcy default [[1,0],[1,0],[1,0],[2,0],[2,0]] (5 levels)."""
    def __init__(self, cin, hidden, num_layer=5, num_iteration=((1, 0), (1, 0), (1, 0), (2, 0), (2, 0))):
        super().__init__()
        self.mult = 2 ** (len(num_iteration) - 1)
        self.net = _MgNO_DC(num_layer, hidden, cin, [list(t) for t in num_iteration])

    def forward(self, x):                                  # (B, ng, ng, cin) -> (B, ng, ng)
        B, n1, n2, _ = x.shape
        p1, p2 = (-n1) % self.mult, (-n2) % self.mult
        y = F.pad(x.permute(0, 3, 1, 2), (0, p2, 0, p1))
        return self.net(y)[:, 0, :n1, :n2]
