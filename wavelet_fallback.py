"""Minimal DWTForward / DWTInverse stand-in: separable 2D orthogonal DWT with
periodic boundary handling, built as an explicit orthogonal matrix per level so
perfect reconstruction is exact by construction.

API-compatible with pytorch_wavelets for the calls RealWNO2D makes:
    DWTForward(J, wave, mode)(x) -> (Yl, [Yh_0 .. Yh_{J-1}])   Yh_j: (B,C,3,h,w)
    DWTInverse(wave, mode)((Yl, Yh)) -> (B,C,H,W)
This is NOT bit-identical to pytorch_wavelets; it reproduces the subband
layout, the odd-length padding behaviour and the tensor shapes, which is what
the network plumbing depends on.
"""
import torch, torch.nn as nn

_REC_LO = {
 'db1': [0.7071067811865476, 0.7071067811865476],
 'db4': [0.23037781330885523, 0.7148465705525415, 0.6308807679295904,
         -0.02798376941698385, -0.18703481171888114, 0.030841381835986965,
         0.032883011666982945, -0.010597401784997278],
}

def _qmf(wave):
    h = torch.tensor(_REC_LO[wave], dtype=torch.float64)
    k = torch.arange(h.numel(), dtype=torch.float64)
    g = ((-1.0) ** k) * h.flip(0)
    return h, g

def _wmat(N, wave):
    """Orthogonal periodic analysis matrix: rows 0..N/2-1 low, N/2..N-1 high."""
    h, g = _qmf(wave)
    L, M = h.numel(), N // 2
    W = torch.zeros(N, N, dtype=torch.float64)
    idx = torch.arange(L)
    for k in range(M):
        cols = (2 * k + idx) % N
        W[k].index_add_(0, cols, h)
        W[M + k].index_add_(0, cols, g)
    return W.float()

class _Base(nn.Module):
    def __init__(self, wave):
        super().__init__()
        self.wave, self._cache = wave, {}
    def W(self, N, device):
        key = (N, device)
        if key not in self._cache:
            self._cache[key] = _wmat(N, self.wave).to(device)   # buffer-like, not a Parameter
        return self._cache[key]

class DWTForward(_Base):
    def __init__(self, J=1, wave='db4', mode='periodization'):
        super().__init__(wave); self.J, self.mode = J, mode
    def forward(self, x):
        yh = []
        for _ in range(self.J):
            if x.shape[-1] % 2: x = torch.cat([x, x[..., -1:]], dim=-1)
            if x.shape[-2] % 2: x = torch.cat([x, x[..., -1:, :]], dim=-2)
            H, Wd = x.shape[-2], x.shape[-1]
            Wr, Wc = self.W(H, x.device), self.W(Wd, x.device)
            Y = Wr @ x @ Wc.T
            m, n = H // 2, Wd // 2
            yh.append(torch.stack([Y[..., :m, n:], Y[..., m:, :n], Y[..., m:, n:]], dim=2))
            x = Y[..., :m, :n]
        return x, yh   # yh[0] = finest scale (pytorch_wavelets convention)

class DWTInverse(_Base):
    def __init__(self, wave='db4', mode='periodization'):
        super().__init__(wave); self.mode = mode
    def forward(self, coeffs):
        ll, yh = coeffs
        for band in yh[::-1]:
            lh, hl, hh = band[:, :, 0], band[:, :, 1], band[:, :, 2]
            m, n = lh.shape[-2], lh.shape[-1]
            ll = ll[..., :m, :n]
            Y = torch.cat([torch.cat([ll, lh], -1), torch.cat([hl, hh], -1)], -2)
            Wr, Wc = self.W(2 * m, Y.device), self.W(2 * n, Y.device)
            ll = Wr.T @ Y @ Wc
        return ll
