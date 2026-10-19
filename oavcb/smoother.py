"""The learned operator-consistent smoother (eq. 6 of the paper): the only trainable part of OA-VCB.

    S(b; A) = sigma * net(D^-1 b / sigma),   sigma = rms(D^-1 b),   D = diag(A)

net: encoder -> nu steps  s <- s + L([s, sum_j W_ij V s_j]) + MLP([s, sum_j W_ij V s_j, mean edge message]),  W_ij = A_ij / A_ii
-> decoder.  The operator enters only through W and D of the level the smoother acts on, and S is positively homogeneous in b.
Module and parameter names are those of the paper's checkpoints.
"""
import torch
import torch.nn as nn


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Linear(h, o))


class OpSmoother(nn.Module):
    """b -> correction x for A x = b on one level.
    linear=True keeps only the linear path (residual-linear control: S(b) = L(D^-1 b))."""

    def __init__(self, C, nu, mlp=True, linear=False):
        super().__init__()
        self.C, self.nu, self.linear = C, nu, linear
        self.mlp = mlp and not linear
        self.enc_lin = nn.Linear(2, C, bias=False); self.enc = None if linear else _mlp(2, 2 * C, C)
        self.V = nn.ModuleList([nn.Linear(C, C, bias=False) for _ in range(nu)])
        self.Lw = nn.ModuleList([nn.Linear(2 * C, C, bias=False) for _ in range(nu)])
        if mlp:
            self.msg = nn.ModuleList([_mlp(2 * C + 4, C, C) for _ in range(nu)])
            self.upd = nn.ModuleList([_mlp(3 * C, C, C) for _ in range(nu)])
        self.dec_lin = nn.Linear(C, 1, bias=False); self.dec = None if linear else _mlp(C, C, 1)

    def forward(self, b, lv, Avals):
        B = b.shape[0]; d = Avals[:, lv['diag']]                               # (B, n) diagonal, pattern order = rows
        dg = b.new_zeros(B, lv['n']).index_add_(1, lv['diag_row'], d)
        z = b / dg
        # sigma = max|z| * rms(z / max|z|): no float32 underflow of z^2 for tiny residuals.  An exactly zero right-hand
        # side gives S(0) = 0, with safe values in the masked branch so that the backward pass stays finite.
        mz = z.abs().amax(1, keepdim=True); nz = mz > 0
        mz = torch.where(nz, mz, torch.ones_like(mz))
        q = (z / mz).pow(2).mean(1, keepdim=True)
        sig = mz * torch.where(nz, q, torch.ones_like(q)).sqrt()
        lsd = torch.log(dg / dg.median(1, keepdim=True).values)
        x = torch.stack([z / sig, torch.zeros_like(lsd) if self.linear else lsd], -1)   # (B, n, 2)
        s = self.enc_lin(x) if self.linear else self.enc(x) + self.enc_lin(x)
        r_, c_ = lv['Ar'][lv['off']], lv['Ac'][lv['off']]
        w = (Avals[:, lv['off']] / dg[:, r_])[..., None]                        # W_ij = A_ij / A_ii (row i)
        ef = torch.cat([lv['erel'][None].expand(B, -1, -1), w], -1)
        for t in range(self.nu):
            opa = s.new_zeros(s.shape).index_add_(1, r_, w * self.V[t](s)[:, c_])
            ds = self.Lw[t](torch.cat([s, opa], -1))
            if self.mlp:
                nl = s.new_zeros(s.shape).index_add_(
                    1, r_, self.msg[t](torch.cat([s[:, r_], s[:, c_], ef], -1))) / lv['deg'][None, :, None]
                ds = ds + self.upd[t](torch.cat([s, opa, nl], -1))
            s = s + ds
        y = self.dec_lin(s) if self.linear else self.dec_lin(s) + self.dec(s)
        return torch.where(nz, y[..., 0] * sig, torch.zeros_like(b))
