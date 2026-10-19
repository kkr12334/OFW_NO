"""The OA-VCB network: one learned multiplicative V-cycle with operator-adapted transfers, near-kernel deflation and a
Galerkin (energy-optimal) output.  Section 3 of the paper:

    x0 = Z (Z^T A Z)^-1 Z^T r;   b = r - A x0;   (x, x_c, x_p) = Cycle(L, b)
    c  = x0 + W y,   W = [x, x_c, x_p, Z],   W^T A W y = W^T b

Only the two shared smoothers are learned; transfers, Galerkin levels, the exact coarsest solve, deflation and the output
projection are prescribed algebra on the per-sample operator data (OAData).
"""
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .smoother import OpSmoother
from .operator_data import OAData

# project: False (plain cycle) | True (Galerkin output in training and evaluation) | 'eval' (output projection at
# evaluation only) | 'both' (loss on the projected AND the plain output; evaluation uses the projection).
# OA-VCB is project='both' with deflation.  The others are the ablations of the paper (2 x 2 factorial, linear controls).
VARIANTS = {
    'oa-vc':   dict(project=False,  deflate=False, linear=False),
    'oa-vcp':  dict(project=True,   deflate=True,  linear=False),
    'oa-vcd':  dict(project='eval', deflate=True,  linear=False),
    'oa-vcb':  dict(project='both', deflate=True,  linear=False),
    'oa-vcz':  dict(project=False,  deflate=True,  linear=False),
    'oa-vcq':  dict(project='both', deflate=False, linear=False),
    'oa-vcl':  dict(project=False,  deflate=True,  linear=True),
    'oa-vcbl': dict(project='both', deflate=True,  linear=True),
}


class OAVCycle(nn.Module):
    def __init__(self, data: OAData, width=16, nu=4, share=True, mlp=True, project=False, deflate=False, ckpt=False,
                 linear=False):
        super().__init__()
        self.ckpt = ckpt                                  # activation checkpointing of the smoothers (training only)
        self.project, self.deflate, self._Z = project, deflate, None
        self.lv, self.n_levels = data.lv, data.n_levels
        nets = 1 if share else self.n_levels
        self.pre = nn.ModuleList([OpSmoother(width, nu, mlp, linear) for _ in range(nets)])
        self.post = nn.ModuleList([OpSmoother(width, nu, mlp, linear) for _ in range(nets)])
        self.share = share

    def set_ops(self, Av, Xv, A0inv, Z=None, A64=None):
        """Operator data of the current batch (OAData.batch / batch_z / batch_a64).  A64: float64 fine operator for the
        safeguard; without it the stored operator cast to double is used."""
        self._A, self._X, self._A0inv, self._Z, self._A64 = Av, Xv, A0inv, Z, A64

    # -- prescribed algebra --------------------------------------------------------------------------------------
    def _matvec64(self, x):
        lv = self.lv[0]; Av = self._A64 if getattr(self, '_A64', None) is not None else self._A[0].double()
        return x.new_zeros(x.shape[0], lv['n']).index_add_(1, lv['Ar'], Av * x[:, lv['Ac']])

    def _matvec(self, l, x):
        lv = self.lv[l]
        return x.new_zeros(x.shape[0], lv['n']).index_add_(1, lv['Ar'], self._A[l] * x[:, lv['Ac']])

    def energy_sq(self, x):
        """x^T A x per sample with the (float32) fine operator of the current batch; used by the energy loss."""
        return (x * self._matvec(0, x)).sum(1)

    def _restrict(self, l, r):                                                  # Phi^T r = r_C + Phat^T r_F
        lv = self.lv[l]; nc = lv['nc']
        return r[:, :nc].index_add(1, lv['Xc'], self._X[l] * r[:, nc:][:, lv['Xr']])

    def _prolong(self, l, e):                                                   # Phi e = [e; Phat e]
        lv = self.lv[l]
        f = e.new_zeros(e.shape[0], lv['n'] - lv['nc']).index_add_(1, lv['Xr'], self._X[l] * e[:, lv['Xc']])
        return torch.cat([e, f], 1)

    def _smooth(self, S, b, l):
        if self.ckpt and self.training and torch.is_grad_enabled():
            return checkpoint(lambda b_: S(b_, self.lv[l], self._A[l]), b, use_reentrant=False)
        return S(b, self.lv[l], self._A[l])

    def _cycle(self, l, b, parts=False):
        j = 0 if self.share else l
        x = self._smooth(self.pre[j], b, l)
        rc = self._restrict(l, b - self._matvec(l, x))
        if l + 1 < self.n_levels:
            ec = self._cycle(l + 1, rc)
        else:
            ec = torch.bmm(self._A0inv, rc[..., None])[..., 0]
        xc = self._prolong(l, ec)
        xp = self._smooth(self.post[j], b - self._matvec(l, x + xc), l)
        return (x, xc, xp) if parts else x + xc + xp

    def _galerkin(self, W, b):
        """Correction d = W y from the Jacobi-scaled, ridge-regularised normal equations of min ||e - W y||_A, A e = b.
        Ideal: the A-orthogonal projection of e onto span W.  Implemented in float64 against the assembled fine operator,
        with a ridge of 1e-10 on the scaled Gram matrix (zero / padding columns masked) and an acceptance test on the
        correction: if 2 d^T b - d^T A d < 0 (round-off in a near-singular solve) d is replaced by the exact line-search step
        d max(d^T b, 0) / d^T A d.  Energy non-increase is therefore kept up to rounding; exact minimisation over span W
        is not.  No eigendecomposition in the forward pass.  Returns d in float64."""
        B, N, m = W.shape
        Wd, bd = W.double(), b.double()
        AW = torch.stack([self._matvec64(Wd[..., j]) for j in range(m)], -1)
        G = Wd.transpose(1, 2) @ AW; g = Wd.transpose(1, 2) @ bd[..., None]
        G = 0.5 * (G + G.transpose(1, 2))
        dg = torch.diagonal(G, dim1=1, dim2=2); live = dg > 1e-30 * dg.amax(1, keepdim=True).clamp_min(1e-300)
        # "double where": the masked branch gets a safe input, otherwise rsqrt'(0) = inf and 0 * inf = NaN in backward
        sc = torch.where(live, torch.where(live, dg, torch.ones_like(dg)).rsqrt(), torch.zeros_like(dg))
        Gs = sc[:, :, None] * G * sc[:, None, :] + torch.diag_embed((~live).double() + 1e-10)
        y = sc[..., None] * torch.linalg.solve(Gs, sc[..., None] * g)
        d = (Wd @ y)[..., 0]
        db = (d * bd).sum(1); dAd = (d * self._matvec64(d)).sum(1)
        bad = (2 * db - dAd) < 0
        alpha = torch.where(bad, db.clamp_min(0) / torch.where(bad, dAd, torch.ones_like(dAd)), torch.ones_like(db))
        st = self.__dict__.setdefault('stats', dict(calls=0, samples=0, backstop=0, rejected=0, cols=0, rank_deficit=0))
        st['calls'] += 1; st['samples'] += B; st['backstop'] += int(bad.sum()); st['rejected'] += int((bad & (db <= 0)).sum())
        if getattr(self, 'collect_stats', False):
            with torch.no_grad():
                ev = torch.linalg.eigvalsh(Gs - torch.diag_embed((~live).double() + 1e-10))
                rk = (ev > 1e-8 * ev.amax(1, keepdim=True).clamp_min(1e-300)).sum(1)
                st['cols'] += int(live.sum()); st['rank_deficit'] += int((live.sum(1) - rk).clamp_min(0).sum())
        return alpha[:, None] * d

    def forward(self, r):
        """r: residual F - A u_c (float64 recommended), shape (B, N); returns the correction c, same dtype.
        The cycle runs in the network's dtype; deflation and the Galerkin output in float64."""
        pdt = next(self.parameters()).dtype
        rn = r.to(pdt)
        if not (self.project or self.deflate):
            return self._cycle(0, rn).to(r.dtype)
        r64 = r.double()
        Z = self._Z if (self.deflate and self._Z is not None) else None
        x0 = self._galerkin(Z, r64) if Z is not None else torch.zeros_like(r64)
        b64 = r64 - self._matvec64(x0); bn = b64.to(pdt)
        self.aux_out = None
        if not self.project or (self.project == 'eval' and self.training):
            return (x0 + self._cycle(0, bn).double()).to(r.dtype)   # plain cycle (project='eval' in training)
        parts = self._cycle(0, bn, parts=True)
        W = torch.stack(parts, -1)
        if Z is not None:
            W = torch.cat([W, Z.to(W.dtype)], -1)
        if self.project == 'both' and self.training:
            self.aux_out = (x0 + (parts[0] + parts[1] + parts[2]).double()).to(r.dtype)   # plain output: second loss term
        return (x0 + self._galerkin(W, b64)).to(r.dtype)


def n_params(model):
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def build_model(data: OAData, variant='oa-vcb', nu=4, params=22000, width=0, act_ckpt=False):
    """OAVCycle of the named variant.  width = 0: the largest latent width whose parameter count fits `params`."""
    v = VARIANTS[variant]
    mk = lambda w: OAVCycle(data, width=w, nu=nu, project=v['project'], deflate=v['deflate'], ckpt=act_ckpt,
                            linear=v['linear'])
    if not width:
        lo, hi = 4, 256
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if n_params(mk(mid)) <= params:
                lo = mid
            else:
                hi = mid
        width = lo
    return mk(width)
