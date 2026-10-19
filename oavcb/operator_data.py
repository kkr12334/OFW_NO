"""OAData: the per-sample operator data of all samples as tensors on fixed sparsity patterns, ready for batching."""
import numpy as np
import torch

from .transfers import oa_hierarchy, structural_patterns, on_pattern, near_kernel_basis


class OAData:
    """For every sample: the level operators A_k and transfer blocks Phat_k as values on a per-level fixed pattern, the
    inverse of the coarsest operator A_0, and (optionally) the near-kernel vectors Z, padded with zero columns.

    H            Hierarchy;   A_list  iterable of fine operators (any SPD stiffness on H.nodes[L]; a tensor coefficient
                 or an elasticity matrix works as well, with `A_struct` giving the structural pattern of the class)
    near_kernel  detect Z for every sample (scalar diffusion only)
    """

    def __init__(self, H, A_list, n_levels, m, device, near_kernel=True, A_struct=None, theta=0.1, tau=0.05,
                 dtype=torch.float32):
        pats, n0 = structural_patterns(H, n_levels, m, A_struct)
        self.n_levels, self.m, self.L, self.lv = n_levels, m, H.L, []
        for l, (pA, pX) in enumerate(pats):
            k = H.L - l
            kA = pA.row.astype(np.int64) * pA.shape[1] + pA.col; oA = np.argsort(kA)
            kX = pX.row.astype(np.int64) * pX.shape[1] + pX.col; oX = np.argsort(kX)
            r, c = pA.row[oA], pA.col[oA]; off = r != c
            xy = H.level_coords(k); rel = (xy[c] - xy[r]) / H.h[k]
            T = lambda z, dt=torch.long: torch.tensor(np.ascontiguousarray(z), dtype=dt, device=device)
            self.lv.append(dict(k=k, n=pA.shape[0], nc=H.P[k].shape[1],
                                Ar=T(r), Ac=T(c), off=T(np.nonzero(off)[0]), diag=T(np.nonzero(~off)[0]),
                                diag_row=T(r[~off]),
                                erel=T(np.column_stack([rel, np.linalg.norm(rel, axis=1)])[off], torch.float32),
                                deg=T(np.bincount(r[off], minlength=pA.shape[0]).clip(min=1), torch.float32),
                                Xr=T(pX.row[oX]), Xc=T(pX.col[oX])))
        Avals = [[] for _ in pats]; Xvals = [[] for _ in pats]; A0inv = []; Zs = []
        for A_ in A_list:
            levs, A0 = oa_hierarchy(H, A_, n_levels, m)
            if near_kernel:
                Zs.append(near_kernel_basis(levs[0][0], theta, tau))
            for l, ((Ak, X), (pA, pX)) in enumerate(zip(levs, pats)):
                Avals[l].append(on_pattern(Ak, pA)[0]); Xvals[l].append(on_pattern(X, pX)[0])
            A0inv.append(np.linalg.inv(A0.toarray()))
        f32 = lambda z: torch.tensor(np.array(z), dtype=dtype, device=device)   # float32 for training; float64 for checks
        self.Av = [f32(v) for v in Avals]; self.Xv = [f32(v) for v in Xvals]; self.A0inv = f32(A0inv); self.n0 = n0
        # the fine operator also in float64, exactly as assembled: the output safeguard is evaluated with it
        self.A64 = torch.tensor(np.array(Avals[0]), dtype=torch.float64, device=device)
        self.Z, self.k_nk = None, [z.shape[1] for z in Zs]
        if near_kernel:                                        # padded with zero columns (masked in the projection)
            km = max(1, max(self.k_nk)); N = pats[0][0].shape[0]
            Zp = np.zeros((len(Zs), N, km))
            for i, z in enumerate(Zs):
                Zp[i, :, :z.shape[1]] = z
            self.Z = f32(Zp)

    def __len__(self):
        return self.A0inv.shape[0]

    def batch(self, idx):
        return [v[idx] for v in self.Av], [v[idx] for v in self.Xv], self.A0inv[idx]

    def batch_z(self, idx):
        return None if self.Z is None else self.Z[idx]

    def batch_a64(self, idx):
        return self.A64[idx]

    def set_batch(self, model, idx):
        """Point `model` at the operators of samples idx."""
        model.set_ops(*self.batch(idx), Z=self.batch_z(idx), A64=self.batch_a64(idx))
