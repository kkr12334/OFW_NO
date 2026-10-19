"""Networks for the OFW-NO baseline comparison.

Model classes are taken verbatim from the user's untitled9.py; the only
additions are the import guard (falls back to wavelet_fallback when
pytorch_wavelets is absent) and the exact mesh<->grid bijection helpers.
"""
import inspect
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from typing import Dict

try:
    from pytorch_wavelets import DWTForward, DWTInverse
    _HAVE_PYTORCH_WAVELETS = True
    _WAVELET_BACKEND = "pytorch_wavelets"
except ImportError:
    from wavelet_fallback import DWTForward, DWTInverse
    _HAVE_PYTORCH_WAVELETS = True
    _WAVELET_BACKEND = "wavelet_fallback (separable orthogonal DWT)"

class PerLevelBlock(nn.Module):
    def __init__(self, in_dim, mu_dim, out_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, out_dim),
        )
        self.skip = nn.Linear(mu_dim, out_dim)

    def forward(self, x, mu):
        return self.net(x) + self.skip(mu)


class CrossLevelCoupling(nn.Module):
    def __init__(self, coarse_dim, proj_dim=16):
        super().__init__()
        self.proj = nn.Linear(coarse_dim, proj_dim) if coarse_dim > 0 else None
        self.proj_dim = proj_dim if coarse_dim > 0 else 0

    def forward(self, coarse_concat):
        if self.proj is None:
            return torch.zeros(coarse_concat.shape[0], 0, device=coarse_concat.device)
        return torch.tanh(self.proj(coarse_concat))


class OFWNeuralOperator(nn.Module):
    def __init__(self, mu_dim, level_dims, hidden=64, proj_dim=16):
        super().__init__()
        self.max_level = max(level_dims.keys())
        self.level_dims = level_dims
        self.mu_encoder = nn.Sequential(
            nn.Linear(mu_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
        )
        self.blocks = nn.ModuleDict()
        self.couplings = nn.ModuleDict()
        cumulative_dim = 0
        for lvl in range(self.max_level + 1):
            coupling = CrossLevelCoupling(cumulative_dim, proj_dim=proj_dim)
            self.couplings[str(lvl)] = coupling
            in_dim = hidden + coupling.proj_dim
            self.blocks[str(lvl)] = PerLevelBlock(in_dim, mu_dim, level_dims[lvl], hidden=hidden)
            cumulative_dim += level_dims[lvl]

    def forward(self, mu):
        feat = self.mu_encoder(mu)
        outputs = {}
        coarse_concat = torch.zeros(mu.shape[0], 0, device=mu.device)
        for lvl in range(self.max_level + 1):
            coupling_feat = self.couplings[str(lvl)](coarse_concat)
            block_in = torch.cat([feat, coupling_feat], dim=-1)
            c_lvl = self.blocks[str(lvl)](block_in, mu)
            outputs[lvl] = c_lvl
            coarse_concat = torch.cat([coarse_concat, c_lvl.detach()], dim=-1)
        return outputs


class LinearSkipOFWNeuralOperator(nn.Module):
    def __init__(self, mu_dim, level_dims, hidden=64, proj_dim=16):
        super().__init__()
        self.base = OFWNeuralOperator(mu_dim, level_dims, hidden=hidden, proj_dim=proj_dim)
        self.max_level = self.base.max_level
        self.linear_heads = nn.ModuleDict({
            str(lvl): nn.Linear(mu_dim, level_dims[lvl], bias=False)
            for lvl in range(self.max_level + 1)
        })
        sample_block = next(iter(self.base.blocks.values()))
        n_params = len(inspect.signature(sample_block.forward).parameters)
        self._block_needs_mu = n_params >= 2

    def forward(self, mu):
        feat = self.base.mu_encoder(mu)
        outputs = {}
        coarse_concat = torch.zeros(mu.shape[0], 0, device=mu.device)
        for lvl in range(self.max_level + 1):
            coupling_feat = self.base.couplings[str(lvl)](coarse_concat)
            block_in = torch.cat([feat, coupling_feat], dim=-1)
            block = self.base.blocks[str(lvl)]
            nonlinear_out = block(block_in, mu) if self._block_needs_mu else block(block_in)
            c_lvl = nonlinear_out + self.linear_heads[str(lvl)](mu)
            outputs[lvl] = c_lvl
            coarse_concat = torch.cat([coarse_concat, c_lvl.detach()], dim=-1)
        return outputs

    def warm_start_linear(self, mu_train: np.ndarray, c_train: Dict[int, np.ndarray]):
        with torch.no_grad():
            for lvl in range(self.max_level + 1):
                Y = c_train[lvl]
                if Y.shape[1] == 0:
                    continue
                W, *_ = np.linalg.lstsq(mu_train, Y, rcond=None)
                self.linear_heads[str(lvl)].weight.copy_(torch.tensor(W.T, dtype=torch.float32))

    def shrink_nonlinear_branch(self, scale: float = 1e-3):
        for block in self.base.blocks.values():
            final_layer = block.net[-1]
            nn.init.uniform_(final_layer.weight, -scale, scale)
            if final_layer.bias is not None:
                nn.init.zeros_(final_layer.bias)
            if hasattr(block, "skip") and isinstance(block.skip, nn.Linear):
                nn.init.zeros_(block.skip.weight)
                if block.skip.bias is not None:
                    nn.init.zeros_(block.skip.bias)


def build_calibrated_model(mu_dim, level_dims, mu_train, c_train, hidden=64, proj_dim=16):
    model = LinearSkipOFWNeuralOperator(mu_dim, level_dims, hidden=hidden, proj_dim=proj_dim)
    model.warm_start_linear(mu_train, c_train)
    model.shrink_nonlinear_branch()
    return model


# =====================================================================
# 1. EXACT mesh <-> grid bijection (no interpolation, ever)
# =====================================================================
def build_grid_bijection(mesh, max_level: int):
    """Returns (grid_to_node, node_to_grid, n, grid_xy).

    grid_to_node : (n, n) int array, grid_to_node[i, j] = mesh node index
    node_to_grid : (n_nodes, 2) int array, node_to_grid[k] = (i, j)
    n            : grid side length, 2**max_level + 1
    grid_xy      : (n, n, 2) float array of the actual (x, y) coordinates

    Raises AssertionError if the mesh is not an exact uniform grid at this
    level (it always is for MeshHierarchy's criss-cross refinement, but we
    check rather than assume, since this is the linchpin of the whole
    "no interpolation" fix).
    """
    coords = np.asarray(mesh.coords)
    n = 2 ** max_level + 1
    ij = np.rint(coords * (2 ** max_level)).astype(int)
    assert ij.min() >= 0 and ij.max() <= 2 ** max_level, \
        "mesh coordinates fall outside the expected dyadic grid"
    grid_to_node = -np.ones((n, n), dtype=int)
    for node_idx, (i, j) in enumerate(ij):
        assert grid_to_node[i, j] == -1, f"duplicate node at grid cell ({i},{j})"
        grid_to_node[i, j] = node_idx
    assert (grid_to_node >= 0).all(), "mesh does not cover the full uniform grid"
    node_to_grid = ij
    lin = np.linspace(0.0, 1.0, n)
    gx, gy = np.meshgrid(lin, lin, indexing='ij')
    grid_xy = np.stack([gx, gy], axis=-1)
    # sanity: coordinates recovered via the bijection must match mesh.coords exactly
    recovered = grid_xy[node_to_grid[:, 0], node_to_grid[:, 1]]
    assert np.allclose(recovered, coords, atol=1e-10), "bijection does not round-trip exactly"
    return grid_to_node, node_to_grid, n, grid_xy


def nodal_to_grid(nodal_batch: np.ndarray, grid_to_node: np.ndarray) -> np.ndarray:
    """(B, n_nodes) -> (B, n, n). Exact, no interpolation."""
    return nodal_batch[:, grid_to_node]


def grid_to_nodal(grid_batch: np.ndarray, node_to_grid: np.ndarray) -> np.ndarray:
    """(B, n, n) -> (B, n_nodes). Exact, no interpolation."""
    ii, jj = node_to_grid[:, 0], node_to_grid[:, 1]
    return grid_batch[:, ii, jj]


def build_a_field_batch(mu_batch: np.ndarray, grid_xy: np.ndarray,
                         n_cx: int, n_cy: int, a0: float) -> np.ndarray:
    """Exact piecewise-constant coefficient field a(x,y;mu) on the grid --
    same channel-assignment rule as ChannelizedOperatorAssembly, evaluated
    analytically at each grid point (which IS a mesh node), not interpolated.
    Returns (B, n, n).
    """
    x, y = grid_xy[..., 0], grid_xy[..., 1]
    cx = np.minimum((x * n_cx).astype(int), n_cx - 1)
    cy = np.minimum((y * n_cy).astype(int), n_cy - 1)
    channel_idx = cy * n_cx + cx  # (n, n)
    return a0 + mu_batch[:, channel_idx]  # (B, n, n)


# =====================================================================
# 2. REAL spectral FNO (actual FFT layers, correct spatial input)
# =====================================================================
class FourierBlock2DProper(nn.Module):
    """Standard FNO Fourier layer: spectral conv + spatial skip + GELU."""
    def __init__(self, channels, modes, dropout=0.0):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (channels * channels)
        self.weights = nn.Parameter(
            scale * torch.randn(channels, channels, modes, modes, dtype=torch.cfloat)
        )
        self.skip = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        b, c, h, w = x.shape
        x_skip = x
        x_ft = torch.fft.rfft2(x, norm='ortho')
        out_ft = torch.zeros(b, c, h, w // 2 + 1, dtype=torch.cfloat, device=x.device)
        m = min(self.modes, h, w // 2 + 1)
        out_ft[:, :, :m, :m] = torch.einsum('bixy,ioxy->boxy', x_ft[:, :, :m, :m], self.weights[:, :, :m, :m])
        x = torch.fft.irfft2(out_ft, s=(h, w), norm='ortho')
        x = x + self.skip(x_skip)
        x = F.gelu(x)
        x = self.dropout(x)
        return x


class RealFNO2D(nn.Module):
    """Real FNO: takes the EXACT (a_field, x, y) grid tensor as input
    (no mu-broadcast), has genuine spectral convolution layers."""
    def __init__(self, in_channels=3, hidden=64, modes=12, n_layers=4, dropout=0.0):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden, 1)
        self.layers = nn.ModuleList(
            [FourierBlock2DProper(hidden, modes, dropout) for _ in range(n_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, x):
        # x: (B, n, n, in_channels)
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x.squeeze(1)  # (B, n, n)


# =====================================================================
# 3. REAL wavelet-domain WNO (genuine DWT/IDWT, not a plain CNN)
# =====================================================================
class WaveletBlock2D(nn.Module):
    """DWT -> learnable per-subband mixing -> IDWT, with a spatial skip.
    This is the wavelet-domain analogue of FNO's spectral-conv layer:
    instead of truncating/mixing Fourier modes, it truncates/mixes wavelet
    subbands."""
    def __init__(self, channels, wave='db4', J=2, mode='periodization'):
        if not _HAVE_PYTORCH_WAVELETS:
            raise ImportError(
                "pytorch_wavelets is required for a real WNO baseline: "
                "pip install pytorch_wavelets"
            )
        super().__init__()
        self.J = J
        self.dwt = DWTForward(J=J, wave=wave, mode=mode)
        self.idwt = DWTInverse(wave=wave, mode=mode)
        self.w_low = nn.Conv2d(channels, channels, 1)
        self.w_high = nn.ModuleList(
            [nn.Conv2d(channels * 3, channels * 3, 1) for _ in range(J)]
        )
        self.skip = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        x_skip = x
        Yl, Yh = self.dwt(x)
        Yl = self.w_low(Yl)
        Yh_new = []
        for j in range(self.J):
            b, c, three, h, w = Yh[j].shape
            yh = Yh[j].reshape(b, c * three, h, w)
            yh = self.w_high[j](yh)
            Yh_new.append(yh.reshape(b, c, three, h, w))
        x_rec = self.idwt((Yl, Yh_new))
        # 'periodization' mode pads odd-length inputs to even length internally;
        # the reconstruction is (h+1, w+1) with samples [0:h, 0:w] matching the
        # original grid to ~1e-7 precision and the extra trailing row/col being
        # the padding artifact. CROP it -- do not resize, which would blur/
        # misalign every pixel (verified: resizing introduces error on the
        # order of the signal's own magnitude; cropping does not).
        h, w = x_skip.shape[-2:]
        x_rec = x_rec[..., :h, :w]
        return F.gelu(x_rec + self.skip(x_skip))


class RealWNO2D(nn.Module):
    """Real WNO: genuine multi-level 2D DWT/IDWT per layer via
    pytorch_wavelets, matching the input representation used by RealFNO2D
    (exact grid, no interpolation)."""
    def __init__(self, in_channels=3, hidden=32, n_layers=4, wave='db4', J=2):
        super().__init__()
        self.input_proj = nn.Conv2d(in_channels, hidden, 1)
        self.blocks = nn.ModuleList(
            [WaveletBlock2D(hidden, wave=wave, J=J) for _ in range(n_layers)]
        )
        self.output_proj = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1), nn.GELU(), nn.Conv2d(hidden, 1, 1)
        )

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.input_proj(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.output_proj(x)
        return x.squeeze(1)


# =====================================================================
# 4. MLP/DeepONet baseline -- already predicts on native nodal ordering,
#    kept as-is except scoring is unified below.
# =====================================================================
class MLPBaseline(nn.Module):
    def __init__(self, mu_dim, n_outputs, hidden_dims=(256, 256, 128)):
        super().__init__()
        layers, d = [], mu_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.GELU()]
            d = h
        layers += [nn.Linear(d, n_outputs)]
        self.net = nn.Sequential(*layers)

    def forward(self, mu):
        return self.net(mu)


# =====================================================================
# 5. Spatially-encoded OFW-NO
# =====================================================================
# Diagnosis behind this class: the original OFWNeuralOperator maps a 16-dim mu
# through a dense MLP to every wavelet coefficient.  Capacity-matched at level
# 4 it lands on 6.8e-1 relative L2 -- indistinguishable from plain linear
# least squares on log a (6.7e-1) -- while FNO/WNO reach 2.1e-1 from the same
# 150 training samples.  The gap is NOT the output basis (reconstruction from
# exact coefficients is 7e-17); it is the input encoding.  FNO/WNO share
# weights across space, so they generalise from few samples; a dense mu-to-
# coefficient map has to learn every coefficient independently.
#
# This class keeps the operator-adapted wavelet OUTPUT basis -- the actual
# contribution -- and gives the INPUT the same spatial weight sharing:
#   (a_field, x, y) grid -> spectral stem -> gather features at each level's
#   node positions -> per-level shared head -> that level's coefficients.
# Each level's head is applied to every node in that level, so its parameter
# count is independent of the number of coefficients.
class SpatialOFWNO(nn.Module):
    def __init__(self, level_nodes_grid, in_channels=3, hidden=32, modes=12,
                 n_layers=4, head_hidden=64):
        """level_nodes_grid: {level: LongTensor (n_l, 2) of (i, j) grid indices,
        in the same order as that level's coefficient vector}."""
        super().__init__()
        self.levels = sorted(level_nodes_grid)
        for l in self.levels:
            self.register_buffer(f"ij_{l}", level_nodes_grid[l])
        self.input_proj = nn.Conv2d(in_channels, hidden, 1)
        self.stem = nn.ModuleList(
            [FourierBlock2DProper(hidden, modes) for _ in range(n_layers)])
        self.heads = nn.ModuleDict({
            str(l): nn.Sequential(nn.Linear(hidden, head_hidden), nn.GELU(),
                                  nn.Linear(head_hidden, 1))
            for l in self.levels})

    def forward(self, x):
        x = self.input_proj(x.permute(0, 3, 1, 2))
        for blk in self.stem:
            x = blk(x)                                  # (B, H, n, n)
        out = {}
        for l in self.levels:
            ij = getattr(self, f"ij_{l}")
            if ij.numel() == 0:
                out[l] = x.new_zeros(x.shape[0], 0); continue
            f = x[:, :, ij[:, 0], ij[:, 1]].permute(0, 2, 1)   # (B, n_l, H)
            out[l] = self.heads[str(l)](f).squeeze(-1)         # (B, n_l)
        return out


def spatial_ofwno_params(h, modes, layers, head_hidden, n_levels, cin=3):
    return (cin*h + h
            + layers*(2*h*h*modes*modes + h*h + h)
            + n_levels*(h*head_hidden + head_hidden + head_hidden + 1))
