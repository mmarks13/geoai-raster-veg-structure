"""
Soft pillar conv decoder for Path B (`cross_attn_soft_pillar`).

This module contains a verbatim copy of `SoftPillarConvDecoder` and its helper
`_GridRefineBlock` from `src/models/raster_head.py`. The legacy file is left
in place for checkpoint migration; this copy is the version wired into the new
`raster_heads` package.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_sum, scatter_max


class _GridRefineBlock(nn.Module):
    """
    Depthwise + pointwise ConvNeXt-style block for small grids.
    Uses GroupNorm(1, C) which is always valid and batch-size agnostic.
    """
    def __init__(self, channels: int, expansion: int = 2, dropout: float = 0.0):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels)
        self.gn = nn.GroupNorm(1, channels)
        hidden = channels * expansion
        self.pw1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.pw2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = x
        x = self.dw(x)
        x = self.gn(x)
        x = F.gelu(x)
        x = self.pw1(x)
        x = F.gelu(x)
        x = self.drop(x)
        x = self.pw2(x)
        return r + x


class SoftPillarConvDecoder(nn.Module):
    """
    Point-to-raster decoder using:
      - bilinear soft splatting to 5x5 pillars (reduces aliasing at cell boundaries)
      - dual-stream aggregation (weighted mean semantics + hard max semantics)
      - explicit density/vertical structure stats
      - lightweight convolutional refinement on the grid

    This decoder performs its own point-to-grid aggregation internally via bilinear
    soft-splatting, bypassing the attention-based PointToGridAggregator.

    Inputs:
      point_features:  [N, F]
      point_positions: [N, 3] z-score normalized (x=y=0 is tile center)
      batch_indices:   [N]
      norm_params:     list of dicts with 'coord_mean' and 'coord_std' (tensors)

    Output:
      - if output_variance=False: [B, n_bands, 5, 5]
      - if output_variance=True:  (mean, log_var), each [B, n_bands, 5, 5]
    """
    def __init__(
        self,
        feature_dim: int = 256,
        grid_size: int = 5,
        tile_extent: float = 10.0,
        n_bands: int = 8,
        decoder_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.10,
        z_bin_edges_m: tuple = (2.0, 5.0),
        output_variance: bool = False,
        z_embed_dim: int = 64,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.tile_extent = tile_extent
        self.cell_size = tile_extent / grid_size
        self.n_bands = n_bands
        self.output_variance = output_variance
        self.z_bin_edges_m = z_bin_edges_m

        # Fixed grid coordinate channels (x_norm, y_norm, r_norm)
        centers_1d = torch.linspace(
            -tile_extent / 2 + self.cell_size / 2,
            tile_extent / 2 - self.cell_size / 2,
            grid_size
        )
        yy, xx = torch.meshgrid(centers_1d, centers_1d, indexing="ij")  # [H, W]
        x_norm = (xx / (tile_extent / 2)).clamp(-1, 1)
        y_norm = (yy / (tile_extent / 2)).clamp(-1, 1)
        r_norm = torch.sqrt(x_norm ** 2 + y_norm ** 2).clamp(0, 1)
        grid_xyz = torch.stack([x_norm, y_norm, r_norm], dim=0)  # [3, H, W]
        self.register_buffer("grid_pos_channels", grid_xyz)

        # Height encoder (uses z in meters after denormalization)
        self.z_encoder = nn.Sequential(
            nn.Linear(1, z_embed_dim),
            nn.GELU(),
            nn.Linear(z_embed_dim, feature_dim),
        )

        # Combine original features and z-encoded features
        self.feat_fusion = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # Density / structure stats:
        # - hard_count (log1p), soft_count (log1p)
        # - z_mean, z_std, z_max
        # - occupancy
        # - vertical bin soft counts (3 bins): low, mid, high (log1p)
        # - grid position channels: 3 (x_norm, y_norm, r_norm)
        self.num_density = 1 + 1 + 1 + 1 + 1 + 1 + 3  # = 9
        in_channels = (feature_dim * 2) + self.num_density + 3

        self.grid_stem = nn.Conv2d(in_channels, decoder_dim, kernel_size=1)

        self.blocks = nn.ModuleList([
            _GridRefineBlock(decoder_dim, expansion=2, dropout=dropout)
            for _ in range(num_blocks)
        ])

        out_ch = n_bands * 2 if output_variance else n_bands
        self.head = nn.Conv2d(decoder_dim, out_ch, kernel_size=1)

    def _denorm_positions(
        self,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: list
    ) -> torch.Tensor:
        """Denormalize positions from z-score to bbox-normalized (meter) space."""
        device = point_positions.device
        dtype = point_positions.dtype

        coord_mean_batch = torch.stack([p["coord_mean"] for p in norm_params]).to(device=device, dtype=dtype)
        coord_std_batch = torch.stack([p["coord_std"] for p in norm_params]).to(device=device, dtype=dtype)

        coord_means = coord_mean_batch[batch_indices]  # [N, 3]
        coord_stds = coord_std_batch[batch_indices]   # [N, 3]
        return point_positions * coord_stds + coord_means  # [N, 3] in meters

    def forward(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: list,
    ):
        """
        Forward pass: aggregate points to grid via bilinear soft-splatting
        and refine with ConvNeXt blocks.

        Args:
            point_features: [N_total, feature_dim] - concatenated point features
            point_positions: [N_total, 3] - Z-SCORE NORMALIZED point positions
            batch_indices: [N_total] - which batch each point belongs to
            norm_params: List of dicts with 'coord_mean', 'coord_std'

        Returns:
            If output_variance=False:
                raster: [batch_size, n_bands, 5, 5]
            If output_variance=True:
                Tuple of (mean, log_var): each [batch_size, n_bands, 5, 5]
        """
        device = point_features.device
        N, F = point_features.shape
        B = len(norm_params)
        H = W = self.grid_size
        total_cells = B * H * W
        eps = 1e-6

        # ---- 1) Denormalize positions to meters for correct grid assignment ----
        pos_m = self._denorm_positions(point_positions, batch_indices, norm_params)
        x_m, y_m, z_m = pos_m[:, 0], pos_m[:, 1], pos_m[:, 2]

        # ---- 2) Inject explicit height into point features (in meters) ----
        z_feat = self.z_encoder(z_m.view(-1, 1))
        point_features = self.feat_fusion(point_features + z_feat)

        # ---- 3) Continuous grid coordinates in [0, grid_size) ----
        # Tile assumed centered at 0 with extent [-E/2, E/2]
        u = (x_m + self.tile_extent / 2) / self.cell_size  # ~[0, 5]
        v = (y_m + self.tile_extent / 2) / self.cell_size  # ~[0, 5]

        # Clamp to stay strictly inside [0, grid_size) to avoid boundary idx==grid_size
        u = u.clamp(0.0, self.grid_size - 1e-6)
        v = v.clamp(0.0, self.grid_size - 1e-6)

        ix0 = torch.floor(u).long()
        iy0 = torch.floor(v).long()
        ix1 = (ix0 + 1).clamp(max=self.grid_size - 1)
        iy1 = (iy0 + 1).clamp(max=self.grid_size - 1)

        fx = (u - ix0.float()).clamp(0.0, 1.0)
        fy = (v - iy0.float()).clamp(0.0, 1.0)
        wx0, wx1 = (1.0 - fx), fx
        wy0, wy1 = (1.0 - fy), fy

        # 4-neighbor bilinear weights
        w00 = wx0 * wy0
        w10 = wx1 * wy0
        w01 = wx0 * wy1
        w11 = wx1 * wy1

        # Unique cell indices across batch
        batch_offset = batch_indices * (H * W)
        c00 = batch_offset + (iy0 * W + ix0)
        c10 = batch_offset + (iy0 * W + ix1)
        c01 = batch_offset + (iy1 * W + ix0)
        c11 = batch_offset + (iy1 * W + ix1)

        # ---- 4) Soft splat aggregations (weighted mean + soft count + z moments) ----
        def _wsum(val: torch.Tensor, idx: torch.Tensor, w: torch.Tensor, dim_size: int):
            return scatter_sum(val * w.unsqueeze(-1), idx, dim=0, dim_size=dim_size)

        # Weighted feature sum
        feat_sum = (
            _wsum(point_features, c00, w00, total_cells) +
            _wsum(point_features, c10, w10, total_cells) +
            _wsum(point_features, c01, w01, total_cells) +
            _wsum(point_features, c11, w11, total_cells)
        )  # [C, F]

        # Weighted "soft count"
        soft_count = (
            scatter_sum(w00, c00, dim=0, dim_size=total_cells) +
            scatter_sum(w10, c10, dim=0, dim_size=total_cells) +
            scatter_sum(w01, c01, dim=0, dim_size=total_cells) +
            scatter_sum(w11, c11, dim=0, dim_size=total_cells)
        )  # [C]

        feat_mean = feat_sum / (soft_count.unsqueeze(-1) + eps)  # [C, F]

        # z moments (soft)
        z_sum = (
            scatter_sum(z_m * w00, c00, dim=0, dim_size=total_cells) +
            scatter_sum(z_m * w10, c10, dim=0, dim_size=total_cells) +
            scatter_sum(z_m * w01, c01, dim=0, dim_size=total_cells) +
            scatter_sum(z_m * w11, c11, dim=0, dim_size=total_cells)
        )
        z2_sum = (
            scatter_sum((z_m ** 2) * w00, c00, dim=0, dim_size=total_cells) +
            scatter_sum((z_m ** 2) * w10, c10, dim=0, dim_size=total_cells) +
            scatter_sum((z_m ** 2) * w01, c01, dim=0, dim_size=total_cells) +
            scatter_sum((z_m ** 2) * w11, c11, dim=0, dim_size=total_cells)
        )

        z_mean = z_sum / (soft_count + eps)
        z_var = (z2_sum / (soft_count + eps)) - z_mean ** 2
        z_std = torch.sqrt(torch.clamp(z_var, min=0.0) + eps)

        # ---- 5) Hard max semantics + hard occupancy/count (stability for max ops) ----
        # Primary hard cell assignment
        c_primary = c00  # (iy0, ix0)

        # Hard count
        hard_ones = torch.ones((N,), device=device, dtype=torch.float32)
        hard_count = scatter_sum(hard_ones, c_primary, dim=0, dim_size=total_cells)  # [C]
        occupancy = (hard_count > 0).float()  # [C]

        # Hard max feature per cell (fix empty cells)
        feat_max, _ = scatter_max(point_features, c_primary, dim=0, dim_size=total_cells)  # [C, F]
        feat_max = torch.nan_to_num(feat_max, nan=0.0, posinf=0.0, neginf=0.0)
        feat_max = feat_max * occupancy.unsqueeze(-1)  # zero empty cells

        # Hard max height per cell
        z_max, _ = scatter_max(z_m, c_primary, dim=0, dim_size=total_cells)  # [C]
        z_max = torch.nan_to_num(z_max, nan=0.0, posinf=0.0, neginf=0.0) * occupancy

        # ---- 6) Vertical bin soft counts (low/mid/high) ----
        h1, h2 = self.z_bin_edges_m
        low = (z_m <= h1).float()
        mid = ((z_m > h1) & (z_m <= h2)).float()
        high = (z_m > h2).float()

        def _bin_soft(bin_mask: torch.Tensor):
            return (
                scatter_sum(bin_mask * w00, c00, dim=0, dim_size=total_cells) +
                scatter_sum(bin_mask * w10, c10, dim=0, dim_size=total_cells) +
                scatter_sum(bin_mask * w01, c01, dim=0, dim_size=total_cells) +
                scatter_sum(bin_mask * w11, c11, dim=0, dim_size=total_cells)
            )

        low_c = _bin_soft(low)
        mid_c = _bin_soft(mid)
        high_c = _bin_soft(high)

        # ---- 7) Assemble grid tensor ----
        log_hard = torch.log1p(hard_count.clamp(min=0.0))
        log_soft = torch.log1p(soft_count.clamp(min=0.0))

        # Density/structure channels: [C, 9]
        density = torch.stack(
            [
                log_hard,
                log_soft,
                z_mean,
                z_std,
                z_max,
                occupancy,
                torch.log1p(low_c.clamp(min=0.0)),
                torch.log1p(mid_c.clamp(min=0.0)),
                torch.log1p(high_c.clamp(min=0.0)),
            ],
            dim=1
        )

        # Combine semantics + density
        grid_cells = torch.cat([feat_mean, feat_max, density], dim=1)  # [C, 2F + num_density]

        # Reshape to [B, C, H, W]
        grid = grid_cells.view(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # [B, Ch, H, W]

        # Append fixed positional channels
        pos_ch = self.grid_pos_channels.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 3, H, W]
        grid = torch.cat([grid, pos_ch], dim=1)  # [B, Ch+3, H, W]

        # ---- 8) Convolutional refinement ----
        x = self.grid_stem(grid)
        for blk in self.blocks:
            x = blk(x)

        out = self.head(x)  # [B, n_bands or 2*n_bands, H, W]

        if self.output_variance:
            mean = out[:, :self.n_bands]
            log_var = out[:, self.n_bands:]
            return mean, log_var

        return out
