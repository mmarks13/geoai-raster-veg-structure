"""
DEPRECATED — see src/models/raster_heads/ for active code.
Retained for checkpoint migration and legacy class construction.

Raster prediction head for fuel metrics prediction.

This module implements query-based aggregation to convert point cloud features
into a regular grid, followed by a decoder to predict fuel metrics rasters.

Architecture:
1. LearnableGridQueries: Learnable positional embeddings for each grid cell
2. DistanceMaskedAttention: Cross-attention with distance-based masking (Flash Attention)
3. PointToGridAggregator: Aggregates point features to regular grid
4. RasterDecoder: MLP to predict fuel metrics from grid features
5. RasterPredictionHead: Complete pipeline from points to raster predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_sum, scatter_max
from torchvision.ops import StochasticDepth
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# Import LocalGlobalPointAttentionBlock for pre-aggregation refinement
from .multimodal_model import LocalGlobalPointAttentionBlock


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

    Config parameter mapping (when raster_use_wide_decoder=True):
      - raster_hidden_dim → decoder_dim (ConvNeXt refinement dimension)
      - raster_decoder_layers → num_blocks (number of refinement blocks)
      - raster_decoder_dropout → dropout (refinement block dropout)
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


class LearnableGridQueries(nn.Module):
    """
    Learnable query embeddings for regular grid cells with positional encoding.

    Creates fixed 5×5 grid over 10m tile with learnable feature embeddings
    combined with sinusoidal 2D positional encodings. This helps queries
    represent their spatial position (e.g., "I'm a corner cell" vs "I'm center").

    Grid centers are at [-4, -2, 0, 2, 4] meters in bbox-normalized space.

    Args:
        feature_dim: Feature dimension for query embeddings
        grid_size: Number of cells per side (default 5 for 5×5 grid)
        tile_extent: Tile extent in meters (default 10.0)
    """

    def __init__(self, feature_dim: int = 256, grid_size: int = 5, tile_extent: float = 10.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.tile_extent = tile_extent
        self.num_queries = grid_size * grid_size

        # Calculate grid centers (in bbox-normalized space: meters)
        cell_size = tile_extent / grid_size  # 2.0 meters
        centers_1d = np.linspace(-tile_extent/2 + cell_size/2, tile_extent/2 - cell_size/2, grid_size)
        # centers_1d = [-4, -2, 0, 2, 4] meters

        # Create 2D grid of centers (X, Y only - no Z component for 2D grid)
        grid_y, grid_x = np.meshgrid(centers_1d, centers_1d, indexing='ij')
        grid_centers = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)  # [25, 2]

        # Register as buffer (not trainable)
        self.register_buffer('grid_centers', torch.from_numpy(grid_centers).float())  # [25, 2]

        # Learnable query embeddings
        self.query_embed = nn.Parameter(torch.randn(self.num_queries, feature_dim))

        # Initialize with appropriate std (0.02 is standard for transformers)
        nn.init.normal_(self.query_embed, std=0.02)

        # Create 2D sinusoidal positional encoding
        pos_encoding = self._create_2d_sinusoidal_encoding(
            grid_centers, feature_dim, tile_extent
        )
        self.register_buffer('pos_encoding', torch.from_numpy(pos_encoding).float())  # [25, feature_dim]

    def _create_2d_sinusoidal_encoding(
        self,
        positions: np.ndarray,
        feature_dim: int,
        tile_extent: float
    ) -> np.ndarray:
        """
        Create 2D sinusoidal positional encoding for grid positions.

        Uses separate frequency bands for X and Y dimensions, interleaved
        to create the full encoding. Standard approach from ViT/DETR.

        Args:
            positions: [num_queries, 2] grid centers (X, Y) in meters
            feature_dim: Output dimension
            tile_extent: Tile extent for normalization

        Returns:
            pos_encoding: [num_queries, feature_dim]
        """
        num_queries = positions.shape[0]

        # Normalize positions to [0, 1] range
        positions_norm = (positions + tile_extent / 2) / tile_extent  # [0, 1]

        # Half the dimensions for X, half for Y
        half_dim = feature_dim // 2

        # Create frequency bands (exponential spacing)
        # div_term: [half_dim // 2]
        div_term = np.exp(np.arange(0, half_dim, 2) * (-np.log(10000.0) / half_dim))

        # Compute X and Y encodings
        x_pos = positions_norm[:, 0:1]  # [num_queries, 1]
        y_pos = positions_norm[:, 1:2]  # [num_queries, 1]

        # PE(x, 2i) = sin(x / 10000^(2i/d))
        # PE(x, 2i+1) = cos(x / 10000^(2i/d))
        pe_x = np.zeros((num_queries, half_dim))
        pe_y = np.zeros((num_queries, half_dim))

        pe_x[:, 0::2] = np.sin(x_pos * div_term)
        pe_x[:, 1::2] = np.cos(x_pos * div_term)
        pe_y[:, 0::2] = np.sin(y_pos * div_term)
        pe_y[:, 1::2] = np.cos(y_pos * div_term)

        # Concatenate X and Y encodings
        pos_encoding = np.concatenate([pe_x, pe_y], axis=1)  # [num_queries, feature_dim]

        return pos_encoding

    def forward(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate query embeddings and grid centers for a batch.

        Args:
            batch_size: Number of tiles in batch
            device: Device to create tensors on

        Returns:
            Tuple of:
            - query_embeddings: [batch_size, num_queries, feature_dim]
            - grid_centers: [num_queries, 2] (X, Y in meters, bbox-normalized space)
        """
        # Combine learnable embeddings with positional encoding
        queries = self.query_embed + self.pos_encoding  # [num_queries, feature_dim]

        # Expand query embeddings for batch
        query_embeddings = queries.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embeddings, self.grid_centers


class DistanceMaskedAttention(nn.Module):
    """
    Multi-head cross-attention with soft Gaussian distance weighting.

    Queries (grid cells) attend to keys/values (point features) with attention
    weights modulated by a Gaussian decay based on distance. This replaces
    the previous hard radius cutoff to handle sparse tiles gracefully.

    Supports per-head sigma values for multi-scale attention, allowing different
    heads to focus on different spatial scales (e.g., local vs global context).

    Uses manual scaled dot-product attention implementation (bypassing
    PyTorch's F.scaled_dot_product_attention for full compatibility).

    Args:
        feature_dim: Feature dimension
        num_heads: Number of attention heads (must divide feature_dim)
        distance_sigma: Gaussian sigma for distance weighting (meters).
                       Can be a single float (same sigma for all heads) or
                       a list of floats (one sigma per head for multi-scale).
                       Points at distance=sigma have weight ~0.61.
                       Points at distance=2*sigma have weight ~0.14.
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        distance_sigma: float | List[float] = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.dropout = dropout

        # Handle per-head sigma values (multi-scale attention)
        if isinstance(distance_sigma, (list, tuple)):
            assert len(distance_sigma) == num_heads, \
                f"distance_sigma list length ({len(distance_sigma)}) must match num_heads ({num_heads})"
            self.per_head_sigma = True
            # Register as buffer for proper device handling
            self.register_buffer('distance_sigmas', torch.tensor(distance_sigma, dtype=torch.float32))
        else:
            self.per_head_sigma = False
            self.distance_sigma = distance_sigma

        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        query_positions: torch.Tensor,
        key_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        return_coverage: bool = False
    ):
        """
        Apply distance-masked cross-attention.

        Args:
            queries: [batch_size, num_queries, feature_dim] - grid query embeddings
            keys: [N_total, feature_dim] - point features (concatenated across batch)
            values: [N_total, feature_dim] - point features (same as keys)
            query_positions: [num_queries, 2] - grid centers (X, Y in METERS)
            key_positions: [N_total, 2] - point positions (X, Y in METERS, bbox-normalized)
            batch_indices: [N_total] - which batch each point belongs to (required)
            return_coverage: If True, return (output, coverage_stats) tuple

        Returns:
            output: [batch_size, num_queries, feature_dim] - aggregated grid features
            coverage_stats (optional): Dict with attention coverage statistics
        """
        batch_size, num_queries, _ = queries.shape
        
        # 1. Create dense batches using torch_geometric
        # keys: [N_total, F] -> [B, max_points, F], mask: [B, max_points]
        K_dense, mask = to_dense_batch(keys, batch_indices, batch_size=batch_size)
        V_dense, _ = to_dense_batch(values, batch_indices, batch_size=batch_size)
        key_pos_dense, _ = to_dense_batch(key_positions, batch_indices, batch_size=batch_size)
        
        max_points = K_dense.size(1)

        # 2. Project to Q, K, V
        Q = self.q_proj(queries)  # [B, num_queries, feature_dim]
        K = self.k_proj(K_dense)  # [B, max_points, feature_dim]
        V = self.v_proj(V_dense)  # [B, max_points, feature_dim]

        # 3. Reshape for multi-head attention
        # [B, N, F] -> [B, N, num_heads, head_dim] -> [B, num_heads, N, head_dim]
        Q = Q.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, max_points, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, max_points, self.num_heads, self.head_dim).transpose(1, 2)

        # 4. Compute soft Gaussian distance weights
        # query_positions: [num_queries, 2] -> [B, num_queries, 2]
        # key_pos_dense: [B, max_points, 2]
        distances = torch.cdist(
            query_positions.unsqueeze(0).expand(batch_size, -1, -1),
            key_pos_dense,
            p=2
        )  # [B, num_queries, max_points]

        # Soft Gaussian distance weighting (replaces hard radius cutoff)
        # weight = exp(-d² / 2σ²)
        # This ensures weights are always > 0, preventing NaN from softmax
        # when grid cells have no nearby points (sparse tile handling)

        if self.per_head_sigma:
            # Multi-scale attention: different sigma per head
            # distances: [B, num_queries, max_points]
            # sigmas: [num_heads] -> need [1, num_heads, 1, 1] for broadcasting
            sigmas = self.distance_sigmas.view(1, self.num_heads, 1, 1)  # [1, H, 1, 1]
            # Expand distances for per-head computation: [B, 1, num_queries, max_points]
            distances_expanded = distances.unsqueeze(1)  # [B, 1, Q, P]
            # Compute per-head distance weights: [B, H, Q, P]
            distance_weights_per_head = torch.exp(-distances_expanded**2 / (2 * sigmas**2))
            # Convert to log-space
            log_distance_weights = torch.log(distance_weights_per_head + 1e-10)
            # [B, num_heads, num_queries, max_points]
            # Store for coverage stats (use mean across heads)
            distance_weights = distance_weights_per_head.mean(dim=1)  # [B, Q, P]
        else:
            # Single-scale attention: same sigma for all heads
            distance_weights = torch.exp(-distances**2 / (2 * self.distance_sigma**2))
            # [B, num_queries, max_points], values in (0, 1]
            # Convert to log-space for addition to attention scores
            log_distance_weights = torch.log(distance_weights + 1e-10)
            # [B, num_queries, max_points]
        
        # Padding mask: True if padded (should be masked with -inf)
        # mask from to_dense_batch is True for real points, False for padding
        pad_mask = ~mask  # [B, max_points], True for padded positions

        # Apply padding mask: set log weights to -inf for padded positions
        if self.per_head_sigma:
            # log_distance_weights: [B, num_heads, num_queries, max_points]
            # Expand pad_mask: [B, max_points] -> [B, 1, 1, max_points]
            log_distance_weights = log_distance_weights.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, max_points]
                float('-inf')
            )
        else:
            # log_distance_weights: [B, num_queries, max_points]
            # Expand for broadcasting: [B, max_points] -> [B, 1, max_points]
            log_distance_weights = log_distance_weights.masked_fill(
                pad_mask.unsqueeze(1),  # [B, 1, max_points]
                float('-inf')
            )

        # 5. Manual Scaled Dot-Product Attention
        # NOTE: Bypassing F.scaled_dot_product_attention entirely due to persistent
        # cuDNN compatibility issues with custom attention patterns.
        # Manual implementation is more robust and gives full control.

        # Q: [B, num_heads, num_queries, head_dim]
        # K: [B, num_heads, max_points, head_dim]
        # V: [B, num_heads, max_points, head_dim]
        # log_distance_weights: [B, num_queries, max_points]

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, num_heads, num_queries, max_points]

        # Add log distance weights
        if self.per_head_sigma:
            # log_distance_weights already [B, num_heads, num_queries, max_points]
            scores = scores + log_distance_weights
        else:
            # Broadcast over heads dimension
            # [B, num_queries, max_points] -> [B, 1, num_queries, max_points]
            scores = scores + log_distance_weights.unsqueeze(1)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Apply dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        # output: [B, num_heads, num_queries, head_dim]

        # 6. Reshape back
        # [B, num_heads, num_queries, head_dim] -> [B, num_queries, feature_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.feature_dim)

        # Final projection
        output = self.out_proj(output)

        # Compute coverage statistics if requested
        if return_coverage:
            coverage_stats = self._compute_coverage_stats(
                distance_weights, mask, batch_size, num_queries
            )
            return output, coverage_stats
        else:
            return output

    def _compute_coverage_stats(
        self,
        distance_weights: torch.Tensor,
        mask: torch.Tensor,
        batch_size: int,
        num_queries: int
    ) -> Dict[str, float]:
        """
        Compute attention coverage statistics.

        Args:
            distance_weights: [B, num_queries, max_points] - Gaussian weights
            mask: [B, max_points] - True for real points, False for padding
            batch_size: Batch size
            num_queries: Number of grid queries (25)

        Returns:
            Dict with coverage statistics
        """
        # Threshold for "effective" points (weight > 0.05 means d < 2.45*sigma)
        weight_threshold = 0.05

        # Count effective points per query (using first batch for efficiency)
        weights_b0 = distance_weights[0]  # [num_queries, max_points]
        mask_b0 = mask[0]  # [max_points]

        # Mask out padding
        weights_b0 = weights_b0 * mask_b0.unsqueeze(0)  # [num_queries, max_points]

        # Count effective points per query
        effective_counts = (weights_b0 > weight_threshold).sum(dim=1).float()  # [num_queries]

        mean_coverage = effective_counts.mean().item()
        min_coverage = effective_counts.min().item()
        max_coverage = effective_counts.max().item()

        # Spatial pattern: edge vs center cells
        # Grid is 5×5, so indices 0-4,20-24 are edges, 5-19 contains some interior
        # More precise: corners are 0,4,20,24; edges are perimeter; center is 12
        grid_size = 5
        center_idx = 12  # (2, 2) in 5×5 grid
        corner_indices = [0, 4, 20, 24]
        edge_indices = [1, 2, 3, 5, 9, 10, 14, 15, 19, 21, 22, 23]  # Edges excluding corners
        interior_indices = [6, 7, 8, 11, 12, 13, 16, 17, 18]  # Interior 3×3

        # Mean coverage by position type
        corner_coverage = effective_counts[corner_indices].mean().item()
        edge_coverage = effective_counts[edge_indices].mean().item()
        interior_coverage = effective_counts[interior_indices].mean().item()

        return {
            'coverage_mean': mean_coverage,
            'coverage_min': min_coverage,
            'coverage_max': max_coverage,
            'coverage_corner': corner_coverage,
            'coverage_edge': edge_coverage,
            'coverage_interior': interior_coverage,
        }


class PointToGridAggregator(nn.Module):
    """
    Aggregates point cloud features to a regular grid using learnable queries.

    Combines LearnableGridQueries and DistanceMaskedAttention to transform
    irregular point cloud features into a regular 5×5 grid.

    Args:
        feature_dim: Feature dimension (must match point features)
        num_heads: Number of attention heads
        distance_sigma: Gaussian sigma for distance weighting in meters (default 2.0).
                       Can be a list of floats for per-head multi-scale attention.
        grid_size: Grid size per side (default 5)
        tile_extent: Tile extent in meters (default 10.0)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        distance_sigma: float | List[float] = 2.0,
        grid_size: int = 5,
        tile_extent: float = 10.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size

        # Grid queries
        self.grid_queries = LearnableGridQueries(
            feature_dim=feature_dim,
            grid_size=grid_size,
            tile_extent=tile_extent
        )

        # Distance-weighted attention (soft Gaussian weighting)
        self.attention = DistanceMaskedAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            distance_sigma=distance_sigma,
            dropout=dropout,
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict],
        return_coverage: bool = False
    ):
        """
        Aggregate point features to regular grid.

        Args:
            point_features: [N_total, feature_dim] - concatenated point features
            point_positions: [N_total, 3] - Z-SCORE NORMALIZED point positions (X, Y, Z)
            batch_indices: [N_total] - which batch each point belongs to
            norm_params: List of dicts (length batch_size) with 'coord_mean', 'coord_std'
            return_coverage: If True, return (grid_features, coverage_stats) tuple

        Returns:
            grid_features: [batch_size, grid_size, grid_size, feature_dim]
            coverage_stats (optional): Dict with attention coverage statistics
        """
        batch_size = len(norm_params)
        device = point_features.device

        # Generate grid queries and centers
        query_embeddings, grid_centers = self.grid_queries(batch_size, device)
        # query_embeddings: [batch_size, num_queries=25, feature_dim]
        # grid_centers: [25, 2] (X, Y in meters)

        # Denormalize point positions from z-score to bbox-normalized (meter space)
        # point_positions: [N_total, 3] z-score normalized
        # We need: point_pos_phys = point_positions * coord_std + coord_mean

        # Stack tensors directly (norm_params now contains tensors, not lists)
        coord_mean_batch = torch.stack([p['coord_mean'] for p in norm_params]).to(device=device, dtype=point_positions.dtype)
        coord_std_batch = torch.stack([p['coord_std'] for p in norm_params]).to(device=device, dtype=point_positions.dtype)

        # Index into batch tensors using batch_indices (vectorized)
        coord_means = coord_mean_batch[batch_indices]  # [N_total, 3]
        coord_stds = coord_std_batch[batch_indices]  # [N_total, 3]

        # Denormalize: z-score → bbox-normalized (meters)
        point_pos_phys = point_positions * coord_stds + coord_means  # [N_total, 3]

        # Extract X, Y only (grid is 2D)
        point_pos_xy = point_pos_phys[:, :2]  # [N_total, 2]

        # Apply distance-masked attention
        attn_result = self.attention(
            queries=query_embeddings,
            keys=point_features,
            values=point_features,
            query_positions=grid_centers,
            key_positions=point_pos_xy,
            batch_indices=batch_indices,
            return_coverage=return_coverage
        )

        if return_coverage:
            grid_features, coverage_stats = attn_result
        else:
            grid_features = attn_result
        # grid_features: [batch_size, num_queries=25, feature_dim]

        # Reshape to 2D grid
        grid_features = grid_features.view(batch_size, self.grid_size, self.grid_size, self.feature_dim)
        # [batch_size, 5, 5, feature_dim]

        if return_coverage:
            return grid_features, coverage_stats
        else:
            return grid_features


class RasterDecoder(nn.Module):
    """
    MLP decoder for fuel metrics prediction with configurable depth and width.

    Uses Linear layers (equivalent to 1x1 Conv) to preserve sharp boundaries
    while decoding grid features to fuel metrics. No spatial mixing - each
    cell is processed independently.

    Architecture (configurable):
        grid_features [B, 5, 5, feature_dim]
        → Linear layers with progressive dimension halving
        → Permute to [B, n_bands, 5, 5]

    Args:
        feature_dim: Input feature dimension (default 256)
        n_bands: Number of output fuel metrics bands (default 3)
        hidden_dim: Hidden layer dimension (default 128)
        num_layers: Number of MLP layers (default 3, tunable: 3/4/5)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_bands: int = 3,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.1,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.num_layers = num_layers

        # Build MLP layers dynamically
        layers = []
        in_dim = feature_dim

        for i in range(num_layers - 1):
            # Halve dimension at each layer
            out_dim = hidden_dim // (2 ** i)
            layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),  # Add normalization after linear
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_dim = out_dim

        # Final layer to n_bands. Spectral norm bounds Lipschitz of the final readout for OOD.
        sn = nn.utils.parametrizations.spectral_norm if use_spectral_norm else (lambda m: m)
        layers.append(sn(nn.Linear(in_dim, n_bands)))

        self.mlp = nn.Sequential(*layers)

    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        """
        Decode grid features to fuel metrics raster.

        Args:
            grid_features: [batch_size, 5, 5, feature_dim]

        Returns:
            raster: [batch_size, n_bands, 5, 5] - predicted fuel metrics
        """
        # Input is [B, 5, 5, feature_dim], MLP applies to last dim
        x = self.mlp(grid_features)  # [B, 5, 5, n_bands]

        # Permute to raster format: [B, 5, 5, n_bands] → [B, n_bands, 5, 5]
        x = x.permute(0, 3, 1, 2).contiguous()

        return x


class WideRasterDecoder(nn.Module):
    """
    Wide MLP decoder with Pre-LayerNorm residual blocks and stochastic depth.

    Architecture:
        grid_features [B, 5, 5, feature_dim]
        → Pre-LN Residual Block 1: x = x + DropPath(MLP(LayerNorm(x)))
        → Pre-LN Residual Block 2: x = x + DropPath(MLP(LayerNorm(x)))
        → Final projection: Linear(feature_dim → n_bands or 2*n_bands if output_variance)
        → Permute to [B, n_bands, 5, 5] or split into (mean, log_var) tuple

    Uses GELU activation (smoother than ReLU, better for regression).
    Pre-LN structure provides better gradient flow than Post-LN.
    Stochastic depth rates increase linearly from 0 (first block) to drop_path (last block).

    Args:
        feature_dim: Input/hidden feature dimension (default 256)
        n_bands: Number of output fuel metrics bands (default 2)
        num_layers: Number of residual blocks (default 2)
        dropout: Dropout probability (default 0.35)
        drop_path: Maximum stochastic depth probability (default 0.0, disabled)
        use_spectral_norm: Apply spectral normalization to linear layers
        output_variance: If True, output (mean, log_variance) tuple for heteroscedastic loss
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_bands: int = 2,
        num_layers: int = 2,
        dropout: float = 0.35,
        drop_path: float = 0.0,
        use_spectral_norm: bool = False,
        output_variance: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.num_layers = num_layers
        self.output_variance = output_variance

        # Linearly increasing drop_path rates (0 at first block, max at last)
        drop_path_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]

        # Build Pre-LN residual blocks with stochastic depth
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                PreLNResidualBlock(feature_dim, dropout, drop_path=drop_path_rates[i])
            )

        # Final projection: output n_bands (mean only) or 2*n_bands (mean + log_variance).
        # Spectral norm bounds the Lipschitz constant of the final readout for OOD robustness.
        sn = nn.utils.parametrizations.spectral_norm if use_spectral_norm else (lambda m: m)
        output_dim = n_bands * 2 if output_variance else n_bands
        self.final_proj = sn(nn.Linear(feature_dim, output_dim))

    def forward(self, grid_features: torch.Tensor):
        """
        Decode grid features to fuel metrics raster.

        Args:
            grid_features: [batch_size, 5, 5, feature_dim]

        Returns:
            If output_variance=False:
                raster: [batch_size, n_bands, 5, 5] - predicted fuel metrics
            If output_variance=True:
                Tuple of (mean, log_var):
                - mean: [batch_size, n_bands, 5, 5] - predicted mean
                - log_var: [batch_size, n_bands, 5, 5] - predicted log-variance
        """
        x = grid_features  # [B, 5, 5, feature_dim]

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.final_proj(x)  # [B, 5, 5, n_bands] or [B, 5, 5, 2*n_bands]

        # Permute to raster format: [B, 5, 5, C] → [B, C, 5, 5]
        x = x.permute(0, 3, 1, 2).contiguous()

        if self.output_variance:
            # Split into mean and log_variance
            mean = x[:, :self.n_bands]      # [B, n_bands, 5, 5]
            log_var = x[:, self.n_bands:]   # [B, n_bands, 5, 5]
            return mean, log_var
        else:
            return x


class PreLNResidualBlock(nn.Module):
    """
    Pre-LayerNorm residual block: x = x + DropPath(MLP(LayerNorm(x)))

    Architecture:
        LayerNorm → Linear → GELU → Dropout → Linear
        + stochastic depth on residual connection

    Args:
        feature_dim: Input/output dimension
        dropout: Dropout probability
        drop_path: Stochastic depth probability (0.0 = disabled)
    """

    def __init__(self, feature_dim: int, dropout: float = 0.10, drop_path: float = 0.0):
        super().__init__()
        # Note: Final projection to n_bands is in WideRasterDecoder, not here
        self.norm = nn.LayerNorm(feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
        )
        # Stochastic depth - drops entire residual branch with probability drop_path
        self.drop_path = StochasticDepth(drop_path, mode="row") if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Pre-LN residual with stochastic depth: x = x + DropPath(MLP(LN(x)))"""
        return x + self.drop_path(self.mlp(self.norm(x)))


class RasterPredictionHead(nn.Module):
    """
    Complete raster prediction head: point features → regular grid → fuel metrics.

    Combines optional pre-aggregation LG-PAB blocks, PointToGridAggregator,
    and RasterDecoder into a single module for the multimodal raster model.

    Args:
        feature_dim: Feature dimension (must match point features)
        n_bands: Number of output fuel metrics bands (default 3)
        num_heads: Number of attention heads (default 8)
        distance_sigma: Gaussian sigma for distance weighting in meters (default 2.0).
                       Can be a list of floats for per-head multi-scale attention.
        grid_size: Grid size per side (default 5)
        tile_extent: Tile extent in meters (default 10.0)
        hidden_dim: Decoder hidden dimension (default 128, ignored if use_wide_decoder=True)
        num_decoder_layers: Number of decoder MLP layers (default 3, or residual blocks if wide)
        attention_dropout: Dropout for grid aggregation attention (default 0.1, keep low)
        decoder_dropout: Dropout for decoder MLP (default 0.1, can be higher)
        use_wide_decoder: Use wide decoder with Pre-LN residuals (default False)
        decoder_drop_path: Stochastic depth for WideRasterDecoder (default 0.0, disabled)
        num_pre_agg_blocks: Number of pre-aggregation LG-PAB blocks (default 2)
        pre_agg_lcl_heads: Local attention heads for pre-aggregation (default 4)
        pre_agg_glbl_heads: Global attention heads for pre-aggregation (default 4)
        pre_agg_dropout: Dropout for pre-aggregation blocks (default 0.1)
        pre_agg_k_neighbors: KNN neighbors for pre-aggregation (default 15)
        position_encoding_dim: Position encoding dimension (default 24)
        point_attn_drop_path: Stochastic depth for PosAwareGlobalFlashAttention in pre-agg blocks (default 0.0)
        use_v2_attention: Use PosAwareGlobalFlashAttentionV2 with decoupled Q/K/V (default False)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_bands: int = 3,
        num_heads: int = 8,
        distance_sigma: float | List[float] = 2.0,
        grid_size: int = 5,
        tile_extent: float = 10.0,
        hidden_dim: int = 128,
        num_decoder_layers: int = 3,
        attention_dropout: float = 0.1,
        decoder_dropout: float = 0.1,
        use_wide_decoder: bool = False,
        decoder_drop_path: float = 0.0,
        num_pre_agg_blocks: int = 2,
        pre_agg_lcl_heads: int = 4,
        pre_agg_glbl_heads: int = 4,
        pre_agg_dropout: float = 0.1,
        pre_agg_k_neighbors: int = 15,
        position_encoding_dim: int = 24,
        point_attn_drop_path: float = 0.0,
        use_v2_attention: bool = False,
        use_spectral_norm: bool = False,
        pos_encoder_dropout: float = 0.1,
        stochastic_pos_dropout_prob: float = 0.0,
        output_variance: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.num_pre_agg_blocks = num_pre_agg_blocks
        self.use_wide_decoder = use_wide_decoder
        self.output_variance = output_variance

        if use_wide_decoder:
            # SoftPillarConvDecoder: performs its own point-to-grid aggregation via bilinear
            # soft-splatting, bypassing the attention-based PointToGridAggregator.
            # Pre-aggregation blocks are skipped when using this decoder.
            self.pre_aggregation_blocks = None
            self.aggregator = None  # Not used - decoder does its own aggregation

            # Config parameter mapping:
            # - raster_hidden_dim → decoder_dim (ConvNeXt refinement dimension)
            # - raster_decoder_layers → num_blocks (number of refinement blocks)
            # - raster_decoder_dropout → dropout (refinement block dropout)
            self.decoder = SoftPillarConvDecoder(
                feature_dim=feature_dim,
                grid_size=grid_size,
                tile_extent=tile_extent,
                n_bands=n_bands,
                decoder_dim=hidden_dim,  # Maps from raster_hidden_dim
                num_blocks=num_decoder_layers,  # Maps from raster_decoder_layers
                dropout=decoder_dropout,  # Maps from raster_decoder_dropout
                z_bin_edges_m=(2.0, 5.0),  # Hardcoded vertical strata boundaries
                output_variance=output_variance,
                z_embed_dim=64,  # Hardcoded height encoder hidden dim
            )
        else:
            # Original path: Pre-aggregation LG-PAB blocks + PointToGridAggregator + RasterDecoder

            # Pre-aggregation refinement blocks (optional)
            if num_pre_agg_blocks > 0:
                self.pre_aggregation_blocks = nn.ModuleList([
                    LocalGlobalPointAttentionBlock(
                        in_channels=feature_dim,
                        out_channels=feature_dim,
                        num_lcl_heads=pre_agg_lcl_heads,
                        num_glbl_heads=pre_agg_glbl_heads,
                        pos_encoding_dim=position_encoding_dim,
                        dropout=pre_agg_dropout,
                        up_ratio=None,  # No upsampling
                        k_neighbors=pre_agg_k_neighbors,
                        global_drop_path=point_attn_drop_path,
                        use_v2_attention=use_v2_attention,
                        pos_encoder_dropout=pos_encoder_dropout,
                        stochastic_pos_dropout_prob=stochastic_pos_dropout_prob
                    )
                    for _ in range(num_pre_agg_blocks)
                ])
            else:
                self.pre_aggregation_blocks = None

            # Point-to-grid aggregation (soft Gaussian distance weighting)
            # Uses attention_dropout (keep low to preserve sparse signal)
            self.aggregator = PointToGridAggregator(
                feature_dim=feature_dim,
                num_heads=num_heads,
                distance_sigma=distance_sigma,
                grid_size=grid_size,
                tile_extent=tile_extent,
                dropout=attention_dropout,  # Split dropout: attention
            )

            # Original narrow decoder with dimension halving
            # Note: RasterDecoder does not support output_variance (use SoftPillarConvDecoder for heteroscedastic loss)
            self.decoder = RasterDecoder(
                feature_dim=feature_dim,
                n_bands=n_bands,
                hidden_dim=hidden_dim,
                num_layers=num_decoder_layers,
                dropout=decoder_dropout,  # Split dropout: decoder
                use_spectral_norm=use_spectral_norm,
            )

    def forward(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict]
    ) -> torch.Tensor:
        """
        Predict fuel metrics raster from point features.

        Args:
            point_features: [N_total, feature_dim] - concatenated point features
            point_positions: [N_total, 3] - Z-SCORE NORMALIZED point positions
            batch_indices: [N_total] - which batch each point belongs to
            norm_params: List of dicts with 'coord_mean', 'coord_std'

        Returns:
            raster: [batch_size, n_bands, 5, 5] - predicted fuel metrics (z-score normalized)
            Or if output_variance=True:
                Tuple of (mean, log_var): each [batch_size, n_bands, 5, 5]
        """
        if self.use_wide_decoder:
            # SoftPillarConvDecoder path: decoder does its own aggregation internally
            # No pre-aggregation blocks, no separate aggregator
            raster = self.decoder(
                point_features=point_features,
                point_positions=point_positions,
                batch_indices=batch_indices,
                norm_params=norm_params
            )
        else:
            # Original path: pre-aggregation → aggregator → decoder
            x_feat = point_features
            if self.pre_aggregation_blocks is not None:
                for block in self.pre_aggregation_blocks:
                    x_feat, _ = block(x_feat, point_positions, edge_index=None)
                    # edge_index=None → block builds KNN graph internally

            # Aggregate to grid
            grid_features = self.aggregator(
                point_features=x_feat,
                point_positions=point_positions,
                batch_indices=batch_indices,
                norm_params=norm_params
            )  # [batch_size, 5, 5, feature_dim]

            # Decode to raster
            raster = self.decoder(grid_features)  # [batch_size, n_bands, 5, 5]

        return raster

    def forward_with_diagnostics(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict]
    ) -> tuple:
        """
        Forward pass with diagnostic information about feature diversity.

        This helps identify whether the bottleneck is in aggregation (low diversity)
        or decoder (high diversity but poor predictions).

        NOTE: Only supported when use_wide_decoder=False. SoftPillarConvDecoder
        does not support diagnostics mode.

        Returns:
            raster: [batch_size, n_bands, 5, 5] - predicted fuel metrics
            diagnostics: Dict with diversity metrics
        """
        if self.use_wide_decoder:
            raise NotImplementedError(
                "forward_with_diagnostics is not supported when use_wide_decoder=True. "
                "SoftPillarConvDecoder does not support diagnostics mode."
            )

        # Apply pre-aggregation LG-PAB blocks (if enabled)
        x_feat = point_features
        if self.pre_aggregation_blocks is not None:
            for block in self.pre_aggregation_blocks:
                x_feat, _ = block(x_feat, point_positions, edge_index=None)

        # Aggregate to grid (with coverage stats)
        grid_features, coverage_stats = self.aggregator(
            point_features=x_feat,
            point_positions=point_positions,
            batch_indices=batch_indices,
            norm_params=norm_params,
            return_coverage=True
        )  # grid_features: [batch_size, 5, 5, feature_dim], coverage_stats: Dict

        # Compute point and grid feature diversity diagnostics
        point_diagnostics = self._compute_point_feature_diversity(x_feat, batch_indices)
        grid_diagnostics = self._compute_feature_diversity(grid_features)

        # Compute query embedding norm diagnostics (detect positional encoding washout)
        query_diagnostics = self._compute_query_norm_diagnostics()

        # Combine all diagnostics
        diagnostics = {**point_diagnostics, **grid_diagnostics, **coverage_stats, **query_diagnostics}

        # Decode to raster
        raster = self.decoder(grid_features)  # [batch_size, n_bands, 5, 5]

        return raster, diagnostics

    @staticmethod
    def _compute_point_feature_diversity(point_features: torch.Tensor, batch_indices: torch.Tensor) -> Dict[str, float]:
        """
        Compute diversity metrics for point features before aggregation.

        This helps identify if the encoder is producing diverse enough features.

        Args:
            point_features: [N_total, feature_dim]
            batch_indices: [N_total] batch assignment

        Returns:
            Dict with diversity metrics
        """
        # Sample one batch for efficiency (batch 0)
        mask_b0 = (batch_indices == 0)
        if mask_b0.sum() == 0:
            return {}

        features_b0 = point_features[mask_b0]  # [N_0, F]
        N, F = features_b0.shape

        # 1. Feature std across points
        feature_std = features_b0.std(dim=0).mean().item()

        # 2. Feature range
        feature_max = features_b0.max(dim=0).values
        feature_min = features_b0.min(dim=0).values
        feature_range = (feature_max - feature_min).mean().item()

        # 3. Coefficient of variation
        feature_mean_abs = features_b0.abs().mean(dim=0)
        feature_std_vec = features_b0.std(dim=0)
        cv = (feature_std_vec / (feature_mean_abs + 1e-8)).mean().item()

        # 4. Pairwise cosine similarity (subsample if too many points)
        max_points_for_sim = 200
        if N > max_points_for_sim:
            # Random sample
            indices = torch.randperm(N)[:max_points_for_sim]
            features_sample = features_b0[indices]
        else:
            features_sample = features_b0

        # Normalize
        features_norm = features_sample / (features_sample.norm(dim=1, keepdim=True) + 1e-8)
        sim_matrix = torch.matmul(features_norm, features_norm.t())

        # Upper triangle excluding diagonal
        N_sample = features_sample.shape[0]
        triu_indices = torch.triu_indices(N_sample, N_sample, offset=1)
        similarities = sim_matrix[triu_indices[0], triu_indices[1]]
        mean_cosine_sim = similarities.mean().item()

        # 5. Feature magnitude
        feature_norm = features_b0.norm(dim=1).mean().item()

        return {
            'point_feature_std': feature_std,
            'point_feature_range': feature_range,
            'point_feature_cv': cv,
            'point_cosine_similarity': mean_cosine_sim,
            'point_feature_norm': feature_norm
        }

    def _compute_feature_diversity(self, grid_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute diversity metrics for grid features after aggregation.

        Low diversity → aggregation is bottleneck (features too similar)
        High diversity → decoder is bottleneck (features diverse but predictions poor)

        Args:
            grid_features: [batch_size, 5, 5, feature_dim]

        Returns:
            Dict with diversity metrics
        """
        B, H, W, F = grid_features.shape

        # Flatten spatial dimensions for analysis
        features_spatial = grid_features.view(B, H*W, F)  # [B, 25, F]

        # 1. Feature std across spatial positions (per feature dim)
        # High std → features vary across grid positions (good)
        feature_std_spatial = features_spatial.std(dim=1).mean().item()  # Mean std across batch and feature dims

        # 2. Feature range (max - min across spatial positions)
        feature_max = features_spatial.max(dim=1).values  # [B, F]
        feature_min = features_spatial.min(dim=1).values  # [B, F]
        feature_range = (feature_max - feature_min).mean().item()

        # 3. Coefficient of variation (std/mean, normalized measure)
        feature_mean_abs = features_spatial.abs().mean(dim=1)  # [B, F]
        feature_std = features_spatial.std(dim=1)  # [B, F]
        cv = (feature_std / (feature_mean_abs + 1e-8)).mean().item()

        # 4. Pairwise cosine similarity between grid cells (within each batch)
        # Low similarity → diverse representations (good)
        # High similarity → collapsed representations (bad)
        cosine_sims = []
        for b in range(B):
            features_b = features_spatial[b]  # [25, F]
            # Normalize
            features_norm = features_b / (features_b.norm(dim=1, keepdim=True) + 1e-8)
            # Compute cosine similarity matrix [25, 25]
            sim_matrix = torch.matmul(features_norm, features_norm.t())
            # Get upper triangle (excluding diagonal)
            triu_indices = torch.triu_indices(H*W, H*W, offset=1)
            similarities = sim_matrix[triu_indices[0], triu_indices[1]]
            cosine_sims.append(similarities.mean().item())

        mean_cosine_sim = sum(cosine_sims) / len(cosine_sims)

        # 5. Feature magnitude (L2 norm per cell)
        feature_norm = features_spatial.norm(dim=2).mean().item()  # Mean L2 norm across batch and spatial

        return {
            'grid_feature_std_spatial': feature_std_spatial,
            'grid_feature_range': feature_range,
            'grid_feature_cv': cv,
            'grid_cosine_similarity': mean_cosine_sim,
            'grid_feature_norm': feature_norm
        }

    def _compute_query_norm_diagnostics(self) -> Dict[str, float]:
        """
        Compute query embedding norm diagnostics to detect positional encoding washout.

        If learned query embeddings dominate positional encodings (query_norm >> pos_norm),
        the model may lose spatial position information over training.

        Returns:
            Dict with norm diagnostics:
            - query_embed_norm: Mean L2 norm of learned query embeddings
            - pos_encoding_norm: Mean L2 norm of positional encodings
            - query_pos_ratio: Ratio of query_norm / pos_norm (should stay ~1.0)
        """
        # Access the grid queries from the aggregator
        grid_queries = self.aggregator.grid_queries

        # Learned query embeddings
        query_embed = grid_queries.query_embed  # [25, feature_dim]
        query_norm = query_embed.norm(dim=1).mean().item()

        # Positional encodings (buffer)
        pos_encoding = grid_queries.pos_encoding  # [25, feature_dim]
        pos_norm = pos_encoding.norm(dim=1).mean().item()

        # Ratio (high ratio = positional info being washed out)
        ratio = query_norm / (pos_norm + 1e-8)

        return {
            'query_embed_norm': query_norm,
            'pos_encoding_norm': pos_norm,
            'query_pos_ratio': ratio
        }
