"""
Grid Cross-Attention Fusion for unified multimodal aggregation.

This module implements a single-step fusion where learnable grid queries
attend to all modalities simultaneously (point features + NAIP patches +
UAVSAR patches), replacing the two-stage CrossAttentionFusion → PointToGridAggregator
approach.

Architecture:
    25 Learnable Grid Queries
            ↓
    Distance-Weighted Cross-Attention
            ↓
    [N points + 16 NAIP patches + 16 UAVSAR patches]

Output: Grid features [B, 5, 5, feature_dim] → RasterDecoder → [B, n_bands, 5, 5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import math


class PatchPositionEncoding(nn.Module):
    """
    Sinusoidal 2D positional encoding for image patches.

    NAIP/UAVSAR patches cover 20×20m with 4×4 grid (5m spacing per patch).
    Patch centers: [-7.5, -2.5, 2.5, 7.5]m in both x and y.

    Args:
        feature_dim: Output encoding dimension
        patch_grid_size: Number of patches per side (default 4 for 4×4 grid)
        patch_extent: Total extent covered by patches in meters (default 20.0)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        patch_grid_size: int = 4,
        patch_extent: float = 20.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_grid_size = patch_grid_size
        self.num_patches = patch_grid_size * patch_grid_size

        # Calculate patch centers (in meters, centered at 0)
        patch_spacing = patch_extent / patch_grid_size  # 5.0 meters
        centers_1d = np.linspace(
            -patch_extent/2 + patch_spacing/2,
            patch_extent/2 - patch_spacing/2,
            patch_grid_size
        )
        # centers_1d = [-7.5, -2.5, 2.5, 7.5] meters

        # Create 2D grid of patch centers
        grid_y, grid_x = np.meshgrid(centers_1d, centers_1d, indexing='ij')
        patch_centers = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1)  # [16, 2]

        # Register as buffer
        self.register_buffer('patch_centers', torch.from_numpy(patch_centers).float())  # [16, 2]

        # Create sinusoidal position encoding
        pos_encoding = self._create_2d_sinusoidal_encoding(
            patch_centers, feature_dim, patch_extent
        )
        self.register_buffer('pos_encoding', torch.from_numpy(pos_encoding).float())  # [16, feature_dim]

    def _create_2d_sinusoidal_encoding(
        self,
        positions: np.ndarray,
        feature_dim: int,
        extent: float
    ) -> np.ndarray:
        """Create 2D sinusoidal positional encoding for patch positions."""
        num_positions = positions.shape[0]

        # Normalize positions to [0, 1] range
        positions_norm = (positions + extent / 2) / extent

        # Half dimensions for X, half for Y
        half_dim = feature_dim // 2

        # Frequency bands
        div_term = np.exp(np.arange(0, half_dim, 2) * (-np.log(10000.0) / half_dim))

        # X and Y encodings
        x_pos = positions_norm[:, 0:1]
        y_pos = positions_norm[:, 1:2]

        pe_x = np.zeros((num_positions, half_dim))
        pe_y = np.zeros((num_positions, half_dim))

        pe_x[:, 0::2] = np.sin(x_pos * div_term)
        pe_x[:, 1::2] = np.cos(x_pos * div_term)
        pe_y[:, 0::2] = np.sin(y_pos * div_term)
        pe_y[:, 1::2] = np.cos(y_pos * div_term)

        return np.concatenate([pe_x, pe_y], axis=1)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            pos_encoding: [16, feature_dim] sinusoidal position encoding
            patch_centers: [16, 2] patch centers in meters
        """
        return self.pos_encoding, self.patch_centers


class MultiSourceDistanceAttention(nn.Module):
    """
    Cross-attention where grid queries attend to multiple key/value sources
    (points + image patches), each with its own position set for distance weighting.

    Uses multi-scale Gaussian distance weighting with per-head sigma values.

    Args:
        feature_dim: Feature dimension for queries, keys, values
        num_heads: Number of attention heads
        distance_sigma: Gaussian sigma values (float or list of floats per head)
        dropout: Dropout probability
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        distance_sigma: Union[float, List[float]] = 2.0,
        dropout: float = 0.1
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.dropout = dropout

        # Handle per-head sigma values
        if isinstance(distance_sigma, (list, tuple)):
            assert len(distance_sigma) == num_heads, \
                f"distance_sigma list length ({len(distance_sigma)}) must match num_heads ({num_heads})"
            self.register_buffer('distance_sigmas', torch.tensor(distance_sigma, dtype=torch.float32))
        else:
            self.register_buffer('distance_sigmas', torch.tensor([distance_sigma] * num_heads, dtype=torch.float32))

        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        queries: torch.Tensor,          # [B, 25, D] grid queries
        query_positions: torch.Tensor,  # [25, 2] grid centers in meters
        keys: torch.Tensor,             # [B, N_total, D] concatenated keys
        values: torch.Tensor,           # [B, N_total, D] concatenated values
        key_positions: torch.Tensor,    # [B, N_total, 2] positions in meters
        key_mask: torch.Tensor,         # [B, N_total] True for valid, False for padding
    ) -> torch.Tensor:
        """
        Apply distance-weighted cross-attention from grid queries to all sources.

        Args:
            queries: [B, 25, D] grid query embeddings
            query_positions: [25, 2] grid centers (X, Y in meters)
            keys: [B, N_total, D] all key features (points + patches)
            values: [B, N_total, D] all value features
            key_positions: [B, N_total, 2] positions for all keys
            key_mask: [B, N_total] True for valid positions

        Returns:
            output: [B, 25, D] aggregated grid features
        """
        batch_size, num_queries, _ = queries.shape
        _, max_keys, _ = keys.shape

        # Reshape for multi-head attention
        # [B, N, D] -> [B, H, N, head_dim]
        Q = queries.view(batch_size, num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = keys.view(batch_size, max_keys, self.num_heads, self.head_dim).transpose(1, 2)
        V = values.view(batch_size, max_keys, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute distances: [B, 25, N_total]
        distances = torch.cdist(
            query_positions.unsqueeze(0).expand(batch_size, -1, -1),  # [B, 25, 2]
            key_positions,  # [B, N_total, 2]
            p=2
        )

        # Multi-scale Gaussian distance weighting
        # sigmas: [H] -> [1, H, 1, 1]
        sigmas = self.distance_sigmas.view(1, self.num_heads, 1, 1)
        distances_expanded = distances.unsqueeze(1)  # [B, 1, 25, N_total]

        # Distance weights: exp(-d² / 2σ²)
        log_distance_weights = -distances_expanded**2 / (2 * sigmas**2)
        # [B, H, 25, N_total]

        # Apply padding mask
        pad_mask = ~key_mask  # [B, N_total] True for padding
        log_distance_weights = log_distance_weights.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, N_total]
            float('-inf')
        )

        # Scaled dot-product attention with distance bias
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # [B, H, 25, N_total]

        # Add distance weights (in log space)
        scores = scores + log_distance_weights

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)

        # Handle NaN from all -inf (no valid keys)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        # Dropout
        if self.training and self.dropout > 0:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=True)

        # Apply attention
        output = torch.matmul(attn_weights, V)
        # [B, H, 25, head_dim]

        # Reshape back: [B, H, 25, head_dim] -> [B, 25, D]
        output = output.transpose(1, 2).contiguous().view(batch_size, num_queries, self.feature_dim)

        # Output projection
        output = self.out_proj(output)

        return output


class GridCrossAttentionFusion(nn.Module):
    """
    Unified grid-based fusion: 25 grid queries attend to all modalities.

    Replaces CrossAttentionFusion + PointToGridAggregator with a single step
    where grid queries directly attend to points and image patches.

    Args:
        point_dim: Point feature dimension (default 256)
        patch_dim: Image patch embedding dimension (default 128)
        grid_size: Output grid size per side (default 5)
        tile_extent: Tile extent in meters (default 10.0)
        num_heads: Number of attention heads (default 8)
        distance_sigma: Gaussian sigma values for distance weighting
        dropout: Attention dropout probability
        use_naip: Whether to use NAIP features
        use_uavsar: Whether to use UAVSAR features
    """

    def __init__(
        self,
        point_dim: int = 256,
        patch_dim: int = 128,
        grid_size: int = 5,
        tile_extent: float = 10.0,
        num_heads: int = 8,
        distance_sigma: Union[float, List[float]] = 2.0,
        dropout: float = 0.1,
        use_naip: bool = True,
        use_uavsar: bool = True
    ):
        super().__init__()

        self.point_dim = point_dim
        self.patch_dim = patch_dim
        self.grid_size = grid_size
        self.tile_extent = tile_extent
        self.num_queries = grid_size * grid_size
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar

        # Grid queries (reuse LearnableGridQueries pattern)
        from .raster_head import LearnableGridQueries
        self.grid_queries = LearnableGridQueries(
            feature_dim=point_dim,
            grid_size=grid_size,
            tile_extent=tile_extent
        )

        # Patch position encoding (4×4 grid over 20×20m)
        self.patch_pos_encoding = PatchPositionEncoding(
            feature_dim=point_dim,  # Match point_dim for addition
            patch_grid_size=4,
            patch_extent=20.0
        )

        # Point position encoding (matches existing implementation)
        self.point_pos_dim = point_dim

        # Project points to key/value
        # Point features already at point_dim, just need K/V projections
        self.point_key_proj = nn.Linear(point_dim, point_dim)
        self.point_value_proj = nn.Linear(point_dim, point_dim)

        # Project patches to common dimension + K/V
        # Patches start at patch_dim, need to project up to point_dim
        if use_naip:
            self.naip_proj = nn.Linear(patch_dim, point_dim)
            self.naip_key_proj = nn.Linear(point_dim, point_dim)
            self.naip_value_proj = nn.Linear(point_dim, point_dim)

        if use_uavsar:
            self.uavsar_proj = nn.Linear(patch_dim, point_dim)
            self.uavsar_key_proj = nn.Linear(point_dim, point_dim)
            self.uavsar_value_proj = nn.Linear(point_dim, point_dim)

        # Multi-source attention
        self.attention = MultiSourceDistanceAttention(
            feature_dim=point_dim,
            num_heads=num_heads,
            distance_sigma=distance_sigma,
            dropout=dropout
        )

        # LayerNorm and feedforward (standard transformer block pattern)
        self.norm1 = nn.LayerNorm(point_dim)
        self.norm2 = nn.LayerNorm(point_dim)
        self.ffn = nn.Sequential(
            nn.Linear(point_dim, point_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(point_dim * 2, point_dim),
            nn.Dropout(dropout)
        )

        # Pre-compute frequency bands for position encoding (cached as buffer)
        half_dim = point_dim // 2
        div_term = torch.exp(
            torch.arange(0, half_dim, 2, dtype=torch.float32) *
            (-math.log(10000.0) / half_dim)
        )  # [half_dim // 2]
        self.register_buffer('pos_div_term', div_term)

    def _create_point_position_encoding_batched(
        self,
        point_positions: torch.Tensor,  # [B, max_N, 2] in meters
    ) -> torch.Tensor:
        """
        Create sinusoidal position encoding for batched point positions.

        Vectorized implementation - no Python loops, uses cached div_term.

        Args:
            point_positions: [B, max_N, 2] point positions (X, Y in meters)

        Returns:
            pos_encoding: [B, max_N, point_dim]
        """
        B, max_N, _ = point_positions.shape
        half_dim = self.point_dim // 2

        # Normalize to [0, 1] based on tile extent (points are in ~[-5, 5]m)
        extent = self.tile_extent * 1.5  # 15m to handle edge cases
        positions_norm = (point_positions + extent / 2) / extent
        positions_norm = positions_norm.clamp(0, 1)  # Safety clamp

        # Get cached div_term (already on correct device via buffer)
        div_term = self.pos_div_term  # [half_dim // 2]

        # X and Y positions: [B, max_N, 1]
        x_pos = positions_norm[:, :, 0:1]
        y_pos = positions_norm[:, :, 1:2]

        # Compute sin/cos for all positions at once
        # x_pos * div_term: [B, max_N, 1] * [half_dim//2] -> [B, max_N, half_dim//2]
        x_angles = x_pos * div_term  # [B, max_N, half_dim // 2]
        y_angles = y_pos * div_term  # [B, max_N, half_dim // 2]

        # Interleave sin and cos: [B, max_N, half_dim]
        pe_x = torch.stack([torch.sin(x_angles), torch.cos(x_angles)], dim=-1)
        pe_x = pe_x.view(B, max_N, half_dim)  # [B, max_N, half_dim]

        pe_y = torch.stack([torch.sin(y_angles), torch.cos(y_angles)], dim=-1)
        pe_y = pe_y.view(B, max_N, half_dim)  # [B, max_N, half_dim]

        return torch.cat([pe_x, pe_y], dim=-1)  # [B, max_N, point_dim]

    def forward(
        self,
        point_features: torch.Tensor,       # [N_total, point_dim]
        point_positions: torch.Tensor,      # [N_total, 3] z-score normalized
        batch_indices: torch.Tensor,        # [N_total]
        norm_params: List[Dict],            # For denormalization
        naip_embeddings: Optional[torch.Tensor] = None,   # [B, 16, patch_dim] or None
        uavsar_embeddings: Optional[torch.Tensor] = None, # [B, 16, patch_dim] or None
    ) -> torch.Tensor:
        """
        Unified fusion: grid queries attend to all modalities.

        Args:
            point_features: [N_total, point_dim] encoded point features
            point_positions: [N_total, 3] z-score normalized positions
            batch_indices: [N_total] batch assignment
            norm_params: List of dicts with 'coord_mean', 'coord_std'
            naip_embeddings: [B, 16, patch_dim] NAIP patch embeddings or None
            uavsar_embeddings: [B, 16, patch_dim] UAVSAR patch embeddings or None

        Returns:
            grid_features: [B, 5, 5, point_dim]
        """
        batch_size = len(norm_params)
        device = point_features.device
        dtype = point_features.dtype

        # 1. Denormalize point positions to meters
        coord_mean_batch = torch.stack([p['coord_mean'] for p in norm_params]).to(device=device, dtype=dtype)
        coord_std_batch = torch.stack([p['coord_std'] for p in norm_params]).to(device=device, dtype=dtype)
        coord_means = coord_mean_batch[batch_indices]  # [N_total, 3]
        coord_stds = coord_std_batch[batch_indices]    # [N_total, 3]
        point_pos_phys = point_positions * coord_stds + coord_means  # [N_total, 3]
        point_pos_xy = point_pos_phys[:, :2]  # [N_total, 2]

        # 2. Generate grid queries
        query_embeddings, grid_centers = self.grid_queries(batch_size, device)
        # query_embeddings: [B, 25, point_dim]
        # grid_centers: [25, 2] in meters

        # 3. Prepare point keys/values
        # Create dense batch for points
        point_dense, point_mask = to_dense_batch(
            point_features, batch_indices, batch_size=batch_size
        )  # [B, max_points, point_dim], [B, max_points]
        point_pos_dense, _ = to_dense_batch(
            point_pos_xy, batch_indices, batch_size=batch_size
        )  # [B, max_points, 2]

        max_points = point_dense.shape[1]

        # Add position encoding to point features (additive)
        # Vectorized: process entire batch at once (no Python loop)
        point_pos_enc = self._create_point_position_encoding_batched(point_pos_dense)
        # [B, max_points, point_dim]

        # Add position encoding to features
        point_features_with_pos = point_dense + point_pos_enc

        # Project to keys/values
        point_keys = self.point_key_proj(point_features_with_pos)    # [B, max_points, point_dim]
        point_values = self.point_value_proj(point_dense)  # [B, max_points, point_dim]

        # 4. Prepare patch keys/values
        patch_pos_enc, patch_centers = self.patch_pos_encoding()
        # patch_pos_enc: [16, point_dim]
        # patch_centers: [16, 2]

        all_keys = [point_keys]
        all_values = [point_values]
        all_positions = [point_pos_dense]
        all_masks = [point_mask]

        num_patches = 16  # 4x4 grid

        if self.use_naip and naip_embeddings is not None:
            # Project NAIP to common dim + add position encoding
            naip_proj = self.naip_proj(naip_embeddings)  # [B, 16, point_dim]
            naip_with_pos = naip_proj + patch_pos_enc.unsqueeze(0)  # [B, 16, point_dim]
            naip_keys = self.naip_key_proj(naip_with_pos)
            naip_values = self.naip_value_proj(naip_proj)

            # Expand patch positions for batch
            naip_positions = patch_centers.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 16, 2]
            naip_mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)

            all_keys.append(naip_keys)
            all_values.append(naip_values)
            all_positions.append(naip_positions)
            all_masks.append(naip_mask)

        if self.use_uavsar and uavsar_embeddings is not None:
            # Project UAVSAR to common dim + add position encoding
            uavsar_proj = self.uavsar_proj(uavsar_embeddings)  # [B, 16, point_dim]
            uavsar_with_pos = uavsar_proj + patch_pos_enc.unsqueeze(0)  # [B, 16, point_dim]
            uavsar_keys = self.uavsar_key_proj(uavsar_with_pos)
            uavsar_values = self.uavsar_value_proj(uavsar_proj)

            # Expand patch positions for batch
            uavsar_positions = patch_centers.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 16, 2]
            uavsar_mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)

            all_keys.append(uavsar_keys)
            all_values.append(uavsar_values)
            all_positions.append(uavsar_positions)
            all_masks.append(uavsar_mask)

        # 5. Concatenate all sources
        concat_keys = torch.cat(all_keys, dim=1)        # [B, N_total, point_dim]
        concat_values = torch.cat(all_values, dim=1)    # [B, N_total, point_dim]
        concat_positions = torch.cat(all_positions, dim=1)  # [B, N_total, 2]
        concat_mask = torch.cat(all_masks, dim=1)       # [B, N_total]

        # 6. Apply multi-source attention
        grid_features = self.attention(
            queries=query_embeddings,
            query_positions=grid_centers,
            keys=concat_keys,
            values=concat_values,
            key_positions=concat_positions,
            key_mask=concat_mask
        )  # [B, 25, point_dim]

        # 7. Add residual + feedforward (transformer block)
        grid_features = query_embeddings + grid_features  # Residual
        grid_features = self.norm1(grid_features)
        grid_features = grid_features + self.ffn(grid_features)  # FFN with residual
        grid_features = self.norm2(grid_features)

        # 8. Reshape to grid
        grid_features = grid_features.view(batch_size, self.grid_size, self.grid_size, self.point_dim)
        # [B, 5, 5, point_dim]

        return grid_features
