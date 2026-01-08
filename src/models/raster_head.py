"""
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
from torchvision.ops import StochasticDepth
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# Import LocalGlobalPointAttentionBlock for pre-aggregation refinement
from .multimodal_model import LocalGlobalPointAttentionBlock


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
        use_spectral_norm: bool = False
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

        # Apply spectral normalization to out_proj only
        if use_spectral_norm:
            self.out_proj = nn.utils.spectral_norm(self.out_proj)

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
        use_spectral_norm: bool = False
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
            use_spectral_norm=use_spectral_norm
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
        dropout: float = 0.1
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

        # Final layer to n_bands
        layers.append(nn.Linear(in_dim, n_bands))

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
        → Final projection: Linear(feature_dim → n_bands)
        → Permute to [B, n_bands, 5, 5]

    Uses GELU activation (smoother than ReLU, better for regression).
    Pre-LN structure provides better gradient flow than Post-LN.
    Stochastic depth rates increase linearly from 0 (first block) to drop_path (last block).

    Args:
        feature_dim: Input/hidden feature dimension (default 256)
        n_bands: Number of output fuel metrics bands (default 2)
        num_layers: Number of residual blocks (default 2)
        dropout: Dropout probability (default 0.35)
        drop_path: Maximum stochastic depth probability (default 0.0, disabled)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_bands: int = 2,
        num_layers: int = 2,
        dropout: float = 0.35,
        drop_path: float = 0.0,
        use_spectral_norm: bool = False
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.num_layers = num_layers

        # Linearly increasing drop_path rates (0 at first block, max at last)
        drop_path_rates = [drop_path * i / max(num_layers - 1, 1) for i in range(num_layers)]

        # Build Pre-LN residual blocks with stochastic depth
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                PreLNResidualBlock(feature_dim, dropout, drop_path=drop_path_rates[i], use_spectral_norm=use_spectral_norm)
            )

        # Final projection to n_bands (NO spectral norm - this is the output layer)
        self.final_proj = nn.Linear(feature_dim, n_bands)

    def forward(self, grid_features: torch.Tensor) -> torch.Tensor:
        """
        Decode grid features to fuel metrics raster.

        Args:
            grid_features: [batch_size, 5, 5, feature_dim]

        Returns:
            raster: [batch_size, n_bands, 5, 5] - predicted fuel metrics
        """
        x = grid_features  # [B, 5, 5, feature_dim]

        # Apply residual blocks
        for block in self.blocks:
            x = block(x)

        # Final projection
        x = self.final_proj(x)  # [B, 5, 5, n_bands]

        # Permute to raster format: [B, 5, 5, n_bands] → [B, n_bands, 5, 5]
        x = x.permute(0, 3, 1, 2).contiguous()

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

    def __init__(self, feature_dim: int, dropout: float = 0.10, drop_path: float = 0.0, use_spectral_norm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)

        # Build linear layers - optionally apply spectral normalization to BOTH
        # Note: Final projection to n_bands is in WideRasterDecoder, not here
        linear1 = nn.Linear(feature_dim, feature_dim*2)
        linear2 = nn.Linear(feature_dim*2, feature_dim)

        if use_spectral_norm:
            linear1 = nn.utils.spectral_norm(linear1)
            linear2 = nn.utils.spectral_norm(linear2)

        self.mlp = nn.Sequential(
            linear1,
            nn.GELU(),
            nn.Dropout(dropout),
            linear2
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
        stochastic_pos_dropout_prob: float = 0.0
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.num_pre_agg_blocks = num_pre_agg_blocks
        self.use_wide_decoder = use_wide_decoder

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
            use_spectral_norm=use_spectral_norm
        )

        # Raster decoder (uses decoder_dropout, can be higher for regularization)
        if use_wide_decoder:
            # Wide decoder with Pre-LN residuals + stochastic depth: 256→256→256→n_bands
            self.decoder = WideRasterDecoder(
                feature_dim=feature_dim,
                n_bands=n_bands,
                num_layers=num_decoder_layers,  # Number of residual blocks
                dropout=decoder_dropout,
                drop_path=decoder_drop_path,
                use_spectral_norm=use_spectral_norm
            )
        else:
            # Original narrow decoder with dimension halving
            self.decoder = RasterDecoder(
                feature_dim=feature_dim,
                n_bands=n_bands,
                hidden_dim=hidden_dim,
                num_layers=num_decoder_layers,
                dropout=decoder_dropout  # Split dropout: decoder
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
        """
        # Apply pre-aggregation LG-PAB blocks (if enabled)
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

        Returns:
            raster: [batch_size, n_bands, 5, 5] - predicted fuel metrics
            diagnostics: Dict with diversity metrics
        """
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