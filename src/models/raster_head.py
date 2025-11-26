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
from typing import Dict, List, Optional, Tuple
import numpy as np
import math

# Import LocalGlobalPointAttentionBlock for pre-aggregation refinement
from .multimodal_model import LocalGlobalPointAttentionBlock


class LearnableGridQueries(nn.Module):
    """
    Learnable query embeddings for regular grid cells.

    Creates fixed 5×5 grid over 10m tile with learnable feature embeddings.
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

        # Initialize with small values
        nn.init.normal_(self.query_embed, std=0.01)

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
        # Expand query embeddings for batch
        query_embeddings = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1)

        return query_embeddings, self.grid_centers


class DistanceMaskedAttention(nn.Module):
    """
    Multi-head cross-attention with distance-based masking.

    Queries (grid cells) attend to keys/values (point features) within a radius.
    Points outside the radius are masked out.

    Uses manual scaled dot-product attention implementation (bypassing
    PyTorch's F.scaled_dot_product_attention for full compatibility).

    Args:
        feature_dim: Feature dimension
        num_heads: Number of attention heads (must divide feature_dim)
        radius: Distance threshold in meters (points beyond this are masked)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        radius: float = 5.0,
        dropout: float = 0.1
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.radius = radius
        self.dropout = dropout

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
        batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply distance-masked cross-attention.

        Args:
            queries: [batch_size, num_queries, feature_dim] - grid query embeddings
            keys: [N_total, feature_dim] - point features (concatenated across batch)
            values: [N_total, feature_dim] - point features (same as keys)
            query_positions: [num_queries, 2] - grid centers (X, Y in METERS)
            key_positions: [N_total, 2] - point positions (X, Y in METERS, bbox-normalized)
            batch_indices: [N_total] - which batch each point belongs to (required)

        Returns:
            output: [batch_size, num_queries, feature_dim] - aggregated grid features
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

        # 4. Compute distance mask
        # query_positions: [num_queries, 2] -> [B, num_queries, 2]
        # key_pos_dense: [B, max_points, 2]
        distances = torch.cdist(
            query_positions.unsqueeze(0).expand(batch_size, -1, -1),
            key_pos_dense,
            p=2
        )  # [B, num_queries, max_points]

        # Create attention mask
        # True means keep, False means mask out (for scaled_dot_product_attention attn_mask)
        # Note: scaled_dot_product_attention expects mask where True = mask out if boolean.
        
        # Distance mask: True if distance > radius (should be masked)
        dist_mask = (distances > self.radius)
        
        # Padding mask: True if padded (should be masked)
        # mask from to_dense_batch is True for real points, False for padding
        # We need [B, 1, 1, max_points] for broadcasting
        pad_mask = (~mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, max_points]
        
        # Combine masks: [B, 1, num_queries, max_points]
        # Expand dist_mask for heads: [B, 1, num_queries, max_points]
        attn_mask_bool = dist_mask.unsqueeze(1) | pad_mask

        # Convert to float mask for cuDNN compatibility
        # 0.0 for keep, -inf for mask
        # CRITICAL: Use float32 (not half-precision) to avoid cuDNN -inf issues with AMP
        attn_mask = torch.zeros(
            (batch_size, 1, num_queries, max_points),
            dtype=torch.float32,
            device=Q.device
        )
        attn_mask.masked_fill_(attn_mask_bool, float('-inf'))

        # 5. Manual Scaled Dot-Product Attention
        # NOTE: Bypassing F.scaled_dot_product_attention entirely due to persistent
        # cuDNN compatibility issues with custom distance-masked attention patterns.
        # Manual implementation is more robust and gives full control.

        # Q: [B, num_heads, num_queries, head_dim]
        # K: [B, num_heads, max_points, head_dim]
        # V: [B, num_heads, max_points, head_dim]
        # attn_mask: [B, 1, num_queries, max_points] with -inf for masked positions

        # Compute attention scores: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # scores: [B, num_heads, num_queries, max_points]

        # Apply mask (broadcast over heads dimension)
        scores = scores + attn_mask  # attn_mask: [B, 1, num_queries, max_points]

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

        return output


class PointToGridAggregator(nn.Module):
    """
    Aggregates point cloud features to a regular grid using learnable queries.

    Combines LearnableGridQueries and DistanceMaskedAttention to transform
    irregular point cloud features into a regular 5×5 grid.

    Args:
        feature_dim: Feature dimension (must match point features)
        num_heads: Number of attention heads
        radius: Distance threshold in meters (default 5.0)
        grid_size: Grid size per side (default 5)
        tile_extent: Tile extent in meters (default 10.0)
        dropout: Dropout probability (default 0.1)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        radius: float = 5.0,
        grid_size: int = 5,
        tile_extent: float = 10.0,
        dropout: float = 0.1
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

        # Distance-masked attention
        self.attention = DistanceMaskedAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            radius=radius,
            dropout=dropout
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict]
    ) -> torch.Tensor:
        """
        Aggregate point features to regular grid.

        Args:
            point_features: [N_total, feature_dim] - concatenated point features
            point_positions: [N_total, 3] - Z-SCORE NORMALIZED point positions (X, Y, Z)
            batch_indices: [N_total] - which batch each point belongs to
            norm_params: List of dicts (length batch_size) with 'coord_mean', 'coord_std'

        Returns:
            grid_features: [batch_size, grid_size, grid_size, feature_dim]
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
        grid_features = self.attention(
            queries=query_embeddings,
            keys=point_features,
            values=point_features,
            query_positions=grid_centers,
            key_positions=point_pos_xy,
            batch_indices=batch_indices
        )  # [batch_size, num_queries=25, feature_dim]

        # Reshape to 2D grid
        grid_features = grid_features.view(batch_size, self.grid_size, self.grid_size, self.feature_dim)
        # [batch_size, 5, 5, feature_dim]

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


class RasterPredictionHead(nn.Module):
    """
    Complete raster prediction head: point features → regular grid → fuel metrics.

    Combines optional pre-aggregation LG-PAB blocks, PointToGridAggregator,
    and RasterDecoder into a single module for the multimodal raster model.

    Args:
        feature_dim: Feature dimension (must match point features)
        n_bands: Number of output fuel metrics bands (default 3)
        num_heads: Number of attention heads (default 8)
        radius: Distance threshold in meters (default 5.0)
        grid_size: Grid size per side (default 5)
        tile_extent: Tile extent in meters (default 10.0)
        hidden_dim: Decoder hidden dimension (default 128)
        num_decoder_layers: Number of decoder MLP layers (default 3)
        dropout: Dropout probability (default 0.1)
        num_pre_agg_blocks: Number of pre-aggregation LG-PAB blocks (default 2)
        pre_agg_lcl_heads: Local attention heads for pre-aggregation (default 4)
        pre_agg_glbl_heads: Global attention heads for pre-aggregation (default 4)
        pre_agg_dropout: Dropout for pre-aggregation blocks (default 0.1)
        pre_agg_k_neighbors: KNN neighbors for pre-aggregation (default 15)
        position_encoding_dim: Position encoding dimension (default 24)
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_bands: int = 3,
        num_heads: int = 8,
        radius: float = 5.0,
        grid_size: int = 5,
        tile_extent: float = 10.0,
        hidden_dim: int = 128,
        num_decoder_layers: int = 3,
        dropout: float = 0.1,
        num_pre_agg_blocks: int = 2,
        pre_agg_lcl_heads: int = 4,
        pre_agg_glbl_heads: int = 4,
        pre_agg_dropout: float = 0.1,
        pre_agg_k_neighbors: int = 15,
        position_encoding_dim: int = 24
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.num_pre_agg_blocks = num_pre_agg_blocks

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
                    k_neighbors=pre_agg_k_neighbors
                )
                for _ in range(num_pre_agg_blocks)
            ])
        else:
            self.pre_aggregation_blocks = None

        # Point-to-grid aggregation
        self.aggregator = PointToGridAggregator(
            feature_dim=feature_dim,
            num_heads=num_heads,
            radius=radius,
            grid_size=grid_size,
            tile_extent=tile_extent,
            dropout=dropout
        )

        # Raster decoder
        self.decoder = RasterDecoder(
            feature_dim=feature_dim,
            n_bands=n_bands,
            hidden_dim=hidden_dim,
            num_layers=num_decoder_layers,
            dropout=dropout
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