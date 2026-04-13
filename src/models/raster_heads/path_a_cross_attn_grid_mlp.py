"""
Path A — `cross_attn_grid_mlp`.

Receives already-fused point features (from `CrossAttentionFusion`) and produces
a fuel-metrics raster via:

    LearnableGridQueries
        ↓ GaussianDistanceBiasedCrossAttention   (grid queries attend to points)
        ↓ PreLNFFN                               (pointwise capacity bump)
        ↓ SmallMlpDecoder

The aggregator uses a single unified positional-encoding extent so that points,
grid query centers, and image patches sit in a common normalized frame.
"""

from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from ._primitives import (
    DEFAULT_PE_EXTENT_M,
    GaussianDistanceBiasedCrossAttention,
    LearnableGridQueries,
    PreLNFFN,
    SmallMlpDecoder,
)


class CrossAttnGridMlpHead(nn.Module):
    """Path A head: cross-attention grid aggregation + Pre-LN FFN + small MLP decoder."""

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        distance_sigma: Union[float, Sequence[float]] = 2.0,
        dropout: float = 0.1,
        ffn_ratio: int = 2,
        grid_size: int = 5,
        tile_extent_m: float = 10.0,
        n_bands: int = 8,
        output_variance: bool = False,
        pe_extent_m: float = DEFAULT_PE_EXTENT_M,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.tile_extent_m = tile_extent_m

        self.grid_queries = LearnableGridQueries(
            feature_dim=feature_dim,
            grid_size=grid_size,
            tile_extent_m=tile_extent_m,
            pe_extent_m=pe_extent_m,
        )
        self.aggregator = GaussianDistanceBiasedCrossAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            distance_sigma=distance_sigma,
            dropout=dropout,
        )
        self.ffn = PreLNFFN(feature_dim=feature_dim, ffn_ratio=ffn_ratio, dropout=dropout)
        self.decoder = SmallMlpDecoder(
            feature_dim=feature_dim,
            n_bands=n_bands,
            dropout=dropout,
            output_variance=output_variance,
        )

    def forward(
        self,
        point_features: torch.Tensor,   # [N_total, F]
        point_positions: torch.Tensor,  # [N_total, 3] z-score normalized
        batch_indices: torch.Tensor,    # [N_total]
        norm_params: List[Dict],
    ):
        device = point_features.device
        dtype = point_features.dtype
        B = len(norm_params)

        # Denormalize point positions to meters for distance bias.
        coord_mean = torch.stack([p["coord_mean"] for p in norm_params]).to(device=device, dtype=point_positions.dtype)
        coord_std = torch.stack([p["coord_std"] for p in norm_params]).to(device=device, dtype=point_positions.dtype)
        pos_m = point_positions * coord_std[batch_indices] + coord_mean[batch_indices]  # [N_total, 3]
        pos_xy_m = pos_m[:, :2]  # [N_total, 2]

        # Pad to dense [B, max_N, *] and build padding mask (True = valid).
        feat_dense, key_padding_mask = to_dense_batch(point_features, batch_indices, batch_size=B)
        pos_dense, _ = to_dense_batch(pos_xy_m, batch_indices, batch_size=B)
        pos_dense = pos_dense.to(dtype)

        # Grid queries (shared across batch).
        queries, query_centers = self.grid_queries(B, device)  # [B, Q, F], [Q, 2]

        agg = self.aggregator(
            queries=queries,
            keys=feat_dense,
            values=feat_dense,
            query_positions=query_centers,
            key_positions=pos_dense,
            key_padding_mask=key_padding_mask,
        )  # [B, Q, F]

        agg = self.ffn(agg)

        H = W = self.grid_size
        grid_features = agg.view(B, H, W, self.feature_dim)
        return self.decoder(grid_features)
