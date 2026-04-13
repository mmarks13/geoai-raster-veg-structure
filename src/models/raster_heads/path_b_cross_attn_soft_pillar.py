"""
Path B — `cross_attn_soft_pillar`.

Receives already-fused point features (from `CrossAttentionFusion`) and produces
a fuel-metrics raster via `SoftPillarConvDecoder`, which performs its own
bilinear soft-splatting + ConvNeXt refinement on the grid.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from ._soft_pillar import SoftPillarConvDecoder


class CrossAttnSoftPillarHead(nn.Module):
    """Path B head: thin wrapper around SoftPillarConvDecoder."""

    def __init__(
        self,
        feature_dim: int = 256,
        grid_size: int = 5,
        tile_extent_m: float = 10.0,
        n_bands: int = 8,
        decoder_dim: int = 128,
        num_blocks: int = 3,
        dropout: float = 0.10,
        output_variance: bool = False,
    ):
        super().__init__()
        self.decoder = SoftPillarConvDecoder(
            feature_dim=feature_dim,
            grid_size=grid_size,
            tile_extent=tile_extent_m,
            n_bands=n_bands,
            decoder_dim=decoder_dim,
            num_blocks=num_blocks,
            dropout=dropout,
            output_variance=output_variance,
        )

    def forward(
        self,
        point_features: torch.Tensor,
        point_positions: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict],
    ):
        return self.decoder(
            point_features=point_features,
            point_positions=point_positions,
            batch_indices=batch_indices,
            norm_params=norm_params,
        )
