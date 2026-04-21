"""
Grid cross-attention raster head (reference-only, not used by production).

25 learnable grid queries iteratively refine themselves by attending to a
concatenated set of keys: [points, NAIP patches, UAVSAR patches]. Each block is
Pre-LN ``cross-attn → self-attn → FFN`` (the "flipped DETR" ordering, which is
semantically correct for position-anchored grid queries).

Unlike the production raster head, this variant fuses imagery directly inside
the head rather than consuming pre-fused features from `CrossAttentionFusion`.

The per-tile modality presence masks are OR-ed into the cross-attention
key-padding mask so that zero-filled NAIP/UAVSAR patches for tiles missing those
modalities are excluded from attention.
"""

from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch_geometric.utils import to_dense_batch

from src.models.raster_primitives import (
    DEFAULT_PE_EXTENT_M,
    GaussianDistanceBiasedCrossAttention,
    LearnableGridQueries,
    PatchPositionEncoding,
    SmallMlpDecoder,
    build_point_position_encoding,
)


class GridCrossAttnBlock(nn.Module):
    """Pre-LN transformer block: cross-attn → self-attn → FFN with residuals.

    Specific to Path C — not used by any other path.
    """

    def __init__(
        self,
        feature_dim: int,
        num_heads: int,
        distance_sigma: Union[float, Sequence[float]],
        dropout: float,
        use_self_attn: bool = True,
        ffn_ratio: Optional[int] = 2,
    ):
        super().__init__()
        self.norm_cross = nn.LayerNorm(feature_dim)
        self.cross_attn = GaussianDistanceBiasedCrossAttention(
            feature_dim=feature_dim,
            num_heads=num_heads,
            distance_sigma=distance_sigma,
            dropout=dropout,
        )

        self.use_self_attn = use_self_attn
        if self.use_self_attn:
            self.norm_self = nn.LayerNorm(feature_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=feature_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.use_ffn = ffn_ratio is not None
        if self.use_ffn:
            self.norm_ffn = nn.LayerNorm(feature_dim)
            self.ffn = nn.Sequential(
                nn.Linear(feature_dim, feature_dim * ffn_ratio),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(feature_dim * ffn_ratio, feature_dim),
            )

    def forward(
        self,
        queries: torch.Tensor,           # [B, Q, F]
        query_positions: torch.Tensor,   # [Q, 2] or [B, Q, 2]
        keys: torch.Tensor,              # [B, K, F]
        key_positions: torch.Tensor,     # [B, K, 2]
        key_padding_mask: torch.Tensor,  # [B, K], True = valid
    ) -> torch.Tensor:
        # Pre-LN cross-attention
        q_norm = self.norm_cross(queries)
        attn_out = self.cross_attn(
            queries=q_norm,
            keys=keys,
            values=keys,
            query_positions=query_positions,
            key_positions=key_positions,
            key_padding_mask=key_padding_mask,
        )
        queries = queries + attn_out

        if self.use_self_attn:
            # Optional query-query mixing across the fixed 5x5 grid.
            q_norm = self.norm_self(queries)
            sa_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
            queries = queries + sa_out

        if self.use_ffn:
            queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class GridCrossAttnHead(nn.Module):
    """Path C head: stacked Pre-LN cross-attn blocks over a multi-source key stack."""

    def __init__(
        self,
        feature_dim: int = 256,
        depth: int = 2,
        num_heads: int = 8,
        distance_sigma: Union[float, Sequence[float]] = 2.0,
        dropout: float = 0.1,
        use_self_attn: bool = True,
        ffn_ratio: Optional[int] = 2,
        grid_size: int = 5,
        tile_extent_m: float = 10.0,
        n_bands: int = 8,
        output_variance: bool = False,
        use_naip: bool = False,
        use_uavsar: bool = False,
        patch_dim: int = 128,
        patch_grid_size: int = 4,
        patch_extent_m: float = 20.0,
        pe_extent_m: float = DEFAULT_PE_EXTENT_M,
        use_spectral_norm: bool = False,
    ):
        super().__init__()
        if not (1 <= depth <= 4):
            raise ValueError(f"GridCrossAttnHead: depth must be in [1, 4], got {depth}")

        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.tile_extent_m = tile_extent_m
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar
        self.pe_extent_m = pe_extent_m

        self.grid_queries = LearnableGridQueries(
            feature_dim=feature_dim,
            grid_size=grid_size,
            tile_extent_m=tile_extent_m,
            pe_extent_m=pe_extent_m,
        )

        if use_naip:
            self.naip_proj = nn.Linear(patch_dim, feature_dim)
            self.naip_pe = PatchPositionEncoding(
                feature_dim=feature_dim,
                patch_grid_size=patch_grid_size,
                patch_extent_m=patch_extent_m,
                pe_extent_m=pe_extent_m,
            )
        if use_uavsar:
            self.uavsar_proj = nn.Linear(patch_dim, feature_dim)
            self.uavsar_pe = PatchPositionEncoding(
                feature_dim=feature_dim,
                patch_grid_size=patch_grid_size,
                patch_extent_m=patch_extent_m,
                pe_extent_m=pe_extent_m,
            )

        self.blocks = nn.ModuleList([
            GridCrossAttnBlock(
                feature_dim=feature_dim,
                num_heads=num_heads,
                distance_sigma=distance_sigma,
                dropout=dropout,
                use_self_attn=use_self_attn,
                ffn_ratio=ffn_ratio,
            )
            for _ in range(depth)
        ])

        self.decoder = SmallMlpDecoder(
            feature_dim=feature_dim,
            n_bands=n_bands,
            dropout=dropout,
            output_variance=output_variance,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(
        self,
        point_features: torch.Tensor,             # [N_total, F]
        point_positions: torch.Tensor,            # [N_total, 3] z-score normalized
        batch_indices: torch.Tensor,              # [N_total]
        norm_params: List[Dict],
        naip_stacked: Optional[torch.Tensor] = None,    # [B, P, patch_dim]
        uavsar_stacked: Optional[torch.Tensor] = None,  # [B, P, patch_dim]
        naip_present_mask: Optional[torch.Tensor] = None,    # [B] bool
        uavsar_present_mask: Optional[torch.Tensor] = None,  # [B] bool
    ):
        device = point_features.device
        dtype = point_features.dtype
        B = len(norm_params)

        # ---- 1) Denormalize point xy to meters ----
        coord_mean = torch.stack([p["coord_mean"] for p in norm_params]).to(device=device, dtype=point_positions.dtype)
        coord_std = torch.stack([p["coord_std"] for p in norm_params]).to(device=device, dtype=point_positions.dtype)
        pos_m = point_positions * coord_std[batch_indices] + coord_mean[batch_indices]
        pos_xy_m = pos_m[:, :2]

        # ---- 2) Pad points to dense; add point positional encoding ----
        pt_feat_dense, pt_mask = to_dense_batch(point_features, batch_indices, batch_size=B)
        pt_pos_dense, _ = to_dense_batch(pos_xy_m, batch_indices, batch_size=B)
        pt_pos_dense = pt_pos_dense.to(dtype)
        pt_pe = build_point_position_encoding(pt_pos_dense, self.feature_dim, self.pe_extent_m)
        pt_feat_dense = pt_feat_dense + pt_pe

        keys_list = [pt_feat_dense]
        positions_list = [pt_pos_dense]
        masks_list = [pt_mask]

        # ---- 3) NAIP patches (projected, with patch PE, modality-masked) ----
        if self.use_naip and naip_stacked is not None:
            naip_proj = self.naip_proj(naip_stacked)  # [B, P, F]
            patch_pe, patch_centers = self.naip_pe()  # [P, F], [P, 2]
            naip_proj = naip_proj + patch_pe.unsqueeze(0).to(naip_proj.dtype)
            P = patch_centers.shape[0]
            patch_pos = patch_centers.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)

            if naip_present_mask is not None:
                naip_key_mask = naip_present_mask.to(device=device).unsqueeze(1).expand(-1, P)
            else:
                naip_key_mask = torch.ones(B, P, dtype=torch.bool, device=device)

            keys_list.append(naip_proj)
            positions_list.append(patch_pos)
            masks_list.append(naip_key_mask)

        # ---- 4) UAVSAR patches (projected, with patch PE, modality-masked) ----
        if self.use_uavsar and uavsar_stacked is not None:
            uavsar_proj = self.uavsar_proj(uavsar_stacked)  # [B, P, F]
            patch_pe, patch_centers = self.uavsar_pe()
            uavsar_proj = uavsar_proj + patch_pe.unsqueeze(0).to(uavsar_proj.dtype)
            P = patch_centers.shape[0]
            patch_pos = patch_centers.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)

            if uavsar_present_mask is not None:
                uavsar_key_mask = uavsar_present_mask.to(device=device).unsqueeze(1).expand(-1, P)
            else:
                uavsar_key_mask = torch.ones(B, P, dtype=torch.bool, device=device)

            keys_list.append(uavsar_proj)
            positions_list.append(patch_pos)
            masks_list.append(uavsar_key_mask)

        keys = torch.cat(keys_list, dim=1)            # [B, K, F]
        key_positions = torch.cat(positions_list, dim=1)  # [B, K, 2]
        key_padding_mask = torch.cat(masks_list, dim=1)   # [B, K] bool

        # ---- 5) Grid queries + stacked transformer blocks ----
        queries, query_centers = self.grid_queries(B, device)  # [B, Q, F], [Q, 2]
        for block in self.blocks:
            queries = block(
                queries=queries,
                query_positions=query_centers,
                keys=keys,
                key_positions=key_positions,
                key_padding_mask=key_padding_mask,
            )

        # ---- 6) Reshape and decode ----
        H = W = self.grid_size
        grid_features = queries.view(B, H, W, self.feature_dim)
        return self.decoder(grid_features)
