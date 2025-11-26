"""
Multimodal raster prediction model for fuel metrics.

This module implements a raster-based decoder that predicts fuel hazard metrics
directly from sparse LiDAR + imagery. Shares the same encoder as the point cloud
upsampling model but uses a query-based grid aggregation decoder.

Architecture:
1. Feature Extraction: LocalGlobalPointAttentionBlock (shared with point cloud model)
2. Image Encoding: NAIPEncoder + UAVSAREncoder (shared)
3. Fusion: CrossAttentionFusion (shared, with denormalization support)
4. Raster Decoder: RasterPredictionHead (new, raster-specific)

Key differences from point cloud model:
- Uses z-score normalized coordinates (not just bbox-normalized)
- Predicts fuel metrics raster [n_bands, 5, 5] (not dense point cloud)
- No feature expansion/refinement (no upsampling)
- Requires norm_params for denormalization in distance computations
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional

# Import shared encoder components
from .encoders import NAIPEncoder, UAVSAREncoder
from .multimodal_model import LocalGlobalPointAttentionBlock
from .cross_attn_fusion import CrossAttentionFusion
from .raster_head import RasterPredictionHead


@dataclass
class MultimodalRasterConfig:
    """
    Configuration for multimodal raster prediction model.

    Inherits most parameters from point cloud model config, but adds
    raster-specific parameters and removes upsampling-related ones.
    """
    # Core model parameters
    k: int = 15
    feature_dim: int = 256
    pos_mlp_hdn: int = 16

    # Point Transformer parameters
    pt_attn_dropout: float = 0.05

    # Feature extractor attention heads
    extractor_lcl_heads: int = 4
    extractor_glbl_heads: int = 4

    # Attribute dimension (intensity, return number, number of returns)
    attr_dim: int = 3

    # Modality flags
    use_naip: bool = False
    use_uavsar: bool = False

    # Imagery encoder parameters
    img_embed_dim: int = 128
    img_num_patches: int = 16

    # Fusion parameters (cross-attention only for raster model)
    fusion_type: str = "cross_attention"
    max_dist_ratio: float = 5.0  # Maximum distance in METERS for cross-attention masking (note: parameter name is misleading)
    fusion_num_heads: int = 4
    fusion_dropout: float = 0.1
    position_encoding_dim: int = 24  # Must be divisible by 6 for 3D positions (2 * D_pos = 2 * 3 = 6)

    # Encoder dropouts
    naip_dropout: float = 0.1
    uavsar_dropout: float = 0.1
    temporal_encoder: str = "gru"

    # Raster-specific parameters
    n_bands: int = 3  # Number of fuel metrics bands to predict (default: Height, TFL, Total_cover)
    target_band_indices: List[int] = None  # Indices of target bands (set in __post_init__)
    grid_size: int = 5  # Grid size per side (5×5 grid)
    tile_extent: float = 10.0  # Tile extent in meters
    raster_num_heads: int = 8  # Number of attention heads in raster aggregator
    raster_radius: float = 5.0  # Distance threshold in meters for grid query attention (matches cross_attention max_dist_ratio)
    raster_hidden_dim: int = 128  # Hidden dimension in raster decoder
    raster_decoder_layers: int = 3  # Number of MLP layers in raster decoder (tunable: 3/4/5)
    raster_dropout: float = 0.1  # Dropout in raster decoder

    # Pre-aggregation refinement parameters
    num_pre_agg_blocks: int = 2  # Number of pre-aggregation LG-PAB blocks (0-5 configurable)
    pre_agg_lcl_heads: int = 4  # Local attention heads for pre-aggregation blocks
    pre_agg_glbl_heads: int = 4  # Global attention heads for pre-aggregation blocks
    pre_agg_dropout: float = 0.1  # Dropout for pre-aggregation blocks
    pre_agg_k_neighbors: int = 15  # KNN neighbors for pre-aggregation blocks

    # Checkpoint loading parameters
    checkpoint_path: str = None
    layers_to_load: list = None
    layers_to_freeze: list = None

    def __post_init__(self):
        """Set default target_band_indices if not provided."""
        if self.target_band_indices is None:
            self.target_band_indices = [2, 7, 14]  # Default: Height, TFL, Total_cover

    def __reduce__(self):
        """Custom reduce method for multiprocessing compatibility."""
        return (
            self.__class__,
            (
                self.k,
                self.feature_dim,
                self.pos_mlp_hdn,
                self.pt_attn_dropout,
                self.extractor_lcl_heads,
                self.extractor_glbl_heads,
                self.attr_dim,
                self.use_naip,
                self.use_uavsar,
                self.img_embed_dim,
                self.img_num_patches,
                self.fusion_type,
                self.max_dist_ratio,
                self.fusion_num_heads,
                self.fusion_dropout,
                self.position_encoding_dim,
                self.naip_dropout,
                self.uavsar_dropout,
                self.temporal_encoder,
                self.n_bands,
                self.target_band_indices,
                self.grid_size,
                self.tile_extent,
                self.raster_num_heads,
                self.raster_radius,
                self.raster_hidden_dim,
                self.raster_decoder_layers,
                self.raster_dropout,
                self.num_pre_agg_blocks,
                self.pre_agg_lcl_heads,
                self.pre_agg_glbl_heads,
                self.pre_agg_dropout,
                self.pre_agg_k_neighbors,
                self.checkpoint_path,
                self.layers_to_load,
                self.layers_to_freeze,
            )
        )


class MultimodalRasterPredictor(nn.Module):
    """
    Multimodal raster prediction model for fuel metrics.

    Combines sparse LiDAR point clouds with optical (NAIP) and SAR (UAVSAR)
    imagery to predict fuel hazard metrics rasters directly.

    Architecture:
    1. Feature Extraction: LocalGlobalPointAttentionBlock on point cloud
    2. Image Encoding: ViT-based encoders with temporal aggregation
    3. Fusion: Cross-attention between point features and image patch embeddings
    4. Raster Prediction: Query-based grid aggregation + 1×1 Conv decoder

    Args:
        config: MultimodalRasterConfig with model hyperparameters
    """

    def __init__(self, config: MultimodalRasterConfig):
        """Initialize the multimodal raster predictor."""
        super().__init__()
        self.config = config

        # Track which modalities are being used
        self.use_naip = config.use_naip
        self.use_uavsar = config.use_uavsar

        # Get extractor dropout
        extractor_dropout = getattr(config, 'extractor_dropout', config.pt_attn_dropout)

        # Get position generation hidden dimension
        pos_gen_hidden_dim = getattr(config, 'pos_gen_hidden_dim', 64)

        # ====== 1) Feature Extractor (shared with point cloud model) ======
        self.feature_extractor = LocalGlobalPointAttentionBlock(
            in_channels=6,  # 3 attributes + 3 coordinates
            out_channels=config.feature_dim,
            num_lcl_heads=config.extractor_lcl_heads,
            num_glbl_heads=config.extractor_glbl_heads,
            pos_encoding_dim=config.position_encoding_dim,
            dropout=extractor_dropout,
            k_neighbors=config.k
        )

        # ====== 2) Imagery Encoders (shared with point cloud model) ======
        if self.use_naip:
            self.naip_encoder = NAIPEncoder(
                in_channels=4,  # RGB + NIR
                image_size=40,  # 40×40 pixels
                patch_size=10,  # 10×10 pixel patches
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.naip_dropout,
                temporal_encoder_type=config.temporal_encoder
            )

        if self.use_uavsar:
            self.uavsar_encoder = UAVSAREncoder(
                in_channels=6,  # 6 polarization bands
                image_size=4,   # 4×4 pixels
                patch_size=1,   # 1×1 pixel patches
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.uavsar_dropout,
                temporal_encoder_type=config.temporal_encoder
            )

        # ====== 3) Fusion Module (shared, with norm_params support) ======
        self.fusion = CrossAttentionFusion(
            point_dim=config.feature_dim,
            patch_dim=config.img_embed_dim,
            use_naip=self.use_naip,
            use_uavsar=self.use_uavsar,
            num_patches=config.img_num_patches,
            max_dist_ratio=config.max_dist_ratio,
            num_heads=config.fusion_num_heads,
            attention_dropout=config.fusion_dropout,
            position_encoding_dim=config.position_encoding_dim
        )

        # ====== 4) Raster Prediction Head (raster-specific) ======
        self.raster_head = RasterPredictionHead(
            feature_dim=config.feature_dim,
            n_bands=config.n_bands,
            num_heads=config.raster_num_heads,
            radius=config.raster_radius,
            grid_size=config.grid_size,
            tile_extent=config.tile_extent,
            hidden_dim=config.raster_hidden_dim,
            num_decoder_layers=config.raster_decoder_layers,
            dropout=config.raster_dropout,
            num_pre_agg_blocks=config.num_pre_agg_blocks,
            pre_agg_lcl_heads=config.pre_agg_lcl_heads,
            pre_agg_glbl_heads=config.pre_agg_glbl_heads,
            pre_agg_dropout=config.pre_agg_dropout,
            pre_agg_k_neighbors=config.pre_agg_k_neighbors,
            position_encoding_dim=config.position_encoding_dim
        )

    def forward(
        self,
        dep_points: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict],
        dep_attr: Optional[torch.Tensor] = None,
        naip: Optional[List[Dict]] = None,
        uavsar: Optional[List[Dict]] = None,
        bbox: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of the multimodal raster predictor.

        Args:
            dep_points: Concatenated 3DEP point coordinates [N_total, 3] (Z-SCORE NORMALIZED)
            edge_index: Edge indices for graph connectivity [2, E_total]
            batch_indices: Batch assignment for each point [N_total]
            norm_params: List of dicts (length batch_size) with 'coord_mean', 'coord_std', etc.
            dep_attr: 3DEP point attributes [N_total, attr_dim] (normalized)
            naip: List of dicts (length batch_size) with NAIP imagery data or None
                Each dict contains:
                - 'images': NAIP images [n_images, 4, 40, 40]
                - 'img_bbox': Bounding box for spatial alignment
                - 'relative_dates': Temporal information
            uavsar: List of dicts (length batch_size) with UAVSAR imagery data or None
                Each dict contains:
                - 'images': UAVSAR images [n_images, 6, 4, 4]
                - 'img_bbox': Bounding box for spatial alignment
                - 'attention_mask': Mask for invalid data
                - 'relative_dates': Temporal information
            bbox: Bounding boxes [batch_size, 4] - currently not used but kept for compatibility

        Returns:
            pred_raster: Predicted fuel metrics [batch_size, n_bands, 5, 5] (Z-SCORE NORMALIZED)
        """
        batch_size = len(norm_params)
        device = dep_points.device

        # Clamp extreme Z values (bird returns)
        dep_points[:, 2] = torch.clamp(dep_points[:, 2], -10, 10)  # In z-score space, ±10 is ~70m from mean

        # Concatenate attributes and positions
        dep_points_and_attr = torch.cat([dep_attr, dep_points], dim=1)  # [N_total, 6]

        # ====== 1) Point Cloud Feature Extraction ======
        x_feat, _ = self.feature_extractor(dep_points_and_attr, dep_points, edge_index)
        # x_feat: [N_total, feature_dim]

        # ====== 2) Imagery Feature Extraction ======
        # Process each tile separately since imagery data is list of dicts
        naip_embeddings_list = []
        uavsar_embeddings_list = []

        for b in range(batch_size):
            # NAIP encoding for tile b
            if self.use_naip and naip is not None and naip[b] is not None:
                naip_b = naip[b]
                if 'images' in naip_b and naip_b['images'] is not None:
                    # Convert to float32 if needed (preprocessed data may be float16)
                    naip_images = naip_b['images'].to(device).float()
                    rel_dates = naip_b.get('relative_dates', None)
                    if rel_dates is not None:
                        rel_dates = rel_dates.to(device)

                    naip_emb = self.naip_encoder(
                        naip_images,
                        naip_b.get('img_bbox', None),
                        rel_dates
                    )  # [num_patches, img_embed_dim]
                    naip_embeddings_list.append(naip_emb)
                else:
                    naip_embeddings_list.append(None)
            else:
                naip_embeddings_list.append(None)

            # UAVSAR encoding for tile b
            if self.use_uavsar and uavsar is not None and uavsar[b] is not None:
                uavsar_b = uavsar[b]
                if 'images' in uavsar_b and uavsar_b['images'] is not None:
                    # Convert to float32 if needed (preprocessed data may be float16)
                    uavsar_images = uavsar_b['images'].to(device).float()
                    mask = uavsar_b.get('attention_mask', None)
                    if mask is not None:
                        mask = mask.to(device)
                    rel_dates = uavsar_b.get('relative_dates', None)
                    if rel_dates is not None:
                        rel_dates = rel_dates.to(device)

                    uavsar_emb = self.uavsar_encoder(
                        uavsar_images,
                        attention_mask=mask,
                        img_bbox=uavsar_b.get('img_bbox', None),
                        relative_dates=rel_dates
                    )  # [num_patches, img_embed_dim]
                    uavsar_embeddings_list.append(uavsar_emb)
                else:
                    uavsar_embeddings_list.append(None)
            else:
                uavsar_embeddings_list.append(None)

        # ====== 3) Fusion (per-tile with denormalization) ======
        # Process each tile separately since fusion needs per-tile norm_params and imagery data
        fused_features = []

        for b in range(batch_size):
            # Get points and features for this tile
            mask_b = (batch_indices == b)
            dep_points_b = dep_points[mask_b]  # [N_b, 3]
            x_feat_b = x_feat[mask_b]  # [N_b, feature_dim]

            # Get imagery embeddings for this tile
            naip_emb_b = naip_embeddings_list[b] if len(naip_embeddings_list) > 0 else None
            uavsar_emb_b = uavsar_embeddings_list[b] if len(uavsar_embeddings_list) > 0 else None

            # Get imagery bboxes
            naip_bbox_b = None
            if naip is not None and naip[b] is not None:
                naip_bbox_b = naip[b].get('img_bbox', None)

            uavsar_bbox_b = None
            if uavsar is not None and uavsar[b] is not None:
                uavsar_bbox_b = uavsar[b].get('img_bbox', None)

            # Apply fusion with norm_params for denormalization
            # Note: edge_index not used in CrossAttentionFusion, can pass None
            x_fused_b = self.fusion(
                point_features=x_feat_b,
                edge_index=None,
                point_positions=dep_points_b,
                naip_embeddings=naip_emb_b,
                uavsar_embeddings=uavsar_emb_b,
                main_bbox=None,  # Not used in cross-attention fusion
                naip_bbox=naip_bbox_b,
                uavsar_bbox=uavsar_bbox_b,
                center=None,
                scale=None,
                norm_params=norm_params[b]  # Pass norm_params for denormalization
            )  # [N_b, feature_dim]

            fused_features.append(x_fused_b)

        # Concatenate fused features back to full batch
        x_fused = torch.cat(fused_features, dim=0)  # [N_total, feature_dim]

        # ====== 4) Raster Prediction ======
        pred_raster = self.raster_head(
            point_features=x_fused,
            point_positions=dep_points,
            batch_indices=batch_indices,
            norm_params=norm_params
        )  # [batch_size, n_bands, 5, 5]

        return pred_raster
