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
import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Union
import warnings


# Field order in the OLD positional `__reduce__` (pre-refactor). Used to remap
# legacy checkpoint pickles into kwargs at unpickle time. DO NOT REORDER.
_LEGACY_REDUCE_FIELDS = (
    "k", "feature_dim", "pos_mlp_hdn", "pt_attn_dropout",
    "extractor_lcl_heads", "extractor_glbl_heads", "use_global_only",
    "use_v2_attention", "attr_dim", "use_naip", "use_uavsar",
    "img_embed_dim", "img_num_patches", "fusion_type", "max_dist_ratio",
    "fusion_num_heads", "fusion_dropout", "position_encoding_dim",
    "use_batched_fusion", "grid_fusion_num_heads", "grid_fusion_distance_sigma",
    "grid_fusion_dropout", "naip_dropout", "uavsar_dropout", "temporal_encoder",
    "pos_encoder_dropout", "stochastic_pos_dropout_prob", "n_bands",
    "target_band_indices", "grid_size", "tile_extent", "raster_num_heads",
    "raster_distance_sigma", "raster_hidden_dim", "raster_decoder_layers",
    "raster_attention_dropout", "raster_decoder_dropout",
    "raster_use_wide_decoder", "num_pre_agg_blocks", "pre_agg_lcl_heads",
    "pre_agg_glbl_heads", "pre_agg_dropout", "pre_agg_k_neighbors",
    "checkpoint_path", "layers_to_load", "layers_to_freeze",
    "correlation_loss_weight", "huber_delta", "transfer_lr_multiplier",
    "use_heteroscedastic_loss", "heteroscedastic_min_var",
    "heteroscedastic_overconfidence_weight", "encoder_drop_path",
    "decoder_drop_path", "extractor_point_attn_drop_path",
    "pre_agg_point_attn_drop_path", "use_spectral_norm", "swa_enabled",
    "swa_start_epoch", "swa_update_freq", "training_augmentation_enabled",
    "aug_coord_jitter_sigma_xy", "aug_coord_jitter_sigma_z",
    "aug_coord_jitter_prob", "aug_intensity_noise_sigma",
    "aug_intensity_noise_prob", "aug_intensity_outlier_prob",
    "aug_bird_outlier_prob", "aug_bird_z_offset_range",
    "aug_point_dup_tile_prob", "aug_point_dup_min_point_prob",
    "aug_point_dup_max_point_prob", "aug_point_dup_min_offset",
    "aug_point_dup_max_offset", "aug_omni_outlier_tile_prob",
    "aug_omni_outlier_point_prob", "aug_omni_outlier_min_magnitude",
    "aug_omni_outlier_max_magnitude", "aug_temporal_shift_prob",
    "aug_temporal_max_shift_days", "coordinate_normalization_stats_path",
    "aug_return_scale_prob", "aug_return_scale_range", "aug_return_noise_prob",
    "aug_return_noise_sigma", "aug_return_zero_prob", "aug_return_shuffle_prob",
    "aug_naip_noise_sigma", "aug_naip_noise_prob", "aug_naip_blur_kernel",
    "aug_naip_blur_sigma", "aug_naip_blur_prob", "aug_naip_motion_blur_kernel",
    "aug_naip_motion_blur_angle", "aug_naip_motion_blur_prob",
    "aug_naip_erasing_scale", "aug_naip_erasing_prob",
    "aug_naip_sharpness_range", "aug_naip_sharpness_prob",
    "aug_naip_equalize_prob", "aug_uavsar_noise_sigma", "aug_uavsar_noise_prob",
    "aug_uavsar_blur_kernel", "aug_uavsar_blur_sigma", "aug_uavsar_blur_prob",
    "aug_uavsar_motion_blur_kernel", "aug_uavsar_motion_blur_angle",
    "aug_uavsar_motion_blur_prob", "aug_uavsar_erasing_scale",
    "aug_uavsar_erasing_prob", "aug_geometric_enabled", "aug_rotation_prob",
    "aug_reflection_prob", "aug_temporal_enabled", "aug_naip_subsample_prob",
    "aug_naip_min_frames", "aug_uavsar_t_subsample_prob",
    "aug_uavsar_t_min_frames", "aug_uavsar_g_mask_prob",
    "aug_uavsar_g_min_images", "aug_modality_dropout_enabled",
    "aug_naip_dropout_prob", "aug_uavsar_dropout_prob",
    "aug_point_removal_enabled", "aug_point_removal_prob",
    "aug_point_min_removal_ratio", "aug_point_max_removal_ratio",
    "aug_point_min_points", "ood_val_enabled", "ood_val_tiles_path",
    "ood_val_metadata_path", "ood_val_every_n_epochs",
    "ood_val_band_config_path",
)


def _unpickle_raster_config(state: dict):
    """Stable, kwargs-based unpickler for `MultimodalRasterConfig`.

    Filters out fields no longer present on the current class so configs
    pickled by future versions can still load on older versions, and vice
    versa. Used by the new `__reduce__`.
    """
    valid = {f.name for f in dataclasses.fields(MultimodalRasterConfig)}
    return MultimodalRasterConfig(**{k: v for k, v in state.items() if k in valid})

# Import shared encoder components
from .encoders import NAIPEncoder, UAVSAREncoder
from .multimodal_model import LocalGlobalPointAttentionBlock
from .cross_attn_fusion import CrossAttentionFusion
from .raster_head import RasterPredictionHead
from .raster_heads import (
    CrossAttnGridMlpHead,
    CrossAttnSoftPillarHead,
    GridCrossAttnHead,
)
from .training_augmentation import TrainingAugmentation


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

    # Global-only attention mode for online point removal augmentation
    # When True, uses two consecutive global-only attention blocks (no local KNN)
    # This enables online point removal since global attention doesn't need pre-computed KNN edges
    use_global_only: bool = False

    # V2 attention: decoupled Q/K/V projections (Q/K are position-aware, V is pure semantics)
    # When True, uses PosAwareGlobalFlashAttentionV2 instead of V1
    use_v2_attention: bool = True  # Default True for raster model

    # Attribute dimension:
    # [0] Intensity, [1] ReturnNumber, [2] NumberOfReturns,
    # [3] Planarity, [4] Sphericity, [5] Verticality
    attr_dim: int = 6

    # Modality flags
    use_naip: bool = False
    use_uavsar: bool = False

    # Imagery encoder parameters
    img_embed_dim: int = 128
    img_num_patches: int = 16

    # ===== Architecture selector =====
    # Single authoritative selector for the raster prediction head. One of:
    #   - "cross_attn_grid_mlp"    (Path A, default — CrossAttentionFusion → grid agg → FFN → MLP)
    #   - "cross_attn_soft_pillar" (Path B          — CrossAttentionFusion → SoftPillarConvDecoder)
    #   - "grid_cross_attn"        (Path C          — 25 grid queries × stacked Pre-LN cross-attn blocks)
    raster_architecture: Literal[
        "cross_attn_grid_mlp",
        "cross_attn_soft_pillar",
        "grid_cross_attn",
    ] = "cross_attn_grid_mlp"

    # Fusion parameters
    # DEPRECATED: use `raster_architecture` instead. Retained so legacy checkpoint
    # configs and ad-hoc scripts that instantiate legacy classes still construct.
    fusion_type: str = "cross_attention"
    max_dist_ratio: float = 5.0  # Maximum distance in METERS for cross-attention masking (note: parameter name is misleading)
    fusion_num_heads: int = 4
    fusion_dropout: float = 0.1
    position_encoding_dim: int = 48  # Must be divisible by 6 for 3D positions (2 * D_pos = 2 * 3 = 6)
    use_batched_fusion: bool = True  # Use batched fusion (2x faster) vs original per-tile loop

    # DEPRECATED: legacy GridCrossAttention parameters (folded into `grid_cross_attn_*`).
    # Retained so legacy `GridCrossAttentionFusion` instances still construct.
    grid_fusion_num_heads: int = 8  # 8 heads → 32 dim per head (with feature_dim=256)
    grid_fusion_distance_sigma: Union[float, List[float]] = field(
        default_factory=lambda: [1.0, 1.5, 2.5, 3.0, 3.5, 4.0, 6.0, 8.0]
    )  # 2 local (point-dominant), 4 balanced, 2 wide (patch-inclusive)
    grid_fusion_dropout: float = 0.1

    # ===== Path C (grid_cross_attn) parameters =====
    # Path C reuses raster_num_heads / raster_distance_sigma / raster_attention_dropout
    # for its attention configuration. Only the unique fields live here.
    grid_cross_attn_depth: int = 2  # Stacked transformer blocks (1..4)
    grid_cross_attn_ffn_ratio: int = 2

    # Encoder dropouts
    naip_dropout: float = 0.1
    uavsar_dropout: float = 0.1
    temporal_encoder: str = "gru"

    # Position encoder regularization
    pos_encoder_dropout: float = 0.1
    stochastic_pos_dropout_prob: float = 0.0

    # Raster-specific parameters
    n_bands: int = 3  # Number of fuel metrics bands to predict (default: Height, TFL, Total_cover)
    target_band_indices: List[int] = None  # Indices of target bands (set in __post_init__)
    grid_size: int = 5  # Grid size per side (5×5 grid)
    tile_extent: float = 10.0  # Tile extent in meters
    raster_num_heads: int = 8  # Number of attention heads in raster aggregator
    # RASTER MODEL: Soft Gaussian distance weighting (σ) for grid query attention.
    # Replaces hard radius cutoff to handle sparse tiles without NaN.
    # σ=2.0m matches grid cell size: weight at 0m=1.0, 2m=0.61, 3m=0.32, 5m=0.04
    # Can be a single float (same for all heads) or list of floats (per-head multi-scale)
    # Multi-scale example: [0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0] for 8 heads
    raster_distance_sigma: Union[float, List[float]] = 2.0
    raster_hidden_dim: int = 128  # Hidden dimension in raster decoder
    raster_decoder_layers: int = 3  # Number of MLP layers in raster decoder (tunable: 2/3)
    # Split dropout: separate values for attention (preserve sparse signal) and decoder MLP (regularize)
    raster_attention_dropout: float = 0.1  # Dropout for grid aggregation attention (keep low)
    raster_decoder_dropout: float = 0.1  # Dropout for decoder MLP (can be higher)
    # DEPRECATED: folded into `raster_architecture`. Retained for legacy class construction.
    raster_use_wide_decoder: bool = False  # Use wide decoder with Pre-LN residuals (256→256→256→n_bands)

    # DEPRECATED: pre-aggregation LG-PAB blocks are not used by any new path.
    # Retained so legacy `RasterPredictionHead` still constructs.
    num_pre_agg_blocks: int = 2  # Number of pre-aggregation LG-PAB blocks (0-5 configurable)
    pre_agg_lcl_heads: int = 4  # Local attention heads for pre-aggregation blocks
    pre_agg_glbl_heads: int = 4  # Global attention heads for pre-aggregation blocks
    pre_agg_dropout: float = 0.1  # Dropout for pre-aggregation blocks
    pre_agg_k_neighbors: int = 15  # KNN neighbors for pre-aggregation blocks

    # Checkpoint loading parameters
    checkpoint_path: str = None
    layers_to_load: list = None
    layers_to_freeze: list = None

    # Correlation loss weight (addresses variance collapse)
    # Total loss = MSE + correlation_loss_weight * (1 - pearson_r)
    # Set to 0.0 to disable, typical values: 0.1-0.5
    correlation_loss_weight: float = 0.0

    # Huber loss delta threshold (for robust loss)
    # Errors > delta use linear penalty instead of quadratic
    huber_delta: float = 1.0

    # Transfer learning rate multiplier (for fine-tuning pretrained layers)
    # Transferred layers get lr * transfer_lr_multiplier
    transfer_lr_multiplier: float = 0.1

    # ===== Heteroscedastic (Gaussian NLL) Loss =====
    # When enabled, model outputs (mean, log_variance) per pixel
    # Loss = 0.5 * log(σ²) + (y - μ)² / (2σ²)
    # High variance = "I don't know" (e.g., canopy-blocked pixels)
    use_heteroscedastic_loss: bool = False
    heteroscedastic_min_var: float = 1e-6  # Minimum variance for numerical stability
    heteroscedastic_overconfidence_weight: float = 0.0  # Penalty for σ² < 1 (prevents overconfidence)

    # Stochastic depth (DropPath) - separate configs for different components
    encoder_drop_path: float = 0.0  # Drop path for image encoder TransformerBlocks (NAIPEncoder/UAVSAREncoder)
    decoder_drop_path: float = 0.0  # Drop path for WideRasterDecoder residual blocks
    extractor_point_attn_drop_path: float = 0.0  # Drop path for feature extractor's PosAwareGlobalFlashAttention
    pre_agg_point_attn_drop_path: float = 0.0  # Drop path for pre-aggregation blocks' PosAwareGlobalFlashAttention

    # Spectral normalization (Lipschitz constraint for OOD robustness)
    use_spectral_norm: bool = False  # Apply spectral norm to PreLNResidualBlock and DistanceMaskedAttention.out_proj

    # Stochastic Weight Averaging (SWA) for OOD robustness
    swa_enabled: bool = False  # Enable SWA model averaging
    swa_start_epoch: int = 50  # Epoch to start averaging (typically 50-75% into training)
    swa_update_freq: int = 1   # Update average every N epochs (1 = every epoch)

    # ===== GPU Training Augmentation (raster model only) =====
    # Master switch for GPU-native augmentation (Kornia + custom PyTorch ops)
    # See docs/training_augmentation.md for full documentation
    training_augmentation_enabled: bool = False

    # Point cloud augmentation
    aug_coord_jitter_sigma_xy: float = 0.03  # Separate sigma for x,y
    aug_coord_jitter_sigma_z: float = 0.01   # Separate sigma for z
    aug_coord_jitter_prob: float = 0.5
    aug_intensity_noise_sigma: float = 0.05
    aug_intensity_noise_prob: float = 0.3
    aug_intensity_outlier_prob: float = 0.01
    aug_bird_outlier_prob: float = 0.05  # Per-tile prob of bird simulation
    aug_bird_z_offset_range: tuple = (5.0, 15.0)  # Z-score offset (≈25-75m physical)

    # Point duplication augmentation
    aug_point_dup_tile_prob: float = 0.3  # Tile-level probability
    aug_point_dup_min_point_prob: float = 0.05  # Min point-level probability
    aug_point_dup_max_point_prob: float = 0.20  # Max point-level probability
    aug_point_dup_min_offset: float = 0.001  # Min offset (z-score units)
    aug_point_dup_max_offset: float = 0.2  # Max offset (z-score units)

    # Omnidirectional outlier augmentation
    aug_omni_outlier_tile_prob: float = 0.2  # Tile-level probability
    aug_omni_outlier_point_prob: float = 0.01  # Point-level probability
    aug_omni_outlier_min_magnitude: float = 2.0  # Min offset (std devs)
    aug_omni_outlier_max_magnitude: float = 20.0  # Max offset (std devs)

    # Temporal shift augmentation
    aug_temporal_shift_prob: float = 0.5  # Tile-level probability
    aug_temporal_max_shift_days: float = 180.0  # Max shift in days (±)

    # Return attribute augmentation (return_num, n_returns)
    # Stats loaded from file at runtime - only need path and behavior params in config
    coordinate_normalization_stats_path: str = None  # Path to JSON file with attr_mean/std
    aug_return_scale_prob: float = 0.5
    aug_return_scale_range: tuple = (0.5, 1.5)  # Multiplier for raw integer values
    aug_return_noise_prob: float = 0.3
    aug_return_noise_sigma: float = 0.1  # In z-score units
    aug_return_zero_prob: float = 0.15
    aug_return_shuffle_prob: float = 0.1

    # NAIP augmentation
    aug_naip_noise_sigma: float = 0.03
    aug_naip_noise_prob: float = 0.3
    aug_naip_blur_kernel: int = 3
    aug_naip_blur_sigma: tuple = (0.1, 2.0)
    aug_naip_blur_prob: float = 0.2
    aug_naip_motion_blur_kernel: int = 5
    aug_naip_motion_blur_angle: tuple = (-45.0, 45.0)
    aug_naip_motion_blur_prob: float = 0.1
    aug_naip_erasing_scale: tuple = (0.02, 0.15)
    aug_naip_erasing_prob: float = 0.1
    aug_naip_sharpness_range: tuple = (0.5, 1.5)
    aug_naip_sharpness_prob: float = 0.2
    aug_naip_equalize_prob: float = 0.1

    # UAVSAR augmentation
    aug_uavsar_noise_sigma: float = 0.05
    aug_uavsar_noise_prob: float = 0.3
    aug_uavsar_blur_kernel: int = 3
    aug_uavsar_blur_sigma: tuple = (0.1, 1.5)
    aug_uavsar_blur_prob: float = 0.2
    aug_uavsar_motion_blur_kernel: int = 3
    aug_uavsar_motion_blur_angle: tuple = (-30.0, 30.0)
    aug_uavsar_motion_blur_prob: float = 0.1
    aug_uavsar_erasing_scale: tuple = (0.02, 0.10)
    aug_uavsar_erasing_prob: float = 0.1

    # Synchronized geometric augmentation (rotation, reflection)
    aug_geometric_enabled: bool = True
    aug_rotation_prob: float = 0.5
    aug_reflection_prob: float = 0.3

    # Temporal subsampling augmentation
    aug_temporal_enabled: bool = True
    aug_naip_subsample_prob: float = 0.5
    aug_naip_min_frames: int = 1
    aug_uavsar_t_subsample_prob: float = 0.5
    aug_uavsar_t_min_frames: int = 1
    aug_uavsar_g_mask_prob: float = 0.3
    aug_uavsar_g_min_images: int = 1

    # Modality dropout augmentation
    aug_modality_dropout_enabled: bool = True
    aug_naip_dropout_prob: float = 0.15
    aug_uavsar_dropout_prob: float = 0.15

    # Point cloud sparse augmentation (only with global-only mode)
    aug_point_removal_enabled: bool = False
    aug_point_removal_prob: float = 0.3
    aug_point_min_removal_ratio: float = 0.05
    aug_point_max_removal_ratio: float = 0.7
    aug_point_min_points: int = 20

    # ===== OOD Forest-Plot Validation =====
    # When enabled, runs a small set of forest plot tiles every N epochs and
    # logs ood_<band>_mae / r2 / ... metrics. Can be used as the early-stopping
    # metric (e.g. early_stopping_metric="ood_canopy_cover_mae").
    ood_val_enabled: bool = False
    ood_val_tiles_path: Optional[str] = None       # .pt file from build_ood_validation_set.py
    ood_val_metadata_path: Optional[str] = None    # .json file from build_ood_validation_set.py
    ood_val_every_n_epochs: int = 5
    ood_val_band_config_path: str = "src/evaluation/configs/raster/veg_structure_8band.json"

    def __post_init__(self):
        """Set default target_band_indices and validate architecture selection."""
        if self.target_band_indices is None:
            self.target_band_indices = [2, 7, 14]  # Default: Height, TFL, Total_cover

        valid_archs = {"cross_attn_grid_mlp", "cross_attn_soft_pillar", "grid_cross_attn"}
        if self.raster_architecture not in valid_archs:
            raise ValueError(
                f"raster_architecture must be one of {sorted(valid_archs)}, "
                f"got {self.raster_architecture!r}"
            )

        if self.raster_architecture == "grid_cross_attn":
            if not (1 <= self.grid_cross_attn_depth <= 4):
                raise ValueError(
                    f"grid_cross_attn_depth must be in [1, 4], got {self.grid_cross_attn_depth}"
                )

        # Warn on stale deprecated fields so users know they're being ignored.
        if self.fusion_type != "cross_attention":
            warnings.warn(
                f"`fusion_type={self.fusion_type!r}` is DEPRECATED and ignored when "
                f"`raster_architecture` is set. Use raster_architecture instead.",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.raster_use_wide_decoder:
            warnings.warn(
                "`raster_use_wide_decoder` is DEPRECATED and ignored. "
                "Use raster_architecture='cross_attn_soft_pillar' instead.",
                DeprecationWarning,
                stacklevel=2,
            )

    def __reduce__(self):
        """Stable kwargs-based pickler.

        Serializes the config as a `(field_name -> value)` dict so future
        field additions, removals, and reorderings can't silently corrupt
        loaded checkpoints. See `_unpickle_raster_config`.
        """
        state = {f.name: getattr(self, f.name) for f in dataclasses.fields(self)}
        return (_unpickle_raster_config, (state,))


# ---------------------------------------------------------------------------
# Backwards-compat shim for unpickling legacy checkpoints.
#
# Old checkpoints were pickled with a positional `__reduce__` that listed
# 133 fields in `_LEGACY_REDUCE_FIELDS` order. Unpickling those calls
# `MultimodalRasterConfig(*args)` with 133 positional values that no longer
# match the current dataclass field layout. Detect that exact shape and
# remap to kwargs (filtering out fields that have since been removed).
# ---------------------------------------------------------------------------
_dataclass_init = MultimodalRasterConfig.__init__


def _bc_init(self, *args, **kwargs):
    # Detect legacy positional unpickle: only positional args, count matches
    # the historical 133-field layout *or* is shorter (older even). When the
    # leading positional run begins with the legacy `k` field, treat as legacy.
    if args and not kwargs and len(args) <= len(_LEGACY_REDUCE_FIELDS):
        legacy_kwargs = dict(zip(_LEGACY_REDUCE_FIELDS, args))
        valid = {f.name for f in dataclasses.fields(MultimodalRasterConfig)}
        kwargs = {k: v for k, v in legacy_kwargs.items() if k in valid}
        args = ()
    _dataclass_init(self, *args, **kwargs)


MultimodalRasterConfig.__init__ = _bc_init


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
        self.use_heteroscedastic_loss = config.use_heteroscedastic_loss

        # Get extractor dropout
        extractor_dropout = getattr(config, 'extractor_dropout', config.pt_attn_dropout)

        # Get position generation hidden dimension
        pos_gen_hidden_dim = getattr(config, 'pos_gen_hidden_dim', 64)

        # Get position encoder parameters with fallback
        pos_encoder_dropout = getattr(config, 'pos_encoder_dropout', 0.1)
        stochastic_pos_dropout_prob = getattr(config, 'stochastic_pos_dropout_prob', 0.0)

        # Track global-only mode
        self.use_global_only = getattr(config, 'use_global_only', False)

        # ====== 1) Feature Extractor ======
        if self.use_global_only:
            # Global-only mode: four consecutive global attention blocks (no local KNN)
            # This enables online point removal since global attention computes positions dynamically
            self.feature_extractor_1 = LocalGlobalPointAttentionBlock(
                in_channels=9,  # 6 attributes + 3 coordinates
                out_channels=config.feature_dim // 8,
                num_lcl_heads=0,  # Global-only: no local attention
                num_glbl_heads=config.extractor_glbl_heads,
                pos_encoding_dim=config.position_encoding_dim // 8,
                dropout=extractor_dropout,
                k_neighbors=config.k,  # Not used when num_lcl_heads=0
                global_drop_path=0,
                use_v2_attention=config.use_v2_attention,
                pos_encoder_dropout=0,
                stochastic_pos_dropout_prob=0
            )
            self.feature_extractor_2 = LocalGlobalPointAttentionBlock(
                in_channels=config.feature_dim // 8,  # Takes output of last block
                out_channels=config.feature_dim // 4,
                num_lcl_heads=0,  # Global-only: no local attention
                num_glbl_heads=config.extractor_glbl_heads,
                pos_encoding_dim=config.position_encoding_dim // 4,
                dropout=extractor_dropout,
                k_neighbors=config.k,  # Not used when num_lcl_heads=0
                global_drop_path=0,
                use_v2_attention=config.use_v2_attention,
                pos_encoder_dropout=pos_encoder_dropout,
                stochastic_pos_dropout_prob=stochastic_pos_dropout_prob
            )
            self.feature_extractor_3 = LocalGlobalPointAttentionBlock(
                in_channels=config.feature_dim // 4,  # Takes output of last block
                out_channels=config.feature_dim // 2,
                num_lcl_heads=0,  # Global-only: no local attention
                num_glbl_heads=config.extractor_glbl_heads,
                pos_encoding_dim=config.position_encoding_dim // 2,
                dropout=extractor_dropout,
                k_neighbors=config.k,  # Not used when num_lcl_heads=0
                global_drop_path=config.extractor_point_attn_drop_path/2,
                use_v2_attention=config.use_v2_attention,
                pos_encoder_dropout=pos_encoder_dropout,
                stochastic_pos_dropout_prob=stochastic_pos_dropout_prob
            )
            self.feature_extractor_4 = LocalGlobalPointAttentionBlock(
                in_channels=config.feature_dim // 2,  # Takes output of last block
                out_channels=config.feature_dim,
                num_lcl_heads=0,  # Global-only: no local attention
                num_glbl_heads=config.extractor_glbl_heads,
                pos_encoding_dim=config.position_encoding_dim,
                dropout=extractor_dropout,
                k_neighbors=config.k,  # Not used when num_lcl_heads=0
                global_drop_path=config.extractor_point_attn_drop_path,
                use_v2_attention=config.use_v2_attention,
                pos_encoder_dropout=pos_encoder_dropout,
                stochastic_pos_dropout_prob=stochastic_pos_dropout_prob
            )
        else:
            # Standard mode: Single block with local+global attention
            self.feature_extractor = LocalGlobalPointAttentionBlock(
                in_channels=9,  # 6 attributes + 3 coordinates
                out_channels=config.feature_dim,
                num_lcl_heads=config.extractor_lcl_heads,
                num_glbl_heads=config.extractor_glbl_heads,
                pos_encoding_dim=config.position_encoding_dim,
                dropout=extractor_dropout,
                k_neighbors=config.k,
                global_drop_path=config.extractor_point_attn_drop_path,
                use_v2_attention=config.use_v2_attention,
                pos_encoder_dropout=pos_encoder_dropout,
                stochastic_pos_dropout_prob=stochastic_pos_dropout_prob
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
                temporal_encoder_type=config.temporal_encoder,
                drop_path=config.encoder_drop_path
            )

        if self.use_uavsar:
            self.uavsar_encoder = UAVSAREncoder(
                in_channels=6,  # 6 polarization bands
                image_size=4,   # 4×4 pixels
                patch_size=1,   # 1×1 pixel patches
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.uavsar_dropout,
                temporal_encoder_type=config.temporal_encoder,
                drop_path=config.encoder_drop_path
            )

        # ====== 3) Fusion (Path A & Path B only) ======
        # Path A and Path B share `CrossAttentionFusion` (also used by the point cloud
        # model). Path C does its own multi-source fusion inside the head.
        self.architecture = config.raster_architecture
        self.uses_cross_attn_fusion = self.architecture in (
            "cross_attn_grid_mlp",
            "cross_attn_soft_pillar",
        )

        if self.uses_cross_attn_fusion:
            self.fusion = CrossAttentionFusion(
                point_dim=config.feature_dim,
                patch_dim=config.img_embed_dim,
                use_naip=self.use_naip,
                use_uavsar=self.use_uavsar,
                num_patches=config.img_num_patches,
                max_dist_ratio=config.max_dist_ratio,
                num_heads=config.fusion_num_heads,
                attention_dropout=config.fusion_dropout,
                position_encoding_dim=config.position_encoding_dim,
            )
        else:
            self.fusion = None

        # ====== 4) Raster Head ======
        if self.architecture == "cross_attn_grid_mlp":
            self.head = CrossAttnGridMlpHead(
                feature_dim=config.feature_dim,
                num_heads=config.raster_num_heads,
                distance_sigma=config.raster_distance_sigma,
                dropout=config.raster_decoder_dropout,
                ffn_ratio=2,
                grid_size=config.grid_size,
                tile_extent_m=config.tile_extent,
                n_bands=config.n_bands,
                output_variance=config.use_heteroscedastic_loss,
            )
        elif self.architecture == "cross_attn_soft_pillar":
            self.head = CrossAttnSoftPillarHead(
                feature_dim=config.feature_dim,
                grid_size=config.grid_size,
                tile_extent_m=config.tile_extent,
                n_bands=config.n_bands,
                decoder_dim=config.raster_hidden_dim,
                num_blocks=config.raster_decoder_layers,
                dropout=config.raster_decoder_dropout,
                output_variance=config.use_heteroscedastic_loss,
            )
        elif self.architecture == "grid_cross_attn":
            self.head = GridCrossAttnHead(
                feature_dim=config.feature_dim,
                depth=config.grid_cross_attn_depth,
                num_heads=config.raster_num_heads,
                distance_sigma=config.raster_distance_sigma,
                dropout=config.raster_attention_dropout,
                ffn_ratio=config.grid_cross_attn_ffn_ratio,
                grid_size=config.grid_size,
                tile_extent_m=config.tile_extent,
                n_bands=config.n_bands,
                output_variance=config.use_heteroscedastic_loss,
                use_naip=self.use_naip,
                use_uavsar=self.use_uavsar,
                patch_dim=config.img_embed_dim,
            )
        else:  # pragma: no cover — validated in __post_init__
            raise ValueError(f"Unknown raster_architecture: {self.architecture}")

        # Legacy attribute aliases, set to None so old code paths checking these don't break.
        self.raster_head = None
        self.grid_fusion = None
        self.raster_decoder = None

        # ====== 5) GPU Training Augmentation (raster model only) ======
        self.training_aug = TrainingAugmentation(config)

    def forward(
        self,
        dep_points: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[Dict],
        dep_attr: Optional[torch.Tensor] = None,
        naip: Optional[List[Dict]] = None,
        uavsar: Optional[List[Dict]] = None,
        bbox: Optional[torch.Tensor] = None,
        debug_logging: bool = False
    ):
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
            If use_heteroscedastic_loss=False:
                pred_raster: Predicted fuel metrics [batch_size, n_bands, 5, 5] (Z-SCORE NORMALIZED)
            If use_heteroscedastic_loss=True:
                Tuple of (mean, log_var):
                - mean: [batch_size, n_bands, 5, 5] - predicted mean values
                - log_var: [batch_size, n_bands, 5, 5] - predicted log-variance
        """
        batch_size = len(norm_params)
        device = dep_points.device

        # Clamp extreme Z values (bird returns)
        dep_points[:, 2] = torch.clamp(dep_points[:, 2], -10, 10)  # In z-score space, ±10 is ~70m from mean

        # ====== GPU Training Augmentation (Point Count Changes - Global-Only Mode) ======
        # Only when use_global_only=True, since global attention doesn't need pre-computed KNN
        # Both point removal and point duplication change point count, breaking KNN
        if self.training and self.use_global_only:
            dep_points, dep_attr, batch_indices = self.training_aug.apply_point_removal(
                dep_points, dep_attr, batch_indices
            )
            dep_points, dep_attr, batch_indices = self.training_aug.apply_point_duplication(
                dep_points, dep_attr, batch_indices
            )

        # ====== GPU Training Augmentation (Point Cloud) ======
        # Applied during training only, disabled during validation/inference
        if self.training:
            dep_points, dep_attr = self.training_aug.augment_points(dep_points, dep_attr)

        # Concatenate attributes and positions
        dep_points_and_attr = torch.cat([dep_attr, dep_points], dim=1)  # [N_total, 9]

        # ====== 1) Point Cloud Feature Extraction ======
        if self.use_global_only:
            # Global-only mode: n consecutive global attention blocks
            x_feat, _ = self.feature_extractor_1(dep_points_and_attr, dep_points, edge_index)
            x_feat, _ = self.feature_extractor_2(x_feat, dep_points, edge_index)
            x_feat, _ = self.feature_extractor_3(x_feat, dep_points, edge_index)
            x_feat, _ = self.feature_extractor_4(x_feat, dep_points, edge_index)
        else:
            # Standard mode: Single block with local+global attention
            x_feat, _ = self.feature_extractor(dep_points_and_attr, dep_points, edge_index)
        # x_feat: [N_total, feature_dim]

        if debug_logging:
            has_nan = torch.isnan(x_feat).any().item()
            print(f"  [DEBUG] After feature_extractor: NaN={has_nan}")
            if has_nan:
                print(f"    NaN count: {torch.isnan(x_feat).sum().item()}/{x_feat.numel()}")

        # ====== 2) Imagery Feature Extraction ======
        # Process each tile separately since imagery data is list of dicts

        # ====== GPU Training Augmentation (Temporal Subsampling) ======
        # Note: Modality dropout is applied AFTER encoding via apply_embedding_dropout()
        # This ensures encoder params always get gradients (no find_unused_parameters needed)
        if self.training:
            # Temporal subsampling (per-tile)
            for b in range(batch_size):
                if naip is not None and naip[b] is not None:
                    naip[b], _ = self.training_aug.augment_temporal(naip[b], None)
                if uavsar is not None and uavsar[b] is not None:
                    _, uavsar[b] = self.training_aug.augment_temporal(None, uavsar[b])

            # Temporal shift (vectorized batch)
            self.training_aug.augment_temporal_shift_batch(naip, uavsar, device)

        naip_embeddings_list = []
        uavsar_embeddings_list = []

        for b in range(batch_size):
            # NAIP encoding for tile b
            if self.use_naip and naip is not None and naip[b] is not None:
                naip_b = naip[b]
                if 'images' in naip_b and naip_b['images'] is not None:
                    # Convert to float32 if needed (preprocessed data may be float16)
                    naip_images = naip_b['images'].to(device).float()

                    # GPU Training Augmentation (NAIP)
                    if self.training:
                        naip_images = self.training_aug.augment_naip(naip_images)

                    rel_dates = naip_b.get('relative_dates', None)
                    if rel_dates is not None:
                        rel_dates = rel_dates.to(device)

                    naip_emb = self.naip_encoder(
                        naip_images,
                        naip_b.get('img_bbox', None),
                        rel_dates
                    )  # [num_patches, img_embed_dim]

                    if debug_logging:
                        has_nan = torch.isnan(naip_emb).any().item()
                        print(f"  [DEBUG] After naip_encoder (tile {b}): NaN={has_nan}")
                        if has_nan:
                            print(f"    NaN count: {torch.isnan(naip_emb).sum().item()}/{naip_emb.numel()}")

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

                    # GPU Training Augmentation (UAVSAR)
                    if self.training:
                        uavsar_images = self.training_aug.augment_uavsar(uavsar_images)

                    mask = uavsar_b.get('attention_mask', None)
                    if mask is not None:
                        mask = mask.to(device)
                    rel_dates = uavsar_b.get('relative_dates', None)
                    if rel_dates is not None:
                        rel_dates = rel_dates.to(device)

                    # Filter out UAVSAR acquisitions with NaN values
                    # Check NaN per acquisition: uavsar_images shape is [n_images, 6, 4, 4]
                    n_acquisitions = uavsar_images.shape[0]
                    pixels_per_acquisition = uavsar_images.shape[1] * uavsar_images.shape[2] * uavsar_images.shape[3]  # 6*4*4 = 96

                    # Count NaN pixels per acquisition
                    nan_counts_per_image = torch.isnan(uavsar_images).view(n_acquisitions, -1).sum(dim=1)  # [n_images]
                    has_nan_per_image = nan_counts_per_image > 0

                    if debug_logging:
                        print(f"  [DEBUG] UAVSAR (tile {b}): {n_acquisitions} acquisitions")
                        for acq_idx in range(n_acquisitions):
                            nan_count = nan_counts_per_image[acq_idx].item()
                            status = "✗ REMOVE" if has_nan_per_image[acq_idx] else "✓ KEEP"
                            print(f"    Acquisition {acq_idx}: {nan_count}/{pixels_per_acquisition} NaN pixels ({100*nan_count/pixels_per_acquisition:.1f}%) - {status}")

                    if has_nan_per_image.any():
                        # Some acquisitions have NaN - filter them out
                        valid_mask = ~has_nan_per_image

                        if valid_mask.any():
                            # Keep only valid acquisitions
                            n_removed = has_nan_per_image.sum().item()
                            n_kept = valid_mask.sum().item()

                            uavsar_images = uavsar_images[valid_mask]
                            if mask is not None:
                                mask = mask[valid_mask]
                            if rel_dates is not None:
                                rel_dates = rel_dates[valid_mask]

                            if debug_logging:
                                print(f"  [DEBUG] UAVSAR filtering: Removed {n_removed}, kept {n_kept}")
                        else:
                            # All acquisitions have NaN - skip UAVSAR for this tile
                            if debug_logging:
                                print(f"  [DEBUG] UAVSAR filtering: All {n_acquisitions} acquisitions have NaN, skipping UAVSAR")
                            uavsar_embeddings_list.append(None)
                            continue

                    uavsar_emb = self.uavsar_encoder(
                        uavsar_images,
                        attention_mask=mask,
                        img_bbox=uavsar_b.get('img_bbox', None),
                        relative_dates=rel_dates
                    )  # [num_patches, img_embed_dim]

                    if debug_logging:
                        has_nan = torch.isnan(uavsar_emb).any().item()
                        print(f"  [DEBUG] After uavsar_encoder (tile {b}): NaN={has_nan}")
                        if has_nan:
                            print(f"    NaN count: {torch.isnan(uavsar_emb).sum().item()}/{uavsar_emb.numel()}")

                    uavsar_embeddings_list.append(uavsar_emb)
                else:
                    uavsar_embeddings_list.append(None)
            else:
                uavsar_embeddings_list.append(None)

        # ====== GPU Training Augmentation (Per-Sample Embedding Dropout) ======
        # Applied AFTER encoding to ensure encoder params always get gradients
        # Uses * 0.0 to zero embeddings (gradient-safe, no find_unused_parameters needed)
        if self.training:
            naip_embeddings_list, uavsar_embeddings_list = self.training_aug.apply_embedding_dropout(
                naip_embeddings_list, uavsar_embeddings_list, device
            )

        # ====== 3) Stack imagery embeddings + per-tile presence masks ======
        # Tiles missing a modality have a zero-filled slot here. The presence masks
        # (naip_present_mask / uavsar_present_mask) are how we tell every downstream
        # consumer which slots are real vs. zero-filled, so embedding dropout and
        # missing modalities behave identically.
        naip_stacked = None
        uavsar_stacked = None
        naip_present_mask = None
        uavsar_present_mask = None

        if self.use_naip and len(naip_embeddings_list) > 0:
            num_patches = self.config.img_num_patches
            embed_dim = self.config.img_embed_dim
            naip_stacked = torch.zeros(batch_size, num_patches, embed_dim, device=device)
            naip_present_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for b, emb in enumerate(naip_embeddings_list):
                if emb is not None:
                    naip_stacked[b] = emb
                    naip_present_mask[b] = True

        if self.use_uavsar and len(uavsar_embeddings_list) > 0:
            num_patches = self.config.img_num_patches
            embed_dim = self.config.img_embed_dim
            uavsar_stacked = torch.zeros(batch_size, num_patches, embed_dim, device=device)
            uavsar_present_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
            for b, emb in enumerate(uavsar_embeddings_list):
                if emb is not None:
                    uavsar_stacked[b] = emb
                    uavsar_present_mask[b] = True

        # ====== 4) Optional CrossAttentionFusion (Path A & Path B) ======
        if self.uses_cross_attn_fusion:
            modality_mask = {}
            if naip_present_mask is not None:
                modality_mask['naip'] = naip_present_mask
            if uavsar_present_mask is not None:
                modality_mask['uavsar'] = uavsar_present_mask

            x_feat = self.fusion.forward_batched(
                point_features=x_feat,
                point_positions=dep_points,
                batch_indices=batch_indices,
                norm_params=norm_params,
                naip_embeddings=naip_stacked,
                uavsar_embeddings=uavsar_stacked,
                modality_mask=modality_mask if modality_mask else None,
            )  # [N_total, feature_dim]

            if debug_logging:
                has_nan = torch.isnan(x_feat).any().item()
                print(f"  [DEBUG] After cross-attn fusion: NaN={has_nan}")
                if has_nan:
                    print(f"    NaN count: {torch.isnan(x_feat).sum().item()}/{x_feat.numel()}")

        # ====== 5) Raster head ======
        if self.architecture == "grid_cross_attn":
            pred_raster = self.head(
                point_features=x_feat,
                point_positions=dep_points,
                batch_indices=batch_indices,
                norm_params=norm_params,
                naip_stacked=naip_stacked,
                uavsar_stacked=uavsar_stacked,
                naip_present_mask=naip_present_mask,
                uavsar_present_mask=uavsar_present_mask,
            )
        else:
            pred_raster = self.head(
                point_features=x_feat,
                point_positions=dep_points,
                batch_indices=batch_indices,
                norm_params=norm_params,
            )

        if debug_logging:
            debug_tensor = pred_raster[0] if isinstance(pred_raster, tuple) else pred_raster
            has_nan = torch.isnan(debug_tensor).any().item()
            print(f"  [DEBUG] After raster head ({self.architecture}): NaN={has_nan}")
            if has_nan:
                print(f"    NaN count: {torch.isnan(debug_tensor).sum().item()}/{debug_tensor.numel()}")

        return pred_raster
