"""
Image Encoder Pre-training Model for independent NAIP/UAVSAR encoder training.

This module provides a minimal model that trains image encoders independently
on the fuel metrics prediction task, enabling pre-training before fusion with
the point cloud encoder.

Usage:
    python run_pretrain_image_encoders.py --encoder naip
    python run_pretrain_image_encoders.py --encoder uavsar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import List, Optional, Union

from src.models.encoders import NAIPEncoder, UAVSAREncoder


@dataclass
class ImageEncoderPretrainConfig:
    """Configuration for image encoder pre-training.

    This config contains only the parameters needed for encoder pre-training,
    matching the relevant subset of MultimodalRasterConfig for compatibility.
    """
    # Which encoder to train
    encoder_type: str = "naip"  # "naip" or "uavsar"

    # Encoder parameters (match MultimodalRasterConfig)
    img_embed_dim: int = 256
    img_num_patches: int = 16
    temporal_encoder: str = "gru"
    naip_dropout: float = 0.01
    uavsar_dropout: float = 0.01
    encoder_drop_path: float = 0.0

    # Basic raster head parameters
    n_bands: int = 5
    target_band_indices: List[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    head_hidden_dims: List[int] = field(default_factory=lambda: [128, 64])
    head_dropout: float = 0.1
    grid_size: int = 5

    # Augmentation parameters (passed through to training)
    # These match the aug_* params in MultimodalRasterConfig
    training_augmentation_enabled: bool = True
    aug_geometric_enabled: bool = True
    aug_rotation_prob: float = 0.5
    aug_reflection_prob: float = 0.3
    aug_temporal_enabled: bool = True
    aug_naip_subsample_prob: float = 0.3
    aug_naip_min_frames: int = 1
    aug_uavsar_t_subsample_prob: float = 0.3
    aug_uavsar_t_min_frames: int = 1
    aug_uavsar_g_mask_prob: float = 0.2
    aug_uavsar_g_min_images: int = 1

    # NAIP augmentation
    aug_naip_noise_sigma: float = 0.03
    aug_naip_noise_prob: float = 0.1
    aug_naip_blur_kernel: int = 3
    aug_naip_blur_sigma: tuple = (0.1, 2.0)
    aug_naip_blur_prob: float = 0.1
    aug_naip_motion_blur_kernel: int = 5
    aug_naip_motion_blur_angle: tuple = (-45.0, 45.0)
    aug_naip_motion_blur_prob: float = 0.1
    aug_naip_erasing_scale: tuple = (0.02, 0.15)
    aug_naip_erasing_prob: float = 0.0
    aug_naip_sharpness_range: tuple = (0.5, 1.5)
    aug_naip_sharpness_prob: float = 0.1
    aug_naip_radiometric_prob: float = 0.0   # Master probability for z-score gain/bias augmentation
    aug_naip_radiometric_strength: float = 1.0  # 1.0 = base ranges; <1 shrinks them, >1 widens them
    aug_naip_post_clip_range: tuple = (-4.0, 4.0)  # Final z-score clamp after the radiometric step

    # UAVSAR augmentation
    aug_uavsar_noise_sigma: float = 0.05
    aug_uavsar_noise_prob: float = 0.1
    aug_uavsar_blur_kernel: int = 3
    aug_uavsar_blur_sigma: tuple = (0.1, 1.0)
    aug_uavsar_blur_prob: float = 0.1
    aug_uavsar_motion_blur_kernel: int = 3
    aug_uavsar_motion_blur_angle: tuple = (-30.0, 30.0)
    aug_uavsar_motion_blur_prob: float = 0.1
    aug_uavsar_erasing_scale: tuple = (0.02, 0.10)
    aug_uavsar_erasing_prob: float = 0.0

    # Disable modality dropout for pre-training (we're training one encoder)
    aug_modality_dropout_enabled: bool = False
    aug_naip_dropout_prob: float = 0.0
    aug_uavsar_dropout_prob: float = 0.0

    # These are required by training but not used by this model
    # Set reasonable defaults for compatibility
    use_naip: bool = False  # Set dynamically based on encoder_type
    use_uavsar: bool = False  # Set dynamically based on encoder_type
    use_global_only: bool = True  # Match main model default
    feature_dim: int = 256  # Not used but needed for training loop compatibility
    k: int = 15  # Not used but needed for compatibility

    # Transfer learning (not used for pre-training, but needed by training loop)
    checkpoint_path: Optional[str] = None
    layers_to_load: Optional[List[str]] = None
    layers_to_freeze: Optional[List[str]] = None

    # SWA (Stochastic Weight Averaging)
    swa_enabled: bool = False
    swa_start_epoch: int = 50
    swa_update_freq: int = 1

    # Loss function parameters
    correlation_loss_weight: float = 0.0
    huber_delta: float = 1.0

    # Not used by this model but needed for augmentation pipeline compatibility
    aug_coord_jitter_sigma_xy: float = 0.0
    aug_coord_jitter_sigma_z: float = 0.0
    aug_coord_jitter_prob: float = 0.0
    aug_intensity_noise_sigma: float = 0.0
    aug_intensity_noise_prob: float = 0.0
    aug_intensity_outlier_prob: float = 0.0
    aug_bird_outlier_prob: float = 0.0
    aug_bird_z_offset_range: tuple = (5.0, 15.0)
    aug_point_dup_tile_prob: float = 0.0
    aug_point_dup_min_point_prob: float = 0.0
    aug_point_dup_max_point_prob: float = 0.0
    aug_point_dup_min_offset: float = 0.0
    aug_point_dup_max_offset: float = 0.0
    aug_omni_outlier_tile_prob: float = 0.0
    aug_omni_outlier_point_prob: float = 0.0
    aug_omni_outlier_min_magnitude: float = 0.0
    aug_omni_outlier_max_magnitude: float = 0.0
    aug_temporal_shift_prob: float = 0.0
    aug_temporal_max_shift_days: float = 0.0
    coordinate_normalization_stats_path: str = None
    aug_return_scale_prob: float = 0.0
    aug_return_scale_range: tuple = (1.0, 1.0)
    aug_return_noise_prob: float = 0.0
    aug_return_noise_sigma: float = 0.0
    aug_return_zero_prob: float = 0.0
    aug_return_shuffle_prob: float = 0.0
    aug_point_removal_enabled: bool = False
    aug_point_removal_prob: float = 0.0
    aug_point_min_removal_ratio: float = 0.0
    aug_point_max_removal_ratio: float = 0.0
    aug_point_min_points: int = 20

    # OOD validation — disabled for pretraining. The training loop reads
    # these unconditionally (even when OOD is off), so they must exist.
    ood_val_enabled: bool = False
    ood_val_tiles_path: Optional[str] = None
    ood_val_metadata_path: Optional[str] = None
    ood_val_every_n_epochs: int = 5
    ood_val_band_config_path: Optional[str] = None

    def __post_init__(self):
        """Set modality flags based on encoder_type."""
        if self.encoder_type == "naip":
            self.use_naip = True
            self.use_uavsar = False
        elif self.encoder_type == "uavsar":
            self.use_naip = False
            self.use_uavsar = True
        else:
            raise ValueError(f"encoder_type must be 'naip' or 'uavsar', got '{self.encoder_type}'")


class BasicImageRasterHead(nn.Module):
    """Simple raster prediction head for image encoder pre-training.

    Takes patch embeddings [16, embed_dim] and predicts fuel metrics [n_bands, 5, 5].

    Spatial conventions match the production raster head for transfer learning:
    - Patches at [-7.5, -2.5, 2.5, 7.5]m (20×20m imagery, 4×4 grid)
    - Target grid at [-4, -2, 0, 2, 4]m (10×10m target, 5×5 grid)
    - Sinusoidal positional encoding ADDED to embeddings (not concatenated)

    Architecture:
        1. Add sinusoidal positional encoding to patch embeddings
        2. Reshape 16 patches to 4×4 grid
        3. grid_sample at correct target positions (not naive interpolation)
        4. MLP: embed_dim → hidden_dims → n_bands
    """

    def __init__(
        self,
        embed_dim: int,
        n_bands: int,
        hidden_dims: List[int],
        dropout: float = 0.1
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_bands = n_bands
        self.grid_in = 4  # 16 patches = 4×4 grid
        self.grid_out = 5  # Output 5×5 grid

        # Reuse PatchPositionEncoding from the production raster head primitives
        # to ensure identical positional encoding for transfer learning.
        from src.models.raster_primitives import PatchPositionEncoding
        self.patch_pos_encoding = PatchPositionEncoding(
            feature_dim=embed_dim,
            patch_grid_size=4,
            patch_extent_m=20.0,
        )

        # Precompute grid_sample coordinates for correct spatial mapping
        # With align_corners=True on 4×4 grid:
        #   - grid_coord -1.0 → pixel 0 center → -7.5m
        #   - grid_coord +1.0 → pixel 3 center → +7.5m
        # So: grid_coord = physical_position / 7.5
        target_centers_1d = torch.tensor([-4.0, -2.0, 0.0, 2.0, 4.0])
        tyy, txx = torch.meshgrid(target_centers_1d, target_centers_1d, indexing='ij')
        grid_coords = torch.stack([txx, tyy], dim=-1) / 7.5  # [-0.533, -0.267, 0, 0.267, 0.533]
        self.register_buffer('sample_grid', grid_coords.view(1, self.grid_out, self.grid_out, 2))

        # Build MLP layers (input is embed_dim since we ADD pos encoding, not concatenate)
        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Final projection to n_bands
        layers.append(nn.Linear(in_dim, n_bands))

        self.mlp = nn.Sequential(*layers)

    def forward(self, patch_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_embeddings: [16, embed_dim] from image encoder

        Returns:
            raster: [n_bands, 5, 5] predicted fuel metrics
        """
        # Add positional encoding (matches the raster head's additive approach)
        pos_enc, _ = self.patch_pos_encoding()  # [16, embed_dim]
        patch_with_pos = patch_embeddings + pos_enc  # Additive, not concatenation

        # Reshape to 4×4 grid: [16, embed_dim] → [1, embed_dim, 4, 4]
        x = patch_with_pos.view(self.grid_in, self.grid_in, self.embed_dim)
        x = x.permute(2, 0, 1).unsqueeze(0)  # [1, embed_dim, 4, 4]

        # Sample at correct target positions using grid_sample
        # Patches at [-7.5,-2.5,2.5,7.5]m, targets at [-4,-2,0,2,4]m
        x = F.grid_sample(x, self.sample_grid, mode='bilinear', align_corners=True)
        x = x.squeeze(0)  # [embed_dim, 5, 5]

        # Apply MLP per pixel: [embed_dim, 5, 5] → [n_bands, 5, 5]
        x = x.permute(1, 2, 0)  # [5, 5, embed_dim]
        x = self.mlp(x)  # [5, 5, n_bands]
        x = x.permute(2, 0, 1)  # [n_bands, 5, 5]

        return x


class ImageEncoderPretrainModel(nn.Module):
    """Image encoder + basic raster head for pre-training.

    This model trains a single image encoder (NAIP or UAVSAR) to predict
    fuel metrics directly from imagery, without point cloud features.

    The forward() method matches the interface expected by train_raster_model(),
    accepting the same batch format but only using the relevant imagery data.
    """

    def __init__(self, config: ImageEncoderPretrainConfig):
        super().__init__()

        self.config = config
        self.encoder_type = config.encoder_type

        # Create the appropriate encoder
        if config.encoder_type == "naip":
            self.encoder = NAIPEncoder(
                in_channels=4,
                image_size=40,
                patch_size=10,
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.naip_dropout,
                temporal_encoder_type=config.temporal_encoder,
                drop_path=config.encoder_drop_path,
            )
        else:  # uavsar
            self.encoder = UAVSAREncoder(
                in_channels=6,
                image_size=4,
                patch_size=1,
                embed_dim=config.img_embed_dim,
                num_patches=config.img_num_patches,
                dropout=config.uavsar_dropout,
                temporal_encoder_type=config.temporal_encoder,
                drop_path=config.encoder_drop_path,
            )

        # Basic raster head
        self.raster_head = BasicImageRasterHead(
            embed_dim=config.img_embed_dim,
            n_bands=config.n_bands,
            hidden_dims=config.head_hidden_dims,
            dropout=config.head_dropout,
        )

        # Training augmentation (reuse existing infrastructure)
        # Will be initialized by training loop if training_augmentation_enabled=True
        self.training_aug = None

    def _init_augmentation(self, config):
        """Initialize augmentation module (called by training loop)."""
        if config.training_augmentation_enabled:
            from src.models.training_augmentation import TrainingAugmentation
            self.training_aug = TrainingAugmentation(config)

    def forward(
        self,
        dep_points: torch.Tensor,
        edge_index: torch.Tensor,
        batch_indices: torch.Tensor,
        norm_params: List[dict],
        dep_attr: torch.Tensor,
        naip: Optional[List[dict]],
        uavsar: Optional[List[dict]],
        bbox: Optional[torch.Tensor] = None,
        debug_logging: bool = False,
    ) -> torch.Tensor:
        """Forward pass matching MultimodalRasterPredictor interface.

        This model ignores point cloud data and only processes imagery.

        Args:
            dep_points: [N_total, 3] - Ignored
            edge_index: [2, E_total] - Ignored
            batch_indices: [N_total] - Used only for batch_size
            norm_params: List of dicts - Ignored
            dep_attr: [N_total, attr_dim] - Ignored
            naip: List of dicts with NAIP imagery (used if encoder_type='naip')
            uavsar: List of dicts with UAVSAR imagery (used if encoder_type='uavsar')
            bbox: [batch_size, 4] - Ignored
            debug_logging: Enable debug output

        Returns:
            pred_raster: [batch_size, n_bands, 5, 5] predicted fuel metrics
        """
        batch_size = len(norm_params)
        device = dep_points.device

        # Select imagery based on encoder type
        if self.encoder_type == "naip":
            imagery_list = naip
        else:
            imagery_list = uavsar

        # Process each tile
        predictions = []
        for b in range(batch_size):
            if imagery_list is not None and imagery_list[b] is not None:
                img_data = imagery_list[b]

                if 'images' in img_data and img_data['images'] is not None:
                    images = img_data['images'].to(device).float()

                    # Apply augmentation if training
                    if self.training and self.training_aug is not None:
                        if self.encoder_type == "naip":
                            images = self.training_aug.augment_naip(images)
                        else:
                            images = self.training_aug.augment_uavsar(images)

                    # Get optional metadata
                    rel_dates = img_data.get('relative_dates', None)
                    if rel_dates is not None:
                        rel_dates = rel_dates.to(device)

                    # Encode
                    if self.encoder_type == "naip":
                        embeddings = self.encoder(
                            images,
                            img_data.get('img_bbox', None),
                            rel_dates
                        )  # [16, embed_dim]
                    else:  # uavsar
                        mask = img_data.get('attention_mask', None)
                        if mask is not None:
                            mask = mask.to(device)
                        embeddings = self.encoder(
                            images,
                            attention_mask=mask,
                            img_bbox=img_data.get('img_bbox', None),
                            relative_dates=rel_dates
                        )  # [16, embed_dim]

                    # Predict raster
                    pred = self.raster_head(embeddings)  # [n_bands, 5, 5]
                    predictions.append(pred)
                else:
                    # No imagery - predict zeros
                    predictions.append(torch.zeros(
                        self.config.n_bands, 5, 5, device=device
                    ))
            else:
                # No imagery - predict zeros
                predictions.append(torch.zeros(
                    self.config.n_bands, 5, 5, device=device
                ))

        # Stack predictions: [batch_size, n_bands, 5, 5]
        pred_raster = torch.stack(predictions, dim=0)

        return pred_raster


# Convenience function for creating encoder name for checkpoints
def get_encoder_checkpoint_prefix(encoder_type: str) -> str:
    """Get the prefix used for encoder weights in checkpoint state_dict.

    When loading pre-trained encoder weights into full model:
    - naip_encoder.* matches NAIP encoder weights
    - uavsar_encoder.* matches UAVSAR encoder weights
    """
    if encoder_type == "naip":
        return "naip_encoder"
    elif encoder_type == "uavsar":
        return "uavsar_encoder"
    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}")
