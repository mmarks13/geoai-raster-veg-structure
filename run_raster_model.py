"""
Training entry point for multimodal raster prediction model.

Usage:
    python run_raster_model.py                 # 3-band target (default)
    python run_raster_model.py --bands 13      # 13-band target

The --bands flag switches only `n_bands`, `target_band_indices`, and the
output-directory suffix; all other hyperparameters are identical between the
two runs. To customize further, edit `build_config()` below.
"""

import argparse
import datetime

import torch

from src.models.multimodal_raster_model import MultimodalRasterConfig
from src.training.raster_training import train_raster_model

# Mapping from --bands choice to the corresponding (n_bands, target_band_indices).
# Source band indices follow the comment block in build_config() below.
# 3-band:  output 0 -> src 3 (canopy_cover), output 1 -> src 5 (midstory_density),
#          output 2 -> src 7 (foliage height diversity).
# 13-band: heights (0-2) + densities (3-6) + FHD (7) + height percentiles (8-12).
#          Matches src/evaluation/configs/raster/veg_structure_13band.json.
BAND_PRESETS = {
    3: [3, 5, 7],
    13: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}

# ====== CUDA Performance Optimizations ======
# Enable cuDNN benchmark for faster convolution algorithm selection
torch.backends.cudnn.benchmark = True

# Enable TensorCore optimization for float32 matmul operations
# 'high' uses TF32 on Ampere+ GPUs for ~3x faster matmul with minimal precision loss
torch.set_float32_matmul_precision('high')

# Enable TF32 for CUDA matmul and cuDNN operations (complementary to set_float32_matmul_precision)
# These flags ensure TF32 is used at the CUDA backend level
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def build_config(bands: int = 3) -> MultimodalRasterConfig:
    """Build the canonical raster model config used for training.

    Args:
        bands: Number of output bands. Must be a key in BAND_PRESETS (3 or 13).
            Only `n_bands` and `target_band_indices` are affected by this
            argument; every other hyperparameter is identical between the two.
    """
    if bands not in BAND_PRESETS:
        raise ValueError(
            f"Unsupported --bands value: {bands}. "
            f"Supported: {sorted(BAND_PRESETS)}."
        )
    target_band_indices = BAND_PRESETS[bands]

    return MultimodalRasterConfig(
        # Model architecture
        k=15,
        feature_dim=256,  # Keep feature_dim // extractor_glbl_heads <= ~64; much larger head_dim can OOM batch-aware V2 global attention.
        pt_attn_dropout=0.01,
        pt_block_dropout=0.10,  # Standard block dropout for extractor output + FFN paths

        # Feature extractor heads
        extractor_lcl_heads=4,
        extractor_glbl_heads=4,

        # Modality selection (set to True to enable)
        use_naip=True,
        use_uavsar=False,

        # Image encoder parameters
        img_embed_dim=128,
        img_num_patches=16,
        naip_dropout=0.05,
        uavsar_dropout=0.05,
        temporal_encoder="gru",

        # Fusion parameters
        fusion_num_heads=8,
        fusion_dropout=0.10,
        max_dist_ratio=8.0,  # Distance ratio in meters for cross-attention masking

        # Position Encoders
        position_encoding_dim=48,  # Must be divisible by 6 for 3D positions

        # Raster-specific parameters

        # Vegetation Structure Metrics - Target bands (0-indexed):
        # Basic Statistics (Bands 0-2):
        #   Band 0:  Maximum height (m)
        #   Band 1:  Mean height (m)
        #   Band 2:  Std dev height (m)
        # Cover and Density (Bands 3-6):
        #   Band 3:  Canopy cover - proportion of all returns above canopy threshold (fraction 0-1)
        #   Band 4:  Canopy density - proportion of vegetation returns in canopy layer >3m (fraction 0-1)
        #   Band 5:  Mid-story density - proportion of vegetation returns 1-3m (fraction 0-1)
        #   Band 6:  Understory density - proportion of vegetation returns <1m (fraction 0-1)
        # Complexity (Band 7):
        #   Band 7:  Foliage Height Diversity (FHD) - Shannon-Wiener index of vertical layering
        # Height Percentiles (Bands 8-12):
        #   Band 8:  10th percentile height (m)
        #   Band 9:  25th percentile height (m)
        #   Band 10: 50th percentile (median) height (m)
        #   Band 11: 75th percentile height (m)
        #   Band 12: 90th percentile height (m)
        # Density Proportions (Bands 13-22):
        #   Band 13-22: Proportion of returns in each 2.5m vertical layer (0-25m range)
        # n_bands and target_band_indices are driven by the --bands CLI flag;
        # see BAND_PRESETS at the top of this file.
        n_bands=bands,
        target_band_indices=target_band_indices,

        grid_size=5,
        tile_extent=10.0,
        raster_num_heads=8,
        # MULTI-SCALE ATTENTION: Per-head sigma values
        # 2 heads @ σ=0.5m (very local), 4 heads @ σ=2.0m (medium), 2 heads @ σ=5.0m (wide)
        raster_distance_sigma=[0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0],
        # raster_distance_sigma=[0.5, 2.0, 2.0, 5.0],
        # Raster head dropout knobs:
        #   raster_attention_dropout → grid aggregator attention map,
        #   raster_ffn_dropout      → post-aggregator PreLN FFN,
        #   raster_decoder_dropout  → final SmallMlpDecoder.
        raster_attention_dropout=0.05,
        raster_ffn_dropout=0.5,
        raster_decoder_dropout=0.05,

        # Optional: Transfer learning (only loads weights, not optimizer state)
        # Multi-checkpoint loading with key remapping for pretrained encoders
        # Format: [(path, layers_to_load, key_prefix_map), ...]
        # checkpoint_path=None,
        # checkpoint_path=[
        #     ("data/output/raster_cross_attn_grid_mlp_sweep_20260416_221443/ptbd_0p1__fd_0p1__rdd_0p15__wd_0p1/checkpoints/ptbd_0p1__fd_0p1__rdd_0p15__wd_0p1__epoch_20.pth",
        #     None,  # load all layers
        #     None),
        # ],

        checkpoint_path= [
            # # Baseline: feature extractors (no remapping needed)
            # ("data/output/raster_model_baseline_20260415_041056/checkpoints/epoch_10.pth",
            #  ["feature_extractor"], None),
            # NAIP encoder: remap "encoder.*" → "naip_encoder.*"
            ("data/output/pretrained_naip_encoder_d128/checkpoints/epoch_20.pth",
             ["encoder"], {"encoder": "naip_encoder"}),
            # # UAVSAR encoder: remap "encoder.*" → "uavsar_encoder.*"
            # ("data/output/uavsar_encoder_pretrain_d128_20260415_044952/checkpoints/epoch_40.pth",
            #  ["encoder"], {"encoder": "uavsar_encoder"}),
        ],

        # checkpoint_path= [
        #     # # Baseline: feature extractors (no remapping needed)
        #     # ("data/output/raster_model_baseline_20260415_041056/checkpoints/epoch_10.pth",
        #     #  ["feature_extractor"], None),
        #     # NAIP encoder: remap "encoder.*" → "naip_encoder.*"
        #     ("data/output/naip_encoder_pretrain_d128_20260414_233516/checkpoints/epoch_20.pth",
        #      ["encoder"], {"encoder": "naip_encoder"}),
        #     # UAVSAR encoder: remap "encoder.*" → "uavsar_encoder.*"
        #     ("data/output/uavsar_encoder_pretrain_d128_20260415_044952/checkpoints/epoch_40.pth",
        #      ["encoder"], {"encoder": "uavsar_encoder"}),
        # ],
        layers_to_load=None,  # Ignored for multi-checkpoint (use per-checkpoint filters above)
        layers_to_freeze=None,  # Set to freeze loaded layers if desired

        # Stochastic depth (DropPath) - regularization for residual connections
        encoder_drop_path=0.0,      # Drop path for image encoder TransformerBlocks
        extractor_point_attn_drop_path=0.0,  # Drop path for feature extractor

        # Position encoder regularization
        pos_encoder_dropout=0.0,              # Dropout for position encoder (MLP + embedding) (point attn blocks 2+)
        stochastic_pos_dropout_prob=0.0,     # Probability of zeroing position embedding (point attn blocks 2+)

        # Huber loss - robust to outliers in fuel metrics
        huber_delta=2.0,  # Delta threshold (errors > delta use linear penalty)

        # Correlation loss weight (addresses variance collapse)

        # Higher values encourage model to preserve variance in predictions
        correlation_loss_weight=0.0,

        # Transfer learning rate multiplier (for fine-tuning pretrained layers)
        # Transferred layers get lr * transfer_lr_multiplier
        transfer_lr_multiplier=0.1,

        # Heteroscedastic (Gaussian NLL) loss - outputs mean + variance per pixel
        # When enabled, model predicts uncertainty alongside values
        # High variance = "I don't know" (gradients naturally down-weighted)
        use_heteroscedastic_loss=True,  # Set to True to enable
        heteroscedastic_min_var=1e-2,    # Minimum variance for numerical stability
        heteroscedastic_overconfidence_weight=0.01,  # Penalty for σ² < 1 (prevents overconfidence). Increases as σ² → 0.

        # ===============================================================
        # GPU Training Augmentation (Kornia + Custom PyTorch)
        # ===============================================================
        # Applied during training only (disabled in model.eval())
        # See docs/training_augmentation.md for full documentation

        training_augmentation_enabled=True,  # Master switch for all augmentations

        # --- Point Cloud Augmentation ---
        # Coordinate jitter: adds Gaussian noise to x,y,z coords
        # x,y,z standard deviations are    2.9, 2.9, 6.2  meters respectively
        # Physical effect: simulates point position uncertainty
        aug_coord_jitter_sigma_xy=0.03,  # Noise std for x,y in z-score units
        aug_coord_jitter_sigma_z=0.015,   # Noise std for z in z-score units
        aug_coord_jitter_prob=0.6,      # Probability of applying jitter per tile

        # Intensity noise: adds Gaussian noise to intensity values
        # Physical effect: simulates sensor noise and atmospheric effects
        aug_intensity_noise_sigma=0.05,  # Noise std in z-score units
        aug_intensity_noise_prob=0.10,    # Probability per tile

        # Intensity outliers: randomly replaces intensity values with extreme values
        # Physical effect: simulates sensor saturation, multipath returns
        aug_intensity_outlier_prob=0.001,  # Per-point probability of outlier

        # Bird simulation: adds extreme z-offset to 1 random point
        # Physical effect: simulates bird/drone flyover returns in LiDAR
        aug_bird_outlier_prob=0.001,           # Probability per tile
        aug_bird_z_offset_range=(5.0, 30.0),  # Z-score offset (≈25-75m physical)


        # Point duplication (models redundant LiDAR returns)
        # Note: Only active when use_global_only=True (breaks precomputed KNN)
        aug_point_dup_tile_prob=0.0,          # Probability tiles get duplication
        aug_point_dup_min_point_prob=0.01,    # min probability of points duplicated per tile
        aug_point_dup_max_point_prob=0.40,    # max probability of points duplicated per tile
        aug_point_dup_min_offset=0.001,       # Offset range: z-score units
        aug_point_dup_max_offset=0.2,


        # Omnidirectional outliers
        aug_omni_outlier_tile_prob=0.001,       # fraction of tiles that get outliers
        aug_omni_outlier_point_prob=0.003,     # fraction of points become outliers
        aug_omni_outlier_min_magnitude=2.0,   # Magnitude: 2-20 std dev
        aug_omni_outlier_max_magnitude=20.0,

        # --- Return Attribute Augmentation (return_num, n_returns) ---
        # Makes model robust to variations in LiDAR return patterns
        coordinate_normalization_stats_path="data/processed/model_data_veg_structure/coordinate_normalization_stats_train.json",
        aug_return_scale_prob=0.20,            # Probability of scaling (stretch/shrink)
        aug_return_scale_range=(0.5, 1.8),    # Scale multiplier for raw integer values
        aug_return_noise_prob=0.10,            # Probability of adding Gaussian noise
        aug_return_noise_sigma=0.01,           # Noise std in z-score units
        aug_return_zero_prob=0.10,            # Probability of zeroing out return attrs
        aug_return_shuffle_prob=0.10,          # Probability of shuffling among points

        # --- NAIP Augmentation (4-channel optical: RGBN) ---
        # Gaussian noise: simulates sensor noise
        aug_naip_noise_sigma=0.03,
        aug_naip_noise_prob=0.10,

        # Gaussian blur: simulates atmospheric haze, focus issues
        aug_naip_blur_kernel=3,
        aug_naip_blur_sigma=(0.1, 2.0),
        aug_naip_blur_prob=0.10,

        # Motion blur: simulates aircraft motion, wind effects
        aug_naip_motion_blur_kernel=5,
        aug_naip_motion_blur_angle=(-45.0, 45.0),
        aug_naip_motion_blur_prob=0.10,

        # Random erasing (sets value to 0, mean is z-score space)
        aug_naip_erasing_scale=(0.02, 0.15),  # Erased area as fraction of image
        aug_naip_erasing_prob=0.10,

        # Sharpness: simulates varying focus quality
        aug_naip_sharpness_range=(0.5, 1.5),
        aug_naip_sharpness_prob=0.0,

        # Z-score radiometric augmentation: master probability for global/per-channel gain+bias.
        # Strength = 1.0 uses the base ranges; <1 shrinks them and >1 widens them.
        # Post-clip bounds the final z-score values after the radiometric step.
        aug_naip_radiometric_prob=0.40,
        aug_naip_radiometric_strength=1,
        aug_naip_post_clip_range=(-4.0, 4.0),

        # --- UAVSAR Augmentation (6-channel SAR: polarimetric) ---
        # Gaussian noise: simulates thermal/system noise (valid in dB domain)
        aug_uavsar_noise_sigma=0.05,
        aug_uavsar_noise_prob=0.20,

        # Gaussian blur: simulates multi-looking (speckle filtering)
        aug_uavsar_blur_kernel=3,
        aug_uavsar_blur_sigma=(0.1, 1.0),
        aug_uavsar_blur_prob=0.20,

        # Motion blur: simulates platform motion effects
        aug_uavsar_motion_blur_kernel=3,
        aug_uavsar_motion_blur_angle=(-30.0, 30.0),
        aug_uavsar_motion_blur_prob=0.10,

        # Random erasing (sets value to 0, mean is z-score space)
        aug_uavsar_erasing_scale=(0.10, 0.20),
        aug_uavsar_erasing_prob=0.0,

        # --- Synchronized Geometric Augmentation ---
        # Applies identical rotation/reflection to points, images, and targets
        aug_geometric_enabled=True,
        aug_rotation_prob=0.75,  # Probability of 9°/180°/270° rotation
        aug_reflection_prob=0.3,  # Probability of X or Y axis reflection

        # --- Temporal Subsampling Augmentation ---
        # Randomly subsamples temporal dimension to improve generalization
        aug_temporal_enabled=True,
        aug_naip_subsample_prob=0.4,  # Probability of NAIP temporal subsampling
        aug_naip_min_frames=1,  # Minimum NAIP frames to keep
        aug_uavsar_t_subsample_prob=0.3,  # Probability of UAVSAR T-dim subsampling
        aug_uavsar_t_min_frames=1,  # Minimum UAVSAR temporal groups to keep
        aug_uavsar_g_mask_prob=0.3,  # Probability of G-dim masking within groups
        aug_uavsar_g_min_images=1,  # Minimum images to keep per group


        aug_temporal_shift_prob=0.9,          # 80% of tiles get temporal shift
        aug_temporal_max_shift_days=730,    # ±730 days


        # --- Modality Dropout Augmentation ---
        # Randomly drops entire modalities for robustness (e.g., Laguna has no UAVSAR)
        aug_modality_dropout_enabled=True,
        aug_naip_dropout_prob=0.25,  # Probability of dropping NAIP entirely
        aug_uavsar_dropout_prob=0.35,  # Probability of dropping UAVSAR entirely

        # --- Point Cloud Sparse Augmentation (Global-Only Mode) ---
        # Randomly removes points (only when use_global_only=True)
        aug_point_removal_enabled=True,  # Enable only with use_global_only=True
        aug_point_removal_prob=0.4,  # Probability of applying point removal
        aug_point_min_removal_ratio=0.01,  # Min fraction of points to remove
        aug_point_max_removal_ratio=0.9,  # Max fraction of points to remove
        aug_point_min_points=20,  # Minimum points to keep

        # --- Global-Only Attention Mode ---
        # Uses four consecutive global attention blocks (no local KNN)
        # Enables online point removal since global attention computes positions dynamically
        use_global_only=True,

        # --- OOD Robustness (Spectral Normalization + SWA) ---
        # Spectral Normalization (Lipschitz constraint for OOD robustness)
        use_spectral_norm=True,  # Set to True to enable

        # Stochastic Weight Averaging (model ensemble for OOD robustness)
        swa_enabled=False,         # Set to True to enable
        swa_start_epoch=200,       # Start averaging at this epoch (w/checkpoint this ignores the restart epoch. indexes at 0 again)
        swa_update_freq=1,         # Update every epoch

        # --- In-training OOD forest-plot validation ---

        # epochs, reusing the §7 stitching/extraction/stats helpers. Can be
        # wired into early stopping via early_stopping_metric='ood_overall_mean_tsmae'
        # so the loop tracks OOD generalization rather than in-distribution val.
        ood_val_enabled=True,
        ood_val_tiles_path="data/processed/forest_plot_data/ood_validation/ood_validation_tiles.pt",
        ood_val_metadata_path="data/processed/forest_plot_data/ood_validation/ood_validation_metadata.json",
        ood_val_every_n_epochs=1,
        ood_val_band_config_path="src/evaluation/configs/raster/veg_structure_3band_v2.json",
    )


def main():
    """Main training function."""

    # ====== Command-line arguments ======
    parser = argparse.ArgumentParser(description="Train multimodal raster prediction model")
    parser.add_argument(
        "--bands",
        type=int,
        default=3,
        choices=sorted(BAND_PRESETS),
        help="Number of output bands (3 or 13). Selects n_bands + target_band_indices preset.",
    )
    args = parser.parse_args()

    # ====== Configuration ======
    config = build_config(bands=args.bands)

    # ====== Data Paths ======
    train_data_path = "data/processed/model_data_veg_structure/precomputed_training_tiles_raster_32bit.pt"
    val_data_path = "data/processed/model_data_veg_structure/precomputed_validation_tiles_raster_32bit.pt"

    # ====== Output Directory ======
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    modality_str = ""
    if config.use_naip and config.use_uavsar:
        modality_str = "fused"
    elif config.use_naip:
        modality_str = "naip"
    elif config.use_uavsar:
        modality_str = "uavsar"
    else:
        modality_str = "baseline"

    output_dir = f"data/output/raster_model_{modality_str}_{args.bands}band_{timestamp}"

    # ====== Training Hyperparameters ======
    num_epochs = 100
    batch_size = 80  # Batch size per GPU
    learning_rate = 4.5e-3  # AdamWScheduleFree takes a higher learning rate than regular AdamW (does not update on checkpoint)
    weight_decay = 0.5  # Weight regularization
    beta1 = 0.95  # AdamW momentum (exponential moving average of gradients)
    beta2 = 0.999  # AdamW momentum (exponential moving average of squared gradients)
    gradient_accumulation_steps = 1  # Gradient accumulation for effective larger batches
    max_grad_norm = 12  # prevent large gradient updates
    save_every_n_epochs = 5  # Save checkpoint every N epochs
    use_amp = True  # Automatic mixed precision (bfloat16)
    # Patience is counted in OOD-eval units when the metric is an ood_* metric:
    # e.g., patience=20 with ood_val_every_n_epochs=5 == 100 actual epochs of
    # no OOD improvement before stopping.
    early_stopping_patience = 50
    early_stopping_metric = "mse" #"ood_overall_mean_tsmae" #trimmed, standardized mae
    warmup_steps_percentage = 0.05
    seed = 42
    num_gpus = None  # None = use all available GPUs


    # ====== Resume Training from Checkpoint (Optional) ======
    # Set this to resume training from a checkpoint (loads model weights + optimizer state)
    # This is different from checkpoint_path in config (which only loads weights for transfer learning)
    # Note: Use best_model.pth (has correct epoch) instead of final_model.pth (had a bug with epoch number)
    resume_checkpoint_path = None #"data/output/raster_cross_attn_grid_mlp_sweep_20260418_223806/fd_0p2__hcw_0p01__uavsar_off__wd_0p4/checkpoints/fd_0p2__hcw_0p01__uavsar_off__wd_0p4__epoch_65.pth"

    # ====== Print Configuration ======
    print("=" * 80)
    print("RASTER MODEL TRAINING")
    print("=" * 80)
    print(f"Modalities: {modality_str}")
    print(f"Output directory: {output_dir}")
    print(f"Training data: {train_data_path}")
    print(f"Validation data: {val_data_path}")
    print(f"\nHyperparameters:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size per GPU: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Beta1: {beta1}")
    print(f"  Beta2: {beta2}")
    print(f"  Warmup steps: {warmup_steps_percentage*100:.1f}% of total")
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Early stopping patience: {early_stopping_patience}"
          + (f" (× ood_val_every_n_epochs={config.ood_val_every_n_epochs} "
             f"= {early_stopping_patience * config.ood_val_every_n_epochs} epochs)"
             if early_stopping_metric.startswith('ood_') else ""))
    print(f"  Early stopping metric: {early_stopping_metric}")
    if config.ood_val_enabled:
        print(f"  OOD validation: every {config.ood_val_every_n_epochs} epochs on "
              f"{config.ood_val_tiles_path}")
    print(f"  Use AMP: {use_amp}")
    print(f"  Save every N epochs: {save_every_n_epochs}")
    print(f"  Seed: {seed}")
    if resume_checkpoint_path:
        print(f"\n  RESUMING FROM: {resume_checkpoint_path}")
    print(f"\nModel configuration:")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  KNN neighbors: {config.k}")
    print(f"  Point-attn dropout: {config.pt_attn_dropout:.1%}")
    print(f"  Point-block dropout: {config.pt_block_dropout:.1%}")
    print(f"  Target bands: {config.target_band_indices}")
    print(f"  Grid size: {config.grid_size}×{config.grid_size}")
    print(f"  Distance sigma: {config.raster_distance_sigma}m (Gaussian weighting)")
    print(f"\nRaster head configuration:")
    print(f"  Attention dropout: {config.raster_attention_dropout:.1%}")
    print(f"  FFN dropout: {config.raster_ffn_dropout:.1%}")
    print(f"  Decoder dropout: {config.raster_decoder_dropout:.1%}")
    print(f"\nStochastic depth (DropPath):")
    print(f"  Encoder drop path: {config.encoder_drop_path:.1%}")
    print(f"  Extractor point attn drop path: {config.extractor_point_attn_drop_path:.1%}")
    print(f"\nLoss configuration:")
    print(f"  Huber delta: {config.huber_delta}")
    print(f"  Correlation loss weight: {config.correlation_loss_weight}")
    print(f"  Heteroscedastic (GNLL) loss: {config.use_heteroscedastic_loss}")
    if config.use_heteroscedastic_loss:
        print(f"    Minimum variance: {config.heteroscedastic_min_var}")
        print(f"    Overconfidence penalty weight: {config.heteroscedastic_overconfidence_weight}")
        print(f"  Loss = GNLL + {config.heteroscedastic_overconfidence_weight} * overconf_penalty + {config.correlation_loss_weight} * (1 - pearson_r)")
    else:
        print(f"  Loss = Huber(delta={config.huber_delta}) + {config.correlation_loss_weight} * (1 - pearson_r)")
    print(f"  (MSE also logged to TensorBoard for comparison)")
    print(f"\nTransfer learning:")
    print(f"  LR multiplier: {config.transfer_lr_multiplier}")
    print(f"\nRegularization (OOD Robustness):")
    print(f"  Spectral normalization: {config.use_spectral_norm}")
    print(f"  SWA enabled: {config.swa_enabled}")
    if config.swa_enabled:
        print(f"    SWA start epoch: {config.swa_start_epoch}")
        print(f"    SWA update frequency: every {config.swa_update_freq} epoch(s)")
    print("=" * 80)

    # ====== Start Training ======
    train_raster_model(
        config=config,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_every_n_epochs=save_every_n_epochs,
        use_amp=use_amp,
        early_stopping_patience=early_stopping_patience,
        early_stopping_metric=early_stopping_metric,
        seed=seed,
        num_gpus=num_gpus,
        beta1=beta1,
        beta2=beta2,
        max_grad_norm=max_grad_norm,
        warmup_steps_percentage=warmup_steps_percentage,
        resume_checkpoint_path=resume_checkpoint_path
    )

    print("\nTraining complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
