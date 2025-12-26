"""
Training entry point for multimodal raster prediction model.

Usage:
    python run_raster_model.py

This script trains a single raster prediction model with default hyperparameters.
For ablation studies or custom configurations, modify the config parameters below.
"""

import torch
from src.models.multimodal_raster_model import MultimodalRasterConfig
from src.training.raster_training import train_raster_model
from pathlib import Path
import datetime


def main():
    """Main training function."""

    # ====== Configuration ======
    config = MultimodalRasterConfig(
        # Model architecture
        k=15,
        feature_dim=512,
        pt_attn_dropout=0.02,

        # Feature extractor heads
        extractor_lcl_heads=4,
        extractor_glbl_heads=4,

        # Modality selection (set to True to enable)
        use_naip=True,
        use_uavsar=True,

        # Image encoder parameters
        img_embed_dim=256,
        img_num_patches=16,
        naip_dropout=0.02,
        uavsar_dropout=0.02,
        temporal_encoder="gru",

        # Fusion parameters
        fusion_type="cross_attention",
        fusion_num_heads=4,
        fusion_dropout=0.05,
        max_dist_ratio=8.0,  # Distance ratio in meters for cross-attention masking

        # Position Encoders
        position_encoding_dim=96,  # Must be divisible by 6 for 3D positions

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
        n_bands=8,
        target_band_indices=[0, 1, 2, 3, 4, 5, 6, 7],
        grid_size=5,
        tile_extent=10.0,
        raster_num_heads=8,
        # MULTI-SCALE ATTENTION: Per-head sigma values
        # 2 heads @ σ=0.5m (very local), 4 heads @ σ=2.0m (medium), 2 heads @ σ=5.0m (wide)
        raster_distance_sigma=[0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0],
        raster_hidden_dim=512,
        raster_decoder_layers=4,
        raster_attention_dropout=0.1,
        raster_decoder_dropout=0.5,
        raster_use_wide_decoder=True,  

        # Pre-aggregation LG-PAB refinement parameters
        num_pre_agg_blocks=2,  # Number of pre-aggregation LG-PAB blocks (0-5 configurable)
        pre_agg_lcl_heads=0,  # Local attention heads for pre-aggregation
        pre_agg_glbl_heads=4,  # Global attention heads for pre-aggregation
        pre_agg_dropout=0.2,  # Dropout for pre-aggregation blocks
        pre_agg_k_neighbors=15,  # KNN neighbors for pre-aggregation

        # Optional: Transfer learning (only loads weights, not optimizer state)
        checkpoint_path=None,
        layers_to_load=None,
        layers_to_freeze=None,

        # Training-time dropout (applied during training only, not validation)
        # Edge dropout: randomly mask KNN edges for sparse-robustness
        training_edge_dropout=0.0,  # 0% edge dropout
        # Modality dropout: randomly drop modalities per-tile for modality-robustness
        training_modality_dropout_naip=0.1,  # 0% NAIP dropout
        training_modality_dropout_uavsar=0.2,  # 0% UAVSAR dropout

        # Stochastic depth (DropPath) - regularization for residual connections
        encoder_drop_path=0.0,      # Drop path for image encoder TransformerBlocks
        decoder_drop_path=0.3,      # Drop path for WideRasterDecoder (linearly increasing per block)
        extractor_point_attn_drop_path=0.0,  # Drop path for feature extractor (keep stable)
        pre_agg_point_attn_drop_path=0.3,    # Drop path for pre-aggregation refinement blocks

        # Huber loss - robust to outliers in fuel metrics
        huber_delta=1.0,  # Delta threshold (errors > delta use linear penalty)

        # Correlation loss weight (addresses variance collapse)
        # Total loss = MSE + correlation_loss_weight * (1 - pearson_r)
        # Higher values encourage model to preserve variance in predictions
        correlation_loss_weight=0.3,

        # ===============================================================
        # GPU Training Augmentation (Kornia + Custom PyTorch)
        # ===============================================================
        # Applied during training only (disabled in model.eval())
        # See docs/training_augmentation.md for full documentation

        training_augmentation_enabled=True,  # Master switch for all augmentations

        # --- Point Cloud Augmentation ---
        # Coordinate jitter: adds Gaussian noise to x,y,z coords
        # Physical effect: simulates point position uncertainty (~0.1m)
        aug_coord_jitter_sigma=0.02,    # Noise std in z-score units (~0.1m physical)
        aug_coord_jitter_prob=0.33,      # Probability of applying jitter per tile

        # Intensity noise: adds Gaussian noise to intensity values
        # Physical effect: simulates sensor noise and atmospheric effects
        aug_intensity_noise_sigma=0.05,  # Noise std in z-score units
        aug_intensity_noise_prob=0.2,    # Probability per tile

        # Intensity outliers: randomly replaces intensity values with extreme values
        # Physical effect: simulates sensor saturation, multipath returns
        aug_intensity_outlier_prob=0.002,  # Per-point probability of outlier

        # Bird simulation: adds extreme z-offset to 1 random point
        # Physical effect: simulates bird/drone flyover returns in LiDAR
        aug_bird_outlier_prob=0.003,           # Probability per tile
        aug_bird_z_offset_range=(5.0, 15.0),  # Z-score offset (≈25-75m physical)

        # --- NAIP Augmentation (4-channel optical: RGBN) ---
        # Gaussian noise: simulates sensor noise
        aug_naip_noise_sigma=0.03,
        aug_naip_noise_prob=0.2,

        # Gaussian blur: simulates atmospheric haze, focus issues
        aug_naip_blur_kernel=3,
        aug_naip_blur_sigma=(0.1, 2.0),
        aug_naip_blur_prob=0.1,

        # Motion blur: simulates aircraft motion, wind effects
        aug_naip_motion_blur_kernel=5,
        aug_naip_motion_blur_angle=(-45.0, 45.0),
        aug_naip_motion_blur_prob=0.05,

        # Random erasing: simulates cloud shadows, occlusions
        aug_naip_erasing_scale=(0.02, 0.15),  # Erased area as fraction of image
        aug_naip_erasing_prob=0.1,

        # Sharpness: simulates varying focus quality
        aug_naip_sharpness_range=(0.5, 1.5),
        aug_naip_sharpness_prob=0.1,

        # Histogram equalization: simulates varying exposure/contrast
        aug_naip_equalize_prob=0.1,

        # --- UAVSAR Augmentation (6-channel SAR: polarimetric) ---
        # Gaussian noise: simulates thermal/system noise (valid in dB domain)
        aug_uavsar_noise_sigma=0.05,
        aug_uavsar_noise_prob=0.2,

        # Gaussian blur: simulates multi-looking (speckle filtering)
        aug_uavsar_blur_kernel=3,
        aug_uavsar_blur_sigma=(0.1, 1.5),
        aug_uavsar_blur_prob=0.1,

        # Motion blur: simulates platform motion effects
        aug_uavsar_motion_blur_kernel=3,
        aug_uavsar_motion_blur_angle=(-30.0, 30.0),
        aug_uavsar_motion_blur_prob=0.05,

        # Random erasing: simulates RFI (radio freq interference), dropouts
        aug_uavsar_erasing_scale=(0.02, 0.10),
        aug_uavsar_erasing_prob=0.05,
    )

    # ====== Data Paths ======
    train_data_path = "data/processed/model_data_veg_structure/precomputed_training_tiles_raster_32bit.pt"
    val_data_path = "data/processed/model_data_veg_structure/precomputed_validation_tiles_raster_32bit.pt"
    augmented_data_path = "data/processed/model_data_veg_structure/augmented_tiles_raster_32bit.pt" 

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

    output_dir = f"data/output/raster_model_{modality_str}_{timestamp}"

    # ====== Training Hyperparameters ======
    num_epochs = 120
    batch_size = 30  # Batch size per GPU
    learning_rate = 2e-3  # AdamWScheduleFree takes a higher learning rate than regular AdamW (does not update on checkpoint)
    weight_decay = 0.01  # Weight regularization
    beta1 = 0.95  # AdamW momentum (exponential moving average of gradients)
    beta2 = 0.999  # AdamW momentum (exponential moving average of squared gradients)
    gradient_accumulation_steps = 2  # Gradient accumulation for effective larger batches
    max_grad_norm = 3  # prevent large gradient updates
    save_every_n_epochs = 2  # Save checkpoint every N epochs
    use_amp = True  # Automatic mixed precision (bfloat16)
    early_stopping_patience = 200  # Epochs without improvement before stopping
    early_stopping_metric = "loss"  # Metric to monitor for early stopping
    warmup_steps_percentage = 0.15
    seed = 42
    num_gpus = None  # None = use all available GPUs



    # ====== Resume Training from Checkpoint (Optional) ======
    # Set this to resume training from a checkpoint (loads model weights + optimizer state)
    # This is different from checkpoint_path in config (which only loads weights for transfer learning)
    # Note: Use best_model.pth (has correct epoch) instead of final_model.pth (had a bug with epoch number)
    resume_checkpoint_path = None #"data/output/raster_model_fused_20251226_065021/checkpoints/epoch_2.pth"


    # ====== Print Configuration ======
    print("=" * 80)
    print("RASTER MODEL TRAINING")
    print("=" * 80)
    print(f"Modalities: {modality_str}")
    print(f"Output directory: {output_dir}")
    print(f"Training data: {train_data_path}")
    print(f"Validation data: {val_data_path}")
    print(f"Augmented data: {augmented_data_path}")
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
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Early stopping metric: {early_stopping_metric}")
    print(f"  Use AMP: {use_amp}")
    print(f"  Save every N epochs: {save_every_n_epochs}")
    print(f"  Seed: {seed}")
    if resume_checkpoint_path:
        print(f"\n  RESUMING FROM: {resume_checkpoint_path}")
    print(f"\nModel configuration:")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  KNN neighbors: {config.k}")
    print(f"  Target bands: {config.target_band_indices}")
    print(f"  Grid size: {config.grid_size}×{config.grid_size}")
    print(f"  Distance sigma: {config.raster_distance_sigma}m (Gaussian weighting)")
    print(f"\nRaster head configuration:")
    print(f"  Wide decoder: {config.raster_use_wide_decoder}")
    print(f"  Decoder layers: {config.raster_decoder_layers}")
    print(f"  Attention dropout: {config.raster_attention_dropout:.1%}")
    print(f"  Decoder dropout: {config.raster_decoder_dropout:.1%}")
    print(f"\nTraining-time dropout:")
    print(f"  Edge dropout: {config.training_edge_dropout:.1%}")
    print(f"  NAIP modality dropout: {config.training_modality_dropout_naip:.1%}")
    print(f"  UAVSAR modality dropout: {config.training_modality_dropout_uavsar:.1%}")
    print(f"\nStochastic depth (DropPath):")
    print(f"  Encoder drop path: {config.encoder_drop_path:.1%}")
    print(f"  Decoder drop path: {config.decoder_drop_path:.1%}")
    print(f"  Extractor point attn drop path: {config.extractor_point_attn_drop_path:.1%}")
    print(f"  Pre-agg point attn drop path: {config.pre_agg_point_attn_drop_path:.1%}")
    print(f"\nLoss configuration:")
    print(f"  Huber delta: {config.huber_delta}")
    print(f"  Correlation loss weight: {config.correlation_loss_weight}")
    print(f"  Loss = Huber(delta={config.huber_delta}) + {config.correlation_loss_weight} * (1 - pearson_r)")
    print(f"  (MSE also logged to TensorBoard for comparison)")
    print("=" * 80)

    # ====== Start Training ======
    train_raster_model(
        config=config,
        train_data_path=train_data_path,
        val_data_path=val_data_path,
        output_dir=output_dir,
        augmented_data_path=augmented_data_path,
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
