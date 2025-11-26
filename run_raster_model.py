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
        pos_mlp_hdn=16,
        pt_attn_dropout=0.01,

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
        fusion_dropout=0.02,
        max_dist_ratio=5.0,  # Updated: 1.5m → 5.0m (matches raster_radius for consistency)
        position_encoding_dim=24,  # Must be divisible by 6 for 3D positions

        # Raster-specific parameters
        # Target bands (0-indexed):
        #   Index 11 = Band 12 (Canopy_cover): Raw range 0.000-0.999, Mean=0.520, Std=0.325, Units: Fraction (0-1)
        #   Index 7  = Band 8  (TFL):          Raw range 0.000-39.5,  Mean=4.029, Std=4.189, Units: kg/m²
        n_bands=2,  # Canopy_cover, TFL
        target_band_indices=[11, 7],  # 0-indexed: Bands 12, 8
        grid_size=5,
        tile_extent=10.0,
        raster_num_heads=8,
        raster_radius=5.0,  # Updated: 3.0m → 5.0m to prevent empty grid cell NaN
        raster_hidden_dim=512,
        raster_decoder_layers=4,  # Number of decoder MLP layers (tunable: 3/4/5)
        raster_dropout=0.02,

        # Pre-aggregation LG-PAB refinement parameters
        num_pre_agg_blocks=1,  # Number of pre-aggregation LG-PAB blocks (0-5 configurable)
        pre_agg_lcl_heads=4,  # Local attention heads for pre-aggregation
        pre_agg_glbl_heads=4,  # Global attention heads for pre-aggregation
        pre_agg_dropout=0.02,  # Dropout for pre-aggregation blocks
        pre_agg_k_neighbors=15,  # KNN neighbors for pre-aggregation

        # Optional: Load from checkpoint
        checkpoint_path=None,
        layers_to_load=None,
        layers_to_freeze=None
    )

    # ====== Data Paths ======
    train_data_path = "data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt"
    val_data_path = "data/processed/model_data_raster/precomputed_validation_tiles_raster_32bit.pt"
    augmented_data_path = "data/processed/model_data_raster/augmented_tiles_raster_32bit.pt" 

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
    num_epochs = 100
    batch_size = 60  # Batch size per GPU
    learning_rate = 3.5e-3  # Updated: 5e-4 → 3e-3 for normalized raster data
    weight_decay = 0.0001  # L2 regularization
    beta1 = 0.90  # AdamW momentum (exponential moving average of gradients)
    beta2 = 0.999  # AdamW momentum (exponential moving average of squared gradients)
    gradient_accumulation_steps = 1  # Gradient accumulation for effective larger batches
    max_grad_norm = 10.0  # Gradient clipping threshold (L2 norm)
    save_every_n_epochs = 10  # Save checkpoint every N epochs
    use_amp = True  # Automatic mixed precision (bfloat16)
    early_stopping_patience = 20  # Epochs without improvement before stopping
    warmup_steps_percentage = 0.05  # Linear LR warmup for 5% of total training steps
    seed = 42
    num_gpus = None  # None = use all available GPUs

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
    print(f"  Use AMP: {use_amp}")
    print(f"  Save every N epochs: {save_every_n_epochs}")
    print(f"  Seed: {seed}")
    print(f"\nModel configuration:")
    print(f"  Feature dim: {config.feature_dim}")
    print(f"  KNN neighbors: {config.k}")
    print(f"  Target bands: {config.target_band_indices}")
    print(f"  Grid size: {config.grid_size}×{config.grid_size}")
    print(f"  Raster radius: {config.raster_radius}m")
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
        seed=seed,
        num_gpus=num_gpus,
        beta1=beta1,
        beta2=beta2,
        max_grad_norm=max_grad_norm,
        warmup_steps_percentage=warmup_steps_percentage
    )

    print("\nTraining complete!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
