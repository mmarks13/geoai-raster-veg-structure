"""
Pre-training entry point for image encoders (NAIP or UAVSAR).

This script trains a single image encoder independently on the vegetation
structure metrics prediction task, enabling pre-training before fusion
with the point cloud encoder.

Usage:
    python run_pretrain_image_encoders.py --encoder naip
    python run_pretrain_image_encoders.py --encoder uavsar

After pre-training, combine with point cloud encoder using:
    checkpoint_path=[
        ("path/to/baseline/best_model.pth", ["feature_extractor", "raster_head"]),
        ("path/to/naip_pretrain/best_model.pth", ["encoder"]),
        ("path/to/uavsar_pretrain/best_model.pth", ["encoder"]),
    ]
"""

import argparse
import datetime
import torch

from src.models.image_encoder_pretrain_model import (
    ImageEncoderPretrainConfig,
    ImageEncoderPretrainModel,
)
from src.training.raster_training import train_raster_model

# ====== CUDA Performance Optimizations ======
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def main():
    """Main training function."""

    # ====== Command Line Arguments ======
    parser = argparse.ArgumentParser(description="Pre-train image encoders")
    parser.add_argument('--encoder', choices=['naip', 'uavsar'], required=True,
                        help='Which encoder to pre-train')
    args = parser.parse_args()

    # ====== Configuration ======
    config = ImageEncoderPretrainConfig(
        # Encoder selection (from CLI)
        encoder_type=args.encoder,

        # Encoder architecture
        img_embed_dim=64,
        img_num_patches=16,
        temporal_encoder="gru",
        naip_dropout=0.01,
        uavsar_dropout=0.0,
        encoder_drop_path=0.0,

        # Raster head
        n_bands=3,
        target_band_indices=[3, 4, 5],
        head_hidden_dims=[64, 32],
        head_dropout=0.05,
        grid_size=5,

        # Augmentation
        training_augmentation_enabled=True,
        aug_geometric_enabled=True,
        aug_rotation_prob=0.75,
        aug_reflection_prob=0.3,
        aug_temporal_enabled=True,
        aug_naip_subsample_prob=0.3,
        aug_naip_min_frames=1,
        aug_uavsar_t_subsample_prob=0.3,
        aug_uavsar_t_min_frames=1,
        aug_uavsar_g_mask_prob=0.2,
        aug_uavsar_g_min_images=1,

        # NAIP augmentation
        aug_naip_noise_sigma=0.03,
        aug_naip_noise_prob=0.10,
        aug_naip_blur_kernel=3,
        aug_naip_blur_sigma=(0.1, 2.0),
        aug_naip_blur_prob=0.10,
        aug_naip_motion_blur_kernel=5,
        aug_naip_motion_blur_angle=(-45.0, 45.0),
        aug_naip_motion_blur_prob=0.05,
        aug_naip_sharpness_range=(0.5, 1.5),
        aug_naip_sharpness_prob=0.05,
        aug_naip_equalize_prob=0.20,

        # UAVSAR augmentation
        aug_uavsar_noise_sigma=0.05,
        aug_uavsar_noise_prob=0.05,
        aug_uavsar_blur_kernel=3,
        aug_uavsar_blur_sigma=(0.1, 1.0),
        aug_uavsar_blur_prob=0.05,
        aug_uavsar_motion_blur_kernel=3,
        aug_uavsar_motion_blur_angle=(-30.0, 30.0),
        aug_uavsar_motion_blur_prob=0.05,
    )

    # ====== Data Paths ======
    train_data_path = "data/processed/model_data_veg_structure/precomputed_training_tiles_raster_32bit.pt"
    val_data_path = "data/processed/model_data_veg_structure/precomputed_validation_tiles_raster_32bit.pt"

    # ====== Output Directory ======
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/output/{args.encoder}_encoder_pretrain_{timestamp}"

    # ====== Training Hyperparameters ======
    num_epochs = 200
    batch_size = 16  # Batch size per GPU
    learning_rate = 2e-3
    weight_decay = 1e-2
    beta1 = 0.9
    beta2 = 0.999
    max_grad_norm = 1.0
    gradient_accumulation_steps = 1
    save_every_n_epochs = 10
    use_amp = True
    early_stopping_patience = 30
    early_stopping_metric = "loss"
    warmup_steps_percentage = 0.04
    seed = 42
    num_gpus = None  # None = use all available GPUs

    # ====== Print Configuration ======
    print("=" * 80)
    print(f"IMAGE ENCODER PRE-TRAINING: {args.encoder.upper()}")
    print("=" * 80)
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
    print(f"  Max grad norm: {max_grad_norm}")
    print(f"  Warmup steps: {warmup_steps_percentage*100:.1f}% of total")
    print(f"  Early stopping patience: {early_stopping_patience}")
    print(f"  Use AMP: {use_amp}")
    print(f"  Save every N epochs: {save_every_n_epochs}")
    print(f"  Seed: {seed}")
    print(f"\nModel architecture:")
    print(f"  Encoder type: {args.encoder}")
    print(f"  Embedding dim: {config.img_embed_dim}")
    print(f"  Head hidden dims: {config.head_hidden_dims}")
    print(f"  Head dropout: {config.head_dropout}")
    print(f"\nTarget bands: {config.target_band_indices}")
    print(f"  n_bands: {config.n_bands}")
    print(f"\nAugmentation: Enabled")
    print(f"  Geometric (rotation/reflection): {config.aug_geometric_enabled}")
    print(f"  Temporal subsampling: {config.aug_temporal_enabled}")
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
        resume_checkpoint_path=None,
        model_class=ImageEncoderPretrainModel,
    )

    print("\nPre-training complete!")
    print(f"Results saved to: {output_dir}")
    print(f"\nTo use in final fine-tuning, add to checkpoint_path:")
    print(f'  ("{output_dir}/checkpoints/best_model.pth", ["encoder"])')


if __name__ == "__main__":
    main()
