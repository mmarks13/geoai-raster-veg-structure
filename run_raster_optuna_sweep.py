"""
Optuna-based hyperparameter optimization for raster fuel metrics prediction model.

Usage:
    python run_raster_optuna_sweep.py

This script runs Bayesian hyperparameter optimization using Optuna to search over:
- Memory-safe model configs: Bundles (feature_dim, img_embed_dim,
  batch_size, grad_accum) to ensure GPU memory safety and fair comparison (constant
  effective batch size ≈ 60-80)
- Regularization: raster_attention_dropout, raster_decoder_dropout
- Training: learning_rate, weight_decay, correlation_loss_weight

Fixed parameters match the current best config in run_raster_model.py:
- Multi-scale attention (8 heads with σ=[0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0])
- Target bands: [15, 21] (TFL, Cover)
- 15 epochs with early stopping (patience=5)
"""

import optuna
from optuna.pruners import MedianPruner
import joblib
import datetime
from src.models.multimodal_raster_model import MultimodalRasterConfig
from src.training.raster_training import train_raster_model


def objective(trial):
    """
    Optuna objective function - defines search space and returns validation loss.

    Args:
        trial: Optuna trial object for sampling hyperparameters

    Returns:
        float: Best validation MSE loss from training (to minimize)
    """

    # ====== Sample Hyperparameters from Search Space ======

    # Memory-safe model configurations (model_size, batch_size, grad_accum)
    # All configs target effective batch size ≈ 60 samples
    # Format: (feature_dim, img_embed_dim, batch_size, grad_accum_steps)
    model_config = trial.suggest_categorical("model_config", [
        # # Medium-Large model
        # (512, 256, 15, 4),   # Effective: 60

        # Large model
        # (768, 384, 10, 6),   # Effective: 60
        (768, 384, 6, 10),   # Effective: 60

        # # Extra-Large model
        # (1024, 512, 6, 10)  # Effective: 60
    ])

    # Unpack configuration
    feature_dim, img_embed_dim, batch_size, gradient_accumulation_steps = model_config

    # Architecture - Attention Heads
    extractor_lcl_heads = trial.suggest_categorical("extractor_lcl_heads", [4, 8, 12])
    extractor_glbl_heads = trial.suggest_categorical("extractor_glbl_heads", [4, 8, 12])
    fusion_num_heads = 4

    # Raster Head - Multi-scale attention
    raster_num_heads = 8  # Fixed: multi-scale requires 8 heads

    # Raster Dropout (separate for attention and decoder)
    raster_attention_dropout = 0.05
    raster_decoder_dropout = trial.suggest_categorical("raster_decoder_dropout", [0.2, 0.3, 0.4])

    # FIXED Feature Extractor Regularization
    pt_attn_dropout = 0.01
    naip_dropout = 0.01
    uavsar_dropout = 0.01
    fusion_dropout = 0.01

    # Training Hyperparameters to Search
    learning_rate = 0.002 #trial.suggest_categorical("learning_rate", [0.002, 0.003, 0.004])
    weight_decay = 0.0001

    # Correlation loss weight (addresses variance collapse)
    correlation_loss_weight = 1.0

    # Note: batch_size and gradient_accumulation_steps come from model_config above


    # ====== Create Model Config ======
    config = MultimodalRasterConfig(
        # SAMPLED architecture
        feature_dim=feature_dim,
        img_embed_dim=img_embed_dim,
        extractor_lcl_heads=extractor_lcl_heads,
        extractor_glbl_heads=extractor_glbl_heads,
        raster_num_heads=raster_num_heads,
        fusion_num_heads=fusion_num_heads,

        # SAMPLED regularization
        pt_attn_dropout=pt_attn_dropout,
        naip_dropout=naip_dropout,
        uavsar_dropout=uavsar_dropout,
        fusion_dropout=fusion_dropout,
        raster_attention_dropout=raster_attention_dropout,
        raster_decoder_dropout=raster_decoder_dropout,

        # SAMPLED loss configuration
        correlation_loss_weight=correlation_loss_weight,

        # Fixed spatial/data params (updated to match current best config)
        k=15,
        pos_mlp_hdn=16,
        use_naip=True,
        use_uavsar=True,
        img_num_patches=16,
        temporal_encoder="gru",
        max_dist_ratio=8.0,
        position_encoding_dim=48,
        n_bands=2,
        target_band_indices=[15, 21],  # TFL, Cover (updated from [11, 7])
        grid_size=5,
        tile_extent=10.0,
        # MULTI-SCALE ATTENTION: Per-head sigma values (requires 8 heads)
        # 2 heads @ σ=0.5m (very local), 4 heads @ σ=2.0m (medium), 2 heads @ σ=5.0m (wide)
        raster_distance_sigma=[0.5, 0.5, 2.0, 2.0, 2.0, 2.0, 5.0, 5.0],
    )


    # ====== Data Paths ======
    train_data_path = "data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt"
    val_data_path = "data/processed/model_data_raster/precomputed_validation_tiles_raster_32bit.pt"


    # ====== Training Hyperparameters ======
    num_epochs = 35
    beta1 = 0.90
    beta2 = 0.999
    # gradient_accumulation_steps comes from model_config (already set above)
    max_grad_norm = 5.0
    save_every_n_epochs = 2
    use_amp = True
    early_stopping_patience = 5
    early_stopping_metric = "loss"  # Optimize MSE loss, not MAE
    warmup_steps_percentage = 0.05
    seed = 42
    num_gpus = None


    # ====== Output Directory ======
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/output/raster_model_optuna_trial{trial.number:03d}_{timestamp}"


    # ====== Train Model ======
    effective_batch_size = batch_size * gradient_accumulation_steps
    print(f"\n{'='*80}")
    print(f"TRIAL {trial.number}")
    print(f"{'='*80}")
    print(f"Memory Configuration:")
    print(f"  Model size: {feature_dim}D (img={img_embed_dim})")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {effective_batch_size}")
    print(f"\nArchitecture:")
    print(f"  raster_num_heads: {raster_num_heads} (multi-scale)")
    print(f"  extractor_lcl_heads: {extractor_lcl_heads}")
    print(f"  extractor_glbl_heads: {extractor_glbl_heads}")
    print(f"\nRegularization:")
    print(f"  raster_attention_dropout: {raster_attention_dropout:.3f}")
    print(f"  raster_decoder_dropout: {raster_decoder_dropout:.3f}")
    print(f"\nTraining Hyperparameters:")
    print(f"  learning_rate: {learning_rate:.4e}")
    print(f"  weight_decay: {weight_decay:.4e}")
    print(f"  correlation_loss_weight: {correlation_loss_weight}")
    print(f"\nOutput directory: {output_dir}")
    print(f"{'='*80}\n")

    try:
        best_val_loss = train_raster_model(
            config=config,
            train_data_path=train_data_path,
            val_data_path=val_data_path,
            output_dir=output_dir,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_grad_norm=max_grad_norm,
            save_every_n_epochs=save_every_n_epochs,
            use_amp=use_amp,
            early_stopping_patience=early_stopping_patience,
            early_stopping_metric=early_stopping_metric,
            warmup_steps_percentage=warmup_steps_percentage,
            seed=seed,
            num_gpus=num_gpus
        )
        return best_val_loss
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"\n{'='*80}")
            print(f"Trial {trial.number} failed with OOM - pruning")
            print(f"{'='*80}\n")
            raise optuna.TrialPruned()
        else:
            raise
    except Exception as e:
        # Catch any other exceptions (NCCL timeout, ProcessExitedException, etc.)
        error_msg = str(e)
        print(f"\n{'='*80}")
        print(f"Trial {trial.number} failed with exception: {type(e).__name__}")
        print(f"Error: {error_msg[:200]}")
        print(f"Pruning this trial and continuing sweep...")
        print(f"{'='*80}\n")
        raise optuna.TrialPruned()


def main():
    """Run Optuna hyperparameter optimization."""

    print("=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)
    print("Starting Bayesian optimization with automatic pruning")
    print("=" * 80)
    print()

    # Create study with pruning (automatically stops bad trials early)
    study = optuna.create_study(
        direction="minimize",  # Minimize validation loss
        pruner=MedianPruner(
            n_startup_trials=3,  # Don't prune first 3 trials
            n_warmup_steps=15      # Evaluate 5 epochs before considering pruning (15 epoch runs)
        ),
        study_name="raster_model_hyperparameter_sweep"
    )

    # Run optimization
    n_trials = 10  # Number of trials to run
    n_jobs = 1     # Parallel trials (1=sequential, 2=2 trials in parallel, etc.)

    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Print results
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print(f"\nTotal trials: {len(study.trials)}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best validation MSE loss: {study.best_value:.4f}")

    print("\n" + "-" * 80)
    print("BEST HYPERPARAMETERS:")
    print("-" * 80)
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:.4e}")
        else:
            print(f"  {key:20s}: {value}")

    # Save study for later analysis
    study_file = "optuna_study_hyperparameter_sweep.pkl"
    joblib.dump(study, study_file)
    print(f"\n{'='*80}")
    print(f"Study saved to: {study_file}")
    print(f"Load with: study = joblib.load('{study_file}')")
    print(f"Visualize: optuna.visualization.plot_param_importances(study)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
