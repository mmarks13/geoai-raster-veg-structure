# Training

Training infrastructure with distributed data parallel (DDP) support for multi-GPU training.

## Active Files

- `multimodal_training.py` - Main training loop and ablation study orchestration
  - `train_multimodal_model()`: Train a single model configuration
  - `run_ablation_studies()`: Run all ablation experiments (baseline, NAIP, UAVSAR, fused)
  - Implements DDP training, gradient accumulation, mixed precision
  - Handles data loading with sharded datasets
  - Checkpoint saving and validation loss tracking

- `ddp_training.py` - Internal DDP utility module
  - Provides: `setup_ddp()`, `cleanup()`, `find_free_port()`, `monitor_gpu_stats()`, `ModelConfig`
  - Helper functions for distributed training setup
  - Not user-facing - imported by `multimodal_training.py`

- `optuna.py` - Hyperparameter tuning with Optuna framework
  - Not used in published workflow but available for future optimization

## Entry Points

Training is initiated through root-level scripts:

### Run Ablation Study

```bash
python run_ablation_study.py
```

Runs all four ablation experiments:
1. Baseline (LiDAR only)
2. LiDAR + NAIP
3. LiDAR + UAVSAR
4. LiDAR + NAIP + UAVSAR (fused)

### Train Single Model

```bash
python run_model_test.py
```

Trains a single model configuration with custom hyperparameters.

## Training Environment

**Hardware:** 4x NVIDIA L40 GPUs (48GB each)
**Batch Size:** 15 tiles per GPU (60 total)
**Training Time:** ~7 hours per model
**CUDA Version:** 12.4
**Optimizer:** ScheduleFreeAdamW (base lr: 5e-4, weight-decay: 1e-4, no external scheduler)
**Loss Function:** Density-aware Chamfer distance (α=4)
**Evaluation Metric:** Standard Chamfer distance

**Data:**
- Input: `data/processed/model_data/precomputed_training_tiles_32bit.pt`
- Input: `data/processed/model_data/augmented_tiles_32bit_16k_no_repl.pt`
- Validation: `data/processed/model_data/precomputed_validation_tiles_32bit.pt`

**Output:**
- Checkpoints: `data/output/checkpoints/`
- Best model selected by lowest validation loss (not final epoch)

## Subfolders

### `legacy/`

Superseded training scripts replaced by current DDP infrastructure:

- `train.py` - Early training script with hard-coded data paths
  - **Data:** `augmented_training_tiles_60k.pt` (old dataset)
  - **Replaced by:** `run_ablation_study.py` and `run_model_test.py` (root-level entry points)

- `training.py` - Mid-stage training script with Optuna integration
  - **Model:** Uses legacy `PointUpsampler` from `src/models/model.py`
  - **Dataset:** Custom `PointCloudUpsampleDataset` class
  - **Replaced by:** `multimodal_training.py` with improved data loading and model support

- `preprocess.py` - Point cloud normalization utilities
  - **Functions:** `normalize_pair()`, `normalize_pair_with_bbox()` for point cloud normalization
  - **Replaced by:** Normalization now handled in `train_test_split_and_precompute.py` during data preparation

- `run_model_test.py` - Duplicate training entry point
  - **Replaced by:** Root-level `run_model_test.py` (current version)

---

See [../../README.md](../../README.md) for complete workflow documentation.
