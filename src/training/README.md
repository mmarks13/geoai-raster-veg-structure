# Training

Training infrastructure. DDP with manual per-GPU sharding, mixed precision, TF32.

## Active: raster training

- `raster_training.py` — training loop for the active raster vegetation-structure model.
  - Primary loss: heteroscedastic Gaussian Negative Log-Likelihood with an overconfidence penalty (`gaussian_nll_loss`). Huber available behind `use_heteroscedastic_loss=False`; code default is Huber for backward compatibility and active runs override to NLL.
  - Optional correlation-loss term.
  - Stochastic Weight Averaging via `torch.optim.swa_utils` (implemented but MC Dropout at inference is the retained ensembling method).
  - Checkpointing under `data/output/raster_model_<tag>_<YYYYMMDD>_<HHMMSS>/checkpoints/`.
- `raster_dataset.py` — `ShardedRasterDataset` and a variable-size raster collate. Loads precomputed `.pt` tiles, handles the augmented-tile file alongside originals, and produces batches with a PyG-style point `batch_indices` tensor.
- `ddp_training.py` — DDP utility module: `setup_ddp()`, `cleanup()`, `find_free_port()`, `monitor_gpu_stats()`.

**Entry points** (at repo root):

```bash
python -u run_pretrain_image_encoders.py      # Pretrain NAIP/UAVSAR encoders
python run_raster_model.py                    # Main raster training
python run_raster_cross_attn_grid_mlp_sweep.py  # Architecture/hparam sweep
```

**Data (raster):**
- `data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt`
- `data/processed/model_data_raster/precomputed_validation_tiles_raster_32bit.pt`
- `data/processed/model_data_raster/augmented_tiles_raster_32bit.pt`
- Normalization stats: `coordinate_normalization_stats.json`, `fuel_metrics_normalization_stats.json` (target-band stats; legacy name).

## Historical: point cloud upsampling (published)

- `multimodal_training.py` — `train_multimodal_model()` and `run_ablation_studies()`. Density-aware Chamfer distance loss (α=4, meter-scale).
- `optuna.py` — hyperparameter tuning scaffolding (not used in published work).

**Entry points (historical):** `run_model_test.py`, `run_ablation_study.py` at repo root.

**Data (historical):** `data/processed/model_data/precomputed_*_tiles_32bit.pt`, `augmented_tiles_32bit_16k_no_repl.pt`.

## Training environment (reference)

- 4× NVIDIA L40 (48 GB)
- CUDA 12.4, PyTorch 2.5.1
- AMP mixed precision; TF32 on Ampere+
- Best model selected by lowest validation loss (not final epoch)

Specific hyperparameters (learning rate, batch size, dropout rates, epoch count, augmentation probabilities) drift between experiments — read the entry-point script for the current values.

## Subfolders

### `legacy/`
Superseded training scripts from earlier point-cloud upsampling iterations:
- `train.py`, `training.py`, `preprocess.py`, `run_model_test.py` — all replaced by `multimodal_training.py` and the root-level entry points.

---

See [../../README.md](../../README.md) and [../../CLAUDE.md](../../CLAUDE.md). See [model_data_readme.md](model_data_readme.md) for the precomputed tile data schema.
