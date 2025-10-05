# Source Code

Source code for the multimodal LiDAR point cloud enhancement model using Local-Global Point Attention Blocks (LG-PAB).

## Directory Structure

### `data_prep/`
Data acquisition, preprocessing, and training data generation. Downloads remote sensing data via STAC catalogs, generates 10m×10m training tiles, splits into train/val/test sets, and precomputes KNN graphs and normalized features. Called by `scripts/get_data.sh` and `scripts/process_data.sh`.

### `models/`
Model architecture implementation including the main multimodal LG-PAB model, Vision Transformer encoders for NAIP/UAVSAR imagery, and cross-attention fusion modules.

### `training/`
Training infrastructure with distributed data parallel (DDP) support. Main training loop, ablation study orchestration, and utility functions for multi-GPU training.

### `evaluation/`
Inference, statistical analysis, and figure generation. Runs model inference on test sets, generates evaluation dataframes, performs statistical tests for research questions, and creates manuscript figures.

### `utils/`
Core utility functions for point cloud processing, Chamfer distance metric computation, and GPU-accelerated KNN graph generation.

## Legacy and Unused Code

Each subdirectory may contain:
- `legacy/` - Superseded implementations replaced by improved versions
- `unused_alternatives/` - Alternative approaches explored but not used in published work

See individual folder READMEs for details.

## Entry Points

Model training is initiated through root-level scripts:
- `run_ablation_study.py` - Runs all ablation experiments (baseline, NAIP-only, UAVSAR-only, fused)
- `run_model_test.py` - Trains a single model configuration

Both scripts:
1. Import `MultimodalModelConfig` from `src/models/multimodal_model`
2. Import training functions from `src/training/multimodal_training`
3. Load precomputed data from `data/processed/model_data/`
4. Configure model parameters and training hyperparameters
5. Save checkpoints to `data/output/checkpoints/`

---

See [../README.md](../README.md) for complete workflow documentation.
