# Source Code

Source code for multimodal fusion of sparse 3DEP LiDAR, NAIP optical imagery, and UAVSAR L-band SAR.

Two pipelines share the encoder backbone:

- **Raster vegetation-structure prediction (active).** Predicts a multi-band vegetation-structure raster per tile using the standardized metric family of Moudry et al. (2023). Entry point: `run_raster_model.py`.
- **Point cloud upsampling (published historical).** Predicts dense UAV-quality point clouds from sparse 3DEP. Published in *Remote Sensing* (2025). Entry points: `run_model_test.py`, `run_ablation_study.py`.

## Directory Structure

### `data_prep/`
Data acquisition, preprocessing, and training-data generation. STAC catalog builders for NAIP, UAVSAR, 3DEP, and UAV LiDAR; tile grid creation; per-tile data extraction (raster and point-cloud variants); train/val/test split; precompute. Called from `scripts/` shell pipelines.

### `models/`
Shared encoder and raster decoder. Includes the multimodal point-attention model (LG-PAB), ViT image encoders for NAIP / UAVSAR, cross-attention fusion, the raster decoder (learnable grid queries + distance-biased cross-attention + MLP head), and online GPU augmentation.

### `training/`
Training loops with DDP support. `raster_training.py` is the active loop (heteroscedastic Gaussian NLL / Huber, MC Dropout, SWA, spectral norm, online augmentation). `multimodal_training.py` is the historical PC-upsampling loop. Datasets and DDP utilities live here too.

### `evaluation/`
Inference, forest-plot evaluation (4 OOD sites), 3DEP-only baseline (same Moudry metric pipeline applied to sparse 3DEP for isolating fusion value-add), band configs, and statistical/figure tooling from the published paper.

### `utils/`
Core utilities: vegetation-structure metric computation (`compute_vegetation_structure_metrics`), Chamfer distance (published PC loss), GPU KNN graphs, point cloud helpers.

### `raster_mapping/`
Forest plot visualization helpers.

### `fuel_metrics/` (legacy)
LidarForFuel Python wrappers and orchestration. Was the initial raster target before the pivot to Moudry vegetation-structure metrics. Code retained; may be revisited.

## Legacy and Unused Code

Each subdirectory may contain:
- `legacy/` — superseded implementations (mostly from the PC-upsampling era).
- `unused_alternatives/` — explored but not used in the published work.

## Entry Points

All entry points live at the repo root:

- `run_raster_model.py` — train the active raster model.
- `run_raster_cross_attn_grid_mlp_sweep.py` — architecture/hyperparameter sweep.
- `run_pretrain_image_encoders.py` — pretrain NAIP / UAVSAR encoders before the main raster training.
- `run_model_test.py` — train a single point-cloud upsampling model (historical).
- `run_ablation_study.py` — published PC-upsampling ablations (baseline / NAIP / UAVSAR / fused).

---

See [../README.md](../README.md) for the public-facing project overview and [../CLAUDE.md](../CLAUDE.md) for internal project memory.
