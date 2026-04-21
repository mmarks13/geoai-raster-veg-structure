# Utilities

Core utility functions for point clouds, metrics, and raster ground-truth generation.

## Active files

- `point_cloud_utils.py` — point cloud helpers and the vegetation-structure metric computation.
  - `compute_vegetation_structure_metrics()` — computes the standardized metric raster from Moudry et al. (2023): max/mean/std height, canopy cover, canopy/mid-story/understory density, foliage height diversity (Shannon–Wiener), height percentiles, and per-layer density proportions.
  - Used both for generating UAV-LiDAR ground truth (`scripts/veg_structure_metrics/`) and the 3DEP-only baseline (`src/evaluation/compute_3dep_baseline_metrics.py`).
  - Also contains normalization / transformation / sampling utilities.
- `chamfer_distance.py` — point cloud reconstruction metric (GPU, PyTorch3D-backed). Density-aware α=4 variant for training the published PC-upsampling model; standard bidirectional variant for evaluation.
- `knn_graph_gpu.py` — GPU-accelerated k-nearest-neighbor graph construction. Used by the PC-upsampling pipeline (precomputed during data prep to accelerate training).

## Subfolders

### `unused_alternatives/`
- `infocd.py` — information-theoretic Chamfer distance (alternative training loss; not shipped).
- `training_data_eval.py` — filter invalid tiles; quality filtering now happens in `train_test_split_and_precompute*.py`.
- `dtm_calc.py` — DTM generation utilities (project focuses on vegetation structure, not DTMs).
- `octree_downsampling.py` — octree downsampling (anisotropic voxel downsampling used instead).

---

See [../../README.md](../../README.md) and [../../CLAUDE.md](../../CLAUDE.md).
