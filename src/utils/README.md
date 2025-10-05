# Utilities

Core utility functions for point cloud processing and metrics.

## Active Files

- `chamfer_distance.py` - Point cloud reconstruction metric
  - GPU-accelerated Chamfer distance computation using PyTorch3D
  - Bidirectional nearest neighbor distance between predicted and ground truth point clouds
  - Primary evaluation metric used throughout training and evaluation

- `knn_graph_gpu.py` - GPU-accelerated KNN graph generation
  - Efficient k-nearest neighbor graph construction on GPU
  - Used by model for local attention computation
  - Precomputed during data preparation to accelerate training

- `point_cloud_utils.py` - Point cloud processing utilities
  - Various helper functions for point cloud manipulation
  - Normalization, transformation, and sampling utilities

## Subfolders

### `unused_alternatives/`

Alternative approaches and utilities not used in published work:

- `infocd.py` - InfoCD loss function with repulsion term
  - **Purpose:** Alternative to Chamfer Distance for point cloud reconstruction
  - **Features:** Information-theoretic Chamfer Distance (Lin et al., NeurIPS 2023) with softmax temperature and repulsion loss
  - **Why unused:** Chamfer Distance proved sufficient for published work; InfoCD adds complexity without significant improvement for this application

- `training_data_eval.py` - Training data quality evaluation and filtering
  - **Purpose:** Filter invalid training tiles based on point count thresholds and shape validation
  - **Functions:** `filter_invalid_shapes()` for detecting empty/malformed point clouds, `load_tile_from_h5()` for inspecting HDF5 tiles
  - **Why unused:** Quality filtering now handled in `train_test_split_and_precompute.py` during data preparation

- `dtm_calc.py` - Digital Terrain Model (DTM) calculation utilities
  - **Purpose:** Generate DTMs from point clouds using various interpolation methods
  - **Methods:** Raster minimum-z gridding, Progressive Morphological Filter, kriging
  - **Why unused:** Published work focuses on point cloud reconstruction, not DTM generation

- `octree_downsampling.py` - Octree-based point cloud downsampling
  - **Purpose:** Hierarchical octree downsampling for reducing point cloud density
  - **Why unused:** Anisotropic voxel downsampling (in `train_test_split_and_precompute.py`) used instead

---

See [../../README.md](../../README.md) for complete workflow documentation.
