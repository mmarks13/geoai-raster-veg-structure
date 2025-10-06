# Data Preparation

Data acquisition, preprocessing, and training data generation using STAC-based workflows.

## Main Workflow

Scripts called by `scripts/get_data.sh` and `scripts/process_data.sh` to create training-ready PyTorch files.

### Data Acquisition (STAC Catalog Creation)

- `make_local_naip_stac.py` - Query Planetary Computer for NAIP optical imagery, download full-resolution COGs, crop to bounding boxes, create local STAC catalog
- `make_local_uavsar_stac.py` - Download UAVSAR L-band SAR data from Alaska Satellite Facility, create local STAC catalog
- `make_local_3dep_stac.py` - Download 3DEP LiDAR point clouds from Planetary Computer, create local STAC catalog
- `make_local_uavlidar_stac.py` - Catalog UAV LiDAR ground truth point clouds from local files

###Training Data Generation

- `generate_training_data.py` - Generate 10m×10m training tiles by querying STAC catalogs for each tile geometry, combining UAV LiDAR, 3DEP LiDAR, UAVSAR, and NAIP data into HDF5 files
- `train_test_split_and_precompute.py` - Split combined training data into train/val/test sets based on spatial polygons, apply quality filters (minimum points, coverage), precompute KNN graphs and normalized features, save as PyTorch files

### Data Augmentation

- `data_augmentation.py` - Generate augmented training tiles using geometric transformations and point perturbations

## Supporting Utilities

Internal utilities called by main workflow scripts:

- `process_uav_lidar.py` - Process raw UAV LiDAR files
- `create_training_tile_bboxes.py` - Generate tile bounding box geometries
- `h5_chunk_loader.py` - Combine HDF5 training data chunks into single PyTorch file
- `pointcloud_footprints_to_geojson.py` - Export point cloud footprints to GeoJSON for visualization
- `bbox_tile_filter.py` - Filter tiles by bounding box region
- `imagery_stac.py` - STAC utilities for imagery data loading (imported by generate_training_data.py)
- `imagery_training_data.py` - Imagery data extraction for training tiles
- `las_to_copc_stac.py` - Convert LAS files to Cloud-Optimized Point Cloud (COPC) format and create STAC entries
- `process_pointcloud_stac.py` - Point cloud processing utilities (gridding, aggregation)
- `compress_las.py` - Compress .las files to .laz format

## Subfolders

### `legacy/`

Superseded implementations replaced by improved versions:

- `split_train_test_val_tiles.py` - Splits PyTorch tiles into train/val/test sets based on spatial polygons with quality filters
  - **Replaced by:** `train_test_split_and_precompute.py` (combines splitting and precomputation in one script)

- `precompute_data.py` - Precomputes KNN graphs and normalized features for training tiles
  - **Replaced by:** `train_test_split_and_precompute.py` (combines splitting and precomputation in one script)

### `unused_alternatives/`

Alternative approaches explored but not used in published work:

- `uavsar_to_stac.py` - Creates STAC catalog for UAVSAR products from local COG files
  - **Why unused:** Alternative workflow that assumes pre-downloaded UAVSAR data; published workflow uses `make_local_uavsar_stac.py` which downloads directly from Alaska Satellite Facility

- `wv2_to_stac.py` - Processes WorldView-2 satellite imagery with orthorectification and TOA reflectance conversion
  - **Why unused:** WorldView-2 is a commercial high-resolution optical alternative to NAIP; published work uses NAIP (freely available from Planetary Computer)

## Workflow

1. **Download remote sensing data:**
   ```bash
   ./scripts/get_data.sh
   ```
   Calls: `make_local_naip_stac.py`, `make_local_uavsar_stac.py`, `make_local_uavlidar_stac.py`

2. **Generate and process training tiles:**
   ```bash
   ./scripts/process_data.sh
   ```
   Calls: `generate_training_data.py`, `h5_chunk_loader.py`, `train_test_split_and_precompute.py`

3. **Output:**
   - `data/processed/model_data/precomputed_training_tiles_32bit.pt`
   - `data/processed/model_data/precomputed_validation_tiles_32bit.pt`
   - `data/processed/model_data/precomputed_test_tiles_32bit.pt`

---

See [../../README.md](../../README.md) for complete workflow documentation.
