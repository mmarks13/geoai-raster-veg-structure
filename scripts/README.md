# Data Pipeline Scripts

End-to-end shell scripts for orchestrating data acquisition and preprocessing.

## Scripts

### get_data.sh

Downloads and catalogs remote sensing data for the study regions.

**Calls:**
- `src/data_prep/make_local_uavsar_stac.py` - Downloads UAVSAR L-band SAR from Alaska Satellite Facility
- `src/data_prep/make_local_uavlidar_stac.py` - Catalogs UAV LiDAR ground truth point clouds
- `src/data_prep/make_local_naip_stac.py` - Downloads NAIP optical imagery from Microsoft Planetary Computer
- `src/data_prep/make_local_3dep_stac.py` - Downloads 3DEP LiDAR (commented out, data pre-downloaded)

**Requirements:**
- EARTHDATA credentials (username/password) for UAVSAR downloads
- Study region bounding boxes: Southern California (2 regions)
- Date range: 2014-2025

**Output:** Local STAC catalogs in `data/stac/` for each data source

---

### process_data.sh

Processes downloaded data into training-ready PyTorch files.

**Active step:**
- `src/data_prep/train_test_split_and_precompute.py` - Splits combined training data into train/val/test sets, filters by quality criteria, precomputes KNN graphs and normalized features

**Commented workflow (already completed):**
- `src/data_prep/process_uav_lidar.py` - Process raw UAV LiDAR
- `src/data_prep/create_training_tile_bboxes.py` - Generate tile bounding boxes
- `src/data_prep/generate_training_data.py` - Generate 10m×10m training tiles from STAC catalogs
- `src/data_prep/h5_chunk_loader.py` - Combine data chunks into single PyTorch file
- `src/data_prep/pointcloud_footprints_to_geojson.py` - Export footprints for visualization
- `src/data_prep/data_augmentation.py` - Generate augmented training data
- `src/data_prep/bbox_tile_filter.py` - Filter tiles by region

**Note:** Most steps are commented out as they represent the initial data preparation workflow that has been completed. The active step (`train_test_split_and_precompute.py`) can be re-run to regenerate splits with different quality criteria.

**Input:** `data/processed/model_data/combined_training_data_v3.pt`

**Output:**
- `data/processed/model_data/precomputed_training_tiles_32bit.pt`
- `data/processed/model_data/precomputed_validation_tiles_32bit.pt`
- `data/processed/model_data/precomputed_test_tiles_32bit.pt`

---

### compress_las_files.sh

Utility script for compressing LAS point cloud files to LAZ format using `laszip`.

**Purpose:** Reduces storage requirements for raw UAV LiDAR data.

**Input:** `uavlidar/original_las/*.las`

**Output:** `uavlidar/original_las/compressed/*.laz`

**Usage:** Run once to compress raw data files.

---

## Typical Workflow

1. **First time setup:**
   ```bash
   ./scripts/get_data.sh          # Download all remote sensing data
   ./scripts/compress_las_files.sh # (Optional) Compress raw LiDAR
   ```

2. **Data processing** (most steps already completed):
   ```bash
   ./scripts/process_data.sh      # Run train/test split and precomputation
   ```

3. **Training:** See root-level `run_ablation_study.py` or `run_model_test.py`

---

See [../README.md](../README.md) for complete workflow documentation.
