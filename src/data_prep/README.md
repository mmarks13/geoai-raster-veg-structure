# Data Preparation

Data acquisition, preprocessing, and training-data generation using STAC-based workflows. Supports both the **raster vegetation-structure pipeline (active)** and the **point-cloud upsampling pipeline (historical)**.

## STAC catalog creation (shared)

- `make_local_naip_stac.py` — NAIP optical imagery from Microsoft Planetary Computer.
- `make_local_uavsar_stac.py` — UAVSAR L-band SAR from Alaska Satellite Facility.
- `make_local_3dep_stac.py` — 3DEP LiDAR. Supports building a catalog from the pre-processed 3DEP HAG features (`--mode processed --input-dir data/processed/3dep_hag_features`).
- `make_local_uavlidar_stac.py` — catalog UAV LiDAR ground truth from local `.las/.laz`.

## 3DEP HAG + enhanced features

- `download_3dep_for_sites.py`, `download_and_process_3dep_sites.py` — pull 3DEP tiles from Planetary Computer and run the PDAL pipeline that adds Height Above Ground, Planarity, Sphericity, Verticality, and writes a COPC. Driven by `scripts/process_3dep_hag_features.sh`.

## Raster pipeline (active)

- `create_tile_grid.py` — pixel-aligned 10 m tile grid with 20 % overlap, per site.
- `generate_training_data_raster.py` — per-tile extraction of 3DEP points, NAIP stacks, UAVSAR stacks, and the vegetation-structure target raster.
- `train_test_split_and_precompute_raster.py` — spatial train/val split, two-stage coord normalization (bbox → z-score), precompute tile dicts. Writes global normalization stats.
- `data_augmentation_raster.py` — offline augmented-tile generation (complement to online GPU augmentation during training).
- `validate_preprocessed_raster.py`, `validate_raster_training_data.py` — sanity checks.
- `create_forest_plot_tile_grid.py`, `preprocess_forest_plots_for_inference.py`, `build_ood_validation_set.py` — tiles and inputs for the 4 OOD forest-plot validation sites.

Driven by `scripts/process_data_raster_v2.sh`.

## Point cloud pipeline (historical)

- `generate_training_data.py` — generate 10 m × 10 m training tiles (point-cloud targets).
- `train_test_split_and_precompute.py` — split + precompute KNN graphs + normalize (single-stage bbox normalization only).
- `data_augmentation.py` — augmented tile generation.

Driven by `scripts/process_data.sh`.

## Supporting utilities

- `process_uav_lidar.py` — process raw UAV LiDAR files.
- `create_training_tile_bboxes.py` — tile bounding box generator (used by the historical pipeline).
- `h5_chunk_loader.py` — combine HDF5 training data chunks into a single `.pt`.
- `imagery_stac.py`, `imagery_training_data.py` — imagery STAC loading and per-tile extraction helpers.
- `las_to_copc_stac.py` — convert LAS to COPC and create STAC entries.
- `process_pointcloud_stac.py` — point-cloud STAC gridding/aggregation utilities.
- `pointcloud_footprints_to_geojson.py` — export point cloud footprints.
- `bbox_tile_filter.py` — filter tiles by bbox region.
- `compress_las.py` — `.las` → `.laz`.

## Subfolders

### `legacy/`
Superseded implementations (split and precompute were previously separate scripts, etc.).

### `unused_alternatives/`
- `uavsar_to_stac.py` — builds a STAC from pre-downloaded UAVSAR; the active workflow downloads directly from ASF.
- `wv2_to_stac.py` — WorldView-2 ingestion; the project uses NAIP instead.

## Typical raster workflow

```bash
bash scripts/get_data.sh                       # STAC catalogs (shared)
bash scripts/process_3dep_hag_features.sh      # 3DEP HAG + geometric features
bash scripts/veg_structure_metrics/run_all_sites.sh   # UAV-LiDAR ground truth rasters
bash scripts/process_data_raster_v2.sh         # Tile grid → extraction → split → precompute
```

---

See [../../README.md](../../README.md) and [../../CLAUDE.md](../../CLAUDE.md).
