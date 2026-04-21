# Data Pipeline Scripts

End-to-end shell scripts for data acquisition, preprocessing, ground-truth generation, and evaluation.

## Shared acquisition

### `get_data.sh`
Downloads and catalogs remote sensing data for the study regions.
- NAIP optical (Planetary Computer)
- UAVSAR L-band SAR (Alaska Satellite Facility — requires EarthData credentials)
- 3DEP LiDAR (Planetary Computer)
- UAV LiDAR (cataloged from `data/raw/uavlidar/study_las/`)

**Output:** local STAC catalogs in `data/stac/`.

## Raster pipeline (active)

### `process_data_raster_v2.sh`
Full raster-pipeline preprocessing: tile grid → per-tile 3DEP/NAIP/UAVSAR + vegetation-structure-target extraction → spatial train/val split → two-stage coord normalization (bbox → z-score) → precompute. Writes `data/processed/model_data_raster/precomputed_*_tiles_raster_32bit.pt` and normalization stats JSONs.

### `process_3dep_hag_features.sh`
Runs the PDAL pipeline that adds Height Above Ground, Planarity, Sphericity, and Verticality to each 3DEP site and writes a COPC-format output. Enables efficient per-tile spatial queries during tile extraction. Output: `data/processed/3dep_hag_features/<site>/<site>_hag_features.copc.laz`.

### `veg_structure_metrics/`
Generates UAV-LiDAR ground-truth metric rasters (Moudry et al.).
- `run_all_sites.sh` — process all training sites.
- `process_single_site.sh` — per-site: ground classification (SMRF), HAG, outlier filtering, metric computation at 2 m resolution, output GeoTIFF + visualization.
- `process_large_site.sh` — variant for larger-extent sites.
- `pdal/` — PDAL pipeline templates.

### `evaluate_forest_plots.sh`
Forest plot evaluation on the 4 OOD sites (BluffMesa, NorthBigBear, ReyesPeak, Laguna). Supports multi-GPU inference and MC-Dropout sampling.
```bash
bash scripts/evaluate_forest_plots.sh \
    --model <checkpoint.pth> \
    --band-config src/evaluation/configs/raster/<config>.json \
    --multi-gpu \
    --mc-samples <N> \
    --batch-size <B>
```

## Point cloud pipeline (historical)

### `process_data.sh`
Preprocessing for the published point-cloud upsampling model: tile generation, spatial split, KNN-graph precompute, single-stage bbox normalization.

## Utility

### `compress_las_files.sh`
Compress `.las` → `.laz` with `laszip`. Reduces storage of raw UAV LiDAR.

## Legacy: fuel metrics

### `fuel_metrics/`
LidarForFuel (R) pipeline scripts: ground classification + tiling in a single PDAL pass, pretreatment, per-tile fuel metric computation, merging. The project pivoted away from this target; code is retained for possible revisit. See `src/fuel_metrics/README.md` for operational specifics.

---

See [../README.md](../README.md) and [../CLAUDE.md](../CLAUDE.md).
