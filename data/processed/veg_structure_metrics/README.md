# Phase 2: Vegetation Structure Metrics Rasters

**Goal:** Create vegetation structure metrics rasters for 5 UAV LiDAR study sites using `compute_vegetation_structure_metrics()`.

## Sites Processed

| Site | Input File | Max HAG Filter | Raster Size |
|------|-----------|----------------|-------------|
| t01_t09 | `T01-T09_LIDAR_20231025_Pre_LAS.las` | 25m | 213×275 |
| t03_t13 | `T03-T13_LIDAR_20231025_Pre_LAS.las` | 25m | 171×187 |
| t06_t14 | `T06-T14_LIDAR_20231025_Pre_LAS.las` | 25m | 228×243 |
| trex | `TRex_20231025_LAS.las` | 25m | 406×428 |
| volcan_mtn | `VolcanMt_20231025_LAS.las` | 60m | 715×1040 |

## Output Structure

```
data/processed/veg_structure_metrics/<site_name>/
└── merged/
    ├── <site_name>_veg_metrics_2m.tif   # 24-band GeoTIFF
    └── <site_name>_visualization.png    # 6-panel figure
```

## Output Format

**24-band GeoTIFF** at 2m resolution:
- Bands 0-22: Moudry et al. (2023) vegetation structure metrics
- Band 23: Point count per pixel

**Minimum point count filtering:** Pixels with <20% of median point count set to NaN (bands 0-22).

## Processing Commands

```bash
# Small sites
bash scripts/veg_structure_metrics/process_single_site.sh \
    "data/raw/uavlidar/study_las/<input>.las" "<site_name>" 25

# Volcan Mountain (large file, taller trees)
bash scripts/veg_structure_metrics/process_single_site.sh \
    "data/raw/uavlidar/full_volcan_mtn_las/VolcanMt_20231025_LAS.las" "volcan_mtn" 60
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `resolution` | 2.0m | Standard for landscape analysis |
| `density_range` | (0, 25)m | Fixed for cross-site comparability |
| `point_filter_max_hag` | 25m or 60m | Site-specific outlier removal |
| `min_points_per_pixel` | Auto (20% median) | Quality filtering |

## Visualization

6-panel figures: Max Height, Canopy Cover, Foliage Height Diversity, Canopy Density, Median Height, Understory Density

- **Gray** = NaN (no data / filtered)
- **Colormap** = Valid measurements

## Code Locations

- Function: `src/utils/point_cloud_utils.py::compute_vegetation_structure_metrics()`
- Scripts: `scripts/veg_structure_metrics/`
- Visualization: `src/veg_structure_metrics/visualize_metrics.py`
- Phase 1 docs: `docs/vegetation_structure_metrics.md`
