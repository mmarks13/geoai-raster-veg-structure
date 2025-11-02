# Fuel Metrics Module

Wildfire fuel hazard mapping from UAV LiDAR using the LidarForFuel R package.

## Overview

This module provides tools for computing fuel metrics from UAV LiDAR point clouds using physics-based Beer-Lambert radiative transfer modeling. It generates 173-band rasters containing:

- **23 summary metrics**: Canopy height, canopy base height (CBH), fuel strata gap (FSG), fuel loads, cover percentages, vertical complexity index (VCI), entropy
- **150 bulk density layers**: Vertical fuel density profile at 1.5m resolution

## Quick Start

### 1. Installation

First, install the R environment and LidarForFuel package:

```bash
# Create R environment (one-time setup)
conda env create -f environment_r_fuel_metrics.yml

# Install LidarForFuel package
bash scripts/fuel_metrics/install_lidarforfuel.sh
```

### 2. Run Complete Pipeline

Process any UAV LiDAR file with a single command:

```bash
bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
  --input data/raw/uavlidar/my_site.las \
  --output-name my_site \
  --species "Mixed" \
  --resolution 5.0
```

This will:
1. Ground classify the point cloud (SMRF filter)
2. Tile into 200m × 200m chunks
3. Pretreatment (normalization + trait attribution)
4. Compute fuel metrics for each tile
5. Merge into seamless mosaic
6. Generate visualization

## Module Components

### Python Scripts

- **`lidarforfuel_wrapper.py`**: Python-R interface for LidarForFuel
- **`process_fuel_metrics.py`**: Main orchestration for single tiles
- **`batch_processing.py`**: Parallel batch processing with progress tracking
- **`visualize_bounds.py`**: Spatial coverage validation
- **`visualize_metrics.py`**: Fuel metrics visualization

### Shell Scripts (in `scripts/fuel_metrics/`)

- **`run_fuel_metrics_pipeline.sh`**: Main entry point (complete pipeline)
- **`run_batch_fuel_metrics.sh`**: Batch processing wrapper
- **`pdal/run_ground_classification_and_tiling.sh`**: Consolidated PDAL pipeline
- **`install_lidarforfuel.sh`**: R package installation script

### R Scripts (in `scripts/fuel_metrics/r/`)

- **`run_pretreatment.R`**: Wrapper for fPCpretreatment
- **`run_fuel_metrics.R`**: Wrapper for fCBDprofile_fuelmetrics

## Data Organization

Outputs are organized by site name:

```
data/processed/fuel_metrics/<site_name>/
├── tiles/              # Ground-classified LAZ tiles (200m × 200m)
├── pretreated/         # Normalized LAZ with LMA/WD attributes
├── rasters/            # 173-band fuel metric TIFFs (per tile)
├── merged/             # Seamless mosaic + visualization
├── logs/               # Processing logs and summary CSV
└── validation/         # Spatial coverage checks
```

## Species Trait Lookup

Leaf Mass Area (LMA) and Wood Density (WD) values by species:

| Species | LMA (canopy) | WD (canopy) | LMA (understory) | WD (understory) |
|---------|--------------|-------------|------------------|-----------------|
| Mixed   | 140 g/m²     | 591 kg/m³   | 130 g/m²         | 550 kg/m³       |
| Coast live oak | 111 g/m² | 825 kg/m³ | 130 g/m² | 550 kg/m³ |
| Black oak | 108 g/m² | 562 kg/m³ | 130 g/m² | 550 kg/m³ |
| Ceanothus | 208 g/m² | 600 kg/m³ | 208 g/m² | 600 kg/m³ |
| Coulter pine | 207 g/m² | 400 kg/m³ | 130 g/m² | 550 kg/m³ |
| Incense cedar | 208 g/m² | 380 kg/m³ | 130 g/m² | 550 kg/m³ |

See `data/processed/fuel_metrics/trait_lookup.csv` for complete table.

## Advanced Usage

### Process Individual Steps

```bash
# Step 1: Ground classification + tiling only
bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
  input.las output/tiles/ 200 10

# Step 2: Batch fuel metrics computation
bash scripts/fuel_metrics/run_batch_fuel_metrics.sh \
  output/tiles/ output/ Mixed 5.0 6

# Step 3: Merge tiles
conda run -p /home/jovyan/geoai_env gdal_merge.py \
  -o output/merged/fuel_metrics.tif \
  -a_nodata nan -co COMPRESS=LZW -co TILED=YES \
  --optfile <(find output/rasters -name "*.tif" | sort)
```

### Single Tile Processing

```bash
python src/fuel_metrics/process_fuel_metrics.py \
  --input tiles/tile_0_1cm.laz \
  --output_dir output/ \
  --species "Coast live oak" \
  --resolution 1.0 \
  --export_mode full  # 173 bands
```

### List Available Species

```bash
python src/fuel_metrics/process_fuel_metrics.py --list_species
```

## Output Metrics

### Summary Metrics (Bands 1-23)

1. Canopy height (H, m)
2. Canopy base height (CBH, m)
3. Fuel strata gap (FSG, m)
4. Canopy fuel load (kg/m²)
5. Total fuel load (kg/m²)
6. Midstorey fuel load (kg/m²)
7. Surface fuel load (kg/m²)
8. Canopy cover (%)
9. Midstorey cover (%)
10. Understory cover (%)
11. Vertical complexity index (VCI)
12. Shannon entropy
13-23. Additional structural metrics

### Bulk Density Profile (Bands 24-173)

Fuel bulk density (kg/m³) at 150 vertical layers (1.5m resolution).

## Technical Details

### Ground Classification

Uses SMRF (Simple Morphological Filter) via PDAL with default parameters:
- Cell size: 1.0m
- Slope: 0.15
- Threshold: 0.5m
- Window: 18.0m

### Computational Requirements

- **RAM**: ~4GB per tile for pretreatment, ~8GB for fuel metrics
- **Time**: ~2-5 minutes per tile (depends on point density)
- **Disk**: ~100MB per tile (pretreated LAZ + 173-band raster)

### Parallel Processing

Default: 6 parallel jobs. Adjust based on available cores and RAM:
- 8 cores / 64GB RAM → 6 jobs
- 16 cores / 128GB RAM → 12 jobs
- 4 cores / 32GB RAM → 3 jobs

## Troubleshooting

### NULL from fPCpretreatment
**Cause**: Tile contains only ground points (no vegetation)
**Solution**: Normal for edge tiles; check coverage validation

### Memory errors
**Cause**: Insufficient RAM for large tiles
**Solution**: Reduce tile size (e.g., 100m instead of 200m)

### Missing bulk density layers
**Cause**: `--export_mode summary` (23 bands only)
**Solution**: Use `--export_mode full` for all 173 bands

## References

- **LidarForFuel**: https://github.com/oliviermartin7/lidarforfuel
- **Beer-Lambert Model**: Martin-Ducup & Pimont (2024)
- **SMRF Filter**: Pingel et al. (2013)

## License

See main repository LICENSE file.

## Output Band Reference

### Summary Mode (23 Bands + 150 Profile)

**IMPORTANT:** Band 3 contains the threshold value (e.g., 0.02 kg/m³), so metric indices are offset by +1 from naive expectations.

| Band | Name | Description | Unit |
|------|------|-------------|------|
| 1 | Profil_Type | Vertical profile classification (1-5) | - |
| 2 | Profil_Type_L | Labeled profile type (simplified) | - |
| 3 | **threshold** | **CBD threshold used for filtering** | **kg/m³** |
| 4 | Height | Canopy height (max vegetation) | m |
| 5 | CBH | Canopy Base Height | m |
| 6 | FSG | Fuel Strata Gap (largest discontinuity) | m |
| 7 | Top_Fuel | Highest vegetation above threshold | m |
| 8 | H_Bush | Understory height (CBH - FSG) | m |
| 9 | continuity | Binary flag (1=continuous, 0=gaps) | - |
| 10 | VCI_PAD | Vertical Complexity Index (PAD-based) | - |
| 11 | VCI_lidr | Vertical Complexity Index (lidR-based) | - |
| 12 | entropy_lidr | Shannon entropy (lidR method) | - |
| 13 | PAI_tot | Total Plant Area Index | m²/m² |
| 14 | CBD_max | Maximum Canopy Bulk Density | kg/m³ |
| 15 | CFL | Canopy Fuel Load | kg/m² |
| 16 | TFL | Total Fuel Load | kg/m² |
| 17 | MFL | Midstory Fuel Load | kg/m² |
| 18 | FL_1_3 | Fuel Load 1-3m height | kg/m² |
| 19 | GSFL | Gap Strata Fuel Load | kg/m² |
| 20 | FL_0_1 | Surface Fuel Load (0-1m) | kg/m² |
| 21 | FMA | Fuel Mass Area | g/m² |
| 22 | date | Acquisition date (GPS time) | - |
| 23 | Cover | Total cover percentage | % |
| 24-25 | Cover_4, Cover_6 | Cover at 4m, 6m thresholds | % |
| 26-175 | CBD_1 to CBD_150 | Bulk density profile (150 layers) | kg/m³ |

### Visualization Bands

The default visualization (`visualize_metrics.py`) plots:
- Band 4: Height
- Band 5: CBH  
- Band 6: FSG
- Band 16: TFL
- Band 15: CFL
- Band 23: Cover

