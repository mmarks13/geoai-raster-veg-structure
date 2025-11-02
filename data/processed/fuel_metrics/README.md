# Fuel Metrics Pipeline

Complete workflow for processing UAV LiDAR point clouds into wildfire fuel hazard raster maps using the [LidarForFuel](https://github.com/oliviermartin7/LidarForFuel) R package. Generates comprehensive fuel characterization including bulk density profiles, canopy structure, and standardized fuel layers.

---

## Overview

This pipeline integrates LidarForFuel into a Python-based workflow through thin R wrappers while maintaining parity with the published methods (Martin-Ducup & Pimont, 2024).

**Workflow:**
```
Raw UAV LiDAR → Ground Classification + Tiling → Pretreatment → Fuel Metrics → Merge → Visualization
```

**Data Flow:**
```
input.las (unclassified point cloud)
    ↓
[PDAL: SMRF ground classification + tiling in single pass]
    ↓
tiles/tile_*.laz (ground-classified tiles, 200m × 200m)
    ↓
[LidarForFuel: Pretreatment - fPCpretreatment]
    ↓
pretreated/tile_*_pretreated.laz (normalized, with LMA/WD attributes)
    ↓
[LidarForFuel: Fuel Metrics - fCBDprofile_fuelmetrics]
    ↓
rasters/tile_*_fuel_metrics.tif (173 bands per tile)
    ↓
[GDAL Merge]
    ↓
merged/my_site_fuel_metrics_5m.tif (seamless mosaic)
    ↓
[Visualization]
    ↓
merged/my_site_visualization.png (6-panel figure)
```

**Key Outputs:**
- **23 summary metrics**: Canopy height, CBH, fuel loads, cover, vertical complexity
- **150 bulk density layers**: Vertical distribution profile (1m resolution by default)

**Note:** PDAL ground classification is enabled by default to handle unclassified UAV LiDAR. LidarForFuel's built-in classification has a bug where it checks for ground points before performing classification. Use `--no-pdal-classification` to disable if your data is already classified.

---

## Quick Start

### Single Command (Recommended)

Process any UAV LiDAR file with the main orchestrator script:

```bash
bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
  --input data/raw/uavlidar/my_site.las \
  --output-name my_site \
  --species "Mixed" \
  --resolution 5.0 \
  --tile-size 200 \
  --parallel-jobs 6
```

This runs the complete pipeline automatically:
1. Ground classification (SMRF filter)
2. Tiling (200m × 200m chunks)
3. Pretreatment (normalization + trait attribution)
4. Fuel metrics computation (173-band rasters)
5. Merge into seamless mosaic
6. Generate visualization

**Output:** `data/processed/fuel_metrics/my_site/` with organized subdirectories

---

## Installation

### One-Time Setup

1. **Create R environment:**
```bash
conda env create -f environment_r_fuel_metrics.yml
```

2. **Install LidarForFuel package:**
```bash
bash scripts/fuel_metrics/install_lidarforfuel.sh
```

This installs:
- R base packages (lidR, terra, sf, remotes)
- Rfast (CRAN)
- VoxR (CRAN, auto-installed)
- lidarforfuel (GitHub)

3. **Verify installation:**
```bash
conda run -n r_fuel_metrics R -e "library(lidarforfuel); packageVersion('lidarforfuel')"
```

The Python wrapper automatically uses the `r_fuel_metrics` environment via `conda run`, so you don't need to manually activate it.

---

## Step-by-Step Workflow

### Step 1: Ground Classification + Tiling

**Script:** `scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh`

**What it does:**
- Classifies ground points using SMRF (Simple Morphological Filter)
- Tiles point cloud into manageable chunks (default: 200m × 200m)
- Saves ground-classified tiles with 1cm precision
- **Single I/O pass** (no intermediate large temporary file)

**Usage:**
```bash
bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
  data/raw/uavlidar/my_site.las \
  data/processed/fuel_metrics/my_site/tiles \
  200 \
  10
```

**Parameters:**
- `input_las`: Path to input LAS/LAZ file (classified or unclassified)
- `output_tiles_dir`: Directory for output tiles
- `tile_size`: Tile size in meters (default: 200)
- `buffer`: Buffer overlap in meters (default: 10)

**SMRF Parameters** (PDAL defaults):
```json
{
  "type": "filters.smrf",
  "cell": 1.0,
  "slope": 0.15,
  "threshold": 0.5,
  "window": 18.0
}
```

**Output:**
- `tiles/tile_0_1cm.laz`, `tile_1_1cm.laz`, ... (ground-classified)
- Expected tiles: ~75 for 200m spacing on 1.4km × 2.1km area

---

### Step 2: Pretreatment + Fuel Metrics (Batch)

**Script:** `scripts/fuel_metrics/run_batch_fuel_metrics.sh`

**What it does:**
- Processes all tiles in parallel (default: 6 jobs)
- For each tile:
  1. **Pretreatment** (fPCpretreatment):
     - Normalize height above ground
     - Add LMA (Leaf Mass Area) attribute
     - Add WD (Wood Density) attribute
     - Filter by height (<60m)
  2. **Fuel Metrics** (fCBDprofile_fuelmetrics):
     - Compute 23 summary metrics
     - Compute 150 bulk density layers
     - Output 173-band GeoTIFF
- Tracks progress via summary CSV
- Per-tile logging for debugging

**Usage:**
```bash
bash scripts/fuel_metrics/run_batch_fuel_metrics.sh \
  data/processed/fuel_metrics/my_site/tiles \
  data/processed/fuel_metrics/my_site \
  "Mixed" \
  5.0 \
  6
```

**Parameters:**
- `tiles_dir`: Directory containing tile LAZ files
- `output_base_dir`: Base output directory
- `species`: Species name for trait lookup (see below)
- `resolution`: Output raster resolution in meters (default: 5.0)
- `parallel_jobs`: Number of parallel jobs (default: 6)
- `clumping`: Clumping factor Ω for Beer-Lambert model (default: 0.77)
- `projection_factor`: Projection factor G for fuel metrics (default: 0.5)

**Species Trait Values (TRY Database, Updated 2025-10-31):**

| Species | LMA (canopy) | WD (canopy) | LMA (understory) | WD (understory) | Source |
|---------|--------------|-------------|------------------|-----------------|--------|
| Mixed | 182 g/m² | 600 kg/m³ | 130 g/m² | 550 kg/m³ | TRY mean |
| Quercus agrifolia | 182 g/m² | 643 kg/m³ | 182 g/m² | 550 kg/m³ | TRY (n=19) |
| Quercus kelloggii | 109 g/m² | 580 kg/m³ | 109 g/m² | 550 kg/m³ | TRY (n=8) |
| Ceanothus palmeri | 130 g/m² | 550 kg/m³ | 130 g/m² | 550 kg/m³ | Literature |
| Pinus coulteri | 282 g/m² | 381 kg/m³ | 130 g/m² | 550 kg/m³ | TRY (n=11) |
| Pinus jeffreyi | 282 g/m² | 381 kg/m³ | 130 g/m² | 550 kg/m³ | TRY (n=11) |
| Calocedrus decurrens | 282 g/m² | 434 kg/m³ | 130 g/m² | 550 kg/m³ | TRY/Pinus proxy |
| Eriogonum fasciculatum | 115 g/m² | 625 kg/m³ | 115 g/m² | 625 kg/m³ | TRY (n=17) |

**Trait Value Sources:**
- **TRY Plant Trait Database** (https://www.try-db.org): Primary source for California species
- **LMA conversions**: SLA ↔ LMA using standard leaf area/mass relationship
- **Wood Density**: TRY database where available, literature where needed

**Monitoring Progress:**
```bash
# Watch summary CSV (updates every 5 seconds)
watch -n 5 'tail -10 data/processed/fuel_metrics/my_site/logs/tile_processing_summary.csv'

# Check individual tile log
tail -f data/processed/fuel_metrics/my_site/logs/tile_0_1cm.log
```

**Output:**
- `pretreated/tile_*_pretreated.laz` (normalized LAZ with traits)
- `rasters/tile_*_fuel_metrics.tif` (173 bands per tile)
- `logs/tile_processing_summary.csv` (success/failure tracking)
- `logs/tile_*.log` (per-tile processing logs)

---

### Step 3: Merge Tiles into Seamless Mosaic

**Tool:** GDAL `gdal_merge.py`

**What it does:**
- Mosaics all tile rasters into single seamless GeoTIFF
- Preserves all 173 bands
- Applies LZW compression and tiled layout
- **CRITICAL:** Uses explicit file list (not wildcards) to avoid data loss

**Usage (Recommended with --optfile):**
```bash
find data/processed/fuel_metrics/my_site/rasters -name "*.tif" | sort > /tmp/tiles.txt

conda run -p /home/jovyan/geoai_env gdal_merge.py \
  -o data/processed/fuel_metrics/my_site/merged/my_site_fuel_metrics_5m.tif \
  -a_nodata nan \
  -co COMPRESS=LZW \
  -co TILED=YES \
  -co BIGTIFF=YES \
  --optfile /tmp/tiles.txt
```

**Critical Flags:**
- `-a_nodata nan`: Set output nodata value to NaN
- **DO NOT use `-n nan`**: This ignores all NaN pixels from input (causes data loss)
- `-co COMPRESS=LZW`: LZW compression
- `-co TILED=YES`: Tiled layout for faster access
- `-co BIGTIFF=YES`: Support files >4GB
- `--optfile`: Read file list from text file (avoids shell glob expansion issues)

**⚠️ Common Pitfall:**
```bash
# ❌ WRONG (loses ~60% of data with 70+ tiles):
gdal_merge.py -o output.tif rasters/tile_*_fuel_metrics.tif

# ✓ CORRECT (preserves all data):
gdal_merge.py -o output.tif --optfile <(find rasters -name "*.tif" | sort)
```

Why: Shell glob expansion has file count limits. With 70+ tiles, only first ~28-30 files match. No error message—silently loses entire regions.

**Output:**
- `merged/my_site_fuel_metrics_5m.tif` (173 bands, seamless mosaic)

---

### Step 4: Visualization

**Script:** `src/fuel_metrics/visualize_metrics.py`

**Usage:**
```bash
conda run -p /home/jovyan/geoai_env python src/fuel_metrics/visualize_metrics.py \
  data/processed/fuel_metrics/my_site/merged/my_site_fuel_metrics_5m.tif \
  data/processed/fuel_metrics/my_site/merged/my_site_visualization.png
```

**Output Panels:**
1. Canopy Height (H)
2. Canopy Base Height (CBH)
3. Fuel Strata Gap (FSG)
4. Canopy Fuel Load
5. Total Fuel Load
6. Vertical Complexity Index (VCI)

---

## Complete Pipeline Example

### Method 1: Single Command (Recommended)

```bash
bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
  --input data/raw/uavlidar/volcan_mountain.las \
  --output-name volcan_mountain \
  --species "Mixed" \
  --resolution 5.0 \
  --tile-size 200 \
  --parallel-jobs 6 \
  --clumping 0.77 \
  --projection-factor 0.5
```

### Method 2: Individual Steps

```bash
# 1. Ground classification + tiling (single pass)
bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
  data/raw/uavlidar/volcan_mountain.las \
  data/processed/fuel_metrics/volcan_mountain/tiles \
  200 \
  10

# 2. Batch pretreatment + fuel metrics
bash scripts/fuel_metrics/run_batch_fuel_metrics.sh \
  data/processed/fuel_metrics/volcan_mountain/tiles \
  data/processed/fuel_metrics/volcan_mountain \
  "Mixed" \
  5.0 \
  6 \
  0.77 \
  0.5

# 3. Merge tiles (after all tiles complete)
find data/processed/fuel_metrics/volcan_mountain/rasters -name "*.tif" | sort > /tmp/tiles.txt

conda run -p /home/jovyan/geoai_env gdal_merge.py \
  -o data/processed/fuel_metrics/volcan_mountain/merged/volcan_mountain_fuel_metrics_5m.tif \
  -a_nodata nan \
  -co COMPRESS=LZW \
  -co TILED=YES \
  -co BIGTIFF=YES \
  --optfile /tmp/tiles.txt

# 4. Generate visualization
conda run -p /home/jovyan/geoai_env python src/fuel_metrics/visualize_metrics.py \
  data/processed/fuel_metrics/volcan_mountain/merged/volcan_mountain_fuel_metrics_5m.tif \
  data/processed/fuel_metrics/volcan_mountain/merged/volcan_mountain_visualization.png
```

---

## Directory Structure

```
data/processed/fuel_metrics/
├── README.md                       # This file
├── TRAIT_VALUE_RESOURCES.md        # Database references for updating traits
├── trait_lookup.csv                # LMA/WD values by species
├── volcan_mtn/                     # Example site outputs
│   ├── tiles/                      # Ground-classified LAZ tiles
│   ├── pretreated/                 # Intermediate: pretreated LAZ files
│   ├── rasters/                    # Final: fuel metrics GeoTIFFs (173 bands)
│   ├── merged/                     # Seamless mosaic + visualization
│   ├── logs/                       # Processing logs + summary CSV
│   └── validation/                 # Spatial validation checks
└── [site_name]/                    # Additional sites follow same structure
```

---

## Output Reference

### Summary Metrics (Bands 1-23)

| Band | Name | Description | Units |
|------|------|-------------|-------|
| 1 | Profil_Type | Detailed fuel profile type (1-4) | - |
| 2 | Profil_Type_L | Simplified profile type (A-D) | - |
| 3 | Height | Canopy height (max Z) | m |
| 4 | CBH | Canopy Base Height | m |
| 5 | FSG | Fuel Strata Gap (vertical discontinuity) | m |
| 6 | VCI_PAD | Vertical Complexity Index (PAD) | - |
| 7 | VCI_CBD | Vertical Complexity Index (CBD) | - |
| 8 | TFL | Total Fuel Load (1m to top) | kg/m² |
| 9 | CFL | Canopy Fuel Load | kg/m² |
| 10 | MFL | Midstorey Fuel Load | kg/m² |
| 11 | surf_fuel_load | Surface fuel load (<1m) | kg/m² |
| 12 | Canopy_cover | Canopy cover percentage | % |
| 13 | MidStorey_cover | Midstorey cover percentage | % |
| 14 | understory_cover | Understory cover percentage | % |
| 15 | Total_cover | Total vegetation cover | % |
| 16 | Total_cover_2m | Cover above 2m | % |
| 17 | entropy_CBD | Shannon entropy of CBD profile | - |
| 18 | entropy_PAD | Shannon entropy of PAD profile | - |
| 19 | PAI | Plant Area Index (total) | - |
| 20 | PAI_upper | PAI in upper canopy | - |
| 21 | PAI_mid | PAI in midstorey | - |
| 22 | PAI_understory | PAI in understory | - |
| 23 | max_CBD | Maximum CBD value | kg/m³ |

### Bulk Density Profile (Bands 24-173)

150 bands of Canopy Bulk Density (CBD) for vertical layers:
- **Layer depth**: 1m by default (configurable via `--layer_depth`)
- **Band 24**: 0-1m layer CBD
- **Band 25**: 1-2m layer CBD
- ...
- **Band 173**: 149-150m layer CBD

**Units**: kg/m³ (kilograms of vegetation per cubic meter)

### Profile Types (Band 2: Profil_Type_L)

- **A**: Surface fuel only (no canopy)
- **B**: Canopy only (no surface fuel)
- **C**: Canopy + surface, with gap (discontinuous)
- **D**: Continuous fuel (no gap, high fire risk)

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Tile size | 200m | Grid spacing for tiling |
| Buffer | 10m | Overlap between adjacent tiles |
| Species | Mixed | Species trait lookup |
| Resolution | 5.0m | Output raster pixel size |
| Clumping (Ω) | 0.77 | Clumping factor for Beer-Lambert model |
| Projection factor (G) | 0.5 | Projection factor for fuel metrics |
| Export mode | summary | Output format (summary=23 bands, full=173 bands) |
| Parallel jobs | 6 | Concurrent tile processing jobs |
| Height filter | 60m | Maximum vegetation height to retain |
| Understory threshold | 2m | Height threshold for understory traits |

---

## Trait Values & Customization

### Where Trait Values Come From

Trait values (LMA and WD) are critical parameters that determine how the Beer-Lambert inversion converts Plant Area Density (PAD) to fuel mass density (CBD). They come from:

1. **TRY Plant Trait Database** (https://www.try-db.org)
   - Primary source for California species
   - Provides species-specific LMA and wood density measurements
   - Sample sizes documented in trait table above

2. **Global Wood Density Database** (https://doi.org/10.5061/dryad.234)
   - Zanne et al. (2009) compilation of wood density measurements

3. **Literature estimates**
   - Used for species/regions with limited TRY data
   - References documented in trait_lookup.csv

### How to Customize Trait Values

Edit `trait_lookup.csv`:

```csv
species,common_name,lma_gm2,wd_kgm3,lma_understory_gm2,wd_understory_kgm3,lma_source,wd_source,data_quality,notes
Pinus ponderosa,Ponderosa pine,150,420,130,550,TRY study,Wood density database,High,Moderate LMA
```

**Guidelines:**
- **LMA (Leaf Mass Area)**: g/m² (typical range: 50-250)
  - Evergreen sclerophylls: 120-180
  - Deciduous broadleaf: 60-120
  - Conifers: 120-200
- **WD (Wood Density)**: kg/m³ (typical range: 300-800)
  - Softwoods (pines, cedars): 350-500
  - Hardwoods (oaks): 600-800
  - Shrubs: 450-650
- **Understory values**: Often lower LMA, similar/higher WD

### External Resources

- **TRY Plant Trait Database**: https://www.try-db.org (requires registration for data access)
- **Global Wood Density Database**: Zanne et al. (2009) doi:10.5061/dryad.234
- **USFS Silvics Manual**: https://www.srs.fs.usda.gov/pubs/misc/ag_654/
- See TRAIT_VALUE_RESOURCES.md for detailed database access instructions

---

## Methodology

LidarForFuel uses a physics-based approach to derive fuel metrics from LiDAR:

### Pretreatment (fPCpretreatment)

**Normalization:**
- Height above ground (HAG) computed via DTM subtraction
- Outlier removal (noise classification)
- Height filtering (retain points <60m by default)

**Trait Attribution:**
- LMA and WD assigned to each point via lookup table by species
- Understory (<2m) can have different traits than canopy

**Outputs:**
- Normalized point cloud with attributes: `LMA`, `WD`, `Zref`, `Easting`, `Northing`, `Elevation`

### Fuel Metrics Computation (fCBDprofile_fuelmetrics)

**Beer-Lambert Inversion:**
- **Plant Area Density (PAD)**: Estimated from point density using radiative transfer model
- Formula: `PAD(z) = -ln(P_gap(z)) / ΔZ`, where `P_gap` is gap probability
- Accounts for scanning angle, clumping (ω, default 0.77), and extinction (G, default 0.5)

**Bulk Density Conversion:**
- `CBD(z) = PAD(z) × LMA / 2`
- Converts area density to mass density using leaf traits

**Fuel Metrics Derivation:**
- **CBH** (Canopy Base Height): Lowest height where CBD > threshold (0.02 kg/m³)
- **FSG** (Fuel Strata Gap): Maximum vertical gap in CBD profile
- **Fuel loads**: Integrated CBD over height intervals
- **Cover**: Fraction of pixels with vegetation at height threshold
- **VCI**: Coefficient of variation of CBD profile (vertical heterogeneity)

**Rasterization:**
- Applied via `lidR::pixel_metrics` at specified resolution (1m default)
- Minimum 400 points per pixel required for reliable estimates

---

## Troubleshooting

### Incomplete spatial coverage after merge

**Symptoms:** Merged GeoTIFF covers only ~40% of expected area

**Root Cause:** Shell glob expansion of wildcard pattern hits argument limits, processing only first ~28-30 tiles

**Solution:** Use explicit file list or --optfile:
```bash
# ❌ WRONG (loses ~60% of data):
gdal_merge.py -o output.tif data/rasters/tile_*_fuel_metrics.tif

# ✓ CORRECT (preserves all data):
gdal_merge.py -o output.tif --optfile <(find data/rasters -name "*.tif" | sort)
```

### NaN values in merged output

**Symptoms:** Large NaN regions where tiles should have data

**Cause:** Using `-n nan` flag which ignores all NaN pixels

**Solution:** Remove `-n nan`; only use `-a_nodata nan`:
```bash
# ❌ WRONG (ignores NaN pixels, creates gaps):
gdal_merge.py -o output.tif -n nan -a_nodata nan tiles/*.tif

# ✓ CORRECT (preserves data, sets nodata value):
gdal_merge.py -o output.tif -a_nodata nan tiles/*.tif
```

### Fewer output tiles than expected

**Symptoms:** Only 50 tiles generated when expecting 75

**Cause:** Edge tiles with sparse vegetation fail processing

**Solution:** This is normal. Check `tile_processing_summary.csv` for failed tiles and verify they're at dataset edges.

### Memory errors during processing

**Symptoms:** Process killed, "Killed" message in logs

**Cause:** Insufficient RAM for large tiles (>50M points per tile)

**Solution:**
1. Reduce tile size: Use 100m instead of 200m
2. Reduce parallel jobs: Use 3 instead of 6
3. Increase system swap space

**Memory requirements:**
- Pretreatment: ~4GB RAM per tile
- Fuel metrics: ~8GB RAM per tile
- Safe configuration for 64GB system: 6 parallel jobs with 200m tiles

### Missing bulk density layers in output

**Symptoms:** Output raster has only 23 bands instead of 173

**Cause:** Using `--export_mode summary` (default)

**Solution:** Use `--export_mode full` for all 173 bands (creates larger files ~2-3x)

---

## File Size Estimates

| Dataset | Size | Resolution | Extent |
|---------|------|------------|--------|
| Raw UAV LiDAR | 24 GB | 1cm point spacing | 1.4 × 2.1 km |
| Tile LAZ (200m) | 50-150 MB each | 1cm point spacing | 200 × 200 m |
| Pretreated LAZ | Similar to input | 1cm point spacing | 200 × 200 m |
| Fuel metrics TIF (summary) | 20-40 KB each | 5m pixels | 40 × 40 pixels |
| Fuel metrics TIF (full) | 60-120 KB each | 5m pixels | 40 × 40 pixels |
| Merged TIF (compressed) | 2-3 MB | 5m pixels | 1.4 × 1.5 km |

---

## Performance Benchmarks

**Volcan Mountain Dataset:**
- Input: 24GB LAS, 1.66 billion points, 1.4 × 2.1 km
- Tiles: 70 tiles (200m × 200m)
- Hardware: 8-core CPU, 64GB RAM

| Step | Time | Parallelization |
|------|------|-----------------|
| Ground classification + tiling | 45 min | Single-threaded (PDAL) |
| Pretreatment (70 tiles) | 90 min | 6 parallel jobs |
| Fuel metrics (70 tiles) | 120 min | 6 parallel jobs |
| Merge tiles | 2 min | Single-threaded (GDAL) |
| Visualization | 30 sec | Single-threaded |
| **Total** | **~4 hours** | Mixed |

**Scaling:**
- Larger datasets: Time scales linearly with point count and area
- More cores: Increase parallel jobs (watch RAM usage)
- Smaller tiles (100m): 4× more tiles, ~1.5× longer total time

---

## References

### LidarForFuel Package

- **GitHub**: https://github.com/oliviermartin7/LidarForFuel
- **Paper**: Martin-Ducup, O., & Pimont, F. (2024). Unlocking the potential of Airborne LiDAR for direct assessment of fuel bulk density and load distributions for wildfire hazard mapping. *Agricultural and Forest Meteorology*, 357, 110204. https://doi.org/10.1016/j.agrformet.2024.110204

### Methods & Theory

- **Beer-Lambert Law**: Radiative transfer model for canopy structure estimation
- **lidR Package**: Roussel, J.-R., et al. (2020). lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. *Remote Sensing of Environment*, 251, 112061.
- **SMRF ground classification**: Pingel, T.J., Clarke, K.C., McBride, W.A. (2013). An improved simple morphological filter for the terrain classification of airborne LIDAR data.
- **PDAL filters.smrf**: https://pdal.io/en/stable/stages/filters.smrf.html
- **GDAL gdal_merge**: https://gdal.org/programs/gdal_merge.html

### Trait Databases

- **TRY Plant Trait Database**: https://www.try-db.org
- **Global Wood Density Database**: Zanne, A. E., et al. (2009). doi:10.5061/dryad.234
- **USFS Silvics Manual**: https://www.srs.fs.usda.gov/pubs/misc/ag_654/

---

## Future Enhancements

### Near-term
- [ ] LAScatalog parallel processing for large datasets
- [ ] Automated species map generation from NAIP/classification
- [ ] Integration with existing raster metrics pipeline
- [ ] Validation against field-measured fuel loads

### Long-term
- [ ] Raster-based LMA/WD maps (spatially varying traits)
- [ ] Fusion with NAIP/UAVSAR for species-trait attribution
- [ ] Time-series fuel monitoring (multi-temporal LiDAR)
- [ ] Fire behavior modeling inputs (FlamMap, FARSITE integration)

---

## Contact & Support

For issues with:
- **LidarForFuel package**: https://github.com/oliviermartin7/LidarForFuel/issues
- **This pipeline**: Open an issue in the main repo or consult project documentation

---

**Last updated:** 2025-11-02
**Pipeline version:** 2.1 (Consolidated documentation, configurable Beer-Lambert parameters)
