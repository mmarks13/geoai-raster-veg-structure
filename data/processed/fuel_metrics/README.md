## Fuel Metrics Pipeline

Fuel metrics computation from UAV LiDAR point clouds using the [LidarForFuel](https://github.com/oliviermartin7/LidarForFuel) R package. Generates comprehensive wildfire fuel characterization rasters including bulk density profiles, canopy structure, and standardized fuel layers.

---

## Overview

This pipeline integrates LidarForFuel into the Python-based workflow through thin R wrappers. It maintains parity with the published method (Martin-Ducup & Pimont, 2024) while minimizing re-implementation complexity.

**Workflow:**
```
Raw UAV LiDAR (.las)
    ↓
[fPCpretreatment]  ← Normalization + Trait Attribution
    ↓
Pretreated Point Cloud (.laz)
    ↓
[fCBDprofile_fuelmetrics]  ← Beer-Lambert Inversion + Fuel Metrics
    ↓
173-Band Fuel Raster (.tif)
```

**Key Outputs:**
- **23 summary metrics**: Canopy height, CBH, fuel loads, cover, vertical complexity
- **150 bulk density layers**: Vertical distribution profile (1m resolution by default)

---

## Quick Start

### Prerequisites

**Note:** R packages are now in a separate conda environment (`r_fuel_metrics`) to avoid conflicts with the main Python environment.

```bash
# 1. Create the R fuel metrics environment
conda env create -f environment_r_fuel_metrics.yml

# 2. Install LidarForFuel R package
conda activate r_fuel_metrics
R -e "remotes::install_github('oliviermartin7/lidarforfuel')"

# 3. Verify installation
Rscript scripts/r/run_pretreatment.R --help

# 4. Return to main environment for Python work (optional)
conda activate geoai_env
```

The Python wrapper (`src/data_prep/lidarforfuel_wrapper.py`) automatically uses the `r_fuel_metrics` environment via `conda run`, so you don't need to manually activate it when running the fuel metrics pipeline from Python.

### Single File Processing

```bash
# Using default trait values (Mixed woodland)
python src/data_prep/process_uav_fuel_metrics.py \
    --input data/raw/uavlidar/study_las/20241025_151528.las

# Specify species from trait lookup
python src/data_prep/process_uav_fuel_metrics.py \
    --input data/raw/uavlidar/study_las/20241025_151528.las \
    --species "Quercus agrifolia" \
    --resolution 1.0

# Export summary metrics only (23 bands instead of 173)
python src/data_prep/process_uav_fuel_metrics.py \
    --input data/raw/uavlidar/study_las/20241025_151528.las \
    --export_mode summary
```

### Batch Processing

```bash
# Process all LAS files in directory
python src/data_prep/process_uav_fuel_metrics.py \
    --input_dir data/raw/uavlidar/study_las \
    --pattern "*.las" \
    --species "Mixed"

# Test on first 3 files
python src/data_prep/process_uav_fuel_metrics.py \
    --input_dir data/raw/uavlidar/study_las \
    --max_files 3
```

### List Available Species

```bash
python src/data_prep/process_uav_fuel_metrics.py --list_species
```

---

## Directory Structure

```
data/processed/fuel_metrics/
├── README.md                    # This file
├── trait_lookup.csv             # LMA/WD values by species
├── volcan/                      # Volcan Mountain outputs
│   ├── pretreated/              # Intermediate: pretreated LAZ files
│   │   └── 20241025_151528_pretreated.laz
│   └── rasters/                 # Final: fuel metrics GeoTIFFs
│       └── 20241025_151528_fuel_metrics.tif  (173 bands)
└── [site_name]/                 # Additional sites
    ├── pretreated/
    └── rasters/
```

---

## Trait Lookup Table

The `trait_lookup.csv` file defines Leaf Mass Area (LMA) and Wood Density (WD) values for different vegetation types. These traits are critical for converting Plant Area Density (PAD) to bulk density.

**Species included:**
- **Quercus agrifolia** (Coast live oak): LMA 150, WD 750
- **Quercus kelloggii** (California black oak): LMA 100, WD 650
- **Ceanothus spp.** (Ceanothus chaparral): LMA 130, WD 550
- **Pinus coulteri** (Coulter pine): LMA 180, WD 450
- **Calocedrus decurrens** (Incense cedar): LMA 140, WD 384
- **Mixed** woodland: LMA 140, WD 591 (recommended default)

**Trait sources:**
- Mediterranean sclerophyll literature (evergreen oaks)
- USFS Silvics Manual (incense cedar)
- Chaparral fire ecology literature
- LidarForFuel package defaults

### Adding Custom Species

Edit `trait_lookup.csv`:

```csv
species,common_name,lma_gm2,wd_kgm3,lma_understory_gm2,wd_understory_kgm3,notes,references
Pinus ponderosa,Ponderosa pine,150,420,130,550,Moderate LMA; WD from global database,Wood density database
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

**Resources:**
- [TRY Plant Trait Database](https://www.try-db.org)
- [Global Wood Density Database](https://doi.org/10.5061/dryad.234)
- Published trait studies for your region

---

## Output Raster Bands

### Summary Metrics (Bands 1-23)

LidarForFuel produces 23 summary fuel metrics (always included):

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

### Profile Types

**Profil_Type_L** (Band 2) classifies vertical structure:
- **A**: Surface fuel only (no canopy)
- **B**: Canopy only (no surface fuel)
- **C**: Canopy + surface, with gap (discontinuous)
- **D**: Continuous fuel (no gap, high fire risk)

---

## Methodology

LidarForFuel uses a physics-based approach to derive fuel metrics from LiDAR:

### 1. Pretreatment (`fPCpretreatment`)

**Normalization:**
- Height above ground (HAG) computed via DTM subtraction
- Outlier removal (noise classification)
- Height filtering (retain points <60m by default)

**Trait Attribution:**
- LMA and WD assigned to each point via:
  - **Constant values** (current implementation): Lookup table by species
  - **Raster maps** (future): Spatial intersection with species/trait maps
- Understory (<2m) can have different traits than canopy

**Outputs:**
- Normalized point cloud with attributes: `LMA`, `WD`, `Zref`, `Easting`, `Northing`, `Elevation`

### 2. Fuel Metrics Computation (`fCBDprofile_fuelmetrics`)

**Beer-Lambert Inversion:**
- **Plant Area Density (PAD)**: Estimated from point density using radiative transfer model
- Formula: `PAD(z) = -ln(P_gap(z)) / ΔZ`, where `P_gap` is gap probability
- Accounts for scanning angle, clumping (ω=0.77), and extinction (G=0.5)

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

## Performance & Scalability

### Resource Requirements

**Memory:**
- R/lidR is RAM-intensive for large point clouds
- Estimate: 8-16 GB per parallel worker
- Volcan Mountain full dataset (~36 GB compressed): 64-128 GB recommended

**Runtime (single file, ~1M points):**
- Pretreatment: 30-90 seconds
- Fuel metrics (1m resolution): 2-5 minutes
- **Total**: ~3-6 minutes per file

**Disk:**
- Pretreated LAZ: ~same size as input (compression efficient)
- Fuel metrics GeoTIFF: 50-200 MB per file (full 173 bands, LZW compressed)
- Summary mode (23 bands): ~10-30 MB

### Scaling Strategies

**For large datasets:**

1. **Batch processing** (current implementation):
   ```bash
   python src/data_prep/process_uav_fuel_metrics.py \
       --input_dir data/raw/uavlidar/study_las \
       --pattern "*.las"
   ```

2. **LAScatalog** (future enhancement):
   - Process tiled point clouds in parallel via lidR
   - Automatic chunking and stitching
   - Requires R script modification

3. **Cleanup intermediate files**:
   ```bash
   python src/data_prep/process_uav_fuel_metrics.py \
       --input file.las \
       --cleanup  # Deletes pretreated LAZ after metrics computation
   ```

4. **Summary mode** (reduce output size):
   ```bash
   python src/data_prep/process_uav_fuel_metrics.py \
       --input file.las \
       --export_mode summary  # Only 23 bands
   ```

---

## Parameter Tuning

### Resolution Trade-offs

| Resolution | Detail | File Size | Runtime | Use Case |
|------------|--------|-----------|---------|----------|
| 0.5 m | Very high | Large | Slow | Individual tree analysis |
| 1.0 m | High (default) | Moderate | Moderate | Stand-level fuel mapping |
| 2.0 m | Moderate | Small | Fast | Landscape-scale assessment |
| 5.0 m | Coarse | Very small | Very fast | Regional overview |

### Layer Depth

- **1.0 m** (default): Standard for wildfire modeling
- **0.5 m**: Fine-scale vertical structure analysis
- **2.0 m**: Coarser profiles (reduces band count to 75)

### Threshold (Strata Detection)

- **0.02 kg/m³** (default): Standard threshold for canopy base
- **0.01**: More sensitive (lower CBH estimates)
- **0.05**: More conservative (higher CBH estimates)

### Height Cover

- **2.0 m** (default): Standard threshold for "cover" computation
- **1.5 m**: Include shorter vegetation
- **3.0 m**: Only taller vegetation

---

## Validation & Interpretation

### Expected Value Ranges (Volcan Mountain)

Based on California oak woodland/chaparral:

| Metric | Typical Range | Notes |
|--------|---------------|-------|
| Height | 5-25 m | Oaks 10-20m, chaparral 1-5m |
| CBH | 1-8 m | Higher for pine/oak, lower for chaparral |
| FSG | 0-10 m | Stratified stands have larger gaps |
| CFL | 0.2-2.0 kg/m² | Higher in dense oak woodland |
| TFL | 0.5-5.0 kg/m² | Includes all strata |
| Canopy cover | 20-80% | Variable by stand type |

**Sanity checks:**
- CBH < Height (always)
- CFL < TFL (always)
- max_CBD typically 0.1-0.5 kg/m³ for forests

### Comparison with Existing Metrics

Your existing pipeline ([point_cloud_utils.py](../../src/utils/point_cloud_utils.py)) computes simpler metrics:
- Max/mean/percentile heights → Compare with LidarForFuel **Height** (band 3)
- Canopy/midstory density → Compare with **Canopy_cover** (band 12)
- Foliage Height Diversity (FHD) → Compare with **entropy_PAD** (band 18) or **VCI_PAD** (band 6)

**Differences:**
- LidarForFuel uses Beer-Lambert inversion (physics-based) vs. simple binning
- Bulk density profiles vs. point count ratios
- Species-specific traits vs. uniform assumptions

**Recommended:**
- Run both pipelines on same tiles
- Document correlation and systematic differences
- Use for cross-validation or fusion

---

## Troubleshooting

### "Rscript not found in PATH"

```bash
# Check R installation
which Rscript
R --version

# Install R if missing
conda install -c conda-forge r-base

# Verify
Rscript --version
```

### "Error loading libraries: there is no package called 'lidarforfuel'"

```bash
# Install lidR and dependencies
conda install -c conda-forge r-lidr r-remotes r-terra

# Install LidarForFuel from GitHub
R -e "remotes::install_github('oliviermartin7/lidarforfuel')"

# Test installation
R -e "library(lidarforfuel)"
```

### "Missing required attributes: gpstime"

**Cause:** Input LAS file lacks GPS time stamps

**Solutions:**
1. Use original LAS files (not processed/stripped versions)
2. Add synthetic timestamps via PDAL:
   ```bash
   pdal translate input.las output.las --writers.las.system_id="FAKED" \
       --writers.las.a_srs="EPSG:32611" --metadata gpstime="1"
   ```
3. Modify R script to set `start_date` and `season_filter` to disable time filtering

### "Pretreatment failed: not enough points"

**Cause:** Very sparse point cloud or aggressive height filtering

**Solutions:**
1. Check input density: `pdal info input.las --stats`
2. Increase `height_filter` in R script (default 60m)
3. Use coarser resolution for metrics computation

### Memory errors during metrics computation

**Cause:** Large point clouds exceed available RAM

**Solutions:**
1. Reduce resolution (e.g., 2m instead of 1m)
2. Process smaller tiles/chunks
3. Increase available RAM
4. Use LAScatalog approach (future enhancement)

### Output raster has many NA/nodata pixels

**Causes:**
- Insufficient points per pixel (`limit_N_points = 400` threshold)
- Point cloud gaps or coverage issues

**Solutions:**
1. Reduce resolution (larger pixels → more points)
2. Lower `limit_N_points` in R script (caution: less reliable)
3. Check input coverage: `pdal info input.las --boundary`

---

## References

### LidarForFuel Package

- **GitHub**: https://github.com/oliviermartin7/LidarForFuel
- **Paper**: Martin-Ducup, O., & Pimont, F. (2024). Unlocking the potential of Airborne LiDAR for direct assessment of fuel bulk density and load distributions for wildfire hazard mapping. *Agricultural and Forest Meteorology*, 357, 110204. https://doi.org/10.1016/j.agrformet.2024.110204

### Methods & Theory

- **Beer-Lambert Law**: Radiative transfer model for canopy structure estimation
- **lidR Package**: Roussel, J.-R., et al. (2020). lidR: An R package for analysis of Airborne Laser Scanning (ALS) data. *Remote Sensing of Environment*, 251, 112061.
- **PAD/CBD**: Plant Area Density and Canopy Bulk Density concepts in fire ecology

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

**Last updated:** 2025-10-19
