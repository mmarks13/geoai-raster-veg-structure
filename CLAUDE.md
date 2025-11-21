# CLAUDE.md - Project Memory

## Documentation Map (Read These When Needed)
- [README.md](README.md) - Project overview, repo structure, getting started, workflow
- [data/README.md](data/README.md) - **Data directory structure** (complete layout, provenance, file formats, storage)
- [src/README.md](src/README.md) - Source code organization, entry points
- [scripts/README.md](scripts/README.md) - Data pipeline scripts (get_data.sh, process_data.sh)
- [src/data_prep/README.md](src/data_prep/README.md) - STAC catalogs, tile generation, train/test splits
- [src/models/README.md](src/models/README.md) - LG-PAB architecture, ViT encoders, fusion modules
- [src/training/README.md](src/training/README.md) - DDP training, ablation studies
- [src/evaluation/README.md](src/evaluation/README.md) - Inference, statistical tests, figure generation
- [src/utils/README.md](src/utils/README.md) - Point cloud utilities, Chamfer distance, KNN graphs
- [src/training/model_data_readme.md](src/training/model_data_readme.md) - **Data structure spec** (read before working with training data)
- [data/processed/fuel_metrics/README.md](data/processed/fuel_metrics/README.md) - **Fuel metrics pipeline** (LidarForFuel integration, wildfire hazard mapping)

## 1. Project Overview
Multi-modal LiDAR point cloud enhancement using attention mechanisms. Fuses sparse 3DEP LiDAR with NAIP optical and UAVSAR L-band SAR imagery to upsample vegetation structure. Published in Remote Sensing 2025.

**Task:** Point cloud upsampling (regression) - predict dense 3D point cloud from sparse input
**Stack:** PyTorch 2.5.1, CUDA 12.4, PyTorch Geometric, PyTorch3D
**Model:** Local-Global Point Attention Blocks (LG-PAB) with Vision Transformer encoders

## 2. Essential Commands

### Training
```bash
# Single model (full multi-modal)
python run_model_test.py

# Ablation study (baseline, NAIP, UAVSAR, fused)
python run_ablation_study.py
```

### Data Preparation
```bash
# Download data (NAIP, UAVSAR, 3DEP, UAV LiDAR STAC catalogs)
bash scripts/get_data.sh

# Preprocess & split (train/val/test)
bash scripts/process_data.sh
```

### Evaluation
```bash
# Inference on test set
python src/evaluation/inference_eval.py --model_path <checkpoint> --test_data <test_tiles.pt>

# Statistical tests
python src/evaluation/RQ_test_v2.py --eval_data <eval_df.pt>

# Generate manuscript figures
python src/evaluation/manuscript_figures.py --eval_data <eval_df.pt>
```

### Fuel Metrics (Wildfire Hazard Mapping)
```bash
# Complete pipeline (recommended) - Single command for entire workflow
bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
    --input data/raw/uavlidar/my_site.las \
    --output-name my_site \
    --species "Mixed" \
    --resolution 5.0 \
    --tile-size 200 \
    --parallel-jobs 6

# Installation (one-time setup)
conda env create -f environment_r_fuel_metrics.yml
bash scripts/fuel_metrics/install_lidarforfuel.sh

# Individual pipeline steps:

# Step 1: Ground classification + tiling (consolidated PDAL pipeline)
bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
    data/raw/uavlidar/my_site.las \
    data/processed/fuel_metrics/my_site/tiles \
    200 \
    10

# Step 2: Batch pretreatment + fuel metrics
bash scripts/fuel_metrics/run_batch_fuel_metrics.sh \
    data/processed/fuel_metrics/my_site/tiles \
    data/processed/fuel_metrics/my_site \
    "Mixed" \
    5.0 \
    6

# Step 3: Merge tiles (CRITICAL: use explicit file list, NOT wildcards)
conda run -p /home/jovyan/geoai_env gdal_merge.py \
    -o data/processed/fuel_metrics/my_site/merged/my_site_fuel_metrics_5m.tif \
    -a_nodata nan \
    -co COMPRESS=LZW -co TILED=YES -co BIGTIFF=YES \
    --optfile <(find data/processed/fuel_metrics/my_site/rasters -name "*.tif" | sort)

# Step 4: Generate visualization
conda run -p /home/jovyan/geoai_env python src/fuel_metrics/visualize_metrics.py \
    data/processed/fuel_metrics/my_site/merged/my_site_fuel_metrics_5m.tif \
    data/processed/fuel_metrics/my_site/merged/my_site_visualization.png

# List available species/traits
python src/fuel_metrics/process_fuel_metrics.py --list_species

# See data/processed/fuel_metrics/PIPELINE.md for complete documentation
```

## 3. Code Standards & Style

### Core Development Principles
- **Minimal viable implementation** - Build exactly what's requested, no anticipatory features
- **Ask before assuming** - Present options when ambiguous; flag assumptions explicitly
- **Optimize for readability** - Clear code over micro-optimizations unless performance is critical
- **No secrets in code/config** - All credentials via environment variables or secret managers
- **Examine data first, theorize second** - Always check actual data values and distributions before forming hypotheses about bugs; assumptions about what "should" be happening often blind you to what's actually in the data

### Python Language Standards
- **Descriptive names** - Full words: `customer_email` over `cust_email` or `ce`
- **Type hints** - All function signatures for clarity and IDE support
- **Docstrings** - All public functions/classes with purpose, parameters, return values
- **Simple comprehensions** - List/dict comprehensions for simple cases only; multi-line/nested → regular loops

### Code Organization
- **Organize by purpose** - Top-level folders by system purpose, flat files within, subfolders only for 3-4+ files
- **Python scripts go in `src/`** - Organized by purpose (data_prep, models, training, etc.)
- **Shell and R scripts go in `scripts/`** - Use appropriate subdirectories (`scripts/r/` for R, root of `scripts/` for shell)
- **Target <40 lines per function** - Split on distinct responsibilities, not just line counts; 50+ acceptable if readable
- **Abstract on second use** - Extract truly identical logic on second occurrence; wait for patterns if purposes differ
- **Named constants** - Config thresholds, timeouts, business values; skip obvious one-offs
- **Return early** - Guard clauses for edge cases first, reduces nesting

**Note on directory separation:** This strict organization prevents mixing languages and keeps dependencies clean. Python ecosystem (pip/conda) is separate from R packages, and shell scripts are configuration/orchestration only.

### Error Handling & Validation
- **Fail fast** - Let exceptions propagate; specific catches (`except SpecificError`), re-raise with context
- **Contextual error messages** - Include what failed + actionable context: `ValueError(f"Config '{key}' not found. Check env vars.")`
- **Avoid fallback values** - Don't mask failures unless explicitly instructed

### Documentation & Comments
- **Comment flow/context** - One-line labels for multi-step processes; "why" for non-obvious decisions/trade-offs
- **Clarify complex logic** - Explain what isn't self-evident; avoid restating obvious code

### CRS & Geospatial Data Standards

**Core Principle:** CRS mismatches must fail immediately with clear errors - never silently propagate.

- **Explicit CRS declaration** - All geospatial files MUST include CRS metadata in file format
- **Fail fast validation** - Check CRS on file read, reject mismatches immediately with clear error
- **No silent reprojection** - Fix CRS issues in source files/scripts upstream, not in processing code
- **Clear error messages** - State expected CRS, actual CRS, and which file to fix
- **Datum awareness** - Never assume datum shifts (WGS84 vs NAD83) are acceptable - verify requirements explicitly
- **User control** - Provide `--crs` arguments with documented defaults; fail if input doesn't match

**Example validation pattern:**
```python
expected_crs = 'EPSG:32611'
if str(gdf.crs) != expected_crs:
    raise ValueError(
        f"CRS mismatch! Expected: {expected_crs}, Got: {gdf.crs}. "
        f"Reproject input file {input_path} to {expected_crs} before processing."
    )
```

### Testing & Dependencies
- **Test critical paths** - Complex logic, edge cases, public APIs after implementation; skip trivial code
- **Use established libraries** - Prefer mature, maintained libraries over custom implementations

### Data Analysis Standards (ML/ETL Pipelines)
- **Validate data upfront** - Check missing values, types, ranges at start; fail loudly if expectations unmet
- **Explicit missing data handling** - Never silent `.dropna()`/`.fillna()`; document strategy (drop/impute/flag), justify, note biases
- **Surface all assumptions** - Document analytical decisions (ordering, thresholds, transforms, methods); comment why, don't hide implicit choices

## 4. PyTorch & ML Patterns

### Training Infrastructure
- **Training loop:** Custom DDP implementation in `src/training/multimodal_training.py`
- **Entry points:** `run_ablation_study.py`, `run_model_test.py` (root level)
- **Checkpoints:** Saved to `data/output/checkpoints/`, best model by validation loss
- **Config:** `MultimodalModelConfig` dataclass in `src/models/multimodal_model.py`
- **Tracking:** Not using TensorBoard in published workflow
- **Device:** Multi-GPU DDP with gradient accumulation, mixed precision (AMP)

### Data Loading
- **Dataset:** List of dicts loaded from `.pt` files (NOT PyTorch Dataset class)
- **Sharding:** Manual per-GPU sharding for DDP (not DistributedSampler)
- **DataLoader:** Standard PyTorch DataLoader with custom collate
- **Augmentation:** Precomputed augmented tiles in separate `.pt` file

### Tensor Conventions
- Point clouds: `[N, 3]` for xyz coordinates, `[N, F]` for features
- Batched points: Use batch indexing tensor `[N]` (NOT `[B, N, 3]`)
- Edge indices: `[2, E]` (PyTorch Geometric convention)
- Images: `[n_images, C, H, W]` (NAIP: 4 bands 40×40, UAVSAR: 6 bands 4×4)

### Key Hyperparameters
- `feature_dim`: 256 (feature dimension throughout model)
- `k`: 16 (KNN neighbors for local attention)
- `up_ratio`: 2 (upsampling ratio, not heavily used)
- `batch_size`: 15 tiles per GPU (60 total on 4 GPUs)
- `optimizer`: ScheduleFreeAdamW (base lr: 5e-4, weight-decay: 1e-4, β₁,₂=(0.9,0.999); no external LR schedule)
- `loss`: Density-aware Chamfer distance (α=4, adapted for meter-scale coordinates)
- `epochs`: 100-400 depending on experiment

## 5. Project Structure & Critical Paths

### Key Directories
- `src/data_prep/` - STAC downloads, tile generation, train/test split
- `src/fuel_metrics/` - **Wildfire fuel hazard mapping module** (Python-R interface, batch processing, visualization)
- `src/models/` - Model architecture (multimodal_model.py, encoders.py, cross_attn_fusion.py, fusion.py)
- `src/training/` - Training loop (multimodal_training.py, ddp_training.py)
- `src/evaluation/` - Inference, stats, figures
- `src/utils/` - Chamfer distance, KNN graphs, point cloud utilities
- `src/raster_mapping/` - Forest plot visualization utilities
- `scripts/` - **Shell scripts only** (get_data.sh, process_data.sh, compress_las_files.sh)
- `scripts/fuel_metrics/` - **Fuel metrics pipeline scripts** (run_fuel_metrics_pipeline.sh, PDAL/R wrappers)
- `scripts/fuel_metrics/pdal/` - Consolidated PDAL pipeline (ground classification + tiling)
- `scripts/fuel_metrics/r/` - R wrapper scripts (run_pretreatment.R, run_fuel_metrics.R)
- `data/processed/fuel_metrics/<site_name>/` - Per-site fuel metrics outputs (tiles, pretreated, rasters, merged, logs, validation)
- `manuscript/` - LaTeX source and figures
- `run_*.py` - **Training entry points (root level)**

### NEVER Modify/Read
- `data/stac/` - STAC catalog files (downloaded)
- `data/raw/` - Raw downloaded datasets
- `data/processed/training_data_chunks/` - Intermediate HDF5 files
- `data/output/checkpoints/*.pth` - Model checkpoint weights
- `data/output/logs/` - Training logs (except for debugging)
- `.cache/` - Cached intermediate results

**Exception:** `data/README.md` documents the structure and is safe to read.

### Legacy/Unused Code
- `src/*/legacy/` - Superseded implementations
- `src/*/unused_alternatives/` - Explored but not published

## 6. Data Structure (Critical for Training)

### Precomputed Tile Format
**Location:** `data/processed/model_data/precomputed_{training,validation,test}_tiles_32bit.pt`

**Each tile:** Dict with keys:
- `dep_points_norm`: [N_dep, 3] - Normalized sparse 3DEP points (input)
- `uav_points_norm`: [N_uav, 3] - Normalized dense UAV points (ground truth, max 20k)
- `dep_points_attr`: [N_dep, 3] - Intensity, ReturnNumber, NumberOfReturns
- `center`: [1, 3] - Normalization center
- `knn_edge_indices`: Dict[k, Tensor[2, E]] - Precomputed KNN graphs (k=15)
- `naip`: Dict or None - NAIP imagery (2-6 images, 4 bands, 40×40 pixels)
- `uavsar`: Dict or None - UAVSAR imagery (4-30 images, 6 bands, 4×4 pixels)
- `tile_id`: String - Unique identifier
- `bbox`: [4] - Original bbox [xmin, ymin, xmax, ymax] in EPSG:32611

### Normalization
- **Points:** x,y centered at (0,0), z min=0, units in meters
- **Bbox:** 10×10m tile, point coords in [-5, 5] for x,y
- **Imagery bbox:** 20×20m sharing same centroid

## 7. Model Architecture (LG-PAB)

### Components (in order)
1. **Feature Extractor (LG-PAB):** PointTransformerConv (local k-NN) → Global position-aware attention
2. **Image Encoders (if enabled):** Separate ViT encoders for NAIP/UAVSAR with temporal GRU aggregation
3. **Cross-Attention Fusion:** Multi-head cross-attention (point features ← image patch embeddings)
4. **Feature Expansion (LG-PAB):** Feature-guided upsampling (2×) → Local k-NN → Global attention
5. **Feature Refinement (LG-PAB):** Local k-NN → Global attention on upsampled points
6. **Coordinate Decoder:** MLP predicting per-point residual offsets

### Ablation Variants (use_naip, use_uavsar flags)
- **Baseline:** LiDAR only (both False)
- **NAIP:** LiDAR + optical (use_naip=True)
- **UAVSAR:** LiDAR + SAR (use_uavsar=True)
- **Fused:** LiDAR + optical + SAR (both True)

### Fusion Types
- `fusion_type='cross_attention'` - Multi-head cross-attention (used in paper)
- `fusion_type='spatial'` - Distance-weighted proximity fusion (alternative)

## 8. Critical Gotchas

### Data
- **Tiles are NOT batched as [B, N, 3]** - use batch indexing tensors like PyTorch Geometric
- **KNN graphs precomputed** - don't recompute during training (slow)
- **Imagery dates don't align** - NAIP and UAVSAR have different acquisition dates
- **Variable point counts** - dep: 1k-10k, uav: up to 50k (downsampled)

### Training
- **Best model != final epoch** - save by validation loss, not epoch number
- **Sharding not DistributedSampler** - manual per-GPU data splits for DDP
- **Augmented data in separate file** - combine with original in training loop
- **GPU memory:** 4×48GB required for published batch size (15/GPU)

### Model
- **Attention head configs granular** - separate for extractor/expansion/refinement (extractor_lcl_heads, etc.)
- **Legacy params ignored** - num_lcl_heads, up_attn_hds kept for backward compatibility only
- **Checkpoint loading selective** - can load specific layers via `layers_to_load` list

### Metrics & Coordinates
- **Training loss:** Density-aware Chamfer distance (α=4, handles meter-scale coordinates)
- **Evaluation metric:** Standard Chamfer distance (PyTorch3D GPU implementation)
- **Direction:** Bidirectional nearest neighbor distance
- **Critical:** Point clouds are in **meter-scale coordinates**, NOT normalized to unit cube
  - x,y ∈ [-5, 5] meters (centered at tile center)
  - z ∈ [0, tile_height] meters (min z = 0)
  - This aids interpretability and alignment but requires α=4 (not α=1000) for density-aware loss to prevent gradient saturation

### LAS Coordinate Precision
- PDAL automatically adjusts coordinate scale based on data extent
- When using PDAL to create new LAS/LAZ files, include precision suffix in filename (e.g., `_1cm.laz`, `_1mm.laz`)
- Check precision with: `pdal info --summary file.las | grep scale_`

## 9. Git Workflow
- **Main branch:** `cleanup` (current), not `main`
- **Commit message format:** Standard descriptive commits
- **Data directory:** Only `.gitkeep`, `README.md`, and critical config files (e.g., `test_val_polygons.geojson`) are tracked
- **Never commit:** `*.pt`, `*.pth`, `*.las`, `*.laz`, `*.tif`, checkpoints, logs, STAC catalog contents, `.cache/`
- **Granular .gitignore:** Uses extension-based patterns to allow structure while ignoring data files

## 10. Environment

### Main Environment (Python/ML Stack)
- **Environment name:** `geoai_env`
- **Environment file:** `environment.yml`
- **Python:** 3.11.11
- **PyTorch:** 2.5.1 (CUDA 12.4)
- **Package manager:** Conda
- **Hardware (published):** 4× NVIDIA L40 (48GB)
- **CUDA required:** 12.4 toolkit
- **Usage:** Point cloud upsampling, model training, evaluation

### R Environment (Fuel Metrics)
- **Environment name:** `r_fuel_metrics`
- **Environment file:** `environment_r_fuel_metrics.yml`
- **R:** 4.5.1
- **Key packages:** r-lidr, r-terra, r-sf, LidarForFuel (from GitHub)
- **Usage:** Wildfire fuel hazard mapping via LidarForFuel
- **Note:** Python wrapper automatically uses this environment via `conda run`

**Why separate environments?**
- Prevents R/Python package conflicts
- Cleaner dependency management
- Smaller, faster environment creation
- R packages won't interfere with PyTorch/CUDA stack

## 11. Critical File Paths (Frequently Referenced)

### Training Data
```
data/processed/model_data/precomputed_training_tiles_32bit.pt
data/processed/model_data/augmented_tiles_32bit_16k_no_repl.pt
data/processed/model_data/precomputed_validation_tiles_32bit.pt
data/processed/model_data/precomputed_test_tiles_32bit.pt
```

### Model Components
```
src/models/multimodal_model.py        # Main model + config
src/models/encoders.py                # NAIPEncoder, UAVSAREncoder
src/models/cross_attn_fusion.py       # CrossAttentionFusion
src/training/multimodal_training.py   # train_multimodal_model(), run_ablation_studies()
```

### Entry Points
```
run_ablation_study.py                 # Train all variants
run_model_test.py                     # Train single model
```

## 12. Data Directory Structure

The `data/` directory contains all downloaded, processed, and generated data. See [data/README.md](data/README.md) for complete structure documentation.

**Key locations:**
- `data/stac/` - STAC catalogs (NAIP, UAVSAR, 3DEP, UAV LiDAR)
- `data/raw/uavlidar/study_las/` - **USER-PROVIDED** UAV LiDAR ground truth (.las/.laz files)
- `data/processed/model_data/` - Training-ready `.pt` files
- `data/processed/test_val_polygons.geojson` - Spatial train/test split polygons (created in QGIS)
- `data/output/checkpoints/` - Model checkpoints (best by val loss, per-epoch saves)
- `data/output/cached_shards/` - Per-GPU data shards for DDP training

**Git tracking:** Only structure (`.gitkeep`), documentation (`data/README.md`), and critical config files (`test_val_polygons.geojson`) are tracked. All data files are ignored to prevent repo bloat and merge conflicts.

**Data provenance:**
- **Downloaded:** NAIP (Planetary Computer), UAVSAR (ASF), 3DEP (Planetary Computer or pre-downloaded)
- **User-provided:** UAV LiDAR ground truth in `data/raw/uavlidar/study_las/`
- **Generated:** Training tiles, checkpoints, evaluation results
- **Manual:** `test_val_polygons.geojson` (created in QGIS for spatial splits)

## 13. Fuel Metrics Pipeline (LidarForFuel Integration)

**Purpose:** Compute wildfire fuel hazard metrics from UAV LiDAR using physics-based Beer-Lambert inversion.

**Module location:** `src/fuel_metrics/` (separate from point cloud upsampling work)

**Key components:**
- `src/fuel_metrics/lidarforfuel_wrapper.py` - Python-R interface (preserves SMRF ground classification params)
- `src/fuel_metrics/process_fuel_metrics.py` - Main orchestration script (renamed from process_uav_fuel_metrics.py)
- `src/fuel_metrics/batch_processing.py` - Parallel batch processing with progress tracking
- `src/fuel_metrics/visualize_bounds.py` - Spatial coverage validation
- `src/fuel_metrics/visualize_metrics.py` - Fuel metrics visualization
- `scripts/fuel_metrics/run_fuel_metrics_pipeline.sh` - **Main entry point** (complete pipeline orchestrator)
- `scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh` - **Consolidated PDAL pipeline** (ground classification + tiling in single pass)
- `scripts/fuel_metrics/r/run_pretreatment.R` - R wrapper for fPCpretreatment
- `scripts/fuel_metrics/r/run_fuel_metrics.R` - R wrapper for fCBDprofile_fuelmetrics
- `scripts/fuel_metrics/install_lidarforfuel.sh` - R package installation script
- `data/processed/fuel_metrics/trait_lookup.csv` - LMA/WD values by species

**Complete workflow (single command):**
```bash
bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
  --input data/raw/uavlidar/my_site.las \
  --output-name my_site \
  --species "Mixed" \
  --resolution 5.0 \
  --tile-size 200 \
  --parallel-jobs 6
```

**Pipeline steps:**
1. **Ground classification + tiling** (PDAL, single I/O pass): SMRF filter + 200m tiles
2. **Pretreatment** (LidarForFuel): Normalize point cloud, add LMA/WD attributes
3. **Fuel metrics** (LidarForFuel): Compute 173-band raster (23 summary + 150 bulk density)
4. **Merge** (GDAL): Seamless mosaic using explicit file list (NOT wildcards - glob expansion fails with 70+ files)
5. **Visualization**: 6-panel figure (H, CBH, FSG, fuel loads, VCI)

**Data organization (per site):**
```
data/processed/fuel_metrics/<site_name>/
├── tiles/              # Ground-classified LAZ tiles (200m × 200m)
├── pretreated/         # Normalized LAZ with LMA/WD attributes
├── rasters/            # 173-band fuel metric TIFFs (per tile)
├── merged/             # Seamless mosaic + visualization
├── logs/               # Processing logs and summary CSV
└── validation/         # Spatial coverage checks
```

**Key metrics:**
- Canopy height, Canopy Base Height (CBH), Fuel Strata Gap (FSG)
- Fuel loads (Canopy, Total, Midstorey, Surface)
- Cover percentages (Canopy, Midstorey, Understory)
- Vertical Complexity Index (VCI), entropy
- Bulk density profile (150 vertical layers at 1.5m resolution)

**Trait values (default: Mixed woodland):**
- LMA: 140 g/m² (canopy), 130 g/m² (understory <2m)
- WD: 591 kg/m³ (canopy), 550 kg/m³ (understory)
- Species-specific values in `trait_lookup.csv` (Coast live oak, Black oak, Ceanothus, Coulter pine, Incense cedar)

**Ground classification method:** SMRF (Simple Morphological Filter) via PDAL with default parameters:
```json
{
  "type": "filters.smrf",
  "cell": 1.0,
  "slope": 0.15,
  "threshold": 0.5,
  "window": 18.0
}
```

**R environment:** Separate conda environment (`r_fuel_metrics`) to avoid conflicts with PyTorch stack
```bash
# Installation
conda env create -f environment_r_fuel_metrics.yml
bash scripts/fuel_metrics/install_lidarforfuel.sh
```

**R package:** [LidarForFuel](https://github.com/oliviermartin7/LidarForFuel) (Martin-Ducup & Pimont 2024)

**Documentation:**
- **Complete pipeline guide:** [data/processed/fuel_metrics/PIPELINE.md](data/processed/fuel_metrics/PIPELINE.md) (fully rewritten for v2.0)
- **Module README:** [src/fuel_metrics/README.md](src/fuel_metrics/README.md)
- **Trait lookup:** [data/processed/fuel_metrics/trait_lookup.csv](data/processed/fuel_metrics/trait_lookup.csv)

**Critical notes:**
- **Separate feature:** Fuel metrics is distinct from point cloud upsampling (different use case, different methods)
- **Requires R runtime:** Uses LidarForFuel R package (not pure Python)
- **Physics-based:** Beer-Lambert radiative transfer model (not ML)
- **Consolidated pipeline:** Ground classification + tiling in single PDAL pass (eliminates 24GB intermediate file)
- **Dynamic:** Works with any UAV LiDAR file (not site-specific like original implementation)
- **Production-ready:** Comprehensive logging, validation, error handling, parallel processing

**Common pitfall:**
- **gdal_merge wildcards:** Using `tile_*_fuel_metrics.tif` pattern with 70+ tiles loses ~60% of data due to shell glob expansion limits. **Always use explicit file lists or --optfile**

## 14. Common Issues & Fixes

### NaN Losses
- Check normalization (points should be in [-5, 5] for x,y)
- Verify no empty point clouds (should be filtered in preprocessing)
- Reduce learning rate or check gradient clipping
