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
- [docs/raster_model_implementation_plan.md](docs/raster_model_implementation_plan.md) - **Raster prediction model** architecture and implementation (in development)

## 1. Project Overview
Multi-modal LiDAR point cloud enhancement using attention mechanisms. Fuses sparse 3DEP LiDAR with NAIP optical and UAVSAR L-band SAR imagery. **Two modeling approaches:** point cloud upsampling (published in Remote Sensing 2025) and raster fuel metrics prediction (in development).

**Point Cloud Upsampling (Published):**
- **Task:** Predict dense 3D point cloud from sparse input
- **Output:** Dense UAV-quality point clouds [N,3]
- **Ground truth:** High-density UAV LiDAR

**Raster Fuel Metrics Prediction (In Development):**
- **Task:** Directly predict fuel hazard rasters from sparse LiDAR + imagery
- **Output:** Fuel metrics raster [n_bands, 5×5] (Height, TFL, Total_cover, etc.)
- **Ground truth:** Fuel metrics from LidarForFuel pipeline (Section 13)

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
- `dep_points_norm`: [N_dep, 3] - Normalized sparse 3DEP points (input, Z is Height Above Ground if HAG-processed)
- `uav_points_norm`: [N_uav, 3] - Normalized dense UAV points (ground truth, max 20k)
- `dep_points_attr`: [N_dep, 6] - Point attributes (see below)
- `center`: [1, 3] - Normalization center
- `knn_edge_indices`: Dict[k, Tensor[2, E]] - Precomputed KNN graphs (k=15)
- `naip`: Dict or None - NAIP imagery (2-6 images, 4 bands, 40×40 pixels)
- `uavsar`: Dict or None - UAVSAR imagery (4-30 images, 6 bands, 4×4 pixels)
- `tile_id`: String - Unique identifier
- `bbox`: [4] - Original bbox [xmin, ymin, xmax, ymax] in EPSG:32611

### Point Attributes (6 features)
**Format:** `dep_points_attr` is `[N_dep, 6]` tensor

| Index | Name | Range | Description | Neutral Value |
|-------|------|-------|-------------|---------------|
| 0 | Intensity | [0, 65535] | Return intensity | N/A |
| 1 | ReturnNumber | [1, N] | Return number within pulse | N/A |
| 2 | NumberOfReturns | [1, N] | Total returns for this pulse | N/A |
| 3 | Planarity | [0, 1] | How planar the local neighborhood | 0.0 |
| 4 | Sphericity | [0, 1] | How spherical the local neighborhood | 1.0 |
| 5 | Verticality | [0, 1] | How vertical the local structure | 0.5 |

**Normalization:** Z-score normalization applied during preprocessing.

**Neutral values:** Used when features are missing (e.g., legacy data without HAG) or when insufficient neighbors for eigenvalue computation.

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

### Decoder Variants: Point Cloud vs Raster

**Point Cloud Upsampling Decoder (Published):**
1. **Feature Expansion (LG-PAB):** Feature-guided upsampling (2×) → Local k-NN → Global attention
2. **Feature Refinement (LG-PAB):** Local k-NN → Global attention on upsampled points
3. **Coordinate Decoder:** MLP predicting per-point residual offsets
- Output: Dense point cloud [2*N_dep, 3]
- Loss: Density-aware Chamfer distance

**Raster Prediction Decoder (In Development):**
1. **Query-Based Grid Aggregation:** Learnable grid queries [25] attend to point features via distance-masked cross-attention (R=3m radius)
2. **Raster Decoder:** 1×1 Conv MLP [256→128→64→n_bands] preserves sharp boundaries
- Output: Fuel metrics raster [n_bands, 5, 5]
- Loss: MSE (L2) regression loss

**Key Difference:** Shared encoder (LG-PAB + ViT + Fusion), separate task-specific decoders

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

### Raster Pipeline
- **Two-stage normalization:** Raster pipeline uses bbox normalization (meter-scale) → z-score normalization (mean=0, std=1), unlike point cloud pipeline (bbox only)
- **Denormalization required:** Before distance computations in query attention and cross-attention fusion, denormalize from z-score back to bbox-normalized (meter) space
- **Backward compatibility:** Point cloud and raster pipelines coexist; cross-attention fusion supports both normalized/unnormalized inputs
- **Grid centers:** For 5×5 grid over 10m tile: `[-4, -2, 0, 2, 4]` meters in bbox-normalized space (cell_size = 2m)
- **Forest plot validation:** 5 out-of-distribution sites (BluffMesa, Laguna, NorthBigBear, ReyesPeak, TecuyaRidge) for generalization testing with human field measurements

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

### Raster Pipeline Data (In Development)
```
data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt
data/processed/model_data_raster/augmented_tiles_raster_32bit.pt
data/processed/model_data_raster/precomputed_validation_tiles_raster_32bit.pt
data/processed/model_data_raster/fuel_metrics_normalization_stats.json
data/processed/model_data_raster/coordinate_normalization_stats.json
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

## 15. Raster Fuel Metrics Prediction Pipeline (In Development)

**Purpose:** Directly predict fuel hazard metrics rasters from sparse 3DEP LiDAR + NAIP/UAVSAR imagery, enabling large-scale inference and validation on forest plots with human field measurements.

**Parallel to point cloud upsampling:** Both approaches share the same encoder (LG-PAB + ViT + Fusion), but differ in decoder and output format.

| Aspect | Point Cloud Pipeline | Raster Pipeline |
|--------|---------------------|------------------|
| **Output** | Dense point clouds [N,3] | Fuel metrics raster [n_bands, 5×5] |
| **Decoder** | Feature expansion + refinement (LG-PAB) | Query attention + 1×1 Conv MLP |
| **Loss** | Density-aware Chamfer distance | MSE (L2) regression |
| **Ground truth** | UAV LiDAR point clouds | Fuel metrics from LidarForFuel (Section 13) |
| **Scale** | Training site only | Extensible to 100s km² forest plots |

### Architecture

**Encoder (Shared with Point Cloud Model):**
- LocalGlobalPointAttentionBlock (feature extraction)
- NAIPEncoder + UAVSAREncoder (image encoding)
- CrossAttentionFusion (multi-modal fusion)
- Output: Fused point features [N_dep, feature_dim=256]

**Decoder (Raster-Specific):**
1. **Query-Based Grid Aggregation:**
   - Learnable queries [25, feature_dim] (one per 2m × 2m grid cell)
   - Distance-masked cross-attention: queries attend to points within R=3m radius
   - Handles temporal mismatch (2015 geometry vs 2016-2024 imagery) and empty cells
2. **Raster Decoder:**
   - 1×1 Conv MLP (preserves sharp boundaries: roads, firebreaks)
   - Projects [256, 5, 5] → [n_bands, 5, 5]
   - Initially targets 3 bands: Height (Band 3), Total Fuel Load (Band 8), Total Cover (Band 15)

### Data Pipeline Status

**Completed:**
- Tile generation with fuel metrics ground truth (`generate_training_data_raster.py`)
- Two-stage normalization preprocessing (`train_test_split_and_precompute_raster.py`)
- Data augmentation (`data_augmentation_raster.py`)
- Training/validation splits: 19,019 training tiles, 2,114 validation tiles
- Augmented dataset: 19,019 tiles (1:1 ratio)
- Total training samples: 38,038 tiles
- Model implementation (`src/models/raster_head.py`, `src/models/multimodal_raster_model.py`)
- Training pipeline (`src/training/raster_training.py`)
- Entry point (`run_raster_model.py`)

See [docs/raster_model_implementation_plan.md](docs/raster_model_implementation_plan.md) for complete implementation plan.

### Two-Stage Normalization (Critical Difference)

**Point cloud pipeline:** Bbox normalization only (X,Y ∈ [-5,5]m, Z ∈ [0,max]m)

**Raster pipeline:** Bbox normalization → Z-score normalization
```
Raw coords → normalize_point_clouds_with_bbox()
  → X,Y ∈ [-5,5]m, Z ∈ [0,max]m (bbox-normalized)
  → apply z-score (mean=0, std=1) (z-score normalized)
```

**Why:** Resolves scale imbalance (Z was 8.9× larger than X/Y), enables:
- Faster training (lr=1e-3 vs 5e-4)
- Balanced gradient flow across dimensions
- Spatial attention mechanisms to work correctly

**Denormalization:** Before distance/positional encoding operations, denormalize from z-score → bbox-normalized (meter) space:
```python
point_pos_phys = point_positions_norm * coord_std + coord_mean
distances = torch.cdist(point_pos_phys[:, :2], grid_centers)  # Meters!
```

**Statistics computed on bbox-normalized coords** (not raw UTM coords):
- `coord_mean`, `coord_std` from `coordinate_normalization_stats.json`
- Stored in tile `norm_params` dict alongside bbox `center`/`scale`

### Optimizer Configuration (ScheduleFree AdamW)

- **Optimizer:** AdamWScheduleFree ([The Road Less Scheduled](https://arxiv.org/abs/2405.15682))
- **Learning rate range:** 1x-10x larger than scheduled approaches (typical: 6e-4 to 5e-3)
- **Current setting:** 1.5e-3 (within recommended range)
- **Tuning:** Can increase to 2e-3 or 3e-3 if convergence is slow, or decrease to 8e-4 if training is unstable
- **Note:** ScheduleFree optimizers eliminate the need for learning rate schedules through implicit scheduling
- **References:** [GitHub Implementation](https://github.com/facebookresearch/schedule_free)

### Forest Plot Validation Sites

**4 out-of-distribution sites** (Southern California montane forests):
- BluffMesa, NorthBigBear, ReyesPeak (full imagery), Laguna (NAIP only)

**Available data per site:**

| Site | 3DEP | NAIP | UAVSAR | Field Plots |
|------|------|------|--------|-------------|
| BluffMesa | ✅ | ✅ | ✅ | 10 |
| NorthBigBear | ✅ | ✅ | ✅ | 19 |
| ReyesPeak | ✅ | ✅ | ✅ | 21 |
| Laguna | ✅ | ✅ | ❌ | 64 |

**⚠️ UAVSAR Gap:** Laguna (56% of plots) has **no UAVSAR coverage**. The model's cross-attention fusion handles this gracefully (skips UAVSAR branch when `uavsar=None`), but training data augmentation should include modality dropout to ensure robustness.

**Modality dropout augmentation (recommended):**
```python
# During training, randomly drop modalities (10-15% of tiles)
if random.random() < 0.15:
    tile['uavsar'] = None  # or tile['naip'] = None
```

**Validation workflow:**
1. Generate tiles + extract imagery (same pipeline as Volcan training site)
2. Run trained raster model → predictions [n_bands, 5×5] per tile
3. Compare to field measurements
4. Assess generalization to different forest types, elevations, management histories
5. **Track Laguna separately** to assess NAIP-only prediction quality

### Backward Compatibility

- **Point cloud pipeline unchanged:** Both pipelines coexist
- **Shared encoder:** Can transfer encoder weights between pipelines
- **CrossAttentionFusion updated:** Supports both normalized/unnormalized inputs (denormalization optional)
- **Training entry points:** `run_model_test.py` (point cloud), `run_raster_model.py` (raster, pending)

### Fuel Metrics Bands (Ground Truth)

**Source:** Generated by LidarForFuel pipeline (Section 13) applied to UAV LiDAR

**Summary metrics** (23 bands, Band 22 removed due to corruption):
- Structural: Height (Band 3), Canopy Base Height, Fuel Strata Gap, Vertical Complexity Index
- Fuel loads: Total (Band 8), Canopy, Midstorey, Surface
- Cover: Total (Band 15), Canopy, Midstorey, Understory
- Additional: Plant Area Index, entropy metrics, max bulk density

**Bulk density profile:** 150 vertical layers (1m resolution) - not used in initial implementation

**NA handling:**
- Bands 3, 8, 15 (initial targets): NA→0 before normalization (physical meaning: absence of vegetation)
- Structural bands (1-2, 4-7, 17-18): Keep NA→-999 after normalization (undefined without vegetation)

## 16. 3DEP HAG and Enhanced Features Pipeline

**Purpose:** Replace raw Z elevation with Height Above Ground (HAG) and add enhanced point features for better model performance.

### Features Computed

| Feature | Source | Description |
|---------|--------|-------------|
| HeightAboveGround | SMRF + Delaunay | True height above ground surface |
| Planarity | Eigenvalues (knn=15) | How planar the local neighborhood [0,1] |
| Sphericity | Eigenvalues (knn=15) | How spherical the local neighborhood [0,1] |
| Verticality | Eigenvalues (knn=15) | How vertical the local structure [0,1] |

### Processing Pipeline

```bash
# Process all sites (training + validation)
bash scripts/process_3dep_hag_features.sh

# Process single site with custom bbox
python src/data_prep/download_and_process_3dep_sites.py \
    --site my_site \
    --bbox "-116.5,33.0,-116.4,33.1" \
    --output-dir data/processed/3dep_hag_features/my_site

# Verify processed files
python scripts/verify_3dep_hag_features.py --dir data/processed/3dep_hag_features/

# Create STAC catalog from processed files
python src/data_prep/make_local_3dep_stac.py \
    --mode processed \
    --input-dir data/processed/3dep_hag_features \
    --output data/stac/3dep_hag
```

### PDAL Pipeline Stages

1. **readers.copc** - Read from Planetary Computer COPC tiles
2. **filters.reprojection** - Reproject to EPSG:32611 (UTM 11N)
3. **filters.assign** - Reset classification to 0
4. **filters.smrf** - Ground classification (cell=1.0, slope=0.15, threshold=0.5, window=18.0)
5. **filters.hag_delaunay** - Compute Height Above Ground
6. **filters.covariancefeatures** - Compute Planarity, Scattering, Verticality (knn=15)
7. **filters.ferry** - Rename Scattering to Sphericity (PDAL uses different name)
8. **writers.copc** - Write COPC with extra dimensions (enables efficient spatial queries)

### Output Structure

```
data/processed/3dep_hag_features/
├── {site_name}/
│   ├── {site_name}_hag_features.copc.laz    # Processed point cloud (COPC format)
│   ├── {site_name}_processing_metadata.json
│   ├── {site_name}_pipeline.json             # PDAL pipeline for debugging
│   └── processing_log.txt
└── logs/
    ├── processing_{timestamp}.log
    └── verification_{timestamp}.json
```

**Why COPC format?** COPC (Cloud Optimized Point Cloud) has an internal octree that enables spatial queries without reading the entire file. This is critical when generating tens of thousands of training tiles - each tile only reads ~1-10MB instead of the full 1-2GB file.

### Model Input Changes

- **Model input dimension:** Changes from `[N, 6]` to `[N, 9]` (3 coords + 6 attrs)
- **Config:** `attr_dim: int = 6` in `MultimodalRasterConfig`
- **Requires retraining:** Cannot load checkpoints trained with 3 attributes

### Backward Compatibility

- **Fallback to raw Z:** If HeightAboveGround not present in source, uses raw Z
- **Neutral attribute values:** Missing features (Planarity, Sphericity, Verticality) filled with neutral values
- **Legacy data:** Old tiles with 3 attributes still work but won't have enhanced features
