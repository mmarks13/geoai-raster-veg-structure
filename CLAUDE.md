# CLAUDE.md - Project Memory

## Documentation Map (Read These When Needed)
- [README.md](README.md) - Public-facing project overview (oriented around the published point-cloud-upsampling paper)
- [data/README.md](data/README.md) - Data directory structure, provenance, file formats, storage
- [src/README.md](src/README.md) - Source code organization, entry points
- [scripts/README.md](scripts/README.md) - Shell-script pipelines (data prep, 3DEP HAG, forest plot evaluation, veg structure metrics)
- [src/data_prep/README.md](src/data_prep/README.md) - STAC catalogs, tile generation, train/test splits
- [src/models/README.md](src/models/README.md) - Shared encoder (LG-PAB, ViT, fusion) and raster decoder
- [src/training/README.md](src/training/README.md) - DDP training (raster + legacy point-cloud)
- [src/evaluation/README.md](src/evaluation/README.md) - Inference, 3DEP baseline, forest plot evaluation
- [src/utils/README.md](src/utils/README.md) - Point cloud utilities, vegetation structure metrics, Chamfer, KNN
- [src/training/model_data_readme.md](src/training/model_data_readme.md) - Precomputed tile data structure spec

## 1. Project Overview

Multimodal fusion of sparse 3DEP LiDAR with NAIP optical and UAVSAR L-band SAR to recover vegetation structure at a quality comparable to dense UAV LiDAR. Two modeling approaches, with a shared encoder lineage.

**Active: Raster vegetation-structure prediction**
- **Task:** predict a small multi-band vegetation-structure raster per tile.
- **Output family:** the standardized vegetation structure variables of Moudry et al. (2023) computed from dense UAV LiDAR point clouds (max/mean/std height, canopy cover, canopy / mid-story / understory density, foliage height diversity, height percentiles, per-layer density). The exact subset of bands trained in any given run is governed by a band config under `src/evaluation/configs/raster/` and is not fixed.
- **Output scale:** 2 m pixels on a small grid per 10 m tile.
- **Ground truth:** metrics rasters computed by `src/utils/point_cloud_utils.py::compute_vegetation_structure_metrics` applied to UAV LiDAR; pipeline under `scripts/veg_structure_metrics/`.

**Historical: Point cloud upsampling (published)**
- **Task:** predict dense UAV-quality point clouds from sparse 3DEP.
- **Status:** published (Marks et al., *Remote Sensing* 2025). Code and checkpoints retained. The LG-PAB point-attention encoder, ViT image encoders, and cross-attention fusion from this work are reused as the shared encoder in the active raster model. Entry points: `run_ablation_study.py`, `run_model_test.py`.

**Stack:** PyTorch 2.5.1, CUDA 12.4, PyTorch Geometric, PyTorch3D.

## 2. Essential Commands

### Data Preparation
```bash
# Remote sensing acquisition (NAIP, UAVSAR, 3DEP, UAV LiDAR STAC catalogs)
bash scripts/get_data.sh

# Raster pipeline: tile grid, per-tile data extraction, train/val split, precompute
bash scripts/process_data_raster_v2.sh

# 3DEP HAG + geometric features (Height Above Ground, Planarity, Sphericity, Verticality)
bash scripts/process_3dep_hag_features.sh

# Vegetation structure metric rasters from dense UAV LiDAR (ground truth)
bash scripts/veg_structure_metrics/run_all_sites.sh
```

### Training
```bash
# Pretrain NAIP / UAVSAR image encoders (feeds the main raster model)
python -u run_pretrain_image_encoders.py

# Main raster training entry point
python run_raster_model.py

# Architecture/hyperparameter sweep
python run_raster_cross_attn_grid_mlp_sweep.py
```

### Evaluation
```bash
# Forest plot evaluation (4 OOD sites, MC-Dropout inference, multi-GPU)
bash scripts/evaluate_forest_plots.sh \
    --model data/output/raster_model_<tag>_<YYYYMMDD>_<HHMMSS>/checkpoints/epoch_<N>.pth \
    --band-config src/evaluation/configs/raster/<band_config>.json \
    --multi-gpu \
    --mc-samples <N> \
    --batch-size <B>

# 3DEP-only baseline: apply the Moudry metric pipeline directly to sparse 3DEP
# at the validation sites to isolate the multimodal-fusion value-add
python src/evaluation/compute_3dep_baseline_metrics.py
```

### Historical (point-cloud upsampling, published model)
```bash
python run_model_test.py          # Single model
python run_ablation_study.py      # Baseline / NAIP / UAVSAR / fused ablations
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

## 4. Architecture

### 4.1 Shared encoder (reused from the published point-cloud model)
- **LG-PAB point feature extractor** — local k-NN point attention + global position-aware attention. `src/models/multimodal_model.py`
- **Image encoders** — separate ViT encoders for NAIP and UAVSAR with temporal GRU aggregation. `src/models/encoders.py`
- **Cross-attention fusion** — multi-head cross-attention from point features into image patch embeddings. `src/models/cross_attn_fusion.py`
- **Alternative fusion (spatial/proximity)** available as a baseline. `src/models/fusion.py`

### 4.2 Raster decoder (new)
- Learnable per-cell grid queries attend into fused point features via Gaussian distance-biased cross-attention (within-tile only).
- Pre-LN FFN + small 1×1-Conv MLP head. Optional heteroscedastic output (mean + log-variance).
- Files: `src/models/raster_head.py`, `src/models/raster_primitives.py`, `src/models/multimodal_raster_model.py`.

### 4.3 Differences from the published model
These are intentional improvements that apply to both pipelines via the shared encoder:
- **Cleaner global attention math.** Q and K are position-aware, V carries pure semantic features — no positional leakage into aggregated values. See `PosAwareGlobalFlashAttentionV2` in `src/models/multimodal_model.py`.
- **No cross-tile contamination.** Global attention and k-NN neighborhoods are restricted to within-tile context via `batch_indices` + `to_dense_batch` key-padding masks. Enforced in both the encoder and raster head.
- **Learned NAIP tokenizer.** `PatchEmbeddingV2` in `src/models/encoders.py` uses a learned Conv2d patchifier (kernel=stride=patch_size) that preserves within-patch texture, replacing the earlier average-pool tokenizer.
- **Terrain-relative heights.** Z is Height-Above-Ground from the SMRF + Delaunay HAG pipeline (`scripts/process_3dep_hag_features.sh`), not raw Z.
- **Two-stage coordinate normalization.** Per-tile bbox normalization (X,Y ∈ [-5,5]m, Z ∈ [0,max]m) followed by z-score per axis. Stats live in `data/processed/model_data_raster/coordinate_normalization_stats.json`. Denormalize to meter-space before any distance-based operation (query attention, fusion cross-attention).
- **Richer per-point inputs (partial).** The 3DEP HAG pipeline computes Planarity / Sphericity / Verticality alongside Intensity / ReturnNumber / NumberOfReturns, but the raster dataset (`src/training/raster_dataset.py`) currently only exposes the 3 non-geometric attributes to the model. **Reason:** these features are eigenvalues of the knn=15 neighborhood computed once at preprocessing time, so they go stale under point-removal sparsification (up to 90% dropout) and are also inconsistent across sites with different 3DEP densities (6–22 pts/m²). Feeding the stored values would undercut the density invariance that sparsification aug is meant to induce. Re-enabling would require on-the-fly recomputation after augmentation, a multi-scale / radius-based variant, or explicit conditioning on density.

### 4.4 Regularization and OOD generalization
High-level list; see files for specifics.
- **Spectral normalization** on decoder linear/conv layers — `src/models/raster_primitives.py`, `src/models/raster_head.py`.
- **Stochastic depth / DropPath** in encoder residual branches — `src/models/encoders.py`, `src/models/multimodal_model.py`.
- **Heteroscedastic Gaussian NLL loss** with an overconfidence penalty is the primary training loss in current runs (`gaussian_nll_loss` in `src/training/raster_training.py`). Huber remains implemented behind a config flag (`use_heteroscedastic_loss`); the code default is currently Huber for backward compatibility, and active raster runs override to NLL.
- **MC Dropout at inference** — `src/evaluation/raster_inference.py::enable_mc_dropout` keeps dropout active and aggregates over multiple stochastic passes.
- **SWA** is implemented (`torch.optim.swa_utils` in `src/training/raster_training.py`) but MC Dropout is the retained inference-time ensembling method.
- **Online GPU augmentation** — `src/models/training_augmentation.py` provides modality dropout (NAIP / UAVSAR), synchronized geometric (rotation + reflection across points, imagery, and targets), temporal subsampling + date shift, point-cloud sparsification (requires global-only attention mode — no precomputed KNN graphs), point-level perturbations (jitter, noise, bird/outlier simulation, duplication, attribute augmentation), and image-level perturbations including z-score radiometric gain/bias, blur, erasing, and sharpness.

## 5. Data Pipeline

### 5.1 Raster training data
- `data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt`
- `data/processed/model_data_raster/precomputed_validation_tiles_raster_32bit.pt`
- `data/processed/model_data_raster/augmented_tiles_raster_32bit.pt`
- `data/processed/model_data_raster/coordinate_normalization_stats.json` — per-axis mean/std used for Stage-2 z-score normalization
- `data/processed/model_data_raster/fuel_metrics_normalization_stats.json` — target-band normalization stats (name is a legacy carry-over from the LidarForFuel era; the stats are now the vegetation-structure metric stats)

### 5.2 Raster tile schema (stable keys)
Live spec is `src/training/raster_dataset.py` + `src/training/model_data_readme.md`.
- `dep_points_norm` [N, 3] — z-scored points (X,Y centered at tile center; Z = Height Above Ground; per-axis z-scored using Stage-2 stats).
- `dep_points_attr_norm` [N, 3] — z-scored point attributes: Intensity, ReturnNumber, NumberOfReturns. Geometric attributes (Planarity, Sphericity, Verticality) are computed upstream in the 3DEP HAG pipeline but deliberately not consumed here; they go stale under point-removal augmentation and vary with 3DEP density across sites (see §4.3).
- `naip` — dict of multi-temporal 4-band (RGBN) chips over a 20 m × 20 m concentric bbox, or `None`.
- `uavsar` — dict of multi-temporal 6-channel polarimetric chips, or `None` (Laguna has no UAVSAR).
- `target` [n_bands, H, W] — vegetation-structure raster over the 2 m grid. Active bands are governed by the band config.
- `norm_params` — bbox center / scale plus Stage-2 coord mean/std.

### 5.3 Two-stage coordinate normalization
1. **Bbox normalize** (per tile): center X,Y at 0; set min Z to 0. Units = meters.
2. **Z-score** (global): subtract mean, divide by std per axis. Stats come from `coordinate_normalization_stats.json`.

Always denormalize to bbox (meter) space before distance-based operations so attention geometries are interpretable.

### 5.4 Tensor conventions
- Point clouds: `[N, 3]` for coords, `[N, F]` for features.
- Batched points use a **batch index tensor `[N]`**, not `[B, N, 3]` (PyG convention). Within-tile attention / KNN is enforced using this index.
- Edge indices: `[2, E]`.
- Images: `[n_images, C, H, W]` (NAIP: 4 bands, 40×40 at 0.5 m; UAVSAR: 6 bands, 4×4 at ~5 m).

### 5.5 Data loading notes
- Tiles are **lists of dicts loaded from `.pt` files**, not a `torch.utils.data.Dataset` subclass.
- DDP uses **manual per-GPU sharding** (not `DistributedSampler`).
- Augmented tiles are stored in a **separate `.pt` file** and combined with the originals in the training loop.

## 6. Training Infrastructure

- **Loop:** `src/training/raster_training.py` (raster), `src/training/multimodal_training.py` (historical PC upsampling).
- **Dataset:** `src/training/raster_dataset.py` (`ShardedRasterDataset`, raster variable-size collate).
- **DDP utils:** `src/training/ddp_training.py`.
- **Config:** `MultimodalRasterConfig` dataclass in `src/models/multimodal_raster_model.py` (raster); `MultimodalModelConfig` in `src/models/multimodal_model.py` (PC).
- **Checkpoints:** saved per run under `data/output/raster_model_<tag>_<YYYYMMDD>_<HHMMSS>/checkpoints/epoch_<N>.pth`. Best model selected by validation loss, not final epoch.
- **Hardware (reference):** 4× NVIDIA L40 (48 GB). Mixed-precision (AMP) + TF32 on Ampere+.

For specific hyperparameters (learning rate, batch size, dropout rates, augmentation probabilities, feature_dim, epoch count), read the config in `run_raster_model.py`. Values drift between experiments.

## 7. Validation Sites (OOD)

Four Southern California montane forest plot sites used to assess generalization beyond the two training sites (Sedgwick / Midland and Volcan Mountain). Total 114 field plots.

| Site | 3DEP | NAIP | UAVSAR | Field plots |
|------|------|------|--------|-------------|
| BluffMesa | ✅ | ✅ | ✅ | 10 |
| NorthBigBear | ✅ | ✅ | ✅ | 19 |
| ReyesPeak | ✅ | ✅ | ✅ | 21 |
| Laguna | ✅ | ✅ | ❌ | 64 |

Laguna (56 % of plots) has **no UAVSAR coverage**. Robustness to missing UAVSAR is enforced via modality dropout during training and graceful degradation at inference (cross-attention fusion skips the UAVSAR branch when `uavsar=None`).

Forest plot evaluation entry point: `scripts/evaluate_forest_plots.sh`.

## 8. 3DEP Baseline

To isolate the multimodal-fusion value-add, the same Moudry vegetation-structure pipeline used for UAV ground truth is applied directly to the sparse 3DEP point clouds at the validation sites. This produces a LiDAR-only reference that can be compared against both field measurements and the trained multimodal model.

Entry point: `src/evaluation/compute_3dep_baseline_metrics.py`. Uses `compute_vegetation_structure_metrics()` from `src/utils/point_cloud_utils.py`.

## 9. 3DEP HAG + Enhanced Features Pipeline

Replaces raw Z elevation with Height Above Ground (HAG) and adds geometric features on the 3DEP point cloud.

**Features computed**

| Feature | Source | Description |
|---------|--------|-------------|
| HeightAboveGround | SMRF + Delaunay | True height above ground surface |
| Planarity | Eigenvalues (knn=15) | How planar the local neighborhood |
| Sphericity | Eigenvalues (knn=15) | How spherical the local neighborhood |
| Verticality | Eigenvalues (knn=15) | How vertical the local structure |

**Pipeline commands**
```bash
# Process all sites (training + validation)
bash scripts/process_3dep_hag_features.sh

# Single site
python src/data_prep/download_and_process_3dep_sites.py \
    --site my_site \
    --bbox "-116.5,33.0,-116.4,33.1" \
    --output-dir data/processed/3dep_hag_features/my_site

# Verify
python scripts/verify_3dep_hag_features.py --dir data/processed/3dep_hag_features/

# Build STAC from processed files
python src/data_prep/make_local_3dep_stac.py \
    --mode processed \
    --input-dir data/processed/3dep_hag_features \
    --output data/stac/3dep_hag
```

**PDAL stages:** `readers.copc` → `filters.reprojection` (EPSG:32611) → `filters.assign` (reset class) → `filters.smrf` → `filters.hag_delaunay` → `filters.covariancefeatures` (knn=15) → `filters.ferry` (rename Scattering→Sphericity) → `writers.copc`.

**COPC output:** the octree enables efficient per-tile spatial queries, reading ~1–10 MB per tile rather than the full 1–2 GB file.

**Note:** the raster dataset currently consumes only the 3 non-geometric attributes (Intensity, ReturnNumber, NumberOfReturns). Planarity/Sphericity/Verticality are persisted in the COPC outputs but deliberately not wired through `raster_dataset.py` — see §4.3 for the reasoning (sparsification staleness + cross-site density variance).

**Output layout**
```
data/processed/3dep_hag_features/
├── {site_name}/
│   ├── {site_name}_hag_features.copc.laz
│   ├── {site_name}_processing_metadata.json
│   ├── {site_name}_pipeline.json
│   └── processing_log.txt
└── logs/
    ├── processing_{timestamp}.log
    └── verification_{timestamp}.json
```

## 10. Environment

### Main (Python / ML stack)
- **Env:** `geoai_env` (`environment.yml`)
- **Python:** 3.11.11
- **PyTorch:** 2.5.1 (CUDA 12.4)
- **Package manager:** Conda

### R (legacy fuel-metrics exploration)
- **Env:** `r_fuel_metrics` (`environment_r_fuel_metrics.yml`)
- **R:** 4.5.1; key packages: `r-lidr`, `r-terra`, `r-sf`, `LidarForFuel` (from GitHub)
- **Usage:** LidarForFuel-based fuel hazard mapping. Used only if revisiting the legacy fuel-metrics path (see §12). Invoked automatically via `conda run` from the wrapper scripts.

Separate environments are kept to prevent R/Python dependency conflicts.

## 11. Project Structure & Critical Paths

### Key directories
- `src/data_prep/` — STAC downloads, tile generation (raster + legacy PC), train/test splits
- `src/models/` — shared encoder + raster head + training augmentation
- `src/training/` — training loops, datasets, DDP utils
- `src/evaluation/` — inference, forest plot evaluation, 3DEP baseline, band configs
- `src/utils/` — vegetation structure metrics, Chamfer distance, KNN graph utilities
- `src/raster_mapping/` — forest plot visualization helpers
- `src/fuel_metrics/` — legacy LidarForFuel Python wrappers (see §12)
- `scripts/` — shell pipelines; subdirs: `veg_structure_metrics/`, `fuel_metrics/` (legacy)
- `data/processed/model_data_raster/` — raster training tiles
- `data/processed/3dep_hag_features/` — HAG + geometric-feature COPC files per site
- `data/processed/veg_structure_metrics/` — ground-truth metric rasters per site
- `data/output/` — checkpoints and logs per run
- `manuscript/` — published paper LaTeX and figures
- `run_*.py` — training and pretraining entry points at repo root

### Never modify unless directed
- `data/stac/`, `data/raw/`, `data/processed/training_data_chunks/`, `data/output/checkpoints/*.pth`, `data/output/logs/`, `.cache/`

**Exception:** `data/README.md` documents the structure and is safe to read.

### Legacy / historical code
- `src/*/legacy/` — superseded implementations (PC upsampling era). Still exist for `src/models/`, `src/training/`, `src/data_prep/`.
- `src/*/unused_alternatives/` — explored but not published. Exist for `src/models/`, `src/data_prep/`, `src/utils/`.
- `run_model_test.py`, `run_ablation_study.py` — published PC-upsampling entry points; kept for reproducibility.

## 12. Legacy: LidarForFuel fuel-metrics pipeline

**Status:** **demoted / exploratory.** The project originally targeted the LidarForFuel fuel-metric raster (173 bands from Beer-Lambert radiative transfer: canopy height, CBH, FSG, fuel loads, VCI, bulk-density profile). Early OOD prediction attempts with this target were unreliable, so the raster target was simplified to the Moudry vegetation-structure family (§1). The fuel-metrics code is retained and may be revisited — the original hypothesis that the fuel metrics were unreliable was never conclusively confirmed.

**Code preserved at:**
- `src/fuel_metrics/` — Python wrappers, orchestration, visualization
- `scripts/fuel_metrics/` — PDAL + R pipeline
- `data/processed/fuel_metrics/` — per-site outputs

Follow [`src/fuel_metrics/README.md`](src/fuel_metrics/README.md) for operational specifics if revisiting.

## 13. Gotchas

### Data
- Tiles are **not batched as `[B, N, 3]`** — use PyG-style batch index tensors.
- **Variable point counts** per tile; collate accordingly.
- **NAIP and UAVSAR dates do not align** — each has its own temporal stack and relative-date vector.
- **Laguna has no UAVSAR** — the model must handle `uavsar=None`.

### Training
- Best model != final epoch. Save by val loss.
- DDP uses **manual per-GPU sharding**, not `DistributedSampler`.
- Augmented tiles live in a **separate `.pt` file**; combine in the loader.
- Attention head configs are granular (separate for extractor / expansion / refinement). A few legacy params are ignored for backward compatibility.

### Raster-specific
- **Denormalize before distances.** The two-stage normalization means coords in the model are z-score-space, not meters. Always go back to bbox (meter) space for query cross-attention, fusion, and anywhere `torch.cdist` or positional encodings are applied.
- **Loss config flag.** The code default `use_heteroscedastic_loss=False` falls back to Huber; active raster runs override to NLL. Don't assume the default matches the current experiment.
- **Point-attr mismatch.** `dep_points_attr_norm` is 3-dim in the raster dataset despite the 3DEP HAG pipeline producing 6 attributes. This is deliberate (see §4.3) — re-enabling the geometric features would require handling their staleness under sparsification augmentation. If the model config advertises `attr_dim=6`, the last 3 channels are effectively unused; check this before interpreting attribution or importance analyses.

### Metrics & coordinates
- Point clouds are in **meter-scale coordinates** pre-normalization. Density-aware Chamfer distance (α=4) — used in the published PC model — was tuned for meter-scale; it is not applicable to the raster model, which trains with regression losses (NLL / Huber).
- All CRS is **EPSG:32611** (UTM 11N). CRS mismatches fail loudly upstream per the geospatial standards above.

### LAS coordinate precision
- PDAL auto-adjusts coordinate scale based on extent. When writing new LAS/LAZ, include a precision suffix in the filename (e.g., `_1cm.laz`, `_1mm.laz`). Verify with `pdal info --summary file.las | grep scale_`.
