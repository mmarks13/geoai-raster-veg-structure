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

## 3. Code Standards & Style

### Core Development Principles
- **Minimal viable implementation** - Build exactly what's requested, no anticipatory features
- **Ask before assuming** - Present options when ambiguous; flag assumptions explicitly
- **Optimize for readability** - Clear code over micro-optimizations unless performance is critical
- **No secrets in code/config** - All credentials via environment variables or secret managers

### Python Language Standards
- **Descriptive names** - Full words: `customer_email` over `cust_email` or `ce`
- **Type hints** - All function signatures for clarity and IDE support
- **Docstrings** - All public functions/classes with purpose, parameters, return values
- **Simple comprehensions** - List/dict comprehensions for simple cases only; multi-line/nested → regular loops

### Code Organization
- **Organize by purpose** - Top-level folders by system purpose, flat files within, subfolders only for 3-4+ files
- **Python scripts go in `src/`** - Organized by purpose (data_prep, models, training, etc.); `scripts/` is for shell scripts only
- **Target <40 lines per function** - Split on distinct responsibilities, not just line counts; 50+ acceptable if readable
- **Abstract on second use** - Extract truly identical logic on second occurrence; wait for patterns if purposes differ
- **Named constants** - Config thresholds, timeouts, business values; skip obvious one-offs
- **Return early** - Guard clauses for edge cases first, reduces nesting

### Error Handling & Validation
- **Fail fast** - Let exceptions propagate; specific catches (`except SpecificError`), re-raise with context
- **Contextual error messages** - Include what failed + actionable context: `ValueError(f"Config '{key}' not found. Check env vars.")`
- **Avoid fallback values** - Don't mask failures unless explicitly instructed

### Documentation & Comments
- **Comment flow/context** - One-line labels for multi-step processes; "why" for non-obvious decisions/trade-offs
- **Clarify complex logic** - Explain what isn't self-evident; avoid restating obvious code

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
- `src/models/` - Model architecture (multimodal_model.py, encoders.py, cross_attn_fusion.py, fusion.py)
- `src/training/` - Training loop (multimodal_training.py, ddp_training.py)
- `src/evaluation/` - Inference, stats, figures
- `src/utils/` - Chamfer distance, KNN graphs, point cloud utilities
- `src/raster_mapping/` - Forest plot visualization utilities
- `scripts/` - **Shell scripts only** (get_data.sh, process_data.sh, compress_las_files.sh)
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

## 9. Git Workflow
- **Main branch:** `cleanup` (current), not `main`
- **Commit message format:** Standard descriptive commits
- **Data directory:** Only `.gitkeep`, `README.md`, and critical config files (e.g., `test_val_polygons.geojson`) are tracked
- **Never commit:** `*.pt`, `*.pth`, `*.las`, `*.laz`, `*.tif`, checkpoints, logs, STAC catalog contents, `.cache/`
- **Granular .gitignore:** Uses extension-based patterns to allow structure while ignoring data files

## 10. Environment
- **Python:** 3.11.11
- **PyTorch:** 2.5.1 (CUDA 12.4)
- **Package manager:** Conda
- **Environment file:** `environment.yml`
- **Hardware (published):** 4× NVIDIA L40 (48GB)
- **CUDA required:** 12.4 toolkit

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

## 13. Common Issues & Fixes

### NaN Losses
- Check normalization (points should be in [-5, 5] for x,y)
- Verify no empty point clouds (should be filtered in preprocessing)
- Reduce learning rate or check gradient clipping
