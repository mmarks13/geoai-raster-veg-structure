# Internal Documentation - geoai_veg_map

**Purpose**: Track repository structure, legacy files, and architectural decisions for maintenance and cleanup.

**Last Updated**: 2025-10-04

---

## Active Architecture

**Entry Points:**
- `run_model_test.py` - Train single model
- `run_ablation_study.py` - Run all ablations (baseline, NAIP, UAVSAR, fused)

**Core Model (`src/models/`):**
- `multimodal_model.py` - LG-PAB architecture, main model
- `encoders.py` - ViT encoders for NAIP/UAVSAR
- `cross_attn_fusion.py` - Cross-attention fusion (used in paper)
- `fusion.py` - SpatialFusion (alternative, not used in paper but kept as configurable option)

**Training (`src/training/`):**
- `multimodal_training.py` - Main training loop, calls ddp_training utilities
- `ddp_training.py` - **Internal utility module** (NOT in README)
  - Provides: `setup_ddp()`, `cleanup()`, `find_free_port()`, `monitor_gpu_stats()`, `ModelConfig`
  - Imported by multimodal_training.py line 32
  - Users interact with multimodal_training.py, not this file directly
- `optuna.py` - Hyperparameter tuning (active but not in published workflow, keep for future use)

**Data Pipeline:**
1. `scripts/get_data.sh` → Download UAVSAR, NAIP, 3DEP, create STAC catalogs
2. `scripts/process_data.sh` → Generate tiles, split train/test, precompute KNN graphs, augmentation
3. Output: `data/processed/model_data/*.pt` files
4. Training: `run_ablation_study.py` → `multimodal_training.py` → `multimodal_model.py`
5. Evaluation: `inference_eval.py`, `RQ_test_v2.py`, `manuscript_figures.py`

---

## Legacy Files (Candidates for Deletion)

**Phase 1: Safe to Delete Now (no imports found):**
1. `src/training/train.py` - Early training script, hard-coded paths to old data (`augmented_training_tiles_60k.pt`)
2. `src/training/preprocess.py` - Old preprocessing, replaced by `src/data_prep/`
3. `src/training/run_model_test.py` - Duplicate of root-level version

**Phase 2: Verify Before Deletion:**
4. `src/training/training.py` - Mid-stage training script, imports legacy model.py
5. `src/models/model.py` - Original PointTransformerConv model (LiDAR-only, no multimodal fusion)
   - Imported by `ddp_training.py` line 24: `from src.models.model import PointUpsampler`
   - Need to verify if ddp_training.py actually uses it (likely just legacy import)

**Verification Commands:**
```bash
grep -n "PointUpsampler" src/training/ddp_training.py
grep -r "training.py" --include="*.py" --include="*.sh"
grep -n "from src.models.model import" src/training/multimodal_training.py
```

**Import Dependency Tree:**
- Active: `run_ablation_study.py` → `multimodal_training.py` → `ddp_training.py` (utils) + `multimodal_model.py`
- Legacy: `model.py` ← `training.py`, `ddp_training.py` (line 24, likely unused)

---

## Model Evolution

**Phase 1 (Legacy):** `model.py` - PointTransformerConv, LiDAR-only, trained with `train.py`/`training.py`

**Phase 2 (Current):** `multimodal_model.py` - LG-PAB, LiDAR+NAIP+UAVSAR, ViT encoders, cross-attention, GRU temporal aggregation, Flash attention

---

## Fusion Strategies (Both Active)

1. **CrossAttentionFusion** (`cross_attn_fusion.py`) - Used in paper, multi-head attention, position-aware
2. **SpatialFusion** (`fusion.py`) - Alternative, distance-weighted proximity fusion
   - Config: `fusion_type='cross_attention'` (default) vs `'spatial'`
   - Published work uses cross_attention exclusively
   - Keep SpatialFusion as intentional architectural option for future experiments

---

## Key Facts

- **Training**: 4x NVIDIA L40 (48GB), batch=15/GPU (60 total), ~7hrs/model, CUDA 12.4
- **Data**: STAC-based organization, precomputed KNN graphs (32-bit), 10m×10m tiles
- **Selection**: Best model = lowest validation loss (not final epoch)
- **Ablations**: baseline (LiDAR), +NAIP, +UAVSAR, fused (all)
- **File counts removed from README** to avoid maintenance burden

---

## README Philosophy

**Included**: Conceptual workflow steps (user-facing scripts)
**Excluded**:
- Internal utilities (ddp_training.py, h5_chunk_loader.py, bbox_tile_filter.py, etc.)
- Development tools (optuna.py)
- Supporting scripts called by main workflow

**Rationale**: README shows what users need to understand/run, not implementation details

---

## Complete File Inventory

### Data Preparation Files (`src/data_prep/`)

**Active - In Main Workflow:**
- `make_local_naip_stac.py` - Download NAIP imagery, create STAC catalog
- `make_local_uavsar_stac.py` - Download UAVSAR data, create STAC catalog
- `make_local_3dep_stac.py` - Download 3DEP LiDAR, create STAC catalog
- `make_local_uavlidar_stac.py` - Catalog UAV LiDAR ground truth
- `generate_training_data.py` - Generate 10m×10m training tiles
- `train_test_split_and_precompute.py` - Combined split + precompute (current workflow)
- `data_augmentation.py` - Geometric & point perturbations

**Active - Supporting Utilities (called by main workflow, not in README):**
- `process_uav_lidar.py` - Process raw UAV LiDAR data
- `create_training_tile_bboxes.py` - Generate training tile bounding boxes
- `h5_chunk_loader.py` - Combine training data chunks into single file
- `pointcloud_footprints_to_geojson.py` - Export point cloud footprints
- `bbox_tile_filter.py` - Filter tiles by bounding box
- `imagery_stac.py` - STAC utilities for imagery loading (imported by generate_training_data.py)
- `imagery_training_data.py` - Imagery data extraction for training tiles
- `las_to_copc_stac.py` - Convert LAS to Cloud-Optimized Point Cloud + STAC
- `process_pointcloud_stac.py` - Point cloud processing utilities (gridding, aggregation)
- `compress_las.py` - Simple utility to compress .las to .laz

**Legacy/Superseded:**
- `split_train_test_val_tiles.py` - Old separate train/test split (replaced by train_test_split_and_precompute.py)
- `precompute_data.py` - Old separate precompute step (replaced by train_test_split_and_precompute.py)

**Experimental/Alternative Data Sources (not used in published work):**
- `uavsar_to_stac.py` - Alternative UAVSAR STAC catalog creation (standalone script)
- `wv2_to_stac.py` - WorldView-2 satellite imagery processing (not used, alternative to NAIP)

### Evaluation Files (`src/evaluation/`)

**Active - In Main Workflow:**
- `inference_eval.py` - Run inference on test set, generate predictions
- `manuscript_figures.py` - Generate publication figures
- `RQ_test_v2.py` - Statistical tests for research questions
- `generate_eval_df.py` - Generate evaluation dataframes

**Active - Development/Analysis Tools (not in README):**
- `val_eval.py` - Validation evaluation utilities (imported by model_val_report.py)
- `model_val_report.py` - Generate validation reports (imports from val_eval.py)
- `model_comparison_report.py` - Multi-model comparison reports with 3D visualizations
- `df_based_model_comparison_report.py` - DataFrame-based model comparison visualization

### Utilities Files (`src/utils/`)

**Active - Core Utilities:**
- `chamfer_distance.py` - Point cloud reconstruction metric (used throughout)
- `knn_graph_gpu.py` - GPU-accelerated KNN graphs (used in training)
- `point_cloud_utils.py` - Point cloud processing utilities

**Active - Experimental/Optional:**
- `infocd.py` - InfoCD loss + repulsion loss (commented out in multimodal_training.py line 32)
  - Experimental loss function, not used in published work but available
- `training_data_eval.py` - Training data quality evaluation/filtering utilities
- `dtm_calc.py` - Digital Terrain Model calculation utilities (raster gridding, kriging)
- `octree_downsampling.py` - Octree-based point cloud downsampling (not currently used)

---

## Changelog

**2025-10-04:**
- Initial documentation during README cleanup
- Identified 5 legacy files for deletion
- Documented ddp_training.py as internal utility (excluded from README)
- Removed file counts from README
- Clarified SpatialFusion vs CrossAttentionFusion (both active, different purposes)
- **Added complete file inventory** covering all 46 Python files in repository
- Categorized files: main workflow, supporting utilities, legacy, experimental
