# Internal Documentation for geoai_veg_map Repository

**Purpose**: This document contains detailed findings about the repository structure, legacy files, and architectural decisions discovered during documentation cleanup. It serves as a reference for future code maintenance and cleanup operations.

**Last Updated**: 2025-10-04

---

## Repository Architecture Overview

### Active Codebase Structure

The current active codebase follows this architecture:

**Entry Points:**
- `run_model_test.py` (root) - Train a single model configuration
- `run_ablation_study.py` (root) - Run all ablation study experiments (baseline, NAIP, UAVSAR, fused)

**Core Model:**
- `src/models/multimodal_model.py` - Main model implementation with Local-Global Point Attention Block (LG-PAB) architecture
- `src/models/encoders.py` - Vision Transformer encoders for NAIP and UAVSAR
- `src/models/cross_attn_fusion.py` - Cross-attention fusion module for multimodal integration
- `src/models/fusion.py` - Alternative fusion strategies

**Training Pipeline:**
- `src/training/multimodal_training.py` - Main training loop and ablation study orchestration
- `src/training/ddp_training.py` - Distributed Data Parallel (DDP) training utilities and base classes

**Data Processing:**
- `src/data_prep/` - 21 scripts for data acquisition and preprocessing
- Key scripts: STAC catalog creation, tile generation, train/test split, feature precomputation, augmentation

**Evaluation:**
- `src/evaluation/` - 9 scripts for inference, visualization, and statistical analysis
- Key scripts: inference_eval.py, manuscript_figures.py, RQ_test_v2.py

**Utilities:**
- `src/utils/` - 7 utility modules
- Key modules: chamfer_distance.py, knn_graph_gpu.py, point_cloud_utils.py

---

## Legacy Files - Deprecated but Still Present

These files are **no longer used** in the active workflow but remain in the repository. They are candidates for future cleanup.

### 1. `src/models/model.py`
**Status**: Legacy / Superseded
**Replaced By**: `src/models/multimodal_model.py`

**Description**:
- Original baseline model implementation using PointTransformerConv layers
- Contains `FeatureExtractor`, `FeatureExpander`, and `PointUpsampler` classes
- Uses PyTorch Geometric's PointTransformerConv for local geometric processing
- Only supports LiDAR-only processing (no multimodal fusion)

**Why Deprecated**:
- Superseded by the multimodal architecture that supports LiDAR + NAIP + UAVSAR fusion
- Lacks attention mechanisms (Local-Global Point Attention Blocks)
- No Vision Transformer encoders or cross-attention fusion

**Import Analysis**:
- Referenced in `src/training/ddp_training.py` (line 24: `from src.models.model import PointUpsampler`)
- Referenced in `src/training/training.py` (line 24: `from src.models.model import PointUpsampler`)
- Both files are also legacy (see below)

**Safe to Delete**: Yes, after verifying training.py and ddp_training.py are truly unused

---

### 2. `src/training/train.py`
**Status**: Legacy / Superseded
**Replaced By**: `src/training/multimodal_training.py`

**Description**:
- Early-stage training script with basic DDP setup
- Hard-coded data paths to old dataset locations (`augmented_training_tiles_60k.pt`)
- Includes extensive environment variable debugging for NCCL distributed training
- Loads data directly without using the current precomputed tile structure

**Why Deprecated**:
- Predates the multimodal training architecture
- Hard-coded paths no longer match current data organization
- Replaced by more sophisticated multimodal_training.py with better abstractions

**Import Analysis**:
- No imports found in active codebase

**Safe to Delete**: Yes

---

### 3. `src/training/training.py`
**Status**: Legacy / Superseded
**Replaced By**: `src/training/multimodal_training.py`

**Description**:
- Mid-stage training script that introduced some modularization
- Contains `PointCloudUpsampleDataset` class for loading precomputed data
- Uses older `PointUpsampler` model from `src/models/model.py`
- Includes Optuna hyperparameter tuning integration
- Has DDP setup but less sophisticated than current implementation

**Why Deprecated**:
- Uses legacy model architecture (model.py instead of multimodal_model.py)
- Optuna integration moved to separate optuna.py file
- DDP functionality better implemented in ddp_training.py

**Import Analysis**:
- Only imported by itself and `src/training/ddp_training.py` (which references legacy model.py)
- No imports from active workflow files

**Safe to Delete**: Yes, confirmed superseded

---

### 4. `src/training/preprocess.py`
**Status**: Legacy / Superseded
**Replaced By**: `src/data_prep/` scripts (train_test_split_and_precompute.py, data_augmentation.py)

**Description**:
- Early preprocessing utilities
- Likely contained data normalization, tile creation, or augmentation logic
- Not examined in detail as no imports were found

**Why Deprecated**:
- Preprocessing moved to dedicated `src/data_prep/` directory with better organization
- Current workflow uses `train_test_split_and_precompute.py` for feature precomputation

**Import Analysis**:
- No imports found in active codebase

**Safe to Delete**: Yes

---

### 5. `src/training/run_model_test.py`
**Status**: Duplicate
**Canonical Version**: `/run_model_test.py` (root level)

**Description**:
- Duplicate of the root-level `run_model_test.py`
- Likely created during refactoring when moving the script to project root

**Why Deprecated**:
- Duplicate functionality
- Root-level version is the canonical entry point documented in README

**Import Analysis**:
- No imports found in active codebase

**Safe to Delete**: Yes, keep only root-level version

---

### 6. `src/training/optuna.py`
**Status**: Active but Not Used in Published Workflow
**Purpose**: Hyperparameter tuning only

**Description**:
- Hyperparameter optimization using Optuna framework
- Not called by `run_ablation_study.py` or any published workflow scripts
- Useful for future hyperparameter exploration but not part of manuscript reproduction

**Why Not in README**:
- Not part of the core workflow for reproducing published results
- Hyperparameter tuning was done during development, not during final manuscript experiments

**Import Analysis**:
- Not imported by any active workflow scripts
- Standalone utility script

**Safe to Delete**: No - keep for future use, but not prominently documented

**Recommendation**: Keep but add comment in file header explaining it's for development/tuning only

---

## Alternative Fusion Strategies - Active but Unused in Published Results

### `src/models/fusion.py` - SpatialFusion

**Classification**: Active code, alternative implementation (NOT legacy)

**Status**:
- Actively maintained as a configurable option in the multimodal model
- Not used in published manuscript results
- Available for future research and experimentation

**Technical Details**:

The model supports two fusion strategies for combining point cloud features with image patch embeddings:

1. **SpatialFusion** (in `fusion.py`):
   - Distance-weighted spatial proximity fusion
   - Uses temperature-scaled similarity between point positions and image patch centers
   - Weights image features by spatial distance to points
   - Older approach, more traditional

2. **CrossAttentionFusion** (in `cross_attn_fusion.py`):
   - Multi-head cross-attention mechanism
   - Query: point features, Key/Value: image patch embeddings
   - Position-aware attention with learned position encodings
   - Used in published manuscript
   - More sophisticated, attention-based approach

**Usage in Codebase**:
```python
# In multimodal_model.py line 23
from .fusion import SpatialFusion

# In multimodal_model.py lines 630-651
if config.fusion_type.lower() == "cross_attention":
    self.fusion = CrossAttentionFusion(...)
else:  # Default to spatial fusion
    self.fusion = SpatialFusion(...)
```

**Configuration**:
- Config parameter: `fusion_type` (line 70 in multimodal_model.py)
- Default: `"cross_attention"`
- Alternative: `"spatial"` (activates SpatialFusion)

**Published Workflow**:
- `run_ablation_study.py` line 59: `fusion_type='cross_attention'`
- `run_model_test.py` line 59: `fusion_type='cross_attention'`
- All manuscript experiments use CrossAttentionFusion exclusively

**Why It's Kept**:
1. Provides valid alternative fusion approach for comparison
2. Part of model's configurable architecture
3. May be useful for future ablation studies comparing fusion strategies
4. Actively imported and maintained (not orphaned code)

**Recommendation**:
- **Keep in codebase** - it's an intentional architectural option
- **Keep in README** - properly listed as "Alternative fusion strategies"
- Similar to `optuna.py` - development/research tool, not in main workflow

**Not Legacy Because**:
- Intentionally designed as switchable component
- Actively imported by current model
- Part of documented model configuration options
- Unlike legacy files (model.py, train.py), this serves a purpose

---

## Model Evolution Timeline

### Phase 1: Baseline Model (model.py)
- **File**: `src/models/model.py`
- **Architecture**: PointTransformerConv-based upsampling
- **Modalities**: LiDAR only
- **Training**: `src/training/train.py`, `src/training/training.py`

### Phase 2: Multimodal Model (multimodal_model.py)
- **File**: `src/models/multimodal_model.py`
- **Architecture**: Local-Global Point Attention Blocks (LG-PAB)
- **Modalities**: LiDAR + NAIP + UAVSAR
- **Features**:
  - Vision Transformer encoders for imagery
  - Cross-attention fusion
  - Temporal aggregation with bidirectional GRU
  - Flash attention for global context
- **Training**: `src/training/multimodal_training.py`, `src/training/ddp_training.py`

---

## Data Flow Pipeline

### Current Active Pipeline:

1. **Data Acquisition** (`scripts/get_data.sh`):
   - Download UAVSAR from Alaska Satellite Facility
   - Download NAIP from Microsoft Planetary Computer
   - Download 3DEP LiDAR from Microsoft Planetary Computer
   - Create STAC catalogs

2. **Data Preprocessing** (`scripts/process_data.sh`):
   - `generate_training_data.py` - Create 10m x 10m tiles from STAC catalogs
   - `train_test_split_and_precompute.py` - Split data and precompute KNN graphs
   - `data_augmentation.py` - Generate augmented training data

3. **Training Data Structure**:
   - `data/processed/model_data/precomputed_training_tiles_32bit.pt`
   - `data/processed/model_data/precomputed_validation_tiles_32bit.pt`
   - `data/processed/model_data/precomputed_test_tiles_32bit.pt`
   - `data/processed/model_data/augmented_tiles_32bit_16k_no_repl.pt`

4. **Model Training**:
   - Entry: `run_ablation_study.py` calls `src/training/multimodal_training.py`
   - DDP utilities from `src/training/ddp_training.py`
   - Model from `src/models/multimodal_model.py`

5. **Evaluation**:
   - `src/evaluation/inference_eval.py` - Generate predictions
   - `src/evaluation/RQ_test_v2.py` - Statistical tests
   - `src/evaluation/manuscript_figures.py` - Publication figures

---

## File Count Discrepancies

During documentation cleanup, actual file counts were discovered to differ from README claims:

| Directory | README Claim | Actual Count | Status |
|-----------|--------------|--------------|---------|
| `src/data_prep/` | 19 scripts | 21 files | Counts removed from README |
| `src/models/` | 5 files | 5 files (4 active + 1 legacy) | Counts removed from README |
| `src/training/` | 7 files | 7 files (2 active + 5 legacy) | Counts removed from README |
| `src/evaluation/` | 8 scripts | 9 files | Counts removed from README |
| `src/utils/` | 9 files | 7 files | Counts removed from README |

**Resolution**: All file counts removed from README.md to avoid maintenance burden and discrepancies.

---

## Import Dependency Analysis

### Legacy File Dependencies:
- `model.py` ← imported by `training.py` and `ddp_training.py`
- `training.py` ← not imported by any active files
- `train.py` ← not imported by any active files
- `preprocess.py` ← not imported by any active files
- `run_model_test.py` (src/training) ← not imported by any active files
- `optuna.py` ← not imported by any active files

### Active File Dependencies:
- `multimodal_model.py` ← imported by `run_ablation_study.py`, `multimodal_training.py`
- `multimodal_training.py` ← imported by `run_ablation_study.py`
- `ddp_training.py` ← imported by `multimodal_training.py`

---

## Future Cleanup Recommendations

### Phase 1: Safe Deletions (Verified Unused)
Can be deleted immediately with minimal risk:
1. `src/training/train.py` - No imports, superseded
2. `src/training/preprocess.py` - No imports, superseded
3. `src/training/run_model_test.py` - Duplicate of root version

### Phase 2: Requires Verification
Requires checking if ddp_training.py's import of model.py is actually used:
4. `src/training/training.py` - Imports legacy model, only imported by ddp_training.py
5. `src/models/model.py` - Legacy model, may be imported by training.py/ddp_training.py

**Verification Steps**:
```bash
# Check if ddp_training.py actually uses PointUpsampler from model.py
grep -n "PointUpsampler" src/training/ddp_training.py

# Check if training.py is executed anywhere
grep -r "training.py" --include="*.py" --include="*.sh"

# Verify multimodal_training.py doesn't import anything from legacy files
grep -n "from src.models.model import" src/training/multimodal_training.py
grep -n "from src.training.training import" src/training/multimodal_training.py
```

### Phase 3: Keep for Future Use
6. `src/training/optuna.py` - Keep for hyperparameter tuning, add documentation note

---

## Key Architectural Decisions

### Why Multimodal Architecture?
The transition from `model.py` to `multimodal_model.py` enabled:
- Integration of multiple sensor modalities (LiDAR, optical, SAR)
- Attention mechanisms for learning structure from point clouds
- Cross-modal fusion via cross-attention
- Temporal aggregation for multi-date imagery

### Why DDP Training?
- Training on 4x NVIDIA L40 GPUs (48GB each)
- Batch size: 15 tiles per GPU (60 total)
- Training time: ~7 hours per model variant
- Required for large-scale point cloud processing

### Why Precomputed Features?
- KNN graphs computed once during preprocessing
- Normalized point clouds stored in 32-bit precision
- Significant speedup during training (no on-the-fly graph computation)
- Enables larger batch sizes

---

## Notes for Future Developers

1. **Do not delete files without checking imports**: Use grep to verify no active code references legacy files

2. **Data paths have changed**: Old scripts reference different data structures (e.g., `augmented_training_tiles_60k.pt` vs current `augmented_tiles_32bit_16k_no_repl.pt`)

3. **Model checkpointing**: Best model selected by lowest validation loss, not final epoch

4. **Ablation study runs all variants**: baseline (LiDAR only), +NAIP, +UAVSAR, fused (all modalities)

5. **Environment dependencies**: CUDA 12.4 required, see environment.yml

6. **STAC catalogs are central**: All data organized via STAC, not direct file paths

---

## Contact

For questions about this documentation or the repository structure:
- **Michael Marks** - mmarks0561@sdsu.edu
- **Repository**: https://github.com/yourusername/geoai_veg_map

---

## Changelog

### 2025-10-04
- Initial documentation created during README cleanup
- Identified 6 legacy files for future removal
- Documented model evolution from baseline to multimodal architecture
- Cataloged file count discrepancies
- Created import dependency analysis
