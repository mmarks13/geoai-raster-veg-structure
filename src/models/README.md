# Models

Model architecture components for multimodal LiDAR point cloud enhancement using Local-Global Point Attention Blocks (LG-PAB).

## Active Files

### Core Model

- `multimodal_model.py` - Main `MultimodalPointUpsampler` model with LG-PAB architecture
  - Implements Local-Global Point Attention Blocks (feature extraction, expansion, refinement)
  - Integrates Vision Transformer encoders for imagery modalities
  - Configurable fusion strategies and modality flags for ablation studies
  - Defines `MultimodalModelConfig` dataclass for model hyperparameters

### Encoders

- `encoders.py` - Vision Transformer encoders for remote sensing imagery
  - `NAIPEncoder`: Processes NAIP optical imagery (4-band RGB-NIR)
  - `UAVSAREncoder`: Processes UAVSAR L-band SAR imagery (6-band polarimetric)
  - Both use patch-based ViT architecture with temporal GRU aggregation for multi-date image stacks

### Fusion Modules

- `cross_attn_fusion.py` - Cross-attention fusion module (used in published work)
  - Multi-head cross-attention mechanism with position encoding
  - Query: point features, Key/Value: image patch embeddings
  - Position-aware attention for spatial alignment

- `fusion.py` - Spatial fusion module (alternative approach)
  - Distance-weighted proximity fusion with temperature scaling
  - Configurable alternative to cross-attention fusion

## Configuration

Model behavior is controlled via `MultimodalModelConfig`:

**Modality Flags** (for ablation studies):
- `use_naip`: Enable/disable NAIP optical imagery input
- `use_uavsar`: Enable/disable UAVSAR SAR imagery input

**Fusion Type**:
- `fusion_type='cross_attention'`: Use cross-attention fusion (default, used in paper)
- `fusion_type='spatial'`: Use spatial distance-weighted fusion

## Usage

Models are instantiated in `run_ablation_study.py` and `run_model_test.py`:

```python
from src.models.multimodal_model import MultimodalModelConfig, MultimodalPointUpsampler

config = MultimodalModelConfig(
    feature_dim=256,
    k=16,
    use_naip=True,
    use_uavsar=True,
    fusion_type='cross_attention'
)

model = MultimodalPointUpsampler(config)
```

## Subfolders

### `legacy/`

Superseded implementation replaced by current multimodal architecture:

- `model.py` - Original point cloud upsampling model using PointTransformerConv layers
  - **Architecture:** FeatureExtractor (2-layer PointTransformerConv) → FeatureExpansion (3-layer MLP with interpolation) → PointRefiner (coordinate prediction MLP)
  - **Input:** LiDAR only (no multimodal fusion)
  - **Replaced by:** `multimodal_model.py` with LG-PAB architecture supporting NAIP/UAVSAR imagery fusion

---

See [../../README.md](../../README.md) for complete workflow documentation.
