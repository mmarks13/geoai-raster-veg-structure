# Models

Model architecture components for multimodal LiDAR + imagery fusion. The encoder is shared across two pipelines; the decoder differs.

- **Raster vegetation-structure prediction (active).** Shared encoder → raster decoder (learnable grid queries + distance-biased cross-attention + small MLP head). Predicts a multi-band raster per tile.
- **Point cloud upsampling (published historical).** Shared encoder → feature expansion + refinement (LG-PAB) → MLP coordinate decoder. Predicts dense point clouds.

## Shared encoder

- `multimodal_model.py` — LG-PAB point feature extractor and the `MultimodalPointUpsampler` (published point-cloud model). Defines `PosAwareGlobalFlashAttentionV2`, which cleanly separates position-aware Q/K from semantic-only V and enforces within-tile attention via `batch_indices` + `to_dense_batch` masking. Config: `MultimodalModelConfig`.
- `encoders.py` — ViT encoders for imagery.
  - `NAIPEncoder`: 4-band (RGBN) 40×40 chips, multi-temporal with GRU aggregation.
  - `UAVSAREncoder`: 6-channel polarimetric 4×4 chips, multi-temporal with GRU aggregation.
  - `PatchEmbeddingV2`: learned Conv2d patchifier (kernel=stride=patch_size) replacing the earlier average-pool tokenizer.
  - Stochastic-depth / DropPath hooks for regularization.
- `cross_attn_fusion.py` — multi-head cross-attention. Query = point features; Key/Value = image patch embeddings. Supports both normalized and denormalized point inputs (denormalize to meters before distance operations).
- `fusion.py` — distance-weighted spatial fusion (alternative to cross-attention).

## Raster decoder

- `multimodal_raster_model.py` — top-level raster model. Config: `MultimodalRasterConfig`.
- `raster_head.py` — composes the raster decoder: `LearnableGridQueries` + `GaussianDistanceBiasedCrossAttention` + `PreLNFFN` + `SmallMlpDecoder`. `SmallMlpDecoder` optionally produces mean + log-variance for heteroscedastic output. Spectral normalization applied on the decoder MLP.
- `raster_primitives.py` — reusable blocks (spectral-norm-wrapped MLPs, Pre-LN feedforward, small MLP decoder).

## Augmentation

- `training_augmentation.py` — online GPU augmentation orchestrator. Includes:
  - `PointCloudAugmentation` (coord jitter, intensity noise, bird-strike / omnidirectional outliers, point duplication).
  - `ReturnAttributeAugmentation`.
  - `NAIPAugmentation`, `UAVSARAugmentation` (Kornia-based blur, erasing, sharpness, z-score radiometric gain/bias, etc.).
  - `SynchronizedGeometricAugmentation` (rotation + reflection applied identically to points, imagery, and target rasters).
  - `TemporalSubsamplingAugmentation` (NAIP/UAVSAR T-dim and UAVSAR G-dim subsampling; date shift).
  - `ModalityDropoutAugmentation` (drops NAIP and/or UAVSAR entirely — mirrors the Laguna OOD site with no UAVSAR).
  - `PointCloudSparseAugmentation` (random point removal; requires global-only attention mode, not the KNN-precomputed mode).

## Configuration

**Raster (`MultimodalRasterConfig`):** modality flags, point attr dim, feature dim, decoder query grid size, heteroscedastic flag, augmentation flags.

**Point cloud (`MultimodalModelConfig`):**
- `use_naip`, `use_uavsar` — modality flags for ablations.
- `fusion_type`: `'cross_attention'` (published) or `'spatial'`.

## Subfolders

### `legacy/`
- `model.py` — original point-cloud upsampling model (PointTransformerConv) prior to LG-PAB and multimodal fusion. Kept for reference.

### `unused_alternatives/`
Explored but not shipped.

---

See [../../README.md](../../README.md) and [../../CLAUDE.md](../../CLAUDE.md).
