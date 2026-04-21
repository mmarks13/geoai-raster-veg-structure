# GeoAI Raster Vegetation Structure

Multimodal GeoAI for predicting fine-scale vegetation structure rasters from sparse airborne LiDAR, optical imagery, and SAR.

## Overview

This repository contains the active raster-model branch of the broader `geoai_veg_map` project. The goal is to recover vegetation structure variables from sparse USGS 3DEP LiDAR by fusing it with multi-temporal NAIP imagery and UAVSAR L-band SAR.

For a broad AI/ML audience, the core idea is straightforward: learn a shared representation over 3D geometry and imagery, then decode that representation into a small raster of vegetation structure metrics for each tile.

### Inputs

- Sparse 3DEP airborne LiDAR
- Multi-temporal NAIP optical imagery
- Multi-temporal UAVSAR L-band SAR imagery

### Outputs

- Small multi-band vegetation-structure rasters on a 2 m grid per 10 m tile
- Configurable target-band subsets, including height, cover, density, and related vegetation-structure variables

## What Is New Here

- Point-attention encoder for sparse LiDAR geometry
- Multimodal fusion of LiDAR, optical imagery, and SAR
- Raster decoder with learnable grid queries and distance-biased cross-attention
- Out-of-distribution evaluation on independent Southern California forest sites
- 3DEP-only baseline and MC-dropout uncertainty estimation

## Project Status

- Active: raster vegetation-structure prediction
- Historical: point-cloud upsampling code from the published 2025 *Remote Sensing* paper is retained where it provides encoder lineage and reusable components

The original published point-cloud project lives in the related repository: `geoai_veg_map`.

## Repository Layout

```text
geoai-raster-veg-structure/
├── src/data_prep/          # Tile generation, preprocessing, train/val splits
├── src/models/             # Shared encoder, fusion modules, raster decoder
├── src/training/           # DDP training and dataset loading
├── src/evaluation/         # Inference, OOD evaluation, 3DEP baseline
├── src/utils/              # Point-cloud and vegetation-structure utilities
├── scripts/                # Data prep, feature pipelines, evaluation entry points
├── run_pretrain_image_encoders.py
├── run_raster_model.py
└── run_raster_cross_attn_grid_mlp_sweep.py
```

## Getting Started

### 1. Environment

```bash
conda env create -f environment.yml
conda activate geoai_env
```

If you need the vegetation-structure metric generation pipeline, the repository also includes the supporting scripts and environment files for that workflow.

### 2. Data Preparation

```bash
bash scripts/get_data.sh
bash scripts/process_data_raster_v2.sh
bash scripts/process_3dep_hag_features.sh
bash scripts/veg_structure_metrics/run_all_sites.sh
```

### 3. Training

```bash
python -u run_pretrain_image_encoders.py
python run_raster_model.py
python run_raster_cross_attn_grid_mlp_sweep.py
```

### 4. Evaluation

```bash
bash scripts/evaluate_forest_plots.sh \
    --model data/output/raster_model_<tag>_<YYYYMMDD>_<HHMMSS>/checkpoints/epoch_<N>.pth \
    --band-config src/evaluation/configs/raster/<band_config>.json

python src/evaluation/compute_3dep_baseline_metrics.py
```

## Model Framing

The active model reuses a shared encoder lineage from the earlier point-cloud work:

- Local-global point attention over sparse LiDAR
- ViT-based image encoders for NAIP and UAVSAR
- Cross-attention fusion between point features and image patch embeddings

On top of that shared encoder, this repository adds a raster-specific decoder that predicts vegetation-structure rasters instead of dense point clouds.

## Related Publication

The encoder lineage comes from:

Marks, M.; Sousa, D.; Franklin, J. "Attention-Based Enhancement of Airborne LiDAR Across Vegetated Landscapes Using SAR and Optical Imagery Fusion." *Remote Sensing* 2025, 17, 3278. https://doi.org/10.3390/rs17193278

## Notes

- This repo is intended to stand alone as the raster vegetation-structure project.
- Some historical code remains because it is still part of the model lineage or evaluation context.
- Public-facing documentation will continue to tighten around the raster workflow as the project stabilizes.
