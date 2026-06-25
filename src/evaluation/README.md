# Evaluation

Inference, forest plot evaluation, baselines, and manuscript figure generation.

## Active: raster model evaluation

- `raster_inference.py` — inference on the raster vegetation-structure model. `enable_mc_dropout()` keeps dropout active at eval time so multiple stochastic forward passes produce a predictive mean and uncertainty.
- `compute_3dep_baseline_metrics.py` — **3DEP-only baseline.** Applies the Moudry vegetation-structure pipeline (`compute_vegetation_structure_metrics` from `src/utils/point_cloud_utils.py`) directly to sparse 3DEP point clouds at the validation sites. Produces a LiDAR-only reference to isolate the multimodal-fusion value-add of the trained model.
- `configs/raster/` — band configs governing which vegetation-structure bands are trained/evaluated in a given run (e.g., `veg_structure_4band.json`, `veg_structure_3band.json`, `veg_structure_baseline.json`, plus OOD variants).

**Forest plot evaluation** (4 OOD sites: BluffMesa, NorthBigBear, ReyesPeak, Laguna) is orchestrated by:

```bash
bash scripts/evaluate_forest_plots.sh \
    --model <checkpoint.pth> \
    --band-config <config.json> \
    --multi-gpu \
    --mc-samples <N> \
    --batch-size <B>
```

Laguna has no UAVSAR — the model handles this via graceful degradation at inference (skips the UAVSAR fusion branch when `uavsar=None`).

## Polygon → map → fuel-treatment priority

End-to-end map production for an arbitrary AOI, layered on the forest-plot inference path above.

- `predict_polygon.py` — orchestrates the full chain for one AOI polygon (reproject to EPSG:32611, tile grid, NAIP + 3DEP fetch, H5 build, MC-dropout inference, mosaic) into a vegetation-structure map. Writes a per-AOI `OUTPUT_README.md`.
- `predict_polygon_blocks.py` — block-tiled runner for large AOIs: splits the polygon into ~1 km blocks (edge-to-edge with overlap to avoid mosaic seams), runs `predict_polygon.py` per block across GPUs, and mosaics the result. Resumable. `run_laguna.sh` (repo root) is an idempotent, self-resuming launcher for the full Laguna run.
- `fuel_treatment_priority.py` — combines the three predicted structure bands (canopy cover, mid-story density, FHD) into a crown-fire fuel-treatment priority index: canopy crown-fuel × ladder boost × neighborhood fuel continuity, percentile-ranked within forest. Reads the mosaic, writes a multi-band `fuel_priority.tif`.
- `build_laguna_handoff.py` — assembles the shareable deliverable from `fuel_priority.tif` + the raw band mosaic: a continuous + 1–5 tiered priority raster, the raw structure COGs, an overview PNG, and a plain-language README.

```bash
# Full-AOI map (block-tiled, resumable)
bash run_laguna.sh        # wraps: python src/evaluation/predict_polygon_blocks.py --polygon <AOI> ...

# Fuel-treatment priority index from the mosaic
python src/evaluation/fuel_treatment_priority.py

# Package the deliverable
python src/evaluation/build_laguna_handoff.py
```

Outputs land under `data/output/polygon_predictions/<NAME>/` (gitignored — regenerate from these scripts).

## Historical: point cloud upsampling (published)

- `inference_eval.py` — model inference on the point-cloud test set; Chamfer distance metrics.
- `generate_eval_df.py` — aggregates inference results into evaluation dataframes.
- `RQ_test_v2.py` — statistical tests (Wilcoxon, effect sizes) for the published research questions.
- `manuscript_figures.py` — figures for the *Remote Sensing* (2025) paper.

## Development tools (not in any published workflow)

- `val_eval.py`, `model_val_report.py` — validation evaluation utilities and PDF reporting.
- `model_comparison_report.py`, `df_based_model_comparison_report.py` — multi-model comparison reports with 3D point-cloud visualizations.

---

See [../../README.md](../../README.md) and [../../CLAUDE.md](../../CLAUDE.md).
