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
