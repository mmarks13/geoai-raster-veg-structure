# Running on this VM and migrating off

A working note (not part of the public README) describing the data layout
introduced by the `add-runnable-deps` branch — what's hard-copied, what's
symlinked, how to make the repo portable when leaving this VM.

## Entry points

- Training: `run_raster_model.py`
- Image-encoder pre-training: `run_pretrain_image_encoders.py --encoder {naip,uavsar}`
- Forest-plot evaluation: `scripts/evaluate_forest_plots.sh`

The training scripts load from `data/processed/model_data_veg_structure/` and
`data/processed/forest_plot_data/ood_validation/`. They load a band config
from `src/evaluation/configs/raster/`.

The forest-plot evaluation pipeline additionally needs:

- The trained model checkpoint at
  `data/output/pretrained_3band_naip/checkpoints/epoch_100.pth`
  (the final 3-band NAIP-fused run — confirmed by being the largest output
  under `data/output/forest_plot_evaluations/`).
- Inference-ready forest-plot tiles
  (`data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt`).
- The structural baseline comparison data under
  `data/processed/veg_structure_baseline/comparison/` (used by Step 5 of
  `evaluate_forest_plots.sh` to compare the model against the 3DEP-only baseline).

Reference invocation:

```bash
bash scripts/evaluate_forest_plots.sh \
    --model data/output/pretrained_3band_naip/checkpoints/epoch_100.pth \
    --band-config src/evaluation/configs/raster/veg_structure_3band_v2.json \
    --multi-gpu --mc-samples 30 --batch-size 800
```

## Pretrained models in this repo

Two pretrained model directories are intended to live here long-term, both
for internal reuse and for public release alongside the corresponding band
configs:

| Symlink path | Output bands | Band config |
|---|---|---|
| `data/output/pretrained_3band_naip/` | canopy_cover, midstory_density, FHD | `src/evaluation/configs/raster/veg_structure_3band_v2.json` |
| `data/output/pretrained_13band_naip/` | heights + all densities + FHD + height percentiles (13 bands) | `src/evaluation/configs/raster/veg_structure_13band.json` |

The 3-band model is trained (currently a symlink into the source repo's
April 22 run); the 13-band model will be trained from scratch in this
repo by running `python run_raster_model.py --bands 13`.

## Data layout

Bulk training and validation data lives outside git. Under `data/processed/`:

### Hard-copied (small, travels with the repo)

These files are physically present and gitignored at the data-extension level:

| Path | Used by |
|---|---|
| `model_data_veg_structure/coordinate_normalization_stats_train.json` | training |
| `model_data_veg_structure/target_raster_normalization_stats_train.json` | band configs (via `stats_file`) |
| `forest_plot_data/ood_validation/ood_validation_metadata.json` | OOD validation, forest-plot eval (OOD exclude list) |
| `forest_plot_data/ood_validation/ood_validation_config.json` | informational |
| `forest_plot_data/ood_validation/ood_validation_plot_ids.txt` | informational |
| `forest_plot_data/forest_plots_processed.csv` | forest-plot evaluation |
| `forest_plot_data/forest_plots_processed.gpkg` | forest-plot evaluation |
| `forest_plot_data/plot_footprints.gpkg` | mapping/visualization |
| `forest_plot_data/site_polygons.gpkg` | mapping/visualization |
| `forest_plot_data/site_bboxes.txt` | data prep / inference |
| `forest_plot_tiles.geojson` (23 MB) | tile geometry |

### Symlinked (large, point into `/home/jovyan/geoai_veg_map/data/`)

`torch.load` follows symlinks transparently, so runtime is identical to
having real files in place.

| Symlink path | Size | Used by |
|---|---|---|
| `model_data_veg_structure/precomputed_training_tiles_raster_32bit.pt` | 22.5 GB | training |
| `model_data_veg_structure/precomputed_validation_tiles_raster_32bit.pt` | 3.9 GB | training |
| `model_data_veg_structure/combined_training_data_veg_structure.pt` | 5.7 GB | precompute |
| `forest_plot_data/ood_validation/ood_validation_tiles.pt` | 95 MB | OOD validation |
| `3dep_hag_features/` (whole dir) | 8.4 GB | data prep (COPC tiles) |
| `veg_structure_metrics/` (per-site subdirs) | 48 MB | ground-truth metric rasters |
| `forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt` | 21 GB | forest-plot eval (Step 1: inference) |
| `veg_structure_baseline/` (whole dir) | 201 MB | forest-plot eval (Step 5: vs 3DEP baseline) |
| `data/output/pretrained_3band_naip/` (whole run dir) | ~430 MB | trained 3-band model checkpoints + logs |
| `data/output/pretrained_naip_encoder_d128/` (whole run dir) | ~100 MB | NAIP image-encoder pretrain (transfer source for `run_raster_model.py`) |

## Migrating off this VM

Symlinks store paths, not contents. `git add` of a symlink would push a
broken pointer; `tar` / `rsync` without dereferencing flags would copy
dangling links. Before moving the repo to a new host:

```bash
bash scripts/materialize_data.sh
```

The script walks `data/processed/`, replaces every symlink with a real
copy of its target, and is idempotent (regular files and broken symlinks
are skipped).

After materialization, the repo is independent of this VM and can be
`tar`/`rsync`'d normally. Beware: the directory will then weigh roughly
40+ GB on disk.

## Public vs. private inputs

- **Volcan Mountain UAV LiDAR** is now publicly hosted on
  [OpenTopography](https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042026.32611.1).
  See `scripts/fetch_public_data.sh` for download notes (currently a
  documentation placeholder; wire in the OT API call when ready).
- **NAIP** is fetched via earth-search STAC by `scripts/get_data.sh`.
- **USGS 3DEP** is fetched and processed by `scripts/process_3dep_hag_features.sh`.
- **UAVSAR** and other UAV LiDAR collections may remain private; source
  them separately at a new host.

## Band configurations

Trainable raster runs read a band-config JSON from
`src/evaluation/configs/raster/`. Three are checked in:

- `veg_structure_3band_v2.json` — canopy_cover, midstory_density, FHD.
  Default OOD-validation target for `run_raster_model.py`.
- `veg_structure_ood_2band_density.json` — canopy_density, midstory_density.
  OOD-validation target for `run_pretrain_image_encoders.py`.
- `veg_structure_13band.json` — heights (max/mean/std), all four density
  bands, FHD, and height percentiles p10/p25/p50/p75/p90. Same field-column
  linkage and `model_units: "normalized"` convention as the 3-band v2 file,
  so it plugs into the same training/eval path. (Mirrors the first 13 bands
  of `veg_structure_baseline.json` but parameterized for trainable use.)

## Verifying

After symlinks and copies are in place, the entry scripts should resolve
all paths cleanly. Quick check:

```bash
cd /home/jovyan/geoai-raster-veg-structure
for p in \
  data/processed/model_data_veg_structure/precomputed_training_tiles_raster_32bit.pt \
  data/processed/model_data_veg_structure/precomputed_validation_tiles_raster_32bit.pt \
  data/processed/model_data_veg_structure/coordinate_normalization_stats_train.json \
  data/processed/model_data_veg_structure/target_raster_normalization_stats_train.json \
  data/processed/forest_plot_data/ood_validation/ood_validation_tiles.pt \
  data/processed/forest_plot_data/ood_validation/ood_validation_metadata.json \
  src/evaluation/configs/raster/veg_structure_3band_v2.json \
  src/evaluation/configs/raster/veg_structure_ood_2band_density.json \
  src/evaluation/configs/raster/veg_structure_13band.json ; do
  [[ -e "$p" ]] && echo "OK   $p" || echo "MISS $p"
done
```
