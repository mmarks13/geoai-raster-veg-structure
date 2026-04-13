#!/bin/bash
# Forest Plot Data Processing Pipeline
# Prepares forest plot tiles for model inference (no ground truth fuel metrics)
#
# Prerequisites:
#   - NAIP/UAVSAR/3DEP data downloaded (scripts/get_forest_plot_data.sh)
#   - Forest plots processed (src/raster_mapping/process_forest_plots.py)
#   - Training normalization stats available (from raster training pipeline)
#
# Output:
#   - data/processed/forest_plot_tiles.geojson (tile grid)
#   - data/processed/forest_plot_data/combined_forest_plots.pt (raw combined data)
#   - data/processed/forest_plot_data/precomputed_forest_plot_tiles_32bit.pt (inference-ready)
#
# See forest_plot_vs_prediction_comparison.md for full documentation

set -e  # Exit on error

cd /home/jovyan/geoai_veg_map

echo "========================================"
echo "FOREST PLOT DATA PROCESSING PIPELINE"
echo "========================================"
echo "Start time: $(date)"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Generate tile grid from site polygons
# -----------------------------------------------------------------------------
echo "Step 3: Generate tile grid from site polygons..."
# Use 20% overlap (stride=8m for 10m tiles) to reduce tile edge discontinuities
# Overlapping predictions will be averaged in compare_predictions_to_plots.py
python src/data_prep/create_forest_plot_tile_grid.py \
  --input data/processed/forest_plot_data/site_polygons.gpkg \
  --output data/processed/forest_plot_tiles.geojson \
  --tile-size 10.0 \
  --stride 8.0

echo "✓ Tile grid created (20% overlap)"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Extract 3DEP + NAIP + UAVSAR for each tile
# -----------------------------------------------------------------------------
echo "Step 4: Extract tile data (3DEP, NAIP, UAVSAR)..."

# Create output directory if it doesn't exist
mkdir -p data/processed/forest_plot_data_chunks

# Check if --resume flag should be used (if chunks already exist)
RESUME_FLAG=""
if [ -n "$(ls -A data/processed/forest_plot_data_chunks/*.h5 2>/dev/null)" ]; then
  echo "  Found existing chunk files - using --resume to skip already-processed tiles"
  RESUME_FLAG="--resume"
fi

# Use generate_training_data_raster.py (same as training pipeline) but without
# --fuel-metrics-raster since forest plots don't have ground truth fuel metrics
python src/data_prep/generate_training_data_raster.py \
  --tiles_geojson data/processed/forest_plot_tiles.geojson \
  --outdir data/processed/forest_plot_data_chunks \
  --skip-uav-lidar \
  --uavsar_stac_source data/stac/uavsar/catalog.json \
  --naip_stac_source data/stac/naip/catalog.json \
  --dep_stac_source data/stac/3dep_hag/catalog.json \
  --start_date 2016-01-01 \
  --end_date 2030-12-31 \
  --chunk_size 100 \
  --threads 12 \
  --max-api-retries 20 \
  $RESUME_FLAG

echo "✓ Tile data extracted"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Combine H5 chunks into single .pt file
# -----------------------------------------------------------------------------
echo "Step 5: Combine H5 chunks into .pt file..."
python src/data_prep/h5_chunk_loader.py \
  --input_dir data/processed/forest_plot_data_chunks \
  --output_path data/processed/forest_plot_data/combined_forest_plots.pt

echo "✓ H5 chunks combined"
echo ""

# -----------------------------------------------------------------------------
# Step 6: Preprocess for inference (normalize, precompute KNN)
# -----------------------------------------------------------------------------
echo "Step 6: Preprocess for inference..."

# Create inference output directory
mkdir -p data/processed/forest_plot_data/inference_ready

python src/data_prep/preprocess_forest_plots_for_inference.py \
  --pt-file data/processed/forest_plot_data/combined_forest_plots.pt \
  --training-stats-dir data/processed/model_data_veg_structure \
  --output-dir data/processed/forest_plot_data/inference_ready \
  --min-dep-points 50 \
  --precision 32

echo "✓ Preprocessing complete"
echo ""
Okay. phase 1 is complete 

Output: data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt
2025-12-05 22:27:44,956 - INFO - Distribution shift report: data/processed/forest_plot_data/inference_ready/distribution_shift_report.json
2025-12-05 22:27:44,956 - INFO - Forest plot stats: data/processed/forest_plot_data/inference_ready/forest_plot_coordinate_stats.json
✓ Preprocessing complete 

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "========================================"
echo "✓ PIPELINE COMPLETE"
echo "========================================"
echo "End time: $(date)"
echo ""
echo "Output files:"
echo "  - data/processed/forest_plot_tiles.geojson"
echo "  - data/processed/forest_plot_data/combined_forest_plots.pt"
echo "  - data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt"
echo "  - data/processed/forest_plot_data/inference_ready/distribution_shift_report.json"
echo ""
echo "Next steps:"
echo "  1. Run raster model inference:"
echo "     python src/evaluation/raster_inference.py \\"
echo "       --tiles data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt \\"
echo "       --checkpoint data/output/checkpoints/best_raster_model.pth \\"
echo "       --output-dir data/processed/forest_plot_data/predictions"
echo ""
echo "  2. Create plot footprints:"
echo "     python src/raster_mapping/create_plot_footprints.py \\"
echo "       --input data/processed/forest_plot_data/forest_plots_processed.gpkg \\"
echo "       --output data/processed/forest_plot_data/plot_footprints.gpkg"
echo ""
echo "  3. Compare predictions to ground truth:"
echo "     python src/evaluation/compare_predictions_to_plots.py \\"
echo "       --predictions data/processed/forest_plot_data/predictions \\"
echo "       --footprints data/processed/forest_plot_data/plot_footprints.gpkg \\"
echo "       --ground-truth data/processed/forest_plot_data/forest_plots_processed.gpkg"
echo "========================================"
