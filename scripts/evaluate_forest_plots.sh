#!/bin/bash
#
# Forest Plot Evaluation Pipeline
#
# This script runs the complete evaluation pipeline for trained raster models:
#   1. Run inference on forest plot tiles
#   2. Build per-site rasters
#   3. Compare predictions to field measurements
#   4. Generate QGIS export (footprints + VRT)
#
# Usage:
#   # Single model
#   bash scripts/evaluate_forest_plots.sh \
#       --model data/output/raster_model_naip_20251213_173144/checkpoints/best_model.pth \
#       --band-config src/evaluation/configs/raster/cover_only.json
#
#   # Multiple models
#   bash scripts/evaluate_forest_plots.sh \
#       --model path1/best_model.pth \
#       --model path2/best_model.pth \
#       --band-config src/evaluation/configs/raster/cover_only.json
#
# Options:
#   --model PATH           Model checkpoint path (required, can be specified multiple times)
#   --band-config PATH     Band configuration JSON (required)
#   --batch-size N         Batch size for inference (default: 64)
#   --device DEVICE        Device to use (default: cuda)
#   --skip-inference       Skip inference step (reuse existing predictions)
#   --help                 Show this help message

set -e  # Exit on error

# Default values
BATCH_SIZE=64
DEVICE="cuda"
SKIP_INFERENCE=false
BAND_CONFIG=""
MODELS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODELS+=("$2")
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --skip-inference)
            SKIP_INFERENCE=true
            shift
            ;;
        --band-config)
            BAND_CONFIG="$2"
            shift 2
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate inputs
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "Error: At least one --model must be specified"
    echo "Use --help for usage information"
    exit 1
fi

if [ -z "$BAND_CONFIG" ]; then
    echo "Error: --band-config must be specified"
    echo "Use --help for usage information"
    exit 1
fi

if [ ! -f "$BAND_CONFIG" ]; then
    echo "Error: Band config not found: $BAND_CONFIG"
    exit 1
fi

# Check required input files
TILES_FILE="data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt"
FIELD_DATA="data/processed/forest_plot_data/forest_plots_processed.gpkg"

# Read stats file path from band config
FUEL_STATS=$(python -c "import json; print(json.load(open('$BAND_CONFIG'))['stats_file'])")
if [ -z "$FUEL_STATS" ]; then
    echo "Error: Could not read 'stats_file' from band config: $BAND_CONFIG"
    exit 1
fi
echo "Using stats file from band config: $FUEL_STATS"

# Note: Using ls for file existence check due to -f test issues with large files
if ! ls "$TILES_FILE" &>/dev/null; then
    echo "Error: Forest plot tiles not found: $TILES_FILE"
    exit 1
fi

if ! ls "$FIELD_DATA" &>/dev/null; then
    echo "Error: Field data not found: $FIELD_DATA"
    exit 1
fi

if ! ls "$FUEL_STATS" &>/dev/null; then
    echo "Error: Fuel stats not found: $FUEL_STATS"
    exit 1
fi

# Main pipeline
echo "========================================"
echo "FOREST PLOT EVALUATION PIPELINE"
echo "========================================"
echo "Models to evaluate: ${#MODELS[@]}"
echo "Batch size: $BATCH_SIZE"
echo "Device: $DEVICE"
echo ""

for MODEL_PATH in "${MODELS[@]}"; do
    echo ""
    echo "========================================"
    echo "Processing model: $MODEL_PATH"
    echo "========================================"

    # Validate model checkpoint exists
    if [ ! -f "$MODEL_PATH" ]; then
        echo "Error: Model checkpoint not found: $MODEL_PATH"
        continue
    fi

    # Extract model name from path
    # Example: data/output/raster_model_fused_20251203_083411/checkpoints/best_model.pth
    # -> raster_model_fused_20251203_083411
    MODEL_DIR=$(dirname "$MODEL_PATH")
    MODEL_DIR=$(dirname "$MODEL_DIR")  # Go up two levels
    MODEL_NAME=$(basename "$MODEL_DIR")

    echo "Model name: $MODEL_NAME"

    # Create output directory
    OUTPUT_BASE="data/output/forest_plot_evaluations/$MODEL_NAME"
    mkdir -p "$OUTPUT_BASE"

    PREDICTIONS_DIR="$OUTPUT_BASE/predictions"
    RASTERS_DIR="$OUTPUT_BASE/site_rasters"
    COMPARISON_DIR="$OUTPUT_BASE/comparison"
    QGIS_DIR="$OUTPUT_BASE/qgis_export"

    mkdir -p "$PREDICTIONS_DIR"
    mkdir -p "$RASTERS_DIR"
    mkdir -p "$COMPARISON_DIR"
    mkdir -p "$QGIS_DIR"

    # Find config.json in same directory as checkpoint
    CONFIG_PATH="$(dirname "$MODEL_PATH")/config.json"
    if [ ! -f "$CONFIG_PATH" ]; then
        # Try parent directory (for legacy structure)
        CONFIG_PATH="$(dirname "$(dirname "$MODEL_PATH")")/config.json"
    fi

    # Step 1: Run inference
    if [ "$SKIP_INFERENCE" = false ]; then
        echo ""
        echo "Step 1: Running inference..."

        INFERENCE_CMD="python src/evaluation/raster_inference.py \
            --checkpoint $MODEL_PATH \
            --input $TILES_FILE \
            --output $PREDICTIONS_DIR \
            --fuel-stats $FUEL_STATS \
            --batch-size $BATCH_SIZE \
            --device $DEVICE"

        # Add config if available
        if [ -f "$CONFIG_PATH" ]; then
            INFERENCE_CMD="$INFERENCE_CMD --config $CONFIG_PATH"
        fi

        eval $INFERENCE_CMD

        if [ $? -ne 0 ]; then
            echo "Error: Inference failed for $MODEL_NAME"
            continue
        fi
    else
        echo ""
        echo "Step 1: Skipping inference (--skip-inference flag)"
    fi

    # Find most recent prediction files
    LATEST_PT=$(ls -t "$PREDICTIONS_DIR"/forest_plot_predictions_*.pt 2>/dev/null | head -1)
    LATEST_CSV=$(ls -t "$PREDICTIONS_DIR"/forest_plot_predictions_*.csv 2>/dev/null | head -1)

    if [ -z "$LATEST_PT" ] || [ -z "$LATEST_CSV" ]; then
        echo "Error: Prediction files not found in $PREDICTIONS_DIR"
        continue
    fi

    echo "Using predictions:"
    echo "  PT:  $LATEST_PT"
    echo "  CSV: $LATEST_CSV"

    # Step 2: Build site rasters
    echo ""
    echo "Step 2: Building site rasters from predictions..."

    python src/evaluation/build_prediction_rasters.py \
        --predictions-pt "$LATEST_PT" \
        --predictions-csv "$LATEST_CSV" \
        --output-dir "$RASTERS_DIR" \
        --field-plots "$FIELD_DATA"

    if [ $? -ne 0 ]; then
        echo "Error: Raster building failed for $MODEL_NAME"
        continue
    fi

    # Step 3: Compare to field data
    echo ""
    echo "Step 3: Comparing predictions to field measurements..."

    python src/evaluation/compare_predictions_to_plots.py \
        --site-rasters-dir "$RASTERS_DIR" \
        --field-data "$FIELD_DATA" \
        --band-config "$BAND_CONFIG" \
        --output "$COMPARISON_DIR"

    if [ $? -ne 0 ]; then
        echo "Error: Comparison failed for $MODEL_NAME"
        continue
    fi

    # Step 4: Create QGIS export
    echo ""
    echo "Step 4: Creating QGIS export..."

    python src/evaluation/create_qgis_export.py \
        --comparison-dir "$COMPARISON_DIR" \
        --output-dir "$QGIS_DIR" \
        --band-config "$BAND_CONFIG"

    if [ $? -ne 0 ]; then
        echo "Error: QGIS export failed for $MODEL_NAME"
        continue
    fi

    echo ""
    echo "✓ Evaluation complete for $MODEL_NAME"
    echo "  Output: $OUTPUT_BASE"
done

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo ""
echo "Results saved to: data/output/forest_plot_evaluations/"
echo ""
echo "Next steps:"
echo "  - Review comparison statistics: */comparison/comparison_stats.json"
echo "  - View figures: */comparison/comparison_scatter.png"
echo "  - Compare models: bash scripts/compare_forest_plot_models.sh"
echo "  - Load in QGIS: */qgis_export/"
