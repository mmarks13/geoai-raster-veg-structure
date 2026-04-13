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
#   # Single model (single GPU)
#   bash scripts/evaluate_forest_plots.sh \
#       --model data/output/raster_model_naip_20251213_173144/checkpoints/best_model.pth \
#       --band-config src/evaluation/configs/raster/cover_only.json
#
#   # Single model (multi-GPU)
#   bash scripts/evaluate_forest_plots.sh \
#       --model data/output/raster_model_naip_20251213_173144/checkpoints/best_model.pth \
#       --band-config src/evaluation/configs/raster/cover_only.json \
#       --multi-gpu
#
#   # Multiple models with specific GPU count
#   bash scripts/evaluate_forest_plots.sh \
#       --model path1/best_model.pth \
#       --model path2/best_model.pth \
#       --band-config src/evaluation/configs/raster/cover_only.json \
#       --multi-gpu --num-gpus 2
#
# Options:
#   --model PATH           Model checkpoint path (required, can be specified multiple times)
#   --band-config PATH     Band configuration JSON (required)
#   --batch-size N         Batch size for inference (default: 64)
#   --device DEVICE        Device to use for single-GPU mode (default: cuda)
#   --multi-gpu            Enable multi-GPU inference using DDP
#   --num-gpus N           Number of GPUs to use (default: all available)
#   --mc-samples N         Number of MC dropout samples for uncertainty (default: 1 = deterministic)
#   --skip-inference       Skip inference step (reuse existing predictions)
#   --help                 Show this help message

set -e  # Exit on error

# Default values
BATCH_SIZE=64
DEVICE="cuda"
MULTI_GPU=false
NUM_GPUS=""
SKIP_INFERENCE=false
BAND_CONFIG=""
MC_SAMPLES=1
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
        --multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --skip-inference)
            SKIP_INFERENCE=true
            shift
            ;;
        --mc-samples)
            MC_SAMPLES="$2"
            shift 2
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

# Optional: plots held out for in-training OOD validation. When present, these
# plots are excluded from §7 final evaluation to prevent double-counting.
OOD_EXCLUDE_FILE="data/processed/forest_plot_data/ood_validation/ood_validation_plot_ids.txt"

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
if [ "$MULTI_GPU" = true ]; then
    if [ -n "$NUM_GPUS" ]; then
        echo "Mode: Multi-GPU ($NUM_GPUS GPUs)"
    else
        echo "Mode: Multi-GPU (all available)"
    fi
else
    echo "Device: $DEVICE (single-GPU)"
fi
if [ "$MC_SAMPLES" -gt 1 ]; then
    echo "MC Dropout: $MC_SAMPLES samples (uncertainty enabled)"
else
    echo "MC Dropout: disabled (deterministic)"
fi
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

    # Add MC suffix if MC dropout is enabled
    if [ "$MC_SAMPLES" -gt 1 ]; then
        MODEL_NAME="${MODEL_NAME}_mc${MC_SAMPLES}"
    fi

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
            --mc-samples $MC_SAMPLES"

        # Add multi-GPU or single-GPU device args
        if [ "$MULTI_GPU" = true ]; then
            INFERENCE_CMD="$INFERENCE_CMD --multi-gpu"
            if [ -n "$NUM_GPUS" ]; then
                INFERENCE_CMD="$INFERENCE_CMD --num-gpus $NUM_GPUS"
            fi
        else
            INFERENCE_CMD="$INFERENCE_CMD --device $DEVICE"
        fi

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
    LATEST_PT=$(ls -t "$PREDICTIONS_DIR"/forest_plot_predictions_*.pt 2>/dev/null | grep -v '_std_' | head -1)
    LATEST_CSV=$(ls -t "$PREDICTIONS_DIR"/forest_plot_predictions_*.csv 2>/dev/null | head -1)

    if [ -z "$LATEST_PT" ] || [ -z "$LATEST_CSV" ]; then
        echo "Error: Prediction files not found in $PREDICTIONS_DIR"
        continue
    fi

    # Find std predictions file if it exists (from MC dropout)
    LATEST_STD_PT=$(ls -t "$PREDICTIONS_DIR"/forest_plot_predictions_std_*.pt 2>/dev/null | head -1)

    echo "Using predictions:"
    echo "  PT:  $LATEST_PT"
    echo "  CSV: $LATEST_CSV"
    if [ -n "$LATEST_STD_PT" ]; then
        echo "  STD: $LATEST_STD_PT (uncertainty)"
    fi

    # Step 2: Build site rasters
    echo ""
    echo "Step 2: Building site rasters from predictions..."

    RASTER_CMD="python src/evaluation/build_prediction_rasters.py \
        --predictions-pt \"$LATEST_PT\" \
        --predictions-csv \"$LATEST_CSV\" \
        --output-dir \"$RASTERS_DIR\" \
        --field-plots \"$FIELD_DATA\""

    # Add std predictions if available
    if [ -n "$LATEST_STD_PT" ]; then
        RASTER_CMD="$RASTER_CMD --predictions-std-pt \"$LATEST_STD_PT\""
    fi

    eval $RASTER_CMD

    if [ $? -ne 0 ]; then
        echo "Error: Raster building failed for $MODEL_NAME"
        continue
    fi

    # Step 3: Compare to field data
    echo ""
    echo "Step 3: Comparing predictions to field measurements..."

    EXCLUDE_ARG=""
    if [ -f "$OOD_EXCLUDE_FILE" ]; then
        EXCLUDE_ARG="--exclude-plots-file $OOD_EXCLUDE_FILE"
        echo "  → Excluding OOD-training plots listed in $OOD_EXCLUDE_FILE"
    fi

    python src/evaluation/compare_predictions_to_plots.py \
        --site-rasters-dir "$RASTERS_DIR" \
        --field-data "$FIELD_DATA" \
        --band-config "$BAND_CONFIG" \
        --output "$COMPARISON_DIR" \
        $EXCLUDE_ARG

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
