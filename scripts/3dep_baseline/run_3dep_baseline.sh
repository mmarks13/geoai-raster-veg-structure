#!/bin/bash
# run_3dep_baseline.sh
#
# 3DEP Baseline Vegetation Structure Comparison Pipeline
#
# This script computes vegetation structure metrics from 3DEP LiDAR and
# compares them to field measurements, establishing a baseline for
# multimodal model comparison.
#
# Usage:
#   bash scripts/3dep_baseline/run_3dep_baseline.sh [OPTIONS]
#
# Options:
#   --site SITE_NAME    Process only specified site (BluffMesa, Laguna, NorthBigBear, ReyesPeak)
#   --skip-download     Skip 3DEP download step (use existing LAZ files)
#   --skip-metrics      Skip metrics computation (use existing rasters)
#   --skip-compare      Skip comparison step (use existing results)
#   --model-rasters DIR Include model predictions for 3-way comparison
#   --resolution FLOAT  Raster resolution in meters (default: 2.0)
#   --max-hag FLOAT     Maximum height above ground filter (default: 60.0)
#   --dry-run           Print commands without executing
#   --help              Show this help message

set -e  # Exit on error

# Default configuration
SITES="BluffMesa Laguna NorthBigBear ReyesPeak"
DEP_DATA_DIR="data/processed/fuel_metrics/3dep_baseline"
OUTPUT_BASE="data/processed/veg_structure_baseline"
FIELD_DATA="data/processed/forest_plot_data/forest_plots_processed.gpkg"
BAND_CONFIG="src/evaluation/configs/raster/veg_structure_baseline.json"
RESOLUTION=2.0
MAX_HAG=100.0

# Flags
SKIP_DOWNLOAD=false
SKIP_METRICS=false
SKIP_COMPARE=false
MODEL_RASTERS=""
DRY_RUN=false
SINGLE_SITE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --site)
            SINGLE_SITE="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-metrics)
            SKIP_METRICS=true
            shift
            ;;
        --skip-compare)
            SKIP_COMPARE=true
            shift
            ;;
        --model-rasters)
            MODEL_RASTERS="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --max-hag)
            MAX_HAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            head -n 25 "$0" | tail -n +2 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Override sites if single site specified
if [[ -n "$SINGLE_SITE" ]]; then
    SITES="$SINGLE_SITE"
fi

# Validate sites
VALID_SITES="BluffMesa Laguna NorthBigBear ReyesPeak"
for site in $SITES; do
    if ! echo "$VALID_SITES" | grep -qw "$site"; then
        echo "Error: Invalid site '$site'. Valid sites: $VALID_SITES"
        exit 1
    fi
done

# Helper function to run commands
run_cmd() {
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "[RUNNING] $*"
        "$@"
    fi
}

# Print banner
echo ""
echo "============================================="
echo "3DEP BASELINE VEGETATION STRUCTURE PIPELINE"
echo "============================================="
echo ""
echo "Sites: $SITES"
echo "Resolution: ${RESOLUTION}m"
echo "Max HAG: ${MAX_HAG}m"
echo "Output: $OUTPUT_BASE"
echo "Skip download: $SKIP_DOWNLOAD"
echo "Skip metrics: $SKIP_METRICS"
echo "Skip compare: $SKIP_COMPARE"
if [[ -n "$MODEL_RASTERS" ]]; then
    echo "Model rasters: $MODEL_RASTERS (3-way comparison enabled)"
fi
echo ""

# ============================================
# Phase 1: Download 3DEP Data (if needed)
# ============================================
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 1: Checking 3DEP Downloads"
    echo "============================================="
    echo ""

    for site in $SITES; do
        LAZ_FILE="$DEP_DATA_DIR/$site/3dep_merged.laz"

        if [[ -f "$LAZ_FILE" ]]; then
            echo "✓ $site: 3DEP data exists at $LAZ_FILE"
        else
            echo "⚠ $site: 3DEP data not found, downloading..."
            run_cmd conda run -p /home/jovyan/geoai_env python src/data_prep/download_3dep_for_sites.py \
                --site "$site" \
                --output-dir "$DEP_DATA_DIR/$site"
        fi
    done

    echo ""
    echo "✓ Phase 1 complete: 3DEP data ready"
else
    echo ""
    echo "⊗ Phase 1 skipped: Download"
fi

# ============================================
# Phase 2: Compute Vegetation Structure Metrics
# ============================================
if [[ "$SKIP_METRICS" == "false" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 2: Computing Vegetation Structure Metrics"
    echo "============================================="
    echo ""

    for site in $SITES; do
        LAZ_FILE="$DEP_DATA_DIR/$site/3dep_merged.laz"
        OUTPUT_TIF="$OUTPUT_BASE/$site/veg_structure_2m.tif"

        if [[ ! -f "$LAZ_FILE" ]]; then
            echo "Error: Input LAZ not found: $LAZ_FILE"
            echo "Run without --skip-download first"
            exit 1
        fi

        echo ""
        echo "--- Processing $site ---"
        echo ""

        run_cmd conda run -p /home/jovyan/geoai_env python src/evaluation/compute_3dep_baseline_metrics.py \
            --input "$LAZ_FILE" \
            --output "$OUTPUT_BASE/$site" \
            --resolution "$RESOLUTION" \
            --max-hag "$MAX_HAG"
    done

    echo ""
    echo "✓ Phase 2 complete: Vegetation structure metrics generated"
else
    echo ""
    echo "⊗ Phase 2 skipped: Metrics computation"
fi

# ============================================
# Phase 3: Compare Baseline to Field Measurements
# ============================================
if [[ "$SKIP_COMPARE" == "false" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 3: Comparing Baseline to Field Measurements"
    echo "============================================="
    echo ""

    if [[ ! -f "$FIELD_DATA" ]]; then
        echo "Error: Field data not found: $FIELD_DATA"
        exit 1
    fi

    COMPARISON_DIR="$OUTPUT_BASE/comparison"

    run_cmd conda run -p /home/jovyan/geoai_env python src/evaluation/compare_baseline_to_plots.py \
        --baseline-rasters-dir "$OUTPUT_BASE" \
        --field-data "$FIELD_DATA" \
        --band-config "$BAND_CONFIG" \
        --output "$COMPARISON_DIR"

    echo ""
    echo "✓ Phase 3 complete: Baseline comparison generated"
else
    echo ""
    echo "⊗ Phase 3 skipped: Comparison"
fi

# ============================================
# Phase 4: 3-Way Comparison (Optional)
# ============================================
if [[ -n "$MODEL_RASTERS" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 4: 3-Way Comparison (Baseline vs Model vs Field)"
    echo "============================================="
    echo ""

    if [[ ! -d "$MODEL_RASTERS" ]]; then
        echo "Error: Model rasters directory not found: $MODEL_RASTERS"
        exit 1
    fi

    MODEL_COMPARISON_DIR="$OUTPUT_BASE/model_comparison"

    run_cmd conda run -p /home/jovyan/geoai_env python src/evaluation/compare_predictions_to_plots.py \
        --site-rasters-dir "$MODEL_RASTERS" \
        --baseline-rasters-dir "$OUTPUT_BASE" \
        --field-data "$FIELD_DATA" \
        --band-config "src/evaluation/configs/raster/veg_structure_8band.json" \
        --output "$MODEL_COMPARISON_DIR"

    echo ""
    echo "✓ Phase 4 complete: 3-way comparison generated"
else
    echo ""
    echo "⊗ Phase 4 skipped: 3-way comparison (no --model-rasters provided)"
fi

# ============================================
# Summary
# ============================================
echo ""
echo "============================================="
echo "PIPELINE COMPLETE"
echo "============================================="
echo ""
echo "Results:"

if [[ "$SKIP_COMPARE" == "false" ]]; then
    echo "  Comparison statistics: $OUTPUT_BASE/comparison/baseline_comparison_stats.json"
    echo "  Comparison figures: $OUTPUT_BASE/comparison/baseline_comparison_scatter.png"
    echo "  Comparison results: $OUTPUT_BASE/comparison/baseline_comparison_results.csv"
fi

echo ""
echo "Per-site rasters:"
for site in $SITES; do
    RASTER_PATH="$OUTPUT_BASE/$site/veg_structure_2m.tif"
    if [[ -f "$RASTER_PATH" || "$DRY_RUN" == "true" ]]; then
        echo "  $site: $RASTER_PATH"
    fi
done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] No actions were performed. Remove --dry-run to execute."
fi
