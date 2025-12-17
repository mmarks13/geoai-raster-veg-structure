#!/bin/bash
# run_3dep_baseline.sh
#
# Complete 3DEP baseline generation pipeline for forest plot sites.
# Downloads 3DEP COPC data from Planetary Computer, processes through fuel metrics
# pipeline, extracts at plot locations, and generates comparison statistics.
#
# Usage:
#   bash scripts/run_3dep_baseline.sh [OPTIONS]
#
# Options:
#   --site SITE_NAME    Process only specified site (BluffMesa, Laguna, NorthBigBear, ReyesPeak)
#   --skip-download     Skip download step (use existing LAZ files)
#   --skip-fuel         Skip fuel metrics step (use existing rasters)
#   --skip-extract      Skip extraction step (use existing predictions)
#   --dry-run           Print commands without executing
#   --help              Show this help message

set -e  # Exit on error

# Default configuration
SITES="BluffMesa Laguna NorthBigBear ReyesPeak"
OUTPUT_BASE="data/processed/fuel_metrics/3dep_baseline"
RESOLUTION=5.0
TILE_SIZE=200
PARALLEL_JOBS=6
SPECIES="Mixed"

# Flags
SKIP_DOWNLOAD=false
SKIP_FUEL=false
SKIP_EXTRACT=false
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
        --skip-fuel)
            SKIP_FUEL=true
            shift
            ;;
        --skip-extract)
            SKIP_EXTRACT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n +2 | sed 's/^# //'
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
echo "============================================="
echo "3DEP Baseline Fuel Metrics Pipeline"
echo "============================================="
echo ""
echo "Sites: $SITES"
echo "Resolution: ${RESOLUTION}m"
echo "Output: $OUTPUT_BASE"
echo "Skip download: $SKIP_DOWNLOAD"
echo "Skip fuel metrics: $SKIP_FUEL"
echo "Skip extraction: $SKIP_EXTRACT"
echo ""

# Phase 1: Download 3DEP data
if [[ "$SKIP_DOWNLOAD" == "false" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 1: Downloading 3DEP COPC Data"
    echo "============================================="
    echo ""

    for site in $SITES; do
        echo ""
        echo "--- Processing $site ---"
        echo ""

        run_cmd conda run -p /home/jovyan/geoai_env python src/data_prep/download_3dep_for_sites.py \
            --site "$site" \
            --output-dir "$OUTPUT_BASE/$site"
    done

    echo ""
    echo "✓ Phase 1 complete: 3DEP data downloaded"
else
    echo ""
    echo "⊗ Phase 1 skipped: Download"
fi

# Phase 2: Run fuel metrics pipeline
if [[ "$SKIP_FUEL" == "false" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 2: Running Fuel Metrics Pipeline"
    echo "============================================="
    echo ""

    for site in $SITES; do
        echo ""
        echo "--- Processing $site ---"
        echo ""

        INPUT_LAZ="$OUTPUT_BASE/$site/3dep_merged.laz"

        if [[ ! -f "$INPUT_LAZ" ]]; then
            echo "Error: Input LAZ not found: $INPUT_LAZ"
            echo "Run without --skip-download first"
            exit 1
        fi

        run_cmd bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
            --input "$INPUT_LAZ" \
            --output-name "3dep_$site" \
            --output-base "$OUTPUT_BASE" \
            --resolution "$RESOLUTION" \
            --tile-size "$TILE_SIZE" \
            --parallel-jobs "$PARALLEL_JOBS" \
            --species "$SPECIES" \
            --skip-interactive-prompts
    done

    echo ""
    echo "✓ Phase 2 complete: Fuel metrics generated"
else
    echo ""
    echo "⊗ Phase 2 skipped: Fuel metrics"
fi

# Phase 3: Extract at plots and compare
if [[ "$SKIP_EXTRACT" == "false" ]]; then
    echo ""
    echo "============================================="
    echo "Phase 3: Extracting at Plot Locations"
    echo "============================================="
    echo ""

    FIELD_DATA="data/processed/forest_plot_data/forest_plots_processed.csv"
    COMPARISON_DIR="$OUTPUT_BASE/comparison"

    if [[ ! -f "$FIELD_DATA" ]]; then
        echo "Error: Field data not found: $FIELD_DATA"
        exit 1
    fi

    run_cmd conda run -p /home/jovyan/geoai_env python src/evaluation/extract_fuel_metrics_at_plots.py \
        --raster-dir "$OUTPUT_BASE" \
        --field-data "$FIELD_DATA" \
        --output-dir "$COMPARISON_DIR"

    echo ""
    echo "✓ Phase 3 complete: Predictions extracted and compared"
else
    echo ""
    echo "⊗ Phase 3 skipped: Extraction"
fi

# Phase 4: QA Report
echo ""
echo "============================================="
echo "Phase 4: Generating QA Report"
echo "============================================="
echo ""

QA_OUTPUT="$OUTPUT_BASE/qa_report.json"

run_cmd conda run -p /home/jovyan/geoai_env python src/evaluation/qa_3dep_baseline.py \
    --data-dir "$OUTPUT_BASE" \
    --output "$QA_OUTPUT"

echo ""
echo "✓ Phase 4 complete: QA report generated"

# Final summary
echo ""
echo "============================================="
echo "Pipeline Complete!"
echo "============================================="
echo ""
echo "Results:"
echo "  Comparison statistics: $OUTPUT_BASE/comparison/baseline_comparison_stats.json"
echo "  Comparison figures: $OUTPUT_BASE/comparison/baseline_comparison_scatter.png"
echo "  Predictions CSV: $OUTPUT_BASE/comparison/baseline_predictions.csv"
echo "  QA Report: $QA_OUTPUT"
echo ""
echo "Per-site rasters:"
for site in $SITES; do
    RASTER_PATH="$OUTPUT_BASE/3dep_$site/merged/3dep_${site}_fuel_metrics_${RESOLUTION}m.tif"
    if [[ -f "$RASTER_PATH" || "$DRY_RUN" == "true" ]]; then
        echo "  $site: $RASTER_PATH"
    fi
done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY-RUN] No actions were performed. Remove --dry-run to execute."
fi
