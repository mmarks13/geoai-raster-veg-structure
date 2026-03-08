#!/bin/bash
# Download and process 3DEP for all sites with HAG and enhanced features
#
# This script processes 3DEP point clouds from Planetary Computer for both
# training sites (with vegetation structure ground truth) and validation sites
# (forest plots with field measurements).
#
# Processing includes:
# - SMRF ground classification
# - Height Above Ground (HAG) computation
# - Eigenvalue features (Planarity, Sphericity, Verticality)
# - PointsAbove (canopy density metric)
# - ReturnRatio (normalized return position)
#
# Output: LAZ files with all extra dimensions in data/processed/3dep_hag_features/
#
# Usage:
#     bash scripts/process_3dep_hag_features.sh          # Process all sites
#     bash scripts/process_3dep_hag_features.sh --validation-only  # Validation sites only
#     bash scripts/process_3dep_hag_features.sh --training-only    # Training sites only

set -e

# Configuration
OUTPUT_BASE="data/processed/3dep_hag_features"
STAC_OUTPUT="data/stac/3dep_hag"
LOG_DIR="${OUTPUT_BASE}/logs"

# Create directories
mkdir -p "$OUTPUT_BASE"
mkdir -p "$STAC_OUTPUT"
mkdir -p "$LOG_DIR"

# Parse arguments
PROCESS_TRAINING=true
PROCESS_VALIDATION=true

for arg in "$@"; do
    case $arg in
        --validation-only)
            PROCESS_TRAINING=false
            shift
            ;;
        --training-only)
            PROCESS_VALIDATION=false
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--validation-only] [--training-only]"
            echo ""
            echo "Options:"
            echo "  --validation-only  Process only validation sites (forest plots)"
            echo "  --training-only    Process only training sites"
            echo ""
            exit 0
            ;;
    esac
done

# Training sites (from vegetation structure metrics rasters)
# Bboxes extracted from fuel metrics raster footprints in data/processed/veg_structure_metrics/
declare -A TRAINING_SITES=(
    ["volcan_mtn"]="-116.609486,33.118992,-116.594239,33.137803"
    ["t01_t09"]="-120.082581,34.726003,-120.077750,34.730836" 
    ["t03_t13"]="-120.082303,34.722297,-120.078447,34.725553" 
    ["t06_t14"]="-120.086636,34.720675,-120.081500,34.724925" 
    ["trex"]="-120.056031,34.688081,-120.046894,34.695569"
)

# Validation sites (forest plots with field measurements)
declare -A VALIDATION_SITES=(
    ["BluffMesa"]="-116.959308,34.215152,-116.951654,34.222287"
    ["Laguna"]="-116.438215,32.844384,-116.424106,32.862321"
    ["NorthBigBear"]="-116.937442,34.287520,-116.917939,34.298950"
    ["ReyesPeak"]="-119.341524,34.632405,-119.282198,34.643359"
)

# Start logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
MAIN_LOG="${LOG_DIR}/processing_${TIMESTAMP}.log"
echo "3DEP HAG Feature Processing - Started at $(date)" | tee "$MAIN_LOG"
echo "========================================================" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Track results
declare -A RESULTS

# Process training sites
if [ "$PROCESS_TRAINING" = true ]; then
    echo "Processing training sites..." | tee -a "$MAIN_LOG"
    echo "----------------------------" | tee -a "$MAIN_LOG"

    for site in "${!TRAINING_SITES[@]}"; do
        bbox="${TRAINING_SITES[$site]}"
        echo "" | tee -a "$MAIN_LOG"
        echo "Processing $site with bbox: $bbox" | tee -a "$MAIN_LOG"

        if python src/data_prep/download_and_process_3dep_sites.py \
            --site "$site" \
            --bbox="$bbox" \
            --output-dir "$OUTPUT_BASE/$site" \
            --buffer 3.0 2>&1 | tee -a "$MAIN_LOG"; then
            RESULTS[$site]="SUCCESS"
            echo "✓ $site completed successfully" | tee -a "$MAIN_LOG"
        else
            RESULTS[$site]="FAILED"
            echo "✗ $site failed" | tee -a "$MAIN_LOG"
        fi
    done
fi

# Process validation sites
if [ "$PROCESS_VALIDATION" = true ]; then
    echo "" | tee -a "$MAIN_LOG"
    echo "Processing validation sites..." | tee -a "$MAIN_LOG"
    echo "------------------------------" | tee -a "$MAIN_LOG"

    for site in "${!VALIDATION_SITES[@]}"; do
        bbox="${VALIDATION_SITES[$site]}"
        echo "" | tee -a "$MAIN_LOG"
        echo "Processing $site with bbox: $bbox" | tee -a "$MAIN_LOG"

        if python src/data_prep/download_and_process_3dep_sites.py \
            --site "$site" \
            --bbox="$bbox" \
            --output-dir "$OUTPUT_BASE/$site" \
            --buffer 5.0 2>&1 | tee -a "$MAIN_LOG"; then
            RESULTS[$site]="SUCCESS"
            echo "✓ $site completed successfully" | tee -a "$MAIN_LOG"
        else
            RESULTS[$site]="FAILED"
            echo "✗ $site failed" | tee -a "$MAIN_LOG"
        fi
    done
fi

# Run verification on all processed files
echo "" | tee -a "$MAIN_LOG"
echo "Running verification..." | tee -a "$MAIN_LOG"
echo "----------------------" | tee -a "$MAIN_LOG"

python scripts/verify_3dep_hag_features.py --dir "$OUTPUT_BASE" \
    --output-json "${LOG_DIR}/verification_${TIMESTAMP}.json" 2>&1 | tee -a "$MAIN_LOG"

# Create local STAC catalog for the processed files
echo "" | tee -a "$MAIN_LOG"
echo "Creating local STAC catalog..." | tee -a "$MAIN_LOG"
echo "------------------------------" | tee -a "$MAIN_LOG"

# Only create STAC if make_local_3dep_stac.py supports processed mode
if python src/data_prep/make_local_3dep_stac.py --help 2>&1 | grep -q "processed"; then
    python src/data_prep/make_local_3dep_stac.py \
        --input-dir "$OUTPUT_BASE" \
        --output "$STAC_OUTPUT" \
        --mode processed 2>&1 | tee -a "$MAIN_LOG"
    echo "STAC catalog created at: $STAC_OUTPUT/catalog.json" | tee -a "$MAIN_LOG"
else
    echo "Note: make_local_3dep_stac.py does not yet support processed mode." | tee -a "$MAIN_LOG"
    echo "STAC catalog creation skipped." | tee -a "$MAIN_LOG"
fi

# Summary
echo "" | tee -a "$MAIN_LOG"
echo "========================================================" | tee -a "$MAIN_LOG"
echo "PROCESSING SUMMARY" | tee -a "$MAIN_LOG"
echo "========================================================" | tee -a "$MAIN_LOG"

SUCCESS_COUNT=0
FAIL_COUNT=0

for site in "${!RESULTS[@]}"; do
    status="${RESULTS[$site]}"
    if [ "$status" = "SUCCESS" ]; then
        echo "✓ $site: SUCCESS" | tee -a "$MAIN_LOG"
        ((SUCCESS_COUNT++))
    else
        echo "✗ $site: FAILED" | tee -a "$MAIN_LOG"
        ((FAIL_COUNT++))
    fi
done

echo "" | tee -a "$MAIN_LOG"
echo "Successful: $SUCCESS_COUNT" | tee -a "$MAIN_LOG"
echo "Failed: $FAIL_COUNT" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "Output directory: $OUTPUT_BASE" | tee -a "$MAIN_LOG"
echo "Log file: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "3DEP HAG Feature Processing - Completed at $(date)" | tee -a "$MAIN_LOG"

# Exit with error code if any failed
if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
