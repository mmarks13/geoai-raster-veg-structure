#!/bin/bash
# run_batch_fuel_metrics.sh - Batch process LiDAR tiles through fuel metrics pipeline
#
# Processes all tiles in parallel with per-tile logging and summary tracking.
#
# Usage: bash scripts/run_batch_fuel_metrics.sh <tiles_dir> <output_dir> [species] [resolution] [num_jobs]
#
# Arguments:
#   tiles_dir   - Directory containing tile_*.laz files
#   output_dir  - Output directory for processed tiles
#   species     - Species/category for traits (default: Mixed)
#   resolution  - Output resolution in meters (default: 5.0)
#   num_jobs    - Number of parallel jobs (default: 6)
#
# Example:
#   bash scripts/run_batch_fuel_metrics.sh \
#     data/processed/fuel_metrics/volcan_tiles \
#     data/processed/fuel_metrics/volcan_tiles_output \
#     Mixed \
#     5.0 \
#     6

set -e

# Parse arguments
if [ $# -lt 2 ]; then
    echo "ERROR: Missing required arguments"
    echo ""
    echo "Usage: $0 <tiles_dir> <output_dir> [species] [resolution] [num_jobs] [clumping] [projection_factor]"
    echo ""
    echo "Arguments:"
    echo "  tiles_dir         - Directory containing tile_*.laz files"
    echo "  output_dir        - Output directory for processed tiles"
    echo "  species           - Species/category for traits (default: Mixed)"
    echo "  resolution        - Output resolution in meters (default: 5.0)"
    echo "  num_jobs          - Number of parallel jobs (default: 6)"
    echo "  clumping          - Clumping factor Ω (default: 0.77)"
    echo "  projection_factor - Projection factor G (default: 0.5)"
    echo ""
    echo "Example:"
    echo "  $0 data/processed/fuel_metrics/volcan_tiles data/processed/fuel_metrics/volcan_tiles_output"
    exit 1
fi

TILES_DIR="$1"
OUTPUT_DIR="$2"
SPECIES="${3:-Mixed}"
RESOLUTION="${4:-5.0}"
NUM_JOBS="${5:-6}"
CLUMPING="${6:-0.77}"
PROJECTION_FACTOR="${7:-0.5}"

# Use site-specific log directory
LOG_DIR="${OUTPUT_DIR}/logs"
CSV_SUMMARY="${OUTPUT_DIR}/logs/tile_processing_summary.csv"

echo "=== Batch Fuel Metrics Processing ==="
echo ""
echo "Configuration:"
echo "  Tiles directory: $TILES_DIR"
echo "  Output directory: $OUTPUT_DIR"
echo "  Species: $SPECIES"
echo "  Resolution: ${RESOLUTION}m"
echo "  Parallel jobs: $NUM_JOBS"
echo "  Clumping (Ω): $CLUMPING"
echo "  Projection (G): $PROJECTION_FACTOR"
echo "  Log directory: $LOG_DIR"
echo "  Summary CSV: $CSV_SUMMARY"
echo ""

# Validate tiles directory
if [ ! -d "$TILES_DIR" ]; then
    echo "ERROR: Tiles directory not found: $TILES_DIR"
    exit 1
fi

# Count tiles
tile_count=$(find "$TILES_DIR" -maxdepth 1 -name 'tile_*_1cm.laz' -type f | wc -l)
if [ "$tile_count" -eq 0 ]; then
    echo "ERROR: No tiles found in $TILES_DIR (pattern: tile_*_1cm.laz)"
    exit 1
fi

echo "Found $tile_count tiles to process"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Check for existing output files
existing_pretreated=$(find "$OUTPUT_DIR/pretreated" -name "*.laz" 2>/dev/null | wc -l)
existing_rasters=$(find "$OUTPUT_DIR/rasters" -name "*.tif" 2>/dev/null | wc -l)

if [ "$existing_pretreated" -gt 0 ] || [ "$existing_rasters" -gt 0 ]; then
    echo "WARNING: Existing output files found"
    echo "  Pretreated LAZ files: $existing_pretreated"
    echo "  Raster TIF files: $existing_rasters"
    read -p "Delete existing output files and start fresh? (y/N): " confirm
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        echo "Removing existing output files..."
        rm -rf "$OUTPUT_DIR/pretreated" "$OUTPUT_DIR/rasters"
        mkdir -p "$OUTPUT_DIR/pretreated" "$OUTPUT_DIR/rasters"
        echo "Output directories cleaned"
    fi
fi

# Check for existing summary CSV and auto-archive
if [ -f "$CSV_SUMMARY" ]; then
    existing_count=$(tail -n +2 "$CSV_SUMMARY" 2>/dev/null | wc -l)
    if [ "$existing_count" -gt 0 ]; then
        backup_csv="${CSV_SUMMARY}.$(date +%Y%m%d_%H%M%S).bak"
        mv "$CSV_SUMMARY" "$backup_csv"
        echo "Found existing CSV with $existing_count entries - archived to: $backup_csv"
    fi
fi

echo "Starting parallel processing..."
echo "Progress can be monitored with:"
echo "  watch -n 5 'tail -1 $CSV_SUMMARY'"
echo ""
echo "Press Ctrl+C to stop (running jobs will complete)"
echo ""

# Run in parallel using GNU parallel if available, otherwise fall back to xargs
if command -v parallel &> /dev/null; then
    echo "Using GNU parallel for job control"
    find "$TILES_DIR" -maxdepth 1 -name 'tile_*_1cm.laz' -type f -print0 | \
      parallel -0 -j "$NUM_JOBS" --bar \
      "python src/fuel_metrics/batch_processing.py \
        --input {} \
        --output_dir $OUTPUT_DIR \
        --species $SPECIES \
        --resolution $RESOLUTION \
        --clumping $CLUMPING \
        --projection_factor $PROJECTION_FACTOR \
        --log_dir $LOG_DIR \
        --csv_summary $CSV_SUMMARY"
else
    echo "Using xargs for job control (install GNU parallel for better progress tracking)"
    find "$TILES_DIR" -maxdepth 1 -name 'tile_*_1cm.laz' -type f -print0 | \
      xargs -0 -n1 -P"$NUM_JOBS" -I{} \
      python src/fuel_metrics/batch_processing.py \
        --input {} \
        --output_dir "$OUTPUT_DIR" \
        --species "$SPECIES" \
        --resolution "$RESOLUTION" \
        --clumping "$CLUMPING" \
        --projection_factor "$PROJECTION_FACTOR" \
        --log_dir "$LOG_DIR" \
        --csv_summary "$CSV_SUMMARY"
fi

# Generate summary report
echo ""
echo "=== Processing Complete ==="
echo ""

if [ -f "$CSV_SUMMARY" ]; then
    total=$(tail -n +2 "$CSV_SUMMARY" | wc -l)
    success=$(tail -n +2 "$CSV_SUMMARY" | grep -c ",SUCCESS," || true)
    failed=$((total - success))

    echo "Results:"
    echo "  Total tiles: $total"
    echo "  Successful: $success"
    echo "  Failed: $failed"

    if [ "$failed" -gt 0 ]; then
        echo ""
        echo "Failed tiles:"
        tail -n +2 "$CSV_SUMMARY" | grep -v ",SUCCESS," | cut -d',' -f1,2,13 | head -10
        if [ "$failed" -gt 10 ]; then
            echo "  ... and $((failed - 10)) more (see $CSV_SUMMARY)"
        fi
    fi

    echo ""
    echo "Summary CSV: $CSV_SUMMARY"
    echo "Logs directory: $LOG_DIR"
else
    echo "WARNING: Summary CSV not found. Check for errors."
fi

echo ""
echo "Next steps:"
echo "  1. Review summary CSV: $CSV_SUMMARY"
echo "  2. Merge tiles: gdal_merge.py -o <output.tif> -n nan -a_nodata nan $OUTPUT_DIR/rasters/*.tif"
echo "  3. Visualize: python scripts/visualize_fuel_metrics_simple.py <output.tif> <output.png>"
