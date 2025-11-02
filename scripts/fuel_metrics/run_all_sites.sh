#!/bin/bash
# run_all_sites.sh
#
# Batch process multiple UAV LiDAR sites through complete fuel metrics pipeline.
# Each site runs sequentially to avoid overwhelming system resources.
#
# Usage:
#   bash scripts/fuel_metrics/run_all_sites.sh
#
# To enable/disable specific sites, comment/uncomment the corresponding lines
# in the SITES array definition below.

set -e  # Exit on error (but we'll trap errors per-site)

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Shared processing parameters
SPECIES="Mixed"
RESOLUTION=5.0
TILE_SIZE=200
PARALLEL_JOBS=6
CLUMPING=0.65
PROJECTION_FACTOR=0.5

# Define sites as array of "input_file|output_name" pairs
# Comment out lines to skip specific sites
SITES=(
    "data/raw/uavlidar/full_volcan_mtn_las/VolcanMt_20231025_LAS.las|volcan_mtn"  # Already processed
    # "data/raw/uavlidar/study_las/T01-T09_LiDAR_20230928_Pre_LAS.las|sedgwick_t01_t09"
    # "data/raw/uavlidar/study_las/T03-T13_LIDAR_20231025_Pre_LAS.las|sedgwick_t03_t13"
    # "data/raw/uavlidar/study_las/T06-T14_LIDAR_20231025_Pre_LAS.las|sedgwick_t06_t14"
    # "data/raw/uavlidar/study_las/TREX_LIDAR_20230630_Pre_LAS.las|sedgwick_trex"
)

# Pipeline script
PIPELINE_SCRIPT="scripts/fuel_metrics/run_fuel_metrics_pipeline.sh"

# Validate pipeline script exists
if [ ! -f "$PIPELINE_SCRIPT" ]; then
    echo -e "${RED}ERROR: Pipeline script not found: $PIPELINE_SCRIPT${NC}"
    exit 1
fi

# Print configuration
echo ""
echo "================================================================================"
echo -e "${BLUE}Multi-Site Fuel Metrics Processing${NC}"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Species:         $SPECIES"
echo "  Resolution:      ${RESOLUTION}m"
echo "  Tile size:       ${TILE_SIZE}m × ${TILE_SIZE}m"
echo "  Parallel jobs:   $PARALLEL_JOBS"
echo "  Clumping (Ω):    $CLUMPING"
echo "  Projection (G):  $PROJECTION_FACTOR"
echo ""
echo "Sites to process:"
for i in "${!SITES[@]}"; do
    IFS='|' read -r input_file output_name <<< "${SITES[$i]}"
    echo "  $((i+1)). $output_name"
    echo "     Input: $input_file"
done
echo ""
echo "================================================================================"
echo ""

# Validate all input files exist before starting
echo "Validating input files..."
ALL_VALID=true
for site in "${SITES[@]}"; do
    IFS='|' read -r input_file output_name <<< "$site"
    if [ ! -f "$input_file" ]; then
        echo -e "${RED}ERROR: Input file not found: $input_file${NC}"
        ALL_VALID=false
    else
        file_size=$(du -h "$input_file" | cut -f1)
        echo -e "  ${GREEN}✓${NC} $output_name: $file_size"
    fi
done

if [ "$ALL_VALID" = false ]; then
    echo ""
    echo -e "${RED}ERROR: One or more input files not found. Exiting.${NC}"
    exit 1
fi

echo ""
echo "All input files validated."
echo ""

# Confirmation prompt
read -p "Proceed with processing ${#SITES[@]} sites? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Processing cancelled."
    exit 0
fi

echo ""
echo "================================================================================"
echo "Starting multi-site processing..."
echo "================================================================================"
echo ""

# Start overall timing
OVERALL_START=$(date +%s)

# Track results
SUCCESSFUL_SITES=()
FAILED_SITES=()

# Process each site
for i in "${!SITES[@]}"; do
    IFS='|' read -r input_file output_name <<< "${SITES[$i]}"

    SITE_NUM=$((i+1))
    TOTAL_SITES=${#SITES[@]}

    echo ""
    echo "================================================================================"
    echo -e "${BLUE}[$SITE_NUM/$TOTAL_SITES] Processing: $output_name${NC}"
    echo "================================================================================"
    echo ""
    echo "Input:  $input_file"
    echo "Output: data/processed/fuel_metrics/$output_name"
    echo ""

    SITE_START=$(date +%s)

    # Run pipeline (don't exit on error, capture result)
    # Use --skip-interactive-prompts to prevent mid-processing confirmations
    set +e
    bash "$PIPELINE_SCRIPT" \
        --input "$input_file" \
        --output-name "$output_name" \
        --species "$SPECIES" \
        --resolution "$RESOLUTION" \
        --tile-size "$TILE_SIZE" \
        --parallel-jobs "$PARALLEL_JOBS" \
        --clumping "$CLUMPING" \
        --projection-factor "$PROJECTION_FACTOR" \
        --skip-interactive-prompts

    PIPELINE_EXIT_CODE=$?
    set -e

    SITE_END=$(date +%s)
    SITE_DURATION=$((SITE_END - SITE_START))
    SITE_HOURS=$((SITE_DURATION / 3600))
    SITE_MINUTES=$(((SITE_DURATION % 3600) / 60))
    SITE_SECONDS=$((SITE_DURATION % 60))

    if [ $PIPELINE_EXIT_CODE -eq 0 ]; then
        echo ""
        echo -e "${GREEN}✓ Site $output_name completed successfully${NC}"
        echo "  Duration: ${SITE_HOURS}h ${SITE_MINUTES}m ${SITE_SECONDS}s"
        SUCCESSFUL_SITES+=("$output_name")
    else
        echo ""
        echo -e "${RED}✗ Site $output_name FAILED (exit code: $PIPELINE_EXIT_CODE)${NC}"
        echo "  Duration: ${SITE_HOURS}h ${SITE_MINUTES}m ${SITE_SECONDS}s"
        FAILED_SITES+=("$output_name")
    fi

    echo ""
done

# Overall timing
OVERALL_END=$(date +%s)
OVERALL_DURATION=$((OVERALL_END - OVERALL_START))
OVERALL_HOURS=$((OVERALL_DURATION / 3600))
OVERALL_MINUTES=$(((OVERALL_DURATION % 3600) / 60))
OVERALL_SECONDS=$((OVERALL_DURATION % 60))

# Final summary
echo ""
echo "================================================================================"
echo -e "${BLUE}Multi-Site Processing Complete${NC}"
echo "================================================================================"
echo ""
echo "Overall duration: ${OVERALL_HOURS}h ${OVERALL_MINUTES}m ${OVERALL_SECONDS}s"
echo ""
echo "Results:"
echo "  Successful: ${#SUCCESSFUL_SITES[@]}/${#SITES[@]}"
echo "  Failed:     ${#FAILED_SITES[@]}/${#SITES[@]}"
echo ""

if [ ${#SUCCESSFUL_SITES[@]} -gt 0 ]; then
    echo "Successful sites:"
    for site in "${SUCCESSFUL_SITES[@]}"; do
        echo -e "  ${GREEN}✓${NC} $site"
    done
    echo ""
fi

if [ ${#FAILED_SITES[@]} -gt 0 ]; then
    echo "Failed sites:"
    for site in "${FAILED_SITES[@]}"; do
        echo -e "  ${RED}✗${NC} $site"
    done
    echo ""
fi

echo "Output locations:"
echo "  data/processed/fuel_metrics/<site_name>/merged/*.tif"
echo "  data/processed/fuel_metrics/<site_name>/merged/*_visualization.png"
echo ""

if [ ${#FAILED_SITES[@]} -gt 0 ]; then
    echo -e "${YELLOW}WARNING: Some sites failed. Check logs in:${NC}"
    echo "  data/processed/fuel_metrics/<site_name>/logs/"
    echo ""
    exit 1
fi

echo -e "${GREEN}✓ All sites processed successfully!${NC}"
echo ""

exit 0
