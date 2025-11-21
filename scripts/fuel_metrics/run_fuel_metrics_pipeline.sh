#!/bin/bash
# run_fuel_metrics_pipeline.sh
#
# Main orchestration script for complete fuel metrics pipeline
# Combines ground classification, tiling, pretreatment, fuel metrics computation, and merging
#
# Usage:
#   bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
#     --input <las_file> \
#     --output-name <site_name> \
#     [--species <species>] \
#     [--resolution <meters>] \
#     [--tile-size <meters>] \
#     [--parallel-jobs <n>]
#
# Example:
#   bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \
#     --input data/raw/uavlidar/my_site.las \
#     --output-name my_site \
#     --species "Mixed" \
#     --resolution 5.0 \
#     --tile-size 200 \
#     --parallel-jobs 6

set -e  # Exit on error

# Default values
SPECIES="Mixed"
RESOLUTION=5.0
TILE_SIZE=200
BUFFER=10
PARALLEL_JOBS=6
OUTPUT_BASE="data/processed/fuel_metrics"
CLUMPING=0.6
PROJECTION_FACTOR=0.5

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Usage message
usage() {
    echo ""
    echo "Usage: $0 --input <las_file> --output-name <site_name> [OPTIONS]"
    echo ""
    echo "Required arguments:"
    echo "  --input FILE          Input LAS/LAZ file path"
    echo "  --output-name NAME    Site name for output directory"
    echo ""
    echo "Optional arguments:"
    echo "  --species SPECIES           Species for trait lookup (default: Mixed)"
    echo "  --resolution METERS         Output raster resolution (default: 5.0)"
    echo "  --tile-size METERS          Tile size for processing (default: 200)"
    echo "  --parallel-jobs N           Number of parallel jobs (default: 6)"
    echo "  --output-base DIR           Base output directory (default: data/processed/fuel_metrics)"
    echo "  --clumping VALUE            Clumping factor Ω (default: 0.77)"
    echo "  --projection-factor VALUE   Projection factor G (default: 0.5)"
    echo "  --skip-classify             Skip ground classification step (use if already classified)"
    echo "  --skip-merge                Skip final merge step (process tiles only)"
    echo "  --skip-interactive-prompts  Skip confirmation prompts and overwrite existing outputs"
    echo "  --help                      Show this help message"
    echo ""
    echo "Available species (see data/processed/fuel_metrics/trait_lookup.csv):"
    echo "  Mixed, Coast live oak, Black oak, Ceanothus, Coulter pine, Incense cedar"
    echo ""
    echo "Example:"
    echo "  $0 --input data/raw/my_site.las --output-name my_site --species Mixed --resolution 5.0"
    echo ""
    exit 1
}

# Parse command-line arguments
INPUT_LAS=""
OUTPUT_NAME=""
SKIP_CLASSIFY=false
SKIP_MERGE=false
SKIP_INTERACTIVE_PROMPTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_LAS="$2"
            shift 2
            ;;
        --output-name)
            OUTPUT_NAME="$2"
            shift 2
            ;;
        --species)
            SPECIES="$2"
            shift 2
            ;;
        --resolution)
            RESOLUTION="$2"
            shift 2
            ;;
        --tile-size)
            TILE_SIZE="$2"
            shift 2
            ;;
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --output-base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --clumping)
            CLUMPING="$2"
            shift 2
            ;;
        --projection-factor)
            PROJECTION_FACTOR="$2"
            shift 2
            ;;
        --skip-classify)
            SKIP_CLASSIFY=true
            shift
            ;;
        --skip-merge)
            SKIP_MERGE=true
            shift
            ;;
        --skip-interactive-prompts)
            SKIP_INTERACTIVE_PROMPTS=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            echo -e "${RED}ERROR: Unknown argument: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$INPUT_LAS" ]; then
    echo -e "${RED}ERROR: --input is required${NC}"
    usage
fi

if [ -z "$OUTPUT_NAME" ]; then
    echo -e "${RED}ERROR: --output-name is required${NC}"
    usage
fi

# Check input file exists
if [ ! -f "$INPUT_LAS" ]; then
    echo -e "${RED}ERROR: Input file not found: $INPUT_LAS${NC}"
    exit 1
fi

# Set up output directories
SITE_DIR="$OUTPUT_BASE/$OUTPUT_NAME"
TILES_DIR="$SITE_DIR/tiles"
PRETREATED_DIR="$SITE_DIR/pretreated"
RASTERS_DIR="$SITE_DIR/rasters"
MERGED_DIR="$SITE_DIR/merged"
LOGS_DIR="$SITE_DIR/logs"
VALIDATION_DIR="$SITE_DIR/validation"

# Print configuration
echo ""
echo "================================================================================"
echo -e "${BLUE}Fuel Metrics Pipeline${NC}"
echo "================================================================================"
echo ""
echo "Configuration:"
echo "  Input:           $INPUT_LAS"
echo "  Output name:     $OUTPUT_NAME"
echo "  Species:         $SPECIES"
echo "  Resolution:      ${RESOLUTION}m"
echo "  Tile size:       ${TILE_SIZE}m × ${TILE_SIZE}m (buffer: ${BUFFER}m)"
echo "  Parallel jobs:   $PARALLEL_JOBS"
echo "  Output base:     $OUTPUT_BASE"
echo "  Clumping (Ω):    $CLUMPING"
echo "  Projection (G):  $PROJECTION_FACTOR"
echo ""
echo "Output directories:"
echo "  Site directory:  $SITE_DIR"
echo "  Tiles:           $TILES_DIR"
echo "  Pretreated:      $PRETREATED_DIR"
echo "  Rasters:         $RASTERS_DIR"
echo "  Merged:          $MERGED_DIR"
echo "  Logs:            $LOGS_DIR"
echo "  Validation:      $VALIDATION_DIR"
echo ""

# Interactive step detection (unless --force flag is used)
# Check for existing outputs and prompt user
FORCE_OVERWRITE=false
OVERWRITE_FROM_STEP=0  # 0=run all, 1=skip classify, 2=skip pretreat, 3=skip merge, 4=all done

if [ -d "$SITE_DIR" ] && [ "$SKIP_CLASSIFY" = false ] && [ "$SKIP_MERGE" = false ] && [ "$SKIP_INTERACTIVE_PROMPTS" = false ]; then
    echo "================================================================================"
    echo -e "${BLUE}Existing Output Detection${NC}"
    echo "================================================================================"
    echo ""

    # Check Step 1: Tiles
    tile_count=$(ls "$TILES_DIR"/*.laz 2>/dev/null | wc -l)
    if [ "$tile_count" -gt 0 ] && [ "$OVERWRITE_FROM_STEP" -eq 0 ]; then
        echo -e "${YELLOW}Found $tile_count existing tiles in: $TILES_DIR${NC}"
        echo "Use existing tiles or re-run ground classification?"
        echo "  [u] Use existing (skip Step 1)"
        echo "  [r] Re-run (overwrite tiles and ALL downstream: pretreated, rasters, merged)"
        echo "  [q] Quit"
        read -p "Choice (u/r/q): " tile_choice
        case "$tile_choice" in
            u|U)
                SKIP_CLASSIFY=true
                echo -e "${GREEN}Using existing tiles${NC}"
                ;;
            r|R)
                OVERWRITE_FROM_STEP=1
                echo -e "${YELLOW}Will re-run from Step 1 (all downstream will be overwritten)${NC}"
                ;;
            q|Q)
                echo "Pipeline cancelled"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        echo ""
    fi

    # Check Step 2: Pretreated files
    pretreated_count=$(ls "$PRETREATED_DIR"/*.laz 2>/dev/null | wc -l)
    if [ "$pretreated_count" -gt 0 ] && [ "$OVERWRITE_FROM_STEP" -eq 0 ]; then
        echo -e "${YELLOW}Found $pretreated_count existing pretreated files in: $PRETREATED_DIR${NC}"
        echo "Use existing pretreated files or re-run pretreatment?"
        echo "  [u] Use existing (skip Step 2 pretreatment)"
        echo "  [r] Re-run (overwrite pretreated, rasters, and merged)"
        echo "  [q] Quit"
        read -p "Choice (u/r/q): " pretreat_choice
        case "$pretreat_choice" in
            u|U)
                # Check for rasters too
                raster_count=$(ls "$RASTERS_DIR"/*.tif 2>/dev/null | wc -l)
                if [ "$raster_count" -gt 0 ]; then
                    echo -e "${GREEN}Using existing pretreated files and rasters${NC}"
                    # We'll need to skip to merge step
                    OVERWRITE_FROM_STEP=3
                else
                    echo -e "${YELLOW}Found pretreated but no rasters - will run fuel metrics computation${NC}"
                    OVERWRITE_FROM_STEP=2
                fi
                ;;
            r|R)
                OVERWRITE_FROM_STEP=2
                echo -e "${YELLOW}Will re-run from Step 2 (pretreated, rasters, and merged will be overwritten)${NC}"
                ;;
            q|Q)
                echo "Pipeline cancelled"
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid choice. Exiting.${NC}"
                exit 1
                ;;
        esac
        echo ""
    fi

    # Check Step 3: Rasters (only if we didn't already handle this above)
    if [ "$OVERWRITE_FROM_STEP" -lt 2 ]; then
        raster_count=$(ls "$RASTERS_DIR"/*.tif 2>/dev/null | wc -l)
        if [ "$raster_count" -gt 0 ]; then
            echo -e "${YELLOW}Found $raster_count existing rasters in: $RASTERS_DIR${NC}"
            echo "Use existing rasters or re-run fuel metrics computation?"
            echo "  [u] Use existing (skip to merge step)"
            echo "  [r] Re-run (overwrite rasters and merged)"
            echo "  [q] Quit"
            read -p "Choice (u/r/q): " raster_choice
            case "$raster_choice" in
                u|U)
                    OVERWRITE_FROM_STEP=3
                    echo -e "${GREEN}Using existing rasters${NC}"
                    ;;
                r|R)
                    OVERWRITE_FROM_STEP=3
                    echo -e "${YELLOW}Will re-run fuel metrics computation (rasters and merged will be overwritten)${NC}"
                    ;;
                q|Q)
                    echo "Pipeline cancelled"
                    exit 0
                    ;;
                *)
                    echo -e "${RED}Invalid choice. Exiting.${NC}"
                    exit 1
                    ;;
            esac
            echo ""
        fi
    fi

    # Check Step 4: Merged output
    if [ "$OVERWRITE_FROM_STEP" -lt 3 ]; then
        if [ -f "$MERGED_DIR"/*_fuel_metrics_*.tif 2>/dev/null ]; then
            echo -e "${YELLOW}Found existing merged output in: $MERGED_DIR${NC}"
            echo "Re-run merge or use existing?"
            echo "  [u] Use existing (pipeline complete)"
            echo "  [r] Re-run merge"
            echo "  [q] Quit"
            read -p "Choice (u/r/q): " merge_choice
            case "$merge_choice" in
                u|U)
                    OVERWRITE_FROM_STEP=4
                    echo -e "${GREEN}Pipeline already complete!${NC}"
                    echo ""
                    echo "Output location: $MERGED_DIR"
                    exit 0
                    ;;
                r|R)
                    SKIP_MERGE=false
                    echo -e "${YELLOW}Will re-run merge step${NC}"
                    ;;
                q|Q)
                    echo "Pipeline cancelled"
                    exit 0
                    ;;
                *)
                    echo -e "${RED}Invalid choice. Exiting.${NC}"
                    exit 1
                    ;;
            esac
            echo ""
        fi
    fi

    echo "================================================================================"
    echo ""
fi

# Create output directories
mkdir -p "$TILES_DIR" "$PRETREATED_DIR" "$RASTERS_DIR" "$MERGED_DIR" "$LOGS_DIR" "$VALIDATION_DIR"

# Start pipeline timing
PIPELINE_START=$(date +%s)

# =============================================================================
# STEP 1: Ground Classification + Tiling
# =============================================================================
if [ "$SKIP_CLASSIFY" = false ]; then
    echo "================================================================================"
    echo -e "${BLUE}STEP 1: Ground Classification + Tiling${NC}"
    echo "================================================================================"
    echo ""

    bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
        "$INPUT_LAS" \
        "$TILES_DIR" \
        "$TILE_SIZE" \
        "$BUFFER"

    echo ""
    echo -e "${GREEN}✓ Step 1 complete: Ground classification + tiling${NC}"
    echo ""
else
    echo "================================================================================"
    echo -e "${YELLOW}STEP 1: SKIPPED (--skip-classify)${NC}"
    echo "================================================================================"
    echo ""
    echo "Using existing tiles in: $TILES_DIR"
    tile_count=$(ls "$TILES_DIR"/*.laz 2>/dev/null | wc -l)
    echo "Found $tile_count tiles"
    echo ""
fi

# =============================================================================
# STEP 2: Batch Pretreatment + Fuel Metrics Computation
# =============================================================================
echo "================================================================================"
echo -e "${BLUE}STEP 2: Batch Pretreatment + Fuel Metrics Computation${NC}"
echo "================================================================================"
echo ""

bash scripts/fuel_metrics/run_batch_fuel_metrics.sh \
    "$TILES_DIR" \
    "$SITE_DIR" \
    "$SPECIES" \
    "$RESOLUTION" \
    "$PARALLEL_JOBS" \
    "$CLUMPING" \
    "$PROJECTION_FACTOR"

echo ""
echo -e "${GREEN}✓ Step 2 complete: Pretreatment + fuel metrics${NC}"
echo ""

# =============================================================================
# STEP 3: Merge Tiles into Seamless Mosaic
# =============================================================================
if [ "$SKIP_MERGE" = false ]; then
    echo "================================================================================"
    echo -e "${BLUE}STEP 3: Merge Tiles into Seamless Mosaic${NC}"
    echo "================================================================================"
    echo ""

    # Count available raster tiles
    raster_count=$(ls "$RASTERS_DIR"/*.tif 2>/dev/null | wc -l)
    echo "Found $raster_count fuel metrics rasters to merge"
    echo ""

    if [ "$raster_count" -eq 0 ]; then
        echo -e "${RED}ERROR: No fuel metrics rasters found in $RASTERS_DIR${NC}"
        echo "Cannot proceed with merge step"
        exit 1
    fi

    # Build explicit file list (avoid shell glob expansion issues with 70+ files)
    MERGED_OUTPUT="$MERGED_DIR/${OUTPUT_NAME}_fuel_metrics_${RESOLUTION}m.tif"
    echo "Merging $raster_count tiles into: $MERGED_OUTPUT"
    echo ""

    # Create temporary file list
    TEMP_FILE_LIST="/tmp/fuel_metrics_merge_list_$$.txt"
    find "$RASTERS_DIR" -name "*.tif" -type f | sort > "$TEMP_FILE_LIST"

    echo "Running gdal_merge.py..."
    conda run -p /home/jovyan/geoai_env gdal_merge.py \
        -o "$MERGED_OUTPUT" \
        -a_nodata nan \
        -co COMPRESS=LZW \
        -co TILED=YES \
        -co BIGTIFF=YES \
        --optfile "$TEMP_FILE_LIST" \
        -v

    rm -f "$TEMP_FILE_LIST"

    echo ""
    echo -e "${GREEN}✓ Step 3 complete: Merge successful${NC}"
    echo "  Output: $MERGED_OUTPUT"
    echo ""

    # Generate visualization
    echo "Generating visualization..."
    conda run -p /home/jovyan/geoai_env python src/fuel_metrics/visualize_metrics.py \
        "$MERGED_OUTPUT" \
        "$MERGED_DIR/${OUTPUT_NAME}_visualization.png"

    echo -e "${GREEN}✓ Visualization saved: $MERGED_DIR/${OUTPUT_NAME}_visualization.png${NC}"
    echo ""
else
    echo "================================================================================"
    echo -e "${YELLOW}STEP 3: SKIPPED (--skip-merge)${NC}"
    echo "================================================================================"
    echo ""
fi

# =============================================================================
# Pipeline Summary
# =============================================================================
PIPELINE_END=$(date +%s)
PIPELINE_DURATION=$((PIPELINE_END - PIPELINE_START))
HOURS=$((PIPELINE_DURATION / 3600))
MINUTES=$(((PIPELINE_DURATION % 3600) / 60))
SECONDS=$((PIPELINE_DURATION % 60))

echo "================================================================================"
echo -e "${GREEN}Pipeline Complete!${NC}"
echo "================================================================================"
echo ""
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo ""
echo "Output summary:"
echo "  Site:         $OUTPUT_NAME"
echo "  Tiles:        $(ls "$TILES_DIR"/*.laz 2>/dev/null | wc -l) LAZ files"
echo "  Pretreated:   $(ls "$PRETREATED_DIR"/*.laz 2>/dev/null | wc -l) LAZ files"
echo "  Rasters:      $(ls "$RASTERS_DIR"/*.tif 2>/dev/null | wc -l) TIF files"
if [ "$SKIP_MERGE" = false ]; then
    echo "  Merged:       $MERGED_DIR/${OUTPUT_NAME}_fuel_metrics_${RESOLUTION}m.tif"
    echo "  Visualization: $MERGED_DIR/${OUTPUT_NAME}_visualization.png"
fi
echo "  Logs:         $LOGS_DIR"
echo ""
echo "Next steps:"
echo "  1. View visualization:"
echo "     open $MERGED_DIR/${OUTPUT_NAME}_visualization.png"
echo ""
echo "  2. Load into QGIS/ArcGIS for spatial analysis"
echo ""
echo "  3. Check processing summary:"
echo "     cat $LOGS_DIR/tile_processing_summary.csv"
echo ""
echo "✓ All done!"
echo ""
