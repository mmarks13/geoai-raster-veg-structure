#!/bin/bash
# Process a large site with tiling
# Usage: bash process_large_site.sh <input_las_or_zip> <site_name> <tile_size> <parallel_jobs> <point_filter_max_hag>

INPUT_FILE="$1"
SITE_NAME="$2"
TILE_SIZE="${3:-500}"
PARALLEL_JOBS="${4:-2}"
POINT_FILTER_MAX_HAG="$5"

if [ -z "$POINT_FILTER_MAX_HAG" ]; then
    echo "Error: point_filter_max_hag is required"
    echo "Usage: bash process_large_site.sh <input_las_or_zip> <site_name> <tile_size> <parallel_jobs> <point_filter_max_hag>"
    exit 1
fi

SITE_DIR="data/processed/veg_structure_metrics/$SITE_NAME"
TILES_DIR="$SITE_DIR/tiles"
RASTERS_DIR="$SITE_DIR/rasters"
MERGED_DIR="$SITE_DIR/merged"

echo "  Input: $INPUT_FILE"
echo "  Output: $SITE_DIR"
echo "  Tile size: ${TILE_SIZE}m, Parallel jobs: $PARALLEL_JOBS"

# Create directories
mkdir -p "$TILES_DIR" "$RASTERS_DIR" "$MERGED_DIR" "$SITE_DIR/logs"

# Step 1: Unzip if input is a zip file
if [[ "$INPUT_FILE" == *.zip ]]; then
    echo "  Unzipping..."
    INPUT_LAS="${INPUT_FILE%.zip}.las"
    if [ ! -f "$INPUT_LAS" ]; then
        unzip -o "$INPUT_FILE" -d "$(dirname "$INPUT_FILE")"
    fi
else
    INPUT_LAS="$INPUT_FILE"
fi

# Step 2: Tile with PDAL
echo "  Tiling into ${TILE_SIZE}m chunks..."
bash scripts/veg_structure_metrics/pdal/run_tiling.sh "$INPUT_LAS" "$TILES_DIR" "$TILE_SIZE"

# Step 3: Process tiles in parallel
echo "  Processing tiles with $PARALLEL_JOBS parallel jobs..."
find "$TILES_DIR" -name '*.laz' | parallel -j "$PARALLEL_JOBS" \
    python src/veg_structure_metrics/batch_processing.py \
        --input {} \
        --output_dir "$SITE_DIR" \
        --point_filter_max_hag "$POINT_FILTER_MAX_HAG"

# Step 4: Merge tiles
echo "  Merging tiles..."
find "$RASTERS_DIR" -name '*.tif' | sort > /tmp/merge_list.txt
gdal_merge.py -o "$MERGED_DIR/${SITE_NAME}_veg_metrics_2m.tif" \
    -a_nodata nan -co COMPRESS=LZW -co TILED=YES \
    --optfile /tmp/merge_list.txt

# Step 5: Generate visualization
echo "  Generating visualization..."
python src/veg_structure_metrics/visualize_metrics.py \
    --input "$MERGED_DIR/${SITE_NAME}_veg_metrics_2m.tif" \
    --output "$MERGED_DIR/${SITE_NAME}_visualization.png"

echo "  Done: $SITE_NAME"
