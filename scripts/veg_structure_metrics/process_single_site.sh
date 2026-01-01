#!/bin/bash
# Process a single site (whole file, no tiling)
# Usage: bash process_single_site.sh <input_las> <site_name> <point_filter_max_hag>

INPUT_LAS="$1"
SITE_NAME="$2"
POINT_FILTER_MAX_HAG="$3"

if [ -z "$POINT_FILTER_MAX_HAG" ]; then
    echo "Error: point_filter_max_hag is required"
    echo "Usage: bash process_single_site.sh <input_las> <site_name> <point_filter_max_hag>"
    exit 1
fi

SITE_DIR="data/processed/veg_structure_metrics/$SITE_NAME"
OUTPUT_TIF="$SITE_DIR/merged/${SITE_NAME}_veg_metrics_2m.tif"
OUTPUT_PNG="$SITE_DIR/merged/${SITE_NAME}_visualization.png"

echo "  Input: $INPUT_LAS"
echo "  Output: $SITE_DIR"

# Create directories
mkdir -p "$SITE_DIR/merged" "$SITE_DIR/logs"

# Compute vegetation structure metrics
python -c "
from src.utils.point_cloud_utils import compute_vegetation_structure_metrics, save_metrics_to_geotiff

raster, metadata = compute_vegetation_structure_metrics(
    '$INPUT_LAS',
    resolution=2.0,
    point_filter_max_hag=$POINT_FILTER_MAX_HAG,
    preprocess=True
)
save_metrics_to_geotiff(raster, metadata, '$OUTPUT_TIF')
"

# Generate visualization
python src/veg_structure_metrics/visualize_metrics.py \
    --input "$OUTPUT_TIF" \
    --output "$OUTPUT_PNG"

echo "  Done: $SITE_NAME"
