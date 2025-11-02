#!/bin/bash
# run_ground_classification_and_tiling.sh
#
# Consolidated PDAL pipeline that combines ground classification + tiling in a single pass
# Eliminates the need for intermediate 24GB classified LAS file
#
# Ground classification method: SMRF (Simple Morphological Filter) with default parameters
# (same method used in published work - see src/utils/point_cloud_utils.py)
#
# Usage: bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
#          <input_las> <output_dir> [tile_size] [buffer]
#
# Arguments:
#   input_las   - Path to input LAS/LAZ file (unclassified or classified)
#   output_dir  - Directory for output tile LAZ files
#   tile_size   - Tile size in meters (default: 200)
#   buffer      - Buffer overlap in meters (default: 10)
#
# Example:
#   bash scripts/fuel_metrics/pdal/run_ground_classification_and_tiling.sh \
#     data/raw/my_pointcloud.las \
#     data/processed/fuel_metrics/my_site/tiles \
#     200 10

set -e  # Exit on error

# Parse arguments
if [ $# -lt 2 ]; then
    echo "ERROR: Missing required arguments"
    echo ""
    echo "Usage: $0 <input_las> <output_dir> [tile_size] [buffer]"
    echo ""
    echo "Arguments:"
    echo "  input_las   - Path to input LAS/LAZ file"
    echo "  output_dir  - Directory for output tiles"
    echo "  tile_size   - Tile size in meters (default: 200)"
    echo "  buffer      - Buffer overlap in meters (default: 10)"
    echo ""
    echo "Example:"
    echo "  $0 data/raw/site.las data/processed/fuel_metrics/site/tiles 200 10"
    exit 1
fi

INPUT_LAS="$1"
OUTPUT_DIR="$2"
TILE_SIZE="${3:-200}"
BUFFER="${4:-10}"
TEMP_PIPELINE="/tmp/pdal_classify_tile_pipeline_$$.json"

echo "=== PDAL Ground Classification + Tiling Pipeline ==="
echo ""

# Check if input LAS exists
if [ ! -f "$INPUT_LAS" ]; then
    echo "ERROR: Input LAS not found: $INPUT_LAS"
    exit 1
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "Creating output directory: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
else
    echo "Output directory exists: $OUTPUT_DIR"
fi

# Check if tiles already exist
tile_count=$(ls "$OUTPUT_DIR"/*.laz 2>/dev/null | wc -l)
if [ "$tile_count" -gt 0 ]; then
    echo "WARNING: $tile_count tiles already exist in $OUTPUT_DIR"
    read -p "Do you want to overwrite them? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Tiling cancelled"
        exit 0
    fi
    echo "Removing existing tiles..."
    rm -f "$OUTPUT_DIR"/*.laz
fi

echo ""
echo "Extracting LAS bounds..."
# Extract bounds using pdal info
bounds_json=$(conda run -p /home/jovyan/geoai_env pdal info --summary "$INPUT_LAS" 2>/dev/null | grep -A 8 '"bounds"')

# Parse minx and miny using grep and awk
minx=$(echo "$bounds_json" | grep '"minx"' | awk -F': ' '{print $2}' | tr -d ',')
miny=$(echo "$bounds_json" | grep '"miny"' | awk -F': ' '{print $2}' | tr -d ',')
maxx=$(echo "$bounds_json" | grep '"maxx"' | awk -F': ' '{print $2}' | tr -d ',')
maxy=$(echo "$bounds_json" | grep '"maxy"' | awk -F': ' '{print $2}' | tr -d ',')

echo "Configuration:"
echo "  Input:      $INPUT_LAS"
echo "  Output:     $OUTPUT_DIR"
echo "  Tile size:  ${TILE_SIZE}m × ${TILE_SIZE}m"
echo "  Buffer:     ${BUFFER}m"
echo "  Method:     SMRF (Simple Morphological Filter)"
echo ""
echo "Input LAS bounds:"
echo "  X: $minx → $maxx"
echo "  Y: $miny → $maxy"
echo ""
echo "Tile grid origin:"
echo "  origin_x: $minx"
echo "  origin_y: $miny"
echo ""

# Calculate expected tiles
x_extent=$(awk "BEGIN {printf \"%.2f\", $maxx - $minx}")
y_extent=$(awk "BEGIN {printf \"%.2f\", $maxy - $miny}")
x_tiles=$(awk "BEGIN {print int(($maxx - $minx) / $TILE_SIZE) + 1}")
y_tiles=$(awk "BEGIN {print int(($maxy - $miny) / $TILE_SIZE) + 1}")
expected_tiles=$(awk "BEGIN {print $x_tiles * $y_tiles}")

echo "Expected output:"
echo "  Extent: ${x_extent}m × ${y_extent}m"
echo "  Grid: ${x_tiles} × ${y_tiles} tiles"
echo "  Total tiles: ~${expected_tiles}"
echo ""

# Generate PDAL pipeline JSON
cat > "$TEMP_PIPELINE" << EOF
{
  "pipeline": [
    {
      "type": "readers.las",
      "filename": "$INPUT_LAS"
    },
    {
      "type": "filters.assign",
      "assignment": "Classification[:]=0"
    },
    {
      "type": "filters.smrf"
    },
    {
      "type": "filters.splitter",
      "length": $TILE_SIZE,
      "buffer": $BUFFER,
      "origin_x": $minx,
      "origin_y": $miny
    },
    {
      "type": "writers.las",
      "filename": "$OUTPUT_DIR/tile_#_1cm.laz",
      "compression": "laszip",
      "scale_x": 0.01,
      "scale_y": 0.01,
      "scale_z": 0.01,
      "offset_x": "auto",
      "offset_y": "auto",
      "offset_z": "auto"
    }
  ]
}
EOF

echo "Generated PDAL pipeline: $TEMP_PIPELINE"
echo ""
echo "Pipeline stages:"
echo "  1. Read LAS file"
echo "  2. Reset classifications (filters.assign)"
echo "  3. Ground classification (filters.smrf with defaults)"
echo "  4. Tile into ${TILE_SIZE}m × ${TILE_SIZE}m chunks (filters.splitter)"
echo "  5. Write LAZ tiles with 1cm precision"
echo ""
echo "Starting PDAL execution..."
echo ""

# Run PDAL pipeline
start_time=$(date +%s)
conda run -p /home/jovyan/geoai_env pdal pipeline "$TEMP_PIPELINE" --verbose 4

end_time=$(date +%s)
duration=$((end_time - start_time))
minutes=$((duration / 60))
seconds=$((duration % 60))

echo ""
echo "=== Pipeline Complete ==="
echo "Duration: ${minutes}m ${seconds}s"
echo ""

# Validate output
actual_tiles=$(ls "$OUTPUT_DIR"/tile_*.laz 2>/dev/null | wc -l)
echo "Output validation:"
echo "  Expected tiles: ~${expected_tiles}"
echo "  Actual tiles:   $actual_tiles"

if [ "$actual_tiles" -eq 0 ]; then
    echo ""
    echo "ERROR: No tiles were generated!"
    echo "Check the PDAL output above for errors."
    exit 1
elif [ "$actual_tiles" -lt "$expected_tiles" ]; then
    # Allow some tolerance (edge tiles might be empty)
    tolerance=$( awk "BEGIN {print int($expected_tiles * 0.1)}" )
    diff=$((expected_tiles - actual_tiles))
    if [ "$diff" -gt "$tolerance" ]; then
        echo ""
        echo "WARNING: Fewer tiles than expected (missing $diff tiles)"
        echo "This might indicate missing data or edge effects"
        echo "Check spatial coverage with visualize_bounds.py"
    else
        echo "  Status: OK (within tolerance for edge tiles)"
    fi
else
    echo "  Status: OK"
fi

# Show tile size statistics
echo ""
echo "Tile size distribution:"
du -h "$OUTPUT_DIR"/tile_*.laz | awk '{sum+=$1; count++} END {print "  Average: " sum/count " per tile"}'
du -sh "$OUTPUT_DIR" | awk '{print "  Total: " $1}'

echo ""
echo "Output location: $OUTPUT_DIR"
echo ""
echo "✓ Ground classification + tiling complete!"
echo ""
echo "Next steps:"
echo "  1. Run fuel metrics pretreatment + computation:"
echo "     bash scripts/fuel_metrics/run_batch_fuel_metrics.sh \\"
echo "       $OUTPUT_DIR \\"
echo "       ${OUTPUT_DIR%/tiles}/rasters \\"
echo "       Mixed 5.0 6"
echo ""

# Cleanup
rm -f "$TEMP_PIPELINE"
