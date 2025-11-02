#!/bin/bash
# run_tiling.sh - Tiles large classified point cloud into manageable chunks
#
# This script dynamically extracts LAS bounds and generates a pipeline with
# proper origin alignment to ensure complete spatial coverage
#
# Usage: bash scripts/pdal/run_tiling.sh <input_las> <output_dir> [tile_size] [buffer]
#
# Arguments:
#   input_las   - Path to input LAS/LAZ file
#   output_dir  - Directory for output tiles
#   tile_size   - Tile size in meters (default: 100)
#   buffer      - Buffer overlap in meters (default: 10)
#
# Example:
#   bash scripts/pdal/run_tiling.sh data/raw/my_pointcloud.las data/processed/tiles 100 10

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
    echo "  tile_size   - Tile size in meters (default: 100)"
    echo "  buffer      - Buffer overlap in meters (default: 10)"
    echo ""
    echo "Example:"
    echo "  $0 data/raw/my_pointcloud.las data/processed/tiles 100 10"
    exit 1
fi

INPUT_LAS="$1"
OUTPUT_DIR="$2"
TILE_SIZE="${3:-100}"
BUFFER="${4:-10}"
TEMP_PIPELINE="/tmp/pdal_tiling_pipeline_$$.json"

echo "=== PDAL Tiling Script ==="
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
echo ""
echo "Input LAS bounds:"
echo "  X: $minx → $maxx"
echo "  Y: $miny → $maxy"
echo ""
echo "Tile grid origin will be set to:"
echo "  origin_x: $minx"
echo "  origin_y: $miny"
echo ""

# Calculate expected tiles
x_extent=$(awk "BEGIN {printf \"%.2f\", $maxx - $minx}")
y_extent=$(awk "BEGIN {printf \"%.2f\", $maxy - $miny}")
tiles_x=$(awk "BEGIN {printf \"%d\", ($x_extent / $TILE_SIZE) + 1}")
tiles_y=$(awk "BEGIN {printf \"%d\", ($y_extent / $TILE_SIZE) + 1}")
tiles_expected=$(awk "BEGIN {printf \"%d\", $tiles_x * $tiles_y}")

echo "Expected tile coverage:"
echo "  ${tiles_x} tiles (X) × ${tiles_y} tiles (Y) = ~${tiles_expected} tiles"
echo ""

# Generate dynamic pipeline JSON
cat > "$TEMP_PIPELINE" << EOF
{
  "_comment": "Dynamically generated pipeline - tiles point cloud into ${TILE_SIZE}m × ${TILE_SIZE}m chunks",
  "_generated_by": "run_tiling.sh",
  "pipeline": [
    "$INPUT_LAS",
    {
      "type": "filters.splitter",
      "length": $TILE_SIZE,
      "buffer": $BUFFER,
      "origin_x": "$minx",
      "origin_y": "$miny"
    },
    {
      "type": "writers.las",
      "filename": "$OUTPUT_DIR/tile_#_1cm.laz",
      "compression": "laszip"
    }
  ]
}
EOF

echo "Starting PDAL tiling process..."
echo ""

# Run PDAL tiling
conda run -p /home/jovyan/geoai_env pdal pipeline "$TEMP_PIPELINE"

# Clean up temporary pipeline
rm -f "$TEMP_PIPELINE"

# Verify output
final_count=$(ls "$OUTPUT_DIR"/*.laz 2>/dev/null | wc -l)
echo ""
echo "=== Tiling Complete ==="
echo "Tiles created: $final_count"
echo "Expected:      ~$tiles_expected tiles"
echo "Location:      $OUTPUT_DIR"
echo ""

# Quality checks
if [ "$final_count" -eq 0 ]; then
    echo "ERROR: No tiles were created!"
    echo "Possible causes:"
    echo "  - PDAL pipeline failed silently"
    echo "  - Output directory permissions issue"
    echo "  - Input LAS file is empty or corrupted"
    exit 1
fi

# Validate spatial coverage by checking actual tile extents
echo "Validating spatial coverage..."
echo "(This may take a moment for large tile counts)"
echo ""

# Extract bounds from all tiles
tiles_bounds_temp="/tmp/tiles_bounds_$$.txt"
for tile in "$OUTPUT_DIR"/tile_*_1cm.laz; do
    conda run -p /home/jovyan/geoai_env pdal info --summary "$tile" 2>/dev/null | \
        grep -E '"(minx|maxx|miny|maxy)"' >> "$tiles_bounds_temp"
done

# Calculate overall extent of all tiles
read tiles_minx tiles_maxx tiles_miny tiles_maxy < <(
    awk -F': ' '
    {
        gsub(/,/, "", $2)
        if ($1 ~ /minx/) {if ($2 < minx || minx == "") minx = $2}
        if ($1 ~ /maxx/) {if ($2 > maxx) maxx = $2}
        if ($1 ~ /miny/) {if ($2 < miny || miny == "") miny = $2}
        if ($1 ~ /maxy/) {if ($2 > maxy) maxy = $2}
    }
    END {print minx, maxx, miny, maxy}
    ' "$tiles_bounds_temp"
)
rm -f "$tiles_bounds_temp"

echo "Original LAS extent:"
echo "  X: $minx → $maxx"
echo "  Y: $miny → $maxy"
echo ""
echo "Tiles combined extent:"
echo "  X: $tiles_minx → $tiles_maxx"
echo "  Y: $tiles_miny → $tiles_maxy"
echo ""

# Compare extents (allow 1m tolerance for floating point differences)
x_coverage_ok=0
y_coverage_ok=0

x_diff_min=$(awk "BEGIN {printf \"%.2f\", ($tiles_minx - $minx) < 0 ? ($minx - $tiles_minx) : ($tiles_minx - $minx)}")
x_diff_max=$(awk "BEGIN {printf \"%.2f\", ($maxx - $tiles_maxx) < 0 ? ($tiles_maxx - $maxx) : ($maxx - $tiles_maxx)}")
y_diff_min=$(awk "BEGIN {printf \"%.2f\", ($tiles_miny - $miny) < 0 ? ($miny - $tiles_miny) : ($tiles_miny - $miny)}")
y_diff_max=$(awk "BEGIN {printf \"%.2f\", ($maxy - $tiles_maxy) < 0 ? ($tiles_maxy - $maxy) : ($maxy - $tiles_maxy)}")

if awk "BEGIN {exit !($x_diff_min <= 1 && $x_diff_max <= 1)}"; then
    x_coverage_ok=1
fi

if awk "BEGIN {exit !($y_diff_min <= 1 && $y_diff_max <= 1)}"; then
    y_coverage_ok=1
fi

if [ "$x_coverage_ok" -eq 1 ] && [ "$y_coverage_ok" -eq 1 ]; then
    echo "✓ Spatial coverage: COMPLETE"
    echo "  Tiles cover 100% of input LAS data extent"
    if [ "$final_count" -lt "$tiles_expected" ]; then
        echo ""
        echo "Note: $((tiles_expected - final_count)) fewer tiles than bounding box grid suggests."
        echo "This is normal - PDAL skips empty grid cells where no point data exists."
    fi
else
    echo "✗ Spatial coverage: INCOMPLETE"
    echo ""
    echo "Missing extent (meters):"
    [ $(awk "BEGIN {print ($x_diff_min > 1)}") -eq 1 ] && echo "  X min: $x_diff_min m gap"
    [ $(awk "BEGIN {print ($x_diff_max > 1)}") -eq 1 ] && echo "  X max: $x_diff_max m gap"
    [ $(awk "BEGIN {print ($y_diff_min > 1)}") -eq 1 ] && echo "  Y min: $y_diff_min m gap"
    [ $(awk "BEGIN {print ($y_diff_max > 1)}") -eq 1 ] && echo "  Y max: $y_diff_max m gap"
    echo ""
    echo "WARNING: Tiles do not cover full input extent!"
    echo "Possible causes:"
    echo "  - Origin alignment issue"
    echo "  - PDAL pipeline error"
    echo "  - Tile size too large for data extent"
fi

echo ""

# Check for suspiciously small tiles (< 100KB for 1cm precision LAZ)
echo "Checking tile sizes..."
small_tiles=$(find "$OUTPUT_DIR" -name "*.laz" -size -100k | wc -l)
if [ "$small_tiles" -gt 0 ]; then
    echo "WARNING: Found $small_tiles tiles smaller than 100KB"
    echo "These may be edge tiles with sparse data:"
    find "$OUTPUT_DIR" -name "*.laz" -size -100k -exec ls -lh {} \; | head -10
    if [ "$small_tiles" -gt 10 ]; then
        echo "... and $((small_tiles - 10)) more"
    fi
    echo ""
    echo "This is usually normal for boundary tiles with low point density."
else
    echo "✓ All tiles have reasonable file sizes (>100KB)"
fi

echo ""

if [ "$final_count" -gt 0 ]; then
    echo "Sample tiles:"
    ls -lh "$OUTPUT_DIR"/*.laz | head -5
    if [ "$final_count" -gt 5 ]; then
        echo "... and $((final_count - 5)) more"
    fi
fi

echo ""
echo "=== Summary ==="
if [ "$x_coverage_ok" -eq 1 ] && [ "$y_coverage_ok" -eq 1 ]; then
    echo "Status:   ✓ SUCCESS - Complete spatial coverage"
else
    echo "Status:   ✗ FAILED - Incomplete spatial coverage"
fi
echo "Tiles:    $final_count created"
echo "Expected: ~$tiles_expected tiles (based on bounding box grid)"
echo "Coverage: Verified against actual data extent"
