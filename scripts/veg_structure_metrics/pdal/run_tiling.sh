#!/bin/bash
# PDAL pipeline: ground classification + tiling
# Usage: bash run_tiling.sh <input_las> <output_dir> <tile_size>

INPUT_LAS="$1"
OUTPUT_DIR="$2"
TILE_SIZE="${3:-500}"

echo "  Running PDAL tiling pipeline..."

# Build and run PDAL JSON pipeline
# Pipeline: read → reset classification → SMRF ground filter → HAG → split → write tiles
conda run -p /home/jovyan/geoai_env pdal pipeline <(cat <<EOF
{
  "pipeline": [
    {"type": "readers.las", "filename": "$INPUT_LAS"},
    {"type": "filters.reprojection", "out_srs": "EPSG:32611"},
    {"type": "filters.assign", "assignment": "Classification[:]=0"},
    {"type": "filters.smrf", "cell": 1.0, "slope": 0.15, "threshold": 0.5, "window": 18.0},
    {"type": "filters.hag_nn"},
    {"type": "filters.splitter", "length": $TILE_SIZE, "buffer": 10},
    {"type": "writers.las", "filename": "$OUTPUT_DIR/tile_#.laz", "compression": "laszip"}
  ]
}
EOF
)

echo "  Tiling complete: $(ls $OUTPUT_DIR/*.laz | wc -l) tiles created"
