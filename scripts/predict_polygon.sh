#!/bin/bash
#
# Polygon → Vegetation-Structure Map
#
# Thin wrapper around src/evaluation/predict_polygon.py. Takes an AOI polygon and
# produces a polygon-masked Cloud-Optimized GeoTIFF + per-band GeoTIFFs + band-stack
# VRT + OUTPUT_README under data/output/polygon_predictions/<NAME>/.
#
# Mirrors scripts/evaluate_forest_plots.sh's argument and multi-GPU pattern.
#
# Usage:
#   bash scripts/predict_polygon.sh --polygon path/to/aoi.gpkg [options]
#
#   # Tiny smoke test (clip a 100 m box from a corner; live NAIP + 3DEP fetch):
#   bash scripts/predict_polygon.sh \
#       --polygon data/raw/inference_aoi/LagunaProjectArea.zip \
#       --name LagunaSmoke100 \
#       --clip-box-corner -116.4343358 32.8358660 --clip-box-size 100 \
#       --tile-filter intersects --mc-samples 30 --batch-size 64
#
# Options (forwarded to predict_polygon.py):
#   --polygon PATH            AOI polygon: GeoJSON / GPKG / SHP / zipped SHP (required)
#   --name NAME               Output + cache identifier (default: polygon stem)
#   --model PATH              Checkpoint (default: pretrained 3-band NAIP epoch_100)
#   --band-config PATH        Band config JSON (default: veg_structure_3band_v2.json)
#   --mc-samples N            MC-dropout samples (default: 30)
#   --batch-size N            Inference batch size (default: 64)
#   --stride METERS           Tile-grid stride (default: 8.0 = 20% overlap)
#   --tile-filter MODE        centroid | contains | intersects (default: intersects)
#   --min-dep-points N        Minimum 3DEP points per tile (default: 50)
#   --multi-gpu               Enable multi-GPU DDP inference (auto-detect GPUs)
#   --num-gpus N              GPU count for multi-GPU mode
#   --device DEVICE           Single-device override (cuda/cpu; auto-detected)
#   --naip-start DATE         NAIP search start (default: 2016-01-01)
#   --naip-end DATE           NAIP search end (default: 2024-12-31)
#   --dep-date-range RANGE    3DEP STAC date range (default: 2015-01-01/2024-12-31)
#   --clip-box-corner LON LAT WGS84 top-left corner to clip a fixed square (smoke test)
#   --clip-box-size METERS    Side length of the clip square
#   --help                    Show this help message
#
# NOTE: --tile-filter defaults to 'intersects' here, which differs from the
#       create_forest_plot_tile_grid.py helper's 'contains'-only behavior; the
#       filtering is applied in the driver against the true AOI geometry.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

# Resolve the project Python interpreter (prefer the geoai_env interpreter).
PYTHON="${PYTHON:-/home/jovyan/geoai_env/bin/python}"
if [ ! -x "$PYTHON" ]; then
    PYTHON="python"
fi

if [[ "$*" == *"--help"* || $# -eq 0 ]]; then
    grep "^#" "${BASH_SOURCE[0]}" | grep -v "#!/bin/bash" | sed 's/^# \{0,1\}//'
    exit 0
fi

# Auto-detect multi-GPU: if --multi-gpu is requested without --num-gpus and more
# than one CUDA device is visible, log it; the driver does the real fallback logic.
echo "========================================"
echo "POLYGON → VEGETATION-STRUCTURE MAP"
echo "========================================"
echo "Interpreter: $PYTHON"
echo "Args: $*"
echo ""

exec "$PYTHON" src/evaluation/predict_polygon.py "$@"
