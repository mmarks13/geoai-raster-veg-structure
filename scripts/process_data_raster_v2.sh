#!/bin/bash

# Multi-Site Raster Prediction Pipeline - Training Data Generation
#
# Generates training data for vegetation structure metrics prediction model
# from 5 UAV LiDAR sites.
#
# Input: Vegetation structure metrics rasters (2m resolution, 24 bands)
# Output: Precomputed training/validation tiles in .pt format
#
# Sites:
#   - t01_t09 (Sedgwick, validation)
#   - t03_t13 (Sedgwick, training)
#   - t06_t14 (Sedgwick, training)
#   - trex (training)
#   - volcan_mtn (partial validation, partial training)
#
# Steps:
#   0. Create validation polygons (shrink Volcan by 50% vertically)
#   1. Generate tile grids per site (pixel-aligned to 2m raster)
#   2. Merge tile grids into single GeoJSON
#   3. Extract target raster + NAIP + UAVSAR per tile
#   4. Combine H5 chunks into single .pt file
#   5. Normalize and split (spatial train/val)
#   6. Data augmentation

set -e

# === Site Configuration ===
# Edit these paths if your rasters are in different locations

declare -A SITES=(
    ["t01_t09"]="data/processed/veg_structure_metrics/t01_t09/merged/t01_t09_veg_metrics_2m.tif"
    ["t03_t13"]="data/processed/veg_structure_metrics/t03_t13/merged/t03_t13_veg_metrics_2m.tif"
    ["t06_t14"]="data/processed/veg_structure_metrics/t06_t14/merged/t06_t14_veg_metrics_2m.tif"
    ["trex"]="data/processed/veg_structure_metrics/trex/merged/trex_veg_metrics_2m.tif"
    ["volcan_mtn"]="data/processed/veg_structure_metrics/volcan_mtn/merged/volcan_mtn_veg_metrics_2m.tif"
)

# === Output Configuration ===
OUTPUT_DIR="${1:-data/processed/training_data_chunks_veg_structure}"
MODEL_DATA_DIR="data/processed/model_data_veg_structure"

# --------------------------------------------------
# Step 0: Create validation polygons
# --------------------------------------------------
echo "Step 0: Create validation polygons..."

if [ ! -f data/processed/test_val_polygons_v2.geojson ]; then
    python scripts/update_val_polygons_v2.py
else
    echo "  Using existing test_val_polygons_v2.geojson"
fi

# --------------------------------------------------
# Step 1: Generate tile grids per site
# --------------------------------------------------
echo "Step 1: Generate tile grids per site..."

# Clean up old files to prevent duplicates on restart
rm -f data/processed/boundaries/*_boundary.geojson
rm -f data/processed/tiles/*_tiles.geojson
rm -f data/processed/tiles_veg_structure_all.geojson

mkdir -p data/processed/boundaries data/processed/tiles

for site in "${!SITES[@]}"; do
    raster="${SITES[$site]}"
    echo "  Processing $site..."

    gdal_footprint -b 1 "$raster" "data/processed/boundaries/${site}_boundary.geojson"

    python src/data_prep/create_tile_grid.py \
        --input "data/processed/boundaries/${site}_boundary.geojson" \
        --output "data/processed/tiles/${site}_tiles.geojson" \
        --tile-size 10.0 --overlap 0.20 \
        --raster-for-alignment "$raster" \
        --site-name "$site"
done

# --------------------------------------------------
# Step 2: Merge tile grids into single GeoJSON
# --------------------------------------------------
echo "Step 2: Merge tile grids..."

python3 << 'EOF'
import geopandas as gpd
import pandas as pd
import glob

files = sorted(glob.glob('data/processed/tiles/*_tiles.geojson'))
gdfs = [gpd.read_file(f) for f in files]
merged = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
merged.to_file('data/processed/tiles_veg_structure_all.geojson', driver='GeoJSON')

print(f"  Merged {len(merged)} tiles from {len(files)} sites:")
for site in sorted(merged['site'].unique()):
    print(f"    {site}: {len(merged[merged['site'] == site])} tiles")
EOF

# --------------------------------------------------
# Step 3: Generate training data from tiles
# --------------------------------------------------
echo "Step 3: Generate training data from tiles..."

mkdir -p "$OUTPUT_DIR"

# Build site:path mapping string from SITES array
RASTER_MAP=""
for site in "${!SITES[@]}"; do
    RASTER_MAP="${RASTER_MAP}${site}:${SITES[$site]},"
done
RASTER_MAP="${RASTER_MAP%,}"  # Remove trailing comma

python src/data_prep/generate_training_data_raster.py \
    --tiles_geojson data/processed/tiles_veg_structure_all.geojson \
    --target-raster-map "$RASTER_MAP" \
    --outdir "$OUTPUT_DIR" \
    --naip_stac_source data/stac/naip/catalog.json \
    --uavsar_stac_source data/stac/uavsar/catalog.json \
    --start_date 2014-01-01 \
    --end_date 2025-12-31 \
    --chunk_size 100 \
    --threads 14 \
    --skip-uav-lidar

# --------------------------------------------------
# Step 4: Combine H5 chunks into single .pt file
# --------------------------------------------------
echo "Step 4: Combine training data chunks..."

mkdir -p "$MODEL_DATA_DIR"

python src/data_prep/h5_chunk_loader.py \
    --input_dir "$OUTPUT_DIR" \
    --output_path "${MODEL_DATA_DIR}/combined_training_data_veg_structure.pt"

# --------------------------------------------------
# Step 5: Normalize, split, and precompute
# --------------------------------------------------
echo "Step 5: Precompute and split training data..."

python src/data_prep/train_test_split_and_precompute_raster.py \
    --pt-file "${MODEL_DATA_DIR}/combined_training_data_veg_structure.pt" \
    --geojson-file data/processed/test_val_polygons_v2.geojson \
    --output-dir "$MODEL_DATA_DIR" \
    --precision 32

# --------------------------------------------------
# Step 6: Data augmentation
# --------------------------------------------------
echo "Step 6: Data augmentation..."

python src/data_prep/data_augmentation_raster.py \
    --training_tiles "${MODEL_DATA_DIR}/precomputed_training_tiles_raster_32bit.pt" \
    --output_path "${MODEL_DATA_DIR}/augmented_tiles_raster_32bit.pt" \
    --n_augmentations 2

echo ""
echo "Pipeline complete! Training data ready in ${MODEL_DATA_DIR}/"
