#!/bin/bash

# Raster Prediction Pipeline - Training Data Generation
#
# Generates training data for fuel metrics raster prediction model
# Input: Fuel metrics raster (2.0m or 5.0m resolution)
# Output: Precomputed training/validation tiles in .pt format
#
# Steps:
#   0. Extract UAV LiDAR coverage boundary (gdal_footprint)
#   1. Generate tile grid filtered by boundary
#   2. Extract fuel metrics + NAIP + UAVSAR per tile
#   3. Combine H5 chunks into single .pt file
#   4. Normalize and split (90/10 train/val)
#   5. Data augmentation

# Extract UAV LiDAR coverage boundary from fuel metrics raster
echo "Step 0: Extract UAV LiDAR coverage boundary..."
gdal_footprint \
  -b 15 \
  -max_points unlimited \
  -simplify 0 \
  -min_ring_area 20000 \
  -t_srs EPSG:32611 \
  data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \
  data/processed/uav_lidar_coverage_boundary.geojson

# Add 'site' column to boundary GeoJSON (required by create_tile_grid.py)
echo "Injecting 'site' column into boundary GeoJSON..."
python3 << 'EOF'
import geopandas as gpd
from pathlib import Path

gdf = gpd.read_file('data/processed/uav_lidar_coverage_boundary.geojson')

# Extract site name from raster filename in 'location' field
if 'location' in gdf.columns and len(gdf) > 0:
    location = gdf['location'].iloc[0]
    basename = Path(location).stem  # e.g., "volcan_mtn_fuel_metrics_2.0m"
    if '_fuel_metrics' in basename:
        site = basename.split('_fuel_metrics')[0]  # "volcan_mtn"
    else:
        site = 'volcan_mtn'  # fallback
else:
    site = 'volcan_mtn'  # fallback if no location field

gdf['site'] = site
gdf.to_file('data/processed/uav_lidar_coverage_boundary.geojson', driver='GeoJSON')
print(f"✓ Added 'site' column: {site}")
EOF

# Generate tile grid from boundary polygon
echo "Step 1: Generate tile grid from UAV boundary..."
python src/data_prep/create_tile_grid.py \
  --input data/processed/uav_lidar_coverage_boundary.geojson \
  --output data/processed/tiles_raster.geojson \
  --tile-size 10.0 \
  --overlap 0.15

# Generate training data: fuel metrics + 3DEP + NAIP + UAVSAR per tile
echo "Step 2: Generate training data from tiles..."
python src/data_prep/generate_training_data_raster.py \
  --tiles_geojson data/processed/tiles_raster.geojson \
  --fuel-metrics-raster data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \
  --outdir data/processed/training_data_chunks_raster \
  --naip_stac_source data/stac/naip/catalog.json \
  --uavsar_stac_source data/stac/uavsar/catalog.json \
  --start_date 2014-01-01 \
  --end_date 2025-12-31 \
  --chunk_size 100 \
  --threads 8 \
  --skip-uav-lidar

# Combine H5 chunks into single .pt file
echo "Step 3: Combine training data chunks..."
python src/data_prep/h5_chunk_loader.py \
  --input_dir data/processed/training_data_chunks_raster \
  --output_path data/processed/model_data_raster/combined_training_data_raster.pt

# Validate combined training data
echo "Step 3.5: Validate training data quality..."
python src/data_prep/validate_raster_training_data.py \
  --input data/processed/model_data_raster/combined_training_data_raster.pt \
  --fuel-metrics-raster data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \
  --output-dir data/processed/model_data_raster/validation_report \
  --max-na-ratio 0.5 \
  --min-dep-points 50 \
  --verbose

# Normalize, split 90/10, and precompute
echo "Step 4: Precompute and split training data..."
python src/data_prep/train_test_split_and_precompute_raster.py \
  --pt-file data/processed/model_data_raster/combined_training_data_raster.pt \
  --output-dir data/processed/model_data_raster \
  --train-val-ratio 0.9 \
  --min-dep-points 50 \
  --max-na-ratio 0.5 \
  --random-seed 42 \
  --precision 32 \
  --no-log-tfl

# Data augmentation
echo "Step 5: Data augmentation..."
python src/data_prep/data_augmentation_raster.py \
  --training_tiles data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt \
  --output_path data/processed/model_data_raster/augmented_tiles_raster_32bit.pt \
  --n_augmentations 2

echo ""
echo "Pipeline complete! Training data ready in data/processed/model_data_raster/"

