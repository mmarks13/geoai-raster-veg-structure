#!/bin/bash
# Process forest plot tiles (no UAV LiDAR ground truth)
# These tiles will be used for model validation against fuel metrics

python src/data_prep/generate_training_data.py \
  --tiles_geojson data/processed/forest_plot_tiles.geojson \
  --outdir data/processed/forest_plot_data_chunks \
  --skip-uav-lidar \
  --uavsar_stac_source data/stac/uavsar/catalog.json \
  --naip_stac_source data/stac/naip/catalog.json \
  --start_date 2016-01-01 \
  --end_date 2025-12-27 \
  --chunk_size 100 \
  --threads 8 \
  --max-api-retries 20 \
  --initial-voxel-size-cm 4 \
  --max-points 20000 30000 40000 \
  --verbose
