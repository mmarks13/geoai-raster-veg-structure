#!/bin/bash
# Process all veg structure metrics sites
# Usage: bash scripts/veg_structure_metrics/run_all_sites.sh

set -e
SCRIPT_DIR="$(dirname "$0")"

echo "=== Vegetation Structure Metrics Pipeline ==="
echo ""

# Process small sites (whole file, max_hag=25 for shorter vegetation)
echo "Processing t01_t09..."
bash "$SCRIPT_DIR/process_single_site.sh" \
    "data/raw/uavlidar/study_las/T01-T09_LiDAR_20230928_Pre_LAS.las" \
    "t01_t09" \
    25

echo "Processing t03_t13..."
bash "$SCRIPT_DIR/process_single_site.sh" \
    "data/raw/uavlidar/study_las/T03-T13_LIDAR_20231025_Pre_LAS.las" \
    "t03_t13" \
    25

echo "Processing t06_t14..."
bash "$SCRIPT_DIR/process_single_site.sh" \
    "data/raw/uavlidar/study_las/T06-T14_LIDAR_20231025_Pre_LAS.las" \
    "t06_t14" \
    25

echo "Processing trex..."
bash "$SCRIPT_DIR/process_single_site.sh" \
    "data/raw/uavlidar/study_las/TREX_LIDAR_20230630_Pre_LAS.las" \
    "trex" \
    25

echo "Processing volcan_mtn..."
bash scripts/veg_structure_metrics/process_single_site.sh \
    "data/raw/uavlidar/full_volcan_mtn_las/VolcanMt_20231025_wHAG.laz" \
    "volcan_mtn" \
    60

echo ""
echo "=== All sites complete ==="
