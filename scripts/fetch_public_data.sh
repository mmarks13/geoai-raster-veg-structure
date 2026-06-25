#!/usr/bin/env bash
# Fetch publicly-available source data for this repo.
#
# Currently documents the Volcan Mountain UAV LiDAR dataset, which is now
# hosted on OpenTopography. Other public sources (NAIP, USGS 3DEP) are
# fetched by the existing pipeline scripts under scripts/ (get_data.sh,
# process_3dep_hag_features.sh).
#
# This script is a placeholder. Wire in the actual download once an
# OpenTopography API key is configured (or use the OT web portal to
# download into data/raw/uavlidar/volcan_mtn/ manually).
set -euo pipefail

cd "$(dirname "$0")/.."

mkdir -p data/raw/uavlidar/volcan_mtn

cat <<'EOF'
Volcan Mountain UAV LiDAR
-------------------------
Hosted at: https://portal.opentopography.org/datasetMetadata?otCollectionID=OT.042026.32611.1

To populate locally:

  1. Manual:
     Visit the URL above, download the LAS/LAZ files via the OT web portal,
     and place them under: data/raw/uavlidar/volcan_mtn/

  2. Programmatic (requires OT API key):
     Set OT_API_KEY in your environment, then use OT's bulk download endpoint.
     Example (uncomment and adapt once you have the exact endpoint URL):

       # curl -L \
       #   -H "X-API-Key: $OT_API_KEY" \
       #   -o data/raw/uavlidar/volcan_mtn/volcan_mtn.laz \
       #   "https://portal.opentopography.org/API/getLidarFromCollection?id=OT.042026.32611.1"

After populating, run scripts/process_3dep_hag_features.sh (which also
processes the UAV LiDAR through the SMRF + HAG pipeline, modulo paths).
EOF
