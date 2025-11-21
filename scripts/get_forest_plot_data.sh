#!/bin/bash
# Download NAIP, UAVSAR, and 3DEP data for forest plot sites
# This script appends to existing STAC catalogs without reprocessing existing data

cd /home/jovyan/geoai_veg_map/

# Set environment variables for UAVSAR authentication
export EARTHDATA_USERNAME=mmarks13
export EARTHDATA_PASSWORD=vuj@zmp2CQX5bkp2kbd

# Date range (covers all forest plot years)
START_DATE="2014-01-01"
END_DATE="2025-12-31"

# Output directories (same as existing catalogs to enable appending)
NAIP_OUT="/home/jovyan/geoai_veg_map/data/stac/naip"
UAVSAR_OUT="/home/jovyan/geoai_veg_map/data/stac/uavsar"
UAVSAR_TEMP="data/raw/uavsar"
THREEDEP_OUT="/home/jovyan/geoai_veg_map/data/stac/3dep"

echo "================================================================================"
echo "DOWNLOADING DATA FOR FOREST PLOT SITES"
echo "================================================================================"
echo "Start time: $(date)"
echo ""
echo "Sites to process: BluffMesa, Laguna, NorthBigBear, ReyesPeak, TecuyaRidge"
echo "Data types: NAIP imagery, UAVSAR SAR, 3DEP LiDAR"
echo ""
echo "This will APPEND to existing STAC catalogs (no reprocessing of existing data)"
echo "================================================================================"
echo ""

# -----------------------------------------------------------------------------
# Site 1: BluffMesa
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "[1/5] PROCESSING BLUFFMESA"
echo "================================================================================"
echo "Bbox: -116.959147 34.215288 -116.951819 34.222152"
echo "Plots: 10 | Area: 32.91 ha"
echo ""

echo ">>> Downloading NAIP imagery..."
python src/data_prep/make_local_naip_stac.py \
  --bbox -116.959147 34.215288 -116.951819 34.222152 \
  --start $START_DATE --end $END_DATE \
  --output $NAIP_OUT

echo ""
echo ">>> Downloading UAVSAR data..."
python src/data_prep/make_local_uavsar_stac.py \
  --bbox -116.959147 34.215288 -116.951819 34.222152 \
  --start $START_DATE --end 2024-12-31 \
  --output $UAVSAR_OUT --temp $UAVSAR_TEMP

echo ""
echo ">>> Downloading 3DEP LiDAR..."
python src/data_prep/make_local_3dep_stac.py \
  --bbox -116.959147 34.215288 -116.951819 34.222152 \
  --start $START_DATE --end $END_DATE \
  --output $THREEDEP_OUT

echo ""
echo "✓ BluffMesa complete"
echo ""

# -----------------------------------------------------------------------------
# Site 2: Laguna
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "[2/5] PROCESSING LAGUNA"
echo "================================================================================"
echo "Bbox: -116.438054 32.844520 -116.424266 32.862186"
echo "Plots: 64 | Area: 157.41 ha"
echo ""

echo ">>> Downloading NAIP imagery..."
python src/data_prep/make_local_naip_stac.py \
  --bbox -116.438054 32.844520 -116.424266 32.862186 \
  --start $START_DATE --end $END_DATE \
  --output $NAIP_OUT

echo ""
echo ">>> Downloading UAVSAR data..."
python src/data_prep/make_local_uavsar_stac.py \
  --bbox -116.438054 32.844520 -116.424266 32.862186 \
  --start $START_DATE --end 2024-12-31 \
  --output $UAVSAR_OUT --temp $UAVSAR_TEMP

echo ""
echo ">>> Downloading 3DEP LiDAR..."
python src/data_prep/make_local_3dep_stac.py \
  --bbox -116.438054 32.844520 -116.424266 32.862186 \
  --start $START_DATE --end $END_DATE \
  --output $THREEDEP_OUT

echo ""
echo "✓ Laguna complete"
echo ""

# -----------------------------------------------------------------------------
# Site 3: NorthBigBear
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "[3/5] PROCESSING NORTHBIGBEAR"
echo "================================================================================"
echo "Bbox: -116.937281 34.287655 -116.918104 34.304225"
echo "Plots: 19 | Area: 171.45 ha"
echo ""

echo ">>> Downloading NAIP imagery..."
python src/data_prep/make_local_naip_stac.py \
  --bbox -116.937281 34.287655 -116.918104 34.304225 \
  --start $START_DATE --end $END_DATE \
  --output $NAIP_OUT

echo ""
echo ">>> Downloading UAVSAR data..."
python src/data_prep/make_local_uavsar_stac.py \
  --bbox -116.937281 34.287655 -116.918104 34.304225 \
  --start $START_DATE --end 2024-12-31 \
  --output $UAVSAR_OUT --temp $UAVSAR_TEMP

echo ""
echo ">>> Downloading 3DEP LiDAR..."
python src/data_prep/make_local_3dep_stac.py \
  --bbox -116.937281 34.287655 -116.918104 34.304225 \
  --start $START_DATE --end $END_DATE \
  --output $THREEDEP_OUT

echo ""
echo "✓ NorthBigBear complete"
echo ""

# -----------------------------------------------------------------------------
# Site 4: ReyesPeak
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "[4/5] PROCESSING REYESPEAK"
echo "================================================================================"
echo "Bbox: -119.341359 34.632541 -119.282359 34.643225"
echo "Plots: 21 | Area: 199.48 ha"
echo ""

echo ">>> Downloading NAIP imagery..."
python src/data_prep/make_local_naip_stac.py \
  --bbox -119.341359 34.632541 -119.282359 34.643225 \
  --start $START_DATE --end $END_DATE \
  --output $NAIP_OUT

echo ""
echo ">>> Downloading UAVSAR data..."
python src/data_prep/make_local_uavsar_stac.py \
  --bbox -119.341359 34.632541 -119.282359 34.643225 \
  --start $START_DATE --end 2024-12-31 \
  --output $UAVSAR_OUT --temp $UAVSAR_TEMP

echo ""
echo ">>> Downloading 3DEP LiDAR..."
python src/data_prep/make_local_3dep_stac.py \
  --bbox -119.341359 34.632541 -119.282359 34.643225 \
  --start $START_DATE --end $END_DATE \
  --output $THREEDEP_OUT

echo ""
echo "✓ ReyesPeak complete"
echo ""

# -----------------------------------------------------------------------------
# Site 5: TecuyaRidge
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "[5/5] PROCESSING TECUYARIDGE"
echo "================================================================================"
echo "Bbox: -119.025374 34.840597 -118.967664 34.848034"
echo "Plots: 21 | Area: 297.57 ha"
echo ""

echo ">>> Downloading NAIP imagery..."
python src/data_prep/make_local_naip_stac.py \
  --bbox -119.025374 34.840597 -118.967664 34.848034 \
  --start $START_DATE --end $END_DATE \
  --output $NAIP_OUT

echo ""
echo ">>> Downloading UAVSAR data..."
python src/data_prep/make_local_uavsar_stac.py \
  --bbox -119.025374 34.840597 -118.967664 34.848034 \
  --start $START_DATE --end 2024-12-31 \
  --output $UAVSAR_OUT --temp $UAVSAR_TEMP

echo ""
echo ">>> Downloading 3DEP LiDAR..."
python src/data_prep/make_local_3dep_stac.py \
  --bbox -119.025374 34.840597 -118.967664 34.848034 \
  --start $START_DATE --end $END_DATE \
  --output $THREEDEP_OUT

echo ""
echo "✓ TecuyaRidge complete"
echo ""

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
echo "================================================================================"
echo "✓ ALL FOREST PLOT SITES PROCESSED"
echo "================================================================================"
echo "End time: $(date)"
echo ""
echo "Sites completed: 5/5"
echo "  - BluffMesa (10 plots, 32.91 ha)"
echo "  - Laguna (64 plots, 157.41 ha)"
echo "  - NorthBigBear (19 plots, 171.45 ha)"
echo "  - ReyesPeak (21 plots, 199.48 ha)"
echo "  - TecuyaRidge (21 plots, 297.57 ha)"
echo ""
echo "Data downloaded to:"
echo "  - NAIP: $NAIP_OUT"
echo "  - UAVSAR: $UAVSAR_OUT"
echo "  - 3DEP: $THREEDEP_OUT"
echo ""
echo "Next steps:"
echo "  1. Verify STAC catalog item counts"
echo "  2. Check spatial coverage in QGIS"
echo "  3. Run tile generation for new sites"
echo "================================================================================"
