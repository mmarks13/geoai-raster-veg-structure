#!/bin/bash
# install_lidarforfuel.sh
# Install R and LidarForFuel package in conda environment

set -e  # Exit on error

echo "========================================================================"
echo "Installing R and LidarForFuel Dependencies"
echo "========================================================================"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found in PATH"
    exit 1
fi

# Install R and core packages via conda
echo ""
echo "Step 1: Installing R and core packages via conda..."
echo "------------------------------------------------------------------------"
conda install -y -c conda-forge \
    r-base \
    r-lidr \
    r-remotes \
    r-sf \
    r-terra \
    r-raster \
    r-data.table \
    r-future \
    r-future.apply

# Verify R installation
echo ""
echo "Step 2: Verifying R installation..."
echo "------------------------------------------------------------------------"
if ! command -v Rscript &> /dev/null; then
    echo "Error: Rscript not found after installation"
    exit 1
fi

echo "R version:"
R --version | head -1

# Install LidarForFuel from GitHub
echo ""
echo "Step 3: Installing LidarForFuel from GitHub..."
echo "------------------------------------------------------------------------"
R -e "remotes::install_github('oliviermartin7/lidarforfuel', quiet = FALSE)"

# Verify LidarForFuel installation
echo ""
echo "Step 4: Verifying LidarForFuel installation..."
echo "------------------------------------------------------------------------"
R -e "if (require('lidarforfuel', quietly = TRUE)) { cat('✓ lidarforfuel installed successfully\n') } else { stop('lidarforfuel installation failed') }"

# Test R wrapper scripts exist
echo ""
echo "Step 5: Checking R wrapper scripts..."
echo "------------------------------------------------------------------------"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/r/run_pretreatment.R" ]; then
    echo "✓ run_pretreatment.R found"
else
    echo "✗ run_pretreatment.R not found"
    exit 1
fi

if [ -f "$SCRIPT_DIR/r/run_fuel_metrics.R" ]; then
    echo "✓ run_fuel_metrics.R found"
else
    echo "✗ run_fuel_metrics.R not found"
    exit 1
fi

# Make R scripts executable
chmod +x "$SCRIPT_DIR/r/run_pretreatment.R"
chmod +x "$SCRIPT_DIR/r/run_fuel_metrics.R"

echo ""
echo "========================================================================"
echo "Installation Complete!"
echo "========================================================================"
echo ""
echo "Next steps:"
echo "  1. Test on a small LAS file:"
echo "     python src/data_prep/process_uav_fuel_metrics.py \\"
echo "         --input data/raw/uavlidar/study_las/20241025_151528.las"
echo ""
echo "  2. List available species:"
echo "     python src/data_prep/process_uav_fuel_metrics.py --list_species"
echo ""
echo "  3. See full documentation:"
echo "     cat data/processed/fuel_metrics/README.md"
echo ""
