#!/bin/bash
# install_lidarforfuel.sh
# Reproducible installation script for LidarForFuel R package and dependencies
#
# Usage:
#   bash scripts/install_lidarforfuel.sh
#
# Requirements:
#   - r_fuel_metrics conda environment must exist (create with: conda env create -f scripts/fuel_metrics/environment_r_fuel_metrics.yml)
#   - Internet connection for downloading packages

set -e  # Exit on error

CONDA_ENV="r_fuel_metrics"
LOG_FILE="lidarforfuel_install_$(date +%Y%m%d_%H%M%S).log"

echo "========================================================================"
echo "LidarForFuel Installation Script"
echo "========================================================================"
echo ""
echo "Target environment: $CONDA_ENV"
echo "Log file: $LOG_FILE"
echo ""

# Function to run R commands in the conda environment
run_r() {
    local cmd="$1"
    echo "Running: $cmd" | tee -a "$LOG_FILE"
    conda run -n "$CONDA_ENV" R --quiet --vanilla -e "$cmd" 2>&1 | tee -a "$LOG_FILE"
    return ${PIPESTATUS[0]}
}

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "ERROR: conda not found in PATH"
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "^$CONDA_ENV "; then
    echo "ERROR: Conda environment '$CONDA_ENV' not found"
    echo ""
    echo "Please create it first:"
    echo "  conda env create -f scripts/fuel_metrics/environment_r_fuel_metrics.yml"
    exit 1
fi

# Step 1: Verify base packages
echo "[1/4] Checking conda-installed base packages..."
echo "------------------------------------------------------------------------"
run_r "
pkgs <- c('lidR', 'terra', 'sf', 'remotes')
all_ok <- TRUE
for (pkg in pkgs) {
  if (require(pkg, character.only=TRUE, quietly=TRUE)) {
    cat(sprintf('  ✓ %s %s\n', pkg, packageVersion(pkg)))
  } else {
    cat(sprintf('  ✗ %s MISSING\n', pkg))
    all_ok <- FALSE
  }
}
if (!all_ok) {
  cat('\nERROR: Base packages missing. Recreate environment:\n')
  cat('  conda env remove -n r_fuel_metrics\n')
  cat('  conda env create -f scripts/fuel_metrics/environment_r_fuel_metrics.yml\n')
  quit(status=1)
}
"

# Step 2: Clean lock files
echo ""
echo "[2/4] Removing package lock files..."
echo "------------------------------------------------------------------------"
conda run -n "$CONDA_ENV" bash -c "rm -rf \$CONDA_PREFIX/lib/R/library/00LOCK-* && echo '  ✓ Lock files removed'" || echo "  (No lock files to remove)"

# Step 3: Install Rfast
echo ""
echo "[3/4] Installing Rfast from CRAN..."
echo "------------------------------------------------------------------------"
echo "  NOTE: Source compilation may take 5-10 minutes"
run_r "
options(repos = c(CRAN = 'https://cloud.r-project.org'))
if (!require('Rfast', quietly=TRUE)) {
  cat('  Installing Rfast (this will take several minutes)...\n')
  install.packages('Rfast', Ncpus = parallel::detectCores())
  library(Rfast)
  cat('  ✓ Rfast', as.character(packageVersion('Rfast')), 'installed\n')
} else {
  cat('  ✓ Rfast', as.character(packageVersion('Rfast')), 'already installed\n')
}
"

# Step 4: Install lidarforfuel (VoxR will be auto-installed as dependency)
echo ""
echo "[4/4] Installing lidarforfuel from GitHub..."
echo "------------------------------------------------------------------------"
echo "  NOTE: VoxR will be automatically installed from CRAN as a dependency"
run_r "
if (!require('lidarforfuel', quietly=TRUE)) {
  cat('  Installing lidarforfuel from GitHub...\n')
  cat('  (VoxR will be automatically installed from CRAN)\n')
  remotes::install_github('oliviermartin7/lidarforfuel', upgrade='never', quiet=FALSE)
  library(lidarforfuel)
  cat('  ✓ lidarforfuel', as.character(packageVersion('lidarforfuel')), 'installed\n')
  cat('  ✓ VoxR', as.character(packageVersion('VoxR')), 'auto-installed\n')
} else {
  cat('  ✓ lidarforfuel', as.character(packageVersion('lidarforfuel')), 'already installed\n')
}
"

# Final verification
echo ""
echo "========================================================================"
echo "Verifying Installation"
echo "========================================================================"
run_r "
required_pkgs <- c('Rfast', 'VoxR', 'lidarforfuel')
all_ok <- TRUE
for (pkg in required_pkgs) {
  if (require(pkg, character.only=TRUE, quietly=TRUE)) {
    cat(sprintf('✓ %-15s %s\n', pkg, packageVersion(pkg)))
  } else {
    cat(sprintf('✗ %-15s FAILED\n', pkg))
    all_ok <- FALSE
  }
}
if (!all_ok) {
  cat('\nERROR: Some packages failed to install. Check log file.\n')
  quit(status=1)
}
"


echo ""
echo "========================================================================"
echo "Checking R Wrapper Scripts"
echo "========================================================================"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/r/run_pretreatment.R" ]; then
    echo "✓ run_pretreatment.R found"
    chmod +x "$SCRIPT_DIR/r/run_pretreatment.R"
else
    echo "✗ run_pretreatment.R not found at: $SCRIPT_DIR/r/"
    exit 1
fi

if [ -f "$SCRIPT_DIR/r/run_fuel_metrics.R" ]; then
    echo "✓ run_fuel_metrics.R found"
    chmod +x "$SCRIPT_DIR/r/run_fuel_metrics.R"
else
    echo "✗ run_fuel_metrics.R not found at: $SCRIPT_DIR/r/"
    exit 1
fi

echo ""
echo "========================================================================"
echo "Installation Complete!"
echo "========================================================================"
echo ""
echo "✓ All packages installed successfully in conda environment: $CONDA_ENV"
echo "✓ Log saved to: $LOG_FILE"
echo ""
echo "Installed packages:"
echo "  • Rfast $(conda run -n $CONDA_ENV R --quiet --vanilla -e "cat(as.character(packageVersion('Rfast')))" 2>/dev/null || echo 'check log')"
echo "  • VoxR $(conda run -n $CONDA_ENV R --quiet --vanilla -e "cat(as.character(packageVersion('VoxR')))" 2>/dev/null || echo 'check log')"
echo "  • lidarforfuel $(conda run -n $CONDA_ENV R --quiet --vanilla -e "cat(as.character(packageVersion('lidarforfuel')))" 2>/dev/null || echo 'check log')"
echo ""
echo "Next steps:"
echo "  1. Test R scripts directly:"
echo "     conda run -n r_fuel_metrics Rscript scripts/fuel_metrics/r/run_pretreatment.R \\"
echo "       input.las output.laz 140 591 130 550"
echo ""
echo "  2. Test Python wrapper (auto-switches environments):"
echo "     python src/fuel_metrics/process_fuel_metrics.py --list_species"
echo ""
echo "  3. Process UAV LiDAR for fuel metrics:"
echo "     python src/fuel_metrics/process_fuel_metrics.py \\"
echo "         --input data/raw/uavlidar/study_las/your_file.las"
echo ""
echo "  4. Run complete pipeline:"
echo "     bash scripts/fuel_metrics/run_fuel_metrics_pipeline.sh \\"
echo "         --input data/raw/uavlidar/my_site.las --output-name my_site"
echo ""
echo "  5. Read full documentation:"
echo "     cat data/processed/fuel_metrics/README.md"
echo ""
echo "Note: Files with only ground points will return NULL from fPCpretreatment."
echo "      Use LAS files containing vegetation classification for fuel metrics."
echo ""
