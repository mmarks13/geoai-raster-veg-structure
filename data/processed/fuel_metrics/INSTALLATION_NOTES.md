# LidarForFuel Installation Notes

## Current Status

The LidarForFuel integration has been fully implemented with:
- ✅ Python-R interface ([src/data_prep/lidarforfuel_wrapper.py](../../../src/data_prep/lidarforfuel_wrapper.py))
- ✅ R wrapper scripts ([scripts/r/run_pretreatment.R](../../../scripts/r/run_pretreatment.R), [scripts/r/run_fuel_metrics.R](../../../scripts/r/run_fuel_metrics.R))
- ✅ Main orchestration script ([src/data_prep/process_uav_fuel_metrics.py](../../../src/data_prep/process_uav_fuel_metrics.py))
- ✅ Trait lookup table ([trait_lookup.csv](trait_lookup.csv))
- ✅ Comprehensive documentation ([README.md](README.md))

**Installation Status:** Pending due to environment dependency conflicts (see below).

---

## Installation Challenges Encountered

### Issue: BLAS Conflict

The existing conda environment has a BLAS conflict that prevents installing R spatial packages via conda:

```
PackagesNotFoundError: The following packages are not available from current channels:
  - blas*.*
```

This occurs because the environment already has MKL BLAS (for PyTorch/NumPy) and conda-forge r-lidr requires a different BLAS implementation.

### Issue: R Package Compilation

When installing R packages from source (via `install.packages()` in R), several spatial packages fail due to missing system libraries:
- `s2`: Requires Abseil (C++ library)
- `units`: Requires `libudunits2`
- `terra`: Requires `gdal-config`
- `sf`: Requires GDAL, GEOS, PROJ

While `udunits2` and `gdal` can be installed via conda, the compilation process is very time-consuming (10+ minutes per package).

---

## Recommended Installation Approaches

### Option 1: Separate Conda Environment (Recommended)

Create a dedicated R environment for fuel metrics:

```bash
# Create separate R environment
conda create -n r_fuel_metrics -c conda-forge \
    r-base r-lidr r-terra r-sf r-remotes python

# Activate and install LidarForFuel
conda activate r_fuel_metrics
R -e "remotes::install_github('oliviermartin7/lidarforfuel')"

# Run fuel metrics from this environment
python src/data_prep/process_uav_fuel_metrics.py --input file.las
```

**Pros:**
- Clean installation without conflicts
- Pre-compiled packages (fast)
- Isolated from main PyTorch environment

**Cons:**
- Need to switch environments
- Duplicate Python installation

### Option 2: Docker Container

Use a pre-configured Docker image with R and Python:

```bash
# Create Dockerfile
FROM rocker/geospatial:latest

# Install Python and LidarForFuel
RUN apt-get update && apt-get install -y python3-pip
RUN R -e "remotes::install_github('oliviermartin7/lidarforfuel')"

# Copy scripts
COPY src /app/src
COPY scripts /app/scripts
COPY data/processed/fuel_metrics/trait_lookup.csv /app/data/processed/fuel_metrics/

# Run
WORKDIR /app
CMD ["python3", "src/data_prep/process_uav_fuel_metrics.py"]
```

**Pros:**
- Reproducible environment
- No conda conflicts
- Easy sharing/deployment

**Cons:**
- Requires Docker
- File I/O overhead

### Option 3: Manual Compilation (Advanced)

If you must use the existing environment:

```bash
# Install system dependencies
conda install -c conda-forge udunits2 gdal proj geos

# Install R packages from source (slow, 20-30 minutes)
R -e "install.packages(c('sf', 'terra', 'lidR'), repos='https://cloud.r-project.org')"

# Install LidarForFuel
R -e "remotes::install_github('oliviermartin7/lidarforfuel')"
```

**Pros:**
- Single environment

**Cons:**
- Very slow compilation
- May still fail due to BLAS conflicts
- Fragile (breaks with environment updates)

---

## Testing Status

**Pipeline Implementation:** ✅ Complete and ready
**R Installation:** ⏳ Pending due to environment issues
**Functional Test:** ⏳ Blocked by installation

### What Has Been Tested:
- ✅ Python wrapper scripts (syntax, imports, error handling)
- ✅ R script structure (reviewed for correctness)
- ✅ Trait lookup table (validated data)
- ✅ Documentation completeness

### What Needs Testing (Once R is Installed):
- [ ] Run pretreatment on test file ([data/raw/uavlidar/study_las/20241025_151528.las](../../../data/raw/uavlidar/study_las/20241025_151528.las))
- [ ] Verify pretreated LAZ attributes (LMA, WD, Zref)
- [ ] Run fuel metrics computation
- [ ] Verify 173-band output raster
- [ ] Validate metric ranges (CBH, fuel loads, cover)
- [ ] Performance benchmarking

---

## Quick Test Once R is Installed

```bash
# Verify R and lidR installation
Rscript -e "library(lidR); library(lidarforfuel); cat('✓ Packages loaded\n')"

# Test pretreatment wrapper
Rscript scripts/r/run_pretreatment.R \
    data/raw/uavlidar/study_las/20241025_151528.las \
    data/processed/fuel_metrics/volcan/pretreated/test_pretreated.laz \
    140 591 130 550

# Test full pipeline
python src/data_prep/process_uav_fuel_metrics.py \
    --input data/raw/uavlidar/study_las/20241025_151528.las \
    --species "Mixed" \
    --resolution 1.0

# Verify outputs
ls -lh data/processed/fuel_metrics/volcan/pretreated/
ls -lh data/processed/fuel_metrics/volcan/rasters/
gdalinfo data/processed/fuel_metrics/volcan/rasters/*_fuel_metrics.tif
```

---

## Workaround for Testing (If Installation Blocked)

If you need to test the concept without full installation:

1. **Mock Test** - Verify Python logic without R:
   ```bash
   python -c "from src.data_prep.lidarforfuel_wrapper import load_trait_lookup, get_default_traits; \
              print(load_trait_lookup()); \
              print(get_default_traits())"
   ```

2. **Manual R Test** - Run R scripts directly (if R is installed):
   ```bash
   # Skip wrapper, test R directly
   R -e "source('scripts/r/run_pretreatment.R')"
   ```

3. **Alternative System** - Test on a different machine/cluster with R already configured

---

## Expected Timeline

Once installation issue is resolved:
- **Installation (Option 1):** 10-15 minutes
- **Test run (single file):** 3-6 minutes
- **Full validation:** 30-60 minutes

---

## Contact for Installation Support

For this specific environment:
- Check if a separate R environment is acceptable
- Consider using university HPC/cluster resources (often have R pre-installed)
- Docker option if reproducibility is priority

For LidarForFuel package issues:
- GitHub: https://github.com/oliviermartin7/LidarForFuel/issues

---

**Last Updated:** 2025-10-19
**Status:** Implementation complete, awaiting R environment resolution for testing
