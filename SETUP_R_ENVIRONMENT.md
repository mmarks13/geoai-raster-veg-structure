# R Environment Setup for Fuel Metrics

## Summary

A separate conda environment (`r_fuel_metrics`) has been created for R-based fuel metrics processing using LidarForFuel. This prevents conflicts between R and Python packages in the main `geoai_env`.

## What Was Done

### 1. Created Separate R Environment
- **Environment file**: `environment_r_fuel_metrics.yml`
- **Environment name**: `r_fuel_metrics`
- **R version**: 4.5.1
- **Key packages**: r-lidr, r-terra, r-sf, r-raster, r-remotes

### 2. Cleaned Up Main Environment
- Removed all R packages from `geoai_env`
- Updated `environment.yml` to remove R dependencies
- Verified clean separation between Python and R stacks

### 3. Updated Python Wrapper
- Modified `src/data_prep/lidarforfuel_wrapper.py` to use `conda run`
- Automatic environment switching (no manual activation required)
- Backwards compatible with PATH-based Rscript

### 4. Updated Documentation
- `data/processed/fuel_metrics/README.md`: New installation instructions
- `CLAUDE.md`: Dual-environment workflow notes

## Manual Installation Step Required

The LidarForFuel package installation requires manual completion due to dependency complexity:

```bash
# 1. Activate the R environment
conda activate r_fuel_metrics

# 2. Remove any lock files
rm -rf /opt/conda/envs/r_fuel_metrics/lib/R/library/00LOCK-*

# 3. Install dependencies (may take 5-10 minutes)
R -e "install.packages(c('Rfast', 'R.matlab'), repos='https://cloud.r-project.org')"

# 4. Install VoxR from GitHub
R -e "remotes::install_github('Blundeman/VoxR')"

# 5. Install LidarForFuel
R -e "remotes::install_github('oliviermartin7/lidarforfuel')"

# 6. Verify installation
R -e "library(lidarforfuel)"
```

## Usage

### From Python (Automatic)
The wrapper automatically uses the `r_fuel_metrics` environment:

```python
from src.data_prep.lidarforfuel_wrapper import process_point_cloud

# No need to manually activate r_fuel_metrics
process_point_cloud(
    input_las=Path("data/raw/uavlidar/study_las/file.las"),
    output_dir=Path("data/processed/fuel_metrics/volcan"),
    species="Mixed",
    resolution=1.0
)
```

### From Command Line
```bash
# Option 1: Manual activation
conda activate r_fuel_metrics
python src/data_prep/process_uav_fuel_metrics.py --input file.las

# Option 2: Let wrapper handle it (from any environment)
conda activate geoai_env
python src/data_prep/process_uav_fuel_metrics.py --input file.las
```

## Testing

Once LidarForFuel is installed, test with:

```bash
conda activate r_fuel_metrics

# Test R scripts directly
Rscript scripts/r/run_pretreatment.R --help

# Test Python wrapper
python src/data_prep/process_uav_fuel_metrics.py --list_species
```

## Benefits of This Approach

✅ **Clean separation**: R won't interfere with PyTorch/CUDA  
✅ **No conflicts**: Python and R packages isolated  
✅ **Automatic switching**: Wrapper handles environment activation  
✅ **Smaller environments**: Each contains only what it needs  
✅ **Better reproducibility**: Clear dependency boundaries  

## Troubleshooting

### "Rscript not found"
Ensure `r_fuel_metrics` environment exists:
```bash
conda env list | grep r_fuel_metrics
```

### Package installation fails
Remove lock files and try again:
```bash
conda activate r_fuel_metrics
rm -rf /opt/conda/envs/r_fuel_metrics/lib/R/library/00LOCK-*
```

### Wrapper can't find environment
Check that `conda run` works:
```bash
conda run -n r_fuel_metrics Rscript --version
```

## Files Created/Modified

**Created:**
- `environment_r_fuel_metrics.yml` - R environment specification
- `SETUP_R_ENVIRONMENT.md` - This file

**Modified:**
- `environment.yml` - Removed R packages
- `src/data_prep/lidarforfuel_wrapper.py` - Added conda-aware execution
- `data/processed/fuel_metrics/README.md` - Updated installation instructions
- `CLAUDE.md` - Added dual-environment documentation

## Next Steps

1. Complete LidarForFuel installation (see above)
2. Test the fuel metrics pipeline
3. Commit changes to version control

---

**Date**: 2025-10-19  
**Status**: Environment created, LidarForFuel installation pending manual completion
