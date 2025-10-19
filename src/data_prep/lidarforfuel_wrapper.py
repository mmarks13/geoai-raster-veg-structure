"""
lidarforfuel_wrapper.py

Python-R interface for the LidarForFuel R package.
Provides Python functions to call R scripts for fuel metrics computation.

This module orchestrates:
1. fPCpretreatment: Point cloud preprocessing (normalization, trait attribution)
2. fCBDprofile_fuelmetrics: Fuel metrics computation (bulk density profiles, fire risk metrics)
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict
import csv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to R scripts
SCRIPTS_DIR = Path(__file__).parents[2] / "scripts" / "r"
PRETREAT_SCRIPT = SCRIPTS_DIR / "run_pretreatment.R"
METRICS_SCRIPT = SCRIPTS_DIR / "run_fuel_metrics.R"

# Path to trait lookup table
TRAIT_LOOKUP_PATH = Path(__file__).parents[2] / "data" / "processed" / "fuel_metrics" / "trait_lookup.csv"


class LidarForFuelError(Exception):
    """Custom exception for LidarForFuel wrapper errors."""
    pass


def load_trait_lookup(csv_path: Optional[Path] = None) -> Dict[str, Dict[str, float]]:
    """
    Load LMA and wood density values from trait lookup table.

    Args:
        csv_path: Path to trait lookup CSV. If None, uses default location.

    Returns:
        Dictionary mapping species names to trait dictionaries with keys:
        'lma_gm2', 'wd_kgm3', 'lma_understory_gm2', 'wd_understory_kgm3'

    Raises:
        FileNotFoundError: If trait lookup file doesn't exist
        ValueError: If CSV is malformed
    """
    if csv_path is None:
        csv_path = TRAIT_LOOKUP_PATH

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Trait lookup table not found: {csv_path}\n"
            f"Expected at: {TRAIT_LOOKUP_PATH}"
        )

    traits = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            species = row['species']
            traits[species] = {
                'lma_gm2': float(row['lma_gm2']),
                'wd_kgm3': float(row['wd_kgm3']),
                'lma_understory_gm2': float(row['lma_understory_gm2']),
                'wd_understory_kgm3': float(row['wd_understory_kgm3']),
                'common_name': row['common_name'],
                'notes': row.get('notes', '')
            }

    logger.info(f"Loaded trait data for {len(traits)} species/categories")
    return traits


def get_default_traits() -> Tuple[float, float, float, float]:
    """
    Get default LMA and wood density values.

    Returns:
        Tuple of (lma, wd, lma_bush, wd_bush) as floats

    Uses the 'Default' entry from trait lookup table, or hardcoded
    LidarForFuel Mediterranean defaults if lookup unavailable.
    """
    try:
        traits = load_trait_lookup()
        default = traits.get('Default', traits.get('Mixed'))
        if default:
            return (
                default['lma_gm2'],
                default['wd_kgm3'],
                default['lma_understory_gm2'],
                default['wd_understory_kgm3']
            )
    except FileNotFoundError:
        logger.warning("Trait lookup not found, using hardcoded defaults")

    # Fallback to LidarForFuel Mediterranean defaults
    return (140.0, 591.0, 140.0, 591.0)


def check_rscript_available(r_env_name: str = "r_fuel_metrics") -> bool:
    """
    Check if Rscript is available in the specified conda environment.

    Args:
        r_env_name: Name of the conda environment containing R (default: r_fuel_metrics)

    Returns:
        True if Rscript found, False otherwise
    """
    try:
        # First try conda run
        result = subprocess.run(
            ['conda', 'run', '-n', r_env_name, 'Rscript', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.debug(f"Found Rscript in conda environment: {r_env_name}")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Fallback to PATH (for backwards compatibility)
    try:
        result = subprocess.run(
            ['Rscript', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.debug("Found Rscript in PATH")
            return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    return False


def run_pretreatment(
    input_las: Path,
    output_laz: Path,
    lma: float,
    wd: float,
    lma_bush: float,
    wd_bush: float,
    h_strata_bush: float = 2.0,
    height_filter: float = 60.0,
    classify: bool = False,
    timeout: int = 3600,
    r_env_name: str = "r_fuel_metrics"
) -> None:
    """
    Run LidarForFuel fPCpretreatment on a LAS/LAZ file.

    Preprocessing steps:
    1. Normalize point cloud (height above ground)
    2. Add LMA (Leaf Mass Area) attribute
    3. Add WD (Wood Density) attribute
    4. Add spatial coordinates (Easting, Northing, Elevation)
    5. Preserve original Z coordinate (Zref)
    6. Optionally classify ground points

    Args:
        input_las: Path to input LAS/LAZ file
        output_laz: Path to output pretreated LAZ file
        lma: Leaf Mass Area (g/m²) for canopy
        wd: Wood Density (kg/m³) for canopy
        lma_bush: Leaf Mass Area (g/m²) for understory (<h_strata_bush)
        wd_bush: Wood Density (kg/m³) for understory
        h_strata_bush: Height threshold (m) for understory (default: 2.0)
        height_filter: Maximum height (m) to retain (default: 60.0)
        classify: Whether to classify ground points (default: False)
        timeout: Maximum execution time in seconds (default: 3600)
        r_env_name: Name of conda environment containing R (default: r_fuel_metrics)

    Raises:
        FileNotFoundError: If input file or Rscript not found
        LidarForFuelError: If R script fails
        subprocess.TimeoutExpired: If execution exceeds timeout
    """
    # Validate inputs
    if not input_las.exists():
        raise FileNotFoundError(f"Input LAS file not found: {input_las}")

    if not check_rscript_available(r_env_name):
        raise FileNotFoundError(
            f"Rscript not found in conda environment '{r_env_name}'. Please ensure the environment exists:\n"
            f"  conda env create -f environment_r_fuel_metrics.yml\n"
            f"  conda activate {r_env_name}\n"
            f"  R -e \"remotes::install_github('oliviermartin7/lidarforfuel')\""
        )

    if not PRETREAT_SCRIPT.exists():
        raise FileNotFoundError(f"R pretreatment script not found: {PRETREAT_SCRIPT}")

    # Ensure output directory exists
    output_laz.parent.mkdir(parents=True, exist_ok=True)

    # Build command (use conda run to execute in r_fuel_metrics environment)
    cmd = [
        'conda', 'run', '-n', r_env_name, 'Rscript',
        str(PRETREAT_SCRIPT),
        str(input_las),
        str(output_laz),
        str(lma),
        str(wd),
        str(lma_bush),
        str(wd_bush),
        str(h_strata_bush),
        str(height_filter),
        str(classify).upper()  # R expects TRUE/FALSE
    ]

    logger.info(f"Running fPCpretreatment on {input_las.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    # Execute R script
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  R: {line}")

        # Check for errors
        if result.returncode != 0:
            error_msg = f"fPCpretreatment failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr:\n{result.stderr}"
            raise LidarForFuelError(error_msg)

        # Verify output exists
        if not output_laz.exists():
            raise LidarForFuelError(f"Output file not created: {output_laz}")

        logger.info(f"Pretreatment completed: {output_laz}")

    except subprocess.TimeoutExpired:
        logger.error(f"Pretreatment timed out after {timeout} seconds")
        raise


def run_fuel_metrics(
    input_laz: Path,
    output_tif: Path,
    resolution: float = 1.0,
    layer_depth: float = 1.0,
    height_cover: float = 2.0,
    threshold: float = 0.02,
    export_mode: str = "full",
    timeout: int = 7200,
    r_env_name: str = "r_fuel_metrics"
) -> None:
    """
    Run LidarForFuel fCBDprofile_fuelmetrics to compute fuel metrics.

    Computes 173-band fuel metrics raster including:
    - Summary metrics (23 bands): canopy height, CBH, fuel loads, cover, etc.
    - Bulk density profile (150 bands): vertical distribution by layer

    Args:
        input_laz: Path to pretreated LAZ file (from run_pretreatment)
        output_tif: Path to output GeoTIFF raster
        resolution: Output raster resolution in meters (default: 1.0)
        layer_depth: Vertical layer depth for bulk density profile (default: 1.0)
        height_cover: Height threshold for cover computation (default: 2.0)
        threshold: Bulk density threshold for strata detection (default: 0.02)
        export_mode: 'full' (173 bands) or 'summary' (23 bands) (default: 'full')
        timeout: Maximum execution time in seconds (default: 7200)
        r_env_name: Name of conda environment containing R (default: r_fuel_metrics)

    Raises:
        FileNotFoundError: If input file or Rscript not found
        LidarForFuelError: If R script fails
        subprocess.TimeoutExpired: If execution exceeds timeout
    """
    # Validate inputs
    if not input_laz.exists():
        raise FileNotFoundError(f"Input LAZ file not found: {input_laz}")

    if not check_rscript_available(r_env_name):
        raise FileNotFoundError(f"Rscript not found in conda environment '{r_env_name}'")

    if not METRICS_SCRIPT.exists():
        raise FileNotFoundError(f"R metrics script not found: {METRICS_SCRIPT}")

    if export_mode not in ['full', 'summary']:
        raise ValueError("export_mode must be 'full' or 'summary'")

    # Ensure output directory exists
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    # Build command (use conda run to execute in r_fuel_metrics environment)
    cmd = [
        'conda', 'run', '-n', r_env_name, 'Rscript',
        str(METRICS_SCRIPT),
        str(input_laz),
        str(output_tif),
        str(resolution),
        str(layer_depth),
        str(height_cover),
        str(threshold),
        export_mode
    ]

    logger.info(f"Computing fuel metrics for {input_laz.name}")
    logger.debug(f"Command: {' '.join(cmd)}")

    # Execute R script
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False
        )

        # Log output
        if result.stdout:
            for line in result.stdout.splitlines():
                logger.info(f"  R: {line}")

        # Check for errors
        if result.returncode != 0:
            error_msg = f"fCBDprofile_fuelmetrics failed with exit code {result.returncode}"
            if result.stderr:
                error_msg += f"\nStderr:\n{result.stderr}"
            raise LidarForFuelError(error_msg)

        # Verify output exists
        if not output_tif.exists():
            raise LidarForFuelError(f"Output file not created: {output_tif}")

        logger.info(f"Fuel metrics computation completed: {output_tif}")

    except subprocess.TimeoutExpired:
        logger.error(f"Fuel metrics computation timed out after {timeout} seconds")
        raise


def process_point_cloud(
    input_las: Path,
    output_dir: Path,
    species: str = "Default",
    resolution: float = 1.0,
    export_mode: str = "full",
    cleanup_intermediate: bool = False
) -> Tuple[Path, Path]:
    """
    Complete pipeline: pretreatment + fuel metrics computation.

    Convenience function that runs both steps sequentially.

    Args:
        input_las: Path to input LAS/LAZ file
        output_dir: Directory for outputs
        species: Species name from trait lookup table (default: 'Default')
        resolution: Output raster resolution in meters (default: 1.0)
        export_mode: 'full' (173 bands) or 'summary' (23 bands) (default: 'full')
        cleanup_intermediate: Delete pretreated LAZ after completion (default: False)

    Returns:
        Tuple of (pretreated_laz_path, fuel_metrics_tif_path)

    Raises:
        FileNotFoundError: If input file not found or trait lookup missing
        LidarForFuelError: If processing fails
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get trait values
    traits = load_trait_lookup()
    if species not in traits:
        logger.warning(f"Species '{species}' not found in lookup, using 'Default'")
        species = "Default"

    trait_data = traits[species]
    logger.info(f"Using traits for: {trait_data['common_name']}")

    # Define output paths
    stem = input_las.stem
    pretreated_laz = output_dir / "pretreated" / f"{stem}_pretreated.laz"
    fuel_metrics_tif = output_dir / "rasters" / f"{stem}_fuel_metrics.tif"

    # Step 1: Pretreatment
    run_pretreatment(
        input_las=input_las,
        output_laz=pretreated_laz,
        lma=trait_data['lma_gm2'],
        wd=trait_data['wd_kgm3'],
        lma_bush=trait_data['lma_understory_gm2'],
        wd_bush=trait_data['wd_understory_kgm3']
    )

    # Step 2: Fuel metrics
    run_fuel_metrics(
        input_laz=pretreated_laz,
        output_tif=fuel_metrics_tif,
        resolution=resolution,
        export_mode=export_mode
    )

    # Optional cleanup
    if cleanup_intermediate and pretreated_laz.exists():
        logger.info(f"Removing intermediate file: {pretreated_laz}")
        pretreated_laz.unlink()

    return pretreated_laz, fuel_metrics_tif
