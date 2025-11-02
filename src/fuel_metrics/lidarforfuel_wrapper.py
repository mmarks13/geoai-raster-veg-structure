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
import tempfile
import gc
import psutil
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Path to R scripts
SCRIPTS_DIR = Path(__file__).parents[2] / "scripts" / "fuel_metrics" / "r"
PRETREAT_SCRIPT = SCRIPTS_DIR / "run_pretreatment.R"
METRICS_SCRIPT = SCRIPTS_DIR / "run_fuel_metrics.R"

# Path to trait lookup table
TRAIT_LOOKUP_PATH = Path(__file__).parents[2] / "data" / "processed" / "fuel_metrics" / "trait_lookup.csv"


def log_memory_usage(step_name: str) -> None:
    """
    Log current memory usage statistics.

    Args:
        step_name: Descriptive name of the current processing step
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    rss_gb = mem_info.rss / (1024**3)
    available_gb = vm.available / (1024**3)
    total_gb = vm.total / (1024**3)
    swap_used_gb = swap.used / (1024**3)
    swap_total_gb = swap.total / (1024**3)

    logger.info(f"[{step_name}] Memory Usage:")
    logger.info(f"  Process RSS: {rss_gb:.2f} GB")
    logger.info(f"  System Available: {available_gb:.2f} GB / {total_gb:.2f} GB ({vm.percent:.1f}% used)")
    logger.info(f"  Swap Used: {swap_used_gb:.2f} GB / {swap_total_gb:.2f} GB ({swap.percent:.1f}% used)")


def log_file_size(file_path: Path, description: str = "File") -> None:
    """
    Log file size in human-readable format.

    Args:
        file_path: Path to file
        description: Description of the file (e.g., "Input LAS", "Output LAZ")
    """
    if file_path.exists():
        size_bytes = file_path.stat().st_size
        size_gb = size_bytes / (1024**3)
        logger.info(f"{description}: {size_gb:.2f} GB ({file_path.name})")
    else:
        logger.warning(f"{description}: File not found ({file_path})")


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
                'notes': row.get('notes', ''),
                'lma_source': row.get('lma_source', ''),
                'wd_source': row.get('wd_source', ''),
                'data_quality': row.get('data_quality', '')
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


def check_rscript_available(r_env_path: str = "/home/jovyan/r_fuel_metrics") -> bool:
    """
    Check if Rscript is available in the specified conda environment.

    Args:
        r_env_path: Path to the conda environment containing R (default: /home/jovyan/r_fuel_metrics)

    Returns:
        True if Rscript found, False otherwise
    """
    try:
        # First try conda run with path
        result = subprocess.run(
            ['conda', 'run', '-p', r_env_path, 'Rscript', '--version'],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            logger.debug(f"Found Rscript in conda environment: {r_env_path}")
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
    height_filter: float = 80.0,
    classify: bool = False,
    timeout: int = 3600,
    r_env_path: str = "/home/jovyan/r_fuel_metrics"
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
        height_filter: Maximum height (m) to retain (default: 80.0)
        classify: Whether to classify ground points (default: False)
        timeout: Maximum execution time in seconds (default: 3600)
        r_env_path: Path to conda environment containing R (default: /home/jovyan/r_fuel_metrics)

    Raises:
        FileNotFoundError: If input file or Rscript not found
        LidarForFuelError: If R script fails
        subprocess.TimeoutExpired: If execution exceeds timeout
    """
    # Validate inputs
    if not input_las.exists():
        raise FileNotFoundError(f"Input LAS file not found: {input_las}")

    if not check_rscript_available(r_env_path):
        raise FileNotFoundError(
            f"Rscript not found in conda environment '{r_env_path}'. Please ensure the environment exists:\n"
            f"  conda env create -f environment_r_fuel_metrics.yml -p /home/jovyan/r_fuel_metrics\n"
            f"  conda run -p /home/jovyan/r_fuel_metrics R -e \"remotes::install_github('oliviermartin7/lidarforfuel')\""
        )

    if not PRETREAT_SCRIPT.exists():
        raise FileNotFoundError(f"R pretreatment script not found: {PRETREAT_SCRIPT}")

    # Ensure output directory exists
    output_laz.parent.mkdir(parents=True, exist_ok=True)

    # Build command (use conda run to execute in r_fuel_metrics environment)
    cmd = [
        'conda', 'run', '-p', r_env_path, 'Rscript',
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

    import time
    start_time = time.time()

    logger.info(f"Running fPCpretreatment on {input_las.name}")
    logger.info(f"  LMA (canopy/understory): {lma:.1f} / {lma_bush:.1f} g/m²")
    logger.info(f"  WD (canopy/understory): {wd:.1f} / {wd_bush:.1f} kg/m³")
    logger.info(f"  Height filter: {height_filter:.1f} m, Classify: {classify}")
    logger.debug(f"Command: {' '.join(cmd)}")

    # Log input file size and memory before pretreatment
    log_file_size(input_las, "Input LAS")
    log_memory_usage("Before fPCpretreatment")

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

        elapsed = time.time() - start_time
        logger.info(f"Pretreatment completed in {elapsed:.1f} seconds: {output_laz}")

        # Log output file size and memory after pretreatment
        log_file_size(output_laz, "Output pretreated LAZ")
        log_memory_usage("After fPCpretreatment")

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
    omega: float = 0.77,
    projection_factor: float = 0.5,
    export_mode: str = "full",
    timeout: int = 7200,
    r_env_path: str = "/home/jovyan/r_fuel_metrics"
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
        omega: Clumping factor Ω for Beer-Lambert model (default: 0.77)
        projection_factor: Projection factor G for fuel metrics (default: 0.5)
        export_mode: 'full' (173 bands) or 'summary' (23 bands) (default: 'full')
        timeout: Maximum execution time in seconds (default: 7200)
        r_env_path: Path to conda environment containing R (default: /home/jovyan/r_fuel_metrics)

    Raises:
        FileNotFoundError: If input file or Rscript not found
        LidarForFuelError: If R script fails
        subprocess.TimeoutExpired: If execution exceeds timeout
    """
    # Validate inputs
    if not input_laz.exists():
        raise FileNotFoundError(f"Input LAZ file not found: {input_laz}")

    if not check_rscript_available(r_env_path):
        raise FileNotFoundError(f"Rscript not found in conda environment '{r_env_path}'")

    if not METRICS_SCRIPT.exists():
        raise FileNotFoundError(f"R metrics script not found: {METRICS_SCRIPT}")

    if export_mode not in ['full', 'summary']:
        raise ValueError("export_mode must be 'full' or 'summary'")

    # Ensure output directory exists
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    # Build command (use conda run to execute in r_fuel_metrics environment)
    cmd = [
        'conda', 'run', '-p', r_env_path, 'Rscript',
        str(METRICS_SCRIPT),
        str(input_laz),
        str(output_tif),
        str(resolution),
        str(layer_depth),
        str(height_cover),
        str(threshold),
        str(omega),
        str(projection_factor),
        export_mode
    ]

    import time
    start_time = time.time()

    logger.info(f"Computing fuel metrics for {input_laz.name}")
    logger.info(f"  Resolution: {resolution} m, Export mode: {export_mode}")
    logger.info(f"  Layer depth: {layer_depth} m, Threshold: {threshold}")
    logger.info(f"  Omega (Ω): {omega}, Projection factor (G): {projection_factor}")
    logger.debug(f"Command: {' '.join(cmd)}")

    # Log input file size and memory before fuel metrics
    log_file_size(input_laz, "Input pretreated LAZ")
    log_memory_usage("Before Fuel Metrics Computation")

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

        elapsed = time.time() - start_time
        logger.info(f"Fuel metrics computation completed in {elapsed:.1f} seconds: {output_tif}")

        # Log output file size and memory after fuel metrics
        log_file_size(output_tif, "Output fuel metrics GeoTIFF")
        log_memory_usage("After Fuel Metrics Computation")

    except subprocess.TimeoutExpired:
        logger.error(f"Fuel metrics computation timed out after {timeout} seconds")
        raise


def preprocess_with_pdal(
    input_las: Path,
    output_las: Path,
    min_hag: float = 0.0,
    max_hag: float = 80.0
) -> Path:
    """
    Preprocess LAS file with PDAL: ground classification and HAG computation.

    This function uses PDAL's SMRF (Simple Morphological Filter) algorithm to classify
    ground points before passing to LidarForFuel. This works around a bug in
    fPCpretreatment where it checks for ground points before performing classification.

    The pipeline:
    1. Reset all classifications to 0 (unclassified)
    2. Classify ground points using SMRF (sets class 2)
    3. Compute Height Above Ground (HAG) using Delaunay triangulation
    4. Filter points by HAG range
    5. Write classified LAS file

    NOTE: When using PDAL to create new LAS/LAZ files, include precision suffix
    in filename (e.g., _1cm.laz, _1mm.laz) to indicate coordinate scale.
    Check precision: pdal info --summary file.las | grep scale_

    Args:
        input_las: Path to input LAS/LAZ file (unclassified)
        output_las: Path for output classified LAS file
        min_hag: Minimum height above ground in meters (default: 0.0)
        max_hag: Maximum height above ground in meters (default: 80.0)

    Returns:
        Path to classified output file

    Raises:
        ImportError: If PDAL is not available
        RuntimeError: If PDAL pipeline execution fails
    """
    try:
        # Preferred explicit absolute import
        from src.utils.point_cloud_utils import process_and_classify_las
    except ImportError:
        try:
            # Fallback to a relative import when the package root is not on sys.path
            from ..utils.point_cloud_utils import process_and_classify_las
        except Exception as e:
            raise ImportError(
                "PDAL preprocessing requires src.utils.point_cloud_utils module. "
                "This project expects the repository root on PYTHONPATH or to be installed as a package. "
                "If running scripts directly, run them from the repository root or set PYTHONPATH=.\n"
                "Original import error: {}".format(e)
            ) from e

    import time
    start_time = time.time()

    logger.info(f"Running PDAL ground classification on {input_las.name}")
    logger.debug(f"  HAG range: [{min_hag}, {max_hag}] meters")

    # Log input file size and memory before PDAL
    log_file_size(input_las, "Input LAS")
    log_memory_usage("Before PDAL Classification")

    # Ensure output directory exists
    output_las.parent.mkdir(parents=True, exist_ok=True)

    # Create and execute PDAL pipeline
    pipeline = process_and_classify_las(
        input_las=str(input_las),
        output_las=str(output_las),
        min_hag=min_hag,
        max_hag=max_hag,
        filter_noise=False  # LidarForFuel does its own noise filtering
    )

    try:
        pipeline.execute()
        elapsed = time.time() - start_time
        logger.info(f"PDAL classification completed in {elapsed:.1f} seconds: {output_las}")

        # Log output file size and memory after PDAL
        log_file_size(output_las, "Output classified LAS")
        log_memory_usage("After PDAL Classification")

    except Exception as e:
        raise RuntimeError(f"PDAL pipeline execution failed: {e}") from e

    # Verify output exists
    if not output_las.exists():
        raise RuntimeError(f"PDAL did not create output file: {output_las}")

    return output_las


def process_point_cloud(
    input_las: Path,
    output_dir: Path,
    species: str = "Default",
    resolution: float = 1.0,
    clumping: float = 0.77,
    projection_factor: float = 0.5,
    export_mode: str = "full",
    cleanup_intermediate: bool = False,
    use_pdal_classification: bool = True,
    min_hag: float = 0.1,
    max_hag: float = 60.0
) -> Tuple[Path, Path]:
    """
    Complete pipeline: pretreatment + fuel metrics computation.

    Convenience function that runs both steps sequentially.

    Args:
        input_las: Path to input LAS/LAZ file
        output_dir: Directory for outputs
        species: Species name from trait lookup table (default: 'Default')
        resolution: Output raster resolution in meters (default: 1.0)
        clumping: Clumping factor Ω for Beer-Lambert model (default: 0.77)
        projection_factor: Projection factor G for fuel metrics (default: 0.5)
        export_mode: 'full' (173 bands) or 'summary' (23 bands) (default: 'full')
        cleanup_intermediate: Delete pretreated LAZ after completion (default: False)
        use_pdal_classification: Use PDAL for ground classification before LidarForFuel (default: True)
        min_hag: Minimum height above ground for PDAL filtering in meters (default: 0.0)
        max_hag: Maximum height above ground for PDAL filtering in meters (default: 80.0)

    Returns:
        Tuple of (pretreated_laz_path, fuel_metrics_tif_path)

    Raises:
        FileNotFoundError: If input file not found or trait lookup missing
        LidarForFuelError: If processing fails

    Notes:
        - PDAL classification is enabled by default to work around a bug in fPCpretreatment
          where it checks for ground points before performing classification
        - When use_pdal_classification=True, the PDAL SMRF algorithm classifies ground points,
          then LidarForFuel performs normalization and trait attribution (classify=False)
        - When use_pdal_classification=False, LidarForFuel attempts classification (may fail
          on unclassified data due to the bug)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Log pipeline configuration
    logger.info("=" * 80)
    logger.info("Fuel Metrics Pipeline Configuration")
    logger.info("=" * 80)
    logger.info(f"Input LAS: {input_las}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Species: {species}")
    logger.info(f"Resolution: {resolution} m")
    logger.info(f"Clumping (Ω): {clumping}")
    logger.info(f"Projection factor (G): {projection_factor}")
    logger.info(f"Export mode: {export_mode}")
    logger.info(f"Use PDAL classification: {use_pdal_classification}")
    logger.info(f"HAG range: [{min_hag}, {max_hag}] m")
    logger.info(f"Cleanup intermediate: {cleanup_intermediate}")
    logger.info("=" * 80)
    log_memory_usage("Pipeline Start")

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
    ground_classified_las = output_dir / "ground_classified" / f"{stem}_classified.las"

    # Step 1a: PDAL classification (if enabled)
    if use_pdal_classification:
        logger.info("Classification method: PDAL (SMRF algorithm)")

        # Check if ground classification checkpoint exists
        if ground_classified_las.exists():
            logger.info(f"Found existing ground-classified checkpoint: {ground_classified_las}")
            logger.info("Skipping PDAL classification (using checkpoint)")
            log_file_size(ground_classified_las, "Checkpoint classified LAS")
        else:
            # If input appears to be a tiled, already-processed LAZ (e.g. tile_*_1cm.laz)
            # skip running PDAL over many small tiles in parallel. Use the input file
            # directly as the ground-classified checkpoint.
            in_name = input_las.name.lower()
            if in_name.startswith('tile_') and '_1cm' in in_name:
                logger.info("Detected tiled input; skipping PDAL classification for tile and using input as checkpoint")
                ground_classified_las = input_las
            else:
                # Run PDAL classification and save to permanent checkpoint
                logger.info("Running PDAL ground classification (creating checkpoint)")
                preprocess_with_pdal(
                    input_las=input_las,
                    output_las=ground_classified_las,
                    min_hag=min_hag,
                    max_hag=max_hag
                )
                logger.info(f"PDAL checkpoint saved: {ground_classified_las}")

            # Force garbage collection after PDAL
            gc.collect()
            log_memory_usage("After PDAL and GC")

        # Step 1b: Pretreatment with pre-classified data
        run_pretreatment(
            input_las=ground_classified_las,
            output_laz=pretreated_laz,
            lma=trait_data['lma_gm2'],
            wd=trait_data['wd_kgm3'],
            lma_bush=trait_data['lma_understory_gm2'],
            wd_bush=trait_data['wd_understory_kgm3'],
            classify=False  # Already classified by PDAL
        )

        # Force garbage collection after pretreatment
        gc.collect()
        log_memory_usage("After Pretreatment and GC")
    else:
        logger.info("Classification method: LidarForFuel (may fail on unclassified data)")
        # Step 1b: Pretreatment with LidarForFuel classification
        run_pretreatment(
            input_las=input_las,
            output_laz=pretreated_laz,
            lma=trait_data['lma_gm2'],
            wd=trait_data['wd_kgm3'],
            lma_bush=trait_data['lma_understory_gm2'],
            wd_bush=trait_data['wd_understory_kgm3'],
            classify=True  # Let LidarForFuel attempt classification
        )

    # Step 2: Fuel metrics
    run_fuel_metrics(
        input_laz=pretreated_laz,
        output_tif=fuel_metrics_tif,
        resolution=resolution,
        omega=clumping,
        projection_factor=projection_factor,
        export_mode=export_mode
    )

    # Force garbage collection after fuel metrics
    gc.collect()
    log_memory_usage("After Fuel Metrics and GC")

    # Optional cleanup
    if cleanup_intermediate and pretreated_laz.exists():
        logger.info(f"Removing intermediate file: {pretreated_laz}")
        pretreated_laz.unlink()

    logger.info("=" * 80)
    logger.info("Pipeline completed successfully")
    logger.info("=" * 80)
    log_memory_usage("Pipeline End")

    return pretreated_laz, fuel_metrics_tif
