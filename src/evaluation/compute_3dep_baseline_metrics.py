#!/usr/bin/env python
"""
Compute vegetation structure metrics from 3DEP LiDAR for baseline comparison.

This script wraps compute_vegetation_structure_metrics() to process 3DEP point clouds
and generate 24-band vegetation structure rasters for forest plot validation sites.

Usage:
    python src/evaluation/compute_3dep_baseline_metrics.py \
        --input data/processed/fuel_metrics/3dep_baseline/BluffMesa/3dep_merged.laz \
        --output data/processed/veg_structure_baseline/BluffMesa \
        --resolution 2.0

    # Process all sites
    python src/evaluation/compute_3dep_baseline_metrics.py --all
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.point_cloud_utils import (
    compute_vegetation_structure_metrics,
    save_metrics_to_geotiff,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Site configurations - reuse from download_3dep_for_sites.py
SITE_CONFIGS = {
    "BluffMesa": {
        "input": "data/processed/fuel_metrics/3dep_baseline/BluffMesa/3dep_merged.laz",
        "expected_crs": "EPSG:32611",
    },
    "Laguna": {
        "input": "data/processed/fuel_metrics/3dep_baseline/Laguna/3dep_merged.laz",
        "expected_crs": "EPSG:32611",
    },
    "NorthBigBear": {
        "input": "data/processed/fuel_metrics/3dep_baseline/NorthBigBear/3dep_merged.laz",
        "expected_crs": "EPSG:32611",
    },
    "ReyesPeak": {
        "input": "data/processed/fuel_metrics/3dep_baseline/ReyesPeak/3dep_merged.laz",
        "expected_crs": "EPSG:32611",
    },
}


def validate_input_laz(
    laz_path: Path,
    expected_crs: str = "EPSG:32611"
) -> Dict:
    """
    Validate 3DEP LAZ file before processing.

    Args:
        laz_path: Path to LAZ file
        expected_crs: Expected CRS string

    Returns:
        Dict with: point_count, bounds, crs_valid, density_pts_m2

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If CRS mismatch or insufficient points
    """
    import pdal

    if not laz_path.exists():
        raise FileNotFoundError(f"Input LAZ not found: {laz_path}")

    logger.info(f"Validating input: {laz_path}")

    # Read metadata
    pipeline_json = json.dumps({
        "pipeline": [
            {"type": "readers.las", "filename": str(laz_path)}
        ]
    })

    pipeline = pdal.Pipeline(pipeline_json)
    pipeline.execute()

    # Parse metadata
    metadata_str = pipeline.metadata
    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str

    # Get point array
    points = pipeline.arrays[0]
    point_count = len(points)

    # Calculate bounds and area
    minx, maxx = float(points['X'].min()), float(points['X'].max())
    miny, maxy = float(points['Y'].min()), float(points['Y'].max())
    area_m2 = (maxx - minx) * (maxy - miny)
    density = point_count / area_m2 if area_m2 > 0 else 0

    # Check CRS
    las_metadata = metadata.get("metadata", {}).get("readers.las", {})
    srs_wkt = las_metadata.get("srs", {}).get("wkt", "")
    crs_valid = "32611" in srs_wkt or "UTM zone 11N" in srs_wkt

    results = {
        "file": str(laz_path),
        "point_count": int(point_count),
        "bounds": {
            "minx": minx, "maxx": maxx,
            "miny": miny, "maxy": maxy
        },
        "area_m2": area_m2,
        "area_ha": area_m2 / 10000,
        "point_density_pts_m2": density,
        "crs_valid": crs_valid,
    }

    logger.info(f"  Point count: {point_count:,}")
    logger.info(f"  Area: {results['area_ha']:.2f} ha")
    logger.info(f"  Point density: {density:.2f} pts/m^2")
    logger.info(f"  CRS valid: {crs_valid}")

    # Validation checks
    if not crs_valid:
        raise ValueError(
            f"CRS mismatch! Expected {expected_crs}, got WKT containing: "
            f"{srs_wkt[:100]}... Reproject input file before processing."
        )

    if point_count < 1000:
        raise ValueError(
            f"Insufficient points: {point_count}. Need at least 1000 points."
        )

    return results


def add_hag_to_classified_las(
    input_laz: str,
    output_laz: str,
    target_crs: str = "EPSG:32611",
    min_hag: float = 0.0,
    max_hag: float = 100.0
) -> Tuple[int, str]:
    """
    Add HeightAboveGround to a LAS file that already has ground classification.

    Uses existing Classification (Class 2 = ground) without re-running SMRF.
    This is appropriate for 3DEP data which comes pre-classified.

    Tries hag_delaunay first (more accurate), falls back to hag_nn if it fails
    (handles numerical precision issues from poor LAS offset encoding).

    Args:
        input_laz: Path to input LAZ with existing classification
        output_laz: Path to output LAZ with HAG added
        target_crs: Target CRS for reprojection
        min_hag: Minimum HAG filter
        max_hag: Maximum HAG filter

    Returns:
        Tuple of (point_count, hag_method_used)
    """
    import pdal

    def build_pipeline(hag_filter: str) -> str:
        pipeline_def = [
            {
                "type": "readers.las",
                "filename": input_laz
            },
            {
                "type": "filters.reprojection",
                "out_srs": target_crs
            },
            {
                # Use existing ground classification (Class 2)
                "type": hag_filter
            },
            {
                "type": "filters.range",
                "limits": f"HeightAboveGround[{min_hag}:{max_hag}]"
            },
            {
                "type": "writers.las",
                "filename": output_laz,
                "compression": "laszip",
                "extra_dims": "all"
            }
        ]
        return json.dumps({"pipeline": pipeline_def})

    # Try hag_delaunay first (more accurate triangulation-based interpolation)
    try:
        pipeline = pdal.Pipeline(build_pipeline("filters.hag_delaunay"))
        count = pipeline.execute()
        return count, "hag_delaunay"
    except RuntimeError as e:
        if "collinear" in str(e).lower():
            logger.warning(f"hag_delaunay failed (collinear points - likely LAS offset encoding issue)")
            logger.warning("Falling back to hag_nn (nearest neighbor interpolation)")
        else:
            raise

    # Fallback to hag_nn (nearest neighbor - more robust to numerical issues)
    pipeline = pdal.Pipeline(build_pipeline("filters.hag_nn"))
    count = pipeline.execute()
    return count, "hag_nn"


def compute_baseline_for_site(
    input_laz: Path,
    output_dir: Path,
    site_name: str,
    resolution: float = 2.0,
    point_filter_max_hag: float = 100.0,
    target_crs: str = "EPSG:32611"
) -> Tuple[np.ndarray, dict]:
    """
    Compute vegetation structure metrics from 3DEP LiDAR.

    Args:
        input_laz: Path to 3DEP LAZ file
        output_dir: Output directory for this site
        site_name: Name of the site
        resolution: Raster resolution in meters
        point_filter_max_hag: Maximum height above ground filter
        target_crs: Target CRS

    Returns:
        Tuple of (raster array, metadata dict)
    """
    import tempfile
    import laspy

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing site: {site_name}")
    logger.info(f"Input: {input_laz}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Resolution: {resolution}m")
    logger.info(f"{'='*60}\n")

    # Validate input
    validation_results = validate_input_laz(Path(input_laz), target_crs)

    # Check if 3DEP data has existing ground classification
    logger.info("\nChecking for existing ground classification...")
    las = laspy.read(str(input_laz))
    has_hag = 'HeightAboveGround' in [dim.name for dim in las.point_format.dimensions]
    has_ground_class = 2 in np.unique(las.classification)
    ground_count = (las.classification == 2).sum()
    del las

    hag_method = None  # Track which HAG method was used

    if has_hag:
        logger.info("  HeightAboveGround already exists - using directly")
        working_laz = str(input_laz)
        hag_method = "existing"
    elif has_ground_class:
        logger.info(f"  Found existing ground classification: {ground_count:,} ground points")
        logger.info("  Adding HAG using existing classification (no SMRF re-run)")

        # Create temp file with HAG added
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.laz',
            dir=output_dir,
            delete=False
        )
        temp_file.close()
        working_laz = temp_file.name

        count, hag_method = add_hag_to_classified_las(
            str(input_laz),
            working_laz,
            target_crs=target_crs,
            min_hag=0.0,
            max_hag=point_filter_max_hag
        )
        logger.info(f"  HAG computed: {count:,} points after filtering (method: {hag_method})")
    else:
        logger.info("  No existing classification - will run full preprocessing (SMRF + HAG)")
        working_laz = str(input_laz)

    # Compute vegetation structure metrics
    logger.info("\nComputing vegetation structure metrics...")
    start_time = datetime.now()

    raster, metadata = compute_vegetation_structure_metrics(
        working_laz,
        resolution=resolution,
        canopy_min_hag=3.0,
        understory_max_hag=1.0,
        point_filter_min_hag=0.0,
        point_filter_max_hag=point_filter_max_hag,
        density_range=(0, 25),
        num_density_layers=10,
        percentiles=[10, 25, 50, 75, 90],
        preprocess=not (has_hag or has_ground_class),  # Only preprocess if no existing classification
        target_crs=target_crs,
    )

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(f"\nMetrics computation completed in {elapsed:.1f}s")
    logger.info(f"  Raster shape: {raster.shape}")
    logger.info(f"  Bands: {metadata['n_bands']}")

    # Save raster
    output_tif = output_dir / "veg_structure_2m.tif"
    logger.info(f"\nSaving raster to: {output_tif}")
    save_metrics_to_geotiff(raster, metadata, str(output_tif))

    # Clean up temp file if created
    if has_ground_class and not has_hag and working_laz != str(input_laz):
        try:
            Path(working_laz).unlink()
            logger.info(f"Cleaned up temp file: {working_laz}")
        except Exception as e:
            logger.warning(f"Failed to clean up temp file: {e}")

    # Save preprocessing log
    preprocess_mode = "existing_hag" if has_hag else ("existing_classification" if has_ground_class else "full_smrf")
    log_data = {
        "site_name": site_name,
        "input_file": str(input_laz),
        "output_file": str(output_tif),
        "processing_timestamp": datetime.now().isoformat(),
        "processing_time_seconds": elapsed,
        "preprocess_mode": preprocess_mode,
        "hag_method": hag_method,  # "existing", "hag_delaunay", or "hag_nn"
        "had_existing_hag": has_hag,
        "had_existing_ground_classification": has_ground_class,
        "ground_point_count": int(ground_count) if has_ground_class else None,
        "parameters": {
            "resolution": resolution,
            "canopy_min_hag": 3.0,
            "understory_max_hag": 1.0,
            "point_filter_max_hag": point_filter_max_hag,
            "density_range": [0, 25],
            "num_density_layers": 10,
            "percentiles": [10, 25, 50, 75, 90],
            "target_crs": target_crs,
        },
        "input_validation": validation_results,
        "output_shape": list(raster.shape),
        "band_names": metadata.get("band_names", []),
    }

    log_file = output_dir / "preprocessing_log.json"
    with open(log_file, 'w') as f:
        json.dump(log_data, f, indent=2)

    logger.info(f"Preprocessing log saved to: {log_file}")

    # Generate visualization
    try:
        from src.veg_structure_metrics.visualize_metrics import create_visualization
        vis_path = output_dir / "visualization.png"
        create_visualization(str(output_tif), str(vis_path))
        logger.info(f"Visualization saved to: {vis_path}")
    except ImportError:
        logger.warning("Visualization module not available, skipping")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")

    logger.info(f"\n{'='*60}")
    logger.info(f"Site {site_name} processing complete!")
    logger.info(f"{'='*60}\n")

    return raster, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Compute vegetation structure metrics from 3DEP LiDAR for baseline comparison"
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to input 3DEP LAZ file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output directory for this site"
    )
    parser.add_argument(
        "--site",
        type=str,
        choices=list(SITE_CONFIGS.keys()),
        help="Site name (uses default paths if --input/--output not provided)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all sites"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=2.0,
        help="Raster resolution in meters (default: 2.0)"
    )
    parser.add_argument(
        "--max-hag",
        type=float,
        default=100.0,
        help="Maximum height above ground filter (default: 100.0)"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("data/processed/veg_structure_baseline"),
        help="Base output directory when using --all or --site"
    )

    args = parser.parse_args()

    # Validate arguments
    if args.all:
        sites_to_process = list(SITE_CONFIGS.keys())
    elif args.site:
        sites_to_process = [args.site]
    elif args.input and args.output:
        # Custom input/output
        site_name = args.output.name
        compute_baseline_for_site(
            args.input,
            args.output,
            site_name,
            resolution=args.resolution,
            point_filter_max_hag=args.max_hag,
        )
        return
    else:
        parser.error("Must specify --all, --site, or both --input and --output")

    # Process sites
    results = {}
    for site_name in sites_to_process:
        config = SITE_CONFIGS[site_name]
        input_path = Path(config["input"])
        output_path = args.output_base / site_name

        if not input_path.exists():
            logger.warning(f"Skipping {site_name}: Input file not found at {input_path}")
            logger.warning(f"  Run: python src/data_prep/download_3dep_for_sites.py --site {site_name}")
            results[site_name] = "SKIPPED (input not found)"
            continue

        try:
            compute_baseline_for_site(
                input_path,
                output_path,
                site_name,
                resolution=args.resolution,
                point_filter_max_hag=args.max_hag,
            )
            results[site_name] = "SUCCESS"
        except Exception as e:
            logger.error(f"Failed to process {site_name}: {e}")
            results[site_name] = f"FAILED: {e}"

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PROCESSING SUMMARY")
    logger.info("="*60)
    for site, status in results.items():
        logger.info(f"  {site:20s}: {status}")


if __name__ == "__main__":
    main()
