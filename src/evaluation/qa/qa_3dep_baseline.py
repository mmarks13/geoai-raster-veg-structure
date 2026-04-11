#!/usr/bin/env python
"""
Quality assurance validation for 3DEP baseline fuel metrics.

This script validates:
1. Point density statistics (expected range 0.5-20 pts/m²)
2. Coverage statistics (percentage of valid fuel metrics)
3. CRS validation (all files should be EPSG:32611)
4. Bounds validation (data covers expected areas)

Usage:
    python src/evaluation/qa_3dep_baseline.py \
        --data-dir data/processed/fuel_metrics/3dep_baseline \
        --output data/processed/fuel_metrics/3dep_baseline/qa_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pdal
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Expected CRS
EXPECTED_CRS = "EPSG:32611"  # UTM 11N

# Expected point density range (pts/m²)
DENSITY_MIN = 0.5
DENSITY_MAX = 20.0


def compute_point_density_stats(laz_path: Path) -> Dict:
    """
    Compute point density statistics for LAZ file.

    Args:
        laz_path: Path to LAZ file

    Returns:
        Dict with point_count, area_m2, density, classification_distribution
    """
    logger.info(f"\nAnalyzing point cloud: {laz_path.name}")

    try:
        # Read point cloud
        pipeline_json = json.dumps({
            "pipeline": [
                {"type": "readers.las", "filename": str(laz_path)}
            ]
        })

        pipeline = pdal.Pipeline(pipeline_json)
        pipeline.execute()

        # pipeline.metadata returns a string, need to parse it
        metadata_str = pipeline.metadata
        metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str

        # Navigate metadata structure carefully
        if "metadata" in metadata:
            meta_section = metadata["metadata"]
            if isinstance(meta_section, str):
                meta_section = json.loads(meta_section)
            las_metadata = meta_section.get("readers.las", {})
        else:
            las_metadata = {}

        # Get point array
        points = pipeline.arrays[0]
        point_count = len(points)

        # Get bounds
        if "comp_spatialreference" in las_metadata and "bbox" in las_metadata["comp_spatialreference"]:
            bounds = las_metadata["comp_spatialreference"]["bbox"]
        else:
            # Fallback: compute bounds from points
            bounds = {
                "minx": float(points['X'].min()),
                "maxx": float(points['X'].max()),
                "miny": float(points['Y'].min()),
                "maxy": float(points['Y'].max())
            }
        minx, maxx = bounds["minx"], bounds["maxx"]
        miny, maxy = bounds["miny"], bounds["maxy"]

        # Calculate area and density
        area_m2 = (maxx - minx) * (maxy - miny)
        area_ha = area_m2 / 10000
        density = point_count / area_m2 if area_m2 > 0 else 0

        # Classification distribution
        classification_dist = {}
        if 'Classification' in points.dtype.names:
            unique, counts = np.unique(points['Classification'], return_counts=True)
            classification_dist = {int(c): int(cnt) for c, cnt in zip(unique, counts)}

        # Validation checks
        density_in_range = DENSITY_MIN <= density <= DENSITY_MAX
        has_points = point_count > 1000

        results = {
            "file": str(laz_path),
            "point_count": int(point_count),
            "bounds": {
                "minx": float(minx),
                "maxx": float(maxx),
                "miny": float(miny),
                "maxy": float(maxy)
            },
            "area_m2": float(area_m2),
            "area_ha": float(area_ha),
            "point_density_pts_m2": float(density),
            "classification_distribution": classification_dist,
            "density_in_range": bool(density_in_range),
            "has_points": bool(has_points),
            "checks_passed": bool(density_in_range and has_points)
        }

        logger.info(f"  Point count: {point_count:,}")
        logger.info(f"  Area: {area_ha:.2f} ha")
        logger.info(f"  Point density: {density:.2f} pts/m²")
        logger.info(f"  Density in range ({DENSITY_MIN}-{DENSITY_MAX}): {density_in_range}")
        logger.info(f"  Has sufficient points (>1000): {has_points}")
        logger.info(f"  Classification: {classification_dist}")
        logger.info(f"  Validation: {'PASS' if results['checks_passed'] else 'FAIL'}")

        return results

    except Exception as e:
        error_msg = f"Failed to analyze {laz_path}: {e}"
        logger.error(error_msg)
        return {
            "file": str(laz_path),
            "error": error_msg,
            "checks_passed": False
        }


def compute_coverage_stats(
    raster_path: Path,
    site_polygon: Optional[gpd.GeoDataFrame] = None
) -> Dict:
    """
    Compute coverage statistics for fuel metrics raster.

    Args:
        raster_path: Path to fuel metrics GeoTIFF
        site_polygon: Optional site boundary polygon for clipping

    Returns:
        Dict with total_pixels, valid_pixels, coverage_percent, nan_regions
    """
    logger.info(f"\nAnalyzing raster coverage: {raster_path.name}")

    try:
        with rasterio.open(raster_path) as src:
            # Read first band as indicator (all bands should have same coverage)
            band = src.read(1)

            # Count valid vs NaN pixels
            total_pixels = band.size
            valid_pixels = np.sum(~np.isnan(band))
            nan_pixels = np.sum(np.isnan(band))
            coverage_percent = 100 * valid_pixels / total_pixels if total_pixels > 0 else 0

            # Find NaN regions (vectorize)
            nan_mask = np.isnan(band).astype('uint8')

            # Extract NaN region polygons
            nan_regions = []
            for geom, value in shapes(nan_mask, mask=(nan_mask == 1), transform=src.transform):
                if value == 1:  # NaN region
                    poly = shape(geom)
                    nan_regions.append({
                        "bounds": list(poly.bounds),
                        "area_m2": float(poly.area)
                    })

            # Sort by area (largest first)
            nan_regions = sorted(nan_regions, key=lambda x: x['area_m2'], reverse=True)[:10]  # Top 10

            # Coverage quality assessment
            if coverage_percent > 95:
                quality = "Excellent"
            elif coverage_percent > 80:
                quality = "Good"
            elif coverage_percent > 50:
                quality = "Marginal"
            else:
                quality = "Poor"

            results = {
                "file": str(raster_path),
                "total_pixels": int(total_pixels),
                "valid_pixels": int(valid_pixels),
                "nan_pixels": int(nan_pixels),
                "coverage_percent": float(coverage_percent),
                "coverage_quality": quality,
                "largest_nan_regions": nan_regions,
                "checks_passed": bool(coverage_percent > 80)
            }

            logger.info(f"  Total pixels: {total_pixels:,}")
            logger.info(f"  Valid pixels: {valid_pixels:,} ({coverage_percent:.1f}%)")
            logger.info(f"  NaN pixels: {nan_pixels:,}")
            logger.info(f"  Coverage quality: {quality}")
            logger.info(f"  Validation: {'PASS' if results['checks_passed'] else 'FAIL'}")

            return results

    except Exception as e:
        error_msg = f"Failed to analyze {raster_path}: {e}"
        logger.error(error_msg)
        return {
            "file": str(raster_path),
            "error": error_msg,
            "checks_passed": False
        }


def validate_crs(file_path: Path, expected_crs: str = EXPECTED_CRS) -> Dict:
    """
    Validate file CRS matches expected.

    Args:
        file_path: Path to LAZ or GeoTIFF file
        expected_crs: Expected CRS string

    Returns:
        Dict with crs, crs_valid, checks_passed
    """
    logger.info(f"\nValidating CRS: {file_path.name}")

    try:
        # Try as raster first
        if file_path.suffix in ['.tif', '.tiff']:
            with rasterio.open(file_path) as src:
                crs = src.crs.to_string() if src.crs else "None"
        else:
            # Try as point cloud
            pipeline_json = json.dumps({
                "pipeline": [
                    {"type": "readers.las", "filename": str(file_path)}
                ]
            })

            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()

            # pipeline.metadata returns a string, need to parse it
            metadata_str = pipeline.metadata
            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
            las_metadata = metadata["metadata"]["readers.las"]
            crs_wkt = las_metadata.get("srs", {}).get("wkt", "")
            crs = crs_wkt if crs_wkt else "None"

        # Check if CRS matches
        crs_valid = "32611" in crs or "UTM zone 11N" in crs

        results = {
            "file": str(file_path),
            "crs": crs,
            "expected_crs": expected_crs,
            "crs_valid": bool(crs_valid),
            "checks_passed": bool(crs_valid)
        }

        logger.info(f"  CRS: {crs[:100]}...")  # Truncate for readability
        logger.info(f"  Expected: {expected_crs}")
        logger.info(f"  Valid: {crs_valid}")
        logger.info(f"  Validation: {'PASS' if crs_valid else 'FAIL'}")

        return results

    except Exception as e:
        error_msg = f"Failed to validate CRS for {file_path}: {e}"
        logger.error(error_msg)
        return {
            "file": str(file_path),
            "error": error_msg,
            "checks_passed": False
        }


def validate_bounds(
    data_bounds: Tuple[float, float, float, float],
    expected_bounds: Tuple[float, float, float, float],
    tolerance_m: float = 100
) -> Dict:
    """
    Check if data bounds cover expected area.

    Args:
        data_bounds: (minx, miny, maxx, maxy) from actual data
        expected_bounds: (minx, miny, maxx, maxy) expected
        tolerance_m: Tolerance in meters

    Returns:
        Dict with bounds_match, coverage_percent, missing_regions
    """
    logger.info("\nValidating bounds coverage")

    data_box = box(*data_bounds)
    expected_box = box(*expected_bounds)

    # Check intersection
    intersection = data_box.intersection(expected_box)
    coverage_percent = 100 * intersection.area / expected_box.area if expected_box.area > 0 else 0

    # Check if data fully covers expected (within tolerance)
    data_minx, data_miny, data_maxx, data_maxy = data_bounds
    exp_minx, exp_miny, exp_maxx, exp_maxy = expected_bounds

    bounds_match = (
        abs(data_minx - exp_minx) <= tolerance_m and
        abs(data_miny - exp_miny) <= tolerance_m and
        abs(data_maxx - exp_maxx) <= tolerance_m and
        abs(data_maxy - exp_maxy) <= tolerance_m
    )

    results = {
        "data_bounds": list(data_bounds),
        "expected_bounds": list(expected_bounds),
        "bounds_match": bool(bounds_match),
        "coverage_percent": float(coverage_percent),
        "tolerance_m": tolerance_m,
        "checks_passed": bool(coverage_percent > 90)
    }

    logger.info(f"  Data bounds: {data_bounds}")
    logger.info(f"  Expected bounds: {expected_bounds}")
    logger.info(f"  Coverage: {coverage_percent:.1f}%")
    logger.info(f"  Bounds match (±{tolerance_m}m): {bounds_match}")
    logger.info(f"  Validation: {'PASS' if results['checks_passed'] else 'FAIL'}")

    return results


def run_qa_for_site(site_dir: Path) -> Dict:
    """
    Run complete QA validation for a single site.

    Args:
        site_dir: Directory containing site data (tiles, merged rasters, etc.)

    Returns:
        Dict with QA results for this site
    """
    site_name = site_dir.name
    logger.info(f"\n{'='*60}")
    logger.info(f"Running QA for site: {site_name}")
    logger.info(f"{'='*60}")

    results = {
        "site_name": site_name,
        "site_dir": str(site_dir),
        "checks": {},
        "all_checks_passed": False
    }

    # Find 3dep_merged.laz
    merged_laz = site_dir / "3dep_merged.laz"
    if merged_laz.exists():
        logger.info("\n--- Point Density QA ---")
        results['checks']['point_density'] = compute_point_density_stats(merged_laz)

        logger.info("\n--- CRS Validation (LAZ) ---")
        results['checks']['crs_laz'] = validate_crs(merged_laz)
    else:
        logger.warning(f"Merged LAZ not found: {merged_laz}")
        results['checks']['point_density'] = {"error": "File not found", "checks_passed": False}

    # Find merged fuel metrics raster
    merged_dir = site_dir.parent / f"3dep_{site_name}" / "merged"
    if merged_dir.exists():
        raster_files = list(merged_dir.glob("*.tif"))
        if raster_files:
            raster_path = raster_files[0]

            logger.info("\n--- Coverage QA ---")
            results['checks']['coverage'] = compute_coverage_stats(raster_path)

            logger.info("\n--- CRS Validation (Raster) ---")
            results['checks']['crs_raster'] = validate_crs(raster_path)
        else:
            logger.warning(f"No raster files found in {merged_dir}")
            results['checks']['coverage'] = {"error": "File not found", "checks_passed": False}
    else:
        logger.warning(f"Merged directory not found: {merged_dir}")
        results['checks']['coverage'] = {"error": "Directory not found", "checks_passed": False}

    # Overall validation
    all_passed = all(check.get('checks_passed', False) for check in results['checks'].values())
    results['all_checks_passed'] = all_passed

    logger.info(f"\n{'='*60}")
    logger.info(f"Site {site_name} QA: {'PASS' if all_passed else 'FAIL'}")
    logger.info(f"{'='*60}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Quality assurance validation for 3DEP baseline fuel metrics"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Base directory containing site subdirectories (e.g., data/processed/fuel_metrics/3dep_baseline)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for QA report JSON"
    )
    parser.add_argument(
        "--site",
        type=str,
        help="Optional: only run QA for specific site"
    )

    args = parser.parse_args()

    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)

    # Find site directories
    if args.site:
        site_dirs = [args.data_dir / args.site]
    else:
        # Find all site directories (those containing 3dep_merged.laz)
        site_dirs = [d for d in args.data_dir.iterdir() if d.is_dir() and (d / "3dep_merged.laz").exists()]

    if not site_dirs:
        logger.error(f"No site directories found in {args.data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(site_dirs)} sites to validate: {[d.name for d in site_dirs]}")

    # Run QA for each site
    qa_results = {
        "qa_timestamp": pd.Timestamp.now().isoformat(),
        "data_dir": str(args.data_dir),
        "sites": {},
        "summary": {}
    }

    for site_dir in site_dirs:
        site_results = run_qa_for_site(site_dir)
        qa_results['sites'][site_results['site_name']] = site_results

    # Summary statistics
    total_sites = len(site_dirs)
    passed_sites = sum(1 for r in qa_results['sites'].values() if r['all_checks_passed'])
    failed_sites = total_sites - passed_sites

    qa_results['summary'] = {
        "total_sites": total_sites,
        "passed_sites": passed_sites,
        "failed_sites": failed_sites,
        "pass_rate": 100 * passed_sites / total_sites if total_sites > 0 else 0
    }

    # Save report
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(qa_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info("QA SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total sites: {total_sites}")
    logger.info(f"Passed: {passed_sites}")
    logger.info(f"Failed: {failed_sites}")
    logger.info(f"Pass rate: {qa_results['summary']['pass_rate']:.1f}%")
    logger.info(f"\nQA report saved to: {args.output}")

    # Exit with error code if any failed
    if failed_sites > 0:
        sys.exit(1)


# Add pandas import
import pandas as pd


if __name__ == "__main__":
    main()
