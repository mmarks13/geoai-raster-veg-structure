#!/usr/bin/env python
"""
Download 3DEP COPC point clouds from Planetary Computer for forest plot sites.

This script queries the Planetary Computer STAC API for 3DEP COPC tiles covering
specified forest plot sites, downloads and merges them using PDAL, and validates
the output. Supports streaming COPC read with fallback to download-first approach.

Usage:
    python src/data_prep/download_3dep_for_sites.py --site BluffMesa --output-dir data/processed/fuel_metrics/3dep_baseline/BluffMesa
    python src/data_prep/download_3dep_for_sites.py --all  # Process all sites
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pdal
import pystac_client
import planetary_computer
from shapely.geometry import box, mapping


# Site bounding boxes (WGS84: [minlon, minlat, maxlon, maxlat])
SITE_BBOXES = {
    "BluffMesa": [-116.959308, 34.215152, -116.951654, 34.222287],
    "Laguna": [-116.438215, 32.844384, -116.424106, 32.862321],
    "NorthBigBear": [-116.937442, 34.287520, -116.917939, 34.298950],
    "ReyesPeak": [-119.341524, 34.632405, -119.282198, 34.643359],
}

TARGET_CRS = "EPSG:32611"  # UTM 11N to match existing data


def setup_logging(output_dir: Path) -> logging.Logger:
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "download_log.txt"

    logger = logging.getLogger("3dep_download")
    logger.setLevel(logging.INFO)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def bbox_to_geojson_polygon(bbox: List[float]) -> dict:
    """Convert bbox [minlon, minlat, maxlon, maxlat] to GeoJSON Polygon."""
    minlon, minlat, maxlon, maxlat = bbox
    return {
        "type": "Polygon",
        "coordinates": [[
            [minlon, minlat],
            [maxlon, minlat],
            [maxlon, maxlat],
            [minlon, maxlat],
            [minlon, minlat]
        ]]
    }


def search_3dep_copc(
    catalog_client,
    bbox: List[float],
    logger: logging.Logger
) -> List:
    """
    Search Planetary Computer for 3DEP COPC tiles covering bbox.

    Args:
        catalog_client: pystac_client.Client connected to Planetary Computer
        bbox: Bounding box [minlon, minlat, maxlon, maxlat] in WGS84
        logger: Logger instance

    Returns:
        List of STAC items with 'data' asset
    """
    polygon = bbox_to_geojson_polygon(bbox)

    logger.info(f"Searching for 3DEP COPC tiles covering bbox {bbox}")

    try:
        search = catalog_client.search(
            collections=["3dep-lidar-copc"],
            intersects=polygon,
            datetime="2015-01-01/2024-12-31",  # Wide date range
            limit=100
        )
        items = list(search.items())

        # Filter items that have 'data' asset
        valid_items = [item for item in items if 'data' in item.assets]

        logger.info(f"Found {len(valid_items)} COPC tiles with data assets")

        if not valid_items:
            logger.warning("No 3DEP COPC tiles found for this bbox")

        return valid_items

    except Exception as e:
        logger.error(f"Error searching Planetary Computer: {e}")
        raise


def download_and_merge_streaming(
    items: List,
    bbox: List[float],
    output_path: Path,
    logger: logging.Logger,
    requests_threads: int = 8
) -> bool:
    """
    Download and merge COPC tiles using streaming approach.

    Args:
        items: List of STAC items with COPC data
        bbox: Bounding box for spatial filtering
        output_path: Path to output LAZ file
        logger: Logger instance
        requests_threads: Number of parallel HTTP threads

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Attempting streaming download with {len(items)} tiles")

    try:
        # Build PDAL pipeline
        polygon = bbox_to_geojson_polygon(bbox)
        pipeline_stages = []

        # Add a reader for each COPC tile
        for item in items:
            asset_url = item.assets['data'].href
            logger.info(f"  Adding reader for {item.id}")

            pipeline_stages.append({
                "type": "readers.copc",
                "filename": asset_url,
                "requests": requests_threads,
                "polygon": json.dumps(polygon)
            })

        # Merge if multiple tiles
        if len(items) > 1:
            pipeline_stages.append({
                "type": "filters.merge"
            })

        # Reproject to target CRS
        pipeline_stages.append({
            "type": "filters.reprojection",
            "out_srs": TARGET_CRS
        })

        # Write to LAZ
        pipeline_stages.append({
            "type": "writers.las",
            "filename": str(output_path),
            "compression": "laszip"
        })

        pipeline_dict = {"pipeline": pipeline_stages}
        pipeline_json = json.dumps(pipeline_dict, indent=2)

        logger.info("Executing PDAL pipeline...")
        logger.debug(f"Pipeline JSON:\n{pipeline_json}")

        pipeline = pdal.Pipeline(pipeline_json)
        count = pipeline.execute()

        logger.info(f"Successfully processed {count} points")
        logger.info(f"Output written to {output_path}")

        return True

    except Exception as e:
        logger.error(f"Streaming download failed: {e}")
        return False


def download_and_merge_fallback(
    items: List,
    bbox: List[float],
    output_path: Path,
    logger: logging.Logger,
    temp_dir: Path
) -> bool:
    """
    Download COPC tiles to temp files, then merge (fallback approach).

    Args:
        items: List of STAC items with COPC data
        bbox: Bounding box for spatial filtering
        output_path: Path to output LAZ file
        logger: Logger instance
        temp_dir: Directory for temporary files

    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Attempting fallback download-first approach with {len(items)} tiles")

    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_files = []

    try:
        polygon = bbox_to_geojson_polygon(bbox)

        # Download each tile separately
        for i, item in enumerate(items):
            asset_url = item.assets['data'].href
            temp_file = temp_dir / f"tile_{i:03d}.laz"

            logger.info(f"  Downloading tile {i+1}/{len(items)}: {item.id}")

            pipeline_stages = [
                {
                    "type": "readers.copc",
                    "filename": asset_url,
                    "requests": 4,  # Conservative for fallback
                    "polygon": json.dumps(polygon)
                },
                {
                    "type": "writers.las",
                    "filename": str(temp_file),
                    "compression": "laszip"
                }
            ]

            pipeline_json = json.dumps({"pipeline": pipeline_stages})
            pipeline = pdal.Pipeline(pipeline_json)
            count = pipeline.execute()

            logger.info(f"    Downloaded {count} points")
            temp_files.append(temp_file)

        # Merge all tiles
        logger.info("Merging downloaded tiles...")

        merge_stages = []
        for temp_file in temp_files:
            merge_stages.append({
                "type": "readers.las",
                "filename": str(temp_file)
            })

        if len(temp_files) > 1:
            merge_stages.append({"type": "filters.merge"})

        merge_stages.append({
            "type": "filters.reprojection",
            "out_srs": TARGET_CRS
        })

        merge_stages.append({
            "type": "writers.las",
            "filename": str(output_path),
            "compression": "laszip"
        })

        merge_json = json.dumps({"pipeline": merge_stages})
        merge_pipeline = pdal.Pipeline(merge_json)
        total_count = merge_pipeline.execute()

        logger.info(f"Successfully merged {total_count} total points")
        logger.info(f"Output written to {output_path}")

        # Clean up temp files
        for temp_file in temp_files:
            temp_file.unlink()

        return True

    except Exception as e:
        logger.error(f"Fallback download failed: {e}")
        return False
    finally:
        # Clean up temp directory
        if temp_dir.exists():
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def validate_point_cloud(
    laz_path: Path,
    expected_crs: str = TARGET_CRS,
    logger: Optional[logging.Logger] = None
) -> Dict:
    """
    Validate downloaded point cloud meets requirements.

    Args:
        laz_path: Path to LAZ file
        expected_crs: Expected CRS string
        logger: Optional logger instance

    Returns:
        Dictionary with validation results
    """
    if logger:
        logger.info(f"Validating point cloud: {laz_path}")

    try:
        # Read point cloud metadata
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
        las_metadata = metadata["metadata"]["readers.las"]

        # Extract key info
        point_count = pipeline.arrays[0].shape[0]
        bounds = las_metadata["comp_spatialreference"]["bbox"]
        srs_wkt = las_metadata.get("srs", {}).get("wkt", "")

        # Calculate area and density
        minx, maxx = bounds["minx"], bounds["maxx"]
        miny, maxy = bounds["miny"], bounds["maxy"]
        area_m2 = (maxx - minx) * (maxy - miny)
        point_density = point_count / area_m2 if area_m2 > 0 else 0

        # Validation checks
        crs_valid = "32611" in srs_wkt or "UTM zone 11N" in srs_wkt
        density_in_range = 0.5 <= point_density <= 20
        has_points = point_count > 1000

        results = {
            "file": str(laz_path),
            "point_count": point_count,
            "bounds": {"minx": minx, "maxx": maxx, "miny": miny, "maxy": maxy},
            "area_m2": area_m2,
            "area_ha": area_m2 / 10000,
            "point_density_pts_m2": point_density,
            "crs_wkt": srs_wkt,
            "crs_valid": crs_valid,
            "density_in_range": density_in_range,
            "has_points": has_points,
            "all_checks_passed": crs_valid and density_in_range and has_points
        }

        if logger:
            logger.info(f"  Point count: {point_count:,}")
            logger.info(f"  Area: {results['area_ha']:.2f} ha")
            logger.info(f"  Point density: {point_density:.2f} pts/m²")
            logger.info(f"  CRS valid: {crs_valid}")
            logger.info(f"  Density in range (0.5-20): {density_in_range}")
            logger.info(f"  Has sufficient points (>1000): {has_points}")
            logger.info(f"  Overall validation: {'PASS' if results['all_checks_passed'] else 'FAIL'}")

        return results

    except Exception as e:
        error_msg = f"Validation failed: {e}"
        if logger:
            logger.error(error_msg)
        return {"error": error_msg, "all_checks_passed": False}


def download_site(
    site_name: str,
    output_dir: Path,
    catalog_client,
    logger: logging.Logger
) -> bool:
    """
    Download 3DEP COPC data for a single site.

    Args:
        site_name: Name of site (must be in SITE_BBOXES)
        output_dir: Output directory for this site
        catalog_client: pystac_client.Client connected to Planetary Computer
        logger: Logger instance

    Returns:
        True if successful, False otherwise
    """
    if site_name not in SITE_BBOXES:
        logger.error(f"Unknown site: {site_name}. Valid sites: {list(SITE_BBOXES.keys())}")
        return False

    bbox = SITE_BBOXES[site_name]
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "3dep_merged.laz"

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing site: {site_name}")
    logger.info(f"Bounding box: {bbox}")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}\n")

    # Search for COPC tiles
    items = search_3dep_copc(catalog_client, bbox, logger)

    if not items:
        logger.error(f"No COPC tiles found for {site_name}")
        return False

    # Try streaming approach first
    logger.info("\n--- Attempting streaming download ---")
    success = download_and_merge_streaming(items, bbox, output_path, logger)

    # Fallback to download-first if streaming fails
    if not success:
        logger.warning("\n--- Streaming failed, trying fallback approach ---")
        temp_dir = output_dir / "temp_tiles"
        success = download_and_merge_fallback(items, bbox, output_path, logger, temp_dir)

    if not success:
        logger.error(f"Failed to download data for {site_name}")
        return False

    # Validate output
    logger.info("\n--- Validating output ---")
    validation_results = validate_point_cloud(output_path, TARGET_CRS, logger)

    # Save validation results
    validation_file = output_dir / "validation_results.json"
    with open(validation_file, 'w') as f:
        json.dump(validation_results, f, indent=2)

    logger.info(f"Validation results saved to {validation_file}")

    if validation_results.get("all_checks_passed", False):
        logger.info(f"\n✓ Successfully downloaded and validated {site_name}")
        return True
    else:
        logger.warning(f"\n⚠ Downloaded {site_name} but validation failed")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download 3DEP COPC point clouds from Planetary Computer for forest plot sites"
    )
    parser.add_argument(
        "--site",
        type=str,
        choices=list(SITE_BBOXES.keys()),
        help="Site name to process"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all sites"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: data/processed/fuel_metrics/3dep_baseline/{site})"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("data/processed/fuel_metrics/3dep_baseline"),
        help="Base output directory when using --all"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.site and not args.all:
        parser.error("Must specify either --site or --all")

    if args.site and args.all:
        parser.error("Cannot specify both --site and --all")

    # Determine sites to process
    if args.all:
        sites_to_process = list(SITE_BBOXES.keys())
    else:
        sites_to_process = [args.site]

    # Connect to Planetary Computer
    print("Connecting to Planetary Computer STAC API...")
    catalog_client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Process each site
    results = {}
    for site in sites_to_process:
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = args.output_base / site

        # Setup logging for this site
        logger = setup_logging(output_dir)

        # Download site
        success = download_site(site, output_dir, catalog_client, logger)
        results[site] = success

    # Summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    for site, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{site:20s}: {status}")

    # Exit with error code if any failed
    if not all(results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
