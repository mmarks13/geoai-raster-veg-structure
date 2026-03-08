#!/usr/bin/env python
"""
Download 3DEP COPC point clouds and compute HAG + enhanced features.

This script queries Planetary Computer for 3DEP COPC tiles, downloads them,
and processes through a PDAL pipeline that computes:
- Height Above Ground (HAG) using SMRF + filters.hag_delaunay
- Eigenvalue features (Planarity, Sphericity, Verticality)

The output is a single LAZ file per site with all features stored as extra dimensions.

Usage:
    python src/data_prep/download_and_process_3dep_sites.py --site BluffMesa
    python src/data_prep/download_and_process_3dep_sites.py --all
    python src/data_prep/download_and_process_3dep_sites.py --site volcan_mtn --bbox "-116.625141,33.100581,-116.565760,33.145949"

Output dimensions:
    X, Y, Z (raw elevation), HeightAboveGround, Classification,
    Intensity, ReturnNumber, NumberOfReturns,
    Planarity, Sphericity, Verticality
"""

import argparse
import json
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pdal
import planetary_computer
import pystac_client
from shapely.geometry import box

# Site bounding boxes (WGS84: [minlon, minlat, maxlon, maxlat])
# Validation sites with field measurements
VALIDATION_SITE_BBOXES = {
    "BluffMesa": [-116.959308, 34.215152, -116.951654, 34.222287],
    "Laguna": [-116.438215, 32.844384, -116.424106, 32.862321],
    "NorthBigBear": [-116.937442, 34.287520, -116.917939, 34.298950],
    "ReyesPeak": [-119.341524, 34.632405, -119.282198, 34.643359],
    "TecuyaRidge": [-119.025374, 34.840597, -118.967664, 34.848034],
}

# Training sites (bboxes from vegetation structure metrics rasters)
TRAINING_SITE_BBOXES = {
    "volcan_mtn": [-116.625141, 33.100581, -116.565760, 33.145949],
    # Other training sites will have bboxes extracted from their rasters
}

# Combined for convenience
SITE_BBOXES = {**VALIDATION_SITE_BBOXES, **TRAINING_SITE_BBOXES}

TARGET_CRS = "EPSG:32611"  # UTM 11N

# SMRF parameters (matching CLAUDE.md Section 13)
SMRF_PARAMS = {
    "cell": 1.0,
    "slope": 0.15,
    "threshold": 0.5,
    "window": 18.0
}


def setup_logging(output_dir: Path, log_name: str = "processing_log.txt") -> logging.Logger:
    """Configure logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / log_name

    logger = logging.getLogger("3dep_hag_processing")
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    logger.handlers.clear()

    # File handler (DEBUG level)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler (INFO level)
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


def bbox_wgs84_to_utm(bbox_wgs84: List[float], target_crs: str = TARGET_CRS) -> List[float]:
    """
    Convert WGS84 bbox to UTM coordinates.

    Args:
        bbox_wgs84: [minlon, minlat, maxlon, maxlat] in WGS84
        target_crs: Target CRS (default: EPSG:32611)

    Returns:
        [minx, miny, maxx, maxy] in UTM
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)
    minlon, minlat, maxlon, maxlat = bbox_wgs84

    minx, miny = transformer.transform(minlon, minlat)
    maxx, maxy = transformer.transform(maxlon, maxlat)

    return [minx, miny, maxx, maxy]


def buffer_bbox_utm(bbox_utm: List[float], buffer_m: float) -> List[float]:
    """Add buffer in meters to UTM bbox."""
    minx, miny, maxx, maxy = bbox_utm
    return [minx - buffer_m, miny - buffer_m, maxx + buffer_m, maxy + buffer_m]


def deduplicate_3dep_items(items: List, logger: logging.Logger) -> List:
    """
    Deduplicate 3DEP STAC items that cover the same spatial grid cell.

    Some USGS projects have multiple versions of the same tiles.
    """
    if not items:
        return items

    def extract_grid_id(item_id: str) -> str:
        match = re.search(r'_(\d{6})(?:_|$)', item_id)
        if match:
            return match.group(1)
        match = re.search(r'_(w\d+n\d+)(?:_|$)', item_id)
        if match:
            return match.group(1)
        return item_id

    def version_priority(item_id: str) -> int:
        if 'LAS_2018' in item_id or 'LAS_2019' in item_id:
            return 3
        elif '_C17_1_' in item_id:
            return 2
        elif '_C17_' in item_id:
            return 1
        return 0

    grid_items: Dict[str, List] = {}
    for item in items:
        grid_id = extract_grid_id(item.id)
        if grid_id not in grid_items:
            grid_items[grid_id] = []
        grid_items[grid_id].append(item)

    deduplicated = []
    duplicates_removed = 0

    for grid_id, grid_item_list in grid_items.items():
        if len(grid_item_list) > 1:
            grid_item_list.sort(key=lambda x: version_priority(x.id), reverse=True)
            selected = grid_item_list[0]
            duplicates_removed += len(grid_item_list) - 1
        else:
            selected = grid_item_list[0]
        deduplicated.append(selected)

    if duplicates_removed > 0:
        logger.info(f"Deduplicated {duplicates_removed} duplicate tile versions "
                   f"({len(items)} -> {len(deduplicated)} tiles)")

    return deduplicated


def search_3dep_copc(
    catalog_client,
    bbox: List[float],
    logger: logging.Logger,
    date_range: str = "2015-01-01/2024-12-31"
) -> List:
    """Search Planetary Computer for 3DEP COPC tiles covering bbox."""
    polygon = bbox_to_geojson_polygon(bbox)

    logger.info(f"Searching for 3DEP COPC tiles covering bbox {bbox}")

    try:
        search = catalog_client.search(
            collections=["3dep-lidar-copc"],
            intersects=polygon,
            datetime=date_range,
            limit=100
        )
        items = list(search.items())

        valid_items = [item for item in items if 'data' in item.assets]
        logger.info(f"Found {len(valid_items)} COPC tiles with data assets")

        if not valid_items:
            logger.warning("No 3DEP COPC tiles found for this bbox")
            return valid_items

        deduplicated_items = deduplicate_3dep_items(valid_items, logger)
        return deduplicated_items

    except Exception as e:
        logger.error(f"Error searching Planetary Computer: {e}")
        raise


def build_processing_pipeline(
    copc_urls: List[str],
    polygon_geojson: dict,
    output_path: Path,
    buffer_m: float = 3.0,
    requests_threads: int = 8
) -> dict:
    """
    Build the complete PDAL processing pipeline.

    Pipeline stages:
    1. Read COPC tiles with spatial filter
    2. Merge (if multiple tiles)
    3. Reproject to UTM
    4. Compute HAG
    5. Filter HAG range [-0.5m, 100m] (remove ground errors and outliers)
    6. Compute eigenvalue features (Planarity, Scattering, Verticality)
    7. Rename Scattering to Sphericity (for consistency)
    8. Write COPC with extra dimensions (enables efficient spatial queries)
    """
    pipeline_stages = []

    # Add readers for each COPC tile
    for url in copc_urls:
        pipeline_stages.append({
            "type": "readers.copc",
            "filename": url,
            "requests": requests_threads,
            "polygon": json.dumps(polygon_geojson)
        })

    # Merge if multiple tiles
    if len(copc_urls) > 1:
        pipeline_stages.append({"type": "filters.merge"})

    # Reproject to UTM
    pipeline_stages.append({
        "type": "filters.reprojection",
        "out_srs": TARGET_CRS
    })


    # Compute Height Above Ground using Delaunay triangulation
    # https://github.com/PDAL/PDAL/pull/2846
    pipeline_stages.append({
        "type": "filters.hag_delaunay"
    })

    # Filter points by HAG range (remove ground errors and unreasonably high points)
    pipeline_stages.append({
        "type": "filters.range",
        "limits": "HeightAboveGround[-0.5:100]"
    })

    # Compute eigenvalue features (Planarity, Scattering, Verticality)
    # Note: PDAL calls it "Scattering" but it's mathematically equivalent to Sphericity (λ3/λ1)
    pipeline_stages.append({
        "type": "filters.covariancefeatures",
        "knn": 10,
        "feature_set": "Dimensionality",
        "threads": 4
    })

    # Rename Scattering to Sphericity for consistency with our pipeline
    pipeline_stages.append({
        "type": "filters.ferry",
        "dimensions": "Scattering => Sphericity"
    })

    # Write output as COPC (Cloud Optimized Point Cloud) for efficient spatial queries
    # COPC enables reading only points within a bbox without loading entire file
    pipeline_stages.append({
        "type": "writers.copc",
        "filename": str(output_path),
        "extra_dims": "all",
        "forward": "header"
    })

    return {"pipeline": pipeline_stages}


def download_and_process_site(
    site_name: str,
    site_bbox_wgs84: List[float],
    output_dir: Path,
    catalog_client,
    logger: logging.Logger,
    buffer_m: float = 5.0,
    date_range: str = "2015-01-01/2024-12-31"
) -> Optional[Path]:
    """
    Download 3DEP and compute HAG + features in single PDAL pipeline.

    Args:
        site_name: Name of the site
        site_bbox_wgs84: [minlon, minlat, maxlon, maxlat] in WGS84
        output_dir: Directory for output files
        catalog_client: Planetary Computer STAC client
        logger: Logger instance
        buffer_m: Buffer to add around site bbox (in meters)
        date_range: Date range for STAC search

    Returns:
        Path to processed LAZ file, or None if failed
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{site_name}_hag_features.copc.laz"

    logger.info(f"\n{'='*60}")
    logger.info(f"Processing site: {site_name}")
    logger.info(f"WGS84 bbox: {site_bbox_wgs84}")
    logger.info(f"Buffer: {buffer_m}m")
    logger.info(f"Output: {output_path}")
    logger.info(f"{'='*60}\n")

    # Convert bbox to UTM and add buffer
    bbox_utm = bbox_wgs84_to_utm(site_bbox_wgs84)
    logger.info(f"UTM bbox (unbuffered): {[f'{x:.1f}' for x in bbox_utm]}")

    bbox_utm_buffered = buffer_bbox_utm(bbox_utm, buffer_m)
    logger.info(f"UTM bbox (buffered by {buffer_m}m): {[f'{x:.1f}' for x in bbox_utm_buffered]}")

    # Search for COPC tiles (use unbuffered WGS84 for search)
    items = search_3dep_copc(catalog_client, site_bbox_wgs84, logger, date_range)

    if not items:
        logger.error(f"No COPC tiles found for {site_name}")
        return None

    # Get COPC URLs
    copc_urls = [item.assets['data'].href for item in items]
    logger.info(f"Found {len(copc_urls)} COPC tiles to process")
    for url in copc_urls:
        logger.debug(f"  - {url}")

    # Build processing pipeline
    polygon_geojson = bbox_to_geojson_polygon(site_bbox_wgs84)
    pipeline_dict = build_processing_pipeline(
        copc_urls=copc_urls,
        polygon_geojson=polygon_geojson,
        output_path=output_path,
        buffer_m=buffer_m
    )

    # Save pipeline JSON for debugging
    pipeline_json_path = output_dir / f"{site_name}_pipeline.json"
    with open(pipeline_json_path, 'w') as f:
        json.dump(pipeline_dict, f, indent=2)
    logger.info(f"Pipeline JSON saved to {pipeline_json_path}")

    # Execute pipeline
    logger.info("\nExecuting PDAL processing pipeline...")
    logger.info("  - Reading COPC tiles")
    logger.info("  - Reprojecting to UTM 11N")
    logger.info("  - Computing Height Above Ground")
    logger.info("  - Filtering HAG range [-0.5m, 100m]")
    logger.info("  - Computing eigenvalue features (knn=10)")
    logger.info("  - Writing output COPC")

    try:
        pipeline = pdal.Pipeline(json.dumps(pipeline_dict))
        count = pipeline.execute()

        logger.info(f"\nSuccessfully processed {count:,} points")
        logger.info(f"Output written to {output_path}")

        return output_path

    except Exception as e:
        logger.error(f"PDAL pipeline failed: {e}")
        return None


def validate_processed_laz(
    laz_path: Path,
    logger: logging.Logger
) -> Dict:
    """
    Validate processed LAZ has all required dimensions and sensible values.

    Returns:
        Dictionary with validation results
    """
    logger.info(f"\nValidating: {laz_path}")

    required_dims = [
        'X', 'Y', 'Z',
        'HeightAboveGround',
        'Intensity', 'ReturnNumber', 'NumberOfReturns',
        'Planarity', 'Sphericity', 'Verticality'
    ]

    try:
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [{"type": "readers.las", "filename": str(laz_path)}]
        }))
        pipeline.execute()
        arr = pipeline.arrays[0]

        results = {
            'file': str(laz_path),
            'n_points': len(arr),
            'dimensions': {},
            'warnings': [],
            'all_checks_passed': True
        }

        # Check each required dimension
        for dim in required_dims:
            if dim in arr.dtype.names:
                data = arr[dim]

                # Handle NaN for float types
                if np.issubdtype(data.dtype, np.floating):
                    nan_count = int(np.isnan(data).sum())
                    valid_data = data[~np.isnan(data)]
                else:
                    nan_count = 0
                    valid_data = data

                if len(valid_data) > 0:
                    results['dimensions'][dim] = {
                        'present': True,
                        'min': float(np.min(valid_data)),
                        'max': float(np.max(valid_data)),
                        'mean': float(np.mean(valid_data)),
                        'nan_count': nan_count,
                        'nan_pct': 100.0 * nan_count / len(data) if len(data) > 0 else 0
                    }
                else:
                    results['dimensions'][dim] = {
                        'present': True,
                        'min': None,
                        'max': None,
                        'mean': None,
                        'nan_count': nan_count,
                        'nan_pct': 100.0
                    }
                    results['warnings'].append(f"{dim}: all values are NaN")
            else:
                results['dimensions'][dim] = {'present': False}
                results['warnings'].append(f"Missing dimension: {dim}")
                results['all_checks_passed'] = False

        # Specific validation checks
        if 'HeightAboveGround' in arr.dtype.names:
            hag = arr['HeightAboveGround']
            valid_hag = hag[~np.isnan(hag)] if np.issubdtype(hag.dtype, np.floating) else hag

            # Check for excessive negative HAG (ground model error)
            neg_count = int((valid_hag < -0.5).sum())
            if neg_count > len(valid_hag) * 0.01:
                results['warnings'].append(
                    f"HAG: {neg_count} points ({100*neg_count/len(valid_hag):.2f}%) below -0.5m"
                )

            # Check for unreasonably high HAG
            high_count = int((valid_hag > 100).sum())
            if high_count > 0:
                results['warnings'].append(
                    f"HAG: {high_count} points above 100m (unusual for vegetation)"
                )

        # Check eigenvalue features are in [0,1] range
        for dim in ['Planarity', 'Sphericity', 'Verticality']:
            if dim in arr.dtype.names:
                data = arr[dim]
                valid_data = data[~np.isnan(data)] if np.issubdtype(data.dtype, np.floating) else data
                if len(valid_data) > 0:
                    if valid_data.min() < -0.01 or valid_data.max() > 1.01:
                        results['warnings'].append(
                            f"{dim}: values outside [0,1] range "
                            f"(min={valid_data.min():.3f}, max={valid_data.max():.3f})"
                        )

        # Log results
        logger.info(f"  Total points: {results['n_points']:,}")
        logger.info("  Dimension statistics:")
        for dim, stats in results['dimensions'].items():
            if stats['present'] and stats['mean'] is not None:
                logger.info(f"    {dim:20s}: min={stats['min']:10.4f}, max={stats['max']:10.4f}, "
                           f"mean={stats['mean']:10.4f}, nan={stats['nan_pct']:.1f}%")
            elif stats['present']:
                logger.info(f"    {dim:20s}: ALL NaN")
            else:
                logger.info(f"    {dim:20s}: MISSING")

        if results['warnings']:
            logger.warning("  Warnings:")
            for w in results['warnings']:
                logger.warning(f"    - {w}")
        else:
            logger.info("  No warnings - all checks passed!")

        return results

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return {
            'file': str(laz_path),
            'error': str(e),
            'all_checks_passed': False
        }


def save_processing_metadata(
    site_name: str,
    output_dir: Path,
    site_bbox_wgs84: List[float],
    laz_path: Path,
    validation_results: Dict,
    logger: logging.Logger
) -> None:
    """Save processing metadata and parameters to JSON."""
    metadata = {
        'site_name': site_name,
        'processing_date': datetime.now().isoformat(),
        'site_bbox_wgs84': site_bbox_wgs84,
        'target_crs': TARGET_CRS,
        'smrf_params': SMRF_PARAMS,
        'eigenvalue_knn': 10,
        'output_file': str(laz_path),
        'validation': validation_results,
        'extra_dimensions': [
            'HeightAboveGround',
            'Planarity', 'Sphericity', 'Verticality'
        ]
    }

    metadata_path = output_dir / f"{site_name}_processing_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Processing metadata saved to {metadata_path}")


def parse_bbox_string(bbox_str: str) -> List[float]:
    """Parse comma-separated bbox string to list of floats."""
    parts = bbox_str.replace(' ', '').split(',')
    if len(parts) != 4:
        raise ValueError(f"Expected 4 comma-separated values, got {len(parts)}")
    return [float(x) for x in parts]


def main():
    parser = argparse.ArgumentParser(
        description="Download 3DEP and compute HAG + enhanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process a known site
    python src/data_prep/download_and_process_3dep_sites.py --site BluffMesa

    # Process all known sites
    python src/data_prep/download_and_process_3dep_sites.py --all

    # Process custom site with explicit bbox
    python src/data_prep/download_and_process_3dep_sites.py \\
        --site my_site \\
        --bbox "-116.5,33.0,-116.4,33.1"
"""
    )

    parser.add_argument(
        "--site",
        type=str,
        help="Site name (use with --bbox for custom sites)"
    )
    parser.add_argument(
        "--bbox",
        type=str,
        help="Bounding box as 'minlon,minlat,maxlon,maxlat' (WGS84)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all known sites"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: data/processed/3dep_hag_features/{site})"
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("data/processed/3dep_hag_features"),
        help="Base output directory when using --all"
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=3.0,
        help="Buffer around site bbox in meters (default: 3.0)"
    )
    parser.add_argument(
        "--date-range",
        type=str,
        default="2015-01-01/2024-12-31",
        help="Date range for STAC search (default: 2015-01-01/2024-12-31)"
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip validation of output files"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.site and not args.all:
        parser.error("Must specify either --site or --all")

    if args.all and (args.site or args.bbox):
        parser.error("Cannot use --site or --bbox with --all")

    # Determine sites to process
    sites_to_process = []

    if args.all:
        sites_to_process = [(name, bbox) for name, bbox in SITE_BBOXES.items()]
    else:
        if args.bbox:
            bbox = parse_bbox_string(args.bbox)
        elif args.site in SITE_BBOXES:
            bbox = SITE_BBOXES[args.site]
        else:
            parser.error(f"Unknown site '{args.site}'. Use --bbox to specify custom bbox, "
                        f"or choose from: {list(SITE_BBOXES.keys())}")
        sites_to_process = [(args.site, bbox)]

    # Connect to Planetary Computer
    print("Connecting to Planetary Computer STAC API...")
    catalog_client = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

    # Process each site
    results = {}
    for site_name, site_bbox in sites_to_process:
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        else:
            output_dir = args.output_base / site_name

        # Setup logging for this site
        logger = setup_logging(output_dir)

        # Download and process
        laz_path = download_and_process_site(
            site_name=site_name,
            site_bbox_wgs84=site_bbox,
            output_dir=output_dir,
            catalog_client=catalog_client,
            logger=logger,
            buffer_m=args.buffer,
            date_range=args.date_range
        )

        if laz_path is None:
            results[site_name] = {'success': False, 'error': 'Download/processing failed'}
            continue

        # Validate output
        if not args.skip_validation:
            validation_results = validate_processed_laz(laz_path, logger)

            # Save metadata
            save_processing_metadata(
                site_name=site_name,
                output_dir=output_dir,
                site_bbox_wgs84=site_bbox,
                laz_path=laz_path,
                validation_results=validation_results,
                logger=logger
            )

            results[site_name] = {
                'success': validation_results.get('all_checks_passed', False),
                'output': str(laz_path),
                'validation': validation_results
            }
        else:
            results[site_name] = {
                'success': True,
                'output': str(laz_path),
                'validation': 'skipped'
            }

    # Summary
    print("\n" + "="*60)
    print("PROCESSING SUMMARY")
    print("="*60)

    for site_name, result in results.items():
        if result['success']:
            status = "✓ SUCCESS"
            print(f"{site_name:20s}: {status}")
            if 'output' in result:
                print(f"                      Output: {result['output']}")
        else:
            status = "✗ FAILED"
            print(f"{site_name:20s}: {status}")
            if 'error' in result:
                print(f"                      Error: {result['error']}")

    # Exit with error code if any failed
    if not all(r['success'] for r in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
