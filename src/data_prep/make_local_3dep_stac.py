#!/usr/bin/env python
"""
Create local STAC catalogs for 3DEP point cloud data.

Supports two modes:
1. Download mode (default): Query Planetary Computer, download, crop, and create STAC catalog
2. Processed mode: Create STAC catalog from locally processed LAZ files with HAG/features

Usage:
  # Download mode (original behavior)
  python make_local_3dep_stac.py --bbox minx miny maxx maxy --start YYYY-MM-DD --end YYYY-MM-DD

  # Processed mode (for HAG-processed LAZ files)
  python make_local_3dep_stac.py --mode processed --input-dir data/processed/3dep_hag_features --output data/stac/3dep_hag
"""

import argparse
import os
import json
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pdal
import numpy as np
from shapely.geometry import mapping, box

import pystac
import pystac_client
import planetary_computer
from pystac import Catalog, CatalogType, Item, Asset


def bounding_box_to_geojson(bbox):
    """Convert [minx, miny, maxx, maxy] to a GeoJSON Polygon."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [bbox[0], bbox[1]],
            [bbox[2], bbox[1]],
            [bbox[2], bbox[3]],
            [bbox[0], bbox[3]],
            [bbox[0], bbox[1]]
        ]]
    }


def load_or_create_catalog(output_dir):
    """
    Loads an existing STAC catalog from the output directory if it exists,
    otherwise creates a new one.
    """
    catalog_file = os.path.join(output_dir, "catalog.json")
    if os.path.exists(catalog_file):
        print("Existing catalog found; loading catalog from", catalog_file)
        return pystac.read_file(catalog_file)
    else:
        print("No existing catalog found; creating a new catalog.")
        return Catalog(
            id="3dep_lidar_copc_local_catalog",
            description="Local STAC catalog for processed 3DEP Lidar COPC point clouds",
            title="3DEP Lidar COPC Local Catalog"
        )


def process_bbox(bbox, date_range, output_dir, stac_catalog, catalog_client, threads, target_crs):
    """
    For a given bounding box and date range:
      - Query the Planetary Computer for COPC point cloud items intersecting the area.
      - For each item that has a "data" asset, build and execute a PDAL pipeline that:
            * Reads the asset,
            * Crops to the bounding box,
            * Optionally reprojects to a target CRS,
            * Writes the processed point cloud to a local COPC file.
      - Create a new STAC item referencing the processed file.
    """
    # Create GeoJSON polygon from bbox.
    polygon = bounding_box_to_geojson(bbox)
    print(f"\nSearching for 3DEP Lidar COPC point clouds intersecting bbox {bbox} and date range {date_range}...")
    
    # Query the STAC API.
    search = catalog_client.search(
        collections=["3dep-lidar-copc"],
        intersects=polygon,
        datetime=date_range,
        limit=100
    )
    items = list(search.items())
    if not items:
        print("No 3DEP Lidar COPC point clouds found for this bbox and date range.")
        return

    # Process each item separately.
    for item in items:
        if 'data' not in item.assets:
            print(f"Item {item.id} does not have a 'data' asset; skipping.")
            continue
        
        asset_url = item.assets['data'].href
        print(f"\nProcessing item {item.id} with asset: {asset_url}")
        
        # Build the PDAL pipeline for this item.
        pipeline_stages = [
            {
                "type": "readers.copc",
                "filename": asset_url,
                "threads": threads,
                "polygon": json.dumps(polygon)
            },
            {
                "type": "filters.crop",
                "polygon": json.dumps(polygon)
            }
        ]
        
        if target_crs:
            pipeline_stages.append({
                "type": "filters.reprojection",
                "out_srs": target_crs
            })
        
        # Define the output filename using the item ID.
        output_filename = os.path.join(output_dir, f"{item.id}.copc")
        pipeline_stages.append({
            "type": "writers.copc",
            "filename": output_filename
        })
        
        pipeline_dict = {"pipeline": pipeline_stages}
        pipeline_json = json.dumps(pipeline_dict, indent=2)
        print("PDAL pipeline:")
        print(pipeline_json)
        
        try:
            pipeline = pdal.Pipeline(pipeline_json)
            pipeline.execute()
            print(f"Processed COPC file written to {output_filename}")
        except Exception as e:
            print(f"Error processing point cloud for item {item.id}: {e}")
            continue
        finally:
            del pipeline
            gc.collect()
        
        # Create a new STAC item for the processed COPC file.
        stac_item = Item(
            id=item.id,
            geometry=item.geometry,
            bbox=item.bbox,
            datetime=item.datetime,
            properties=item.properties,
        )
        
        asset = Asset(
            href=output_filename,
            media_type="application/octet-stream",
            roles=["data"],
            title="Processed 3DEP Lidar COPC Point Cloud"
        )
        stac_item.add_asset("data", asset)
        
        if stac_catalog.get_item(stac_item.id) is None:
            stac_catalog.add_item(stac_item)
            print(f"STAC item {stac_item.id} added to catalog.")
        else:
            print(f"STAC item {stac_item.id} already exists in the catalog; skipping addition.")


def get_laz_metadata(laz_path: Path) -> Dict:
    """
    Read metadata from a LAZ file using PDAL.

    Returns:
        Dict with bbox, point_count, crs, and available dimensions
    """
    pipeline = pdal.Pipeline(json.dumps({
        "pipeline": [{"type": "readers.las", "filename": str(laz_path)}]
    }))
    pipeline.execute()

    arr = pipeline.arrays[0]
    metadata_str = pipeline.metadata
    metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str

    # Get bounds from the data
    X = arr['X']
    Y = arr['Y']
    Z = arr['Z']

    bbox = [float(X.min()), float(Y.min()), float(X.max()), float(Y.max())]

    # Get CRS from metadata
    las_metadata = metadata.get("metadata", {}).get("readers.las", {})
    srs_wkt = las_metadata.get("srs", {}).get("wkt", "")

    # Get available dimensions
    dimensions = list(arr.dtype.names)

    return {
        'bbox': bbox,
        'point_count': len(arr),
        'crs_wkt': srs_wkt,
        'dimensions': dimensions,
        'z_min': float(Z.min()),
        'z_max': float(Z.max())
    }


def create_stac_item_from_processed_laz(
    laz_path: Path,
    site_name: str,
    processing_metadata_path: Optional[Path] = None
) -> Item:
    """
    Create STAC item from locally processed LAZ file with HAG features.

    Args:
        laz_path: Path to processed LAZ file
        site_name: Name of the site (used as item ID)
        processing_metadata_path: Optional path to JSON with processing parameters

    Returns:
        pystac.Item with proper geometry, bbox, and custom properties
    """
    # Get LAZ metadata
    laz_meta = get_laz_metadata(laz_path)

    # Create geometry from bbox
    minx, miny, maxx, maxy = laz_meta['bbox']
    geometry = {
        "type": "Polygon",
        "coordinates": [[
            [minx, miny],
            [maxx, miny],
            [maxx, maxy],
            [minx, maxy],
            [minx, miny]
        ]]
    }

    # Load processing metadata if available
    processing_params = {}
    if processing_metadata_path and processing_metadata_path.exists():
        with open(processing_metadata_path) as f:
            processing_params = json.load(f)

    # Build properties
    properties = {
        'geoai:has_hag': 'HeightAboveGround' in laz_meta['dimensions'],
        'geoai:has_eigenvalues': all(d in laz_meta['dimensions'] for d in ['Planarity', 'Sphericity', 'Verticality']),
        'geoai:has_points_above': 'PointsAbove' in laz_meta['dimensions'],
        'geoai:has_return_ratio': 'ReturnRatio' in laz_meta['dimensions'],
        'geoai:point_count': laz_meta['point_count'],
        'geoai:dimensions': laz_meta['dimensions'],
        'geoai:z_range': [laz_meta['z_min'], laz_meta['z_max']],
    }

    # Add processing parameters if available
    if processing_params:
        if 'smrf_params' in processing_params:
            properties['geoai:smrf_params'] = processing_params['smrf_params']
        if 'processing_date' in processing_params:
            properties['geoai:processing_date'] = processing_params['processing_date']

    # Create STAC item
    item = Item(
        id=f"3dep_hag_{site_name}",
        geometry=geometry,
        bbox=laz_meta['bbox'],
        datetime=datetime.now(),
        properties=properties
    )

    # Add the LAZ file as the data asset
    asset = Asset(
        href=str(laz_path.absolute()),
        media_type="application/vnd.laszip",
        roles=["data"],
        title=f"3DEP with HAG features - {site_name}"
    )
    item.add_asset("data", asset)

    return item


def create_catalog_from_processed_files(input_dir: Path, output_dir: Path) -> Catalog:
    """
    Create STAC catalog from directory of processed LAZ files.

    Expected directory structure:
        input_dir/
            site1/
                site1_hag_features.copc.laz
                site1_processing_metadata.json (optional)
            site2/
                ...

    Args:
        input_dir: Directory containing processed site subdirectories
        output_dir: Directory to save the STAC catalog

    Returns:
        pystac.Catalog
    """
    catalog = Catalog(
        id="3dep_hag_local_catalog",
        description="Local STAC catalog for 3DEP point clouds with HAG and enhanced features (COPC format)",
        title="3DEP HAG Features Local Catalog"
    )

    # Find all processed COPC files
    laz_files = list(input_dir.glob("**/*_hag_features.copc.laz"))

    if not laz_files:
        # Try looking for any LAZ/COPC files
        laz_files = list(input_dir.glob("**/*.laz"))

    print(f"Found {len(laz_files)} LAZ files to catalog")

    for laz_path in laz_files:
        # Determine site name from parent directory
        site_name = laz_path.parent.name

        # Look for processing metadata
        metadata_path = laz_path.parent / f"{site_name}_processing_metadata.json"

        print(f"Processing {site_name}: {laz_path}")

        try:
            item = create_stac_item_from_processed_laz(
                laz_path=laz_path,
                site_name=site_name,
                processing_metadata_path=metadata_path if metadata_path.exists() else None
            )

            # Check if item already exists
            if catalog.get_item(item.id) is None:
                catalog.add_item(item)
                print(f"  Added STAC item: {item.id}")

                # Print feature summary
                props = item.properties
                features = []
                if props.get('geoai:has_hag'):
                    features.append('HAG')
                if props.get('geoai:has_eigenvalues'):
                    features.append('Eigenvalues')
                if props.get('geoai:has_points_above'):
                    features.append('PointsAbove')
                if props.get('geoai:has_return_ratio'):
                    features.append('ReturnRatio')
                print(f"  Features: {', '.join(features) if features else 'None'}")
            else:
                print(f"  Item {item.id} already exists; skipping")

        except Exception as e:
            print(f"  Error processing {laz_path}: {e}")
            continue

    return catalog


def main():
    parser = argparse.ArgumentParser(
        description="Create local STAC catalogs for 3DEP point cloud data. "
                    "Supports both downloading from Planetary Computer and cataloging processed local files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download mode (original behavior)
    python make_local_3dep_stac.py --bbox -116.5 33.0 -116.4 33.1 --start 2015-01-01 --end 2024-12-31

    # Processed mode (for HAG-processed LAZ files)
    python make_local_3dep_stac.py --mode processed --input-dir data/processed/3dep_hag_features --output data/stac/3dep_hag
"""
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["download", "processed"],
        default="download",
        help="Mode: 'download' to query Planetary Computer, 'processed' to catalog local LAZ files"
    )

    # Arguments for processed mode
    parser.add_argument(
        "--input-dir",
        type=Path,
        help="Input directory containing processed LAZ files (required for --mode processed)"
    )

    # Arguments for download mode (original behavior)
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("minx", "miny", "maxx", "maxy"),
        action="append",
        help="Bounding box coordinates (minx miny maxx maxy). Use multiple --bbox for multiple areas."
    )
    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD) for the date range."
    )
    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD) for the date range."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/stac/3dep_local/",
        help="Directory to write the local STAC catalog."
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Number of threads to use in PDAL processing."
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default="",
        help="Optional target CRS for reprojection (e.g., 'EPSG:4326')."
    )
    args = parser.parse_args()

    # Handle processed mode
    if args.mode == "processed":
        if not args.input_dir:
            parser.error("--input-dir is required for --mode processed")

        if not args.input_dir.exists():
            parser.error(f"Input directory does not exist: {args.input_dir}")

        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Creating STAC catalog from processed LAZ files in {args.input_dir}")

        catalog = create_catalog_from_processed_files(args.input_dir, output_dir)

        # Save catalog
        catalog.normalize_and_save(str(output_dir), catalog_type=CatalogType.SELF_CONTAINED)
        print(f"\nLocal STAC catalog saved to {output_dir}")
        print(f"Total items: {len(list(catalog.get_all_items()))}")

    # Handle download mode (original behavior)
    else:
        if not args.bbox:
            parser.error("--bbox is required for download mode")
        if not args.start or not args.end:
            parser.error("--start and --end are required for download mode")

        os.makedirs(args.output, exist_ok=True)
        date_range = f"{args.start}/{args.end}"

        # Open the Planetary Computer STAC API client.
        catalog_client = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )

        # Load (or create) the local STAC catalog.
        stac_catalog = load_or_create_catalog(args.output)

        # Process each bounding box.
        for bbox in args.bbox:
            process_bbox(bbox, date_range, args.output, stac_catalog, catalog_client, args.threads, args.target_crs)

        # Save the updated catalog.
        stac_catalog.normalize_and_save(args.output, catalog_type=CatalogType.SELF_CONTAINED)
        print(f"\nLocal STAC catalog saved to {args.output}")


if __name__ == "__main__":
    main()
