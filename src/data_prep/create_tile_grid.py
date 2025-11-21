#!/usr/bin/env python3
"""
Unified Tile Grid Generation Script

Creates a grid of tiles from either:
- Raster input (GeoTIFF): Extracts outline of valid data, generates tiles
- Vector input (GeoJSON/GeoPackage): Uses polygon boundaries, generates tiles

Replaces both create_forest_plot_tile_grid.py and create_training_tile_bboxes.py
with a single, flexible implementation.

Usage:
    # Raster input (fuel metrics)
    python src/data_prep/create_tile_grid.py \
      --input data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \
      --output data/processed/tiles_raster.geojson \
      --tile-size 10.0 \
      --overlap 0.15

    # Vector input (forest plots)
    python src/data_prep/create_tile_grid.py \
      --input data/processed/forest_plot_sites.gpkg \
      --output data/processed/forest_plot_tiles.geojson \
      --tile-size 10.0 \
      --overlap 0.0
"""

import os
import sys
import argparse
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union
import numpy as np
import logging
from pyproj import CRS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def detect_input_type(input_path: str) -> str:
    """
    Auto-detect whether input is raster or vector.

    Parameters
    ----------
    input_path : str
        Path to input file

    Returns
    -------
    str
        'raster' or 'vector'

    Raises
    ------
    ValueError
        If file cannot be opened as either raster or vector
    """
    # Try rasterio first (for GeoTIFFs)
    try:
        with rasterio.open(input_path) as src:
            _ = src.crs
        return 'raster'
    except Exception:
        pass

    # Try geopandas (for GeoJSON, GeoPackage, etc.)
    try:
        gdf = gpd.read_file(input_path)
        if len(gdf) > 0:
            return 'vector'
    except Exception:
        pass

    # Both failed
    raise ValueError(
        f"Could not open '{input_path}' as raster (rasterio) or vector (geopandas). "
        f"Ensure file exists and is a valid GeoTIFF, GeoJSON, or GeoPackage."
    )


def extract_site_from_raster_filename(input_path: str) -> str:
    """
    Extract site name from fuel metrics raster filename.

    Expected format: {site_name}_fuel_metrics_{resolution}.tif
    Example: volcan_mtn_fuel_metrics_2.0m.tif → volcan_mtn

    Parameters
    ----------
    input_path : str
        Path to raster file

    Returns
    -------
    str
        Site name

    Raises
    ------
    ValueError
        If filename doesn't contain '_fuel_metrics'
    """
    basename = os.path.basename(input_path)

    if '_fuel_metrics' not in basename:
        raise ValueError(
            f"Raster filename does not contain '_fuel_metrics': {basename}\n"
            f"Expected format: {{site_name}}_fuel_metrics_{{resolution}}.tif\n"
            f"Example: volcan_mtn_fuel_metrics_2.0m.tif"
        )

    site = basename.split('_fuel_metrics')[0]
    return site


def extract_raster_outline(raster_path: str) -> tuple:
    """
    Extract outline of valid data from a raster file.

    Parameters
    ----------
    raster_path : str
        Path to raster file

    Returns
    -------
    tuple
        (outline_polygon, crs_string)

    Raises
    ------
    ValueError
        If CRS is not EPSG:32611 or if no valid data found
    """
    with rasterio.open(raster_path) as src:
        # Validate CRS - must be based on EPSG:32611 (WGS 84 / UTM zone 11N)
        # Accept both 2D (EPSG:32611) and 3D variants (compound CRS based on 32611)
        actual_crs = CRS.from_user_input(src.crs)

        # Try to get EPSG code (works for standard EPSG CRS)
        epsg_code = actual_crs.to_epsg()

        # Check if it's EPSG:32611 or derived from it
        is_valid_crs = False
        if epsg_code == 32611:
            is_valid_crs = True
        elif 'UTM zone 11N' in actual_crs.name and 'WGS 84' in actual_crs.name:
            # 3D variant or compound CRS based on EPSG:32611
            is_valid_crs = True
            logger.info(f"Detected 3D CRS variant of EPSG:32611: {actual_crs.name}")

        if not is_valid_crs:
            raise ValueError(
                f"\n❌ CRS MISMATCH!\n"
                f"Expected: EPSG:32611 (WGS 84 / UTM zone 11N) or 3D variant\n"
                f"Got: {actual_crs.name} (EPSG:{epsg_code})\n"
                f"Reproject input file to EPSG:32611 before processing."
            )

        # Read first band as mask
        image = src.read(1)

        # Create mask of valid data
        if src.nodata is not None:
            mask = image != src.nodata
        else:
            mask = image != 0

        # Generate polygons from mask
        transform = src.transform
        shapes_gen = features.shapes(
            mask.astype(np.uint8),
            mask=mask,
            transform=transform
        )

        # Collect all valid polygons
        all_polygons = []
        for geom, value in shapes_gen:
            if value == 1:  # Valid data
                polygon = shape(geom)
                if not polygon.is_empty:
                    all_polygons.append(polygon)

        if not all_polygons:
            raise ValueError(f"No valid data found in raster: {raster_path}")

        # Union all polygons
        if len(all_polygons) == 1:
            outline = all_polygons[0]
        else:
            logger.warning(f"Raster has {len(all_polygons)} disconnected regions. Merging into single outline.")
            outline = unary_union(all_polygons)

        return outline, 'EPSG:32611'


def create_tiles_from_polygon(
    polygon,
    tile_size: float,
    overlap_ratio: float,
    site: str
) -> list:
    """
    Create tiles within a polygon boundary.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Boundary polygon
    tile_size : float
        Tile size in meters
    overlap_ratio : float
        Overlap ratio (0.0 to 1.0)
    site : str
        Site name for tile IDs

    Returns
    -------
    list
        List of dicts with 'tile_id', 'site', and 'geometry'
    """
    # Get bounding box
    minx, miny, maxx, maxy = polygon.bounds

    # Calculate step size
    if not (0 <= overlap_ratio < 1):
        raise ValueError(f"overlap_ratio must be in [0, 1), got {overlap_ratio}")

    step = tile_size * (1 - overlap_ratio)

    # Generate tiles
    tiles = []
    x = minx

    while x + tile_size <= maxx:
        y = miny
        while y + tile_size <= maxy:
            # Create candidate tile
            tile_geom = box(x, y, x + tile_size, y + tile_size)

            # Check if completely within boundary (strict containment)
            if polygon.contains(tile_geom):
                # Create tile ID from top-left corner
                xmin_int = int(round(x))
                ymin_int = int(round(y))
                tile_id = f"{site}_{xmin_int}_{ymin_int}"

                tiles.append({
                    'tile_id': tile_id,
                    'site': site,
                    'geometry': tile_geom
                })

            y += step
        x += step

    return tiles


def process_raster_input(
    input_path: str,
    tile_size: float,
    overlap_ratio: float
) -> list:
    """
    Process raster input to generate tiles.

    Parameters
    ----------
    input_path : str
        Path to raster file
    tile_size : float
        Tile size in meters
    overlap_ratio : float
        Overlap ratio

    Returns
    -------
    list
        List of tile dicts
    """
    logger.info(f"Processing raster input: {input_path}")

    # Extract site name
    site = extract_site_from_raster_filename(input_path)
    logger.info(f"Extracted site name: {site}")

    # Extract outline
    logger.info("Extracting raster outline...")
    outline, crs = extract_raster_outline(input_path)
    logger.info(f"Outline extracted. CRS: {crs}")

    # Generate tiles
    logger.info(f"Generating tiles (size={tile_size}m, overlap={overlap_ratio})...")
    tiles = create_tiles_from_polygon(outline, tile_size, overlap_ratio, site)

    return tiles


def process_vector_input(
    input_path: str,
    tile_size: float,
    overlap_ratio: float
) -> list:
    """
    Process vector input to generate tiles.

    Parameters
    ----------
    input_path : str
        Path to vector file
    tile_size : float
        Tile size in meters
    overlap_ratio : float
        Overlap ratio

    Returns
    -------
    list
        List of tile dicts
    """
    logger.info(f"Processing vector input: {input_path}")

    # Read vector file
    gdf = gpd.read_file(input_path)

    # Validate CRS using pyproj for robust comparison
    expected_crs = CRS.from_epsg(32611)  # WGS 84 / UTM zone 11N
    actual_crs = CRS.from_user_input(gdf.crs)

    if not actual_crs.equals(expected_crs):
        raise ValueError(
            f"\n❌ CRS MISMATCH!\n"
            f"Expected: EPSG:32611 (WGS 84 / UTM zone 11N)\n"
            f"Got: {actual_crs}\n"
            f"Reproject input file to EPSG:32611 before processing."
        )

    # Check for 'site' column
    if 'site' not in gdf.columns:
        raise ValueError(
            f"Vector input missing 'site' column.\n"
            f"Available columns: {list(gdf.columns)}\n"
            f"Add a 'site' column with site names for each polygon."
        )

    logger.info(f"Found {len(gdf)} polygons")

    # Process each polygon
    all_tiles = []
    for idx, row in gdf.iterrows():
        site = row['site']
        polygon = row['geometry']

        logger.info(f"Processing polygon {idx+1}/{len(gdf)}: site={site}")

        tiles = create_tiles_from_polygon(polygon, tile_size, overlap_ratio, site)
        all_tiles.extend(tiles)

        logger.info(f"  Generated {len(tiles)} tiles")

    return all_tiles


def main():
    parser = argparse.ArgumentParser(
        description="Generate tile grid from raster or vector input",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Raster input (fuel metrics)
  python %(prog)s \\
    --input data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \\
    --output data/processed/tiles_raster.geojson \\
    --tile-size 10.0 \\
    --overlap 0.15

  # Vector input (forest plots)
  python %(prog)s \\
    --input data/processed/forest_plot_sites.gpkg \\
    --output data/processed/forest_plot_tiles.geojson \\
    --tile-size 10.0 \\
    --overlap 0.0
        """
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to input raster (GeoTIFF) or vector (GeoJSON/GeoPackage) file'
    )
    parser.add_argument(
        '--output',
        required=True,
        help='Output GeoJSON path'
    )
    parser.add_argument(
        '--tile-size',
        type=float,
        default=10.0,
        help='Tile size in meters (default: 10.0)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.15,
        help='Overlap ratio 0.0-1.0 (default: 0.15)'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    if args.tile_size <= 0:
        logger.error(f"Tile size must be positive, got {args.tile_size}")
        sys.exit(1)

    if not (0 <= args.overlap < 1):
        logger.error(f"Overlap must be in [0, 1), got {args.overlap}")
        sys.exit(1)

    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Detect input type
    logger.info("="*80)
    logger.info("UNIFIED TILE GRID GENERATION")
    logger.info("="*80)

    try:
        input_type = detect_input_type(args.input)
        logger.info(f"Detected input type: {input_type}")
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Process based on type
    try:
        if input_type == 'raster':
            tiles = process_raster_input(args.input, args.tile_size, args.overlap)
        else:  # vector
            tiles = process_vector_input(args.input, args.tile_size, args.overlap)
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Check if any tiles were generated
    if not tiles:
        logger.error("No tiles generated within boundary. Check input data coverage.")
        sys.exit(1)

    if len(tiles) < 10:
        logger.warning(f"Only {len(tiles)} tiles generated. Expected more for typical use cases.")

    logger.info(f"Total tiles generated: {len(tiles)}")

    # Convert to GeoDataFrame
    geometries = [tile['geometry'] for tile in tiles]
    properties = [{'tile_id': tile['tile_id'], 'site': tile['site']} for tile in tiles]

    gdf = gpd.GeoDataFrame(
        properties,
        geometry=geometries,
        crs='EPSG:32611'
    )

    # Save to GeoJSON
    logger.info(f"Saving to {args.output}...")
    gdf.to_file(args.output, driver='GeoJSON')

    logger.info("="*80)
    logger.info("TILE GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Output: {args.output}")
    logger.info(f"Tiles: {len(tiles)}")
    logger.info(f"Tile size: {args.tile_size}m")
    logger.info(f"Overlap: {args.overlap * 100:.1f}%")

    # Show sample tile IDs
    sample_size = min(5, len(tiles))
    logger.info(f"\nSample tile IDs (first {sample_size}):")
    for tile in tiles[:sample_size]:
        logger.info(f"  {tile['tile_id']}")


if __name__ == '__main__':
    main()
