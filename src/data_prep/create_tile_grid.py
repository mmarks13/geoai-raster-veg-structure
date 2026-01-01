#!/usr/bin/env python3
"""
Unified Tile Grid Generation Script

Creates a grid of tiles from either:
- Raster input (GeoTIFF): Extracts outline of valid data, generates tiles
- Vector input (GeoJSON/GeoPackage): Uses polygon boundaries, generates tiles

Supports pixel-aligned snapping to ensure tiles align with raster pixel boundaries.

Replaces both create_forest_plot_tile_grid.py and create_training_tile_bboxes.py
with a single, flexible implementation.

Usage:
    # Raster input (auto pixel alignment)
    python src/data_prep/create_tile_grid.py \
      --input data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \
      --output data/processed/tiles_raster.geojson \
      --tile-size 10.0 \
      --overlap 0.15

    # Vector input with explicit pixel alignment to a target raster
    python src/data_prep/create_tile_grid.py \
      --input data/processed/boundaries/t01_t09_boundary.geojson \
      --output data/processed/tiles/t01_t09_tiles.geojson \
      --tile-size 10.0 \
      --overlap 0.20 \
      --site-name t01_t09 \
      --raster-for-alignment data/processed/veg_structure_metrics/t01_t09/merged/t01_t09_veg_metrics_2m.tif
"""

import os
import sys
import argparse
import math
import rasterio
from rasterio import features
import geopandas as gpd
from shapely.geometry import box, shape, mapping
from shapely.ops import unary_union
import numpy as np
import logging
from pyproj import CRS
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def get_raster_pixel_grid_params(raster_path: str) -> Tuple[float, float, float, float]:
    """
    Get pixel grid parameters from a raster's transform.

    Parameters
    ----------
    raster_path : str
        Path to raster file

    Returns
    -------
    tuple
        (x_origin, y_origin, x_res, y_res)
        - x_origin: x-coordinate of upper-left corner
        - y_origin: y-coordinate of upper-left corner
        - x_res: pixel width (positive)
        - y_res: pixel height (positive, absolute value)
    """
    with rasterio.open(raster_path) as src:
        transform = src.transform
        x_origin = transform.c  # Upper-left x
        y_origin = transform.f  # Upper-left y
        x_res = transform.a     # Pixel width (positive)
        y_res = abs(transform.e)  # Pixel height (make positive)
        return x_origin, y_origin, x_res, y_res


def snap_to_raster_pixel_grid(
    coord: float,
    origin: float,
    resolution: float,
    snap_type: str = 'floor'
) -> float:
    """
    Snap a coordinate to align with a raster's actual pixel grid.

    This differs from simple resolution-divisibility because it accounts
    for the raster's origin point. A raster starting at x=533412.5 has
    pixel boundaries at 533412.5, 533414.5, 533416.5, etc. (assuming 2m res),
    NOT at 533412, 533414, 533416.

    Parameters
    ----------
    coord : float
        Coordinate to snap
    origin : float
        Raster origin (transform.c for x, transform.f for y)
    resolution : float
        Pixel size (transform.a for x, abs(transform.e) for y)
    snap_type : str
        'floor' to snap down, 'ceil' to snap up

    Returns
    -------
    float
        Snapped coordinate aligned to pixel grid
    """
    # Calculate which pixel this coordinate falls into relative to origin
    pixel_offset = (coord - origin) / resolution

    if snap_type == 'floor':
        snapped_offset = math.floor(pixel_offset)
    elif snap_type == 'ceil':
        snapped_offset = math.ceil(pixel_offset)
    else:
        raise ValueError(f"snap_type must be 'floor' or 'ceil', got {snap_type}")

    # Convert back to physical coordinates
    return origin + snapped_offset * resolution


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
    site: str,
    pixel_grid_params: Optional[Tuple[float, float, float, float]] = None
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
    pixel_grid_params : tuple, optional
        (x_origin, y_origin, x_res, y_res) from raster transform.
        If provided, tile coordinates are snapped to align with raster pixel boundaries.

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

    # Generate tiles with pixel-aligned coordinates if params provided
    if pixel_grid_params is not None:
        x_origin, y_origin, x_res, y_res = pixel_grid_params

        # Convert tile size and step to integer pixel counts
        tile_size_pixels_x = round(tile_size / x_res)
        tile_size_pixels_y = round(tile_size / y_res)
        step_pixels_x = round(step / x_res)
        step_pixels_y = round(step / y_res)

        # Compute actual tile size and step in meters (pixel-aligned)
        tile_size_x = tile_size_pixels_x * x_res
        tile_size_y = tile_size_pixels_y * y_res
        step_x = step_pixels_x * x_res
        step_y = step_pixels_y * y_res

        logger.debug(f"Pixel-aligned: tile={tile_size_pixels_x}x{tile_size_pixels_y}px, step={step_pixels_x}x{step_pixels_y}px")

        # Find starting pixel indices that cover the bounding box
        # Start from the first pixel boundary <= minx, miny
        start_pixel_x = math.floor((minx - x_origin) / x_res)
        start_pixel_y = math.floor((miny - y_origin) / y_res)

        # Find ending pixel indices
        end_pixel_x = math.ceil((maxx - x_origin) / x_res)
        end_pixel_y = math.ceil((maxy - y_origin) / y_res)

        # Generate all tiles on the pixel grid, then filter by boundary
        tiles = []
        pixel_x = start_pixel_x
        while pixel_x * x_res + x_origin + tile_size_x <= maxx + x_res:
            pixel_y = start_pixel_y
            while pixel_y * y_res + y_origin + tile_size_y <= maxy + y_res:
                # Compute tile coordinates from pixel indices
                x = x_origin + pixel_x * x_res
                y = y_origin + pixel_y * y_res

                # Create candidate tile
                tile_geom = box(x, y, x + tile_size_x, y + tile_size_y)

                # Check if completely within boundary (strict containment)
                if polygon.contains(tile_geom):
                    xmin_int = int(round(x))
                    ymin_int = int(round(y))
                    tile_id = f"{site}_{xmin_int}_{ymin_int}"

                    tiles.append({
                        'tile_id': tile_id,
                        'site': site,
                        'geometry': tile_geom
                    })

                pixel_y += step_pixels_y
            pixel_x += step_pixels_x
    else:
        # Non-aligned mode: use floating-point step
        start_x = minx
        start_y = miny

        tiles = []
        x = start_x

        while x + tile_size <= maxx:
            y = start_y
            while y + tile_size <= maxy:
                tile_geom = box(x, y, x + tile_size, y + tile_size)

                if polygon.contains(tile_geom):
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
    overlap_ratio: float,
    site_name_override: Optional[str] = None,
    raster_for_alignment: Optional[str] = None
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
    site_name_override : str, optional
        Override automatic site name extraction
    raster_for_alignment : str, optional
        Path to raster for pixel grid alignment (uses input_path if not specified)

    Returns
    -------
    list
        List of tile dicts
    """
    logger.info(f"Processing raster input: {input_path}")

    # Extract or use override site name
    if site_name_override:
        site = site_name_override
        logger.info(f"Using provided site name: {site}")
    else:
        site = extract_site_from_raster_filename(input_path)
        logger.info(f"Extracted site name: {site}")

    # Extract outline
    logger.info("Extracting raster outline...")
    outline, crs = extract_raster_outline(input_path)
    logger.info(f"Outline extracted. CRS: {crs}")

    # Get pixel grid params for alignment
    alignment_raster = raster_for_alignment or input_path
    pixel_grid_params = get_raster_pixel_grid_params(alignment_raster)
    x_origin, y_origin, x_res, y_res = pixel_grid_params
    logger.info(f"Pixel grid alignment: origin=({x_origin:.2f}, {y_origin:.2f}), res=({x_res:.3f}, {y_res:.3f})")

    # Generate tiles with pixel alignment
    logger.info(f"Generating tiles (size={tile_size}m, overlap={overlap_ratio})...")
    tiles = create_tiles_from_polygon(outline, tile_size, overlap_ratio, site, pixel_grid_params)

    return tiles


def process_vector_input(
    input_path: str,
    tile_size: float,
    overlap_ratio: float,
    site_name_override: Optional[str] = None,
    raster_for_alignment: Optional[str] = None
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
    site_name_override : str, optional
        Override site name (uses this for ALL polygons)
    raster_for_alignment : str, optional
        Path to raster for pixel grid alignment

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

    # Determine site name source
    if site_name_override:
        use_override = True
        logger.info(f"Using provided site name for all polygons: {site_name_override}")
    elif 'site' in gdf.columns:
        use_override = False
        logger.info("Using 'site' column from vector file")
    else:
        raise ValueError(
            f"Vector input missing 'site' column and no --site-name provided.\n"
            f"Available columns: {list(gdf.columns)}\n"
            f"Either add a 'site' column or use --site-name argument."
        )

    logger.info(f"Found {len(gdf)} polygons")

    # Get pixel grid params if raster provided
    pixel_grid_params = None
    if raster_for_alignment:
        pixel_grid_params = get_raster_pixel_grid_params(raster_for_alignment)
        x_origin, y_origin, x_res, y_res = pixel_grid_params
        logger.info(f"Pixel grid alignment: origin=({x_origin:.2f}, {y_origin:.2f}), res=({x_res:.3f}, {y_res:.3f})")

    # Process each polygon
    all_tiles = []
    for idx, row in gdf.iterrows():
        site = site_name_override if use_override else row['site']
        polygon = row['geometry']

        logger.info(f"Processing polygon {idx+1}/{len(gdf)}: site={site}")

        tiles = create_tiles_from_polygon(polygon, tile_size, overlap_ratio, site, pixel_grid_params)
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
    parser.add_argument(
        '--site-name',
        type=str,
        default=None,
        help='Override site name (required for vector input without site column)'
    )
    parser.add_argument(
        '--raster-for-alignment',
        type=str,
        default=None,
        help='Path to raster for pixel grid alignment. Tiles will be snapped to this raster\'s pixel boundaries.'
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

    # Validate raster-for-alignment if provided
    if args.raster_for_alignment and not os.path.exists(args.raster_for_alignment):
        logger.error(f"Raster for alignment not found: {args.raster_for_alignment}")
        sys.exit(1)

    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

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
            tiles = process_raster_input(
                args.input,
                args.tile_size,
                args.overlap,
                site_name_override=args.site_name,
                raster_for_alignment=args.raster_for_alignment
            )
        else:  # vector
            tiles = process_vector_input(
                args.input,
                args.tile_size,
                args.overlap,
                site_name_override=args.site_name,
                raster_for_alignment=args.raster_for_alignment
            )
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
