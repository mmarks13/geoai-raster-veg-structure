#!/usr/bin/env python3
"""
Create a grid of 10x10m tiles covering the convex hull polygons of forest plot sites.

This script generates GeoJSON tiles for forest plot locations to be used
as test-only data (no UAV LiDAR ground truth available).
"""

import argparse
import geopandas as gpd
from shapely.geometry import box
from pathlib import Path
import numpy as np


def create_tile_grid_from_polygon(
    polygon_gpkg: str,
    output_geojson: str,
    tile_size: float = 10.0,
    stride: float = None,
    site_filter: str = None
) -> None:
    """
    Create a grid of square tiles covering convex hull polygons.

    Parameters
    ----------
    polygon_gpkg : str
        Path to GeoPackage with site polygons
    output_geojson : str
        Path to output GeoJSON file
    tile_size : float
        Side length of square tiles in meters (default: 10.0)
    stride : float, optional
        Distance between tile origins in meters. If None, defaults to tile_size
        (no overlap). For 20% overlap on 10m tiles, use stride=8.0
    site_filter : str, optional
        If provided, only process this site
    """
    # Use tile_size as default stride (no overlap)
    if stride is None:
        stride = tile_size

    # Calculate overlap percentage for logging
    overlap_pct = (1 - stride / tile_size) * 100 if stride < tile_size else 0
    # Target CRS - must match training data
    target_crs = "EPSG:32611"
    
    # Read site polygons
    print(f"Reading site polygons from: {polygon_gpkg}")
    gdf_sites = gpd.read_file(polygon_gpkg)
    
    # Reproject to target CRS FIRST, before any coordinate operations
    # This ensures we work in the same CRS as training data throughout
    if str(gdf_sites.crs) != target_crs:
        print(f"Reprojecting from {gdf_sites.crs} to {target_crs}...")
        gdf_sites = gdf_sites.to_crs(target_crs)
    
    # Filter by site if requested
    if site_filter:
        print(f"Filtering for site: {site_filter}")
        gdf_sites = gdf_sites[gdf_sites['Site'] == site_filter].copy()
    
    print(f"\nProcessing {len(gdf_sites)} site(s)")
    for idx, row in gdf_sites.iterrows():
        print(f"  {row['Site']:15s} | Area: {row['area_ha']:7.2f} ha | Plots: {row['plot_count']:3d}")
    
    # Create tiles for each site
    all_tiles = []
    
    for idx, site_row in gdf_sites.iterrows():
        site_name = site_row['Site']
        site_poly = site_row['geometry']
        
        print(f"\nGenerating tiles for {site_name}...")
        
        # Get bounding box of the polygon (now in EPSG:32611)
        raw_minx, raw_miny, raw_maxx, raw_maxy = site_poly.bounds
        
        # Snap bounding box to 0.5m grid to ensure exact tile dimensions
        # This prevents floating-point precision issues that cause 41x40 pixel images
        # Training data uses exact coordinates like 537540.5, 537550.5 (10m tiles)
        # and NAIP bbox 537535.5, 537555.5 (20m, centered on tile)
        grid_snap = 0.5  # 0.5m grid alignment
        minx = np.floor(raw_minx / grid_snap) * grid_snap
        miny = np.floor(raw_miny / grid_snap) * grid_snap
        maxx = np.ceil(raw_maxx / grid_snap) * grid_snap
        maxy = np.ceil(raw_maxy / grid_snap) * grid_snap
        
        # Calculate number of tiles needed in each direction
        # With overlap, we step by stride (not tile_size), but ensure last tile fits
        nx = int(np.ceil((maxx - minx - tile_size) / stride)) + 1
        ny = int(np.ceil((maxy - miny - tile_size) / stride)) + 1

        print(f"  Bounding box: ({minx:.1f}, {miny:.1f}) to ({maxx:.1f}, {maxy:.1f})")
        print(f"  Grid origin snapped to 0.5m grid for NAIP alignment")
        print(f"  Stride: {stride}m (overlap: {overlap_pct:.0f}%)")
        print(f"  Grid dimensions: {nx} × {ny} = {nx * ny:,} potential tiles")
        
        # Generate grid of tiles
        site_tiles = []
        tiles_inside = 0
        
        for i in range(nx):
            for j in range(ny):
                # Calculate tile bounds using stride for positioning
                # Tiles are still tile_size × tile_size, but origins are stride apart
                tile_minx = minx + (i * stride)
                tile_miny = miny + (j * stride)
                tile_maxx = tile_minx + tile_size
                tile_maxy = tile_miny + tile_size
                
                # Verify exact dimensions (should be exactly tile_size)
                assert abs((tile_maxx - tile_minx) - tile_size) < 1e-10, \
                    f"Tile width {tile_maxx - tile_minx} != {tile_size}"
                assert abs((tile_maxy - tile_miny) - tile_size) < 1e-10, \
                    f"Tile height {tile_maxy - tile_miny} != {tile_size}"
                
                # Create tile geometry
                tile_geom = box(tile_minx, tile_miny, tile_maxx, tile_maxy)

                # Only keep tiles that are completely within the site polygon
                if site_poly.contains(tile_geom):
                    tiles_inside += 1
                    
                    # Calculate tile center
                    center_x = (tile_minx + tile_maxx) / 2
                    center_y = (tile_miny + tile_maxy) / 2
                    
                    # Create tile properties
                    tile_props = {
                        'tile_id': f"{site_name}_{i:04d}_{j:04d}",
                        'site': site_name,
                        'grid_i': i,
                        'grid_j': j,
                        'center_x': center_x,
                        'center_y': center_y,
                        'tile_size': tile_size,
                        'xmin': tile_minx,
                        'xmax': tile_maxx,
                        'ymin': tile_miny,
                        'ymax': tile_maxy
                    }
                    
                    site_tiles.append({
                        'geometry': tile_geom,
                        'properties': tile_props
                    })
        
        print(f"  Tiles intersecting polygon: {tiles_inside:,}")
        all_tiles.extend(site_tiles)
    
    # Create GeoDataFrame - already in target CRS since we reprojected at the start
    gdf = gpd.GeoDataFrame(
        [t['properties'] for t in all_tiles],
        geometry=[t['geometry'] for t in all_tiles],
        crs=target_crs  # Use target_crs directly since tiles were created in this CRS
    )

    # Save to GeoJSON
    output_path = Path(output_geojson)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Writing {len(gdf):,} tiles to: {output_geojson}")
    gdf.to_file(output_geojson, driver='GeoJSON')
    
    # Print summary statistics by site
    print(f"\n{'='*60}")
    print("Tile summary by site:")
    site_counts = gdf.groupby('site').size()
    for site, count in site_counts.items():
        site_area_ha = gdf_sites[gdf_sites['Site'] == site]['area_ha'].iloc[0]
        print(f"  {site:15s} | Tiles: {count:6,} | Area: {site_area_ha:7.2f} ha | Coverage: {(count * 0.01):.2f} ha")
    
    print(f"\n{'='*60}")
    print(f"Total tiles: {len(gdf):,}")
    print(f"Tile size: {tile_size}m × {tile_size}m")
    print(f"Stride: {stride}m (overlap: {overlap_pct:.0f}%)")
    print(f"Total area covered: {(len(gdf) * 0.01):.2f} ha")
    print(f"CRS: {gdf.crs}")


def main():
    parser = argparse.ArgumentParser(
        description='Create grid of tiles covering forest plot site polygons'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Input GeoPackage file with site polygons'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output GeoJSON file path'
    )
    parser.add_argument(
        '--tile-size',
        type=float,
        default=10.0,
        help='Side length of square tiles in meters (default: 10.0)'
    )
    parser.add_argument(
        '--stride',
        type=float,
        default=None,
        help='Distance between tile origins in meters (default: tile_size, no overlap). '
             'For 20%% overlap on 10m tiles, use stride=8.0'
    )
    parser.add_argument(
        '--site',
        type=str,
        default=None,
        help='Filter for specific site (e.g., BluffMesa, Laguna, etc.)'
    )
    
    args = parser.parse_args()
    
    create_tile_grid_from_polygon(
        polygon_gpkg=args.input,
        output_geojson=args.output,
        tile_size=args.tile_size,
        stride=args.stride,
        site_filter=args.site
    )


if __name__ == '__main__':
    main()
