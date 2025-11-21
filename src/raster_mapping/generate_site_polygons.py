#!/usr/bin/env python3
"""
Generate site polygons and bounding boxes from existing forest plot data.

Quick script to create site_polygons.gpkg and site_bboxes.txt from
the existing forest_plots_processed.gpkg file.
"""

import sys
from pathlib import Path
import numpy as np
import geopandas as gpd


def create_site_polygons(gdf: gpd.GeoDataFrame, buffer_meters: float = 35.0,
                         method: str = 'alpha_shape', alpha: float = None,
                         alpha_scale: float = None) -> gpd.GeoDataFrame:
    """
    Create site polygons using various methods.

    Args:
        gdf: GeoDataFrame with plot point geometries and Site column
        buffer_meters: Buffer distance in meters (default: 35m)
        method: Polygon method - 'convex_hull', 'alpha_shape', or 'buffer'
        alpha: Fixed alpha parameter in meters (if None, uses alpha_scale)
        alpha_scale: Scaling factor for auto-alpha (alpha = alpha_scale × mean_distance)
                    Default: 0.8 (adapts to each site's point spacing)
                    Auto-scaling calculates alpha per-site based on mean pairwise distance,
                    allowing tighter polygons for densely-spaced plots (e.g., Laguna)
                    and looser polygons for widely-spaced plots (e.g., ReyesPeak).

    Returns:
        GeoDataFrame with one polygon per site
    """
    # Determine alpha mode
    # Changed from convex hull to alpha shapes with auto-scaling (Nov 2025)
    # This creates tighter polygons that follow plot distribution more closely
    if alpha is None and alpha_scale is None:
        alpha_scale = 0.8  # Default auto-scaling (tuned for forest plot spacing)
        use_auto_alpha = True
    elif alpha_scale is not None:
        use_auto_alpha = True
    else:
        use_auto_alpha = False

    print(f"\nCreating site polygons using method='{method}'")
    if method == 'alpha_shape':
        if use_auto_alpha:
            print(f"  Alpha mode: AUTO-SCALED (scale factor = {alpha_scale})")
            print(f"  Alpha will be calculated per-site as: {alpha_scale} × mean_distance")
        else:
            print(f"  Alpha mode: FIXED ({alpha}m for all sites)")
    print(f"  Final buffer: {buffer_meters}m")

    site_polygons = []

    for site_name, site_plots in gdf.groupby('Site'):
        # Calculate site-specific alpha if using auto-scaling
        if method == 'alpha_shape' and use_auto_alpha:
            from scipy.spatial.distance import pdist
            coords = np.array([(geom.x, geom.y) for geom in site_plots.geometry])
            if len(coords) >= 2:
                distances = pdist(coords)
                mean_distance = np.mean(distances)
                site_alpha = alpha_scale * mean_distance
            else:
                site_alpha = 50.0  # Fallback for single point
        else:
            site_alpha = alpha if alpha is not None else 50.0

        # Create base polygon using selected method
        if method == 'convex_hull':
            base_polygon = site_plots.geometry.unary_union.convex_hull

        elif method == 'alpha_shape':
            # Alpha shape implementation
            from scipy.spatial import Delaunay
            from shapely.ops import cascaded_union, polygonize
            from shapely.geometry import LineString

            # Extract point coordinates
            coords = np.array([(geom.x, geom.y) for geom in site_plots.geometry])

            if len(coords) < 4:
                # Fall back to convex hull for small point sets
                base_polygon = site_plots.geometry.unary_union.convex_hull
            else:
                # Compute Delaunay triangulation
                tri = Delaunay(coords)

                # Filter triangles by edge length (alpha parameter)
                edges = []
                for simplex in tri.simplices:
                    # Get triangle vertices
                    pts = coords[simplex]
                    # Check all edge lengths
                    edge_lengths = [
                        np.linalg.norm(pts[0] - pts[1]),
                        np.linalg.norm(pts[1] - pts[2]),
                        np.linalg.norm(pts[2] - pts[0])
                    ]
                    # Keep triangle if all edges are shorter than site_alpha
                    if all(length < site_alpha for length in edge_lengths):
                        edges.append(tuple(sorted([simplex[0], simplex[1]])))
                        edges.append(tuple(sorted([simplex[1], simplex[2]])))
                        edges.append(tuple(sorted([simplex[2], simplex[0]])))

                # Build polygon from edges
                from collections import Counter
                edge_counts = Counter(edges)
                # Boundary edges appear once, interior edges appear twice
                boundary_edges = [edge for edge, count in edge_counts.items() if count == 1]

                # Create line segments from boundary edges
                lines = [LineString([coords[edge[0]], coords[edge[1]]]) for edge in boundary_edges]

                if lines:
                    # Polygonize the boundary edges
                    result = list(polygonize(lines))
                    if result:
                        base_polygon = cascaded_union(result) if len(result) > 1 else result[0]
                    else:
                        # Fall back to convex hull
                        base_polygon = site_plots.geometry.unary_union.convex_hull
                else:
                    base_polygon = site_plots.geometry.unary_union.convex_hull

        elif method == 'buffer':
            # Buffer each point then merge
            buffered_points = [geom.buffer(buffer_meters) for geom in site_plots.geometry]
            base_polygon = gpd.GeoSeries(buffered_points).unary_union
            # Don't add additional buffer for this method
            buffer_meters = 0

        else:
            raise ValueError(f"Unknown method: {method}")

        # Add final buffer (if not already applied)
        if buffer_meters > 0:
            buffered_polygon = base_polygon.buffer(buffer_meters)
        else:
            buffered_polygon = base_polygon

        # Calculate statistics
        plot_count = len(site_plots)
        area_m2 = buffered_polygon.area
        area_ha = area_m2 / 10000

        site_polygons.append({
            'Site': site_name,
            'plot_count': plot_count,
            'area_m2': area_m2,
            'area_ha': area_ha,
            'buffer_m': buffer_meters,
            'method': method,
            'alpha_m': site_alpha if method == 'alpha_shape' else None,
            'alpha_scale': alpha_scale if (method == 'alpha_shape' and use_auto_alpha) else None,
            'geometry': buffered_polygon
        })

        if method == 'alpha_shape' and use_auto_alpha:
            print(f"  {site_name}: {plot_count} plots, {area_ha:.2f} ha (alpha={site_alpha:.0f}m)")
        else:
            print(f"  {site_name}: {plot_count} plots, {area_ha:.2f} ha")

    site_gdf = gpd.GeoDataFrame(site_polygons, crs=gdf.crs)
    print(f"  Total sites: {len(site_gdf)}")

    return site_gdf


def save_site_bboxes(site_polygons: gpd.GeoDataFrame, output_file: Path) -> None:
    """Save bounding boxes in get_data.sh format (EPSG:4326)."""
    print(f"\nSaving site bounding boxes to: {output_file}")
    
    # Transform to WGS84
    site_polygons_geo = site_polygons.to_crs(epsg=4326)
    
    with open(output_file, 'w') as f:
        f.write("# Site bounding boxes for get_data.sh\n")
        f.write("# Format: --bbox minlon minlat maxlon maxlat\n")
        f.write("# CRS: EPSG:4326 (WGS84 geographic coordinates)\n")
        f.write("#\n")
        
        for _, row in site_polygons_geo.iterrows():
            site_name = row['Site']
            bounds = row['geometry'].bounds
            
            bbox_str = f"--bbox {bounds[0]:.6f} {bounds[1]:.6f} {bounds[2]:.6f} {bounds[3]:.6f}"
            f.write(f"# {site_name}\n")
            f.write(f"{bbox_str}\n")
            f.write("\n")
            
            print(f"  {site_name}: lon [{bounds[0]:.6f}, {bounds[2]:.6f}], lat [{bounds[1]:.6f}, {bounds[3]:.6f}]")


def main():
    """Generate site polygons and bboxes."""
    import argparse

    parser = argparse.ArgumentParser(description='Generate site polygons with configurable methods')
    parser.add_argument('--method', type=str, default='alpha_shape',
                       choices=['convex_hull', 'alpha_shape', 'buffer'],
                       help='Polygon generation method (default: alpha_shape)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Fixed alpha parameter in meters (overrides --alpha-scale)')
    parser.add_argument('--alpha-scale', type=float, default=None,
                       help='Auto-scale alpha per site (alpha = scale × mean_distance). '
                            'Default: 0.8 if neither --alpha nor --alpha-scale specified.')
    parser.add_argument('--buffer', type=float, default=35.0,
                       help='Final buffer distance in meters (default: 35.0)')
    args = parser.parse_args()

    # Convert alpha-scale to alpha_scale for Python naming
    alpha_scale = getattr(args, 'alpha_scale', None)

    repo_root = Path(__file__).parent.parent.parent

    # Paths
    input_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plots_processed.gpkg'
    output_polygons = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'site_polygons.gpkg'
    output_bboxes = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'site_bboxes.txt'

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("GENERATE SITE POLYGONS")
    print("=" * 80)
    print(f"\nInput: {input_file}")
    print(f"\nPolygon settings:")
    print(f"  Method: {args.method}")
    if args.method == 'alpha_shape':
        if args.alpha is not None:
            print(f"  Alpha:  {args.alpha}m (fixed)")
        elif alpha_scale is not None:
            print(f"  Alpha:  AUTO-SCALED (factor={alpha_scale})")
        else:
            print(f"  Alpha:  AUTO-SCALED (factor=0.8, default)")
    print(f"  Buffer: {args.buffer}m")

    # Load data
    print("\nLoading forest plot data...")
    gdf = gpd.read_file(input_file)
    print(f"  Loaded {len(gdf)} plots from {len(gdf['Site'].unique())} sites")
    print(f"  CRS: {gdf.crs}")

    # Create site polygons
    site_polygons = create_site_polygons(
        gdf,
        buffer_meters=args.buffer,
        method=args.method,
        alpha=args.alpha,
        alpha_scale=alpha_scale
    )

    # Save outputs
    print(f"\nSaving site polygons to: {output_polygons}")
    site_polygons.to_file(output_polygons, driver='GPKG')

    save_site_bboxes(site_polygons, output_bboxes)

    print("\n" + "=" * 80)
    print("✓ Complete")
    print(f"  - Site polygons: {output_polygons}")
    print(f"  - Site bboxes: {output_bboxes}")
    print("=" * 80)


if __name__ == '__main__':
    main()
