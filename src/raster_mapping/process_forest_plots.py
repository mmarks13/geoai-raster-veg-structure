#!/usr/bin/env python3
"""
Process forest plot data from Excel to georeferenced CSV.

Reads raw Excel file, filters to specified sites/years, and creates
a georeferenced output file with proper CRS metadata. Also generates
site-level bounding polygons encompassing all plots per site.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point


def load_filter_criteria(filter_file: Path) -> list[dict]:
    """
    Load filter criteria from text file.

    Expected format (CSV):
    Year,Site,District,Forest

    Lines starting with # are ignored as comments.

    Args:
        filter_file: Path to filter criteria file

    Returns:
        List of filter criteria dicts
    """
    if not filter_file.exists():
        print(f"Error: Filter file not found: {filter_file}", file=sys.stderr)
        print("Create the file with format: Year,Site,District,Forest", file=sys.stderr)
        sys.exit(1)

    criteria = []
    with open(filter_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            parts = [p.strip() for p in line.split(',')]
            if len(parts) != 4:
                print(f"Warning: Skipping malformed line: {line}", file=sys.stderr)
                continue

            criteria.append({
                'year': parts[0],
                'site': parts[1],
                'district': parts[2],
                'forest': parts[3]
            })

    if not criteria:
        print(f"Error: No valid filter criteria found in {filter_file}", file=sys.stderr)
        sys.exit(1)

    return criteria


def parse_site_year_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse concatenated Site_Year column into separate Site and Year columns.

    Expected formats:
    - "SiteName2024"
    - "SiteName2023"
    - etc.

    Args:
        df: Input dataframe with Site_Year column

    Returns:
        DataFrame with added 'Site' and 'Year' columns
    """
    # Find the column that contains site/year information
    site_year_col = None
    for col in df.columns:
        col_lower = col.lower().replace(' ', '').replace('_', '')
        if 'site' in col_lower and 'year' in col_lower:
            site_year_col = col
            break

    if site_year_col is None:
        print("Error: Could not find Site_Year column", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    print(f"Using column '{site_year_col}' for site/year parsing")

    # Extract year (last 4 digits) and site (everything before the year)
    df = df.copy()

    # Parse the concatenated column
    df['Year'] = df[site_year_col].str.extract(r'(\d{4})$')[0]
    df['Site'] = df[site_year_col].str.replace(r'\d{4}$', '', regex=True)

    # Clean up Site names (remove trailing underscores or spaces)
    df['Site'] = df['Site'].str.rstrip('_').str.rstrip()

    return df


def filter_plots(df: pd.DataFrame, criteria: list[dict]) -> pd.DataFrame:
    """
    Filter dataframe to only include plots matching the criteria.

    Args:
        df: Input dataframe with Site, Year, District, Forest columns
        criteria: List of dicts with filter criteria

    Returns:
        Filtered dataframe
    """
    # Create a boolean mask for each criterion
    masks = []

    for criterion in criteria:
        mask = (
            (df['Year'] == criterion['year']) &
            (df['Site'] == criterion['site'])
        )
        masks.append(mask)

    # Combine all masks with OR
    combined_mask = masks[0]
    for mask in masks[1:]:
        combined_mask = combined_mask | mask

    filtered_df = df[combined_mask].copy()

    print(f"\nFiltering results:")
    print(f"  Total input rows: {len(df)}")
    print(f"  Rows matching criteria: {len(filtered_df)}")

    # Print summary by site/year
    if len(filtered_df) > 0:
        print(f"\nFiltered data by Site and Year:")
        summary = filtered_df.groupby(['Site', 'Year']).size().reset_index(name='Count')
        for _, row in summary.iterrows():
            print(f"    {row['Site']} {row['Year']}: {row['Count']} plots")

    return filtered_df


def create_geodataframe(df: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Create GeoDataFrame from dataframe with Easting/Northing columns.

    Args:
        df: DataFrame with Easting and Northing columns

    Returns:
        GeoDataFrame with geometry in UTM Zone 11N (EPSG:26911)
    """
    # Check for coordinate columns
    easting_col = None
    northing_col = None

    for col in df.columns:
        col_lower = col.lower()
        if 'easting' in col_lower:
            easting_col = col
        if 'northing' in col_lower:
            northing_col = col

    if easting_col is None or northing_col is None:
        print("Error: Could not find Easting/Northing columns", file=sys.stderr)
        print(f"Available columns: {df.columns.tolist()}", file=sys.stderr)
        sys.exit(1)

    print(f"Using columns: Easting='{easting_col}', Northing='{northing_col}'")

    # Filter to rows with valid coordinates
    valid_coords = df[[easting_col, northing_col]].dropna()
    df_valid = df.loc[valid_coords.index].copy()

    if len(df_valid) == 0:
        print("Error: No valid coordinates found", file=sys.stderr)
        sys.exit(1)

    print(f"  Valid coordinates: {len(df_valid)} / {len(df)}")

    # Create Point geometries
    geometry = [Point(xy) for xy in zip(df_valid[easting_col], df_valid[northing_col])]
    gdf = gpd.GeoDataFrame(df_valid, geometry=geometry)

    # Set CRS to UTM Zone 11N (EPSG:26911)
    gdf.set_crs(epsg=26911, inplace=True)

    print(f"  CRS: EPSG:26911 (UTM Zone 11N)")

    return gdf


def create_site_polygons(gdf: gpd.GeoDataFrame, buffer_meters: float = 35.0,
                         method: str = 'alpha_shape', alpha: float = None,
                         alpha_scale: float = None) -> gpd.GeoDataFrame:
    """
    Create site polygons using alpha shapes with auto-scaling.

    Uses alpha shapes instead of convex hull to create tighter polygons that
    follow the actual distribution of plot points. Auto-scales alpha parameter
    per-site based on mean pairwise distance between plots.

    Args:
        gdf: GeoDataFrame with plot point geometries and Site column
        buffer_meters: Buffer distance in meters (default: 35m)
        method: Polygon method - 'convex_hull', 'alpha_shape', or 'buffer'
        alpha: Fixed alpha parameter in meters (overrides alpha_scale if set)
        alpha_scale: Scaling factor (alpha = alpha_scale × mean_distance)
                    Default: 0.8 (tuned for forest plot spacing)

    Returns:
        GeoDataFrame with one polygon per site
    """
    # Auto-scaling setup (changed from fixed convex hull, Nov 2025)
    if alpha is None and alpha_scale is None:
        alpha_scale = 0.8  # Default: adapts to each site's point spacing
        use_auto_alpha = True
    elif alpha_scale is not None:
        use_auto_alpha = True
    else:
        use_auto_alpha = False
    if 'Site' not in gdf.columns:
        print("Error: No 'Site' column found for grouping", file=sys.stderr)
        sys.exit(1)

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
                from shapely.geometry import LineString, MultiLineString
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
        area_ha = area_m2 / 10000  # Convert to hectares

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

    # Create GeoDataFrame
    site_gdf = gpd.GeoDataFrame(site_polygons, crs=gdf.crs)

    print(f"  Total sites: {len(site_gdf)}")

    return site_gdf


def save_site_bboxes(site_polygons: gpd.GeoDataFrame, output_file: Path) -> None:
    """
    Save bounding boxes for each site in get_data.sh format.
    
    Transforms from UTM (EPSG:26911) to geographic (EPSG:4326) to match
    the format used in get_data.sh.
    
    Format: --bbox minlon minlat maxlon maxlat (one line per site)
    
    Args:
        site_polygons: GeoDataFrame with site polygons in EPSG:26911
        output_file: Path to output text file
    """
    print(f"\nSaving site bounding boxes to: {output_file}")
    
    # Transform to WGS84 (EPSG:4326) for geographic coordinates
    site_polygons_geo = site_polygons.to_crs(epsg=4326)
    
    with open(output_file, 'w') as f:
        f.write("# Site bounding boxes for get_data.sh\n")
        f.write("# Format: --bbox minlon minlat maxlon maxlat\n")
        f.write("# CRS: EPSG:4326 (WGS84 geographic coordinates)\n")
        f.write("#\n")
        
        for _, row in site_polygons_geo.iterrows():
            site_name = row['Site']
            bounds = row['geometry'].bounds  # (minx, miny, maxx, maxy) = (minlon, minlat, maxlon, maxlat)
            
            # Write bbox in get_data.sh format (geographic coordinates)
            bbox_str = f"--bbox {bounds[0]:.6f} {bounds[1]:.6f} {bounds[2]:.6f} {bounds[3]:.6f}"
            f.write(f"# {site_name}\n")
            f.write(f"{bbox_str}\n")
            f.write("\n")
            
            print(f"  {site_name}: lon [{bounds[0]:.6f}, {bounds[2]:.6f}], lat [{bounds[1]:.6f}, {bounds[3]:.6f}]")


def main():
    """Process forest plot Excel file and create filtered georeferenced output."""
    import argparse

    parser = argparse.ArgumentParser(description='Process forest plot data with configurable polygon generation')
    parser.add_argument('--method', type=str, default='alpha_shape',
                       choices=['convex_hull', 'alpha_shape', 'buffer'],
                       help='Polygon generation method (default: alpha_shape)')
    parser.add_argument('--alpha', type=float, default=None,
                       help='Fixed alpha parameter in meters (overrides auto-scaling)')
    parser.add_argument('--alpha-scale', type=float, default=None,
                       help='Auto-scale factor (alpha = scale × mean_distance). Default: 0.8')
    parser.add_argument('--buffer', type=float, default=35.0,
                       help='Final buffer distance in meters (default: 35.0)')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent

    # Define paths
    input_file = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'forest_plots_raw.xlsx'
    filter_file = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'site_filter.txt'
    output_csv = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plots_processed.csv'
    output_gpkg = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plots_processed.gpkg'
    output_site_polygons = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'site_polygons.gpkg'
    output_site_bboxes = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'site_bboxes.txt'

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Handle alpha_scale argument
    alpha_scale = getattr(args, 'alpha_scale', None)

    print("=" * 80)
    print("FOREST PLOT DATA PROCESSING")
    print("=" * 80)
    print(f"\nInput: {input_file}")
    print(f"Filter: {filter_file}")
    print(f"\nPolygon settings:")
    print(f"  Method: {args.method}")
    if args.method == 'alpha_shape':
        if args.alpha is not None:
            print(f"  Alpha: {args.alpha}m (fixed)")
        elif alpha_scale is not None:
            print(f"  Alpha: AUTO-SCALED (factor={alpha_scale})")
        else:
            print(f"  Alpha: AUTO-SCALED (factor=0.8, default)")
    print(f"  Buffer: {args.buffer}m")

    # Load filter criteria
    print("\nLoading filter criteria...")
    filter_criteria = load_filter_criteria(filter_file)
    print(f"  Loaded {len(filter_criteria)} site/year combinations")

    # Load Excel file
    print("\nLoading Excel file...")
    df = pd.read_excel(input_file)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {df.columns.tolist()}")

    # Parse Site_Year column
    print("\nParsing Site_Year column...")
    df = parse_site_year_column(df)

    # Filter to specified sites/years
    print("\nApplying filters...")
    df_filtered = filter_plots(df, filter_criteria)

    if len(df_filtered) == 0:
        print("\nWarning: No plots matched the filter criteria", file=sys.stderr)
        print("Check that Site and Year values match the expected format")
        sys.exit(1)

    # Create GeoDataFrame
    print("\nCreating georeferenced data...")
    gdf = create_geodataframe(df_filtered)

    # Save as CSV (with coordinate columns)
    print(f"\nSaving CSV to: {output_csv}")
    gdf.drop(columns=['geometry']).to_csv(output_csv, index=False)

    # Save as GeoPackage (with geometry)
    print(f"Saving GeoPackage to: {output_gpkg}")
    gdf.to_file(output_gpkg, driver='GPKG')

    # Create site-level polygons with specified method
    site_polygons = create_site_polygons(
        gdf,
        buffer_meters=args.buffer,
        method=args.method,
        alpha=args.alpha,
        alpha_scale=alpha_scale
    )

    # Save site polygons
    print(f"\nSaving site polygons to: {output_site_polygons}")
    site_polygons.to_file(output_site_polygons, driver='GPKG')

    # Save site bounding boxes
    save_site_bboxes(site_polygons, output_site_bboxes)

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total plots processed: {len(gdf)}")

    if 'Site' in gdf.columns and 'Year' in gdf.columns:
        print("\nPlots by Site and Year:")
        summary = gdf.groupby(['Site', 'Year']).size().reset_index(name='Count')
        for _, row in summary.iterrows():
            print(f"  {row['Site']} {row['Year']}: {row['Count']}")

    # Print coordinate bounds
    bounds = gdf.total_bounds
    print(f"\nCoordinate bounds (UTM 11N):")
    print(f"  Easting:  {bounds[0]:.2f} to {bounds[2]:.2f}")
    print(f"  Northing: {bounds[1]:.2f} to {bounds[3]:.2f}")

    # Print site polygon summary
    print(f"\nSite polygon coverage:")
    for _, row in site_polygons.iterrows():
        print(f"  {row['Site']}: {row['area_ha']:.2f} ha ({row['plot_count']} plots)")

    print("\n" + "=" * 80)
    print("✓ Processing complete")
    print(f"  - Plot points: {output_gpkg}")
    print(f"  - Site polygons: {output_site_polygons}")
    print(f"  - Site bboxes: {output_site_bboxes}")
    print("=" * 80)


if __name__ == '__main__':
    main()
