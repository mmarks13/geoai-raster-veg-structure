"""
Create circular or square plot footprints from forest plot center coordinates.

This script generates polygon geometries representing the spatial extent of
USFS forest plots based on their center coordinates and plot size.

For 0.1-acre plots:
- Circular: radius = 11.35m (37.2 ft)
- Square: 20m × 20m (66 × 66 ft)

Usage:
    python src/raster_mapping/create_plot_footprints.py \
        --input data/processed/forest_plot_data/forest_plots_processed.gpkg \
        --output data/processed/forest_plot_data/plot_footprints.gpkg \
        --method circular \
        --radius 11.35

References:
    - FIREMON Field Sampling Manual, RMRS-GTR-164
    - CSE / FSVeg preparation and design guide
"""

import argparse
from pathlib import Path
import geopandas as gpd
from shapely.geometry import box


def create_circular_footprints(gdf: gpd.GeoDataFrame, radius: float) -> gpd.GeoDataFrame:
    """
    Create circular plot footprints by buffering point geometries.
    
    Args:
        gdf: GeoDataFrame with point geometries (plot centers)
        radius: Buffer radius in meters (11.35m for 0.1-acre circular plot)
    
    Returns:
        GeoDataFrame with polygon geometries (circular footprints)
    """
    footprints = gdf.copy()
    footprints['geometry'] = footprints.geometry.buffer(radius)
    return footprints


def create_square_footprints(gdf: gpd.GeoDataFrame, half_side: float) -> gpd.GeoDataFrame:
    """
    Create square plot footprints centered on point geometries.
    
    Args:
        gdf: GeoDataFrame with point geometries (plot centers)
        half_side: Half the side length in meters (10.06m for 0.1-acre square plot)
    
    Returns:
        GeoDataFrame with polygon geometries (square footprints)
    """
    footprints = gdf.copy()
    
    def point_to_square(point, half_side):
        return box(
            point.x - half_side,
            point.y - half_side,
            point.x + half_side,
            point.y + half_side
        )
    
    footprints['geometry'] = footprints.geometry.apply(
        lambda pt: point_to_square(pt, half_side)
    )
    return footprints


def main():
    parser = argparse.ArgumentParser(
        description="Create plot footprints from forest plot center coordinates"
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Path to input GeoPackage with plot center points"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Path to output GeoPackage with plot footprint polygons"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["circular", "square"],
        default="circular",
        help="Footprint shape method (default: circular)"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=11.35,
        help="Buffer radius for circular plots in meters (default: 11.35m for 0.1-acre)"
    )
    parser.add_argument(
        "--half-side",
        type=float,
        default=10.06,
        help="Half side length for square plots in meters (default: 10.06m for 0.1-acre)"
    )
    parser.add_argument(
        "--target-crs",
        type=str,
        default="EPSG:32611",
        help="Target CRS for output (default: EPSG:32611, UTM Zone 11N WGS84)"
    )
    
    args = parser.parse_args()
    
    # Load input data
    print(f"Loading forest plot data from: {args.input}")
    gdf = gpd.read_file(args.input)
    print(f"  Loaded {len(gdf)} plots")
    print(f"  Input CRS: {gdf.crs}")
    
    # Reproject to target CRS if needed (buffer operations require projected CRS)
    if str(gdf.crs) != args.target_crs:
        print(f"  Reprojecting to {args.target_crs}...")
        gdf = gdf.to_crs(args.target_crs)
    
    # Create footprints based on method
    if args.method == "circular":
        print(f"\nCreating circular footprints with radius={args.radius}m...")
        footprints = create_circular_footprints(gdf, args.radius)
        area_expected = 3.14159 * args.radius**2
        print(f"  Expected area per plot: {area_expected:.1f} m² ({area_expected/4046.86:.3f} acres)")
    else:
        print(f"\nCreating square footprints with half-side={args.half_side}m...")
        footprints = create_square_footprints(gdf, args.half_side)
        area_expected = (2 * args.half_side)**2
        print(f"  Expected area per plot: {area_expected:.1f} m² ({area_expected/4046.86:.3f} acres)")
    
    # Verify geometry types
    geom_types = footprints.geometry.geom_type.unique()
    print(f"  Geometry types: {geom_types}")
    
    # Calculate actual areas
    footprints['area_m2'] = footprints.geometry.area
    footprints['area_acres'] = footprints['area_m2'] / 4046.86
    
    print(f"\nArea statistics:")
    print(f"  Mean: {footprints['area_m2'].mean():.1f} m² ({footprints['area_acres'].mean():.3f} acres)")
    print(f"  Min:  {footprints['area_m2'].min():.1f} m²")
    print(f"  Max:  {footprints['area_m2'].max():.1f} m²")
    
    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving footprints to: {args.output}")
    footprints.to_file(args.output, driver="GPKG")

    # Summary by site
    if 'Site' in footprints.columns:
        print(f"\nFootprints by site:")
        site_counts = footprints.groupby('Site').size()
        for site, count in site_counts.items():
            print(f"  {site}: {count} plots")
    
    print(f"\n✓ Created {len(footprints)} plot footprints")
    print(f"  Output CRS: {footprints.crs}")


if __name__ == "__main__":
    main()
