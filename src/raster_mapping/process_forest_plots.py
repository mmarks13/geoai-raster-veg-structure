#!/usr/bin/env python3
"""
Process forest plot data from Excel to georeferenced CSV.

Reads raw Excel file, filters to specified sites/years, and creates
a georeferenced output file with proper CRS metadata.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd
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


def main():
    """Process forest plot Excel file and create filtered georeferenced output."""
    repo_root = Path(__file__).parent.parent.parent

    # Define paths
    input_file = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'forest_plots_raw.xlsx'
    filter_file = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'site_filter.txt'
    output_csv = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plots_processed.csv'
    output_gpkg = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plots_processed.gpkg'

    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Check input file exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("FOREST PLOT DATA PROCESSING")
    print("=" * 80)
    print(f"\nInput: {input_file}")
    print(f"Filter: {filter_file}")

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

    print("\n" + "=" * 80)
    print("✓ Processing complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
