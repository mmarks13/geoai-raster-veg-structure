#!/usr/bin/env python3
"""
Create maps of forest plot locations.

Generates visualization of forest plot data with optional basemap.
Can work with both obfuscated and non-obfuscated coordinates.
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd

# Use non-interactive backend for matplotlib (must be before pyplot import)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from shapely.geometry import Point
import urllib.request
import io
from PIL import Image
import numpy as np

# Try to import contextily, but make it optional
try:
    import contextily as ctx
    HAS_CONTEXTILY = True
except (ImportError, Exception) as e:
    HAS_CONTEXTILY = False
    print(f"Warning: contextily not available ({e}), will skip basemap")


def create_plot_map(csv_path: Path, output_dir: Path, site_filter: str = None,
                    obfuscated: bool = True) -> None:
    """
    Create map visualization of forest plots.

    Args:
        csv_path: Path to CSV file
        output_dir: Directory to save output plots
        site_filter: Filter to specific site(s) (e.g., 'L,O' or 'T')
        obfuscated: Whether coordinates are obfuscated
    """
    # Load data
    print(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)

    # Clean column names
    df.columns = df.columns.str.replace(' ', '_').str.replace('[^A-Za-z0-9_]', '', regex=True)

    # Filter to rows with valid coordinates
    valid_coords = df[['Easting', 'Northing']].dropna()
    df_valid = df.loc[valid_coords.index].copy()

    if len(df_valid) == 0:
        print("Error: No valid coordinates found", file=sys.stderr)
        sys.exit(1)

    # Apply site filter if specified
    if site_filter and 'Site' in df_valid.columns:
        sites = [s.strip() for s in site_filter.split(',')]
        df_valid = df_valid[df_valid['Site'].isin(sites)]
        site_label = f"_{site_filter.replace(',', '_')}"
        print(f"Filtered to site(s): {', '.join(sites)} ({len(df_valid)} plots)")
    else:
        site_label = ""

    if len(df_valid) == 0:
        print(f"Error: No plots found for site filter '{site_filter}'", file=sys.stderr)
        return

    print(f"Found {len(df_valid)} plots with valid coordinates")

    # Create Point geometries
    geometry = [Point(xy) for xy in zip(df_valid['Easting'], df_valid['Northing'])]
    gdf = gpd.GeoDataFrame(df_valid, geometry=geometry)

    # Set CRS to UTM Zone 11N (EPSG:26911)
    gdf.set_crs(epsg=26911, inplace=True)

    # Convert to Web Mercator for basemap
    webmerc = gdf.to_crs(epsg=3857)

    # Create figure with basemap attempt
    fig, ax = plt.subplots(figsize=(14, 10))

    # Plot points colored by year with visually distinct colors
    if 'Year' in webmerc.columns:
        years = sorted(webmerc['Year'].dropna().unique())
        # Use distinct colors: orange for 2023, blue for 2024
        color_map = {2023: '#FF6B35', 2024: '#004E89'}  # Orange and Blue

        for year in years:
            year_data = webmerc[webmerc['Year'] == year]
            year_data.plot(
                ax=ax,
                color=color_map.get(year, '#FF6B35'),
                markersize=80,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.8,
                label=str(int(year))
            )

        legend_title = 'Year'
    else:
        # No year information, plot all points in one color
        webmerc.plot(
            ax=ax,
            color='#004E89',
            markersize=80,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.8
        )
        legend_title = None

    # Add basemap - try multiple providers
    basemap_note = ""
    if HAS_CONTEXTILY:
        basemap_added = False
        # Try Stamen Terrain first (good for natural areas)
        providers_to_try = [
            ('Stamen.Terrain', lambda: ctx.add_basemap(ax, source=ctx.providers.Stamen.Terrain)),
            ('CartoDB Positron', lambda: ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)),
            ('OpenStreetMap', lambda: ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik)),
        ]

        for provider_name, add_func in providers_to_try:
            try:
                add_func()
                basemap_added = True
                print(f"Added basemap: {provider_name}")
                break
            except Exception as e:
                print(f"Could not add {provider_name} basemap: {e}")

        if not basemap_added:
            basemap_note = " (no basemap available)"
    else:
        basemap_note = " (no basemap - contextily unavailable)"

    # Styling
    loc_type = "Obfuscated" if obfuscated else "Actual"
    title = f'Forest Plots - {loc_type} Locations'
    if site_filter:
        title += f' (Site: {site_filter})'
    ax.set_title(f'{title}{basemap_note}', fontsize=16, fontweight='bold')
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)

    if legend_title:
        ax.legend(title=legend_title, loc='best', fontsize=11, title_fontsize=12)

    # Save figure with basemap
    output_path = output_dir / f'forest_plots_map{site_label}.png'
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nMap saved to {output_path}")
    plt.close(fig)

    # Also create a simple version without basemap (always works)
    fig2, ax2 = plt.subplots(figsize=(12, 10))

    if 'Year' in gdf.columns:
        years = sorted(gdf['Year'].dropna().unique())
        color_map = {2023: '#FF6B35', 2024: '#004E89'}
        for year in years:
            year_data = gdf[gdf['Year'] == year]
            year_data.plot(
                ax=ax2,
                color=color_map.get(year, '#FF6B35'),
                markersize=80,
                alpha=0.7,
                edgecolor='black',
                linewidth=0.8,
                label=str(int(year))
            )
    else:
        gdf.plot(
            ax=ax2,
            color='#004E89',
            markersize=80,
            alpha=0.7,
            edgecolor='black',
            linewidth=0.8
        )

    title_simple = f'Forest Plots - {loc_type} Locations'
    if site_filter:
        title_simple += f' (Site: {site_filter})'
    ax2.set_title(title_simple, fontsize=16, fontweight='bold')
    ax2.set_xlabel('Easting (UTM Zone 11N, meters)', fontsize=12)
    ax2.set_ylabel('Northing (UTM Zone 11N, meters)', fontsize=12)

    if legend_title:
        ax2.legend(title=legend_title, loc='best', fontsize=11, title_fontsize=12)

    ax2.grid(True, alpha=0.3)

    output_path_simple = output_dir / f'forest_plots_map_simple{site_label}.png'
    plt.tight_layout()
    plt.savefig(output_path_simple, dpi=300, bbox_inches='tight')
    print(f"Simple map saved to {output_path_simple}")
    plt.close(fig2)

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total plots: {len(df_valid)}")
    if 'Year' in df_valid.columns:
        year_counts = df_valid['Year'].value_counts().sort_index()
        print(f"  Plots by year:")
        for year, count in year_counts.items():
            print(f"    {int(year)}: {count}")
    if 'Site' in df_valid.columns:
        site_counts = df_valid['Site'].value_counts()
        print(f"  Plots by site:")
        for site, count in site_counts.items():
            print(f"    {site}: {count}")


def main():
    """Create forest plot maps from data."""
    repo_root = Path(__file__).parent.parent.parent

    # Define file paths
    obfuscated_csv = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plot_sample_obfuscated.csv'
    actual_csv = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'forest_plot_sample.csv'

    output_dir_obf = repo_root / 'temp' / 'forest_plots' / 'obfuscated'
    output_dir_actual = repo_root / 'temp' / 'forest_plots' / 'actual'

    # Create output directories
    output_dir_obf.mkdir(parents=True, exist_ok=True)
    output_dir_actual.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("OBFUSCATED DATA - Safe to share")
    print("=" * 80)

    if obfuscated_csv.exists():
        # All sites together
        print("\n--- All Sites (Obfuscated) ---")
        create_plot_map(obfuscated_csv, output_dir_obf, site_filter=None, obfuscated=True)

        # L and O sites together
        print("\n--- L and O Sites (Obfuscated) ---")
        create_plot_map(obfuscated_csv, output_dir_obf, site_filter='L,O', obfuscated=True)

        # T site separately
        print("\n--- T Site (Obfuscated) ---")
        create_plot_map(obfuscated_csv, output_dir_obf, site_filter='T', obfuscated=True)
    else:
        print(f"Obfuscated file not found: {obfuscated_csv}")
        print("Run transform_coordinates.py first")

    print("\n" + "=" * 80)
    print("ACTUAL DATA - SENSITIVE - Do not share")
    print("=" * 80)

    if actual_csv.exists():
        # All sites together
        print("\n--- All Sites (Actual) ---")
        create_plot_map(actual_csv, output_dir_actual, site_filter=None, obfuscated=False)

        # L and O sites together
        print("\n--- L and O Sites (Actual) ---")
        create_plot_map(actual_csv, output_dir_actual, site_filter='L,O', obfuscated=False)

        # T site separately
        print("\n--- T Site (Actual) ---")
        create_plot_map(actual_csv, output_dir_actual, site_filter='T', obfuscated=False)
    else:
        print(f"Actual file not found: {actual_csv}")
        print("This is expected if you haven't added the sensitive data file")

    print("\n" + "=" * 80)
    print("✓ Map generation complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
