#!/usr/bin/env python3
"""
Visualize forest plot sites with plots, convex hulls, and bounding boxes.

Creates a grid of subplots, one per site, showing:
1. Individual plot points (dots)
2. Convex hull polygon (25m buffer)
3. Bounding box rectangle
"""

import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def create_site_visualizations(plots_file: Path, polygons_file: Path, output_file: Path) -> None:
    """
    Create visualization grid showing plots, convex hulls, and bboxes for each site.
    
    Args:
        plots_file: Path to forest plots GeoPackage
        polygons_file: Path to site polygons GeoPackage
        output_file: Path to save output figure
    """
    # Load data
    print("Loading data...")
    plots = gpd.read_file(plots_file)
    polygons = gpd.read_file(polygons_file)
    
    # Get unique sites
    sites = sorted(plots['Site'].unique())
    n_sites = len(sites)
    
    print(f"Found {n_sites} sites: {sites}")
    
    # Calculate grid dimensions (try to make roughly square)
    n_cols = int(np.ceil(np.sqrt(n_sites)))
    n_rows = int(np.ceil(n_sites / n_cols))
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
    
    # Flatten axes for easy iteration
    if n_sites == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_sites > 1 else [axes]
    
    # Plot each site
    for idx, site_name in enumerate(sites):
        ax = axes[idx]
        
        # Filter data for this site
        site_plots = plots[plots['Site'] == site_name]
        site_polygon = polygons[polygons['Site'] == site_name]
        
        if len(site_polygon) == 0:
            print(f"Warning: No polygon found for site {site_name}")
            continue
        
        # Get bounding box
        bounds = site_polygon.iloc[0].geometry.bounds  # (minx, miny, maxx, maxy)
        
        # Plot 1: Individual plots (dots)
        site_plots.plot(ax=ax, color='darkgreen', markersize=50, alpha=0.7, 
                       label=f'Plots (n={len(site_plots)})', zorder=3)
        
        # Plot 2: Convex hull polygon (25m buffer)
        site_polygon.boundary.plot(ax=ax, color='blue', linewidth=2, 
                                  label='Convex Hull + 25m', zorder=2)
        site_polygon.plot(ax=ax, color='blue', alpha=0.1, zorder=1)
        
        # Plot 3: Bounding box
        bbox_width = bounds[2] - bounds[0]
        bbox_height = bounds[3] - bounds[1]
        bbox_rect = Rectangle((bounds[0], bounds[1]), bbox_width, bbox_height,
                             linewidth=2, edgecolor='red', facecolor='none',
                             linestyle='--', label='Bounding Box', zorder=2)
        ax.add_patch(bbox_rect)
        
        # Formatting
        ax.set_title(f'{site_name}\n{len(site_plots)} plots, {site_polygon.iloc[0]["area_ha"]:.2f} ha',
                    fontsize=12, fontweight='bold')
        ax.set_xlabel('Easting (m)', fontsize=10)
        ax.set_ylabel('Northing (m)', fontsize=10)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Add some padding to the view
        x_pad = bbox_width * 0.1
        y_pad = bbox_height * 0.1
        ax.set_xlim(bounds[0] - x_pad, bounds[2] + x_pad)
        ax.set_ylim(bounds[1] - y_pad, bounds[3] + y_pad)
    
    # Hide empty subplots
    for idx in range(n_sites, len(axes)):
        axes[idx].axis('off')
    
    # Overall title
    fig.suptitle('Forest Plot Sites: Individual Plots, Convex Hulls, and Bounding Boxes',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Save figure
    print(f"\nSaving figure to: {output_file}")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close('all')  # Close all figures to free memory
    print(f"✓ Figure saved successfully")


def main():
    """Create forest plot site visualizations."""
    import argparse

    parser = argparse.ArgumentParser(description='Visualize forest plot sites')
    parser.add_argument('--suffix', type=str, default='',
                       help='Suffix to add to output filename (e.g., "alpha50")')
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent.parent

    # Define paths
    plots_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plots_processed.gpkg'
    polygons_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'site_polygons.gpkg'

    # Add suffix to filename if provided
    if args.suffix:
        output_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / f'site_visualizations_{args.suffix}.png'
    else:
        output_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'site_visualizations.png'

    # Check files exist
    if not plots_file.exists():
        print(f"Error: Plots file not found: {plots_file}", file=sys.stderr)
        print("Run process_forest_plots.py first", file=sys.stderr)
        sys.exit(1)

    if not polygons_file.exists():
        print(f"Error: Polygons file not found: {polygons_file}", file=sys.stderr)
        print("Run process_forest_plots.py first", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print("FOREST PLOT SITE VISUALIZATION")
    print("=" * 80)

    # Create visualizations
    create_site_visualizations(plots_file, polygons_file, output_file)

    print("\n" + "=" * 80)
    print("✓ Visualization complete")
    print("=" * 80)


if __name__ == '__main__':
    main()
