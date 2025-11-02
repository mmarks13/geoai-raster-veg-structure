#!/usr/bin/env python3
"""
Visualize fuel metrics tile bounds and coverage.

Creates a visualization showing:
1. All individual tile raster bounds
2. Original LAS point cloud extent
3. Merged GeoTIFF extent
4. Coverage gaps and overlaps
"""

import subprocess
import json
import re
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import numpy as np

def get_raster_bounds(tif_file):
    """Extract bounds from a GeoTIFF using gdalinfo."""
    result = subprocess.run(
        ["conda", "run", "-p", "/home/jovyan/geoai_env", "gdalinfo", str(tif_file)],
        capture_output=True,
        text=True,
        timeout=30
    )

    lines = result.stdout.split('\n')
    ul = lr = None
    for line in lines:
        if 'Upper Left' in line:
            match = re.search(r'\(\s*([-\d.]+),\s*([-\d.]+)\)', line)
            if match:
                ul = (float(match.group(1)), float(match.group(2)))
        elif 'Lower Right' in line:
            match = re.search(r'\(\s*([-\d.]+),\s*([-\d.]+)\)', line)
            if match:
                lr = (float(match.group(1)), float(match.group(2)))

    if ul and lr:
        return {
            'minx': ul[0],
            'maxx': lr[0],
            'miny': lr[1],
            'maxy': ul[1]
        }
    return None

def get_las_bounds():
    """Get bounds from the original LAS file."""
    result = subprocess.run(
        ["conda", "run", "-p", "/home/jovyan/geoai_env", "pdal", "info", "--summary",
         "/home/jovyan/geoai_veg_map/data/processed/fuel_metrics/volcan_full/ground_classified/VolcanMt_20231025_LAS_classified.las"],
        capture_output=True,
        text=True,
        timeout=60
    )

    info = json.loads(result.stdout)
    bounds = info['summary']['bounds']
    return {
        'minx': bounds['minx'],
        'maxx': bounds['maxx'],
        'miny': bounds['miny'],
        'maxy': bounds['maxy']
    }

def get_merged_bounds():
    """Get bounds from the merged GeoTIFF."""
    result = subprocess.run(
        ["conda", "run", "-p", "/home/jovyan/geoai_env", "gdalinfo",
         "/home/jovyan/geoai_veg_map/data/processed/fuel_metrics/volcan_full/volcan_fuel_metrics_5m.tif"],
        capture_output=True,
        text=True,
        timeout=30
    )

    lines = result.stdout.split('\n')
    ul = lr = None
    for line in lines:
        if 'Upper Left' in line:
            match = re.search(r'\(\s*([-\d.]+),\s*([-\d.]+)\)', line)
            if match:
                ul = (float(match.group(1)), float(match.group(2)))
        elif 'Lower Right' in line:
            match = re.search(r'\(\s*([-\d.]+),\s*([-\d.]+)\)', line)
            if match:
                lr = (float(match.group(1)), float(match.group(2)))

    if ul and lr:
        return {
            'minx': ul[0],
            'maxx': lr[0],
            'miny': lr[1],
            'maxy': ul[1]
        }
    return None

def main():
    rasters_dir = Path("/home/jovyan/geoai_veg_map/data/processed/fuel_metrics/volcan_tiles_output/rasters")
    rasters = sorted([f for f in rasters_dir.glob("tile_*_fuel_metrics.tif")])

    print(f"Extracting bounds from {len(rasters)} rasters...")

    # Get all raster bounds
    tile_bounds = {}
    for i, raster in enumerate(rasters):
        bounds = get_raster_bounds(raster)
        if bounds:
            tile_id = raster.stem.replace('_fuel_metrics', '')
            tile_bounds[tile_id] = bounds
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(rasters)}")

    print(f"Successfully extracted {len(tile_bounds)} tile bounds\n")

    # Get reference extents
    print("Extracting reference extents...")
    las_bounds = get_las_bounds()
    merged_bounds = get_merged_bounds()
    print(f"  LAS extent: X=[{las_bounds['minx']:.2f}, {las_bounds['maxx']:.2f}] Y=[{las_bounds['miny']:.2f}, {las_bounds['maxy']:.2f}]")
    print(f"  Merged GeoTIFF: X=[{merged_bounds['minx']:.2f}, {merged_bounds['maxx']:.2f}] Y=[{merged_bounds['miny']:.2f}, {merged_bounds['maxy']:.2f}]\n")

    # Create visualization
    print("Creating visualization...")
    fig, ax = plt.subplots(1, 1, figsize=(16, 14))

    # Plot individual tile bounds
    for tile_id, bounds in tile_bounds.items():
        width = bounds['maxx'] - bounds['minx']
        height = bounds['maxy'] - bounds['miny']
        rect = Rectangle((bounds['minx'], bounds['miny']), width, height,
                         linewidth=0.5, edgecolor='blue', facecolor='none', alpha=0.6)
        ax.add_patch(rect)

    # Plot LAS extent
    las_width = las_bounds['maxx'] - las_bounds['minx']
    las_height = las_bounds['maxy'] - las_bounds['miny']
    las_rect = Rectangle((las_bounds['minx'], las_bounds['miny']), las_width, las_height,
                         linewidth=3, edgecolor='red', facecolor='none', label='Original LAS', linestyle='--')
    ax.add_patch(las_rect)

    # Plot merged GeoTIFF extent
    merged_width = merged_bounds['maxx'] - merged_bounds['minx']
    merged_height = merged_bounds['maxy'] - merged_bounds['miny']
    merged_rect = Rectangle((merged_bounds['minx'], merged_bounds['miny']), merged_width, merged_height,
                           linewidth=3, edgecolor='green', facecolor='none', label='Merged GeoTIFF', linestyle='-.')
    ax.add_patch(merged_rect)

    # Set axis properties
    ax.set_aspect('equal')
    ax.set_xlabel('Easting (UTM Zone 11N, meters)', fontsize=12)
    ax.set_ylabel('Northing (UTM Zone 11N, meters)', fontsize=12)
    ax.set_title('Fuel Metrics Tile Bounds Coverage Analysis', fontsize=14, fontweight='bold')

    # Add legend
    ax.legend(['Original LAS (Expected)', 'Merged GeoTIFF (Actual)', 'Individual Tiles'],
             loc='upper right', fontsize=11)

    # Set limits with some padding
    padding = 100
    ax.set_xlim(las_bounds['minx'] - padding, las_bounds['maxx'] + padding)
    ax.set_ylim(las_bounds['miny'] - padding, las_bounds['maxy'] + padding)

    # Add grid
    ax.grid(True, alpha=0.3, linestyle=':')

    # Save figure
    output_file = "/home/jovyan/geoai_veg_map/tile_bounds_coverage.png"
    fig.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved to: {output_file}\n")

    # Print statistics
    print("=" * 100)
    print("COVERAGE STATISTICS")
    print("=" * 100)

    # Calculate combined extent of all tiles
    all_tiles_minx = min(b['minx'] for b in tile_bounds.values())
    all_tiles_maxx = max(b['maxx'] for b in tile_bounds.values())
    all_tiles_miny = min(b['miny'] for b in tile_bounds.values())
    all_tiles_maxy = max(b['maxy'] for b in tile_bounds.values())

    print(f"\nIndividual tiles combined extent:")
    print(f"  X: [{all_tiles_minx:.2f}, {all_tiles_maxx:.2f}] (width: {all_tiles_maxx - all_tiles_minx:.2f} m)")
    print(f"  Y: [{all_tiles_miny:.2f}, {all_tiles_maxy:.2f}] (height: {all_tiles_maxy - all_tiles_miny:.2f} m)")
    print(f"  Area: {(all_tiles_maxx - all_tiles_minx) * (all_tiles_maxy - all_tiles_miny):,.0f} m²")

    print(f"\nOriginal LAS extent:")
    print(f"  X: [{las_bounds['minx']:.2f}, {las_bounds['maxx']:.2f}] (width: {las_bounds['maxx'] - las_bounds['minx']:.2f} m)")
    print(f"  Y: [{las_bounds['miny']:.2f}, {las_bounds['maxy']:.2f}] (height: {las_bounds['maxy'] - las_bounds['miny']:.2f} m)")
    print(f"  Area: {(las_bounds['maxx'] - las_bounds['minx']) * (las_bounds['maxy'] - las_bounds['miny']):,.0f} m²")

    print(f"\nMerged GeoTIFF extent:")
    print(f"  X: [{merged_bounds['minx']:.2f}, {merged_bounds['maxx']:.2f}] (width: {merged_bounds['maxx'] - merged_bounds['minx']:.2f} m)")
    print(f"  Y: [{merged_bounds['miny']:.2f}, {merged_bounds['maxy']:.2f}] (height: {merged_bounds['maxy'] - merged_bounds['miny']:.2f} m)")
    print(f"  Area: {(merged_bounds['maxx'] - merged_bounds['minx']) * (merged_bounds['maxy'] - merged_bounds['miny']):,.0f} m²")

    # Coverage analysis
    print(f"\nCoverage Analysis:")
    tiles_area = (all_tiles_maxx - all_tiles_minx) * (all_tiles_maxy - all_tiles_miny)
    las_area = (las_bounds['maxx'] - las_bounds['minx']) * (las_bounds['maxy'] - las_bounds['miny'])
    merged_area = (merged_bounds['maxx'] - merged_bounds['minx']) * (merged_bounds['maxy'] - merged_bounds['miny'])

    print(f"  Tiles vs LAS:  {tiles_area / las_area * 100:.1f}% coverage")
    print(f"  Merged vs LAS: {merged_area / las_area * 100:.1f}% coverage")
    print(f"  ⚠️  DATA LOSS: {100 - (merged_area / las_area * 100):.1f}% of expected area missing in merged GeoTIFF")

    # Undercoverage per dimension
    print(f"\nUndercoverage (merged vs LAS):")
    x_left = max(0, las_bounds['minx'] - merged_bounds['minx'])
    x_right = max(0, merged_bounds['maxx'] - las_bounds['maxx'])
    y_bottom = max(0, las_bounds['miny'] - merged_bounds['miny'])
    y_top = max(0, merged_bounds['maxy'] - las_bounds['maxy'])

    if x_left > 0.1:
        print(f"  LEFT edge:   {x_left:.2f} m missing")
    if x_right > 0.1:
        print(f"  RIGHT edge:  {x_right:.2f} m missing")
    if y_bottom > 0.1:
        print(f"  BOTTOM edge: {y_bottom:.2f} m missing")
    if y_top > 0.1:
        print(f"  TOP edge:    {y_top:.2f} m missing")

    print("\n" + "=" * 100)

if __name__ == '__main__':
    main()
