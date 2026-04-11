#!/usr/bin/env python3
"""
Analyze tile edge discontinuities in raster predictions.

This script:
1. Loads site rasters from evaluation output
2. Identifies tile boundaries (10m grid aligned)
3. Computes prediction differences at edges
4. Reports discontinuity statistics and visualizes issues

Usage:
    python src/evaluation/check_tile_edges.py \
        --raster-dir data/output/forest_plot_evaluations/<model>/comparison/site_rasters \
        --output-dir data/output/forest_plot_evaluations/<model>/diagnostics/tile_edges

Example:
    python src/evaluation/check_tile_edges.py \
        --raster-dir data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/comparison/site_rasters \
        --output-dir data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/diagnostics/tile_edges
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    logger.warning("rasterio not available, some features disabled")


def load_raster(path: str) -> Tuple[np.ndarray, dict]:
    """Load raster and metadata."""
    if not HAS_RASTERIO:
        raise ImportError("rasterio required for raster loading")

    with rasterio.open(path) as src:
        data = src.read()
        meta = {
            'transform': src.transform,
            'crs': str(src.crs),
            'width': src.width,
            'height': src.height,
            'bounds': src.bounds,
            'res': src.res,
            'nodata': src.nodata
        }
    return data, meta


def find_tile_boundaries(
    meta: dict,
    tile_size: float = 10.0
) -> Tuple[List[int], List[int]]:
    """
    Find tile boundary indices based on 10m grid alignment.

    The raster is built from 10m tiles with 2m cells (5×5 grid per tile).
    Tile boundaries occur every 5 pixels.

    Args:
        meta: Raster metadata with transform info
        tile_size: Tile size in meters (default 10m)

    Returns:
        (horizontal_boundary_cols, vertical_boundary_rows)
    """
    # Calculate pixel size
    res_x = abs(meta['res'][0])
    res_y = abs(meta['res'][1])

    # Pixels per tile
    pixels_per_tile_x = int(tile_size / res_x)
    pixels_per_tile_y = int(tile_size / res_y)

    # Tile boundaries are at indices that are multiples of pixels_per_tile
    # For 10m tiles with 2m cells: boundaries at cols 5, 10, 15, ...
    h_boundaries = list(range(pixels_per_tile_x, meta['width'], pixels_per_tile_x))
    v_boundaries = list(range(pixels_per_tile_y, meta['height'], pixels_per_tile_y))

    return h_boundaries, v_boundaries


def compute_horizontal_edge_diffs(
    data: np.ndarray,
    boundary_cols: List[int]
) -> np.ndarray:
    """
    Compute differences across horizontal tile boundaries.

    At each boundary column, compare pixels on left vs right.

    Args:
        data: Raster data [bands, height, width]
        boundary_cols: List of column indices where tiles meet

    Returns:
        Array of absolute differences at boundaries
    """
    diffs = []

    for col in boundary_cols:
        if col <= 0 or col >= data.shape[2]:
            continue

        # Compare column col-1 to column col
        left = data[:, :, col - 1]  # [bands, height]
        right = data[:, :, col]

        # Compute absolute difference where both are valid
        valid = ~(np.isnan(left) | np.isnan(right))
        if np.any(valid):
            diff = np.abs(right - left)[valid]
            diffs.extend(diff.flatten())

    return np.array(diffs)


def compute_vertical_edge_diffs(
    data: np.ndarray,
    boundary_rows: List[int]
) -> np.ndarray:
    """
    Compute differences across vertical tile boundaries.

    At each boundary row, compare pixels above vs below.

    Args:
        data: Raster data [bands, height, width]
        boundary_rows: List of row indices where tiles meet

    Returns:
        Array of absolute differences at boundaries
    """
    diffs = []

    for row in boundary_rows:
        if row <= 0 or row >= data.shape[1]:
            continue

        # Compare row row-1 to row row
        above = data[:, row - 1, :]  # [bands, width]
        below = data[:, row, :]

        # Compute absolute difference where both are valid
        valid = ~(np.isnan(above) | np.isnan(below))
        if np.any(valid):
            diff = np.abs(below - above)[valid]
            diffs.extend(diff.flatten())

    return np.array(diffs)


def compute_interior_diffs(data: np.ndarray) -> np.ndarray:
    """
    Compute differences between adjacent pixels in interior (not at boundaries).

    This provides baseline for comparison with edge differences.

    Args:
        data: Raster data [bands, height, width]

    Returns:
        Array of absolute differences in interior
    """
    diffs = []

    # Horizontal differences (interior only - skip every 5th column)
    for col in range(1, data.shape[2]):
        if col % 5 == 0:  # Skip tile boundaries
            continue

        left = data[:, :, col - 1]
        right = data[:, :, col]
        valid = ~(np.isnan(left) | np.isnan(right))
        if np.any(valid):
            diff = np.abs(right - left)[valid]
            diffs.extend(diff.flatten())

    # Vertical differences (interior only - skip every 5th row)
    for row in range(1, data.shape[1]):
        if row % 5 == 0:  # Skip tile boundaries
            continue

        above = data[:, row - 1, :]
        below = data[:, row, :]
        valid = ~(np.isnan(above) | np.isnan(below))
        if np.any(valid):
            diff = np.abs(below - above)[valid]
            diffs.extend(diff.flatten())

    return np.array(diffs)


def analyze_site_raster(
    raster_path: str,
    tile_size: float = 10.0
) -> Dict:
    """
    Analyze tile edge discontinuities for a single site raster.

    Args:
        raster_path: Path to site raster GeoTIFF
        tile_size: Tile size in meters

    Returns:
        Dict with edge discontinuity statistics
    """
    logger.info(f"Analyzing {Path(raster_path).name}")

    data, meta = load_raster(raster_path)

    # Find tile boundaries
    h_bounds, v_bounds = find_tile_boundaries(meta, tile_size)
    logger.info(f"  Found {len(h_bounds)} horizontal, {len(v_bounds)} vertical boundaries")

    # Compute edge differences
    h_diffs = compute_horizontal_edge_diffs(data, h_bounds)
    v_diffs = compute_vertical_edge_diffs(data, v_bounds)
    edge_diffs = np.concatenate([h_diffs, v_diffs]) if len(h_diffs) > 0 or len(v_diffs) > 0 else np.array([])

    # Compute interior differences for comparison
    interior_diffs = compute_interior_diffs(data)

    # Compute statistics
    results = {
        'raster_path': str(raster_path),
        'shape': list(data.shape),
        'n_horizontal_boundaries': len(h_bounds),
        'n_vertical_boundaries': len(v_bounds),
        'edge_diffs': {
            'n': len(edge_diffs),
            'mean': float(np.mean(edge_diffs)) if len(edge_diffs) > 0 else np.nan,
            'std': float(np.std(edge_diffs)) if len(edge_diffs) > 0 else np.nan,
            'median': float(np.median(edge_diffs)) if len(edge_diffs) > 0 else np.nan,
            'max': float(np.max(edge_diffs)) if len(edge_diffs) > 0 else np.nan,
            'p90': float(np.percentile(edge_diffs, 90)) if len(edge_diffs) > 0 else np.nan,
            'p95': float(np.percentile(edge_diffs, 95)) if len(edge_diffs) > 0 else np.nan
        },
        'interior_diffs': {
            'n': len(interior_diffs),
            'mean': float(np.mean(interior_diffs)) if len(interior_diffs) > 0 else np.nan,
            'std': float(np.std(interior_diffs)) if len(interior_diffs) > 0 else np.nan,
            'median': float(np.median(interior_diffs)) if len(interior_diffs) > 0 else np.nan,
            'max': float(np.max(interior_diffs)) if len(interior_diffs) > 0 else np.nan
        }
    }

    # Compute ratio of edge to interior differences
    if len(edge_diffs) > 0 and len(interior_diffs) > 0:
        results['edge_interior_ratio'] = {
            'mean_ratio': results['edge_diffs']['mean'] / results['interior_diffs']['mean'],
            'median_ratio': results['edge_diffs']['median'] / results['interior_diffs']['median']
        }

    logger.info(f"  Edge diff: mean={results['edge_diffs']['mean']:.4f}, max={results['edge_diffs']['max']:.4f}")
    logger.info(f"  Interior diff: mean={results['interior_diffs']['mean']:.4f}")

    return results


def visualize_edge_discontinuities(
    raster_path: str,
    output_path: str,
    tile_size: float = 10.0,
    band: int = 0
) -> None:
    """
    Create visualization of tile edge discontinuities.

    Args:
        raster_path: Path to site raster
        output_path: Output path for figure
        tile_size: Tile size in meters
        band: Which band to visualize (0=Canopy, 1=TFL)
    """
    data, meta = load_raster(raster_path)
    h_bounds, v_bounds = find_tile_boundaries(meta, tile_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Raster with tile boundaries
    ax1 = axes[0]
    band_data = data[band]
    vmin, vmax = np.nanpercentile(band_data, [2, 98])
    im = ax1.imshow(band_data, cmap='viridis', vmin=vmin, vmax=vmax)

    # Draw tile boundaries
    for col in h_bounds:
        ax1.axvline(col - 0.5, color='red', linewidth=0.5, alpha=0.7)
    for row in v_bounds:
        ax1.axhline(row - 0.5, color='red', linewidth=0.5, alpha=0.7)

    ax1.set_title(f'Band {band} with tile boundaries (red)')
    plt.colorbar(im, ax=ax1, label='Prediction value')

    # Right: Edge difference map
    ax2 = axes[1]
    edge_diff_map = np.full_like(band_data, np.nan)

    # Fill in horizontal edge diffs
    for col in h_bounds:
        if col <= 0 or col >= data.shape[2]:
            continue
        diff = np.abs(band_data[:, col] - band_data[:, col - 1])
        edge_diff_map[:, col] = diff
        edge_diff_map[:, col - 1] = diff

    # Fill in vertical edge diffs
    for row in v_bounds:
        if row <= 0 or row >= data.shape[1]:
            continue
        diff = np.abs(band_data[row, :] - band_data[row - 1, :])
        edge_diff_map[row, :] = diff
        edge_diff_map[row - 1, :] = diff

    vmax_diff = np.nanpercentile(edge_diff_map, 95)
    im2 = ax2.imshow(edge_diff_map, cmap='Reds', vmin=0, vmax=vmax_diff)
    ax2.set_title('Tile edge discontinuities')
    plt.colorbar(im2, ax=ax2, label='|Δ prediction|')

    site_name = Path(raster_path).stem.replace('_raster', '')
    plt.suptitle(f'{site_name} - Tile Edge Analysis', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"  Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze tile edge discontinuities in raster predictions'
    )
    parser.add_argument(
        '--raster-dir',
        type=str,
        required=True,
        help='Directory containing site raster GeoTIFFs'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )
    parser.add_argument(
        '--tile-size',
        type=float,
        default=10.0,
        help='Tile size in meters (default: 10.0)'
    )

    args = parser.parse_args()

    if not HAS_RASTERIO:
        logger.error("rasterio required for this script")
        sys.exit(1)

    raster_dir = Path(args.raster_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all site rasters
    raster_files = list(raster_dir.glob('*_raster.tif'))
    if not raster_files:
        # Try alternative pattern
        raster_files = list(raster_dir.glob('*.tif'))

    logger.info(f"Found {len(raster_files)} raster files")

    all_results = {}
    for raster_path in raster_files:
        try:
            results = analyze_site_raster(str(raster_path), args.tile_size)
            site_name = raster_path.stem.replace('_raster', '')
            all_results[site_name] = results

            # Create visualization
            viz_path = output_dir / f'{site_name}_edge_analysis.png'
            visualize_edge_discontinuities(
                str(raster_path), str(viz_path),
                args.tile_size, band=0
            )

        except Exception as e:
            logger.error(f"Error processing {raster_path}: {e}")
            continue

    # Save combined results
    with open(output_dir / 'edge_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create summary table
    summary_rows = []
    for site, results in all_results.items():
        summary_rows.append({
            'site': site,
            'edge_mean': results['edge_diffs']['mean'],
            'edge_max': results['edge_diffs']['max'],
            'edge_p95': results['edge_diffs']['p95'],
            'interior_mean': results['interior_diffs']['mean'],
            'ratio_mean': results.get('edge_interior_ratio', {}).get('mean_ratio', np.nan)
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / 'edge_analysis_summary.csv', index=False)

    logger.info(f"\nSaved results to {output_dir}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("EDGE DISCONTINUITY SUMMARY")
    logger.info("="*60)
    logger.info(f"{'Site':<15} {'Edge Mean':>10} {'Edge Max':>10} {'Edge P95':>10} {'Ratio':>8}")
    logger.info("-"*60)
    for _, row in summary_df.iterrows():
        logger.info(
            f"{row['site']:<15} {row['edge_mean']:>10.4f} {row['edge_max']:>10.4f} "
            f"{row['edge_p95']:>10.4f} {row['ratio_mean']:>8.2f}x"
        )


# Need pandas for summary table
try:
    import pandas as pd
except ImportError:
    pd = None


if __name__ == '__main__':
    main()
