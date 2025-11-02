#!/usr/bin/env python3
"""
Validate LidarForFuel fuel metrics against forest plot field measurements.

This script compares the distribution of fuel metrics from LidarForFuel raster
outputs with ground truth measurements from forest plots. Note that this is a
non-spatially-matched comparison (no spatial overlap between datasets).

Usage:
    python src/fuel_metrics/validate_fuel_metrics.py \
        --raster data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_5.0m.tif \
        --forest-plots data/processed/forest_plot_data/forest_plots_processed.csv \
        --output-dir data/processed/fuel_metrics/volcan_mtn/validation
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import seaborn as sns
from scipy.ndimage import uniform_filter


# Unit conversion constants
TONS_ACRE_TO_KG_M2 = 0.2242  # 1 ton/acre = 0.2242 kg/m²

# Forest plot spatial parameters
PLOT_SIZE_ACRES = 0.1
ACRES_TO_M2 = 4046.86  # 1 acre = 4046.86 m²
PLOT_AREA_M2 = PLOT_SIZE_ACRES * ACRES_TO_M2  # 404.7 m²
PLOT_DIAMETER_M = 2 * np.sqrt(PLOT_AREA_M2 / np.pi)  # ~22.7 m for circular plot


def load_forest_plot_data(csv_path: Path, site_filter: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and prepare forest plot data.

    Args:
        csv_path: Path to forest plot CSV
        site_filter: Optional site name to filter to (e.g., "Laguna")

    Returns:
        Tuple of (fuels_df, cover_df) where:
        - fuels_df: All plots with TotalFuels (converted to kg/m²)
        - cover_df: Only plots with TreeCover (% values)
    """
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} forest plots from {csv_path}")

    # Apply site filter if specified
    if site_filter:
        df = df[df['Site'] == site_filter].copy()
        print(f"Filtered to site '{site_filter}': {len(df)} plots remaining")

    # Prepare TotalFuels data (all plots, convert units)
    fuels_df = df[['Plot_ID', 'Site', 'TotalFuels']].copy()
    fuels_df['TotalFuels_kg_m2'] = fuels_df['TotalFuels'] * TONS_ACRE_TO_KG_M2
    fuels_df = fuels_df.dropna(subset=['TotalFuels_kg_m2'])

    print(f"TotalFuels: {len(fuels_df)} plots with data")
    print(f"  Original units (tons/acre): {fuels_df['TotalFuels'].min():.2f} - {fuels_df['TotalFuels'].max():.2f}")
    print(f"  Converted units (kg/m²): {fuels_df['TotalFuels_kg_m2'].min():.2f} - {fuels_df['TotalFuels_kg_m2'].max():.2f}")

    # Prepare TreeCover data (only plots with values)
    cover_df = df[['Plot_ID', 'Site', 'TreeCover']].copy()
    cover_df = cover_df.dropna(subset=['TreeCover'])

    print(f"TreeCover: {len(cover_df)} plots with data (excluded {len(df) - len(cover_df)} NaN values)")
    print(f"  Range: {cover_df['TreeCover'].min():.1f}% - {cover_df['TreeCover'].max():.1f}%")

    return fuels_df, cover_df


def aggregate_raster_to_plot_scale(
    raster_data: np.ndarray,
    pixel_size: float,
    plot_diameter: float
) -> np.ndarray:
    """
    Aggregate fine-resolution raster to match forest plot spatial scale.

    Uses moving window averaging to simulate the spatial integration
    that occurs in field plot measurements. NaN-aware aggregation.

    Args:
        raster_data: 2D array of raster values
        pixel_size: Resolution of input raster in meters
        plot_diameter: Diameter of forest plot in meters

    Returns:
        Aggregated raster at plot scale
    """
    # Calculate aggregation window size (in pixels)
    window_size_pixels = int(np.round(plot_diameter / pixel_size))

    # Ensure odd window size for symmetric kernel
    if window_size_pixels % 2 == 0:
        window_size_pixels += 1

    print(f"  Aggregating with {window_size_pixels}×{window_size_pixels} pixel window")
    print(f"  Window covers {(window_size_pixels * pixel_size):.1f}m diameter")

    # NaN-aware moving window average
    # Replace NaN with 0 for uniform_filter, then correct using count of valid pixels
    mask = ~np.isnan(raster_data)
    filled_data = np.where(mask, raster_data, 0.0)

    # Sum of valid values in each window
    sum_filtered = uniform_filter(filled_data, size=window_size_pixels, mode='constant', cval=0.0)

    # Count of valid values in each window
    count_filtered = uniform_filter(mask.astype(float), size=window_size_pixels, mode='constant', cval=0.0)

    # Average = sum / count (avoid division by zero)
    aggregated = np.where(count_filtered > 0, sum_filtered / count_filtered, np.nan)

    return aggregated


def load_raster_data(raster_path: Path, tfl_band: int = 16, surface_band: int = 20, cover_band: int = 23) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and extract valid values from LidarForFuel raster.

    Args:
        raster_path: Path to fuel metrics raster
        tfl_band: Band number for Total Fuel Load >1m (kg/m²)
        surface_band: Band number for Surface Fuel Load 0-1m (kg/m²)
        cover_band: Band number for Canopy Cover (%)

    Returns:
        Tuple of (combined_fuels, cover_values_fine, cover_values_aggregated)
        where combined_fuels = surface (0-1m) + standing (>1m)
    """
    with rasterio.open(raster_path) as src:
        pixel_size = src.res[0]  # Assuming square pixels

        print(f"\nLoaded raster: {raster_path}")
        print(f"  Size: {src.width} × {src.height} pixels")
        print(f"  Resolution: {pixel_size} m")
        print(f"  Bands: {src.count}")
        print(f"  CRS: {src.crs}")

        # Read fuel bands and combine
        tfl = src.read(tfl_band)  # Standing fuels >1m
        fl_0_1 = src.read(surface_band)  # Surface fuels 0-1m

        # Combined total fuels (surface + standing)
        combined_fuels = tfl + fl_0_1
        combined_valid = combined_fuels[~np.isnan(combined_fuels) & (combined_fuels > 0)]

        # Read Cover band (Band 23) as 2D array for aggregation
        cover = src.read(cover_band)

        # Convert from fraction (0-1) to percentage (0-100) if needed
        if np.nanmax(cover) <= 1.0:
            print(f"\nBand {cover_band} (Cover): Converting from fraction to percentage")
            cover = cover * 100.0

        print(f"\nFuel Load Calculation:")
        print(f"  Band {surface_band} (FL_0_1, 0-1m) + Band {tfl_band} (TFL, >1m) = Combined Total")
        print(f"  Valid pixels: {len(combined_valid):,} / {combined_fuels.size:,}")
        print(f"  Range: {combined_valid.min():.2f} - {combined_valid.max():.2f} kg/m²")
        print(f"  Mean: {combined_valid.mean():.2f} kg/m², Median: {np.median(combined_valid):.2f} kg/m²")

        # Extract fine-scale cover values
        cover_valid_fine = cover[~np.isnan(cover) & (cover >= 0)]

        print(f"\nBand {cover_band} (Cover at {pixel_size}m resolution):")
        print(f"  Valid pixels: {len(cover_valid_fine):,} / {cover.size:,}")
        print(f"  Range: {cover_valid_fine.min():.1f} - {cover_valid_fine.max():.1f}%")

        # Aggregate cover to plot scale
        print(f"\nAggregating cover to plot scale ({PLOT_DIAMETER_M:.1f}m diameter)...")
        pixels_per_plot = PLOT_AREA_M2 / (pixel_size ** 2)
        print(f"  Forest plot covers {pixels_per_plot:.1f} pixels at {pixel_size}m resolution")

        cover_aggregated = aggregate_raster_to_plot_scale(
            cover,
            pixel_size,
            PLOT_DIAMETER_M
        )

        # Extract valid aggregated values
        cover_valid_aggregated = cover_aggregated[~np.isnan(cover_aggregated) & (cover_aggregated >= 0)]

        print(f"\nAggregated cover (at ~{PLOT_DIAMETER_M:.1f}m scale):")
        print(f"  Valid pixels: {len(cover_valid_aggregated):,}")
        print(f"  Range: {cover_valid_aggregated.min():.1f} - {cover_valid_aggregated.max():.1f}%")

        return combined_valid, cover_valid_fine, cover_valid_aggregated


def calculate_statistics(data: np.ndarray, name: str) -> Dict[str, float]:
    """Calculate summary statistics for a dataset."""
    stats = {
        'name': name,
        'count': len(data),
        'mean': np.mean(data),
        'median': np.median(data),
        'std': np.std(data),
        'min': np.min(data),
        'max': np.max(data),
        'q25': np.percentile(data, 25),
        'q75': np.percentile(data, 75)
    }
    return stats


def plot_distribution_comparison(
    fuels_plots: np.ndarray,
    fuels_raster: np.ndarray,
    cover_plots: np.ndarray,
    cover_raster_fine: np.ndarray,
    cover_raster_aggregated: np.ndarray,
    output_path: Path
):
    """
    Create 3×2 comparison figure with histograms and box plots at multiple scales.

    Args:
        fuels_plots: Forest plot TotalFuels (kg/m²)
        fuels_raster: Raster TFL values (kg/m²)
        cover_plots: Forest plot TreeCover (%)
        cover_raster_fine: Raster Cover values at fine scale (5m) (%)
        cover_raster_aggregated: Raster Cover values at plot scale (20m) (%)
        output_path: Output PNG path
    """
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['font.size'] = 10

    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))
    fig.suptitle('LidarForFuel Validation: Distribution Comparison with Spatial Scale Analysis\n(Non-spatially matched)',
                 fontsize=14, fontweight='bold', y=0.995)

    # Colors
    plot_color = '#2E86AB'  # Blue
    raster_fine_color = '#E63946'  # Red
    raster_agg_color = '#F77F00'   # Orange

    # ==================== Row 1: Total Fuel Load ====================

    # Histogram (overlaid)
    ax = axes[0, 0]
    ax.hist(fuels_plots, bins=30, alpha=0.6, label=f'Forest Plots (n={len(fuels_plots)})',
            color=plot_color, edgecolor='black', linewidth=0.5)
    ax.hist(fuels_raster, bins=30, alpha=0.6, label=f'LiDAR (FL_0_1+TFL) (n={len(fuels_raster):,})',
            color=raster_fine_color, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Total Fuel Load (kg/m²)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Total Fuel Load Distribution\n(Forest Plots vs LiDAR Surface+Standing)', fontweight='bold', fontsize=10)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f'Forest Plots: μ={np.mean(fuels_plots):.2f}, σ={np.std(fuels_plots):.2f}\n'
    stats_text += f'LiDAR (combined): μ={np.mean(fuels_raster):.2f}, σ={np.std(fuels_raster):.2f}\n'
    stats_text += f'Ratio: {np.mean(fuels_raster) / np.mean(fuels_plots):.2f}×'
    ax.text(0.98, 0.70, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Box plot
    ax = axes[0, 1]
    bp = ax.boxplot([fuels_plots, fuels_raster],
                     labels=['Forest Plots', 'LiDAR\n(FL_0_1+TFL)'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black', markersize=6))
    bp['boxes'][0].set_facecolor(plot_color)
    bp['boxes'][1].set_facecolor(raster_fine_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('Total Fuel Load (kg/m²)', fontweight='bold')
    ax.set_title('Total Fuel Load Box Plots', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # ==================== Row 2: Canopy Cover - Scale Effect ====================

    # Histogram - Three scales
    ax = axes[1, 0]
    ax.hist(cover_raster_fine, bins=25, alpha=0.5, label=f'5m pixels (n={len(cover_raster_fine):,})',
            color=raster_fine_color, edgecolor='black', linewidth=0.5, range=(0, 100))
    ax.hist(cover_raster_aggregated, bins=25, alpha=0.5, label=f'20m aggregated (n={len(cover_raster_aggregated):,})',
            color=raster_agg_color, edgecolor='black', linewidth=0.5, range=(0, 100))
    ax.hist(cover_plots, bins=25, alpha=0.5, label=f'Field plots 20m (n={len(cover_plots)})',
            color=plot_color, edgecolor='black', linewidth=0.5, range=(0, 100))
    ax.set_xlabel('Canopy Cover (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Canopy Cover: All Spatial Scales', fontweight='bold')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # Box plot - Three scales
    ax = axes[1, 1]
    bp = ax.boxplot([cover_raster_fine, cover_raster_aggregated, cover_plots],
                     labels=['5m\nLiDAR', '20m\nLiDAR', '20m\nField'],
                     patch_artist=True,
                     widths=0.5,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black', markersize=6))
    bp['boxes'][0].set_facecolor(raster_fine_color)
    bp['boxes'][1].set_facecolor(raster_agg_color)
    bp['boxes'][2].set_facecolor(plot_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('Canopy Cover (%)', fontweight='bold')
    ax.set_title('Distribution Comparison Across Scales', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # ==================== Row 3: Matched Scale Comparison ====================

    # Histogram - Matched scales only
    ax = axes[2, 0]
    ax.hist(cover_raster_aggregated, bins=25, alpha=0.6, label=f'20m LiDAR (n={len(cover_raster_aggregated):,})',
            color=raster_agg_color, edgecolor='black', linewidth=0.5, range=(0, 100))
    ax.hist(cover_plots, bins=25, alpha=0.6, label=f'20m Field (n={len(cover_plots)})',
            color=plot_color, edgecolor='black', linewidth=0.5, range=(0, 100))
    ax.set_xlabel('Canopy Cover (%)', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title('Matched Spatial Scale (20m)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)

    # Add statistics text
    scale_effect = np.mean(cover_raster_aggregated) - np.mean(cover_raster_fine)
    method_diff = np.mean(cover_raster_aggregated) - np.mean(cover_plots)
    stats_text = f'LiDAR 20m: μ={np.mean(cover_raster_aggregated):.1f}%, σ={np.std(cover_raster_aggregated):.1f}%\n'
    stats_text += f'Field 20m: μ={np.mean(cover_plots):.1f}%, σ={np.std(cover_plots):.1f}%\n'
    stats_text += f'Difference: {method_diff:+.1f}%\n'
    stats_text += f'Scale effect (5m→20m): {scale_effect:+.1f}%'
    ax.text(0.98, 0.70, stats_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Box plot - Matched scales with ratio annotation
    ax = axes[2, 1]
    bp = ax.boxplot([cover_raster_aggregated, cover_plots],
                     labels=['20m LiDAR\n(aggregated)', '20m Field\n(plots)'],
                     patch_artist=True,
                     widths=0.6,
                     showmeans=True,
                     meanprops=dict(marker='D', markerfacecolor='yellow', markeredgecolor='black', markersize=6))
    bp['boxes'][0].set_facecolor(raster_agg_color)
    bp['boxes'][1].set_facecolor(plot_color)
    for box in bp['boxes']:
        box.set_alpha(0.6)
    ax.set_ylabel('Canopy Cover (%)', fontweight='bold')
    ax.set_title('Method Comparison at Equal Scale', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    # Add ratio annotation
    ratio = np.mean(cover_raster_aggregated) / np.mean(cover_plots)
    ax.text(0.5, 0.95, f'LiDAR/Field ratio: {ratio:.2f}×',
            transform=ax.transAxes, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7),
            fontsize=10, fontweight='bold')

    # Adjust layout and save
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved figure to: {output_path}")
    plt.close()


def save_statistics_table(
    fuels_plots: np.ndarray,
    fuels_raster: np.ndarray,
    cover_plots: np.ndarray,
    cover_raster_fine: np.ndarray,
    cover_raster_aggregated: np.ndarray,
    output_path: Path
):
    """Save summary statistics to CSV."""
    stats = []

    # Total Fuel Load
    stats.append(calculate_statistics(fuels_plots, 'TotalFuels_ForestPlots_kg_m2'))
    stats.append(calculate_statistics(fuels_raster, 'TotalFuelLoad_LiDAR_FL01+TFL_kg_m2'))

    # Canopy Cover - Multiple scales
    stats.append(calculate_statistics(cover_raster_fine, 'CanopyCover_LiDAR_5m_percent'))
    stats.append(calculate_statistics(cover_raster_aggregated, 'CanopyCover_LiDAR_20m_percent'))
    stats.append(calculate_statistics(cover_plots, 'TreeCover_FieldPlots_20m_percent'))

    df = pd.DataFrame(stats)
    df = df.round(2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved statistics to: {output_path}")

    # Print to console
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Validate LidarForFuel fuel metrics against forest plot measurements'
    )
    parser.add_argument(
        '--raster',
        type=Path,
        default=Path('data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_5.0m.tif'),
        help='Path to LidarForFuel fuel metrics raster'
    )
    parser.add_argument(
        '--forest-plots',
        type=Path,
        default=Path('data/processed/forest_plot_data/forest_plots_processed.csv'),
        help='Path to forest plot CSV'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/processed/fuel_metrics/volcan_mtn/validation'),
        help='Output directory for validation results'
    )
    parser.add_argument(
        '--tfl-band',
        type=int,
        default=16,
        help='Band number for Total Fuel Load (default: 16)'
    )
    parser.add_argument(
        '--cover-band',
        type=int,
        default=23,
        help='Band number for Canopy Cover (default: 23)'
    )
    parser.add_argument(
        '--site-filter',
        type=str,
        default=None,
        help='Filter forest plots to specific site (e.g., "Laguna", "BluffMesa")'
    )

    args = parser.parse_args()

    print("="*80)
    print("LIDARFORFUEL VALIDATION")
    print("="*80)
    print(f"Raster: {args.raster}")
    print(f"Forest plots: {args.forest_plots}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Load data
    print("\n[1/4] Loading forest plot data...")
    fuels_df, cover_df = load_forest_plot_data(args.forest_plots, args.site_filter)

    print("\n[2/4] Loading raster data and aggregating to plot scale...")
    tfl_raster, cover_raster_fine, cover_raster_aggregated = load_raster_data(args.raster, args.tfl_band, 20, args.cover_band)

    # Extract arrays
    fuels_plots = fuels_df['TotalFuels_kg_m2'].values
    cover_plots = cover_df['TreeCover'].values

    # Generate visualizations
    print("\n[3/4] Generating multi-scale distribution comparison figure...")
    plot_path = args.output_dir / 'distribution_comparison.png'
    plot_distribution_comparison(fuels_plots, tfl_raster, cover_plots, cover_raster_fine, cover_raster_aggregated, plot_path)

    # Save statistics
    print("\n[4/4] Saving summary statistics...")
    stats_path = args.output_dir / 'summary_statistics.csv'
    save_statistics_table(fuels_plots, tfl_raster, cover_plots, cover_raster_fine, cover_raster_aggregated, stats_path)

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print(f"Outputs saved to: {args.output_dir}")
    print("  - distribution_comparison.png")
    print("  - summary_statistics.csv")


if __name__ == '__main__':
    main()
