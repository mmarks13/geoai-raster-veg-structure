#!/usr/bin/env python3
"""
Compare 3DEP baseline raster predictions to forest plot field measurements.

This script compares vegetation structure metrics computed directly from 3DEP
LiDAR to field measurements, establishing a baseline for multimodal model comparison.

Usage:
    python src/evaluation/compare_baseline_to_plots.py \
        --baseline-rasters-dir data/processed/veg_structure_baseline \
        --field-data data/processed/forest_plot_data/forest_plots_processed.gpkg \
        --band-config src/evaluation/configs/raster/veg_structure_baseline.json \
        --output data/processed/veg_structure_baseline/comparison

Output:
    - baseline_comparison_results.csv: Per-plot baseline and field measurements
    - baseline_comparison_stats.json: Summary statistics per band
    - baseline_comparison_scatter.png: Scatter plots per band
    - baseline_comparison_by_site.png: Per-site comparison plots
    - baseline_per_site_summary.csv: Per-site statistics
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from scipy import stats as scipy_stats
from scipy.stats import spearmanr
from shapely.geometry import box

# Add script directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))
from band_config import load_band_config, BandConfig, BandInfo

# Import functions from existing comparison script
from compare_predictions_to_plots import (
    extract_raster_values_at_footprint,
    create_plot_footprints,
    compute_statistics,
    compute_per_site_summary,
    plot_comparisons,
    plot_by_site,
    plot_rank_scatter,
    plot_rank_by_site,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_baseline_rasters(rasters_dir: Path) -> Dict[str, str]:
    """
    Load paths to baseline raster files.

    Baseline rasters are stored as {site}/veg_structure_2m.tif

    Returns:
        Dict mapping site_name to raster file path
    """
    logger.info(f"Loading baseline rasters from {rasters_dir}")

    # Look for pattern: {site}/veg_structure_2m.tif
    raster_files = list(rasters_dir.glob('*/veg_structure_2m.tif'))

    if not raster_files:
        raise FileNotFoundError(
            f"No baseline raster files found in {rasters_dir}. "
            "Expected pattern: {site}/veg_structure_2m.tif"
        )

    site_rasters = {}
    for raster_path in raster_files:
        # Extract site name from parent directory
        site_name = raster_path.parent.name
        site_rasters[site_name] = str(raster_path)

    logger.info(f"Found {len(site_rasters)} baseline rasters")
    for site_name in sorted(site_rasters.keys()):
        logger.info(f"  - {site_name}: {site_rasters[site_name]}")

    return site_rasters


def compare_baseline_to_field(
    field_gdf: gpd.GeoDataFrame,
    site_rasters: Dict[str, str],
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> pd.DataFrame:
    """
    Compare baseline raster values to field measurements.

    Args:
        field_gdf: GeoDataFrame with plot footprints and field measurements
        site_rasters: Dict mapping site_name to raster file path
        band_config: Band configuration
        coverage_threshold: Minimum coverage fraction (default 0.99 = 99%)

    Returns:
        DataFrame with per-plot baseline and field measurements
    """
    logger.info(f"Comparing baseline to {len(field_gdf)} field plots")
    logger.info(f"Coverage threshold: {coverage_threshold:.0%}")

    results = []

    for idx, row in field_gdf.iterrows():
        plot_id = row.get('PlotID', row.get('plot_id', idx))
        site_name = row.get('Site', row.get('site_name', 'unknown'))
        footprint = row.geometry

        # Skip if site raster not available
        if site_name not in site_rasters:
            logger.debug(f"No raster found for site {site_name}, skipping plot {plot_id}")
            continue

        raster_path = site_rasters[site_name]

        # Base result dict
        result = {
            'plot_id': plot_id,
            'site_name': site_name,
            'plot_x': footprint.centroid.x,
            'plot_y': footprint.centroid.y,
        }

        # Extract baseline values for each band
        for band in band_config.bands:
            band_name = band.name

            try:
                # Extract raster values
                extraction = extract_raster_values_at_footprint(
                    raster_path, footprint, band.output_index, band.aggregation_method
                )

                # Raw values
                baseline_value = extraction['weighted_mean']
                result[f'{band_name}_pred_raw'] = baseline_value
                result[f'{band_name}_n_pixels'] = extraction['n_pixels']
                result[f'{band_name}_coverage_fraction'] = extraction['coverage_fraction']

                # Convert to display units if not NaN
                if not np.isnan(baseline_value):
                    baseline_display = band.convert_to_display_units(baseline_value)
                    result[f'{band_name}_pred'] = baseline_display
                else:
                    result[f'{band_name}_pred'] = np.nan

            except Exception as e:
                logger.warning(f"Error extracting {band_name} for plot {plot_id}: {e}")
                result[f'{band_name}_pred_raw'] = np.nan
                result[f'{band_name}_pred'] = np.nan
                result[f'{band_name}_n_pixels'] = 0
                result[f'{band_name}_coverage_fraction'] = 0.0

            # Field measurement (if exists)
            if band.field_column and band.field_column in row:
                result[f'{band_name}_field'] = row[band.field_column]
            else:
                result[f'{band_name}_field'] = np.nan

        results.append(result)

    df = pd.DataFrame(results)
    logger.info(f"Extracted baseline values for {len(df)} plots")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3DEP baseline raster predictions to forest plot field measurements"
    )
    parser.add_argument(
        '--baseline-rasters-dir',
        type=str,
        required=True,
        help='Directory containing per-site baseline raster GeoTIFFs (pattern: {site}/veg_structure_2m.tif)'
    )
    parser.add_argument(
        '--field-data',
        type=str,
        required=True,
        help='Path to field measurements GeoPackage'
    )
    parser.add_argument(
        '--band-config',
        type=str,
        required=True,
        help='Path to band configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for comparison results'
    )
    parser.add_argument(
        '--footprint-radius',
        type=float,
        default=11.35,
        help='Plot footprint radius in meters (default: 11.35 = 0.1-acre)'
    )
    parser.add_argument(
        '--coverage-threshold',
        type=float,
        default=0.99,
        help='Minimum coverage fraction for inclusion (default: 0.99)'
    )
    parser.add_argument(
        '--crs',
        type=str,
        default='EPSG:32611',
        help='Coordinate reference system (default: EPSG:32611)'
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load band configuration
    band_config = load_band_config(args.band_config)
    logger.info(f"Loaded band config: {band_config.name}")
    logger.info(f"Bands with field mapping: {', '.join([b.display_name for b in band_config.get_bands_with_field_mapping()])}")

    # Load baseline rasters
    rasters_dir = Path(args.baseline_rasters_dir)
    site_rasters = load_baseline_rasters(rasters_dir)

    # Load field data
    logger.info(f"Loading field data from {args.field_data}")
    field_gdf = gpd.read_file(args.field_data)
    logger.info(f"Loaded {len(field_gdf)} field plots")

    # Reproject if needed
    if str(field_gdf.crs) != args.crs:
        logger.info(f"Reprojecting field plots from {field_gdf.crs} to {args.crs}")
        field_gdf = field_gdf.to_crs(args.crs)

    # Create footprints
    field_footprints = create_plot_footprints(field_gdf, radius_m=args.footprint_radius)

    # Compare baseline to field
    comparison_df = compare_baseline_to_field(
        field_footprints,
        site_rasters,
        band_config,
        coverage_threshold=args.coverage_threshold
    )

    # Save comparison results
    results_path = output_dir / 'baseline_comparison_results.csv'
    comparison_df.to_csv(results_path, index=False)
    logger.info(f"Saved comparison results to {results_path}")

    # Compute statistics
    stats_dict = compute_statistics(
        comparison_df,
        band_config,
        coverage_threshold=args.coverage_threshold
    )

    # Save statistics
    stats_path = output_dir / 'baseline_comparison_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    # Compute and save per-site summary
    site_summary_df = compute_per_site_summary(
        comparison_df,
        band_config,
        coverage_threshold=args.coverage_threshold
    )
    site_summary_path = output_dir / 'baseline_per_site_summary.csv'
    site_summary_df.to_csv(site_summary_path, index=False)
    logger.info(f"Saved per-site summary to {site_summary_path}")

    # Generate plots
    logger.info("Generating comparison figures")

    # Rename output files for baseline
    plot_comparisons(
        comparison_df,
        band_config,
        stats_dict,
        output_dir,
        coverage_threshold=args.coverage_threshold
    )
    # Rename the default output file
    default_scatter = output_dir / 'comparison_scatter.png'
    baseline_scatter = output_dir / 'baseline_comparison_scatter.png'
    if default_scatter.exists():
        default_scatter.rename(baseline_scatter)

    plot_by_site(
        comparison_df,
        band_config,
        output_dir / 'baseline_comparison_by_site.png',
        coverage_threshold=args.coverage_threshold
    )

    plot_rank_scatter(
        comparison_df,
        band_config,
        stats_dict,
        output_dir / 'baseline_comparison_rank_scatter.png',
        coverage_threshold=args.coverage_threshold
    )

    plot_rank_by_site(
        comparison_df,
        band_config,
        output_dir / 'baseline_comparison_rank_by_site.png',
        coverage_threshold=args.coverage_threshold
    )

    logger.info("\n" + "=" * 60)
    logger.info("BASELINE COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nFiles generated:")
    logger.info(f"  - {results_path.name}")
    logger.info(f"  - {stats_path.name}")
    logger.info(f"  - {site_summary_path.name}")
    logger.info(f"  - baseline_comparison_scatter.png")
    logger.info(f"  - baseline_comparison_by_site.png")
    logger.info(f"  - baseline_comparison_rank_scatter.png")
    logger.info(f"  - baseline_comparison_rank_by_site.png")


if __name__ == '__main__':
    main()
