#!/usr/bin/env python3
"""
Compare model raster predictions to forest plot field measurements.

This script compares per-site prediction rasters to field measurement data,
computing statistics and generating comparison figures. Uses band configuration
to support arbitrary numbers of prediction bands.

Usage:
    python src/evaluation/compare_predictions_to_plots.py \
        --site-rasters-dir comparison/site_rasters \
        --field-data forest_plots_processed.gpkg \
        --band-config src/evaluation/configs/raster/cover_only.json \
        --output comparison/

Output:
    - comparison_results.csv: Per-plot predictions and field measurements
    - comparison_stats.json: Summary statistics per band
    - comparison_scatter.png: Scatter plots per band
    - comparison_by_site.png: Per-site comparison plots
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

# Add script directory to path for local imports (avoids PYTHONPATH requirement)
sys.path.insert(0, str(Path(__file__).parent))
from band_config import load_band_config, BandConfig, BandInfo

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_site_rasters(rasters_dir: Path) -> Dict[str, str]:
    """
    Load paths to site raster files.
    
    Returns:
        Dict mapping site_name to raster file path
    """
    logger.info(f"Loading site rasters from {rasters_dir}")
    
    raster_files = list(rasters_dir.glob('*_predictions_raster.tif'))
    
    if not raster_files:
        raise FileNotFoundError(f"No raster files found in {rasters_dir}")
    
    site_rasters = {}
    for raster_path in raster_files:
        # Extract site name from filename (e.g., "BluffMesa_predictions_raster.tif" -> "BluffMesa")
        site_name = raster_path.stem.replace('_predictions_raster', '')
        site_rasters[site_name] = str(raster_path)
    
    logger.info(f"Found {len(site_rasters)} site rasters")
    for site_name in sorted(site_rasters.keys()):
        logger.info(f"  - {site_name}")
    
    return site_rasters


def extract_raster_values_at_footprint(
    raster_path: str,
    footprint: 'Polygon',
    band_idx: int,
    aggregation_method: str = 'mean'
) -> Dict[str, float]:
    """
    Extract raster values within plot footprint.

    Args:
        raster_path: Path to raster file
        footprint: Shapely polygon (plot boundary)
        band_idx: Band index to extract (0-indexed)
        aggregation_method: 'mean' for area-weighted average, 'max' for maximum value across pixels with >=50% overlap

    Returns:
        Dict with weighted_mean (or max_value for method='max'), n_pixels, coverage_fraction
    """
    with rasterio.open(raster_path) as src:
        # Get footprint bounds with padding to ensure all edge pixels are captured
        minx, miny, maxx, maxy = footprint.bounds
        padding = 1.5  # meters - ensures window captures all pixels touching circular footprint

        # Read raster window
        window = rasterio.windows.from_bounds(
            minx - padding, miny - padding,
            maxx + padding, maxy + padding,
            transform=src.transform
        )

        # Read band data
        data = src.read(band_idx + 1, window=window)  # rasterio uses 1-indexed bands

        # Get pixel coordinates
        transform = src.window_transform(window)

        # Compute pixel area (assuming square pixels)
        pixel_area = abs(transform.a * transform.e)

        # Initialize accumulators
        weighted_sum = 0.0
        weight_sum = 0.0
        n_pixels = 0
        max_value = -np.inf

        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                # Pixel center coordinates
                pixel_x, pixel_y = rasterio.transform.xy(transform, row, col, offset='center')

                # Pixel bounds
                pixel_minx = pixel_x - transform.a / 2
                pixel_maxx = pixel_x + transform.a / 2
                pixel_miny = pixel_y - abs(transform.e) / 2
                pixel_maxy = pixel_y + abs(transform.e) / 2

                pixel_box = box(pixel_minx, pixel_miny, pixel_maxx, pixel_maxy)

                # Compute intersection area
                try:
                    intersection = footprint.intersection(pixel_box)
                    if intersection.is_empty:
                        continue
                    area = intersection.area
                except Exception:
                    continue

                if area < 1e-6:
                    continue

                # Get pixel value
                value = data[row, col]
                if np.isnan(value):
                    continue

                # Accumulate for mean calculation
                weighted_sum += value * area
                weight_sum += area
                n_pixels += 1

                # For max aggregation: check if pixel has >=50% overlap
                if aggregation_method == 'max':
                    overlap_fraction = area / pixel_area
                    if overlap_fraction >= 0.5:
                        max_value = max(max_value, value)

        # Compute results
        footprint_area = footprint.area
        coverage_fraction = weight_sum / footprint_area if footprint_area > 0 else 0.0

        if aggregation_method == 'max':
            result_value = max_value if max_value != -np.inf else np.nan
        else:
            result_value = weighted_sum / weight_sum if weight_sum > 0 else np.nan

        return {
            'weighted_mean': result_value,
            'n_pixels': n_pixels,
            'coverage_fraction': coverage_fraction
        }


def create_plot_footprints(plots_gdf: gpd.GeoDataFrame, radius_m: float = 11.35) -> gpd.GeoDataFrame:
    """Convert plot point geometries to circular footprint polygons."""
    logger.info(f"Creating circular plot footprints (radius={radius_m}m)")
    
    result = plots_gdf.copy()
    footprints = [geom.buffer(radius_m) for geom in result.geometry]
    result['geometry'] = footprints
    result['footprint_area_m2'] = result.geometry.area
    
    logger.info(f"Created {len(footprints)} footprints (mean area: {result['footprint_area_m2'].mean():.1f} m²)")
    
    return result


def compare_predictions_to_field(
    field_gdf: gpd.GeoDataFrame,
    site_rasters: Dict[str, str],
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> pd.DataFrame:
    """
    Compare raster predictions to field measurements.
    
    Args:
        field_gdf: GeoDataFrame with plot footprints and field measurements
        site_rasters: Dict mapping site_name to raster file path
        band_config: Band configuration
        coverage_threshold: Minimum coverage fraction (default 0.99 = 99%)
    
    Returns:
        DataFrame with per-plot predictions and field measurements
    """
    logger.info(f"Comparing predictions to {len(field_gdf)} field plots")
    logger.info(f"Coverage threshold: {coverage_threshold:.0%}")
    
    results = []
    
    for idx, row in field_gdf.iterrows():
        plot_id = row.get('PlotID', row.get('plot_id', idx))
        site_name = row.get('Site', row.get('site_name', 'unknown'))
        footprint = row.geometry
        
        # Skip if site raster not available
        if site_name not in site_rasters:
            logger.warning(f"No raster found for site {site_name}, skipping plot {plot_id}")
            continue
        
        raster_path = site_rasters[site_name]
        
        # Base result dict
        result = {
            'plot_id': plot_id,
            'site_name': site_name,
            'plot_x': footprint.centroid.x,
            'plot_y': footprint.centroid.y,
        }
        
        # Extract predictions for each band
        for band in band_config.bands:
            band_name = band.name

            # Extract raster values
            extraction = extract_raster_values_at_footprint(
                raster_path, footprint, band.output_index, band.aggregation_method
            )
            
            # Raw model units
            pred_value = extraction['weighted_mean']
            result[f'{band_name}_pred_raw'] = pred_value
            result[f'{band_name}_n_pixels'] = extraction['n_pixels']
            result[f'{band_name}_coverage_fraction'] = extraction['coverage_fraction']
            
            # Convert to display units if not NaN
            if not np.isnan(pred_value):
                pred_display = band.convert_to_display_units(pred_value)
                result[f'{band_name}_pred'] = pred_display
            else:
                result[f'{band_name}_pred'] = np.nan
            
            # Field measurement (if exists)
            if band.field_column and band.field_column in row:
                result[f'{band_name}_field'] = row[band.field_column]
            else:
                result[f'{band_name}_field'] = np.nan
        
        results.append(result)
    
    df = pd.DataFrame(results)
    logger.info(f"Extracted predictions for {len(df)} plots")
    
    return df


def compute_statistics(
    df: pd.DataFrame,
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> Dict:
    """
    Compute comparison statistics for each band.
    
    Args:
        df: DataFrame with predictions and field measurements
        band_config: Band configuration
        coverage_threshold: Minimum coverage fraction
    
    Returns:
        Dict with statistics per band
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING COMPARISON STATISTICS")
    logger.info("=" * 60)
    
    stats_dict = {}
    
    for band in band_config.get_bands_with_field_mapping():
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        coverage_col = f'{band_name}_coverage_fraction'
        
        # Skip if no field data
        if field_col not in df.columns or pred_col not in df.columns:
            if band.field_column is not None:  # Only warn if field_column was specified
                logger.warning(f"Skipping {band_name}: missing field or prediction columns")
            continue
        
        # Filter valid data
        valid_mask = (
            df[field_col].notna() &
            df[pred_col].notna() &
            (df[coverage_col] >= coverage_threshold)
        )
        
        n_valid = valid_mask.sum()
        n_total = len(df)
        n_missing_field = df[field_col].isna().sum()
        n_missing_pred = df[pred_col].isna().sum()
        n_low_coverage = (df[coverage_col] < coverage_threshold).sum()
        
        logger.info(f"\n{band.display_name}:")
        logger.info(f"  Total plots: {n_total}")
        logger.info(f"  Missing field data: {n_missing_field}")
        logger.info(f"  Missing prediction: {n_missing_pred}")
        logger.info(f"  Coverage <{coverage_threshold:.0%}: {n_low_coverage}")
        logger.info(f"  Included in analysis: {n_valid}")
        
        if n_valid < 3:
            logger.warning(f"  Insufficient data (n={n_valid}<3) for statistics")
            continue
        
        # Extract valid values (convert field data from string to numeric)
        field_vals = pd.to_numeric(df.loc[valid_mask, field_col], errors='coerce').values
        pred_vals = df.loc[valid_mask, pred_col].values

        # Compute statistics
        r_val, p_pearson = scipy_stats.pearsonr(field_vals, pred_vals)
        rho_val, p_spearman = spearmanr(field_vals, pred_vals)
        rmse = np.sqrt(np.mean((pred_vals - field_vals) ** 2))
        mae = np.mean(np.abs(pred_vals - field_vals))
        bias = np.mean(pred_vals - field_vals)
        
        stats_dict[band_name] = {
            'n': int(n_valid),
            'pearson_r': float(r_val),
            'r_squared': float(r_val ** 2),
            'pearson_p_value': float(p_pearson),
            'spearman_rho': float(rho_val),
            'spearman_p_value': float(p_spearman),
            'rmse': float(rmse),
            'mae': float(mae),
            'bias': float(bias),
            'field_mean': float(field_vals.mean()),
            'field_std': float(field_vals.std()),
            'pred_mean': float(pred_vals.mean()),
            'pred_std': float(pred_vals.std()),
            'units': band.display_units,
            'display_name': band.display_name
        }
        
        logger.info(f"  R² = {stats_dict[band_name]['r_squared']:.3f}")
        logger.info(f"  RMSE = {stats_dict[band_name]['rmse']:.2f} {band.display_units}")
        logger.info(f"  Bias = {stats_dict[band_name]['bias']:.2f} {band.display_units}")
    
    logger.info("=" * 60)
    
    return stats_dict


def compute_per_site_summary(
    df: pd.DataFrame,
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> pd.DataFrame:
    """
    Compute per-site summary statistics for each band.

    Args:
        df: DataFrame with per-plot predictions and field measurements
        band_config: Band configuration
        coverage_threshold: Minimum coverage fraction for inclusion

    Returns:
        DataFrame with per-site summary statistics
    """
    from scipy.stats import spearmanr

    sites = df['site_name'].unique()
    summary_rows = []

    for site in sites:
        site_df = df[df['site_name'] == site].copy()
        n_total = len(site_df)

        row = {'site_name': site, 'n_total_plots': n_total}

        # Compute statistics for each band
        for band in band_config.get_bands_with_field_mapping():
            band_name = band.name
            field_col = f'{band_name}_field'
            pred_col = f'{band_name}_pred'
            coverage_col = f'{band_name}_coverage_fraction'

            if field_col not in df.columns or pred_col not in df.columns:
                continue

            # Count plots by category
            valid_mask = (
                site_df[field_col].notna() &
                site_df[pred_col].notna() &
                (site_df[coverage_col] >= coverage_threshold)
            )

            n_included = valid_mask.sum()
            n_missing_field = site_df[field_col].isna().sum()
            n_missing_pred = site_df[pred_col].isna().sum()
            n_low_coverage = ((site_df[coverage_col] < coverage_threshold) &
                            (site_df[coverage_col] > 0)).sum()

            row[f'{band_name}_n_included'] = n_included
            row[f'{band_name}_n_missing_field'] = n_missing_field
            row[f'{band_name}_n_missing_pred'] = n_missing_pred
            row[f'{band_name}_n_low_coverage'] = n_low_coverage

            # Compute statistics if sufficient data
            if n_included >= 3:
                field_vals = pd.to_numeric(site_df.loc[valid_mask, field_col], errors='coerce').values
                pred_vals = site_df.loc[valid_mask, pred_col].values

                r_val, p_pearson = scipy_stats.pearsonr(field_vals, pred_vals)
                rho_val, p_spearman = spearmanr(field_vals, pred_vals)

                row[f'{band_name}_r_squared'] = float(r_val ** 2)
                row[f'{band_name}_pearson_r'] = float(r_val)
                row[f'{band_name}_pearson_p'] = float(p_pearson)
                row[f'{band_name}_spearman_rho'] = float(rho_val)
                row[f'{band_name}_spearman_p'] = float(p_spearman)
                row[f'{band_name}_rmse'] = float(np.sqrt(np.mean((pred_vals - field_vals) ** 2)))
                row[f'{band_name}_mae'] = float(np.mean(np.abs(pred_vals - field_vals)))
                row[f'{band_name}_bias'] = float(np.mean(pred_vals - field_vals))
                row[f'{band_name}_field_mean'] = float(np.mean(field_vals))
                row[f'{band_name}_field_std'] = float(np.std(field_vals, ddof=1))
                row[f'{band_name}_pred_mean'] = float(np.mean(pred_vals))
                row[f'{band_name}_pred_std'] = float(np.std(pred_vals, ddof=1))
            else:
                # Insufficient data
                for key in ['r_squared', 'pearson_r', 'pearson_p', 'spearman_rho', 'spearman_p',
                           'rmse', 'mae', 'bias', 'field_mean', 'field_std', 'pred_mean', 'pred_std']:
                    row[f'{band_name}_{key}'] = np.nan

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def plot_by_site(
    df: pd.DataFrame,
    band_config: BandConfig,
    output_path: Path,
    coverage_threshold: float = 0.99
) -> None:
    """
    Generate per-site comparison scatter plots.

    Args:
        df: DataFrame with predictions and field measurements
        band_config: Band configuration
        output_path: Path to save figure
        coverage_threshold: Minimum coverage fraction
    """
    from scipy.stats import spearmanr

    sites = sorted(df['site_name'].unique())
    n_sites = len(sites)

    if n_sites == 0:
        logger.warning("No sites to plot")
        return

    bands_with_mapping = band_config.get_bands_with_field_mapping()
    n_bands = len(bands_with_mapping)

    if n_bands == 0:
        logger.warning("No bands with field mapping for plotting")
        return

    # Create figure
    fig, axes = plt.subplots(n_bands, n_sites, figsize=(4 * n_sites, 4 * n_bands))
    if n_bands == 1 and n_sites == 1:
        axes = np.array([[axes]])
    elif n_bands == 1:
        axes = axes.reshape(1, -1)
    elif n_sites == 1:
        axes = axes.reshape(-1, 1)

    for band_idx, band in enumerate(bands_with_mapping):
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        coverage_col = f'{band_name}_coverage_fraction'

        for site_idx, site in enumerate(sites):
            ax = axes[band_idx, site_idx]
            site_df = df[df['site_name'] == site]

            # Filter valid data
            valid_mask = (
                site_df[field_col].notna() &
                site_df[pred_col].notna() &
                (site_df[coverage_col] >= coverage_threshold)
            )

            if valid_mask.sum() >= 2:
                field_vals = pd.to_numeric(site_df.loc[valid_mask, field_col], errors='coerce')
                pred_vals = site_df.loc[valid_mask, pred_col]

                ax.scatter(field_vals, pred_vals, alpha=0.6, s=50)

                # 1:1 line
                val_min = min(field_vals.min(), pred_vals.min())
                val_max = max(field_vals.max(), pred_vals.max())
                ax.plot([val_min, val_max], [val_min, val_max], 'k--', alpha=0.5)

                # Statistics if n >= 3
                if len(field_vals) >= 3:
                    r_val = scipy_stats.pearsonr(field_vals, pred_vals)[0]
                    rho_val = spearmanr(field_vals, pred_vals)[0]
                    ax.set_title(f'{site}\nR²={r_val**2:.3f}, ρ={rho_val:.3f}\n(n={len(field_vals)})',
                               fontsize=9)
                else:
                    ax.set_title(f'{site}\n(n={len(field_vals)}, insufficient)', fontsize=9)
            else:
                ax.set_title(f'{site}\n(no data)', fontsize=9)

            ax.set_xlabel(f'Field {band.display_name} ({band.display_units})', fontsize=8)
            ax.set_ylabel(f'Pred. {band.display_name} ({band.display_units})', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-site comparison to {output_path}")


def plot_by_site_with_uncertainty(
    df: pd.DataFrame,
    band_config: BandConfig,
    output_path: Path,
    coverage_threshold: float = 0.99
) -> None:
    """
    Generate per-site comparison scatter plots with MC dropout uncertainty error bars.

    Same layout as plot_by_site(), but uses ax.errorbar() with uncertainty
    from band_X_mc_std_mean columns.

    Args:
        df: DataFrame with predictions, field measurements, and MC uncertainty columns
        band_config: Band configuration
        output_path: Path to save figure
        coverage_threshold: Minimum coverage fraction
    """
    from scipy.stats import spearmanr

    sites = sorted(df['site_name'].unique())
    n_sites = len(sites)

    if n_sites == 0:
        logger.warning("No sites to plot")
        return

    bands_with_mapping = band_config.get_bands_with_field_mapping()
    n_bands = len(bands_with_mapping)

    if n_bands == 0:
        logger.warning("No bands with field mapping for plotting")
        return

    # Check if uncertainty columns exist
    first_band = bands_with_mapping[0].name
    if f'{first_band}_mc_std_mean' not in df.columns:
        logger.warning("No MC uncertainty columns found, skipping uncertainty plot")
        return

    # Create figure
    fig, axes = plt.subplots(n_bands, n_sites, figsize=(4 * n_sites, 4 * n_bands))
    if n_bands == 1 and n_sites == 1:
        axes = np.array([[axes]])
    elif n_bands == 1:
        axes = axes.reshape(1, -1)
    elif n_sites == 1:
        axes = axes.reshape(-1, 1)

    for band_idx, band in enumerate(bands_with_mapping):
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        coverage_col = f'{band_name}_coverage_fraction'
        mc_std_col = f'{band_name}_mc_std_mean'

        for site_idx, site in enumerate(sites):
            ax = axes[band_idx, site_idx]
            site_df = df[df['site_name'] == site]

            # Filter valid data
            valid_mask = (
                site_df[field_col].notna() &
                site_df[pred_col].notna() &
                (site_df[coverage_col] >= coverage_threshold)
            )

            if valid_mask.sum() >= 2:
                field_vals = pd.to_numeric(site_df.loc[valid_mask, field_col], errors='coerce')
                pred_vals = site_df.loc[valid_mask, pred_col]
                mc_std_vals = site_df.loc[valid_mask, mc_std_col]

                # Error bars using MC uncertainty
                ax.errorbar(
                    field_vals, pred_vals,
                    yerr=mc_std_vals,
                    fmt='o', alpha=0.6, markersize=6,
                    capsize=2, capthick=1, elinewidth=1,
                    color='forestgreen', ecolor='gray'
                )

                # 1:1 line
                val_min = min(field_vals.min(), pred_vals.min())
                val_max = max(field_vals.max(), pred_vals.max())
                ax.plot([val_min, val_max], [val_min, val_max], 'k--', alpha=0.5)

                # Statistics if n >= 3
                if len(field_vals) >= 3:
                    r_val = scipy_stats.pearsonr(field_vals, pred_vals)[0]
                    rho_val = spearmanr(field_vals, pred_vals)[0]
                    mean_unc = mc_std_vals.mean()
                    ax.set_title(
                        f'{site}\nR²={r_val**2:.3f}, ρ={rho_val:.3f}\n'
                        f'(n={len(field_vals)}, unc={mean_unc:.2f})',
                        fontsize=9
                    )
                else:
                    ax.set_title(f'{site}\n(n={len(field_vals)}, insufficient)', fontsize=9)
            else:
                ax.set_title(f'{site}\n(no data)', fontsize=9)

            ax.set_xlabel(f'Field {band.display_name} ({band.display_units})', fontsize=8)
            ax.set_ylabel(f'Pred. {band.display_name} ({band.display_units})', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-site comparison with uncertainty to {output_path}")


def plot_uncertainty_distribution(
    df: pd.DataFrame,
    band_config: BandConfig,
    output_path: Path
) -> None:
    """
    Generate histograms of MC dropout uncertainty per band.

    Shows the distribution of prediction uncertainty across all tiles,
    optionally split by site to identify high-uncertainty regions.

    Args:
        df: DataFrame with MC uncertainty columns
        band_config: Band configuration
        output_path: Path to save figure
    """
    bands_with_mapping = band_config.get_bands_with_field_mapping()
    n_bands = len(bands_with_mapping)

    if n_bands == 0:
        logger.warning("No bands with field mapping for plotting")
        return

    # Check if uncertainty columns exist
    first_band = bands_with_mapping[0].name
    if f'{first_band}_mc_std_mean' not in df.columns:
        logger.warning("No MC uncertainty columns found, skipping uncertainty distribution plot")
        return

    sites = sorted(df['site_name'].unique())
    n_sites = len(sites)

    # Create figure: one row per band, columns for overall + per-site
    fig, axes = plt.subplots(n_bands, 1 + n_sites, figsize=(4 * (1 + n_sites), 4 * n_bands))
    if n_bands == 1:
        axes = axes.reshape(1, -1)

    colors = plt.cm.tab10(np.linspace(0, 1, n_sites))

    for band_idx, band in enumerate(bands_with_mapping):
        band_name = band.name
        mc_std_col = f'{band_name}_mc_std_mean'
        mc_cv_col = f'{band_name}_mc_cv'

        # Column 0: Overall distribution
        ax = axes[band_idx, 0]
        all_unc = df[mc_std_col].dropna()
        if len(all_unc) > 0:
            ax.hist(all_unc, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
            ax.axvline(all_unc.mean(), color='red', linestyle='--', label=f'Mean: {all_unc.mean():.3f}')
            ax.axvline(all_unc.median(), color='orange', linestyle=':', label=f'Median: {all_unc.median():.3f}')
            ax.set_title(f'{band.display_name}\nAll Sites (n={len(all_unc)})', fontsize=10)
            ax.set_xlabel(f'MC Std ({band.display_units})', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        # Columns 1+: Per-site distributions
        for site_idx, site in enumerate(sites):
            ax = axes[band_idx, 1 + site_idx]
            site_df = df[df['site_name'] == site]
            site_unc = site_df[mc_std_col].dropna()

            if len(site_unc) > 0:
                ax.hist(site_unc, bins=20, alpha=0.7, color=colors[site_idx], edgecolor='black')
                ax.axvline(site_unc.mean(), color='red', linestyle='--')
                ax.set_title(f'{site}\n(n={len(site_unc)}, mean={site_unc.mean():.3f})', fontsize=9)
            else:
                ax.set_title(f'{site}\n(no data)', fontsize=9)

            ax.set_xlabel(f'MC Std ({band.display_units})', fontsize=8)
            if site_idx == 0:
                ax.set_ylabel('Count', fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved uncertainty distribution plot to {output_path}")


def plot_rank_scatter(
    df: pd.DataFrame,
    band_config: BandConfig,
    stats_dict: Dict,
    output_path: Path,
    coverage_threshold: float = 0.99
) -> None:
    """
    Generate rank-rank scatter plots (percentile-based).

    Args:
        df: DataFrame with predictions and field measurements
        band_config: Band configuration
        stats_dict: Statistics dict from compute_statistics()
        output_path: Path to save figure
        coverage_threshold: Coverage threshold
    """
    bands_with_stats = [b for b in band_config.get_bands_with_field_mapping() if b.name in stats_dict]
    n_bands = len(bands_with_stats)

    if n_bands == 0:
        logger.warning("No bands with statistics for rank plotting")
        return

    fig, axes = plt.subplots(1, n_bands, figsize=(6 * n_bands, 5))
    if n_bands == 1:
        axes = [axes]

    for idx, band in enumerate(bands_with_stats):
        ax = axes[idx]
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        coverage_col = f'{band_name}_coverage_fraction'

        # Filter valid data
        valid_mask = (
            df[field_col].notna() &
            df[pred_col].notna() &
            (df[coverage_col] >= coverage_threshold)
        )

        if valid_mask.sum() < 3:
            continue

        field_vals = pd.to_numeric(df.loc[valid_mask, field_col], errors='coerce')
        pred_vals = df.loc[valid_mask, pred_col]

        # Compute ranks (percentiles)
        field_ranks = scipy_stats.rankdata(field_vals, method='average') / len(field_vals) * 100
        pred_ranks = scipy_stats.rankdata(pred_vals, method='average') / len(pred_vals) * 100

        ax.scatter(field_ranks, pred_ranks, alpha=0.6, s=50)
        ax.plot([0, 100], [0, 100], 'k--', alpha=0.5)

        # Statistics
        band_stats = stats_dict[band_name]
        rho = band_stats['spearman_rho']
        ax.set_title(f'{band.display_name}\nSpearman ρ={rho:.3f}, n={band_stats["n"]}')
        ax.set_xlabel('Field Percentile')
        ax.set_ylabel('Prediction Percentile')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved rank scatter plot to {output_path}")


def plot_rank_by_site(
    df: pd.DataFrame,
    band_config: BandConfig,
    output_path: Path,
    coverage_threshold: float = 0.99
) -> None:
    """
    Generate per-site rank-rank scatter plots.

    Args:
        df: DataFrame with predictions and field measurements
        band_config: Band configuration
        output_path: Path to save figure
        coverage_threshold: Minimum coverage fraction
    """
    from scipy.stats import spearmanr

    sites = sorted(df['site_name'].unique())
    n_sites = len(sites)

    if n_sites == 0:
        logger.warning("No sites to plot")
        return

    bands_with_mapping = band_config.get_bands_with_field_mapping()
    n_bands = len(bands_with_mapping)

    if n_bands == 0:
        logger.warning("No bands with field mapping for plotting")
        return

    fig, axes = plt.subplots(n_bands, n_sites, figsize=(4 * n_sites, 4 * n_bands))
    if n_bands == 1 and n_sites == 1:
        axes = np.array([[axes]])
    elif n_bands == 1:
        axes = axes.reshape(1, -1)
    elif n_sites == 1:
        axes = axes.reshape(-1, 1)

    for band_idx, band in enumerate(bands_with_mapping):
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        coverage_col = f'{band_name}_coverage_fraction'

        for site_idx, site in enumerate(sites):
            ax = axes[band_idx, site_idx]
            site_df = df[df['site_name'] == site]

            # Filter valid data
            valid_mask = (
                site_df[field_col].notna() &
                site_df[pred_col].notna() &
                (site_df[coverage_col] >= coverage_threshold)
            )

            if valid_mask.sum() >= 3:
                field_vals = pd.to_numeric(site_df.loc[valid_mask, field_col], errors='coerce').values
                pred_vals = site_df.loc[valid_mask, pred_col].values

                # Compute ranks
                field_ranks = scipy_stats.rankdata(field_vals, method='average') / len(field_vals) * 100
                pred_ranks = scipy_stats.rankdata(pred_vals, method='average') / len(pred_vals) * 100

                ax.scatter(field_ranks, pred_ranks, alpha=0.6, s=50)
                ax.plot([0, 100], [0, 100], 'k--', alpha=0.5)

                rho_val = spearmanr(field_vals, pred_vals)[0]
                ax.set_title(f'{site}\nρ={rho_val:.3f}, n={len(field_vals)}', fontsize=9)
            else:
                ax.set_title(f'{site}\n(insufficient data)', fontsize=9)

            ax.set_xlabel('Field Percentile', fontsize=8)
            ax.set_ylabel('Pred. Percentile', fontsize=8)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved per-site rank plot to {output_path}")


def load_uncertainty_rasters(rasters_dir: Path) -> Dict[str, str]:
    """
    Load paths to uncertainty (std) raster files.

    Returns:
        Dict mapping site_name to uncertainty raster file path
    """
    logger.info(f"Loading uncertainty rasters from {rasters_dir}")

    raster_files = list(rasters_dir.glob('*_predictions_std_raster.tif'))

    if not raster_files:
        return {}  # No uncertainty rasters is not an error

    unc_rasters = {}
    for raster_path in raster_files:
        # Extract site name from filename (e.g., "BluffMesa_predictions_std_raster.tif" -> "BluffMesa")
        site_name = raster_path.stem.replace('_predictions_std_raster', '')
        unc_rasters[site_name] = str(raster_path)

    logger.info(f"Found {len(unc_rasters)} uncertainty rasters")
    for site_name in sorted(unc_rasters.keys()):
        logger.info(f"  - {site_name}")

    return unc_rasters


def extract_uncertainty_values(
    field_gdf: gpd.GeoDataFrame,
    unc_rasters: Dict[str, str],
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> pd.DataFrame:
    """
    Extract uncertainty raster values at plot footprints.

    Returns DataFrame with columns: plot_id, site_name, {band}_mc_std_mean
    """
    logger.info(f"Extracting uncertainty values for {len(field_gdf)} plots")

    results = []
    for idx, row in field_gdf.iterrows():
        plot_id = row.get('PlotID', row.get('plot_id', idx))
        site_name = row.get('Site', row.get('site_name', 'unknown'))
        footprint = row.geometry

        if site_name not in unc_rasters:
            continue

        raster_path = unc_rasters[site_name]
        result = {'plot_id': plot_id, 'site_name': site_name}

        for band in band_config.bands:
            band_name = band.name
            try:
                extraction = extract_raster_values_at_footprint(
                    raster_path, footprint, band.output_index, 'mean'  # Always use mean for std
                )
                unc_value = extraction['weighted_mean']
                if not np.isnan(unc_value):
                    # Apply same unit conversion as for predictions
                    unc_display = band.convert_to_display_units(unc_value)
                    result[f'{band_name}_mc_std_mean'] = unc_display
                else:
                    result[f'{band_name}_mc_std_mean'] = np.nan
            except Exception:
                result[f'{band_name}_mc_std_mean'] = np.nan

        results.append(result)

    return pd.DataFrame(results)


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
        site_name = raster_path.parent.name
        site_rasters[site_name] = str(raster_path)

    logger.info(f"Found {len(site_rasters)} baseline rasters")
    for site_name in sorted(site_rasters.keys()):
        logger.info(f"  - {site_name}")

    return site_rasters


def extract_baseline_values(
    field_gdf: gpd.GeoDataFrame,
    baseline_rasters: Dict[str, str],
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> pd.DataFrame:
    """
    Extract baseline raster values at plot footprints.

    Returns DataFrame with columns: plot_id, site_name, {band}_baseline, {band}_baseline_coverage
    """
    logger.info(f"Extracting baseline values for {len(field_gdf)} plots")

    results = []
    for idx, row in field_gdf.iterrows():
        plot_id = row.get('PlotID', row.get('plot_id', idx))
        site_name = row.get('Site', row.get('site_name', 'unknown'))
        footprint = row.geometry

        if site_name not in baseline_rasters:
            continue

        raster_path = baseline_rasters[site_name]
        result = {'plot_id': plot_id, 'site_name': site_name}

        for band in band_config.bands:
            band_name = band.name
            try:
                extraction = extract_raster_values_at_footprint(
                    raster_path, footprint, band.output_index, band.aggregation_method
                )
                baseline_value = extraction['weighted_mean']
                if not np.isnan(baseline_value):
                    baseline_display = band.convert_to_display_units(baseline_value)
                    result[f'{band_name}_baseline'] = baseline_display
                else:
                    result[f'{band_name}_baseline'] = np.nan
                result[f'{band_name}_baseline_coverage'] = extraction['coverage_fraction']
            except Exception:
                result[f'{band_name}_baseline'] = np.nan
                result[f'{band_name}_baseline_coverage'] = 0.0

        results.append(result)

    return pd.DataFrame(results)


def compute_3way_statistics(
    df: pd.DataFrame,
    band_config: BandConfig,
    coverage_threshold: float = 0.99
) -> Dict:
    """
    Compute 3-way comparison statistics: model vs field, baseline vs field, model vs baseline.

    Returns dict with {band}_model_vs_field, {band}_baseline_vs_field, {band}_model_vs_baseline
    """
    logger.info("\n" + "=" * 60)
    logger.info("COMPUTING 3-WAY COMPARISON STATISTICS")
    logger.info("=" * 60)

    stats_dict = {}

    for band in band_config.get_bands_with_field_mapping():
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        baseline_col = f'{band_name}_baseline'
        coverage_col = f'{band_name}_coverage_fraction'
        baseline_coverage_col = f'{band_name}_baseline_coverage'

        # Skip if columns missing
        required_cols = [field_col, pred_col, baseline_col, coverage_col, baseline_coverage_col]
        if not all(col in df.columns for col in required_cols):
            logger.warning(f"Skipping {band_name}: missing required columns")
            continue

        # Filter valid data for all three comparisons
        valid_all = (
            df[field_col].notna() &
            df[pred_col].notna() &
            df[baseline_col].notna() &
            (df[coverage_col] >= coverage_threshold) &
            (df[baseline_coverage_col] >= coverage_threshold)
        )

        n_valid = valid_all.sum()
        logger.info(f"\n{band.display_name}: {n_valid} plots with all three values")

        if n_valid < 3:
            continue

        field_vals = pd.to_numeric(df.loc[valid_all, field_col], errors='coerce').values
        pred_vals = df.loc[valid_all, pred_col].values
        baseline_vals = df.loc[valid_all, baseline_col].values

        # Model vs Field
        r_mf, p_mf = scipy_stats.pearsonr(field_vals, pred_vals)
        rho_mf, _ = spearmanr(field_vals, pred_vals)
        rmse_mf = np.sqrt(np.mean((pred_vals - field_vals) ** 2))
        mae_mf = np.mean(np.abs(pred_vals - field_vals))
        bias_mf = np.mean(pred_vals - field_vals)

        # Baseline vs Field
        r_bf, p_bf = scipy_stats.pearsonr(field_vals, baseline_vals)
        rho_bf, _ = spearmanr(field_vals, baseline_vals)
        rmse_bf = np.sqrt(np.mean((baseline_vals - field_vals) ** 2))
        mae_bf = np.mean(np.abs(baseline_vals - field_vals))
        bias_bf = np.mean(baseline_vals - field_vals)

        # Model vs Baseline
        r_mb, p_mb = scipy_stats.pearsonr(baseline_vals, pred_vals)
        rho_mb, _ = spearmanr(baseline_vals, pred_vals)
        rmse_mb = np.sqrt(np.mean((pred_vals - baseline_vals) ** 2))

        stats_dict[band_name] = {
            'n': int(n_valid),
            'units': band.display_units,
            'display_name': band.display_name,
            'model_vs_field': {
                'r_squared': float(r_mf ** 2),
                'pearson_r': float(r_mf),
                'spearman_rho': float(rho_mf),
                'rmse': float(rmse_mf),
                'mae': float(mae_mf),
                'bias': float(bias_mf),
            },
            'baseline_vs_field': {
                'r_squared': float(r_bf ** 2),
                'pearson_r': float(r_bf),
                'spearman_rho': float(rho_bf),
                'rmse': float(rmse_bf),
                'mae': float(mae_bf),
                'bias': float(bias_bf),
            },
            'model_vs_baseline': {
                'r_squared': float(r_mb ** 2),
                'pearson_r': float(r_mb),
                'spearman_rho': float(rho_mb),
                'rmse': float(rmse_mb),
            },
            'improvement': {
                'rmse_reduction': float(rmse_bf - rmse_mf),
                'r2_improvement': float(r_mf ** 2 - r_bf ** 2),
            }
        }

        logger.info(f"  Model vs Field:    R²={r_mf**2:.3f}, RMSE={rmse_mf:.2f}")
        logger.info(f"  Baseline vs Field: R²={r_bf**2:.3f}, RMSE={rmse_bf:.2f}")
        logger.info(f"  Improvement:       ΔR²={r_mf**2 - r_bf**2:+.3f}, ΔRMSE={rmse_bf - rmse_mf:+.2f}")

    return stats_dict


def plot_3way_comparison(
    df: pd.DataFrame,
    band_config: BandConfig,
    stats_dict: Dict,
    output_path: Path,
    coverage_threshold: float = 0.99
) -> None:
    """Generate 3-column scatter plot: Baseline vs Field, Model vs Field, Model vs Baseline."""
    bands_with_stats = [b for b in band_config.get_bands_with_field_mapping() if b.name in stats_dict]
    n_bands = len(bands_with_stats)

    if n_bands == 0:
        logger.warning("No bands with 3-way statistics for plotting")
        return

    fig, axes = plt.subplots(n_bands, 3, figsize=(15, 4 * n_bands))
    if n_bands == 1:
        axes = axes.reshape(1, -1)

    for band_idx, band in enumerate(bands_with_stats):
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        baseline_col = f'{band_name}_baseline'
        coverage_col = f'{band_name}_coverage_fraction'
        baseline_coverage_col = f'{band_name}_baseline_coverage'

        valid_mask = (
            df[field_col].notna() &
            df[pred_col].notna() &
            df[baseline_col].notna() &
            (df[coverage_col] >= coverage_threshold) &
            (df[baseline_coverage_col] >= coverage_threshold)
        )

        if valid_mask.sum() < 3:
            continue

        field_vals = pd.to_numeric(df.loc[valid_mask, field_col], errors='coerce')
        pred_vals = df.loc[valid_mask, pred_col]
        baseline_vals = df.loc[valid_mask, baseline_col]

        stats = stats_dict[band_name]

        # Column 1: Baseline vs Field
        ax = axes[band_idx, 0]
        ax.scatter(field_vals, baseline_vals, alpha=0.6, s=50, c='orange', label='Baseline')
        max_val = max(field_vals.max(), baseline_vals.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
        ax.set_xlabel(f'Field ({band.display_units})')
        ax.set_ylabel(f'Baseline ({band.display_units})')
        ax.set_title(f'{band.display_name}\nBaseline vs Field\nR²={stats["baseline_vs_field"]["r_squared"]:.3f}')
        ax.grid(True, alpha=0.3)

        # Column 2: Model vs Field
        ax = axes[band_idx, 1]
        ax.scatter(field_vals, pred_vals, alpha=0.6, s=50, c='forestgreen', label='Model')
        max_val = max(field_vals.max(), pred_vals.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
        ax.set_xlabel(f'Field ({band.display_units})')
        ax.set_ylabel(f'Model ({band.display_units})')
        ax.set_title(f'{band.display_name}\nModel vs Field\nR²={stats["model_vs_field"]["r_squared"]:.3f}')
        ax.grid(True, alpha=0.3)

        # Column 3: Model vs Baseline
        ax = axes[band_idx, 2]
        ax.scatter(baseline_vals, pred_vals, alpha=0.6, s=50, c='steelblue', label='Comparison')
        max_val = max(baseline_vals.max(), pred_vals.max())
        ax.plot([0, max_val], [0, max_val], 'k--', linewidth=1.5)
        ax.set_xlabel(f'Baseline ({band.display_units})')
        ax.set_ylabel(f'Model ({band.display_units})')
        ax.set_title(f'{band.display_name}\nModel vs Baseline\nR²={stats["model_vs_baseline"]["r_squared"]:.3f}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved 3-way comparison plot to {output_path}")


def plot_comparisons(
    df: pd.DataFrame,
    band_config: BandConfig,
    stats_dict: Dict,
    output_dir: Path,
    coverage_threshold: float = 0.99
) -> None:
    """
    Generate comparison scatter plots.
    
    Args:
        df: DataFrame with predictions and field measurements
        band_config: Band configuration
        stats_dict: Statistics dict from compute_statistics()
        output_dir: Output directory
        coverage_threshold: Coverage threshold used for filtering
    """
    logger.info("Generating comparison figures")
    
    bands_with_stats = [b for b in band_config.get_bands_with_field_mapping() if b.name in stats_dict]
    n_bands = len(bands_with_stats)
    
    if n_bands == 0:
        logger.warning("No bands with valid statistics for plotting")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_bands, figsize=(6 * n_bands, 5))
    if n_bands == 1:
        axes = [axes]
    
    for idx, band in enumerate(bands_with_stats):
        ax = axes[idx]
        band_name = band.name
        field_col = f'{band_name}_field'
        pred_col = f'{band_name}_pred'
        coverage_col = f'{band_name}_coverage_fraction'
        
        # Filter valid data
        valid_mask = (
            df[field_col].notna() &
            df[pred_col].notna() &
            (df[coverage_col] >= coverage_threshold)
        )
        
        if valid_mask.sum() < 3:
            continue

        field_vals = pd.to_numeric(df.loc[valid_mask, field_col], errors='coerce')
        pred_vals = df.loc[valid_mask, pred_col]

        # Scatter plot
        ax.scatter(field_vals, pred_vals, alpha=0.6, s=50, c='forestgreen')
        
        # 1:1 line
        max_val = max(field_vals.max(), pred_vals.max())
        ax.plot([0, max_val], [0, max_val], 'k--', label='1:1 line', linewidth=1.5)
        
        # Stats text
        stats = stats_dict[band_name]
        stats_txt = f"R² = {stats['r_squared']:.3f}\n"
        stats_txt += f"RMSE = {stats['rmse']:.2f} {stats['units']}\n"
        stats_txt += f"Bias = {stats['bias']:.2f} {stats['units']}\n"
        stats_txt += f"n = {stats['n']}"
        ax.text(0.05, 0.95, stats_txt, transform=ax.transAxes,
               verticalalignment='top', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(f'Field {band.display_name} ({stats["units"]})', fontsize=12)
        ax.set_ylabel(f'Predicted {band.display_name} ({stats["units"]})', fontsize=12)
        ax.set_title(f'{band.display_name} Comparison', fontsize=14)
        ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save
    output_path = output_dir / 'comparison_scatter.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison scatter plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model raster predictions to forest plot field measurements"
    )
    parser.add_argument(
        '--site-rasters-dir',
        type=str,
        required=True,
        help='Directory containing per-site prediction raster GeoTIFFs'
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
    parser.add_argument(
        '--baseline-rasters-dir',
        type=str,
        default=None,
        help='Optional: Directory containing baseline raster GeoTIFFs for 3-way comparison (pattern: {site}/veg_structure_2m.tif)'
    )

    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load band configuration
    band_config = load_band_config(args.band_config)
    logger.info(f"Loaded band config: {band_config.name}")
    logger.info(f"Bands: {', '.join([b.display_name for b in band_config.bands])}")
    
    # Load site rasters
    rasters_dir = Path(args.site_rasters_dir)
    site_rasters = load_site_rasters(rasters_dir)

    # Check for uncertainty rasters (from MC dropout)
    unc_rasters = load_uncertainty_rasters(rasters_dir)
    has_uncertainty = len(unc_rasters) > 0
    if has_uncertainty:
        logger.info(f"Found {len(unc_rasters)} uncertainty rasters - will extract for visualization")

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
    
    # Compare predictions to field
    comparison_df = compare_predictions_to_field(
        field_footprints,
        site_rasters,
        band_config,
        coverage_threshold=args.coverage_threshold
    )

    # Merge uncertainty values if available (from MC dropout)
    if has_uncertainty:
        logger.info("Extracting uncertainty values from std rasters")
        unc_df = extract_uncertainty_values(
            field_footprints,
            unc_rasters,
            band_config,
            coverage_threshold=args.coverage_threshold
        )
        if len(unc_df) > 0:
            comparison_df = comparison_df.merge(
                unc_df,
                on=['plot_id', 'site_name'],
                how='left'
            )
            logger.info(f"Merged uncertainty values for {len(unc_df)} plots")

    # Save comparison results
    results_path = output_dir / 'comparison_results.csv'
    comparison_df.to_csv(results_path, index=False)
    logger.info(f"Saved comparison results to {results_path}")
    
    # Compute statistics
    stats_dict = compute_statistics(
        comparison_df,
        band_config,
        coverage_threshold=args.coverage_threshold
    )
    
    # Save statistics
    stats_path = output_dir / 'comparison_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    # Compute and save per-site summary
    site_summary_df = compute_per_site_summary(
        comparison_df,
        band_config,
        coverage_threshold=args.coverage_threshold
    )
    site_summary_path = output_dir / 'comparison_per_site_summary.csv'
    site_summary_df.to_csv(site_summary_path, index=False)
    logger.info(f"Saved per-site summary to {site_summary_path}")

    # Generate plots
    logger.info("Generating comparison figures")
    plot_comparisons(
        comparison_df,
        band_config,
        stats_dict,
        output_dir,
        coverage_threshold=args.coverage_threshold
    )

    plot_by_site(
        comparison_df,
        band_config,
        output_dir / 'comparison_by_site.png',
        coverage_threshold=args.coverage_threshold
    )

    plot_rank_scatter(
        comparison_df,
        band_config,
        stats_dict,
        output_dir / 'comparison_rank_scatter.png',
        coverage_threshold=args.coverage_threshold
    )

    plot_rank_by_site(
        comparison_df,
        band_config,
        output_dir / 'comparison_rank_by_site.png',
        coverage_threshold=args.coverage_threshold
    )

    # MC dropout uncertainty visualizations (if uncertainty columns exist)
    # Check for mc_samples column to determine if MC dropout was used
    if 'mc_samples' in comparison_df.columns or any(
        col.endswith('_mc_std_mean') for col in comparison_df.columns
    ):
        logger.info("Generating MC dropout uncertainty figures")

        plot_by_site_with_uncertainty(
            comparison_df,
            band_config,
            output_dir / 'comparison_by_site_uncertainty.png',
            coverage_threshold=args.coverage_threshold
        )

        plot_uncertainty_distribution(
            comparison_df,
            band_config,
            output_dir / 'uncertainty_distribution.png'
        )

    # 3-way comparison (if baseline provided)
    if args.baseline_rasters_dir:
        logger.info("\n" + "=" * 60)
        logger.info("3-WAY COMPARISON MODE")
        logger.info("=" * 60)

        baseline_rasters_dir = Path(args.baseline_rasters_dir)
        baseline_rasters = load_baseline_rasters(baseline_rasters_dir)

        # Extract baseline values
        baseline_df = extract_baseline_values(
            field_footprints,
            baseline_rasters,
            band_config,
            coverage_threshold=args.coverage_threshold
        )

        # Merge baseline with predictions
        comparison_df = comparison_df.merge(
            baseline_df,
            on=['plot_id', 'site_name'],
            how='left'
        )

        # Save updated comparison results
        results_3way_path = output_dir / '3way_comparison_results.csv'
        comparison_df.to_csv(results_3way_path, index=False)
        logger.info(f"Saved 3-way comparison results to {results_3way_path}")

        # Compute 3-way statistics
        stats_3way = compute_3way_statistics(
            comparison_df,
            band_config,
            coverage_threshold=args.coverage_threshold
        )

        # Save 3-way statistics
        stats_3way_path = output_dir / '3way_comparison_stats.json'
        with open(stats_3way_path, 'w') as f:
            json.dump(stats_3way, f, indent=2)
        logger.info(f"Saved 3-way statistics to {stats_3way_path}")

        # Generate 3-way comparison plot
        plot_3way_comparison(
            comparison_df,
            band_config,
            stats_3way,
            output_dir / '3way_comparison_scatter.png',
            coverage_threshold=args.coverage_threshold
        )

        # Generate improvement summary CSV
        improvement_rows = []
        for band_name, stats in stats_3way.items():
            improvement_rows.append({
                'band': band_name,
                'display_name': stats['display_name'],
                'units': stats['units'],
                'n': stats['n'],
                'baseline_r2': stats['baseline_vs_field']['r_squared'],
                'model_r2': stats['model_vs_field']['r_squared'],
                'baseline_rmse': stats['baseline_vs_field']['rmse'],
                'model_rmse': stats['model_vs_field']['rmse'],
                'r2_improvement': stats['improvement']['r2_improvement'],
                'rmse_reduction': stats['improvement']['rmse_reduction'],
            })
        improvement_df = pd.DataFrame(improvement_rows)
        improvement_path = output_dir / '3way_improvement_summary.csv'
        improvement_df.to_csv(improvement_path, index=False)
        logger.info(f"Saved improvement summary to {improvement_path}")

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")


if __name__ == '__main__':
    main()
