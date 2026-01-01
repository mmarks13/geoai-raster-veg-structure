#!/usr/bin/env python3
"""
Analyze forest plot prediction accuracy by site and year.

This script generates:
1. Per-site statistics (R², RMSE, bias)
2. Site × Year interaction analysis
3. Distribution comparisons
4. Visualization figures

Usage:
    python src/evaluation/analyze_predictions_by_site.py \
        --comparison-results <path_to_comparison_results.csv> \
        --field-data <path_to_forest_plots_processed.csv> \
        --output-dir <output_directory>

Example:
    python src/evaluation/analyze_predictions_by_site.py \
        --comparison-results data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/comparison/comparison_results.csv \
        --field-data data/processed/forest_plot_data/forest_plots_processed.csv \
        --output-dir data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/diagnostics
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def compute_metrics(
    field_values: np.ndarray,
    pred_values: np.ndarray
) -> Dict[str, float]:
    """
    Compute regression metrics between field and predicted values.

    Args:
        field_values: Array of field measurements
        pred_values: Array of model predictions

    Returns:
        Dict with R², RMSE, MAE, bias, correlation metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(field_values) | np.isnan(pred_values))
    field = field_values[mask]
    pred = pred_values[mask]

    if len(field) < 3:
        return {
            'n': len(field),
            'r': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'field_mean': np.nan,
            'field_std': np.nan,
            'pred_mean': np.nan,
            'pred_std': np.nan
        }

    # Correlation
    r, p_value = stats.pearsonr(field, pred)

    # Error metrics
    errors = pred - field
    rmse = np.sqrt(np.mean(errors ** 2))
    mae = np.mean(np.abs(errors))
    bias = np.mean(errors)

    return {
        'n': len(field),
        'r': r,
        'r_squared': r ** 2,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'bias': bias,
        'field_mean': np.mean(field),
        'field_std': np.std(field),
        'pred_mean': np.mean(pred),
        'pred_std': np.std(pred)
    }


def analyze_by_site(
    df: pd.DataFrame,
    target: str,
    field_col: str,
    pred_col: str
) -> pd.DataFrame:
    """
    Compute per-site statistics for a target variable.

    Args:
        df: DataFrame with comparison results
        target: Name of target variable (for labeling)
        field_col: Column name for field measurements
        pred_col: Column name for predictions

    Returns:
        DataFrame with per-site statistics
    """
    results = []

    for site in df['site_name_field'].unique():
        site_df = df[df['site_name_field'] == site]

        # Only include plots with valid predictions (coverage > 0)
        valid_df = site_df[site_df['canopy_coverage_fraction'] > 0]

        if len(valid_df) == 0:
            continue

        field_vals = valid_df[field_col].values
        pred_vals = valid_df[pred_col].values

        metrics = compute_metrics(field_vals, pred_vals)
        metrics['site'] = site
        metrics['target'] = target
        metrics['n_total'] = len(site_df)
        metrics['n_valid'] = len(valid_df)

        results.append(metrics)

    return pd.DataFrame(results)


def analyze_by_site_year(
    df: pd.DataFrame,
    field_data: pd.DataFrame,
    target: str,
    field_col: str,
    pred_col: str
) -> pd.DataFrame:
    """
    Compute site × year interaction statistics.

    Args:
        df: DataFrame with comparison results (merged with year info)
        field_data: Source field data with Year column
        target: Name of target variable
        field_col: Column name for field measurements
        pred_col: Column name for predictions

    Returns:
        DataFrame with site × year statistics
    """
    # Merge year information from field data
    if 'Year' not in df.columns:
        # Try to match by plot coordinates or ID
        if 'Year' in field_data.columns:
            year_map = field_data.set_index(['Site', 'Easting', 'Northing'])['Year'].to_dict()
            df['Year'] = df.apply(
                lambda row: year_map.get(
                    (row['site_name_field'],
                     round(row['plot_x'], 0),
                     round(row['plot_y'], 0)),
                    np.nan
                ),
                axis=1
            )

    results = []

    for site in df['site_name_field'].unique():
        site_df = df[df['site_name_field'] == site]

        if 'Year' not in site_df.columns or site_df['Year'].isna().all():
            # No year data, analyze as single group
            valid_df = site_df[site_df['canopy_coverage_fraction'] > 0]
            if len(valid_df) > 0:
                metrics = compute_metrics(
                    valid_df[field_col].values,
                    valid_df[pred_col].values
                )
                metrics['site'] = site
                metrics['year'] = 'Unknown'
                metrics['target'] = target
                results.append(metrics)
            continue

        for year in site_df['Year'].dropna().unique():
            year_df = site_df[site_df['Year'] == year]
            valid_df = year_df[year_df['canopy_coverage_fraction'] > 0]

            if len(valid_df) < 3:
                continue

            metrics = compute_metrics(
                valid_df[field_col].values,
                valid_df[pred_col].values
            )
            metrics['site'] = site
            metrics['year'] = int(year)
            metrics['target'] = target
            results.append(metrics)

    return pd.DataFrame(results)


def create_site_scatter_plots(
    df: pd.DataFrame,
    output_path: Path,
    target: str,
    field_col: str,
    pred_col: str,
    units: str
) -> None:
    """Create scatter plots by site with regression lines."""
    sites = df['site_name_field'].unique()
    n_sites = len(sites)

    fig, axes = plt.subplots(1, n_sites, figsize=(4*n_sites, 4))
    if n_sites == 1:
        axes = [axes]

    colors = plt.cm.tab10(np.linspace(0, 1, n_sites))

    for ax, site, color in zip(axes, sites, colors):
        site_df = df[df['site_name_field'] == site]
        valid_df = site_df[site_df['canopy_coverage_fraction'] > 0]

        if len(valid_df) == 0:
            ax.text(0.5, 0.5, f'{site}\nNo valid data',
                   transform=ax.transAxes, ha='center')
            continue

        field_vals = valid_df[field_col].values
        pred_vals = valid_df[pred_col].values

        # Remove NaN
        mask = ~(np.isnan(field_vals) | np.isnan(pred_vals))
        field_vals = field_vals[mask]
        pred_vals = pred_vals[mask]

        if len(field_vals) < 3:
            ax.text(0.5, 0.5, f'{site}\nn<3',
                   transform=ax.transAxes, ha='center')
            continue

        # Scatter plot
        ax.scatter(field_vals, pred_vals, c=[color], alpha=0.7, s=50)

        # 1:1 line
        lims = [
            min(min(field_vals), min(pred_vals)),
            max(max(field_vals), max(pred_vals))
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1')

        # Regression line
        if len(field_vals) >= 3:
            slope, intercept, r, p, se = stats.linregress(field_vals, pred_vals)
            x_reg = np.array(lims)
            y_reg = slope * x_reg + intercept
            ax.plot(x_reg, y_reg, color=color, linestyle='-',
                   label=f'R²={r**2:.3f}')

        ax.set_xlabel(f'Field {target} ({units})')
        ax.set_ylabel(f'Predicted {target} ({units})')
        ax.set_title(f'{site} (n={len(field_vals)})')
        ax.legend(loc='upper left')
        ax.set_aspect('equal', adjustable='box')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved scatter plots to {output_path}")


def create_site_year_heatmap(
    site_year_df: pd.DataFrame,
    output_path: Path,
    target: str
) -> None:
    """Create heatmap of R² values by site × year."""
    if len(site_year_df) == 0:
        logger.warning(f"No site×year data for {target}")
        return

    # Pivot for heatmap
    pivot = site_year_df.pivot(index='site', columns='year', values='r_squared')

    fig, ax = plt.subplots(figsize=(8, 6))

    # Handle NaN values in heatmap
    mask = pivot.isna()

    sns.heatmap(
        pivot,
        annot=True,
        fmt='.3f',
        cmap='RdYlGn',
        vmin=0,
        vmax=1,
        mask=mask,
        ax=ax,
        cbar_kws={'label': 'R²'}
    )

    ax.set_title(f'{target} Prediction R² by Site × Year')
    ax.set_xlabel('Year')
    ax.set_ylabel('Site')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved heatmap to {output_path}")


def create_distribution_comparison(
    df: pd.DataFrame,
    output_path: Path,
    target: str,
    field_col: str,
    pred_col: str,
    units: str
) -> None:
    """Create distribution comparison plots (histograms) by site."""
    sites = df['site_name_field'].unique()
    n_sites = len(sites)

    fig, axes = plt.subplots(2, n_sites, figsize=(4*n_sites, 6))
    if n_sites == 1:
        axes = axes.reshape(2, 1)

    for i, site in enumerate(sites):
        site_df = df[df['site_name_field'] == site]
        valid_df = site_df[site_df['canopy_coverage_fraction'] > 0]

        # Field distribution
        ax_field = axes[0, i]
        field_vals = valid_df[field_col].dropna().values
        if len(field_vals) > 0:
            ax_field.hist(field_vals, bins=15, alpha=0.7, color='steelblue')
            ax_field.axvline(np.mean(field_vals), color='red', linestyle='--',
                           label=f'μ={np.mean(field_vals):.1f}')
        ax_field.set_title(f'{site} - Field')
        ax_field.set_xlabel(f'{target} ({units})')
        ax_field.legend()

        # Prediction distribution
        ax_pred = axes[1, i]
        pred_vals = valid_df[pred_col].dropna().values
        if len(pred_vals) > 0:
            ax_pred.hist(pred_vals, bins=15, alpha=0.7, color='darkorange')
            ax_pred.axvline(np.mean(pred_vals), color='red', linestyle='--',
                          label=f'μ={np.mean(pred_vals):.1f}')
        ax_pred.set_title(f'{site} - Predicted')
        ax_pred.set_xlabel(f'{target} ({units})')
        ax_pred.legend()

    plt.suptitle(f'{target} Distribution Comparison', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved distribution plots to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze forest plot predictions by site and year'
    )
    parser.add_argument(
        '--comparison-results',
        type=str,
        required=True,
        help='Path to comparison_results.csv'
    )
    parser.add_argument(
        '--field-data',
        type=str,
        required=True,
        help='Path to forest_plots_processed.csv (source field data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for analysis results'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading comparison results from {args.comparison_results}")
    df = pd.read_csv(args.comparison_results)
    logger.info(f"Loaded {len(df)} plots")

    logger.info(f"Loading field data from {args.field_data}")
    field_data = pd.read_csv(args.field_data)
    logger.info(f"Loaded {len(field_data)} field records")

    # Merge year info
    if 'Year' in field_data.columns:
        # Create lookup by rounded coordinates
        field_data['x_round'] = field_data['Easting'].round(0)
        field_data['y_round'] = field_data['Northing'].round(0)
        df['x_round'] = df['plot_x'].round(0)
        df['y_round'] = df['plot_y'].round(0)

        year_lookup = field_data.set_index(['Site', 'x_round', 'y_round'])['Year'].to_dict()
        df['Year'] = df.apply(
            lambda row: year_lookup.get(
                (row['site_name_field'], row['x_round'], row['y_round']),
                np.nan
            ),
            axis=1
        )
        logger.info(f"Matched {df['Year'].notna().sum()} plots with year info")

    # Define analysis targets
    targets = [
        {
            'name': 'TreeCover',
            'field_col': 'TreeCover_field',
            'pred_col': 'Canopy_cover_pred_pct',
            'units': '%'
        },
        {
            'name': 'TotalFuels',
            'field_col': 'TotalFuels_field',
            'pred_col': 'TFL_pred_tons_acre',
            'units': 'tons/acre'
        }
    ]

    all_site_stats = []
    all_site_year_stats = []

    for target_info in targets:
        target = target_info['name']
        field_col = target_info['field_col']
        pred_col = target_info['pred_col']
        units = target_info['units']

        logger.info(f"\n{'='*50}")
        logger.info(f"Analyzing {target}")
        logger.info('='*50)

        # Per-site analysis
        site_stats = analyze_by_site(df, target, field_col, pred_col)
        all_site_stats.append(site_stats)

        logger.info(f"\nPer-site {target} statistics:")
        print(site_stats[['site', 'n', 'r_squared', 'rmse', 'bias']].to_string(index=False))

        # Site × Year analysis
        site_year_stats = analyze_by_site_year(df, field_data, target, field_col, pred_col)
        all_site_year_stats.append(site_year_stats)

        if len(site_year_stats) > 0:
            logger.info(f"\nSite×Year {target} statistics:")
            print(site_year_stats[['site', 'year', 'n', 'r_squared', 'rmse', 'bias']].to_string(index=False))

        # Create visualizations
        create_site_scatter_plots(
            df, output_dir / f'{target}_scatter_by_site.png',
            target, field_col, pred_col, units
        )

        if len(site_year_stats) > 0 and 'year' in site_year_stats.columns:
            create_site_year_heatmap(
                site_year_stats,
                output_dir / f'{target}_heatmap_site_year.png',
                target
            )

        create_distribution_comparison(
            df, output_dir / f'{target}_distributions_by_site.png',
            target, field_col, pred_col, units
        )

    # Save combined statistics
    site_stats_combined = pd.concat(all_site_stats, ignore_index=True)
    site_stats_combined.to_csv(output_dir / 'site_statistics.csv', index=False)
    logger.info(f"\nSaved site statistics to {output_dir / 'site_statistics.csv'}")

    if all_site_year_stats:
        site_year_combined = pd.concat(all_site_year_stats, ignore_index=True)
        site_year_combined.to_csv(output_dir / 'site_year_statistics.csv', index=False)
        logger.info(f"Saved site×year statistics to {output_dir / 'site_year_statistics.csv'}")

    # Summary JSON
    summary = {
        'targets': [t['name'] for t in targets],
        'n_sites': len(df['site_name_field'].unique()),
        'sites': list(df['site_name_field'].unique()),
        'n_plots_total': len(df),
        'n_plots_with_predictions': len(df[df['canopy_coverage_fraction'] > 0]),
    }

    with open(output_dir / 'analysis_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()
