#!/usr/bin/env python3
"""
Compare model predictions to 3DEP baseline predictions at forest plot locations.

This script:
1. Loads model predictions (from compare_predictions_to_plots.py output)
2. Loads 3DEP baseline predictions (from extract_fuel_metrics_at_plots.py output)
3. Computes per-site comparison statistics for both
4. Creates merged plot-level CSV for detailed inspection
5. Generates comparison visualizations (scatter plots + bar charts)

Usage:
    python src/evaluation/compare_model_to_baseline.py \
        --model-output data/output/forest_plot_evaluations/run_c_multiscale_20251209_120227

    # Override baseline path if needed
    python src/evaluation/compare_model_to_baseline.py \
        --model-output data/output/forest_plot_evaluations/run_c_multiscale_20251209_120227 \
        --baseline-dir data/processed/fuel_metrics/3dep_baseline/comparison

Output (in {model_output}/summary/):
    - site_comparison_summary.csv: Per-site stats with model + baseline + deltas
    - site_comparison_summary.json: Same data in JSON format
    - merged_plot_predictions.csv: Plot-level with model_pred, baseline_pred, field
    - comparison_scatter.png: Scatter plots (pred vs field) for model and baseline
    - comparison_barchart.png: Bar charts of R²/RMSE by site and method
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Default paths
DEFAULT_BASELINE_DIR = Path("data/processed/fuel_metrics/3dep_baseline/comparison")

# Sites to skip (known out-of-scope)
SKIP_SITES = {"TecuyaRidge"}

# Coverage threshold
MIN_COVERAGE_FRACTION = 0.95


def load_model_predictions(model_output_dir: Path) -> pd.DataFrame:
    """
    Load model predictions from comparison_results.csv.

    Args:
        model_output_dir: Path to model evaluation output folder

    Returns:
        DataFrame with model predictions
    """
    comparison_path = model_output_dir / "comparison" / "comparison_results.csv"

    if not comparison_path.exists():
        raise FileNotFoundError(f"Model comparison results not found: {comparison_path}")

    logger.info(f"Loading model predictions from {comparison_path}")
    df = pd.read_csv(comparison_path)

    # Standardize column names
    df = df.rename(columns={
        'site_name_field': 'Site',
        'Canopy_cover_pred_pct': 'model_TreeCover_pred',
        'TFL_pred_tons_acre': 'model_TotalFuels_pred',
        'canopy_coverage_fraction': 'model_TreeCover_coverage',
        'tfl_coverage_fraction': 'model_TotalFuels_coverage',
        'canopy_n_pixels': 'model_TreeCover_n_pixels',
        'tfl_n_pixels': 'model_TotalFuels_n_pixels',
    })

    logger.info(f"Loaded {len(df)} model predictions")
    logger.info(f"Sites: {df['Site'].unique()}")

    return df


def load_baseline_predictions(baseline_dir: Path) -> pd.DataFrame:
    """
    Load 3DEP baseline predictions from baseline_predictions.csv.

    Args:
        baseline_dir: Path to baseline comparison folder

    Returns:
        DataFrame with baseline predictions
    """
    predictions_path = baseline_dir / "baseline_predictions.csv"

    if not predictions_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found: {predictions_path}")

    logger.info(f"Loading baseline predictions from {predictions_path}")
    df = pd.read_csv(predictions_path)

    # Standardize column names
    df = df.rename(columns={
        'Cover_pct_pred': 'baseline_TreeCover_pred_raw',
        'TFL_pred_tons_acre': 'baseline_TotalFuels_pred',
        'Cover_pct_coverage': 'baseline_TreeCover_coverage',
        'TFL_kg_m2_coverage': 'baseline_TotalFuels_coverage',
        'Cover_pct_n_pixels': 'baseline_TreeCover_n_pixels',
        'TFL_kg_m2_n_pixels': 'baseline_TotalFuels_n_pixels',
        'TotalFuels_field_tons_acre': 'TotalFuels_field',
    })

    # Convert baseline TreeCover from fraction (0-1) to percent (0-100)
    if 'baseline_TreeCover_pred_raw' in df.columns:
        df['baseline_TreeCover_pred'] = df['baseline_TreeCover_pred_raw'] * 100
        df = df.drop(columns=['baseline_TreeCover_pred_raw'])

    logger.info(f"Loaded {len(df)} baseline predictions")

    return df


def merge_predictions(
    model_df: pd.DataFrame,
    baseline_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge model and baseline predictions on plot coordinates.

    Args:
        model_df: Model predictions DataFrame
        baseline_df: Baseline predictions DataFrame

    Returns:
        Merged DataFrame with both predictions
    """
    logger.info("Merging model and baseline predictions...")

    # Round coordinates for matching (handle floating point precision)
    model_df = model_df.copy()
    baseline_df = baseline_df.copy()

    model_df['plot_x_round'] = model_df['plot_x'].round(0)
    model_df['plot_y_round'] = model_df['plot_y'].round(0)
    baseline_df['plot_x_round'] = baseline_df['Easting'].round(0)
    baseline_df['plot_y_round'] = baseline_df['Northing'].round(0)

    # Select columns to merge from baseline
    baseline_cols = [
        'Site', 'Plot_ID', 'plot_x_round', 'plot_y_round',
        'baseline_TreeCover_pred', 'baseline_TotalFuels_pred',
        'baseline_TreeCover_coverage', 'baseline_TotalFuels_coverage',
        'baseline_TreeCover_n_pixels', 'baseline_TotalFuels_n_pixels',
    ]

    # Filter baseline columns that exist
    baseline_cols = [c for c in baseline_cols if c in baseline_df.columns]
    baseline_subset = baseline_df[baseline_cols].copy()

    # Merge on Site and rounded coordinates
    merged = model_df.merge(
        baseline_subset,
        on=['Site', 'plot_x_round', 'plot_y_round'],
        how='left',
        suffixes=('', '_baseline')
    )

    # Clean up
    merged = merged.drop(columns=['plot_x_round', 'plot_y_round'], errors='ignore')

    logger.info(f"Merged {len(merged)} plots")

    # Check merge success
    n_with_baseline = merged['baseline_TreeCover_pred'].notna().sum()
    logger.info(f"Plots with baseline data: {n_with_baseline}")

    return merged


def filter_by_coverage(
    df: pd.DataFrame,
    min_coverage: float = MIN_COVERAGE_FRACTION
) -> pd.DataFrame:
    """
    Filter plots by coverage threshold.

    Args:
        df: Merged predictions DataFrame
        min_coverage: Minimum coverage fraction (default 0.95)

    Returns:
        Filtered DataFrame
    """
    logger.info(f"Filtering plots with ≥{min_coverage*100:.0f}% coverage...")

    # Model coverage filter
    model_mask = (
        (df['model_TreeCover_coverage'] >= min_coverage) |
        (df['model_TotalFuels_coverage'] >= min_coverage)
    )

    n_before = len(df)
    df_filtered = df[model_mask].copy()
    n_after = len(df_filtered)

    logger.info(f"Filtered: {n_before} -> {n_after} plots ({n_after/n_before*100:.1f}% retained)")

    # Skip known out-of-scope sites
    df_filtered = df_filtered[~df_filtered['Site'].isin(SKIP_SITES)]

    if len(df_filtered) < n_after:
        logger.info(f"Skipped {n_after - len(df_filtered)} plots from out-of-scope sites: {SKIP_SITES}")

    return df_filtered


def compute_site_stats(
    df: pd.DataFrame,
    field_col: str,
    pred_col: str,
    coverage_col: str,
    min_coverage: float = MIN_COVERAGE_FRACTION
) -> Dict:
    """
    Compute comparison statistics for a single site/variable combination.

    Args:
        df: DataFrame with predictions
        field_col: Field measurement column name
        pred_col: Prediction column name
        coverage_col: Coverage fraction column name
        min_coverage: Minimum coverage threshold

    Returns:
        Dict with statistics
    """
    # Filter valid data
    valid_mask = (
        df[field_col].notna() &
        df[pred_col].notna() &
        (df[coverage_col] >= min_coverage)
    )

    valid_df = df[valid_mask]
    n = len(valid_df)

    if n < 3:
        return {
            'n': n,
            'n_full_coverage': int((df[coverage_col] >= 0.95).sum()),
            'n_partial_coverage': int(((df[coverage_col] >= 0.5) & (df[coverage_col] < 0.95)).sum()),
            'n_poor_coverage': int((df[coverage_col] < 0.5).sum()),
            'mean_coverage': float(df[coverage_col].mean()) if len(df) > 0 else np.nan,
            'r': np.nan,
            'r_squared': np.nan,
            'p_value': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan,
            'field_mean': np.nan,
            'field_std': np.nan,
            'pred_mean': np.nan,
            'pred_std': np.nan,
        }

    field_vals = valid_df[field_col].values
    pred_vals = valid_df[pred_col].values

    # Compute statistics
    r_val, p_val = stats.pearsonr(field_vals, pred_vals)
    rmse = np.sqrt(np.mean((pred_vals - field_vals) ** 2))
    mae = np.mean(np.abs(pred_vals - field_vals))
    bias = np.mean(pred_vals - field_vals)

    return {
        'n': n,
        'n_full_coverage': int((df[coverage_col] >= 0.95).sum()),
        'n_partial_coverage': int(((df[coverage_col] >= 0.5) & (df[coverage_col] < 0.95)).sum()),
        'n_poor_coverage': int((df[coverage_col] < 0.5).sum()),
        'mean_coverage': float(df[coverage_col].mean()),
        'r': float(r_val),
        'r_squared': float(r_val ** 2),
        'p_value': float(p_val),
        'rmse': float(rmse),
        'mae': float(mae),
        'bias': float(bias),
        'field_mean': float(field_vals.mean()),
        'field_std': float(field_vals.std()),
        'pred_mean': float(pred_vals.mean()),
        'pred_std': float(pred_vals.std()),
    }


def compute_matched_stats(
    df: pd.DataFrame,
    metric: str,
    min_coverage: float = MIN_COVERAGE_FRACTION
) -> Tuple[Dict, Dict, int]:
    """
    Compute matched-sample statistics (same plots for both methods).

    For apples-to-apples comparison, filter to plots where BOTH model AND
    baseline have ≥95% coverage, then compute stats for both methods on
    that identical set of plots.

    Args:
        df: Merged predictions DataFrame
        metric: 'TreeCover' or 'TotalFuels'
        min_coverage: Minimum coverage threshold

    Returns:
        Tuple of (model_stats, baseline_stats, n_matched_plots)
    """
    field_col = f'{metric}_field'
    model_pred_col = f'model_{metric}_pred'
    baseline_pred_col = f'baseline_{metric}_pred'
    model_cov_col = f'model_{metric}_coverage'
    baseline_cov_col = f'baseline_{metric}_coverage'

    # Filter to plots where BOTH have ≥95% coverage
    matched_mask = (
        df[field_col].notna() &
        df[model_pred_col].notna() &
        df[baseline_pred_col].notna() &
        (df[model_cov_col] >= min_coverage) &
        (df[baseline_cov_col] >= min_coverage)
    )

    matched_df = df[matched_mask]
    n_matched = len(matched_df)

    if n_matched < 3:
        empty_stats = {
            'n': n_matched, 'r': np.nan, 'r_squared': np.nan, 'p_value': np.nan,
            'rmse': np.nan, 'mae': np.nan, 'bias': np.nan,
            'field_mean': np.nan, 'field_std': np.nan,
            'pred_mean': np.nan, 'pred_std': np.nan,
        }
        return empty_stats, empty_stats, n_matched

    field_vals = matched_df[field_col].values
    model_vals = matched_df[model_pred_col].values
    baseline_vals = matched_df[baseline_pred_col].values

    def calc_stats(pred_vals):
        r_val, p_val = stats.pearsonr(field_vals, pred_vals)
        return {
            'n': n_matched,
            'r': float(r_val),
            'r_squared': float(r_val ** 2),
            'p_value': float(p_val),
            'rmse': float(np.sqrt(np.mean((pred_vals - field_vals) ** 2))),
            'mae': float(np.mean(np.abs(pred_vals - field_vals))),
            'bias': float(np.mean(pred_vals - field_vals)),
            'field_mean': float(field_vals.mean()),
            'field_std': float(field_vals.std()),
            'pred_mean': float(pred_vals.mean()),
            'pred_std': float(pred_vals.std()),
        }

    model_stats = calc_stats(model_vals)
    baseline_stats = calc_stats(baseline_vals)

    return model_stats, baseline_stats, n_matched


def compute_all_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute statistics for all sites and both methods.

    Args:
        df: Merged predictions DataFrame

    Returns:
        DataFrame with per-site statistics
    """
    logger.info("Computing per-site statistics...")

    results = []

    sites = sorted(df['Site'].unique())

    for site in sites:
        site_df = df[df['Site'] == site]

        row = {'Site': site}

        # Model TreeCover stats
        model_tc = compute_site_stats(
            site_df,
            'TreeCover_field',
            'model_TreeCover_pred',
            'model_TreeCover_coverage'
        )
        for k, v in model_tc.items():
            row[f'model_TreeCover_{k}'] = v

        # Model TotalFuels stats
        model_tf = compute_site_stats(
            site_df,
            'TotalFuels_field',
            'model_TotalFuels_pred',
            'model_TotalFuels_coverage'
        )
        for k, v in model_tf.items():
            row[f'model_TotalFuels_{k}'] = v

        # Baseline TreeCover stats (if available)
        if 'baseline_TreeCover_pred' in site_df.columns:
            baseline_tc = compute_site_stats(
                site_df,
                'TreeCover_field',
                'baseline_TreeCover_pred',
                'baseline_TreeCover_coverage'
            )
            for k, v in baseline_tc.items():
                row[f'baseline_TreeCover_{k}'] = v

            # Delta (model - baseline)
            if not np.isnan(model_tc['r_squared']) and not np.isnan(baseline_tc['r_squared']):
                row['delta_TreeCover_r_squared'] = model_tc['r_squared'] - baseline_tc['r_squared']
                row['delta_TreeCover_rmse'] = model_tc['rmse'] - baseline_tc['rmse']
                row['delta_TreeCover_mae'] = model_tc['mae'] - baseline_tc['mae']
            else:
                row['delta_TreeCover_r_squared'] = np.nan
                row['delta_TreeCover_rmse'] = np.nan
                row['delta_TreeCover_mae'] = np.nan
        else:
            # No baseline data
            for k in ['n', 'r', 'r_squared', 'p_value', 'rmse', 'mae', 'bias',
                      'field_mean', 'field_std', 'pred_mean', 'pred_std',
                      'n_full_coverage', 'n_partial_coverage', 'n_poor_coverage', 'mean_coverage']:
                row[f'baseline_TreeCover_{k}'] = np.nan
            row['delta_TreeCover_r_squared'] = np.nan
            row['delta_TreeCover_rmse'] = np.nan
            row['delta_TreeCover_mae'] = np.nan

        # Baseline TotalFuels stats (if available)
        if 'baseline_TotalFuels_pred' in site_df.columns:
            baseline_tf = compute_site_stats(
                site_df,
                'TotalFuels_field',
                'baseline_TotalFuels_pred',
                'baseline_TotalFuels_coverage'
            )
            for k, v in baseline_tf.items():
                row[f'baseline_TotalFuels_{k}'] = v

            # Delta (model - baseline)
            if not np.isnan(model_tf['r_squared']) and not np.isnan(baseline_tf['r_squared']):
                row['delta_TotalFuels_r_squared'] = model_tf['r_squared'] - baseline_tf['r_squared']
                row['delta_TotalFuels_rmse'] = model_tf['rmse'] - baseline_tf['rmse']
                row['delta_TotalFuels_mae'] = model_tf['mae'] - baseline_tf['mae']
            else:
                row['delta_TotalFuels_r_squared'] = np.nan
                row['delta_TotalFuels_rmse'] = np.nan
                row['delta_TotalFuels_mae'] = np.nan
        else:
            for k in ['n', 'r', 'r_squared', 'p_value', 'rmse', 'mae', 'bias',
                      'field_mean', 'field_std', 'pred_mean', 'pred_std',
                      'n_full_coverage', 'n_partial_coverage', 'n_poor_coverage', 'mean_coverage']:
                row[f'baseline_TotalFuels_{k}'] = np.nan
            row['delta_TotalFuels_r_squared'] = np.nan
            row['delta_TotalFuels_rmse'] = np.nan
            row['delta_TotalFuels_mae'] = np.nan

        results.append(row)

    # Add "All Sites" row
    all_row = {'Site': 'All Sites'}

    # Model TreeCover - all sites combined
    model_tc_all = compute_site_stats(
        df, 'TreeCover_field', 'model_TreeCover_pred', 'model_TreeCover_coverage'
    )
    for k, v in model_tc_all.items():
        all_row[f'model_TreeCover_{k}'] = v

    # Model TotalFuels - all sites combined
    model_tf_all = compute_site_stats(
        df, 'TotalFuels_field', 'model_TotalFuels_pred', 'model_TotalFuels_coverage'
    )
    for k, v in model_tf_all.items():
        all_row[f'model_TotalFuels_{k}'] = v

    # Baseline - all sites combined
    if 'baseline_TreeCover_pred' in df.columns:
        baseline_tc_all = compute_site_stats(
            df, 'TreeCover_field', 'baseline_TreeCover_pred', 'baseline_TreeCover_coverage'
        )
        for k, v in baseline_tc_all.items():
            all_row[f'baseline_TreeCover_{k}'] = v

        if not np.isnan(model_tc_all['r_squared']) and not np.isnan(baseline_tc_all['r_squared']):
            all_row['delta_TreeCover_r_squared'] = model_tc_all['r_squared'] - baseline_tc_all['r_squared']
            all_row['delta_TreeCover_rmse'] = model_tc_all['rmse'] - baseline_tc_all['rmse']
            all_row['delta_TreeCover_mae'] = model_tc_all['mae'] - baseline_tc_all['mae']
        else:
            all_row['delta_TreeCover_r_squared'] = np.nan
            all_row['delta_TreeCover_rmse'] = np.nan
            all_row['delta_TreeCover_mae'] = np.nan

    if 'baseline_TotalFuels_pred' in df.columns:
        baseline_tf_all = compute_site_stats(
            df, 'TotalFuels_field', 'baseline_TotalFuels_pred', 'baseline_TotalFuels_coverage'
        )
        for k, v in baseline_tf_all.items():
            all_row[f'baseline_TotalFuels_{k}'] = v

        if not np.isnan(model_tf_all['r_squared']) and not np.isnan(baseline_tf_all['r_squared']):
            all_row['delta_TotalFuels_r_squared'] = model_tf_all['r_squared'] - baseline_tf_all['r_squared']
            all_row['delta_TotalFuels_rmse'] = model_tf_all['rmse'] - baseline_tf_all['rmse']
            all_row['delta_TotalFuels_mae'] = model_tf_all['mae'] - baseline_tf_all['mae']
        else:
            all_row['delta_TotalFuels_r_squared'] = np.nan
            all_row['delta_TotalFuels_rmse'] = np.nan
            all_row['delta_TotalFuels_mae'] = np.nan

    results.append(all_row)

    return pd.DataFrame(results)


def plot_scatter_comparison(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate scatter plots comparing model and baseline to field measurements.

    Args:
        df: Merged predictions DataFrame
        output_path: Path to save figure
    """
    logger.info(f"Generating scatter plots: {output_path}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Color by site
    sites = sorted(df['Site'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(sites)))
    site_colors = dict(zip(sites, colors))

    # TreeCover - Model
    ax = axes[0, 0]
    for site in sites:
        site_df = df[df['Site'] == site]
        mask = site_df['TreeCover_field'].notna() & site_df['model_TreeCover_pred'].notna()
        if mask.sum() > 0:
            ax.scatter(
                site_df.loc[mask, 'TreeCover_field'],
                site_df.loc[mask, 'model_TreeCover_pred'],
                c=[site_colors[site]], label=site, alpha=0.7, s=40
            )

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel('Field TreeCover (%)')
    ax.set_ylabel('Model Prediction (%)')
    ax.set_title('Model: TreeCover')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # TreeCover - Baseline
    ax = axes[0, 1]
    if 'baseline_TreeCover_pred' in df.columns:
        for site in sites:
            site_df = df[df['Site'] == site]
            mask = (
                site_df['TreeCover_field'].notna() &
                site_df['baseline_TreeCover_pred'].notna() &
                (site_df['baseline_TreeCover_coverage'] >= MIN_COVERAGE_FRACTION)
            )
            if mask.sum() > 0:
                ax.scatter(
                    site_df.loc[mask, 'TreeCover_field'],
                    site_df.loc[mask, 'baseline_TreeCover_pred'],
                    c=[site_colors[site]], label=site, alpha=0.7, s=40
                )

    ax.plot([0, 100], [0, 100], 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel('Field TreeCover (%)')
    ax.set_ylabel('3DEP Baseline Prediction (%)')
    ax.set_title('3DEP Baseline: TreeCover')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    # TotalFuels - Model
    ax = axes[1, 0]
    for site in sites:
        site_df = df[df['Site'] == site]
        mask = site_df['TotalFuels_field'].notna() & site_df['model_TotalFuels_pred'].notna()
        if mask.sum() > 0:
            ax.scatter(
                site_df.loc[mask, 'TotalFuels_field'],
                site_df.loc[mask, 'model_TotalFuels_pred'],
                c=[site_colors[site]], label=site, alpha=0.7, s=40
            )

    max_val = max(
        df['TotalFuels_field'].max(),
        df['model_TotalFuels_pred'].max()
    )
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel('Field TotalFuels (tons/acre)')
    ax.set_ylabel('Model Prediction (tons/acre)')
    ax.set_title('Model: TotalFuels')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # TotalFuels - Baseline
    ax = axes[1, 1]
    if 'baseline_TotalFuels_pred' in df.columns:
        for site in sites:
            site_df = df[df['Site'] == site]
            mask = (
                site_df['TotalFuels_field'].notna() &
                site_df['baseline_TotalFuels_pred'].notna() &
                (site_df['baseline_TotalFuels_coverage'] >= MIN_COVERAGE_FRACTION)
            )
            if mask.sum() > 0:
                ax.scatter(
                    site_df.loc[mask, 'TotalFuels_field'],
                    site_df.loc[mask, 'baseline_TotalFuels_pred'],
                    c=[site_colors[site]], label=site, alpha=0.7, s=40
                )

    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='1:1')
    ax.set_xlabel('Field TotalFuels (tons/acre)')
    ax.set_ylabel('3DEP Baseline Prediction (tons/acre)')
    ax.set_title('3DEP Baseline: TotalFuels')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved scatter plots to {output_path}")


def plot_bar_comparison(
    stats_df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Generate bar charts comparing R² and RMSE across sites and methods.

    Args:
        stats_df: Statistics DataFrame
        output_path: Path to save figure
    """
    logger.info(f"Generating bar charts: {output_path}")

    # Filter to individual sites (exclude "All Sites" row)
    site_stats = stats_df[stats_df['Site'] != 'All Sites'].copy()

    if len(site_stats) == 0:
        logger.warning("No site-level statistics to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    sites = site_stats['Site'].tolist()
    x = np.arange(len(sites))
    width = 0.35

    # TreeCover R²
    ax = axes[0, 0]
    model_r2 = site_stats['model_TreeCover_r_squared'].fillna(0).values
    baseline_r2 = site_stats['baseline_TreeCover_r_squared'].fillna(0).values

    bars1 = ax.bar(x - width/2, model_r2, width, label='Model', color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_r2, width, label='3DEP Baseline', color='coral')

    ax.set_ylabel('R²')
    ax.set_title('TreeCover: R² by Site')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        if bar.get_height() > 0:
            ax.annotate(f'{bar.get_height():.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        if bar.get_height() > 0:
            ax.annotate(f'{bar.get_height():.2f}',
                       xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                       ha='center', va='bottom', fontsize=8)

    # TreeCover RMSE
    ax = axes[0, 1]
    model_rmse = site_stats['model_TreeCover_rmse'].fillna(0).values
    baseline_rmse = site_stats['baseline_TreeCover_rmse'].fillna(0).values

    bars1 = ax.bar(x - width/2, model_rmse, width, label='Model', color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_rmse, width, label='3DEP Baseline', color='coral')

    ax.set_ylabel('RMSE (%)')
    ax.set_title('TreeCover: RMSE by Site')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # TotalFuels R²
    ax = axes[1, 0]
    model_r2 = site_stats['model_TotalFuels_r_squared'].fillna(0).values
    baseline_r2 = site_stats['baseline_TotalFuels_r_squared'].fillna(0).values

    bars1 = ax.bar(x - width/2, model_r2, width, label='Model', color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_r2, width, label='3DEP Baseline', color='coral')

    ax.set_ylabel('R²')
    ax.set_title('TotalFuels: R² by Site')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')

    # TotalFuels RMSE
    ax = axes[1, 1]
    model_rmse = site_stats['model_TotalFuels_rmse'].fillna(0).values
    baseline_rmse = site_stats['baseline_TotalFuels_rmse'].fillna(0).values

    bars1 = ax.bar(x - width/2, model_rmse, width, label='Model', color='steelblue')
    bars2 = ax.bar(x + width/2, baseline_rmse, width, label='3DEP Baseline', color='coral')

    ax.set_ylabel('RMSE (tons/acre)')
    ax.set_title('TotalFuels: RMSE by Site')
    ax.set_xticks(x)
    ax.set_xticklabels(sites, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved bar charts to {output_path}")


def save_merged_predictions(
    df: pd.DataFrame,
    output_path: Path
) -> None:
    """
    Save merged plot-level predictions CSV.

    Args:
        df: Merged predictions DataFrame
        output_path: Path to save CSV
    """
    # Select and order columns for output
    output_cols = [
        'plot_id', 'Site', 'plot_x', 'plot_y',
        'TreeCover_field', 'model_TreeCover_pred', 'baseline_TreeCover_pred',
        'model_TreeCover_coverage', 'baseline_TreeCover_coverage',
        'TotalFuels_field', 'model_TotalFuels_pred', 'baseline_TotalFuels_pred',
        'model_TotalFuels_coverage', 'baseline_TotalFuels_coverage',
    ]

    # Filter to columns that exist
    output_cols = [c for c in output_cols if c in df.columns]

    df[output_cols].to_csv(output_path, index=False)
    logger.info(f"Saved merged predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare model predictions to 3DEP baseline at forest plot locations"
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        required=True,
        help="Path to model evaluation output folder"
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=DEFAULT_BASELINE_DIR,
        help=f"Path to baseline comparison folder (default: {DEFAULT_BASELINE_DIR})"
    )

    args = parser.parse_args()

    # Validate paths
    if not args.model_output.exists():
        raise FileNotFoundError(f"Model output folder not found: {args.model_output}")

    if not args.baseline_dir.exists():
        logger.warning(f"Baseline folder not found: {args.baseline_dir}")
        logger.warning("Proceeding without baseline comparison")
        baseline_available = False
    else:
        baseline_available = True

    # Create output directory
    output_dir = args.model_output / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MODEL VS BASELINE COMPARISON")
    logger.info("=" * 60)
    logger.info(f"Model output: {args.model_output}")
    logger.info(f"Baseline dir: {args.baseline_dir}")
    logger.info(f"Output dir: {output_dir}")
    logger.info(f"Coverage threshold: ≥{MIN_COVERAGE_FRACTION*100:.0f}%")
    logger.info("")

    # Load data
    model_df = load_model_predictions(args.model_output)

    if baseline_available:
        baseline_df = load_baseline_predictions(args.baseline_dir)
        merged_df = merge_predictions(model_df, baseline_df)
    else:
        merged_df = model_df.copy()

    # Filter by coverage
    filtered_df = filter_by_coverage(merged_df)

    # Compute statistics
    stats_df = compute_all_stats(filtered_df)

    # Save outputs
    # 1. Summary CSV
    summary_csv_path = output_dir / "site_comparison_summary.csv"
    stats_df.to_csv(summary_csv_path, index=False)
    logger.info(f"\nSaved summary CSV to {summary_csv_path}")

    # 2. Summary JSON
    summary_json_path = output_dir / "site_comparison_summary.json"
    stats_dict = stats_df.set_index('Site').to_dict(orient='index')
    with open(summary_json_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    logger.info(f"Saved summary JSON to {summary_json_path}")

    # 3. Merged plot-level CSV
    merged_csv_path = output_dir / "merged_plot_predictions.csv"
    save_merged_predictions(filtered_df, merged_csv_path)

    # 4. Scatter plots
    scatter_path = output_dir / "comparison_scatter.png"
    plot_scatter_comparison(filtered_df, scatter_path)

    # 5. Bar charts
    barchart_path = output_dir / "comparison_barchart.png"
    plot_bar_comparison(stats_df, barchart_path)

    # Compute matched-sample statistics (apples-to-apples)
    matched_stats = {}
    if baseline_available:
        logger.info("\nComputing matched-sample (apples-to-apples) statistics...")

        tc_model, tc_baseline, tc_n = compute_matched_stats(filtered_df, 'TreeCover')
        tf_model, tf_baseline, tf_n = compute_matched_stats(filtered_df, 'TotalFuels')

        matched_stats = {
            'TreeCover': {
                'n_matched': tc_n,
                'model': tc_model,
                'baseline': tc_baseline,
                'delta_r_squared': tc_model['r_squared'] - tc_baseline['r_squared'] if not np.isnan(tc_model['r_squared']) else np.nan,
                'delta_rmse': tc_model['rmse'] - tc_baseline['rmse'] if not np.isnan(tc_model['rmse']) else np.nan,
            },
            'TotalFuels': {
                'n_matched': tf_n,
                'model': tf_model,
                'baseline': tf_baseline,
                'delta_r_squared': tf_model['r_squared'] - tf_baseline['r_squared'] if not np.isnan(tf_model['r_squared']) else np.nan,
                'delta_rmse': tf_model['rmse'] - tf_baseline['rmse'] if not np.isnan(tf_model['rmse']) else np.nan,
            }
        }

        logger.info(f"  TreeCover: {tc_n} matched plots")
        logger.info(f"  TotalFuels: {tf_n} matched plots")

        # Save matched stats to JSON
        matched_json_path = output_dir / "matched_sample_comparison.json"
        with open(matched_json_path, 'w') as f:
            json.dump(matched_stats, f, indent=2)
        logger.info(f"Saved matched-sample stats to {matched_json_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
    logger.info("=" * 60)

    all_row = stats_df[stats_df['Site'] == 'All Sites'].iloc[0]

    logger.info("\nTreeCover (All Sites Combined):")
    logger.info(f"  Model:    R²={all_row['model_TreeCover_r_squared']:.3f}, RMSE={all_row['model_TreeCover_rmse']:.1f}%, n={all_row['model_TreeCover_n']:.0f}")
    if baseline_available and not np.isnan(all_row.get('baseline_TreeCover_r_squared', np.nan)):
        logger.info(f"  Baseline: R²={all_row['baseline_TreeCover_r_squared']:.3f}, RMSE={all_row['baseline_TreeCover_rmse']:.1f}%, n={all_row['baseline_TreeCover_n']:.0f}")
        logger.info(f"  Delta:    R²={all_row['delta_TreeCover_r_squared']:+.3f}, RMSE={all_row['delta_TreeCover_rmse']:+.1f}%")

    logger.info("\nTotalFuels (All Sites Combined):")
    logger.info(f"  Model:    R²={all_row['model_TotalFuels_r_squared']:.3f}, RMSE={all_row['model_TotalFuels_rmse']:.1f} tons/acre, n={all_row['model_TotalFuels_n']:.0f}")
    if baseline_available and not np.isnan(all_row.get('baseline_TotalFuels_r_squared', np.nan)):
        logger.info(f"  Baseline: R²={all_row['baseline_TotalFuels_r_squared']:.3f}, RMSE={all_row['baseline_TotalFuels_rmse']:.1f} tons/acre, n={all_row['baseline_TotalFuels_n']:.0f}")
        logger.info(f"  Delta:    R²={all_row['delta_TotalFuels_r_squared']:+.3f}, RMSE={all_row['delta_TotalFuels_rmse']:+.1f} tons/acre")

    # Print matched-sample (apples-to-apples) comparison
    if baseline_available and matched_stats:
        logger.info("\n" + "-" * 60)
        logger.info("MATCHED-SAMPLE COMPARISON (Apples-to-Apples)")
        logger.info("-" * 60)
        logger.info("Same plots evaluated for both methods:")

        tc = matched_stats['TreeCover']
        logger.info(f"\nTreeCover (n={tc['n_matched']} matched plots):")
        logger.info(f"  Model:    R²={tc['model']['r_squared']:.3f}, RMSE={tc['model']['rmse']:.1f}%, MAE={tc['model']['mae']:.1f}%, Bias={tc['model']['bias']:+.1f}%")
        logger.info(f"  Baseline: R²={tc['baseline']['r_squared']:.3f}, RMSE={tc['baseline']['rmse']:.1f}%, MAE={tc['baseline']['mae']:.1f}%, Bias={tc['baseline']['bias']:+.1f}%")
        logger.info(f"  Delta:    R²={tc['delta_r_squared']:+.3f}, RMSE={tc['delta_rmse']:+.1f}%")

        tf = matched_stats['TotalFuels']
        logger.info(f"\nTotalFuels (n={tf['n_matched']} matched plots):")
        logger.info(f"  Model:    R²={tf['model']['r_squared']:.3f}, RMSE={tf['model']['rmse']:.1f}, MAE={tf['model']['mae']:.1f}, Bias={tf['model']['bias']:+.1f} tons/acre")
        logger.info(f"  Baseline: R²={tf['baseline']['r_squared']:.3f}, RMSE={tf['baseline']['rmse']:.1f}, MAE={tf['baseline']['mae']:.1f}, Bias={tf['baseline']['bias']:+.1f} tons/acre")
        logger.info(f"  Delta:    R²={tf['delta_r_squared']:+.3f}, RMSE={tf['delta_rmse']:+.1f} tons/acre")

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
