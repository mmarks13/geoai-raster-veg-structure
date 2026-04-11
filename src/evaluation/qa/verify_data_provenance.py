#!/usr/bin/env python3
"""
Verify data provenance through the forest plot evaluation pipeline.

This script traces sample plots from source CSV through preprocessing,
inference, and comparison to verify data integrity at each stage.

Usage:
    python src/evaluation/verify_data_provenance.py \
        --source-csv data/processed/forest_plot_data/forest_plots_processed.csv \
        --tiles-pt data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt \
        --predictions-dir data/output/forest_plot_evaluations/<model>/predictions \
        --comparison-csv data/output/forest_plot_evaluations/<model>/comparison/comparison_results.csv \
        --output data/output/forest_plot_evaluations/<model>/diagnostics/provenance_report.json

Example:
    python src/evaluation/verify_data_provenance.py \
        --source-csv data/processed/forest_plot_data/forest_plots_processed.csv \
        --tiles-pt data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt \
        --predictions-dir data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/predictions \
        --comparison-csv data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/comparison/comparison_results.csv \
        --output data/output/forest_plot_evaluations/raster_model_naip_20251203_190008/diagnostics/provenance_report.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_source_csv(path: str) -> pd.DataFrame:
    """Load source field measurement CSV."""
    logger.info(f"Loading source CSV from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Shape: {df.shape}")
    return df


def load_tiles(path: str) -> List[Dict]:
    """Load preprocessed tiles from .pt file."""
    logger.info(f"Loading tiles from {path}")
    tiles = torch.load(path, map_location='cpu', weights_only=False)
    logger.info(f"  Loaded {len(tiles)} tiles")
    return tiles


def load_predictions(predictions_dir: str) -> Dict[str, pd.DataFrame]:
    """Load prediction CSVs from predictions directory."""
    predictions_dir = Path(predictions_dir)
    predictions = {}

    for csv_path in predictions_dir.glob('forest_plot_predictions_*.csv'):
        site = csv_path.stem.replace('forest_plot_predictions_', '')
        df = pd.read_csv(csv_path)
        predictions[site] = df
        logger.info(f"  Loaded {len(df)} predictions for {site}")

    return predictions


def load_comparison_results(path: str) -> pd.DataFrame:
    """Load final comparison results."""
    logger.info(f"Loading comparison results from {path}")
    df = pd.read_csv(path)
    logger.info(f"  Shape: {df.shape}")
    return df


def trace_sample_plot(
    plot_idx: int,
    source_df: pd.DataFrame,
    tiles: List[Dict],
    predictions: Dict[str, pd.DataFrame],
    comparison_df: pd.DataFrame
) -> Dict:
    """
    Trace a single plot through all pipeline stages.

    Args:
        plot_idx: Index in source CSV
        source_df: Source field measurements
        tiles: Preprocessed tiles
        predictions: Prediction DataFrames by site
        comparison_df: Final comparison results

    Returns:
        Dict with values at each stage and any discrepancies
    """
    trace = {
        'plot_idx': plot_idx,
        'stages': {},
        'discrepancies': []
    }

    # Stage 1: Source CSV
    source_row = source_df.iloc[plot_idx]
    trace['stages']['source'] = {
        'Site': source_row.get('Site', 'N/A'),
        'Easting': float(source_row.get('Easting', np.nan)),
        'Northing': float(source_row.get('Northing', np.nan)),
        'TreeCover': float(source_row.get('TreeCover', np.nan)),
        'TotalFuels': float(source_row.get('TotalFuels', np.nan)),
        'Year': int(source_row.get('Year', 0)) if pd.notna(source_row.get('Year')) else None
    }

    site = trace['stages']['source']['Site']
    easting = trace['stages']['source']['Easting']
    northing = trace['stages']['source']['Northing']

    # Stage 2: Find matching tile(s)
    matching_tiles = []
    for tile in tiles:
        if tile.get('site_name') != site:
            continue

        bbox = tile.get('bbox', [])
        if len(bbox) >= 4:
            xmin, ymin, xmax, ymax = bbox[:4]
            if xmin <= easting <= xmax and ymin <= northing <= ymax:
                matching_tiles.append({
                    'tile_id': tile.get('tile_id', 'unknown'),
                    'bbox': list(bbox[:4]),
                    'center': list(tile.get('center', [])),
                    'has_naip': tile.get('naip') is not None,
                    'has_uavsar': tile.get('uavsar') is not None,
                    'dep_points_shape': list(tile['dep_points_norm'].shape) if 'dep_points_norm' in tile else None
                })

    trace['stages']['tiles'] = {
        'n_matching': len(matching_tiles),
        'tiles': matching_tiles[:5]  # Limit to first 5
    }

    if len(matching_tiles) == 0:
        trace['discrepancies'].append(f"No tiles found covering plot at ({easting}, {northing})")

    # Stage 3: Find in predictions
    pred_info = None
    if site in predictions:
        pred_df = predictions[site]
        # Find tiles that overlap with this plot location
        for _, row in pred_df.iterrows():
            bbox = [row['bbox_xmin'], row['bbox_ymin'], row['bbox_xmax'], row['bbox_ymax']]
            if bbox[0] <= easting <= bbox[2] and bbox[1] <= northing <= bbox[3]:
                pred_info = {
                    'tile_id': row.get('tile_id', 'unknown'),
                    'Canopy_cover_pred': row.get('Canopy_cover_pred'),
                    'TFL_pred_kg_m2': row.get('TFL_pred_kg_m2'),
                    'bbox': bbox
                }
                break

    trace['stages']['predictions'] = pred_info or {'found': False}

    if pred_info is None:
        trace['discrepancies'].append(f"No predictions found for plot at site {site}")

    # Stage 4: Find in comparison results
    comparison_info = None
    # Match by coordinates (rounded)
    for _, row in comparison_df.iterrows():
        if row['site_name_field'] != site:
            continue
        if abs(row['plot_x'] - easting) < 1.0 and abs(row['plot_y'] - northing) < 1.0:
            comparison_info = {
                'plot_id': row.get('plot_id'),
                'TreeCover_field': row.get('TreeCover_field'),
                'TotalFuels_field': row.get('TotalFuels_field'),
                'Canopy_cover_pred_pct': row.get('Canopy_cover_pred_pct'),
                'TFL_pred_tons_acre': row.get('TFL_pred_tons_acre'),
                'coverage_fraction': row.get('canopy_coverage_fraction')
            }
            break

    trace['stages']['comparison'] = comparison_info or {'found': False}

    if comparison_info is None:
        trace['discrepancies'].append(f"Plot not found in comparison results")

    # Verify value consistency
    if comparison_info and 'TreeCover_field' in comparison_info:
        source_tc = trace['stages']['source']['TreeCover']
        comp_tc = comparison_info['TreeCover_field']
        if pd.notna(source_tc) and pd.notna(comp_tc):
            if abs(source_tc - comp_tc) > 0.01:
                trace['discrepancies'].append(
                    f"TreeCover mismatch: source={source_tc}, comparison={comp_tc}"
                )

    if comparison_info and 'TotalFuels_field' in comparison_info:
        source_tfl = trace['stages']['source']['TotalFuels']
        comp_tfl = comparison_info['TotalFuels_field']
        if pd.notna(source_tfl) and pd.notna(comp_tfl):
            if abs(source_tfl - comp_tfl) > 0.01:
                trace['discrepancies'].append(
                    f"TotalFuels mismatch: source={source_tfl}, comparison={comp_tfl}"
                )

    return trace


def select_sample_plots(source_df: pd.DataFrame, n_per_site: int = 2) -> List[int]:
    """Select sample plot indices for tracing."""
    indices = []

    for site in source_df['Site'].unique():
        site_indices = source_df[source_df['Site'] == site].index.tolist()
        # Take first and last plot from each site
        if len(site_indices) >= n_per_site:
            indices.append(site_indices[0])
            indices.append(site_indices[-1])
        else:
            indices.extend(site_indices)

    return sorted(set(indices))


def create_summary(traces: List[Dict]) -> Dict:
    """Create summary statistics from all traces."""
    n_plots = len(traces)
    n_with_tiles = sum(1 for t in traces if t['stages']['tiles']['n_matching'] > 0)
    n_with_predictions = sum(1 for t in traces if t['stages']['predictions'].get('found', True))
    n_with_comparison = sum(1 for t in traces if t['stages']['comparison'].get('found', True))
    n_with_discrepancies = sum(1 for t in traces if len(t['discrepancies']) > 0)

    all_discrepancies = []
    for t in traces:
        for d in t['discrepancies']:
            all_discrepancies.append(d)

    return {
        'n_plots_traced': n_plots,
        'n_with_tiles': n_with_tiles,
        'n_with_predictions': n_with_predictions,
        'n_with_comparison': n_with_comparison,
        'n_with_discrepancies': n_with_discrepancies,
        'unique_discrepancy_types': list(set(
            d.split(':')[0] for d in all_discrepancies
        )),
        'all_discrepancies': all_discrepancies
    }


def main():
    parser = argparse.ArgumentParser(
        description='Verify data provenance through forest plot evaluation pipeline'
    )
    parser.add_argument(
        '--source-csv',
        type=str,
        required=True,
        help='Path to source forest_plots_processed.csv'
    )
    parser.add_argument(
        '--tiles-pt',
        type=str,
        required=True,
        help='Path to precomputed tiles .pt file'
    )
    parser.add_argument(
        '--predictions-dir',
        type=str,
        required=True,
        help='Directory containing prediction CSVs'
    )
    parser.add_argument(
        '--comparison-csv',
        type=str,
        required=True,
        help='Path to comparison_results.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output path for provenance report JSON'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=2,
        help='Number of sample plots per site to trace'
    )

    args = parser.parse_args()

    # Load data
    source_df = load_source_csv(args.source_csv)
    tiles = load_tiles(args.tiles_pt)
    predictions = load_predictions(args.predictions_dir)
    comparison_df = load_comparison_results(args.comparison_csv)

    # Select sample plots
    sample_indices = select_sample_plots(source_df, args.n_samples)
    logger.info(f"\nTracing {len(sample_indices)} sample plots")

    # Trace each plot
    traces = []
    for idx in sample_indices:
        trace = trace_sample_plot(
            idx, source_df, tiles, predictions, comparison_df
        )
        traces.append(trace)

        # Log progress
        site = trace['stages']['source']['Site']
        n_discrep = len(trace['discrepancies'])
        status = "OK" if n_discrep == 0 else f"{n_discrep} issues"
        logger.info(f"  Plot {idx} ({site}): {status}")

    # Create summary
    summary = create_summary(traces)

    # Build report
    report = {
        'summary': summary,
        'traces': traces
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nSaved provenance report to {output_path}")

    # Print summary
    logger.info("\n" + "="*50)
    logger.info("PROVENANCE SUMMARY")
    logger.info("="*50)
    logger.info(f"Plots traced: {summary['n_plots_traced']}")
    logger.info(f"  With tiles: {summary['n_with_tiles']}")
    logger.info(f"  With predictions: {summary['n_with_predictions']}")
    logger.info(f"  With comparison results: {summary['n_with_comparison']}")
    logger.info(f"  With discrepancies: {summary['n_with_discrepancies']}")

    if summary['all_discrepancies']:
        logger.info("\nDiscrepancy types found:")
        for d_type in summary['unique_discrepancy_types']:
            logger.info(f"  - {d_type}")


if __name__ == '__main__':
    main()
