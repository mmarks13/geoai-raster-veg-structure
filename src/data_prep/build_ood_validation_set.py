#!/usr/bin/env python3
"""
Build a small out-of-distribution validation set for in-training forest plot
validation.

Selects ~3 plots per site (stratified by canopy cover quantile) from the
inclusion-filtered set used by the final §7 evaluation, then gathers all tiles
whose bbox overlaps the 11.35 m plot footprint plus a 2 m search buffer.

Outputs (under --output-dir):
    ood_validation_tiles.pt        - List[dict] of unique tile dicts
    ood_validation_metadata.json   - Per-plot record (field measurements + tile_ids)
    ood_validation_plot_ids.txt    - Flat list of selected plot IDs (for §7 exclusion)
    ood_validation_config.json     - Reproducibility / build config

Usage:
    python src/data_prep/build_ood_validation_set.py \
        --per-site 3 \
        --seed 42 \
        --output-dir data/processed/forest_plot_data/ood_validation
"""

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import List, Set

import numpy as np
import pandas as pd
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


# Inclusion filter: matches the final §7 eval used in compare_predictions_to_plots.py
INCLUSION_FIELD_COL = 'canopy_cover_field'
INCLUSION_COVERAGE_COL = 'canopy_cover_coverage_fraction'
INCLUSION_COVERAGE_THRESHOLD = 0.99

# Plot footprint geometry
FIELD_PLOT_RADIUS_M = 11.35              # 0.1-acre circular plot
TILE_OVERLAP_BUFFER_M = 2.0              # extra buffer when searching for tiles
SEARCH_RADIUS_M = FIELD_PLOT_RADIUS_M + TILE_OVERLAP_BUFFER_M  # 13.35 m

# Field columns from baseline_comparison_results.csv (already in display units)
FIELD_COLUMNS = {
    'field_max_height': 'max_height_field',
    'field_canopy_cover': 'canopy_cover_field',
    'field_midstory_density': 'midstory_density_field',
    'field_understory_density': 'understory_density_field',
}


def load_inclusion_filtered_plots(csv_path: Path) -> pd.DataFrame:
    """Load baseline comparison CSV and apply the §7 inclusion filter."""
    df = pd.read_csv(csv_path)
    n_total = len(df)

    if INCLUSION_FIELD_COL not in df.columns:
        raise KeyError(
            f"Required column '{INCLUSION_FIELD_COL}' not in {csv_path}. "
            f"Columns: {list(df.columns)[:20]}"
        )
    if INCLUSION_COVERAGE_COL not in df.columns:
        raise KeyError(
            f"Required column '{INCLUSION_COVERAGE_COL}' not in {csv_path}."
        )

    valid_mask = (
        df[INCLUSION_FIELD_COL].notna()
        & (df[INCLUSION_COVERAGE_COL] >= INCLUSION_COVERAGE_THRESHOLD)
    )
    df = df[valid_mask].copy()

    site_counts = df['site_name'].value_counts().to_dict()
    logger.info(
        f"Inclusion filter: {len(df)}/{n_total} plots pass "
        f"({INCLUSION_FIELD_COL}.notna() AND "
        f"{INCLUSION_COVERAGE_COL} >= {INCLUSION_COVERAGE_THRESHOLD})"
    )
    logger.info(f"Per-site inclusion counts: {site_counts}")
    return df


def stratified_sample_per_site(
    df: pd.DataFrame, per_site: int, seed: int
) -> pd.DataFrame:
    """Sample `per_site` plots per site, stratified by canopy cover quantile."""
    rng = np.random.default_rng(seed)
    selected_rows = []

    for site_name in sorted(df['site_name'].unique()):
        site_df = df[df['site_name'] == site_name].copy()
        n_avail = len(site_df)

        if n_avail <= per_site:
            logger.info(
                f"  {site_name}: only {n_avail} plots available "
                f"(requested {per_site}) — taking all"
            )
            selected_rows.append(site_df)
            continue

        # Stratify by canopy cover into 3 buckets (low/mid/high)
        n_bins = min(3, per_site)
        site_df['_strata'] = pd.qcut(
            site_df[INCLUSION_FIELD_COL],
            q=n_bins,
            labels=False,
            duplicates='drop',
        )
        strata_avail = sorted(site_df['_strata'].dropna().unique().tolist())

        site_picked: List[int] = []
        for i in range(per_site):
            stratum_label = strata_avail[i % len(strata_avail)]
            pool = site_df[
                (site_df['_strata'] == stratum_label)
                & (~site_df.index.isin(site_picked))
            ]
            if len(pool) == 0:
                # Stratum exhausted — pull any unpicked plot
                pool = site_df[~site_df.index.isin(site_picked)]
                if len(pool) == 0:
                    break
            pick_idx = pool.index[int(rng.integers(0, len(pool)))]
            site_picked.append(pick_idx)

        site_selected = site_df.loc[site_picked].drop(columns='_strata')
        selected_rows.append(site_selected)
        logger.info(
            f"  {site_name}: selected {len(site_selected)}/{n_avail} plots"
        )

    result = pd.concat(selected_rows, axis=0).reset_index(drop=True)
    logger.info(f"Total selected plots: {len(result)}")
    return result


def find_overlapping_tile_indices(
    tiles: List[dict],
    plot_x: float,
    plot_y: float,
    search_radius_m: float = SEARCH_RADIUS_M,
) -> List[int]:
    """Return indices of tiles whose bbox intersects the plot search box."""
    sxmin = plot_x - search_radius_m
    sxmax = plot_x + search_radius_m
    symin = plot_y - search_radius_m
    symax = plot_y + search_radius_m

    overlapping: List[int] = []
    for i, tile in enumerate(tiles):
        bbox = tile['bbox']
        if torch.is_tensor(bbox):
            xmin, ymin, xmax, ymax = bbox.tolist()
        else:
            xmin, ymin, xmax, ymax = bbox

        # 2D box-box intersection test
        if (
            xmax >= sxmin
            and xmin <= sxmax
            and ymax >= symin
            and ymin <= symax
        ):
            overlapping.append(i)

    return overlapping


def union_bbox_covers_circle(
    tile_idxs: List[int],
    tiles: List[dict],
    plot_x: float,
    plot_y: float,
    radius_m: float = FIELD_PLOT_RADIUS_M,
) -> bool:
    """Cheap pre-check: union of tile bboxes contains the circle's bbox."""
    if not tile_idxs:
        return False

    boxes = []
    for idx in tile_idxs:
        bbox = tiles[idx]['bbox']
        if torch.is_tensor(bbox):
            boxes.append(bbox.tolist())
        else:
            boxes.append(list(bbox))

    union_xmin = min(b[0] for b in boxes)
    union_xmax = max(b[2] for b in boxes)
    union_ymin = min(b[1] for b in boxes)
    union_ymax = max(b[3] for b in boxes)

    return (
        union_xmin <= plot_x - radius_m
        and union_xmax >= plot_x + radius_m
        and union_ymin <= plot_y - radius_m
        and union_ymax >= plot_y + radius_m
    )


def build_plot_record(plot_row: pd.Series, tile_ids: List[str]) -> dict:
    """Convert a CSV row + assigned tile IDs into the metadata JSON record."""
    record = {
        'plot_id': int(plot_row['plot_id']),
        'site': str(plot_row['site_name']),
        'plot_easting': float(plot_row['plot_x']),
        'plot_northing': float(plot_row['plot_y']),
        'tile_ids': tile_ids,
    }
    for field_key, csv_col in FIELD_COLUMNS.items():
        val = plot_row.get(csv_col)
        record[field_key] = float(val) if pd.notna(val) else None
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-site', type=int, default=3,
                        help='Plots per site (default: 3)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument(
        '--baseline-csv',
        type=str,
        default='data/processed/veg_structure_baseline/comparison/baseline_comparison_results.csv',
        help='Baseline comparison CSV with inclusion filter columns',
    )
    parser.add_argument(
        '--inference-tiles',
        type=str,
        default='data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt',
        help='Pre-computed forest plot tile file (~20 GB)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/processed/forest_plot_data/ood_validation',
        help='Output directory',
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load + filter plots
    logger.info(f"Loading baseline CSV: {args.baseline_csv}")
    plots_df = load_inclusion_filtered_plots(Path(args.baseline_csv))

    # 2. Stratified sample
    logger.info(
        f"Stratified sampling {args.per_site} plots per site (seed={args.seed})"
    )
    selected_df = stratified_sample_per_site(plots_df, args.per_site, args.seed)

    # 3. Load inference tiles (slow on the full 20 GB file)
    logger.info(f"Loading inference tiles: {args.inference_tiles}")
    t0 = time.time()
    tiles = torch.load(args.inference_tiles, map_location='cpu', weights_only=False)
    logger.info(f"Loaded {len(tiles)} tiles in {time.time() - t0:.1f}s")
    if not isinstance(tiles, list):
        raise TypeError(
            f"Expected list of tile dicts, got {type(tiles).__name__}"
        )

    # 4. For each selected plot, find overlapping tiles and build records
    selected_tile_idxs: Set[int] = set()
    plot_records: List[dict] = []
    coverage_gaps: List[int] = []

    for _, plot_row in selected_df.iterrows():
        plot_id = int(plot_row['plot_id'])
        plot_x = float(plot_row['plot_x'])
        plot_y = float(plot_row['plot_y'])
        site = str(plot_row['site_name'])

        tile_idxs = find_overlapping_tile_indices(tiles, plot_x, plot_y)
        if not tile_idxs:
            logger.warning(
                f"  plot_id={plot_id} site={site}: NO overlapping tiles "
                f"(at {plot_x:.1f}, {plot_y:.1f}) — skipping"
            )
            continue

        coverage_ok = union_bbox_covers_circle(tile_idxs, tiles, plot_x, plot_y)
        if not coverage_ok:
            coverage_gaps.append(plot_id)

        marker = 'OK ' if coverage_ok else 'GAP'
        plot_tile_ids = [
            tiles[idx].get('tile_id', f'tile_{idx}') for idx in tile_idxs
        ]
        plot_records.append(build_plot_record(plot_row, plot_tile_ids))
        selected_tile_idxs.update(tile_idxs)

        logger.info(
            f"  plot_id={plot_id} site={site}: {len(tile_idxs)} tiles [{marker}]"
        )

    if coverage_gaps:
        logger.warning(
            f"{len(coverage_gaps)} plots have a tile-bbox-union gap: "
            f"{coverage_gaps}"
        )

    # 5. Save artifacts
    logger.info(f"\nDeduplicated tile count: {len(selected_tile_idxs)}")
    sorted_idxs = sorted(selected_tile_idxs)
    selected_tiles: List[dict] = []
    for idx in sorted_idxs:
        tile = tiles[idx]
        if 'tile_id' not in tile:
            tile = dict(tile)
            tile['tile_id'] = f'tile_{idx}'
        selected_tiles.append(tile)

    tiles_path = output_dir / 'ood_validation_tiles.pt'
    logger.info(f"Saving tile list -> {tiles_path}")
    torch.save(selected_tiles, tiles_path)

    metadata_path = output_dir / 'ood_validation_metadata.json'
    logger.info(f"Saving plot metadata -> {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(plot_records, f, indent=2)

    plot_ids_path = output_dir / 'ood_validation_plot_ids.txt'
    logger.info(f"Saving plot IDs -> {plot_ids_path}")
    with open(plot_ids_path, 'w') as f:
        for record in plot_records:
            f.write(f"{record['plot_id']}\n")

    config_path = output_dir / 'ood_validation_config.json'
    config_data = {
        'per_site': args.per_site,
        'seed': args.seed,
        'build_timestamp': datetime.now().isoformat(),
        'n_plots': len(plot_records),
        'n_tiles': len(selected_tiles),
        'baseline_csv': args.baseline_csv,
        'inference_tiles': args.inference_tiles,
        'inclusion_filter': {
            'field_column': INCLUSION_FIELD_COL,
            'coverage_column': INCLUSION_COVERAGE_COL,
            'coverage_threshold': INCLUSION_COVERAGE_THRESHOLD,
        },
        'search_radius_m': SEARCH_RADIUS_M,
        'field_plot_radius_m': FIELD_PLOT_RADIUS_M,
        'coverage_gap_plot_ids': coverage_gaps,
    }
    with open(config_path, 'w') as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"Saving build config -> {config_path}")

    logger.info("\n" + "=" * 60)
    logger.info("BUILD COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Plots:    {len(plot_records)}")
    logger.info(f"Tiles:    {len(selected_tiles)}")
    logger.info(f"Output:   {output_dir}")


if __name__ == '__main__':
    main()
