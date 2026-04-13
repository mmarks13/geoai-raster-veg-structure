"""
In-training OOD forest-plot validation hook.

Runs a tiny forward pass on a fixed subset of forest plot tiles every N epochs,
stitches the per-tile predictions into per-site rasters using the existing
``build_site_rasters`` helper, and reuses ``compare_predictions_to_field`` /
``compute_statistics`` from the §7 evaluation path so the in-training OOD metric
is byte-identical to the final reported OOD metric.

The cleanest reuse pattern is to write the per-site rasters into a
``tempfile.TemporaryDirectory()`` and then call the existing comparison helpers
with that directory — every spatial convention is then inherited automatically.

Public API:
    OODValidationSet  - holds tiles + per-plot metadata loaded from disk
    evaluate_ood      - runs forward pass + extraction + stats; returns flat metrics
"""

import json
import logging
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from shapely.geometry import Point
from torch.amp import autocast

from src.evaluation.band_config import BandConfig, load_band_config
from src.evaluation.build_prediction_rasters import build_site_rasters
from src.evaluation.compare_predictions_to_plots import (
    compare_predictions_to_field,
    compute_statistics,
    create_plot_footprints,
    load_site_rasters,
)
from src.evaluation.raster_inference import (
    collate_inference_batch,
    denormalize_predictions,
)

logger = logging.getLogger(__name__)


# Maps the band config field column → metadata.json field key
# (used when constructing the in-memory field GeoDataFrame for the comparison
# step). Field columns must match the band config's `field_column` strings —
# `compare_predictions_to_field` looks them up by exact name.
FIELD_COLUMN_FROM_METADATA = {
    'CrownHt': 'field_max_height',
    'TreeCover': 'field_canopy_cover',
    'ShrubCover': 'field_midstory_density',
    'HerbCover': 'field_understory_density',
}


class OODValidationSet:
    """Loads OOD validation tiles + per-plot metadata once per training run."""

    def __init__(self, tiles_path: str, metadata_path: str):
        self.tiles_path = Path(tiles_path)
        self.metadata_path = Path(metadata_path)

        if not self.tiles_path.exists():
            raise FileNotFoundError(f"OOD tiles file not found: {self.tiles_path}")
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"OOD metadata file not found: {self.metadata_path}"
            )

        logger.info(f"Loading OOD tiles from {self.tiles_path}")
        self.tiles: List[dict] = torch.load(
            self.tiles_path, map_location='cpu', weights_only=False
        )
        if not isinstance(self.tiles, list):
            raise TypeError(
                f"Expected list of tile dicts, got {type(self.tiles).__name__}"
            )

        logger.info(f"Loading OOD plot metadata from {self.metadata_path}")
        with open(self.metadata_path) as f:
            self.plots: List[dict] = json.load(f)

        # Sanity: every plot's tile_ids must exist in self.tiles
        tile_id_set = {t.get('tile_id') for t in self.tiles}
        missing_tiles = []
        for plot in self.plots:
            for tid in plot['tile_ids']:
                if tid not in tile_id_set:
                    missing_tiles.append((plot['plot_id'], tid))
        if missing_tiles:
            raise ValueError(
                f"OOD metadata references {len(missing_tiles)} tile IDs that are "
                f"not in the tiles file. First 5: {missing_tiles[:5]}"
            )

        # Site-level groupings: union of tile IDs used by any plot in that site
        self.site_to_tile_ids: Dict[str, set] = {}
        for plot in self.plots:
            site = plot['site']
            self.site_to_tile_ids.setdefault(site, set()).update(plot['tile_ids'])

        # Tile ID → tile dict for fast lookup
        self.tile_by_id: Dict[str, dict] = {
            t['tile_id']: t for t in self.tiles if 'tile_id' in t
        }
        # Tile ID → site (used to write the metadata CSV for build_site_rasters)
        self.tile_id_to_site: Dict[str, str] = {}
        for site, tile_ids in self.site_to_tile_ids.items():
            for tid in tile_ids:
                # If a tile is used by multiple sites (shouldn't happen but be
                # defensive), take the first one — sites are hundreds of miles
                # apart so this is a build-script bug if it occurs.
                if tid not in self.tile_id_to_site:
                    self.tile_id_to_site[tid] = site

        n_sites = len(self.site_to_tile_ids)
        n_unique_tiles = len(self.tile_by_id)
        logger.info(
            f"OOD validation set: {len(self.plots)} plots, {n_unique_tiles} unique "
            f"tiles, {n_sites} sites"
        )
        for site in sorted(self.site_to_tile_ids.keys()):
            n_plots = sum(1 for p in self.plots if p['site'] == site)
            n_tiles = len(self.site_to_tile_ids[site])
            logger.info(f"  {site}: {n_plots} plots, {n_tiles} tiles")


@contextmanager
def _logger_at(logger_obj: logging.Logger, level: int):
    """Temporarily raise a logger's level (used to silence chatty helpers)."""
    prev = logger_obj.level
    logger_obj.setLevel(level)
    try:
        yield
    finally:
        logger_obj.setLevel(prev)


def _move_naip_to_device(naip_dict: Optional[dict], device: torch.device) -> Optional[dict]:
    if naip_dict is None or 'images' not in naip_dict:
        return naip_dict
    return {
        'images': naip_dict['images'].to(device),
        'img_bbox': naip_dict.get('img_bbox'),
        'relative_dates': naip_dict.get('relative_dates'),
    }


def _move_uavsar_to_device(
    uavsar_dict: Optional[dict], device: torch.device
) -> Optional[dict]:
    if uavsar_dict is None or 'images' not in uavsar_dict:
        return uavsar_dict
    return {
        'images': uavsar_dict['images'].to(device),
        'img_bbox': uavsar_dict.get('img_bbox'),
        'attention_mask': uavsar_dict.get('attention_mask'),
        'relative_dates': uavsar_dict.get('relative_dates'),
    }


def _run_ood_forward_pass(
    model,
    ood_set: OODValidationSet,
    device: torch.device,
    fuel_stats: dict,
    target_band_indices: List[int],
    batch_size: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Forward-pass all OOD tiles in mixed-modality batches.

    Returns dict mapping tile_id → np.ndarray[n_bands, 5, 5] in physical units.
    """
    predictions_dict: Dict[str, np.ndarray] = {}
    tiles = ood_set.tiles

    # Chunk tiles into batches regardless of site or UAVSAR availability —
    # mixed-modality batching exercises the model's native None-handling.
    for batch_start in range(0, len(tiles), batch_size):
        batch_tiles = tiles[batch_start: batch_start + batch_size]
        batch = collate_inference_batch(batch_tiles)

        dep_points = batch['dep_points'].to(device)
        dep_attr = batch['dep_attr'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_indices = batch['batch_indices'].to(device)
        norm_params = batch['norm_params']
        bbox = batch['bbox'].to(device)

        naip = [_move_naip_to_device(n, device) for n in batch['naip']]
        uavsar = [_move_uavsar_to_device(u, device) for u in batch['uavsar']]

        with autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                dep_points=dep_points,
                edge_index=edge_index,
                batch_indices=batch_indices,
                norm_params=norm_params,
                dep_attr=dep_attr,
                naip=naip,
                uavsar=uavsar,
                bbox=bbox,
                debug_logging=False,
            )

        # Heteroscedastic models return (mean, log_var) — discard log_var
        if isinstance(output, tuple):
            predictions = output[0]
        else:
            predictions = output

        # Cast back to float32 for denormalization (denorm reads dtype from preds)
        predictions = predictions.detach().float().cpu()

        # z-score → physical units
        predictions_denorm = denormalize_predictions(
            predictions, fuel_stats, target_band_indices
        )

        for i, tile_id in enumerate(batch['tile_ids']):
            predictions_dict[tile_id] = predictions_denorm[i].numpy()

    return predictions_dict


def _build_tile_metadata_df(
    ood_set: OODValidationSet, predictions_dict: Dict[str, np.ndarray]
) -> pd.DataFrame:
    """Build the tile metadata DataFrame expected by build_site_rasters()."""
    rows = []
    for tile_id, _pred in predictions_dict.items():
        tile = ood_set.tile_by_id[tile_id]
        bbox = tile['bbox']
        if torch.is_tensor(bbox):
            xmin, ymin, xmax, ymax = bbox.tolist()
        else:
            xmin, ymin, xmax, ymax = bbox
        rows.append({
            'tile_id': tile_id,
            'site_name': ood_set.tile_id_to_site[tile_id],
            'bbox_xmin': xmin,
            'bbox_ymin': ymin,
            'bbox_xmax': xmax,
            'bbox_ymax': ymax,
        })
    return pd.DataFrame(rows)


def _build_field_gdf(ood_set: OODValidationSet, crs: str = 'EPSG:32611') -> gpd.GeoDataFrame:
    """
    Build an in-memory GeoDataFrame mirroring the gpkg used by §7 eval.

    Columns mirror what compare_predictions_to_field looks for:
      - geometry (Point in `crs`)
      - Site
      - CrownHt, TreeCover, ShrubCover, HerbCover (field measurements)

    The DataFrame index is set to the integer plot_id so that
    compare_predictions_to_field falls through `row.get('PlotID', row.get('plot_id', idx))`
    to the index — matching the §7 eval, which also falls through to idx because
    the gpkg lacks both 'PlotID' and 'plot_id' columns.
    """
    rows = []
    for plot in ood_set.plots:
        row = {
            '__plot_id_index': int(plot['plot_id']),
            'Site': plot['site'],
            'geometry': Point(plot['plot_easting'], plot['plot_northing']),
        }
        for field_col, meta_key in FIELD_COLUMN_FROM_METADATA.items():
            row[field_col] = plot.get(meta_key)
        rows.append(row)

    df = pd.DataFrame(rows).set_index('__plot_id_index')
    df.index.name = None
    return gpd.GeoDataFrame(df, geometry='geometry', crs=crs)


def _flatten_stats_to_metrics(
    overall_stats: Dict, per_site_df: pd.DataFrame, band_config: BandConfig
) -> Dict[str, float]:
    """
    Flatten compute_statistics output + per-site DataFrame into a flat metric dict
    suitable for TensorBoard / early-stopping consumption.

    Output keys:
      ood_overall_mae                              - mean of mapped-band MAEs
      ood_<band>_mae / r2 / rmse / bias / n        - per-band overall metrics
      ood_per_site/<site>/<band>_mae               - per-band per-site metrics
    """
    metrics: Dict[str, float] = {}

    mapped_band_names = [b.name for b in band_config.get_bands_with_field_mapping()]

    band_maes = []
    for band_name in mapped_band_names:
        band_stats = overall_stats.get(band_name)
        if band_stats is None:
            continue
        metrics[f'ood_{band_name}_mae'] = float(band_stats['mae'])
        metrics[f'ood_{band_name}_rmse'] = float(band_stats['rmse'])
        metrics[f'ood_{band_name}_r2'] = float(band_stats['r_squared'])
        metrics[f'ood_{band_name}_bias'] = float(band_stats['bias'])
        metrics[f'ood_{band_name}_n'] = float(band_stats['n'])
        band_maes.append(float(band_stats['mae']))

    if band_maes:
        metrics['ood_overall_mae'] = float(np.mean(band_maes))
    else:
        metrics['ood_overall_mae'] = float('nan')

    # Per-site flattening (skip rows where MAE is NaN due to small-n sites)
    for _, site_row in per_site_df.iterrows():
        site = site_row['site_name']
        for band_name in mapped_band_names:
            mae_key = f'{band_name}_mae'
            if mae_key in site_row and pd.notna(site_row[mae_key]):
                metrics[f'ood_per_site/{site}/{band_name}_mae'] = float(
                    site_row[mae_key]
                )

    return metrics


@torch.no_grad()
def evaluate_ood(
    model,
    ood_set: OODValidationSet,
    device: torch.device,
    config,
    band_config: BandConfig,
    fuel_stats: Optional[dict] = None,
) -> Dict[str, float]:
    """
    Run a single OOD eval pass and return a flat metric dict.

    Args:
        model: MultimodalRasterPredictor (DDP-unwrapped on rank 0).
        ood_set: Loaded OODValidationSet.
        device: Device to run on (rank-0 GPU).
        config: MultimodalRasterConfig — used to read target_band_indices.
        band_config: Loaded BandConfig (8-band veg structure).
        fuel_stats: Optional pre-loaded normalization stats dict. If None,
            loads from band_config.stats_file.

    Returns:
        Flat dict of OOD metrics (keys all start with 'ood_').
    """
    if fuel_stats is None:
        if band_config.stats_file is None:
            raise ValueError(
                "band_config.stats_file is None — must pass fuel_stats explicitly"
            )
        with open(band_config.stats_file) as f:
            fuel_stats = json.load(f)

    target_band_indices = list(config.target_band_indices)

    was_training = model.training
    model.eval()

    try:
        # 1. Forward pass on all OOD tiles → predictions_dict
        predictions_dict = _run_ood_forward_pass(
            model=model,
            ood_set=ood_set,
            device=device,
            fuel_stats=fuel_stats,
            target_band_indices=target_band_indices,
        )

        # 2. Stitch + extract via existing helpers, using a temp directory
        # so the rasters never touch persistent storage.
        with tempfile.TemporaryDirectory(prefix='ood_eval_') as tmpdir:
            tmpdir_path = Path(tmpdir)

            tile_meta_df = _build_tile_metadata_df(ood_set, predictions_dict)
            tile_meta_csv = tmpdir_path / 'ood_tile_metadata.csv'
            tile_meta_df.to_csv(tile_meta_csv, index=False)

            # Silence the chatty stitcher / extractor / stats loggers during
            # in-training eval. They are useful for the standalone CLI but
            # would dominate training output.
            stitch_logger = logging.getLogger(
                'src.evaluation.build_prediction_rasters'
            )
            compare_logger = logging.getLogger(
                'src.evaluation.compare_predictions_to_plots'
            )
            with _logger_at(stitch_logger, logging.WARNING), _logger_at(
                compare_logger, logging.WARNING
            ):
                build_site_rasters(
                    predictions_dict=predictions_dict,
                    tile_metadata_csv=str(tile_meta_csv),
                    site_column='site_name',
                    crs='EPSG:32611',
                    output_dir=tmpdir_path,
                )

                site_rasters = load_site_rasters(tmpdir_path)
                field_gdf = _build_field_gdf(ood_set, crs='EPSG:32611')
                field_footprints = create_plot_footprints(
                    field_gdf, radius_m=11.35
                )
                merged_df = compare_predictions_to_field(
                    field_footprints,
                    site_rasters,
                    band_config,
                    coverage_threshold=0.99,
                )
                overall_stats = compute_statistics(
                    merged_df, band_config, coverage_threshold=0.99
                )

                from src.evaluation.compare_predictions_to_plots import (
                    compute_per_site_summary,
                )
                per_site_df = compute_per_site_summary(
                    merged_df, band_config, coverage_threshold=0.99
                )

        return _flatten_stats_to_metrics(overall_stats, per_site_df, band_config)

    finally:
        if was_training:
            model.train()
        else:
            model.eval()
