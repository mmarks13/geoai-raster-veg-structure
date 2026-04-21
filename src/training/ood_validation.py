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
import math
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
from torch.utils.tensorboard import SummaryWriter

from src.evaluation.band_config import BandConfig, load_band_config
from src.evaluation.build_prediction_rasters import build_site_rasters
from src.evaluation.compare_predictions_to_plots import (
    compare_predictions_to_field,
    create_plot_footprints,
    extract_raster_values_at_footprint,
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
) -> tuple[Dict[str, np.ndarray], Optional[Dict[str, np.ndarray]]]:
    """
    Forward-pass all OOD tiles in mixed-modality batches.

    Returns:
        predictions_dict: tile_id → np.ndarray[n_bands, 5, 5] in physical units.
        variances_dict:   tile_id → np.ndarray[n_bands, 5, 5] of σ² in z-score
                          (target-normalized) space, or None if the model does
                          not emit log-variance.
    """
    predictions_dict: Dict[str, np.ndarray] = {}
    variances_dict: Dict[str, np.ndarray] = {}
    has_variance = False
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

        # Heteroscedastic models return (mean, log_var)
        if isinstance(output, tuple):
            predictions = output[0]
            log_var = output[1]
            has_variance = True
        else:
            predictions = output
            log_var = None

        # Cast back to float32 for denormalization (denorm reads dtype from preds)
        predictions = predictions.detach().float().cpu()

        # z-score → physical units for the mean predictions
        predictions_denorm = denormalize_predictions(
            predictions, fuel_stats, target_band_indices
        )

        # Keep σ² in z-score (target-normalized) space. Calibration ratios are
        # scale-invariant, so we intentionally avoid denormalizing here — it
        # sidesteps the log-TFL inverse-transform issue and keeps the downstream
        # math simple.
        if log_var is not None:
            variance_z = torch.exp(log_var.detach().float().cpu())

        for i, tile_id in enumerate(batch['tile_ids']):
            predictions_dict[tile_id] = predictions_denorm[i].numpy()
            if log_var is not None:
                variances_dict[tile_id] = variance_z[i].numpy()

    return predictions_dict, (variances_dict if has_variance else None)


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


def _trimmed_spearman_from_residuals(
    field_vals: np.ndarray,
    pred_vals: np.ndarray,
    n_trim: int = 2,
) -> float:
    """Compute Spearman rho after dropping the worst residual outliers."""
    if len(field_vals) < 3:
        return float('nan')

    trimmed_field, trimmed_pred = _trim_worst_residuals(
        field_vals, pred_vals, n_trim=n_trim
    )
    if len(trimmed_field) < 3:
        return float('nan')

    from scipy.stats import spearmanr

    rho_val, _ = spearmanr(trimmed_field, trimmed_pred)
    return float(rho_val)


def _trim_worst_residuals(
    field_vals: np.ndarray,
    pred_vals: np.ndarray,
    n_trim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop the largest absolute residuals from aligned field/pred arrays."""
    if len(field_vals) != len(pred_vals):
        raise ValueError("field_vals and pred_vals must have the same length")

    if n_trim <= 0:
        return field_vals, pred_vals
    if len(field_vals) <= n_trim:
        return np.array([]), np.array([])

    abs_residuals = np.abs(pred_vals - field_vals)
    keep_indices = np.argsort(abs_residuals)[: len(abs_residuals) - n_trim]
    return field_vals[keep_indices], pred_vals[keep_indices]


def _compute_robust_ood_stats(
    merged_df: pd.DataFrame,
    band_config: BandConfig,
    fuel_stats: dict,
    target_band_indices: List[int],
    coverage_threshold: float = 0.99,
    n_trim: int = 2,
) -> tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
    """Compute OOD-only robust metrics from extracted plot predictions."""
    overall_stats: Dict[str, Dict[str, float]] = {}
    per_site_rows: List[Dict[str, float]] = []
    mapped_bands = band_config.get_bands_with_field_mapping()

    for band in mapped_bands:
        source_band_idx = target_band_indices[band.output_index]
        train_std_display_units = (
            float(fuel_stats[f'band_{source_band_idx}_std']) *
            band.unit_conversion_factor
        )
        field_col = f'{band.name}_field'
        pred_col = f'{band.name}_pred'
        coverage_col = f'{band.name}_coverage_fraction'
        valid_mask = (
            merged_df[field_col].notna() &
            merged_df[pred_col].notna() &
            (merged_df[coverage_col] >= coverage_threshold)
        )

        field_vals = pd.to_numeric(
            merged_df.loc[valid_mask, field_col], errors='coerce'
        ).to_numpy()
        pred_vals = pd.to_numeric(
            merged_df.loc[valid_mask, pred_col], errors='coerce'
        ).to_numpy()
        finite_mask = np.isfinite(field_vals) & np.isfinite(pred_vals)
        field_vals = field_vals[finite_mask]
        pred_vals = pred_vals[finite_mask]
        if len(field_vals) < 3:
            continue

        residuals = pred_vals - field_vals
        abs_residuals = np.abs(residuals)
        trimmed_field_vals, trimmed_pred_vals = _trim_worst_residuals(
            field_vals, pred_vals, n_trim=n_trim
        )
        trimmed_abs_residuals = np.abs(trimmed_pred_vals - trimmed_field_vals)
        field_std = float(np.std(field_vals))
        medae = float(np.median(abs_residuals))
        overall_stats[band.name] = {
            'n': int(len(field_vals)),
            'medae': medae,
            'trimmed_mae': (
                float(np.mean(trimmed_abs_residuals))
                if len(trimmed_abs_residuals) > 0 else float('nan')
            ),
            'median_bias': float(np.median(residuals)),
            'field_mean': float(np.mean(field_vals)),
            'field_std': field_std,
            'pred_mean': float(np.mean(pred_vals)),
            'pred_std': float(np.std(pred_vals)),
            'spearman_rho': _trimmed_spearman_from_residuals(
                field_vals, pred_vals, n_trim=n_trim
            ),
            'normalized_medae': (
                medae / field_std if field_std > 1e-8 else float('nan')
            ),
            'train_std_display_units': train_std_display_units,
            'trimmed_standardized_mae': (
                float(np.mean(trimmed_abs_residuals)) / train_std_display_units
                if len(trimmed_abs_residuals) > 0 and train_std_display_units > 1e-8
                else float('nan')
            ),
            'display_name': band.display_name,
            'units': band.display_units,
        }

    for site_name, site_df in merged_df.groupby('site_name'):
        row: Dict[str, float] = {'site_name': site_name}
        for band in mapped_bands:
            field_col = f'{band.name}_field'
            pred_col = f'{band.name}_pred'
            coverage_col = f'{band.name}_coverage_fraction'
            valid_mask = (
                site_df[field_col].notna() &
                site_df[pred_col].notna() &
                (site_df[coverage_col] >= coverage_threshold)
            )
            field_vals = pd.to_numeric(
                site_df.loc[valid_mask, field_col], errors='coerce'
            ).to_numpy()
            pred_vals = pd.to_numeric(
                site_df.loc[valid_mask, pred_col], errors='coerce'
            ).to_numpy()
            finite_mask = np.isfinite(field_vals) & np.isfinite(pred_vals)
            field_vals = field_vals[finite_mask]
            pred_vals = pred_vals[finite_mask]
            if len(field_vals) == 0:
                continue
            row[f'{band.name}_medae'] = float(np.median(np.abs(pred_vals - field_vals)))
        per_site_rows.append(row)

    return overall_stats, pd.DataFrame(per_site_rows)


def _extract_per_plot_variance_z(
    variance_site_rasters: Dict[str, str],
    field_footprints: gpd.GeoDataFrame,
    band_config: BandConfig,
    coverage_threshold: float = 0.99,
) -> pd.DataFrame:
    """
    Extract footprint-weighted mean σ² (z-score space) per plot, per band.

    Returns a DataFrame keyed by plot_id (index) with columns
    '<band>_var_z' and '<band>_var_coverage' for each mapped band.
    """
    rows: List[Dict[str, float]] = []
    for idx, row in field_footprints.iterrows():
        plot_id = row.get('PlotID', row.get('plot_id', idx))
        site_name = row.get('Site', row.get('site_name', 'unknown'))
        footprint = row.geometry
        if site_name not in variance_site_rasters:
            continue

        raster_path = variance_site_rasters[site_name]
        out: Dict[str, float] = {
            'plot_id': plot_id,
            'site_name': site_name,
        }
        for band in band_config.get_bands_with_field_mapping():
            extraction = extract_raster_values_at_footprint(
                raster_path, footprint, band.output_index, 'mean'
            )
            var_value = extraction['weighted_mean']
            coverage = extraction['coverage_fraction']
            out[f'{band.name}_var_z'] = (
                float(var_value) if not np.isnan(var_value) else float('nan')
            )
            out[f'{band.name}_var_coverage'] = float(coverage)
        rows.append(out)

    return pd.DataFrame(rows)


def _compute_ood_calibration_metrics(
    merged_df: pd.DataFrame,
    plot_variances_df: pd.DataFrame,
    band_config: BandConfig,
    fuel_stats: dict,
    target_band_indices: List[int],
    coverage_threshold: float = 0.99,
) -> Dict[str, float]:
    """
    Compute per-band / per-site / overall calibration ratios on OOD plots.

    ratio = mean_plots((y - μ)² / σ²) equivalent (using mean(sq_err)/mean(σ²)
    so a few low-σ² plots don't dominate). Everything is computed in z-score
    (target-normalized) space so ratios are comparable across bands.

    Skips log-transformed bands (e.g. TFL with use_log_tfl=True) — z-score
    calibration isn't well-defined there without a delta-method correction.

    Returns a flat dict of metrics with keys:
        ood_<band>_calibration_ratio
        ood_<band>_mean_sq_err_z
        ood_<band>_mean_pred_var_z
        ood_overall_calibration_ratio
        ood_per_site/<site>/<band>_calibration_ratio
    """
    metrics: Dict[str, float] = {}
    mapped_bands = band_config.get_bands_with_field_mapping()

    use_log_tfl = fuel_stats.get('use_log_tfl', False)
    tfl_band_index = fuel_stats.get('tfl_band_index', 15)

    joined = merged_df.merge(
        plot_variances_df, on=['plot_id', 'site_name'], how='inner'
    )

    per_band_sq_err_all: List[float] = []
    per_band_var_all: List[float] = []

    for band in mapped_bands:
        source_band_idx = target_band_indices[band.output_index]
        if use_log_tfl and source_band_idx == tfl_band_index:
            continue

        band_mean = float(fuel_stats[f'band_{source_band_idx}_mean'])
        band_std = float(fuel_stats[f'band_{source_band_idx}_std'])
        if band_std <= 1e-8:
            continue

        field_col = f'{band.name}_field'
        pred_col = f'{band.name}_pred'
        cov_col = f'{band.name}_coverage_fraction'
        var_col = f'{band.name}_var_z'
        var_cov_col = f'{band.name}_var_coverage'

        required_cols = [field_col, pred_col, cov_col, var_col, var_cov_col]
        if not all(c in joined.columns for c in required_cols):
            continue

        valid = (
            joined[field_col].notna()
            & joined[pred_col].notna()
            & joined[var_col].notna()
            & (joined[cov_col] >= coverage_threshold)
            & (joined[var_cov_col] >= coverage_threshold)
            & (joined[var_col] > 0)
        )
        sub = joined.loc[valid, ['site_name', field_col, pred_col, var_col]]
        if len(sub) < 3:
            continue

        factor = band.unit_conversion_factor
        # display → model units → z-score
        field_z = (sub[field_col].to_numpy() / factor - band_mean) / band_std
        pred_z = (sub[pred_col].to_numpy() / factor - band_mean) / band_std
        var_z = sub[var_col].to_numpy()

        sq_err_z = (field_z - pred_z) ** 2
        finite = np.isfinite(sq_err_z) & np.isfinite(var_z) & (var_z > 0)
        if finite.sum() < 3:
            continue

        mean_sq_err = float(np.mean(sq_err_z[finite]))
        mean_var = float(np.mean(var_z[finite]))
        ratio = mean_sq_err / mean_var if mean_var > 1e-12 else float('nan')

        metrics[f'ood_{band.name}_calibration_ratio'] = ratio
        metrics[f'ood_{band.name}_mean_sq_err_z'] = mean_sq_err
        metrics[f'ood_{band.name}_mean_pred_var_z'] = mean_var
        metrics[f'ood_{band.name}_calibration_n_plots'] = float(int(finite.sum()))

        per_band_sq_err_all.append(mean_sq_err)
        per_band_var_all.append(mean_var)

        # Per-site
        site_names = sub['site_name'].to_numpy()[finite]
        sq_err_finite = sq_err_z[finite]
        var_finite = var_z[finite]
        for site in np.unique(site_names):
            site_mask = (site_names == site)
            if site_mask.sum() < 3:
                continue
            site_mean_var = float(np.mean(var_finite[site_mask]))
            if site_mean_var <= 1e-12:
                continue
            metrics[f'ood_per_site/{site}/{band.name}_calibration_ratio'] = (
                float(np.mean(sq_err_finite[site_mask])) / site_mean_var
            )

    if per_band_sq_err_all and per_band_var_all:
        # Overall: mean of per-band ratios (equal weighting across bands)
        per_band_ratios = [
            s / v for s, v in zip(per_band_sq_err_all, per_band_var_all) if v > 1e-12
        ]
        if per_band_ratios:
            metrics['ood_overall_calibration_ratio'] = float(np.mean(per_band_ratios))

    return metrics


def _flatten_stats_to_metrics(
    overall_stats: Dict, per_site_df: pd.DataFrame, band_config: BandConfig
) -> Dict[str, float]:
    """
    Flatten compute_statistics output + per-site DataFrame into a flat metric dict
    suitable for TensorBoard / early-stopping consumption.

    Output keys:
      ood_overall_mean_tsmae                      - mean of mapped-band trimmed standardized MAEs
      ood_overall_median_spearman_rho             - median of mapped-band trimmed Spearman rhos
      ood_<band>_medae / tsmae / spearman_rho / bias
                                                  - per-band overall metrics
      ood_per_site/<site>/<band>_medae            - per-band per-site metrics
    """
    metrics: Dict[str, float] = {}

    mapped_band_names = [b.name for b in band_config.get_bands_with_field_mapping()]

    band_trimmed_standardized_maes = []
    band_spearman_rhos = []
    for band_name in mapped_band_names:
        band_stats = overall_stats.get(band_name)
        if band_stats is None:
            continue
        metrics[f'ood_{band_name}_medae'] = float(band_stats['medae'])
        metrics[f'ood_{band_name}_tsmae'] = float(
            band_stats['trimmed_standardized_mae']
        )
        metrics[f'ood_{band_name}_spearman_rho'] = float(band_stats['spearman_rho'])
        metrics[f'ood_{band_name}_bias'] = float(band_stats['median_bias'])
        trimmed_standardized_mae = float(band_stats['trimmed_standardized_mae'])
        if np.isfinite(trimmed_standardized_mae):
            band_trimmed_standardized_maes.append(trimmed_standardized_mae)
        spearman_rho = float(band_stats['spearman_rho'])
        if np.isfinite(spearman_rho):
            band_spearman_rhos.append(spearman_rho)

    if band_trimmed_standardized_maes:
        metrics['ood_overall_mean_tsmae'] = float(
            np.mean(band_trimmed_standardized_maes)
        )
    else:
        metrics['ood_overall_mean_tsmae'] = float('nan')

    if band_spearman_rhos:
        metrics['ood_overall_median_spearman_rho'] = float(np.median(band_spearman_rhos))
    else:
        metrics['ood_overall_median_spearman_rho'] = float('nan')

    # Per-site flattening (skip rows where MAE is NaN due to small-n sites)
    for _, site_row in per_site_df.iterrows():
        site = site_row['site_name']
        for band_name in mapped_band_names:
            medae_key = f'{band_name}_medae'
            if medae_key in site_row and pd.notna(site_row[medae_key]):
                metrics[f'ood_per_site/{site}/{band_name}_medae'] = float(
                    site_row[medae_key]
                )

    return metrics


def _log_ood_scatterplots(
    writer: SummaryWriter,
    epoch: int,
    merged_df: pd.DataFrame,
    band_config: BandConfig,
    coverage_threshold: float = 0.99,
) -> None:
    """Log per-band OOD field-vs-prediction scatterplots to TensorBoard."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping OOD scatterplot logging")
        return

    site_names = sorted(merged_df['site_name'].dropna().unique())
    cmap = plt.get_cmap('tab10')
    site_colors = {
        site: cmap(i % cmap.N)
        for i, site in enumerate(site_names)
    }

    for band in band_config.get_bands_with_field_mapping():
        field_col = f'{band.name}_field'
        pred_col = f'{band.name}_pred'
        coverage_col = f'{band.name}_coverage_fraction'

        valid_mask = (
            merged_df[field_col].notna() &
            merged_df[pred_col].notna() &
            (merged_df[coverage_col] >= coverage_threshold)
        )
        plot_df = merged_df.loc[valid_mask, ['site_name', field_col, pred_col]]
        if plot_df.empty:
            continue

        field_vals = pd.to_numeric(plot_df[field_col], errors='coerce').to_numpy()
        pred_vals = pd.to_numeric(plot_df[pred_col], errors='coerce').to_numpy()

        finite_mask = np.isfinite(field_vals) & np.isfinite(pred_vals)
        plot_df = plot_df.loc[finite_mask].copy()
        if plot_df.empty:
            continue

        field_vals = field_vals[finite_mask]
        pred_vals = pred_vals[finite_mask]
        vmin = float(min(np.min(field_vals), np.min(pred_vals)))
        vmax = float(max(np.max(field_vals), np.max(pred_vals)))
        if math.isclose(vmin, vmax):
            pad = 1.0
            vmin -= pad
            vmax += pad

        fig, ax = plt.subplots(figsize=(6, 6))
        for site in site_names:
            site_df = plot_df[plot_df['site_name'] == site]
            if site_df.empty:
                continue
            ax.scatter(
                site_df[field_col],
                site_df[pred_col],
                label=site,
                s=48,
                alpha=0.9,
                color=site_colors[site],
                edgecolors='white',
                linewidths=0.5,
            )

        ax.plot([vmin, vmax], [vmin, vmax], linestyle='--', color='0.4', linewidth=1.0)
        ax.set_xlim(vmin, vmax)
        ax.set_ylim(vmin, vmax)
        ax.set_xlabel(f'Field {band.display_name} ({band.display_units})')
        ax.set_ylabel(f'Predicted {band.display_name} ({band.display_units})')
        ax.set_title(f'OOD {band.display_name}: field vs prediction')
        ax.grid(alpha=0.2)
        ax.legend(title='Site', fontsize=8, title_fontsize=9, loc='best')
        fig.tight_layout()
        writer.add_figure(f'OOD_Scatter/{band.name}', fig, global_step=epoch)
        plt.close(fig)


@torch.no_grad()
def evaluate_ood(
    model,
    ood_set: OODValidationSet,
    device: torch.device,
    config,
    band_config: BandConfig,
    fuel_stats: Optional[dict] = None,
    writer: Optional[SummaryWriter] = None,
    epoch: Optional[int] = None,
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
        # 1. Forward pass on all OOD tiles → (predictions_dict, variances_dict)
        predictions_dict, variances_dict = _run_ood_forward_pass(
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
                overall_stats, per_site_df = _compute_robust_ood_stats(
                    merged_df=merged_df,
                    band_config=band_config,
                    fuel_stats=fuel_stats,
                    target_band_indices=target_band_indices,
                    coverage_threshold=0.99,
                )

                # Optional: if the model emitted variance, build a parallel
                # set of σ² rasters (z-score space) and extract plot-level
                # variance for calibration metrics.
                calibration_metrics: Dict[str, float] = {}
                if variances_dict is not None:
                    variance_dir = tmpdir_path / 'variance'
                    variance_dir.mkdir(exist_ok=True)
                    build_site_rasters(
                        predictions_dict=variances_dict,
                        tile_metadata_csv=str(tile_meta_csv),
                        site_column='site_name',
                        crs='EPSG:32611',
                        output_dir=variance_dir,
                        raster_suffix='_variance_raster',
                    )
                    variance_site_rasters = {
                        p.stem.replace('_variance_raster', ''): str(p)
                        for p in variance_dir.glob('*_variance_raster.tif')
                    }
                    if variance_site_rasters:
                        plot_variances_df = _extract_per_plot_variance_z(
                            variance_site_rasters,
                            field_footprints,
                            band_config,
                            coverage_threshold=0.99,
                        )
                        calibration_metrics = _compute_ood_calibration_metrics(
                            merged_df=merged_df,
                            plot_variances_df=plot_variances_df,
                            band_config=band_config,
                            fuel_stats=fuel_stats,
                            target_band_indices=target_band_indices,
                            coverage_threshold=0.99,
                        )

            if writer is not None and epoch is not None:
                _log_ood_scatterplots(
                    writer=writer,
                    epoch=epoch,
                    merged_df=merged_df,
                    band_config=band_config,
                    coverage_threshold=0.99,
                )

        flat = _flatten_stats_to_metrics(overall_stats, per_site_df, band_config)
        flat.update(calibration_metrics)
        return flat

    finally:
        if was_training:
            model.train()
        else:
            model.eval()
