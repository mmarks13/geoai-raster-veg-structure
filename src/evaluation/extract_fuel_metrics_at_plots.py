#!/usr/bin/env python
"""
Extract fuel metrics at forest plot locations from GeoTIFF rasters.

This script:
1. Loads fuel metrics rasters (merged GeoTIFFs from 3DEP baseline or UAV processing)
2. Loads field measurements from CSV
3. Creates circular plot footprints (0.1-acre = 11.35m radius)
4. Extracts fuel metrics using pixel-weighted averaging
5. Computes comparison statistics (R², RMSE, MAE, bias)
6. Generates scatter plots

Usage:
    python src/evaluation/extract_fuel_metrics_at_plots.py \
        --raster-dir data/processed/fuel_metrics/3dep_baseline \
        --field-data data/processed/forest_plot_data/forest_plots_processed.csv \
        --output-dir data/processed/fuel_metrics/3dep_baseline/comparison

    # Verify band structure first
    python src/evaluation/extract_fuel_metrics_at_plots.py \
        --verify-bands data/processed/fuel_metrics/3dep_BluffMesa/merged/3dep_BluffMesa_fuel_metrics_2m.tif
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.crs import CRS
from scipy import stats
from shapely.geometry import Point, Polygon, box


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Unit conversion constants
KG_M2_TO_TONS_ACRE = 4.461  # 1 kg/m² = 4.461 tons/acre


# Band indices (0-indexed, based on src/fuel_metrics/README.md)
# Band 16 (1-indexed) = Band 15 (0-indexed) = TFL (Total Fuel Load) kg/m²
# Band 23 (1-indexed) = Band 22 (0-indexed) = Cover (Total cover %)
BAND_TFL = 15  # Total Fuel Load (kg/m²)
BAND_COVER = 22  # Total Cover (%)


def verify_raster_bands(raster_path: Path):
    """
    Print raster metadata to verify band structure.

    Args:
        raster_path: Path to fuel metrics GeoTIFF
    """
    logger.info(f"\nVerifying raster bands: {raster_path}")

    with rasterio.open(raster_path) as src:
        logger.info(f"  CRS: {src.crs}")
        logger.info(f"  Bounds: {src.bounds}")
        logger.info(f"  Shape: {src.height} × {src.width} pixels")
        logger.info(f"  Resolution: {src.transform[0]:.2f}m × {abs(src.transform[4]):.2f}m")
        logger.info(f"  Number of bands: {src.count}")

        # Read a sample pixel from each band to verify structure
        logger.info("\nSample pixel values (center of raster):")
        center_row = src.height // 2
        center_col = src.width // 2

        for band_idx in range(min(src.count, 25)):  # Show first 25 bands
            band_data = src.read(band_idx + 1, window=((center_row, center_row + 1), (center_col, center_col + 1)))
            value = band_data[0, 0]
            logger.info(f"  Band {band_idx + 1} (index {band_idx}): {value:.4f}")

        logger.info("\nExpected bands (based on src/fuel_metrics/README.md):")
        logger.info("  Band 16 (index 15) = TFL (Total Fuel Load) kg/m²")
        logger.info("  Band 23 (index 22) = Cover (Total cover %)")


def load_fuel_metrics_rasters(
    raster_dir: Path,
    pattern: str = "*_fuel_metrics_*.tif"
) -> Dict[str, Tuple[str, rasterio.DatasetReader]]:
    """
    Load fuel metrics rasters from directory.

    Args:
        raster_dir: Base directory containing per-site subdirectories
        pattern: Glob pattern for raster files

    Returns:
        Dict mapping site_name to (raster_path, rasterio_dataset)
    """
    logger.info(f"Loading fuel metrics rasters from {raster_dir}")

    raster_files = list(Path(raster_dir).rglob(pattern))
    logger.info(f"Found {len(raster_files)} raster files matching pattern '{pattern}'")

    site_rasters = {}

    for raster_path in raster_files:
        # Extract site name from path
        # Expecting structure: .../3dep_BluffMesa/merged/3dep_BluffMesa_fuel_metrics_2m.tif
        # Or: .../uav_BluffMesa/merged/uav_BluffMesa_fuel_metrics_5m.tif
        parts = raster_path.parts

        # Try to find site name
        site_name = None
        for part in reversed(parts):
            if part.endswith("Mesa") or part.endswith("Bear") or part.endswith("Peak") or "Laguna" in part:
                # Extract site name (remove prefixes like "3dep_" or "uav_")
                if "_" in part:
                    site_name = part.split("_", 1)[1]
                else:
                    site_name = part
                break

        if not site_name:
            logger.warning(f"Could not extract site name from {raster_path}, skipping")
            continue

        logger.info(f"  Loading {site_name}: {raster_path}")

        try:
            dataset = rasterio.open(raster_path)
            site_rasters[site_name] = (str(raster_path), dataset)
            logger.info(f"    CRS: {dataset.crs}, Shape: {dataset.height} × {dataset.width}, Bands: {dataset.count}")
        except Exception as e:
            logger.error(f"    Failed to open {raster_path}: {e}")

    logger.info(f"Loaded {len(site_rasters)} site rasters: {list(site_rasters.keys())}")

    return site_rasters


def load_field_data(field_data_path: Path) -> pd.DataFrame:
    """
    Load field measurement data from CSV.

    Args:
        field_data_path: Path to forest_plots_processed.csv

    Returns:
        DataFrame with columns: Site, Easting, Northing, TreeCover, TotalFuels, etc.
    """
    logger.info(f"Loading field data from {field_data_path}")

    df = pd.read_csv(field_data_path)
    logger.info(f"Loaded {len(df)} field plots")
    logger.info(f"Sites: {df['Site'].unique()}")
    logger.info(f"Key columns: {[c for c in df.columns if c in ['Site', 'Easting', 'Northing', 'TreeCover', 'TotalFuels']]}")

    return df


def create_plot_footprints(
    field_data: pd.DataFrame,
    radius_m: float = 11.35,
    crs: str = "EPSG:32611"
) -> gpd.GeoDataFrame:
    """
    Convert plot point locations to circular footprint polygons.

    Args:
        field_data: DataFrame with Easting, Northing columns
        radius_m: Plot footprint radius (default 11.35m = 0.1-acre circular)
        crs: Coordinate reference system

    Returns:
        GeoDataFrame with circular Polygon geometries
    """
    logger.info(f"Creating circular plot footprints (radius={radius_m}m)")

    # Create point geometries
    geometry = [Point(row['Easting'], row['Northing']) for _, row in field_data.iterrows()]
    gdf = gpd.GeoDataFrame(field_data, geometry=geometry, crs=crs)

    # Buffer to create circular footprints
    gdf['geometry'] = gdf.geometry.buffer(radius_m)
    gdf['footprint_area_m2'] = gdf.geometry.area

    logger.info(f"Created {len(gdf)} footprints (mean area: {gdf['footprint_area_m2'].mean():.1f} m²)")

    return gdf


def compute_weighted_prediction(
    plot_footprint: Polygon,
    dataset: rasterio.DatasetReader,
    band_idx: int,
    transform: Affine
) -> Dict[str, float]:
    """
    Compute weighted average prediction for single plot and band.

    Weights each pixel by its intersection area with the plot footprint.

    Args:
        plot_footprint: Shapely Polygon (plot boundary)
        dataset: Rasterio dataset reader
        band_idx: Which band to extract (0-indexed)
        transform: Affine transform from dataset

    Returns:
        Dict with weighted_prediction, n_pixels, coverage_fraction, weight_sum
    """
    # Get plot bounds
    minx, miny, maxx, maxy = plot_footprint.bounds

    # Convert to pixel coordinates
    inv_transform = ~transform

    col_min_f, row_min_f = inv_transform * (minx, maxy)  # Upper-left
    col_max_f, row_max_f = inv_transform * (maxx, miny)  # Lower-right

    col_min = max(0, int(np.floor(col_min_f)))
    col_max = min(dataset.width, int(np.ceil(col_max_f)) + 1)
    row_min = max(0, int(np.floor(row_min_f)))
    row_max = min(dataset.height, int(np.ceil(row_max_f)) + 1)

    # Read data window
    if row_max <= row_min or col_max <= col_min:
        return {
            'weighted_prediction': np.nan,
            'n_pixels': 0,
            'coverage_fraction': 0.0,
            'weight_sum': 0.0
        }

    window = ((row_min, row_max), (col_min, col_max))
    try:
        raster_data = dataset.read(band_idx + 1, window=window)  # rasterio uses 1-indexed bands
    except Exception as e:
        logger.warning(f"Failed to read band {band_idx + 1}: {e}")
        return {
            'weighted_prediction': np.nan,
            'n_pixels': 0,
            'coverage_fraction': 0.0,
            'weight_sum': 0.0
        }

    # Iterate over pixels
    weighted_sum = 0.0
    weight_sum = 0.0
    n_pixels = 0

    for row in range(row_min, row_max):
        for col in range(col_min, col_max):
            # Pixel bounding box in CRS coordinates
            pixel_x_min, pixel_y_max = transform * (col, row)  # Upper-left
            pixel_x_max, pixel_y_min = transform * (col + 1, row + 1)  # Lower-right

            pixel_box = box(pixel_x_min, pixel_y_min, pixel_x_max, pixel_y_max)

            # Compute intersection area
            try:
                intersection = plot_footprint.intersection(pixel_box)
            except Exception:
                continue

            if intersection.is_empty:
                continue

            area = intersection.area
            if area < 1e-6:  # Skip negligible intersections
                continue

            # Get pixel value
            pixel_value = raster_data[row - row_min, col - col_min]
            if np.isnan(pixel_value):
                continue  # Skip NaN predictions

            # Accumulate weighted sum
            weight = area
            weighted_sum += pixel_value * weight
            weight_sum += weight
            n_pixels += 1

    # Compute results
    plot_area = plot_footprint.area
    coverage_fraction = weight_sum / plot_area if plot_area > 0 else 0.0
    weighted_prediction = weighted_sum / weight_sum if weight_sum > 0 else np.nan

    return {
        'weighted_prediction': weighted_prediction,
        'n_pixels': n_pixels,
        'coverage_fraction': coverage_fraction,
        'weight_sum': weight_sum
    }


def extract_at_plots(
    plot_footprints: gpd.GeoDataFrame,
    site_rasters: Dict[str, Tuple[str, rasterio.DatasetReader]],
    bands_to_extract: Dict[str, int] = None
) -> pd.DataFrame:
    """
    Extract fuel metrics at all plot locations.

    Args:
        plot_footprints: GeoDataFrame with plot polygons and Site column
        site_rasters: Dict mapping site_name to (raster_path, dataset)
        bands_to_extract: Dict mapping metric_name to band_index

    Returns:
        DataFrame with extracted predictions per plot
    """
    if bands_to_extract is None:
        bands_to_extract = {
            'TFL_kg_m2': BAND_TFL,
            'Cover_pct': BAND_COVER
        }

    logger.info(f"Extracting fuel metrics at {len(plot_footprints)} plot locations")
    logger.info(f"Bands to extract: {bands_to_extract}")

    results = []

    for _, plot_row in plot_footprints.iterrows():
        site_name = plot_row['Site']
        plot_id = plot_row.get('Plot_ID', 'unknown')
        footprint = plot_row.geometry

        result = {
            'Site': site_name,
            'Plot_ID': plot_id,
            'Easting': plot_row['Easting'],
            'Northing': plot_row['Northing'],
            'TreeCover_field': plot_row.get('TreeCover', np.nan),
            'TotalFuels_field_tons_acre': plot_row.get('TotalFuels', np.nan),
        }

        # Check if we have a raster for this site
        if site_name not in site_rasters:
            logger.warning(f"No raster found for site {site_name}, plot {plot_id}")
            for metric_name in bands_to_extract.keys():
                result[f'{metric_name}_pred'] = np.nan
                result[f'{metric_name}_n_pixels'] = 0
                result[f'{metric_name}_coverage'] = 0.0
        else:
            raster_path, dataset = site_rasters[site_name]
            transform = dataset.transform

            # Extract each band
            for metric_name, band_idx in bands_to_extract.items():
                if band_idx >= dataset.count:
                    logger.warning(f"Band {band_idx} not in dataset (count={dataset.count})")
                    result[f'{metric_name}_pred'] = np.nan
                    result[f'{metric_name}_n_pixels'] = 0
                    result[f'{metric_name}_coverage'] = 0.0
                    continue

                extraction = compute_weighted_prediction(footprint, dataset, band_idx, transform)

                result[f'{metric_name}_pred'] = extraction['weighted_prediction']
                result[f'{metric_name}_n_pixels'] = extraction['n_pixels']
                result[f'{metric_name}_coverage'] = extraction['coverage_fraction']

        results.append(result)

    results_df = pd.DataFrame(results)

    # Convert TFL from kg/m² to tons/acre
    results_df['TFL_pred_tons_acre'] = results_df['TFL_kg_m2_pred'] * KG_M2_TO_TONS_ACRE

    logger.info(f"Extracted predictions for {len(results_df)} plots")

    return results_df


def compute_comparison_stats(
    predictions: pd.DataFrame,
    field_col: str,
    pred_col: str,
    metric_name: str
) -> Dict:
    """
    Compute comparison statistics between predictions and field measurements.

    Args:
        predictions: DataFrame with predictions and field data
        field_col: Column name for field measurements
        pred_col: Column name for predictions
        metric_name: Human-readable metric name

    Returns:
        Dict with R², RMSE, MAE, bias, n_samples
    """
    # Filter valid pairs (no NaN)
    valid = predictions[[field_col, pred_col]].dropna()

    if len(valid) == 0:
        logger.warning(f"No valid pairs for {metric_name}")
        return {
            'metric': metric_name,
            'n': 0,
            'r_squared': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'bias': np.nan
        }

    field_vals = valid[field_col].values
    pred_vals = valid[pred_col].values

    # R² (Pearson correlation coefficient squared)
    r, _ = stats.pearsonr(field_vals, pred_vals)
    r_squared = r ** 2

    # RMSE
    rmse = np.sqrt(np.mean((pred_vals - field_vals) ** 2))

    # MAE
    mae = np.mean(np.abs(pred_vals - field_vals))

    # Bias (mean error: positive = overprediction)
    bias = np.mean(pred_vals - field_vals)

    stats_dict = {
        'metric': metric_name,
        'n': len(valid),
        'r_squared': r_squared,
        'rmse': rmse,
        'mae': mae,
        'bias': bias
    }

    logger.info(f"\n{metric_name} statistics:")
    logger.info(f"  N: {stats_dict['n']}")
    logger.info(f"  R²: {r_squared:.3f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  Bias: {bias:.2f}")

    return stats_dict


def plot_comparison_figures(
    predictions: pd.DataFrame,
    output_path: Path,
    comparisons: List[Tuple[str, str, str, str]]
):
    """
    Generate scatter plots comparing predictions to field measurements.

    Args:
        predictions: DataFrame with predictions and field data
        output_path: Path to save figure
        comparisons: List of (field_col, pred_col, title, units) tuples
    """
    logger.info(f"Generating comparison figures: {output_path}")

    n_comparisons = len(comparisons)
    fig, axes = plt.subplots(1, n_comparisons, figsize=(6 * n_comparisons, 5))

    if n_comparisons == 1:
        axes = [axes]

    for ax, (field_col, pred_col, title, units) in zip(axes, comparisons):
        # Filter valid pairs
        valid = predictions[[field_col, pred_col]].dropna()

        if len(valid) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        field_vals = valid[field_col].values
        pred_vals = valid[pred_col].values

        # Scatter plot
        ax.scatter(field_vals, pred_vals, alpha=0.6, s=50)

        # 1:1 line
        lims = [
            min(field_vals.min(), pred_vals.min()),
            max(field_vals.max(), pred_vals.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, label='1:1 line')

        # Regression line
        slope, intercept, r, _, _ = stats.linregress(field_vals, pred_vals)
        line_x = np.array(lims)
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, 'r-', alpha=0.7, label=f'Fit: y={slope:.2f}x+{intercept:.2f}')

        # Stats
        r_squared = r ** 2
        rmse = np.sqrt(np.mean((pred_vals - field_vals) ** 2))
        bias = np.mean(pred_vals - field_vals)

        stats_text = f'R² = {r_squared:.3f}\nRMSE = {rmse:.2f}\nBias = {bias:.2f}\nn = {len(valid)}'
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax.set_xlabel(f'Field Measurement ({units})')
        ax.set_ylabel(f'Prediction ({units})')
        ax.set_title(title)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved comparison figures to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract fuel metrics at forest plot locations from GeoTIFF rasters"
    )
    parser.add_argument(
        "--raster-dir",
        type=Path,
        help="Base directory containing per-site fuel metrics rasters"
    )
    parser.add_argument(
        "--field-data",
        type=Path,
        help="Path to forest_plots_processed.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for comparison results"
    )
    parser.add_argument(
        "--verify-bands",
        type=Path,
        help="Path to raster file to verify band structure (prints metadata and exits)"
    )

    args = parser.parse_args()

    # Verify bands mode
    if args.verify_bands:
        verify_raster_bands(args.verify_bands)
        return

    # Validate arguments
    if not args.raster_dir or not args.field_data or not args.output_dir:
        parser.error("Must specify --raster-dir, --field-data, and --output-dir (or use --verify-bands)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    site_rasters = load_fuel_metrics_rasters(args.raster_dir)
    field_data = load_field_data(args.field_data)
    plot_footprints = create_plot_footprints(field_data)

    # Extract predictions
    predictions = extract_at_plots(plot_footprints, site_rasters)

    # Save predictions CSV
    predictions_path = args.output_dir / "baseline_predictions.csv"
    predictions.to_csv(predictions_path, index=False)
    logger.info(f"\nSaved predictions to {predictions_path}")

    # Compute comparison statistics
    stats_list = []

    # TreeCover comparison
    cover_stats = compute_comparison_stats(
        predictions,
        field_col='TreeCover_field',
        pred_col='Cover_pct_pred',
        metric_name='TreeCover'
    )
    cover_stats['units'] = 'percent'
    stats_list.append(cover_stats)

    # TotalFuels comparison (tons/acre)
    fuel_stats = compute_comparison_stats(
        predictions,
        field_col='TotalFuels_field_tons_acre',
        pred_col='TFL_pred_tons_acre',
        metric_name='TotalFuels'
    )
    fuel_stats['units'] = 'tons/acre'
    stats_list.append(fuel_stats)

    # Save statistics JSON
    stats_dict = {stat['metric']: {k: v for k, v in stat.items() if k != 'metric'} for stat in stats_list}
    stats_path = args.output_dir / "baseline_comparison_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats_dict, f, indent=2)
    logger.info(f"Saved statistics to {stats_path}")

    # Generate scatter plots
    comparisons = [
        ('TreeCover_field', 'Cover_pct_pred', 'Tree Cover', '%'),
        ('TotalFuels_field_tons_acre', 'TFL_pred_tons_acre', 'Total Fuel Load', 'tons/acre')
    ]
    figures_path = args.output_dir / "baseline_comparison_scatter.png"
    plot_comparison_figures(predictions, figures_path, comparisons)

    logger.info("\n" + "="*60)
    logger.info("EXTRACTION COMPLETE")
    logger.info("="*60)
    logger.info(f"Predictions: {predictions_path}")
    logger.info(f"Statistics: {stats_path}")
    logger.info(f"Figures: {figures_path}")


if __name__ == "__main__":
    main()
