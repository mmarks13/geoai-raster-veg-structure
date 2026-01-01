#!/usr/bin/env python3
"""
Build prediction rasters from model inference results.

This script generates per-site GeoTIFF rasters from tile-level predictions.
It can be used independently for creating rasters to share with researchers,
without requiring forest plot comparison.

Usage:
    python src/evaluation/build_prediction_rasters.py \
        --predictions-pt predictions.pt \
        --predictions-csv predictions.csv \
        --output-dir output/site_rasters \
        --site-column site_name

Output:
    Per-site GeoTIFF files with N bands (one per prediction band)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from affine import Affine
from rasterio.crs import CRS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def build_site_rasters(
    predictions_dict: Dict[str, np.ndarray],
    tile_metadata_csv: str,
    field_plots_path: Optional[str] = None,
    site_column: str = 'site_name',
    crs: str = 'EPSG:32611',
    output_dir: Optional[Path] = None
) -> Dict[str, Tuple[np.ndarray, Affine, CRS]]:
    """
    Build separate raster for each site from tile predictions.

    Sites are hundreds of miles apart, so unified raster would be massive.
    This builds one raster per site to avoid 99.9% NaN values.

    Args:
        predictions_dict: Dict mapping tile_id to prediction array [n_bands, 5, 5]
        tile_metadata_csv: CSV with tile_id, site_name, bbox_xmin/ymin/xmax/ymax
        field_plots_path: Optional path to field plots GeoPackage (for site spatial join if site_name='unknown')
        site_column: Column name containing site names (default: 'site_name')
        crs: Coordinate reference system (default EPSG:32611)
        output_dir: If provided, save rasters per site as GeoTIFF

    Returns:
        Dict mapping site_name to (raster_array, transform, crs) tuple
    """
    logger.info("Building per-site rasters from tile predictions")

    # Load tile metadata
    df = pd.read_csv(tile_metadata_csv)
    df = df[df['tile_id'].isin(predictions_dict.keys())]  # Only tiles with predictions

    logger.info(f"Loaded metadata for {len(df)} tiles")

    # If site_name is 'unknown', do spatial join with field plots
    if site_column in df.columns and (df[site_column] == 'unknown').any():
        if field_plots_path is None:
            raise ValueError(
                "Site names are 'unknown' but no field_plots_path provided for spatial join"
            )

        logger.info("Site names not populated - performing spatial join with field plots")

        # Load field plots to get site boundaries
        field_plots = gpd.read_file(field_plots_path)

        # Reproject field plots to match tile CRS if needed
        if str(field_plots.crs) != crs:
            logger.info(f"Reprojecting field plots from {field_plots.crs} to {crs}")
            field_plots = field_plots.to_crs(crs)

        # Create tile geometries (centroids)
        from shapely.geometry import Point
        tile_geom = [
            Point((row['bbox_xmin'] + row['bbox_xmax']) / 2,
                  (row['bbox_ymin'] + row['bbox_ymax']) / 2)
            for _, row in df.iterrows()
        ]
        tiles_gdf = gpd.GeoDataFrame(df, geometry=tile_geom, crs=crs)

        # Create site boundaries (convex hull + 1km buffer per site)
        site_boundaries = field_plots.dissolve(by='Site').copy()
        site_boundaries['geometry'] = site_boundaries.geometry.buffer(1000)  # 1km buffer

        # Spatial join
        tiles_with_sites = gpd.sjoin(tiles_gdf, site_boundaries, how='left', predicate='within')

        # Update site_name column
        df[site_column] = tiles_with_sites['Site'].fillna('unknown').values

        logger.info(f"Spatial join complete - sites: {df[site_column].unique()}")
        logger.info(f"Tiles per site:\n{df[site_column].value_counts()}")

    logger.info(f"Sites: {df[site_column].unique()}")

    # Grid cell configuration
    cell_size = 2.0  # 2m × 2m cells
    grid_offsets = np.array([-4, -2, 0, 2, 4])  # 5×5 grid centers

    # Get number of bands from first prediction
    n_bands = next(iter(predictions_dict.values())).shape[0]
    logger.info(f"Prediction bands: {n_bands}")

    # Build raster per site
    site_rasters = {}
    crs_obj = CRS.from_string(crs)

    for site_name in df[site_column].unique():
        logger.info(f"\nProcessing site: {site_name}")
        site_df = df[df[site_column] == site_name].copy()

        # Compute site extent
        xmin_site = site_df['bbox_xmin'].min()
        ymin_site = site_df['bbox_ymin'].min()
        xmax_site = site_df['bbox_xmax'].max()
        ymax_site = site_df['bbox_ymax'].max()

        logger.info(f"  Extent: [{xmin_site:.0f}, {ymin_site:.0f}] to [{xmax_site:.0f}, {ymax_site:.0f}]")

        # Compute raster dimensions
        width = int(np.ceil((xmax_site - xmin_site) / cell_size))
        height = int(np.ceil((ymax_site - ymin_site) / cell_size))

        logger.info(f"  Raster size: {width} × {height} pixels ({width*height*n_bands*4 / 1e6:.1f} MB)")

        # Initialize accumulator arrays for averaging overlapping tiles
        # Using sum and count arrays allows proper averaging of overlapping predictions
        raster_sum = np.zeros((n_bands, height, width), dtype=np.float64)
        raster_count = np.zeros((height, width), dtype=np.int32)

        # Create Affine transform: upper-left corner origin
        transform = Affine.translation(xmin_site, ymax_site) * Affine.scale(cell_size, -cell_size)

        # Accumulate tiles into raster
        n_placed = 0
        n_overlaps = 0

        for _, row in site_df.iterrows():
            tile_id = row['tile_id']
            if tile_id not in predictions_dict:
                continue

            pred = predictions_dict[tile_id]  # [n_bands, 5, 5]

            # Tile center in CRS coordinates
            tile_center_x = (row['bbox_xmin'] + row['bbox_xmax']) / 2
            tile_center_y = (row['bbox_ymin'] + row['bbox_ymax']) / 2

            # For each cell in 5×5 grid
            for i, offset_y in enumerate(grid_offsets):
                for j, offset_x in enumerate(grid_offsets):
                    # Cell center in CRS coordinates
                    cell_x = tile_center_x + offset_x
                    cell_y = tile_center_y + offset_y

                    # Convert to pixel coordinates (raster uses upper-left origin)
                    col = int((cell_x - xmin_site) / cell_size)
                    row_idx = int((ymax_site - cell_y) / cell_size)

                    # Bounds check
                    if 0 <= row_idx < height and 0 <= col < width:
                        # Track overlaps for logging
                        if raster_count[row_idx, col] > 0:
                            n_overlaps += 1

                        # Accumulate prediction (will be averaged later)
                        raster_sum[:, row_idx, col] += pred[:, i, j]
                        raster_count[row_idx, col] += 1
                        n_placed += 1

        # Compute final raster as average of overlapping predictions
        # Where count > 0, divide sum by count; otherwise leave as NaN
        raster_array = np.full((n_bands, height, width), np.nan, dtype=np.float32)
        valid_mask = raster_count > 0
        for b in range(n_bands):
            raster_array[b, valid_mask] = (raster_sum[b, valid_mask] / raster_count[valid_mask]).astype(np.float32)

        logger.info(f"  Placed {n_placed} pixels from {len(site_df)} tiles")
        if n_overlaps > 0:
            logger.info(f"  {n_overlaps} overlapping pixels averaged (expected with tile overlap)")

        # Calculate coverage
        n_valid = np.sum(~np.isnan(raster_array[0]))
        coverage_pct = 100 * n_valid / (height * width)
        logger.info(f"  Coverage: {n_valid}/{height*width} pixels ({coverage_pct:.1f}%)")

        # Optionally save to GeoTIFF
        if output_dir is not None:
            import rasterio
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{site_name}_predictions_raster.tif"

            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=n_bands,
                dtype=raster_array.dtype,
                crs=crs_obj,
                transform=transform,
                nodata=np.nan,
                compress='LZW',
                tiled=True
            ) as dst:
                dst.write(raster_array)

            logger.info(f"  Saved to {output_path}")

        site_rasters[site_name] = (raster_array, transform, crs_obj)

    logger.info(f"\nBuilt {len(site_rasters)} site-specific rasters")
    return site_rasters


def main():
    parser = argparse.ArgumentParser(
        description="Build prediction rasters from model inference results"
    )
    parser.add_argument(
        '--predictions-pt',
        type=str,
        required=True,
        help='Path to predictions .pt file (from raster_inference.py)'
    )
    parser.add_argument(
        '--predictions-csv',
        type=str,
        required=True,
        help='Path to predictions CSV with tile metadata (from raster_inference.py)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for site rasters'
    )
    parser.add_argument(
        '--field-plots',
        type=str,
        default=None,
        help='Optional path to field plots GeoPackage (for site spatial join if needed)'
    )
    parser.add_argument(
        '--site-column',
        type=str,
        default='site_name',
        help='Column name containing site names (default: site_name)'
    )
    parser.add_argument(
        '--crs',
        type=str,
        default='EPSG:32611',
        help='Coordinate reference system (default: EPSG:32611)'
    )

    args = parser.parse_args()

    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions dict
    logger.info(f"Loading predictions from {args.predictions_pt}")
    predictions_dict = torch.load(args.predictions_pt, weights_only=False)
    logger.info(f"Loaded {len(predictions_dict)} tile predictions")

    # Build rasters
    site_rasters = build_site_rasters(
        predictions_dict,
        args.predictions_csv,
        field_plots_path=args.field_plots,
        site_column=args.site_column,
        crs=args.crs,
        output_dir=output_dir
    )

    logger.info("\n" + "=" * 60)
    logger.info("RASTER GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Site rasters: {len(site_rasters)}")
    for site_name in sorted(site_rasters.keys()):
        logger.info(f"  - {site_name}_predictions_raster.tif")


if __name__ == '__main__':
    main()
