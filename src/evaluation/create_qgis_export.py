#!/usr/bin/env python3
"""
Create QGIS-ready export from forest plot evaluation results.

This script:
1. Creates a GeoPackage with plot footprints, predictions, and coverage diagnostics
2. Builds an N-band VRT combining all site rasters
3. Copies site rasters to export folder with relative paths

Usage:
    python src/evaluation/create_qgis_export.py \
        --comparison-dir data/output/forest_plot_evaluations/model_xxx/comparison \
        --output-dir data/output/forest_plot_evaluations/model_xxx/qgis_export \
        --band-config src/evaluation/configs/raster/cover_only.json
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
from osgeo import gdal
from shapely.geometry import Point

# Add script directory to path for local imports (avoids PYTHONPATH requirement)
sys.path.insert(0, str(Path(__file__).parent))
from band_config import load_band_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_footprints_gpkg(
    comparison_results_csv: str,
    output_path: str,
    band_config_path: Optional[str] = None
) -> None:
    """
    Create GeoPackage with plot footprints and predictions.

    Args:
        comparison_results_csv: Path to comparison_results.csv
        output_path: Output path for footprints.gpkg
        band_config_path: Optional path to band config (for band names)
    """
    logger.info("Creating plot footprints GeoPackage")

    # Load comparison results
    df = pd.read_csv(comparison_results_csv)
    logger.info(f"Loaded {len(df)} plot comparisons")

    # Load band config if provided
    if band_config_path:
        band_config = load_band_config(band_config_path)
        band_names = [b.name for b in band_config.bands]
    else:
        # Infer band names from columns
        band_names = list(set([col.replace('_pred', '').replace('_field', '') 
                              for col in df.columns 
                              if col.endswith('_pred') or col.endswith('_field')]))

    # Create circular footprints (11.35m radius = 0.1 acre)
    footprint_radius = 11.35
    geometries = [Point(row['plot_x'], row['plot_y']).buffer(footprint_radius)
                  for _, row in df.iterrows()]

    # Build GeoDataFrame
    footprints_gdf = gpd.GeoDataFrame(
        df,
        geometry=geometries,
        crs='EPSG:32611'
    )

    # Add coverage quality classifications
    for band_name in band_names:
        coverage_col = f'{band_name}_coverage_fraction'
        if coverage_col in footprints_gdf.columns:
            def classify_coverage(frac):
                if pd.isna(frac):
                    return None
                elif frac >= 0.95:
                    return 'full'
                elif frac >= 0.50:
                    return 'partial'
                else:
                    return 'poor'
            
            footprints_gdf[f'{band_name}_coverage_quality'] = \
                footprints_gdf[coverage_col].apply(classify_coverage)

    # Save to GeoPackage
    footprints_gdf.to_file(output_path, driver='GPKG')
    logger.info(f"Saved footprints to {output_path}")
    logger.info(f"  Total plots: {len(footprints_gdf)}")
    logger.info(f"  CRS: {footprints_gdf.crs}")


def create_combined_vrt(
    site_raster_paths: List[str],
    output_vrt_path: str,
    band_config_path: Optional[str] = None
) -> None:
    """
    Create N-band VRT combining all site rasters.

    Uses relative paths for portability.

    Args:
        site_raster_paths: List of paths to site raster GeoTIFFs
        output_vrt_path: Output path for VRT file
        band_config_path: Optional path to band config (for band names)
    """
    logger.info("Creating combined VRT from site rasters")

    output_vrt = Path(output_vrt_path)
    output_dir = output_vrt.parent

    # Get number of bands from first raster
    if not site_raster_paths:
        raise ValueError("No site rasters provided")

    first_raster = gdal.Open(site_raster_paths[0])
    if first_raster is None:
        raise ValueError(f"Could not open raster: {site_raster_paths[0]}")
    
    n_bands = first_raster.RasterCount
    projection = first_raster.GetProjection()
    first_raster = None

    logger.info(f"Number of bands: {n_bands}")

    # Load band names from config
    band_names = []
    if band_config_path:
        band_config = load_band_config(band_config_path)
        band_names = [b.display_name for b in band_config.bands]
    else:
        band_names = [f'Band_{i}' for i in range(n_bands)]

    # Get overall extent and resolution
    logger.info(f"Processing {len(site_raster_paths)} site rasters")

    all_bounds = []
    resolutions = []

    for raster_path in site_raster_paths:
        if not Path(raster_path).exists():
            logger.warning(f"Raster not found: {raster_path}")
            continue

        ds = gdal.Open(raster_path)
        if ds is None:
            continue

        width = ds.RasterXSize
        height = ds.RasterYSize
        geotransform = ds.GetGeoTransform()

        # Extract bounds
        top_left_x = geotransform[0]
        pixel_width = geotransform[1]
        top_left_y = geotransform[3]
        pixel_height = -geotransform[5]

        xmin = top_left_x
        xmax = top_left_x + width * pixel_width
        ymin = top_left_y - height * pixel_height
        ymax = top_left_y

        all_bounds.append((xmin, ymin, xmax, ymax))
        resolutions.append((pixel_width, pixel_height))

        ds = None

    if not all_bounds:
        raise ValueError("No valid rasters found")

    # Compute overall extent
    xmin = min(b[0] for b in all_bounds)
    ymin = min(b[1] for b in all_bounds)
    xmax = max(b[2] for b in all_bounds)
    ymax = max(b[3] for b in all_bounds)

    # Use first raster's resolution
    pixel_width, pixel_height = resolutions[0]

    # Compute VRT dimensions
    vrt_width = int((xmax - xmin) / pixel_width)
    vrt_height = int((ymax - ymin) / pixel_height)

    logger.info(f"VRT extent: [{xmin:.0f}, {ymin:.0f}] to [{xmax:.0f}, {ymax:.0f}]")
    logger.info(f"VRT size: {vrt_width} × {vrt_height} pixels")
    logger.info(f"Resolution: {pixel_width}m × {pixel_height}m")

    # Build VRT XML
    vrt_xml = f'''<VRTDataset rasterXSize="{vrt_width}" rasterYSize="{vrt_height}">
  <SRS>{projection}</SRS>
  <GeoTransform>{xmin}, {pixel_width}, 0.0, {ymax}, 0.0, {-pixel_height}</GeoTransform>
'''

    # Add bands
    for band_idx in range(1, n_bands + 1):
        band_name = band_names[band_idx - 1] if band_idx <= len(band_names) else f'Band_{band_idx}'

        vrt_xml += f'''  <VRTRasterBand dataType="Float32" band="{band_idx}">
    <Description>{band_name}</Description>
    <NoDataValue>nan</NoDataValue>
'''

        # Add source for each site raster
        for raster_path in site_raster_paths:
            if not Path(raster_path).exists():
                continue

            # Get relative path from VRT to source raster
            try:
                rel_path = Path(raster_path).relative_to(output_dir)
            except ValueError:
                # If not relative, use absolute path
                rel_path = Path(raster_path)

            ds = gdal.Open(raster_path)
            if ds is None:
                continue

            width = ds.RasterXSize
            height = ds.RasterYSize
            geotransform = ds.GetGeoTransform()

            # Source bounds in georeferenced coordinates
            src_xmin = geotransform[0]
            src_ymax = geotransform[3]
            src_xmax = src_xmin + width * geotransform[1]
            src_ymin = src_ymax - height * (-geotransform[5])

            # Destination pixel coordinates in VRT space
            dst_xoff = int((src_xmin - xmin) / pixel_width)
            dst_yoff = int((ymax - src_ymax) / pixel_height)

            vrt_xml += f'''    <ComplexSource>
      <SourceFilename relativeToVRT="1">{rel_path.as_posix()}</SourceFilename>
      <SourceBand>{band_idx}</SourceBand>
      <SrcRect xOff="0" yOff="0" xSize="{width}" ySize="{height}" />
      <DstRect xOff="{dst_xoff}" yOff="{dst_yoff}" xSize="{width}" ySize="{height}" />
      <NODATA>nan</NODATA>
    </ComplexSource>
'''

            ds = None

        vrt_xml += '  </VRTRasterBand>\n'

    vrt_xml += '</VRTDataset>\n'

    # Write VRT file
    with open(output_vrt_path, 'w') as f:
        f.write(vrt_xml)

    logger.info(f"Created VRT at {output_vrt_path}")

    # Verify VRT is valid
    ds = gdal.Open(output_vrt_path)
    if ds is None:
        raise ValueError(f"Failed to create valid VRT at {output_vrt_path}")

    logger.info(f"  Verified VRT: {ds.RasterCount} bands, {ds.RasterXSize}×{ds.RasterYSize} pixels")
    ds = None


def prepare_qgis_export(
    comparison_dir: Path,
    output_dir: Path,
    band_config_path: Optional[str] = None
) -> None:
    """
    Main function: create export folder with footprints + VRT + source rasters.

    Args:
        comparison_dir: Path to comparison directory
        output_dir: Path to output directory for QGIS export
        band_config_path: Optional path to band config
    """
    logger.info("=" * 60)
    logger.info("CREATING QGIS EXPORT")
    logger.info("=" * 60)

    comparison_dir = Path(comparison_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Locate required files
    comparison_csv = comparison_dir / 'comparison_results.csv'
    site_rasters_dir = comparison_dir.parent / 'site_rasters'

    if not comparison_csv.exists():
        raise FileNotFoundError(f"Comparison results not found: {comparison_csv}")
    if not site_rasters_dir.exists():
        raise FileNotFoundError(f"Site rasters directory not found: {site_rasters_dir}")

    # Find all site rasters
    site_rasters = sorted(site_rasters_dir.glob('*_predictions_raster.tif'))
    logger.info(f"Found {len(site_rasters)} site rasters")

    if not site_rasters:
        raise ValueError("No site rasters found in comparison directory")

    # Copy site rasters to export directory
    logger.info("\nCopying site rasters to export directory")
    copied_rasters = []
    for raster_path in site_rasters:
        dest_path = output_dir / raster_path.name
        shutil.copy2(raster_path, dest_path)
        logger.info(f"  Copied {raster_path.name}")
        copied_rasters.append(str(dest_path))

    # Create footprints GeoPackage
    logger.info("\nCreating plot footprints GeoPackage")
    footprints_path = output_dir / 'plot_footprints.gpkg'
    create_footprints_gpkg(
        str(comparison_csv),
        str(footprints_path),
        band_config_path=band_config_path
    )

    # Create combined VRT
    logger.info("\nCreating combined VRT")
    vrt_path = output_dir / 'predictions_combined.vrt'
    create_combined_vrt(
        copied_rasters,
        str(vrt_path),
        band_config_path=band_config_path
    )

    logger.info("\n" + "=" * 60)
    logger.info("QGIS EXPORT COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"\nFiles created:")
    logger.info(f"  - plot_footprints.gpkg ({len(gpd.read_file(footprints_path))} plots)")
    
    # Get band count from VRT
    ds = gdal.Open(str(vrt_path))
    n_bands = ds.RasterCount if ds else 0
    ds = None
    
    logger.info(f"  - predictions_combined.vrt ({n_bands} bands)")
    logger.info(f"  - {len(site_rasters)} site raster GeoTIFFs")
    logger.info(f"\nTo use in QGIS:")
    logger.info(f"  1. Drag and drop the entire folder: {output_dir.name}/")
    logger.info(f"  2. Load predictions_combined.vrt for raster visualization")
    logger.info(f"  3. Load plot_footprints.gpkg for plot locations and comparison")


def main():
    parser = argparse.ArgumentParser(
        description="Create QGIS-ready export from forest plot evaluation results"
    )
    parser.add_argument(
        '--comparison-dir',
        type=str,
        required=True,
        help='Path to comparison directory (contains comparison_results.csv; site_rasters/ should be in parent directory)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for QGIS export'
    )
    parser.add_argument(
        '--band-config',
        type=str,
        default=None,
        help='Optional path to band config JSON (for band names)'
    )

    args = parser.parse_args()

    prepare_qgis_export(
        Path(args.comparison_dir),
        Path(args.output_dir),
        band_config_path=args.band_config
    )


if __name__ == '__main__':
    main()
