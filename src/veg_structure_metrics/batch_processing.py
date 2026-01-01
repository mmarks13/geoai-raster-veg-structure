#!/usr/bin/env python3
"""Process a single tile for veg structure metrics."""

import argparse
from pathlib import Path
from src.utils.point_cloud_utils import compute_vegetation_structure_metrics, save_metrics_to_geotiff

def main():
    parser = argparse.ArgumentParser(description='Process a single tile for vegetation structure metrics')
    parser.add_argument('--input', type=Path, required=True, help='Input LAZ tile')
    parser.add_argument('--output_dir', type=Path, required=True, help='Site output directory')
    parser.add_argument('--point_filter_max_hag', type=float, default=60.0, help='Maximum HAG for point filtering (default: 60.0)')
    args = parser.parse_args()

    tile_name = args.input.stem
    output_tif = args.output_dir / 'rasters' / f"{tile_name}_veg_metrics.tif"
    output_tif.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing {tile_name}...")

    raster, metadata = compute_vegetation_structure_metrics(
        str(args.input),
        resolution=2.0,
        point_filter_max_hag=args.point_filter_max_hag,
        preprocess=False  # Tiles already have HAG from PDAL
    )
    save_metrics_to_geotiff(raster, metadata, str(output_tif))

    print(f"  Saved: {output_tif}")

if __name__ == '__main__':
    main()
