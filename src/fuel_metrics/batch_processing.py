#!/usr/bin/env python3
"""
Batch tile processing with per-tile logging and summary tracking.

Processes LiDAR tiles through the fuel metrics pipeline with parallel execution,
detailed per-tile logs, and a summary CSV for tracking results.

Usage:
    # Process specific tile
    python scripts/process_tiles_batch.py \
        --input data/processed/fuel_metrics/volcan_tiles/tile_1_1cm.laz \
        --output_dir data/processed/fuel_metrics/volcan_tiles_output \
        --species Mixed \
        --resolution 5.0

    # Batch processing with GNU parallel
    ls data/processed/fuel_metrics/volcan_tiles/tile_*_1cm.laz | \
      parallel -j 6 \
      "python scripts/process_tiles_batch.py \
        --input {} \
        --output_dir data/processed/fuel_metrics/volcan_tiles_output \
        --species Mixed \
        --resolution 5.0"

    # Batch processing with xargs
    find data/processed/fuel_metrics/volcan_tiles -name 'tile_*_1cm.laz' -print0 | \
      xargs -0 -n1 -P6 -I{} \
      python scripts/process_tiles_batch.py \
        --input {} \
        --output_dir data/processed/fuel_metrics/volcan_tiles_output \
        --species Mixed \
        --resolution 5.0
"""

import argparse
import csv
import json
import logging
import subprocess
import sys
import time
import fcntl
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def get_tile_metadata(las_file: Path) -> dict:
    """Extract bounds and metadata from LAS file using PDAL."""
    try:
        result = subprocess.run(
            ['conda', 'run', '-p', '/home/jovyan/geoai_env',
             'pdal', 'info', '--summary', str(las_file)],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode != 0:
            return {'error': f'PDAL failed: {result.stderr}'}

        info = json.loads(result.stdout)
        summary = info.get('summary', {})
        bounds = summary.get('bounds', {})

        # Calculate area from bounds
        area_m2 = None
        if bounds:
            area_m2 = (bounds['maxx'] - bounds['minx']) * (bounds['maxy'] - bounds['miny'])

        file_size_mb = las_file.stat().st_size / (1024 * 1024)

        return {
            'point_count': summary.get('count', None),
            'bounds_minx': bounds.get('minx', None),
            'bounds_miny': bounds.get('miny', None),
            'bounds_maxx': bounds.get('maxx', None),
            'bounds_maxy': bounds.get('maxy', None),
            'bounds_minz': bounds.get('minz', None),
            'bounds_maxz': bounds.get('maxz', None),
            'area_m2': area_m2,
            'file_size_mb': file_size_mb,
        }
    except Exception as e:
        return {'error': str(e)}


def append_to_summary_csv(csv_file: Path, row_data: dict):
    """Append row to CSV with file locking for concurrent writes."""
    fieldnames = [
        'tile_name', 'status', 'point_count', 'area_m2', 'file_size_mb',
        'bounds_minx', 'bounds_miny', 'bounds_maxx', 'bounds_maxy',
        'bounds_minz', 'bounds_maxz',
        'duration_sec', 'error_message', 'log_file', 'timestamp'
    ]

    csv_file.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_file.exists()

    with open(csv_file, 'a', newline='') as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row_data)
            f.flush()
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(
        description='Process LiDAR tile through fuel metrics pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--input', type=Path, required=True,
                       help='Input LAS/LAZ tile')
    parser.add_argument('--output_dir', type=Path, required=True,
                       help='Output directory for processed tiles')
    parser.add_argument('--species', type=str, default='Mixed',
                       help='Species/category for trait lookup (default: Mixed)')
    parser.add_argument('--resolution', type=float, default=5.0,
                       help='Output raster resolution in meters (default: 5.0)')
    parser.add_argument('--clumping', type=float, default=0.77,
                       help='Clumping factor Ω for Beer-Lambert model (default: 0.77)')
    parser.add_argument('--projection_factor', type=float, default=0.5,
                       help='Projection factor G for fuel metrics (default: 0.5)')
    parser.add_argument('--export_mode', type=str, default='summary',
                       choices=['summary', 'profile'],
                       help='Export mode: summary (23 bands) or profile (173 bands)')
    parser.add_argument('--log_dir', type=Path,
                       default=Path('data/logs/tile_processing'),
                       help='Log directory for per-tile logs')
    parser.add_argument('--csv_summary', type=Path,
                       default=Path('data/logs/tile_processing_summary.csv'),
                       help='Summary CSV file for batch tracking')
    args = parser.parse_args()

    # Validate input exists
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        return 1

    # Set up per-tile logging
    args.log_dir.mkdir(parents=True, exist_ok=True)
    tile_name = args.input.stem
    log_file = args.log_dir / f"{tile_name}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)

    # Initialize result tracking
    result = {
        'tile_name': tile_name,
        'status': 'UNKNOWN',
        'point_count': None,
        'area_m2': None,
        'file_size_mb': None,
        'bounds_minx': None,
        'bounds_miny': None,
        'bounds_maxx': None,
        'bounds_maxy': None,
        'bounds_minz': None,
        'bounds_maxz': None,
        'duration_sec': None,
        'error_message': None,
        'log_file': str(log_file),
        'timestamp': datetime.now().isoformat()
    }

    start_time = time.time()

    try:
        logger.info("=" * 80)
        logger.info(f"Processing: {tile_name}")
        logger.info("=" * 80)
        logger.info(f"Input:               {args.input}")
        logger.info(f"Output dir:          {args.output_dir}")
        logger.info(f"Species:             {args.species}")
        logger.info(f"Resolution:          {args.resolution}m")
        logger.info(f"Clumping (Ω):        {args.clumping}")
        logger.info(f"Projection factor (G): {args.projection_factor}")
        logger.info(f"Mode:                {args.export_mode}")
        logger.info("")

        # Extract tile metadata
        logger.info("Extracting tile metadata...")
        metadata = get_tile_metadata(args.input)

        if 'error' in metadata:
            logger.error(f"Metadata extraction failed: {metadata['error']}")
            result['status'] = 'METADATA_ERROR'
            result['error_message'] = metadata['error']
        else:
            # Log metadata
            if metadata.get('file_size_mb'):
                logger.info(f"  File size: {metadata['file_size_mb']:.2f} MB")
            if metadata.get('area_m2'):
                logger.info(f"  Area: {metadata['area_m2']:.2f} m²")
            if metadata.get('bounds_minx') is not None:
                logger.info(f"  Bounds X: [{metadata['bounds_minx']:.2f}, {metadata['bounds_maxx']:.2f}]")
                logger.info(f"  Bounds Y: [{metadata['bounds_miny']:.2f}, {metadata['bounds_maxy']:.2f}]")
            logger.info("")

            result.update(metadata)

            # Run fuel metrics pipeline
            logger.info("Running fuel metrics pipeline...")

            cmd = [
                'python', 'src/fuel_metrics/process_fuel_metrics.py',
                '--input', str(args.input),
                '--output_dir', str(args.output_dir),
                '--species', args.species,
                '--resolution', str(args.resolution),
                '--clumping', str(args.clumping),
                '--projection_factor', str(args.projection_factor),
                '--export_mode', args.export_mode
            ]

            pipeline_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200  # 2 hour timeout per tile
            )

            # Log pipeline output
            if pipeline_result.stdout:
                logger.info("Pipeline output:")
                logger.info(pipeline_result.stdout)

            if pipeline_result.stderr:
                logger.warning("Pipeline warnings/errors:")
                logger.warning(pipeline_result.stderr)

            if pipeline_result.returncode == 0:
                result['status'] = 'SUCCESS'
                logger.info("")
                logger.info("✓ Pipeline completed successfully")
            else:
                result['status'] = 'PIPELINE_ERROR'
                result['error_message'] = f"Exit code {pipeline_result.returncode}"
                logger.error(f"✗ Pipeline failed with exit code {pipeline_result.returncode}")

    except subprocess.TimeoutExpired:
        result['status'] = 'TIMEOUT'
        result['error_message'] = 'Exceeded 2 hour timeout'
        logger.error("✗ Processing timed out after 2 hours")

    except Exception as e:
        result['status'] = 'EXCEPTION'
        result['error_message'] = str(e)
        logger.exception("✗ Unexpected exception:")

    finally:
        # Calculate duration
        result['duration_sec'] = time.time() - start_time

        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Summary: {tile_name}")
        logger.info("=" * 80)
        logger.info(f"Status:   {result['status']}")
        logger.info(f"Duration: {result['duration_sec']:.1f} seconds")
        if result['error_message']:
            logger.info(f"Error:    {result['error_message']}")
        logger.info(f"Log:      {log_file}")
        logger.info("=" * 80)

        # Append to summary CSV
        try:
            append_to_summary_csv(args.csv_summary, result)
            logger.info(f"Summary: {args.csv_summary}")
        except Exception as e:
            logger.error(f"Failed to write summary CSV: {e}")

        # Return appropriate exit code
        return 0 if result['status'] == 'SUCCESS' else 1


if __name__ == '__main__':
    sys.exit(main())
