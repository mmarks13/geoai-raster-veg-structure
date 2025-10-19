"""
process_uav_fuel_metrics.py

Main orchestration script for computing fuel metrics from UAV LiDAR point clouds.
Uses LidarForFuel R package via Python wrapper.

Usage:
    # Single file
    python process_uav_fuel_metrics.py --input data/raw/uavlidar/study_las/file.las

    # Multiple files
    python process_uav_fuel_metrics.py --input_dir data/raw/uavlidar/study_las --pattern "*.las"

    # Custom species and resolution
    python process_uav_fuel_metrics.py --input file.las --species "Quercus agrifolia" --resolution 0.5

    # Summary metrics only (23 bands instead of 173)
    python process_uav_fuel_metrics.py --input file.las --export_mode summary
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data_prep.lidarforfuel_wrapper import (
    process_point_cloud,
    load_trait_lookup,
    check_rscript_available,
    LidarForFuelError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fuel_metrics_processing.log')
    ]
)
logger = logging.getLogger(__name__)


def find_las_files(input_dir: Path, pattern: str = "*.las") -> List[Path]:
    """
    Find all LAS/LAZ files matching pattern in directory.

    Args:
        input_dir: Directory to search
        pattern: Glob pattern (default: "*.las")

    Returns:
        List of Path objects
    """
    files = list(input_dir.glob(pattern))
    laz_files = list(input_dir.glob(pattern.replace(".las", ".laz")))
    all_files = files + laz_files

    logger.info(f"Found {len(all_files)} files matching {pattern} in {input_dir}")
    return sorted(all_files)


def print_available_species():
    """Print available species from trait lookup table."""
    try:
        traits = load_trait_lookup()
        print("\nAvailable species/categories:")
        print("=" * 80)
        for species, data in traits.items():
            print(f"{species:30s}  {data['common_name']}")
            print(f"  LMA: {data['lma_gm2']:.1f} g/m²  |  WD: {data['wd_kgm3']:.1f} kg/m³")
            if data['notes']:
                print(f"  Notes: {data['notes']}")
            print()
    except Exception as e:
        print(f"Error loading trait lookup: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute fuel metrics from UAV LiDAR using LidarForFuel",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input specification
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input',
        type=Path,
        help='Single input LAS/LAZ file'
    )
    input_group.add_argument(
        '--input_dir',
        type=Path,
        help='Directory containing multiple LAS/LAZ files'
    )
    input_group.add_argument(
        '--list_species',
        action='store_true',
        help='List available species from trait lookup table and exit'
    )

    # Processing parameters
    parser.add_argument(
        '--pattern',
        type=str,
        default='*.las',
        help='Glob pattern for input files (default: *.las)'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('data/processed/fuel_metrics/volcan'),
        help='Output directory (default: data/processed/fuel_metrics/volcan)'
    )
    parser.add_argument(
        '--species',
        type=str,
        default='Mixed',
        help='Species name from trait lookup table (default: Mixed)'
    )
    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='Output raster resolution in meters (default: 1.0)'
    )
    parser.add_argument(
        '--export_mode',
        type=str,
        choices=['full', 'summary'],
        default='full',
        help='Export mode: full (173 bands) or summary (23 bands) (default: full)'
    )
    parser.add_argument(
        '--cleanup',
        action='store_true',
        help='Delete intermediate pretreated LAZ files after processing'
    )
    parser.add_argument(
        '--max_files',
        type=int,
        default=None,
        help='Maximum number of files to process (for testing)'
    )

    args = parser.parse_args()

    # Handle list_species
    if args.list_species:
        print_available_species()
        return 0

    # Pre-flight checks
    logger.info("=" * 80)
    logger.info("UAV LiDAR Fuel Metrics Processing")
    logger.info("=" * 80)

    if not check_rscript_available():
        logger.error("Rscript not found in PATH")
        logger.error("Please install R and required packages:")
        logger.error("  conda install -c conda-forge r-base r-lidr r-remotes")
        logger.error("  R -e \"remotes::install_github('oliviermartin7/lidarforfuel')\"")
        return 1

    # Collect input files
    if args.input:
        if not args.input.exists():
            logger.error(f"Input file not found: {args.input}")
            return 1
        input_files = [args.input]
    else:
        if not args.input_dir.exists():
            logger.error(f"Input directory not found: {args.input_dir}")
            return 1
        input_files = find_las_files(args.input_dir, args.pattern)

    if not input_files:
        logger.error("No input files found")
        return 1

    # Limit files if requested
    if args.max_files:
        input_files = input_files[:args.max_files]
        logger.info(f"Limited to first {args.max_files} files")

    logger.info(f"Processing {len(input_files)} file(s)")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Species: {args.species}")
    logger.info(f"Resolution: {args.resolution} m")
    logger.info(f"Export mode: {args.export_mode}")
    logger.info("=" * 80)

    # Process each file
    results = []
    failures = []

    for i, input_file in enumerate(input_files, 1):
        logger.info(f"\n[{i}/{len(input_files)}] Processing: {input_file.name}")
        start_time = time.time()

        try:
            pretreated_laz, fuel_metrics_tif = process_point_cloud(
                input_las=input_file,
                output_dir=args.output_dir,
                species=args.species,
                resolution=args.resolution,
                export_mode=args.export_mode,
                cleanup_intermediate=args.cleanup
            )

            elapsed = time.time() - start_time
            logger.info(f"✓ Completed in {elapsed:.1f} seconds")
            logger.info(f"  Pretreated LAZ: {pretreated_laz}")
            logger.info(f"  Fuel metrics:   {fuel_metrics_tif}")

            results.append({
                'input': input_file,
                'pretreated': pretreated_laz,
                'metrics': fuel_metrics_tif,
                'elapsed': elapsed
            })

        except (LidarForFuelError, FileNotFoundError, Exception) as e:
            logger.error(f"✗ Failed: {e}")
            failures.append({
                'input': input_file,
                'error': str(e)
            })

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Processing Summary")
    logger.info("=" * 80)
    logger.info(f"Successful: {len(results)}/{len(input_files)}")
    logger.info(f"Failed:     {len(failures)}/{len(input_files)}")

    if results:
        total_time = sum(r['elapsed'] for r in results)
        avg_time = total_time / len(results)
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Average:    {avg_time:.1f} seconds/file")

    if failures:
        logger.info("\nFailed files:")
        for f in failures:
            logger.info(f"  {f['input'].name}: {f['error']}")

    logger.info("=" * 80)

    return 0 if not failures else 1


if __name__ == '__main__':
    sys.exit(main())
