#!/usr/bin/env python
"""
Verify 3DEP HAG and enhanced features are correctly computed.

This script validates that processed LAZ files contain all required dimensions
with sensible values. It performs both structural checks (dimension presence)
and statistical checks (value ranges, distributions).

Usage:
    python scripts/verify_3dep_hag_features.py data/processed/3dep_hag_features/volcan_mtn/volcan_mtn_hag_features.copc.laz
    python scripts/verify_3dep_hag_features.py --dir data/processed/3dep_hag_features/

Output:
    - Console summary of validation results
    - JSON file with detailed statistics (optional)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pdal


# Required dimensions after processing
REQUIRED_DIMS = [
    'X', 'Y', 'Z',
    'HeightAboveGround',
    'Intensity', 'ReturnNumber', 'NumberOfReturns',
    'Planarity', 'Sphericity', 'Verticality'
]

# Expected value ranges for validation
EXPECTED_RANGES = {
    'HeightAboveGround': (-2.0, 100.0),  # Slight negative allowed for ground model error
    'Planarity': (0.0, 1.0),
    'Sphericity': (0.0, 1.0),
    'Verticality': (0.0, 1.0),
}


def verify_processed_laz(laz_path: str, verbose: bool = False) -> Dict:
    """
    Verify processed LAZ file has all required dimensions and sensible values.

    Args:
        laz_path: Path to processed LAZ file
        verbose: If True, print detailed statistics

    Returns:
        Dictionary with validation results
    """
    results = {
        'file': laz_path,
        'n_points': 0,
        'dimensions': {},
        'warnings': [],
        'errors': [],
        'all_checks_passed': True
    }

    # Read the LAZ file
    try:
        pipeline = pdal.Pipeline(json.dumps({
            "pipeline": [{"type": "readers.las", "filename": laz_path}]
        }))
        pipeline.execute()
        arr = pipeline.arrays[0]
        results['n_points'] = len(arr)

    except Exception as e:
        results['errors'].append(f"Failed to read LAZ file: {e}")
        results['all_checks_passed'] = False
        return results

    if results['n_points'] == 0:
        results['errors'].append("File contains 0 points")
        results['all_checks_passed'] = False
        return results

    # Check each required dimension
    available_dims = list(arr.dtype.names)

    for dim in REQUIRED_DIMS:
        if dim not in available_dims:
            results['dimensions'][dim] = {'present': False}
            results['errors'].append(f"Missing required dimension: {dim}")
            results['all_checks_passed'] = False
            continue

        data = arr[dim]

        # Compute statistics
        is_float = np.issubdtype(data.dtype, np.floating)
        if is_float:
            nan_mask = np.isnan(data)
            nan_count = int(nan_mask.sum())
            valid_data = data[~nan_mask]
        else:
            nan_count = 0
            valid_data = data

        if len(valid_data) == 0:
            results['dimensions'][dim] = {
                'present': True,
                'dtype': str(data.dtype),
                'all_nan': True,
                'nan_count': nan_count,
                'nan_pct': 100.0
            }
            results['warnings'].append(f"{dim}: all values are NaN/invalid")
            continue

        stats = {
            'present': True,
            'dtype': str(data.dtype),
            'min': float(np.min(valid_data)),
            'max': float(np.max(valid_data)),
            'mean': float(np.mean(valid_data)),
            'std': float(np.std(valid_data)),
            'median': float(np.median(valid_data)),
            'nan_count': nan_count,
            'nan_pct': 100.0 * nan_count / len(data) if len(data) > 0 else 0.0
        }

        # Add percentiles for key features
        if dim in ['HeightAboveGround', 'Planarity', 'Sphericity', 'Verticality']:
            stats['p01'] = float(np.percentile(valid_data, 1))
            stats['p05'] = float(np.percentile(valid_data, 5))
            stats['p25'] = float(np.percentile(valid_data, 25))
            stats['p75'] = float(np.percentile(valid_data, 75))
            stats['p95'] = float(np.percentile(valid_data, 95))
            stats['p99'] = float(np.percentile(valid_data, 99))

        results['dimensions'][dim] = stats

        # Validate against expected ranges
        if dim in EXPECTED_RANGES:
            expected_min, expected_max = EXPECTED_RANGES[dim]

            # Allow some tolerance for floating point
            tolerance = 0.01

            if stats['min'] < expected_min - tolerance:
                results['warnings'].append(
                    f"{dim}: min value {stats['min']:.4f} below expected {expected_min}"
                )

            if stats['max'] > expected_max + tolerance:
                results['warnings'].append(
                    f"{dim}: max value {stats['max']:.4f} above expected {expected_max}"
                )

    # Additional specific checks

    # HAG validation
    if 'HeightAboveGround' in results['dimensions'] and results['dimensions']['HeightAboveGround'].get('present'):
        hag_stats = results['dimensions']['HeightAboveGround']

        # Check for excessive negative HAG
        hag = arr['HeightAboveGround']
        valid_hag = hag[~np.isnan(hag)] if np.issubdtype(hag.dtype, np.floating) else hag

        neg_count = int((valid_hag < -0.5).sum())
        neg_pct = 100.0 * neg_count / len(valid_hag) if len(valid_hag) > 0 else 0

        if neg_pct > 1.0:
            results['warnings'].append(
                f"HAG: {neg_count:,} points ({neg_pct:.2f}%) below -0.5m - check ground classification"
            )

        # Check for extremely high HAG
        high_count = int((valid_hag > 80).sum())
        if high_count > 0:
            results['warnings'].append(
                f"HAG: {high_count:,} points above 80m - verify if expected for this site"
            )

    # Classification check
    if 'Classification' in available_dims:
        classification = arr['Classification']
        unique_classes, class_counts = np.unique(classification, return_counts=True)
        class_dist = {int(c): int(cnt) for c, cnt in zip(unique_classes, class_counts)}

        results['classification_distribution'] = class_dist

        # Check for ground points (class 2)
        ground_count = class_dist.get(2, 0)
        ground_pct = 100.0 * ground_count / len(classification) if len(classification) > 0 else 0

        results['ground_point_count'] = ground_count
        results['ground_point_pct'] = ground_pct

        if ground_pct < 5.0:
            results['warnings'].append(
                f"Only {ground_pct:.1f}% points classified as ground (class 2) - "
                "SMRF may not have run correctly"
            )
        elif ground_pct > 80.0:
            results['warnings'].append(
                f"{ground_pct:.1f}% points classified as ground - unusually high for vegetated site"
            )

    # Eigenvalue feature validation
    for dim in ['Planarity', 'Sphericity', 'Verticality']:
        if dim in results['dimensions'] and results['dimensions'][dim].get('present'):
            stats = results['dimensions'][dim]

            # Check for excessive NaN
            if stats['nan_pct'] > 10.0:
                results['warnings'].append(
                    f"{dim}: {stats['nan_pct']:.1f}% NaN values - check knn neighborhood computation"
                )

    return results


def print_results(results: Dict, verbose: bool = False) -> None:
    """Print validation results to console."""
    print(f"\n{'='*70}")
    print(f"Verification Results: {results['file']}")
    print(f"{'='*70}")

    print(f"\nTotal points: {results['n_points']:,}")

    if 'ground_point_pct' in results:
        print(f"Ground points: {results['ground_point_count']:,} ({results['ground_point_pct']:.1f}%)")

    print("\nDimension Statistics:")
    print("-" * 70)
    print(f"{'Dimension':<20} {'Present':^8} {'Min':>12} {'Max':>12} {'Mean':>12} {'NaN%':>8}")
    print("-" * 70)

    for dim in REQUIRED_DIMS:
        if dim in results['dimensions']:
            stats = results['dimensions'][dim]
            present = "✓" if stats.get('present') else "✗"

            if stats.get('all_nan'):
                print(f"{dim:<20} {present:^8} {'ALL NaN':>12} {'-':>12} {'-':>12} {'100.0':>8}")
            elif stats.get('present'):
                min_val = f"{stats['min']:.4f}" if stats['min'] is not None else '-'
                max_val = f"{stats['max']:.4f}" if stats['max'] is not None else '-'
                mean_val = f"{stats['mean']:.4f}" if stats['mean'] is not None else '-'
                nan_pct = f"{stats['nan_pct']:.1f}" if stats['nan_pct'] is not None else '-'
                print(f"{dim:<20} {present:^8} {min_val:>12} {max_val:>12} {mean_val:>12} {nan_pct:>8}")
            else:
                print(f"{dim:<20} {present:^8} {'MISSING':>12}")
        else:
            print(f"{dim:<20} {'✗':^8} {'MISSING':>12}")

    if verbose and 'classification_distribution' in results:
        print(f"\nClassification Distribution:")
        for cls, count in sorted(results['classification_distribution'].items()):
            pct = 100.0 * count / results['n_points']
            print(f"  Class {cls}: {count:,} ({pct:.1f}%)")

    if results['errors']:
        print(f"\n❌ ERRORS ({len(results['errors'])}):")
        for error in results['errors']:
            print(f"  - {error}")

    if results['warnings']:
        print(f"\n⚠️  WARNINGS ({len(results['warnings'])}):")
        for warning in results['warnings']:
            print(f"  - {warning}")

    if results['all_checks_passed'] and not results['warnings']:
        print("\n✅ All validation checks PASSED")
    elif results['all_checks_passed']:
        print(f"\n✅ Core checks PASSED (with {len(results['warnings'])} warnings)")
    else:
        print(f"\n❌ Validation FAILED")


def main():
    parser = argparse.ArgumentParser(
        description="Verify 3DEP HAG and enhanced features in processed LAZ files"
    )
    parser.add_argument(
        'laz_path',
        type=str,
        nargs='?',
        help='Path to processed LAZ file'
    )
    parser.add_argument(
        '--dir',
        type=Path,
        help='Directory containing processed LAZ files (processes all *.laz)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed statistics'
    )
    parser.add_argument(
        '--output-json',
        type=Path,
        help='Save detailed results to JSON file'
    )

    args = parser.parse_args()

    if not args.laz_path and not args.dir:
        parser.error("Must specify either laz_path or --dir")

    # Collect files to process
    files_to_check: List[Path] = []

    if args.dir:
        if not args.dir.exists():
            print(f"Error: Directory {args.dir} does not exist")
            sys.exit(1)

        # Find all LAZ files
        files_to_check = list(args.dir.rglob("*_hag_features.copc.laz"))
        if not files_to_check:
            files_to_check = list(args.dir.rglob("*.laz"))

        if not files_to_check:
            print(f"No LAZ files found in {args.dir}")
            sys.exit(1)

        print(f"Found {len(files_to_check)} LAZ files to verify")

    if args.laz_path:
        files_to_check.append(Path(args.laz_path))

    # Process each file
    all_results = {}
    any_failed = False

    for laz_path in files_to_check:
        if not laz_path.exists():
            print(f"Error: File {laz_path} does not exist")
            any_failed = True
            continue

        results = verify_processed_laz(str(laz_path), verbose=args.verbose)
        all_results[str(laz_path)] = results

        print_results(results, verbose=args.verbose)

        if not results['all_checks_passed']:
            any_failed = True

    # Save to JSON if requested
    if args.output_json:
        with open(args.output_json, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to {args.output_json}")

    # Summary for multiple files
    if len(files_to_check) > 1:
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)

        passed = sum(1 for r in all_results.values() if r['all_checks_passed'])
        failed = len(all_results) - passed

        print(f"Files checked: {len(all_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed > 0:
            print("\nFailed files:")
            for path, result in all_results.items():
                if not result['all_checks_passed']:
                    print(f"  - {path}")

    # Exit with error code if any failed
    if any_failed:
        sys.exit(1)


if __name__ == '__main__':
    main()
