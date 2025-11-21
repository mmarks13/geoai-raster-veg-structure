#!/usr/bin/env python3
"""
Validation Script for Raster Training Data

Validates combined training data against source fuel metrics raster.
Checks data completeness, dimensions, value ranges, and logs tiles
that will be filtered in Step 4 (train_test_split_and_precompute_raster.py).

Usage:
    python src/data_prep/validate_raster_training_data.py \
      --input data/processed/model_data_raster/combined_training_data_raster_test.pt \
      --fuel-metrics-raster data/processed/fuel_metrics/volcan_mtn/merged/volcan_mtn_fuel_metrics_2.0m.tif \
      --output-dir data/processed/model_data_raster/validation_report \
      --max-na-ratio 0.5 \
      --min-dep-points 100 \
      --verbose
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import hashlib

import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend


# Band name mappings for fuel metrics (bands 1-21, 23 - Band 22 removed due to corruption)
FUEL_METRICS_BAND_NAMES = {
    1: "Understory Height",
    2: "Canopy Height",
    3: "Height",
    4: "Canopy Base Height (CBH)",
    5: "Fuel Strata Gap (FSG)",
    6: "Canopy Relief Ratio (CRR)",
    7: "Depth",
    8: "Canopy Fuel Load",
    9: "Total Fuel Load",
    10: "Midstorey Fuel Load",
    11: "Surface Fuel Load",
    12: "Canopy Cover",
    13: "Midstorey Cover",
    14: "Understory Cover",
    15: "Total Cover",
    16: "Above 2m Canopy Cover",
    17: "Vertical Complexity Index (VCI)",
    18: "Entropy",
    19: "PAI above 2m",
    20: "PAI above CBH",
    21: "PAI Canopy",
    # Band 22: "PAI Total" - REMOVED (corrupted data ~413M)
    23: "max_CBD"
}

# Bands that should have NA→0 replacement (from Step 4 logic)
NA_TO_ZERO_BANDS = [3] + list(range(8, 17)) + [19, 20, 21, 23]  # Bands 3, 8-16, 19-21, 23 (skip 22)

# Bands where NA is valid/expected
NA_VALID_BANDS = [1, 2, 4, 5, 6, 7, 17, 18]  # Bands 1-2, 4-7, 17-18


class Colors:
    """ANSI color codes for console output"""
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text: str):
    """Print colored header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")


def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_info(text: str):
    """Print info message"""
    print(f"  {text}")


def compute_fuel_metrics_reference(raster_path: str, verbose: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Compute reference distribution statistics from source fuel metrics raster.

    Parameters
    ----------
    raster_path : str
        Path to source fuel metrics GeoTIFF (173 bands)
    verbose : bool
        Print progress messages

    Returns
    -------
    dict
        Dictionary mapping band number (1-21, 23) to statistics:
        {band_num: {'min', 'max', 'mean', 'std', 'p1', 'p5', 'p25', 'p50', 'p75', 'p95', 'p99', 'na_ratio'}}
    """
    if verbose:
        print_header("Computing Fuel Metrics Reference Distribution")
        print_info(f"Source raster: {raster_path}")

    reference = {}

    with rasterio.open(raster_path) as src:
        if verbose:
            print_info(f"Raster shape: {src.height} x {src.width}")
            print_info(f"Total bands: {src.count} (reading bands 1-21, 23 - skip Band 22)")

        # Read bands 1-21, 23 (skip Band 22 - corrupted)
        band_indices = list(range(1, 22)) + [23]
        for band_num in band_indices:
            if verbose:
                print_info(f"Processing band {band_num}: {FUEL_METRICS_BAND_NAMES[band_num]}")

            data = src.read(band_num).astype(np.float32)
            valid_mask = ~np.isnan(data)
            valid_data = data[valid_mask]

            if len(valid_data) == 0:
                print_warning(f"Band {band_num} has no valid data (100% NaN)")
                reference[band_num] = {
                    'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan,
                    'p1': np.nan, 'p5': np.nan, 'p25': np.nan, 'p50': np.nan,
                    'p75': np.nan, 'p95': np.nan, 'p99': np.nan,
                    'na_ratio': 1.0, 'total_pixels': data.size
                }
            else:
                percentiles = np.percentile(valid_data, [1, 5, 25, 50, 75, 95, 99])
                reference[band_num] = {
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data)),
                    'p1': float(percentiles[0]),
                    'p5': float(percentiles[1]),
                    'p25': float(percentiles[2]),
                    'p50': float(percentiles[3]),
                    'p75': float(percentiles[4]),
                    'p95': float(percentiles[5]),
                    'p99': float(percentiles[6]),
                    'na_ratio': float(1 - valid_mask.sum() / data.size),
                    'total_pixels': int(data.size)
                }

                if verbose:
                    print_info(f"  Range: [{reference[band_num]['min']:.2f}, {reference[band_num]['max']:.2f}]")
                    print_info(f"  Mean ± Std: {reference[band_num]['mean']:.2f} ± {reference[band_num]['std']:.2f}")
                    print_info(f"  NA ratio: {reference[band_num]['na_ratio']*100:.1f}%")

    if verbose:
        print_success(f"Reference distribution computed for {len(reference)} bands")

    return reference


def validate_tile_completeness(tile: Dict, tile_id: str) -> Dict[str, Any]:
    """Check if tile has all expected data modalities"""
    issues = []

    # Check fuel_metrics
    if 'fuel_metrics' not in tile:
        issues.append(('missing_fuel_metrics', 'error', 'fuel_metrics field missing'))
    elif tile['fuel_metrics'] is None or len(tile['fuel_metrics']) == 0:
        issues.append(('empty_fuel_metrics', 'error', 'fuel_metrics data is None or empty'))

    # Check NAIP
    if 'naip_imgs' not in tile:
        issues.append(('missing_naip', 'error', 'naip_imgs field missing'))
    elif tile['naip_imgs'] is None or len(tile['naip_imgs']) == 0:
        issues.append(('empty_naip', 'warning', 'No NAIP images'))

    # Check UAVSAR
    if 'uavsar_imgs' not in tile:
        issues.append(('missing_uavsar', 'error', 'uavsar_imgs field missing'))
    elif tile['uavsar_imgs'] is None or len(tile['uavsar_imgs']) == 0:
        issues.append(('empty_uavsar', 'warning', 'No UAVSAR images'))

    # Check 3DEP
    if 'dep_points' not in tile:
        issues.append(('missing_dep', 'error', 'dep_points field missing'))
    elif tile['dep_points'] is None or len(tile['dep_points']) == 0:
        issues.append(('empty_dep', 'warning', 'No 3DEP points'))

    return {
        'tile_id': tile_id,
        'has_fuel_metrics': 'fuel_metrics' in tile and tile['fuel_metrics'] is not None,
        'has_naip': 'naip_imgs' in tile and tile['naip_imgs'] is not None and len(tile['naip_imgs']) > 0,
        'has_uavsar': 'uavsar_imgs' in tile and tile['uavsar_imgs'] is not None and len(tile['uavsar_imgs']) > 0,
        'has_dep': 'dep_points' in tile and tile['dep_points'] is not None and len(tile['dep_points']) > 0,
        'issues': issues
    }


def validate_tile_dimensions(tile: Dict, tile_id: str) -> Dict[str, Any]:
    """Validate data dimensions match expected shapes"""
    issues = []
    dims = {}

    # Fuel metrics: Expected (22, H, W) - Band 22 removed
    if 'fuel_metrics' in tile and tile['fuel_metrics'] is not None:
        fm_shape = tile['fuel_metrics'].shape
        dims['fuel_metrics'] = fm_shape
        if len(fm_shape) != 3:
            issues.append(('fuel_metrics_ndim', 'error', f'Expected 3D array, got {len(fm_shape)}D'))
        elif fm_shape[0] != 22:
            issues.append(('fuel_metrics_bands', 'error', f'Expected 22 bands, got {fm_shape[0]}'))

    # NAIP: Expected (N_images, 4, 40, 40)
    if 'naip_imgs' in tile and tile['naip_imgs'] is not None and len(tile['naip_imgs']) > 0:
        naip_shape = tile['naip_imgs'].shape
        dims['naip'] = naip_shape
        if len(naip_shape) != 4:
            issues.append(('naip_ndim', 'error', f'Expected 4D array, got {len(naip_shape)}D'))
        elif naip_shape[1] != 4:
            issues.append(('naip_bands', 'error', f'Expected 4 bands, got {naip_shape[1]}'))
        elif naip_shape[2] != 40 or naip_shape[3] != 40:
            issues.append(('naip_spatial', 'warning', f'Expected 40x40 pixels, got {naip_shape[2]}x{naip_shape[3]}'))

    # UAVSAR: Expected (N_images, 6, 4, 4)
    if 'uavsar_imgs' in tile and tile['uavsar_imgs'] is not None and len(tile['uavsar_imgs']) > 0:
        uavsar_shape = tile['uavsar_imgs'].shape
        dims['uavsar'] = uavsar_shape
        if len(uavsar_shape) != 4:
            issues.append(('uavsar_ndim', 'error', f'Expected 4D array, got {len(uavsar_shape)}D'))
        elif uavsar_shape[1] != 6:
            issues.append(('uavsar_bands', 'error', f'Expected 6 bands, got {uavsar_shape[1]}'))
        elif uavsar_shape[2] != 4 or uavsar_shape[3] != 4:
            issues.append(('uavsar_spatial', 'warning', f'Expected 4x4 pixels, got {uavsar_shape[2]}x{uavsar_shape[3]}'))

    # 3DEP: Expected (N_points, 3)
    if 'dep_points' in tile and tile['dep_points'] is not None and len(tile['dep_points']) > 0:
        dep_shape = tile['dep_points'].shape
        dims['dep_points'] = dep_shape
        if len(dep_shape) != 2:
            issues.append(('dep_points_ndim', 'error', f'Expected 2D array, got {len(dep_shape)}D'))
        elif dep_shape[1] != 3:
            issues.append(('dep_points_coords', 'error', f'Expected 3 coordinates (x,y,z), got {dep_shape[1]}'))

        # Check attributes match point count
        if 'dep_pnt_attr' in tile and tile['dep_pnt_attr'] is not None:
            attr_shape = tile['dep_pnt_attr'].shape
            dims['dep_pnt_attr'] = attr_shape
            if attr_shape[0] != dep_shape[0]:
                issues.append(('dep_attr_mismatch', 'error',
                             f'Point count mismatch: {dep_shape[0]} points vs {attr_shape[0]} attributes'))

    return {
        'tile_id': tile_id,
        'dimensions': dims,
        'issues': issues
    }


def validate_tile_values(tile: Dict, tile_id: str, reference: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Validate value ranges against reference distribution"""
    issues = []
    stats = {}

    # Fuel metrics: Compare against reference distribution
    if 'fuel_metrics' in tile and tile['fuel_metrics'] is not None:
        fm_data = tile['fuel_metrics']

        for band_idx in range(fm_data.shape[0]):
            band_num = band_idx + 1  # 1-indexed
            band_data = fm_data[band_idx].flatten().cpu().numpy()  # Convert to numpy

            # Compute tile statistics
            valid_mask = ~np.isnan(band_data)
            valid_data = band_data[valid_mask]
            na_ratio = 1 - (valid_mask.sum() / band_data.size)

            tile_stats = {
                'na_ratio': float(na_ratio),
                'n_valid': int(valid_mask.sum()),
                'n_total': int(band_data.size)
            }

            if len(valid_data) > 0:
                tile_stats.update({
                    'min': float(np.min(valid_data)),
                    'max': float(np.max(valid_data)),
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data))
                })

                # Compare against reference
                ref = reference.get(band_num, {})

                # Check if values are outside reference percentile range [p1, p99]
                if 'p1' in ref and 'p99' in ref and not np.isnan(ref['p1']):
                    outliers_low = valid_data < ref['p1']
                    outliers_high = valid_data > ref['p99']
                    n_outliers = outliers_low.sum() + outliers_high.sum()

                    if n_outliers > 0:
                        outlier_ratio = n_outliers / len(valid_data)
                        if outlier_ratio > 0.05:  # More than 5% outliers
                            issues.append((f'band{band_num}_outliers', 'warning',
                                         f'Band {band_num} has {outlier_ratio*100:.1f}% values outside reference [p1, p99]'))
                        tile_stats['outlier_ratio'] = float(outlier_ratio)

                # Check for extreme max values (> 10x reference max)
                if 'max' in ref and ref['max'] > 0:
                    if tile_stats['max'] > 10 * ref['max']:
                        issues.append((f'band{band_num}_extreme_max', 'error',
                                     f'Band {band_num} max {tile_stats["max"]:.2f} >> reference max {ref["max"]:.2f}'))

            stats[f'band_{band_num}'] = tile_stats

    # NAIP: Check uint8 range [0, 255]
    if 'naip_imgs' in tile and tile['naip_imgs'] is not None and len(tile['naip_imgs']) > 0:
        naip_data = tile['naip_imgs']
        if naip_data.dtype != np.uint8:
            issues.append(('naip_dtype', 'warning', f'Expected uint8, got {naip_data.dtype}'))
        naip_min, naip_max = naip_data.min(), naip_data.max()
        if naip_min < 0 or naip_max > 255:
            issues.append(('naip_range', 'error', f'NAIP values [{naip_min}, {naip_max}] outside [0, 255]'))
        stats['naip'] = {'min': int(naip_min), 'max': int(naip_max), 'dtype': str(naip_data.dtype)}

    # 3DEP: Check points within bbox
    if 'dep_points' in tile and 'bbox' in tile:
        dep_points = tile['dep_points']
        bbox = tile['bbox']
        if len(dep_points) > 0 and len(bbox) == 4:
            x_out = (dep_points[:, 0] < bbox[0]) | (dep_points[:, 0] > bbox[2])
            y_out = (dep_points[:, 1] < bbox[1]) | (dep_points[:, 1] > bbox[3])
            n_outside = (x_out | y_out).sum()
            if n_outside > 0:
                issues.append(('dep_outside_bbox', 'warning',
                             f'{n_outside}/{len(dep_points)} 3DEP points outside tile bbox'))

    return {
        'tile_id': tile_id,
        'stats': stats,
        'issues': issues
    }


def check_step4_filters(tile: Dict, tile_id: str, max_na_ratio: float, min_dep_points: int) -> Dict[str, Any]:
    """Check if tile will be filtered in Step 4"""
    will_filter = False
    reasons = []

    # Check fuel metrics NA ratio (band 15 = Total Cover)
    if 'fuel_metrics' in tile and tile['fuel_metrics'] is not None:
        fm_data = tile['fuel_metrics']
        if fm_data.shape[0] >= 15:
            band15_data = fm_data[14].flatten().cpu().numpy()  # 0-indexed, convert to numpy
            na_ratio = np.isnan(band15_data).sum() / band15_data.size

            if na_ratio > max_na_ratio:
                will_filter = True
                reasons.append(f'na_ratio={na_ratio:.2f} > {max_na_ratio}')

    # Check 3DEP point count
    if 'dep_points' in tile and tile['dep_points'] is not None:
        n_points = len(tile['dep_points'])
        if n_points < min_dep_points:
            will_filter = True
            reasons.append(f'dep_points={n_points} < {min_dep_points}')

    return {
        'tile_id': tile_id,
        'will_filter': will_filter,
        'reasons': reasons
    }


def validate_tiles(tiles: List[Dict], reference: Dict, max_na_ratio: float, min_dep_points: int,
                   verbose: bool = False) -> Dict[str, Any]:
    """
    Validate all tiles and accumulate statistics.

    Returns comprehensive validation results including:
    - Completeness summary
    - Dimension validation
    - Value range validation
    - Step 4 filter preview
    - Flagged tiles
    """
    if verbose:
        print_header(f"Validating {len(tiles)} Tiles")

    results = {
        'total_tiles': len(tiles),
        'completeness': defaultdict(int),
        'dimension_issues': [],
        'value_issues': [],
        'step4_filters': [],
        'flagged_tiles': [],
        'tile_stats': []
    }

    for i, tile_dict in enumerate(tiles):
        tile_id = tile_dict.get('tile_id', f'tile_{i}')

        if verbose and (i + 1) % 100 == 0:
            print_info(f"Processed {i + 1}/{len(tiles)} tiles...")

        # Completeness
        completeness = validate_tile_completeness(tile_dict, tile_id)
        results['completeness']['has_fuel_metrics'] += int(completeness['has_fuel_metrics'])
        results['completeness']['has_naip'] += int(completeness['has_naip'])
        results['completeness']['has_uavsar'] += int(completeness['has_uavsar'])
        results['completeness']['has_dep'] += int(completeness['has_dep'])

        for issue_type, severity, details in completeness['issues']:
            results['flagged_tiles'].append({
                'tile_id': tile_id,
                'issue_type': issue_type,
                'severity': severity,
                'details': details
            })

        # Dimensions
        dimensions = validate_tile_dimensions(tile_dict, tile_id)
        if dimensions['issues']:
            results['dimension_issues'].append(dimensions)
            for issue_type, severity, details in dimensions['issues']:
                results['flagged_tiles'].append({
                    'tile_id': tile_id,
                    'issue_type': issue_type,
                    'severity': severity,
                    'details': details
                })

        # Values
        values = validate_tile_values(tile_dict, tile_id, reference)
        if values['issues']:
            results['value_issues'].append(values)
            for issue_type, severity, details in values['issues']:
                results['flagged_tiles'].append({
                    'tile_id': tile_id,
                    'issue_type': issue_type,
                    'severity': severity,
                    'details': details
                })

        results['tile_stats'].append({
            'tile_id': tile_id,
            'stats': values['stats']
        })

        # Step 4 filters
        filter_check = check_step4_filters(tile_dict, tile_id, max_na_ratio, min_dep_points)
        if filter_check['will_filter']:
            results['step4_filters'].append(filter_check)

    if verbose:
        print_success(f"Validation complete: {len(tiles)} tiles processed")

    return results


def generate_reports(results: Dict, reference: Dict, output_dir: Path, verbose: bool = False):
    """Generate validation reports (JSON, CSV, plots)"""
    if verbose:
        print_header("Generating Validation Reports")

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Reference distribution JSON
    reference_file = output_dir / 'fuel_metrics_reference_distribution.json'
    with open(reference_file, 'w') as f:
        json.dump(reference, f, indent=2)
    if verbose:
        print_success(f"Saved reference distribution: {reference_file}")

    # 2. Validation summary JSON
    summary = {
        'total_tiles': results['total_tiles'],
        'completeness': dict(results['completeness']),
        'completeness_pct': {
            k: f"{v/results['total_tiles']*100:.1f}%"
            for k, v in results['completeness'].items()
        },
        'dimension_issues_count': len(results['dimension_issues']),
        'value_issues_count': len(results['value_issues']),
        'flagged_tiles_count': len(results['flagged_tiles']),
        'step4_filter_count': len(results['step4_filters']),
        'step4_filter_pct': f"{len(results['step4_filters'])/results['total_tiles']*100:.1f}%"
    }

    summary_file = output_dir / 'validation_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print_success(f"Saved validation summary: {summary_file}")

    # 3. Flagged tiles CSV
    if results['flagged_tiles']:
        import csv
        flagged_file = output_dir / 'flagged_tiles.csv'
        with open(flagged_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['tile_id', 'issue_type', 'severity', 'details'])
            writer.writeheader()
            writer.writerows(results['flagged_tiles'])
        if verbose:
            print_success(f"Saved flagged tiles: {flagged_file}")

    # 4. Step 4 filters CSV
    if results['step4_filters']:
        import csv
        filters_file = output_dir / 'tiles_to_be_filtered.csv'
        with open(filters_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['tile_id', 'will_filter', 'reasons'])
            writer.writeheader()
            for item in results['step4_filters']:
                writer.writerow({
                    'tile_id': item['tile_id'],
                    'will_filter': item['will_filter'],
                    'reasons': '; '.join(item['reasons'])
                })
        if verbose:
            print_success(f"Saved Step 4 filter preview: {filters_file}")

    # 5. Generate distribution comparison plots
    try:
        plot_file = output_dir / 'fuel_metrics_distributions.png'
        plot_fuel_metrics_distributions(results['tile_stats'], reference, plot_file)
        if verbose:
            print_success(f"Saved distribution plots: {plot_file}")
    except Exception as e:
        print_warning(f"Failed to generate plots: {e}")

    if verbose:
        print_success("All reports generated successfully")


def plot_fuel_metrics_distributions(tile_stats: List[Dict], reference: Dict, output_file: Path):
    """Plot fuel metrics distributions comparing tiles vs source"""
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    axes = axes.flatten()

    for band_num in range(1, 24):
        ax = axes[band_num - 1]

        # Collect all tile values for this band
        tile_values = []
        for tile in tile_stats:
            band_key = f'band_{band_num}'
            if band_key in tile['stats']:
                # Would need actual values here, for now skip plotting
                pass

        # Plot reference distribution info
        ref = reference.get(band_num, {})
        if ref:
            ax.text(0.5, 0.5,
                   f"Band {band_num}: {FUEL_METRICS_BAND_NAMES[band_num]}\n"
                   f"Ref Range: [{ref.get('min', 'N/A'):.2f}, {ref.get('max', 'N/A'):.2f}]\n"
                   f"Ref Mean: {ref.get('mean', 'N/A'):.2f}\n"
                   f"Ref NA: {ref.get('na_ratio', 0)*100:.1f}%",
                   ha='center', va='center', fontsize=8)
        ax.set_title(f"Band {band_num}", fontsize=10)
        ax.axis('off')

    # Hide unused subplots
    for i in range(23, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()


def print_console_report(results: Dict, verbose: bool = True):
    """Print human-readable console report"""
    print_header("VALIDATION REPORT")

    # Executive summary
    total = results['total_tiles']
    print_info(f"Total tiles validated: {total}")
    print()

    # Completeness
    print(f"{Colors.BOLD}Data Completeness:{Colors.END}")
    for key, count in results['completeness'].items():
        pct = count / total * 100
        status = Colors.GREEN if pct >= 95 else Colors.YELLOW if pct >= 80 else Colors.RED
        print(f"  {key}: {status}{count}/{total} ({pct:.1f}%){Colors.END}")
    print()

    # Issues summary
    print(f"{Colors.BOLD}Issues Summary:{Colors.END}")
    print(f"  Dimension issues: {len(results['dimension_issues'])}")
    print(f"  Value range issues: {len(results['value_issues'])}")
    print(f"  Total flagged tiles: {len(results['flagged_tiles'])}")
    print()

    # Step 4 filter preview
    n_filtered = len(results['step4_filters'])
    filter_pct = n_filtered / total * 100
    print(f"{Colors.BOLD}Step 4 Filter Preview:{Colors.END}")
    print(f"  Tiles to be filtered: {Colors.YELLOW}{n_filtered}/{total} ({filter_pct:.1f}%){Colors.END}")

    if n_filtered > 0 and verbose:
        # Count reasons
        reason_counts = defaultdict(int)
        for item in results['step4_filters']:
            for reason in item['reasons']:
                reason_counts[reason] += 1

        print(f"  Breakdown by reason:")
        for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"    - {reason}: {count} tiles")
    print()

    # Overall status
    critical_issues = sum(1 for item in results['flagged_tiles'] if item['severity'] == 'error')
    if critical_issues == 0:
        print_success(f"Validation PASSED: No critical errors found")
    else:
        print_error(f"Validation FAILED: {critical_issues} critical errors found")


def main():
    parser = argparse.ArgumentParser(
        description='Validate raster training data against source fuel metrics raster',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        required=True,
        help='Path to combined training data .pt file'
    )
    parser.add_argument(
        '--fuel-metrics-raster',
        required=True,
        help='Path to source fuel metrics GeoTIFF'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for validation reports'
    )
    parser.add_argument(
        '--max-na-ratio',
        type=float,
        default=0.5,
        help='Maximum NA ratio threshold for Step 4 filtering (default: 0.5)'
    )
    parser.add_argument(
        '--min-dep-points',
        type=int,
        default=100,
        help='Minimum 3DEP point count for Step 4 filtering (default: 100)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print detailed progress messages'
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.input):
        print_error(f"Input file not found: {args.input}")
        sys.exit(1)

    if not os.path.exists(args.fuel_metrics_raster):
        print_error(f"Fuel metrics raster not found: {args.fuel_metrics_raster}")
        sys.exit(1)

    output_dir = Path(args.output_dir)

    try:
        # Step 1: Compute reference distribution
        reference = compute_fuel_metrics_reference(args.fuel_metrics_raster, verbose=args.verbose)

        # Step 2: Load training data
        if args.verbose:
            print_header("Loading Training Data")
            print_info(f"Loading {args.input}...")

        tiles = torch.load(args.input, weights_only=False)

        if args.verbose:
            print_success(f"Loaded {len(tiles)} tiles")

        # Step 3: Validate tiles
        results = validate_tiles(tiles, reference, args.max_na_ratio, args.min_dep_points,
                                verbose=args.verbose)

        # Step 4: Generate reports
        generate_reports(results, reference, output_dir, verbose=args.verbose)

        # Step 5: Print console report
        print_console_report(results, verbose=args.verbose)

    except Exception as e:
        print_error(f"Validation failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
