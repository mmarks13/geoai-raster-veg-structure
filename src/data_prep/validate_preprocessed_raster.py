#!/usr/bin/env python3
"""
Validation Script for Preprocessed Raster Training Data

Validates precomputed tiles after preprocessing with two-stage normalization.
Checks for presence of normalized coordinates, proper statistics, and NaN values.

Usage:
    python src/data_prep/validate_preprocessed_raster.py \
      --training-tiles data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt \
      --validation-tiles data/processed/model_data_raster/precomputed_validation_tiles_raster_32bit.pt \
      --output-dir data/processed/model_data_raster/validation_preprocessed
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_tile_normalization(tile: Dict, tile_id: str) -> Dict[str, Any]:
    """
    Validate normalization in a single tile.

    Returns:
        dict: Validation results with keys 'valid', 'errors', 'warnings', 'stats'
    """
    result = {
        'tile_id': tile_id,
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Check for required keys
    required_keys = ['dep_points_norm', 'norm_params']
    for key in required_keys:
        if key not in tile:
            result['valid'] = False
            result['errors'].append(f"Missing required key: {key}")

    if not result['valid']:
        return result

    # Check dep_points_norm
    dep_norm = tile['dep_points_norm']
    if not isinstance(dep_norm, torch.Tensor):
        result['valid'] = False
        result['errors'].append(f"dep_points_norm is not a tensor: {type(dep_norm)}")
        return result

    if dep_norm.ndim != 2 or dep_norm.shape[1] != 3:
        result['valid'] = False
        result['errors'].append(f"dep_points_norm has wrong shape: {dep_norm.shape}, expected [N, 3]")
        return result

    # Check for NaN values
    if torch.isnan(dep_norm).any():
        n_nan = torch.isnan(dep_norm).sum().item()
        result['errors'].append(f"dep_points_norm contains {n_nan} NaN values")
        result['valid'] = False

    # Check for Inf values
    if torch.isinf(dep_norm).any():
        n_inf = torch.isinf(dep_norm).sum().item()
        result['errors'].append(f"dep_points_norm contains {n_inf} Inf values")
        result['valid'] = False

    # Compute and store statistics
    if dep_norm.shape[0] > 0:
        result['stats']['dep_points_norm'] = {
            'shape': list(dep_norm.shape),
            'mean': dep_norm.mean(dim=0).tolist(),
            'std': dep_norm.std(dim=0).tolist(),
            'min': dep_norm.min(dim=0).values.tolist(),
            'max': dep_norm.max(dim=0).values.tolist()
        }

    # Check norm_params
    norm_params = tile['norm_params']
    if not isinstance(norm_params, dict):
        result['valid'] = False
        result['errors'].append(f"norm_params is not a dict: {type(norm_params)}")
        return result

    required_param_keys = ['coord_mean', 'coord_std', 'attr_mean', 'attr_std']
    for key in required_param_keys:
        if key not in norm_params:
            result['errors'].append(f"norm_params missing key: {key}")

    result['stats']['norm_params'] = norm_params

    # Check dep_points_attr_norm if present
    if 'dep_points_attr_norm' in tile and tile['dep_points_attr_norm'] is not None:
        attr_norm = tile['dep_points_attr_norm']
        if not isinstance(attr_norm, torch.Tensor):
            result['errors'].append(f"dep_points_attr_norm is not a tensor: {type(attr_norm)}")
        elif attr_norm.ndim != 2 or attr_norm.shape[1] not in (3, 6):
            result['errors'].append(f"dep_points_attr_norm has wrong shape: {attr_norm.shape}, expected [N, 3] or [N, 6]")
        elif torch.isnan(attr_norm).any() or torch.isinf(attr_norm).any():
            result['errors'].append(f"dep_points_attr_norm contains NaN/Inf values")
        else:
            result['stats']['dep_points_attr_norm'] = {
                'shape': list(attr_norm.shape),
                'mean': attr_norm.mean(dim=0).tolist(),
                'std': attr_norm.std(dim=0).tolist()
            }

    return result


def validate_dataset(tiles_path: Path, dataset_name: str = 'training') -> Dict[str, Any]:
    """Validate all tiles in a dataset."""
    logger.info(f"\n{'='*80}")
    logger.info(f"VALIDATING {dataset_name.upper()} DATASET: {tiles_path.name}")
    logger.info(f"{'='*80}")

    # Load tiles
    logger.info(f"Loading {dataset_name} tiles...")
    tiles = torch.load(str(tiles_path), weights_only=False)
    logger.info(f"Loaded {len(tiles)} {dataset_name} tiles")

    # Validate all tiles
    results = {
        'dataset': dataset_name,
        'total_tiles': len(tiles),
        'valid_tiles': 0,
        'invalid_tiles': 0,
        'tiles_with_errors': [],
        'tiles_with_warnings': [],
        'global_stats': {
            'coord_mean_global': None,
            'coord_std_global': None,
            'attr_mean_global': None,
            'attr_std_global': None
        }
    }

    all_coords = []
    all_attrs = []

    for idx, tile in enumerate(tiles):
        tile_id = tile.get('tile_id', f'tile_{idx}')
        result = validate_tile_normalization(tile, tile_id)

        if result['valid']:
            results['valid_tiles'] += 1
        else:
            results['invalid_tiles'] += 1
            results['tiles_with_errors'].append(result)

        if result['warnings']:
            results['tiles_with_warnings'].append(result)

        # Collect coordinates for global statistics
        if 'stats' in result and 'dep_points_norm' in result['stats']:
            if 'dep_points_norm' in tile:
                all_coords.append(tile['dep_points_norm'])

        if idx % 500 == 0:
            logger.info(f"  Validated {idx+1}/{len(tiles)} tiles...")

    # Compute global statistics
    if all_coords:
        all_coords = torch.cat(all_coords, dim=0)
        results['global_stats']['coord_mean_global'] = all_coords.mean(dim=0).tolist()
        results['global_stats']['coord_std_global'] = all_coords.std(dim=0).tolist()

        logger.info(f"\nGlobal coordinate statistics ({dataset_name}):")
        logger.info(f"  X: mean={results['global_stats']['coord_mean_global'][0]:.6f}, "
                   f"std={results['global_stats']['coord_std_global'][0]:.6f}")
        logger.info(f"  Y: mean={results['global_stats']['coord_mean_global'][1]:.6f}, "
                   f"std={results['global_stats']['coord_std_global'][1]:.6f}")
        logger.info(f"  Z: mean={results['global_stats']['coord_mean_global'][2]:.6f}, "
                   f"std={results['global_stats']['coord_std_global'][2]:.6f}")

    # Summary
    logger.info(f"\n{dataset_name.upper()} VALIDATION SUMMARY:")
    logger.info(f"  Valid tiles: {results['valid_tiles']}/{results['total_tiles']}")
    logger.info(f"  Invalid tiles: {results['invalid_tiles']}/{results['total_tiles']}")

    if results['invalid_tiles'] > 0:
        logger.error(f"  ✗ {results['invalid_tiles']} tiles have errors!")
        for tile_result in results['tiles_with_errors'][:5]:  # Show first 5
            logger.error(f"    Tile {tile_result['tile_id']}: {tile_result['errors']}")
    else:
        logger.info(f"  ✓ All tiles valid!")

    return results


def main():
    parser = argparse.ArgumentParser(description="Validate preprocessed raster training data")
    parser.add_argument('--training-tiles', required=True, help='Path to training tiles .pt file')
    parser.add_argument('--validation-tiles', required=True, help='Path to validation tiles .pt file')
    parser.add_argument('--output-dir', required=True, help='Output directory for validation report')

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Validate both datasets
    train_results = validate_dataset(Path(args.training_tiles), 'training')
    val_results = validate_dataset(Path(args.validation_tiles), 'validation')

    # Combined results
    combined_results = {
        'training': train_results,
        'validation': val_results,
        'overall': {
            'total_tiles': train_results['total_tiles'] + val_results['total_tiles'],
            'valid_tiles': train_results['valid_tiles'] + val_results['valid_tiles'],
            'invalid_tiles': train_results['invalid_tiles'] + val_results['invalid_tiles']
        }
    }

    # Save report
    report_path = output_dir / 'validation_report.json'
    with open(report_path, 'w') as f:
        json.dump(combined_results, f, indent=2)
    logger.info(f"\n✓ Validation report saved to {report_path}")

    # Final summary
    logger.info(f"\n{'='*80}")
    logger.info("OVERALL VALIDATION SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"Total tiles: {combined_results['overall']['total_tiles']}")
    logger.info(f"Valid tiles: {combined_results['overall']['valid_tiles']}")
    logger.info(f"Invalid tiles: {combined_results['overall']['invalid_tiles']}")

    if combined_results['overall']['invalid_tiles'] == 0:
        logger.info("\n✓ All tiles passed validation!")
    else:
        logger.error(f"\n✗ {combined_results['overall']['invalid_tiles']} tiles failed validation")


if __name__ == '__main__':
    main()
