#!/usr/bin/env python3
"""
Data Augmentation for Raster-Based Model

Extends point cloud augmentation functions to handle fuel metrics rasters [22, h, w].

Key principle:
  - Geometric transforms (rotation, reflection): Applied to BOTH inputs AND fuel_metrics
  - Intensity/noise augmentation: Applied to inputs ONLY
  - Ensures spatial correspondence between inputs and targets

Usage:
    python data_augmentation_raster.py \
        --training_tiles precomputed_training_tiles_raster_32bit.pt \
        --output_path augmented_tiles_raster_32bit.pt
"""

import torch
import sys
import argparse
from pathlib import Path
import logging

# Import base augmentation functions
sys.path.insert(0, str(Path(__file__).parent))
from data_augmentation import (
    randomly_remove_points,
    add_nearby_points,
    mask_points,
    shift_temporal_sequence,
    augment_attributes,
    rotate_tile,
    reflect_tile,
    jitter_points,
    augment_spectral_bands,
    simulate_sensor_effects,
    remove_horizontal_slice,
    validate_augmented_tile,
    augment_dataset,
    randomly_augment_tile
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def rotate_fuel_metrics(tile, angle_degrees=None):
    """
    Rotate fuel metrics raster [22, h, w] by specified angle.

    Parameters:
        tile (dict): Tile with 'fuel_metrics' key
        angle_degrees (float, optional): Rotation angle in degrees (90, 180, 270)

    Returns:
        dict: Modified tile with rotated fuel_metrics
    """
    import copy
    import random
    import numpy as np

    tile_copy = copy.deepcopy(tile)

    if 'fuel_metrics' not in tile_copy or tile_copy['fuel_metrics'] is None:
        return tile_copy

    if angle_degrees is None:
        angle_degrees = random.choice([90, 180, 270])

    fuel_metrics = tile_copy['fuel_metrics']  # [22, h, w]

    # Apply rotation to spatial dimensions (height, width)
    if angle_degrees == 90:
        # Permute h,w then flip along width
        rotated = fuel_metrics.permute(0, 2, 1).flip(dims=[2])
    elif angle_degrees == 180:
        # Flip along both height and width
        rotated = fuel_metrics.flip(dims=[1, 2])
    elif angle_degrees == 270:
        # Permute h,w then flip along height
        rotated = fuel_metrics.permute(0, 2, 1).flip(dims=[1])
    else:
        rotated = fuel_metrics

    tile_copy['fuel_metrics'] = rotated
    return tile_copy


def reflect_fuel_metrics(tile, axis='x'):
    """
    Reflect fuel metrics raster [22, h, w] across specified axis.

    Parameters:
        tile (dict): Tile with 'fuel_metrics' key
        axis (str): Axis to reflect ('x', 'y', or 'both')

    Returns:
        dict: Modified tile with reflected fuel_metrics
    """
    import copy

    tile_copy = copy.deepcopy(tile)

    if 'fuel_metrics' not in tile_copy or tile_copy['fuel_metrics'] is None:
        return tile_copy

    fuel_metrics = tile_copy['fuel_metrics']  # [22, h, w]

    # Determine which dimensions to flip
    if axis == 'x':
        flip_dims = [2]  # Flip along width
    elif axis == 'y':
        flip_dims = [1]  # Flip along height
    elif axis == 'both':
        flip_dims = [1, 2]  # Flip both
    else:
        return tile_copy

    tile_copy['fuel_metrics'] = fuel_metrics.flip(dims=flip_dims)
    return tile_copy


def augment_tile_with_rasters(tile, config=None):
    """
    Apply augmentation to tile with synchronized geometric transforms.

    Geometric transforms (rotate, reflect) are applied to BOTH inputs and targets.
    Intensity/noise augmentations are applied to inputs only.

    Parameters:
        tile (dict): Tile to augment
        config (dict): Augmentation configuration

    Returns:
        dict: Augmented tile
    """
    import copy
    import random

    if config is None:
        config = {
            'rotate_probability': 1.0,
            'reflect_probability': 0.5,
            'jitter_probability': 0.3,
            'add_points_probability': 0.2,
            'remove_points_probability': 0.5,
            'mask_points_probability': 0.5,
            'remove_horizontal_slice_probability': 0.5,
            'temporal_shift_probability': 0.4,
            'attribute_augment_probability': 0.4,
            'spectral_band_probability': 0.3,
            'sensor_effects_probability': 0.3,
            'max_shift_days': 30,
            'jitter_xy_scale': 0.02,
            'jitter_z_scale': 0.01,
            'attribute_scale_range': (0.9, 1.1),
            'attribute_shift_range': (-0.1, 0.1),
            'band_scale_range': (0.9, 1.1),
            'add_points_ratio': 0.1,
            'add_points_max_distance': 0.02,
            'remove_points_ratio': 0.1,
            'mask_min_radius': 0.05,
            'mask_max_radius': 0.2,
            'mask_count': 1,
            'mask_min_removal_ratio': 0.7,
            'mask_max_removal_ratio': 1.0,
            'sensor_effect_strength': 0.2,
            'uavsar_noise_variance': 0.1,
            'horizontal_slice_min_height': 0.05,
            'horizontal_slice_max_height': 0.2,
            'horizontal_slice_max_position': 0.5,
            'horizontal_slice_min_removal_ratio': 0.7,
            'horizontal_slice_max_removal_ratio': 1.0,
        }

    augmented_tile = copy.deepcopy(tile)

    try:
        # Store original fuel_metrics for synchronized transforms
        original_fm = augmented_tile.get('fuel_metrics')

        # Randomly choose geometric transform
        random_value = random.random()

        if random_value < config['rotate_probability']:
            # Apply synchronized rotation
            angle = random.choice([90, 180, 270])
            augmented_tile = rotate_tile(augmented_tile, angle_degrees=angle)
            if original_fm is not None:
                # Also rotate fuel_metrics with same angle
                augmented_tile = rotate_fuel_metrics(augmented_tile, angle_degrees=angle)

        elif random_value < config['rotate_probability'] + config['reflect_probability']:
            # Apply synchronized reflection
            axis = random.choice(['x', 'y', 'both'])
            augmented_tile = reflect_tile(augmented_tile, axis=axis)
            if original_fm is not None:
                # Also reflect fuel_metrics with same axis
                augmented_tile = reflect_fuel_metrics(augmented_tile, axis=axis)

        # Input-only augmentations (NOT applied to fuel_metrics)
        if random.random() < config['jitter_probability']:
            augmented_tile = jitter_points(
                augmented_tile,
                xy_scale=config['jitter_xy_scale'],
                z_scale=config['jitter_z_scale']
            )

        if random.random() < config['add_points_probability']:
            augmented_tile = add_nearby_points(
                augmented_tile,
                ratio=config['add_points_ratio'],
                max_distance=config['add_points_max_distance']
            )

        if random.random() < config['remove_points_probability']:
            augmented_tile = randomly_remove_points(
                augmented_tile,
                ratio=config['remove_points_ratio']
            )

        if random.random() < config['mask_points_probability']:
            augmented_tile = mask_points(
                augmented_tile,
                min_radius=config['mask_min_radius'],
                max_radius=config['mask_max_radius'],
                n_masks=config['mask_count'],
                min_removal_ratio=config['mask_min_removal_ratio'],
                max_removal_ratio=config['mask_max_removal_ratio']
            )

        if random.random() < config['remove_horizontal_slice_probability']:
            augmented_tile = remove_horizontal_slice(
                augmented_tile,
                min_slice_height=config['horizontal_slice_min_height'],
                max_slice_height=config['horizontal_slice_max_height'],
                max_slice_position=config['horizontal_slice_max_position'],
                min_removal_ratio=config['horizontal_slice_min_removal_ratio'],
                max_removal_ratio=config['horizontal_slice_max_removal_ratio']
            )

        if random.random() < config['temporal_shift_probability']:
            augmented_tile = shift_temporal_sequence(
                augmented_tile,
                max_shift_days=config['max_shift_days']
            )

        if random.random() < config['attribute_augment_probability']:
            augmented_tile = augment_attributes(
                augmented_tile,
                scale_range=config['attribute_scale_range'],
                shift_range=config['attribute_shift_range']
            )

        if random.random() < config['spectral_band_probability']:
            augmented_tile = augment_spectral_bands(
                augmented_tile,
                band_scale_range=config['band_scale_range']
            )

        if random.random() < config['sensor_effects_probability']:
            augmented_tile = simulate_sensor_effects(
                augmented_tile,
                effect_strength=config['sensor_effect_strength'],
                speckle_variance=config['uavsar_noise_variance']
            )

        # Regenerate KNN indices
        if 'knn_edge_indices' in augmented_tile:
            from torch_geometric.nn import knn_graph
            from torch_geometric.utils import to_undirected

            for k in augmented_tile['knn_edge_indices']:
                edge_index = knn_graph(augmented_tile['dep_points_norm'], k=k, loop=False)
                edge_index = to_undirected(edge_index, num_nodes=augmented_tile['dep_points_norm'].size(0))
                augmented_tile['knn_edge_indices'][k] = edge_index

        return augmented_tile

    except Exception as e:
        logger.error(f"Error during augmentation: {str(e)}")
        return copy.deepcopy(tile)


def validate_augmented_tile_raster(original_tile, augmented_tile, max_allowed_value=500.0):
    """
    Validate augmented tile with raster targets.

    Parameters:
        original_tile (dict): Original tile
        augmented_tile (dict): Augmented tile to validate
        max_allowed_value (float): Max allowed value for any tensor

    Returns:
        dict: Validated augmented tile or raises ValueError
    """
    # Check basic keys
    essential_keys = ['dep_points_norm']
    for key in essential_keys:
        if key in original_tile and key not in augmented_tile:
            raise ValueError(f"Missing essential key: {key}")

    # Check DEP points
    if 'dep_points_norm' in augmented_tile:
        points = augmented_tile['dep_points_norm']
        if points.shape[0] < 100:
            raise ValueError(f"Too few DEP points after augmentation: {points.shape[0]}")
        if torch.isnan(points).any():
            raise ValueError("NaN values in dep_points_norm")
        if torch.isinf(points).any():
            raise ValueError("Inf values in dep_points_norm")
        if torch.abs(points).max() > max_allowed_value:
            raise ValueError(f"Extreme values in dep_points_norm: {torch.abs(points).max().item()}")

    # Check fuel metrics
    if 'fuel_metrics' in augmented_tile:
        fm = augmented_tile['fuel_metrics']
        if fm is not None:
            # Check dimensions [22, h, w]
            if len(fm.shape) != 3 or fm.shape[0] != 22:
                raise ValueError(f"Invalid fuel_metrics shape: {fm.shape}, expected [22, h, w]")
            # Check for NaN/Inf (but allow -999 sentinel value)
            if torch.isnan(fm).any():
                raise ValueError("NaN values in fuel_metrics")
            if torch.isinf(fm).any():
                raise ValueError("Inf values in fuel_metrics")
            # Note: -999 is a valid sentinel value for missing data, not an error

    return augmented_tile


def main():
    parser = argparse.ArgumentParser(
        description="Data augmentation for raster-based model"
    )
    parser.add_argument(
        '--training_tiles',
        required=True,
        help='Path to precomputed training tiles .pt file'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help='Output path for augmented tiles'
    )
    parser.add_argument(
        '--n_augmentations',
        type=int,
        default=1,
        help='Number of augmentations per tile (default: 1)'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("DATA AUGMENTATION FOR RASTER MODEL")
    logger.info("="*80)

    # Load training tiles
    logger.info(f"Loading training tiles from {args.training_tiles}...")
    training_tiles = torch.load(args.training_tiles, weights_only=False)
    logger.info(f"Loaded {len(training_tiles)} tiles")

    # Configure augmentation
    config = {
        'rotate_probability': 1.0,
        'reflect_probability': 0.5,
        'jitter_probability': 0.2,
        'add_points_probability': 0.1,
        'remove_points_probability': 0.3,
        'mask_points_probability': 0.2,
        'remove_horizontal_slice_probability': 0.2,
        'temporal_shift_probability': 0.0,
        'attribute_augment_probability': 0.0,
        'spectral_band_probability': 0.2,
        'sensor_effects_probability': 0.2,
        'max_shift_days': 30,
        'jitter_xy_scale': 0.02,
        'jitter_z_scale': 0.01,
        'attribute_scale_range': (0.9, 1.1),
        'attribute_shift_range': (-0.1, 0.1),
        'band_scale_range': (0.9, 1.1),
        'add_points_ratio': 0.05,
        'add_points_max_distance': 0.02,
        'remove_points_ratio': 0.05,
        'mask_min_radius': 0.05,
        'mask_max_radius': 0.2,
        'mask_count': 1,
        'mask_min_removal_ratio': 0.5,
        'mask_max_removal_ratio': 0.8,
        'sensor_effect_strength': 0.1,
        'uavsar_noise_variance': 0.1,
        'horizontal_slice_min_height': 0.05,
        'horizontal_slice_max_height': 0.2,
        'horizontal_slice_max_position': 0.5,
        'horizontal_slice_min_removal_ratio': 0.5,
        'horizontal_slice_max_removal_ratio': 0.8,
    }

    # Augment dataset
    logger.info(f"Augmenting dataset ({args.n_augmentations} per tile)...")
    augmented_tiles = []

    for tile_idx, tile in enumerate(training_tiles):
        for aug_idx in range(args.n_augmentations):
            try:
                augmented_tile = augment_tile_with_rasters(tile, config)
                validate_augmented_tile_raster(tile, augmented_tile)

                # Add augmentation index to tile_id if present
                if 'tile_id' in augmented_tile:
                    augmented_tile['tile_id'] = f"{augmented_tile['tile_id']}_aug_{aug_idx + 1}"

                augmented_tiles.append(augmented_tile)

            except Exception as e:
                logger.warning(f"Augmentation failed for tile {tile_idx}, aug {aug_idx}: {str(e)}")
                continue

        if (tile_idx + 1) % 100 == 0:
            logger.info(f"  Augmented {tile_idx + 1}/{len(training_tiles)} tiles ({len(augmented_tiles)} total augmentations)")

    logger.info(f"\nAugmentation complete: {len(augmented_tiles)} augmented tiles created")

    # Save augmented dataset
    logger.info(f"Saving augmented tiles to {args.output_path}...")
    torch.save(augmented_tiles, args.output_path)
    logger.info(f"Saved to {args.output_path}")

    logger.info("\n" + "="*80)
    logger.info("AUGMENTATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Original tiles: {len(training_tiles)}")
    logger.info(f"Augmented tiles: {len(augmented_tiles)}")


if __name__ == '__main__':
    main()
