#!/usr/bin/env python3
"""
Data Augmentation for Raster-Based Model

Extends point cloud augmentation functions to handle fuel metrics rasters [n_bands, h, w].

Key principle:
  - Geometric transforms (rotation, reflection): Applied to BOTH inputs AND fuel_metrics
  - Intensity/noise augmentation: Applied to inputs ONLY
  - Ensures spatial correspondence between inputs and targets

RASTER MODEL AUGMENTATION STRATEGY:
  Addresses distribution gaps between training data and inference sites:
  - Gap 2: Aggressive sparse point removal (simulates ultra-sparse 3DEP tiles)
  - Gap 3: Temporal subsampling for NAIP and UAVSAR (handles natural variation: NAIP 5-10 images, UAVSAR 4-15 frames)
  - Gap 1: Modality dropout handled in training loop (not here)

Usage:
    python data_augmentation_raster.py \\
        --training_tiles precomputed_training_tiles_raster_32bit.pt \\
        --output_path augmented_tiles_raster_32bit.pt
"""

import torch
import sys
import argparse
from pathlib import Path
import logging
import random
import copy

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


# ============================================================================
# RASTER MODEL: Augmentation Configuration (Single Source of Truth)
# ============================================================================

DEFAULT_RASTER_AUGMENTATION_CONFIG = {
    # Geometric transforms (applied to BOTH inputs and targets)
    'rotate_probability': 1.0,
    'reflect_probability': 0.5,
    
    # Point cloud augmentations (inputs only)
    'jitter_probability': 0.0,
    'jitter_xy_scale': 0.05,
    'jitter_z_scale': 0.02,
    
    'add_points_probability': 0.15,
    'add_points_ratio': 0.05,
    'add_points_max_distance': 0.06,
    
    # Regular point removal (mild sparsification)
    'remove_points_probability': 0.0,
    'remove_points_ratio': 0.05,
    
    # RASTER MODEL: Aggressive sparse removal (Gap 2 - simulates ultra-sparse 3DEP tiles)
    # X% of tiles get heavy point removal 
    # Floor of X points ensures grid attention still works
    'aggressive_sparse_probability': 0.7,
    'aggressive_sparse_min_ratio': 0.05,  # Remove at least X% of points
    'aggressive_sparse_max_ratio': 0.9,  # Remove up to X% of points
    'aggressive_sparse_min_points': 20,  # Never go below X points

    # Spatial masking
    'mask_points_probability': 0.0,
    'mask_min_radius': 0.05,
    'mask_max_radius': 0.2,
    'mask_count': 1,
    'mask_min_removal_ratio': 0.5,
    'mask_max_removal_ratio': 0.8,
    
    # Horizontal slice removal
    'remove_horizontal_slice_probability': 0.0,
    'horizontal_slice_min_height': 0.05,
    'horizontal_slice_max_height': 0.2,
    'horizontal_slice_max_position': 0.5,
    'horizontal_slice_min_removal_ratio': 0.5,
    'horizontal_slice_max_removal_ratio': 0.8,
    
    # Temporal augmentations (inputs only)
    'temporal_shift_probability': 0.2,
    'max_shift_days': 90,

    # RASTER MODEL: Temporal sequence subsampling (addresses natural variation in acquisition counts)
    # NAIP: 5-10 images depending on quad boundaries; UAVSAR: 4-15 images across sites
    'naip_temporal_subsample_probability': 0.2,  # Subsample NAIP sequences
    'naip_temporal_subsample_min_frames': 3,  # Minimum NAIP frames 
    'uavsar_temporal_subsample_probability': 0.2,  # Subsample UAVSAR sequences
    'uavsar_temporal_subsample_min_frames': 1,  # Minimum UAVSAR frames
    
    # Attribute augmentations (inputs only)
    'attribute_augment_probability': 0.0,  # Disabled
    'attribute_scale_range': (0.9, 1.1),
    'attribute_shift_range': (-0.1, 0.1),
    
    # Spectral augmentations (imagery inputs only)
    'spectral_band_probability': 0.0,
    'band_scale_range': (0.9, 1.1),
    
    # Sensor effects (imagery inputs only)
    'sensor_effects_probability': 0.0,
    'sensor_effect_strength': 0.1,
    'uavsar_noise_variance': 0.1,
    
    # Validation
    'min_points_after_augmentation': 20,  # RASTER MODEL: Lowered from 100 for sparse tile handling

    # RASTER MODEL: Modality dropout moved to training time (raster_training.py)
    # Set training_modality_dropout_naip and training_modality_dropout_uavsar in model config
    # Keeping these disabled here to avoid double dropout
    'modality_dropout_enabled': False,  # DISABLED - use training-time dropout instead
    'uavsar_dropout': 0.0,
    'naip_dropout': 0.0

}


# ============================================================================
# RASTER MODEL: Aggressive Sparse Point Removal (Gap 2)
# ============================================================================

def aggressive_sparse_removal(tile, min_ratio=0.5, max_ratio=0.9, min_points=50):
    """
    RASTER MODEL: Aggressive point removal to simulate ultra-sparse 3DEP tiles.
    
    This addresses Gap 2: Forest plot tiles may have much sparser point clouds
    than training data (Volcan site). By aggressively removing points during
    augmentation, the model learns to handle sparse inputs.
    
    Parameters:
        tile (dict): Tile with 'dep_points_norm' and optionally 'dep_points_attr_norm'
        min_ratio (float): Minimum ratio of points to remove (0.5 = remove 50%)
        max_ratio (float): Maximum ratio of points to remove (0.9 = remove 90%)
        min_points (int): Floor - never go below this many points (default 50)
        
    Returns:
        dict: Tile with reduced point count (floor of min_points)
    """
    tile_copy = copy.deepcopy(tile)
    
    if 'dep_points_norm' not in tile_copy:
        return tile_copy
    
    dep_points = tile_copy['dep_points_norm']
    n_original = dep_points.shape[0]
    
    # Skip if already very sparse
    if n_original <= min_points:
        return tile_copy
    
    # Random removal ratio in [min_ratio, max_ratio]
    removal_ratio = random.uniform(min_ratio, max_ratio)
    n_keep = int(n_original * (1.0 - removal_ratio))
    
    # Enforce floor
    n_keep = max(n_keep, min_points)
    
    if n_keep < n_original:
        # Random selection of points to keep
        keep_indices = torch.randperm(n_original)[:n_keep]
        
        tile_copy['dep_points_norm'] = dep_points[keep_indices]
        
        # Also update attributes if present
        if 'dep_points_attr_norm' in tile_copy and tile_copy['dep_points_attr_norm'] is not None:
            tile_copy['dep_points_attr_norm'] = tile_copy['dep_points_attr_norm'][keep_indices]
    
    return tile_copy


# ============================================================================
# RASTER MODEL: UAVSAR Temporal Sequence Augmentation (Gap 3)
# ============================================================================

def duplicate_uavsar_temporally(tile, target_frames=15):
    """
    RASTER MODEL: Duplicate UAVSAR temporal sequence to match training distribution.
    
    This addresses Gap 3: Training data has 15 UAVSAR acquisition events (T), but forest plots
    (BluffMesa, NorthBigBear) have only 4. By duplicating acquisition events during
    augmentation, the model learns to handle the temporal encoder with
    varying effective sequence lengths.
    
    UAVSAR format: [T, G_max, n_bands, h, w] where:
      - T = number of acquisition events (temporal dimension)
      - G_max = max images per acquisition event (typically 8)
      - n_bands = 6 polarization bands
      - h, w = spatial dimensions (4×4 pixels)
    
    Strategy: Repeat acquisition events cyclically until reaching target_frames.
    Example: T=4 events → T=15 by repeating [0,1,2,3,0,1,2,3,0,1,2,3,0,1,2]
    
    Parameters:
        tile (dict): Tile with 'uavsar' dict containing 'images' [T, G_max, n_bands, h, w]
        target_frames (int): Target number of acquisition events (default 15)
        
    Returns:
        dict: Tile with duplicated UAVSAR acquisition events
    """
    tile_copy = copy.deepcopy(tile)
    
    if 'uavsar' not in tile_copy or tile_copy['uavsar'] is None:
        return tile_copy
    
    uavsar_data = tile_copy['uavsar']
    
    if 'images' not in uavsar_data or uavsar_data['images'] is None:
        return tile_copy
    
    images = uavsar_data['images']  # [T, G_max, n_bands, h, w]
    
    # Verify 5D format
    if images.ndim != 5:
        logger.warning(f"UAVSAR images have {images.ndim}D, expected 5D. Skipping temporal duplication.")
        return tile_copy
    
    T = images.shape[0]  # Number of acquisition events
    
    # Skip if already at or above target
    if T >= target_frames:
        return tile_copy
    
    # Calculate how many times to repeat
    n_repeats = (target_frames + T - 1) // T  # Ceiling division
    
    # Repeat along temporal dimension (dim=0) and truncate to target
    duplicated = images.repeat(n_repeats, 1, 1, 1, 1)[:target_frames]
    
    tile_copy['uavsar']['images'] = duplicated
    
    # Also duplicate attention_mask if present [T, G_max]
    if 'attention_mask' in uavsar_data and uavsar_data['attention_mask'] is not None:
        mask = uavsar_data['attention_mask']
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            duplicated_mask = mask.repeat(n_repeats, 1)[:target_frames]
            tile_copy['uavsar']['attention_mask'] = duplicated_mask
    
    # Also duplicate relative_dates if present [T, 1]
    if 'relative_dates' in uavsar_data and uavsar_data['relative_dates'] is not None:
        rel_dates = uavsar_data['relative_dates']
        if isinstance(rel_dates, torch.Tensor):
            if rel_dates.ndim == 2:  # [T, 1]
                duplicated_dates = rel_dates.repeat(n_repeats, 1)[:target_frames]
            else:  # [T]
                duplicated_dates = rel_dates.repeat(n_repeats)[:target_frames]
            tile_copy['uavsar']['relative_dates'] = duplicated_dates
    
    return tile_copy


def subsample_uavsar_temporally(tile, min_frames=1):
    """
    RASTER MODEL: Subsample UAVSAR temporal sequence to vary sequence length.
    
    This provides data augmentation by randomly reducing the number of UAVSAR
    acquisition events, helping the model be robust to varying temporal coverage.
    
    UAVSAR format: [T, G_max, n_bands, h, w]
    
    Strategy: Randomly select a subset of acquisition events (preserving temporal order).
    
    Parameters:
        tile (dict): Tile with 'uavsar' dict containing 'images' [T, G_max, n_bands, h, w]
        min_frames (int): Minimum number of acquisition events to keep (default 1)
        
    Returns:
        dict: Tile with subsampled UAVSAR acquisition events
    """
    tile_copy = copy.deepcopy(tile)
    
    if 'uavsar' not in tile_copy or tile_copy['uavsar'] is None:
        return tile_copy
    
    uavsar_data = tile_copy['uavsar']
    
    if 'images' not in uavsar_data or uavsar_data['images'] is None:
        return tile_copy
    
    images = uavsar_data['images']  # [T, G_max, n_bands, h, w]
    
    # Verify 5D format
    if images.ndim != 5:
        logger.warning(f"UAVSAR images have {images.ndim}D, expected 5D. Skipping temporal subsampling.")
        return tile_copy
    
    T = images.shape[0]  # Number of acquisition events
    
    # Skip if already at minimum
    if T <= min_frames:
        return tile_copy
    
    # Random number of events to keep (between min_frames and T-1)
    n_keep = random.randint(min_frames, T - 1)
    
    # Sorted random indices to preserve temporal order
    keep_indices = sorted(random.sample(range(T), n_keep))
    keep_indices = torch.tensor(keep_indices)
    
    tile_copy['uavsar']['images'] = images[keep_indices]
    
    # Also subsample attention_mask if present [T, G_max]
    if 'attention_mask' in uavsar_data and uavsar_data['attention_mask'] is not None:
        mask = uavsar_data['attention_mask']
        if isinstance(mask, torch.Tensor) and mask.ndim == 2:
            tile_copy['uavsar']['attention_mask'] = mask[keep_indices]
    
    # Also subsample relative_dates if present [T, 1] or [T]
    if 'relative_dates' in uavsar_data and uavsar_data['relative_dates'] is not None:
        rel_dates = uavsar_data['relative_dates']
        if isinstance(rel_dates, torch.Tensor):
            tile_copy['uavsar']['relative_dates'] = rel_dates[keep_indices]
    
    return tile_copy


def subsample_naip_temporally(tile, min_frames=2):
    """
    RASTER MODEL: Subsample NAIP temporal sequence to vary sequence length.

    This provides data augmentation by randomly reducing the number of NAIP
    acquisition images, helping the model be robust to varying temporal coverage.
    Natural variation exists (5-10 images depending on NAIP quad boundaries), and
    this augmentation extends that variability.

    NAIP format: [n_images, n_bands, h, w] where n_bands=4 (RGBN)

    Strategy: Randomly select a subset of images (preserving temporal order).

    Parameters:
        tile (dict): Tile with 'naip' dict containing 'images' [n_images, 4, h, w]
        min_frames (int): Minimum number of images to keep (default 2)

    Returns:
        dict: Tile with subsampled NAIP images
    """
    tile_copy = copy.deepcopy(tile)

    if 'naip' not in tile_copy or tile_copy['naip'] is None:
        return tile_copy

    naip_data = tile_copy['naip']

    if 'images' not in naip_data or naip_data['images'] is None:
        return tile_copy

    images = naip_data['images']  # [n_images, 4, h, w]

    # Verify 4D format
    if images.ndim != 4:
        logger.warning(f"NAIP images have {images.ndim}D, expected 4D. Skipping temporal subsampling.")
        return tile_copy

    n_images = images.shape[0]  # Number of NAIP images

    # Skip if already at minimum
    if n_images <= min_frames:
        return tile_copy

    # Random number of images to keep (between min_frames and n_images-1)
    n_keep = random.randint(min_frames, n_images - 1)

    # Sorted random indices to preserve temporal order
    keep_indices = sorted(random.sample(range(n_images), n_keep))
    keep_indices = torch.tensor(keep_indices)

    tile_copy['naip']['images'] = images[keep_indices]

    # Also subsample relative_dates if present [n_images, 1] or [n_images]
    if 'relative_dates' in naip_data and naip_data['relative_dates'] is not None:
        rel_dates = naip_data['relative_dates']
        if isinstance(rel_dates, torch.Tensor):
            tile_copy['naip']['relative_dates'] = rel_dates[keep_indices]

    return tile_copy


# ============================================================================
# Core Functions
# ============================================================================


def denormalize_to_physical_space(tile):
    """
    Convert z-score normalized points back to bbox-normalized (physical space).

    Denormalization: x_bbox_norm = x_zscore * std + mean

    Returns tile with dep_points_norm temporarily replaced with bbox-normalized values.
    Stores original z-score normalized points for later restoration.
    """
    if 'dep_points_norm' not in tile or 'norm_params' not in tile:
        return tile

    tile_copy = tile.copy()
    zscore_points = tile_copy['dep_points_norm']
    norm_params = tile_copy['norm_params']

    # Extract denormalization parameters
    coord_mean = norm_params['coord_mean'].clone().detach().to(dtype=zscore_points.dtype, device=zscore_points.device)
    coord_std = norm_params['coord_std'].clone().detach().to(dtype=zscore_points.dtype, device=zscore_points.device)

    # Denormalize: x_physical = x_zscore * std + mean
    bbox_norm_points = zscore_points * coord_std + coord_mean

    # Store original z-score points and replace with physical space version
    tile_copy['_original_zscore_points'] = zscore_points
    tile_copy['dep_points_norm'] = bbox_norm_points

    return tile_copy


def renormalize_from_physical_space(tile):
    """
    Convert augmented bbox-normalized (physical space) points back to z-score normalized.

    Normalization: x_zscore = (x_bbox_norm - mean) / std

    Restores z-score normalized representation after augmentation.
    """
    if 'norm_params' not in tile or '_original_zscore_points' not in tile:
        return tile

    tile_copy = tile.copy()
    bbox_norm_points = tile_copy['dep_points_norm']
    norm_params = tile_copy['norm_params']

    # Extract normalization parameters
    coord_mean = norm_params['coord_mean'].clone().detach().to(dtype=bbox_norm_points.dtype, device=bbox_norm_points.device)
    coord_std = norm_params['coord_std'].clone().detach().to(dtype=bbox_norm_points.dtype, device=bbox_norm_points.device)

    # Re-normalize: x_zscore = (x_physical - mean) / std
    zscore_points = (bbox_norm_points - coord_mean) / coord_std

    # Restore z-score normalization and clean up temporary storage
    tile_copy['dep_points_norm'] = zscore_points
    del tile_copy['_original_zscore_points']

    return tile_copy


def rotate_fuel_metrics(tile, angle_degrees=None):
    """
    Rotate fuel metrics raster [n_bands, h, w] by specified angle.

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

    fuel_metrics = tile_copy['fuel_metrics']  # [n_bands, h, w]

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
    Reflect fuel metrics raster [n_bands, h, w] across specified axis.

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

    fuel_metrics = tile_copy['fuel_metrics']  # [n_bands, h, w]

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
    
    RASTER MODEL: Includes aggressive sparse removal and UAVSAR temporal augmentation
    to address distribution gaps with forest plot inference data.

    Parameters:
        tile (dict): Tile to augment
        config (dict): Augmentation configuration (uses DEFAULT_RASTER_AUGMENTATION_CONFIG if None)

    Returns:
        dict: Augmented tile
    """
    # Use global default config if not provided
    if config is None:
        config = DEFAULT_RASTER_AUGMENTATION_CONFIG.copy()

    augmented_tile = copy.deepcopy(tile)

    try:
        # CRITICAL: Convert from z-score normalized space to physical (bbox-normalized) space
        # All augmentations happen in physical space, then convert back at the end
        augmented_tile = denormalize_to_physical_space(augmented_tile)

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

        # ================================================================
        # Input-only augmentations (NOT applied to fuel_metrics)
        # ================================================================
        
        # Point jitter
        if random.random() < config['jitter_probability']:
            augmented_tile = jitter_points(
                augmented_tile,
                xy_scale=config['jitter_xy_scale'],
                z_scale=config['jitter_z_scale']
            )

        # Add nearby points
        if random.random() < config['add_points_probability']:
            augmented_tile = add_nearby_points(
                augmented_tile,
                ratio=config['add_points_ratio'],
                max_distance=config['add_points_max_distance']
            )

        # Regular point removal (mild)
        if random.random() < config['remove_points_probability']:
            augmented_tile = randomly_remove_points(
                augmented_tile,
                ratio=config['remove_points_ratio']
            )
        
        # RASTER MODEL: Aggressive sparse point removal (Gap 2)
        # Applied separately from regular removal - simulates ultra-sparse 3DEP tiles
        if random.random() < config.get('aggressive_sparse_probability', 0.0):
            augmented_tile = aggressive_sparse_removal(
                augmented_tile,
                min_ratio=config.get('aggressive_sparse_min_ratio', 0.5),
                max_ratio=config.get('aggressive_sparse_max_ratio', 0.9),
                min_points=config.get('aggressive_sparse_min_points', 50)
            )

        # Spatial masking
        if random.random() < config['mask_points_probability']:
            augmented_tile = mask_points(
                augmented_tile,
                min_radius=config['mask_min_radius'],
                max_radius=config['mask_max_radius'],
                n_masks=config['mask_count'],
                min_removal_ratio=config['mask_min_removal_ratio'],
                max_removal_ratio=config['mask_max_removal_ratio']
            )

        # Horizontal slice removal
        if random.random() < config['remove_horizontal_slice_probability']:
            augmented_tile = remove_horizontal_slice(
                augmented_tile,
                min_slice_height=config['horizontal_slice_min_height'],
                max_slice_height=config['horizontal_slice_max_height'],
                max_slice_position=config['horizontal_slice_max_position'],
                min_removal_ratio=config['horizontal_slice_min_removal_ratio'],
                max_removal_ratio=config['horizontal_slice_max_removal_ratio']
            )

        # Temporal shift (disabled by default)
        if random.random() < config['temporal_shift_probability']:
            augmented_tile = shift_temporal_sequence(
                augmented_tile,
                max_shift_days=config['max_shift_days']
            )

        # RASTER MODEL: Temporal sequence subsampling
        # Subsample NAIP sequences to add temporal variation (natural variation: 5-10 images)
        if random.random() < config.get('naip_temporal_subsample_probability', 0.0):
            augmented_tile = subsample_naip_temporally(
                augmented_tile,
                min_frames=config.get('naip_temporal_subsample_min_frames', 2)
            )

        # Subsample UAVSAR sequences to add temporal variation
        if random.random() < config.get('uavsar_temporal_subsample_probability', 0.0):
            augmented_tile = subsample_uavsar_temporally(
                augmented_tile,
                min_frames=config.get('uavsar_temporal_subsample_min_frames', 2)
            )

        # Attribute augmentation (disabled by default)
        if random.random() < config['attribute_augment_probability']:
            augmented_tile = augment_attributes(
                augmented_tile,
                scale_range=config['attribute_scale_range'],
                shift_range=config['attribute_shift_range']
            )

        # Spectral band augmentation
        if random.random() < config['spectral_band_probability']:
            augmented_tile = augment_spectral_bands(
                augmented_tile,
                band_scale_range=config['band_scale_range']
            )

        # Sensor effects simulation
        if random.random() < config['sensor_effects_probability']:
            augmented_tile = simulate_sensor_effects(
                augmented_tile,
                effect_strength=config['sensor_effect_strength'],
                speckle_variance=config['uavsar_noise_variance']
            )

        # CRITICAL: Convert back from physical (bbox-normalized) space to z-score normalized space
        augmented_tile = renormalize_from_physical_space(augmented_tile)

        # Regenerate KNN indices
        if 'knn_edge_indices' in augmented_tile:
            from torch_geometric.nn import knn_graph
            from torch_geometric.utils import to_undirected

            for k in augmented_tile['knn_edge_indices']:
                edge_index = knn_graph(augmented_tile['dep_points_norm'], k=k, loop=False)
                edge_index = to_undirected(edge_index, num_nodes=augmented_tile['dep_points_norm'].size(0))
                augmented_tile['knn_edge_indices'][k] = edge_index

        # === DEPRECATED: Modality dropout moved to training time ===
        # Modality dropout is now applied in raster_training.py during training
        # using config parameters: training_modality_dropout_naip, training_modality_dropout_uavsar
        # This block is kept for backwards compatibility but disabled by default
        if config.get('modality_dropout_enabled', False):
            uavsar_drop_prob = config.get('uavsar_dropout', 0.0)
            naip_drop_prob = config.get('naip_dropout', 0.0)

            if uavsar_drop_prob > 0 and random.random() < uavsar_drop_prob:
                augmented_tile['uavsar'] = None

            if naip_drop_prob > 0 and random.random() < naip_drop_prob:
                augmented_tile['naip'] = None

        return augmented_tile

    except Exception as e:
        logger.error(f"Error during augmentation: {str(e)}")
        return copy.deepcopy(tile)


def validate_augmented_tile_raster(original_tile, augmented_tile, max_allowed_value=500.0, 
                                    min_points=None):
    """
    Validate augmented tile with raster targets.

    Parameters:
        original_tile (dict): Original tile
        augmented_tile (dict): Augmented tile to validate
        max_allowed_value (float): Max allowed value for any tensor
        min_points (int): Minimum points required (default from config: 50)

    Returns:
        dict: Validated augmented tile or raises ValueError
    """
    # RASTER MODEL: Use config default if not specified
    if min_points is None:
        min_points = DEFAULT_RASTER_AUGMENTATION_CONFIG.get('min_points_after_augmentation', 50)
    
    # Check basic keys
    essential_keys = ['dep_points_norm']
    for key in essential_keys:
        if key in original_tile and key not in augmented_tile:
            raise ValueError(f"Missing essential key: {key}")

    # Check DEP points
    if 'dep_points_norm' in augmented_tile:
        points = augmented_tile['dep_points_norm']
        # RASTER MODEL: Lowered from 100 to support aggressive sparse removal
        if points.shape[0] < min_points:
            raise ValueError(f"Too few DEP points after augmentation: {points.shape[0]} (min: {min_points})")
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
            # Check dimensions [n_bands, h, w] - don't hardcode band count
            if len(fm.shape) != 3:
                raise ValueError(f"Invalid fuel_metrics shape: {fm.shape}, expected [n_bands, h, w]")
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
    parser.add_argument(
        '--disable_aggressive_sparse',
        action='store_true',
        help='Disable aggressive sparse point removal (Gap 2)'
    )
    parser.add_argument(
        '--disable_temporal_subsampling',
        action='store_true',
        help='Disable temporal subsampling for both NAIP and UAVSAR'
    )

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("DATA AUGMENTATION FOR RASTER MODEL")
    logger.info("="*80)

    # Load training tiles
    logger.info(f"Loading training tiles from {args.training_tiles}...")
    training_tiles = torch.load(args.training_tiles, weights_only=False)
    logger.info(f"Loaded {len(training_tiles)} tiles")

    # Use centralized config (single source of truth)
    config = DEFAULT_RASTER_AUGMENTATION_CONFIG.copy()
    
    # Apply command-line overrides if specified
    if args.disable_aggressive_sparse:
        config['aggressive_sparse_probability'] = 0.0
        logger.info("Aggressive sparse point removal DISABLED")

    if args.disable_temporal_subsampling:
        config['naip_temporal_subsample_probability'] = 0.0
        config['uavsar_temporal_subsample_probability'] = 0.0
        logger.info("Temporal subsampling (NAIP and UAVSAR) DISABLED")

    # Log key augmentation settings
    logger.info("\nAugmentation Configuration:")
    logger.info(f"  Geometric: rotate={config['rotate_probability']:.0%}, reflect={config['reflect_probability']:.0%}")
    logger.info(f"  Point removal (mild): {config['remove_points_probability']:.0%} prob, {config['remove_points_ratio']:.0%} ratio")
    logger.info(f"  Point removal (aggressive): {config['aggressive_sparse_probability']:.0%} prob, "
                f"{config['aggressive_sparse_min_ratio']:.0%}-{config['aggressive_sparse_max_ratio']:.0%} ratio, "
                f"floor={config['aggressive_sparse_min_points']} pts")
    logger.info(f"  Temporal subsampling: NAIP={config['naip_temporal_subsample_probability']:.0%} (min={config['naip_temporal_subsample_min_frames']}), "
                f"UAVSAR={config['uavsar_temporal_subsample_probability']:.0%} (min={config['uavsar_temporal_subsample_min_frames']})")
    logger.info(f"  Min points after augmentation: {config['min_points_after_augmentation']}")

    # Augment dataset
    logger.info(f"\nAugmenting dataset ({args.n_augmentations} per tile)...")
    augmented_tiles = []
    failed_count = 0

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
                failed_count += 1
                continue

        if (tile_idx + 1) % 1000 == 0:
            logger.info(f"  Augmented {tile_idx + 1}/{len(training_tiles)} tiles "
                       f"({len(augmented_tiles)} successful, {failed_count} failed)")

    logger.info(f"\nAugmentation complete: {len(augmented_tiles)} augmented tiles created")
    if failed_count > 0:
        logger.warning(f"  {failed_count} augmentations failed")

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
