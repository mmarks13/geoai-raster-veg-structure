#!/usr/bin/env python3
"""
Preprocess forest plot tiles for inference with trained raster model.

This script takes combined forest plot tiles (output of h5_chunk_loader.py) and applies
the SAME normalization as training data, using pre-computed TRAINING statistics.

Key differences from train_test_split_and_precompute_raster.py:
1. NO train/val splitting - all tiles go to one output
2. NO computing new statistics - uses existing TRAINING stats
3. NO fuel_metrics normalization - forest plots don't have fuel_metrics ground truth
4. Computes diagnostic statistics to report distribution shift vs training

Input:
    - Combined .pt file from h5_chunk_loader.py (e.g., combined_forest_plots.pt)
    - Training normalization stats (coordinate_normalization_stats.json)

Output:
    - precomputed_forest_plot_tiles_32bit.pt with:
        - dep_points_norm: z-score normalized coordinates
        - dep_points_bbox_norm: bbox-normalized coordinates (meters)
        - dep_points_attr_norm: z-score normalized attributes
        - knn_edge_indices: precomputed KNN graphs (k=15)
        - naip: preprocessed NAIP imagery dict
        - uavsar: preprocessed UAVSAR imagery dict
        - center, scale: bbox normalization parameters
        - norm_params: training normalization stats (for denormalization)

Usage:
    python src/data_prep/preprocess_forest_plots_for_inference.py \
        --pt-file data/processed/forest_plot_data/tiles/combined_forest_plots.pt \
        --training-stats-dir data/processed/model_data_raster \
        --output-dir data/processed/forest_plot_data/inference_ready \
        --min-dep-points 50
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


##########################################
# Point Cloud Normalization Functions
# (Adapted from train_test_split_and_precompute_raster.py)
##########################################

def normalize_point_clouds_with_bbox(
    dep_points: torch.Tensor,
    uav_points: torch.Tensor,
    bbox: torch.Tensor,
    dtype: torch.dtype = torch.float32
) -> tuple:
    """
    Normalize point clouds using bbox-based centering and scaling.

    X,Y: Centered at bbox center, scaled to [-5, 5] range (for 10m tiles)
    Z: Shifted so minimum = 0, kept in meters

    Args:
        dep_points: [N_dep, 3] raw coordinates
        uav_points: [N_uav, 3] raw coordinates (can be empty for inference)
        bbox: [4] tensor [xmin, ymin, xmax, ymax]
        dtype: Output dtype

    Returns:
        dep_points_norm: Bbox-normalized DEP points
        uav_points_norm: Bbox-normalized UAV points (empty for inference)
        center: [1, 3] normalization center
        scale: [1, 3] normalization scale
    """
    # Get bbox bounds
    xmin, ymin, xmax, ymax = bbox.tolist()

    # Find z range from DEP points
    z_min = dep_points[:, 2].min().item()

    # Calculate x,y centers from the bounding box
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # Create center tensor (for x,y we use bbox center, for z we use min value)
    center = torch.tensor([[center_x, center_y, z_min]],
                         dtype=dep_points.dtype, device=dep_points.device)

    # Calculate x,y scales to map to [-5, 5] range
    # For a 10x10m box, scale factor = 10/10 = 1 (dividing by 1 doesn't change values)
    scale_x = (xmax - xmin) / 10
    scale_y = (ymax - ymin) / 10

    # For z, we use scale of 1 to keep values in meters
    scale_z = 1.0

    # Create scale tensor
    scale = torch.tensor([[scale_x, scale_y, scale_z]],
                         dtype=dep_points.dtype, device=dep_points.device)

    # Apply normalization
    dep_points_norm = dep_points.clone()
    uav_points_norm = uav_points.clone() if uav_points.numel() > 0 else uav_points

    # Handle x,y coordinates (center at 0,0, scale to [-5,5] range)
    dep_points_norm[:, :2] -= center[:, :2]
    dep_points_norm[:, :2] /= scale[:, :2]
    if uav_points_norm.numel() > 0:
        uav_points_norm[:, :2] -= center[:, :2]
        uav_points_norm[:, :2] /= scale[:, :2]

    # Handle z coordinates (shift to make minimum = 0, keep in meters)
    dep_points_norm[:, 2] = dep_points[:, 2] - center[:, 2]
    if uav_points_norm.numel() > 0:
        uav_points_norm[:, 2] = uav_points[:, 2] - center[:, 2]

    # Convert to specified dtype for memory efficiency
    dep_points_norm = dep_points_norm.to(dtype)
    uav_points_norm = uav_points_norm.to(dtype) if uav_points_norm.numel() > 0 else uav_points_norm

    return dep_points_norm, uav_points_norm, center, scale


def apply_zscore_to_bbox_normalized_coords(
    dep_points_bbox_norm: torch.Tensor,
    coord_mean: torch.Tensor,
    coord_std: torch.Tensor
) -> torch.Tensor:
    """
    Apply z-score normalization to bbox-normalized coordinates.

    CRITICAL: Input coordinates are ALREADY bbox-normalized (X,Y ∈ [-5,5]m, Z ∈ [0,max]m).
    This function standardizes the distribution to mean=0, std=1.

    Args:
        dep_points_bbox_norm: [N, 3] bbox-normalized coordinates
        coord_mean: [3] means of bbox-normalized coords (from TRAINING stats)
        coord_std: [3] stds of bbox-normalized coords (from TRAINING stats)

    Returns:
        dep_points_zscore: [N, 3] z-score normalized coordinates
    """
    coord_mean_t = torch.tensor(coord_mean, dtype=dep_points_bbox_norm.dtype, device=dep_points_bbox_norm.device)
    coord_std_t = torch.tensor(coord_std, dtype=dep_points_bbox_norm.dtype, device=dep_points_bbox_norm.device)

    # Z-score: (x - mean) / std
    dep_points_zscore = (dep_points_bbox_norm - coord_mean_t) / coord_std_t

    return dep_points_zscore


def normalize_attributes_zscore(
    dep_pnt_attr: torch.Tensor,
    attr_mean: torch.Tensor,
    attr_std: torch.Tensor,
    dtype: torch.dtype = torch.float16
) -> Optional[torch.Tensor]:
    """
    Normalize point attributes using z-score.

    Handles NaN/Inf by setting to mean before normalization.

    Args:
        dep_pnt_attr: [N, 3] raw attributes
        attr_mean: [3] global means (from TRAINING stats)
        attr_std: [3] global stds (from TRAINING stats)
        dtype: Output dtype (default float16 for memory)

    Returns:
        dep_points_attr_norm: [N, 3] normalized attributes
    """
    if dep_pnt_attr is None:
        return None

    dep_points_attr_norm = dep_pnt_attr.clone()

    # Handle invalid values → set to mean
    invalid_mask = torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)

    attr_mean_t = torch.tensor(attr_mean, dtype=dep_points_attr_norm.dtype, device=dep_points_attr_norm.device)
    attr_std_t = torch.tensor(attr_std, dtype=dep_points_attr_norm.dtype, device=dep_points_attr_norm.device)

    for attr_idx in range(3):
        if invalid_mask[:, attr_idx].any():
            dep_points_attr_norm[:, attr_idx][invalid_mask[:, attr_idx]] = attr_mean_t[attr_idx]

    # Z-score normalization
    dep_points_attr_norm = (dep_points_attr_norm - attr_mean_t) / attr_std_t

    # Set remaining invalid to 0
    dep_points_attr_norm[torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)] = 0.0

    # Convert to specified dtype
    dep_points_attr_norm = dep_points_attr_norm.to(dtype)

    return dep_points_attr_norm


##########################################
# Diagnostic Statistics Functions
##########################################

def compute_forest_plot_statistics(all_tiles: List[Dict]) -> Dict:
    """
    Compute statistics for forest plot tiles (for distribution shift analysis).

    Args:
        all_tiles: List of tile dicts with 'dep_points' and 'dep_pnt_attr'

    Returns:
        dict with coordinate and attribute statistics
    """
    logger.info("Computing forest plot statistics for distribution shift analysis...")

    all_coords_norm = []
    all_attrs = []
    total_points = 0

    for tile_idx, tile in enumerate(all_tiles):
        if 'dep_points' not in tile or tile['dep_points'] is None:
            continue

        dep_points = tile['dep_points'].clone()
        dep_pnt_attr = tile.get('dep_pnt_attr', None)
        bbox = tile.get('bbox', None)

        if bbox is None:
            continue

        # Apply bbox normalization (same as training)
        uav_dummy = torch.empty((0, 3), dtype=torch.float32)
        dep_points_norm, _, _, _ = normalize_point_clouds_with_bbox(
            dep_points, uav_dummy, bbox, dtype=torch.float32
        )

        # Clamp bbox-normalized Z to [0, 150]
        dep_points_norm[:, 2] = torch.clamp(dep_points_norm[:, 2], 0, 150)

        all_coords_norm.append(dep_points_norm)
        total_points += dep_points_norm.shape[0]

        if dep_pnt_attr is not None:
            all_attrs.append(dep_pnt_attr)

        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Processed {tile_idx + 1}/{len(all_tiles)} tiles for stats...")

    if len(all_coords_norm) == 0:
        raise ValueError("No valid tiles found for statistics computation!")

    # Concatenate and compute stats
    all_coords_norm = torch.cat(all_coords_norm, dim=0)

    coord_mean = all_coords_norm.mean(dim=0).numpy()
    coord_std = all_coords_norm.std(dim=0).numpy()

    logger.info(f"  Forest plot coordinate stats (bbox-normalized):")
    logger.info(f"    X: mean={coord_mean[0]:.4f}, std={coord_std[0]:.4f}")
    logger.info(f"    Y: mean={coord_mean[1]:.4f}, std={coord_std[1]:.4f}")
    logger.info(f"    Z: mean={coord_mean[2]:.4f}, std={coord_std[2]:.4f}")

    attr_mean = None
    attr_std = None

    if len(all_attrs) > 0:
        all_attrs = torch.cat(all_attrs, dim=0)
        attr_mean = torch.zeros(3, dtype=torch.float64)
        attr_std = torch.zeros(3, dtype=torch.float64)

        for attr_idx in range(3):
            values = all_attrs[:, attr_idx]
            valid_mask = ~(torch.isnan(values) | torch.isinf(values))
            valid_values = values[valid_mask]

            if len(valid_values) > 0:
                attr_mean[attr_idx] = valid_values.mean()
                attr_std[attr_idx] = valid_values.std()
            else:
                attr_mean[attr_idx] = 0.0
                attr_std[attr_idx] = 1.0

        attr_mean = attr_mean.numpy()
        attr_std = attr_std.numpy()

        logger.info(f"  Forest plot attribute stats:")
        logger.info(f"    Intensity: mean={attr_mean[0]:.4f}, std={attr_std[0]:.4f}")
        logger.info(f"    ReturnNumber: mean={attr_mean[1]:.4f}, std={attr_std[1]:.4f}")
        logger.info(f"    NumberOfReturns: mean={attr_mean[2]:.4f}, std={attr_std[2]:.4f}")

    return {
        'coord_mean': torch.from_numpy(coord_mean).float(),
        'coord_std': torch.from_numpy(coord_std).float(),
        'attr_mean': torch.from_numpy(attr_mean).float() if attr_mean is not None else None,
        'attr_std': torch.from_numpy(attr_std).float() if attr_std is not None else None,
        'total_points': int(total_points),
        'total_tiles': len([t for t in all_tiles if 'dep_points' in t])
    }


def report_distribution_shift(train_stats: Dict, forest_stats: Dict) -> Dict:
    """
    Report distribution shift between training and forest plot data.

    Large shifts (>1.0 std) indicate potential domain mismatch.

    Args:
        train_stats: Dict with 'coord_mean', 'coord_std', 'attr_mean', 'attr_std'
        forest_stats: Dict with same structure

    Returns:
        dict with shift metrics
    """
    logger.info("\n" + "="*80)
    logger.info("DISTRIBUTION SHIFT ANALYSIS (Forest Plots vs Training)")
    logger.info("="*80)

    # Coordinate shift
    coord_shift = []
    for dim, name in enumerate(['X', 'Y', 'Z']):
        mean_diff = abs(forest_stats['coord_mean'][dim].item() - train_stats['coord_mean'][dim])
        std_diff = abs(forest_stats['coord_std'][dim].item() - train_stats['coord_std'][dim])

        # Normalize by training std
        shift_in_std = mean_diff / train_stats['coord_std'][dim]
        std_ratio = forest_stats['coord_std'][dim].item() / train_stats['coord_std'][dim]
        coord_shift.append(shift_in_std)

        logger.info(f"  {name}: mean shift = {shift_in_std:.3f} std, std ratio = {std_ratio:.3f}")

    # Attribute shift (if available)
    attr_shift = []
    if train_stats.get('attr_mean') is not None and forest_stats.get('attr_mean') is not None:
        for dim, name in enumerate(['Intensity', 'ReturnNumber', 'NumberOfReturns']):
            mean_diff = abs(forest_stats['attr_mean'][dim].item() - train_stats['attr_mean'][dim])
            shift_in_std = mean_diff / train_stats['attr_std'][dim]
            attr_shift.append(shift_in_std)

            logger.info(f"  {name}: mean shift = {shift_in_std:.3f} std")

    # Overall assessment
    max_coord_shift = max(coord_shift)
    if max_coord_shift > 1.0:
        logger.warning(f"  ⚠️  SEVERE coordinate shift detected ({max_coord_shift:.2f} std)")
        logger.warning(f"      Forest plot distribution differs significantly from training!")
    elif max_coord_shift > 0.5:
        logger.warning(f"  ⚠️  Moderate coordinate shift detected ({max_coord_shift:.2f} std)")
        logger.warning(f"      Monitor forest plot predictions carefully")
    else:
        logger.info(f"  ✓ Acceptable coordinate shift ({max_coord_shift:.2f} std)")

    logger.info("="*80)

    return {
        'coord_shift': coord_shift,
        'attr_shift': attr_shift,
        'max_shift': max_coord_shift
    }


##########################################
# Date and Imagery Preprocessing (NAIP/UAVSAR)
# (Identical to train_test_split_and_precompute_raster.py)
##########################################

def parse_date(date_str: str) -> datetime:
    """Parse a date string using multiple methods."""
    try:
        return datetime.fromisoformat(date_str)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for {date_str} not recognized.")


def compute_relative_dates(dates: List[str], reference_date: datetime) -> torch.Tensor:
    """Compute relative dates (in days) for a list of date strings."""
    ref_date_only = reference_date.date()
    rel_dates = []
    for date_str in dates:
        d = parse_date(date_str)
        d_date_only = d.date()
        delta_days = (d_date_only - ref_date_only).days
        rel_dates.append(float(delta_days))
    return torch.tensor(rel_dates, dtype=torch.float32).unsqueeze(1)


def preprocess_naip_imagery(
    tile: Dict[str, Any],
    reference_date: datetime,
    dtype: torch.dtype = torch.float16
) -> Optional[Dict[str, Any]]:
    """
    Preprocess NAIP imagery from the flattened data structure.
    Rescales uint8 values from [0, 255] to [0, 1] range.
    """
    if 'naip_imgs' not in tile or tile['naip_imgs'] is None:
        return None

    images = tile['naip_imgs']
    if images.numel() == 0:
        return None

    images = images.clone()

    # Identify invalid values before normalization
    invalid_mask = torch.isnan(images) | torch.isinf(images)

    # Simple rescaling from [0, 255] to [0, 1]
    images = images.float() / 255.0

    # Set any invalid values to 0
    images[invalid_mask] = 0.0

    # Convert to specified dtype for memory efficiency
    images = images.to(dtype)

    # Get dates and compute relative dates
    dates = tile['naip_dates']
    relative_dates = compute_relative_dates(dates, reference_date)

    return {
        'images': images,
        'ids': tile['naip_ids'],
        'dates': dates,
        'relative_dates': relative_dates,
        'img_bbox': tile['naip_img_bbox'],
        'bands': tile['naip_bands']
    }


def preprocess_uavsar_imagery(
    tile: Dict[str, Any],
    reference_date: datetime,
    dtype: torch.dtype = torch.float32,
    max_images_per_group: int = 8
) -> Optional[Dict[str, Any]]:
    """
    Preprocess UAVSAR imagery, handling variable numbers of images
    associated with distinct acquisition events.
    Only keeps acquisition events with two or more images.
    """
    if 'uavsar_imgs' not in tile or tile['uavsar_imgs'] is None:
        return None

    images = tile['uavsar_imgs']
    if images.numel() == 0:
        return None

    images = images.clone()
    dates_str = tile['uavsar_dates']
    ids = tile['uavsar_ids']

    # Step 1: Filter out images with all invalid pixels
    n_images = images.shape[0]
    valid_image_mask = torch.zeros(n_images, dtype=torch.bool, device=images.device)

    for img_idx in range(n_images):
        img = images[img_idx]
        valid_image_mask[img_idx] = not (torch.isnan(img) | torch.isinf(img)).all()

    invalid_count = n_images - valid_image_mask.sum().item()
    if invalid_count > 0:
        logger.debug(f"Removing {invalid_count} UAVSAR images with all invalid values.")

    if valid_image_mask.sum() == 0:
        logger.warning("All UAVSAR images have invalid values only!")
        return None

    # Apply filtering
    images = images[valid_image_mask]
    dates_str = [date for i, date in enumerate(dates_str) if valid_image_mask[i]]
    ids = [id for i, id in enumerate(ids) if valid_image_mask[i]]

    # Step 2: Parse dates and sort chronologically
    parsed_dates = []
    for date_str in dates_str:
        parsed_date = parse_date(date_str).date()
        parsed_dates.append(parsed_date)

    # Create tuples and sort by date
    sorted_data = sorted(zip(images, parsed_dates, ids, dates_str), key=lambda x: x[1])

    sorted_images = [item[0] for item in sorted_data]
    sorted_dates = [item[1] for item in sorted_data]
    sorted_ids = [item[2] for item in sorted_data]
    sorted_date_strs = [item[3] for item in sorted_data]

    # Step 3: Group by consecutive dates (acquisition events)
    groups = []
    current_group = {'images': [], 'dates': [], 'ids': [], 'date_strs': []}

    for i, (img, date, id_, date_str) in enumerate(zip(sorted_images, sorted_dates, sorted_ids, sorted_date_strs)):
        if i == 0:
            current_group['images'].append(img)
            current_group['dates'].append(date)
            current_group['ids'].append(id_)
            current_group['date_strs'].append(date_str)
        else:
            days_diff = (date - current_group['dates'][-1]).days
            if days_diff > 1:
                groups.append(current_group)
                current_group = {'images': [img], 'dates': [date], 'ids': [id_], 'date_strs': [date_str]}
            else:
                current_group['images'].append(img)
                current_group['dates'].append(date)
                current_group['ids'].append(id_)
                current_group['date_strs'].append(date_str)

    if current_group['images']:
        groups.append(current_group)

    # Step 4: Filter groups to keep only those with two or more images
    groups = [group for group in groups if len(group['images']) >= 2]

    if len(groups) == 0:
        logger.warning("No valid UAVSAR acquisition events with 2+ images found!")
        return None

    # Get tensor dimensions
    n_bands, h, w = images[0].shape
    T = len(groups)
    G_max = max_images_per_group

    # Initialize padded tensors and masks
    device = images[0].device
    padded_images = torch.zeros((T, G_max, n_bands, h, w), dtype=images[0].dtype, device=device)
    attention_mask = torch.zeros((T, G_max), dtype=torch.bool, device=device)
    invalid_mask = torch.zeros((T, G_max, n_bands, h, w), dtype=torch.bool, device=device)

    group_date_strs = []
    group_ids = []

    # Step 5: Populate padded tensors
    for t, group in enumerate(groups):
        actual_count = len(group['images'])
        count_to_pad = min(actual_count, G_max)

        for i in range(count_to_pad):
            padded_images[t, i] = group['images'][i]
            attention_mask[t, i] = True
            invalid_mask[t, i] = torch.isnan(group['images'][i]) | torch.isinf(group['images'][i])

        group_date_strs.append(group['date_strs'][0])
        group_ids.append(group['ids'][:count_to_pad])

    # Step 6: Convert to specified dtype (no per-band normalization for inference)
    padded_images = padded_images.to(dtype)

    # Step 7: Compute relative dates
    relative_dates = compute_relative_dates(group_date_strs, reference_date)

    return {
        'images': padded_images,
        'attention_mask': attention_mask,
        'relative_dates': relative_dates,
        'dates': group_date_strs,
        'ids': group_ids,
        'invalid_mask': invalid_mask,
        'img_bbox': tile['uavsar_img_bbox'],
        'bands': tile['uavsar_bands']
    }


##########################################
# Main Processing
##########################################

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess forest plot tiles for inference with trained raster model"
    )
    parser.add_argument(
        '--pt-file',
        required=True,
        help='Path to combined .pt file with forest plot tiles'
    )
    parser.add_argument(
        '--training-stats-dir',
        required=True,
        help='Directory containing training normalization stats (coordinate_normalization_stats.json)'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for precomputed tiles'
    )
    parser.add_argument(
        '--min-dep-points',
        type=int,
        default=50,
        help='Minimum 3DEP points required per tile (default: 50)'
    )
    parser.add_argument(
        '--precision',
        type=int,
        choices=[16, 32, 64],
        default=32,
        help='Float precision for output (default: 32)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("="*80)
    logger.info("FOREST PLOT PREPROCESSING FOR INFERENCE")
    logger.info("="*80)

    # ============================================================================
    # STEP 1: LOAD TRAINING NORMALIZATION STATISTICS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: LOAD TRAINING NORMALIZATION STATISTICS")
    logger.info("="*80)

    training_stats_dir = Path(args.training_stats_dir)

    # Load coordinate normalization stats
    coord_stats_path = training_stats_dir / 'coordinate_normalization_stats.json'
    if not coord_stats_path.exists():
        # Try alternate naming
        coord_stats_path = training_stats_dir / 'coordinate_normalization_stats_train.json'

    if not coord_stats_path.exists():
        logger.error(f"Training coordinate stats not found at {coord_stats_path}")
        return

    with open(coord_stats_path, 'r') as f:
        train_stats = json.load(f)

    logger.info(f"Loaded training coordinate stats from {coord_stats_path}")
    logger.info(f"  Coord mean: {train_stats['coord_mean']}")
    logger.info(f"  Coord std: {train_stats['coord_std']}")

    # ============================================================================
    # STEP 2: LOAD COMBINED FOREST PLOT DATA
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 2: LOAD COMBINED FOREST PLOT DATA")
    logger.info("="*80)

    logger.info(f"Loading forest plot data from {args.pt_file}...")
    combined_data = torch.load(args.pt_file, weights_only=False)
    all_tiles = combined_data if isinstance(combined_data, list) else combined_data['tiles']
    logger.info(f"Loaded {len(all_tiles)} tiles")

    # ============================================================================
    # STEP 3: FIX KEY NAMING AND CONVERT TO TENSORS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: FIX KEY NAMING AND CONVERT TO TENSORS")
    logger.info("="*80)

    numpy_scalar_fields = [
        'fuel_metrics_resolution', 'has_fuel_metrics', 'has_imagery',
        'has_naip', 'has_pointcloud', 'has_uavsar',
        'initial_voxel_size_cm', 'naip_resolution', 'uavsar_resolution'
    ]

    for tile in all_tiles:
        # Convert numpy arrays to tensors
        if 'dep_points' in tile and isinstance(tile['dep_points'], np.ndarray):
            tile['dep_points'] = torch.from_numpy(tile['dep_points']).float()
        if 'dep_pnt_attr' in tile and isinstance(tile['dep_pnt_attr'], np.ndarray):
            tile['dep_pnt_attr'] = torch.from_numpy(tile['dep_pnt_attr']).float()
        if 'bbox' in tile and isinstance(tile['bbox'], np.ndarray):
            tile['bbox'] = torch.from_numpy(tile['bbox']).float()

        # Convert numpy scalar types to Python native types
        for field in numpy_scalar_fields:
            if field in tile:
                value = tile[field]
                if isinstance(value, (np.integer, np.floating)):
                    tile[field] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.bool_):
                    tile[field] = bool(value)

    logger.info("Fixed key naming and converted arrays to tensors")

    # ============================================================================
    # STEP 4: FILTER TILES BY MIN DEP POINTS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info(f"STEP 4: FILTER TILES (min_dep_points={args.min_dep_points})")
    logger.info("="*80)

    filtered_tiles = []
    filtered_out_count = 0
    filter_reasons = {'no_dep': 0, 'low_dep': 0, 'no_bbox': 0}

    for tile in all_tiles:
        if 'dep_points' not in tile or tile['dep_points'] is None:
            filtered_out_count += 1
            filter_reasons['no_dep'] += 1
            continue

        if 'bbox' not in tile or tile['bbox'] is None:
            filtered_out_count += 1
            filter_reasons['no_bbox'] += 1
            continue

        dep_count = tile['dep_points'].shape[0]
        if dep_count < args.min_dep_points:
            filtered_out_count += 1
            filter_reasons['low_dep'] += 1
            continue

        filtered_tiles.append(tile)

    logger.info(f"Filtered to {len(filtered_tiles)} valid tiles (removed {filtered_out_count})")
    logger.info(f"  Filter reasons: no_dep={filter_reasons['no_dep']}, "
               f"low_dep={filter_reasons['low_dep']}, no_bbox={filter_reasons['no_bbox']}")

    if len(filtered_tiles) == 0:
        logger.error("No valid tiles found after filtering!")
        return

    # ============================================================================
    # STEP 5: COMPUTE FOREST PLOT STATISTICS AND DISTRIBUTION SHIFT
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: COMPUTE FOREST PLOT STATISTICS")
    logger.info("="*80)

    forest_stats = compute_forest_plot_statistics(filtered_tiles)

    # Save forest plot statistics (diagnostic)
    forest_stats_path = output_dir / 'forest_plot_coordinate_stats.json'
    forest_stats_json = {
        'coord_mean': forest_stats['coord_mean'].tolist(),
        'coord_std': forest_stats['coord_std'].tolist(),
        'attr_mean': forest_stats['attr_mean'].tolist() if forest_stats['attr_mean'] is not None else None,
        'attr_std': forest_stats['attr_std'].tolist() if forest_stats['attr_std'] is not None else None,
        'total_points': forest_stats['total_points'],
        'total_tiles': forest_stats['total_tiles']
    }
    with open(forest_stats_path, 'w') as f:
        json.dump(forest_stats_json, f, indent=2)
    logger.info(f"Saved forest plot statistics to {forest_stats_path}")

    # Report distribution shift
    shift_report = report_distribution_shift(train_stats, forest_stats)

    # Save shift report
    shift_report_path = output_dir / 'distribution_shift_report.json'
    with open(shift_report_path, 'w') as f:
        json.dump(shift_report, f, indent=2)
    logger.info(f"Saved distribution shift report to {shift_report_path}")

    # ============================================================================
    # STEP 6: APPLY COORDINATE NORMALIZATION USING TRAINING STATS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: APPLY COORDINATE NORMALIZATION (using TRAINING stats)")
    logger.info("="*80)

    for tile_idx, tile in enumerate(filtered_tiles):
        if 'dep_points' in tile and 'dep_points_norm' not in tile:
            dep_points = tile.pop('dep_points')
            dep_pnt_attr = tile.get('dep_pnt_attr', None)

            # Convert to tensors if needed
            if isinstance(dep_points, np.ndarray):
                dep_points = torch.from_numpy(dep_points).float()
            if dep_pnt_attr is not None and isinstance(dep_pnt_attr, np.ndarray):
                dep_pnt_attr = torch.from_numpy(dep_pnt_attr).float()

            bbox = tile.get('bbox', None)

            if bbox is not None:
                if isinstance(bbox, np.ndarray):
                    bbox = torch.from_numpy(bbox).float()

                # Step 1: Apply bbox normalization
                uav_dummy = torch.empty((0, 3), dtype=torch.float32)
                dep_points_bbox_norm, _, center, scale = normalize_point_clouds_with_bbox(
                    dep_points, uav_dummy, bbox, dtype=torch.float32
                )

                # Step 1b: Clamp bbox-normalized Z to [0, 150]
                dep_points_bbox_norm[:, 2] = torch.clamp(dep_points_bbox_norm[:, 2], 0, 150)

                # Step 2: Apply z-score normalization using TRAINING stats
                dep_points_norm = apply_zscore_to_bbox_normalized_coords(
                    dep_points_bbox_norm,
                    train_stats['coord_mean'],
                    train_stats['coord_std']
                )

                # Step 3: Normalize attributes using TRAINING stats
                dep_points_attr_norm = None
                if dep_pnt_attr is not None and train_stats.get('attr_mean') is not None:
                    dep_points_attr_norm = normalize_attributes_zscore(
                        dep_pnt_attr,
                        train_stats['attr_mean'],
                        train_stats['attr_std'],
                        dtype=torch.float16
                    )

                # Store all normalized versions
                tile['dep_points_norm'] = dep_points_norm
                tile['dep_points_bbox_norm'] = dep_points_bbox_norm
                tile['dep_points_attr_norm'] = dep_points_attr_norm
                tile['center'] = center
                tile['scale'] = scale

                # Store norm_params using TRAINING stats (tensors for model compatibility)
                tile['norm_params'] = {
                    'coord_mean': torch.tensor(train_stats['coord_mean'], dtype=torch.float32),
                    'coord_std': torch.tensor(train_stats['coord_std'], dtype=torch.float32),
                    'attr_mean': torch.tensor(train_stats['attr_mean'], dtype=torch.float32) if train_stats.get('attr_mean') else None,
                    'attr_std': torch.tensor(train_stats['attr_std'], dtype=torch.float32) if train_stats.get('attr_std') else None
                }
            else:
                logger.warning(f"No bbox found for tile {tile.get('tile_id', 'unknown')}, skipping")
                tile['dep_points_norm'] = dep_points
                tile['dep_points_attr_norm'] = None

        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(filtered_tiles)} tiles...")

    logger.info("Coordinate normalization complete")

    # ============================================================================
    # STEP 7: PREPROCESS NAIP AND UAVSAR IMAGERY
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 7: PREPROCESS NAIP AND UAVSAR IMAGERY")
    logger.info("="*80)

    naip_count = 0
    uavsar_count = 0

    for tile_idx, tile in enumerate(filtered_tiles):
        # Use 3DEP acquisition date as reference for relative_dates computation
        # This is the correct temporal anchor: imagery dates relative to when 3DEP was captured
        # The model learns to predict current vegetation state from 3DEP + temporal changes in imagery
        if 'dep_meta' in tile and tile['dep_meta'] and 'start_datetime' in tile['dep_meta']:
            reference_date = parse_date(tile['dep_meta']['start_datetime'])
        elif 'naip_dates' in tile and len(tile['naip_dates']) > 0:
            # Fallback to NAIP if 3DEP date not available
            reference_date = parse_date(tile['naip_dates'][0])
            logger.warning(f"Tile {tile.get('tile_id', 'unknown')}: No 3DEP date, using NAIP date as fallback")
        else:
            reference_date = datetime.now()
            logger.warning(f"Tile {tile.get('tile_id', 'unknown')}: No reference date available")

        # Preprocess NAIP imagery
        if 'naip_imgs' in tile and tile['naip_imgs'] is not None and tile['naip_imgs'].numel() > 0:
            naip_preprocessed = preprocess_naip_imagery(tile, reference_date, dtype=torch.float16)
            if naip_preprocessed is not None:
                tile['naip'] = naip_preprocessed
                naip_count += 1
            else:
                tile['naip'] = None
        else:
            tile['naip'] = None

        # Preprocess UAVSAR imagery
        if 'uavsar_imgs' in tile and tile['uavsar_imgs'] is not None and tile['uavsar_imgs'].numel() > 0:
            uavsar_preprocessed = preprocess_uavsar_imagery(tile, reference_date, dtype=torch.float32)
            if uavsar_preprocessed is not None:
                tile['uavsar'] = uavsar_preprocessed
                uavsar_count += 1
            else:
                tile['uavsar'] = None
        else:
            tile['uavsar'] = None

        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Preprocessed {tile_idx + 1}/{len(filtered_tiles)} tiles...")

    logger.info(f"Imagery preprocessing complete: {naip_count} NAIP, {uavsar_count} UAVSAR")

    # ============================================================================
    # STEP 8: PRECOMPUTE KNN INDICES
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 8: PRECOMPUTE KNN INDICES")
    logger.info("="*80)

    try:
        from torch_geometric.nn import knn_graph
        from torch_geometric.utils import to_undirected
    except ImportError:
        logger.error("torch_geometric not available, skipping KNN precomputation")
        logger.error("Install with: pip install torch_geometric")
        return

    k_values = [15]  # Standard KNN value
    processed_count = 0
    knn_failed_count = 0

    for tile in filtered_tiles:
        if 'dep_points_norm' in tile and isinstance(tile['dep_points_norm'], torch.Tensor):
            dep_points = tile['dep_points_norm']

            if dep_points.shape[0] > k_values[0]:
                try:
                    knn_edges = {}
                    for k in k_values:
                        edge_index = knn_graph(dep_points, k=k, loop=False)
                        edge_index = to_undirected(edge_index, num_nodes=dep_points.size(0))
                        knn_edges[k] = edge_index

                    tile['knn_edge_indices'] = knn_edges
                except Exception as e:
                    logger.warning(f"KNN failed for tile {tile.get('tile_id', 'unknown')}: {e}")
                    tile['knn_edge_indices'] = None
                    knn_failed_count += 1
            else:
                logger.warning(f"Tile {tile.get('tile_id', 'unknown')} has only {dep_points.shape[0]} points, "
                             f"skipping KNN (need >{k_values[0]})")
                tile['knn_edge_indices'] = None

        processed_count += 1
        if processed_count % 50 == 0:
            logger.info(f"  Precomputed KNN for {processed_count}/{len(filtered_tiles)} tiles...")

    logger.info(f"KNN precomputation complete ({knn_failed_count} failed)")

    # ============================================================================
    # STEP 9: SAVE OUTPUT
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 9: SAVE OUTPUT")
    logger.info("="*80)

    output_path = output_dir / f'precomputed_forest_plot_tiles_{args.precision}bit.pt'
    torch.save(filtered_tiles, output_path)
    logger.info(f"Saved to {output_path}")

    # ============================================================================
    # SUMMARY
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("="*80)

    # Count tiles per site
    site_counts = {}
    for tile in filtered_tiles:
        tile_id = tile.get('tile_id', 'unknown')
        # Extract site from tile_id (e.g., "BluffMesa_tile_123" -> "BluffMesa")
        parts = tile_id.split('_tile_')
        site = parts[0] if len(parts) > 0 else 'unknown'
        site_counts[site] = site_counts.get(site, 0) + 1

    logger.info(f"Total tiles: {len(filtered_tiles)}")
    logger.info(f"Tiles per site:")
    for site, count in sorted(site_counts.items()):
        logger.info(f"  {site}: {count}")

    logger.info(f"\nOutput: {output_path}")
    logger.info(f"Distribution shift report: {shift_report_path}")
    logger.info(f"Forest plot stats: {forest_stats_path}")


if __name__ == '__main__':
    main()
