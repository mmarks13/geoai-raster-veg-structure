#!/usr/bin/env python3
"""
Train/Validation Split and Precomputation for Raster-Based Model

Loads combined training data, applies quality filtering, splits into train/val,
normalizes fuel metrics bands globally, preprocesses NAIP/UAVSAR imagery, and
precomputes KNN indices.

**IDENTICAL PREPROCESSING to: src/data_prep/train_test_split_and_precompute.py**
(Same model architecture, different prediction head: point cloud → raster)

Input: Combined .pt file with tiles containing fuel_metrics [22, h, w] (Band 22 removed)
Output:
  - precomputed_training_tiles_raster_32bit.pt (normalized, preprocessed)
  - precomputed_validation_tiles_raster_32bit.pt (normalized, preprocessed)
  - fuel_metrics_normalization_stats.json (statistics)
"""

import torch
import json
import argparse
import random
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_fuel_metrics_quality(tile, band_index=14, max_na_ratio=0.5):  # Band 15 (index 14)
    """
    Check if fuel metrics have acceptable coverage in specified band.

    Parameters:
        tile (dict): Tile with 'fuel_metrics' key
        band_index (int): Band to check (0-indexed, 14 = Band 15)
        max_na_ratio (float): Maximum allowed ratio of NaN pixels (0.0-1.0, default: 0.5 = 50%)

    Returns:
        bool: True if NaN pixels <= max_na_ratio, False otherwise
    """
    if 'fuel_metrics' not in tile:
        return False

    fuel_metrics = tile['fuel_metrics']
    if fuel_metrics is None:
        return False

    # Check band 15 (index 14) for NA ratio
    band_data = fuel_metrics[band_index]
    nan_count = torch.isnan(band_data).sum().item()
    total_pixels = band_data.numel()
    na_ratio = nan_count / total_pixels if total_pixels > 0 else 0.0

    return na_ratio <= max_na_ratio


def compute_normalization_stats(tiles):
    """
    Compute per-band global mean and std for fuel metrics (23 bands).

    Parameters:
        tiles (list): List of tiles with 'fuel_metrics' [23, h, w]

    Returns:
        dict: Statistics {'band_1_mean': ..., 'band_1_std': ..., ...}
    """
    logger.info("Computing normalization statistics for 22 fuel metrics bands...")

    # Initialize accumulators for each band
    band_sums = [0.0] * 22
    band_sq_sums = [0.0] * 22
    band_counts = [0] * 22

    # Collect statistics from all tiles
    for tile_idx, tile in enumerate(tiles):
        if 'fuel_metrics' not in tile or tile['fuel_metrics'] is None:
            continue

        fuel_metrics = tile['fuel_metrics']  # [22, h, w]

        for band_idx in range(22):
            band_data = fuel_metrics[band_idx]

            # Ignore NaN values
            valid_mask = ~torch.isnan(band_data)
            valid_data = band_data[valid_mask]

            if len(valid_data) > 0:
                band_sums[band_idx] += valid_data.sum().item()
                band_sq_sums[band_idx] += (valid_data ** 2).sum().item()
                band_counts[band_idx] += valid_data.numel()

        if (tile_idx + 1) % 100 == 0:
            logger.info(f"  Processed {tile_idx + 1}/{len(tiles)} tiles...")

    # Compute mean and std for each band
    stats = {}
    for band_idx in range(22):
        if band_counts[band_idx] > 0:
            mean = band_sums[band_idx] / band_counts[band_idx]
            sq_mean = band_sq_sums[band_idx] / band_counts[band_idx]
            variance = sq_mean - (mean ** 2)
            std = np.sqrt(max(variance, 1e-8))  # Avoid sqrt of negative

            stats[f'band_{band_idx + 1}_mean'] = float(mean)
            stats[f'band_{band_idx + 1}_std'] = float(std)
        else:
            # Default for empty bands
            stats[f'band_{band_idx + 1}_mean'] = 0.0
            stats[f'band_{band_idx + 1}_std'] = 1.0

    logger.info(f"Computed statistics for all 22 bands")
    return stats


def normalize_point_clouds_with_bbox(dep_points: torch.Tensor,
                                     uav_points: torch.Tensor,
                                     bbox: torch.Tensor,
                                     dtype: torch.dtype = torch.float32):
    """
    Normalizes 3DEP and UAV point clouds to a common coordinate system where:
    - x,y coordinates range from -5 to 5 (1 unit = 1 meter)
    - z coordinates are in meters, with minimum z value set to 0

    **MATCHES ORIGINAL:** This function is identical to the non-raster version
    in src/data_prep/train_test_split_and_precompute.py for consistency.

    Inputs:
      dep_points: [N_dep, 3] tensor of 3DEP point coordinates.
      uav_points: [N_uav, 3] tensor of UAV point coordinates.
      bbox: Tensor [xmin, ymin, xmax, ymax] defining the spatial extent.
      dtype: PyTorch dtype to use for the output tensors.

    Returns:
      dep_points_norm: [N_dep, 3] normalized 3DEP points in specified dtype.
      uav_points_norm: [N_uav, 3] normalized UAV points in specified dtype.
      center: [1, 3] tensor representing the normalization center.
      scale: [1, 3] tensor with scale factors for x, y, z.
    """
    xmin, ymin, xmax, ymax = bbox[0].item(), bbox[1].item(), bbox[2].item(), bbox[3].item()

    # Combine both point clouds to find minimum z value
    z_min_dep = dep_points[:, 2].min() if dep_points.shape[0] > 0 else 0
    z_min_uav = uav_points[:, 2].min() if uav_points.shape[0] > 0 else z_min_dep
    z_min = min(z_min_dep, z_min_uav)

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
    uav_points_norm = uav_points.clone()

    # Handle x,y coordinates (center at 0,0, scale to [-5,5] range)
    dep_points_norm[:, :2] -= center[:, :2]
    dep_points_norm[:, :2] /= scale[:, :2]
    uav_points_norm[:, :2] -= center[:, :2]
    uav_points_norm[:, :2] /= scale[:, :2]

    # Handle z coordinates (shift to make minimum = 0, keep in meters)
    dep_points_norm[:, 2] = dep_points[:, 2] - center[:, 2]  # Just subtract minimum z
    uav_points_norm[:, 2] = uav_points[:, 2] - center[:, 2]  # Just subtract minimum z

    # Convert to specified dtype for memory efficiency
    dep_points_norm = dep_points_norm.to(dtype)
    uav_points_norm = uav_points_norm.to(dtype)

    return dep_points_norm, uav_points_norm, center, scale


##########################################
# Date and Imagery Preprocessing (NAIP/UAVSAR)
# IDENTICAL to: src/data_prep/train_test_split_and_precompute.py
##########################################

def parse_date(date_str: str) -> datetime:
    """
    Parse a date string using multiple methods.

    First, attempts to use datetime.fromisoformat (which supports ISO 8601 strings
    with timezone offsets, e.g., "2023-10-25T00:00:00+00:00").
    If that fails, it will try common strptime formats.

    Returns:
      A datetime object.
    """
    try:
        # This handles ISO 8601 with timezone offsets.
        return datetime.fromisoformat(date_str)
    except ValueError:
        pass

    # Fallback: Try common formats.
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    raise ValueError(f"Date format for {date_str} not recognized.")


def compute_relative_dates(dates: List[str], reference_date: datetime) -> torch.Tensor:
    """
    Compute relative dates (in days) for a list of date strings relative to the reference_date.
    Both the reference_date and each date in the list are converted to date objects (ignoring time).

    Returns:
      A tensor of shape [n_images, 1] with the relative day differences.
    """
    ref_date_only = reference_date.date()
    rel_dates = []
    for date_str in dates:
        d = parse_date(date_str)
        d_date_only = d.date()
        delta_days = (d_date_only - ref_date_only).days
        rel_dates.append(float(delta_days))
    return torch.tensor(rel_dates, dtype=torch.float32).unsqueeze(1)


def preprocess_naip_imagery(tile: Dict[str, Any], reference_date: datetime,
                          naip_means=None, naip_stds=None, dtype: torch.dtype = torch.float16) -> Dict[str, Any]:
    """
    Preprocess NAIP imagery from the flattened data structure.
    Rescales uint8 values from [0, 255] to [0, 1] range.
    Handles NA, NaN, and Inf values by setting them to 0 after normalization.
    Converts normalized values to specified precision for memory efficiency.

    **IDENTICAL to:** src/data_prep/train_test_split_and_precompute.py

    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'naip_imgs': Tensor of shape [n_images, 4, h, w] (4 spectral bands)
         - 'naip_dates': List of date strings
         - 'naip_ids': List of image IDs
         - 'naip_img_bbox': NAIP imagery bounding box [minx, miny, maxx, maxy]
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      naip_means: Not used for this normalization method.
      naip_stds: Not used for this normalization method.
      dtype: PyTorch dtype to use for the output tensors.

    Returns:
      A dictionary with:
         - 'images': The normalized NAIP imagery tensor in specified dtype format
         - 'relative_dates': Tensor of shape [n_images, 1] with relative dates (in days)
         - 'img_bbox': The NAIP imagery bounding box
    """
    # Get NAIP imagery tensor
    images = tile['naip_imgs'].clone()  # Tensor: [n_images, 4, h, w]

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
        'images': images,                  # Normalized image tensor: [n_images, 4, h, w] in specified dtype
        'ids': tile['naip_ids'],           # List of image IDs
        'dates': dates,
        'relative_dates': relative_dates,  # Tensor: [n_images, 1]
        'img_bbox': tile['naip_img_bbox'], # Bounding box
        'bands': tile['naip_bands']        # Band information
    }


def preprocess_uavsar_imagery(tile: Dict[str, Any], reference_date: datetime,
                             uavsar_means=None, uavsar_stds=None, dtype: torch.dtype = torch.float32,
                             max_images_per_group: int = 8) -> Dict[str, Any]:
    """
    Preprocess UAVSAR imagery from the flattened data structure, handling variable numbers of images
    associated with distinct acquisition events (where events can span consecutive days).
    Only keeps acquisition events with two or more images.

    **IDENTICAL to:** src/data_prep/train_test_split_and_precompute.py

    Inputs:
      tile: Dictionary containing flattened tile data with keys:
         - 'uavsar_imgs': Tensor of shape [n_images, n_bands, h, w]
         - 'uavsar_dates': List of date strings
         - 'uavsar_ids': List of image IDs
         - 'uavsar_img_bbox': UAVSAR imagery bounding box
      reference_date: UAV LiDAR acquisition date used to compute relative dates.
      uavsar_means: Optional tensor of shape [n_bands] with mean values for each band.
      uavsar_stds: Optional tensor of shape [n_bands] with standard deviation values for each band.
      dtype: PyTorch dtype to use for the output tensors.
      max_images_per_group: Maximum number of images to keep per acquisition event (G_max).

    Returns:
      A dictionary with:
         - 'images': Padded tensor of shape [T, G_max, n_bands, h, w]
         - 'attention_mask': Boolean mask of shape [T, G_max]
         - 'relative_dates': Tensor of shape [T, 1] with relative dates
         - 'dates': List of T representative dates
         - 'ids': List of lists containing image IDs
         - 'invalid_mask': Boolean mask of shape [T, G_max, n_bands, h, w]
         - 'img_bbox': The UAVSAR imagery bounding box
         - 'bands': Band information
    """
    # Get UAVSAR imagery tensor
    images = tile['uavsar_imgs'].clone()  # Tensor: [n_images, n_bands, h, w]
    dates_str = tile['uavsar_dates']
    ids = tile['uavsar_ids']

    # Step 1: Filter out images with all invalid pixels
    n_images = images.shape[0]
    valid_image_mask = torch.zeros(n_images, dtype=torch.bool, device=images.device)

    for img_idx in range(n_images):
        img = images[img_idx]  # Shape: [n_bands, h, w]
        valid_image_mask[img_idx] = not (torch.isnan(img) | torch.isinf(img)).all()

    invalid_count = n_images - valid_image_mask.sum().item()
    if invalid_count > 0:
        logger.info(f"Removing {invalid_count} UAVSAR images with all invalid values.")

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
        parsed_date = parse_date(date_str).date()  # Convert to date object (ignore time)
        parsed_dates.append(parsed_date)

    # Create tuples of (image, date, id) and sort by date
    sorted_data = sorted(zip(images, parsed_dates, ids, dates_str), key=lambda x: x[1])

    # Unpack sorted data
    sorted_images = [item[0] for item in sorted_data]
    sorted_dates = [item[1] for item in sorted_data]
    sorted_ids = [item[2] for item in sorted_data]
    sorted_date_strs = [item[3] for item in sorted_data]

    # Step 3: Group by consecutive dates (acquisition events)
    groups = []
    current_group = {'images': [], 'dates': [], 'ids': [], 'date_strs': []}

    for i, (img, date, id_, date_str) in enumerate(zip(sorted_images, sorted_dates, sorted_ids, sorted_date_strs)):
        if i == 0:  # First image always starts a group
            current_group['images'].append(img)
            current_group['dates'].append(date)
            current_group['ids'].append(id_)
            current_group['date_strs'].append(date_str)
        else:
            # Check if current date is more than 1 day after the last date in the current group
            days_diff = (date - current_group['dates'][-1]).days
            if days_diff > 1:
                # Start a new group
                groups.append(current_group)
                current_group = {'images': [img], 'dates': [date], 'ids': [id_], 'date_strs': [date_str]}
            else:
                # Add to current group
                current_group['images'].append(img)
                current_group['dates'].append(date)
                current_group['ids'].append(id_)
                current_group['date_strs'].append(date_str)

    # Add the last group if it's not empty
    if current_group['images']:
        groups.append(current_group)

    # Step 4: Filter groups to keep only those with two or more images
    original_group_count = len(groups)
    groups = [group for group in groups if len(group['images']) >= 2]
    filtered_count = original_group_count - len(groups)

    if len(groups) == 0:
        logger.warning("No valid UAVSAR acquisition events with 2+ images found after filtering!")
        return None

    # Get tensor dimensions
    n_bands, h, w = images[0].shape
    T = len(groups)  # Number of acquisition events
    G_max = max_images_per_group  # Maximum images per event

    # Initialize padded tensors and masks
    device = images[0].device
    padded_images = torch.zeros((T, G_max, n_bands, h, w), dtype=images[0].dtype, device=device)
    attention_mask = torch.zeros((T, G_max), dtype=torch.bool, device=device)
    invalid_mask = torch.zeros((T, G_max, n_bands, h, w), dtype=torch.bool, device=device)

    # Lists for metadata
    group_dates = []  # Representative date for each group
    group_date_strs = []  # String representation of representative dates
    group_ids = []  # IDs for each group

    # Step 5: Populate padded tensors
    for t, group in enumerate(groups):
        actual_count = len(group['images'])
        count_to_pad = min(actual_count, G_max)

        if actual_count > G_max:
            logger.warning(f"Group {t} has {actual_count} images, but only {G_max} will be used (truncating).")

        # Copy images into padded tensor
        for i in range(count_to_pad):
            padded_images[t, i] = group['images'][i]
            attention_mask[t, i] = True

            # Mark invalid pixels in the original images
            invalid_mask[t, i] = torch.isnan(group['images'][i]) | torch.isinf(group['images'][i])

        # Store representative date (first date in group) and IDs
        group_dates.append(group['dates'][0])
        group_date_strs.append(group['date_strs'][0])
        group_ids.append(group['ids'][:count_to_pad])  # Truncate if necessary

    # Step 6: Normalize the padded images
    if uavsar_means is not None and uavsar_stds is not None:
        # Create copies of the original tensors as we'll modify them
        padded_images_normalized = padded_images.clone()

        # Set invalid values to means temporarily for normalization
        for band_idx in range(n_bands):
            band_invalid_mask = invalid_mask[..., band_idx, :, :]
            if band_invalid_mask.any():
                padded_images_normalized[..., band_idx, :, :][band_invalid_mask] = uavsar_means[band_idx].to(padded_images.dtype)

        # Reshape means and stds for broadcasting: [1, 1, C, 1, 1]
        means = uavsar_means.view(1, 1, -1, 1, 1).to(padded_images.dtype)
        stds = uavsar_stds.view(1, 1, -1, 1, 1).to(padded_images.dtype)

        # Normalize
        padded_images_normalized = (padded_images_normalized - means) / stds

        # Find any new invalid values created during normalization
        new_invalid_mask = torch.isnan(padded_images_normalized) | torch.isinf(padded_images_normalized)
        padded_images_normalized[new_invalid_mask] = 0.0

        # Update the invalid mask
        invalid_mask = invalid_mask | new_invalid_mask

        # Zero out padded positions using the attention mask
        float_attention_mask = attention_mask.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        padded_images_normalized = padded_images_normalized * float_attention_mask

        # Convert to specified dtype
        padded_images = padded_images_normalized.to(dtype)
    else:
        # Convert to specified dtype even if normalization wasn't applied
        padded_images = padded_images.to(dtype)

    # Step 7: Compute relative dates for representative dates
    relative_dates = compute_relative_dates(group_date_strs, reference_date)

    return {
        'images': padded_images,                # Shape: [T, G_max, n_bands, h, w]
        'attention_mask': attention_mask,       # Shape: [T, G_max]
        'relative_dates': relative_dates,       # Shape: [T, 1]
        'dates': group_date_strs,               # List of T representative dates
        'ids': group_ids,                       # List of lists: [[ids for group 0], [ids for group 1], ...]
        'invalid_mask': invalid_mask,           # Shape: [T, G_max, n_bands, h, w]
        'img_bbox': tile['uavsar_img_bbox'],    # Bounding box
        'bands': tile['uavsar_bands']           # Band information
    }


def replace_na_with_defaults(tile):
    """
    Replace NA with 0 for fuel/cover/height bands, keep NA for structural metrics.

    NA → 0 replacement (absence of vegetation = zero value):
      - Band 3: Height (Canopy height, m)
      - Bands 8-16: Fuel loads, cover percentages
      - Bands 19-21, 23: PAI values, max_CBD (Band 22 skipped in source)

    Keep NA (value undefined without vegetation):
      - Bands 1-2: Profile types (categorical)
      - Bands 4-7: CBH, FSG, VCI indices
      - Bands 17-18: Entropy indices

    Parameters:
        tile (dict): Tile with 'fuel_metrics' key

    Returns:
        dict: Modified tile with NAs replaced for appropriate bands
    """
    if 'fuel_metrics' not in tile or tile['fuel_metrics'] is None:
        return tile

    fuel_metrics = tile['fuel_metrics'].clone()  # [22, h, w]

    # Band indices to replace NA → 0 (0-indexed)
    # Band 3 (index 2): Height
    # Bands 8-16 (indices 7-15): Fuel loads + cover
    # Bands 19-21, 23 (indices 18-21): PAI + max_CBD (Band 22 removed)
    replace_bands = [2] + list(range(7, 16)) + list(range(18, 22))

    na_count_per_band = {}
    for band_idx in replace_bands:
        band_data = fuel_metrics[band_idx]
        nan_mask = torch.isnan(band_data)
        na_count = nan_mask.sum().item()
        if na_count > 0:
            fuel_metrics[band_idx, nan_mask] = 0.0
            na_count_per_band[band_idx + 1] = na_count

    tile['fuel_metrics'] = fuel_metrics
    return tile, na_count_per_band


def normalize_fuel_metrics(tile, stats):
    """
    Apply z-score normalization to fuel metrics using pre-computed statistics.

    Parameters:
        tile (dict): Tile to normalize (modified in-place)
        stats (dict): Normalization statistics

    Returns:
        dict: Modified tile with normalized fuel_metrics
    """
    if 'fuel_metrics' not in tile or tile['fuel_metrics'] is None:
        return tile

    fuel_metrics = tile['fuel_metrics'].clone()  # [22, h, w]

    for band_idx in range(22):
        mean_key = f'band_{band_idx + 1}_mean'
        std_key = f'band_{band_idx + 1}_std'

        if mean_key in stats and std_key in stats:
            mean = stats[mean_key]
            std = stats[std_key]

            # Apply z-score normalization, preserving NaN values
            band_data = fuel_metrics[band_idx]
            valid_mask = ~torch.isnan(band_data)

            fuel_metrics[band_idx, valid_mask] = (band_data[valid_mask] - mean) / (std + 1e-8)

    tile['fuel_metrics'] = fuel_metrics
    return tile


def replace_remaining_nans_with_sentinel(tile, sentinel_value=-999.0):
    """
    Replace any remaining NaN values in normalized fuel metrics with a sentinel value.

    This is called AFTER normalization to handle bands where NaN has meaning
    (e.g., canopy height, understory height) and shouldn't be set to 0.

    Parameters:
        tile (dict): Tile with normalized fuel_metrics
        sentinel_value (float): Sentinel value to use for NaN (-999.0 by default)

    Returns:
        dict: Modified tile with NaN replaced by sentinel
    """
    if 'fuel_metrics' not in tile or tile['fuel_metrics'] is None:
        return tile

    fuel_metrics = tile['fuel_metrics']
    nan_mask = torch.isnan(fuel_metrics)

    if nan_mask.any():
        fuel_metrics = fuel_metrics.clone()
        fuel_metrics[nan_mask] = sentinel_value
        tile['fuel_metrics'] = fuel_metrics

    return tile


def main():
    parser = argparse.ArgumentParser(
        description="Train/validation split and precomputation for raster model"
    )
    parser.add_argument(
        '--pt-file',
        required=True,
        help='Path to combined .pt file with all tiles'
    )
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Output directory for precomputed tiles and statistics'
    )
    parser.add_argument(
        '--train-val-ratio',
        type=float,
        default=0.9,
        help='Ratio of training data (default: 0.9 = 90% train, 10% val)'
    )
    parser.add_argument(
        '--min-dep-points',
        type=int,
        default=100,
        help='Minimum 3DEP points required per tile (default: 100)'
    )
    parser.add_argument(
        '--random-seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--precision',
        type=int,
        choices=[16, 32, 64],
        default=32,
        help='Float precision for output (default: 32)'
    )
    parser.add_argument(
        '--max-na-ratio',
        type=float,
        default=0.5,
        help='Maximum allowed ratio of NaN pixels in Band 15 (0.0-1.0, default: 0.5 = 50%)'
    )

    args = parser.parse_args()

    # Validate max_na_ratio
    if not (0.0 <= args.max_na_ratio <= 1.0):
        logger.error(f"--max-na-ratio must be between 0.0 and 1.0, got {args.max_na_ratio}")
        return

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    logger.info("="*80)
    logger.info("TRAIN/VALIDATION SPLIT AND PRECOMPUTATION FOR RASTER MODEL")
    logger.info("="*80)

    # Load combined data
    logger.info(f"Loading combined data from {args.pt_file}...")
    combined_data = torch.load(args.pt_file, weights_only=False)
    all_tiles = combined_data if isinstance(combined_data, list) else combined_data['tiles']
    logger.info(f"Loaded {len(all_tiles)} tiles")

    # Fix key naming issues and normalize point clouds
    for tile in all_tiles:
        # Fix fuel_metrics_fuel_metrics → fuel_metrics
        if 'fuel_metrics_fuel_metrics' in tile and 'fuel_metrics' not in tile:
            tile['fuel_metrics'] = tile.pop('fuel_metrics_fuel_metrics')

        # Normalize point cloud coordinates (matches original train_test_split_and_precompute.py)
        if 'dep_points' in tile and 'dep_points_norm' not in tile:
            dep_points = tile.pop('dep_points')
            if isinstance(dep_points, np.ndarray):
                dep_points = torch.from_numpy(dep_points).float()

            # Get UAV points if available, otherwise create empty tensor
            uav_points = tile.get('uav_points', torch.empty((0, 3), dtype=torch.float32))
            if isinstance(uav_points, np.ndarray):
                uav_points = torch.from_numpy(uav_points).float()

            # Get bbox for normalization
            if 'bbox' in tile:
                bbox = tile['bbox']
                if isinstance(bbox, np.ndarray):
                    bbox = torch.from_numpy(bbox).float()

                # Normalize point clouds using the same function as the original version
                dep_points_norm, uav_points_norm, center, scale = normalize_point_clouds_with_bbox(
                    dep_points, uav_points, bbox, dtype=torch.float32
                )

                tile['dep_points_norm'] = dep_points_norm
                tile['uav_points'] = uav_points_norm
                tile['center'] = center
                tile['scale'] = scale
            else:
                logger.warning(f"No bbox found for tile {tile.get('tile_id', 'unknown')}, skipping point normalization")
                tile['dep_points_norm'] = dep_points

    # Filter tiles by quality
    logger.info(f"\nFiltering tiles (min_dep_points={args.min_dep_points}, max_na_ratio={args.max_na_ratio})...")
    filtered_tiles = []
    filtered_out_count = 0
    total_na_replaced = 0
    na_per_band = {i: 0 for i in range(1, 24)}

    for tile_idx, tile in enumerate(all_tiles):
        # Check DEP points
        if 'dep_points_norm' not in tile or tile['dep_points_norm'] is None:
            filtered_out_count += 1
            continue

        dep_count = tile['dep_points_norm'].shape[0]
        if dep_count < args.min_dep_points:
            filtered_out_count += 1
            continue

        # Check fuel metrics quality (Band 15) with configurable NA ratio
        if not validate_fuel_metrics_quality(tile, band_index=14, max_na_ratio=args.max_na_ratio):
            filtered_out_count += 1
            continue

        # Replace NA with 0 for appropriate bands
        tile, tile_na_counts = replace_na_with_defaults(tile)
        total_na_replaced += sum(tile_na_counts.values())
        for band_idx, count in tile_na_counts.items():
            na_per_band[band_idx] += count

        filtered_tiles.append(tile)

    logger.info(f"Filtered to {len(filtered_tiles)} valid tiles (removed {filtered_out_count})")
    if total_na_replaced > 0:
        logger.info(f"Replaced {total_na_replaced} total NaN pixels with 0 across {len([b for b in na_per_band.values() if b > 0])} bands")

    if len(filtered_tiles) == 0:
        logger.error("No valid tiles found after filtering!")
        return

    # Compute normalization statistics
    logger.info("\nComputing normalization statistics...")
    stats = compute_normalization_stats(filtered_tiles)

    # Save normalization stats to JSON
    stats_json_path = output_dir / 'fuel_metrics_normalization_stats.json'
    with open(stats_json_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved normalization stats to {stats_json_path}")

    # Random shuffle and split
    logger.info(f"\nPerforming random 90/10 train/val split...")
    random.shuffle(filtered_tiles)

    split_idx = int(len(filtered_tiles) * args.train_val_ratio)
    training_tiles = filtered_tiles[:split_idx]
    validation_tiles = filtered_tiles[split_idx:]

    logger.info(f"Training tiles: {len(training_tiles)}")
    logger.info(f"Validation tiles: {len(validation_tiles)}")

    # Normalize fuel metrics
    logger.info("\nNormalizing fuel metrics...")

    for tile_idx, tile in enumerate(training_tiles):
        normalize_fuel_metrics(tile, stats)
        if (tile_idx + 1) % 100 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(training_tiles)} training tiles...")

    for tile_idx, tile in enumerate(validation_tiles):
        normalize_fuel_metrics(tile, stats)
        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(validation_tiles)} validation tiles...")

    # Replace remaining NaN values with sentinel (-999)
    logger.info("\nReplacing remaining NaN values with sentinel (-999)...")
    nan_replacement_count = 0

    for tile in training_tiles + validation_tiles:
        if 'fuel_metrics' in tile and tile['fuel_metrics'] is not None:
            had_nans = torch.isnan(tile['fuel_metrics']).any()
            replace_remaining_nans_with_sentinel(tile, sentinel_value=-999.0)
            if had_nans:
                nan_replacement_count += 1

    logger.info(f"  Replaced NaN values in {nan_replacement_count} tiles")

    # Preprocess NAIP and UAVSAR imagery (IDENTICAL to original train_test_split_and_precompute.py)
    logger.info("\nPreprocessing NAIP imagery...")
    for tile_idx, tile in enumerate(training_tiles + validation_tiles):
        # Use first NAIP image date as reference if available, otherwise use current date
        if 'naip_dates' in tile and len(tile['naip_dates']) > 0:
            reference_date = parse_date(tile['naip_dates'][0])
        elif 'uavsar_dates' in tile and len(tile['uavsar_dates']) > 0:
            reference_date = parse_date(tile['uavsar_dates'][0])
        else:
            reference_date = datetime.now()  # Fallback

        # Preprocess NAIP imagery
        if 'naip_imgs' in tile and tile['naip_imgs'] is not None and tile['naip_imgs'].numel() > 0:
            naip_preprocessed = preprocess_naip_imagery(tile, reference_date, dtype=torch.float16)
            if naip_preprocessed is not None:
                tile['naip'] = naip_preprocessed
            else:
                tile['naip'] = None
        else:
            tile['naip'] = None

        # Preprocess UAVSAR imagery
        if 'uavsar_imgs' in tile and tile['uavsar_imgs'] is not None and tile['uavsar_imgs'].numel() > 0:
            uavsar_preprocessed = preprocess_uavsar_imagery(tile, reference_date, dtype=torch.float32)
            if uavsar_preprocessed is not None:
                tile['uavsar'] = uavsar_preprocessed
            else:
                tile['uavsar'] = None
        else:
            tile['uavsar'] = None

        if (tile_idx + 1) % 100 == 0:
            logger.info(f"  Preprocessed {tile_idx + 1}/{len(training_tiles) + len(validation_tiles)} tiles...")

    # Precompute KNN indices
    logger.info("\nPrecomputing KNN indices...")
    from torch_geometric.nn import knn_graph
    from torch_geometric.utils import to_undirected

    k_values = [15]  # Standard KNN value
    processed_count = 0

    for tile in training_tiles + validation_tiles:
        if 'dep_points_norm' in tile and isinstance(tile['dep_points_norm'], torch.Tensor):
            dep_points = tile['dep_points_norm']

            if dep_points.shape[0] > k_values[0]:
                knn_edges = {}
                for k in k_values:
                    edge_index = knn_graph(dep_points, k=k, loop=False)
                    edge_index = to_undirected(edge_index, num_nodes=dep_points.size(0))
                    knn_edges[k] = edge_index

                tile['knn_edge_indices'] = knn_edges

        processed_count += 1
        if processed_count % 100 == 0:
            logger.info(f"  Precomputed KNN for {processed_count}/{len(training_tiles) + len(validation_tiles)} tiles...")

    # Save training tiles
    logger.info(f"\nSaving precomputed training tiles...")
    train_output_path = output_dir / f'precomputed_training_tiles_raster_{args.precision}bit.pt'
    torch.save(training_tiles, train_output_path)
    logger.info(f"Saved to {train_output_path}")

    # Save validation tiles
    logger.info(f"Saving precomputed validation tiles...")
    val_output_path = output_dir / f'precomputed_validation_tiles_raster_{args.precision}bit.pt'
    torch.save(validation_tiles, val_output_path)
    logger.info(f"Saved to {val_output_path}")

    # Summary
    logger.info("\n" + "="*80)
    logger.info("PRECOMPUTATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Training tiles: {len(training_tiles)}")
    logger.info(f"Validation tiles: {len(validation_tiles)}")
    logger.info(f"Total: {len(training_tiles) + len(validation_tiles)}")
    logger.info(f"Normalization stats: {stats_json_path}")
    logger.info(f"Training output: {train_output_path}")
    logger.info(f"Validation output: {val_output_path}")


if __name__ == '__main__':
    main()
