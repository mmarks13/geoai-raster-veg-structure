#!/usr/bin/env python3
"""
Train/Validation Split and Precomputation for Raster-Based Model

Loads combined training data, applies quality filtering, splits into train/val,
normalizes target raster bands globally, preprocesses NAIP/UAVSAR imagery, and
precomputes KNN indices.

**GENERIC:** Works with any target raster (vegetation structure, fuel metrics, etc.)
- Auto-detects number of bands from data
- Rejects tiles with ANY NaN in target (NA handling must be done upstream)
- Applies z-score normalization to all bands

Input: Combined .pt file with tiles containing target_raster/fuel_metrics [n_bands, h, w]
Output:
  - precomputed_training_tiles_raster_32bit.pt (normalized, preprocessed)
  - precomputed_validation_tiles_raster_32bit.pt (normalized, preprocessed)
  - target_raster_normalization_stats_train.json (statistics)
  - rejected_tiles.log (tiles filtered out with reasons)
"""

import torch
import json
import argparse
import random
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple
import geopandas as gpd
from shapely.geometry import box

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def validate_tile_no_nan(tile, tile_id='unknown'):
    """
    Validate that target raster has NO NaN values (upstream should have handled all NA).

    Parameters:
        tile (dict): Tile with 'fuel_metrics' (or 'target_raster') key
        tile_id (str): Tile identifier for logging

    Returns:
        tuple: (is_valid: bool, nan_bands: list of band indices with NaN)
    """
    # Check for target raster under either key name
    target_key = 'fuel_metrics' if 'fuel_metrics' in tile else 'target_raster'

    if target_key not in tile:
        return False, [-1]  # -1 indicates missing target

    target_raster = tile[target_key]
    if target_raster is None:
        return False, [-1]

    # Check each band for NaN
    nan_bands = []
    n_bands = target_raster.shape[0]

    for band_idx in range(n_bands):
        band_data = target_raster[band_idx]
        if torch.isnan(band_data).any():
            nan_bands.append(band_idx)

    is_valid = len(nan_bands) == 0
    return is_valid, nan_bands


def compute_normalization_stats(tiles):
    """
    Compute per-band global mean and std for target raster.

    Auto-detects number of bands from the data.

    Parameters:
        tiles (list): List of tiles with 'fuel_metrics' [n_bands, h, w]

    Returns:
        dict: Statistics {'n_bands': n, 'band_0_mean': ..., 'band_0_std': ..., ...}

    Raises:
        ValueError: If no valid tiles found or if any band has no valid data
    """
    # Auto-detect number of bands from first valid tile
    n_bands = None
    for tile in tiles:
        if 'fuel_metrics' in tile and tile['fuel_metrics'] is not None:
            n_bands = tile['fuel_metrics'].shape[0]
            break

    if n_bands is None:
        raise ValueError("No valid tiles found to determine band count")

    logger.info(f"Computing normalization statistics for {n_bands} bands...")

    # Initialize accumulators for each band
    band_sums = [0.0] * n_bands
    band_sq_sums = [0.0] * n_bands
    band_counts = [0] * n_bands

    # Collect statistics from all tiles
    for tile_idx, tile in enumerate(tiles):
        if 'fuel_metrics' not in tile or tile['fuel_metrics'] is None:
            continue

        target_raster = tile['fuel_metrics']  # [n_bands, h, w]

        for band_idx in range(n_bands):
            band_data = target_raster[band_idx]
            # All data should be valid (NaN tiles filtered upstream)
            band_sums[band_idx] += band_data.sum().item()
            band_sq_sums[band_idx] += (band_data ** 2).sum().item()
            band_counts[band_idx] += band_data.numel()

        if (tile_idx + 1) % 500 == 0:
            logger.info(f"  Processed {tile_idx + 1}/{len(tiles)} tiles...")

    # Compute mean and std for each band
    stats = {'n_bands': n_bands}
    for band_idx in range(n_bands):
        if band_counts[band_idx] == 0:
            raise ValueError(f"Band {band_idx} has no valid data across all tiles!")

        mean = band_sums[band_idx] / band_counts[band_idx]
        sq_mean = band_sq_sums[band_idx] / band_counts[band_idx]
        variance = sq_mean - (mean ** 2)

        if variance < 0:
            raise ValueError(
                f"Band {band_idx} has negative variance ({variance:.6f}). "
                f"This indicates numerical issues in the data."
            )

        std = np.sqrt(max(variance, 1e-8))  # Small epsilon for numerical stability

        stats[f'band_{band_idx}_mean'] = float(mean)
        stats[f'band_{band_idx}_std'] = float(std)

    logger.info(f"Computed statistics for all {n_bands} bands")
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


def compute_global_coordinate_and_attribute_statistics(all_tiles):
    """
    Compute global mean and std for BBOX-NORMALIZED coordinates (X,Y,Z) and attributes.

    CRITICAL: This computes stats on coordinates AFTER normalize_point_clouds_with_bbox(),
    not on raw coordinates. This maintains spatial structure while standardizing distribution.

    Attribute handling (6 features):
        Index 0: Intensity
        Index 1: ReturnNumber
        Index 2: NumberOfReturns
        Index 3: Planarity [0,1]
        Index 4: Sphericity [0,1]
        Index 5: Verticality [0,1]

    Args:
        all_tiles: List of tile dicts with 'dep_points' and 'dep_pnt_attr'

    Returns:
        dict with keys:
            'coord_mean': [x_mean, y_mean, z_mean] (of bbox-normalized coords)
            'coord_std': [x_std, y_std, z_std] (of bbox-normalized coords)
            'attr_mean': [6] means
            'attr_std': [6] stds
            'total_points': int
            'total_tiles': int
    """
    logger.info("Computing global statistics on bbox-normalized coordinates...")

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

        # Step 1: Apply bbox normalization (existing function)
        # This brings coords to meter-scale: X,Y ∈ [-5,5]m, Z ∈ [0,max]m (relative to ground)
        uav_dummy = torch.empty((0, 3), dtype=torch.float32)
        dep_points_norm, _, _, _ = normalize_point_clouds_with_bbox(
            dep_points, uav_dummy, bbox, dtype=torch.float32
        )

        # Step 2: Clamp bbox-normalized Z to [0, 150] to remove bird returns
        dep_points_norm[:, 2] = torch.clamp(dep_points_norm[:, 2], 0, 150)

        # Collect bbox-normalized coordinates
        all_coords_norm.append(dep_points_norm)
        total_points += dep_points_norm.shape[0]

        # Collect attributes if present
        if dep_pnt_attr is not None:
            all_attrs.append(dep_pnt_attr)

        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Processed {tile_idx + 1}/{len(all_tiles)} tiles for stats computation...")

    if len(all_coords_norm) == 0:
        raise ValueError("No valid tiles found for statistics computation!")

    # Concatenate all bbox-normalized coordinates
    all_coords_norm = torch.cat(all_coords_norm, dim=0)  # [N_total, 3]

    # Compute coordinate statistics
    coord_mean = all_coords_norm.mean(dim=0).numpy()  # [3]
    coord_std = all_coords_norm.std(dim=0).numpy()    # [3]

    logger.info(f"  Coordinate stats (bbox-normalized):")
    logger.info(f"    X: mean={coord_mean[0]:.4f}, std={coord_std[0]:.4f}")
    logger.info(f"    Y: mean={coord_mean[1]:.4f}, std={coord_std[1]:.4f}")
    logger.info(f"    Z: mean={coord_mean[2]:.4f}, std={coord_std[2]:.4f}")

    # Compute attribute statistics if available
    attr_mean = None
    attr_std = None

    if len(all_attrs) > 0:
        all_attrs = torch.cat(all_attrs, dim=0)  # [N_total, num_attrs]
        num_attrs = all_attrs.shape[1]  # Can be 3 (legacy) or 6 (new)

        # Attribute names for logging
        attr_names = ['Intensity', 'ReturnNumber', 'NumberOfReturns',
                      'Planarity', 'Sphericity', 'Verticality']

        attr_mean = torch.zeros(num_attrs, dtype=torch.float64)
        attr_std = torch.zeros(num_attrs, dtype=torch.float64)

        for attr_idx in range(num_attrs):
            values = all_attrs[:, attr_idx].clone()

            # Handle NaN/Inf
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

        logger.info(f"  Attribute stats ({num_attrs} attributes):")
        for idx in range(num_attrs):
            name = attr_names[idx] if idx < len(attr_names) else f'Attr{idx}'
            logger.info(f"    {name}: mean={attr_mean[idx]:.4f}, std={attr_std[idx]:.4f}")

    # Convert to tensors (not lists) for weights_only=True compatibility
    return {
        'coord_mean': torch.from_numpy(coord_mean).float(),  # Tensor[3]
        'coord_std': torch.from_numpy(coord_std).float(),    # Tensor[3]
        'attr_mean': torch.from_numpy(attr_mean).float() if attr_mean is not None else None,  # Tensor[num_attrs]
        'attr_std': torch.from_numpy(attr_std).float() if attr_std is not None else None,     # Tensor[num_attrs]
        'total_points': int(total_points),
        'total_tiles': len([t for t in all_tiles if 'dep_points' in t])
    }


def remove_z_outliers_from_tile(tile, z_mean_bbox_norm, z_std_bbox_norm, n_std=4.0, log_file=None):
    """
    Remove points where bbox-normalized Z value exceeds mean + n_std * std.

    CRITICAL: Operates on bbox-normalized coordinates, not raw coords.

    Args:
        tile: Tile dict with 'dep_points' and 'dep_pnt_attr'
        z_mean_bbox_norm: Mean of Z after bbox normalization
        z_std_bbox_norm: Std of Z after bbox normalization
        n_std: Number of std devs for threshold (default 4.0)
        log_file: File handle for logging

    Returns:
        tile: Modified tile with outliers removed
        n_removed: Number of points removed
    """
    if 'dep_points' not in tile or tile['dep_points'] is None:
        return tile, 0

    dep_points = tile['dep_points'].clone()
    dep_pnt_attr = tile.get('dep_pnt_attr', None)
    bbox = tile.get('bbox', None)
    tile_id = tile.get('tile_id', 'unknown')

    if bbox is None:
        return tile, 0

    # Apply bbox normalization to get Z in same space as stats
    uav_dummy = torch.empty((0, 3), dtype=torch.float32)
    dep_points_norm, _, _, _ = normalize_point_clouds_with_bbox(
        dep_points, uav_dummy, bbox, dtype=torch.float32
    )

    # Clamp bbox-normalized Z to [0, 150]
    dep_points_norm[:, 2] = torch.clamp(dep_points_norm[:, 2], 0, 150)

    # Identify outliers based on bbox-normalized Z
    z_norm = dep_points_norm[:, 2]
    threshold = z_mean_bbox_norm + n_std * z_std_bbox_norm
    outlier_mask = z_norm > threshold
    n_removed = outlier_mask.sum().item()

    if n_removed > 0:
        # Log each outlier
        if log_file is not None:
            outlier_indices = torch.where(outlier_mask)[0]
            for idx in outlier_indices:
                z_val = z_norm[idx].item()
                n_std_above = (z_val - z_mean_bbox_norm) / z_std_bbox_norm if z_std_bbox_norm > 0 else 0
                log_file.write(
                    f"Tile {tile_id}, point {idx.item()}: "
                    f"Z_bbox_norm={z_val:.3f} ({n_std_above:.2f} std above mean)\n"
                )

        # Remove outliers from ORIGINAL coordinates
        keep_mask = ~outlier_mask
        tile['dep_points'] = dep_points[keep_mask]

        if dep_pnt_attr is not None:
            tile['dep_pnt_attr'] = dep_pnt_attr[keep_mask]

    return tile, n_removed


def apply_zscore_to_bbox_normalized_coords(dep_points_bbox_norm, coord_mean, coord_std):
    """
    Apply z-score normalization to bbox-normalized coordinates.

    CRITICAL: Input coordinates are ALREADY bbox-normalized (X,Y ∈ [-5,5]m, Z ∈ [0,max]m).
    This function standardizes the distribution to mean=0, std=1.

    Args:
        dep_points_bbox_norm: [N, 3] bbox-normalized coordinates
        coord_mean: [3] means of bbox-normalized coords (from global stats)
        coord_std: [3] stds of bbox-normalized coords (from global stats)

    Returns:
        dep_points_zscore: [N, 3] z-score normalized coordinates
    """
    coord_mean_t = torch.tensor(coord_mean, dtype=dep_points_bbox_norm.dtype, device=dep_points_bbox_norm.device)
    coord_std_t = torch.tensor(coord_std, dtype=dep_points_bbox_norm.dtype, device=dep_points_bbox_norm.device)

    # Z-score: (x - mean) / std
    dep_points_zscore = (dep_points_bbox_norm - coord_mean_t) / coord_std_t

    return dep_points_zscore


def normalize_attributes_zscore(dep_pnt_attr, attr_mean, attr_std, dtype=torch.float16):
    """
    Normalize point attributes using z-score.

    Handles NaN/Inf by setting to mean before normalization.

    Attribute indices:
        0: Intensity
        1: ReturnNumber
        2: NumberOfReturns
        3: Planarity
        4: Sphericity
        5: Verticality

    Args:
        dep_pnt_attr: [N, num_attrs] raw attributes (3 for legacy, 6 for new)
        attr_mean: [num_attrs] global means
        attr_std: [num_attrs] global stds
        dtype: Output dtype (default float16 for memory)

    Returns:
        dep_points_attr_norm: [N, num_attrs] normalized attributes
    """
    if dep_pnt_attr is None:
        return None

    dep_points_attr_norm = dep_pnt_attr.clone()
    num_attrs = dep_points_attr_norm.shape[1]

    # Handle invalid values → set to mean
    invalid_mask = torch.isnan(dep_points_attr_norm) | torch.isinf(dep_points_attr_norm)

    attr_mean_t = torch.tensor(attr_mean, dtype=dep_points_attr_norm.dtype, device=dep_points_attr_norm.device)
    attr_std_t = torch.tensor(attr_std, dtype=dep_points_attr_norm.dtype, device=dep_points_attr_norm.device)

    for attr_idx in range(num_attrs):
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

    # Center-crop to square if dimensions don't match
    # This maintains centroid alignment and produces expected 40x40 for encoder
    h, w = images.shape[-2:]
    if h != w:
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        images = images[..., start_h:start_h + min_dim, start_w:start_w + min_dim]

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

    # Center-crop to square if dimensions don't match
    # This maintains centroid alignment and produces expected 4x4 for encoder
    h, w = images.shape[-2:]
    if h != w:
        min_dim = min(h, w)
        start_h = (h - min_dim) // 2
        start_w = (w - min_dim) // 2
        images = images[..., start_h:start_h + min_dim, start_w:start_w + min_dim]

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


def normalize_target_raster(tile, stats):
    """
    Apply z-score normalization to target raster using pre-computed statistics.

    Parameters:
        tile (dict): Tile to normalize (modified in-place)
        stats (dict): Normalization statistics with 'n_bands', 'band_0_mean', 'band_0_std', etc.

    Returns:
        dict: Modified tile with normalized fuel_metrics

    Raises:
        ValueError: If stats are missing for any band
    """
    if 'fuel_metrics' not in tile or tile['fuel_metrics'] is None:
        return tile

    target_raster = tile['fuel_metrics'].clone()
    n_bands = target_raster.shape[0]

    # Validate stats has all required keys
    if 'n_bands' not in stats:
        raise ValueError("Stats dict missing 'n_bands' key")
    if stats['n_bands'] != n_bands:
        raise ValueError(
            f"Band count mismatch: tile has {n_bands} bands, stats computed for {stats['n_bands']}"
        )

    for band_idx in range(n_bands):
        mean_key = f'band_{band_idx}_mean'
        std_key = f'band_{band_idx}_std'

        if mean_key not in stats or std_key not in stats:
            raise ValueError(f"Stats missing for band {band_idx}: need '{mean_key}' and '{std_key}'")

        mean = stats[mean_key]
        std = stats[std_key]

        # Apply z-score normalization (no NaN handling - should be filtered upstream)
        target_raster[band_idx] = (target_raster[band_idx] - mean) / std

    tile['fuel_metrics'] = target_raster
    return tile


def report_distribution_shift(train_stats, val_stats, dataset_name="Validation"):
    """
    Report distribution shift between training and validation/test sets.

    Large shifts (>1.0 std) indicate potential domain mismatch.

    Args:
        train_stats: Dict with 'coord_mean', 'coord_std', 'attr_mean', 'attr_std'
        val_stats: Dict with same structure
        dataset_name: Name of comparison dataset (e.g., "Validation", "Forest Plots")

    Returns:
        dict with shift metrics
    """
    logger.info("\n" + "="*80)
    logger.info(f"DISTRIBUTION SHIFT ANALYSIS ({dataset_name} vs Training)")
    logger.info("="*80)

    # Coordinate shift
    coord_shift = []
    for dim, name in enumerate(['X', 'Y', 'Z']):
        mean_diff = abs(val_stats['coord_mean'][dim] - train_stats['coord_mean'][dim])
        std_diff = abs(val_stats['coord_std'][dim] - train_stats['coord_std'][dim])

        # Normalize by training std
        shift_in_std = mean_diff / train_stats['coord_std'][dim]
        std_ratio = val_stats['coord_std'][dim] / train_stats['coord_std'][dim]
        coord_shift.append(shift_in_std)

        logger.info(f"  {name}: mean shift = {shift_in_std:.3f} std, "
                   f"std ratio = {std_ratio:.3f}")

    # Attribute shift (if available)
    attr_shift = []
    if train_stats.get('attr_mean') is not None and val_stats.get('attr_mean') is not None:
        for dim, name in enumerate(['Intensity', 'ReturnNumber', 'NumberOfReturns']):
            mean_diff = abs(val_stats['attr_mean'][dim] - train_stats['attr_mean'][dim])
            shift_in_std = mean_diff / train_stats['attr_std'][dim]
            attr_shift.append(shift_in_std)

            logger.info(f"  {name}: mean shift = {shift_in_std:.3f} std")

    # Overall assessment
    max_coord_shift = max(coord_shift)
    if max_coord_shift > 1.0:
        logger.warning(f"  ⚠️  SEVERE coordinate shift detected ({max_coord_shift:.2f} std)")
        logger.warning(f"      {dataset_name} distribution differs significantly from training!")
    elif max_coord_shift > 0.5:
        logger.warning(f"  ⚠️  Moderate coordinate shift detected ({max_coord_shift:.2f} std)")
        logger.warning(f"      Monitor {dataset_name.lower()} performance carefully")
    else:
        logger.info(f"  ✓ Acceptable coordinate shift ({max_coord_shift:.2f} std)")

    logger.info("="*80)

    return {
        'coord_shift': coord_shift,
        'attr_shift': attr_shift,
        'max_shift': max_coord_shift
    }


def split_tiles_by_spatial_intersection(
    tiles: List[dict],
    geojson_file_path: str,
    crs: str = 'EPSG:32611'
) -> Tuple[List[dict], List[dict]]:
    """
    Split tiles into training and validation sets based on spatial intersection
    with validation polygons from a GeoJSON file.

    Tiles that intersect with validation polygons → validation set
    Tiles that don't intersect → training set

    Parameters:
        tiles: List of tile dicts with 'bbox' key [xmin, ymin, xmax, ymax]
        geojson_file_path: Path to GeoJSON file containing validation polygons
        crs: Coordinate reference system (default: EPSG:32611 UTM 11N)

    Returns:
        Tuple of (training_tiles, validation_tiles)
    """
    logger.info(f"Loading validation polygons from {geojson_file_path}...")
    val_polygons = gpd.read_file(geojson_file_path)
    logger.info(f"Loaded {len(val_polygons)} validation polygons")

    # Check/set CRS
    if val_polygons.crs is None:
        logger.warning(f"GeoJSON CRS is not defined. Assuming {crs}.")
        val_polygons.crs = crs
    elif str(val_polygons.crs) != crs:
        logger.info(f"Reprojecting validation polygons from {val_polygons.crs} to {crs}")
        val_polygons = val_polygons.to_crs(crs)

    # Create GeoDataFrame from tile bounding boxes
    logger.info("Creating GeoDataFrame from tile bounding boxes...")
    tile_geoms = []
    tile_indices = []

    for idx, tile in enumerate(tiles):
        bbox = tile.get('bbox', None)
        if bbox is None:
            logger.warning(f"Tile {tile.get('tile_id', idx)} has no bbox, skipping spatial check")
            continue

        # Handle both tensor and list/array bbox formats
        if hasattr(bbox, 'tolist'):
            bbox = bbox.tolist()
        elif hasattr(bbox, 'numpy'):
            bbox = bbox.numpy().tolist()

        minx, miny, maxx, maxy = bbox
        tile_geoms.append(box(minx, miny, maxx, maxy))
        tile_indices.append(idx)

    tile_gdf = gpd.GeoDataFrame(geometry=tile_geoms, crs=crs)
    tile_gdf['tile_idx'] = tile_indices

    # Perform spatial join to find tiles that intersect with validation polygons
    logger.info("Performing spatial join...")
    joined = gpd.sjoin(tile_gdf, val_polygons, predicate='intersects', how='left')

    # Get indices of tiles that intersect with validation polygons
    val_indices = set(joined.loc[~joined['index_right'].isna(), 'tile_idx'])
    logger.info(f"Found {len(val_indices)} tiles that intersect with validation polygons")

    # Split tiles
    training_tiles = []
    validation_tiles = []

    for idx, tile in enumerate(tiles):
        if idx in val_indices:
            validation_tiles.append(tile)
        else:
            training_tiles.append(tile)

    # Log site breakdown
    train_sites = {}
    val_sites = {}
    for tile in training_tiles:
        site = tile.get('site', 'unknown')
        train_sites[site] = train_sites.get(site, 0) + 1
    for tile in validation_tiles:
        site = tile.get('site', 'unknown')
        val_sites[site] = val_sites.get(site, 0) + 1

    logger.info("Training tiles by site:")
    for site, count in sorted(train_sites.items()):
        logger.info(f"  {site}: {count}")
    logger.info("Validation tiles by site:")
    for site, count in sorted(val_sites.items()):
        logger.info(f"  {site}: {count}")

    return training_tiles, validation_tiles


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
        '--geojson-file',
        type=str,
        default=None,
        help='Path to GeoJSON file containing validation polygons for spatial split. '
             'If provided, uses spatial intersection for train/val split. '
             'If not provided, uses random split based on --train-val-ratio.'
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
        default=20,
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

    args = parser.parse_args()

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

    # ============================================================================
    # STEP 1: FIX KEY NAMING AND CONVERT TO TENSORS
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 1: FIX KEY NAMING AND CONVERT TO TENSORS")
    logger.info("="*80)

    # Convert numpy scalars to Python native types for weights_only=True compatibility
    numpy_scalar_fields = [
        'fuel_metrics_resolution',
        'has_fuel_metrics',
        'has_imagery',
        'has_naip',
        'has_pointcloud',
        'has_uavsar',
        'initial_voxel_size_cm',
        'naip_resolution',
        'uavsar_resolution'
    ]

    for tile in all_tiles:
        # Handle different key naming conventions for fuel metrics
        if 'fuel_metrics_fuel_metrics' in tile and 'fuel_metrics' not in tile:
            tile['fuel_metrics'] = tile.pop('fuel_metrics_fuel_metrics')
        elif 'target_raster' in tile and 'fuel_metrics' not in tile:
            # New pipeline uses 'target_raster' instead of 'fuel_metrics'
            tile['fuel_metrics'] = tile.pop('target_raster')

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
                    # Convert numpy scalars to Python float/int
                    tile[field] = float(value) if isinstance(value, np.floating) else int(value)
                elif isinstance(value, np.bool_):
                    # Convert numpy bool to Python bool
                    tile[field] = bool(value)

    logger.info("Fixed key naming, converted arrays to tensors, and converted numpy scalars to Python types")

    # ============================================================================
    # STEP 3: PRE-FILTER TILES (on raw data, before normalization)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 3: PRE-FILTER TILES BY QUALITY")
    logger.info("="*80)
    logger.info(f"Filtering criteria: min_dep_points={args.min_dep_points}, no NaN in target raster")

    filtered_tiles = []
    rejected_tiles = []  # Track rejected tiles for logging
    missing_dep_count = 0
    low_dep_count = 0
    nan_count = 0
    nan_bands_summary = {}  # band_idx -> count of tiles with NaN in that band

    for tile_idx, tile in enumerate(all_tiles):
        tile_id = tile.get('tile_id', f'tile_{tile_idx}')

        # Check DEP points
        if 'dep_points' not in tile or tile['dep_points'] is None:
            missing_dep_count += 1
            rejected_tiles.append((tile_id, 'missing_dep_points', []))
            continue

        dep_count = tile['dep_points'].shape[0]
        if dep_count < args.min_dep_points:
            low_dep_count += 1
            rejected_tiles.append((tile_id, f'low_dep_points ({dep_count})', []))
            continue

        # Check for NaN in target raster (should be handled upstream)
        is_valid, nan_bands = validate_tile_no_nan(tile, tile_id)
        if not is_valid:
            nan_count += 1
            rejected_tiles.append((tile_id, 'nan_in_target', nan_bands))
            for band_idx in nan_bands:
                nan_bands_summary[band_idx] = nan_bands_summary.get(band_idx, 0) + 1
            continue

        filtered_tiles.append(tile)

    # Log filtering summary
    total_rejected = missing_dep_count + low_dep_count + nan_count
    logger.info(f"Filtered to {len(filtered_tiles)} valid tiles (removed {total_rejected})")
    if total_rejected > 0:
        logger.info(f"  - Missing dep_points: {missing_dep_count}")
        logger.info(f"  - Low dep_points (<{args.min_dep_points}): {low_dep_count}")
        logger.info(f"  - NaN in target raster: {nan_count}")

    # Write rejected tiles log (detailed info for debugging)
    rejected_log_path = output_dir / 'rejected_tiles.log'
    with open(rejected_log_path, 'w') as f:
        f.write("tile_id,reason,nan_bands\n")
        for tile_id, reason, nan_bands in rejected_tiles:
            nan_bands_str = ';'.join(map(str, nan_bands)) if nan_bands else ''
            f.write(f"{tile_id},{reason},{nan_bands_str}\n")
    logger.info(f"Rejected tiles log: {rejected_log_path}")

    if len(filtered_tiles) == 0:
        raise ValueError("No valid tiles found after filtering!")

    # ============================================================================
    # STEP 4: TRAIN/VAL SPLIT (spatial or random)
    # ============================================================================
    logger.info("\n" + "="*80)
    if args.geojson_file:
        logger.info("STEP 4: SPATIAL TRAIN/VAL SPLIT (using validation polygons)")
        logger.info("="*80)

        training_tiles, validation_tiles = split_tiles_by_spatial_intersection(
            filtered_tiles,
            args.geojson_file,
            crs='EPSG:32611'
        )
    else:
        logger.info(f"STEP 4: RANDOM TRAIN/VAL SPLIT ({args.train_val_ratio*100:.0f}/{(1-args.train_val_ratio)*100:.0f})")
        logger.info("="*80)

        random.shuffle(filtered_tiles)
        split_idx = int(len(filtered_tiles) * args.train_val_ratio)
        training_tiles = filtered_tiles[:split_idx]
        validation_tiles = filtered_tiles[split_idx:]

    logger.info(f"Training tiles: {len(training_tiles)}")
    logger.info(f"Validation tiles: {len(validation_tiles)}")

    # ============================================================================
    # STEP 5: COMPUTE COORDINATE/ATTRIBUTE STATISTICS (on TRAINING tiles only)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 5: COMPUTE COORDINATE/ATTRIBUTE STATISTICS (TRAINING ONLY)")
    logger.info("="*80)

    norm_stats_train = compute_global_coordinate_and_attribute_statistics(training_tiles)

    # Save TRAINING coordinate normalization stats (convert tensors to lists for JSON)
    coord_stats_path = output_dir / 'coordinate_normalization_stats_train.json'
    norm_stats_json = {
        'coord_mean': norm_stats_train['coord_mean'].tolist(),
        'coord_std': norm_stats_train['coord_std'].tolist(),
        'attr_mean': norm_stats_train['attr_mean'].tolist() if norm_stats_train['attr_mean'] is not None else None,
        'attr_std': norm_stats_train['attr_std'].tolist() if norm_stats_train['attr_std'] is not None else None,
        'total_points': norm_stats_train['total_points'],
        'total_tiles': norm_stats_train['total_tiles']
    }
    with open(coord_stats_path, 'w') as f:
        json.dump(norm_stats_json, f, indent=2)
    logger.info(f"Saved TRAINING coordinate normalization stats to {coord_stats_path}")

    # ============================================================================
    # STEP 6: COMPUTE TARGET RASTER STATISTICS (on TRAINING tiles only)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 6: COMPUTE TARGET RASTER STATISTICS (TRAINING ONLY)")
    logger.info("="*80)

    fuel_stats_train = compute_normalization_stats(training_tiles)

    # Save TRAINING target raster normalization stats
    fuel_stats_path = output_dir / 'target_raster_normalization_stats_train.json'
    with open(fuel_stats_path, 'w') as f:
        json.dump(fuel_stats_train, f, indent=2)
    logger.info(f"Saved TRAINING target raster normalization stats to {fuel_stats_path}")

    # ============================================================================
    # STEP 7: COMPUTE VALIDATION STATISTICS (for distribution shift analysis)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 7: COMPUTE VALIDATION STATISTICS (diagnostic only)")
    logger.info("="*80)

    norm_stats_val = compute_global_coordinate_and_attribute_statistics(validation_tiles)
    fuel_stats_val = compute_normalization_stats(validation_tiles)

    # Save validation statistics (diagnostic only, NOT used for normalization)
    # Convert tensors to lists for JSON
    coord_stats_val_path = output_dir / 'coordinate_normalization_stats_val.json'
    norm_stats_val_json = {
        'coord_mean': norm_stats_val['coord_mean'].tolist(),
        'coord_std': norm_stats_val['coord_std'].tolist(),
        'attr_mean': norm_stats_val['attr_mean'].tolist() if norm_stats_val['attr_mean'] is not None else None,
        'attr_std': norm_stats_val['attr_std'].tolist() if norm_stats_val['attr_std'] is not None else None,
        'total_points': norm_stats_val['total_points'],
        'total_tiles': norm_stats_val['total_tiles']
    }
    with open(coord_stats_val_path, 'w') as f:
        json.dump(norm_stats_val_json, f, indent=2)
    logger.info(f"Saved validation coordinate stats (diagnostic) to {coord_stats_val_path}")

    fuel_stats_val_path = output_dir / 'target_raster_normalization_stats_val.json'
    with open(fuel_stats_val_path, 'w') as f:
        json.dump(fuel_stats_val, f, indent=2)
    logger.info(f"Saved validation target raster stats (diagnostic) to {fuel_stats_val_path}")

    # Report distribution shift
    report_distribution_shift(norm_stats_train, norm_stats_val, dataset_name="Validation")

    # ============================================================================
    # STEP 8: APPLY COORDINATE NORMALIZATION TO TRAINING TILES
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 8: APPLY COORDINATE NORMALIZATION TO TRAINING TILES")
    logger.info("="*80)

    for tile_idx, tile in enumerate(training_tiles):
        if 'dep_points' in tile and 'dep_points_norm' not in tile:
            # Load coordinates AND attributes
            dep_points = tile.pop('dep_points')
            dep_pnt_attr = tile.get('dep_pnt_attr', None)

            # Convert to tensors
            if isinstance(dep_points, np.ndarray):
                dep_points = torch.from_numpy(dep_points).float()
            if dep_pnt_attr is not None and isinstance(dep_pnt_attr, np.ndarray):
                dep_pnt_attr = torch.from_numpy(dep_pnt_attr).float()

            # Get UAV points and bbox
            uav_points = tile.get('uav_points', torch.empty((0, 3), dtype=torch.float32))
            if isinstance(uav_points, np.ndarray):
                uav_points = torch.from_numpy(uav_points).float()

            bbox = tile.get('bbox', None)

            if bbox is not None:
                if isinstance(bbox, np.ndarray):
                    bbox = torch.from_numpy(bbox).float()

                # Step 1: Apply bbox normalization
                dep_points_bbox_norm, uav_points_norm, center, scale = normalize_point_clouds_with_bbox(
                    dep_points, uav_points, bbox, dtype=torch.float32
                )

                # Step 1b: Clamp bbox-normalized Z to [0, 150]
                dep_points_bbox_norm[:, 2] = torch.clamp(dep_points_bbox_norm[:, 2], 0, 150)

                # Step 2: Apply z-score normalization using TRAINING stats
                dep_points_norm = apply_zscore_to_bbox_normalized_coords(
                    dep_points_bbox_norm,
                    norm_stats_train['coord_mean'],
                    norm_stats_train['coord_std']
                )

                # Step 3: Normalize attributes using TRAINING stats
                dep_points_attr_norm = None
                if dep_pnt_attr is not None and norm_stats_train['attr_mean'] is not None:
                    dep_points_attr_norm = normalize_attributes_zscore(
                        dep_pnt_attr,
                        norm_stats_train['attr_mean'],
                        norm_stats_train['attr_std'],
                        dtype=torch.float16
                    )

                # Store all normalized versions
                tile['dep_points_norm'] = dep_points_norm
                tile['dep_points_bbox_norm'] = dep_points_bbox_norm
                tile['uav_points'] = uav_points_norm
                tile['dep_points_attr_norm'] = dep_points_attr_norm
                tile['center'] = center
                tile['scale'] = scale

                # Store norm_params using TRAINING stats (tensors, not lists)
                tile['norm_params'] = {
                    'coord_mean': norm_stats_train['coord_mean'],  # Tensor[3] float32
                    'coord_std': norm_stats_train['coord_std'],    # Tensor[3] float32
                    'attr_mean': norm_stats_train['attr_mean'],    # Tensor[num_attrs] float32
                    'attr_std': norm_stats_train['attr_std']       # Tensor[num_attrs] float32
                }
            else:
                logger.warning(f"No bbox found for tile {tile.get('tile_id', 'unknown')}, skipping normalization")
                tile['dep_points_norm'] = dep_points
                tile['dep_points_attr_norm'] = None

        if (tile_idx + 1) % 100 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(training_tiles)} training tiles...")

    logger.info("Training coordinate normalization complete")

    # ============================================================================
    # STEP 9: APPLY COORDINATE NORMALIZATION TO VALIDATION TILES (using TRAINING stats)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 9: APPLY COORDINATE NORMALIZATION TO VALIDATION TILES (using TRAINING stats)")
    logger.info("="*80)

    for tile_idx, tile in enumerate(validation_tiles):
        if 'dep_points' in tile and 'dep_points_norm' not in tile:
            # Load coordinates AND attributes
            dep_points = tile.pop('dep_points')
            dep_pnt_attr = tile.get('dep_pnt_attr', None)

            # Convert to tensors
            if isinstance(dep_points, np.ndarray):
                dep_points = torch.from_numpy(dep_points).float()
            if dep_pnt_attr is not None and isinstance(dep_pnt_attr, np.ndarray):
                dep_pnt_attr = torch.from_numpy(dep_pnt_attr).float()

            # Get UAV points and bbox
            uav_points = tile.get('uav_points', torch.empty((0, 3), dtype=torch.float32))
            if isinstance(uav_points, np.ndarray):
                uav_points = torch.from_numpy(uav_points).float()

            bbox = tile.get('bbox', None)

            if bbox is not None:
                if isinstance(bbox, np.ndarray):
                    bbox = torch.from_numpy(bbox).float()

                # Step 1: Apply bbox normalization
                dep_points_bbox_norm, uav_points_norm, center, scale = normalize_point_clouds_with_bbox(
                    dep_points, uav_points, bbox, dtype=torch.float32
                )

                # Step 1b: Clamp bbox-normalized Z to [0, 150]
                dep_points_bbox_norm[:, 2] = torch.clamp(dep_points_bbox_norm[:, 2], 0, 150)

                # Step 2: Apply z-score normalization using TRAINING stats
                dep_points_norm = apply_zscore_to_bbox_normalized_coords(
                    dep_points_bbox_norm,
                    norm_stats_train['coord_mean'],  # TRAINING stats!
                    norm_stats_train['coord_std']    # TRAINING stats!
                )

                # Step 3: Normalize attributes using TRAINING stats
                dep_points_attr_norm = None
                if dep_pnt_attr is not None and norm_stats_train['attr_mean'] is not None:
                    dep_points_attr_norm = normalize_attributes_zscore(
                        dep_pnt_attr,
                        norm_stats_train['attr_mean'],  # TRAINING stats!
                        norm_stats_train['attr_std'],   # TRAINING stats!
                        dtype=torch.float16
                    )

                # Store all normalized versions
                tile['dep_points_norm'] = dep_points_norm
                tile['dep_points_bbox_norm'] = dep_points_bbox_norm
                tile['uav_points'] = uav_points_norm
                tile['dep_points_attr_norm'] = dep_points_attr_norm
                tile['center'] = center
                tile['scale'] = scale

                # Store norm_params using TRAINING stats (tensors, not lists)
                tile['norm_params'] = {
                    'coord_mean': norm_stats_train['coord_mean'],  # Tensor[3] float32
                    'coord_std': norm_stats_train['coord_std'],    # Tensor[3] float32
                    'attr_mean': norm_stats_train['attr_mean'],    # Tensor[num_attrs] float32
                    'attr_std': norm_stats_train['attr_std']       # Tensor[num_attrs] float32
                }
            else:
                logger.warning(f"No bbox found for tile {tile.get('tile_id', 'unknown')}, skipping normalization")
                tile['dep_points_norm'] = dep_points
                tile['dep_points_attr_norm'] = None

        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(validation_tiles)} validation tiles...")

    logger.info("Validation coordinate normalization complete")

    # ============================================================================
    # STEP 10: APPLY TARGET RASTER NORMALIZATION TO TRAINING TILES
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 10: APPLY TARGET RASTER NORMALIZATION TO TRAINING TILES")
    logger.info("="*80)

    for tile_idx, tile in enumerate(training_tiles):
        normalize_target_raster(tile, fuel_stats_train)
        if (tile_idx + 1) % 100 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(training_tiles)} training tiles...")

    logger.info("Training target raster normalization complete")

    # ============================================================================
    # STEP 11: APPLY TARGET RASTER NORMALIZATION TO VALIDATION TILES (using TRAINING stats)
    # ============================================================================
    logger.info("\n" + "="*80)
    logger.info("STEP 11: APPLY TARGET RASTER NORMALIZATION TO VALIDATION TILES (using TRAINING stats)")
    logger.info("="*80)

    for tile_idx, tile in enumerate(validation_tiles):
        normalize_target_raster(tile, fuel_stats_train)  # Use TRAINING stats!
        if (tile_idx + 1) % 50 == 0:
            logger.info(f"  Normalized {tile_idx + 1}/{len(validation_tiles)} validation tiles...")

    # Preprocess NAIP and UAVSAR imagery
    logger.info("\nPreprocessing NAIP imagery...")
    for tile_idx, tile in enumerate(training_tiles + validation_tiles):
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
    logger.info(f"\nNormalization statistics (TRAINING - used for both train/val):")
    logger.info(f"  Coordinate stats: {coord_stats_path}")
    logger.info(f"  Target raster stats: {fuel_stats_path}")
    logger.info(f"\nValidation statistics (diagnostic only):")
    logger.info(f"  Coordinate stats: {coord_stats_val_path}")
    logger.info(f"  Target raster stats: {fuel_stats_val_path}")
    logger.info(f"\nOutput files:")
    logger.info(f"  Training: {train_output_path}")
    logger.info(f"  Validation: {val_output_path}")


if __name__ == '__main__':
    main()
