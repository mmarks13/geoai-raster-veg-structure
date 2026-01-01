"""
Dataset and collate functions for raster fuel metrics prediction model.

This module provides data loading infrastructure for the raster prediction pipeline,
which predicts fuel metrics rasters directly from sparse LiDAR + imagery.

Key differences from point cloud pipeline:
- Uses z-score normalized coordinates (not just bbox-normalized)
- Returns fuel_metrics ground truth (not uav_points)
- Includes norm_params for denormalization in attention layers
"""

import torch
from torch.utils.data import Dataset
from typing import Dict, List, Tuple, Optional
import numpy as np


class ShardedRasterDataset(Dataset):
    """
    Dataset for raster model training with sharded data loading.

    Loads preprocessed tiles from raster pipeline with z-score normalized
    coordinates and fuel metrics ground truth.

    Note:
        Edge indices (KNN graph) are NOT loaded or returned. The raster model uses
        global-only attention (use_global_only=True) which does not require local
        KNN graphs. Global attention computes positions dynamically without edge_index.
        This eliminates significant data loading and collation overhead.

    Args:
        shard_path: Path to .pt file containing list of tile dicts
        use_naip: Whether to include NAIP optical imagery
        use_uavsar: Whether to include UAVSAR L-band SAR imagery
        target_band_indices: Indices of fuel metrics bands to predict (0-indexed)
                           Default [2, 7, 14] = [Height, TFL, Total_cover]

    Returns:
        Tuple of (dep_points_norm, fuel_metrics_norm, dep_points_attr_norm,
                  naip_data, uavsar_data, center, scale, bbox, tile_id, norm_params)

        - dep_points_norm: [N, 3] z-score normalized point coordinates
        - fuel_metrics_norm: [n_bands, 5, 5] z-score normalized target raster
        - dep_points_attr_norm: [N, 3] normalized attributes (intensity, return, nreturns)
        - naip_data: Dict with 'images' [n_imgs, 4, 40, 40] or None
        - uavsar_data: Dict with 'images' [n_imgs, 6, 4, 4] or None
        - center: [1, 3] bbox normalization center
        - scale: [1, 3] bbox normalization scale
        - bbox: [4] original bbox [xmin, ymin, xmax, ymax]
        - tile_id: str unique identifier
        - norm_params: Dict with coord_mean, coord_std, attr_mean, attr_std
    """

    def __init__(
        self,
        shard_path: str,
        use_naip: bool = False,
        use_uavsar: bool = False,
        target_band_indices: List[int] = [2, 7, 14]
    ):
        self.data = torch.load(shard_path, weights_only=False)
        self.use_naip = use_naip
        self.use_uavsar = use_uavsar
        self.target_band_indices = target_band_indices

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a single tile with all inputs and targets.

        Returns:
            10-element tuple for raster model training (edge_index not included -
            global attention does not require KNN graph)
        """
        sample = self.data[idx]

        # Extract z-score normalized point cloud data
        dep_points_norm = sample['dep_points_norm']  # [N, 3] z-score normalized
        dep_points_attr_norm = sample['dep_points_attr_norm']  # [N, 3]

        # Note: edge_index not loaded - global-only attention doesn't use KNN graphs

        # Extract fuel metrics target (z-score normalized) and select target bands
        fuel_metrics_full = sample['fuel_metrics']  # [22, 5, 5]
        fuel_metrics_norm = fuel_metrics_full[self.target_band_indices]  # [n_bands, 5, 5]

        # Extract normalization parameters (needed for denormalization in attention)
        norm_params = sample['norm_params']  # Dict with coord_mean, coord_std, etc.

        # Extract metadata
        center = sample['center']  # [1, 3]
        scale = sample['scale']    # [1, 3]
        bbox = sample['bbox']      # [4]
        tile_id = sample['tile_id']  # str

        # Extract imagery data (already preprocessed in correct format)
        naip_data = None
        if self.use_naip and sample.get('naip') is not None:
            # Data already has 'images' key with shape [n_imgs, 4, 40, 40]
            naip_data = sample['naip']

        uavsar_data = None
        if self.use_uavsar and sample.get('uavsar') is not None:
            # Data already has 'images' key
            uavsar_data = sample['uavsar']

        # Return 10-element tuple (edge_index removed for global-only attention)
        return (
            dep_points_norm,
            fuel_metrics_norm,
            dep_points_attr_norm,
            naip_data,
            uavsar_data,
            center,
            scale,
            bbox,
            tile_id,
            norm_params
        )


def raster_variable_size_collate(batch: List[Tuple]) -> Tuple:
    """
    Collate function for variable-size point clouds in raster model.

    Converts list of tile tuples into batched format suitable for
    PyTorch Geometric-style models with batch indexing.

    Note:
        Edge indices are not collated - the raster model uses global-only attention
        which does not require KNN graphs.

    Args:
        batch: List of 10-element tuples from ShardedRasterDataset

    Returns:
        Tuple of (dep_points_batch, fuel_metrics_batch, dep_points_attr_batch,
                  naip_data_batch, uavsar_data_batch, centers, scales, bboxes,
                  tile_ids, norm_params_batch, batch_indices)

        - dep_points_batch: [N_total, 3] concatenated z-score normalized points
        - fuel_metrics_batch: [B, n_bands, 5, 5] stacked target rasters
        - dep_points_attr_batch: [N_total, 3] concatenated attributes
        - naip_data_batch: List[Dict] or None - per-tile NAIP data
        - uavsar_data_batch: List[Dict] or None - per-tile UAVSAR data
        - centers: [B, 1, 3] bbox centers
        - scales: [B, 1, 3] bbox scales
        - bboxes: [B, 4] original bboxes
        - tile_ids: List[str] of length B
        - norm_params_batch: List[Dict] of length B (each dict has coord_mean, coord_std, etc.)
        - batch_indices: [N_total] batch assignment for each point
    """
    # Single-pass unpack using zip (optimized from 10 list comprehensions)
    (dep_points_list, fuel_metrics_list, dep_points_attr_list,
     naip_data_list, uavsar_data_list, centers, scales, bboxes,
     tile_ids, norm_params_list) = zip(*batch)

    # Concatenate point clouds with batch indexing (PyTorch Geometric style)
    # Vectorized implementation: compute batch indices using repeat_interleave
    sizes = torch.tensor([p.shape[0] for p in dep_points_list], dtype=torch.long)
    batch_indices = torch.repeat_interleave(torch.arange(len(sizes)), sizes)  # [N_total]
    dep_points_batch = torch.cat(dep_points_list, dim=0)  # [N_total, 3]
    dep_points_attr_batch = torch.cat(dep_points_attr_list, dim=0)  # [N_total, 3]

    # Stack fuel metrics (fixed size rasters)
    fuel_metrics_batch = torch.stack(fuel_metrics_list, dim=0)  # [B, n_bands, 5, 5]

    # Stack metadata
    centers = torch.stack(centers, dim=0)  # [B, 1, 3]
    scales = torch.stack(scales, dim=0)    # [B, 1, 3]
    bboxes = torch.stack(bboxes, dim=0)    # [B, 4]

    # Collate NAIP data - pass through list of dicts
    naip_data_batch = list(naip_data_list) if any(item is not None for item in naip_data_list) else None

    # Collate UAVSAR data - pass through list of dicts
    uavsar_data_batch = list(uavsar_data_list) if any(item is not None for item in uavsar_data_list) else None

    # Return 11-element tuple (edge_index removed for global-only attention)
    return (
        dep_points_batch,
        fuel_metrics_batch,
        dep_points_attr_batch,
        naip_data_batch,
        uavsar_data_batch,
        centers,
        scales,
        bboxes,
        tile_ids,
        norm_params_list,
        batch_indices
    )
