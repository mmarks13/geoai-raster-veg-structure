"""
GPU-native training augmentations as nn.Module layers.

Applied during training only (disabled when model.eval()).
Uses Kornia for image augmentations, custom ops for point clouds.

See docs/training_augmentation.md for full documentation.
"""

import json
import random
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import kornia.augmentation as K


class PointCloudAugmentation(nn.Module):
    """GPU-native point cloud augmentation for coordinates and attributes.

    Augmentations:
    - Coordinate jitter: Gaussian noise on x,y,z coords
    - Intensity noise: Gaussian noise on intensity values
    - Intensity outliers: Random extreme intensity values
    - Bird simulation: Extreme z-offset on 1 random point (simulates bird returns)
    """

    def __init__(
        self,
        coord_jitter_sigma_xy: float = 0.03,  # Separate sigma for x,y
        coord_jitter_sigma_z: float = 0.01,   # Separate sigma for z
        coord_jitter_prob: float = 0.5,
        intensity_noise_sigma: float = 0.05,
        intensity_noise_prob: float = 0.3,
        intensity_outlier_prob: float = 0.01,  # Per-point outlier probability
        intensity_outlier_range: Tuple[float, float] = (-2.0, 2.0),  # Z-score range
        # Bird simulation: random extreme z-offset on 1 point
        bird_outlier_prob: float = 0.05,  # Per-tile probability of adding a bird
        bird_z_offset_range: Tuple[float, float] = (5.0, 15.0),  # Z-score offset (5-15σ ≈ 25-75m physical)
        # Point duplication
        aug_point_dup_tile_prob: float = 0.3,
        aug_point_dup_min_point_prob: float = 0.05,
        aug_point_dup_max_point_prob: float = 0.20,
        aug_point_dup_min_offset: float = 0.001,
        aug_point_dup_max_offset: float = 0.2,
        # Omnidirectional outliers
        aug_omni_outlier_tile_prob: float = 0.2,
        aug_omni_outlier_point_prob: float = 0.01,
        aug_omni_outlier_min_magnitude: float = 2.0,
        aug_omni_outlier_max_magnitude: float = 20.0,
    ):
        super().__init__()
        self.coord_jitter_sigma_xy = coord_jitter_sigma_xy
        self.coord_jitter_sigma_z = coord_jitter_sigma_z
        self.coord_jitter_prob = coord_jitter_prob
        self.intensity_noise_sigma = intensity_noise_sigma
        self.intensity_noise_prob = intensity_noise_prob
        self.intensity_outlier_prob = intensity_outlier_prob
        self.intensity_outlier_range = intensity_outlier_range
        self.bird_outlier_prob = bird_outlier_prob
        self.bird_z_offset_range = bird_z_offset_range

        self.aug_point_dup_tile_prob = aug_point_dup_tile_prob
        self.aug_point_dup_min_point_prob = aug_point_dup_min_point_prob
        self.aug_point_dup_max_point_prob = aug_point_dup_max_point_prob
        self.aug_point_dup_min_offset = aug_point_dup_min_offset
        self.aug_point_dup_max_offset = aug_point_dup_max_offset

        self.aug_omni_outlier_tile_prob = aug_omni_outlier_tile_prob
        self.aug_omni_outlier_point_prob = aug_omni_outlier_point_prob
        self.aug_omni_outlier_min_magnitude = aug_omni_outlier_min_magnitude
        self.aug_omni_outlier_max_magnitude = aug_omni_outlier_max_magnitude

    def forward(
        self,
        coords: torch.Tensor,  # [N, 3]
        attrs: torch.Tensor    # [N, 3] - intensity, return_num, n_returns
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return coords, attrs

        # Coordinate jitter (separate xy and z sigmas) - zero GPU→CPU syncs
        do_jitter = (torch.rand(1, device=coords.device) < self.coord_jitter_prob).float()
        noise = torch.randn_like(coords) * do_jitter
        noise[:, :2] *= self.coord_jitter_sigma_xy
        noise[:, 2] *= self.coord_jitter_sigma_z
        coords = coords + noise

        # Intensity noise (first column of attrs only) - zero GPU→CPU syncs
        do_intensity = (torch.rand(1, device=attrs.device) < self.intensity_noise_prob).float()
        intensity_noise = torch.randn(attrs.shape[0], 1, device=attrs.device, dtype=attrs.dtype) * self.intensity_noise_sigma * do_intensity
        attrs = attrs.clone()
        attrs[:, 0] = attrs[:, 0] + intensity_noise.squeeze()

        # Intensity outliers (random extreme values)
        if self.intensity_outlier_prob > 0:
            outlier_mask = torch.rand(attrs.shape[0], device=attrs.device) < self.intensity_outlier_prob
            if outlier_mask.any():
                attrs = attrs.clone() if not attrs.requires_grad else attrs
                outlier_values = torch.empty(outlier_mask.sum(), device=attrs.device, dtype=attrs.dtype).uniform_(
                    self.intensity_outlier_range[0], self.intensity_outlier_range[1]
                )
                attrs[outlier_mask, 0] = outlier_values

        # Bird simulation: add extreme z-offset to 1 random point (simulates bird return)
        # Zero GPU→CPU syncs - always compute, multiply by mask
        if coords.shape[0] > 0:
            do_bird = (torch.rand(1, device=coords.device) < self.bird_outlier_prob).float()
            coords = coords.clone() if not coords.requires_grad else coords
            # Select 1 random point
            bird_idx = torch.randint(0, coords.shape[0], (1,), device=coords.device)
            # Add large positive z-offset (birds are always above the canopy)
            z_offset = torch.empty(1, device=coords.device, dtype=coords.dtype).uniform_(
                self.bird_z_offset_range[0], self.bird_z_offset_range[1]
            ) * do_bird
            coords[bird_idx, 2] = coords[bird_idx, 2] + z_offset

        # Omnidirectional outliers (any 3D direction)
        if self.aug_omni_outlier_tile_prob > 0 and coords.shape[0] > 0:
            coords = self.augment_omnidirectional_outliers(coords, coords.device)

        # Note: Point duplication is handled separately via augment_batch_point_duplication()
        # because it needs to update batch_indices

        return coords, attrs

    def augment_batch_point_duplication(
        self,
        coords: torch.Tensor,        # [N_total, 3]
        attrs: torch.Tensor,         # [N_total, 3]
        batch_indices: torch.Tensor  # [N_total]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Batch-aware point duplication that properly updates batch_indices.
        GPU-native: all tensor operations on GPU, minimal CPU syncs.

        Args:
            coords: [N_total, 3] batched point coordinates
            attrs: [N_total, 3] batched point attributes
            batch_indices: [N_total] tile assignment for each point

        Returns:
            Tuple of (coords, attrs, batch_indices) with duplicates added.
        """
        if not self.training:
            return coords, attrs, batch_indices

        if self.aug_point_dup_tile_prob == 0:
            return coords, attrs, batch_indices

        device = coords.device
        batch_size = batch_indices.max().item() + 1  # Single GPU->CPU sync

        # Pre-compute random decisions for all tiles (single GPU->CPU transfer)
        do_dup_tiles = (torch.rand(batch_size, device=device) < self.aug_point_dup_tile_prob).cpu()
        point_probs = torch.empty(batch_size, device=device).uniform_(
            self.aug_point_dup_min_point_prob, self.aug_point_dup_max_point_prob
        ).cpu()

        # Collect augmented tensors per tile
        new_coords_list = []
        new_attrs_list = []
        new_batch_list = []

        for b in range(batch_size):
            mask_b = (batch_indices == b)
            coords_b = coords[mask_b]
            attrs_b = attrs[mask_b]

            if do_dup_tiles[b] and coords_b.shape[0] > 0:
                # Per-point duplication (GPU operations)
                dup_mask = torch.rand(coords_b.shape[0], device=device) < point_probs[b]
                if dup_mask.any():
                    points_to_dup = coords_b[dup_mask]
                    attrs_to_dup = attrs_b[dup_mask]

                    # Random offsets (GPU)
                    offset_mags = torch.empty(points_to_dup.shape[0], device=device).uniform_(
                        self.aug_point_dup_min_offset, self.aug_point_dup_max_offset
                    )
                    offset_dirs = torch.randn(points_to_dup.shape[0], 3, device=device)
                    offset_dirs = offset_dirs / (offset_dirs.norm(dim=1, keepdim=True) + 1e-8)
                    duplicated_points = points_to_dup + offset_mags.unsqueeze(1) * offset_dirs

                    # Concatenate original + duplicated
                    coords_b = torch.cat([coords_b, duplicated_points], dim=0)
                    attrs_b = torch.cat([attrs_b, attrs_to_dup], dim=0)

            new_coords_list.append(coords_b)
            new_attrs_list.append(attrs_b)
            new_batch_list.append(torch.full((coords_b.shape[0],), b, dtype=batch_indices.dtype, device=device))

        return (
            torch.cat(new_coords_list, dim=0),
            torch.cat(new_attrs_list, dim=0),
            torch.cat(new_batch_list, dim=0)
        )

    def augment_omnidirectional_outliers(
        self,
        coords: torch.Tensor,
        device: torch.device
    ) -> torch.Tensor:
        """
        Add omnidirectional outliers (any 3D direction).

        Two-tier probability:
        1. Tile-level: whether to apply outliers to this tile
        2. Point-level: per-point probability

        Args:
            coords: [N, 3] point coordinates (z-score normalized)
            device: torch device

        Returns:
            augmented_coords: [N, 3] with outliers applied
        """
        # Tile-level probability
        tile_enabled = (torch.rand(1, device=device) < self.aug_omni_outlier_tile_prob).float()

        # Point-level outlier mask
        outlier_mask = (torch.rand(coords.shape[0], device=device)
                       < self.aug_omni_outlier_point_prob).float()

        # Combined mask
        combined_mask = tile_enabled * outlier_mask  # [N]

        # Sample magnitudes from range
        magnitudes = torch.empty(coords.shape[0], device=device).uniform_(
            self.aug_omni_outlier_min_magnitude,
            self.aug_omni_outlier_max_magnitude
        )

        # Sample random 3D directions (uniform on sphere)
        offset_dirs = torch.randn(coords.shape[0], 3, device=device)
        offset_dirs = offset_dirs / (offset_dirs.norm(dim=1, keepdim=True) + 1e-8)

        # Compute offsets
        offsets = magnitudes.unsqueeze(1) * offset_dirs

        # Apply with zero-sync pattern
        return coords + combined_mask.unsqueeze(1) * offsets


class ReturnAttributeAugmentation(nn.Module):
    """GPU-native augmentation for return_num and n_returns attributes.

    These are ordinal integers (1-8) that have been z-score normalized.
    Augmentations help the model not over-rely on these values.

    Augmentations (applied in order, each with independent probability):
    1. Distribution scaling: Stretch/shrink to simulate different return ranges
    2. Gaussian noise: Add uncertainty to exact values
    3. Zeroing: Force model to work without these attributes
    4. Shuffle: Decorrelate return info from point position

    Note: Normalization stats (mean/std) are passed from TrainingAugmentation,
    which loads them from the coordinate_normalization_stats.json file at init time.
    """

    def __init__(
        self,
        # Normalization stats (passed from TrainingAugmentation, loaded from JSON file)
        return_num_mean: float,
        return_num_std: float,
        n_returns_mean: float,
        n_returns_std: float,
        # Scaling params
        scale_prob: float = 0.5,
        scale_range: Tuple[float, float] = (0.5, 1.5),  # Multiplier for raw values
        # Noise params
        noise_prob: float = 0.3,
        noise_sigma: float = 0.1,  # In z-score units
        # Zeroing params
        zero_prob: float = 0.15,
        # Shuffle params
        shuffle_prob: float = 0.1,
    ):
        super().__init__()

        # Register normalization stats as buffers (automatically move to GPU)
        self.register_buffer('return_num_mean', torch.tensor(return_num_mean))
        self.register_buffer('return_num_std', torch.tensor(return_num_std))
        self.register_buffer('n_returns_mean', torch.tensor(n_returns_mean))
        self.register_buffer('n_returns_std', torch.tensor(n_returns_std))

        # Store augmentation parameters
        self.scale_prob = scale_prob
        self.scale_range = scale_range
        self.noise_prob = noise_prob
        self.noise_sigma = noise_sigma
        self.zero_prob = zero_prob
        self.shuffle_prob = shuffle_prob

    def forward(self, attrs: torch.Tensor) -> torch.Tensor:
        """Augment return_num and n_returns attributes.

        Args:
            attrs: [N, 3] point attributes (intensity, return_num, n_returns)
                   All values are z-score normalized.

        Returns:
            Augmented attrs [N, 3] with same shape.
        """
        if not self.training:
            return attrs

        device = attrs.device
        N = attrs.shape[0]

        # Clone to avoid modifying input
        attrs = attrs.clone()

        # Extract return attributes (columns 1 and 2)
        return_num_zscore = attrs[:, 1]  # [N]
        n_returns_zscore = attrs[:, 2]   # [N]

        # === 1. Distribution Scaling (stretch/shrink with integer rounding) ===
        do_scale = (torch.rand(1, device=device) < self.scale_prob).float()

        # Generate scale factor
        scale = self.scale_range[0] + torch.rand(1, device=device) * (self.scale_range[1] - self.scale_range[0])

        # Denormalize to raw integer space
        return_num_raw = return_num_zscore * self.return_num_std + self.return_num_mean
        n_returns_raw = n_returns_zscore * self.n_returns_std + self.n_returns_mean

        # Scale raw values
        return_num_scaled = return_num_raw * scale
        n_returns_scaled = n_returns_raw * scale

        # Round to integers and clamp to valid range [1, 12]
        return_num_int = torch.clamp(torch.round(return_num_scaled), 1, 12)
        n_returns_int = torch.clamp(torch.round(n_returns_scaled), 1, 12)

        # Ensure constraint: return_num <= n_returns
        return_num_int = torch.minimum(return_num_int, n_returns_int)

        # Re-normalize to z-score space
        return_num_new = (return_num_int - self.return_num_mean) / self.return_num_std
        n_returns_new = (n_returns_int - self.n_returns_mean) / self.n_returns_std

        # Apply scaling conditionally (blend with original if not scaling)
        return_num_zscore = return_num_zscore * (1.0 - do_scale) + return_num_new * do_scale
        n_returns_zscore = n_returns_zscore * (1.0 - do_scale) + n_returns_new * do_scale

        # === 2. Gaussian Noise ===
        do_noise = (torch.rand(1, device=device) < self.noise_prob).float()
        noise_rn = torch.randn(N, device=device, dtype=attrs.dtype) * self.noise_sigma * do_noise
        noise_nr = torch.randn(N, device=device, dtype=attrs.dtype) * self.noise_sigma * do_noise
        return_num_zscore = return_num_zscore + noise_rn
        n_returns_zscore = n_returns_zscore + noise_nr

        # === 3. Zeroing (set to mean in raw space = 0 in z-score space) ===
        do_zero = (torch.rand(1, device=device) < self.zero_prob).float()
        return_num_zscore = return_num_zscore * (1.0 - do_zero)
        n_returns_zscore = n_returns_zscore * (1.0 - do_zero)

        # === 4. Shuffle (decorrelate from position) ===
        perm = torch.randperm(N, device=device)
        return_num_shuffled = return_num_zscore[perm]
        n_returns_shuffled = n_returns_zscore[perm]
        do_shuffle = (torch.rand(1, device=device) < self.shuffle_prob).float()
        return_num_zscore = return_num_zscore * (1.0 - do_shuffle) + return_num_shuffled * do_shuffle
        n_returns_zscore = n_returns_zscore * (1.0 - do_shuffle) + n_returns_shuffled * do_shuffle

        # Update attrs tensor
        attrs[:, 1] = return_num_zscore
        attrs[:, 2] = n_returns_zscore

        return attrs


class NAIPAugmentation(nn.Module):
    """GPU-native NAIP imagery augmentation using Kornia.

    Physically-motivated transforms for optical imagery:
    - GaussianNoise: Sensor noise simulation
    - GaussianBlur: Atmospheric haze, focus issues
    - RandomMotionBlur: Aircraft motion, wind effects
    - RandomErasing: Cloud shadows, occlusions
    - RandomSharpness: Focus quality variation
    - RandomEqualize: Exposure, contrast variation
    """

    def __init__(
        self,
        noise_sigma: float = 0.03,
        noise_prob: float = 0.3,
        blur_kernel_size: int = 3,
        blur_sigma: Tuple[float, float] = (0.1, 2.0),
        blur_prob: float = 0.2,
        motion_blur_kernel_size: int = 5,
        motion_blur_angle: Tuple[float, float] = (-45.0, 45.0),
        motion_blur_prob: float = 0.1,
        erasing_scale: Tuple[float, float] = (0.02, 0.15),
        erasing_ratio: Tuple[float, float] = (0.3, 3.0),
        erasing_prob: float = 0.1,
        sharpness_range: Tuple[float, float] = (0.5, 1.5),
        sharpness_prob: float = 0.2,
        equalize_prob: float = 0.1,
    ):
        super().__init__()

        # Build augmentation pipeline
        # Note: RandomEqualize removed - requires [0,1] range but our images are z-score normalized
        self.augmentations = nn.ModuleList([
            K.RandomGaussianNoise(mean=0., std=noise_sigma, p=noise_prob),
            K.RandomGaussianBlur(
                kernel_size=(blur_kernel_size, blur_kernel_size),
                sigma=blur_sigma,
                p=blur_prob
            ),
            K.RandomMotionBlur(
                kernel_size=motion_blur_kernel_size,
                angle=motion_blur_angle,
                direction=(-1.0, 1.0),
                p=motion_blur_prob
            ),
            K.RandomErasing(
                scale=erasing_scale,
                ratio=erasing_ratio,
                value=0.0,  # In z-score space, 0 = mean value (reasonable for cloud shadow simulation)
                p=erasing_prob
            ),
            K.RandomSharpness(sharpness=sharpness_range, p=sharpness_prob),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [..., C, H, W] NAIP imagery (RGBN, C=4)
                    Handles arbitrary leading dimensions (e.g., [n_images, 4, H, W] or [B, n_images, 4, H, W])
        Returns:
            Augmented images same shape
        """
        if not self.training:
            return images

        # Kornia expects [B, C, H, W], so flatten leading dims if needed
        orig_shape = images.shape
        if images.ndim > 4:
            # Flatten all leading dimensions into batch: [..., C, H, W] -> [B, C, H, W]
            C, H, W = orig_shape[-3], orig_shape[-2], orig_shape[-1]
            images = images.reshape(-1, C, H, W)

        for aug in self.augmentations:
            images = aug(images)

        # Restore original shape
        if len(orig_shape) > 4:
            images = images.reshape(orig_shape)

        return images


class UAVSARAugmentation(nn.Module):
    """GPU-native UAVSAR imagery augmentation using Kornia.

    Physically-motivated transforms for SAR data:
    - GaussianNoise: Thermal/system noise (valid in dB domain)
    - GaussianBlur: Multi-looking simulation (speckle filtering)
    - RandomMotionBlur: Platform motion effects
    - RandomErasing: RFI/dropout simulation
    """

    def __init__(
        self,
        noise_sigma: float = 0.05,
        noise_prob: float = 0.3,
        blur_kernel_size: int = 3,
        blur_sigma: Tuple[float, float] = (0.1, 1.5),
        blur_prob: float = 0.2,
        motion_blur_kernel_size: int = 3,
        motion_blur_angle: Tuple[float, float] = (-30.0, 30.0),
        motion_blur_prob: float = 0.1,
        erasing_scale: Tuple[float, float] = (0.02, 0.10),
        erasing_ratio: Tuple[float, float] = (0.5, 2.0),
        erasing_prob: float = 0.1,
    ):
        super().__init__()

        self.augmentations = nn.ModuleList([
            K.RandomGaussianNoise(mean=0., std=noise_sigma, p=noise_prob),
            K.RandomGaussianBlur(
                kernel_size=(blur_kernel_size, blur_kernel_size),
                sigma=blur_sigma,
                p=blur_prob
            ),
            K.RandomMotionBlur(
                kernel_size=motion_blur_kernel_size,
                angle=motion_blur_angle,
                direction=(-1.0, 1.0),
                p=motion_blur_prob
            ),
            K.RandomErasing(
                scale=erasing_scale,
                ratio=erasing_ratio,
                value=0.0,
                p=erasing_prob
            ),
        ])

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [..., C, H, W] UAVSAR imagery (6 polarization bands, C=6)
                    Handles arbitrary leading dimensions (e.g., [n_images, 6, H, W] or [B, n_images, 6, H, W])
        Returns:
            Augmented images same shape
        """
        if not self.training:
            return images

        # Kornia expects [B, C, H, W], so flatten leading dims if needed
        orig_shape = images.shape
        if images.ndim > 4:
            # Flatten all leading dimensions into batch: [..., C, H, W] -> [B, C, H, W]
            C, H, W = orig_shape[-3], orig_shape[-2], orig_shape[-1]
            images = images.reshape(-1, C, H, W)

        for aug in self.augmentations:
            images = aug(images)

        # Restore original shape
        if len(orig_shape) > 4:
            images = images.reshape(orig_shape)

        return images


class SynchronizedGeometricAugmentation(nn.Module):
    """GPU-native synchronized geometric augmentation for multi-modal data.

    Applies IDENTICAL geometric transforms (rotation, reflection) to:
    - Point cloud coordinates [N, 3]
    - NAIP images [..., C, H, W]
    - UAVSAR images [..., C, H, W]
    - Target fuel_metrics [n_bands, H, W] (when training)

    This is critical for maintaining spatial consistency across modalities.
    All transforms use pre-computed rotation matrices as registered buffers.

    Note: Branch decisions require GPU→CPU sync (unavoidable without computing
    all possible transforms). The actual transform computation remains on GPU.
    """

    def __init__(
        self,
        rotation_prob: float = 0.5,
        reflection_prob: float = 0.3,
    ):
        super().__init__()
        self.rotation_prob = rotation_prob
        self.reflection_prob = reflection_prob

        # Pre-compute rotation matrices (90, 180, 270 degrees around Z-axis)
        # Stored as registered buffers for efficient GPU transfer
        angles = torch.tensor([90.0, 180.0, 270.0]) * (torch.pi / 180.0)
        rotation_matrices = []
        for angle in angles:
            c, s = torch.cos(angle), torch.sin(angle)
            # 3x3 rotation matrix around Z-axis
            rot = torch.tensor([
                [c, -s, 0.0],
                [s, c, 0.0],
                [0.0, 0.0, 1.0]
            ])
            rotation_matrices.append(rot)

        # Register as buffers: [3, 3, 3] - 3 rotation matrices, each 3x3
        self.register_buffer('rotation_matrices', torch.stack(rotation_matrices))

    def _rotate_points(self, coords: torch.Tensor, rotation_idx: int) -> torch.Tensor:
        """Apply rotation to point coordinates [N, 3]."""
        rot_matrix = self.rotation_matrices[rotation_idx]  # [3, 3]
        return torch.matmul(coords, rot_matrix.T)

    def _rotate_image(self, image: torch.Tensor, rotation_idx: int) -> torch.Tensor:
        """Apply rotation to image [..., C, H, W].

        rotation_idx: 0=90°, 1=180°, 2=270° clockwise
        """
        # Rotation operations:
        # 90° CW: transpose then flip horizontal
        # 180°: flip both dims
        # 270° CW (= 90° CCW): transpose then flip vertical
        if rotation_idx == 0:  # 90° CW
            return image.transpose(-2, -1).flip(-1)
        elif rotation_idx == 1:  # 180°
            return image.flip(-2).flip(-1)
        else:  # 270° CW
            return image.transpose(-2, -1).flip(-2)

    def _reflect_points(self, coords: torch.Tensor, axis: int) -> torch.Tensor:
        """Reflect points across axis (0=X, 1=Y)."""
        coords = coords.clone()
        coords[:, axis] = -coords[:, axis]
        return coords

    def _reflect_image(self, image: torch.Tensor, axis: int) -> torch.Tensor:
        """Reflect image across axis (0=X/horizontal, 1=Y/vertical)."""
        if axis == 0:  # Reflect across X (flip horizontal)
            return image.flip(-1)
        else:  # Reflect across Y (flip vertical)
            return image.flip(-2)

    def forward(
        self,
        coords: torch.Tensor,  # [N, 3]
        naip: Optional[torch.Tensor] = None,  # [..., C, H, W]
        uavsar: Optional[torch.Tensor] = None,  # [..., C, H, W]
        fuel_metrics: Optional[torch.Tensor] = None,  # [n_bands, H, W]
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply synchronized geometric transforms to all modalities.

        Returns:
            Tuple of (coords, naip, uavsar, fuel_metrics) with identical transforms applied.
        """
        if not self.training:
            return coords, naip, uavsar, fuel_metrics

        device = coords.device

        # Rotation (CPU random avoids GPU→CPU sync, transforms still on GPU)
        if random.random() < self.rotation_prob:
            rotation_idx = random.randint(0, 2)  # 0=90°, 1=180°, 2=270°

            coords = self._rotate_points(coords, rotation_idx)
            if naip is not None:
                naip = self._rotate_image(naip, rotation_idx)
            if uavsar is not None:
                uavsar = self._rotate_image(uavsar, rotation_idx)
            if fuel_metrics is not None:
                fuel_metrics = self._rotate_image(fuel_metrics, rotation_idx)

        # Reflection (CPU random avoids GPU→CPU sync, transforms still on GPU)
        if random.random() < self.reflection_prob:
            axis = random.randint(0, 1)  # 0=X, 1=Y

            coords = self._reflect_points(coords, axis)
            if naip is not None:
                naip = self._reflect_image(naip, axis)
            if uavsar is not None:
                uavsar = self._reflect_image(uavsar, axis)
            if fuel_metrics is not None:
                fuel_metrics = self._reflect_image(fuel_metrics, axis)

        return coords, naip, uavsar, fuel_metrics


class TemporalSubsamplingAugmentation(nn.Module):
    """GPU-native temporal subsampling for multi-temporal imagery.

    Randomly subsamples temporal dimensions to improve generalization.
    All probabilities and limits are configurable.

    Augmentations:
    - NAIP T-dim: Randomly subsample frames (keep at least min_frames)
    - UAVSAR T-dim: Randomly subsample temporal groups (keep at least min_frames)
    - UAVSAR G-dim: Randomly mask images within groups via attention_mask (keep at least min_images)

    Works with dict structures from preprocessing:
    - NAIP: {'images': [n_images, C, H, W], 'relative_dates': [n_images, 1], ...}
    - UAVSAR: {'images': [T, G_max, C, H, W], 'attention_mask': [T, G_max], 'relative_dates': [T, 1], ...}
    """

    def __init__(
        self,
        naip_subsample_prob: float = 0.5,
        naip_min_frames: int = 1,
        uavsar_t_subsample_prob: float = 0.5,
        uavsar_t_min_frames: int = 1,
        uavsar_g_mask_prob: float = 0.3,
        uavsar_g_min_images: int = 1,  # Minimum images to keep per group
    ):
        super().__init__()
        self.naip_subsample_prob = naip_subsample_prob
        self.naip_min_frames = naip_min_frames
        self.uavsar_t_subsample_prob = uavsar_t_subsample_prob
        self.uavsar_t_min_frames = uavsar_t_min_frames
        self.uavsar_g_mask_prob = uavsar_g_mask_prob
        self.uavsar_g_min_images = uavsar_g_min_images

    def forward(
        self,
        naip: Optional[dict] = None,
        uavsar: Optional[dict] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Apply temporal subsampling to imagery dicts.

        Returns:
            Tuple of (naip_dict, uavsar_dict) with potentially reduced temporal dimension.
        """
        if not self.training:
            return naip, uavsar

        # NAIP temporal subsampling
        if naip is not None and naip.get('images') is not None:
            images = naip['images']
            n_frames = images.shape[0]
            if n_frames > self.naip_min_frames:
                if torch.rand(1, device=images.device) < self.naip_subsample_prob:
                    # Keep between min_frames and current count
                    n_keep = torch.randint(
                        self.naip_min_frames, n_frames + 1, (1,), device=images.device
                    ).item()
                    # Random selection of frames to keep
                    perm = torch.randperm(n_frames, device=images.device)[:n_keep]
                    perm = perm.sort().values  # Maintain temporal order

                    # Update dict with subsampled data
                    naip = naip.copy()  # Don't modify original
                    naip['images'] = images[perm]
                    if 'relative_dates' in naip and naip['relative_dates'] is not None:
                        naip['relative_dates'] = naip['relative_dates'][perm]

        # UAVSAR temporal subsampling (T-dim)
        if uavsar is not None and uavsar.get('images') is not None:
            images = uavsar['images']
            T = images.shape[0]  # Number of temporal groups
            if T > self.uavsar_t_min_frames:
                if torch.rand(1, device=images.device) < self.uavsar_t_subsample_prob:
                    n_keep = torch.randint(
                        self.uavsar_t_min_frames, T + 1, (1,), device=images.device
                    ).item()
                    perm = torch.randperm(T, device=images.device)[:n_keep]
                    perm = perm.sort().values

                    # Update dict with subsampled data
                    uavsar = uavsar.copy()
                    uavsar['images'] = images[perm]
                    if 'attention_mask' in uavsar and uavsar['attention_mask'] is not None:
                        uavsar['attention_mask'] = uavsar['attention_mask'][perm]
                    if 'relative_dates' in uavsar and uavsar['relative_dates'] is not None:
                        uavsar['relative_dates'] = uavsar['relative_dates'][perm]

        # UAVSAR G-dim masking (within-group image dropout via attention_mask)
        if uavsar is not None and uavsar.get('attention_mask') is not None:
            if torch.rand(1, device=uavsar['attention_mask'].device) < self.uavsar_g_mask_prob:
                mask = uavsar['attention_mask']  # [T, G_max]
                uavsar = uavsar.copy()

                # For each temporal group, randomly mask some valid images
                # Keep between min_images and current valid count
                new_mask = mask.clone()
                T, G_max = mask.shape

                for t in range(T):
                    valid_indices = torch.where(mask[t])[0]
                    n_valid = len(valid_indices)
                    if n_valid > self.uavsar_g_min_images:
                        # Uniformly random number to keep: [min_images, n_valid]
                        n_keep = torch.randint(
                            self.uavsar_g_min_images, n_valid + 1, (1,), device=mask.device
                        ).item()
                        if n_keep < n_valid:
                            # Randomly select indices to keep
                            keep_perm = torch.randperm(n_valid, device=mask.device)[:n_keep]
                            indices_to_keep = valid_indices[keep_perm]
                            # Mask all valid, then unmask kept ones
                            new_mask[t, valid_indices] = False
                            new_mask[t, indices_to_keep] = True

                uavsar['attention_mask'] = new_mask

        return naip, uavsar


class ModalityDropoutAugmentation(nn.Module):
    """GPU-native modality dropout for robustness to missing data.

    Randomly drops entire modalities (NAIP or UAVSAR) to ensure model
    can handle missing data at inference time (e.g., Laguna site has no UAVSAR).

    This replaces the previous implementation in raster_training.py lines 326-342.
    """

    def __init__(
        self,
        naip_dropout_prob: float = 0.15,
        uavsar_dropout_prob: float = 0.15,
    ):
        super().__init__()
        self.naip_dropout_prob = naip_dropout_prob
        self.uavsar_dropout_prob = uavsar_dropout_prob

    def forward(
        self,
        naip: Optional[dict] = None,
        uavsar: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Apply modality dropout.

        Args:
            naip: NAIP imagery dict or None
            uavsar: UAVSAR imagery dict or None
            device: Device for random number generation (uses naip/uavsar device if not provided)

        Returns:
            Tuple of (naip, uavsar) with randomly dropped modalities set to None.
        """
        if not self.training:
            return naip, uavsar

        # Determine device for RNG
        if device is None:
            if naip is not None and naip.get('images') is not None:
                device = naip['images'].device
            elif uavsar is not None and uavsar.get('images') is not None:
                device = uavsar['images'].device
            else:
                device = torch.device('cpu')

        # NAIP dropout
        if naip is not None:
            if torch.rand(1, device=device) < self.naip_dropout_prob:
                naip = None

        # UAVSAR dropout
        if uavsar is not None:
            if torch.rand(1, device=device) < self.uavsar_dropout_prob:
                uavsar = None

        return naip, uavsar


class PointCloudSparseAugmentation(nn.Module):
    """GPU-native point cloud sparsification for robustness to point density variation.

    Randomly removes points from the point cloud to simulate sparse acquisitions.
    Only used when global-only attention mode is enabled (local attention requires KNN).

    This augmentation trains the model to be robust to:
    - Variable point density across tiles
    - Missing returns in canopy
    - Sparse acquisitions
    """

    def __init__(
        self,
        removal_prob: float = 0.3,  # Probability of applying point removal
        min_removal_ratio: float = 0.05,  # Minimum fraction of points to remove
        max_removal_ratio: float = 0.7,  # Maximum fraction of points to remove
        min_points: int = 20,  # Minimum number of points to keep
    ):
        super().__init__()
        self.removal_prob = removal_prob
        self.min_removal_ratio = min_removal_ratio
        self.max_removal_ratio = max_removal_ratio
        self.min_points = min_points

    def forward(
        self,
        coords: torch.Tensor,  # [N, 3]
        attrs: torch.Tensor,  # [N, 3]
        batch_indices: torch.Tensor,  # [N]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply point removal augmentation.

        Args:
            coords: [N, 3] point coordinates
            attrs: [N, 3] point attributes
            batch_indices: [N] batch index for each point

        Returns:
            Tuple of (coords, attrs, batch_indices) with some points removed.
        """
        if not self.training:
            return coords, attrs, batch_indices

        device = coords.device

        # Check if we should apply removal
        if torch.rand(1, device=device) >= self.removal_prob:
            return coords, attrs, batch_indices

        # Random removal ratio
        removal_ratio = torch.rand(1, device=device) * (
            self.max_removal_ratio - self.min_removal_ratio
        ) + self.min_removal_ratio

        n_points = coords.shape[0]
        n_keep = max(
            self.min_points,
            int(n_points * (1.0 - removal_ratio.item()))
        )

        if n_keep >= n_points:
            return coords, attrs, batch_indices

        # Random selection of points to keep
        perm = torch.randperm(n_points, device=device)[:n_keep]
        perm = perm.sort().values  # Maintain relative order for batch consistency

        return coords[perm], attrs[perm], batch_indices[perm]


class TrainingAugmentation(nn.Module):
    """Combined training augmentation module for raster model.

    Wraps all augmentations into a single module:
    - Point cloud augmentation (jitter, intensity noise, bird simulation)
    - NAIP image augmentation (Kornia-based noise, blur, erasing)
    - UAVSAR image augmentation (Kornia-based noise, blur, erasing)
    - Synchronized geometric augmentation (rotation, reflection)
    - Temporal subsampling (NAIP T-dim, UAVSAR T-dim and G-dim)
    - Modality dropout (randomly drop NAIP or UAVSAR)

    Controlled by config.training_augmentation_enabled master switch.
    """

    def __init__(self, config):
        """
        Args:
            config: MultimodalRasterConfig with augmentation parameters
        """
        super().__init__()

        # Only create if augmentation is enabled
        self.enabled = getattr(config, 'training_augmentation_enabled', False)

        if self.enabled:
            # Point cloud augmentation
            self.point_aug = PointCloudAugmentation(
                coord_jitter_sigma_xy=getattr(config, 'aug_coord_jitter_sigma_xy', 0.03),
                coord_jitter_sigma_z=getattr(config, 'aug_coord_jitter_sigma_z', 0.01),
                coord_jitter_prob=getattr(config, 'aug_coord_jitter_prob', 0.5),
                intensity_noise_sigma=getattr(config, 'aug_intensity_noise_sigma', 0.05),
                intensity_noise_prob=getattr(config, 'aug_intensity_noise_prob', 0.3),
                intensity_outlier_prob=getattr(config, 'aug_intensity_outlier_prob', 0.01),
                bird_outlier_prob=getattr(config, 'aug_bird_outlier_prob', 0.05),
                bird_z_offset_range=getattr(config, 'aug_bird_z_offset_range', (5.0, 15.0)),
                # Point duplication
                aug_point_dup_tile_prob=getattr(config, 'aug_point_dup_tile_prob', 0.3),
                aug_point_dup_min_point_prob=getattr(config, 'aug_point_dup_min_point_prob', 0.05),
                aug_point_dup_max_point_prob=getattr(config, 'aug_point_dup_max_point_prob', 0.20),
                aug_point_dup_min_offset=getattr(config, 'aug_point_dup_min_offset', 0.001),
                aug_point_dup_max_offset=getattr(config, 'aug_point_dup_max_offset', 0.2),
                # Omnidirectional outliers
                aug_omni_outlier_tile_prob=getattr(config, 'aug_omni_outlier_tile_prob', 0.2),
                aug_omni_outlier_point_prob=getattr(config, 'aug_omni_outlier_point_prob', 0.01),
                aug_omni_outlier_min_magnitude=getattr(config, 'aug_omni_outlier_min_magnitude', 2.0),
                aug_omni_outlier_max_magnitude=getattr(config, 'aug_omni_outlier_max_magnitude', 20.0),
            )

            # Return attribute augmentation
            # Load normalization stats from file (one-time at init, not during training)
            coord_norm_stats_path = getattr(config, 'coordinate_normalization_stats_path', None)
            if coord_norm_stats_path is not None:
                with open(coord_norm_stats_path, 'r') as f:
                    norm_stats = json.load(f)
                # attr_mean/std are [intensity, return_num, n_returns]
                return_num_mean = norm_stats['attr_mean'][1]
                return_num_std = norm_stats['attr_std'][1]
                n_returns_mean = norm_stats['attr_mean'][2]
                n_returns_std = norm_stats['attr_std'][2]
            else:
                # Fallback defaults (from training data statistics)
                return_num_mean = 1.3028
                return_num_std = 0.5721
                n_returns_mean = 1.6067
                n_returns_std = 0.7638

            self.return_attr_aug = ReturnAttributeAugmentation(
                return_num_mean=return_num_mean,
                return_num_std=return_num_std,
                n_returns_mean=n_returns_mean,
                n_returns_std=n_returns_std,
                scale_prob=getattr(config, 'aug_return_scale_prob', 0.5),
                scale_range=getattr(config, 'aug_return_scale_range', (0.5, 1.5)),
                noise_prob=getattr(config, 'aug_return_noise_prob', 0.3),
                noise_sigma=getattr(config, 'aug_return_noise_sigma', 0.1),
                zero_prob=getattr(config, 'aug_return_zero_prob', 0.15),
                shuffle_prob=getattr(config, 'aug_return_shuffle_prob', 0.1),
            )

            # NAIP image augmentation (Kornia)
            self.naip_aug = NAIPAugmentation(
                noise_sigma=getattr(config, 'aug_naip_noise_sigma', 0.03),
                noise_prob=getattr(config, 'aug_naip_noise_prob', 0.3),
                blur_kernel_size=getattr(config, 'aug_naip_blur_kernel', 3),
                blur_sigma=getattr(config, 'aug_naip_blur_sigma', (0.1, 2.0)),
                blur_prob=getattr(config, 'aug_naip_blur_prob', 0.2),
                motion_blur_kernel_size=getattr(config, 'aug_naip_motion_blur_kernel', 5),
                motion_blur_angle=getattr(config, 'aug_naip_motion_blur_angle', (-45.0, 45.0)),
                motion_blur_prob=getattr(config, 'aug_naip_motion_blur_prob', 0.1),
                erasing_scale=getattr(config, 'aug_naip_erasing_scale', (0.02, 0.15)),
                erasing_prob=getattr(config, 'aug_naip_erasing_prob', 0.1),
                sharpness_range=getattr(config, 'aug_naip_sharpness_range', (0.5, 1.5)),
                sharpness_prob=getattr(config, 'aug_naip_sharpness_prob', 0.2),
                equalize_prob=getattr(config, 'aug_naip_equalize_prob', 0.1),
            )

            # UAVSAR image augmentation (Kornia)
            self.uavsar_aug = UAVSARAugmentation(
                noise_sigma=getattr(config, 'aug_uavsar_noise_sigma', 0.05),
                noise_prob=getattr(config, 'aug_uavsar_noise_prob', 0.3),
                blur_kernel_size=getattr(config, 'aug_uavsar_blur_kernel', 3),
                blur_sigma=getattr(config, 'aug_uavsar_blur_sigma', (0.1, 1.5)),
                blur_prob=getattr(config, 'aug_uavsar_blur_prob', 0.2),
                motion_blur_kernel_size=getattr(config, 'aug_uavsar_motion_blur_kernel', 3),
                motion_blur_angle=getattr(config, 'aug_uavsar_motion_blur_angle', (-30.0, 30.0)),
                motion_blur_prob=getattr(config, 'aug_uavsar_motion_blur_prob', 0.1),
                erasing_scale=getattr(config, 'aug_uavsar_erasing_scale', (0.02, 0.10)),
                erasing_prob=getattr(config, 'aug_uavsar_erasing_prob', 0.1),
            )

            # Synchronized geometric augmentation (rotation, reflection)
            self.geometric_enabled = getattr(config, 'aug_geometric_enabled', True)
            if self.geometric_enabled:
                self.geometric_aug = SynchronizedGeometricAugmentation(
                    rotation_prob=getattr(config, 'aug_rotation_prob', 0.5),
                    reflection_prob=getattr(config, 'aug_reflection_prob', 0.3),
                )

            # Temporal subsampling augmentation
            self.temporal_enabled = getattr(config, 'aug_temporal_enabled', True)
            if self.temporal_enabled:
                self.temporal_aug = TemporalSubsamplingAugmentation(
                    naip_subsample_prob=getattr(config, 'aug_naip_subsample_prob', 0.5),
                    naip_min_frames=getattr(config, 'aug_naip_min_frames', 1),
                    uavsar_t_subsample_prob=getattr(config, 'aug_uavsar_t_subsample_prob', 0.5),
                    uavsar_t_min_frames=getattr(config, 'aug_uavsar_t_min_frames', 1),
                    uavsar_g_mask_prob=getattr(config, 'aug_uavsar_g_mask_prob', 0.3),
                    uavsar_g_min_images=getattr(config, 'aug_uavsar_g_min_images', 1),
                )

            # Modality dropout augmentation
            self.modality_dropout_enabled = getattr(config, 'aug_modality_dropout_enabled', True)
            if self.modality_dropout_enabled:
                self.modality_dropout = ModalityDropoutAugmentation(
                    naip_dropout_prob=getattr(config, 'aug_naip_dropout_prob', 0.15),
                    uavsar_dropout_prob=getattr(config, 'aug_uavsar_dropout_prob', 0.15),
                )

            # Point cloud sparse augmentation (only with global-only mode)
            self.point_removal_enabled = getattr(config, 'aug_point_removal_enabled', False)
            if self.point_removal_enabled:
                self.point_removal = PointCloudSparseAugmentation(
                    removal_prob=getattr(config, 'aug_point_removal_prob', 0.3),
                    min_removal_ratio=getattr(config, 'aug_point_min_removal_ratio', 0.05),
                    max_removal_ratio=getattr(config, 'aug_point_max_removal_ratio', 0.7),
                    min_points=getattr(config, 'aug_point_min_points', 20),
                )

            # Temporal shift augmentation (shifts relative_dates)
            self.aug_temporal_shift_prob = getattr(config, 'aug_temporal_shift_prob', 0.5)
            self.aug_temporal_max_shift_days = getattr(config, 'aug_temporal_max_shift_days', 180.0)

    def augment_points(self, coords: torch.Tensor, attrs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Augment point cloud coordinates and attributes.

        Args:
            coords: [N, 3] point coordinates (z-score normalized)
            attrs: [N, 3] point attributes (intensity, return_num, n_returns)

        Returns:
            Tuple of (augmented_coords, augmented_attrs)
        """
        if not self.enabled or not self.training:
            return coords, attrs

        # Apply coordinate and intensity augmentation
        coords, attrs = self.point_aug(coords, attrs)

        # Apply return attribute augmentation (return_num, n_returns)
        attrs = self.return_attr_aug(attrs)

        return coords, attrs

    def augment_naip(self, images: torch.Tensor) -> torch.Tensor:
        """Augment NAIP imagery.

        Args:
            images: [n_images, 4, H, W] NAIP imagery (RGBN)

        Returns:
            Augmented images same shape
        """
        if not self.enabled or not self.training:
            return images
        return self.naip_aug(images)

    def augment_uavsar(self, images: torch.Tensor) -> torch.Tensor:
        """Augment UAVSAR imagery.

        Args:
            images: [n_images, 6, H, W] UAVSAR imagery (6 polarization bands)

        Returns:
            Augmented images same shape
        """
        if not self.enabled or not self.training:
            return images
        return self.uavsar_aug(images)

    def augment_temporal_shift(
        self,
        naip_dict: Optional[Dict],
        uavsar_dict: Optional[Dict]
    ) -> None:
        """
        Shift temporal sequences (relative_dates) by random offset in days.

        Models temporal misalignment between LiDAR acquisition and imagery dates.
        Shifts all images in a modality by the same random offset.

        Args:
            naip_dict: NAIP imagery dict (or None)
            uavsar_dict: UAVSAR imagery dict (or None)

        Modifies dicts in-place.
        """
        if self.aug_temporal_max_shift_days == 0:
            return

        # Determine device from available imagery
        device = None
        if naip_dict is not None and 'relative_dates' in naip_dict and naip_dict['relative_dates'] is not None:
            device = naip_dict['relative_dates'].device
        elif uavsar_dict is not None and 'relative_dates' in uavsar_dict and uavsar_dict['relative_dates'] is not None:
            device = uavsar_dict['relative_dates'].device

        if device is None:
            return  # No imagery to augment

        # Tile-level probability (zero-sync pattern)
        do_shift = (torch.rand(1, device=device) < self.aug_temporal_shift_prob).float()

        if do_shift.item() == 0:
            return

        # Shift NAIP relative dates
        if (naip_dict is not None
            and 'relative_dates' in naip_dict
            and naip_dict['relative_dates'] is not None):

            shift_days = torch.empty(1, device=device).uniform_(
                -self.aug_temporal_max_shift_days,
                self.aug_temporal_max_shift_days
            )
            naip_dict['relative_dates'] = naip_dict['relative_dates'] + shift_days

        # Shift UAVSAR relative dates
        if (uavsar_dict is not None
            and 'relative_dates' in uavsar_dict
            and uavsar_dict['relative_dates'] is not None):

            shift_days = torch.empty(1, device=device).uniform_(
                -self.aug_temporal_max_shift_days,
                self.aug_temporal_max_shift_days
            )
            uavsar_dict['relative_dates'] = uavsar_dict['relative_dates'] + shift_days

    def augment_temporal_shift_batch(
        self,
        naip_list: Optional[List[Optional[Dict]]],
        uavsar_list: Optional[List[Optional[Dict]]],
        device: torch.device
    ) -> None:
        """
        Vectorized temporal shift for batched data.

        Args:
            naip_list: List of NAIP dicts (or None)
            uavsar_list: List of UAVSAR dicts (or None)
            device: torch device for random number generation

        Modifies dicts in-place.
        """
        if self.aug_temporal_max_shift_days == 0:
            return

        # Determine batch size
        batch_size = len(naip_list) if naip_list is not None else len(uavsar_list) if uavsar_list is not None else 0
        if batch_size == 0:
            return

        # Generate per-tile shift decisions [batch_size]
        do_shift = torch.rand(batch_size, device=device) < self.aug_temporal_shift_prob

        # Apply shifts vectorized
        for b in range(batch_size):
            if not do_shift[b].item():
                continue

            naip_b = naip_list[b] if naip_list is not None and b < len(naip_list) else None
            uavsar_b = uavsar_list[b] if uavsar_list is not None and b < len(uavsar_list) else None

            # Shift NAIP relative dates
            if (naip_b is not None
                and 'relative_dates' in naip_b
                and naip_b['relative_dates'] is not None):

                shift_days = torch.empty(1, device=device).uniform_(
                    -self.aug_temporal_max_shift_days,
                    self.aug_temporal_max_shift_days
                )
                naip_b['relative_dates'] = naip_b['relative_dates'] + shift_days

            # Shift UAVSAR relative dates
            if (uavsar_b is not None
                and 'relative_dates' in uavsar_b
                and uavsar_b['relative_dates'] is not None):

                shift_days = torch.empty(1, device=device).uniform_(
                    -self.aug_temporal_max_shift_days,
                    self.aug_temporal_max_shift_days
                )
                uavsar_b['relative_dates'] = uavsar_b['relative_dates'] + shift_days

    def augment_geometric(
        self,
        coords: torch.Tensor,
        naip: Optional[torch.Tensor] = None,
        uavsar: Optional[torch.Tensor] = None,
        fuel_metrics: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Apply synchronized geometric transforms to all modalities.

        Args:
            coords: [N, 3] point coordinates
            naip: Optional [..., C, H, W] NAIP imagery
            uavsar: Optional [..., C, H, W] UAVSAR imagery
            fuel_metrics: Optional [n_bands, H, W] target fuel metrics

        Returns:
            Tuple of (coords, naip, uavsar, fuel_metrics) with identical transforms applied.
        """
        if not self.enabled or not self.training:
            return coords, naip, uavsar, fuel_metrics
        if not getattr(self, 'geometric_enabled', False):
            return coords, naip, uavsar, fuel_metrics
        return self.geometric_aug(coords, naip, uavsar, fuel_metrics)

    def augment_temporal(
        self,
        naip: Optional[dict] = None,
        uavsar: Optional[dict] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Apply temporal subsampling to imagery dicts.

        Args:
            naip: Optional NAIP dict with 'images', 'relative_dates', etc.
            uavsar: Optional UAVSAR dict with 'images', 'attention_mask', 'relative_dates', etc.

        Returns:
            Tuple of (naip_dict, uavsar_dict) with potentially reduced temporal dimension.
        """
        if not self.enabled or not self.training:
            return naip, uavsar
        if not getattr(self, 'temporal_enabled', False):
            return naip, uavsar
        return self.temporal_aug(naip, uavsar)

    def apply_modality_dropout(
        self,
        naip: Optional[dict] = None,
        uavsar: Optional[dict] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[dict], Optional[dict]]:
        """Apply modality dropout to randomly drop entire modalities.

        Args:
            naip: Optional NAIP dict or None
            uavsar: Optional UAVSAR dict or None
            device: Device for random number generation

        Returns:
            Tuple of (naip, uavsar) with randomly dropped modalities set to None.
        """
        if not self.enabled or not self.training:
            return naip, uavsar
        if not getattr(self, 'modality_dropout_enabled', False):
            return naip, uavsar
        return self.modality_dropout(naip, uavsar, device)

    def apply_modality_dropout_batch(
        self,
        naip: Optional[List[Optional[Dict]]] = None,
        uavsar: Optional[List[Optional[Dict]]] = None,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[List[Optional[Dict]]], Optional[List[Optional[Dict]]]]:
        """Apply modality dropout for batched data (list of dicts).

        Makes a single batch-wide decision per modality (not per-tile).
        When dropping, replaces entire list with [None] * batch_size.

        Args:
            naip: List of NAIP dicts (one per tile) or None
            uavsar: List of UAVSAR dicts (one per tile) or None
            device: Device for random number generation

        Returns:
            Tuple of (naip, uavsar) with randomly dropped modalities set to list of Nones.
        """
        if not self.enabled or not self.training:
            return naip, uavsar
        if not getattr(self, 'modality_dropout_enabled', False):
            return naip, uavsar

        # Determine device for RNG
        if device is None:
            device = torch.device('cpu')

        # NAIP dropout (batch-wide decision)
        if naip is not None:
            if torch.rand(1, device=device) < self.modality_dropout.naip_dropout_prob:
                naip = [None] * len(naip)

        # UAVSAR dropout (batch-wide decision)
        if uavsar is not None:
            if torch.rand(1, device=device) < self.modality_dropout.uavsar_dropout_prob:
                uavsar = [None] * len(uavsar)

        return naip, uavsar

    def apply_point_removal(
        self,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply point cloud sparse augmentation (only with global-only mode).

        Args:
            coords: [N, 3] point coordinates
            attrs: [N, 3] point attributes
            batch_indices: [N] batch index for each point

        Returns:
            Tuple of (coords, attrs, batch_indices) with some points removed.
        """
        if not self.enabled or not self.training:
            return coords, attrs, batch_indices
        if not getattr(self, 'point_removal_enabled', False):
            return coords, attrs, batch_indices
        return self.point_removal(coords, attrs, batch_indices)

    def apply_point_duplication(
        self,
        coords: torch.Tensor,
        attrs: torch.Tensor,
        batch_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply point duplication augmentation (only with global-only mode).

        Note: Only use when use_global_only=True, since KNN graphs would be invalidated.

        Args:
            coords: [N, 3] point coordinates
            attrs: [N, 3] point attributes
            batch_indices: [N] batch index for each point

        Returns:
            Tuple of (coords, attrs, batch_indices) with duplicated points added.
        """
        if not self.enabled or not self.training:
            return coords, attrs, batch_indices
        return self.point_aug.augment_batch_point_duplication(coords, attrs, batch_indices)

    def augment_batch_geometric(
        self,
        dep_points: torch.Tensor,
        batch_indices: torch.Tensor,
        naip_data: List[Optional[Dict]],
        uavsar_data: List[Optional[Dict]],
        fuel_metrics: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[Optional[Dict]], List[Optional[Dict]], torch.Tensor]:
        """Apply per-tile synchronized geometric transforms with batched random decisions.

        Uses Option A-prime: batch random decisions upfront (4-6 GPU→CPU syncs total),
        then apply per-tile without additional syncs in the loop.

        Args:
            dep_points: [N_total, 3] point coordinates (batched)
            batch_indices: [N_total] batch index for each point
            naip_data: List of dicts with NAIP imagery per tile
            uavsar_data: List of dicts with UAVSAR imagery per tile
            fuel_metrics: [batch_size, n_bands, 5, 5] target rasters

        Returns:
            Tuple of (dep_points, naip_data, uavsar_data, fuel_metrics) with transforms applied.
        """
        if not self.enabled or not self.training:
            return dep_points, naip_data, uavsar_data, fuel_metrics
        if not getattr(self, 'geometric_enabled', False):
            return dep_points, naip_data, uavsar_data, fuel_metrics

        batch_size = fuel_metrics.shape[0]
        device = dep_points.device

        # Batch random decisions upfront (minimizes GPU→CPU syncs)
        do_rotate = (torch.rand(batch_size, device=device) < self.geometric_aug.rotation_prob).cpu()
        rotation_indices = torch.randint(0, 3, (batch_size,), device=device).cpu()
        do_reflect = (torch.rand(batch_size, device=device) < self.geometric_aug.reflection_prob).cpu()
        reflect_axes = torch.randint(0, 2, (batch_size,), device=device).cpu()

        # Apply per-tile (no syncs in loop - decisions already on CPU)
        for b in range(batch_size):
            mask_b = (batch_indices == b)

            if do_rotate[b]:
                rot_idx = rotation_indices[b].item()
                dep_points[mask_b] = self.geometric_aug._rotate_points(dep_points[mask_b], rot_idx)
                fuel_metrics[b] = self.geometric_aug._rotate_image(fuel_metrics[b], rot_idx)
                if naip_data and naip_data[b] and naip_data[b].get('images') is not None:
                    naip_data[b]['images'] = self.geometric_aug._rotate_image(naip_data[b]['images'], rot_idx)
                if uavsar_data and uavsar_data[b] and uavsar_data[b].get('images') is not None:
                    uavsar_data[b]['images'] = self.geometric_aug._rotate_image(uavsar_data[b]['images'], rot_idx)

            if do_reflect[b]:
                axis = reflect_axes[b].item()
                dep_points[mask_b] = self.geometric_aug._reflect_points(dep_points[mask_b], axis)
                fuel_metrics[b] = self.geometric_aug._reflect_image(fuel_metrics[b], axis)
                if naip_data and naip_data[b] and naip_data[b].get('images') is not None:
                    naip_data[b]['images'] = self.geometric_aug._reflect_image(naip_data[b]['images'], axis)
                if uavsar_data and uavsar_data[b] and uavsar_data[b].get('images') is not None:
                    uavsar_data[b]['images'] = self.geometric_aug._reflect_image(uavsar_data[b]['images'], axis)

        return dep_points, naip_data, uavsar_data, fuel_metrics

    def apply_embedding_dropout(
        self,
        naip_embeddings: List[Optional[torch.Tensor]],
        uavsar_embeddings: List[Optional[torch.Tensor]],
        device: torch.device,
    ) -> Tuple[List[Optional[torch.Tensor]], List[Optional[torch.Tensor]]]:
        """Apply per-sample modality dropout by zeroing embeddings (gradient-safe).

        Unlike batch-level dropout (apply_modality_dropout_batch), this:
        - Always runs encoders (gradients flow through all encoder params)
        - Drops per-sample (more augmentation diversity)
        - Doesn't require find_unused_parameters=True in DDP

        The key insight: `embedding * 0.0` still computes gradients (multiplying
        by zero doesn't break autograd), so encoder params always participate
        in the forward pass even when their output is zeroed.

        Args:
            naip_embeddings: List of NAIP embeddings [num_patches, embed_dim] per sample, or None
            uavsar_embeddings: List of UAVSAR embeddings [num_patches, embed_dim] per sample, or None
            device: Device for random number generation

        Returns:
            Tuple of (naip_embeddings, uavsar_embeddings) with randomly zeroed embeddings.
        """
        if not self.enabled or not self.training:
            return naip_embeddings, uavsar_embeddings
        if not getattr(self, 'modality_dropout_enabled', False):
            return naip_embeddings, uavsar_embeddings

        batch_size = len(naip_embeddings)

        # Generate per-sample dropout decisions (single GPU→CPU sync for batch)
        naip_keep = torch.rand(batch_size, device=device) >= self.modality_dropout.naip_dropout_prob
        uavsar_keep = torch.rand(batch_size, device=device) >= self.modality_dropout.uavsar_dropout_prob

        # Zero out dropped embeddings (preserves gradients via * 0.0)
        for b in range(batch_size):
            if naip_embeddings[b] is not None and not naip_keep[b]:
                naip_embeddings[b] = naip_embeddings[b] * 0.0
            if uavsar_embeddings[b] is not None and not uavsar_keep[b]:
                uavsar_embeddings[b] = uavsar_embeddings[b] * 0.0

        return naip_embeddings, uavsar_embeddings
