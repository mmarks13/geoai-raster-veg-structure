"""
GPU-native training augmentations as nn.Module layers.

Applied during training only (disabled when model.eval()).
Uses Kornia for image augmentations, custom ops for point clouds.

See docs/training_augmentation.md for full documentation.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
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
        coord_jitter_sigma: float = 0.02,  # In z-score units (~0.1m physical)
        coord_jitter_prob: float = 0.5,
        intensity_noise_sigma: float = 0.05,
        intensity_noise_prob: float = 0.3,
        intensity_outlier_prob: float = 0.01,  # Per-point outlier probability
        intensity_outlier_range: Tuple[float, float] = (-2.0, 2.0),  # Z-score range
        # Bird simulation: random extreme z-offset on 1 point
        bird_outlier_prob: float = 0.05,  # Per-tile probability of adding a bird
        bird_z_offset_range: Tuple[float, float] = (5.0, 15.0),  # Z-score offset (5-15σ ≈ 25-75m physical)
    ):
        super().__init__()
        self.coord_jitter_sigma = coord_jitter_sigma
        self.coord_jitter_prob = coord_jitter_prob
        self.intensity_noise_sigma = intensity_noise_sigma
        self.intensity_noise_prob = intensity_noise_prob
        self.intensity_outlier_prob = intensity_outlier_prob
        self.intensity_outlier_range = intensity_outlier_range
        self.bird_outlier_prob = bird_outlier_prob
        self.bird_z_offset_range = bird_z_offset_range

    def forward(
        self,
        coords: torch.Tensor,  # [N, 3]
        attrs: torch.Tensor    # [N, 3] - intensity, return_num, n_returns
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.training:
            return coords, attrs

        # Coordinate jitter (all 3 dims)
        if torch.rand(1).item() < self.coord_jitter_prob:
            noise = torch.randn_like(coords) * self.coord_jitter_sigma
            coords = coords + noise

        # Intensity noise (first column of attrs only)
        if torch.rand(1).item() < self.intensity_noise_prob:
            intensity_noise = torch.randn(attrs.shape[0], 1, device=attrs.device) * self.intensity_noise_sigma
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
        if torch.rand(1).item() < self.bird_outlier_prob and coords.shape[0] > 0:
            coords = coords.clone() if not coords.requires_grad else coords
            # Select 1 random point
            bird_idx = torch.randint(0, coords.shape[0], (1,), device=coords.device)
            # Add large positive z-offset (birds are always above the canopy)
            z_offset = torch.empty(1, device=coords.device).uniform_(
                self.bird_z_offset_range[0], self.bird_z_offset_range[1]
            )
            coords[bird_idx, 2] = coords[bird_idx, 2] + z_offset

        return coords, attrs


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


class TrainingAugmentation(nn.Module):
    """Combined training augmentation module for raster model.

    Wraps point cloud, NAIP, and UAVSAR augmentations into a single module.
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
            self.point_aug = PointCloudAugmentation(
                coord_jitter_sigma=getattr(config, 'aug_coord_jitter_sigma', 0.02),
                coord_jitter_prob=getattr(config, 'aug_coord_jitter_prob', 0.5),
                intensity_noise_sigma=getattr(config, 'aug_intensity_noise_sigma', 0.05),
                intensity_noise_prob=getattr(config, 'aug_intensity_noise_prob', 0.3),
                intensity_outlier_prob=getattr(config, 'aug_intensity_outlier_prob', 0.01),
                bird_outlier_prob=getattr(config, 'aug_bird_outlier_prob', 0.05),
                bird_z_offset_range=getattr(config, 'aug_bird_z_offset_range', (5.0, 15.0)),
            )

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
        return self.point_aug(coords, attrs)

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
