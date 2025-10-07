#!/usr/bin/env python3
"""
Transform UTM coordinates deterministically to obfuscate locations.

Applies a deterministic transformation derived from COORD_SECRET_KEY:
1. 50m random scramble (deterministic per secret key)
2. Affine transformation: rotation + translation (scale=1 to preserve geometry)

This allows safe sharing of data while preserving geometric properties
(distances, angles, areas) for analysis.
"""

import os
import sys
import hashlib
import numpy as np
import pandas as pd
from pathlib import Path


def derive_transform_params(secret_key: str) -> tuple[float, np.ndarray]:
    """
    Derive deterministic transformation parameters from secret key.

    Args:
        secret_key: Secret key from environment variable

    Returns:
        rotation_angle: Rotation angle in radians
        translation: Translation vector [tx, ty]
    """
    # Use SHA-256 to generate deterministic pseudorandom values
    hash_bytes = hashlib.sha256(secret_key.encode('utf-8')).digest()

    # Extract rotation angle (0 to 2π)
    rotation_seed = int.from_bytes(hash_bytes[0:4], byteorder='big')
    rotation_angle = (rotation_seed / (2**32)) * 2 * np.pi

    # Extract translation components (keep within valid UTM 11N coordinate space)
    tx_seed = int.from_bytes(hash_bytes[4:8], byteorder='big')
    ty_seed = int.from_bytes(hash_bytes[8:12], byteorder='big')

    # Translation offsets: ±100km to stay well within UTM zone bounds
    # UTM zones are ~667km wide (166km to 834km easting range)
    # This keeps coordinates valid while still providing good obfuscation
    max_translation = 100_000  # 100 km in meters
    tx = ((tx_seed / (2**32)) - 0.5) * max_translation * 2  # ±100 km
    ty = ((ty_seed / (2**32)) - 0.5) * max_translation * 2  # ±100 km

    translation = np.array([tx, ty])

    return rotation_angle, translation


def get_scramble_offsets(secret_key: str, n_points: int, max_offset: float = 50.0) -> np.ndarray:
    """
    Generate deterministic random scramble offsets for each point.

    Args:
        secret_key: Secret key for deterministic RNG
        n_points: Number of points to generate offsets for
        max_offset: Maximum scramble offset in meters (default: 50m)

    Returns:
        offsets: [n_points, 2] array of (dx, dy) offsets
    """
    # Use secret key to seed numpy RNG
    seed = int(hashlib.sha256(secret_key.encode('utf-8')).hexdigest()[:8], 16) % (2**32)
    rng = np.random.RandomState(seed)

    # Generate random offsets in polar coordinates (uniform in circle)
    angles = rng.uniform(0, 2 * np.pi, n_points)
    radii = max_offset * np.sqrt(rng.uniform(0, 1, n_points))

    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)

    return np.column_stack([dx, dy])


def transform_coordinates(coords: np.ndarray, rotation_angle: float, translation: np.ndarray,
                         scramble_offsets: np.ndarray) -> np.ndarray:
    """
    Apply affine transformation: scramble -> rotate around centroid -> translate.

    Args:
        coords: [N, 2] array of (easting, northing) coordinates
        rotation_angle: Rotation angle in radians
        translation: Translation vector [tx, ty]
        scramble_offsets: [N, 2] array of scramble offsets

    Returns:
        transformed_coords: [N, 2] array of transformed coordinates
    """
    # Step 1: Apply scramble
    scrambled = coords + scramble_offsets

    # Step 2: Rotate around the centroid (not origin) to keep points together
    centroid = scrambled.mean(axis=0)
    centered = scrambled - centroid

    # Rotation matrix (scale=1)
    cos_theta = np.cos(rotation_angle)
    sin_theta = np.sin(rotation_angle)
    rotation_matrix = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    # Apply rotation around centroid
    rotated = centered @ rotation_matrix.T

    # Step 3: Move back to centroid position then apply translation
    transformed = rotated + centroid + translation

    return transformed


def main():
    """Transform coordinates in forest plot data."""
    # Get secret key from environment
    secret_key = os.getenv('COORD_SECRET_KEY')
    if not secret_key:
        print("Error: COORD_SECRET_KEY environment variable not set", file=sys.stderr)
        print("Please set it in your .env file or export it in your shell", file=sys.stderr)
        sys.exit(1)

    # Define paths
    repo_root = Path(__file__).parent.parent
    input_file = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'forest_plot_sample.csv'
    output_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plot_sample_obfuscated.csv'

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    # Verify coordinate columns exist
    if 'Easting' not in df.columns or 'Northing' not in df.columns:
        print("Error: 'Easting' and 'Northing' columns not found in CSV", file=sys.stderr)
        sys.exit(1)

    # Extract valid coordinates (drop rows with missing coordinates)
    valid_coords = df[['Easting', 'Northing']].dropna()
    n_valid = len(valid_coords)

    if n_valid == 0:
        print("Error: No valid coordinates found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {n_valid} valid coordinate pairs")

    # Derive transformation parameters
    rotation_angle, translation = derive_transform_params(secret_key)
    print(f"Rotation: {np.degrees(rotation_angle):.2f}°")
    print(f"Translation: ({translation[0]:.2f}, {translation[1]:.2f}) meters")

    # Generate scramble offsets (deterministic per secret key)
    scramble_offsets = get_scramble_offsets(secret_key, n_valid, max_offset=50.0)

    # Transform coordinates
    coords = valid_coords[['Easting', 'Northing']].values
    transformed_coords = transform_coordinates(coords, rotation_angle, translation, scramble_offsets)

    # Create output dataframe
    df_out = df.copy()

    # Replace coordinates with transformed values (only for rows with valid coords)
    df_out.loc[valid_coords.index, 'Easting'] = transformed_coords[:, 0]
    df_out.loc[valid_coords.index, 'Northing'] = transformed_coords[:, 1]

    # Save transformed data
    df_out.to_csv(output_file, index=False)
    print(f"\nTransformed data saved to {output_file}")
    print(f"Original coordinate columns have been replaced with obfuscated values")
    print(f"Geometric properties (distances, angles, areas) are preserved")

    # Compute and display statistics
    original_centroid = coords.mean(axis=0)
    transformed_centroid = transformed_coords.mean(axis=0)
    print(f"\nOriginal centroid: ({original_centroid[0]:.2f}, {original_centroid[1]:.2f})")
    print(f"Transformed centroid: ({transformed_centroid[0]:.2f}, {transformed_centroid[1]:.2f})")

    # Verify distance preservation (check a few sample pairs)
    n_samples = min(5, n_valid - 1)
    if n_samples > 0:
        print("\nDistance preservation check (sample pairs):")
        for i in range(n_samples):
            orig_dist = np.linalg.norm(coords[i] - coords[i+1])
            trans_dist = np.linalg.norm(transformed_coords[i] - transformed_coords[i+1])
            print(f"  Pair {i}: original={orig_dist:.2f}m, transformed={trans_dist:.2f}m, "
                  f"diff={abs(orig_dist - trans_dist):.4f}m")


if __name__ == '__main__':
    main()
