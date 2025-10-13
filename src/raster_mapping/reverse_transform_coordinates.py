#!/usr/bin/env python3
"""
Reverse coordinate transformation to recover original UTM coordinates.

**FOR LOCAL USE ONLY - DO NOT SHARE OR COMMIT THIS SCRIPT TO GIT**

This script reverses the obfuscation applied by transform_coordinates.py.
It requires the same COORD_SECRET_KEY to derive transformation parameters.

WARNING: Running this script will expose the original sensitive locations.
Only use this on your local machine when you need to work with true coordinates.
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
    (Same as forward transform)

    Args:
        secret_key: Secret key from environment variable

    Returns:
        rotation_angle: Rotation angle in radians
        translation: Translation vector [tx, ty]
    """
    hash_bytes = hashlib.sha256(secret_key.encode('utf-8')).digest()

    rotation_seed = int.from_bytes(hash_bytes[0:4], byteorder='big')
    rotation_angle = (rotation_seed / (2**32)) * 2 * np.pi

    tx_seed = int.from_bytes(hash_bytes[4:8], byteorder='big')
    ty_seed = int.from_bytes(hash_bytes[8:12], byteorder='big')

    max_translation = 100_000  # Same as forward transform: ±100 km
    tx = ((tx_seed / (2**32)) - 0.5) * max_translation * 2
    ty = ((ty_seed / (2**32)) - 0.5) * max_translation * 2

    translation = np.array([tx, ty])

    return rotation_angle, translation


def get_scramble_offsets(secret_key: str, n_points: int, max_offset: float = 50.0) -> np.ndarray:
    """
    Generate deterministic random scramble offsets.
    (Same as forward transform)

    Args:
        secret_key: Secret key for deterministic RNG
        n_points: Number of points to generate offsets for
        max_offset: Maximum scramble offset in meters (default: 50m)

    Returns:
        offsets: [n_points, 2] array of (dx, dy) offsets
    """
    seed = int(hashlib.sha256(secret_key.encode('utf-8')).hexdigest()[:8], 16) % (2**32)
    rng = np.random.RandomState(seed)

    angles = rng.uniform(0, 2 * np.pi, n_points)
    radii = max_offset * np.sqrt(rng.uniform(0, 1, n_points))

    dx = radii * np.cos(angles)
    dy = radii * np.sin(angles)

    return np.column_stack([dx, dy])


def reverse_transform_coordinates(transformed_coords: np.ndarray, rotation_angle: float,
                                  translation: np.ndarray, scramble_offsets: np.ndarray) -> np.ndarray:
    """
    Reverse affine transformation: un-translate -> un-rotate around centroid -> un-scramble.

    Args:
        transformed_coords: [N, 2] array of transformed coordinates
        rotation_angle: Rotation angle in radians (same as forward)
        translation: Translation vector [tx, ty] (same as forward)
        scramble_offsets: [N, 2] array of scramble offsets (same as forward)

    Returns:
        original_coords: [N, 2] array of original coordinates
    """
    # Step 1: Remove translation and find centroid
    # The forward transform did: rotated + centroid + translation
    # So we need to figure out the original centroid position

    # First apply scramble to get the scrambled coordinates' centroid
    # This is a bit tricky - we need to work backwards
    # transformed = rotated + centroid + translation
    # We need the centroid of the scrambled coords before rotation

    # For simplicity: subtract translation first
    no_translation = transformed_coords - translation

    # The centroid is at the mean of no_translation (after removing rotation effect)
    centroid = no_translation.mean(axis=0)
    centered = no_translation - centroid

    # Step 2: Reverse rotation (rotate by -angle)
    cos_theta = np.cos(-rotation_angle)
    sin_theta = np.sin(-rotation_angle)
    rotation_matrix_inv = np.array([
        [cos_theta, -sin_theta],
        [sin_theta, cos_theta]
    ])

    rotated_back = centered @ rotation_matrix_inv.T

    # Add centroid back
    scrambled = rotated_back + centroid

    # Step 3: Reverse scramble
    original = scrambled - scramble_offsets

    return original


def main():
    """Reverse transform coordinates to recover original locations."""
    # Get secret key from environment
    secret_key = os.getenv('COORD_SECRET_KEY')
    if not secret_key:
        print("Error: COORD_SECRET_KEY environment variable not set", file=sys.stderr)
        print("Please set it in your .env file or export it in your shell", file=sys.stderr)
        sys.exit(1)

    # Define paths
    repo_root = Path(__file__).parent.parent
    input_file = repo_root / 'data' / 'processed' / 'forest_plot_data' / 'forest_plot_sample_obfuscated.csv'
    output_file = repo_root / 'data' / 'raw' / 'forest_plot_data' / 'forest_plot_sample_recovered.csv'

    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        print("Please run transform_coordinates.py first", file=sys.stderr)
        sys.exit(1)

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load transformed data
    print(f"Loading transformed data from {input_file}")
    df = pd.read_csv(input_file)

    # Verify coordinate columns exist
    if 'Easting' not in df.columns or 'Northing' not in df.columns:
        print("Error: 'Easting' and 'Northing' columns not found in CSV", file=sys.stderr)
        sys.exit(1)

    # Extract valid coordinates
    valid_coords = df[['Easting', 'Northing']].dropna()
    n_valid = len(valid_coords)

    if n_valid == 0:
        print("Error: No valid coordinates found", file=sys.stderr)
        sys.exit(1)

    print(f"Found {n_valid} valid coordinate pairs")

    # Derive transformation parameters (same as forward)
    rotation_angle, translation = derive_transform_params(secret_key)
    print(f"Using rotation: {np.degrees(rotation_angle):.2f}°")
    print(f"Using translation: ({translation[0]:.2f}, {translation[1]:.2f}) meters")

    # Generate scramble offsets (deterministic, same as forward)
    scramble_offsets = get_scramble_offsets(secret_key, n_valid, max_offset=50.0)

    # Reverse transform coordinates
    transformed_coords = valid_coords[['Easting', 'Northing']].values
    original_coords = reverse_transform_coordinates(transformed_coords, rotation_angle,
                                                    translation, scramble_offsets)

    # Create output dataframe
    df_out = df.copy()

    # Replace with recovered original coordinates
    df_out.loc[valid_coords.index, 'Easting'] = original_coords[:, 0]
    df_out.loc[valid_coords.index, 'Northing'] = original_coords[:, 1]

    # Save recovered data
    df_out.to_csv(output_file, index=False)
    print(f"\nRecovered original coordinates saved to {output_file}")
    print(f"\n⚠️  WARNING: This file contains sensitive location data!")
    print(f"   Do NOT commit this file to git or share it externally")

    # Display statistics
    print(f"\nRecovered centroid: ({original_coords.mean(axis=0)[0]:.2f}, "
          f"{original_coords.mean(axis=0)[1]:.2f})")


if __name__ == '__main__':
    main()
