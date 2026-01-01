#!/usr/bin/env python3
"""
Visualize tile alignment between NAIP imagery and target raster.

Shows NAIP imagery (20m bbox, 40x40 pixels) alongside target raster (10m bbox, 5x5 pixels)
to verify spatial alignment and grid conventions.

Usage:
    # Interactive mode - load once, browse with keyboard
    python scripts/visualize_tile_alignment.py \
        --pt-file data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt \
        --interactive

    # Single tile by index
    python scripts/visualize_tile_alignment.py \
        --pt-file data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt \
        --index 100 \
        --output tile_100.png

    # Single tile by tile_id
    python scripts/visualize_tile_alignment.py \
        --pt-file data/processed/model_data_raster/precomputed_training_tiles_raster_32bit.pt \
        --tile-id "t01_0010_0020"

Interactive controls:
    n/Right: Next tile
    p/Left: Previous tile
    j: Jump to index
    s: Save current figure
    q: Quit
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
import numpy as np
import torch


def load_tiles(pt_file: str):
    """Load tiles from .pt file."""
    print(f"Loading tiles from {pt_file}...")
    tiles = torch.load(pt_file, weights_only=False)
    print(f"Loaded {len(tiles)} tiles")
    return tiles


def load_normalization_stats(pt_file: str) -> dict:
    """
    Load target raster normalization stats from JSON file.

    Looks for target_raster_normalization_stats_train.json in same directory as pt_file.
    """
    pt_path = Path(pt_file)
    stats_path = pt_path.parent / "target_raster_normalization_stats_train.json"

    if not stats_path.exists():
        # Try fuel_metrics variant
        stats_path = pt_path.parent / "fuel_metrics_normalization_stats_train.json"

    if stats_path.exists():
        print(f"Loading normalization stats from {stats_path}")
        with open(stats_path) as f:
            return json.load(f)
    else:
        print("Warning: No normalization stats found, displaying raw values")
        return None


def denormalize_band(data: np.ndarray, band_idx: int, stats: dict) -> np.ndarray:
    """Denormalize a single band using z-score stats."""
    if stats is None:
        return data

    mean_key = f"band_{band_idx}_mean"
    std_key = f"band_{band_idx}_std"

    if mean_key in stats and std_key in stats:
        mean = stats[mean_key]
        std = stats[std_key]
        return data * std + mean
    else:
        return data


def get_naip_rgb(tile: dict) -> np.ndarray:
    """Extract NAIP RGB from tile, handling different storage formats."""
    naip_data = None

    # Check for dict format with 'images' subkey first
    if 'naip' in tile and isinstance(tile['naip'], dict) and 'images' in tile['naip']:
        naip_data = tile['naip']['images']
    # Then try direct tensor keys
    elif 'naip_imgs' in tile and tile['naip_imgs'] is not None:
        naip_data = tile['naip_imgs']
    elif 'naip' in tile and tile['naip'] is not None and not isinstance(tile['naip'], dict):
        naip_data = tile['naip']

    if naip_data is None:
        return None

    # Convert to numpy if tensor
    if isinstance(naip_data, torch.Tensor):
        naip_data = naip_data.numpy()

    # Handle different shapes
    # Shape could be: [n_images, bands, H, W] or [bands, H, W]
    if len(naip_data.shape) == 4:
        rgb = naip_data[0, :3, :, :]  # First image, RGB bands
    elif len(naip_data.shape) == 3:
        rgb = naip_data[:3, :, :]
    else:
        return None

    # Transpose from [C, H, W] to [H, W, C] for matplotlib
    rgb = np.transpose(rgb, (1, 2, 0))

    # Normalize to [0, 1] for display
    rgb = rgb.astype(np.float32)
    if rgb.max() > 1:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)

    return rgb


def get_target_raster(tile: dict) -> np.ndarray:
    """Extract target raster from tile."""
    # Check various key names used in different pipeline stages
    data = None
    for key in ['fuel_metrics', 'target_raster', 'fuel_metrics_fuel_metrics']:
        if key in tile and tile[key] is not None:
            data = tile[key]
            break

    if data is None:
        return None

    if isinstance(data, torch.Tensor):
        data = data.numpy()

    return data


def get_bbox(tile: dict) -> tuple:
    """Extract bounding box from tile."""
    if 'bbox' in tile:
        bbox = tile['bbox']
        if isinstance(bbox, torch.Tensor):
            bbox = bbox.numpy()
        if len(bbox) == 4:
            return tuple(bbox)  # (xmin, ymin, xmax, ymax)
    return None


def compute_dsm_from_points(tile: dict, grid_size: int = 5) -> np.ndarray:
    """
    Compute DSM (max Z per cell) from 3DEP point cloud.

    Uses same aggregation as compute_vegetation_structure_metrics():
    - np.max of HeightAboveGround (or bbox-normalized Z) per cell
    - Rasterio convention: row 0 = north (Y inverted)

    Args:
        tile: Tile dict with point cloud data
        grid_size: Number of cells per dimension (default 5 for 2m cells)

    Returns:
        DSM array [grid_size, grid_size] with max Z values, or None if no points
    """
    # Try bbox-normalized points first (X,Y in [-5,5], Z in meters above min)
    points = None
    for key in ['dep_points_bbox_norm', 'dep_points_norm']:
        if key in tile and tile[key] is not None:
            pts = tile[key]
            if isinstance(pts, torch.Tensor):
                pts = pts.numpy()
            if len(pts) > 0:
                points = pts
                break

    if points is None or len(points) == 0:
        return None

    # Grid boundaries: [-5, 5] in both X and Y (10m tile)
    cell_size = 10.0 / grid_size  # 2m for 5x5 grid

    # Initialize DSM with NaN
    dsm = np.full((grid_size, grid_size), np.nan)

    # Bin points into grid cells using rasterio convention:
    # X: [-5, 5] -> col [0, grid_size-1] (west to east)
    # Y: [-5, 5] -> row [0, grid_size-1] BUT row 0 = north (Y inverted)
    cols = np.floor((points[:, 0] + 5) / cell_size).astype(int)
    # Invert Y: row 0 = north (max Y), row 4 = south (min Y)
    rows = np.floor((5 - points[:, 1]) / cell_size).astype(int)

    # Clamp to valid range
    cols = np.clip(cols, 0, grid_size - 1)
    rows = np.clip(rows, 0, grid_size - 1)

    # Compute max Z per cell (same as compute_vegetation_structure_metrics)
    for i in range(grid_size):
        for j in range(grid_size):
            mask = (rows == i) & (cols == j)
            if np.any(mask):
                dsm[i, j] = np.max(points[mask, 2])

    return dsm


def visualize_tile(tile: dict, tile_idx: int, tile_id: str = None, save_path: str = None, norm_stats: dict = None):
    """
    Visualize a single tile showing NAIP, target raster, and 3DEP DSM alignment.

    Layout (3 panels):
    - Left: NAIP RGB with 5x5 grid overlay
    - Middle: Target raster band 0 (denormalized if stats provided)
    - Right: 3DEP DSM (max Z per 2m cell)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Get data
    naip_rgb = get_naip_rgb(tile)
    target = get_target_raster(tile)
    bbox = get_bbox(tile)
    dsm = compute_dsm_from_points(tile)

    # Denormalize target band 0 if stats available
    target_band0 = None
    if target is not None:
        target_band0 = denormalize_band(target[0].copy(), 0, norm_stats)

    title = f"Tile {tile_idx}"
    if tile_id:
        title += f" ({tile_id})"
    if bbox:
        title += f"  |  Bbox: [{bbox[0]:.1f}, {bbox[1]:.1f}] to [{bbox[2]:.1f}, {bbox[3]:.1f}]"

    fig.suptitle(title, fontsize=12)

    # Compute shared vmax for consistent color scale
    shared_vmax = 0
    if target_band0 is not None:
        shared_vmax = max(shared_vmax, np.nanmax(target_band0))
    if dsm is not None:
        shared_vmax = max(shared_vmax, np.nanmax(dsm))
    if shared_vmax == 0:
        shared_vmax = None  # Let matplotlib auto-scale

    # Grid offsets for 5x5 cells
    grid_offsets = np.array([-4, -2, 0, 2, 4])

    # Left panel: NAIP RGB with grid overlay
    ax1 = axes[0]

    if naip_rgb is not None:
        ax1.imshow(naip_rgb, extent=[0, 20, 0, 20], origin='lower')

        # Draw tile bbox (inner 10m centered)
        rect = mpatches.Rectangle((5, 5), 10, 10, linewidth=2, edgecolor='red',
                                   facecolor='none', linestyle='-')
        ax1.add_patch(rect)

        # Draw 5x5 grid within tile bbox (2m cells)
        center_x, center_y = 10, 10
        for i, offset_y in enumerate(grid_offsets):
            for j, offset_x in enumerate(grid_offsets):
                cell_x = center_x + offset_x
                cell_y = center_y + offset_y

                # Cell bounds (2m x 2m cells) - outline only
                cell_rect = mpatches.Rectangle(
                    (cell_x - 1, cell_y - 1), 2, 2,
                    linewidth=1, edgecolor='yellow',
                    facecolor='none'
                )
                ax1.add_patch(cell_rect)

                # Label corner cell indices
                if (i, j) == (0, 0):
                    ax1.text(cell_x, cell_y, '[0,0]',
                            fontsize=9, color='red', fontweight='bold',
                            ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                elif (i, j) in [(0, 4), (4, 0), (4, 4)]:
                    ax1.text(cell_x, cell_y, f'[{i},{j}]',
                            fontsize=8, color='cyan', ha='center', va='center',
                            bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

        ax1.set_xlim(0, 20)
        ax1.set_ylim(0, 20)
        ax1.set_xlabel('X (meters from img bbox min)')
        ax1.set_ylabel('Y (meters from img bbox min)')
        ax1.set_title('NAIP + 5x5 Grid Overlay')
    else:
        ax1.text(0.5, 0.5, 'No NAIP data', ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title("NAIP (not available)")

    # Middle panel: Target raster band 0 (denormalized)
    ax2 = axes[1]

    if target_band0 is not None:
        im2 = ax2.imshow(target_band0, extent=[0, 10, 0, 10], origin='lower', cmap='viridis', vmin=0, vmax=shared_vmax)
        ax2.set_title('Target Band 0 (denormalized)\nDisplayed with origin="lower"')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # Mark corners with indices
        ax2.text(1, 1, '[0,0]', fontsize=10, color='red', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax2.text(9, 1, '[0,4]', fontsize=8, color='blue', ha='center', va='center')
        ax2.text(1, 9, '[4,0]', fontsize=8, color='blue', ha='center', va='center')
        ax2.text(9, 9, '[4,4]', fontsize=8, color='blue', ha='center', va='center')

        # Draw 2m grid
        for x in [2, 4, 6, 8]:
            ax2.axvline(x, color='white', linestyle=':', alpha=0.5)
        for y in [2, 4, 6, 8]:
            ax2.axhline(y, color='white', linestyle=':', alpha=0.5)

        ax2.set_xlabel('X (meters)')
        ax2.set_ylabel('Y (meters)')
    else:
        ax2.text(0.5, 0.5, 'No target raster', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Target (not available)")

    # Right panel: 3DEP DSM
    ax3 = axes[2]

    if dsm is not None:
        im3 = ax3.imshow(dsm, extent=[0, 10, 0, 10], origin='lower', cmap='viridis', vmin=0, vmax=shared_vmax)
        ax3.set_title('3DEP DSM (max Z per 2m cell)\nDisplayed with origin="lower"')
        plt.colorbar(im3, ax=ax3, fraction=0.046)

        # Mark corners with indices
        ax3.text(1, 1, '[0,0]', fontsize=10, color='red', fontweight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax3.text(9, 1, '[0,4]', fontsize=8, color='blue', ha='center', va='center')
        ax3.text(1, 9, '[4,0]', fontsize=8, color='blue', ha='center', va='center')
        ax3.text(9, 9, '[4,4]', fontsize=8, color='blue', ha='center', va='center')

        # Draw 2m grid
        for x in [2, 4, 6, 8]:
            ax3.axvline(x, color='white', linestyle=':', alpha=0.5)
        for y in [2, 4, 6, 8]:
            ax3.axhline(y, color='white', linestyle=':', alpha=0.5)

        ax3.set_xlabel('X (meters)')
        ax3.set_ylabel('Y (meters)')
    else:
        ax3.text(0.5, 0.5, 'No 3DEP points', ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title("3DEP DSM (not available)")

    # Add explanation
    fig.text(0.5, 0.02,
             'Grid indexing: [row, col] where row 0 = SOUTH in display (origin="lower")\n'
             'In rasterio source: row 0 = NORTH. If DSM and Target match, grid alignment is correct.',
             fontsize=9, ha='center', va='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")

    return fig


class InteractiveViewer:
    """Interactive viewer for browsing tiles (CLI-based)."""

    def __init__(self, tiles: list, norm_stats: dict = None, output_dir: str = "/tmp"):
        self.tiles = tiles
        self.norm_stats = norm_stats
        self.current_idx = 0
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def show(self, start_idx: int = 0):
        """Start interactive viewer with CLI menu."""
        self.current_idx = start_idx

        print(f"\nInteractive mode - {len(self.tiles)} tiles loaded")
        print("Commands: n=next, p=prev, j=jump, s=save, q=quit")
        print("-" * 50)

        while True:
            # Generate and save current tile
            self.save_current()

            # Show prompt
            tile = self.tiles[self.current_idx]
            tile_id = tile.get('tile_id', 'unknown')
            print(f"\nTile {self.current_idx}/{len(self.tiles)-1} ({tile_id})")
            print(f"  Saved to: {self.output_dir}/tile_view.png")

            try:
                cmd = input("Command [n/p/j/s/q]: ").strip().lower()
            except EOFError:
                break

            if cmd in ['n', '']:
                self.current_idx = (self.current_idx + 1) % len(self.tiles)
            elif cmd == 'p':
                self.current_idx = (self.current_idx - 1) % len(self.tiles)
            elif cmd == 'j':
                try:
                    idx = int(input(f"  Index (0-{len(self.tiles)-1}): "))
                    if 0 <= idx < len(self.tiles):
                        self.current_idx = idx
                    else:
                        print(f"  Index out of range")
                except ValueError:
                    print("  Invalid index")
            elif cmd == 's':
                save_path = f"tile_{self.current_idx}.png"
                tile = self.tiles[self.current_idx]
                tile_id = tile.get('tile_id', None)
                fig = visualize_tile(tile, self.current_idx, tile_id, save_path, self.norm_stats)
                plt.close(fig)
                print(f"  Saved to {save_path}")
            elif cmd == 'q':
                print("Exiting.")
                break
            else:
                print(f"  Unknown command: {cmd}")

    def save_current(self):
        """Save current tile visualization to temp file."""
        tile = self.tiles[self.current_idx]
        tile_id = tile.get('tile_id', None)
        output_path = self.output_dir / "tile_view.png"
        fig = visualize_tile(tile, self.current_idx, tile_id, str(output_path), self.norm_stats)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize tile alignment between NAIP and target raster'
    )
    parser.add_argument(
        '--pt-file',
        type=str,
        required=True,
        help='Path to preprocessed tiles .pt file'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Interactive mode - browse tiles with keyboard'
    )
    parser.add_argument(
        '--index',
        type=int,
        default=None,
        help='Tile index to visualize'
    )
    parser.add_argument(
        '--tile-id',
        type=str,
        default=None,
        help='Tile ID to visualize (requires loading all tiles)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for saving figure'
    )

    args = parser.parse_args()

    # Load tiles and normalization stats
    tiles = load_tiles(args.pt_file)
    norm_stats = load_normalization_stats(args.pt_file)

    if args.interactive:
        # Interactive mode
        start_idx = args.index if args.index is not None else 0
        viewer = InteractiveViewer(tiles, norm_stats)
        viewer.show(start_idx)
    elif args.tile_id:
        # Find by tile_id
        for idx, tile in enumerate(tiles):
            if tile.get('tile_id') == args.tile_id:
                fig = visualize_tile(tile, idx, args.tile_id, args.output, norm_stats)
                if not args.output:
                    plt.show()
                return
        print(f"Tile ID '{args.tile_id}' not found")
    elif args.index is not None:
        # Single tile by index
        if 0 <= args.index < len(tiles):
            tile = tiles[args.index]
            tile_id = tile.get('tile_id', None)
            fig = visualize_tile(tile, args.index, tile_id, args.output, norm_stats)
            if not args.output:
                plt.show()
        else:
            print(f"Index {args.index} out of range (0-{len(tiles)-1})")
    else:
        # Default: show first tile
        tile = tiles[0]
        tile_id = tile.get('tile_id', None)
        fig = visualize_tile(tile, 0, tile_id, args.output, norm_stats)
        if not args.output:
            plt.show()


if __name__ == '__main__':
    main()
