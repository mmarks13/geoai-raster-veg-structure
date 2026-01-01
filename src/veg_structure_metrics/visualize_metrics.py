#!/usr/bin/env python3
"""Generate 6-panel visualization of vegetation structure metrics."""

import argparse
import matplotlib.pyplot as plt
import rasterio
import numpy as np

# NaN color (light gray) for filtered/no-data pixels
NAN_COLOR = '#D3D3D3'

# 6 panels (band index → display name, colormap)
PANELS = [
    (0, 'Max Height (m)', 'YlGn'),
    (3, 'Canopy Cover', 'Greens'),
    (7, 'Foliage Height Diversity', 'PuBuGn'),
    (4, 'Canopy Density', 'Reds'),
    (10, 'Median Height (m)', 'YlOrBr'),
    (6, 'Understory Density', 'BuGn'),
]

def main():
    parser = argparse.ArgumentParser(description='Generate 6-panel visualization of vegetation metrics')
    parser.add_argument('--input', type=str, required=True, help='Input GeoTIFF')
    parser.add_argument('--output', type=str, required=True, help='Output PNG')
    args = parser.parse_args()

    with rasterio.open(args.input) as src:
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        for ax, (band_idx, title, cmap_name) in zip(axes.flat, PANELS):
            data = src.read(band_idx + 1).astype(np.float32)  # rasterio is 1-indexed

            # Get colormap and set NaN color to gray
            cmap = plt.colormaps[cmap_name].copy()
            cmap.set_bad(color=NAN_COLOR)

            # Display with NaN shown as gray, true zeros shown in colormap
            im = ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=14, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.axis('off')

        # Add legend for NaN color
        fig.text(0.5, 0.02, f'Gray pixels = No data (NaN)', ha='center', fontsize=10, color='gray')

        plt.tight_layout(rect=[0, 0.03, 1, 1])
        plt.savefig(args.output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {args.output}")

if __name__ == '__main__':
    main()
