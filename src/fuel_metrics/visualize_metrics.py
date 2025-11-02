#!/usr/bin/env python3
"""
Visualize fuel metrics rasters - simplified version without basemap.

Plots the 6 most important fuel hazard metrics:
- Canopy Height (Band 4)
- Canopy Base Height / CBH (Band 5)
- Fuel Strata Gap / FSG (Band 6)
- Total Fuel Load / TFL (Band 16)
- Canopy Fuel Load / CFL (Band 15)
- Cover (Band 23)

Note: Band 3 contains the threshold value (e.g., 0.02 kg/m³) used for filtering,
so all metrics after Profil_Type_L are offset by +1 from naive indexing.
"""

import sys
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path

def load_fuel_metrics(input_tif: Path):
    """Load fuel metrics GeoTIFF and extract key bands."""
    with rasterio.open(input_tif) as src:
        # Band indices (1-indexed in rasterio)
        # Full LidarForFuel band order (first 23):
        # 1:Profil_Type, 2:Profil_Type_L, 3:threshold, 4:Height, 5:CBH, 6:FSG,
        # 7:Top_Fuel, 8:H_Bush, 9:continuity, 10:VCI_PAD, 11:VCI_lidr,
        # 12:entropy_lidr, 13:PAI_tot, 14:CBD_max, 15:CFL, 16:TFL, 17:MFL,
        # 18:FL_1_3, 19:GSFL, 20:FL_0_1, 21:FMA, 22:date, 23:Cover
        bands = {
            'Height': 4,           # Canopy height
            'CBH': 5,              # Canopy Base Height
            'FSG': 6,              # Fuel Strata Gap
            'TFL': 16,             # Total Fuel Load
            'CFL': 15,             # Canopy Fuel Load
            'Cover': 23            # Cover percentage
        }

        data = {}
        for name, idx in bands.items():
            band = src.read(idx)
            # Mask NaN and zero values for better visualization
            band = np.ma.masked_where((np.isnan(band)) | (band == 0), band)
            data[name] = band

        # Get metadata
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        shape = src.shape

    return data, transform, crs, bounds, shape


def create_figure(data, transform, crs, bounds, shape, output_path):
    """Create 2x3 subplot figure."""

    # Band configurations (name, colormap, unit, description)
    band_configs = [
        ('Height', 'YlGn', 'm', 'Canopy Height'),
        ('CBH', 'RdYlGn_r', 'm', 'Canopy Base Height'),
        ('FSG', 'PuBuGn', 'm', 'Fuel Strata Gap'),
        ('TFL', 'YlOrRd', 'kg/m²', 'Total Fuel Load'),
        ('CFL', 'Reds', 'kg/m²', 'Canopy Fuel Load'),
        ('Cover', 'Greens', '%', 'Cover')
    ]

    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (band_name, cmap, unit, title) in enumerate(band_configs):
        ax = axes[idx]
        band_data = data[band_name]

        # Plot raster
        im = ax.imshow(
            band_data,
            cmap=cmap,
            interpolation='bilinear'
        )

        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(unit, fontsize=11)

        # Title
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.set_xlabel('Column', fontsize=10)
        ax.set_ylabel('Row', fontsize=10)

        # Add statistics text box
        valid_data = band_data.compressed()  # Get non-masked values
        if len(valid_data) > 0:
            stats_text = (
                f"Min: {valid_data.min():.2f}\n"
                f"Max: {valid_data.max():.2f}\n"
                f"Mean: {valid_data.mean():.2f}\n"
                f"Median: {np.median(valid_data):.2f}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                zorder=3
            )

    # Overall title
    fig.suptitle(
        'Wildfire Fuel Hazard Metrics - LidarForFuel Analysis',
        fontsize=18,
        fontweight='bold',
        y=0.995
    )

    # Add footer with metadata
    resolution = abs(transform[0])
    footer_text = (
        f'Resolution: {resolution:.1f}m | Size: {shape[1]} × {shape[0]} pixels | CRS: {crs.to_string()}'
    )
    fig.text(
        0.5, 0.005, footer_text,
        ha='center',
        fontsize=10,
        style='italic',
        color='gray'
    )

    plt.tight_layout(rect=[0, 0.015, 1, 0.99])

    # Save
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved figure to: {output_path}")

    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_fuel_metrics_simple.py <input_tif> [output_png]")
        print("\nExample:")
        print("  python visualize_fuel_metrics_simple.py data/processed/fuel_metrics/volcan_full/volcan_fuel_metrics_5m.tif")
        sys.exit(1)

    input_tif = Path(sys.argv[1])

    if not input_tif.exists():
        print(f"Error: Input file not found: {input_tif}")
        sys.exit(1)

    # Default output path
    if len(sys.argv) >= 3:
        output_path = Path(sys.argv[2])
    else:
        output_path = input_tif.parent / f"{input_tif.stem}_visualization.png"

    print(f"Loading fuel metrics from: {input_tif}")
    data, transform, crs, bounds, shape = load_fuel_metrics(input_tif)

    print(f"Creating visualization...")
    create_figure(data, transform, crs, bounds, shape, output_path)

    print("Done!")


if __name__ == '__main__':
    main()
