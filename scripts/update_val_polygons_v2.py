#!/usr/bin/env python3
"""
Update validation polygons for Phase 3: Multi-Site Vegetation Structure Metrics.

Changes:
1. Remove T03-T13 (smaller sedgwick site) - now used for training
2. Shrink Volcan Mountain polygons by 50% vertically (identified by name_2 containing "vlcn_")
3. Keep T01-T09 (larger sedgwick site) unchanged for validation

Usage:
    python scripts/update_val_polygons_v2.py

Output:
    data/processed/test_val_polygons_v2.geojson
"""

import geopandas as gpd
from shapely.affinity import scale


def main():
    input_path = 'data/processed/test_val_polygons.geojson'
    output_path = 'data/processed/test_val_polygons_v2.geojson'

    print(f"Reading: {input_path}")
    gdf = gpd.read_file(input_path)
    print(f"  Found {len(gdf)} polygons")

    # Identify rows to remove (T03-T13)
    # T03-T13 is the smaller sedgwick site, identified by 'name' containing 'T03-T13'
    remove_mask = gdf['name'].str.contains('T03-T13', na=False)
    removed_count = remove_mask.sum()
    gdf = gdf[~remove_mask].copy()
    print(f"  Removed {removed_count} T03-T13 polygon(s)")

    # Shrink Volcan polygons by 50% vertically
    # Identified by name_2 containing "vlcn_"
    shrink_count = 0
    for idx, row in gdf.iterrows():
        name_2 = str(row.get('name_2', ''))
        if 'vlcn_' in name_2:
            centroid = row.geometry.centroid
            # Scale by 50% in y-direction (vertical), keep x unchanged
            scaled_geom = scale(row.geometry, xfact=1.0, yfact=0.5, origin=centroid)
            gdf.at[idx, 'geometry'] = scaled_geom
            shrink_count += 1
            print(f"  Shrunk polygon: {name_2}")

    print(f"  Shrunk {shrink_count} Volcan polygon(s) by 50% vertically")

    # Reset index for clean output
    gdf = gdf.reset_index(drop=True)

    # Save output
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"\nSaved: {output_path}")
    print(f"  Final polygon count: {len(gdf)}")

    # Print summary
    print("\nValidation polygons summary:")
    for idx, row in gdf.iterrows():
        print(f"  {idx}: name_2={row.get('name_2', 'N/A')}, bounds={row.geometry.bounds}")


if __name__ == '__main__':
    main()
