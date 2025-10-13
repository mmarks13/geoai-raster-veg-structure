# Raster Mapping

Coordinate transformation and visualization utilities for forest plot data.

## Scripts

### `transform_coordinates.py`
Transforms UTM coordinates deterministically to obfuscate sensitive locations while preserving geometric properties (distances, angles, areas).

**Transformation:**
1. 50m random scramble (deterministic per secret key)
2. Rotation around centroid (deterministic angle)
3. Translation (±100km, deterministic offset)

**Usage:**
```bash
# Requires COORD_SECRET_KEY environment variable
python src/raster_mapping/transform_coordinates.py
```

**Input:** `data/raw/forest_plot_data/forest_plot_sample.csv`  
**Output:** `data/processed/forest_plot_data/forest_plot_sample_obfuscated.csv`

### `reverse_transform_coordinates.py`
⚠️ **SENSITIVE - DO NOT SHARE OR COMMIT**

Reverses the obfuscation to recover original UTM coordinates. Only use locally when true coordinates are needed.

**Usage:**
```bash
# Requires same COORD_SECRET_KEY as forward transform
python src/raster_mapping/reverse_transform_coordinates.py
```

**Input:** `data/processed/forest_plot_data/forest_plot_sample_obfuscated.csv`  
**Output:** `data/raw/forest_plot_data/forest_plot_sample_recovered.csv` (NOT tracked in git)

### `plot_forest_plots.py`
Creates map visualizations of obfuscated forest plot locations.

**Usage:**
```bash
# Requires coord_transform conda environment
conda activate coord_transform
python src/raster_mapping/plot_forest_plots.py
```

**Output:** `temp/forest_plots/`
- `forest_plots_map.png` - Map with OpenStreetMap basemap (if contextily available)
- `forest_plots_map_simple.png` - Simple coordinate plot

**Features:**
- Color-coded by year (2023: orange, 2024: blue)
- Summary statistics by year and site
- Optional OpenStreetMap basemap

## Environment

Uses `coord_transform` conda environment with:
- pandas
- numpy
- geopandas
- shapely
- matplotlib
- contextily (optional, for basemap)

## Security Notes

- **Never commit** `reverse_transform_coordinates.py` output files (`*_recovered.csv`)
- **Never share** the `COORD_SECRET_KEY` environment variable
- **Only use obfuscated data** (`*_obfuscated.csv`) for sharing and version control
- Geometric properties are preserved for analysis, but true locations are protected
