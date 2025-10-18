# Raster Mapping

Visualization utilities for forest plot data.

## Scripts

### `plot_forest_plots.py`
Creates map visualizations of forest plot locations.

**Usage:**
```bash
# Requires coord_transform conda environment
conda activate coord_transform
python src/raster_mapping/plot_forest_plots.py
```

**Input:** `data/raw/forest_plot_data/forest_plot_sample.csv`

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
