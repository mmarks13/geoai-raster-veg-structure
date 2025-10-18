# Raster Mapping

Processing and visualization utilities for forest plot data.

## Scripts

### `process_forest_plots.py`
Processes raw forest plot Excel data into filtered, georeferenced outputs.

**Setup:**
Create a filter configuration file at `data/raw/forest_plot_data/site_filter.txt` with format:
```
# Lines starting with # are comments
Year,Site,District,Forest
```

**Usage:**
```bash
# Requires coord_transform conda environment
conda activate coord_transform
python src/raster_mapping/process_forest_plots.py
```

**Input:**
- `data/raw/forest_plot_data/forest_plots_raw.xlsx` - Raw Excel data
- `data/raw/forest_plot_data/site_filter.txt` - Filter configuration (NOT tracked in git)

**Output:**
- `data/processed/forest_plot_data/forest_plots_processed.csv` - Filtered data with coordinates
- `data/processed/forest_plot_data/forest_plots_processed.gpkg` - Georeferenced GeoPackage (EPSG:26911)

**Processing steps:**
1. Loads filter criteria from `site_filter.txt`
2. Parses concatenated `Site_Year` column into separate Site and Year
3. Filters to specified sites/years from configuration file
4. Creates georeferenced outputs in UTM Zone 11N (EPSG:26911)

### `plot_forest_plots.py`
Creates map visualizations of processed forest plot locations.

**Usage:**
```bash
# Requires coord_transform conda environment
# Run process_forest_plots.py first!
conda activate coord_transform
python src/raster_mapping/plot_forest_plots.py
```

**Input:** `data/processed/forest_plot_data/forest_plots_processed.csv`

**Output:** `temp/forest_plots/`
- `forest_plots_map.png` - Map with OpenStreetMap basemap (if contextily available)
- `forest_plots_map_simple.png` - Simple coordinate plot
- Individual maps per site

**Features:**
- Color-coded by year (2023: orange, 2024: blue)
- Summary statistics by year and site
- Optional OpenStreetMap basemap

## Workflow

1. **Process raw data:**
   ```bash
   conda activate coord_transform
   python src/raster_mapping/process_forest_plots.py
   ```

2. **Create visualizations:**
   ```bash
   python src/raster_mapping/plot_forest_plots.py
   ```

## Environment

Uses `coord_transform` conda environment with:
- pandas
- numpy
- geopandas
- shapely
- matplotlib
- openpyxl (for Excel reading)
- contextily (optional, for basemap)
