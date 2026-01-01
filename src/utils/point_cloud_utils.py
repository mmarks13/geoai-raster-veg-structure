"""
Point Cloud to Raster Aggregation Utilities

Functions for processing LiDAR point clouds and calculating vegetation structure metrics
as proposed by Moudry et al. (2023). Supports aggregating point cloud data into 
raster format with various statistical measures.

Moudry et al. (2023) proposed ten standardized aerial LiDAR-derived vegetation 
structure variables (e.g., maximum vegetation height, mean vegetation height, 
canopy cover percentages, and foliage height diversity) that should be made available 
in common raster formats to assist ecological research. This module calculates each of 
these standardized variables from point clouds at a chosen standard resolution.
"""

import gc
import json
import os
import time
import numpy as np
import pandas as pd
import pdal
import laspy
import rasterio
import earthpy.plot as ep
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import MultiPoint, Polygon, LineString, Point
from shapely import wkt
import matplotlib.colors as mcolors

import rasterio
from rasterio import features
from shapely.geometry import shape, mapping
import numpy as np


# Band descriptions for Moudry et al. (2023) vegetation structure metrics
MOUDRY_BAND_DESCRIPTIONS = {
    0: {'name': 'max_height', 'description': 'Maximum vegetation height (m)', 'unit': 'm'},
    1: {'name': 'mean_height', 'description': 'Mean vegetation height (m)', 'unit': 'm'},
    2: {'name': 'std_height', 'description': 'Standard deviation of vegetation height (m)', 'unit': 'm'},
    3: {'name': 'canopy_cover', 'description': 'Canopy cover - proportion of all returns above canopy threshold', 'unit': 'fraction'},
    4: {'name': 'canopy_density', 'description': 'Canopy density - proportion of vegetation returns in canopy layer', 'unit': 'fraction'},
    5: {'name': 'mid_story_density', 'description': 'Mid-story density - proportion of vegetation returns in mid-story', 'unit': 'fraction'},
    6: {'name': 'understory_density', 'description': 'Understory density - proportion of vegetation returns in understory', 'unit': 'fraction'},
    7: {'name': 'foliage_height_diversity', 'description': 'Foliage Height Diversity - Shannon-Wiener index', 'unit': 'index'},
    # Percentile bands 8-12 are added dynamically based on percentiles parameter
    # Density proportion bands 13-22 are added dynamically based on num_density_layers parameter
}


def get_pointcloud_footprint(las_file_path, simplify_tolerance=None, buffer_distance=None):
    """
    Determine the footprint of a point cloud and return it as a WKT string.
    
    This function extracts the X and Y coordinates from a LAS/LAZ file,
    computes the convex hull to get the footprint, and optionally simplifies
    and/or buffers the resulting polygon.
    
    Parameters:
        las_file_path (str): Path to the LAS/LAZ file
        simplify_tolerance (float, optional): Tolerance for simplifying the polygon.
                                             Larger values result in more simplification.
        buffer_distance (float, optional): Distance to buffer the polygon.
                                          Positive values expand, negative values shrink.
    
    Returns:
        str: WKT string representing the point cloud footprint
    """
    # Start timing
    start_time = time.time()
    
    # Load the LAS file
    print(f"Reading point cloud: {os.path.basename(las_file_path)}")
    las = laspy.read(las_file_path)
    
    # Extract X and Y coordinates - use a sample if the point cloud is very large
    num_points = len(las.x)
    if num_points > 200000:
        # Use a random sample of points if the point cloud is large
        sample_size = 200000
        sample_indices = np.random.choice(num_points, sample_size, replace=False)
        x_points = las.x[sample_indices]
        y_points = las.y[sample_indices]
        print(f"Using {sample_size} random points from {num_points} total")
    else:
        x_points = las.x
        y_points = las.y
    
    # Create MultiPoint and compute convex hull
    points = MultiPoint([(x, y) for x, y in zip(x_points, y_points)])
    hull = points.convex_hull
    
    # Apply simplification if requested
    if simplify_tolerance is not None:
        hull = hull.simplify(simplify_tolerance)
    
    # Apply buffer if requested
    if buffer_distance is not None:
        hull = hull.buffer(buffer_distance)
    
    # End timing
    end_time = time.time()
    execution_time = np.round(end_time - start_time, 1)
    
    # Check if the result is a valid polygon
    if isinstance(hull, Polygon):
        wkt = hull.wkt
        print(f"Footprint calculated in {execution_time} seconds. Number of vertices: {len(hull.exterior.coords)}")
        return wkt
    else:
        print(f"Warning: Generated footprint is not a polygon. Type: {type(hull)}")
        return hull.wkt


def create_dem(input_las, output_tif, dem_type='dtm', resolution=1.0, window_size=None, 
              create_xml=False):
    """
    Create a Digital Terrain Model (DTM) or Digital Surface Model (DSM) from a classified LAS file.
    
    Parameters:
        input_las (str): Path to the input classified LAS file
        output_tif (str): Path for the output GeoTIFF file
        dem_type (str): Type of Digital Elevation Model to create; either 'dtm' for terrain or 'dsm' for surface
        resolution (float): Output resolution in units of the LAS file (typically meters)
        window_size (int, optional): Window size for interpolation. If None, will be 3x resolution.
        create_xml (bool): If True, allows PDAL to create an XML metadata file alongside the GeoTIFF
                          Set to False to prevent creation of the additional XML file
    
    Returns:
        pdal.Pipeline: Configured PDAL pipeline object
    """
    # Set window size if not specified
    if window_size is None:
        window_size = int(3 * resolution)
    
    # Set up GDAL options
    gdalopts = "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256"
    
    # If XML creation should be suppressed
    if not create_xml:
        gdalopts += ",COPY_SRC_OVERVIEWS=YES,WRITE_METADATA=NO"
    
    # Define the pipeline based on DEM type
    if dem_type.lower() == 'dtm':
        # For DTM, use ground points (class 2) only
        pipeline_def = [
            {
                "type": "readers.las",
                "filename": input_las
            },
            {
                "type": "filters.range",
                "limits": "Classification[2:2]"  # Filter for ground points only
            },
            {
                "type": "writers.gdal",
                "filename": output_tif,
                "output_type": "idw",  # Inverse distance weighted interpolation
                "gdaldriver": "GTiff",
                "resolution": resolution,
                "window_size": window_size,
                "dimension": "Z",  # Use Z dimension for elevation
                "gdalopts": gdalopts
            }
        ]
    elif dem_type.lower() == 'dsm':
        # For DSM, use all points and take the maximum Z in each cell
        pipeline_def = [
            {
                "type": "readers.las",
                "filename": input_las
            },
            {
                "type": "filters.range",
                "limits": "ReturnNumber[1:1]"  # Use first returns for DSM
            },
            {
                "type": "writers.gdal",
                "filename": output_tif,
                "output_type": "max",  # Take highest point in each cell
                "gdaldriver": "GTiff",
                "resolution": resolution,
                "window_size": window_size,
                "dimension": "Z",  # Use Z dimension for elevation
                "gdalopts": gdalopts
            }
        ]
    else:
        raise ValueError("dem_type must be either 'dtm' or 'dsm'")
    
    # Create and return the pipeline
    pipeline_json = json.dumps({"pipeline": pipeline_def})
    return pdal.Pipeline(pipeline_json)

# PDAL Pipeline Functions

def process_and_classify_las(input_las, output_las, crop_polygon=None, min_hag=0, max_hag=25, filter_noise=False, target_crs="EPSG:32611"):
    """Process and classify LiDAR data with ground classification and height above ground calculation.

    This function creates a PDAL pipeline that:
    1. Reads the LAS file
    2. Reprojects to target CRS if needed
    3. Optionally crops to a specific polygon
    4. Optionally filters out noise points
    5. Classifies ground points using Simple Morphological Filter (SMRF)
    6. Calculates Height Above Ground (HAG) for each point
    7. Filters points by min/max HAG
    8. Writes the processed data to a new LAS file

    Determining which LiDAR returns are from the ground surface is essential for vegetation structure
    analysis. This method follows similar ground classification approaches to those used in
    the 3DEP program (Pingel et al. 2013).
    """
    pipeline_def = [
        {
            "type": "readers.las",
            "filename": input_las
        },
        {
            "type": "filters.reprojection",
            "out_srs": target_crs
        }
    ]

    if crop_polygon:
        pipeline_def.append({
            "type": "filters.crop",
            "polygon": crop_polygon
        })

    if filter_noise:
        pipeline_def.append({
            "type": "filters.range",
            "limits": "Classification![7:7], Classification![18:18]"
        })

    pipeline_def.extend([
        {
            "type": "filters.assign",
            "assignment": "Classification[:]=0"
        },
        {
            "type": "filters.smrf"
        },
        {
            "type": "filters.hag_delaunay"
        },
        {
            "type": "filters.range",
            "limits": f"HeightAboveGround[{min_hag}:{max_hag}]"
        },
        {
            "type": "writers.las",
            "filename": output_las,
            "compression": "laszip",
            "extra_dims":"all"            
        }
    ])

    pipeline_json = json.dumps({"pipeline": pipeline_def})
    return pdal.Pipeline(pipeline_json)


def process_and_classify_las_to_tif(input_las, output_tif, resolution=1, crop_polygon=None, min_hag=0, max_hag=25, filter_noise=False):
    """Create a PDAL pipeline for generating standard aggregate raster statistics from LiDAR data.
    
    This function creates a raster GeoTIFF using PDAL's writers.gdal with standard statistics:
    - Band 1: Minimum values
    - Band 2: Maximum values
    - Band 3: Mean values
    - Band 4: Inverse Distance Weighted (IDW) values
    - Band 5: Point count
    - Band 6: Standard deviation
    
    Note: This function is more limited than the custom aggregation approach but useful 
    for validation and basic statistics.
    """
    pipeline_def = [
        {
            "type": "readers.las",
            "filename": input_las
        }
    ]

    if crop_polygon:
        pipeline_def.append({
            "type": "filters.crop",
            "polygon": crop_polygon
        })

    if filter_noise:
        pipeline_def.append({
            "type": "filters.range",
            "limits": "Classification![7:7], Classification![18:18]"
        })

    pipeline_def.extend([
        {
            "type": "filters.assign",
            "assignment": "Classification[:]=0"
        },
        {
            "type": "filters.smrf"
        },
        {
            "type": "filters.hag_delaunay"
        },
        {
            "type": "filters.range",
            "limits": f"HeightAboveGround[{min_hag}:{max_hag}]"
        },
        {
        "type": "writers.gdal",
        "filename": output_tif,
        "gdaldriver": "GTiff",
        "dimension": "HeightAboveGround",
        "output_type": "all",
        "binmode": True,
        "resolution": resolution,
        "gdalopts": "COMPRESS=LZW,TILED=YES,blockxsize=256,blockysize=256,COPY_SRC_OVERVIEWS=YES"
        }
    ])    

    pipeline_json = json.dumps({"pipeline": pipeline_def})
    return pdal.Pipeline(pipeline_json)


# Main Aggregation Function

def agg_las_to_array(las_file_path, resolution, dimension="HeightAboveGround", aggregate_func=np.mean, *args, **kwargs):
    """
    Convert a LAS file to a numpy array representing an aggregated raster.
    
    This is the core function that takes a point cloud and creates a raster grid by
    aggregating points that fall within each cell. It allows for custom aggregation
    functions that can return either a single value (creating a 2D raster) or multiple
    values (creating a 3D raster).
    
    The function handles both simple statistics (min, max, mean) and complex metrics
    that may return multiple values per cell (percentiles, density proportions).
    
    Parameters:
        las_file_path: Path to the LAS file
        resolution: Spatial resolution of each cell in X and Y direction
        dimension: LAS point dimension to aggregate (default: "HeightAboveGround")
        aggregate_func: Function to apply to the dimension values
        *args, **kwargs: Additional arguments for aggregate_func
    
    Returns:
        numpy.ndarray: Aggregated raster (2D for single values, 3D for multiple values)
    """
    start_time = time.time()

    las = laspy.read(las_file_path)
    x_points = las.x
    y_points = las.y
    dimension_values = getattr(las, dimension)

    min_x, max_x = np.min(x_points), np.max(x_points)
    min_y, max_y = np.min(y_points), np.max(y_points)

    width = int(np.round((max_x - min_x) / resolution))
    height = int(np.round((max_y - min_y) / resolution))

    x_indices = ((x_points - min_x) / resolution).astype(int)
    y_indices = (height - (y_points - min_y) / resolution).astype(int)

    df = pd.DataFrame({'x_indices': x_indices, 'y_indices': y_indices, 'dimension_values': dimension_values})

    try:
        # For aggregate functions that return a single value per cell (like mean, max, etc.)
        # Group by cell indices and apply the aggregation function to points in each cell
        grouped_df = df.groupby(['x_indices', 'y_indices'])['dimension_values'].agg(
            lambda x: aggregate_func(x.to_numpy(), *args, **kwargs)).reset_index()
        
        # Create empty 2D raster of appropriate size
        raster = np.zeros((height, width))
        
        # Filter out cells that would be outside the raster bounds
        valid_cells = (grouped_df['x_indices'].between(0, width-1)) & (grouped_df['y_indices'].between(0, height-1))
        grouped_df = grouped_df[valid_cells]
        
        # Efficiently assign values to the raster using numpy indexing
        raster[grouped_df['y_indices'].values, grouped_df['x_indices'].values] = grouped_df['dimension_values'].values

    except ValueError as e:
        # Special handling for aggregate functions that return multiple values per cell
        if str(e) == 'Must produce aggregated value':
            # For functions like percentile that return multiple values (e.g., 10th, 25th, 50th percentiles)
            grouped_df = df.groupby(['x_indices', 'y_indices'])['dimension_values'].agg(
                lambda x: aggregate_func(x.to_numpy(), *args, **kwargs).tolist()).reset_index()
            
            # Determine how many values the aggregation function returns per cell
            n_agg_values = len(grouped_df['dimension_values'][0])
            
            # Create empty 3D raster to hold multiple values per cell
            raster = np.zeros((height, width, n_agg_values))

            # Filter out cells outside raster bounds
            valid_cells = (grouped_df['x_indices'].between(0, width-1)) & (grouped_df['y_indices'].between(0, height-1))
            grouped_df = grouped_df[valid_cells]
            
            # Assign values cell by cell (can't use efficient numpy indexing for 3D case)
            for i, row in grouped_df.iterrows():
                if len(row['dimension_values']) > n_agg_values:
                    print(f"Warning: More than {n_agg_values} aggregated values returned. Using first {n_agg_values}.")
                raster[row['y_indices'], row['x_indices']] = row['dimension_values'][0:n_agg_values]
        else:
            # Re-raise any other errors
            raise
    
    end_time = time.time()
    execution_time = np.round(end_time - start_time, 1) 
    n_points = "{:,}".format(np.size(x_points))
    filename = os.path.basename(las_file_path)
    agg_func_name = aggregate_func.__name__
    shape_str = "x".join(map(str, raster.shape))
    print(f"{filename} ({n_points} pts) aggregated via {agg_func_name}() to {shape_str} array. [{execution_time} seconds]")            
    
    return raster


def _aggregate_from_grouped(grouped_series, aggregate_func, raster_shape, *args, **kwargs):
    """
    Apply aggregation function to pre-grouped data and fill a raster.

    Memory-efficient helper that works with already-grouped pandas Series.
    Used internally by compute_vegetation_structure_metrics() to avoid re-reading
    the LAS file for each band.

    Parameters:
        grouped_series: pandas GroupBy object from DataFrame grouped by (y_indices, x_indices)
        aggregate_func: Function to apply to HAG values in each cell
        raster_shape: Tuple (height, width) for output raster
        *args, **kwargs: Additional arguments passed to aggregate_func

    Returns:
        numpy.ndarray: 2D raster (height, width) or 3D if aggregate_func returns multiple values
    """
    height, width = raster_shape

    try:
        # For aggregate functions that return a single value per cell
        grouped_df = grouped_series.agg(
            lambda x: aggregate_func(x.to_numpy(), *args, **kwargs)
        ).reset_index()
        grouped_df.columns = ['y_indices', 'x_indices', 'value']

        # Create empty 2D raster
        raster = np.zeros((height, width), dtype=np.float32)

        # Filter out cells outside raster bounds
        valid_cells = (grouped_df['x_indices'].between(0, width-1)) & \
                      (grouped_df['y_indices'].between(0, height-1))
        grouped_df = grouped_df[valid_cells]

        # Assign values to raster
        raster[grouped_df['y_indices'].values, grouped_df['x_indices'].values] = \
            grouped_df['value'].values.astype(np.float32)

    except ValueError as e:
        if str(e) == 'Must produce aggregated value':
            # For functions that return multiple values (percentiles, density proportions)
            grouped_df = grouped_series.agg(
                lambda x: aggregate_func(x.to_numpy(), *args, **kwargs).tolist()
            ).reset_index()
            grouped_df.columns = ['y_indices', 'x_indices', 'value']

            n_agg_values = len(grouped_df['value'].iloc[0])
            raster = np.zeros((height, width, n_agg_values), dtype=np.float32)

            valid_cells = (grouped_df['x_indices'].between(0, width-1)) & \
                          (grouped_df['y_indices'].between(0, height-1))
            grouped_df = grouped_df[valid_cells]

            for _, row in grouped_df.iterrows():
                raster[row['y_indices'], row['x_indices']] = row['value'][:n_agg_values]
        else:
            raise

    return raster


def plot_georeferenced_rasters_with_geometries(tiff_files, geometries=None, raster_alphas=None,
                                            raster_colormaps=None, raster_labels=None,
                                            geometry_colors=None, geometry_labels=None,
                                            figsize=(12, 12), alpha=0.5, title=None):
    """
    Load and plot multiple georeferenced GeoTIFF files on a single map with correct spatial positioning.
    
    Parameters:
        tiff_files (list): List of paths to GeoTIFF files
        geometries (str or list, optional): WKT string(s) for areas of interest
        raster_alphas (list, optional): Alpha values for each raster
        raster_colormaps (list, optional): Colormap names for each raster
        raster_labels (list, optional): Labels for each raster
        geometry_colors (str or list): Color(s) for geometry overlays
        geometry_labels (str or list): Label(s) for geometries
        figsize (tuple): Figure size in inches
        alpha (float): Transparency for geometry overlays
        title (str): Title for the plot
        
    Returns:
        tuple: (fig, ax) - The figure and axes objects
    """
    # Convert inputs to lists if they're not already
    if not isinstance(tiff_files, list):
        tiff_files = [tiff_files]
    
    if geometries is not None and not isinstance(geometries, list):
        geometries = [geometries]
    
    if raster_labels is not None and not isinstance(raster_labels, list):
        raster_labels = [raster_labels]
    
    if geometry_colors is not None and not isinstance(geometry_colors, list):
        geometry_colors = [geometry_colors]
    
    if geometry_labels is not None and not isinstance(geometry_labels, list):
        geometry_labels = [geometry_labels]
        
    if raster_colormaps is not None and not isinstance(raster_colormaps, list):
        raster_colormaps = [raster_colormaps]
        
    if raster_alphas is not None and not isinstance(raster_alphas, list):
        raster_alphas = [raster_alphas]
    
    # Set default alpha values
    if raster_alphas is None:
        raster_alphas = [1.0] + [max(0.2, 1.0 - (i * 0.15)) for i in range(1, len(tiff_files))]
        
    # Set default colormaps if not provided
    if raster_colormaps is None:
        default_cmaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'terrain']
        raster_colormaps = [default_cmaps[i % len(default_cmaps)] for i in range(len(tiff_files))]
        
    # Set default colors for geometries if not provided
    if geometries is not None and geometry_colors is None:
        default_colors = ['red', 'blue', 'green', 'purple', 'orange', 'teal']
        geometry_colors = [default_colors[i % len(default_colors)] for i in range(len(geometries))]
    
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)
    
    # Store handles for the legend
    handles = []
    
    # Open and plot each raster
    for i, tiff_file in enumerate(tiff_files):
        with rasterio.open(tiff_file) as src:
            # Read the data and mask out no data values
            raster = src.read(1)
            
            # Determine the no data value
            nodata = src.nodata
            if nodata is not None:
                # Create a masked array to handle no data values
                raster = np.ma.masked_where(raster == nodata, raster)
            else:
                # If no explicit nodata value, mask very negative values
                raster = np.ma.masked_where(raster < -1000, raster)
            
            # Get the raster bounds
            bounds = src.bounds
            extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]
            
            # Get the colormap
            cmap_name = raster_colormaps[i % len(raster_colormaps)]
            cmap = plt.cm.get_cmap(cmap_name)
            
            # Plot the raster
            im = ax.imshow(raster, 
                         extent=extent,  # This is key for georeferencing
                         cmap=cmap, 
                         alpha=raster_alphas[i],
                         origin='upper')  # Raster origin is usually upper left
            
            # Add a colorbar
            cbar = plt.colorbar(im, ax=ax, shrink=0.7)
            if raster_labels is not None and i < len(raster_labels):
                cbar.set_label(raster_labels[i])
            
            # Add to legend
            if raster_labels is not None and i < len(raster_labels):
                from matplotlib.patches import Patch
                patch = Patch(color=cmap(0.7), alpha=raster_alphas[i], label=raster_labels[i])
                handles.append(patch)
    
    # Plot geometries if provided
    if geometries is not None:
        for i, wkt_string in enumerate(geometries):
            # Parse the WKT string to a shapely geometry
            geom = wkt.loads(wkt_string)
            
            # Set color for this geometry
            color = geometry_colors[i % len(geometry_colors)]
            
            # Plot the geometry based on its type
            if isinstance(geom, Polygon):
                # Extract exterior coordinates
                x, y = geom.exterior.xy
                poly = MplPolygon(np.column_stack([x, y]), 
                                 facecolor=color, 
                                 alpha=alpha,
                                 edgecolor='black', 
                                 label=geometry_labels[i] if geometry_labels else None)
                ax.add_patch(poly)
                if geometry_labels and i < len(geometry_labels):
                    handles.append(poly)
            
            elif isinstance(geom, LineString):
                x, y = geom.xy
                line, = ax.plot(x, y, color=color, linewidth=2, 
                              label=geometry_labels[i] if geometry_labels else None)
                if geometry_labels and i < len(geometry_labels):
                    handles.append(line)
            
            elif isinstance(geom, Point):
                x, y = geom.x, geom.y
                point, = ax.plot(x, y, 'o', color=color, markersize=8,
                               label=geometry_labels[i] if geometry_labels else None)
                if geometry_labels and i < len(geometry_labels):
                    handles.append(point)
    
    # Add legend if we have handles
    if handles:
        ax.legend(handles=handles, loc='upper right')
    
    # Add title if provided
    if title:
        ax.set_title(title)
    
    # Add gridlines
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Add north arrow and scale bar (optional)
    # This is a simple north arrow - you might want to use a more sophisticated approach
    from matplotlib.patches import Arrow
    arrow_pos = (0.95, 0.05)
    arrow_length = 0.05
    ax.add_patch(Arrow(arrow_pos[0], arrow_pos[1], 0, arrow_length, 
                      width=0.03, transform=ax.transAxes, 
                      facecolor='black', edgecolor='black'))
    ax.text(arrow_pos[0], arrow_pos[1] + arrow_length + 0.01, 'N', 
           transform=ax.transAxes, ha='center')
    
    plt.tight_layout()
    return fig, ax







def visualize_pointclouds(point_clouds, elev=45, azim=45):
    """
    Visualize a list of point clouds in a 3D scatter plot with adjustable viewing angles.

    :param point_clouds: list of NumPy structured arrays, each containing point data (X, Y, Z, etc.).
    :param elev: elevation angle in degrees (default=30).
    :param azim: azimuth angle in degrees (default=45).
    """
    if not point_clouds:
        print("No point clouds to visualize.")
        return

    # Get the common fields across all point clouds
    common_fields = set(point_clouds[0].dtype.names)
    for pc in point_clouds:
        common_fields.intersection_update(pc.dtype.names)

    # Normalize all arrays to have only the common fields
    normalized_point_clouds = []
    for pc in point_clouds:
        # Create a new array with only the common fields
        normalized_pc = np.zeros(pc.shape, dtype=[(field, pc.dtype[field]) for field in common_fields])
        for field in common_fields:
            normalized_pc[field] = pc[field]
        normalized_point_clouds.append(normalized_pc)

    # Concatenate all normalized point clouds into one array
    all_points = np.concatenate(normalized_point_clouds)

    # Extract X, Y, Z
    x = all_points['X']
    y = all_points['Y']
    z = all_points['Z']

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot
    sc = ax.scatter(x, y, z, c=z, s=0.1, cmap='viridis')

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the viewing angle
    ax.view_init(elev=elev, azim=azim)

    # Optionally fix aspect ratio so the axes scales are consistent:
    ax.set_box_aspect((np.ptp(x), np.ptp(y), np.ptp(z)))

    plt.title("Point Cloud Visualization")
    plt.show()



    

def stack_rasters(*rasters):
    """Stack multiple 2D rasters into a 3D array for plotting with earthpy."""
    stacked_image = np.stack(rasters, axis=2)
    return np.transpose(stacked_image, (2, 0, 1))


def plot_rasters(rasters, figsize=(12, 12), titles=None):
    """Plot multiple rasters side by side."""
    ncols = rasters.shape[0]
    ep.plot_bands(
        rasters,
        cols=ncols,
        figsize=figsize,
        cmap="viridis",
        title=titles
    )


def stack_and_plot_rasters(*rasters, titles=None, figsize=(12, 12)):
    """Stack and plot multiple 2D rasters side by side."""
    stacked_rasters = stack_rasters(*rasters)
    plot_rasters(stacked_rasters, titles=titles, figsize=figsize)


def compare_two_rasters(rstr1, rstr2, rstr1_name, rstr2_name, metric_str, 
                       remove_outliers=True, figsize=(12, 12)):
    """Compare two rasters by plotting them with their difference.
    
    Displays three plots side by side:
    1. First raster
    2. Second raster
    3. Difference between them
    
    When remove_outliers=True, extreme values in the difference plot are capped
    at the 95th percentile to make patterns more visible.
    """
    title1 = f"{rstr1_name} {metric_str}"
    title2 = f"{rstr2_name} {metric_str}"
    title3 = f"{rstr1_name}-{rstr2_name} {metric_str}"

    error_rstr = (rstr1 - rstr2)
    if remove_outliers:
        threshold = np.percentile(np.abs(error_rstr), 95)  
        error_rstr[(error_rstr < -threshold)] = -threshold
        error_rstr[(error_rstr > threshold)] = threshold

    stack_and_plot_rasters(rstr1, rstr2, error_rstr, titles=[title1, title2, title3], figsize=figsize)


def create_pctile_plot_titles(percentile):
    """Create titles for percentile plots."""
    return [f"{x:.0f}%" for x in percentile]


def create_density_plot_titles(density_lvl_minmax_hag, density_layers):
    """Create titles for density proportion plots."""
    plot_titles = []
    for i in range(density_layers):
        min_val = density_lvl_minmax_hag[0] + (i * (density_lvl_minmax_hag[1] - density_lvl_minmax_hag[0]) / density_layers)
        max_val = density_lvl_minmax_hag[0] + ((i+1) * (density_lvl_minmax_hag[1] - density_lvl_minmax_hag[0]) / density_layers)
        title = f" {min_val:.0f}-{max_val:.0f}m"
        plot_titles.append(title)
    return plot_titles


def compare_two_n_dim_rasters(rstr1, rstr2, rstr1_name, rstr2_name, plot_title_function, *args, **kwargs):
    """Compare two 3D rasters by plotting each dimension with differences."""
    t_rstr1 = np.transpose(rstr1, (2, 0, 1))
    t_rstr2 = np.transpose(rstr2, (2, 0, 1))
    error_rstr = (t_rstr1 - t_rstr2)

    threshold = np.percentile(np.abs(error_rstr), 95)  
    error_rstr[(error_rstr < -threshold)] = -threshold
    error_rstr[(error_rstr > threshold)] = threshold

    ncols = t_rstr1.shape[0]
    titles1 = [rstr1_name + title for title in plot_title_function(*args, **kwargs)]
    titles2 = [rstr2_name + title for title in plot_title_function(*args, **kwargs)]
    titles3 = [f"{rstr1_name}-{rstr2_name}" + title for title in plot_title_function(*args, **kwargs)]
    figsize = (20, 10)
    
    ep.plot_bands(t_rstr1, figsize=figsize, cols=ncols, cmap="viridis", title=titles1)
    ep.plot_bands(t_rstr2, figsize=figsize, cols=ncols, cmap="viridis", title=titles2)
    ep.plot_bands(error_rstr, figsize=figsize, cols=ncols, cmap="PiYG", title=titles3)


# Vegetation Structure Metric Functions
# The following functions implement the vegetation structure metrics proposed by Moudry et al. (2023)
# These metrics help characterize vegetation structure in terms of height, cover, density, 
# and vertical complexity

def point_count(point_array):
    """Count the number of points in an array.
    
    Used as an auxiliary function to count points in a cell.
    """
    return point_array.size


def canopy_density(hag_array, canopy_min_hag):
    """Calculate proportion of vegetation points in the canopy layer.
    
    Measures the amount of vegetation in the tree/canopy layer.
    A value of 0.65 means that 65% of all vegetation returns came from trees.
    
    Calculation: Number of returns at the top vegetation layer divided by 
    the total number of vegetation returns.
    """
    n_canopy_points = hag_array[(hag_array > canopy_min_hag)].size
    n_veg_points = hag_array[(hag_array > 0.1)].size
    return 0 if n_veg_points == 0 else n_canopy_points / n_veg_points


def mid_story_density(hag_array, understory_max_hag, canopy_min_hag):
    """Calculate proportion of vegetation points in the mid-story layer.
    
    Measures the amount of vegetation in the shrub/mid-story layer.
    A value of 0.25 means that 25% of all vegetation returns came from shrub vegetation.
    
    Calculation: Number of returns at the middle vegetation layer divided by
    the total number of vegetation returns.
    """
    n_mid_story_points = hag_array[(hag_array >= understory_max_hag) & (hag_array < canopy_min_hag)].size
    n_veg_points = hag_array[(hag_array > 0.1)].size
    return 0 if n_veg_points == 0 else n_mid_story_points / n_veg_points


def under_story_density(hag_array, understory_max_hag):
    """Calculate proportion of vegetation points in the understory layer.
    
    Measures the amount of vegetation in the herbaceous/understory layer.
    A value of 0.10 means that 10% of all vegetation returns came from herbaceous vegetation.
    
    Calculation: Number of returns at the lowest vegetation layer divided by
    the total number of vegetation returns.
    """
    n_under_story_points = hag_array[(hag_array <= understory_max_hag) & (hag_array > 0.1)].size
    n_veg_points = hag_array[(hag_array > 0.1)].size
    return 0 if n_veg_points == 0 else n_under_story_points / n_veg_points


def canopy_cover(hag_array, canopy_min_hag):
    """Calculate proportion of all points in the canopy layer.
    
    Measures the extent/percentage of the ground covered by vegetation.
    A value of 0.85 means that 85% of returns were reflected above the height threshold.
    The higher the value, the denser the canopy (closed stands).
    Low values reflect open or scattered stands.
    
    Calculation: Number of returns above a given height cutoff divided by
    the total number of returns.
    """
    n_canopy_points = hag_array[(hag_array > canopy_min_hag)].size
    n_points = hag_array.size
    return 0 if n_points == 0 else n_canopy_points / n_points


def density_proportions(hag_array, hag_rng, num_layers):
    """Calculate proportion of points in each vertical layer.
    
    Measures the vertical distribution of points (vegetation architecture).
    Creates fixed height bins between the minimum and maximum height and
    calculates the proportion of returns in each bin.
    
    Calculation: For each height bin, divide the number of returns in that bin
    by the total number of returns.
    
    Returns:
        np.array: Array of length num_layers containing proportions for each layer
    """
    bins = np.linspace(hag_rng[0], hag_rng[1], num_layers+1)
    layer_indices = np.digitize(hag_array, bins) - 1
    counts = np.bincount(layer_indices, minlength=num_layers)[0:num_layers]
    total_counts = np.sum(counts)
    return np.zeros(num_layers) if total_counts == 0 else counts / total_counts


def foliage_height_diversity(hag_array, hag_rng, num_layers):
    """Calculate Foliage Height Diversity using Shannon-Wiener index.

    A measure of canopy layering complexity (MacArthur & MacArthur, 1961).
    The maximum possible value increases with the number of layers.
    The maximum value occurs when all layers have the same number of returns
    (i.e., the Shannon-Wiener index increases with a more even distribution
    of points over the layers).

    Calculation: FHD = -∑(p_i * ln(p_i))
    where p_i is the proportion of returns in each vertical layer i,
    and n is the total number of layers.
    """
    proportions = density_proportions(hag_array, hag_rng, num_layers)
    mask = proportions > 0
    if np.sum(mask) == 0:
        return 0
    return -np.sum(proportions[mask] * np.log(proportions[mask]))


def compute_vegetation_structure_metrics(
    las_file_path: str,
    resolution: float = 2.0,
    canopy_min_hag: float = 3.0,
    understory_max_hag: float = 1.0,
    point_filter_min_hag: float = 0.0,
    point_filter_max_hag: float = 60.0,
    density_range: tuple = (0, 25),
    num_density_layers: int = 10,
    min_points_per_pixel: int = None,
    percentiles: list = None,
    preprocess: bool = True,
    temp_dir: str = None,
    target_crs: str = "EPSG:32611"
) -> tuple:
    """
    Compute all Moudry et al. (2023) vegetation structure metrics from a LAS file.

    This function computes a comprehensive set of vegetation structure metrics:
    - Basic statistics: max, mean, std of height
    - Cover/density metrics: canopy cover, canopy/mid-story/understory density
    - Complexity metrics: Foliage Height Diversity (Shannon-Wiener)
    - Height percentiles: configurable quantiles (default: 10, 25, 50, 75, 90)
    - Vertical structure: density proportions across height layers

    Parameters:
        las_file_path: Path to the input LAS/LAZ file
        resolution: Spatial resolution of output raster in same units as LAS (default: 2.0m)
        canopy_min_hag: Minimum height above ground for canopy layer (default: 3.0m)
        understory_max_hag: Maximum height for understory layer (default: 1.0m)
        point_filter_min_hag: Minimum HAG for point filtering during preprocessing (default: 0.0m)
        point_filter_max_hag: Maximum HAG for point filtering during preprocessing (default: 60.0m)
                               Site-specific: use 25m for shorter vegetation, 60m for tall forests
        density_range: Tuple of (min, max) height for density binning in metrics (default: (0, 25))
                       FIXED across sites for comparability
        num_density_layers: Number of vertical layers for density proportions (default: 10)
        min_points_per_pixel: Minimum points required per pixel to compute metrics (default: None)
                              If None, auto-calculate as 20% of median pixel point count
                              Pixels below threshold are set to NaN across all bands
        percentiles: List of percentile values to compute (default: [10, 25, 50, 75, 90])
        preprocess: If True, run SMRF ground classification + HAG if not present (default: True)
        temp_dir: Directory for temporary files during preprocessing (default: same as input)
        target_crs: Target coordinate reference system for output (default: "EPSG:32611")

    Returns:
        tuple: (raster, metadata)
            - raster: np.ndarray of shape [n_bands, height, width], dtype=float32
            - metadata: dict with keys:
                - 'band_names': list of band name strings
                - 'band_descriptions': dict mapping band index to description
                - 'crs': coordinate reference system (pyproj CRS object or None)
                - 'transform': rasterio Affine transform
                - 'bounds': tuple (minx, miny, maxx, maxy)
                - 'resolution': float, pixel size
                - 'n_bands': int, total number of bands
                - 'parameters': dict of function parameters used
                - 'min_points_per_pixel_threshold': int, actual threshold value used

    Raises:
        FileNotFoundError: If las_file_path does not exist
        ValueError: If LAS file has no HeightAboveGround dimension and preprocess=False

    Example:
        >>> raster, metadata = compute_vegetation_structure_metrics(
        ...     'data/raw/uavlidar/site.laz',
        ...     resolution=2.0,
        ...     preprocess=True
        ... )
        >>> print(f"Output shape: {raster.shape}")  # [23, height, width]
        >>> print(f"Bands: {metadata['band_names']}")
    """
    import tempfile
    from pathlib import Path

    # Default percentiles
    if percentiles is None:
        percentiles = [10, 25, 50, 75, 90]

    # Validate input file exists
    las_path = Path(las_file_path)
    if not las_path.exists():
        raise FileNotFoundError(f"LAS file not found: {las_file_path}")

    # Check if preprocessing is needed
    working_las = las_file_path
    temp_file = None

    # Check for HeightAboveGround dimension
    las = laspy.read(las_file_path)
    has_hag = 'HeightAboveGround' in [dim.name for dim in las.point_format.dimensions]

    if not has_hag:
        if not preprocess:
            raise ValueError(
                f"LAS file '{las_file_path}' does not have HeightAboveGround dimension. "
                "Set preprocess=True to compute it automatically."
            )

        print(f"HeightAboveGround not found. Running preprocessing (SMRF + HAG)...")

        # Create temp file for preprocessed output
        temp_dir_path = Path(temp_dir) if temp_dir else las_path.parent
        temp_file = tempfile.NamedTemporaryFile(
            suffix='.laz',
            dir=temp_dir_path,
            delete=False
        )
        temp_file.close()
        working_las = temp_file.name

        # Run preprocessing pipeline
        pipeline = process_and_classify_las(
            input_las=las_file_path,
            output_las=working_las,
            min_hag=point_filter_min_hag,
            max_hag=point_filter_max_hag,
            filter_noise=False,
            target_crs=target_crs
        )
        pipeline.execute()
        print(f"Preprocessing complete. Temporary file: {working_las}")

    # Close the laspy file handle
    del las

    try:
        # ========================================
        # MEMORY-EFFICIENT: Load LAS file once and prepare grouped data
        # ========================================

        print(f"Loading point cloud data (single load, float32)...")
        load_start = time.time()

        las = laspy.read(working_las)
        n_points = len(las.x)
        print(f"  Loaded {n_points:,} points")

        # Extract coordinates and HAG as float32 to save memory
        x_points = np.asarray(las.x, dtype=np.float32)
        y_points = np.asarray(las.y, dtype=np.float32)
        hag_values = np.asarray(las.HeightAboveGround, dtype=np.float32)

        # Free the laspy object - we've extracted what we need
        del las
        gc.collect()

        # Compute raster dimensions and pixel indices
        min_x, max_x = float(np.min(x_points)), float(np.max(x_points))
        min_y, max_y = float(np.min(y_points)), float(np.max(y_points))

        width = int(np.round((max_x - min_x) / resolution))
        height = int(np.round((max_y - min_y) / resolution))
        raster_shape = (height, width)

        print(f"  Raster dimensions: {width} x {height} pixels at {resolution}m resolution")

        # Compute pixel indices (int32 is sufficient for indices)
        x_indices = ((x_points - min_x) / resolution).astype(np.int32)
        y_indices = (height - (y_points - min_y) / resolution).astype(np.int32)

        # Free coordinate arrays - no longer needed
        del x_points, y_points
        gc.collect()

        # Create DataFrame with pixel indices and HAG values
        # Using float32 for HAG reduces memory by 50%
        print(f"  Creating grouped data structure...")
        df = pd.DataFrame({
            'y_indices': y_indices,
            'x_indices': x_indices,
            'hag': hag_values
        })

        # Free the original arrays - DataFrame has its own copy
        del x_indices, y_indices, hag_values
        gc.collect()

        # Create the grouped object once - this is reused for all band computations
        grouped = df.groupby(['y_indices', 'x_indices'])['hag']

        load_time = time.time() - load_start
        print(f"  Data preparation complete [{load_time:.1f}s]")

        # ========================================
        # Compute all metrics using pre-loaded grouped data
        # ========================================

        raster_bands = []
        band_names = []
        band_descriptions = {}

        print(f"Computing vegetation structure metrics at {resolution}m resolution...")

        # --- Band 0: Maximum height ---
        print("  [0/24] Computing max height...")
        max_height = _aggregate_from_grouped(grouped, np.max, raster_shape)
        raster_bands.append(max_height)
        band_names.append('max_height')
        band_descriptions[0] = MOUDRY_BAND_DESCRIPTIONS[0]
        del max_height; gc.collect()

        # --- Band 1: Mean height ---
        print("  [1/24] Computing mean height...")
        mean_height = _aggregate_from_grouped(grouped, np.mean, raster_shape)
        raster_bands.append(mean_height)
        band_names.append('mean_height')
        band_descriptions[1] = MOUDRY_BAND_DESCRIPTIONS[1]
        del mean_height; gc.collect()

        # --- Band 2: Std height ---
        print("  [2/24] Computing std height...")
        std_height = _aggregate_from_grouped(grouped, np.std, raster_shape)
        raster_bands.append(std_height)
        band_names.append('std_height')
        band_descriptions[2] = MOUDRY_BAND_DESCRIPTIONS[2]
        del std_height; gc.collect()

        # --- Band 3: Canopy cover ---
        print("  [3/24] Computing canopy cover...")
        canopy_cov = _aggregate_from_grouped(
            grouped, canopy_cover, raster_shape,
            canopy_min_hag=canopy_min_hag
        )
        raster_bands.append(canopy_cov)
        band_names.append('canopy_cover')
        band_descriptions[3] = MOUDRY_BAND_DESCRIPTIONS[3]
        del canopy_cov; gc.collect()

        # --- Band 4: Canopy density ---
        print("  [4/24] Computing canopy density...")
        canopy_dens = _aggregate_from_grouped(
            grouped, canopy_density, raster_shape,
            canopy_min_hag=canopy_min_hag
        )
        raster_bands.append(canopy_dens)
        band_names.append('canopy_density')
        band_descriptions[4] = MOUDRY_BAND_DESCRIPTIONS[4]
        del canopy_dens; gc.collect()

        # --- Band 5: Mid-story density ---
        print("  [5/24] Computing mid-story density...")
        mid_dens = _aggregate_from_grouped(
            grouped, mid_story_density, raster_shape,
            understory_max_hag=understory_max_hag,
            canopy_min_hag=canopy_min_hag
        )
        raster_bands.append(mid_dens)
        band_names.append('mid_story_density')
        band_descriptions[5] = MOUDRY_BAND_DESCRIPTIONS[5]
        del mid_dens; gc.collect()

        # --- Band 6: Understory density ---
        print("  [6/24] Computing understory density...")
        under_dens = _aggregate_from_grouped(
            grouped, under_story_density, raster_shape,
            understory_max_hag=understory_max_hag
        )
        raster_bands.append(under_dens)
        band_names.append('understory_density')
        band_descriptions[6] = MOUDRY_BAND_DESCRIPTIONS[6]
        del under_dens; gc.collect()

        # --- Band 7: Foliage Height Diversity ---
        print("  [7/24] Computing foliage height diversity...")
        fhd = _aggregate_from_grouped(
            grouped, foliage_height_diversity, raster_shape,
            hag_rng=density_range,
            num_layers=num_density_layers
        )
        raster_bands.append(fhd)
        band_names.append('foliage_height_diversity')
        band_descriptions[7] = MOUDRY_BAND_DESCRIPTIONS[7]
        del fhd; gc.collect()

        # --- Bands 8-12: Height percentiles ---
        print(f"  [8-{7+len(percentiles)}/24] Computing height percentiles {percentiles}...")
        height_pctiles = _aggregate_from_grouped(
            grouped, np.percentile, raster_shape,
            q=percentiles
        )
        # height_pctiles shape is [height, width, n_percentiles]
        for i, pct in enumerate(percentiles):
            band_idx = 8 + i
            raster_bands.append(height_pctiles[:, :, i].copy())
            band_names.append(f'height_p{pct}')
            band_descriptions[band_idx] = {
                'name': f'height_p{pct}',
                'description': f'Height {pct}th percentile (m)',
                'unit': 'm'
            }
        del height_pctiles; gc.collect()

        # --- Bands 13-22: Density proportions ---
        first_density_band = 8 + len(percentiles)
        last_density_band = first_density_band + num_density_layers - 1
        print(f"  [{first_density_band}-{last_density_band}/24] Computing density proportions ({num_density_layers} layers)...")

        density_props = _aggregate_from_grouped(
            grouped, density_proportions, raster_shape,
            hag_rng=density_range,
            num_layers=num_density_layers
        )
        # density_props shape is [height, width, num_layers]
        layer_height = (density_range[1] - density_range[0]) / num_density_layers
        for i in range(num_density_layers):
            band_idx = first_density_band + i
            raster_bands.append(density_props[:, :, i].copy())
            layer_min = density_range[0] + i * layer_height
            layer_max = density_range[0] + (i + 1) * layer_height
            band_names.append(f'density_layer_{i}')
            band_descriptions[band_idx] = {
                'name': f'density_layer_{i}',
                'description': f'Density proportion {layer_min:.1f}-{layer_max:.1f}m',
                'unit': 'fraction'
            }
        del density_props; gc.collect()

        # --- Band 23: Point count ---
        print("  [23/24] Computing point count...")
        point_cnt_raster = _aggregate_from_grouped(grouped, point_count, raster_shape)
        raster_bands.append(point_cnt_raster)
        band_names.append('point_count')
        band_descriptions[23] = {
            'name': 'point_count',
            'description': 'Number of points per pixel',
            'unit': 'count'
        }

        # Free the grouped data and DataFrame - no longer needed
        del grouped, df, point_cnt_raster
        gc.collect()

        # ========================================
        # Stack into multi-band array
        # ========================================

        print("  [24/24] Stacking bands...")
        # Stack all bands: shape becomes [n_bands, height, width]
        # Already float32 from _aggregate_from_grouped
        raster = np.stack(raster_bands, axis=0)

        # Free individual band arrays
        del raster_bands
        gc.collect()

        # ========================================
        # Apply minimum point count filtering
        # ========================================

        # Extract point count band (Band 23, last band)
        point_count_band = raster[-1, :, :]

        # Calculate threshold
        if min_points_per_pixel is None:
            # Auto-calculate: 20% of median (excluding empty cells)
            non_zero_counts = point_count_band[point_count_band > 0]

            if non_zero_counts.size == 0:
                raise ValueError(
                    "No pixels contain points. Cannot compute vegetation metrics. "
                    "Check that the LAS file contains valid point data."
                )

            median_count = np.median(non_zero_counts)
            threshold = max(1, int(np.round(median_count * 0.20)))

            print(f"\nApplying minimum point count filter:")
            print(f"  Median points per non-empty pixel: {median_count:.1f}")
            print(f"  Using threshold: {threshold} (20% of median)")
        else:
            # User-provided threshold
            threshold = min_points_per_pixel

            if threshold < 1:
                raise ValueError(
                    f"min_points_per_pixel must be >= 1, got {threshold}"
                )

            print(f"\nApplying minimum point count filter:")
            print(f"  Using threshold: {threshold} (user-provided)")

        # Create mask for pixels below threshold
        low_count_mask = point_count_band < threshold

        # Count affected pixels
        n_total_pixels = point_count_band.size
        n_empty = np.sum(point_count_band == 0)
        n_non_empty = n_total_pixels - n_empty
        n_low_but_not_empty = np.sum((point_count_band > 0) & (point_count_band < threshold))
        n_pass_threshold = np.sum(point_count_band >= threshold)

        # Calculate percentages
        pct_empty = (n_empty / n_total_pixels) * 100
        pct_filtered = (n_low_but_not_empty / n_non_empty) * 100 if n_non_empty > 0 else 0
        pct_pass = (n_pass_threshold / n_non_empty) * 100 if n_non_empty > 0 else 0

        print(f"  Total pixels: {n_total_pixels:,}")
        print(f"  Empty pixels (0 points): {n_empty:,} ({pct_empty:.1f}% of total)")
        print(f"  Non-empty pixels: {n_non_empty:,}")
        print(f"    - Filtered (>0 but <{threshold} points): {n_low_but_not_empty:,} ({pct_filtered:.1f}% of non-empty)")
        print(f"    - Pass threshold (≥{threshold} points): {n_pass_threshold:,} ({pct_pass:.1f}% of non-empty)")

        # Apply mask to bands 0-22 (NOT Band 23 - preserve actual counts)
        raster[:-1, low_count_mask] = np.nan

        # ========================================
        # Build metadata
        # ========================================

        # Re-read LAS to get CRS and bounds
        las = laspy.read(working_las)

        # Get CRS from LAS (required for geospatial output)
        crs = None
        try:
            from pyproj import CRS
            if hasattr(las.header, 'vlrs'):
                for vlr in las.header.vlrs:
                    if vlr.record_id == 2112:  # WKT CRS
                        crs = CRS.from_wkt(vlr.string)
                        break
        except Exception as e:
            raise ValueError(
                f"Failed to extract CRS from LAS file '{las_file_path}'. "
                f"Error: {e}"
            )

        # Validate CRS was extracted
        if crs is None:
            raise ValueError(
                f"Could not extract CRS from LAS file '{las_file_path}'. "
                f"No WKT CRS VLR (record_id 2112) found."
            )

        # Validate CRS matches expected target_crs
        actual_epsg = crs.to_epsg()
        expected_crs = CRS.from_string(target_crs)
        expected_epsg = expected_crs.to_epsg()
        if actual_epsg != expected_epsg:
            raise ValueError(
                f"CRS mismatch in '{las_file_path}'. "
                f"Expected: {target_crs} (EPSG:{expected_epsg}), Got: EPSG:{actual_epsg}. "
                f"Reprojection should have been applied during preprocessing."
            )

        # Compute bounds and transform
        min_x, max_x = np.min(las.x), np.max(las.x)
        min_y, max_y = np.min(las.y), np.max(las.y)

        # Create rasterio-compatible Affine transform
        # Note: raster origin is top-left, y increases downward
        from rasterio.transform import from_bounds
        height_px, width_px = raster.shape[1], raster.shape[2]

        # Snap max bounds to resolution grid to ensure exact pixel size
        # This prevents non-integer pixel sizes like 2.004m instead of 2.0m
        max_x_snapped = min_x + width_px * resolution
        max_y_snapped = min_y + height_px * resolution

        transform = from_bounds(min_x, min_y, max_x_snapped, max_y_snapped, width_px, height_px)

        metadata = {
            'band_names': band_names,
            'band_descriptions': band_descriptions,
            'crs': crs,
            'transform': transform,
            'bounds': (min_x, min_y, max_x_snapped, max_y_snapped),  # Use snapped bounds
            'resolution': resolution,
            'n_bands': len(band_names),
            'parameters': {
                'canopy_min_hag': canopy_min_hag,
                'understory_max_hag': understory_max_hag,
                'point_filter_min_hag': point_filter_min_hag,
                'point_filter_max_hag': point_filter_max_hag,
                'density_range': density_range,
                'num_density_layers': num_density_layers,
                'percentiles': percentiles,
                'target_crs': target_crs,
                'min_points_per_pixel_threshold': threshold
            }
        }

        print(f"\nCompleted: {raster.shape[0]} bands, {raster.shape[1]}x{raster.shape[2]} pixels")

        return raster, metadata

    finally:
        # Clean up temporary file if created
        if temp_file is not None:
            import os
            try:
                os.unlink(temp_file.name)
                print(f"Cleaned up temporary file: {temp_file.name}")
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_file.name}: {e}")


def save_metrics_to_geotiff(
    raster: np.ndarray,
    metadata: dict,
    output_path: str,
    compress: bool = True
) -> None:
    """
    Save vegetation structure metrics raster to a GeoTIFF file.

    Parameters:
        raster: np.ndarray of shape [n_bands, height, width] from compute_vegetation_structure_metrics()
        metadata: dict from compute_vegetation_structure_metrics()
        output_path: Path for output GeoTIFF file
        compress: If True, use LZW compression (default: True)

    Example:
        >>> raster, metadata = compute_vegetation_structure_metrics('site.laz', resolution=2.0)
        >>> save_metrics_to_geotiff(raster, metadata, 'site_metrics.tif')
    """
    from pathlib import Path

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate CRS is present
    if metadata['crs'] is None:
        raise ValueError(
            "Cannot save GeoTIFF without CRS. "
            "Ensure input LAS file has valid CRS metadata."
        )

    # Build rasterio profile
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': raster.shape[2],
        'height': raster.shape[1],
        'count': raster.shape[0],
        'crs': metadata['crs'],
        'transform': metadata['transform'],
        'nodata': np.nan,
    }

    if compress:
        profile.update({
            'compress': 'lzw',
            'tiled': True,
            'blockxsize': 256,
            'blockysize': 256,
        })

    # Write raster
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(raster)

        # Write band descriptions
        for i, name in enumerate(metadata['band_names']):
            dst.set_band_description(i + 1, name)

    print(f"Saved {raster.shape[0]}-band GeoTIFF to {output_path}")


# -----------------------------------------------------------------------------
# Example Usage
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    # Example 1: Process and classify a LAS file
    
    input_las = 'raw_point_cloud.las'
    classified_las = 'classified_with_hag.las'
    
    # Define a crop polygon (in the coordinate system of the LAS file)
    crop_polygon = "POLYGON ((769896 3842418, 769896 3842929, 770189 3842929, 770189 3842418, 769896 3842418))"
    
    # Create and execute the PDAL pipeline
    las_pipeline = process_and_classify_las(
        input_las=input_las,
        output_las=classified_las,
        crop_polygon=crop_polygon,
        min_hag=0,
        max_hag=25,
        filter_noise=True
    )
    las_pipeline.execute()
    print(f"Processed and classified {input_las} to {classified_las}")
    
    # Example 2: Basic raster statistics
    
    # Set the resolution for all rasters
    resolution_m = 1  
    
    # Calculate various basic statistical measures
    max_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.max)
    mean_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.mean)
    min_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.min)
    std_height = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=np.std)
    point_density = agg_las_to_array(classified_las, resolution=resolution_m, aggregate_func=point_count)
    
    # Example 3: Height percentile metrics
    
    # Calculate height percentiles (10%, 25%, 50%, 75%, 90%)
    percentile_values = [10, 25, 50, 75, 90]
    height_percentiles = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=np.percentile, 
        q=percentile_values
    )
    
    # Example 4: Vegetation structure metrics
    
    # Parameters for vegetation structure calculation
    canopy_min_hag = 3      # Minimum height for canopy points (meters)
    understory_max_hag = 1  # Maximum height for understory points (meters)
    
    # Calculate density by vegetation layer
    canopy_dens = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=canopy_density, 
        canopy_min_hag=canopy_min_hag
    )
    
    midstory_dens = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=mid_story_density, 
        understory_max_hag=understory_max_hag, 
        canopy_min_hag=canopy_min_hag
    )
    
    understory_dens = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=under_story_density, 
        understory_max_hag=understory_max_hag
    )
    
    # Calculate canopy cover
    canopy_cov = agg_las_to_array(
        classified_las, 
        resolution=resolution_m, 
        aggregate_func=canopy_cover, 
        canopy_min_hag=2  # Different threshold than canopy density
    )
    
    # Example 5: More complex metrics
    
    # Parameters for density proportions and FHD
    density_layers = 10
    height_range = [0, 18]  # Min and max height for vertical layers (meters)
    
    # Calculate density proportions (vegetation architecture)
    density_props = agg_las_to_array(
        classified_las, 
        resolution=resolution_m,
        aggregate_func=density_proportions, 
        hag_rng=height_range, 
        num_layers=density_layers
    )
    
    # Calculate Foliage Height Diversity (FHD)
    fhd = agg_las_to_array(
        classified_las, 
        resolution=resolution_m,
        aggregate_func=foliage_height_diversity, 
        hag_rng=height_range, 
        num_layers=density_layers
    )
    
    # Example 6: Visualization
    
    # Plot single rasters
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 8))
    plt.imshow(max_height, cmap='viridis')
    plt.colorbar(label='Height (m)')
    plt.title('Maximum Vegetation Height')
    
    # Compare two rasters
    compare_two_rasters(
        canopy_dens, 
        midstory_dens, 
        'Canopy', 
        'Midstory', 
        'Density',
        figsize=(15, 5)
    )
    
    # Example 7: Get point cloud footprint
    
    # Get the footprint of a point cloud as WKT
    footprint_wkt = get_pointcloud_footprint(
        classified_las,
        simplify_tolerance=1.0,  # Simplify the polygon (units in CRS units, e.g., meters)
        buffer_distance=5.0      # Add 5 meter buffer around the point cloud
    )
    
    # Print the WKT string
    print(f"Point cloud footprint WKT: {footprint_wkt[:100]}...")
    
    # Example 8: Plot rasters with geometry overlays
    
    # Plot a single raster with a geometry overlay
    max_height_copy = max_height.copy()  # Create a copy to avoid modifying original
    
    # Get the footprint of the point cloud
    footprint = get_pointcloud_footprint(
        classified_las,
        simplify_tolerance=2.0  # More aggressive simplification for cleaner visualization
    )
    
    # Create a smaller region inside the footprint for demonstration
    interior_geom = wkt.loads(footprint)
    interior_footprint = interior_geom.buffer(-10).wkt  # 10m inside the original footprint
    
    # Plot max height raster with footprints overlaid
    plot_rasters_with_geometries(
        rasters=max_height_copy,
        geometries=[footprint, interior_footprint],
        raster_titles="Maximum Vegetation Height",
        geometry_colors=['red', 'blue'],
        geometry_labels=['Original Footprint', 'Interior Region'],
        figsize=(10, 8),
        alpha=0.3
    )
    
    # Plot multiple rasters with the same geometry overlay
    plot_rasters_with_geometries(
        rasters=[max_height, mean_height, std_height],
        geometries=footprint,
        raster_titles=["Max Height", "Mean Height", "Std Dev"],
        geometry_colors='green',
        geometry_labels='Point Cloud Extent',
        figsize=(15, 5),
        alpha=0.2
    )