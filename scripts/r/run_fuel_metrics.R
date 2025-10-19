#!/usr/bin/env Rscript
# run_fuel_metrics.R
# Wrapper script for LidarForFuel::fCBDprofile_fuelmetrics
# Compute fuel metrics from pretreated UAV LiDAR point clouds

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Usage message
usage <- function() {
  cat("Usage: Rscript run_fuel_metrics.R <input_laz> <output_tif> <resolution> [options]\n")
  cat("\nPositional arguments:\n")
  cat("  input_laz    Path to pretreated LAZ file (from run_pretreatment.R)\n")
  cat("  output_tif   Path to output GeoTIFF (173-band fuel metrics raster)\n")
  cat("  resolution   Output raster resolution in meters (e.g., 1.0)\n")
  cat("\nOptional arguments:\n")
  cat("  layer_depth       Vertical layer depth (m) for bulk density profile (default: 1.0)\n")
  cat("  height_cover      Height threshold for cover computation (m) (default: 2.0)\n")
  cat("  threshold         Bulk density threshold for strata detection (default: 0.02)\n")
  cat("  export_mode       'full' (173 bands) or 'summary' (23 bands only) (default: 'full')\n")
  cat("\nExample:\n")
  cat("  Rscript run_fuel_metrics.R pretreated.laz fuel_metrics.tif 1.0\n")
  cat("  Rscript run_fuel_metrics.R pretreated.laz fuel_metrics.tif 1.0 1.0 2.0 0.02 summary\n")
  quit(status = 1)
}

# Check minimum required arguments
if (length(args) < 3) {
  cat("Error: Missing required arguments\n\n")
  usage()
}

# Parse positional arguments
input_laz <- args[1]
output_tif <- args[2]
resolution <- as.numeric(args[3])

# Parse optional arguments with defaults
layer_depth <- ifelse(length(args) >= 4, as.numeric(args[4]), 1.0)
height_cover <- ifelse(length(args) >= 5, as.numeric(args[5]), 2.0)
threshold <- ifelse(length(args) >= 6, as.numeric(args[6]), 0.02)
export_mode <- ifelse(length(args) >= 7, args[7], "full")

# Validate inputs
if (!file.exists(input_laz)) {
  cat(sprintf("Error: Input file not found: %s\n", input_laz))
  quit(status = 1)
}

if (is.na(resolution) || resolution <= 0) {
  cat("Error: Resolution must be a positive number\n")
  quit(status = 1)
}

if (!(export_mode %in% c("full", "summary"))) {
  cat("Error: export_mode must be 'full' or 'summary'\n")
  quit(status = 1)
}

# Print configuration
cat(rep("=", 80), "\n", sep = "")
cat("LidarForFuel Fuel Metrics\n")
cat(rep("=", 80), "\n", sep = "")
cat(sprintf("Input:           %s\n", input_laz))
cat(sprintf("Output:          %s\n", output_tif))
cat(sprintf("Resolution:      %.2f m\n", resolution))
cat(sprintf("Layer depth:     %.2f m\n", layer_depth))
cat(sprintf("Height cover:    %.2f m\n", height_cover))
cat(sprintf("Threshold:       %.3f\n", threshold))
cat(sprintf("Export mode:     %s (%s bands)\n", export_mode, ifelse(export_mode == "full", "173", "23")))
cat(rep("=", 80), "\n\n", sep = "")

# Load required libraries
cat("Loading libraries...\n")
tryCatch({
  suppressPackageStartupMessages({
    library(lidR)
    library(lidarforfuel)
    library(terra)
  })
  cat("Libraries loaded successfully\n\n")
}, error = function(e) {
  cat(sprintf("Error loading libraries: %s\n", e$message))
  cat("\nPlease install required packages:\n")
  cat("  install.packages(c('remotes', 'terra'))\n")
  cat("  remotes::install_github('oliviermartin7/lidarforfuel')\n")
  quit(status = 1)
})

# Read pretreated point cloud
cat(sprintf("Reading pretreated point cloud: %s\n", input_laz))
start_time <- Sys.time()

tryCatch({
  las <- readLAS(input_laz)
  n_points <- npoints(las)
  cat(sprintf("  Points: %d\n", n_points))

  # Check for required attributes
  required_attrs <- c("X", "Y", "Z", "Zref", "ReturnNumber", "Easting", "Northing", "Elevation", "LMA", "WD", "gpstime")
  present_attrs <- names(las@data)
  missing_attrs <- setdiff(required_attrs, present_attrs)

  if (length(missing_attrs) > 0) {
    cat(sprintf("Error: Missing required attributes: %s\n", paste(missing_attrs, collapse = ", ")))
    cat("The input file must be pretreated with run_pretreatment.R first\n")
    quit(status = 1)
  }

  cat("All required attributes present\n")

}, error = function(e) {
  cat(sprintf("Error reading LAZ file: %s\n", e$message))
  quit(status = 1)
})

read_time <- Sys.time()
cat(sprintf("Read completed in %.2f seconds\n\n", difftime(read_time, start_time, units = "secs")))

# Compute fuel metrics using pixel_metrics
cat("Computing fuel metrics...\n")
cat(sprintf("  Creating %.2f m resolution raster grid\n", resolution))

tryCatch({
  # Use pixel_metrics to apply fCBDprofile_fuelmetrics to raster cells
  fuel_raster <- pixel_metrics(
    las = las,
    func = ~fCBDprofile_fuelmetrics(
      datatype = "Pixel",
      X = X,
      Y = Y,
      Z = Z,
      Zref = Zref,
      ReturnNumber = ReturnNumber,
      Easting = Easting,
      Northing = Northing,
      Elevation = Elevation,
      LMA = LMA,
      WD = WD,
      gpstime = gpstime,
      Height_Cover = height_cover,
      threshold = threshold,
      scanning_angle = TRUE,
      use_cover = FALSE,
      d = layer_depth,
      G = 0.5,          # Extinction coefficient (default)
      omega = 0.77,     # Clumping factor (default)
      H_PAI = 0,        # Height for PAI computation (default)
      limit_N_points = 400,        # Min points per pixel
      limit_flightheight = 800,    # Max flight height (m)
      limit_vegetationheight = 0.1 # Min vegetation height
    ),
    res = resolution
  )

  metrics_time <- Sys.time()
  cat(sprintf("Metrics computation completed in %.2f seconds\n", difftime(metrics_time, read_time, units = "secs")))

  # Convert to terra SpatRaster for better handling
  fuel_raster <- rast(fuel_raster)

  # Get raster dimensions
  n_bands <- nlyr(fuel_raster)
  cat(sprintf("  Output bands: %d\n", n_bands))
  cat(sprintf("  Raster dimensions: %d x %d pixels\n", nrow(fuel_raster), ncol(fuel_raster)))
  cat(sprintf("  Extent: [%.2f, %.2f] x [%.2f, %.2f]\n",
              ext(fuel_raster)[1], ext(fuel_raster)[2],
              ext(fuel_raster)[3], ext(fuel_raster)[4]))

}, error = function(e) {
  cat(sprintf("Error computing fuel metrics: %s\n", e$message))
  cat("\nTraceback:\n")
  print(traceback())
  quit(status = 1)
})

# Select bands for export
if (export_mode == "summary") {
  cat("\nExporting summary metrics only (first 23 bands)\n")
  fuel_raster <- subset(fuel_raster, 1:23)

  # Document band names (first 23 summary metrics)
  band_names <- c(
    "Profil_Type", "Profil_Type_L", "Height", "CBH", "FSG",
    "VCI_PAD", "VCI_CBD", "TFL", "CFL", "MFL",
    "surf_fuel_load", "Canopy_cover", "MidStorey_cover", "understory_cover",
    "Total_cover", "Total_cover_2m", "entropy_CBD", "entropy_PAD",
    "PAI", "PAI_upper", "PAI_mid", "PAI_understory", "max_CBD"
  )

  # Set band names if available
  if (n_bands >= 23) {
    names(fuel_raster) <- band_names
  }
}

# Create output directory if needed
output_dir <- dirname(output_tif)
if (!dir.exists(output_dir)) {
  cat(sprintf("Creating output directory: %s\n", output_dir))
  dir.create(output_dir, recursive = TRUE)
}

# Write output raster
cat(sprintf("\nWriting output: %s\n", output_tif))

tryCatch({
  writeRaster(
    fuel_raster,
    filename = output_tif,
    overwrite = TRUE,
    gdal = c("COMPRESS=LZW", "TILED=YES", "BIGTIFF=IF_SAFER"),
    datatype = "FLT4S"  # 32-bit float
  )

  write_time <- Sys.time()
  cat(sprintf("Write completed in %.2f seconds\n", difftime(write_time, metrics_time, units = "secs")))

  # Verify output file
  if (file.exists(output_tif)) {
    file_size <- file.info(output_tif)$size / (1024^2)  # MB
    cat(sprintf("Output file size: %.2f MB\n", file_size))
  } else {
    cat("Warning: Output file not found after write\n")
    quit(status = 1)
  }

}, error = function(e) {
  cat(sprintf("Error writing output: %s\n", e$message))
  quit(status = 1)
})

# Generate summary statistics
cat("\n", rep("=", 80), "\n", sep = "")
cat("Fuel Metrics Summary Statistics\n")
cat(rep("=", 80), "\n", sep = "")

# Summary of key metrics (if available)
if (export_mode == "summary" || nlyr(fuel_raster) >= 23) {
  # Extract key bands
  key_metrics <- c(
    "Height" = 3,
    "CBH" = 4,
    "FSG" = 5,
    "TFL" = 8,
    "CFL" = 9,
    "Canopy_cover" = 12
  )

  for (metric_name in names(key_metrics)) {
    band_idx <- key_metrics[metric_name]
    if (band_idx <= nlyr(fuel_raster)) {
      band_data <- values(fuel_raster[[band_idx]], na.rm = TRUE)
      if (length(band_data) > 0) {
        cat(sprintf("%s:\n", metric_name))
        cat(sprintf("  Min:    %.3f\n", min(band_data, na.rm = TRUE)))
        cat(sprintf("  Median: %.3f\n", median(band_data, na.rm = TRUE)))
        cat(sprintf("  Mean:   %.3f\n", mean(band_data, na.rm = TRUE)))
        cat(sprintf("  Max:    %.3f\n", max(band_data, na.rm = TRUE)))
        cat(sprintf("  SD:     %.3f\n", sd(band_data, na.rm = TRUE)))
      }
    }
  }
}

# Overall summary
total_time <- Sys.time()
cat("\n", rep("=", 80), "\n", sep = "")
cat("Processing Summary\n")
cat(rep("=", 80), "\n", sep = "")
cat(sprintf("Total time:      %.2f seconds\n", difftime(total_time, start_time, units = "secs")))
cat(sprintf("Input points:    %d\n", n_points))
cat(sprintf("Output bands:    %d\n", nlyr(fuel_raster)))
cat(sprintf("Output pixels:   %d\n", ncell(fuel_raster)))
cat(sprintf("Resolution:      %.2f m\n", resolution))
cat("Status:          SUCCESS\n")
cat(rep("=", 80), "\n", sep = "")

quit(status = 0)
