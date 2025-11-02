#!/usr/bin/env Rscript
# run_fuel_metrics.R
# Wrapper script for LidarForFuel::fCBDprofile_fuelmetrics
# Compute fuel metrics from pretreated UAV LiDAR point clouds

# Parse command-line arguments and filter out empty strings
# (conda run sometimes adds empty string arguments)
raw_args <- commandArgs(trailingOnly = TRUE)
args <- raw_args[nzchar(raw_args)]  # Remove empty strings

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
  cat("  omega             Clumping factor Ω for Beer-Lambert model (default: 0.77)\n")
  cat("  projection_factor Projection factor G for fuel metrics (default: 0.5)\n")
  cat("  export_mode       'full' (173 bands) or 'summary' (23 bands only) (default: 'full')\n")
  cat("\nExample:\n")
  cat("  Rscript run_fuel_metrics.R pretreated.laz fuel_metrics.tif 1.0\n")
  cat("  Rscript run_fuel_metrics.R pretreated.laz fuel_metrics.tif 1.0 1.0 2.0 0.02 0.77 0.5 summary\n")
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
omega <- ifelse(length(args) >= 7, as.numeric(args[7]), 0.77)
projection_factor <- ifelse(length(args) >= 8, as.numeric(args[8]), 0.5)
export_mode <- ifelse(length(args) >= 9, args[9], "full")

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

# Set up R-level logging to site-specific logs directory
# This ensures we have R output even if Python crashes
log_dir <- file.path(dirname(dirname(output_tif)), "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(log_dir, paste0(basename(tools::file_path_sans_ext(output_tif)), "_fuel_metrics.log"))

# Open log file (split=TRUE sends output to both console AND file)
sink(log_file, split = TRUE)

# Helper function to log memory usage
log_memory <- function(step_name) {
  gc_stats <- gc()
  # gc() returns a 2x6 matrix: rows = Ncells/Vcells, cols = used/gc trigger/limit/etc
  # Sum used memory across both rows (Ncells + Vcells in MB)
  used_mb <- sum(gc_stats[, "used"])
  cat(sprintf("[%s] R Memory Used: %.2f MB (Ncells: %.2f MB, Vcells: %.2f MB)\n",
              step_name, used_mb, gc_stats[1, "used"], gc_stats[2, "used"]))
}

# Print configuration
cat(rep("=", 80), "\n", sep = "")
cat("LidarForFuel Fuel Metrics\n")
cat(rep("=", 80), "\n", sep = "")
cat(sprintf("Input:           %s\n", input_laz))
cat(sprintf("Output:          %s\n", output_tif))
cat(sprintf("Log file:        %s\n", log_file))
cat(sprintf("Resolution:      %.2f m\n", resolution))
cat(sprintf("Layer depth:     %.2f m\n", layer_depth))
cat(sprintf("Height cover:    %.2f m\n", height_cover))
cat(sprintf("Threshold:       %.3f\n", threshold))
cat(sprintf("Omega (Ω):       %.2f\n", omega))
cat(sprintf("Projection (G):  %.2f\n", projection_factor))
cat(sprintf("Export mode:     %s (%s bands)\n", export_mode, ifelse(export_mode == "full", "173", "23")))
cat(rep("=", 80), "\n\n", sep = "")
log_memory("Script Start")

# Load required libraries
cat("Loading libraries...\n")
log_memory("Before Library Loading")
tryCatch({
  suppressPackageStartupMessages({
    library(lidR)
    library(lidarforfuel)
    library(terra)
  })

  # Disable progress bars for cleaner diagnostic logs
  options(lidR.progress = FALSE)
  cat("Progress bars disabled for diagnostic mode\n")

  cat("Libraries loaded successfully\n\n")
  log_memory("After Library Loading")
}, error = function(e) {
  cat(sprintf("Error loading libraries: %s\n", e$message))
  cat("\nPlease install required packages:\n")
  cat("  install.packages(c('remotes', 'terra'))\n")
  cat("  remotes::install_github('oliviermartin7/lidarforfuel')\n")
  sink()  # Close log file before exit
  quit(status = 1)
})

# Read pretreated point cloud
cat(sprintf("Reading pretreated point cloud: %s\n", input_laz))

# Log input file size
input_size_mb <- file.info(input_laz)$size / (1024^2)
cat(sprintf("Input file size: %.2f MB\n\n", input_size_mb))

log_memory("Before Reading LAZ")
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
  sink()  # Close log file before exit
  quit(status = 1)
})

read_time <- Sys.time()
cat(sprintf("Read completed in %.2f seconds\n\n", difftime(read_time, start_time, units = "secs")))
log_memory("After Reading LAZ")

# Compute fuel metrics using pixel_metrics
cat("Computing fuel metrics...\n")
cat(sprintf("  Creating %.2f m resolution raster grid\n", resolution))
log_memory("Before pixel_metrics")

tryCatch({
  # Use pixel_metrics to apply fCBDprofile_fuelmetrics to raster cells
  # Following official example from lidarforfuel documentation
  fuel_raster <- pixel_metrics(
    las,
    ~fCBDprofile_fuelmetrics(
      X = X,
      Y = Y,
      Z = Z,
      Zref = Zref,
      gpstime = gpstime,
      ReturnNumber = ReturnNumber,
      Easting = Easting,
      Northing = Northing,
      Elevation = Elevation,
      LMA = LMA,
      threshold = threshold,
      WD = WD,
      limit_N_points = 50,
      scanning_angle = TRUE,   # Enable scanning angle correction for accurate path length
      limit_flightheight = 40, # UAV-appropriate threshold (default 800m is for airborne LiDAR)
      datatype = "Pixel",  # Must be specified explicitly
      omega = omega,
      d = layer_depth,
      G = projection_factor
    ),
    res = resolution
  )

  metrics_time <- Sys.time()
  cat(sprintf("Metrics computation completed in %.2f seconds\n", difftime(metrics_time, read_time, units = "secs")))
  log_memory("After pixel_metrics")

  # pixel_metrics already returns a SpatRaster - no need to convert
  # fuel_raster is already a terra SpatRaster object

  # DIAGNOSTIC: Check raw values before NA conversion
  cat("\n", rep("-", 80), "\n", sep = "")
  cat("DIAGNOSTIC OUTPUT (raw raster before NA conversion)\n")
  cat(rep("-", 80), "\n", sep = "")

  # Check first band for -1 values
  first_band_values <- values(fuel_raster[[1]])
  n_total <- length(first_band_values)
  n_failed <- sum(first_band_values == -1, na.rm = TRUE)
  n_success <- sum(first_band_values != -1, na.rm = TRUE)
  n_na <- sum(is.na(first_band_values))

  cat(sprintf("Total pixels:      %d\n", n_total))
  cat(sprintf("Failed pixels (-1): %d (%.1f%%)\n", n_failed, 100 * n_failed / n_total))
  cat(sprintf("Success pixels:    %d (%.1f%%)\n", n_success, 100 * n_success / n_total))
  cat(sprintf("NA pixels:         %d (%.1f%%)\n", n_na, 100 * n_na / n_total))

  # Show sample of actual values (non -1, non NA)
  success_values <- first_band_values[first_band_values != -1 & !is.na(first_band_values)]
  if (length(success_values) > 0) {
    cat(sprintf("\nSample success values (first 10): %s\n",
                paste(round(head(success_values, 10), 3), collapse = ", ")))
  } else {
    cat("\nNo successful computations found - all pixels returned -1\n")
  }

  cat(rep("-", 80), "\n\n", sep = "")

  # Replace -1 values (failed computation) with NA
  # LidarForFuel returns -1 when computation fails (e.g., insufficient points, flight height check)
  fuel_raster <- subst(fuel_raster, -1, NA)

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
  # NOTE: Band 3 is 'threshold' value used for filtering (e.g., 0.02 kg/m³)
  band_names <- c(
    "Profil_Type", "Profil_Type_L", "threshold", "Height", "CBH",
    "FSG", "Top_Fuel", "H_Bush", "continuity", "VCI_PAD",
    "VCI_lidr", "entropy_lidr", "PAI_tot", "CBD_max", "CFL",
    "TFL", "MFL", "FL_1_3", "GSFL", "FL_0_1",
    "FMA", "date", "Cover"
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
log_memory("Before Writing GeoTIFF")

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
  log_memory("After Writing GeoTIFF")

  # Verify output file
  if (file.exists(output_tif)) {
    file_size <- file.info(output_tif)$size / (1024^2)  # MB
    cat(sprintf("Output file size: %.2f MB\n", file_size))
  } else {
    cat("Warning: Output file not found after write\n")
    sink()  # Close log file before exit
    quit(status = 1)
  }

}, error = function(e) {
  cat(sprintf("Error writing output: %s\n", e$message))
  sink()  # Close log file before exit
  quit(status = 1)
})

# Generate summary statistics
cat("\n", rep("=", 80), "\n", sep = "")
cat("Fuel Metrics Summary Statistics\n")
cat(rep("=", 80), "\n", sep = "")

# Summary of key metrics (if available)
if (export_mode == "summary" || nlyr(fuel_raster) >= 23) {
  # Extract key bands (corrected for threshold at band 3)
  key_metrics <- c(
    "Height" = 4,
    "CBH" = 5,
    "FSG" = 6,
    "TFL" = 16,
    "CFL" = 15,
    "Cover" = 23
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
log_memory("Script End")

warnings()

# Close log file
sink()

quit(status = 0)
