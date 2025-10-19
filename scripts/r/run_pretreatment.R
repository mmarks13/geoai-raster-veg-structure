#!/usr/bin/env Rscript
# run_pretreatment.R
# Wrapper script for LidarForFuel::fPCpretreatment
# Preprocessing UAV LiDAR point clouds for fuel metrics computation

# Parse command-line arguments
args <- commandArgs(trailingOnly = TRUE)

# Usage message
usage <- function() {
  cat("Usage: Rscript run_pretreatment.R <input_las> <output_laz> <lma> <wd> <lma_bush> <wd_bush> [options]\n")
  cat("\nPositional arguments:\n")
  cat("  input_las    Path to input LAS/LAZ file\n")
  cat("  output_laz   Path to output pretreated LAZ file\n")
  cat("  lma          Leaf Mass Area (g/m²) for canopy\n")
  cat("  wd           Wood Density (kg/m³) for canopy\n")
  cat("  lma_bush     Leaf Mass Area (g/m²) for understory (<2m)\n")
  cat("  wd_bush      Wood Density (kg/m³) for understory (<2m)\n")
  cat("\nOptional arguments:\n")
  cat("  h_strata_bush     Height threshold for understory (default: 2m)\n")
  cat("  height_filter     Max height filter (default: 60m)\n")
  cat("  classify          Classify ground points (TRUE/FALSE, default: FALSE)\n")
  cat("\nExample:\n")
  cat("  Rscript run_pretreatment.R input.las output.laz 140 591 130 550\n")
  quit(status = 1)
}

# Check minimum required arguments
if (length(args) < 6) {
  cat("Error: Missing required arguments\n\n")
  usage()
}

# Parse positional arguments
input_las <- args[1]
output_laz <- args[2]
lma <- as.numeric(args[3])
wd <- as.numeric(args[4])
lma_bush <- as.numeric(args[5])
wd_bush <- as.numeric(args[6])

# Parse optional arguments with defaults
h_strata_bush <- ifelse(length(args) >= 7, as.numeric(args[7]), 2)
height_filter <- ifelse(length(args) >= 8, as.numeric(args[8]), 60)
classify <- ifelse(length(args) >= 9, as.logical(args[9]), FALSE)

# Validate inputs
if (!file.exists(input_las)) {
  cat(sprintf("Error: Input file not found: %s\n", input_las))
  quit(status = 1)
}

if (is.na(lma) || is.na(wd) || is.na(lma_bush) || is.na(wd_bush)) {
  cat("Error: LMA and WD values must be numeric\n")
  quit(status = 1)
}

# Print configuration
cat("=" * 80, "\n")
cat("LidarForFuel Pretreatment\n")
cat("=" * 80, "\n")
cat(sprintf("Input:           %s\n", input_las))
cat(sprintf("Output:          %s\n", output_laz))
cat(sprintf("LMA (canopy):    %.1f g/m²\n", lma))
cat(sprintf("WD (canopy):     %.1f kg/m³\n", wd))
cat(sprintf("LMA (understory): %.1f g/m²\n", lma_bush))
cat(sprintf("WD (understory):  %.1f kg/m³\n", wd_bush))
cat(sprintf("Understory threshold: %.1f m\n", h_strata_bush))
cat(sprintf("Height filter:   %.1f m\n", height_filter))
cat(sprintf("Classify ground: %s\n", classify))
cat("=" * 80, "\n\n")

# Load required libraries
cat("Loading libraries...\n")
tryCatch({
  suppressPackageStartupMessages({
    library(lidR)
    library(lidarforfuel)
  })
  cat("Libraries loaded successfully\n\n")
}, error = function(e) {
  cat(sprintf("Error loading libraries: %s\n", e$message))
  cat("\nPlease install required packages:\n")
  cat("  install.packages('remotes')\n")
  cat("  remotes::install_github('oliviermartin7/lidarforfuel')\n")
  quit(status = 1)
})

# Read input point cloud
cat(sprintf("Reading point cloud: %s\n", input_las))
start_time <- Sys.time()

tryCatch({
  las <- readLAS(input_las)
  n_points <- npoints(las)
  cat(sprintf("  Points: %d\n", n_points))
  cat(sprintf("  Extent: [%.2f, %.2f] x [%.2f, %.2f] x [%.2f, %.2f]\n",
              las@bbox[1,1], las@bbox[1,2],
              las@bbox[2,1], las@bbox[2,2],
              las@bbox[3,1], las@bbox[3,2]))
}, error = function(e) {
  cat(sprintf("Error reading LAS file: %s\n", e$message))
  quit(status = 1)
})

read_time <- Sys.time()
cat(sprintf("Read completed in %.2f seconds\n\n", difftime(read_time, start_time, units = "secs")))

# Run fPCpretreatment
cat("Running fPCpretreatment...\n")

tryCatch({
  las_pretreated <- fPCpretreatment(
    chunk = input_las,
    classify = classify,
    LMA = lma,
    WD = wd,
    LMA_bush = lma_bush,
    WD_bush = wd_bush,
    H_strata_bush = h_strata_bush,
    Height_filter = height_filter,
    start_date = "2011-09-14 00:00:00",  # Default from LidarForFuel
    season_filter = 1:12,                 # All months
    deviation_days = "Infinity",          # No date filtering
    plot_hist_days = FALSE                # No plots
  )

  pretreat_time <- Sys.time()
  cat(sprintf("Pretreatment completed in %.2f seconds\n", difftime(pretreat_time, read_time, units = "secs")))

  # Check that required attributes were added
  required_attrs <- c("LMA", "WD", "Zref", "Easting", "Northing", "Elevation")
  present_attrs <- names(las_pretreated@data)
  missing_attrs <- setdiff(required_attrs, present_attrs)

  if (length(missing_attrs) > 0) {
    cat(sprintf("Warning: Missing expected attributes: %s\n", paste(missing_attrs, collapse = ", ")))
  } else {
    cat("All required attributes added successfully\n")
  }

  cat(sprintf("  Output points: %d\n", npoints(las_pretreated)))

}, error = function(e) {
  cat(sprintf("Error in fPCpretreatment: %s\n", e$message))
  cat("\nTraceback:\n")
  print(traceback())
  quit(status = 1)
})

# Create output directory if needed
output_dir <- dirname(output_laz)
if (!dir.exists(output_dir)) {
  cat(sprintf("Creating output directory: %s\n", output_dir))
  dir.create(output_dir, recursive = TRUE)
}

# Write pretreated point cloud
cat(sprintf("\nWriting output: %s\n", output_laz))

tryCatch({
  writeLAS(las_pretreated, output_laz)
  write_time <- Sys.time()
  cat(sprintf("Write completed in %.2f seconds\n", difftime(write_time, pretreat_time, units = "secs")))

  # Verify output file
  if (file.exists(output_laz)) {
    file_size <- file.info(output_laz)$size / (1024^2)  # MB
    cat(sprintf("Output file size: %.2f MB\n", file_size))
  } else {
    cat("Warning: Output file not found after write\n")
    quit(status = 1)
  }

}, error = function(e) {
  cat(sprintf("Error writing output: %s\n", e$message))
  quit(status = 1)
})

# Summary
total_time <- Sys.time()
cat("\n" , "=" * 80, "\n")
cat("Pretreatment Summary\n")
cat("=" * 80, "\n")
cat(sprintf("Total time:      %.2f seconds\n", difftime(total_time, start_time, units = "secs")))
cat(sprintf("Input points:    %d\n", n_points))
cat(sprintf("Output points:   %d\n", npoints(las_pretreated)))
cat(sprintf("Reduction:       %.1f%%\n", 100 * (1 - npoints(las_pretreated)/n_points)))
cat("Status:          SUCCESS\n")
cat("=" * 80, "\n")

quit(status = 0)
