#!/usr/bin/env Rscript
# run_pretreatment.R
# Wrapper script for LidarForFuel::fPCpretreatment
# Preprocessing UAV LiDAR point clouds for fuel metrics computation

# Parse command-line arguments and filter out empty strings
# (conda run sometimes adds empty string arguments)
raw_args <- commandArgs(trailingOnly = TRUE)
args <- raw_args[nzchar(raw_args)]  # Remove empty strings

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
  cat("  height_filter     Max height filter (default: 80m)\n")
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
height_filter <- ifelse(length(args) >= 8, as.numeric(args[8]), 80)
# Convert string TRUE/FALSE to logical
classify <- ifelse(length(args) >= 9, toupper(args[9]) == "TRUE", FALSE)

# Validate inputs
if (!file.exists(input_las)) {
  cat(sprintf("Error: Input file not found: %s\n", input_las))
  quit(status = 1)
}

if (is.na(lma) || is.na(wd) || is.na(lma_bush) || is.na(wd_bush)) {
  cat("Error: LMA and WD values must be numeric\n")
  quit(status = 1)
}

# Set up R-level logging to site-specific logs directory
# This ensures we have R output even if Python crashes
log_dir <- file.path(dirname(dirname(output_laz)), "logs")
dir.create(log_dir, recursive = TRUE, showWarnings = FALSE)
log_file <- file.path(log_dir, paste0(basename(tools::file_path_sans_ext(output_laz)), "_pretreatment.log"))

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
cat(strrep("=", 80), "\n")
cat("LidarForFuel Pretreatment\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Input:           %s\n", input_las))
cat(sprintf("Output:          %s\n", output_laz))
cat(sprintf("Log file:        %s\n", log_file))
cat(sprintf("LMA (canopy):    %.1f g/m²\n", lma))
cat(sprintf("WD (canopy):     %.1f kg/m³\n", wd))
cat(sprintf("LMA (understory): %.1f g/m²\n", lma_bush))
cat(sprintf("WD (understory):  %.1f kg/m³\n", wd_bush))
cat(sprintf("Understory threshold: %.1f m\n", h_strata_bush))
cat(sprintf("Height filter:   %.1f m\n", height_filter))
cat(sprintf("Classify ground: %s\n", classify))
cat(strrep("=", 80), "\n\n")
log_memory("Script Start")

# Load required libraries
cat("Loading libraries...\n")
log_memory("Before Library Loading")
tryCatch({
  suppressPackageStartupMessages({
    library(lidR)
    library(lidarforfuel)
  })

  # Disable progress bars for cleaner diagnostic logs
  options(lidR.progress = FALSE)
  cat("Progress bars disabled for diagnostic mode\n")

  cat("Libraries loaded successfully\n\n")
  log_memory("After Library Loading")
}, error = function(e) {
  cat(sprintf("Error loading libraries: %s\n", e$message))
  cat("\nPlease install required packages:\n")
  cat("  install.packages('remotes')\n")
  cat("  remotes::install_github('oliviermartin7/lidarforfuel')\n")
  sink()  # Close log file before exit
  quit(status = 1)
})

# Read input point cloud
cat(sprintf("Reading point cloud: %s\n", input_las))
cat("readLAS() parameters: (using defaults - no select/filter)\n")
log_memory("Before Reading LAS")
start_time <- Sys.time()

tryCatch({
  las <- readLAS(input_las)

  # Log memory IMMEDIATELY after readLAS completes
  read_time <- Sys.time()
  log_memory("Immediately After readLAS()")

  n_points <- npoints(las)
  cat(sprintf("  Points loaded: %d\n", n_points))

  # Get bounding box using lidR 4.x compatible method
  bbox <- st_bbox(las)
  cat(sprintf("  Extent: X[%.2f, %.2f] Y[%.2f, %.2f] Z[%.2f, %.2f]\n",
              bbox["xmin"], bbox["xmax"],
              bbox["ymin"], bbox["ymax"],
              bbox["zmin"], bbox["zmax"]))

  cat(sprintf("  readLAS() duration: %.2f seconds\n", difftime(read_time, start_time, units = "secs")))
}, error = function(e) {
  cat(sprintf("Error reading LAS file: %s\n", e$message))
  cat(sprintf("Error class: %s\n", class(e)[1]))
  sink()  # Close log file before exit
  quit(status = 1)
})

cat("\n")
log_memory("After readLAS() Processing")

# Log input file size
input_size_mb <- file.info(input_las)$size / (1024^2)
cat(sprintf("Input file size: %.2f MB\n\n", input_size_mb))

# Run fPCpretreatment
cat("Running fPCpretreatment...\n")
log_memory("Before fPCpretreatment")

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
  log_memory("After fPCpretreatment")

  # Check if fPCpretreatment returned NULL
  if (is.null(las_pretreated)) {
    cat("ERROR: fPCpretreatment returned NULL\n")
    cat("This may indicate:\n")
    cat("  - All points were filtered out\n")
    cat("  - Height filter too restrictive\n")
    cat("  - Date/season filter excluded all points\n")
    cat("  - Input file format issue\n")
    sink()  # Close log file before exit
    quit(status = 1)
  }

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

log_memory("Before Writing LAZ")
tryCatch({
  writeLAS(las_pretreated, output_laz)
  write_time <- Sys.time()
  cat(sprintf("Write completed in %.2f seconds\n", difftime(write_time, pretreat_time, units = "secs")))
  log_memory("After Writing LAZ")

  # Verify output file
  if (file.exists(output_laz)) {
    file_size <- file.info(output_laz)$size / (1024^2)  # MB
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

# Summary
total_time <- Sys.time()
cat("\n", strrep("=", 80), "\n")
cat("Pretreatment Summary\n")
cat(strrep("=", 80), "\n")
cat(sprintf("Total time:      %.2f seconds\n", difftime(total_time, start_time, units = "secs")))
cat(sprintf("Input points:    %d\n", n_points))
cat(sprintf("Output points:   %d\n", npoints(las_pretreated)))
cat(sprintf("Reduction:       %.1f%%\n", 100 * (1 - npoints(las_pretreated)/n_points)))
cat("Status:          SUCCESS\n")
cat(strrep("=", 80), "\n")
log_memory("Script End")

# Close log file
sink()

quit(status = 0)
