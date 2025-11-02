"""
Fuel Metrics Module

Wildfire fuel hazard mapping from UAV LiDAR using LidarForFuel R package.

This module provides:
- LidarForFuel wrapper (R package integration via Python)
- Batch processing of tiled point clouds
- Fuel metrics computation (173-band rasters: 23 summary metrics + 150 bulk density layers)
- Visualization tools for validation and QC

Key components:
- lidarforfuel_wrapper: Python-R interface for pretreatment and fuel metrics
- process_fuel_metrics: Main orchestration script for single/batch processing
- batch_processing: Per-tile wrapper with progress tracking
- visualize_bounds: Spatial coverage validation
- visualize_metrics: Fuel metrics visualization

See data/processed/fuel_metrics/README.md for complete pipeline documentation.
"""

__version__ = "1.0.0"
__author__ = "geoai_veg_map"

# Export main functions
from .lidarforfuel_wrapper import (
    run_pretreatment,
    run_fuel_metrics,
    process_point_cloud,
    load_trait_lookup,
    check_rscript_available,
)

__all__ = [
    "run_pretreatment",
    "run_fuel_metrics",
    "process_point_cloud",
    "load_trait_lookup",
    "check_rscript_available",
]
