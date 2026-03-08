"""
Band Configuration for Raster Evaluation Pipeline

This module provides utilities for loading and validating band configuration files
that define how model prediction bands map to field data columns.

Usage:
    from src.evaluation.band_config import load_band_config

    config = load_band_config('src/evaluation/configs/raster/cover_only.json')
    for band in config.bands:
        print(f"{band.name}: {band.display_name} ({band.model_units})")
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class BandInfo:
    """Information about a single prediction band."""
    output_index: int
    name: str
    display_name: str
    model_units: str
    display_units: str
    unit_conversion_factor: float
    field_column: Optional[str]
    field_units: Optional[str]
    aggregation_method: str = 'mean'  # 'mean' or 'max'

    def convert_to_display_units(self, value: float) -> float:
        """Convert value from model units to display units."""
        return value * self.unit_conversion_factor

    def has_field_mapping(self) -> bool:
        """Check if this band has a field column mapping."""
        return self.field_column is not None


@dataclass
class BandConfig:
    """Complete band configuration."""
    name: str
    description: str
    stats_file: Optional[str]  # None for baseline configs (no normalization)
    bands: List[BandInfo]

    def get_band_by_index(self, index: int) -> Optional[BandInfo]:
        """Get band info by output index."""
        for band in self.bands:
            if band.output_index == index:
                return band
        return None

    def get_band_by_name(self, name: str) -> Optional[BandInfo]:
        """Get band info by name."""
        for band in self.bands:
            if band.name == name:
                return band
        return None

    def get_bands_with_field_mapping(self) -> List[BandInfo]:
        """Get list of bands that have field column mappings."""
        return [band for band in self.bands if band.has_field_mapping()]

    def validate(self) -> None:
        """
        Validate the configuration.

        Raises:
            ValueError: If configuration is invalid
        """
        # Check for duplicate output indices
        indices = [band.output_index for band in self.bands]
        if len(indices) != len(set(indices)):
            duplicates = [i for i in indices if indices.count(i) > 1]
            raise ValueError(f"Duplicate output_index values: {set(duplicates)}")

        # Check for duplicate band names
        names = [band.name for band in self.bands]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate band names: {set(duplicates)}")

        # Check that indices are sequential starting from 0
        expected_indices = set(range(len(self.bands)))
        actual_indices = set(indices)
        if expected_indices != actual_indices:
            missing = expected_indices - actual_indices
            extra = actual_indices - expected_indices
            raise ValueError(
                f"Output indices must be sequential starting from 0. "
                f"Missing: {missing}, Extra: {extra}"
            )

        # Check that stats file exists (if provided)
        if self.stats_file is not None:
            stats_path = Path(self.stats_file)
            if not stats_path.exists():
                raise FileNotFoundError(f"Stats file not found: {self.stats_file}")

        # Check that aggregation methods are valid
        valid_methods = {'mean', 'max'}
        for band in self.bands:
            if band.aggregation_method not in valid_methods:
                raise ValueError(
                    f"Invalid aggregation_method '{band.aggregation_method}' for band "
                    f"'{band.name}'. Must be one of {valid_methods}"
                )


def load_band_config(config_path: str) -> BandConfig:
    """
    Load and validate band configuration from JSON file.

    Args:
        config_path: Path to band config JSON file

    Returns:
        BandConfig object

    Raises:
        FileNotFoundError: If config file not found
        ValueError: If config is invalid
        json.JSONDecodeError: If JSON is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Band config not found: {config_path}")

    with open(config_path) as f:
        data = json.load(f)

    # Parse band info objects
    bands = []
    for band_data in data['bands']:
        band = BandInfo(
            output_index=band_data['output_index'],
            name=band_data['name'],
            display_name=band_data['display_name'],
            model_units=band_data['model_units'],
            display_units=band_data['display_units'],
            unit_conversion_factor=band_data['unit_conversion_factor'],
            field_column=band_data.get('field_column'),  # Optional
            field_units=band_data.get('field_units'),  # Optional
            aggregation_method=band_data.get('aggregation_method', 'mean'),  # Optional, default='mean'
        )
        bands.append(band)

    # Create config object
    config = BandConfig(
        name=data['name'],
        description=data['description'],
        stats_file=data['stats_file'],
        bands=bands
    )

    # Validate
    config.validate()

    return config
