"""
Fuel Metrics Band Configuration

Defines handling policies for each of the 22 fuel metrics bands extracted from
LidarForFuel output. This configuration is the single source of truth for:
- Band identities and physical meanings
- NA/NaN handling policies
- Expected value ranges for validation

Source: LidarForFuel GitHub (https://github.com/oliviermartin7/LidarForFuel)
Band order based on fCBDprofile_fuelmetrics() output with band 22 (date) removed.

Usage:
    from src.data_prep.fuel_metrics_band_config import BAND_CONFIG, get_na_replacement_indices

    # Get list of band indices that should have NA → 0
    replace_indices = get_na_replacement_indices()

    # Get info for specific band
    band_info = BAND_CONFIG[3]  # Height
    print(f"{band_info['name']}: {band_info['description']}")
"""

# Band configuration dictionary
# Key: 0-indexed tile band index
# Value: dict with configuration parameters
BAND_CONFIG = {
    0: {
        'name': 'Profil_Type',
        'source_band': 1,  # 1-indexed source band number
        'description': 'Detailed fuel profile type (1-5)',
        'units': 'categorical',
        'na_policy': 'keep',
        'na_justification': 'Categorical value, NA has distinct meaning (no profile)',
        'expected_range': (1, 5),
        'replace_value': None,
    },

    1: {
        'name': 'Profil_Type_L',
        'source_band': 2,
        'description': 'Simplified fuel profile type (A-D, encoded as 1-4)',
        'units': 'categorical',
        'na_policy': 'keep',
        'na_justification': 'Categorical value, NA has distinct meaning',
        'expected_range': (1, 4),
        'replace_value': None,
    },

    2: {
        'name': 'threshold',
        'source_band': 3,
        'description': 'Bulk density threshold used for strata identification',
        'units': 'kg/m³',
        'na_policy': 'replace',
        'na_justification': 'Constant value (0.02), but included for consistency',
        'expected_range': (0.01, 0.03),
        'replace_value': 0.0,
    },

    3: {
        'name': 'Height',
        'source_band': 4,
        'description': 'Canopy height (maximum vegetation height)',
        'units': 'meters',
        'na_policy': 'replace',
        'na_justification': 'No vegetation = ground level = 0m height',
        'expected_range': (0.0, 70.0),
        'replace_value': 0.0,
    },

    4: {
        'name': 'CBH',
        'source_band': 5,
        'description': 'Canopy base height (lowest canopy vegetation)',
        'units': 'meters',
        'na_policy': 'keep',
        'na_justification': 'Undefined without canopy; NA = no canopy structure',
        'expected_range': (0.0, 50.0),
        'replace_value': None,
    },

    5: {
        'name': 'FSG',
        'source_band': 6,
        'description': 'Fuel strata gap (vertical gap between fuel layers)',
        'units': 'meters',
        'na_policy': 'keep',
        'na_justification': 'Undefined without multiple fuel strata',
        'expected_range': (0.0, 50.0),
        'replace_value': None,
    },

    6: {
        'name': 'Top_Fuel',
        'source_band': 7,
        'description': 'Canopy top per bulk density threshold',
        'units': 'meters',
        'na_policy': 'keep',
        'na_justification': 'Undefined without canopy meeting threshold criteria',
        'expected_range': (0.0, 70.0),
        'replace_value': None,
    },

    7: {
        'name': 'H_Bush',
        'source_band': 8,
        'description': 'Midstorey top height (understory vegetation layer)',
        'units': 'meters',
        'na_policy': 'keep',
        'na_justification': 'Undefined without midstorey layer; NA = no midstorey',
        'expected_range': (0.0, 40.0),
        'replace_value': None,
    },

    8: {
        'name': 'continuity',
        'source_band': 9,
        'description': 'Fuel profile type D indicator (continuous canopy)',
        'units': 'binary',
        'na_policy': 'replace',
        'na_justification': 'Binary indicator; no continuity = 0',
        'expected_range': (0.0, 1.0),
        'replace_value': 0.0,
    },

    9: {
        'name': 'VCI_PAD',
        'source_band': 10,
        'description': 'Vertical complexity index from PAD profile',
        'units': 'dimensionless',
        'na_policy': 'keep',
        'na_justification': 'Complexity metric undefined without vegetation',
        'expected_range': (0.0, 1.0),
        'replace_value': None,
    },

    10: {
        'name': 'VCI_lidr',
        'source_band': 11,
        'description': 'Vertical complexity index from point cloud',
        'units': 'dimensionless',
        'na_policy': 'keep',
        'na_justification': 'Complexity metric undefined without sufficient points',
        'expected_range': (0.0, 1.0),
        'replace_value': None,
    },

    11: {
        'name': 'entropy_lidr',
        'source_band': 12,
        'description': 'Shannon entropy from point cloud vertical distribution',
        'units': 'dimensionless',
        'na_policy': 'keep',
        'na_justification': 'Entropy undefined without sufficient points',
        'expected_range': (0.0, 1.0),
        'replace_value': None,
    },

    12: {
        'name': 'PAI_tot',
        'source_band': 13,
        'description': 'Total plant area index (leaf + wood area)',
        'units': 'dimensionless',
        'na_policy': 'replace',
        'na_justification': 'No plants = zero plant area index',
        'expected_range': (0.0, 10.0),
        'replace_value': 0.0,
    },

    13: {
        'name': 'CBD_max',
        'source_band': 14,
        'description': 'Maximum canopy bulk density',
        'units': 'kg/m³',
        'na_policy': 'replace',
        'na_justification': 'No fuel = zero bulk density',
        'expected_range': (0.0, 2.0),
        'replace_value': 0.0,
    },

    14: {
        'name': 'CFL',
        'source_band': 15,
        'description': 'Canopy fuel load (fuel mass in canopy layer)',
        'units': 'kg/m²',
        'na_policy': 'replace',
        'na_justification': 'No canopy = zero canopy fuel load',
        'expected_range': (0.0, 30.0),
        'replace_value': 0.0,
    },

    15: {
        'name': 'TFL',
        'source_band': 16,
        'description': 'Total fuel load (all vegetation fuel mass)',
        'units': 'kg/m²',
        'na_policy': 'replace',
        'na_justification': 'No fuel = zero fuel load',
        'expected_range': (0.0, 50.0),
        'replace_value': 0.0,
    },

    16: {
        'name': 'MFL',
        'source_band': 17,
        'description': 'Midstorey fuel load (fuel in intermediate layer)',
        'units': 'kg/m²',
        'na_policy': 'replace',
        'na_justification': 'No midstorey = zero midstorey fuel load',
        'expected_range': (0.0, 30.0),
        'replace_value': 0.0,
    },

    17: {
        'name': 'FL_1_3',
        'source_band': 18,
        'description': 'Fuel load in 1-3m height stratum',
        'units': 'kg/m²',
        'na_policy': 'replace',
        'na_justification': 'No fuel in stratum = zero fuel load',
        'expected_range': (0.0, 30.0),
        'replace_value': 0.0,
    },

    18: {
        'name': 'GSFL',
        'source_band': 19,
        'description': 'Gap strata fuel load (fuel in vertical gaps)',
        'units': 'kg/m²',
        'na_policy': 'replace',
        'na_justification': 'No gap fuel = zero fuel load',
        'expected_range': (0.0, 30.0),
        'replace_value': 0.0,
    },

    19: {
        'name': 'FL_0_1',
        'source_band': 20,
        'description': 'Surface fuel load in 0-1m height (elevated surface)',
        'units': 'kg/m²',
        'na_policy': 'replace',
        'na_justification': 'No surface fuel = zero fuel load',
        'expected_range': (0.0, 20.0),
        'replace_value': 0.0,
    },

    20: {
        'name': 'FMA',
        'source_band': 21,
        'description': 'Fuel mass area (for bulk density calculation)',
        'units': 'g/m²',
        'na_policy': 'replace',
        'na_justification': 'No fuel = zero fuel mass',
        'expected_range': (0.0, 10000.0),
        'replace_value': 0.0,
    },

    21: {
        'name': 'Cover',
        'source_band': 23,  # Source band 22 (date) was skipped
        'description': 'Total vegetation cover (first returns above 2m / total)',
        'units': 'fraction',
        'na_policy': 'replace',
        'na_justification': 'No vegetation = zero coverage',
        'expected_range': (0.0, 1.0),
        'replace_value': 0.0,
    },
}


def get_na_replacement_indices():
    """
    Get list of band indices (0-indexed) that should have NA → replace_value.

    Returns:
        list: Band indices where na_policy='replace'
    """
    return [idx for idx, config in BAND_CONFIG.items()
            if config['na_policy'] == 'replace']


def get_band_by_name(name):
    """
    Get band configuration by name.

    Args:
        name (str): Band name (e.g., 'Height', 'TFL', 'Cover')

    Returns:
        tuple: (band_index, config_dict) or (None, None) if not found
    """
    for idx, config in BAND_CONFIG.items():
        if config['name'] == name:
            return idx, config
    return None, None


def validate_band_config():
    """
    Validate the band configuration for consistency.

    Raises:
        ValueError: If configuration is invalid
    """
    # Check all 22 bands are defined
    if len(BAND_CONFIG) != 22:
        raise ValueError(f"Expected 22 bands, got {len(BAND_CONFIG)}")

    # Check indices are sequential 0-21
    expected_indices = set(range(22))
    actual_indices = set(BAND_CONFIG.keys())
    if expected_indices != actual_indices:
        missing = expected_indices - actual_indices
        extra = actual_indices - expected_indices
        raise ValueError(f"Index mismatch. Missing: {missing}, Extra: {extra}")

    # Check for duplicate source bands
    source_bands = [config['source_band'] for config in BAND_CONFIG.values()]
    if len(source_bands) != len(set(source_bands)):
        duplicates = [b for b in source_bands if source_bands.count(b) > 1]
        raise ValueError(f"Duplicate source bands: {set(duplicates)}")

    # Check replace_value consistency
    for idx, config in BAND_CONFIG.items():
        if config['na_policy'] == 'replace' and config['replace_value'] is None:
            raise ValueError(f"Band {idx} ({config['name']}): na_policy='replace' but replace_value=None")
        if config['na_policy'] == 'keep' and config['replace_value'] is not None:
            raise ValueError(f"Band {idx} ({config['name']}): na_policy='keep' but replace_value={config['replace_value']}")

    return True


def print_band_summary():
    """Print a summary of the band configuration."""
    print("=" * 100)
    print("FUEL METRICS BAND CONFIGURATION")
    print("=" * 100)
    print(f"{'Idx':<4} {'Source':<7} {'Name':<20} {'Units':<12} {'NA Policy':<10} {'Range':<15}")
    print("-" * 100)

    for idx in range(22):
        config = BAND_CONFIG[idx]
        range_str = f"{config['expected_range'][0]:.1f}-{config['expected_range'][1]:.1f}"
        na_policy_str = f"{config['na_policy']}" + (f"→{config['replace_value']}" if config['replace_value'] is not None else "")

        print(f"{idx:<4} {config['source_band']:<7} {config['name']:<20} {config['units']:<12} "
              f"{na_policy_str:<10} {range_str:<15}")

    print("\n" + "=" * 100)
    print(f"Bands with NA → 0 replacement: {len(get_na_replacement_indices())}/22")
    print(f"Indices: {get_na_replacement_indices()}")
    print("=" * 100)


if __name__ == '__main__':
    # Validate configuration
    try:
        validate_band_config()
        print("✓ Band configuration is valid\n")
    except ValueError as e:
        print(f"✗ Invalid configuration: {e}\n")
        exit(1)

    # Print summary
    print_band_summary()

    # Example usage
    print("\n" + "=" * 100)
    print("EXAMPLE: Target Bands")
    print("=" * 100)
    for name in ['Height', 'TFL', 'Cover']:
        idx, config = get_band_by_name(name)
        if idx is not None:
            print(f"\n{name} (index {idx}):")
            print(f"  Source band: {config['source_band']}")
            print(f"  Description: {config['description']}")
            print(f"  Units: {config['units']}")
            print(f"  NA policy: {config['na_policy']}")
            if config['na_policy'] == 'replace':
                print(f"  Replace value: {config['replace_value']}")
            print(f"  Justification: {config['na_justification']}")
            print(f"  Expected range: {config['expected_range']}")
