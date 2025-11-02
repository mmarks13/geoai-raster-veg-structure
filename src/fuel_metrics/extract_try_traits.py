#!/usr/bin/env python3
"""
Extract LMA and Wood Density values from TRY Plant Trait Database.

This script processes TRY database exports to extract species-level mean trait values
for use in LidarForFuel fuel metrics calculations.

Usage:
    python extract_try_traits.py <input_file> <output_file>

Example:
    python extract_try_traits.py data/raw/plant_traits/TRY_plant_traits.txt \
                                  data/processed/fuel_metrics/species_traits_try.csv
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path


def extract_trait_values(input_file: str, output_file: str, error_risk_threshold: float = 4.0):
    """
    Extract and process LMA and Wood Density values from TRY database.

    Parameters
    ----------
    input_file : str
        Path to TRY database export file (tab-delimited text)
    output_file : str
        Path to output CSV file for processed trait values
    error_risk_threshold : float, optional
        Maximum ErrorRisk value to include (default: 4.0)
        ErrorRisk > 4 likely indicates data quality issues
    """

    print("="*80)
    print("TRY TRAIT EXTRACTION")
    print("="*80)
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Error risk threshold: {error_risk_threshold}")

    # Read TRY database (latin1 encoding per TRY documentation)
    print("\nReading TRY database...")
    df = pd.read_csv(input_file, sep='\t', encoding='latin1', low_memory=False)
    print(f"Total rows: {len(df):,}")

    # Species configuration
    # Key: display name, Value: species name or list of species to combine
    target_species = {
        'Ceanothus palmeri': 'Ceanothus palmeri',
        'Quercus agrifolia': 'Quercus agrifolia',
        'Quercus kelloggii': 'Quercus kelloggii',
        'Calocedrus decurrens': 'Calocedrus decurrens',
        'Pinus spp.': ['Pinus coulteri', 'Pinus jeffreyi'],  # Combined species
        'Eriogonum fasciculatum': 'Eriogonum fasciculatum'
    }

    # Extract trait data
    print("\n" + "="*80)
    print("EXTRACTING TRAIT DATA")
    print("="*80)

    # LMA/SLA traits (TraitID 3115, 3116, 3117)
    # All represent SLA variants with standardized unit mm²/mg
    lma_data = df[df['TraitID'].isin([3115, 3116, 3117])].copy()
    print(f"LMA/SLA observations: {len(lma_data)}")

    # Wood Density (TraitID 4)
    # Standardized unit: g/cm³
    wd_data = df[df['TraitID'] == 4].copy()
    print(f"Wood Density observations: {len(wd_data)}")

    # Convert units
    print("\n" + "="*80)
    print("UNIT CONVERSION")
    print("="*80)

    # LMA (g/m²) = 1000 / SLA (mm²/mg)
    # This is equivalent to: LMA (g/m²) = 1 / SLA (m²/g)
    lma_data['LMA_g_m2'] = 1000.0 / lma_data['StdValue']
    print("LMA: Converted from SLA (mm²/mg) to LMA (g/m²)")
    print(f"  Formula: LMA = 1000 / SLA")

    # WD (kg/m³) = WD (g/cm³) × 1000
    wd_data['WD_kg_m3'] = wd_data['StdValue'] * 1000.0
    print("WD: Converted from g/cm³ to kg/m³")
    print(f"  Formula: WD = value × 1000")

    # Filter by ErrorRisk
    print("\n" + "="*80)
    print("QUALITY FILTERING")
    print("="*80)

    print(f"Before filtering (ErrorRisk < {error_risk_threshold}):")
    print(f"  LMA observations: {len(lma_data)}")
    print(f"  WD observations: {len(wd_data)}")

    # Keep rows with ErrorRisk < threshold OR missing ErrorRisk (no standardization)
    lma_filtered = lma_data[(lma_data['ErrorRisk'].isna()) |
                            (lma_data['ErrorRisk'] < error_risk_threshold)].copy()
    wd_filtered = wd_data[(wd_data['ErrorRisk'].isna()) |
                          (wd_data['ErrorRisk'] < error_risk_threshold)].copy()

    print(f"After filtering:")
    print(f"  LMA observations: {len(lma_filtered)} ({len(lma_data) - len(lma_filtered)} removed)")
    print(f"  WD observations: {len(wd_filtered)} ({len(wd_data) - len(wd_filtered)} removed)")

    # Calculate species-level statistics
    print("\n" + "="*80)
    print("SPECIES-LEVEL STATISTICS")
    print("="*80)

    results = []

    for display_name, species_filter in target_species.items():
        # Handle combined species (e.g., Pinus spp.)
        if isinstance(species_filter, list):
            lma_species = lma_filtered[lma_filtered['AccSpeciesName'].isin(species_filter)]
            wd_species = wd_filtered[wd_filtered['AccSpeciesName'].isin(species_filter)]
            species_name = display_name
        else:
            lma_species = lma_filtered[lma_filtered['AccSpeciesName'] == species_filter]
            wd_species = wd_filtered[wd_filtered['AccSpeciesName'] == species_filter]
            species_name = display_name

        # Calculate LMA statistics
        if len(lma_species) > 0:
            lma_mean = lma_species['LMA_g_m2'].mean()
            lma_std = lma_species['LMA_g_m2'].std()
            lma_n = len(lma_species)
            lma_min = lma_species['LMA_g_m2'].min()
            lma_max = lma_species['LMA_g_m2'].max()

            # Get unique references
            lma_refs = lma_species['Reference'].dropna().unique()
            lma_source = f"{len(lma_refs)} studies"
        else:
            lma_mean = lma_std = lma_n = lma_min = lma_max = np.nan
            lma_source = "No data"

        # Calculate WD statistics
        if len(wd_species) > 0:
            wd_mean = wd_species['WD_kg_m3'].mean()
            wd_std = wd_species['WD_kg_m3'].std()
            wd_n = len(wd_species)
            wd_min = wd_species['WD_kg_m3'].min()
            wd_max = wd_species['WD_kg_m3'].max()

            # Get unique references
            wd_refs = wd_species['Reference'].dropna().unique()
            wd_source = f"{len(wd_refs)} studies"
        else:
            wd_mean = wd_std = wd_n = wd_min = wd_max = np.nan
            wd_source = "No data"

        results.append({
            'species': species_name,
            'lma_mean_g_m2': lma_mean,
            'lma_std_g_m2': lma_std,
            'lma_n': int(lma_n) if not np.isnan(lma_n) else 0,
            'lma_min_g_m2': lma_min,
            'lma_max_g_m2': lma_max,
            'lma_source': lma_source,
            'wd_mean_kg_m3': wd_mean,
            'wd_std_kg_m3': wd_std,
            'wd_n': int(wd_n) if not np.isnan(wd_n) else 0,
            'wd_min_kg_m3': wd_min,
            'wd_max_kg_m3': wd_max,
            'wd_source': wd_source
        })

        # Print summary
        print(f"\n{species_name}:")
        if not np.isnan(lma_mean):
            print(f"  LMA: {lma_mean:.1f} ± {lma_std:.1f} g/m² "
                  f"(n={int(lma_n)}, range: {lma_min:.1f}-{lma_max:.1f})")
        else:
            print(f"  LMA: NO DATA")

        if not np.isnan(wd_mean):
            print(f"  WD:  {wd_mean:.0f} ± {wd_std:.0f} kg/m³ "
                  f"(n={int(wd_n)}, range: {wd_min:.0f}-{wd_max:.0f})")
        else:
            print(f"  WD:  NO DATA")

    # Create output dataframe
    results_df = pd.DataFrame(results)

    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("RESULTS SAVED")
    print("="*80)
    print(f"Output file: {output_file}")
    print(f"Species with complete data: {(results_df['lma_n'] > 0) & (results_df['wd_n'] > 0).sum()}")
    print(f"Species missing LMA: {(results_df['lma_n'] == 0).sum()}")
    print(f"Species missing WD: {(results_df['wd_n'] == 0).sum()}")

    return results_df


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print(__doc__)
        print("\nError: Missing required arguments")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    # Optional: error risk threshold
    error_risk = float(sys.argv[3]) if len(sys.argv) > 3 else 4.0

    extract_trait_values(input_file, output_file, error_risk)
