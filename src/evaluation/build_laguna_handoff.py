#!/usr/bin/env python3
"""
Assemble a clean, self-contained handoff package for the Laguna fuel-treatment
project from the model outputs.

Inputs (already produced upstream):
  - fuel_priority.tif : 6-band index from fuel_treatment_priority.py
                        (band 1 = PRIMARY priority, percentile-ranked within forest)
  - mean_mc20.tif / per_band_mc20/ : the 3 raw model structure bands
                        (canopy cover, mid-story density, FHD)

Outputs (written to <out-dir>, default .../Laguna_full/handoff/):
  - Laguna_fuel_treatment_priority.tif : 2 bands
        band 1 = priority (0-1, within-forest percentile)
        band 2 = treatment tier (1-5; 5 = highest-priority 20% of forest)
  - Laguna_canopy_cover.tif        : raw model band, fraction 0-1
  - Laguna_midstory_density.tif    : raw model band, fraction 0-1
  - Laguna_fhd.tif                 : raw model band, foliage height diversity
  - Laguna_fuel_treatment_priority_overview.png : quick-look map (no GIS needed)
  - README.md                      : one-page plain-language explainer

This is packaging only -- it does not recompute the index. Re-run after any change
to fuel_priority.tif. Everything is reproducible and committed so it can run on
another machine.
"""

import argparse
import os
import shutil
from datetime import date

import numpy as np
import rasterio
from rasterio.enums import Resampling

# Index parameters (must match the values fuel_treatment_priority.py was run with;
# they are surfaced in the README so the handoff is self-documenting).
FOREST_FLOOR = 0.25       # canopy cover below which a pixel is not "forest" (not ranked)
CANOPY_SATURATION = 0.50  # canopy cover at/above which there is "enough" canopy to carry crown fire
CONTINUITY_WINDOW_M = 40  # diameter of the fire-spread neighborhood for horizontal continuity
N_TIERS = 5               # number of discrete treatment tiers
PIXEL_M = 2.0
ACRE_M2 = 4046.86


def _cog_profile(src_profile, count, dtype="float32"):
    prof = src_profile.copy()
    prof.update(driver="COG", count=count, dtype=dtype, nodata=np.nan,
                compress="DEFLATE", overview_resampling="average", blocksize=512)
    for k in ("interleave", "tiled", "blockxsize", "blockysize"):
        prof.pop(k, None)
    return prof


def build_priority_raster(fuel_priority_path, out_path):
    """Continuous priority (band 1) + 1-N_TIERS treatment tier (band 2)."""
    with rasterio.open(fuel_priority_path) as src:
        priority = src.read(1).astype(np.float32)  # band 1 = PRIMARY priority_ladder
        profile = src.profile
        prof = _cog_profile(profile, count=2)

        forest = np.isfinite(priority)
        # Equal-area quintiles of the within-forest percentile: tier 5 = top 20%.
        tier = np.full(priority.shape, np.nan, dtype=np.float32)
        t = np.floor(priority * N_TIERS) + 1.0
        t = np.clip(t, 1, N_TIERS)
        tier[forest] = t[forest]

        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(priority, 1)
            dst.write(tier, 2)
            dst.set_band_description(
                1, "Fuel-treatment priority (0-1; within-forest percentile; 1=highest)")
            dst.set_band_description(
                2, f"Treatment tier (1-{N_TIERS}; {N_TIERS}=highest-priority 20% of forest)")

    n_forest = int(np.isfinite(priority).sum())
    acres = n_forest * (PIXEL_M ** 2) / ACRE_M2
    return n_forest, acres


def copy_raw_band(src_band_path, out_path, description):
    """Re-emit a raw model band as a clean, described single-band COG."""
    with rasterio.open(src_band_path) as src:
        data = src.read(1).astype(np.float32)
        prof = _cog_profile(src.profile, count=1)
        with rasterio.open(out_path, "w", **prof) as dst:
            dst.write(data, 1)
            dst.set_band_description(1, description)


def build_combined_raster(priority_path, per_band_dir, out_path):
    """Single multiband COG with everything: priority + tier + the 3 raw measures.

    This is the one file to load and share in ArcGIS/QGIS. Band descriptions are
    embedded so each band is self-labeling. All inputs share the same 2 m grid.
    """
    measures = [
        ("canopy_cover", "Canopy cover: fraction of returns above 3 m (0-1)"),
        ("midstory_density", "Mid-story density: proportion of vegetation returns 1-3 m (0-1)"),
        ("fhd", "Foliage height diversity (context; not used in priority)"),
    ]
    with rasterio.open(priority_path) as src:
        priority = src.read(1).astype(np.float32)
        tier = src.read(2).astype(np.float32)
        prof = _cog_profile(src.profile, count=2 + len(measures))
        shape = priority.shape

    bands = [
        ("Fuel-treatment priority (0-1; within-forest percentile; 1=highest)", priority),
        (f"Treatment tier (1-{N_TIERS}; {N_TIERS}=highest-priority 20% of forest)", tier),
    ]
    for stem, desc in measures:
        with rasterio.open(os.path.join(per_band_dir, f"{stem}.tif")) as m:
            data = m.read(1).astype(np.float32)
        if data.shape != shape:
            raise ValueError(f"Grid mismatch: {stem} {data.shape} != priority {shape}")
        bands.append((desc, data))

    with rasterio.open(out_path, "w", **prof) as dst:
        for i, (desc, data) in enumerate(bands, start=1):
            dst.write(data, i)
            dst.set_band_description(i, desc)
    return len(bands)


def make_overview_png(priority_path, png_path):
    """Quick-look priority map with tier legend; downsampled so it opens fast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    # Downsample to ~1500 px wide for a light PNG.
    with rasterio.open(priority_path) as src:
        scale = max(1, int(src.width / 1500))
        out_h, out_w = src.height // scale, src.width // scale
        tier = src.read(2, out_shape=(out_h, out_w),
                        resampling=Resampling.nearest).astype(np.float32)
        bounds = src.bounds

    # 5 ascending tiers; non-forest is left blank (transparent over a light backdrop).
    colors = ["#2c7bb6", "#abd9e9", "#ffffbf", "#fdae61", "#d7191c"]  # low->high
    cmap = ListedColormap(colors)
    cmap.set_bad("#e8e8e8")  # non-forest / NoData
    norm = BoundaryNorm(np.arange(0.5, N_TIERS + 1.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(9, 9 * out_h / out_w), dpi=130)
    ax.imshow(np.ma.masked_invalid(tier), cmap=cmap, norm=norm,
              extent=[bounds.left, bounds.right, bounds.bottom, bounds.top])
    ax.set_title("Laguna Project Area — Fuel-Treatment Priority\n"
                 "(crown-fire potential, ranked within forest)", fontsize=12)
    ax.set_xlabel("Easting (m, UTM 11N / EPSG:32611)", fontsize=8)
    ax.set_ylabel("Northing (m)", fontsize=8)
    ax.tick_params(labelsize=7)

    labels = ["Tier 1 — lowest", "Tier 2", "Tier 3", "Tier 4", "Tier 5 — highest"]
    handles = [Patch(facecolor=c, edgecolor="none", label=l) for c, l in zip(colors, labels)]
    handles.append(Patch(facecolor="#e8e8e8", edgecolor="none", label="Not forest (not ranked)"))
    ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.9,
              title="Treatment tier", title_fontsize=8)
    fig.tight_layout()
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)


def write_readme(out_dir, n_forest, acres):
    """One-page, plain-language explainer for fire-literate ecologists / land managers."""
    nbhd = CONTINUITY_WINDOW_M
    readme = f"""# Laguna Project Area — Fuel-Treatment Priority (DRAFT)

_Prepared {date.today().isoformat()} · 2 m resolution · EPSG:32611 (UTM 11N)_

## What this is

A **relative ranking of where to focus fuel-treatment effort** to reduce
crown-fire potential across the Laguna project area. It is a **structure-based
proxy for prioritization, not a calibrated fire-behavior model.** Values rank
locations *against each other within this project area only.*

The ranking combines three things you already think about, derived from the
vegetation-structure model:

1. **Canopy fuel** — is there enough overstory canopy to carry a crown fire?
   (from *canopy cover*)
2. **Ladder fuel** — is there mid-story vegetation that can carry surface fire up
   into the canopy? (from *mid-story density*)
3. **Continuity** — is the surrounding ~{nbhd} m neighborhood also fuel-laden, as
   canopy *or* dense brush, so fire can move stand-to-stand? (neighborhood average)

A location ranks high when it has a real canopy, a ladder into it, **and** sits in
a continuous run of fuel. A closed canopy with no ladder still ranks, just below
canopy-plus-ladder. Open ground, meadow, and shrub-only areas (canopy cover below
**{FOREST_FLOOR:.0%}**) are treated as *not forest* and are left **unranked / blank** —
this metric is about crown-fire treatment in forest, not shrubland conversion.

## How to read it

- **`Laguna_fuel_treatment_priority.tif`, band 1 — priority (0 to 1).**
  A within-forest percentile. **0.90 means higher priority than 90 % of the
  forested area.** Pick your own "treat-the-worst-X %" cutoff.
- **band 2 — treatment tier (1 to 5).** Equal-size fifths of the forest:
  **Tier 5 = the highest-priority 20 % of forest**, Tier 1 = the lowest. Use tiers
  for quick triage; use the continuous band when you need finer separation.
- Ranked forest in this AOI: **{n_forest:,} pixels ≈ {acres:,.0f} acres.**
- Read it at **stand scale**, not single 2 m pixels — aggregate before deciding.

## The raw model inputs (provided so you can see what drives a pixel)

- **`Laguna_canopy_cover.tif`** — fraction of returns above 3 m (overstory cover), 0–1.
- **`Laguna_midstory_density.tif`** — proportion of *vegetation* returns between 1–3 m
  (ladder layer), 0–1.
- **`Laguna_fhd.tif`** — foliage height diversity (vertical layering). Provided for
  context; **not used** in the priority because it tracks canopy cover almost exactly
  and adds no independent ladder signal.

## Honest caveats (please read)

- **Relative, not absolute.** Ranks are internal to this AOI. A Tier 5 here is not
  comparable to a Tier 5 from another project area.
- **Modeled, not measured.** Structure is predicted from sparse public 3DEP LiDAR +
  NAIP imagery, not field cruise data. Expect error at the pixel level.
- **Ladder fuel is the weakest ingredient.** Airborne LiDAR sees poorly *under* a
  closed canopy, and mid-story density is a proportion, not an absolute. Treat the
  ladder contribution as a nudge, not a measurement.
- **Forest line is a choice.** "Forest" = canopy cover ≥ {FOREST_FLOOR:.0%}. Moving
  that line moves what gets ranked.

## Files

**Start here — one file has everything:**

| File | Contents |
|------|----------|
| `Laguna_fuel_priority_and_structure.tif` | **all 5 bands in one — load/share this.** band 1 priority (0–1), band 2 tier (1–5), band 3 canopy cover, band 4 mid-story density, band 5 FHD |

The same layers are also provided as separate single-band files if you prefer:

| File | Contents |
|------|----------|
| `Laguna_fuel_treatment_priority.tif` | band 1 priority (0–1), band 2 tier (1–5) |
| `Laguna_canopy_cover.tif` | raw model canopy cover (0–1) |
| `Laguna_midstory_density.tif` | raw model mid-story density (0–1) |
| `Laguna_fhd.tif` | raw model foliage height diversity |
| `Laguna_fuel_treatment_priority_overview.png` | quick-look map (no GIS needed) |

All rasters: GeoTIFF (COG), 2 m pixels, EPSG:32611, NoData = NaN. Opens in
QGIS/ArcGIS; band descriptions are embedded (each band is self-labeling).

## Method parameters (for the record)

Forest floor (canopy cover) = {FOREST_FLOOR:.2f}; canopy "saturation" = {CANOPY_SATURATION:.2f}
(more canopy beyond this adds no extra risk); continuity neighborhood ≈ {nbhd} m;
priority = percentile rank, within forest, of
`forest-suitability × (baseline + mid-story) × neighborhood fuel continuity`.
"""
    with open(os.path.join(out_dir, "README.md"), "w") as f:
        f.write(readme)


def main():
    p = argparse.ArgumentParser(description="Build the handoff package for the Laguna fuel-treatment project.")
    base = "data/output/polygon_predictions/Laguna_full"
    p.add_argument("--fuel-priority", default=f"{base}/fuel_priority.tif")
    p.add_argument("--per-band-dir", default=f"{base}/per_band_mc20")
    p.add_argument("--out-dir", default=f"{base}/handoff")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    prio_path = os.path.join(args.out_dir, "Laguna_fuel_treatment_priority.tif")
    n_forest, acres = build_priority_raster(args.fuel_priority, prio_path)
    print(f"priority raster -> {prio_path}  ({n_forest:,} forest px ~ {acres:,.0f} acres)")

    raw = [
        ("canopy_cover", "Laguna_canopy_cover.tif",
         "Canopy cover: fraction of returns above 3 m (0-1)"),
        ("midstory_density", "Laguna_midstory_density.tif",
         "Mid-story density: proportion of vegetation returns 1-3 m (0-1)"),
        ("fhd", "Laguna_fhd.tif",
         "Foliage height diversity (context only; not used in priority)"),
    ]
    for stem, out_name, desc in raw:
        src = os.path.join(args.per_band_dir, f"{stem}.tif")
        dst = os.path.join(args.out_dir, out_name)
        copy_raw_band(src, dst, desc)
        print(f"raw band -> {dst}")

    combined_path = os.path.join(args.out_dir, "Laguna_fuel_priority_and_structure.tif")
    n_bands = build_combined_raster(prio_path, args.per_band_dir, combined_path)
    print(f"combined  -> {combined_path}  ({n_bands}-band: priority, tier, canopy, mid-story, FHD)")

    png_path = os.path.join(args.out_dir, "Laguna_fuel_treatment_priority_overview.png")
    make_overview_png(prio_path, png_path)
    print(f"overview  -> {png_path}")

    write_readme(args.out_dir, n_forest, acres)
    print(f"readme    -> {os.path.join(args.out_dir, 'README.md')}")
    print(f"\nHandoff package ready: {args.out_dir}")


if __name__ == "__main__":
    main()
