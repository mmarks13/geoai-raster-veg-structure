# Laguna Fuel-Treatment Priority — Data Note (draft)

_Prepared 2026-06-25_

**File:** `Laguna_fuel_priority_and_structure.tif` — a single multiband GeoTIFF.

## What this is

A map of **where to focus fuel-treatment effort** to reduce crown-fire potential
across the Laguna project area, together with the underlying vegetation-structure
measures it is built from — all in one file. It is a **relative, structure-based
prioritization (a planning aid), not a calibrated fire-behavior model.** Values
rank locations against each other *within this project area only.*

**Please read this up front:** this is **my best current estimate of treatment
priority given the data I have right now, and it is based on vegetation structure
alone.** It is one input to inform planning, not a standalone decision — see
*What this does and doesn't account for* below.

## How this was made: training data → predicted bands → priority

The end-to-end flow:

> **dense drone-LiDAR "ground truth" + public 3DEP LiDAR & NAIP imagery → trained
> model → three predicted structure bands → combined into the priority ranking**

1. **Training (done once, at other sites).** At locations flown with dense,
   drone-mounted LiDAR, a machine-learning model learned how fine-scale canopy
   structure relates to two cheap, widely available inputs: sparse public USGS
   3DEP LiDAR and high-resolution NAIP aerial imagery (sub-meter, color +
   near-infrared).
2. **Prediction (here, at Laguna).** Using *only* those public inputs — no drone
   LiDAR exists for Laguna — the model predicts three vegetation-structure metrics
   on a 2 m grid, following the standardized framework of Moudrý et al. (2023):
   **canopy cover** (fraction of returns above 3 m), **mid-story density**
   (proportion of vegetation returns between 1–3 m), and **foliage height
   diversity** (Shannon–Wiener diversity of returns across 0–25 m height layers).
   These are bands 3–5 in the file.
3. **Priority (here).** The predicted bands are combined into the priority
   (bands 1–2): a location ranks high where there is **enough canopy** to carry a
   crown fire, a **mid-story ladder** into that canopy, and **continuous fuel** in
   the surrounding ~40 m — as canopy or as dense brush. A closed canopy with no
   ladder still ranks, just below canopy-plus-ladder. (Foliage height diversity is
   carried for context but not used in the priority — it tracks canopy cover too
   closely to add independent signal.)

## The five bands

The file is self-labeling — each band carries its description, so it names itself
when loaded in ArcGIS or QGIS.

1. **Fuel-treatment priority (0–1)** — a within-forest percentile. **0.90 means
   higher priority than 90 % of the forested area.** Pick your own
   "treat-the-worst-X %" cutoff.
2. **Treatment tier (1–5)** — equal fifths of the forest; **Tier 5 = the
   highest-priority 20 %.** For quick triage.
3. **Canopy cover (0–1)** — fraction of returns above 3 m (overstory cover).
4. **Mid-story density (0–1)** — proportion of *vegetation* returns 1–3 m (the
   ladder layer).
5. **Foliage height diversity** — vertical layering. Context only; not used in the
   priority.

Bands 1–2 are populated **only where there is forest** (canopy cover ≥ 25 %); open
ground, meadow, and shrub-only areas are left blank, because this metric is about
crown-fire treatment *in forest*. Bands 3–5 are populated across the whole area.
About **3,800 acres** of forest are ranked.

## What this does — and doesn't — account for

The priority reflects **vegetation structure only.** It deliberately does **not**
(yet) account for many things that also drive real treatment decisions, including:
terrain (slope, aspect, position on slope); weather and wind; surface and
dead-and-down fuels; live fuel moisture; fire and treatment history; ignition
likelihood and access; values at risk (communities, infrastructure, watersheds);
land ownership and management constraints; and treatment cost or feasibility.
Use it **alongside** local knowledge and operational judgment, not in place of them.

## Please also keep in mind

- **Relative, not absolute.** A Tier 5 here is not comparable to a Tier 5 from
  another project area.
- **Modeled, not field-measured.** Structure is predicted from public inputs, not
  field cruise data. Expect error at the pixel level — **read it at stand scale,
  not single 2 m pixels.**
- **Ladder fuel is the weakest ingredient.** Airborne LiDAR sees poorly *under* a
  closed canopy, so treat that contribution as a nudge, not a measurement.
- **The 25 % "forest" cutoff is a choice;** moving it changes what gets ranked.

## Format

GeoTIFF (Cloud-Optimized), **2 m pixels, EPSG:32611 (UTM Zone 11N)**, NoData = NaN.
Opens directly in ArcGIS Pro, ArcGIS Online, and QGIS. To symbolize: use **band 1**
for a continuous priority surface, or **band 2** for the 1–5 treatment tiers.
