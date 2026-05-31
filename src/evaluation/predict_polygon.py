#!/usr/bin/env python3
"""
predict_polygon.py — AOI polygon → vegetation-structure map.

Orchestration layer around the forest-plot inference chain. Takes an AOI polygon
(any CRS), reprojects to EPSG:32611, builds a 10 m tile grid, fetches NAIP optical
imagery and 3DEP HAG LiDAR for the AOI, runs the trained raster model with
MC-dropout, stitches the per-tile predictions into a 2 m raster, masks it to the
polygon, and writes a Cloud-Optimized GeoTIFF + per-band single-band GeoTIFFs +
a band-stack VRT + an auto-generated OUTPUT_README.md.

This mirrors the data path of scripts/evaluate_forest_plots.sh
(generate_training_data_raster → h5_chunk_loader → preprocess_forest_plots_for_inference
→ raster_inference → build_prediction_rasters), wrapping it as a single entry point.
See the "Planned: polygon-to-map inference pipeline" section of RUNNING_AND_MIGRATING.md.

Default model: data/output/pretrained_3band_naip/checkpoints/epoch_100.pth
(3-band NAIP-fused: canopy_cover, midstory_density, FHD). The checkpoint is
global-only with attr_dim=6 — all six 3DEP attributes flow end-to-end.

CRS contract: EPSG:32611 (WGS84 UTM 11N) everywhere downstream. Inputs in other
CRS (incl. NAD83 / EPSG:26911) are reprojected at the boundary with a warning.
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
import numpy as np
from shapely.geometry import box, mapping
from shapely.ops import unary_union

# Add project root to path for src imports (avoids PYTHONPATH requirement)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_polygon")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_CRS = "EPSG:32611"
WGS84 = "EPSG:4326"

DEFAULT_MODEL = "data/output/pretrained_3band_naip/checkpoints/epoch_100.pth"
DEFAULT_BAND_CONFIG = "src/evaluation/configs/raster/veg_structure_3band_v2.json"
TRAINING_STATS_DIR = "data/processed/model_data_veg_structure"

# Margin (meters) added around the AOI before fetching NAIP / 3DEP so that the
# 20 m concentric NAIP chips and edge tiles have full coverage.
FETCH_BUFFER_M = 30.0

# Rough Southern California training-domain envelope (WGS84). AOIs outside this
# are rejected (the model is not expected to generalize beyond it).
DOMAIN_BOUNDS_WGS84 = (-122.0, 32.0, -114.0, 36.5)  # minlon, minlat, maxlon, maxlat

# Multipolygon split threshold: components farther apart than this become separate outputs.
MULTIPOLYGON_SPLIT_M = 100.0

# Wide date window for the per-tile H5 build's STAC filtering. It must bracket both
# every downloaded NAIP year and the locally-built 3DEP STAC item (stamped with the
# STAC-build time, since the HAG COPC carries no acquisition date).
GEN_DATE_START = "2000-01-01"
GEN_DATE_END = "2035-12-31"

PYTHON = sys.executable  # run sub-steps under the same interpreter


# ---------------------------------------------------------------------------
# Timing instrumentation
# ---------------------------------------------------------------------------
# Each timed step appends a record here; written to <out_dir>/timing_log.{json,txt}
# at the end of each AOI (including on failure, so partial timings survive a crash).
_TIMINGS: List[dict] = []


@contextmanager
def timed(step_name: str):
    """Time a block and record (step, seconds, start) into the running timing log."""
    start = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    logger.info("⏱  START  %s", step_name)
    try:
        yield
    finally:
        elapsed = time.time() - start
        _TIMINGS.append({"step": step_name, "seconds": round(elapsed, 2), "start": start_iso})
        logger.info("⏱  DONE   %s  (%.2fs / %.2fmin)", step_name, elapsed, elapsed / 60.0)


def write_timing_log(out_dir: Path, name: str, status: str) -> None:
    """Write the accumulated per-step timings to JSON + a human-readable text log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = round(sum(t["seconds"] for t in _TIMINGS), 2)
    payload = {
        "name": name,
        "status": status,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "total_seconds": total,
        "total_minutes": round(total / 60.0, 2),
        "steps": list(_TIMINGS),
    }
    (out_dir / "timing_log.json").write_text(json.dumps(payload, indent=2))

    lines = [
        f"Timing log — {name}  [{status}]",
        f"Generated: {payload['generated']}",
        f"Total: {total:.1f}s ({payload['total_minutes']:.2f} min)",
        "",
        f"{'step':<34} {'seconds':>10} {'minutes':>9}   start",
        "-" * 78,
    ]
    for t in _TIMINGS:
        lines.append(
            f"{t['step']:<34} {t['seconds']:>10.2f} {t['seconds']/60.0:>9.2f}   {t['start']}"
        )
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<34} {total:>10.2f} {total/60.0:>9.2f}")
    (out_dir / "timing_log.txt").write_text("\n".join(lines) + "\n")
    logger.info("Wrote timing log: %s", out_dir / "timing_log.txt")


# ---------------------------------------------------------------------------
# Subprocess helper
# ---------------------------------------------------------------------------
def run_step(cmd: List[str], step_name: str) -> None:
    """Run a sub-pipeline command, streaming output; raise on non-zero exit. Timed."""
    logger.info("=" * 70)
    logger.info(f"STEP: {step_name}")
    logger.info("CMD: %s", " ".join(str(c) for c in cmd))
    logger.info("=" * 70)
    with timed(step_name):
        result = subprocess.run(cmd, cwd=str(_PROJECT_ROOT))
    if result.returncode != 0:
        raise RuntimeError(
            f"Step '{step_name}' failed (exit {result.returncode}). Command: "
            f"{' '.join(str(c) for c in cmd)}"
        )


# ---------------------------------------------------------------------------
# Polygon loading / clipping / domain validation
# ---------------------------------------------------------------------------
def load_polygon(polygon_path: str) -> gpd.GeoDataFrame:
    """Load an AOI polygon (shp / gpkg / geojson, optionally a zipped shapefile)."""
    p = Path(polygon_path)
    if not p.exists():
        raise FileNotFoundError(f"Polygon not found: {polygon_path}")
    read_path = f"zip://{p.resolve()}" if p.suffix.lower() == ".zip" else str(p)
    gdf = gpd.read_file(read_path)
    if gdf.empty:
        raise ValueError(f"Polygon file is empty: {polygon_path}")
    if gdf.crs is None:
        raise ValueError(
            f"Polygon {polygon_path} has no CRS. Inputs must declare a CRS "
            f"(this pipeline reprojects to {TARGET_CRS})."
        )
    return gdf


def reproject_to_target(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Reproject to EPSG:32611, warning loudly on any CRS/datum change."""
    if str(gdf.crs) != TARGET_CRS:
        logger.warning(
            "Reprojecting AOI from %s to %s. If the source datum is NAD83 "
            "(e.g. EPSG:26911), the ~1 m WGS84 datum shift is accepted as "
            "negligible against 2 m pixels.",
            gdf.crs, TARGET_CRS,
        )
        gdf = gdf.to_crs(TARGET_CRS)
    return gdf


def validate_domain(geom_wgs84_bounds: Tuple[float, float, float, float]) -> None:
    """Hard-error if the AOI bbox falls outside the SoCal training envelope."""
    minlon, minlat, maxlon, maxlat = geom_wgs84_bounds
    dminlon, dminlat, dmaxlon, dmaxlat = DOMAIN_BOUNDS_WGS84
    if (minlon < dminlon or maxlon > dmaxlon or minlat < dminlat or maxlat > dmaxlat):
        raise ValueError(
            f"AOI bbox (WGS84) {geom_wgs84_bounds} is outside the supported "
            f"Southern California training domain {DOMAIN_BOUNDS_WGS84}. "
            f"The model is not expected to generalize here."
        )


def apply_clip_box(
    aoi_geom, corner_lonlat: Optional[Tuple[float, float]], box_size_m: Optional[float]
):
    """Intersect the AOI with a fixed square anchored at a top-left WGS84 corner.

    The corner is the north-west corner; the box extends east (+x) and south (-y).
    Used for tiny smoke-test runs.
    """
    if corner_lonlat is None or box_size_m is None:
        return aoi_geom
    lon, lat = corner_lonlat
    # Reproject the corner to EPSG:32611.
    corner = gpd.GeoSeries([box(lon, lat, lon, lat).centroid], crs=WGS84).to_crs(TARGET_CRS)
    x_tl, y_tl = corner.iloc[0].x, corner.iloc[0].y
    clip = box(x_tl, y_tl - box_size_m, x_tl + box_size_m, y_tl)
    clipped = aoi_geom.intersection(clip)
    if clipped.is_empty:
        raise ValueError(
            f"Clip box (corner lon/lat={corner_lonlat}, size={box_size_m} m) does "
            f"not intersect the AOI polygon."
        )
    logger.info(
        "Applied clip box: corner UTM=(%.2f, %.2f), size=%.0f m, "
        "clipped area=%.1f m^2", x_tl, y_tl, box_size_m, clipped.area,
    )
    return clipped


def split_components(aoi_geom) -> List:
    """Apply the multipolygon-split rule.

    Returns a list of AOI geometries. If all polygon components are within
    MULTIPOLYGON_SPLIT_M of their nearest neighbor, returns a single unioned AOI;
    otherwise returns connected-component groups as separate AOIs.
    """
    if aoi_geom.geom_type == "Polygon":
        return [aoi_geom]
    parts = list(aoi_geom.geoms)
    if len(parts) == 1:
        return [parts[0]]

    # Union–find over components within the split distance.
    n = len(parts)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        parent[find(a)] = find(b)

    for i in range(n):
        for j in range(i + 1, n):
            if parts[i].distance(parts[j]) <= MULTIPOLYGON_SPLIT_M:
                union(i, j)

    groups: dict = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(parts[i])

    if len(groups) == 1:
        return [unary_union(parts)]
    logger.warning(
        "AOI has %d component groups farther than %.0f m apart; producing %d "
        "separate outputs.", len(groups), MULTIPOLYGON_SPLIT_M, len(groups),
    )
    return [unary_union(g) for g in groups.values()]


# ---------------------------------------------------------------------------
# Tile grid
# ---------------------------------------------------------------------------
def build_tile_grid(
    aoi_geom, name: str, work_dir: Path, stride: float, tile_filter: str
) -> Path:
    """Build the 10 m tile grid over the AOI bbox and filter it to the polygon.

    The shared grid helper only supports 'contains' filtering against the polygon
    it is given, so we feed it the AOI *bounding box* (keeping all interior tiles)
    and then apply the requested --tile-filter against the true AOI geometry here.
    Returns the path to the filtered tile GeoJSON.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Write a bbox-rectangle GPKG with the columns the helper expects.
    minx, miny, maxx, maxy = aoi_geom.bounds
    bbox_geom = box(minx, miny, maxx, maxy)
    helper_input = work_dir / f"{name}_bbox.gpkg"
    gpd.GeoDataFrame(
        {"Site": [name], "area_ha": [bbox_geom.area / 1e4], "plot_count": [0]},
        geometry=[bbox_geom],
        crs=TARGET_CRS,
    ).to_file(helper_input, driver="GPKG")

    grid_geojson = work_dir / f"{name}_tiles.geojson"
    run_step(
        [
            PYTHON, "src/data_prep/create_forest_plot_tile_grid.py",
            "--input", str(helper_input),
            "--output", str(grid_geojson),
            "--tile-size", "10.0",
            "--stride", str(stride),
            "--site", name,
        ],
        "build tile grid",
    )

    tiles = gpd.read_file(grid_geojson)
    logger.info("Grid helper produced %d tiles over the bbox.", len(tiles))

    # Filter tiles against the true AOI geometry using the chosen predicate.
    if tile_filter == "contains":
        keep = tiles.geometry.apply(aoi_geom.contains)
    elif tile_filter == "centroid":
        keep = tiles.geometry.centroid.apply(aoi_geom.contains)
    elif tile_filter == "intersects":
        keep = tiles.geometry.apply(aoi_geom.intersects)
    else:
        raise ValueError(f"Unknown tile_filter: {tile_filter}")

    filtered = tiles[keep].copy()
    if filtered.empty:
        raise ValueError(
            f"No tiles survived the '{tile_filter}' filter against the AOI. "
            f"The AOI may be smaller than a 10 m tile."
        )
    logger.info("Kept %d tiles after '%s' filter.", len(filtered), tile_filter)

    filtered_path = work_dir / f"{name}_tiles_filtered.geojson"
    filtered.to_file(filtered_path, driver="GeoJSON")
    return filtered_path


# ---------------------------------------------------------------------------
# External-data fetch (NAIP, 3DEP) + local STACs
# ---------------------------------------------------------------------------
def fetch_naip(name: str, bbox_wgs84: Tuple[float, float, float, float],
               start: str, end: str) -> Path:
    """Fetch NAIP into a per-AOI local STAC; skip if the catalog already exists."""
    out_dir = _PROJECT_ROOT / "data" / "stac" / f"naip_{name}"
    catalog = out_dir / "catalog.json"
    if catalog.exists():
        logger.info("NAIP STAC cache hit: %s", catalog)
        return catalog
    run_step(
        [
            PYTHON, "src/data_prep/make_local_naip_stac.py",
            "--bbox", *[str(v) for v in bbox_wgs84],
            "--start", start,
            "--end", end,
            "--output", str(out_dir),
        ],
        "fetch NAIP + build local STAC",
    )
    if not catalog.exists():
        raise RuntimeError(f"NAIP STAC was not created at {catalog}.")
    return catalog


def fetch_3dep(name: str, bbox_wgs84: Tuple[float, float, float, float],
               date_range: str) -> Path:
    """Fetch + process 3DEP HAG features; skip if the COPC already exists."""
    site_dir = _PROJECT_ROOT / "data" / "processed" / "3dep_hag_features" / name
    copc = site_dir / f"{name}_hag_features.copc.laz"
    if copc.exists():
        logger.info("3DEP HAG cache hit: %s", copc)
    else:
        bbox_str = ",".join(str(v) for v in bbox_wgs84)
        # Use --bbox=<value> so argparse does not treat the leading-minus longitude
        # as an option flag ("expected one argument").
        run_step(
            [
                PYTHON, "src/data_prep/download_and_process_3dep_sites.py",
                "--site", name,
                f"--bbox={bbox_str}",
                "--output-dir", str(site_dir),
            ],
            "fetch + process 3DEP HAG",
        )
        if not copc.exists():
            raise RuntimeError(f"3DEP HAG COPC was not created at {copc}.")

    # Build a per-AOI local 3DEP STAC (input dir scoped to this site only).
    stac_out = _PROJECT_ROOT / "data" / "stac" / f"3dep_hag_{name}"
    catalog = stac_out / "catalog.json"
    if catalog.exists():
        logger.info("3DEP STAC cache hit: %s", catalog)
        return catalog
    run_step(
        [
            PYTHON, "src/data_prep/make_local_3dep_stac.py",
            "--mode", "processed",
            "--input-dir", str(site_dir),
            "--output", str(stac_out),
        ],
        "build local 3DEP HAG STAC",
    )
    if not catalog.exists():
        raise RuntimeError(f"3DEP STAC was not created at {catalog}.")
    return catalog


# ---------------------------------------------------------------------------
# Per-tile data build → inference-ready tiles
# ---------------------------------------------------------------------------
def build_inference_tiles(
    name: str, tiles_geojson: Path, dep_stac: Path, naip_stac: Path,
    work_dir: Path, min_dep_points: int, resume: bool = False, threads: int = 2,
) -> Path:
    """Run H5 build → combine → preprocess; return the inference-ready .pt path.

    Uses --strict-attributes on the H5 build so a missing HAG / geometry attribute
    is a hard error (map products must not silently substitute raw Z or neutral
    geometry). All six 3DEP attributes are preserved end-to-end. When resume=True,
    the H5 build skips tiles already present in the chunk dir (for long runs).
    ``threads`` controls the H5-build process pool (the only parallel data-prep step).
    """
    chunks_dir = work_dir / f"training_data_chunks_{name}"
    # generate_training_data_raster filters BOTH the local NAIP and 3DEP STACs by a
    # single [start_date, end_date] window. The 3DEP STAC item is stamped with the
    # STAC-build time (the COPC carries no acquisition date), so the window must
    # extend past "now"; a wide window keeps every downloaded NAIP year as well.
    gen_cmd = [
        PYTHON, "src/data_prep/generate_training_data_raster.py",
        "--tiles_geojson", str(tiles_geojson),
        "--outdir", str(chunks_dir),
        "--dep_stac_source", str(dep_stac),
        "--naip_stac_source", str(naip_stac),
        "--start_date", GEN_DATE_START,
        "--end_date", GEN_DATE_END,
        "--threads", str(threads),
        "--skip-uav-lidar",
        "--strict-attributes",
    ]
    if resume:
        gen_cmd.append("--resume")
    run_step(gen_cmd, "build per-tile H5 chunks")

    combined = work_dir / f"combined_{name}.pt"
    run_step(
        [
            PYTHON, "src/data_prep/h5_chunk_loader.py",
            "--input_dir", str(chunks_dir),
            "--output_path", str(combined),
        ],
        "combine H5 chunks",
    )

    prep_dir = work_dir / "inference_ready"
    run_step(
        [
            PYTHON, "src/data_prep/preprocess_forest_plots_for_inference.py",
            "--pt-file", str(combined),
            "--training-stats-dir", TRAINING_STATS_DIR,
            "--output-dir", str(prep_dir),
            "--min-dep-points", str(min_dep_points),
            "--precision", "32",
        ],
        "preprocess to inference-ready tiles",
    )

    ready = prep_dir / "precomputed_forest_plot_tiles_32bit.pt"
    if not ready.exists():
        raise RuntimeError(f"Inference-ready tiles were not created at {ready}.")
    return ready


def stamp_site_name(ready_pt: Path, name: str) -> None:
    """Set site_name=<name> on every tile so build_prediction_rasters groups them
    into a single named output (avoids the 'unknown' → field-plot spatial join)."""
    import torch
    tiles = torch.load(ready_pt, weights_only=False)
    for tile in tiles:
        tile["site_name"] = name
    torch.save(tiles, ready_pt)
    logger.info("Stamped site_name='%s' on %d tiles.", name, len(tiles))


# ---------------------------------------------------------------------------
# Inference + stitching
# ---------------------------------------------------------------------------
def run_inference(
    ready_pt: Path, model: str, band_config: str, fuel_stats: str,
    preds_dir: Path, mc_samples: int, batch_size: int,
    multi_gpu: bool, num_gpus: Optional[int], device: str,
) -> None:
    """Run MC-dropout inference via raster_inference.py."""
    cmd = [
        PYTHON, "src/evaluation/raster_inference.py",
        "--checkpoint", model,
        "--input", str(ready_pt),
        "--output", str(preds_dir),
        "--band-config", band_config,
        "--fuel-stats", fuel_stats,
        "--mc-samples", str(mc_samples),
        "--batch-size", str(batch_size),
    ]
    if multi_gpu:
        cmd.append("--multi-gpu")
        if num_gpus:
            cmd += ["--num-gpus", str(num_gpus)]
    else:
        cmd += ["--device", device]
    run_step(cmd, "MC-dropout inference")


def _latest(preds_dir: Path, pattern: str, exclude: Optional[str] = None) -> Optional[Path]:
    files = sorted(preds_dir.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    if exclude:
        files = [f for f in files if exclude not in f.name]
    return files[0] if files else None


def dedup_predictions_csv(preds_dir: Path) -> Tuple[Path, Path, Optional[Path]]:
    """Find the latest prediction outputs and write a tile_id-deduplicated CSV.

    DistributedSampler (drop_last=False) can duplicate tiles when the tile count
    is not divisible by the GPU count; the .pt dict dedups by key but the merged
    CSV does not. We drop duplicate tile_id rows before stitching.
    """
    import pandas as pd
    pt = _latest(preds_dir, "forest_plot_predictions_*.pt", exclude="_std_")
    csv = _latest(preds_dir, "forest_plot_predictions_*.csv")
    std_pt = _latest(preds_dir, "forest_plot_predictions_std_*.pt")
    if pt is None or csv is None:
        raise RuntimeError(f"Prediction files not found in {preds_dir}.")

    df = pd.read_csv(csv)
    before = len(df)
    df = df.drop_duplicates(subset="tile_id", keep="first")
    if len(df) != before:
        logger.info("Dropped %d duplicate tile_id rows from predictions CSV.", before - len(df))
    deduped_csv = preds_dir / "predictions_deduped.csv"
    df.to_csv(deduped_csv, index=False)
    return pt, deduped_csv, std_pt


def stitch_rasters(
    pt: Path, csv: Path, std_pt: Optional[Path], rasters_dir: Path, name: str,
) -> Tuple[Path, Optional[Path]]:
    """Stitch per-tile predictions into per-AOI mean (+ std) GeoTIFFs."""
    cmd = [
        PYTHON, "src/evaluation/build_prediction_rasters.py",
        "--predictions-pt", str(pt),
        "--predictions-csv", str(csv),
        "--output-dir", str(rasters_dir),
        "--crs", TARGET_CRS,
    ]
    if std_pt is not None:
        cmd += ["--predictions-std-pt", str(std_pt)]
    run_step(cmd, "stitch per-band rasters")

    mean_tif = rasters_dir / f"{name}_predictions_raster.tif"
    std_tif = rasters_dir / f"{name}_predictions_std_raster.tif"
    if not mean_tif.exists():
        raise RuntimeError(f"Stitched mean raster not found at {mean_tif}.")
    return mean_tif, (std_tif if std_tif.exists() else None)


# ---------------------------------------------------------------------------
# Postprocess: mask → interleave → COG → per-band TIFFs + VRT + README
# ---------------------------------------------------------------------------
def _unit_str(band: dict) -> str:
    """Human-readable unit string for a band, derived from its model_units."""
    mu = band.get("model_units", "")
    if mu == "fraction":
        return "fraction (0-1)"
    if mu == "normalized":
        return "normalized index"
    return mu or "value"


def _band_labels(band_config_path: str) -> List[Tuple[str, str, str]]:
    """Return [(short_name, display_name, units), ...] in output order from the band config."""
    with open(band_config_path) as f:
        cfg = json.load(f)
    bands = sorted(cfg["bands"], key=lambda b: b["output_index"])
    return [(b["name"], b.get("display_name", b["name"]), _unit_str(b)) for b in bands]


def finalize_outputs(
    mean_tif: Path, std_tif: Optional[Path], aoi_geom, out_dir: Path,
    name: str, band_config_path: str, invocation: dict, tag: Optional[str] = None,
) -> Path:
    """Mask to polygon, interleave mean/std bands, write COG + per-band TIFFs + VRT + README.

    When ``tag`` is given (e.g. "mc20"), outputs are suffixed so multiple MC passes do not
    collide: ``mean_<tag>.tif`` and ``per_band_<tag>/``. Returns the COG path.
    """
    import rasterio
    from rasterio.features import geometry_mask

    suffix = f"_{tag}" if tag else ""
    out_dir.mkdir(parents=True, exist_ok=True)
    per_band_dir = out_dir / f"per_band{suffix}"
    per_band_dir.mkdir(parents=True, exist_ok=True)

    labels = _band_labels(band_config_path)

    with rasterio.open(mean_tif) as src:
        mean = src.read()  # [n_bands, H, W]
        profile = src.profile
        transform = src.transform
        crs = src.crs
        height, width = src.height, src.width

    if str(crs) != TARGET_CRS:
        raise ValueError(f"Stitched raster CRS {crs} != {TARGET_CRS}; refusing to proceed.")

    n_bands = mean.shape[0]
    if n_bands != len(labels):
        raise ValueError(
            f"Stitched raster has {n_bands} bands but band config '{band_config_path}' "
            f"describes {len(labels)}."
        )

    std = None
    if std_tif is not None:
        with rasterio.open(std_tif) as s:
            std = s.read()

    # Polygon mask (pixel-center semantics on the 2 m grid).
    poly_mask = geometry_mask(
        [mapping(aoi_geom)], out_shape=(height, width), transform=transform,
        all_touched=False, invert=False,  # True where OUTSIDE polygon
    )

    # Interleave [b0_mean, b0_std, b1_mean, b1_std, ...]; apply NoData outside polygon.
    has_std = std is not None
    out_count = n_bands * 2 if has_std else n_bands
    stack = np.full((out_count, height, width), np.nan, dtype=np.float32)
    descriptions: List[str] = []
    k = 0
    for b in range(n_bands):
        short, disp, units = labels[b]
        band = mean[b].copy()
        band[poly_mask] = np.nan
        stack[k] = band
        descriptions.append(f"{disp} ({units})")
        k += 1
        if has_std:
            sband = std[b].copy()
            sband[poly_mask] = np.nan
            stack[k] = sband
            descriptions.append(f"{disp} MC-std ({units})")
            k += 1

    # --- Cloud-Optimized GeoTIFF (interleaved bands) ---
    cog_path = out_dir / f"mean{suffix}.tif"
    cog_profile = {
        "driver": "COG",
        "dtype": "float32",
        "count": out_count,
        "height": height,
        "width": width,
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "DEFLATE",
        "overview_resampling": "average",
        "blocksize": 512,
    }
    with rasterio.open(cog_path, "w", **cog_profile) as dst:
        dst.write(stack)
        for i, desc in enumerate(descriptions, start=1):
            dst.set_band_description(i, desc)
    logger.info("Wrote COG: %s", cog_path)

    # --- Per-band single-band GeoTIFFs ---
    single_profile = {
        "driver": "GTiff", "dtype": "float32", "count": 1,
        "height": height, "width": width, "crs": crs, "transform": transform,
        "nodata": np.nan, "compress": "DEFLATE", "tiled": True,
    }
    per_band_files: List[Path] = []
    for b in range(n_bands):
        short, disp, units = labels[b]
        mean_path = per_band_dir / f"{short}.tif"
        band = mean[b].copy()
        band[poly_mask] = np.nan
        with rasterio.open(mean_path, "w", **single_profile) as dst:
            dst.write(band, 1)
            dst.set_band_description(1, f"{disp} ({units})")
        per_band_files.append(mean_path)
        if has_std:
            std_path = per_band_dir / f"{short}_std.tif"
            sband = std[b].copy()
            sband[poly_mask] = np.nan
            with rasterio.open(std_path, "w", **single_profile) as dst:
                dst.write(sband, 1)
                dst.set_band_description(1, f"{disp} MC-std ({units})")
            per_band_files.append(std_path)

    # --- Band-stack VRT (stacks the single-band TIFFs; not a spatial mosaic) ---
    vrt_path = per_band_dir / f"{name}{suffix}.vrt"
    gdalbuildvrt = shutil.which("gdalbuildvrt")
    if gdalbuildvrt:
        run_step(
            [gdalbuildvrt, "-separate", "-overwrite", str(vrt_path),
             *[str(p) for p in per_band_files]],
            "build band-stack VRT",
        )
    else:
        logger.warning("gdalbuildvrt not found on PATH; skipping VRT generation.")

    _write_output_readme(out_dir, name, labels, has_std, invocation, band_config_path)
    logger.info("Finalized outputs in %s", out_dir)
    return cog_path


def _write_output_readme(
    out_dir: Path, name: str, labels, has_std: bool, invocation: dict,
    band_config_path: str,
) -> None:
    """Write OUTPUT_README.md describing bands, units, the MC-std caveat, and GIS steps."""
    lines: List[str] = []
    lines.append(f"# Vegetation-structure prediction — {name}\n")
    lines.append(
        "Generated by `src/evaluation/predict_polygon.py` from the raster "
        "vegetation-structure model (sparse 3DEP LiDAR + NAIP optical fusion).\n"
    )
    lines.append("## Files\n")
    lines.append("- `mean.tif` — multi-band Cloud-Optimized GeoTIFF, bands interleaved "
                 "as `[band_mean, band_std, ...]`.")
    lines.append("- `per_band/<band>.tif`" + (" and `per_band/<band>_std.tif`" if has_std else "") +
                 " — single-band GeoTIFFs.")
    lines.append(f"- `per_band/{name}.vrt` — band-stack VRT over the single-band TIFFs "
                 "(convenience stack, NOT a spatial mosaic).\n")

    lines.append("## Bands (COG order)\n")
    idx = 1
    for short, disp, units in labels:
        lines.append(f"{idx}. **{disp}** (`{short}`) — {units}.")
        idx += 1
        if has_std:
            lines.append(f"{idx}. **{disp} MC-std** (`{short}_std`) — {units}.")
            idx += 1
    lines.append("")

    lines.append("## Units\n")
    lines.append("Values are stored **model-native** per the band config: cover/density "
                 "bands as **fractions (0–1)** (multiply by 100 for percent via a QGIS/ArcGIS "
                 "symbology multiplier); diversity bands as a **normalized index**.\n")

    if has_std:
        lines.append("## Uncertainty caveat\n")
        lines.append(
            "The `*_std` bands are the standard deviation **across MC-Dropout samples "
            "only**. They are NOT a calibrated predictive interval: the inference path "
            "discards the model's heteroscedastic `log_var`, so the aleatoric term is "
            "not included. Full predictive variance "
            "(`sample_var + mean(exp(log_var))`) is a follow-up.\n"
        )

    lines.append("## Spatial reference\n")
    lines.append(f"- CRS: **{TARGET_CRS}** (WGS84 UTM 11N).")
    lines.append("- Pixel size: **2 m**; grid aligned to the 10 m tile origin.")
    lines.append("- NoData: NaN, masked to the AOI polygon "
                 "(`all_touched=False`, pixel-center semantics).\n")

    lines.append("## Provenance\n")
    lines.append("```json")
    lines.append(json.dumps(invocation, indent=2, default=str))
    lines.append("```")
    lines.append("")

    lines.append("## Import\n")
    lines.append("- **QGIS**: drag `mean.tif` in; band descriptions appear in Layer "
                 "Properties → Information.")
    lines.append("- **ArcGIS Pro**: Add Data → `mean.tif`; band descriptions show in the "
                 "raster's band list.")

    (out_dir / "OUTPUT_README.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Per-AOI orchestration
# ---------------------------------------------------------------------------
def cleanup_intermediates(name: str, work_dir: Path) -> None:
    """Delete heavy per-AOI intermediates after outputs are written (block runs).

    Removes the whole work/ dir (H5 chunks, combined.pt, inference_ready ~441 KB/tile,
    predictions) plus the per-AOI NAIP/3DEP caches. Keeps the output COGs + timing log.
    Essential for the full-Laguna block run, where keeping inference_ready AOI-wide
    would need ~366 GB.
    """
    targets = [
        work_dir,
        _PROJECT_ROOT / "data" / "stac" / f"naip_{name}",
        _PROJECT_ROOT / "data" / "stac" / f"3dep_hag_{name}",
        _PROJECT_ROOT / "data" / "processed" / "3dep_hag_features" / name,
    ]
    freed = 0
    for t in targets:
        if t.exists():
            for p in t.rglob("*"):
                if p.is_file():
                    freed += p.stat().st_size
            shutil.rmtree(t, ignore_errors=True)
    logger.info("Cleanup: removed intermediates for '%s' (~%.2f GB freed).", name, freed / 1e9)


def process_aoi(aoi_geom, name: str, args, fuel_stats: str) -> None:
    """Run the full pipeline for a single AOI geometry."""
    out_dir = _PROJECT_ROOT / "data" / "output" / "polygon_predictions" / name
    work_dir = out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    # WGS84 + buffered fetch bbox.
    aoi_series = gpd.GeoSeries([aoi_geom], crs=TARGET_CRS)
    bounds_wgs84 = tuple(aoi_series.to_crs(WGS84).total_bounds)
    validate_domain(bounds_wgs84)
    fetch_bbox_wgs84 = tuple(
        aoi_series.buffer(FETCH_BUFFER_M).to_crs(WGS84).total_bounds
    )

    mc_passes = args.mc_passes if args.mc_passes else [args.mc_samples]
    invocation = {
        "name": name,
        "model": args.model,
        "band_config": args.band_config,
        "mc_passes": mc_passes,
        "stride_m": args.stride,
        "tile_filter": args.tile_filter,
        "min_dep_points": args.min_dep_points,
        "threads": args.threads,
        "aoi_bounds_utm32611": [round(v, 2) for v in aoi_geom.bounds],
        "aoi_bounds_wgs84": [round(v, 6) for v in bounds_wgs84],
    }

    # Reset per-AOI timings; write the log even if a step raises (partial timings
    # are useful for diagnosing where a long run died).
    _TIMINGS.clear()
    status = "failed"
    try:
        # 1) tile grid
        tiles_geojson = build_tile_grid(aoi_geom, name, work_dir, args.stride, args.tile_filter)
        # 2) NAIP
        naip_stac = fetch_naip(name, fetch_bbox_wgs84, args.naip_start, args.naip_end)
        # 3) 3DEP HAG + STAC
        dep_stac = fetch_3dep(name, fetch_bbox_wgs84, args.dep_date_range)
        # 4) per-tile build → inference-ready (ONCE; reused by every MC pass)
        ready_pt = build_inference_tiles(
            name, tiles_geojson, dep_stac, naip_stac, work_dir, args.min_dep_points,
            resume=args.resume, threads=args.threads,
        )
        with timed("stamp site_name"):
            stamp_site_name(ready_pt, name)

        # 5-7) one inference → dedup → stitch → finalize per MC pass. Single MC pass
        # ⇒ no tag (mean.tif); multiple passes ⇒ tagged (mean_mc1.tif, mean_mc20.tif).
        for mc in mc_passes:
            tag = f"mc{mc}" if len(mc_passes) > 1 else None
            label = f"mc={mc}"
            preds_dir = work_dir / f"predictions_mc{mc}"
            with timed(f"inference {label}"):
                run_inference(
                    ready_pt, args.model, args.band_config, fuel_stats, preds_dir,
                    mc, args.batch_size, args.multi_gpu, args.num_gpus, args.device,
                )
            with timed(f"dedup {label}"):
                pt, csv, std_pt = dedup_predictions_csv(preds_dir)
            rasters_dir = work_dir / f"site_rasters_mc{mc}"
            mean_tif, std_tif = stitch_rasters(pt, csv, std_pt, rasters_dir, name)
            with timed(f"finalize {label}"):
                finalize_outputs(mean_tif, std_tif, aoi_geom, out_dir, name,
                                 args.band_config, invocation, tag=tag)

        status = "ok"
        logger.info("✓ Done: %s → %s", name, out_dir)
    finally:
        write_timing_log(out_dir, name, status)
        if args.cleanup and status == "ok":
            cleanup_intermediates(name, work_dir)


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------
def resolve_device(args) -> None:
    """Resolve compute mode in place; never blindly default to multi-GPU."""
    import torch
    cuda = torch.cuda.is_available()
    n = torch.cuda.device_count() if cuda else 0
    if args.multi_gpu:
        if not cuda:
            logger.warning("--multi-gpu requested but CUDA is unavailable; falling back to CPU.")
            args.multi_gpu = False
            args.device = "cpu"
        else:
            logger.info("Multi-GPU mode (%s GPUs available).", args.num_gpus or n)
        return
    if args.device is None:
        args.device = "cuda" if cuda else "cpu"
    if args.device.startswith("cuda") and not cuda:
        logger.warning("CUDA unavailable; using CPU.")
        args.device = "cpu"
    logger.info("Single-device mode: %s", args.device)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict a vegetation-structure map for an AOI polygon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--polygon", required=True,
                   help="AOI polygon (GeoJSON / GPKG / SHP, optionally a zipped shapefile).")
    p.add_argument("--name", default=None,
                   help="Short identifier for outputs + cache keys (default: polygon stem).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Model checkpoint.")
    p.add_argument("--band-config", default=DEFAULT_BAND_CONFIG,
                   help="Band-config JSON (must match the checkpoint).")
    p.add_argument("--mc-samples", type=int, default=30,
                   help="MC-dropout samples for a single pass (emits matching *_std bands).")
    p.add_argument("--mc-passes", type=int, nargs="+", default=None,
                   help="Run multiple MC passes on the SAME prepared tiles, e.g. "
                        "'--mc-passes 1 20' for a deterministic map + a 20-sample mean/std map. "
                        "Outputs are tagged (mean_mc1.tif, mean_mc20.tif). Overrides --mc-samples.")
    p.add_argument("--batch-size", type=int, default=64,
                   help="Inference batch size (800 is a benchmark target, not a safe default).")
    p.add_argument("--stride", type=float, default=8.0,
                   help="Tile-grid stride in meters (8.0 = 20%% overlap; 10.0 = none).")
    p.add_argument("--tile-filter", choices=["centroid", "contains", "intersects"],
                   default="intersects",
                   help="How tiles are masked against the polygon.")
    p.add_argument("--min-dep-points", type=int, default=50,
                   help="Minimum 3DEP points per tile; sparser tiles are dropped.")
    p.add_argument("--multi-gpu", action="store_true", help="Enable multi-GPU DDP inference.")
    p.add_argument("--num-gpus", type=int, default=None, help="GPU count for multi-GPU mode.")
    p.add_argument("--device", default=None, help="Single-device override (cuda/cpu; auto-detected).")
    p.add_argument("--naip-start", default="2016-01-01", help="NAIP search start date.")
    p.add_argument("--naip-end", default="2024-12-31", help="NAIP search end date.")
    p.add_argument("--dep-date-range", default="2015-01-01/2024-12-31",
                   help="3DEP STAC search date range.")
    p.add_argument("--resume", action="store_true",
                   help="Resume the per-tile H5 build, skipping tiles already in the chunk "
                        "dir (for large/interrupted runs).")
    p.add_argument("--threads", type=int, default=2,
                   help="Worker processes for the H5 build (the parallel data-prep step).")
    p.add_argument("--cleanup", action="store_true",
                   help="After outputs are written, delete the AOI's work/ dir + NAIP/3DEP "
                        "caches, keeping only the COGs. Required for the full-Laguna block run.")
    p.add_argument("--no-split", action="store_true",
                   help="Treat a multipolygon AOI as one output (skip the >100 m split rule). "
                        "Used by the block runner so a block never fans out into '<name>__NN' dirs.")
    # Smoke-test clip box.
    p.add_argument("--clip-box-corner", type=float, nargs=2, metavar=("LON", "LAT"),
                   default=None, help="WGS84 top-left corner to clip a fixed square (smoke test).")
    p.add_argument("--clip-box-size", type=float, default=None,
                   help="Side length (m) of the clip square anchored at --clip-box-corner.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_name = args.name or Path(args.polygon).stem

    for path, label in [(args.model, "model checkpoint"), (args.band_config, "band config")]:
        if not (_PROJECT_ROOT / path).exists() and not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    with open(args.band_config) as f:
        fuel_stats = json.load(f).get("stats_file")
    if not fuel_stats:
        raise ValueError(f"Band config {args.band_config} has no 'stats_file'.")

    resolve_device(args)

    # Load → reproject → clip → split.
    gdf = reproject_to_target(load_polygon(args.polygon))
    aoi = unary_union(gdf.geometry.values)
    if args.clip_box_corner is not None:
        aoi = apply_clip_box(aoi, tuple(args.clip_box_corner), args.clip_box_size)
    # --no-split keeps the (possibly multipolygon) AOI as a single output. The
    # multipolygon-split rule exists to avoid one huge mostly-NoData bbox spanning
    # far-apart AOIs; inside a bounded block (block runner) it is counterproductive —
    # it would emit '<name>__NN' dirs the block runner doesn't expect.
    aois = [aoi] if args.no_split else split_components(aoi)

    if len(aois) == 1:
        process_aoi(aois[0], base_name, args, fuel_stats)
    else:
        for i, geom in enumerate(aois):
            process_aoi(geom, f"{base_name}__{i:02d}", args, fuel_stats)

    logger.info("All AOIs complete.")


if __name__ == "__main__":
    main()
