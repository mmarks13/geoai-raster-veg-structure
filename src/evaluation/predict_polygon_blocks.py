#!/usr/bin/env python3
"""
predict_polygon_blocks.py — block-tiled driver for large AOIs.

Splits a large AOI into a grid of square blocks (default 1 km) and runs the validated
single-AOI pipeline (src/evaluation/predict_polygon.py) on each block independently, then
mosaics the per-block Cloud-Optimized GeoTIFFs into one map. Designed for AOIs far larger
than a single in-memory run can handle (e.g. the full ~53 km² Laguna project area ≈ 830k
tiles, where keeping inference_ready AOI-wide would need ~366 GB).

Why blocks:
- Bounds 3DEP `hag_delaunay` to ~one site's worth of points per block (no whole-AOI OOM).
- Bounds combine/preprocess memory (a block's tiles, not the whole AOI).
- Each block is independent → resumable (skip finished blocks) and GPU-parallel.
- Per-block `--cleanup` keeps peak disk low (only small output COGs are retained).

Each block reuses predict_polygon.py unchanged: the block square is just the existing
`--clip-box-corner/--clip-box-size` mechanism (interior blocks clip to the full square,
edge blocks to polygon∩square). Blocks run up to `--num-gpus` at a time, one GPU each.

Example (terminal-restart safe; resumable):
    nohup /home/jovyan/geoai_env/bin/python -u \
        src/evaluation/predict_polygon_blocks.py \
        --polygon data/raw/inference_aoi/LagunaProjectArea.zip \
        --name Laguna_full --block-size 1000 --mc-passes 1 20 \
        --threads 4 --num-gpus 3 > laguna_full.log 2>&1 &
"""

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import geopandas as gpd
from shapely.geometry import box
from shapely.ops import unary_union

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation import predict_polygon as pp  # noqa: E402  (reuse helpers)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("predict_polygon_blocks")

PYTHON = sys.executable
OUT_ROOT = _PROJECT_ROOT / "data" / "output" / "polygon_predictions"


# ---------------------------------------------------------------------------
# Block grid
# ---------------------------------------------------------------------------
class Block:
    """A single block: grid indices, name, UTM geometry, and WGS84 NW corner."""

    def __init__(self, row: int, col: int, base_name: str, geom_utm, size_m: float):
        self.row = row
        self.col = col
        self.name = f"{base_name}_blk_{row:03d}_{col:03d}"
        self.geom = geom_utm  # block square in EPSG:32611
        self.size_m = size_m
        nw = gpd.GeoSeries(
            [box(geom_utm.bounds[0], geom_utm.bounds[3], geom_utm.bounds[0], geom_utm.bounds[3])],
            crs=pp.TARGET_CRS,
        ).to_crs(pp.WGS84).iloc[0]
        self.nw_lon = nw.centroid.x
        self.nw_lat = nw.centroid.y

    @property
    def out_dir(self) -> Path:
        return OUT_ROOT / self.name


def build_blocks(aoi_geom, base_name: str, size_m: float,
                 min_overlap_frac: float = 0.01) -> List[Block]:
    """Tile the AOI bbox into edge-to-edge square blocks; keep blocks meaningfully inside the AOI.

    Uses ceil (not //+1) so an exact-multiple bbox does not spawn a degenerate boundary
    row/col, and drops sliver blocks whose overlap with the AOI is below
    ``min_overlap_frac`` of a full block — both would otherwise fail the per-block clip
    (``apply_clip_box`` raises on a non-intersecting / empty clip).
    """
    import math
    minx, miny, maxx, maxy = aoi_geom.bounds
    min_overlap_area = min_overlap_frac * size_m * size_m
    blocks: List[Block] = []
    ny = math.ceil((maxy - miny) / size_m)
    nx = math.ceil((maxx - minx) / size_m)
    for r in range(ny):
        y0 = maxy - (r + 1) * size_m
        y1 = maxy - r * size_m
        for c in range(nx):
            x0 = minx + c * size_m
            x1 = minx + (c + 1) * size_m
            sq = box(x0, y0, x1, y1)
            if sq.intersection(aoi_geom).area > min_overlap_area:
                blocks.append(Block(r, c, base_name, sq, size_m))
    return blocks


def expected_cogs(out_dir: Path, mc_passes: List[int]) -> List[Path]:
    """Per-block output COGs that signal completion (one per MC pass)."""
    multi = len(mc_passes) > 1
    return [out_dir / (f"mean_mc{mc}.tif" if multi else "mean.tif") for mc in mc_passes]


def block_done(block: Block, mc_passes: List[int]) -> bool:
    return all(p.exists() for p in expected_cogs(block.out_dir, mc_passes))


# ---------------------------------------------------------------------------
# Per-block command + scheduler
# ---------------------------------------------------------------------------
def block_command(block: Block, args) -> List[str]:
    cmd = [
        PYTHON, "-u", "src/evaluation/predict_polygon.py",
        "--polygon", args.polygon,
        "--name", block.name,
        "--clip-box-corner", f"{block.nw_lon:.8f}", f"{block.nw_lat:.8f}",
        "--clip-box-size", str(block.size_m),
        "--mc-passes", *[str(m) for m in args.mc_passes],
        "--threads", str(args.threads),
        "--tile-filter", args.tile_filter,
        "--model", args.model,
        "--band-config", args.band_config,
        "--device", "cuda",
        "--resume",
        "--cleanup",
    ]
    return cmd


def run_blocks(blocks: List[Block], args) -> dict:
    """Run blocks with up to num_gpus concurrent, one GPU each. Resumable + status-tracked."""
    status_path = OUT_ROOT / args.name / "blocks_status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def flush_status():
        status_path.write_text(json.dumps({
            "name": args.name,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total_blocks": len(blocks),
            "results": results,
        }, indent=2))

    pending = list(blocks)
    free_gpus = list(range(args.num_gpus))
    running: dict = {}  # Popen -> (block, gpu, logfh, start)

    # Pre-mark already-finished blocks (resume).
    for b in list(pending):
        if block_done(b, args.mc_passes):
            results[b.name] = {"status": "skipped (done)", "gpu": None, "seconds": 0}
            pending.remove(b)
    flush_status()
    logger.info("%d blocks total; %d already complete; %d to run.",
                len(blocks), len(results), len(pending))

    while pending or running:
        while pending and free_gpus:
            b = pending.pop(0)
            gpu = free_gpus.pop(0)
            b.out_dir.mkdir(parents=True, exist_ok=True)
            logfh = open(b.out_dir / "block.log", "w")
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
            logger.info("▶ launch %s on GPU %d", b.name, gpu)
            proc = subprocess.Popen(block_command(b, args), cwd=str(_PROJECT_ROOT),
                                    stdout=logfh, stderr=subprocess.STDOUT, env=env)
            running[proc] = (b, gpu, logfh, time.time())
            results[b.name] = {"status": "running", "gpu": gpu, "seconds": None}
            flush_status()

        for proc in list(running):
            rc = proc.poll()
            if rc is None:
                continue
            b, gpu, logfh, start = running.pop(proc)
            logfh.close()
            free_gpus.append(gpu)
            elapsed = round(time.time() - start, 1)
            ok = rc == 0 and block_done(b, args.mc_passes)
            results[b.name] = {
                "status": "ok" if ok else f"FAILED (rc={rc})",
                "gpu": gpu, "seconds": elapsed,
            }
            logger.info("%s %s in %.1f min (GPU %d)",
                        "✓" if ok else "✗", b.name, elapsed / 60, gpu)
            flush_status()

        if running and not (pending and free_gpus):
            time.sleep(5)

    return results


# ---------------------------------------------------------------------------
# Mosaic
# ---------------------------------------------------------------------------
def mosaic_blocks(blocks: List[Block], args) -> None:
    """Mosaic per-block COGs into one COG per MC pass (gdalbuildvrt → gdal_translate COG)."""
    gdalbuildvrt = shutil.which("gdalbuildvrt")
    gdal_translate = shutil.which("gdal_translate")
    if not (gdalbuildvrt and gdal_translate):
        logger.error("gdalbuildvrt/gdal_translate not on PATH; skipping mosaic.")
        return

    final_dir = OUT_ROOT / args.name
    final_dir.mkdir(parents=True, exist_ok=True)
    multi = len(args.mc_passes) > 1

    for mc in args.mc_passes:
        cog_name = f"mean_mc{mc}.tif" if multi else "mean.tif"
        inputs = [str(b.out_dir / cog_name) for b in blocks if (b.out_dir / cog_name).exists()]
        if not inputs:
            logger.warning("No block COGs found for %s; skipping.", cog_name)
            continue
        vrt = final_dir / f"_mosaic_mc{mc}.vrt"
        out_cog = final_dir / cog_name
        list_file = final_dir / f"_mosaic_mc{mc}_inputs.txt"
        list_file.write_text("\n".join(inputs) + "\n")
        logger.info("Mosaicking %d block COGs → %s", len(inputs), out_cog)
        subprocess.run([gdalbuildvrt, "-overwrite", "-input_file_list", str(list_file), str(vrt)],
                       cwd=str(_PROJECT_ROOT), check=True)
        # Stamp band descriptions onto the VRT *before* translate — gdal_translate copies
        # them to the COG, whereas editing a finished COG in place does not persist.
        _copy_band_descriptions(inputs[0], vrt)
        subprocess.run([gdal_translate, "-of", "COG", "-co", "COMPRESS=DEFLATE",
                        "-co", "OVERVIEW_RESAMPLING=AVERAGE", "-co", "BLOCKSIZE=512",
                        str(vrt), str(out_cog)], cwd=str(_PROJECT_ROOT), check=True)
        vrt.unlink(missing_ok=True)
        list_file.unlink(missing_ok=True)
        logger.info("Wrote mosaic: %s", out_cog)


def _copy_band_descriptions(src_cog: str, dst: Path) -> None:
    """Copy per-band Description metadata from a block COG onto another dataset (e.g. the VRT)."""
    try:
        import rasterio
        with rasterio.open(src_cog) as s:
            descs = list(s.descriptions)
        with rasterio.open(dst, "r+") as d:
            for i, desc in enumerate(descs, start=1):
                if desc:
                    d.set_band_description(i, desc)
    except Exception as e:  # noqa: BLE001 - non-fatal cosmetic step
        logger.warning("Could not copy band descriptions onto %s: %s", dst, e)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Block-tiled vegetation-structure map for a large AOI polygon.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--polygon", required=True, help="AOI polygon (GeoJSON/GPKG/SHP/zipped SHP).")
    p.add_argument("--name", default=None, help="Run name (default: polygon stem + '_full').")
    p.add_argument("--block-size", type=float, default=1000.0, help="Block side length (m).")
    p.add_argument("--mc-passes", type=int, nargs="+", default=[1, 20],
                   help="MC passes per block (deterministic + uncertainty).")
    p.add_argument("--threads", type=int, default=4, help="H5-build workers per block.")
    p.add_argument("--num-gpus", type=int, default=3, help="Concurrent blocks (one GPU each).")
    p.add_argument("--tile-filter", choices=["centroid", "contains", "intersects"],
                   default="intersects")
    p.add_argument("--model", default=pp.DEFAULT_MODEL)
    p.add_argument("--band-config", default=pp.DEFAULT_BAND_CONFIG)
    p.add_argument("--list-blocks", action="store_true",
                   help="Print the block grid (count, GPU est., disk est.) and exit.")
    p.add_argument("--no-mosaic", action="store_true",
                   help="Run blocks but skip the final mosaic step.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.name = args.name or f"{Path(args.polygon).stem}_full"

    gdf = pp.reproject_to_target(pp.load_polygon(args.polygon))
    aoi = unary_union(gdf.geometry.values)
    blocks = build_blocks(aoi, args.name, args.block_size)

    area_km2 = aoi.area / 1e6
    logger.info("AOI area %.1f km²; %d blocks of %.0f m; %d GPU(s); mc-passes %s; threads %d.",
                area_km2, len(blocks), args.block_size, args.num_gpus, args.mc_passes, args.threads)

    if args.list_blocks:
        done = sum(block_done(b, args.mc_passes) for b in blocks)
        # ~6.9 GB inference_ready + ~2.5 GB other per ~1 km² block; peak ≈ concurrent × ~10 GB.
        logger.info("Blocks already complete: %d / %d", done, len(blocks))
        logger.info("Est. peak disk ≈ %.0f GB (%d concurrent × ~10 GB) + small COGs.",
                    args.num_gpus * 10.0, args.num_gpus)
        for b in blocks[:8] + (blocks[-2:] if len(blocks) > 10 else []):
            logger.info("  %s  NW=(%.5f, %.5f)  done=%s",
                        b.name, b.nw_lon, b.nw_lat, block_done(b, args.mc_passes))
        if len(blocks) > 10:
            logger.info("  … %d blocks total", len(blocks))
        return

    t0 = time.time()
    results = run_blocks(blocks, args)
    n_ok = sum(1 for r in results.values() if r["status"] in ("ok", "skipped (done)"))
    n_fail = len(results) - n_ok
    logger.info("Blocks complete: %d ok, %d failed, in %.2f h.",
                n_ok, n_fail, (time.time() - t0) / 3600)

    if n_fail:
        logger.warning("%d blocks failed — see each block.log + blocks_status.json. "
                       "Re-run the same command to retry only the failed/missing blocks.", n_fail)

    if not args.no_mosaic:
        finished = [b for b in blocks if block_done(b, args.mc_passes)]
        mosaic_blocks(finished, args)

    logger.info("Done: %s", OUT_ROOT / args.name)


if __name__ == "__main__":
    main()
