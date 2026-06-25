#!/usr/bin/env python3
"""
Fuel-treatment prioritization index(es) from the vegetation-structure map.

Two indices are produced for comparison (Options 1 & 2). Both use the model's
predicted bands on their NATIVE (raw) scale — no percentile ranking of the inputs
and no "forest threshold" — because `canopy_cover` (fraction of ALL returns above
3 m) is already ~0 over open/treeless land, so open areas screen out on their own.
Only the FINAL priority is percentile-ranked, for "treat-the-worst-X%" display.

  Horizontal continuity (shared):
      H = focal mean of RAW canopy_cover over a ~40 m fire-spread neighborhood
          (crown-to-crown continuity).

  Option 1 — canopy crown-fuel (most defensible; drops mid-story & FHD):
      priority1 = percentile_rank( canopy_cover * H )
      "Where is the most contiguous canopy fuel?"

  Option 2 — canopy crown-fuel x canopy-conditioned ladder boost:
      V2 = canopy_cover * (1 + alpha * midstory_density)      # mid-story enters
      priority2 = percentile_rank( V2 * H )                   # only times canopy,
      so it can add credit ONLY where a canopy exists (open shrub/meadow -> ~0).
      NOTE: the ladder term is a weak, imperfect modifier — airborne LiDAR occludes
      sub-canopy structure and mid-story is a proportion-of-vegetation, so it is
      suppressed under closed canopy. Presented transparently as a minor boost.

Why NOT FHD: FHD ~ canopy_cover at r=+0.98 (redundant), and its canopy-removed
residual correlates NEGATIVELY with mid-story inside forest, so it cannot supply a
ladder signal. Omitted by design.

Robustness (matches the mosaic): interior inter-block seams are filled by
interpolation before computing and flagged low-confidence; AOI-edge pixels get a
`support` band (focal valid fraction).

Relative, structure-based hazard PROXY for prioritization, not a calibrated fire model.
"""

import argparse
import logging
import sys
import warnings
from typing import Dict, List, Tuple

import numpy as np
import rasterio
from rasterio.fill import fillnodata
from scipy.ndimage import binary_fill_holes, convolve
from scipy.stats import rankdata, spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger("fuel_priority")

MEAN_BANDS = {"canopy_cover": "canopy cover", "midstory_density": "mid-story density"}
STD_BANDS = {"canopy_cover_std": "canopy cover mc-std", "midstory_density_std": "mid-story density mc-std"}


def _match_bands(descriptions) -> Dict[str, int]:
    idx: Dict[str, int] = {}
    for i, desc in enumerate(descriptions, start=1):
        d = (desc or "").lower()
        for key, frag in STD_BANDS.items():      # most-specific (std) first
            if frag in d and key not in idx:
                idx[key] = i
                break
        else:
            for key, frag in MEAN_BANDS.items():
                if frag in d and key not in idx:
                    idx[key] = i
                    break
    missing = [k for k in list(MEAN_BANDS) + list(STD_BANDS) if k not in idx]
    if missing:
        raise ValueError(f"Could not locate bands {missing} from descriptions {list(descriptions)}")
    return idx


def percentile_rank(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile-rank valid pixels to [0,1] (ties averaged); NaN elsewhere."""
    out = np.full(arr.shape, np.nan, dtype=np.float32)
    vals = arr[mask].astype(np.float64)
    if vals.size == 0:
        return out
    ranks = rankdata(vals, method="average")
    out[mask] = ((ranks - 1) / (len(vals) - 1)).astype(np.float32) if len(vals) > 1 else 0.0
    return out


def focal_mean(v: np.ndarray, radius_px: int) -> Tuple[np.ndarray, np.ndarray]:
    """nan-aware circular focal mean; returns (mean, support_fraction)."""
    y, x = np.ogrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1]
    kernel = (x * x + y * y <= radius_px * radius_px).astype(np.float32)
    ksum = kernel.sum()
    valid = np.isfinite(v).astype(np.float32)
    filled = np.where(np.isfinite(v), v, 0.0).astype(np.float32)
    num = convolve(filled, kernel, mode="constant", cval=0.0)
    den = convolve(valid, kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, num / den, np.nan)
    return out.astype(np.float32), (den / ksum).astype(np.float32)


def fill_interior_seams(bands: List[np.ndarray], data_valid: np.ndarray):
    """Fill enclosed NoData (inter-block ~14 m seams) inside the AOI by interpolation."""
    aoi_mask = binary_fill_holes(data_valid)
    seam_mask = aoi_mask & ~data_valid
    filled = []
    for b in bands:
        f = fillnodata(np.where(data_valid, b, 0.0).astype(np.float32),
                       mask=data_valid.astype(np.uint8), max_search_distance=15.0)
        filled.append(np.where(aoi_mask, f, np.nan).astype(np.float32))
    return filled, aoi_mask, seam_mask


def main() -> None:
    p = argparse.ArgumentParser(description="Canopy-based fuel-treatment priority (Options 1 & 2).")
    p.add_argument("--input", default="data/output/polygon_predictions/Laguna_full/mean_mc20.tif")
    p.add_argument("--output", default="data/output/polygon_predictions/Laguna_full/fuel_priority.tif")
    p.add_argument("--window-radius-m", type=float, default=20.0,
                   help="Focal radius for horizontal continuity (default 20 m -> ~40 m neighborhood).")
    p.add_argument("--forest-floor", type=float, default=0.25,
                   help="Canopy cover below which it is NOT forest (suitability=0). Default 0.25.")
    p.add_argument("--canopy-saturation", type=float, default=0.50,
                   help="Canopy cover at/above which there is 'enough' canopy to carry crown fire "
                        "(suitability=1; more canopy adds no risk). Default 0.50.")
    p.add_argument("--crown-fuel-baseline", type=float, default=0.10,
                   help="Baseline crown-fuel risk for forest with no detectable ladder (the +beta in "
                        "beta+midstory), so closed canopy still ranks, just below forest-with-ladder.")
    p.add_argument("--continuity-midstory-weight", type=float, default=0.5,
                   help="Weight of mid-story (shrub/brush) relative to canopy in the neighborhood "
                        "fuel-presence used for horizontal continuity. 0 = canopy only; default 0.5 so "
                        "a forest stand adjacent to dense shrubland reads as more exposed.")
    args = p.parse_args()
    floor, sat, beta = args.forest_floor, args.canopy_saturation, args.crown_fuel_baseline
    gamma = args.continuity_midstory_weight
    assert floor < sat, "--forest-floor must be < --canopy-saturation"

    with rasterio.open(args.input) as src:
        bidx = _match_bands(src.descriptions)
        profile = src.profile.copy()
        res = src.res[0]
        canopy = src.read(bidx["canopy_cover"]).astype(np.float32)
        mid = src.read(bidx["midstory_density"]).astype(np.float32)
        c_std = src.read(bidx["canopy_cover_std"]).astype(np.float32)
        m_std = src.read(bidx["midstory_density_std"]).astype(np.float32)

    data_valid = np.isfinite(canopy) & np.isfinite(mid)
    canopy = np.clip(canopy, 0, None); mid = np.clip(mid, 0, None)

    # fill interior seams; flag filled pixels
    (canopy, mid, c_std, m_std), mask, seam_mask = fill_interior_seams(
        [canopy, mid, c_std, m_std], data_valid)
    logger.info("AOI pixels: %d (%.1f%%) | real: %d | seam-filled: %d",
                int(mask.sum()), 100 * mask.mean(), int(data_valid.sum()), int(seam_mask.sum()))

    # Horizontal continuity = focal mean of neighborhood FUEL PRESENCE (canopy + shrub).
    # fuel_presence = canopy_cover + gamma*midstory_density, so continuity is high where the
    # surrounding fuel is continuous as canopy (crown-to-crown) OR as dense mid-story (brush)
    # -- e.g. a forest stand abutting dense shrubland reads as more exposed than one next to
    # bare ground. Computed over the whole AOI (incl. non-forest neighbours) so forest pixels
    # "see" adjacent shrub fuel; only the final ranking is restricted to forest.
    radius_px = max(1, int(round(args.window_radius_m / res)))
    logger.info("Focal radius: %d px (%.0f m) ~%.0f m nbhd | forest_floor=%.2f canopy_sat=%.2f baseline=%.2f midstory_continuity_wt=%.2f",
                radius_px, radius_px * res, 2 * radius_px * res, floor, sat, beta, gamma)
    fuel_presence = np.where(mask, canopy + gamma * mid, np.nan).astype(np.float32)
    H, support = focal_mean(fuel_presence, radius_px)

    # Forest suitability: 0 below floor (not forest), ramps to 1 at saturation (enough
    # canopy to carry crown fire; beyond that, more canopy adds no risk -> non-monotonic).
    suit = np.clip((canopy - floor) / (sat - floor), 0.0, 1.0)

    # Rank WITHIN forest only (canopy >= floor); non-forest is NoData. Otherwise the ~70%
    # non-forest pixels (suit=0 -> priority 0) tie at the percentile midpoint and crush the
    # forest into the top of the 0-1 scale (no contrast). Ranking within forest also matches
    # the intent: "treat the worst X% OF FOREST".
    forest = mask & (suit > 0)
    logger.info("Forest pixels (canopy >= %.2f): %d (%.1f%% of AOI) -- ranked; non-forest = NoData",
                floor, int(forest.sum()), 100 * forest.mean())

    # PRIMARY: forest-suitability x (baseline crown fuel + ladder) x canopy continuity.
    Pl = np.where(forest, suit * (beta + mid) * H, np.nan).astype(np.float32)
    prio_ladder = percentile_rank(Pl, forest & np.isfinite(Pl))

    # BASELINE for comparison: canopy crown-fuel x continuity (monotonic in canopy; no ladder)
    Pc = np.where(forest, canopy * H, np.nan).astype(np.float32)
    prio_canopy = percentile_rank(Pc, forest & np.isfinite(Pc))

    # uncertainty (canopy+mid MC-std rank; seam-filled forced to 1) and edge support.
    # nanmean over all-NaN (non-mask) pixels is expected -> suppress its empty-slice warning.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        U = np.nanmean(np.stack([percentile_rank(c_std, mask), percentile_rank(m_std, mask)]), axis=0).astype(np.float32)
    U = np.where(seam_mask, 1.0, U)
    U = np.where(mask, U, np.nan).astype(np.float32)
    support = np.where(mask, support, np.nan).astype(np.float32)

    out_bands = [
        ("priority_ladder", prio_ladder, f"PRIMARY priority = pctile( suit(canopy;{floor:g},{sat:g}) x ({beta:g}+midstory) x canopy-continuity )"),
        ("priority_canopy", prio_canopy, "Comparison: canopy-only priority = pctile(canopy_cover x canopy-continuity)"),
        ("forest_suitability", np.where(mask, suit, np.nan).astype(np.float32), f"Forest suitability: 0 below canopy {floor:g}, ramps to 1 at {sat:g}"),
        ("fuel_continuity", np.where(mask, H, np.nan).astype(np.float32), f"Focal mean of fuel presence (canopy + {gamma:g}*midstory): canopy + shrub continuity"),
        ("uncertainty", U, "Uncertainty flag (mean ranked MC-std; seam-filled=1; 1=least confident)"),
        ("support", support, "Focal-window valid fraction (1=full neighborhood; low=AOI edge)"),
    ]
    stack = np.stack([b for _, b, _ in out_bands]).astype(np.float32)
    prof = profile.copy()
    prof.update(driver="COG", count=len(out_bands), dtype="float32", nodata=np.nan,
                compress="DEFLATE", overview_resampling="average", blocksize=512)
    for k in ("interleave", "tiled", "blockxsize", "blockysize"):
        prof.pop(k, None)
    with rasterio.open(args.output, "w", **prof) as dst:
        dst.write(stack)
        for i, (name, _, desc) in enumerate(out_bands, start=1):
            dst.set_band_description(i, desc)
    logger.info("Wrote %s", args.output)

    # --- ladder vs canopy-only: how much does the ladder formulation move things? ---
    vm = mask & np.isfinite(prio_ladder) & np.isfinite(prio_canopy)
    rho = spearmanr(prio_ladder[vm], prio_canopy[vm]).correlation
    d = np.abs(prio_ladder[vm] - prio_canopy[vm])
    logger.info("ladder vs canopy-only: Spearman=%.4f | differ>0.10=%.1f%% of AOI", rho, 100 * (d > 0.10).mean())
    for name, b in [("priority_ladder", prio_ladder), ("priority_canopy", prio_canopy)]:
        bv = b[vm]; top = bv >= np.percentile(bv, 90)
        logger.info("  %-18s top-10%%: canopy_mean=%.3f midstory_mean=%.3f non-forest(cc<.25)=%.1f%%",
                    name, canopy[vm][top].mean(), mid[vm][top].mean(), 100 * (canopy[vm][top] < 0.25).mean())


if __name__ == "__main__":
    main()
