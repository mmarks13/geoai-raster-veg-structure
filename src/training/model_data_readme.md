# Precomputed Tile Data Structures

Two pipelines produce precomputed `.pt` tile files with closely related schemas:

- **Raster pipeline (active)** — `data/processed/model_data_raster/precomputed_*_tiles_raster_32bit.pt` and `augmented_tiles_raster_32bit.pt`.
- **Point cloud pipeline (historical)** — `data/processed/model_data/precomputed_{training,validation,test}_tiles_32bit.pt` and `augmented_tiles_32bit_16k_no_repl.pt`.

Each file is a **list of dictionaries**, one per tile. A tile is a 10 m × 10 m region in EPSG:32611 (UTM 11N).

---

## Raster pipeline tile schema

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `dep_points_norm` | Tensor | `[N, 3]` | z-scored 3DEP points. X,Y centered at tile center; Z = Height Above Ground. Stage-2 z-score applied per axis. |
| `dep_points_attr_norm` | Tensor | `[N, 3]` | z-scored point attributes: `[Intensity, ReturnNumber, NumberOfReturns]`. The 3DEP HAG pipeline also produces Planarity / Sphericity / Verticality, but they are deliberately not loaded here: they are knn=15 eigenvalue features computed once at preprocessing time and go stale under point-removal augmentation and across sites with different 3DEP densities. |
| `naip` | Dict or None | — | Multi-temporal NAIP imagery (see NAIP dict below). |
| `uavsar` | Dict or None | — | Multi-temporal UAVSAR imagery. `None` for Laguna. |
| `target` | Tensor | `[n_bands, H, W]` | Vegetation-structure raster over the 2 m grid (5×5 for a 10 m tile). Active band subset governed by the band config. |
| `norm_params` | Dict | — | Contains bbox `center`, bbox `scale`, and Stage-2 coord `mean` / `std`. |
| `tile_id` | str | — | Unique identifier. |
| `bbox` | Tuple | `[4]` | `[xmin, ymin, xmax, ymax]` in EPSG:32611 (10 m × 10 m). |

### Two-stage coordinate normalization
1. **Bbox** (per tile): X,Y centered at 0 (so ∈ [-5, 5] m); min Z set to 0.
2. **Z-score** (global): per-axis mean/std applied, using statistics stored at `data/processed/model_data_raster/coordinate_normalization_stats.json`.

Denormalize back to bbox (meter) space before any distance operation: query attention, cross-attention fusion, positional encoding, `torch.cdist`.

---

## Point cloud pipeline tile schema (historical)

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `dep_points_norm` | Tensor | `[N_dep, 3]` | Normalized sparse 3DEP points. 1k–10k typical. |
| `uav_points_norm` | Tensor | `[N_uav, 3]` | Normalized dense UAV ground-truth points, downsampled to ≤20k. |
| `dep_points_attr` | Tensor | `[N_dep, 3]` | `[Intensity, ReturnNumber, NumberOfReturns]`. |
| `center` | Tensor | `[1, 3]` | Bbox normalization center. |
| `knn_edge_indices` | Dict[int, Tensor] | — | Precomputed KNN graphs keyed by `k` (typically `k=15`). Edge tensor shape `[2, E]` (PyG convention, undirected). |
| `naip` | Dict or None | — | NAIP imagery dict. |
| `uavsar` | Dict or None | — | UAVSAR imagery dict. |
| `tile_id` | str or None | — | Optional identifier. |
| `bbox` | Tuple | `[4]` | `[xmin, ymin, xmax, ymax]` in EPSG:32611. |

### Single-stage coordinate normalization (bbox only)
- X, Y centered at 0 (∈ [-5, 5] m).
- Z shifted so min Z = 0.
- Units are meters throughout (no z-score). This is why the density-aware Chamfer loss uses α=4 rather than the unit-cube α=1000.

---

## NAIP imagery dict

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `images` | Tensor | `[n_images, 4, 40, 40]` | 4-band (RGBN) chips at 0.5 m resolution. |
| `ids` | List[str] | `[n_images]` | NAIP image IDs. |
| `dates` | List[str] | `[n_images]` | Acquisition dates. |
| `relative_dates` | Tensor | `[n_images, 1]` | Days relative to reference acquisition. |
| `img_bbox` | Tuple | `[4]` | 20 m × 20 m bbox sharing centroid with the tile bbox. |
| `bands` | List | — | Band metadata. |

## UAVSAR imagery dict

| Key | Type | Shape | Description |
|-----|------|-------|-------------|
| `images` | Tensor | `[n_images, 6, 4, 4]` | 6-channel polarimetric chips resampled to ~5 m GSD (4×4 over the 20 m bbox). |
| `ids` | List[str] | `[n_images]` | UAVSAR image IDs. |
| `dates` | List[str] | `[n_images]` | Acquisition dates. |
| `relative_dates` | Tensor | `[n_images, 1]` | Days relative to reference acquisition. |
| `img_bbox` | Tuple | `[4]` | 20 m × 20 m bbox sharing centroid with the tile bbox. |
| `bands` | List | — | Polarization labels. |

Imagery counts vary per tile and NAIP / UAVSAR acquisition dates do **not** align.

---

## Batching convention

Tiles are NOT stacked into `[B, N, 3]`. Points across a batch are concatenated and a PyG-style `batch_indices` tensor `[total_N]` identifies which tile each point belongs to. Within-tile attention and k-NN are enforced via this index (+ `to_dense_batch` key-padding masks in the global attention).
