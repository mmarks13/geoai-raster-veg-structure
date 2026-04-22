# GeoAI Raster Vegetation Structure

**A multimodal deep-learning model that measures hard-to-see mid-story vegetation across Southern California by fusing sparse public LiDAR with optical and radar imagery.**

Extends *Marks, Sousa & Franklin, Remote Sensing* (2025) from dense point-cloud reconstruction to ecologically meaningful vegetation-structure rasters, validated against field plots at four out-of-distribution sites.

<!-- TODO: replace with raster-specific hero figure once available -->
<p align="center">
  <img src="manuscript/figures/Overall_Study_Areas_v2.png" width="720" alt="Study areas: two training sites, four out-of-distribution forest-plot sites in Southern California"/>
</p>

---

## Bottom line up front

Mid-story vegetation — roughly the 1–3 m band of shrubs, saplings, and regenerating canopy beneath the main tree layer — is critical for fire behavior, wildlife habitat, and carbon accounting. It is also chronically under-measured: satellite indices don't resolve vertical structure, and sparse airborne LiDAR is biased toward the top of the canopy.

Across 49 field plots at **four held-out Southern California forest sites** (BluffMesa, North Big Bear, Reyes Peak, Laguna — none of which the model trained on), fusing sparse 3DEP LiDAR with NAIP optical and UAVSAR radar imagery substantially improves agreement with field-measured shrub cover:

| Metric (shrub cover, 49 OOD plots) | Raw 3DEP baseline | Multimodal model |
|---|---|---|
| Pearson *r* | 0.24 | **0.45** |
| Spearman rank *r* | 0.18 | **0.36** |

The baseline is **not a competing model**. It is the shrub-layer density computed directly from the raw 3DEP LiDAR point cloud using the same metric pipeline applied to UAV ground truth — i.e., the best a publicly-available sparse-LiDAR-only answer can do. Shrub-layer agreement improved at all four OOD sites. Canopy-cover results were closer to baseline overall, with Laguna a notable exception (Pearson *r* 0.32 → **0.56**).

Field measurements and LiDAR-derived structure are related but non-equivalent indicators, so these numbers reflect stronger *agreement with independent field observations*, not exact one-to-one prediction.

---

## What this is, why it's hard, what it contributes

**What.** Inputs are all publicly available: sparse USGS 3DEP airborne LiDAR, multi-temporal NAIP optical imagery, and multi-temporal UAVSAR L-band SAR. Outputs are small multi-band vegetation-structure rasters on a 2 m grid per 10 m tile — height, cover, and density-by-layer variables from the Moudry et al. (2023) standardized set. Because every input is open data, the pipeline is in principle reproducible and scalable across the continental US.

**Why it's hard.** Dense UAV LiDAR measures vegetation structure accurately but is expensive and spatially limited. Sparse public airborne LiDAR is cheap and continental, but misses the mid- and understory. Satellite indices are frequent and broad but don't resolve vertical structure. Fusing these modalities — different resolutions, different temporal cadences, inconsistent coverage (e.g., UAVSAR is absent at Laguna) — is the core technical problem.

**What this contributes.** A peer-reviewed multimodal fusion approach, extended from point-cloud reconstruction to a more ecologically useful target, and evaluated against independent field data at sites outside the training distribution.

---

## From points to structure — published proof, extended scope

The 2025 *Remote Sensing* paper proved that multimodal fusion of sparse LiDAR, NAIP, and UAVSAR can reconstruct dense UAV-quality LiDAR point clouds. Points, however, are a means rather than an end: ecologists, fire modelers, and land managers work in terms of structure — cover, height, density-by-layer — not raw returns.

This repo closes that gap. It reuses the published encoder lineage and swaps the decoder to predict a gridded raster of vegetation-structure variables directly, then validates the result against field plots at four sites the model never saw during training.

> Marks, M.; Sousa, D.; Franklin, J. *Attention-Based Enhancement of Airborne LiDAR Across Vegetated Landscapes Using SAR and Optical Imagery Fusion.* **Remote Sensing** 2025, 17, 3278. <https://doi.org/10.3390/rs17193278>

---

## Results — out-of-distribution generalization

The model is trained on two Southern California sites (Sedgwick Reserve and Volcan Mountain) and evaluated against **49 matched forest plots at four ecologically distinct held-out sites**: BluffMesa (n = 7), North Big Bear (n = 11), Reyes Peak (n = 18), and Laguna (n = 13). Laguna has no UAVSAR coverage, which exercises the model's robustness to a missing modality at inference time.

Comparing the trained multimodal model against the raw-3DEP structural baseline:

- **Shrub / mid-story agreement improved at all four OOD sites.** Largest site-level gains at BluffMesa and North Big Bear; overall Pearson *r* rose from 0.24 to 0.45, rank-order *r* from 0.18 to 0.36.
- **Canopy-cover results were mixed overall and generally near baseline**, with Laguna the clear exception (*r* 0.32 → 0.56; rank *r* 0.29 → 0.50).

Because the field measurements and LiDAR-derived products describe related but non-identical aspects of vegetation, these findings are best read as *stronger agreement with independent field observations*, not exact one-to-one prediction. Understory (< 1 m) was not evaluated because no matching field measurements were available, not because the model does not produce predictions at that level.

---

## What I built

**Full-stack geospatial ML pipeline.** STAC-driven ingestion of NAIP, UAVSAR, 3DEP, and UAV LiDAR → PDAL-based Height-Above-Ground and geometric-feature computation → tile generation with two-stage coordinate normalization → online on-GPU augmentation → DDP multi-GPU training with mixed precision → Monte-Carlo-dropout inference → field-plot evaluation.

**Novel architecture.** A local-global point-attention encoder (LG-PAB) over sparse LiDAR, ViT-based image encoders for NAIP and UAVSAR with temporal aggregation, cross-attention multimodal fusion between point features and image patch embeddings, and a raster decoder built from learnable per-cell grid queries that attend into fused point features via Gaussian distance-biased cross-attention.

**Rigor aimed at OOD generalization.** Spectral normalization, stochastic depth, heteroscedastic Gaussian NLL loss with an overconfidence penalty, MC-dropout uncertainty at inference, modality dropout that directly mirrors the UAVSAR-missing Laguna site, synchronized geometric augmentation across points/imagery/targets, and point-cloud sparsification up to 90% to induce density invariance.

---

## What changed from the point-cloud upsampling model

The vegetation-structure raster predictor reuses the encoder backbone of the published point-cloud upsampling model but swaps the output target from a dense 3D point cloud to a small gridded raster of vegetation-structure metrics. While retargeting the model, a set of improvements were folded in that benefit both pipelines through the shared encoder.

|                     | Original model                                                  | New model                                                                                        |
| ------------------- | --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ |
| **Output**          | Dense **3D point cloud**                                        | Small **raster** of vegetation-structure metrics                                                 |
| **Prediction head** | Upsamples points, then regresses an xyz coordinate for each one | A fixed grid of learned attention queries, each pooling nearby point features into a raster cell |

**Issues fixed in the published encoder**

- **Cleaner attention math.** The original global point-attention block conflated *where to attend* with *what to aggregate*, letting positional information leak into the aggregated features. The block was rewritten so the two roles are kept separate.
- **No cross-tile contamination.** In the original implementation, when multiple tiles shared a minibatch, points in one tile could attend to points in another through the global attention and nearest-neighbor graphs. Both are now restricted to within-tile context.
- **Higher-fidelity optical tokenizer.** The original NAIP tokenizer averaged each patch into a single value per channel, discarding the within-patch texture that distinguishes vegetation structure at NAIP's resolution. Patches are now formed by a learned projection that preserves it.

**Additional improvements**

- **Terrain-relative heights.** Point heights are now expressed relative to the local ground surface rather than the lowest point of the tile, giving the model a stable prior on the ground.
- **Standardized point coordinates.** The published model trained on raw meter-scale coordinates, where the height axis was typically much larger than the planar axes. Modern deep-learning layers are optimized for standardized inputs; coordinates are now z-scored per axis using statistics computed once over the training set.

---

## Technical appendix

### Architecture

See [`src/models/README.md`](src/models/README.md) for file-level specifics. At a glance:

- **LG-PAB point feature extractor** — local k-NN point attention plus global position-aware attention ([`src/models/multimodal_model.py`](src/models/multimodal_model.py)).
- **Image encoders** — separate ViT encoders for NAIP and UAVSAR with temporal GRU aggregation ([`src/models/encoders.py`](src/models/encoders.py)).
- **Cross-attention fusion** — multi-head cross-attention from point features into image patch embeddings ([`src/models/cross_attn_fusion.py`](src/models/cross_attn_fusion.py)).
- **Raster decoder** — learnable per-cell grid queries, Gaussian distance-biased cross-attention into fused point features, pre-LN FFN, 1×1-conv MLP head with optional heteroscedastic (mean + log-variance) output ([`src/models/raster_head.py`](src/models/raster_head.py), [`src/models/multimodal_raster_model.py`](src/models/multimodal_raster_model.py)).

### Regularization and OOD generalization

The raster model layers the following techniques on top of standard dropout, weight decay, and gradient clipping:

- **Spectral normalization** on decoder linear/conv layers — Lipschitz constraint that bounds the model's sensitivity to input perturbations.
- **Stochastic depth (DropPath)** — randomly drops entire residual branches in the point-attention extractor, regularizing beyond activation-level dropout.
- **Heteroscedastic Gaussian NLL loss** with an overconfidence penalty. The decoder predicts both a mean and a per-pixel variance; uncertain regions are naturally down-weighted in the gradient, and the penalty discourages collapse to over-confident variance estimates.
- **Huber loss (δ = 2.0)** retained behind a config flag as a robust-regression alternative.
- **Auxiliary Foliage Height Diversity (FHD) target** — predicted alongside the evaluated bands even though FHD is not compared against field measurements. The extra task regularizes the shared encoder toward richer vertical-structure features.

### Inference-time ensembling and uncertainty

- **Monte-Carlo Dropout** — dropout layers remain active at inference; multiple stochastic forward passes yield a predictive mean and a pixel-wise uncertainty estimate.
- **Stochastic Weight Averaging (SWA)** is implemented (via `torch.optim.swa_utils`) and was tested, but MC Dropout alone was retained as the final inference-time method.
- MC Dropout surfaced a consistent pattern where UAVSAR input was associated with elevated predictive uncertainty at some sites — a finding that feeds the "what's next" section below.

### Online data augmentation (GPU, per batch)

All augmentation is applied online on the GPU (Kornia plus custom PyTorch ops) rather than precomputed offline, so each tile is freshly perturbed every epoch and sampling probabilities can be tuned without regenerating data.

- **Modality dropout** — NAIP and/or UAVSAR dropped per-tile at 25% / 35%. Directly mirrors the Laguna OOD site, which has no UAVSAR coverage.
- **Temporal augmentation** — random subsampling of NAIP/UAVSAR temporal stacks plus large temporal shifts (±730 days) of acquisition dates, decoupling the model from any single phenological snapshot.
- **Synchronized geometric augmentation** — rotations and X/Y reflections applied identically to points, imagery, and target rasters.
- **Point-cloud sparsification** — random point removal up to 90%, enabled by a global-only attention mode that recomputes neighborhoods on the fly rather than relying on precomputed KNN graphs.
- **Point-level perturbations** — coordinate jitter, intensity noise, return-attribute scaling/shuffling, and rare bird/outlier simulation.
- **Imagery perturbations** — Gaussian/motion blur, random erasing, sharpness, additive noise, and z-score radiometric gain + bias for both NAIP and UAVSAR.

### 3DEP baseline design

To isolate the value-add of multimodal fusion, the same Moudry vegetation-structure metric pipeline used for UAV ground truth is applied directly to the raw sparse 3DEP point clouds at each validation site. This produces a LiDAR-only structural reference — **not a trained model** — that can be compared against both field measurements and the multimodal model's predictions. Entry point: [`src/evaluation/compute_3dep_baseline_metrics.py`](src/evaluation/compute_3dep_baseline_metrics.py).

**3DEP data characterization at validation sites**

| Site          | Acquisition | Unique points | Density (pts/m²) | Pre-classified ground |
| ------------- | ----------- | ------------- | ---------------- | --------------------- |
| BluffMesa     | 2018        | 4.05 M        | 7.3              | Yes (25.3%)           |
| NorthBigBear  | 2018        | 16.9 M        | 7.4              | Yes (31.2%)           |
| ReyesPeak     | 2018        | 56.9 M        | 7.8              | Yes (23.7%)           |
| Laguna        | 2015        | 30.7 M        | 11.5             | Yes (44.1%)           |

---

## Getting started

Environment setup is via Conda:

```bash
conda env create -f environment.yml
conda activate geoai_env
```

Data preparation, training, and evaluation pipelines are documented in [`scripts/README.md`](scripts/README.md) and [`src/README.md`](src/README.md). Primary entry points:

- Training: `run_raster_model.py`
- Forest-plot evaluation: `scripts/evaluate_forest_plots.sh`
- 3DEP-only baseline: `src/evaluation/compute_3dep_baseline_metrics.py`

---

## What's next

**Broader sensor palette.** UAVSAR coverage is sparse and inconsistent, which showed up as elevated predictive uncertainty in MC-dropout runs. Near-term, Sentinel-1 (10 m, open, global, high temporal cadence) is the natural replacement. Longer term, NISAR opens up L-band coverage at a scale that UAVSAR cannot match.

**Foundation-model encoders for generalization.** The more I work on this, the more the evidence points to generalization — not architecture — as the dominant limiter. Imagery foundation models are the natural next step for the image-encoder pathway. An early attempt at using [Clay](https://clay-foundation.github.io/model/release-notes/specification.html) was set aside because its 256 × 256-pixel training scale was too coarse for the 10–40 m tile footprints here. [AnySat](https://github.com/gastruc/AnySat) trains at scales as small as 60 × 60 m, which fits this pipeline's constraints and is a promising candidate for the next iteration.

**A point-cloud foundation model on 3DEP.** To my knowledge no such model exists at continental scale on airborne LiDAR. Given that 3DEP is open, standardized, and near-continental in US coverage, this is plausibly the highest-leverage piece of missing infrastructure for this problem family — and a research direction I would pursue given the opportunity.

---

## Related publication

Marks, M.; Sousa, D.; Franklin, J. **Attention-Based Enhancement of Airborne LiDAR Across Vegetated Landscapes Using SAR and Optical Imagery Fusion.** *Remote Sensing* **2025**, *17*, 3278. <https://doi.org/10.3390/rs17193278>

```bibtex
@article{marks2025attention,
  author  = {Marks, Michael and Sousa, Daniel and Franklin, Janet},
  title   = {Attention-Based Enhancement of Airborne LiDAR Across Vegetated Landscapes Using SAR and Optical Imagery Fusion},
  journal = {Remote Sensing},
  volume  = {17},
  number  = {19},
  pages   = {3278},
  year    = {2025},
  doi     = {10.3390/rs17193278}
}
```

---

## Contact

**Michael Marks** — <mmarks13@gmail.com> — Department of Geography, San Diego State University.
