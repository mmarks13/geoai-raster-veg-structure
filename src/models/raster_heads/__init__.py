"""
Raster prediction heads for the multimodal raster fuel-metrics model.

Three architecture options, selected via `MultimodalRasterConfig.raster_architecture`:

    - "cross_attn_grid_mlp"    → CrossAttnGridMlpHead    (Path A, default)
    - "cross_attn_soft_pillar" → CrossAttnSoftPillarHead (Path B)
    - "grid_cross_attn"        → GridCrossAttnHead       (Path C)

See plan: `/home/jovyan/.claude/plans/recursive-snuggling-shannon.md`.
"""

from .path_a_cross_attn_grid_mlp import CrossAttnGridMlpHead
from .path_b_cross_attn_soft_pillar import CrossAttnSoftPillarHead
from .path_c_grid_cross_attn import GridCrossAttnBlock, GridCrossAttnHead

__all__ = [
    "CrossAttnGridMlpHead",
    "CrossAttnSoftPillarHead",
    "GridCrossAttnHead",
    "GridCrossAttnBlock",
]
