"""
Shared primitives for the new raster prediction heads.

This module provides clean, freshly-authored building blocks used by the three
production raster architectures in `src/models/raster_heads/`:

    - LearnableGridQueries: 5×5 learnable grid query embeddings + 2D sinusoidal PE
    - PatchPositionEncoding: 2D sinusoidal PE for image patch centers
    - GaussianDistanceBiasedCrossAttention: multi-source distance-biased cross-attn
    - PreLNFFN: standalone Pre-LN residual FFN block
    - SmallMlpDecoder: 2-layer pointwise MLP decoder (with optional heteroscedastic head)

Positional encoding extent: a single unified extent (`pe_extent_m`, default 20m)
is used for points, grid query centers, and image patch centers so that
co-located sources produce matching encodings.
"""

from typing import List, Optional, Sequence, Tuple, Union

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Unified positional-encoding extent (meters) used by all primitives below.
# Points sit in roughly [-5, 5] m, grid centers in [-4, 4] m, image patches in
# [-7.5, 7.5] m. A 20 m extent encodes everything in a common normalized frame
# so co-located queries / keys produce matching encodings.
DEFAULT_PE_EXTENT_M: float = 20.0


def _build_2d_sinusoidal_encoding(
    positions_m: np.ndarray,
    feature_dim: int,
    pe_extent_m: float,
) -> np.ndarray:
    """Standard ViT/DETR-style 2D sinusoidal encoding for (x, y) positions in meters.

    Half the dimensions encode x, half encode y. Within each half, the pattern is
    [sin(pos * div_0), cos(pos * div_0), sin(pos * div_1), cos(pos * div_1), ...].
    """
    num = positions_m.shape[0]
    positions_norm = (positions_m + pe_extent_m / 2.0) / pe_extent_m  # → roughly [0, 1]

    half_dim = feature_dim // 2
    div_term = np.exp(np.arange(0, half_dim, 2) * (-np.log(10000.0) / half_dim))

    x_pos = positions_norm[:, 0:1]
    y_pos = positions_norm[:, 1:2]

    pe_x = np.zeros((num, half_dim))
    pe_y = np.zeros((num, half_dim))
    pe_x[:, 0::2] = np.sin(x_pos * div_term)
    pe_x[:, 1::2] = np.cos(x_pos * div_term)
    pe_y[:, 0::2] = np.sin(y_pos * div_term)
    pe_y[:, 1::2] = np.cos(y_pos * div_term)

    return np.concatenate([pe_x, pe_y], axis=1)  # [num, feature_dim] (or feature_dim - 1 if odd)


class LearnableGridQueries(nn.Module):
    """Learnable 5×5 grid query embeddings combined with 2D sinusoidal PE.

    Grid centers live in meter space, centered at the tile origin. For a 5×5 grid
    over a 10 m tile the centers are [-4, -2, 0, 2, 4] in both x and y.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        grid_size: int = 5,
        tile_extent_m: float = 10.0,
        pe_extent_m: float = DEFAULT_PE_EXTENT_M,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.grid_size = grid_size
        self.tile_extent_m = tile_extent_m
        self.num_queries = grid_size * grid_size

        cell_size = tile_extent_m / grid_size
        centers_1d = np.linspace(
            -tile_extent_m / 2 + cell_size / 2,
            tile_extent_m / 2 - cell_size / 2,
            grid_size,
        )
        grid_y, grid_x = np.meshgrid(centers_1d, centers_1d, indexing="ij")
        grid_centers = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1).astype(np.float32)

        self.register_buffer("grid_centers", torch.from_numpy(grid_centers))

        self.query_embed = nn.Parameter(torch.randn(self.num_queries, feature_dim))
        nn.init.normal_(self.query_embed, std=0.02)

        pos_encoding = _build_2d_sinusoidal_encoding(grid_centers, feature_dim, pe_extent_m)
        self.register_buffer("pos_encoding", torch.from_numpy(pos_encoding).float())

    def forward(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        queries = self.query_embed + self.pos_encoding  # [num_queries, F]
        return queries.unsqueeze(0).expand(batch_size, -1, -1), self.grid_centers


class PatchPositionEncoding(nn.Module):
    """2D sinusoidal positional encoding for image patch centers.

    NAIP/UAVSAR patches cover a `patch_extent_m × patch_extent_m` area centered on
    the tile. For a 4×4 grid over 20 m, centers are at [-7.5, -2.5, 2.5, 7.5] m.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        patch_grid_size: int = 4,
        patch_extent_m: float = 20.0,
        pe_extent_m: float = DEFAULT_PE_EXTENT_M,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.patch_grid_size = patch_grid_size
        self.patch_extent_m = patch_extent_m
        self.num_patches = patch_grid_size * patch_grid_size

        spacing = patch_extent_m / patch_grid_size
        centers_1d = np.linspace(
            -patch_extent_m / 2 + spacing / 2,
            patch_extent_m / 2 - spacing / 2,
            patch_grid_size,
        )
        grid_y, grid_x = np.meshgrid(centers_1d, centers_1d, indexing="ij")
        patch_centers = np.stack([grid_x.flatten(), grid_y.flatten()], axis=-1).astype(np.float32)

        self.register_buffer("patch_centers", torch.from_numpy(patch_centers))

        pos_encoding = _build_2d_sinusoidal_encoding(patch_centers, feature_dim, pe_extent_m)
        self.register_buffer("pos_encoding", torch.from_numpy(pos_encoding).float())

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.pos_encoding, self.patch_centers


def build_point_position_encoding(
    point_positions_xy: torch.Tensor,
    feature_dim: int,
    pe_extent_m: float = DEFAULT_PE_EXTENT_M,
) -> torch.Tensor:
    """2D sinusoidal positional encoding for batched point positions in meters.

    Args:
        point_positions_xy: [B, N, 2] (x, y) point positions in meters.
        feature_dim: Output encoding dimension.
        pe_extent_m: Normalization extent in meters; same value used by other
            primitives so co-located sources produce matching encodings.
    Returns:
        Encoded positions [B, N, feature_dim].
    """
    B, N, _ = point_positions_xy.shape
    half_dim = feature_dim // 2

    positions_norm = (point_positions_xy + pe_extent_m / 2.0) / pe_extent_m
    positions_norm = positions_norm.clamp(0.0, 1.0)

    div_term = torch.exp(
        torch.arange(0, half_dim, 2, device=point_positions_xy.device, dtype=torch.float32)
        * (-math.log(10000.0) / half_dim)
    )

    x_pos = positions_norm[:, :, 0:1].float()
    y_pos = positions_norm[:, :, 1:2].float()

    x_angles = x_pos * div_term  # [B, N, half_dim // 2]
    y_angles = y_pos * div_term

    pe_x = torch.stack([torch.sin(x_angles), torch.cos(x_angles)], dim=-1).view(B, N, half_dim)
    pe_y = torch.stack([torch.sin(y_angles), torch.cos(y_angles)], dim=-1).view(B, N, half_dim)

    return torch.cat([pe_x, pe_y], dim=-1).to(point_positions_xy.dtype)  # [B, N, feature_dim]


class GaussianDistanceBiasedCrossAttention(nn.Module):
    """Multi-head cross-attention with multi-scale Gaussian distance bias.

    Queries attend to a stack of keys/values (e.g. concatenated point and image
    patch features). Attention scores receive an additive log-Gaussian bias
    `-d² / (2σ²)` per head, supporting per-head sigma values for multi-scale
    behavior. Padding positions and per-tile modality mask positions are biased
    to `-inf` so they are excluded from the softmax.

    The module assumes inputs are already projected into a single common feature
    dimension (`feature_dim`). It owns its own Q/K/V/out projections.

    Args:
        feature_dim: Common feature dimension for queries, keys, values.
        num_heads: Number of attention heads (must divide feature_dim).
        distance_sigma: float or list/tuple of length num_heads. Sigma in meters.
        dropout: Attention dropout probability.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_heads: int = 8,
        distance_sigma: Union[float, Sequence[float]] = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert feature_dim % num_heads == 0, "feature_dim must be divisible by num_heads"

        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.dropout = dropout

        if isinstance(distance_sigma, (list, tuple)):
            assert len(distance_sigma) == num_heads, (
                f"distance_sigma list length ({len(distance_sigma)}) must match "
                f"num_heads ({num_heads})"
            )
            sigma_tensor = torch.tensor(list(distance_sigma), dtype=torch.float32)
        else:
            sigma_tensor = torch.tensor([float(distance_sigma)] * num_heads, dtype=torch.float32)
        self.register_buffer("distance_sigmas", sigma_tensor)

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(
        self,
        queries: torch.Tensor,           # [B, Q, F]
        keys: torch.Tensor,              # [B, K, F]
        values: torch.Tensor,            # [B, K, F]
        query_positions: torch.Tensor,   # [Q, 2] (shared across batch) or [B, Q, 2]
        key_positions: torch.Tensor,     # [B, K, 2]
        key_padding_mask: Optional[torch.Tensor] = None,  # [B, K] True = valid
    ) -> torch.Tensor:
        """Apply distance-biased multi-head cross-attention.

        `key_padding_mask` must be True for valid positions and False for either
        padding entries from `to_dense_batch` *or* per-tile modality positions
        whose source modality is missing for that tile. The caller is responsible
        for OR-ing those two sources of invalidity together.
        """
        B, Q, _ = queries.shape
        _, K, _ = keys.shape
        H, D = self.num_heads, self.head_dim

        # Q / K / V projections
        q = self.q_proj(queries).view(B, Q, H, D).transpose(1, 2)  # [B, H, Q, D]
        k = self.k_proj(keys).view(B, K, H, D).transpose(1, 2)     # [B, H, K, D]
        v = self.v_proj(values).view(B, K, H, D).transpose(1, 2)   # [B, H, K, D]

        # Pairwise distances [B, Q, K] in meters
        if query_positions.dim() == 2:
            qp = query_positions.unsqueeze(0).expand(B, -1, -1)
        else:
            qp = query_positions
        distances = torch.cdist(qp.float(), key_positions.float(), p=2)  # [B, Q, K]

        # Per-head log Gaussian bias [B, H, Q, K]
        sigmas = self.distance_sigmas.view(1, H, 1, 1)
        log_bias = -(distances.unsqueeze(1) ** 2) / (2.0 * sigmas ** 2)

        # Padding / modality mask → -inf at masked positions
        if key_padding_mask is not None:
            invalid = ~key_padding_mask  # [B, K] True where masked
            log_bias = log_bias.masked_fill(
                invalid.unsqueeze(1).unsqueeze(2),  # [B, 1, 1, K]
                float("-inf"),
            )

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)  # [B, H, Q, K]
        scores = scores + log_bias.to(scores.dtype)

        attn = F.softmax(scores, dim=-1)
        # If a query has no valid keys at all the row is all -inf and softmax → NaN.
        attn = torch.nan_to_num(attn, nan=0.0)

        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout, training=True)

        out = torch.matmul(attn, v)  # [B, H, Q, D]
        out = out.transpose(1, 2).contiguous().view(B, Q, self.feature_dim)
        return self.out_proj(out)


class PreLNFFN(nn.Module):
    """Pre-LN residual FFN: ``x + Linear(Dropout(GELU(Linear(LN(x)))))``.

    Used by Path A as a small capacity bump between the grid aggregator and the
    decoder. Single block, no attention — deliberately distinct from Path C's
    transformer block.
    """

    def __init__(self, feature_dim: int, ffn_ratio: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(feature_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * ffn_ratio, feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class SmallMlpDecoder(nn.Module):
    """Pointwise 2-layer MLP decoder.

    Architecture:
        LayerNorm → Linear(F → F) → GELU → Dropout → Linear(F → out_channels)

    With `output_variance=True`, `out_channels = 2 * n_bands` and the forward
    splits the output into ``(mean, log_var)`` along the channel dimension.

    Operates on ``[B, H, W, F]`` grid features and returns ``[B, n_bands, H, W]``
    (or a tuple of two such tensors in heteroscedastic mode), matching the
    convention used by the rest of the raster model.
    """

    def __init__(
        self,
        feature_dim: int = 256,
        n_bands: int = 8,
        dropout: float = 0.1,
        output_variance: bool = False,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.n_bands = n_bands
        self.output_variance = output_variance

        out_channels = n_bands * 2 if output_variance else n_bands
        self.norm = nn.LayerNorm(feature_dim)
        self.fc1 = nn.Linear(feature_dim, feature_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(feature_dim, out_channels)

    def forward(self, grid_features: torch.Tensor):
        """`grid_features`: [B, H, W, F]."""
        x = self.norm(grid_features)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # [B, H, W, out_channels]
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, out_channels, H, W]

        if self.output_variance:
            mean = x[:, : self.n_bands]
            log_var = x[:, self.n_bands :]
            return mean, log_var
        return x
