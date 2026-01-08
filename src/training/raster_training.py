"""
Training pipeline for multimodal raster prediction model.

Adapted from multimodal_training.py for raster fuel metrics prediction.
Uses MSE loss instead of Chamfer distance, and handles raster-specific data format.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import autocast
from torch.optim.swa_utils import AveragedModel, update_bn
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import gc

try:
    from torch.utils.tensorboard import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False
    print("Warning: TensorBoard not available")


# Import raster-specific components
from src.models.multimodal_raster_model import MultimodalRasterPredictor, MultimodalRasterConfig
from src.training.raster_dataset import ShardedRasterDataset, raster_variable_size_collate

# Import shared utilities
from src.training.ddp_training import setup_logging
from schedulefree import AdamWScheduleFree
import socket


def find_free_port(start_port: int = 12355, max_attempts: int = 100) -> int:
    """
    Find a free port for distributed training.

    Args:
        start_port: Starting port number to try
        max_attempts: Maximum number of ports to try

    Returns:
        Available port number

    Raises:
        RuntimeError: If no free port found after max_attempts
    """
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                return port
        except OSError:
            continue
    raise RuntimeError(f"Could not find free port after {max_attempts} attempts starting from {start_port}")


def compute_correlation_loss(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute correlation loss as (1 - Pearson correlation coefficient).

    Fully vectorized implementation - no Python loops or GPU→CPU syncs.
    This loss penalizes predictions that fail to preserve the ranking/variance
    of the targets, addressing variance collapse where models predict a narrow
    range around the mean.

    Args:
        predictions: [batch_size, n_bands, H, W] predicted values
        targets: [batch_size, n_bands, H, W] target values

    Returns:
        Scalar tensor: mean(1 - r) across all bands, where r is Pearson correlation.
        Range: [0, 2] where 0 = perfect positive correlation, 1 = no correlation, 2 = perfect negative correlation.
    """
    batch_size, n_bands = predictions.shape[:2]

    # Reshape to [n_bands, N] where N = batch_size * H * W
    pred_all = predictions.permute(1, 0, 2, 3).reshape(n_bands, -1)  # [n_bands, N]
    targ_all = targets.permute(1, 0, 2, 3).reshape(n_bands, -1)      # [n_bands, N]

    # Vectorized mean centering: [n_bands, N]
    pred_centered = pred_all - pred_all.mean(dim=1, keepdim=True)
    targ_centered = targ_all - targ_all.mean(dim=1, keepdim=True)

    # Vectorized std: [n_bands]
    pred_std = pred_centered.std(dim=1)
    targ_std = targ_centered.std(dim=1)

    # Vectorized covariance: [n_bands]
    covariance = (pred_centered * targ_centered).mean(dim=1)

    # Vectorized correlation: [n_bands]
    correlation = covariance / (pred_std * targ_std + 1e-8)
    correlation = torch.clamp(correlation, -1.0, 1.0)

    # Handle collapsed variance with torch.where (no if statement, no GPU→CPU sync)
    min_std = 1e-8
    valid_mask = (pred_std > min_std) & (targ_std > min_std)
    corr_loss = torch.where(valid_mask, 1.0 - correlation, torch.ones_like(correlation))

    return corr_loss.mean()


def get_gpu_stats_native(device_id: int = 0) -> dict:
    """Get GPU memory stats using PyTorch native functions (no subprocess).

    Returns dict with memory stats in GB for the specified device.
    Note: GPU utilization % requires nvidia-smi and is not available via PyTorch.
    """
    return {
        'memory_allocated_gb': torch.cuda.memory_allocated(device_id) / (1024 ** 3),
        'memory_reserved_gb': torch.cuda.memory_reserved(device_id) / (1024 ** 3),
        'max_memory_allocated_gb': torch.cuda.max_memory_allocated(device_id) / (1024 ** 3),
    }


def create_raster_shards(data_list: List, world_size: int, temp_dir: str, prefix: str) -> List[str]:
    """
    Split raster dataset into balanced shards and save to disk.

    Uses greedy load-balancing to distribute samples by point cloud size.
    This ensures compute load is balanced across GPUs (not fuel_metrics size).

    Args:
        data_list: List of data samples from torch.load()
        world_size: Number of GPUs to shard for
        temp_dir: Directory to save shard files
        prefix: Prefix for shard filenames (e.g., 'raster_train')

    Returns:
        List of shard file paths
    """
    import gc

    # Create temp directory if needed
    os.makedirs(temp_dir, exist_ok=True)

    # Get sizes for balancing - measure by point cloud size (not fuel_metrics)
    sizes = []
    for i, sample in enumerate(data_list):
        # Use point cloud size as measure of compute load
        if isinstance(sample, dict) and 'dep_points' in sample:
            size = len(sample['dep_points'])
        elif isinstance(sample, dict) and 'dep_points_norm' in sample:
            size = sample['dep_points_norm'].shape[0]
        else:
            # Fallback: use fuel_metrics size
            if 'fuel_metrics_batch' in sample:
                size = sample['fuel_metrics_batch'].numel()
            else:
                size = 1
        sizes.append((i, size))

    # Sort by size (largest first) for better load balancing
    sizes.sort(key=lambda x: x[1], reverse=True)

    # Initialize shards using greedy algorithm
    shards = [[] for _ in range(world_size)]
    shard_sizes = [0] * world_size

    # Distribute samples to shard with smallest total size
    for idx, size in sizes:
        min_shard = shard_sizes.index(min(shard_sizes))
        shards[min_shard].append(idx)
        shard_sizes[min_shard] += size

    # Log balance information
    min_size = min(shard_sizes)
    max_size = max(shard_sizes)
    avg_size = sum(shard_sizes) / world_size
    print(f"Shard balance: min={min_size}, max={max_size}, avg={avg_size:.1f}, "
          f"ratio={max_size/min_size:.2f}")

    # Create each shard file
    shard_paths = []
    for rank in range(world_size):
        # Create the shard data
        shard_data = [data_list[i] for i in shards[rank]]

        # Save to file
        shard_path = os.path.join(temp_dir, f"{prefix}_shard_{rank}.pt")
        torch.save(shard_data, shard_path, _use_new_zipfile_serialization=False)
        shard_paths.append(shard_path)

        gc.collect()
        print(f"  ✓ Created shard {rank} with {len(shard_data)} samples, saved to {shard_path}")

    return shard_paths


def process_raster_batch(
    model: nn.Module,
    batch: Tuple,
    device: torch.device,
    use_naip: bool = False,
    use_uavsar: bool = False,
    is_training: bool = True,
    rank: int = 0,
    logger: Optional[logging.Logger] = None,
    correlation_loss_weight: float = 0.0,
    huber_delta: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Process a single batch for raster prediction with per-tile NaN diagnostics.

    Args:
        model: Raster prediction model
        batch: 11-element tuple from raster_variable_size_collate
               (edge_index removed - global-only attention doesn't use KNN graphs)
        device: Device to run on
        use_naip: Whether NAIP imagery is used
        use_uavsar: Whether UAVSAR imagery is used
        is_training: Whether in training mode
        rank: GPU rank for logging
        logger: Logger instance
        correlation_loss_weight: Weight for correlation loss term (0 = disabled)
        huber_delta: Delta threshold for Huber loss (errors > delta use linear penalty)

    Returns:
        Tuple of (predictions, targets, loss, per_band_losses, diagnostics)
        - predictions: [batch_size, n_bands, 5, 5] predicted rasters
        - targets: [batch_size, n_bands, 5, 5] ground truth rasters
        - loss: scalar combined loss (Huber + correlation_loss_weight * corr_loss)
        - per_band_losses: Dict with per-band MSE losses, huber_loss, and correlation loss
        - diagnostics: Dict with batch status and bad tile info
    """
    # Unpack batch (11 elements - edge_index removed for global-only attention)
    (dep_points_batch, fuel_metrics_batch, dep_points_attr_batch,
     naip_data_batch, uavsar_data_batch, centers, scales, bboxes, tile_ids,
     norm_params_list, batch_indices) = batch

    # Move tensors to device
    dep_points_batch = dep_points_batch.to(device)
    fuel_metrics_batch = fuel_metrics_batch.to(device)
    dep_points_attr_batch = dep_points_attr_batch.to(device)
    batch_indices = batch_indices.to(device)
    bboxes = bboxes.to(device) if bboxes is not None else None

    # Handle imagery data (list of dicts)
    naip_data = naip_data_batch if use_naip else None
    uavsar_data = uavsar_data_batch if use_uavsar else None

    # ====== GPU Training Augmentation (Geometric) ======
    # Applied in training loop because it needs access to targets (fuel_metrics)
    if is_training and hasattr(model, 'training_aug'):
        (dep_points_batch, naip_data, uavsar_data, fuel_metrics_batch) = \
            model.training_aug.augment_batch_geometric(
                dep_points_batch, batch_indices, naip_data, uavsar_data, fuel_metrics_batch
            )

    # Forward pass (edge_index=None for global-only attention)
    predictions = model(
        dep_points=dep_points_batch,
        edge_index=None,
        batch_indices=batch_indices,
        norm_params=norm_params_list,
        dep_attr=dep_points_attr_batch,
        naip=naip_data,
        uavsar=uavsar_data,
        bbox=bboxes
    )  # [batch_size, n_bands, 5, 5]

    # Compute both MSE (for logging/comparison) and Huber (for training)
    mse_loss = nn.functional.mse_loss(predictions, fuel_metrics_batch)
    huber_loss = nn.functional.huber_loss(predictions, fuel_metrics_batch, delta=huber_delta)

    # Compute per-band losses (still use MSE for comparability across runs)
    per_band_losses = {}
    n_bands = predictions.shape[1]
    for band_idx in range(n_bands):
        band_loss = nn.functional.mse_loss(predictions[:, band_idx], fuel_metrics_batch[:, band_idx])
        per_band_losses[f'Band_{band_idx}'] = band_loss

    # Compute correlation loss if enabled
    if correlation_loss_weight > 0:
        corr_loss = compute_correlation_loss(predictions, fuel_metrics_batch)
        per_band_losses['correlation_loss'] = corr_loss
        loss = huber_loss + correlation_loss_weight * corr_loss  # Use Huber as base
    else:
        loss = huber_loss  # Use Huber for actual training

    # Store both MSE and Huber for logging
    per_band_losses['mse_loss'] = mse_loss      # For backward compatibility / comparison
    per_band_losses['huber_loss'] = huber_loss  # New robust loss metric

    # Initialize diagnostics
    diagnostics = {
        'loss_is_valid': not (torch.isnan(loss) or torch.isinf(loss)),
        'all_bad': False,
        'bad_tiles': []
    }

    # If loss is NaN, diagnose which tiles are bad (tile-by-tile only on failure)
    if torch.isnan(loss) or torch.isinf(loss):
        batch_size = predictions.shape[0]
        for tile_idx in range(batch_size):
            tile_id = tile_ids[tile_idx] if isinstance(tile_ids, list) else tile_ids
            pred = predictions[tile_idx]  # [n_bands, 5, 5]
            targ = fuel_metrics_batch[tile_idx]

            bad_info = {}

            # Check predictions for NaN/Inf
            if torch.isnan(pred).any() or torch.isinf(pred).any():
                nan_count = torch.isnan(pred).sum().item()
                inf_count = torch.isinf(pred).sum().item()
                bad_info = {
                    'tile_id': str(tile_id),
                    'reason': 'NaN in predictions',
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'pred_min': float(pred.min().item()) if not torch.isnan(pred).all() else 'all_nan',
                    'pred_max': float(pred.max().item()) if not torch.isnan(pred).all() else 'all_nan'
                }

            # Check targets for NaN/Inf
            elif torch.isnan(targ).any() or torch.isinf(targ).any():
                nan_count = torch.isnan(targ).sum().item()
                inf_count = torch.isinf(targ).sum().item()
                bad_info = {
                    'tile_id': str(tile_id),
                    'reason': 'NaN in targets',
                    'nan_count': nan_count,
                    'inf_count': inf_count,
                    'targ_min': float(targ.min().item()) if not torch.isnan(targ).all() else 'all_nan',
                    'targ_max': float(targ.max().item()) if not torch.isnan(targ).all() else 'all_nan'
                }

            if bad_info:
                diagnostics['bad_tiles'].append(bad_info)
                # Log the bad tile
                msg = f"[GPU {rank}] Tile {bad_info['tile_id']}: {bad_info['reason']} ({bad_info.get('nan_count', 0)} NaNs)"
                print(msg)
                if logger:
                    logger.warning(msg)

        # Mark if all tiles in batch are bad
        if len(diagnostics['bad_tiles']) == batch_size:
            diagnostics['all_bad'] = True
            msg = f"[GPU {rank}] All {batch_size} tiles in batch are bad, skipping entire batch"
            print(msg)
            if logger:
                logger.warning(msg)

    return predictions, fuel_metrics_batch, loss, per_band_losses, diagnostics


def train_one_epoch_ddp(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler,
    config: MultimodalRasterConfig,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    accumulation_steps: int = 1,
    rank: int = 0,
    logger: Optional[logging.Logger] = None,
    max_grad_norm: float = 10.0
) -> Dict[str, float]:
    """
    Train the model for one epoch using DDP with TensorBoard logging.

    Args:
        model: The model to train (DDP wrapped)
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        device: Device to train on
        scaler: GradScaler for mixed precision training
        config: Model configuration
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number
        accumulation_steps: Number of batches to accumulate gradients over
        rank: GPU rank
        logger: Logger instance

    Returns:
        Dict with training metrics
    """
    optimizer.train()
    model.train()

    batch_count = 0
    accumulated_batch_count = 0
    n_good_tiles = 0
    n_bad_tiles = 0

    # Compute steps_per_epoch for global step tracking
    steps_per_epoch = len(train_loader) // accumulation_steps
    current_optimizer_step = epoch * steps_per_epoch  # Global step, not per-epoch
    current_batch_step = epoch * len(train_loader)

    # Zero gradients at start of epoch
    optimizer.zero_grad()

    # Timing trackers
    data_load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    optimizer_time = 0.0
    batch_start_time = time.time()

    # =========================================================================
    # GPU Running Statistics (no CPU copies during training)
    # All metrics computed via running sums, single .item() call at epoch end
    # =========================================================================
    running_loss = torch.zeros(1, device=device)
    running_huber_loss = torch.zeros(1, device=device)
    running_mse_loss = torch.zeros(1, device=device)
    running_corr_loss = torch.zeros(1, device=device)

    # Per-band running statistics (if n_bands is known)
    n_bands = config.n_bands
    running_loss_per_band = torch.zeros(n_bands, device=device)

    # Gradient norm tracking (epoch-level max and mean)
    running_max_grad_norm = torch.zeros(1, device=device)
    running_sum_grad_norm = torch.zeros(1, device=device)
    n_optimizer_steps_local = 0

    # NOTE: Per-module gradient norm tracking DISABLED for optimizer speed
    # Uncomment below if debugging gradient flow issues
    # module_grad_norms = {
    #     'feature_extractor': {'sum': 0.0, 'max': 0.0, 'count': 0},
    #     'naip_encoder': {'sum': 0.0, 'max': 0.0, 'count': 0},
    #     'uavsar_encoder': {'sum': 0.0, 'max': 0.0, 'count': 0},
    #     'fusion': {'sum': 0.0, 'max': 0.0, 'count': 0},
    #     'raster_head': {'sum': 0.0, 'max': 0.0, 'count': 0},
    # }

    # Overall running statistics for variance/correlation metrics
    running_sum_pred = torch.zeros(1, device=device)
    running_sum_targ = torch.zeros(1, device=device)
    running_sum_sq_pred = torch.zeros(1, device=device)
    running_sum_sq_targ = torch.zeros(1, device=device)
    running_sum_cross = torch.zeros(1, device=device)
    running_mae = torch.zeros(1, device=device)
    n_samples = 0

    # Pre-allocate tensors for epoch-end all_reduce (avoids repeated allocations)
    batch_count_tensor = torch.zeros(1, device=device, dtype=torch.long)
    n_samples_tensor = torch.zeros(1, device=device, dtype=torch.long)
    n_optimizer_steps_tensor = torch.zeros(1, device=device, dtype=torch.long)

    for batch_idx, batch in enumerate(train_loader):
        # Track data loading time
        data_time = time.time() - batch_start_time
        data_load_time += data_time

        # Forward pass timing
        forward_start = time.time()
        with autocast(device_type='cuda', dtype=torch.bfloat16):
            predictions, targets, loss, per_band_losses, diagnostics = process_raster_batch(
                model, batch, device,
                use_naip=config.use_naip,
                use_uavsar=config.use_uavsar,
                is_training=True,
                rank=rank,
                logger=logger,
                correlation_loss_weight=config.correlation_loss_weight,
                huber_delta=config.huber_delta
            )
        forward_time += (time.time() - forward_start)

        # Handle invalid loss locally without global synchronization
        # NOTE: Removed per-batch all_reduce for GPU efficiency. NaN losses should be
        # extremely rare after preprocessing. If NaN occurs, we replace with zero loss
        # to avoid DDP deadlock (all GPUs must participate in backward pass).
        if not diagnostics['loss_is_valid'] or diagnostics['all_bad']:
            n_bad_tiles += len(diagnostics['bad_tiles'])
            if rank == 0 and logger:
                logger.warning(f"Batch {batch_idx}: Invalid loss on GPU {rank}, using zero loss to avoid deadlock")
            # Replace NaN loss with zero to keep DDP in sync (all GPUs must do backward)
            loss = torch.zeros(1, device=device, requires_grad=True)
            # Note: This batch contributes zero gradient but DDP stays synchronized

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        n_good_tiles += batch[0].shape[0] - len(diagnostics['bad_tiles'])

        # =====================================================================
        # GPU Running Statistics Update (no CPU copies, no .item() calls)
        # =====================================================================
        with torch.no_grad():
            # Accumulate loss on GPU
            running_loss += loss.detach()
            running_huber_loss += per_band_losses['huber_loss'].detach()
            running_mse_loss += per_band_losses['mse_loss'].detach()
            if 'correlation_loss' in per_band_losses:
                running_corr_loss += per_band_losses['correlation_loss'].detach()

            # Per-band MSE losses
            for band_idx in range(n_bands):
                band_key = f'Band_{band_idx}'
                if band_key in per_band_losses:
                    running_loss_per_band[band_idx] += per_band_losses[band_key].detach()

            # Update running statistics for variance/correlation metrics
            pred_flat = predictions.detach().flatten()
            targ_flat = targets.detach().flatten()
            running_sum_pred += pred_flat.sum()
            running_sum_targ += targ_flat.sum()
            running_sum_sq_pred += (pred_flat ** 2).sum()
            running_sum_sq_targ += (targ_flat ** 2).sum()
            running_sum_cross += (pred_flat * targ_flat).sum()
            running_mae += (pred_flat - targ_flat).abs().sum()
            n_samples += pred_flat.numel()

        # Backward pass timing
        backward_start = time.time()
        scaler.scale(scaled_loss).backward()
        backward_time += (time.time() - backward_start)

        batch_count += 1
        accumulated_batch_count += 1
        current_batch_step += 1

        # NOTE: Per-batch TensorBoard logging removed for GPU efficiency
        # Epoch-level metrics are sufficient for monitoring

        # GPU monitoring via PyTorch native (every 20 batches, no subprocess overhead)
        if batch_idx % 20 == 0 and rank == 0 and writer is not None:
            gpu_stats = get_gpu_stats_native(device.index if device.index is not None else 0)
            writer.add_scalar('GPU/memory_allocated_gb', gpu_stats['memory_allocated_gb'], current_batch_step)
            writer.add_scalar('GPU/memory_reserved_gb', gpu_stats['memory_reserved_gb'], current_batch_step)
            writer.add_scalar('GPU/max_memory_allocated_gb', gpu_stats['max_memory_allocated_gb'], current_batch_step)

        # Optimizer step (every gradient_accumulation_steps)
        is_last_batch = (batch_idx == len(train_loader) - 1)
        if accumulated_batch_count == accumulation_steps or is_last_batch:
            optimizer_step_start = time.time()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients and track on GPU (no .item() calls here)
            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )

            # Update gradient norm running statistics on GPU
            running_max_grad_norm = torch.maximum(running_max_grad_norm, grad_norm_before_clip.unsqueeze(0))
            running_sum_grad_norm += grad_norm_before_clip
            n_optimizer_steps_local += 1

            # NOTE: Per-module gradient norm tracking DISABLED for optimizer speed
            # This was causing 10 CPU syncs per optimizer step on rank 0
            # Uncomment below if debugging gradient flow issues
            # if rank == 0:
            #     base_model = model.module if hasattr(model, 'module') else model
            #     module_prefixes = {
            #         'feature_extractor': 'feature_extractor',
            #         'naip_encoder': 'naip_encoder',
            #         'uavsar_encoder': 'uavsar_encoder',
            #         'fusion': 'fusion',
            #         'raster_head': 'raster_head',
            #     }
            #     for cat, prefix in module_prefixes.items():
            #         # Collect all gradients for this module
            #         grads = [
            #             p.grad.flatten()
            #             for n, p in base_model.named_parameters()
            #             if n.startswith(prefix) and p.grad is not None
            #         ]
            #         if grads:
            #             all_grads = torch.cat(grads)
            #             # Compute stats in bulk on GPU, single .item() call per module
            #             module_norm = all_grads.norm().item()
            #             module_max = all_grads.abs().max().item()
            #             module_grad_norms[cat]['sum'] += module_norm
            #             module_grad_norms[cat]['max'] = max(module_grad_norms[cat]['max'], module_max)
            #             module_grad_norms[cat]['count'] += 1

            # NOTE: Per-batch gradient norm logging removed for GPU efficiency
            # Epoch-level max and mean gradient norms logged at epoch end

            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            optimizer_time += (time.time() - optimizer_step_start)

            current_optimizer_step += 1
            accumulated_batch_count = 0

        # Reset timer for next batch
        batch_start_time = time.time()

    # =========================================================================
    # Epoch-End Metric Computation (single .item() calls here only)
    # All running statistics synchronized across GPUs before CPU transfer
    # =========================================================================
    import math

    # Gather loss from all GPUs (reduce running sums, then average)
    world_size = dist.get_world_size()

    # All-reduce running sums across GPUs (SUM operation)
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_huber_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_mse_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_corr_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_loss_per_band, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_pred, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_targ, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_sq_pred, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_sq_targ, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_cross, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_mae, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_max_grad_norm, op=dist.ReduceOp.MAX)
    dist.all_reduce(running_sum_grad_norm, op=dist.ReduceOp.SUM)

    # Reduce batch_count and n_samples across GPUs (reuse pre-allocated tensors)
    batch_count_tensor.fill_(batch_count)
    n_samples_tensor.fill_(n_samples)
    n_optimizer_steps_tensor.fill_(n_optimizer_steps_local)
    dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_optimizer_steps_tensor, op=dist.ReduceOp.SUM)

    total_batch_count = batch_count_tensor.item()
    total_n_samples = n_samples_tensor.item()
    total_optimizer_steps = n_optimizer_steps_tensor.item()

    # Compute averaged loss (single .item() calls here)
    if total_batch_count > 0:
        train_loss_avg = running_loss.item() / total_batch_count
        train_huber_avg = running_huber_loss.item() / total_batch_count
        train_mse_avg = running_mse_loss.item() / total_batch_count
        train_corr_avg = running_corr_loss.item() / total_batch_count if config.correlation_loss_weight > 0 else 0.0
    else:
        train_loss_avg = float('nan')
        train_huber_avg = float('nan')
        train_mse_avg = float('nan')
        train_corr_avg = float('nan')

    # Compute per-band losses dict (for backward compatibility)
    train_loss_per_band = {}
    for band_idx in range(n_bands):
        if total_batch_count > 0:
            train_loss_per_band[f'Band_{band_idx}'] = running_loss_per_band[band_idx].item() / total_batch_count
        else:
            train_loss_per_band[f'Band_{band_idx}'] = float('nan')
    train_loss_per_band['huber_loss'] = train_huber_avg
    train_loss_per_band['mse_loss'] = train_mse_avg
    if config.correlation_loss_weight > 0:
        train_loss_per_band['correlation_loss'] = train_corr_avg

    # Compute overall variance/correlation metrics from running sums
    if total_n_samples > 0:
        pred_mean = running_sum_pred.item() / total_n_samples
        targ_mean = running_sum_targ.item() / total_n_samples
        pred_var = (running_sum_sq_pred.item() / total_n_samples) - (pred_mean ** 2)
        targ_var = (running_sum_sq_targ.item() / total_n_samples) - (targ_mean ** 2)
        covariance = (running_sum_cross.item() / total_n_samples) - (pred_mean * targ_mean)

        overall_pred_std = math.sqrt(max(pred_var, 0))
        overall_targ_std = math.sqrt(max(targ_var, 0))
        overall_correlation = covariance / (overall_pred_std * overall_targ_std + 1e-8) if overall_pred_std > 1e-8 and overall_targ_std > 1e-8 else 0.0
        overall_variance_ratio = overall_pred_std / (overall_targ_std + 1e-8)
        overall_mae = running_mae.item() / total_n_samples
    else:
        overall_pred_std = 0.0
        overall_targ_std = 0.0
        overall_correlation = 0.0
        overall_variance_ratio = 0.0
        overall_mae = float('nan')

    # Compute gradient norm statistics
    if total_optimizer_steps > 0:
        max_grad_norm_epoch = running_max_grad_norm.item()
        mean_grad_norm_epoch = running_sum_grad_norm.item() / total_optimizer_steps
    else:
        max_grad_norm_epoch = 0.0
        mean_grad_norm_epoch = 0.0

    # Log epoch-level metrics and timing (rank 0 only)
    if rank == 0 and writer is not None:
        # Loss metrics
        writer.add_scalar('Loss/train_epoch', train_loss_avg, epoch)
        writer.add_scalar('Loss/train_huber', train_huber_avg, epoch)
        writer.add_scalar('Loss/train_mse', train_mse_avg, epoch)
        if config.correlation_loss_weight > 0:
            writer.add_scalar('Loss/train_correlation', train_corr_avg, epoch)

        # Per-band losses
        for band_idx in range(n_bands):
            writer.add_scalar(f'Loss_PerBand/train_Band_{band_idx}', train_loss_per_band[f'Band_{band_idx}'], epoch)

        # Gradient norms (epoch-level max and mean)
        writer.add_scalar('Gradients/max_norm_epoch', max_grad_norm_epoch, epoch)
        writer.add_scalar('Gradients/mean_norm_epoch', mean_grad_norm_epoch, epoch)

        # NOTE: Per-module gradient norm logging DISABLED (tracking was disabled above)
        # for module_name, stats in module_grad_norms.items():
        #     if stats['count'] > 0:
        #         avg_norm = stats['sum'] / stats['count']
        #         writer.add_scalar(f'Gradients_PerModule/{module_name}_avg', avg_norm, epoch)
        #         writer.add_scalar(f'Gradients_PerModule/{module_name}_max', stats['max'], epoch)

        # Overall variance/correlation metrics (replacing per-band detailed metrics)
        writer.add_scalar('Variance/train_overall_pred_std', overall_pred_std, epoch)
        writer.add_scalar('Variance/train_overall_target_std', overall_targ_std, epoch)
        writer.add_scalar('Variance/train_overall_ratio', overall_variance_ratio, epoch)
        writer.add_scalar('Correlation/train_overall_pearson_r', overall_correlation, epoch)

        # Log timing breakdown (every 5 epochs to reduce overhead)
        if epoch % 5 == 0:
            total_time = data_load_time + forward_time + backward_time + optimizer_time
            if total_time > 0:
                writer.add_scalar('Timing/data_load_time_s', data_load_time, epoch)
                writer.add_scalar('Timing/forward_time_s', forward_time, epoch)
                writer.add_scalar('Timing/backward_time_s', backward_time, epoch)
                writer.add_scalar('Timing/optimizer_time_s', optimizer_time, epoch)
                writer.add_scalar('Timing/total_time_s', total_time, epoch)

                # Percentages
                writer.add_scalar('Timing/data_load_pct', 100 * data_load_time / total_time, epoch)
                writer.add_scalar('Timing/forward_pct', 100 * forward_time / total_time, epoch)
                writer.add_scalar('Timing/backward_pct', 100 * backward_time / total_time, epoch)
                writer.add_scalar('Timing/optimizer_pct', 100 * optimizer_time / total_time, epoch)

    return {
        'loss': train_loss_avg,
        'per_band': train_loss_per_band,
        'batch_count': batch_count,
        'n_good_tiles': n_good_tiles,
        'n_bad_tiles': n_bad_tiles
    }


def validate_one_epoch_ddp(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: MultimodalRasterConfig,
    writer: Optional[SummaryWriter] = None,
    epoch: int = 0,
    rank: int = 0,
    logger: Optional[logging.Logger] = None
) -> Dict[str, float]:
    """
    Validate the model for one epoch using DDP with comprehensive metrics.

    Uses GPU running statistics - no CPU copies during validation loop.
    Per-band extended metrics computed every 5 epochs using GPU running stats.

    Args:
        model: The model to validate
        val_loader: DataLoader for validation data
        device: Device to validate on
        config: Model configuration
        writer: TensorBoard SummaryWriter (optional)
        epoch: Current epoch number
        rank: GPU rank
        logger: Logger instance

    Returns:
        Dict with validation metrics (loss, MAE, R²)
    """
    import math
    model.eval()

    batch_count = 0
    n_bands = config.n_bands

    # =========================================================================
    # GPU Running Statistics (no CPU copies during validation)
    # =========================================================================
    running_loss = torch.zeros(1, device=device)
    running_huber_loss = torch.zeros(1, device=device)
    running_mse_loss = torch.zeros(1, device=device)
    running_loss_per_band = torch.zeros(n_bands, device=device)

    # Overall running statistics for variance/correlation/R² metrics
    running_sum_pred = torch.zeros(1, device=device)
    running_sum_targ = torch.zeros(1, device=device)
    running_sum_sq_pred = torch.zeros(1, device=device)
    running_sum_sq_targ = torch.zeros(1, device=device)
    running_sum_cross = torch.zeros(1, device=device)
    running_mae = torch.zeros(1, device=device)
    running_ss_res = torch.zeros(1, device=device)  # For R²
    n_samples = 0

    # Pre-allocate tensors for epoch-end all_reduce (avoids repeated allocations)
    batch_count_tensor = torch.zeros(1, device=device, dtype=torch.long)
    n_samples_tensor = torch.zeros(1, device=device, dtype=torch.long)

    # Per-band running statistics (for extended metrics every 5 epochs)
    do_extended_metrics = (epoch % 5 == 0)
    if do_extended_metrics:
        running_mae_per_band = torch.zeros(n_bands, device=device)
        running_sum_pred_per_band = torch.zeros(n_bands, device=device)
        running_sum_targ_per_band = torch.zeros(n_bands, device=device)
        running_sum_sq_pred_per_band = torch.zeros(n_bands, device=device)
        running_sum_sq_targ_per_band = torch.zeros(n_bands, device=device)
        running_sum_cross_per_band = torch.zeros(n_bands, device=device)
        n_samples_per_band = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predictions, targets, loss, per_band_losses, diagnostics = process_raster_batch(
                    model, batch, device,
                    use_naip=config.use_naip,
                    use_uavsar=config.use_uavsar,
                    is_training=False,
                    rank=rank,
                    logger=logger,
                    correlation_loss_weight=config.correlation_loss_weight,
                    huber_delta=config.huber_delta
                )

                # Handle invalid loss locally without global synchronization
                # NOTE: Removed per-batch all_reduce for GPU efficiency. In validation,
                # we simply skip invalid batches locally since no backward pass is needed.
                if not diagnostics['loss_is_valid'] or diagnostics['all_bad']:
                    if rank == 0 and logger:
                        logger.warning(f"Validation batch: Skipping due to invalid loss on GPU {rank}")
                    continue

                # =====================================================================
                # GPU Running Statistics Update (no CPU copies)
                # =====================================================================
                # Accumulate loss on GPU
                running_loss += loss.detach()
                running_huber_loss += per_band_losses['huber_loss'].detach()
                running_mse_loss += per_band_losses['mse_loss'].detach()

                # Per-band MSE losses
                for band_idx in range(n_bands):
                    band_key = f'Band_{band_idx}'
                    if band_key in per_band_losses:
                        running_loss_per_band[band_idx] += per_band_losses[band_key].detach()

                # Overall running statistics
                pred_flat = predictions.detach().flatten()
                targ_flat = targets.detach().flatten()
                diff = pred_flat - targ_flat

                running_sum_pred += pred_flat.sum()
                running_sum_targ += targ_flat.sum()
                running_sum_sq_pred += (pred_flat ** 2).sum()
                running_sum_sq_targ += (targ_flat ** 2).sum()
                running_sum_cross += (pred_flat * targ_flat).sum()
                running_mae += diff.abs().sum()
                running_ss_res += (diff ** 2).sum()
                n_samples += pred_flat.numel()

                # Per-band extended metrics (every 5 epochs)
                if do_extended_metrics:
                    # Sum over batch and spatial dims, keep band dim
                    pred_banded = predictions.detach()  # [B, n_bands, 5, 5]
                    targ_banded = targets.detach()

                    running_mae_per_band += (pred_banded - targ_banded).abs().sum(dim=(0, 2, 3))
                    running_sum_pred_per_band += pred_banded.sum(dim=(0, 2, 3))
                    running_sum_targ_per_band += targ_banded.sum(dim=(0, 2, 3))
                    running_sum_sq_pred_per_band += (pred_banded ** 2).sum(dim=(0, 2, 3))
                    running_sum_sq_targ_per_band += (targ_banded ** 2).sum(dim=(0, 2, 3))
                    running_sum_cross_per_band += (pred_banded * targ_banded).sum(dim=(0, 2, 3))
                    n_samples_per_band += pred_banded.shape[0] * 25  # B * 5 * 5

                batch_count += 1

    # =========================================================================
    # Epoch-End Metric Computation (single .item() calls here only)
    # =========================================================================
    world_size = dist.get_world_size()

    # All-reduce running sums across GPUs
    dist.all_reduce(running_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_huber_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_mse_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_loss_per_band, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_pred, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_targ, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_sq_pred, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_sq_targ, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_sum_cross, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_mae, op=dist.ReduceOp.SUM)
    dist.all_reduce(running_ss_res, op=dist.ReduceOp.SUM)

    if do_extended_metrics:
        dist.all_reduce(running_mae_per_band, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_sum_pred_per_band, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_sum_targ_per_band, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_sum_sq_pred_per_band, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_sum_sq_targ_per_band, op=dist.ReduceOp.SUM)
        dist.all_reduce(running_sum_cross_per_band, op=dist.ReduceOp.SUM)

    # Reduce counts across GPUs (reuse pre-allocated tensors)
    batch_count_tensor.fill_(batch_count)
    n_samples_tensor.fill_(n_samples)
    dist.all_reduce(batch_count_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(n_samples_tensor, op=dist.ReduceOp.SUM)

    total_batch_count = batch_count_tensor.item()
    total_n_samples = n_samples_tensor.item()

    # Compute averaged loss
    if total_batch_count > 0:
        val_loss_avg = running_loss.item() / total_batch_count
        val_huber_avg = running_huber_loss.item() / total_batch_count
        val_mse_avg = running_mse_loss.item() / total_batch_count
    else:
        val_loss_avg = float('nan')
        val_huber_avg = float('nan')
        val_mse_avg = float('nan')

    # Per-band losses dict
    val_loss_per_band = {}
    for band_idx in range(n_bands):
        if total_batch_count > 0:
            val_loss_per_band[f'Band_{band_idx}'] = running_loss_per_band[band_idx].item() / total_batch_count
        else:
            val_loss_per_band[f'Band_{band_idx}'] = float('nan')
    val_loss_per_band['huber_loss'] = val_huber_avg
    val_loss_per_band['mse_loss'] = val_mse_avg

    # Compute overall metrics from running sums
    metrics = {'loss': val_loss_avg, 'per_band': val_loss_per_band}

    if total_n_samples > 0:
        # MAE
        overall_mae = running_mae.item() / total_n_samples
        metrics['mae'] = overall_mae

        # R² using running variance approach (single pass)
        targ_mean = running_sum_targ.item() / total_n_samples
        targ_var = (running_sum_sq_targ.item() / total_n_samples) - (targ_mean ** 2)
        ss_tot = total_n_samples * targ_var
        ss_res = running_ss_res.item()
        r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 1e-8 else 0.0
        metrics['r2'] = r2

        # Overall variance/correlation
        pred_mean = running_sum_pred.item() / total_n_samples
        pred_var = (running_sum_sq_pred.item() / total_n_samples) - (pred_mean ** 2)
        covariance = (running_sum_cross.item() / total_n_samples) - (pred_mean * targ_mean)

        overall_pred_std = math.sqrt(max(pred_var, 0))
        overall_targ_std = math.sqrt(max(targ_var, 0))
        overall_correlation = covariance / (overall_pred_std * overall_targ_std + 1e-8) if overall_pred_std > 1e-8 and overall_targ_std > 1e-8 else 0.0
        overall_variance_ratio = overall_pred_std / (overall_targ_std + 1e-8)
    else:
        overall_mae = float('nan')
        r2 = float('nan')
        overall_pred_std = 0.0
        overall_targ_std = 0.0
        overall_correlation = 0.0
        overall_variance_ratio = 0.0
        metrics['mae'] = overall_mae
        metrics['r2'] = r2

    # Log epoch-level metrics
    if rank == 0 and writer is not None:
        writer.add_scalar('Loss/val_epoch', val_loss_avg, epoch)
        writer.add_scalar('Loss/val_huber', val_huber_avg, epoch)
        writer.add_scalar('Loss/val_mse', val_mse_avg, epoch)

        # Per-band losses
        for band_idx in range(n_bands):
            writer.add_scalar(f'Loss_PerBand/val_Band_{band_idx}', val_loss_per_band[f'Band_{band_idx}'], epoch)

        # Overall metrics
        writer.add_scalar('Val_Metrics/overall_mae', overall_mae, epoch)
        writer.add_scalar('Val_Metrics/overall_r2', r2, epoch)
        writer.add_scalar('Variance/val_overall_pred_std', overall_pred_std, epoch)
        writer.add_scalar('Variance/val_overall_target_std', overall_targ_std, epoch)
        writer.add_scalar('Variance/val_overall_ratio', overall_variance_ratio, epoch)
        writer.add_scalar('Correlation/val_overall_pearson_r', overall_correlation, epoch)

        # Per-band extended metrics (every 5 epochs) - compute for metrics dict only
        # TensorBoard per-band logging removed for GPU efficiency
        if do_extended_metrics and total_n_samples > 0:
            total_n_samples_band = total_n_samples
            if total_n_samples_band > 0:
                for band_idx in range(n_bands):
                    # Per-band MAE (metrics dict only)
                    band_mae = running_mae_per_band[band_idx].item() / (total_n_samples_band / n_bands)
                    metrics[f'band_{band_idx}_mae'] = band_mae

                    # Per-band correlation (metrics dict only)
                    n_per_band = total_n_samples_band / n_bands
                    pred_mean_band = running_sum_pred_per_band[band_idx].item() / n_per_band
                    targ_mean_band = running_sum_targ_per_band[band_idx].item() / n_per_band
                    pred_var_band = (running_sum_sq_pred_per_band[band_idx].item() / n_per_band) - (pred_mean_band ** 2)
                    targ_var_band = (running_sum_sq_targ_per_band[band_idx].item() / n_per_band) - (targ_mean_band ** 2)
                    cov_band = (running_sum_cross_per_band[band_idx].item() / n_per_band) - (pred_mean_band * targ_mean_band)
                    pred_std_band = math.sqrt(max(pred_var_band, 0))
                    targ_std_band = math.sqrt(max(targ_var_band, 0))
                    corr_band = cov_band / (pred_std_band * targ_std_band + 1e-8) if pred_std_band > 1e-8 and targ_std_band > 1e-8 else 0.0
                    metrics[f'band_{band_idx}_correlation'] = corr_band

    return metrics


def train_raster_worker(
    rank: int,
    world_size: int,
    config: MultimodalRasterConfig,
    train_shard_dir: str,
    train_shard_prefix: str,
    val_data_path: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 15,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    save_every_n_epochs: int = 10,
    use_amp: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = 'loss',
    seed: int = 42,
    beta1: float = 0.9,
    beta2: float = 0.999,
    max_grad_norm: float = 10.0,
    warmup_steps_percentage: float = 0.05,
    resume_checkpoint_path: Optional[str] = None
):
    """
    Training worker for distributed data parallel training.

    Args:
        rank: Process rank (GPU ID)
        world_size: Total number of processes (GPUs)
        config: Model configuration
        train_shard_dir: Directory containing pre-sharded training data
        train_shard_prefix: Prefix for training shard files
        val_data_path: Path to validation data .pt file
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimizer
        gradient_accumulation_steps: Steps to accumulate gradients
        save_every_n_epochs: Save checkpoint every N epochs
        use_amp: Use automatic mixed precision
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_metric: Metric to use for early stopping ('loss' or 'mae')
        seed: Random seed
        resume_checkpoint_path: Path to checkpoint to resume training from (optional)
    """
    # Setup distributed training (MASTER_ADDR and MASTER_PORT already set by parent process)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Set seed for reproducibility
    torch.manual_seed(seed + rank)

    # Create output directory
    if rank == 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = Path(output_dir) / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)

    # Validate early stopping metric
    if early_stopping_metric not in ['loss', 'mae']:
        raise ValueError(f"early_stopping_metric must be 'loss' or 'mae', got '{early_stopping_metric}'")

    # Setup logging (rank 0 only)
    logger = None
    if rank == 0:
        log_file = Path(output_dir) / 'training.log'
        logger = setup_logging('raster_training', str(log_file))
        logger.info(f"Starting training on rank 0")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Configuration: use_naip={config.use_naip}, use_uavsar={config.use_uavsar}")
        logger.info(f"Hyperparameters: epochs={num_epochs}, batch_size={batch_size}, "
                   f"lr={learning_rate}, weight_decay={weight_decay}, "
                   f"early_stopping_patience={early_stopping_patience}, "
                   f"early_stopping_metric={early_stopping_metric}")

    # Create TensorBoard writer (rank 0 only)
    writer = None
    if rank == 0 and HAS_TENSORBOARD:
        log_dir = Path(output_dir) / 'logs'
        log_dir.mkdir(exist_ok=True)
        writer = SummaryWriter(str(log_dir))
        print(f"TensorBoard logs will be saved to: {log_dir}")
        if logger:
            logger.info(f"TensorBoard logs will be saved to: {log_dir}")

    # Compute this GPU's shard paths
    train_shard_path = os.path.join(train_shard_dir, f"{train_shard_prefix}_shard_{rank}.pt")

    # Load pre-sharded training data
    # Note: k (KNN neighbors) not passed - global-only attention doesn't use KNN graphs
    print(f"[GPU {rank}] Loading training shard from {train_shard_path}")
    train_dataset = ShardedRasterDataset(
        shard_path=train_shard_path,
        use_naip=config.use_naip,
        use_uavsar=config.use_uavsar,
        target_band_indices=config.target_band_indices
    )

    # Load validation data
    if rank == 0:
        print(f"[GPU {rank}] Loading validation data from {val_data_path}")
    val_dataset = ShardedRasterDataset(
        shard_path=val_data_path,
        use_naip=config.use_naip,
        use_uavsar=config.use_uavsar,
        target_band_indices=config.target_band_indices
    )

    print(f"[GPU {rank}] Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    # Create samplers for distributed training
    if world_size > 1:
        # NOTE: Shards are load-balanced but NOT GPU-exclusive!
        # DistributedSampler is STILL NEEDED to divide each shard across GPUs.
        # Without it, each GPU processes the entire shard (3× more work).
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        collate_fn=raster_variable_size_collate,
        pin_memory=True
        # NOTE: num_workers MUST be 0 with current Dataset architecture
        # Dataset loads entire shard (12k tiles) in __init__, which cannot be pickled
        # to workers without exhausting shared memory file descriptors.
        # To enable num_workers > 0, would need to refactor to lazy loading.
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        collate_fn=raster_variable_size_collate,
        pin_memory=True
        # NOTE: num_workers MUST be 0 (see train_loader comment)
    )

    # Create model
    model = MultimodalRasterPredictor(config).to(device)

    # NOTE: torch.compile() disabled - incompatible with Kornia augmentations
    # Kornia uses .item() calls and CPU/GPU mixed operations that cause graph breaks
    # and compilation failures. If augmentations are disabled, torch.compile() could
    # be re-enabled for additional speedup.

    # Wrap with DDP
    # find_unused_parameters=False is safe because embedding dropout uses * 0.0
    # which preserves gradient flow through encoder params even when modalities are "dropped"
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Create SWA averaged model (if enabled)
    swa_model = None
    if config.swa_enabled:
        # AveragedModel wraps the base model (not DDP) for averaging
        swa_model = AveragedModel(model.module)
        if rank == 0:
            print(f"[GPU {rank}] SWA enabled: averaging starts at epoch {config.swa_start_epoch}")

    # Calculate warmup steps from percentage of total training steps
    total_batches = len(train_loader)
    total_training_steps = total_batches * num_epochs
    warmup_steps = int(warmup_steps_percentage * total_training_steps)

    if rank == 0:
        print(f"[GPU {rank}] Total training steps: {total_training_steps} ({total_batches} batches × {num_epochs} epochs)")
        print(f"[GPU {rank}] Warmup steps: {warmup_steps} ({warmup_steps_percentage*100:.1f}% of total)")

    # Separate parameters: exclude bias and norm layers from weight decay
    # This is a best practice that prevents regularizing shift/scale params
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Exclude: bias terms, LayerNorm weight/bias, any param with 'norm' in name
        if 'bias' in name or 'norm' in name.lower() or name.endswith('.gamma') or name.endswith('.beta'):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    # Create optimizer with parameter groups
    optimizer = AdamWScheduleFree(
        param_groups,
        lr=learning_rate,
        betas=(beta1, beta2),
        warmup_steps=warmup_steps,
        eps=1e-5
    )

    if rank == 0:
        print(f"[GPU {rank}] Params with weight decay: {len(decay_params)}, without: {len(no_decay_params)}")
    print(f"[GPU {rank}] Using AdamWScheduleFree with lr={learning_rate}, weight decay={weight_decay}, betas=({beta1}, {beta2}), warmup_steps={warmup_steps}")

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Resume from checkpoint if provided
    start_epoch = 0
    best_early_stop_metric = float('inf')  # Tracks early_stopping_metric (loss or mae)
    epochs_without_improvement = 0

    if resume_checkpoint_path is not None:
        if rank == 0:
            print(f"\n{'='*80}")
            print(f"RESUMING FROM CHECKPOINT: {resume_checkpoint_path}")
            print(f"{'='*80}")

        checkpoint = torch.load(resume_checkpoint_path, map_location=device, weights_only=False)

        # Load model state
        model.module.load_state_dict(checkpoint['model_state_dict'])
        if rank == 0:
            print(f"  ✓ Loaded model weights")

        # Load optimizer state
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if rank == 0:
            print(f"  ✓ Loaded optimizer state")

        # Override betas with current config values (allows changing mid-training)
        # This preserves momentum buffers but updates the decay rates
        checkpoint_betas = optimizer.param_groups[0]['betas']
        for param_group in optimizer.param_groups:
            param_group['betas'] = (beta1, beta2)
        if rank == 0:
            print(f"  ✓ Overriding betas to ({beta1}, {beta2})")
            if logger:
                logger.info(f"Overriding betas: {checkpoint_betas} -> ({beta1}, {beta2})")

        # Override weight_decay for param groups that have it (not bias/norm layers)
        # Get checkpoint's weight_decay from first group that has it
        checkpoint_wd = None
        for param_group in optimizer.param_groups:
            if param_group['weight_decay'] > 0:
                checkpoint_wd = param_group['weight_decay']
                break

        if checkpoint_wd is not None:
            if weight_decay > checkpoint_wd and rank == 0:
                print(f"  ⚠ WARNING: Increasing weight_decay from {checkpoint_wd} to {weight_decay}")
                print(f"    This may destabilize training. Consider smaller increases.")
                if logger:
                    logger.warning(f"Increasing weight_decay from {checkpoint_wd} to {weight_decay} - may destabilize training")

            for param_group in optimizer.param_groups:
                if param_group['weight_decay'] > 0:
                    param_group['weight_decay'] = weight_decay

            if rank == 0:
                print(f"  ✓ Overriding weight_decay to {weight_decay} (was {checkpoint_wd})")
                if logger:
                    logger.info(f"Overriding weight_decay: {checkpoint_wd} -> {weight_decay}")

        # Get starting epoch (resume from next epoch)
        start_epoch = checkpoint.get('epoch', 0) + 1
        if rank == 0:
            print(f"  ✓ Resuming from epoch {start_epoch} (checkpoint was epoch {checkpoint.get('epoch', 0)})")

        # Load best metric value if available (use the appropriate metric)
        if early_stopping_metric == 'mae' and 'val_mae' in checkpoint and checkpoint['val_mae'] is not None:
            best_early_stop_metric = checkpoint['val_mae']
            if rank == 0:
                print(f"  ✓ Best MAE so far: {best_early_stop_metric:.6f}")
        elif early_stopping_metric == 'loss' and 'val_loss' in checkpoint:
            best_early_stop_metric = checkpoint['val_loss']
            if rank == 0:
                print(f"  ✓ Best loss so far: {best_early_stop_metric:.6f}")
        else:
            # If metric not in checkpoint, start fresh tracking
            if rank == 0:
                print(f"  ⚠ Best {early_stopping_metric} not in checkpoint, starting fresh early stopping tracking")

        if rank == 0:
            remaining_epochs = num_epochs - start_epoch
            print(f"  → Will train for {remaining_epochs} more epochs (epochs {start_epoch} to {num_epochs-1})")
            print(f"{'='*80}\n")
            if logger:
                logger.info(f"Resumed from checkpoint: {resume_checkpoint_path}")
                logger.info(f"Starting at epoch {start_epoch}, best {early_stopping_metric}: {best_early_stop_metric:.6f}")

    training_history = {'train_loss': [], 'val_loss': [], 'val_mae': [], 'learning_rates': []}

    # Pre-allocate tensor for early stopping broadcast (avoids repeated allocations)
    early_stop_tensor = torch.zeros(1, device=device, dtype=torch.int)

    for epoch in range(start_epoch, num_epochs):
        epoch_start_time = time.time()

        # Set epoch for distributed sampler
        if world_size > 1 and train_sampler is not None:
            train_sampler.set_epoch(epoch)

        # Training phase
        train_metrics = train_one_epoch_ddp(
            model, train_loader, optimizer, device, scaler, config,
            writer=writer, epoch=epoch, accumulation_steps=gradient_accumulation_steps,
            rank=rank, logger=logger, max_grad_norm=max_grad_norm
        )

        # Switch optimizer to eval mode before validation (ScheduleFreeAdamW requirement)
        optimizer.eval()

        # Validation phase
        val_metrics = validate_one_epoch_ddp(
            model, val_loader, device, config,
            writer=writer, epoch=epoch, rank=rank, logger=logger
        )

        # Update SWA model if enabled and past start epoch
        if swa_model is not None and epoch >= config.swa_start_epoch:
            if (epoch - config.swa_start_epoch) % config.swa_update_freq == 0:
                swa_model.update_parameters(model.module)
                if rank == 0:
                    print(f"  → SWA: Updated averaged model (epoch {epoch})")

        epoch_time = time.time() - epoch_start_time

        # Log progress (rank 0 only)
        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']

            # Format per-band loss strings
            train_band_str = " | ".join([f"Band_{i}: {train_metrics['per_band'].get(f'Band_{i}', 0):.6f}"
                                         for i in range(config.n_bands)])
            val_band_str = " | ".join([f"Band_{i}: {val_metrics['per_band'].get(f'Band_{i}', 0):.6f}"
                                       for i in range(config.n_bands)])

            log_message = (f"Epoch {epoch+1}/{num_epochs} | "
                          f"Train Loss: {train_metrics['loss']:.6f} ({train_band_str}) | "
                          f"Val Loss: {val_metrics['loss']:.6f} ({val_band_str})")

            # Add metrics if available
            if 'mae' in val_metrics:
                log_message += f" | MAE: {val_metrics['mae']:.6f}"
            if 'r2' in val_metrics:
                log_message += f" | R²: {val_metrics['r2']:.6f}"

            log_message += f" | LR: {current_lr:.2e} | Time: {epoch_time:.1f}s"

            print(log_message)
            if logger:
                logger.info(log_message)

            training_history['train_loss'].append(train_metrics['loss'])
            training_history['val_loss'].append(val_metrics['loss'])
            if 'mae' in val_metrics:
                training_history['val_mae'].append(val_metrics['mae'])
            training_history['learning_rates'].append(current_lr)

            # Early stopping logic
            current_metric_value = val_metrics[early_stopping_metric]
            if current_metric_value < best_early_stop_metric:
                best_early_stop_metric = current_metric_value
                epochs_without_improvement = 0
                checkpoint_path = Path(output_dir) / 'checkpoints' / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_mae': val_metrics.get('mae', None),
                    'config': config
                }, checkpoint_path)
                msg = f"✓ Saved best model ({early_stopping_metric}: {current_metric_value:.6f})"
                print(f"  → {msg}")
                if logger:
                    logger.info(msg)
            else:
                epochs_without_improvement += 1
                if logger:
                    logger.info(f"No improvement in {early_stopping_metric} for {epochs_without_improvement} epochs")

            # Check early stopping
            if epochs_without_improvement >= early_stopping_patience:
                msg = f"Early stopping triggered at epoch {epoch+1} (no improvement in {early_stopping_metric} for {early_stopping_patience} epochs)"
                print(f"  → {msg}")
                if logger:
                    logger.info(msg)
                # Break only on rank 0 is not enough - need to broadcast
                early_stop = True
            else:
                early_stop = False
        else:
            early_stop = False

        # Broadcast early stopping decision to all GPUs (reuse pre-allocated tensor)
        early_stop_tensor.fill_(int(early_stop))
        dist.broadcast(early_stop_tensor, src=0)
        if early_stop_tensor.item():
            if rank == 0 and logger:
                logger.info("Exiting training loop due to early stopping")
            break

        # Save periodic checkpoints
        if rank == 0 and (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = Path(output_dir) / 'checkpoints' / f'epoch_{epoch+1}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'config': config
            }, checkpoint_path)

    # Ensure optimizer is in eval mode before saving final checkpoint (ScheduleFreeAdamW requirement)
    optimizer.eval()

    # Save final model and training history
    if rank == 0:
        final_checkpoint_path = Path(output_dir) / 'checkpoints' / 'final_model.pth'
        torch.save({
            'epoch': epoch,  # Actual last completed epoch (not num_epochs - 1)
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'val_mae': val_metrics.get('mae', None),
            'config': config
        }, final_checkpoint_path)
        if logger:
            logger.info(f"Final model saved to {final_checkpoint_path}")

        # Save SWA averaged model if enabled
        if swa_model is not None:
            # Note: Our model uses LayerNorm (not BatchNorm), so update_bn is not needed
            # If you add BatchNorm layers in the future, uncomment:
            # swa_model.train()
            # update_bn(train_loader, swa_model, device=device)
            # swa_model.eval()

            swa_checkpoint_path = Path(output_dir) / 'checkpoints' / 'swa_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': swa_model.module.state_dict(),
                'config': config
            }, swa_checkpoint_path)
            if logger:
                logger.info(f"SWA averaged model saved to {swa_checkpoint_path}")
            print(f"  → SWA averaged model saved to {swa_checkpoint_path}")

            # Validate SWA model performance
            swa_model.eval()
            swa_val_metrics = validate_one_epoch_ddp(
                swa_model, val_loader, device, config,
                writer=None, epoch=epoch, rank=rank, logger=logger
            )
            swa_mae = swa_val_metrics.get('mae', float('nan'))
            print(f"  → SWA Model Validation - Loss: {swa_val_metrics['loss']:.6f}, MAE: {swa_mae:.6f}")
            if logger:
                logger.info(f"SWA validation - Loss: {swa_val_metrics['loss']:.6f}, MAE: {swa_mae:.6f}")

        history_path = Path(output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        if logger:
            logger.info(f"Training history saved to {history_path}")

        # Close TensorBoard writer
        if writer is not None:
            writer.close()

        completion_msg = (f"\n{'='*80}\nTraining complete!\n"
                         f"Best {early_stopping_metric}: {best_early_stop_metric:.6f}\n"
                         f"Checkpoints saved to: {Path(output_dir) / 'checkpoints'}\n"
                         f"Logs saved to: {Path(output_dir) / 'logs'}\n{'='*80}")
        print(completion_msg)
        if logger:
            logger.info(f"Training complete. Best {early_stopping_metric}: {best_early_stop_metric:.6f}")

        # Save best metric to file for retrieval by main process (rank 0 only)
        best_metric_file = Path(output_dir) / 'best_metric.json'
        with open(best_metric_file, 'w') as f:
            json.dump({
                'best_metric': float(best_early_stop_metric),
                'metric_name': early_stopping_metric
            }, f)

    # Synchronize all ranks before cleanup to prevent deadlock
    if dist.is_initialized():
        dist.barrier()

    # Cleanup (graceful timeout handling for NCCL operations)
    try:
        dist.destroy_process_group()
    except Exception as e:
        if rank == 0 and logger:
            logger.warning(f"Cleanup timeout (training already completed): {str(e)[:100]}")
        # Process group cleanup may timeout after long training sessions, but this doesn't
        # affect the training results since the best model was already saved


def train_raster_model(
    config: MultimodalRasterConfig,
    train_data_path: str,
    val_data_path: str,
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 15,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    save_every_n_epochs: int = 10,
    use_amp: bool = True,
    early_stopping_patience: int = 10,
    early_stopping_metric: str = 'loss',
    seed: int = 42,
    num_gpus: Optional[int] = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    max_grad_norm: float = 10.0,
    warmup_steps_percentage: float = 0.05,
    resume_checkpoint_path: Optional[str] = None
):
    """
    Main entry point for training raster prediction model.

    Args:
        config: Model configuration
        train_data_path: Path to training data
        val_data_path: Path to validation data
        output_dir: Output directory for checkpoints
        num_epochs: Number of epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        weight_decay: Weight decay
        gradient_accumulation_steps: Gradient accumulation steps
        save_every_n_epochs: Save frequency
        use_amp: Use mixed precision
        early_stopping_patience: Epochs without improvement before stopping
        early_stopping_metric: Metric to use for early stopping ('loss' or 'mae')
        seed: Random seed
        num_gpus: Number of GPUs (None = all available)
        beta1: AdamW beta1 parameter
        beta2: AdamW beta2 parameter
        max_grad_norm: Gradient clipping threshold
        warmup_steps_percentage: Percentage of total steps for warmup
        resume_checkpoint_path: Path to checkpoint to resume training from (optional)
    """
    import hashlib

    # Determine world size
    if num_gpus is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(num_gpus, torch.cuda.device_count())

    print(f"Training with {world_size} GPUs")

    # Create persistent cache directory for shards (outside timestamped output_dir)
    shard_cache_dir = Path('data/output/cached_shards')
    shard_cache_dir.mkdir(parents=True, exist_ok=True)

    # Create cache key based on data paths
    cache_key = hashlib.md5((train_data_path + val_data_path).encode()).hexdigest()[:10]
    train_shard_prefix = f"{cache_key}_train"

    # Check if shards already exist (cached)
    train_shard_paths = [str(shard_cache_dir / f"{train_shard_prefix}_shard_{i}.pt")
                         for i in range(world_size)]
    shards_exist = all(os.path.exists(p) for p in train_shard_paths)

    if not shards_exist:
        print(f"\n{'='*80}")
        print("Creating training shards (this is a one-time operation)...")
        print(f"{'='*80}")
        # Load full training data
        print(f"Loading training data from {train_data_path}")
        train_data = torch.load(train_data_path, weights_only=False)
        print(f"  ✓ Loaded {len(train_data)} training samples")

        # Create training shards
        train_shard_paths = create_raster_shards(
            train_data, world_size, str(shard_cache_dir), train_shard_prefix
        )
        del train_data
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("Using cached shards (loading from disk)...")
        print(f"{'='*80}\n")

    # Setup distributed training environment
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        # Find a free port to avoid conflicts with previous runs
        free_port = find_free_port()
        os.environ['MASTER_PORT'] = str(free_port)
        print(f"Using port {free_port} for distributed training")

    # Spawn training workers
    torch.multiprocessing.spawn(
        train_raster_worker,
        args=(world_size, config, str(shard_cache_dir), train_shard_prefix, val_data_path,
              output_dir, num_epochs, batch_size,
              learning_rate, weight_decay, gradient_accumulation_steps,
              save_every_n_epochs, use_amp, early_stopping_patience, early_stopping_metric, seed,
              beta1, beta2, max_grad_norm, warmup_steps_percentage, resume_checkpoint_path),
        nprocs=world_size,
        join=True
    )

    # Read and return best metric from training
    best_metric_file = Path(output_dir) / 'best_metric.json'
    if best_metric_file.exists():
        with open(best_metric_file, 'r') as f:
            metric_data = json.load(f)
        return metric_data['best_metric']
    else:
        raise RuntimeError(f"Training completed but best metric file not found: {best_metric_file}")
