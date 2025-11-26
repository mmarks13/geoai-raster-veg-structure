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

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("Warning: psutil not available for CPU monitoring")

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except:
    HAS_PYNVML = False
    print("Warning: pynvml not available for detailed GPU monitoring")

# Import raster-specific components
from src.models.multimodal_raster_model import MultimodalRasterPredictor, MultimodalRasterConfig
from src.training.raster_dataset import ShardedRasterDataset, raster_variable_size_collate

# Import shared utilities
from src.training.ddp_training import setup_logging
from schedulefree import AdamWScheduleFree


def log_system_metrics(writer: Optional[SummaryWriter], step: int, device: torch.device, rank: int = 0):
    """
    Log GPU and CPU utilization metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter
        step: Global step number for logging
        device: CUDA device
        rank: GPU rank (only rank 0 logs)
    """
    if writer is None or rank != 0:
        return

    # GPU Memory metrics
    if device.type == 'cuda':
        gpu_mem_allocated = torch.cuda.memory_allocated(device) / 1e9  # GB
        gpu_mem_reserved = torch.cuda.memory_reserved(device) / 1e9    # GB
        gpu_mem_free = (torch.cuda.get_device_properties(device).total_memory -
                        torch.cuda.memory_reserved(device)) / 1e9      # GB

        writer.add_scalar('System/GPU_Memory_Allocated_GB', gpu_mem_allocated, step)
        writer.add_scalar('System/GPU_Memory_Reserved_GB', gpu_mem_reserved, step)
        writer.add_scalar('System/GPU_Memory_Free_GB', gpu_mem_free, step)

        # GPU utilization (requires pynvml)
        if HAS_PYNVML:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device.index)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                writer.add_scalar('System/GPU_Compute_Utilization_%', util.gpu, step)
                writer.add_scalar('System/GPU_Memory_Utilization_%', util.memory, step)
            except:
                pass  # Skip if pynvml fails

    # CPU metrics
    if HAS_PSUTIL:
        cpu_percent = psutil.cpu_percent(interval=None)  # Non-blocking
        cpu_per_core = psutil.cpu_percent(interval=None, percpu=True)
        mem = psutil.virtual_memory()

        writer.add_scalar('System/CPU_Utilization_%', cpu_percent, step)
        writer.add_scalar('System/CPU_Memory_Used_GB', mem.used / 1e9, step)
        writer.add_scalar('System/CPU_Memory_Available_GB', mem.available / 1e9, step)

        # Log per-core utilization (first 8 cores to avoid clutter)
        for i, core_util in enumerate(cpu_per_core[:8]):
            writer.add_scalar(f'System/CPU_Core_{i}_Utilization_%', core_util, step)


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
    logger: Optional[logging.Logger] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]:
    """
    Process a single batch for raster prediction with per-tile NaN diagnostics.

    Args:
        model: Raster prediction model
        batch: 12-element tuple from raster_variable_size_collate
        device: Device to run on
        use_naip: Whether NAIP imagery is used
        use_uavsar: Whether UAVSAR imagery is used
        is_training: Whether in training mode
        rank: GPU rank for logging
        logger: Logger instance

    Returns:
        Tuple of (predictions, targets, loss, per_band_losses, diagnostics)
        - predictions: [batch_size, n_bands, 5, 5] predicted rasters
        - targets: [batch_size, n_bands, 5, 5] ground truth rasters
        - loss: scalar MSE loss (or NaN if batch is bad)
        - per_band_losses: Dict with per-band MSE losses
        - diagnostics: Dict with batch status and bad tile info
    """
    # Unpack batch (12 elements)
    (dep_points_batch, fuel_metrics_batch, edge_index_batch, dep_points_attr_batch,
     naip_data_batch, uavsar_data_batch, centers, scales, bboxes, tile_ids,
     norm_params_list, batch_indices) = batch

    # Move tensors to device
    dep_points_batch = dep_points_batch.to(device)
    fuel_metrics_batch = fuel_metrics_batch.to(device)
    edge_index_batch = edge_index_batch.to(device)
    dep_points_attr_batch = dep_points_attr_batch.to(device)
    batch_indices = batch_indices.to(device)
    bboxes = bboxes.to(device) if bboxes is not None else None

    # Handle imagery data (list of dicts)
    naip_data = naip_data_batch if use_naip else None
    uavsar_data = uavsar_data_batch if use_uavsar else None

    # Forward pass
    predictions = model(
        dep_points=dep_points_batch,
        edge_index=edge_index_batch,
        batch_indices=batch_indices,
        norm_params=norm_params_list,
        dep_attr=dep_points_attr_batch,
        naip=naip_data,
        uavsar=uavsar_data,
        bbox=bboxes
    )  # [batch_size, n_bands, 5, 5]

    # Compute MSE loss on normalized values
    loss = nn.functional.mse_loss(predictions, fuel_metrics_batch)

    # Compute per-band losses
    per_band_losses = {}
    n_bands = predictions.shape[1]
    for band_idx in range(n_bands):
        band_loss = nn.functional.mse_loss(predictions[:, band_idx], fuel_metrics_batch[:, band_idx])
        per_band_losses[f'Band_{band_idx}'] = band_loss

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

    train_loss_total = 0.0
    train_loss_per_band = {}
    batch_count = 0
    accumulated_batch_count = 0
    n_good_tiles = 0
    n_bad_tiles = 0

    current_optimizer_step = 0
    current_batch_step = epoch * len(train_loader)

    # Zero gradients at start of epoch
    optimizer.zero_grad()

    # Timing trackers
    data_load_time = 0.0
    forward_time = 0.0
    backward_time = 0.0
    optimizer_time = 0.0
    batch_start_time = time.time()

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
                logger=logger
            )
        forward_time += (time.time() - forward_start)

        # Synchronize skip decision across all GPUs (critical for DDP)
        # If ANY GPU has invalid loss, ALL GPUs must skip to avoid gradient sync deadlock
        skip_batch = int(not diagnostics['loss_is_valid'] or diagnostics['all_bad'])
        skip_batch_tensor = torch.tensor(skip_batch, device=device, dtype=torch.int32)
        dist.all_reduce(skip_batch_tensor, op=dist.ReduceOp.MAX)  # If any GPU wants to skip, all skip

        if skip_batch_tensor.item() > 0:
            # All GPUs skip this batch together
            if not diagnostics['loss_is_valid']:
                n_bad_tiles += len(diagnostics['bad_tiles'])
            if rank == 0 and logger and not diagnostics['loss_is_valid']:
                logger.warning(f"Batch {batch_idx}: Skipping batch due to NaN loss (synchronized across all GPUs)")
            batch_start_time = time.time()  # Reset timer for next batch
            continue

        # Scale loss for gradient accumulation
        scaled_loss = loss / accumulation_steps
        n_good_tiles += batch[0].shape[0] - len(diagnostics['bad_tiles'])

        # Backward pass timing
        backward_start = time.time()
        scaler.scale(scaled_loss).backward()
        backward_time += (time.time() - backward_start)

        # Accumulate unscaled loss for logging
        train_loss_total += loss.item()

        # Accumulate per-band losses
        for band_name, band_loss in per_band_losses.items():
            if band_name not in train_loss_per_band:
                train_loss_per_band[band_name] = 0.0
            train_loss_per_band[band_name] += band_loss.item()

        batch_count += 1
        accumulated_batch_count += 1
        current_batch_step += 1

        # Log per-batch metrics
        if rank == 0 and writer is not None:
            writer.add_scalar('Loss/train_batch', loss.item(), current_batch_step)
            for band_name, band_loss in per_band_losses.items():
                writer.add_scalar(f'Loss_PerBand/train_{band_name}_batch', band_loss.item(), current_batch_step)
            writer.add_scalar('Metrics/learning_rate_batch', optimizer.param_groups[0]['lr'], current_batch_step)

        # Optimizer step (every gradient_accumulation_steps)
        is_last_batch = (batch_idx == len(train_loader) - 1)
        if accumulated_batch_count == accumulation_steps or is_last_batch:
            optimizer_step_start = time.time()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Clip gradients
            grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=max_grad_norm
            )

            effective_grad_norm = min(grad_norm_before_clip.item(), max_grad_norm)

            # Log gradient norms
            if rank == 0 and writer is not None:
                writer.add_scalar('Gradients/norm_pre_clip', grad_norm_before_clip.item(), current_optimizer_step)
                writer.add_scalar('Gradients/norm_post_clip', effective_grad_norm, current_optimizer_step)

            # Update weights
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            optimizer_time += (time.time() - optimizer_step_start)

            # Log system metrics every 10 optimizer steps
            if current_optimizer_step % 10 == 0:
                log_system_metrics(writer, current_batch_step, device, rank)

            current_optimizer_step += 1
            accumulated_batch_count = 0

        # Reset timer for next batch
        batch_start_time = time.time()

    # Average training loss across GPUs (using SUM reduce, then divide)
    if batch_count > 0:
        train_loss_avg = train_loss_total / batch_count
    else:
        train_loss_avg = float('nan')

    # Gather loss from all GPUs
    world_size = dist.get_world_size()
    train_loss_tensor = torch.tensor([train_loss_avg], device=device)
    dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.AVG)
    train_loss_avg = train_loss_tensor.item()

    # Average per-band losses
    for band_name in train_loss_per_band:
        if batch_count > 0:
            train_loss_per_band[band_name] /= batch_count

    # Log epoch-level metrics and timing
    if rank == 0 and writer is not None:
        writer.add_scalar('Loss/train_epoch', train_loss_avg, epoch)
        for band_name in train_loss_per_band:
            writer.add_scalar(f'Loss_PerBand/train_{band_name}', train_loss_per_band[band_name], epoch)

        # Log timing breakdown
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

        # Log one final system snapshot at end of epoch
        log_system_metrics(writer, epoch * len(train_loader), device, rank)

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
        Dict with validation metrics (loss, MAE, R², per-band quantiles)
    """
    model.eval()

    val_loss_total = 0.0
    val_loss_per_band = {}
    batch_count = 0

    # Storage for metrics computation
    all_predictions = []
    all_targets = []
    all_errors = []
    band_errors = [[] for _ in range(config.n_bands)]

    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                predictions, targets, loss, per_band_losses, diagnostics = process_raster_batch(
                    model, batch, device,
                    use_naip=config.use_naip,
                    use_uavsar=config.use_uavsar,
                    is_training=False,
                    rank=rank,
                    logger=logger
                )

                # Synchronize skip decision across all GPUs (critical for DDP)
                skip_batch = int(not diagnostics['loss_is_valid'] or diagnostics['all_bad'])
                skip_batch_tensor = torch.tensor(skip_batch, device=device, dtype=torch.int32)
                dist.all_reduce(skip_batch_tensor, op=dist.ReduceOp.MAX)

                if skip_batch_tensor.item() > 0:
                    if rank == 0 and logger and not diagnostics['loss_is_valid']:
                        logger.warning(f"Validation batch: Skipping due to NaN loss (synchronized)")
                    continue

                val_loss_total += loss.item()

                # Accumulate per-band losses
                for band_name, band_loss in per_band_losses.items():
                    if band_name not in val_loss_per_band:
                        val_loss_per_band[band_name] = 0.0
                    val_loss_per_band[band_name] += band_loss.item()

                # Store predictions/targets for metrics
                all_predictions.append(predictions.cpu())
                all_targets.append(targets.cpu())

                batch_count += 1

    # Average validation loss
    if batch_count > 0:
        val_loss_avg = val_loss_total / batch_count
    else:
        val_loss_avg = float('nan')

    # Gather loss from all GPUs
    world_size = dist.get_world_size()
    val_loss_tensor = torch.tensor([val_loss_avg], device=device)
    dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.AVG)
    val_loss_avg = val_loss_tensor.item()

    # Average per-band losses
    for band_name in val_loss_per_band:
        if batch_count > 0:
            val_loss_per_band[band_name] /= batch_count

    # Compute additional metrics (MAE, R², quantiles)
    metrics = {'loss': val_loss_avg, 'per_band': val_loss_per_band}

    if len(all_predictions) > 0:
        preds_tensor = torch.cat(all_predictions, dim=0)  # [total_samples, n_bands, 5, 5]
        targs_tensor = torch.cat(all_targets, dim=0)

        # Overall metrics
        mae = (preds_tensor - targs_tensor).abs().mean().item()
        ss_res = ((preds_tensor - targs_tensor) ** 2).sum().item()
        ss_tot = ((targs_tensor - targs_tensor.mean()) ** 2).sum().item()
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics['mae'] = mae
        metrics['r2'] = r2

        # Per-band metrics
        for band_idx in range(config.n_bands):
            pred_band = preds_tensor[:, band_idx].flatten()
            targ_band = targs_tensor[:, band_idx].flatten()
            errors = (pred_band - targ_band).abs()

            band_mae = errors.mean().item()
            band_p10 = torch.quantile(errors, 0.1).item()
            band_p50 = torch.quantile(errors, 0.5).item()
            band_p90 = torch.quantile(errors, 0.9).item()

            metrics[f'band_{band_idx}_mae'] = band_mae
            metrics[f'band_{band_idx}_p10'] = band_p10
            metrics[f'band_{band_idx}_p50'] = band_p50
            metrics[f'band_{band_idx}_p90'] = band_p90

            # Log to TensorBoard
            if rank == 0 and writer is not None:
                writer.add_scalar(f'Val_Metrics/band_{band_idx}_mae', band_mae, epoch)
                writer.add_scalar(f'Val_Metrics/band_{band_idx}_p50', band_p50, epoch)

    # Log epoch-level metrics
    if rank == 0 and writer is not None:
        writer.add_scalar('Loss/val_epoch', val_loss_avg, epoch)
        for band_name in val_loss_per_band:
            writer.add_scalar(f'Loss_PerBand/val_{band_name}', val_loss_per_band[band_name], epoch)
        if 'mae' in metrics:
            writer.add_scalar('Val_Metrics/overall_mae', metrics['mae'], epoch)
        if 'r2' in metrics:
            writer.add_scalar('Val_Metrics/overall_r2', metrics['r2'], epoch)

    return metrics


def train_raster_worker(
    rank: int,
    world_size: int,
    config: MultimodalRasterConfig,
    train_shard_dir: str,
    train_shard_prefix: str,
    val_data_path: str,
    aug_shard_dir: Optional[str],
    aug_shard_prefix: Optional[str],
    output_dir: str,
    num_epochs: int = 100,
    batch_size: int = 15,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    save_every_n_epochs: int = 10,
    use_amp: bool = True,
    early_stopping_patience: int = 10,
    seed: int = 42,
    beta1: float = 0.9,
    beta2: float = 0.999,
    max_grad_norm: float = 10.0,
    warmup_steps_percentage: float = 0.05
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
        aug_shard_dir: Directory containing pre-sharded augmented data (optional)
        aug_shard_prefix: Prefix for augmented shard files (optional)
        output_dir: Directory to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size per GPU
        learning_rate: Base learning rate
        weight_decay: Weight decay for optimizer
        gradient_accumulation_steps: Steps to accumulate gradients
        save_every_n_epochs: Save checkpoint every N epochs
        use_amp: Use automatic mixed precision
        early_stopping_patience: Epochs without improvement before stopping
        seed: Random seed
    """
    # Setup distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    # Use environment variable if set, otherwise use default port
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
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
                   f"early_stopping_patience={early_stopping_patience}")

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
    print(f"[GPU {rank}] Loading training shard from {train_shard_path}")
    train_dataset = ShardedRasterDataset(
        shard_path=train_shard_path,
        k=config.k,
        use_naip=config.use_naip,
        use_uavsar=config.use_uavsar,
        target_band_indices=config.target_band_indices
    )

    # Load augmented data if provided
    if aug_shard_dir and aug_shard_prefix:
        aug_shard_path = os.path.join(aug_shard_dir, f"{aug_shard_prefix}_shard_{rank}.pt")
        if os.path.exists(aug_shard_path):
            print(f"[GPU {rank}] Loading augmented shard from {aug_shard_path}")
            aug_dataset = ShardedRasterDataset(
                shard_path=aug_shard_path,
                k=config.k,
                use_naip=config.use_naip,
                use_uavsar=config.use_uavsar,
                target_band_indices=config.target_band_indices
            )
            train_dataset.data = train_dataset.data + aug_dataset.data
            print(f"[GPU {rank}] Combined dataset size: {len(train_dataset)}")

    # Load validation data
    if rank == 0:
        print(f"[GPU {rank}] Loading validation data from {val_data_path}")
    val_dataset = ShardedRasterDataset(
        shard_path=val_data_path,
        k=config.k,
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

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=False)

    # Calculate warmup steps from percentage of total training steps
    total_batches = len(train_loader)
    total_training_steps = total_batches * num_epochs
    warmup_steps = int(warmup_steps_percentage * total_training_steps)

    if rank == 0:
        print(f"[GPU {rank}] Total training steps: {total_training_steps} ({total_batches} batches × {num_epochs} epochs)")
        print(f"[GPU {rank}] Warmup steps: {warmup_steps} ({warmup_steps_percentage*100:.1f}% of total)")

    # Create optimizer
    optimizer = AdamWScheduleFree(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
        warmup_steps=warmup_steps
    )
    print(f"[GPU {rank}] Using AdamWScheduleFree with lr={learning_rate}, weight decay={weight_decay}, betas=({beta1}, {beta2}), warmup_steps={warmup_steps}")

    # AMP scaler for mixed precision
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Training loop
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    training_history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

    for epoch in range(num_epochs):
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
            training_history['learning_rates'].append(current_lr)

            # Early stopping logic
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0
                checkpoint_path = Path(output_dir) / 'checkpoints' / 'best_model.pth'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'config': config
                }, checkpoint_path)
                msg = f"✓ Saved best model (val_loss: {val_metrics['loss']:.6f})"
                print(f"  → {msg}")
                if logger:
                    logger.info(msg)
            else:
                epochs_without_improvement += 1
                if logger:
                    logger.info(f"No improvement for {epochs_without_improvement} epochs")

            # Check early stopping
            if epochs_without_improvement >= early_stopping_patience:
                msg = f"Early stopping triggered at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)"
                print(f"  → {msg}")
                if logger:
                    logger.info(msg)
                # Break only on rank 0 is not enough - need to broadcast
                early_stop = True
            else:
                early_stop = False
        else:
            early_stop = False

        # Broadcast early stopping decision to all GPUs
        early_stop_tensor = torch.tensor(int(early_stop), device=device)
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
            'epoch': num_epochs - 1,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_metrics['loss'],
            'config': config
        }, final_checkpoint_path)
        if logger:
            logger.info(f"Final model saved to {final_checkpoint_path}")

        history_path = Path(output_dir) / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        if logger:
            logger.info(f"Training history saved to {history_path}")

        # Close TensorBoard writer
        if writer is not None:
            writer.close()

        completion_msg = (f"\n{'='*80}\nTraining complete!\n"
                         f"Best validation loss: {best_val_loss:.6f}\n"
                         f"Checkpoints saved to: {Path(output_dir) / 'checkpoints'}\n"
                         f"Logs saved to: {Path(output_dir) / 'logs'}\n{'='*80}")
        print(completion_msg)
        if logger:
            logger.info(f"Training complete. Best validation loss: {best_val_loss:.6f}")

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
    augmented_data_path: Optional[str] = None,
    num_epochs: int = 100,
    batch_size: int = 15,
    learning_rate: float = 5e-4,
    weight_decay: float = 1e-4,
    gradient_accumulation_steps: int = 1,
    save_every_n_epochs: int = 10,
    use_amp: bool = True,
    early_stopping_patience: int = 10,
    seed: int = 42,
    num_gpus: Optional[int] = None,
    beta1: float = 0.9,
    beta2: float = 0.999,
    max_grad_norm: float = 10.0,
    warmup_steps_percentage: float = 0.05
):
    """
    Main entry point for training raster prediction model.

    Args:
        config: Model configuration
        train_data_path: Path to training data
        val_data_path: Path to validation data
        output_dir: Output directory for checkpoints
        augmented_data_path: Path to augmented data (optional)
        num_epochs: Number of epochs
        batch_size: Batch size per GPU
        learning_rate: Learning rate
        weight_decay: Weight decay
        gradient_accumulation_steps: Gradient accumulation steps
        save_every_n_epochs: Save frequency
        use_amp: Use mixed precision
        early_stopping_patience: Epochs without improvement before stopping
        seed: Random seed
        num_gpus: Number of GPUs (None = all available)
        beta1: AdamW beta1 parameter
        beta2: AdamW beta2 parameter
        max_grad_norm: Gradient clipping threshold
        warmup_steps_percentage: Percentage of total steps for warmup
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
    aug_shard_prefix = f"{cache_key}_aug"

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

        # If augmented data provided, create shards for it too
        aug_shard_paths = [None] * world_size
        if augmented_data_path and os.path.exists(augmented_data_path):
            print(f"\nLoading augmented data from {augmented_data_path}")
            aug_data = torch.load(augmented_data_path, weights_only=False)
            print(f"  ✓ Loaded {len(aug_data)} augmented samples")

            # Create augmented shards
            aug_shard_paths = create_raster_shards(
                aug_data, world_size, str(shard_cache_dir), aug_shard_prefix
            )
            del aug_data
        print(f"{'='*80}\n")
    else:
        print(f"\n{'='*80}")
        print("Using cached shards (loading from disk)...")
        print(f"{'='*80}\n")
        # Check for augmented shards
        aug_shard_paths = [str(shard_cache_dir / f"{aug_shard_prefix}_shard_{i}.pt")
                          for i in range(world_size)]
        aug_shard_paths = aug_shard_paths if all(os.path.exists(p) for p in aug_shard_paths) else [None] * world_size

    # Spawn training workers
    torch.multiprocessing.spawn(
        train_raster_worker,
        args=(world_size, config, str(shard_cache_dir), train_shard_prefix, val_data_path,
              str(shard_cache_dir) if aug_shard_paths[0] is not None else None,
              aug_shard_prefix if aug_shard_paths[0] is not None else None,
              output_dir, num_epochs, batch_size,
              learning_rate, weight_decay, gradient_accumulation_steps,
              save_every_n_epochs, use_amp, early_stopping_patience, seed,
              beta1, beta2, max_grad_norm, warmup_steps_percentage),
        nprocs=world_size,
        join=True
    )
