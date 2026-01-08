#!/usr/bin/env python3
"""
Raster model inference for forest plot tiles.

This script loads a trained raster prediction model and runs inference
on preprocessed forest plot tiles, producing predictions that can be
compared to field measurements.

Usage:
    python src/evaluation/raster_inference.py \
        --checkpoint data/output/raster_model_optuna_trial000_20251128_182540/checkpoints/best_model.pth \
        --input data/processed/forest_plot_data/inference_ready/precomputed_forest_plot_tiles_32bit.pt \
        --output data/processed/forest_plot_data/predictions \
        --batch-size 32

Output:
    - forest_plot_predictions.pt: Dict with tile_id -> prediction tensor mapping
    - forest_plot_predictions.csv: CSV with denormalized predictions per tile
    - prediction_raster_<site>.tif: Optional GeoTIFF mosaic per site
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

# Add project root to path for src imports (avoids PYTHONPATH requirement)
# Goes up 3 levels: evaluation -> src -> project_root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Import helper from training module
from src.training.raster_training import find_free_port

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def load_checkpoint_and_config(checkpoint_path: str) -> Tuple[dict, dict]:
    """
    Load model checkpoint and extract configuration.
    
    Args:
        checkpoint_path: Path to best_model.pth or other checkpoint
        
    Returns:
        Tuple of (state_dict, config_dict)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Extract config if available
    config_dict = checkpoint.get('config', {})
    
    # Log training info if available
    if 'epoch' in checkpoint:
        logger.info(f"Checkpoint from epoch {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        logger.info(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    
    return state_dict, config_dict


def build_model_from_checkpoint(checkpoint_path: str, device: torch.device) -> nn.Module:
    """
    Build model architecture and load weights from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model onto
        
    Returns:
        Loaded model in eval mode
    """
    from src.models.multimodal_raster_model import MultimodalRasterConfig, MultimodalRasterPredictor

    state_dict, config_dict = load_checkpoint_and_config(checkpoint_path)

    # Build config from checkpoint or use defaults
    if config_dict:
        # Check if config is already a MultimodalRasterConfig object
        if isinstance(config_dict, MultimodalRasterConfig):
            logger.info("Using config object directly from checkpoint")
            config = config_dict
        else:
            # Reconstruct config from dict
            logger.info("Reconstructing config from checkpoint dict")
            config = MultimodalRasterConfig(
            k=config_dict.get('k', 15),
            feature_dim=config_dict.get('feature_dim', 512),
            pos_mlp_hdn=config_dict.get('pos_mlp_hdn', 16),
            pt_attn_dropout=config_dict.get('pt_attn_dropout', 0.1),
            extractor_lcl_heads=config_dict.get('extractor_lcl_heads', 4),
            extractor_glbl_heads=config_dict.get('extractor_glbl_heads', 4),
            attr_dim=config_dict.get('attr_dim', 3),
            use_naip=config_dict.get('use_naip', True),
            use_uavsar=config_dict.get('use_uavsar', True),
            img_embed_dim=config_dict.get('img_embed_dim', 256),
            img_num_patches=config_dict.get('img_num_patches', 16),
            fusion_type=config_dict.get('fusion_type', 'cross_attention'),
            max_dist_ratio=config_dict.get('max_dist_ratio', 5.0),
            fusion_num_heads=config_dict.get('fusion_num_heads', 4),
            fusion_dropout=config_dict.get('fusion_dropout', 0.1),
            position_encoding_dim=config_dict.get('position_encoding_dim', 24),
            naip_dropout=config_dict.get('naip_dropout', 0.1),
            uavsar_dropout=config_dict.get('uavsar_dropout', 0.1),
            temporal_encoder=config_dict.get('temporal_encoder', 'gru'),
            n_bands=config_dict.get('n_bands', 2),
            target_band_indices=config_dict.get('target_band_indices', [11, 7]),
            grid_size=config_dict.get('grid_size', 5),
            tile_extent=config_dict.get('tile_extent', 10.0),
            raster_num_heads=config_dict.get('raster_num_heads', 8),
            # RASTER MODEL: Support both old 'raster_radius' and new 'raster_distance_sigma'
            # Converts old radius to sigma if found, else uses sigma directly, else default 2.0
            raster_distance_sigma=config_dict.get('raster_distance_sigma', 
                config_dict.get('raster_radius', 2.0)),  # Backwards compat
            raster_hidden_dim=config_dict.get('raster_hidden_dim', 512),
            raster_decoder_layers=config_dict.get('raster_decoder_layers', 4),
            raster_dropout=config_dict.get('raster_dropout', 0.1),
            num_pre_agg_blocks=config_dict.get('num_pre_agg_blocks', 1),
            pre_agg_lcl_heads=config_dict.get('pre_agg_lcl_heads', 4),
            pre_agg_glbl_heads=config_dict.get('pre_agg_glbl_heads', 4),
                pre_agg_dropout=config_dict.get('pre_agg_dropout', 0.1),
                pre_agg_k_neighbors=config_dict.get('pre_agg_k_neighbors', 15),
            )
    else:
        # Use defaults matching run_raster_model.py
        logger.warning("No config found in checkpoint, using defaults")
        config = MultimodalRasterConfig(
            k=15,
            feature_dim=512,
            use_naip=True,
            use_uavsar=True,
            img_embed_dim=256,
            n_bands=2,
            target_band_indices=[11, 7],
            raster_hidden_dim=512,
            raster_decoder_layers=4,
            num_pre_agg_blocks=1,
        )
    
    logger.info(f"Model config: use_naip={config.use_naip}, use_uavsar={config.use_uavsar}")
    logger.info(f"Target bands: {config.target_band_indices} ({config.n_bands} bands)")
    
    # Build model
    model = MultimodalRasterPredictor(config)
    
    # Load weights
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully ({sum(p.numel() for p in model.parameters()):,} parameters)")
    
    return model, config


def collate_inference_batch(batch: List[dict]) -> dict:
    """
    Collate tiles into a batch for inference.
    
    Handles variable point counts using batch indexing (PyTorch Geometric style).
    """
    # Stack/concatenate tensors
    dep_points_list = []
    dep_attr_list = []
    edge_index_list = []
    batch_indices_list = []
    norm_params_list = []
    naip_list = []
    uavsar_list = []
    bbox_list = []
    tile_ids = []
    site_names = []
    
    current_point_offset = 0
    
    for i, tile in enumerate(batch):
        # Points and attributes
        dep_points = tile['dep_points_norm']  # Already z-score normalized
        dep_attr = tile.get('dep_points_attr_norm', tile.get('dep_points_attr'))
        n_points = dep_points.shape[0]
        
        dep_points_list.append(dep_points)
        dep_attr_list.append(dep_attr)
        
        # Edge indices (offset by cumulative point count)
        edge_index = tile['knn_edge_indices'][15]  # k=15
        edge_index_list.append(edge_index + current_point_offset)
        
        # Batch indices
        batch_indices_list.append(torch.full((n_points,), i, dtype=torch.long))
        
        # Normalization params
        norm_params_list.append(tile['norm_params'])
        
        # Imagery (can be None)
        naip_list.append(tile.get('naip'))
        uavsar_list.append(tile.get('uavsar'))
        
        # Metadata
        bbox_list.append(tile['bbox'])
        tile_ids.append(tile.get('tile_id', f'tile_{i}'))
        site_names.append(tile.get('site_name', 'unknown'))
        
        current_point_offset += n_points
    
    return {
        'dep_points': torch.cat(dep_points_list, dim=0),
        'dep_attr': torch.cat(dep_attr_list, dim=0),
        'edge_index': torch.cat(edge_index_list, dim=1),
        'batch_indices': torch.cat(batch_indices_list, dim=0),
        'norm_params': norm_params_list,
        'naip': naip_list,
        'uavsar': uavsar_list,
        'bbox': torch.stack(bbox_list),
        'tile_ids': tile_ids,
        'site_names': site_names,
    }


def denormalize_predictions(
    predictions: torch.Tensor,
    fuel_stats: dict,
    target_band_indices: List[int]
) -> torch.Tensor:
    """
    Denormalize z-score predictions back to original units.

    If TFL was log-transformed during preprocessing (use_log_tfl=True in stats),
    applies exp(x) - 1 to convert back to original units.

    Args:
        predictions: [batch_size, n_bands, 5, 5] z-score normalized
        fuel_stats: Dict with per-band stats (band_1_mean, band_1_std, etc.)
                   May include 'use_log_tfl' and 'tfl_band_index' flags
        target_band_indices: Which bands were predicted (0-indexed)

    Returns:
        Denormalized predictions in original units
    """
    device = predictions.device

    # Get mean/std for target bands
    # Note: fuel_stats uses 0-indexed band names (band_0, band_1, ..., band_23)
    # For tile tensor index i, use band_{i}_mean
    means = torch.tensor([fuel_stats[f'band_{i}_mean'] for i in target_band_indices],
                        device=device, dtype=predictions.dtype)
    stds = torch.tensor([fuel_stats[f'band_{i}_std'] for i in target_band_indices],
                       device=device, dtype=predictions.dtype)

    # Reshape for broadcasting [1, n_bands, 1, 1]
    means = means.view(1, -1, 1, 1)
    stds = stds.view(1, -1, 1, 1)

    # Z-score denormalize: pred_log = pred_norm * std + mean
    denorm = predictions * stds + means

    # Apply inverse log transform to TFL if it was log-transformed during preprocessing
    use_log_tfl = fuel_stats.get('use_log_tfl', False)
    tfl_band_index = fuel_stats.get('tfl_band_index', 15)  # TFL = band index 15 (per band_config.py)

    if use_log_tfl and tfl_band_index in target_band_indices:
        # Find position of TFL in the prediction tensor
        tfl_pred_idx = target_band_indices.index(tfl_band_index)

        # Apply inverse: exp(pred_log) - 1
        denorm[:, tfl_pred_idx, :, :] = torch.exp(denorm[:, tfl_pred_idx, :, :]) - 1

        # Clamp to non-negative (TFL cannot be negative)
        denorm[:, tfl_pred_idx, :, :] = torch.clamp(denorm[:, tfl_pred_idx, :, :], min=0.0)

    return denorm


@torch.no_grad()
def run_inference_on_dataloader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    fuel_stats: Optional[dict] = None,
    target_band_indices: Optional[List[int]] = None
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Run inference on a pre-configured dataloader.

    This function is extracted from run_inference() to allow code reuse
    between single-GPU and multi-GPU inference paths.

    Args:
        model: Trained model in eval mode
        dataloader: Pre-configured DataLoader (with appropriate sampler)
        device: Device to run on
        fuel_stats: Optional dict with normalization stats for denormalization
        target_band_indices: Band indices for labeling

    Returns:
        Tuple of:
        - predictions_dict: {tile_id: prediction_array}
        - predictions_df: DataFrame with tile-level summary statistics
    """
    model.eval()

    predictions_dict = {}
    results_list = []

    logger.info(f"Running inference on {len(dataloader)} batches")

    for batch in tqdm(dataloader, desc="Inference"):
        # Move to device
        dep_points = batch['dep_points'].to(device)
        dep_attr = batch['dep_attr'].to(device)
        edge_index = batch['edge_index'].to(device)
        batch_indices = batch['batch_indices'].to(device)
        norm_params = batch['norm_params']

        # Handle imagery (move to device if present)
        naip = batch['naip']
        uavsar = batch['uavsar']

        for i, naip_dict in enumerate(naip):
            if naip_dict is not None and 'images' in naip_dict:
                naip[i] = {
                    'images': naip_dict['images'].to(device),
                    'img_bbox': naip_dict.get('img_bbox'),
                    'relative_dates': naip_dict.get('relative_dates'),
                }

        for i, uavsar_dict in enumerate(uavsar):
            if uavsar_dict is not None and 'images' in uavsar_dict:
                uavsar[i] = {
                    'images': uavsar_dict['images'].to(device),
                    'img_bbox': uavsar_dict.get('img_bbox'),
                    'attention_mask': uavsar_dict.get('attention_mask'),
                    'relative_dates': uavsar_dict.get('relative_dates'),
                }

        # Check for NaN in UAVSAR and enable debug logging if found
        debug_logging = False
        for i, uavsar_dict in enumerate(uavsar):
            if uavsar_dict is not None and 'images' in uavsar_dict:
                if torch.isnan(uavsar_dict['images']).any():
                    logger.info(f"[DEBUG] Batch contains UAVSAR NaN (tile {i}), enabling debug logging")
                    debug_logging = True
                    break

        # Forward pass
        predictions = model(
            dep_points=dep_points,
            edge_index=edge_index,
            batch_indices=batch_indices,
            norm_params=norm_params,
            dep_attr=dep_attr,
            naip=naip,
            uavsar=uavsar,
            bbox=batch['bbox'].to(device),
            debug_logging=debug_logging
        )

        # predictions shape: [batch_size, n_bands, 5, 5]
        predictions = predictions.cpu()

        # Denormalize if stats provided
        if fuel_stats is not None:
            predictions_denorm = denormalize_predictions(
                predictions, fuel_stats, target_band_indices or [11, 7]
            )
        else:
            predictions_denorm = predictions

        # Store results
        for i, tile_id in enumerate(batch['tile_ids']):
            pred = predictions_denorm[i].numpy()  # [n_bands, 5, 5]
            predictions_dict[tile_id] = pred

            # Compute summary stats for each band
            result = {
                'tile_id': tile_id,
                'site_name': batch['site_names'][i],
                'bbox_xmin': batch['bbox'][i, 0].item(),
                'bbox_ymin': batch['bbox'][i, 1].item(),
                'bbox_xmax': batch['bbox'][i, 2].item(),
                'bbox_ymax': batch['bbox'][i, 3].item(),
            }

            # Add per-band statistics (use generic band_0, band_1, etc. names)
            for j in range(pred.shape[0]):
                band_name = f'band_{j}'
                band_vals = pred[j]
                result[f'{band_name}_mean'] = np.nanmean(band_vals)
                result[f'{band_name}_std'] = np.nanstd(band_vals)
                result[f'{band_name}_min'] = np.nanmin(band_vals)
                result[f'{band_name}_max'] = np.nanmax(band_vals)
                result[f'{band_name}_center'] = band_vals[2, 2]  # Center pixel value

            results_list.append(result)

    predictions_df = pd.DataFrame(results_list)

    return predictions_dict, predictions_df


@torch.no_grad()
def run_inference(
    model: nn.Module,
    tiles: List[dict],
    batch_size: int,
    device: torch.device,
    fuel_stats: Optional[dict] = None,
    target_band_indices: Optional[List[int]] = None
) -> Tuple[Dict[str, np.ndarray], pd.DataFrame]:
    """
    Run inference on all tiles (single-GPU version).

    Args:
        model: Trained model in eval mode
        tiles: List of preprocessed tile dicts
        batch_size: Batch size for inference
        device: Device to run on
        fuel_stats: Optional dict with normalization stats for denormalization
        target_band_indices: Band indices for labeling

    Returns:
        Tuple of:
        - predictions_dict: {tile_id: prediction_array}
        - predictions_df: DataFrame with tile-level summary statistics
    """
    # Create dataloader
    dataloader = DataLoader(
        tiles,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_inference_batch,
        num_workers=0  # Avoid multiprocessing issues
    )

    logger.info(f"Running inference on {len(tiles)} tiles in {len(dataloader)} batches")

    # Use shared inference loop
    return run_inference_on_dataloader(
        model, dataloader, device, fuel_stats, target_band_indices
    )


def inference_worker(
    rank: int,
    world_size: int,
    checkpoint_path: str,
    input_path: str,
    output_dir: str,
    fuel_stats_path: str,
    batch_size: int
):
    """
    DDP worker for distributed multi-GPU inference.

    Each worker:
    1. Initializes its own DDP process group
    2. Loads the model checkpoint onto its GPU
    3. Uses DistributedSampler to get its shard of data
    4. Runs inference on assigned tiles
    5. Saves results to rank-specific file

    Pattern from: src/training/raster_training.py:1009-1080

    Args:
        rank: Process rank (GPU ID)
        world_size: Total number of processes (GPUs)
        checkpoint_path: Path to model checkpoint
        input_path: Path to preprocessed tiles (.pt file)
        output_dir: Directory to save rank-specific predictions
        fuel_stats_path: Path to fuel metrics normalization stats
        batch_size: Batch size per GPU
    """
    # Initialize DDP (pattern from raster_training.py:1057-1059)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    # Setup logging for rank 0 only
    if rank == 0:
        logger.info(f"Starting distributed inference with {world_size} GPUs")

    # Load model (use existing build_model_from_checkpoint)
    if rank == 0:
        logger.info(f"Loading model checkpoint: {checkpoint_path}")
    model, config = build_model_from_checkpoint(checkpoint_path, device)

    # Wrap with DDP
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    model.eval()

    # Load full dataset (NOT pre-sharded - DistributedSampler handles distribution)
    if rank == 0:
        logger.info(f"Loading tiles from {input_path}")
    tiles = torch.load(input_path, weights_only=False)
    if rank == 0:
        logger.info(f"Loaded {len(tiles)} tiles total")

    # Create DistributedSampler (pattern from raster_training.py:1127-1138)
    sampler = DistributedSampler(
        tiles,
        num_replicas=world_size,
        rank=rank,
        shuffle=False  # Keep deterministic order for inference
    )

    # Create DataLoader
    dataloader = DataLoader(
        tiles,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=collate_inference_batch,
        num_workers=0,
        pin_memory=True
    )

    if rank == 0:
        logger.info(f"Each GPU will process ~{len(tiles) // world_size} tiles in {len(dataloader)} batches")

    # Load fuel stats for denormalization
    fuel_stats = None
    if Path(fuel_stats_path).exists():
        with open(fuel_stats_path) as f:
            fuel_stats = json.load(f)
        if rank == 0:
            logger.info(f"Loaded fuel stats from {fuel_stats_path}")
    else:
        if rank == 0:
            logger.warning(f"Fuel stats not found, predictions will be normalized")

    # Run inference (use extracted run_inference_on_dataloader)
    predictions_dict, predictions_df = run_inference_on_dataloader(
        model, dataloader, device, fuel_stats, config.target_band_indices
    )

    # Save rank-specific results
    rank_predictions_path = Path(output_dir) / f'predictions_rank_{rank}.pt'
    torch.save(predictions_dict, rank_predictions_path)

    rank_csv_path = Path(output_dir) / f'predictions_rank_{rank}.csv'
    predictions_df.to_csv(rank_csv_path, index=False)

    if rank == 0:
        logger.info(f"Rank {rank} saved {len(predictions_dict)} predictions")

    # Cleanup (pattern from raster_training.py:1468-1479)
    if dist.is_initialized():
        dist.barrier()

    try:
        dist.destroy_process_group()
    except Exception as e:
        # Graceful timeout handling (NCCL cleanup may timeout)
        if rank == 0:
            logger.warning(f"DDP cleanup timeout (inference completed): {str(e)[:100]}")


def run_multi_gpu_inference(args):
    """
    Main entry point for multi-GPU inference using DDP.

    Spawns worker processes on each GPU, then aggregates results.

    Pattern from: src/training/raster_training.py:1531-1599

    Args:
        args: Parsed command-line arguments
    """
    # Determine world size
    if args.num_gpus is None:
        world_size = torch.cuda.device_count()
    else:
        world_size = min(args.num_gpus, torch.cuda.device_count())

    if world_size == 0:
        raise RuntimeError("No GPUs available for multi-GPU inference")

    logger.info(f"Running multi-GPU inference with {world_size} GPUs")

    # Setup distributed environment
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        free_port = find_free_port()
        os.environ['MASTER_PORT'] = str(free_port)
        logger.info(f"Using port {free_port} for distributed communication")

    # Set NCCL environment variables (pattern from raster_training.py)
    os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
    os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Spawn workers
    logger.info("Spawning worker processes...")
    torch.multiprocessing.spawn(
        inference_worker,
        args=(
            world_size,
            args.checkpoint,
            args.input,
            str(output_dir),
            args.fuel_stats,
            args.batch_size
        ),
        nprocs=world_size,
        join=True
    )

    # Aggregate results from all ranks
    logger.info("Aggregating results from all GPUs...")
    predictions_dict = {}
    predictions_dfs = []

    for rank in range(world_size):
        rank_pred_path = output_dir / f'predictions_rank_{rank}.pt'
        rank_csv_path = output_dir / f'predictions_rank_{rank}.csv'

        if not rank_pred_path.exists():
            logger.warning(f"Rank {rank} predictions not found at {rank_pred_path}")
            continue

        # Load and merge predictions
        rank_preds = torch.load(rank_pred_path, weights_only=False)
        predictions_dict.update(rank_preds)
        logger.info(f"Loaded {len(rank_preds)} predictions from rank {rank}")

        # Load and concatenate DataFrames
        rank_df = pd.read_csv(rank_csv_path)
        predictions_dfs.append(rank_df)

        # Clean up intermediate files
        rank_pred_path.unlink()
        rank_csv_path.unlink()

    # Combine all DataFrames
    predictions_df = pd.concat(predictions_dfs, ignore_index=True)
    logger.info(f"Total predictions: {len(predictions_dict)} tiles")

    # Save final aggregated results (existing pattern)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    final_path = output_dir / f'forest_plot_predictions_{timestamp}.pt'
    torch.save(predictions_dict, final_path)
    logger.info(f"Saved aggregated predictions to {final_path}")

    csv_path = output_dir / f'forest_plot_predictions_{timestamp}.csv'
    predictions_df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")

    # Print summary statistics (same as single-GPU version)
    logger.info("\n" + "=" * 60)
    logger.info("PREDICTION SUMMARY")
    logger.info("=" * 60)

    for site in predictions_df['site_name'].unique():
        site_df = predictions_df[predictions_df['site_name'] == site]
        logger.info(f"\n{site} ({len(site_df)} tiles):")

        # Log statistics for each band
        for j in range(len([col for col in site_df.columns if col.endswith('_mean')])):
            band_name = f'band_{j}'
            if f'{band_name}_mean' in site_df.columns:
                band_vals = site_df[f'{band_name}_mean']
                logger.info(f"  {band_name}: mean={band_vals.mean():.3f}, std={band_vals.std():.3f}, "
                           f"range=[{band_vals.min():.3f}, {band_vals.max():.3f}]")

    logger.info("\n" + "=" * 60)
    logger.info("Multi-GPU inference complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Run raster model inference on forest plot tiles"
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint (best_model.pth)'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to preprocessed tiles (.pt file)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for predictions'
    )
    parser.add_argument(
        '--fuel-stats',
        type=str,
        default='data/processed/model_data_veg_structure/target_raster_normalization_stats_train.json',
        help='Path to fuel metrics normalization stats (for denormalization)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to run inference on (single-GPU mode only)'
    )
    parser.add_argument(
        '--multi-gpu',
        action='store_true',
        help='Enable multi-GPU inference using DDP (default: single GPU)'
    )
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=None,
        help='Number of GPUs to use for multi-GPU inference (default: all available)'
    )

    args = parser.parse_args()

    # Dispatch to appropriate inference mode
    if args.multi_gpu:
        # Multi-GPU distributed inference
        run_multi_gpu_inference(args)
    else:
        # Single-GPU inference (original path)
        # Setup
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        device = torch.device(args.device)

        logger.info(f"Device: {device}")

        # Load model
        model, config = build_model_from_checkpoint(args.checkpoint, device)

        # Load tiles
        logger.info(f"Loading tiles from {args.input}")
        tiles = torch.load(args.input, weights_only=False)
        logger.info(f"Loaded {len(tiles)} tiles")

        # Load fuel stats for denormalization
        fuel_stats = None
        if Path(args.fuel_stats).exists():
            with open(args.fuel_stats) as f:
                fuel_stats = json.load(f)
            logger.info(f"Loaded fuel stats from {args.fuel_stats}")

            # Log if TFL log-transform will be inverted
            if fuel_stats.get('use_log_tfl', False):
                tfl_idx = fuel_stats.get('tfl_band_index', 7)
                logger.info(f"TFL log-transform detected: will apply exp(x)-1 to band index {tfl_idx}")
        else:
            logger.warning(f"Fuel stats not found at {args.fuel_stats}, predictions will be normalized")

        # Run inference
        predictions_dict, predictions_df = run_inference(
            model=model,
            tiles=tiles,
            batch_size=args.batch_size,
            device=device,
            fuel_stats=fuel_stats,
            target_band_indices=config.target_band_indices
        )

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save predictions dict
        predictions_path = output_dir / f'forest_plot_predictions_{timestamp}.pt'
        torch.save(predictions_dict, predictions_path)
        logger.info(f"Saved predictions to {predictions_path}")

        # Save CSV summary
        csv_path = output_dir / f'forest_plot_predictions_{timestamp}.csv'
        predictions_df.to_csv(csv_path, index=False)
        logger.info(f"Saved CSV summary to {csv_path}")

        # Print summary statistics
        logger.info("\n" + "=" * 60)
        logger.info("PREDICTION SUMMARY")
        logger.info("=" * 60)

        for site in predictions_df['site_name'].unique():
            site_df = predictions_df[predictions_df['site_name'] == site]
            logger.info(f"\n{site} ({len(site_df)} tiles):")

            # Log statistics for each band
            for j in range(len([col for col in site_df.columns if col.endswith('_mean')])):
                band_name = f'band_{j}'
                if f'{band_name}_mean' in site_df.columns:
                    band_vals = site_df[f'{band_name}_mean']
                    logger.info(f"  {band_name}: mean={band_vals.mean():.3f}, std={band_vals.std():.3f}, "
                               f"range=[{band_vals.min():.3f}, {band_vals.max():.3f}]")

        logger.info("\n" + "=" * 60)
        logger.info("Inference complete!")


if __name__ == '__main__':
    main()
