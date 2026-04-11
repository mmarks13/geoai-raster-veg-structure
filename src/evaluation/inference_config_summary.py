#!/usr/bin/env python3
"""
Generate inference configuration summary for reproducibility.

Creates a JSON file documenting the model configuration, training info,
inference arguments, and band config used for a particular inference run.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)


def serialize_config(obj: Any) -> Any:
    """
    Recursively serialize config objects to JSON-compatible types.

    Handles dataclasses, tensors, and nested structures.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj):
        return {k: serialize_config(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): serialize_config(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [serialize_config(v) for v in obj]
    # Handle torch tensors
    if hasattr(obj, 'item'):
        return obj.item()
    if hasattr(obj, 'tolist'):
        return obj.tolist()
    # Fallback: convert to string
    return str(obj)


def extract_training_info(checkpoint: dict) -> Dict[str, Any]:
    """
    Extract training information from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict

    Returns:
        Dict with training info (epoch, loss, optimizer, etc.)
    """
    training_info = {}

    # Standard checkpoint keys
    if 'epoch' in checkpoint:
        training_info['epoch'] = checkpoint['epoch']

    if 'best_val_loss' in checkpoint:
        training_info['best_val_loss'] = float(checkpoint['best_val_loss'])

    if 'val_loss' in checkpoint:
        training_info['val_loss'] = float(checkpoint['val_loss'])

    if 'train_loss' in checkpoint:
        training_info['train_loss'] = float(checkpoint['train_loss'])

    # Optimizer info (just note if present, don't include full state)
    if 'optimizer_state_dict' in checkpoint:
        training_info['optimizer_saved'] = True
    else:
        training_info['optimizer_saved'] = False

    # Scheduler info
    if 'scheduler_state_dict' in checkpoint:
        training_info['scheduler_saved'] = True
    else:
        training_info['scheduler_saved'] = False

    return training_info


def extract_model_config(checkpoint: dict) -> Dict[str, Any]:
    """
    Extract model configuration from checkpoint.

    Args:
        checkpoint: Loaded checkpoint dict

    Returns:
        Dict with model configuration
    """
    config = checkpoint.get('config', {})
    return serialize_config(config)


def generate_config_summary(
    checkpoint_path: str,
    checkpoint_data: dict,
    cli_args: Dict[str, Any],
    band_config_path: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """
    Generate a complete configuration summary for the inference run.

    Args:
        checkpoint_path: Path to the model checkpoint
        checkpoint_data: Loaded checkpoint dict
        cli_args: Command-line arguments as dict
        band_config_path: Optional path to band config JSON
        output_path: Optional path to write JSON output

    Returns:
        Dict containing the full configuration summary
    """
    summary = {
        'checkpoint_path': str(checkpoint_path),
        'model_config': extract_model_config(checkpoint_data),
        'training_info': extract_training_info(checkpoint_data),
        'inference_args': serialize_config(cli_args),
    }

    # Load band config if path provided
    if band_config_path and Path(band_config_path).exists():
        try:
            with open(band_config_path) as f:
                band_config = json.load(f)
            summary['band_config'] = band_config
            summary['band_config_path'] = str(band_config_path)
        except Exception as e:
            logger.warning(f"Could not load band config: {e}")
            summary['band_config'] = None
            summary['band_config_path'] = str(band_config_path)

    # Write to file if output_path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Saved config summary to {output_path}")

    return summary
