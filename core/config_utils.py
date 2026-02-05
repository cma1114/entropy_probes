"""
Configuration utilities for reproducible experiment logging.

Provides a standardized way to build config dicts for JSON output files.
Each script passes its local constants to get_config_dict(), which adds
timestamp and git hash for reproducibility.

Usage:
    from core.config_utils import get_config_dict

    results = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            seed=SEED,
            metric=METRIC,
            # ... any other script-specific parameters
        ),
        "results": { ... },
    }
"""

import datetime
import subprocess
from typing import Any, Dict


def _get_git_hash() -> str:
    """Get short git hash of current commit, or 'unknown' if not in a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return "unknown"


def quantization_label(load_in_4bit: bool = False, load_in_8bit: bool = False) -> str:
    """Return a human-readable quantization label for logging."""
    if load_in_4bit:
        return "4bit"
    elif load_in_8bit:
        return "8bit"
    return "fp16"


def get_config_dict(**kwargs: Any) -> Dict[str, Any]:
    """
    Build a standardized config dict for JSON output.

    Accepts arbitrary keyword arguments for experiment parameters.
    Automatically adds timestamp and git hash.

    If ``load_in_4bit`` or ``load_in_8bit`` are passed as kwargs, a derived
    ``quantization`` field ("4bit", "8bit", or "fp16") is added automatically.

    Args:
        **kwargs: Experiment parameters to log (model, dataset, seed, etc.)

    Returns:
        Dict with all parameters plus timestamp, git_hash, and quantization.

    Example:
        get_config_dict(
            model="meta-llama/Llama-3.1-8B-Instruct",
            dataset="TriviaMC_difficulty_filtered",
            seed=42,
            load_in_4bit=True,
            load_in_8bit=False,
            probe_alpha=1000.0,
            probe_pca_components=100,
            train_split=0.8,
            mean_diff_quantile=0.25,
            num_questions=500,
        )
    """
    config = dict(kwargs)
    # Derive quantization label from boolean flags if present
    if "load_in_4bit" in config or "load_in_8bit" in config:
        config["quantization"] = quantization_label(
            config.get("load_in_4bit", False),
            config.get("load_in_8bit", False),
        )
    config["timestamp"] = datetime.datetime.now().isoformat()
    config["git_hash"] = _get_git_hash()
    return config
