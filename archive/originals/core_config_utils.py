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


def get_config_dict(**kwargs: Any) -> Dict[str, Any]:
    """
    Build a standardized config dict for JSON output.

    Accepts arbitrary keyword arguments for experiment parameters.
    Automatically adds timestamp and git hash.

    Args:
        **kwargs: Experiment parameters to log (model, dataset, seed, etc.)

    Returns:
        Dict with all parameters plus timestamp and git_hash.

    Example:
        get_config_dict(
            model="meta-llama/Llama-3.1-8B-Instruct",
            dataset="TriviaMC_difficulty_filtered",
            seed=42,
            probe_alpha=1000.0,
            probe_pca_components=100,
            train_split=0.8,
            mean_diff_quantile=0.25,
            num_questions=500,
        )
    """
    config = dict(kwargs)
    config["timestamp"] = datetime.datetime.now().isoformat()
    config["git_hash"] = _get_git_hash()
    return config
