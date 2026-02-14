"""
Configuration utilities for reproducible experiment logging and output path management.

Provides:
1. Standardized config dicts for JSON output files (get_config_dict)
2. Output path routing to appropriate subdirectories (get_output_path, find_output_file)

Output Directory Structure:
    outputs/
        {model_dir}/             <- Model-specific outputs
            results/             <- Human-readable final outputs (summary JSONs, plots)
            working/             <- Machine data (activations, directions, checkpoints, logs)
        results/                 <- Legacy (no model_dir specified)
        working/                 <- Legacy (no model_dir specified)

Key distinction: results/ is for outputs intended for humans to read/view.
working/ is for machine data (checkpoints, cached computations, intermediate files).

Usage:
    from core.config_utils import get_config_dict, get_output_path, find_output_file
    from core.model_utils import get_model_dir_name

    # Get model directory name
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)

    # For writing files (model prefix no longer needed in filename):
    path = get_output_path(f"{DATASET}_results.json", model_dir=model_dir)
    # -> outputs/Llama-3.1-8B-Instruct/results/TriviaMC_results.json

    # For reading files (with migration fallback):
    path = find_output_file(f"{DATASET}_mc_results.json", model_dir=model_dir)

    # For config dicts:
    results = {
        "config": get_config_dict(model=MODEL, dataset=DATASET, ...),
        "results": { ... },
    }
"""

import datetime
import subprocess
from pathlib import Path
from typing import Any, Dict, List


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


# =============================================================================
# OUTPUT PATH MANAGEMENT
# =============================================================================

# Directory structure:
#   outputs/results/  <- Human-readable outputs (final results, plots)
#   outputs/working/  <- Machine data (activations, directions, checkpoints, logs)
OUTPUT_BASE_DIR = Path(__file__).parent.parent / "outputs"
RESULTS_DIR = OUTPUT_BASE_DIR / "results"
WORKING_DIR = OUTPUT_BASE_DIR / "working"

# Extensions that default to results/ (can be overridden with working=True)
_RESULTS_EXTENSIONS = {'.json', '.png'}


def get_output_path(filename: str, model_dir: str = None, working: bool = False) -> Path:
    """
    Route output file to appropriate subdirectory.
    Creates the target directory if it doesn't exist.

    Routing rules:
    - results/: Human-readable final outputs (summary JSONs, plots)
    - working/: Machine data (activations, directions, checkpoints, probes, logs)

    Args:
        filename: Just the filename (not a path), e.g., "dataset_results.json"
        model_dir: Optional model directory name from get_model_dir_name().
                   If provided, routes to outputs/{model_dir}/results|working/
        working: If True, force routing to working/ regardless of extension.
                 Use for checkpoints, intermediate state, machine-readable data.

    Returns:
        Full path in the appropriate subdirectory:
        - .png -> results/ (always)
        - .json -> results/ (unless working=True)
        - .npz, .joblib, .log, etc. -> working/

    Example:
        >>> get_output_path("TriviaMC_results.json", model_dir="Llama-3.1-8B-Instruct")
        PosixPath('.../outputs/Llama-3.1-8B-Instruct/results/TriviaMC_results.json')
        >>> get_output_path("TriviaMC_checkpoint.json", model_dir="...", working=True)
        PosixPath('.../outputs/Llama-3.1-8B-Instruct/working/TriviaMC_checkpoint.json')
        >>> get_output_path("TriviaMC_activations.npz", model_dir="Llama-3.1-8B-Instruct")
        PosixPath('.../outputs/Llama-3.1-8B-Instruct/working/TriviaMC_activations.npz')
    """
    ext = Path(filename).suffix.lower()
    # Force to working/ if explicitly requested, otherwise route by extension
    subdir = "working" if working else ("results" if ext in _RESULTS_EXTENSIONS else "working")

    if model_dir:
        target_dir = OUTPUT_BASE_DIR / model_dir / subdir
    else:
        target_dir = RESULTS_DIR if ext in _RESULTS_EXTENSIONS else WORKING_DIR

    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename


def find_output_file(filename: str, model_dir: str = None) -> Path:
    """
    Find an existing output file with migration-friendly fallback chain.

    Args:
        filename: Just the filename (not a path)
        model_dir: Optional model directory name. If provided, searches in model-specific dirs first.

    Returns:
        Path to the file. Search order when model_dir is provided:
        1. New model-dir location: outputs/{model_dir}/results|working/{filename}
        2. Legacy structured with model prefix: outputs/results|working/{model_dir}_{filename}
        3. Legacy flat with model prefix: outputs/{model_dir}_{filename}
        4. Returns new location path if none exist (for error messages)

        When model_dir is None (legacy mode):
        1. Structured: outputs/results|working/{filename}
        2. Flat: outputs/{filename}

    Example:
        # With model_dir (new pattern):
        >>> find_output_file("TriviaMC_results.json", model_dir="Llama-3.1-8B-Instruct")
        # Checks: outputs/Llama-3.1-8B-Instruct/results/TriviaMC_results.json
        # Then:   outputs/results/Llama-3.1-8B-Instruct_TriviaMC_results.json
        # Then:   outputs/Llama-3.1-8B-Instruct_TriviaMC_results.json
    """
    ext = Path(filename).suffix.lower()
    subdir = "results" if ext in _RESULTS_EXTENSIONS else "working"

    if model_dir:
        # 1. New model-dir location
        new_path = OUTPUT_BASE_DIR / model_dir / subdir / filename
        if new_path.exists():
            return new_path

        # 2. Legacy structured with model prefix
        legacy_filename = f"{model_dir}_{filename}"
        legacy_structured = (RESULTS_DIR if ext in _RESULTS_EXTENSIONS else WORKING_DIR) / legacy_filename
        if legacy_structured.exists():
            return legacy_structured

        # 3. Legacy flat with model prefix
        legacy_flat = OUTPUT_BASE_DIR / legacy_filename
        if legacy_flat.exists():
            return legacy_flat

        # Return new path if none exist
        return new_path
    else:
        # Legacy mode (no model_dir)
        structured_path = (RESULTS_DIR if ext in _RESULTS_EXTENSIONS else WORKING_DIR) / filename
        if structured_path.exists():
            return structured_path

        flat_path = OUTPUT_BASE_DIR / filename
        if flat_path.exists():
            return flat_path

        return structured_path


def glob_outputs(pattern: str, model_dir: str = None) -> List[Path]:
    """
    Glob for files across output directories with migration support.

    Args:
        pattern: Glob pattern, e.g., "*_mc_activations.npz"
        model_dir: Optional model directory name. If provided, also searches with model prefix.

    Returns:
        List of matching paths, deduplicated. Search locations:

        With model_dir:
        1. outputs/{model_dir}/working/{pattern}
        2. outputs/{model_dir}/results/{pattern}
        3. outputs/working/{model_dir}_{pattern} (legacy structured)
        4. outputs/results/{model_dir}_{pattern} (legacy structured)
        5. outputs/{model_dir}_{pattern} (legacy flat)

        Without model_dir:
        1. outputs/working/{pattern}
        2. outputs/results/{pattern}
        3. outputs/{pattern} (legacy flat, direct children only)

    Example:
        >>> glob_outputs("*_directions.npz", model_dir="Llama-3.1-8B-Instruct")
        [PosixPath('.../outputs/Llama-3.1-8B-Instruct/working/TriviaMC_directions.npz'), ...]
    """
    results = []

    if model_dir:
        # 1-2. New model-dir locations
        model_working = OUTPUT_BASE_DIR / model_dir / "working"
        model_results = OUTPUT_BASE_DIR / model_dir / "results"
        if model_working.exists():
            results.extend(model_working.glob(pattern))
        if model_results.exists():
            results.extend(model_results.glob(pattern))

        # 3-4. Legacy structured with model prefix
        legacy_pattern = f"{model_dir}_{pattern}"
        if WORKING_DIR.exists():
            results.extend(WORKING_DIR.glob(legacy_pattern))
        if RESULTS_DIR.exists():
            results.extend(RESULTS_DIR.glob(legacy_pattern))

        # 5. Legacy flat with model prefix
        if OUTPUT_BASE_DIR.exists():
            for p in OUTPUT_BASE_DIR.glob(legacy_pattern):
                if p.parent == OUTPUT_BASE_DIR:
                    results.append(p)
    else:
        # Legacy mode (no model_dir)
        if WORKING_DIR.exists():
            results.extend(WORKING_DIR.glob(pattern))
        if RESULTS_DIR.exists():
            results.extend(RESULTS_DIR.glob(pattern))

        # Legacy flat (only direct children)
        if OUTPUT_BASE_DIR.exists():
            for p in OUTPUT_BASE_DIR.glob(pattern):
                if p.parent == OUTPUT_BASE_DIR:
                    results.append(p)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for p in results:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique


def discover_model_dirs() -> List[str]:
    """
    Discover all model directories in the outputs folder.

    Returns:
        Sorted list of model directory names (excluding 'results' and 'working' legacy dirs).

    Example:
        >>> discover_model_dirs()
        ['Llama-3.1-8B-Instruct', 'Llama-3.1-8B-Instruct_4bit', 'Llama-3.1-8B-Instruct_adapter-lora']
    """
    if not OUTPUT_BASE_DIR.exists():
        return []

    return sorted([
        p.name for p in OUTPUT_BASE_DIR.iterdir()
        if p.is_dir() and p.name not in ("results", "working")
    ])
