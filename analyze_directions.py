"""
Stage 4. Logit lens analysis and pairwise cosine similarity between probe directions
from different experiments (MC, next-token, introspection, contrastive).

Loads direction files from various probe experiments, computes pairwise cosine
similarities, projects directions through the unembedding matrix (logit lens),
and generates visualizations.

Direction types and their relationships:
- mc_{metric}_{dataset}: Trained on MC questions to predict uncertainty metric
- introspection_{metric}_{dataset}: Also trained on MC questions (direct prompts) to
  predict the same uncertainty metric. The introspection experiment additionally tests
  whether these directions transfer to meta-cognition prompts.
- nexttoken_{metric}: Trained on diverse next-token prediction to predict uncertainty
- contrastive: Difference between high/low uncertainty activations (not a probe)

Inputs:
    outputs/{base}_mc_{metric}_directions.npz               MC uncertainty directions
    outputs/{base}_nexttoken_{metric}_directions.npz        Next-token uncertainty directions
    outputs/{base}_meta_{task}_metaconfdir_directions.npz   Confidence directions
    outputs/{base}_meta_{task}_metamcuncert_directions.npz  Meta→MC uncertainty directions

Outputs:
    outputs/{base}_direction_analysis.json             Pairwise similarity metrics
    outputs/{base}_logit_lens.png                      Logit lens projection plots
    (various additional analysis plots)

Run after: identify_mc_correlate.py, identify_nexttoken_correlate.py
"""

import argparse
import os
import re
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

from core import (
    DEVICE,
    get_model_short_name,
)
from core.config_utils import get_config_dict
from core.plotting import save_figure, GRID_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Set to adapter path if using fine-tuned model
DATASET_FILTER = None#"TriviaMC_difficulty_filtered"  # Only load directions for this dataset (None = all)

# --- Quantization ---
# Must match the setting used when producing the direction files
LOAD_IN_4BIT = False  # Set True for 70B+ models
LOAD_IN_8BIT = False

# --- Script-specific ---
TOP_K_TOKENS = 12  # Number of top tokens to show in logit lens
LAYERS_TO_ANALYZE = None  # None = all layers, or list like [10, 15, 20]
AVAILABLE_METRICS = ["logit_gap"]#"entropy", "top_prob", "margin", "logit_gap", "top_logit"]

# --- Output ---
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(MODEL, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)
    if ADAPTER is not None:
        adapter_short = get_model_short_name(ADAPTER)
        return str(OUTPUT_DIR / f"{model_short}_adapter-{adapter_short}")
    return str(OUTPUT_DIR / f"{model_short}")


def extract_dataset_from_npz(path: Path) -> Optional[str]:
    """
    Extract dataset name from npz file metadata.

    Returns dataset name if stored in metadata, None otherwise.
    """
    try:
        data = np.load(path)
        if "_metadata_dataset" in data.files:
            return str(data["_metadata_dataset"])
    except Exception:
        pass
    return None


def strip_adapter_prefix(s: str) -> str:
    """Strip adapter prefix from a string if present.

    E.g., 'adapter-ect_20251222_SimpleMC' -> 'SimpleMC'
    """
    if s.startswith("adapter-"):
        # Find the dataset part after adapter-xxx_
        # Pattern: adapter-{name}_{dataset}
        # The adapter name can contain underscores, so we look for known dataset patterns
        # or just take everything after the first occurrence of a known dataset prefix
        import re
        # Look for common dataset name patterns
        match = re.search(r'_(TriviaMC|PopMC|SimpleMC|GPQA|MMLU|ARC|Science)', s)
        if match:
            return s[match.start() + 1:]  # +1 to skip the underscore
        # Fallback: try to find where the adapter ends (last contiguous alphanumeric block before dataset)
        # This is fragile, so prefer metadata extraction
    return s


def extract_dataset_from_filename(filename: str, suffix: str) -> Optional[str]:
    """
    Extract dataset name from a direction filename (fallback for old files without metadata).

    Handles patterns like:
    - Llama-3.1-8B-Instruct_SimpleMC_mc_entropy_directions.npz
    - Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000_GPQA_mc_entropy_directions.npz

    The dataset name is the part immediately before the suffix (e.g., _mc_entropy_directions).

    WARNING: This will fail for dataset names containing underscores (e.g., Science_QA).
    Prefer using extract_dataset_from_npz() which reads metadata.
    """
    # Remove .npz extension if present
    if filename.endswith(".npz"):
        filename = filename[:-4]

    # Remove the known suffix
    if not filename.endswith(suffix):
        return None
    filename = filename[:-len(suffix)]

    # Now filename is like:
    # - Llama-3.1-8B-Instruct_SimpleMC
    # - Llama-3.1-8B-Instruct_adapter-ect_20251222_215412_v0uei7y1_2000_GPQA

    # The dataset is the last underscore-separated component
    # Split from the right to get the last part
    parts = filename.rsplit("_", 1)
    if len(parts) == 2:
        return parts[1]  # The dataset name
    return None


def extract_metric_from_npz(path: Path) -> Optional[str]:
    """
    Extract metric name from npz file metadata.

    Returns metric name if stored in metadata, None otherwise.
    """
    try:
        data = np.load(path)
        if "_metadata_metric" in data.files:
            return str(data["_metadata_metric"])
    except Exception:
        pass
    return None


def find_direction_files(output_dir: Path, model_short: str, metric_filter: Optional[str] = None, dataset_filter: Optional[str] = None, exclude_adapters: bool = False) -> Dict[str, Path]:
    """
    Find all direction files for a given model.

    Args:
        output_dir: Directory to search
        model_short: Short model name for pattern matching
        metric_filter: If specified, only include files for this metric
        dataset_filter: If specified, only include files for this dataset
        exclude_adapters: If True, exclude files with "adapter-" in the name

    Returns dict mapping direction_type -> path.
    For dataset-specific files (like mc), includes the dataset in the key.
    For metric-specific files, includes the metric in the key.
    """
    direction_files = {}

    # Patterns that are NOT dataset-specific or metric-specific (single file per model)
    simple_patterns = [
        ("introspection_direction", f"{model_short}*_direction_vectors.npz"),
    ]

    for direction_type, pattern in simple_patterns:
        matches = list(output_dir.glob(pattern))
        if matches:
            # Take the most recent if multiple
            direction_files[direction_type] = max(matches, key=lambda p: p.stat().st_mtime)

    # Contrastive directions from compute_contrastive_directions.py:
    # {model}_{dataset}_{metric}_contrastive_{dir_type}_directions.npz
    # where dir_type is "confidence" or "calibration"
    # Also supports old format: {model}_{dataset}_{metric}_contrastive_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        # New format with direction type suffix
        for dir_type in ["confidence", "calibration"]:
            pattern = f"{model_short}*_{metric}_contrastive_{dir_type}_directions.npz"
            for path in output_dir.glob(pattern):
                dataset = extract_dataset_from_npz(path)
                if dataset is None:
                    # Try to extract from filename
                    name = path.name
                    prefix = f"{model_short}_"
                    suffix = f"_{metric}_contrastive_{dir_type}_directions.npz"
                    if name.startswith(prefix) and name.endswith(suffix):
                        dataset = name[len(prefix):-len(suffix)]
                        dataset = strip_adapter_prefix(dataset)

                if dataset:
                    key = f"{dir_type}_{metric}_{dataset}"
                else:
                    key = f"{dir_type}_{metric}"

                if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                    direction_files[key] = path

        # Old format without direction type (backward compatibility)
        contrastive_pattern = f"{model_short}*_{metric}_contrastive_directions.npz"
        for path in output_dir.glob(contrastive_pattern):
            # Skip if this matches the new format (has _confidence_ or _calibration_)
            if "_confidence_directions.npz" in path.name or "_calibration_directions.npz" in path.name:
                continue

            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                # Try to extract from filename: {model}_{dataset}_{metric}_contrastive_directions.npz
                name = path.name
                prefix = f"{model_short}_"
                suffix = f"_{metric}_contrastive_directions.npz"
                if name.startswith(prefix) and name.endswith(suffix):
                    dataset = name[len(prefix):-len(suffix)]
                    dataset = strip_adapter_prefix(dataset)

            if dataset:
                key = f"contrastive_{metric}_{dataset}"
            else:
                key = f"contrastive_{metric}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Introspection direction files from two sources:
    # 1. run_introspection_experiment.py: {model}_{dataset}_introspection[_{task}]_{metric}_directions.npz
    # 2. run_introspection_probe.py: {model}_{dataset}_introspection[_{task}]_{metric}_probe_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        # Helper to extract task from filename
        def extract_task(filename: str, metric: str, has_probe: bool) -> Optional[str]:
            suffix = f"_{metric}_probe_directions\\.npz$" if has_probe else f"_{metric}_directions\\.npz$"
            task_match = re.search(rf"_introspection(?:_([^_]+))?{suffix}", filename)
            return task_match.group(1) if task_match and task_match.group(1) else None

        # 1. Match files from run_introspection_experiment.py (no _probe suffix)
        # Use negative lookahead to exclude _probe_directions files
        experiment_pattern = f"{model_short}*_introspection*_{metric}_directions.npz"
        for path in output_dir.glob(experiment_pattern):
            # Skip if this is actually a _probe_directions file
            if "_probe_directions.npz" in path.name:
                continue

            dataset = extract_dataset_from_npz(path)
            task = extract_task(path.name, metric, has_probe=False)

            if dataset and task:
                key = f"introspection_{task}_{metric}_{dataset}"
            elif dataset:
                key = f"introspection_{metric}_{dataset}"
            elif task:
                key = f"introspection_{task}_{metric}"
            else:
                key = f"introspection_{metric}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

        # 2. Match files from run_introspection_probe.py (_probe suffix)
        probe_pattern = f"{model_short}*_introspection*_{metric}_probe_directions.npz"
        for path in output_dir.glob(probe_pattern):
            dataset = extract_dataset_from_npz(path)
            task = extract_task(path.name, metric, has_probe=True)

            if dataset and task:
                key = f"introspection_probe_{task}_{metric}_{dataset}"
            elif dataset:
                key = f"introspection_probe_{metric}_{dataset}"
            elif task:
                key = f"introspection_probe_{task}_{metric}"
            else:
                key = f"introspection_probe_{metric}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Backward compatibility: old introspection_entropy/probe patterns without dataset
    # These are ONLY for files with the exact pattern {model}_introspection_entropy_directions.npz
    # (no dataset in the name). Skip if we already found dataset-specific introspection files.
    if not metric_filter or metric_filter == "entropy":
        # Only add these if we found NO dataset-specific introspection files
        has_dataset_specific = any(k.startswith("introspection_") and k.count("_") >= 2
                                   for k in direction_files)
        if not has_dataset_specific:
            old_intro_patterns = [
                ("introspection_entropy", f"{model_short}_introspection_entropy_directions.npz"),
                ("introspection_probe", f"{model_short}_introspection_probe_directions.npz"),
            ]
            for key, pattern in old_intro_patterns:
                if key not in direction_files:
                    matches = list(output_dir.glob(pattern))
                    if matches:
                        direction_files[key] = max(matches, key=lambda p: p.stat().st_mtime)

    # Metric-specific nexttoken patterns
    # Pattern: {model}_nexttoken_{metric}_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        nexttoken_pattern = f"{model_short}*_nexttoken_{metric}_directions.npz"
        matches = list(output_dir.glob(nexttoken_pattern))
        if matches:
            path = max(matches, key=lambda p: p.stat().st_mtime)
            direction_files[f"nexttoken_{metric}"] = path

    # Backward compatibility: old nexttoken_entropy_directions.npz format
    if not metric_filter or metric_filter == "entropy":
        old_pattern = f"{model_short}*_nexttoken_entropy_directions.npz"
        old_matches = list(output_dir.glob(old_pattern))
        for path in old_matches:
            # Check if this is NOT a metric-specific file (old format)
            # Old format: model_nexttoken_entropy_directions.npz
            # New format: model_nexttoken_entropy_directions.npz (same name for entropy)
            # We need to check if we already found it via the new pattern
            if "nexttoken_entropy" not in direction_files:
                direction_files["nexttoken_entropy"] = path

    # Dataset-specific and metric-specific MC patterns
    # New pattern: {model}_{dataset}_mc_{metric}_directions.npz
    # Old pattern: {model}_{dataset}_mc_entropy_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        mc_pattern = f"{model_short}*_mc_{metric}_directions.npz"
        mc_matches = list(output_dir.glob(mc_pattern))
        for path in mc_matches:
            # Try to get dataset from metadata first
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                dataset = extract_dataset_from_filename(path.name, f"_mc_{metric}_directions")

            if dataset:
                key = f"mc_{metric}_{dataset}"
            else:
                key = f"mc_{metric}"

            # If we already have this key, keep the most recent
            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Backward compatibility: old mc_entropy_directions.npz format
    if not metric_filter or metric_filter == "entropy":
        old_mc_pattern = f"{model_short}*_mc_entropy_directions.npz"
        old_mc_matches = list(output_dir.glob(old_mc_pattern))
        for path in old_mc_matches:
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                dataset = extract_dataset_from_filename(path.name, "_mc_entropy_directions")

            if dataset:
                key = f"mc_entropy_{dataset}"
            else:
                key = "mc_entropy"

            # Only add if we don't already have this from the new pattern search
            if key not in direction_files:
                direction_files[key] = path

    # Orthogonal directions from compute_orthogonal_directions.py:
    # {model}_{dataset}_orthogonal_directions.npz
    # Contains: self_confidence_unique_layer_N, other_confidence_unique_layer_N
    # (legacy: introspection_layer_N, surface_layer_N)
    orthogonal_pattern = f"{model_short}*_orthogonal_directions.npz"
    for path in output_dir.glob(orthogonal_pattern):
        dataset = extract_dataset_from_npz(path)
        if dataset is None:
            # Try to extract from filename
            name = path.name
            prefix = f"{model_short}_"
            suffix = "_orthogonal_directions.npz"
            if name.startswith(prefix) and name.endswith(suffix):
                dataset = name[len(prefix):-len(suffix)]
                dataset = strip_adapter_prefix(dataset)

        if dataset:
            key = f"orthogonal_{dataset}"
        else:
            key = "orthogonal"

        if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
            direction_files[key] = path

    # Consensus directions from compare_directions_cross_dataset.py:
    # {model}_consensus_directions.npz or {model}_adapter-{name}_consensus_directions.npz
    # Contains: d_self_confidence_layer_N, d_other_confidence_layer_N,
    #           d_self_confidence_unique_layer_N, d_other_confidence_unique_layer_N,
    #           d_contrast_layer_N, d_mc_{metric}_layer_N (averaged across datasets)
    consensus_pattern = f"{model_short}*_consensus_directions.npz"
    for path in output_dir.glob(consensus_pattern):
        # Use most recent if multiple exist
        if "consensus" not in direction_files or path.stat().st_mtime > direction_files["consensus"].stat().st_mtime:
            direction_files["consensus"] = path

    # Meta confidence directions from test_meta_transfer.py:
    # {model}_{dataset}_meta_confidence_metaconfdir_directions.npz (d_self_confidence)
    # {model}_{dataset}_meta_other_confidence_metaconfdir_directions.npz (d_other_confidence)
    for conf_type, label in [("confidence", "d_self_confidence"), ("other_confidence", "d_other_confidence")]:
        pattern = f"{model_short}*_meta_{conf_type}_metaconfdir_directions.npz"
        for path in output_dir.glob(pattern):
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                # Try to extract from filename
                name = path.name
                prefix = f"{model_short}_"
                suffix = f"_meta_{conf_type}_metaconfdir_directions.npz"
                if name.startswith(prefix) and name.endswith(suffix):
                    dataset = name[len(prefix):-len(suffix)]
                    # Strip adapter prefix if present (e.g., "adapter-xyz_SimpleMC" -> "SimpleMC")
                    dataset = strip_adapter_prefix(dataset)

            if dataset:
                key = f"{label}_{dataset}"
            else:
                key = label

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # selfVother_conf directions from compute_contrast_direction.py:
    # {model}_{dataset}_selfVother_conf_directions.npz (new format)
    # {model}_{dataset}_self_vs_other_confidence_directions.npz (legacy format)
    # {model}_{dataset}_contrast_directions.npz (legacy format)
    # Contains: selfVother_conf_layer_N or self_vs_other_confidence_layer_N or contrast_layer_N
    for suffix_pattern in ["_selfVother_conf_directions.npz", "_self_vs_other_confidence_directions.npz", "_contrast_directions.npz"]:
        pattern = f"{model_short}*{suffix_pattern}"
        for path in output_dir.glob(pattern):
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                # Try to extract from filename
                name = path.name
                prefix = f"{model_short}_"
                if name.startswith(prefix) and name.endswith(suffix_pattern):
                    dataset = name[len(prefix):-len(suffix_pattern)]
                    dataset = strip_adapter_prefix(dataset)

            if dataset:
                key = f"selfVother_conf_{dataset}"
            else:
                key = "selfVother_conf"

            # Use most recent if multiple exist, prefer new format
            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Meta→MC uncertainty directions from test_meta_transfer.py (with FIND_MC_UNCERTAINTY_DIRECTIONS=True):
    # {model}_{dataset}_meta_confidence_metamcuncert_directions.npz
    # {model}_{dataset}_meta_other_confidence_metamcuncert_directions.npz
    for conf_type in ["confidence", "other_confidence"]:
        pattern = f"{model_short}*_meta_{conf_type}_metamcuncert_directions.npz"
        for path in output_dir.glob(pattern):
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                # Try to extract from filename
                name = path.name
                prefix = f"{model_short}_"
                suffix = f"_meta_{conf_type}_metamcuncert_directions.npz"
                if name.startswith(prefix) and name.endswith(suffix):
                    dataset = name[len(prefix):-len(suffix)]
                    dataset = strip_adapter_prefix(dataset)

            if dataset:
                key = f"d_meta_mc_uncert_{conf_type}_{dataset}"
            else:
                key = f"d_meta_mc_uncert_{conf_type}"

            if key not in direction_files or path.stat().st_mtime > direction_files[key].stat().st_mtime:
                direction_files[key] = path

    # Filter by dataset if specified
    if dataset_filter:
        direction_files = {
            k: v for k, v in direction_files.items()
            if dataset_filter in k or dataset_filter in str(v)
        }

    # Filter out adapter files if requested
    if exclude_adapters:
        direction_files = {
            k: v for k, v in direction_files.items()
            if "adapter-" not in str(v.name)
        }

    return direction_files


def load_directions(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load directions from a .npz file.

    Args:
        path: Path to .npz file

    Returns dict mapping layer_idx -> {direction_name: direction_vector}

    Handles two key formats:
    - "layer_N_name" or "layer_N" (old format, e.g., from introspection experiments)
    - "name_layer_N" (new format from identify_mc_correlate.py, e.g., "probe_layer_0", "mean_diff_layer_0")
    """
    data = np.load(path)

    directions = defaultdict(dict)

    for key in data.files:
        # Skip metadata keys
        if key.startswith("_metadata"):
            continue

        parts = key.split("_")

        # Try format 1: "layer_N_name" or "layer_N"
        if parts[0] == "layer" and len(parts) >= 2:
            try:
                layer_idx = int(parts[1])
                if len(parts) > 2:
                    direction_name = "_".join(parts[2:])
                else:
                    # For files with just "layer_N" keys, use "probe" as direction name
                    direction_name = "probe"
                directions[layer_idx][direction_name] = data[key]
                continue
            except ValueError:
                pass

        # Try format 2: "name_layer_N" (from identify_mc_correlate.py)
        # Look for "_layer_" in the key
        if "_layer_" in key:
            # Split on "_layer_" to get (name, layer_idx)
            layer_pos = key.rfind("_layer_")
            direction_name = key[:layer_pos]
            try:
                layer_idx = int(key[layer_pos + 7:])  # len("_layer_") == 7
                # Skip scaler keys (probe_scaler_scale_N, probe_scaler_mean_N)
                if "scaler" in direction_name:
                    continue
                # Skip scalar entries (cosine, residual_norm from orthogonal files)
                if direction_name in ("cosine", "residual_norm"):
                    continue
                arr = data[key]
                # Skip if not a proper direction vector (must be 1D with reasonable size)
                if arr.ndim != 1 or arr.shape[0] < 100:
                    continue
                directions[layer_idx][direction_name] = arr
            except ValueError:
                pass

    return dict(directions)


def compute_cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2)))


def compute_pairwise_similarities(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layer_idx: int
) -> Dict[Tuple[str, str], float]:
    """
    Compute pairwise cosine similarities between all direction types at a given layer.

    Returns dict mapping (type1, type2) -> cosine_similarity
    """
    similarities = {}

    # Flatten to get all (source, name) pairs
    direction_items = []
    for source, layers in all_directions.items():
        if layer_idx in layers:
            for name, direction in layers[layer_idx].items():
                full_name = f"{source}/{name}"
                direction_items.append((full_name, direction))

    # Compute pairwise
    for i, (name1, d1) in enumerate(direction_items):
        for j, (name2, d2) in enumerate(direction_items):
            if i <= j:
                sim = compute_cosine_similarity(d1, d2)
                similarities[(name1, name2)] = sim
                similarities[(name2, name1)] = sim

    return similarities


def load_lm_head_and_norm(model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Load lm_head weight and final norm weight directly from model files.

    This bypasses the model loading and directly loads just the weights needed
    for logit lens from the safetensors files. Much faster and uses less memory
    than loading the full model.

    Returns:
        Tuple of (lm_head_weight, norm_weight)
    """
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    print(f"  Downloading weight index...")

    # Download the index file to find which shards have our weights
    index_file = hf_hub_download(
        repo_id=model_name,
        filename="model.safetensors.index.json",
        token=os.environ.get("HF_TOKEN")
    )

    with open(index_file) as f:
        index = json.load(f)

    weight_map = index.get("weight_map", {})

    # Find which files contain our weights
    lm_head_file = weight_map.get("lm_head.weight")
    norm_file = weight_map.get("model.norm.weight")

    if not lm_head_file:
        raise ValueError(f"Could not find lm_head.weight in model index")

    # Download and load lm_head
    print(f"  Downloading {lm_head_file}...")
    shard_path = hf_hub_download(
        repo_id=model_name,
        filename=lm_head_file,
        token=os.environ.get("HF_TOKEN")
    )

    print(f"  Loading lm_head weight to {DEVICE}...")
    with safe_open(shard_path, framework="pt", device=DEVICE) as f:
        lm_head_weight = f.get_tensor("lm_head.weight")

    print(f"  Loaded lm_head weight: {lm_head_weight.shape}, dtype: {lm_head_weight.dtype}")

    # Download and load norm weight (may be in same or different shard)
    norm_weight = None
    if norm_file:
        if norm_file != lm_head_file:
            print(f"  Downloading {norm_file}...")
            norm_shard_path = hf_hub_download(
                repo_id=model_name,
                filename=norm_file,
                token=os.environ.get("HF_TOKEN")
            )
        else:
            norm_shard_path = shard_path

        print(f"  Loading norm weight...")
        with safe_open(norm_shard_path, framework="pt", device=DEVICE) as f:
            norm_weight = f.get_tensor("model.norm.weight")
        print(f"  Loaded norm weight: {norm_weight.shape}")
    else:
        print(f"  Warning: Could not find model.norm.weight, skipping normalization")

    return lm_head_weight, norm_weight


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm to a vector."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return weight * x_normed


def clean_token_str(s: str) -> str:
    """Clean token string for display - remove non-ASCII and problematic chars."""
    # Remove non-ASCII characters
    s = re.sub(r'[^\x00-\x7F]+', '', str(s))
    # Remove characters that might trigger MathText parsing
    s = re.sub(r'[\$\^\\]', '', s)
    # Replace newlines and tabs with visible representation
    s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    # Limit length
    if len(s) > 12:
        s = s[:10] + '..'
    return s if s.strip() else repr(s)


def logit_lens_for_layer(
    direction: np.ndarray,
    lm_head_weight: torch.Tensor,
    tokenizer,
    top_k: int = 12,
    norm_weight: Optional[torch.Tensor] = None
) -> Tuple[List[str], List[float]]:
    """
    Project a direction through the unembedding matrix.

    Args:
        direction: The direction vector to project
        lm_head_weight: The unembedding matrix
        tokenizer: Tokenizer for decoding
        top_k: Number of top tokens to return
        norm_weight: If provided, apply RMSNorm before unembedding (recommended)

    Returns:
        Tuple of (top_tokens, top_probs) - tokens and their softmax probabilities
    """
    # Project direction through unembedding
    direction_tensor = torch.tensor(
        direction,
        dtype=lm_head_weight.dtype,
        device=lm_head_weight.device
    )

    # Apply RMSNorm if weights provided (matches model's forward pass)
    if norm_weight is not None:
        direction_tensor = rms_norm(direction_tensor, norm_weight)

    logits = direction_tensor @ lm_head_weight.T  # (vocab_size,)

    # Softmax to get probabilities
    probs = torch.nn.functional.softmax(logits, dim=-1)

    # Get top-k
    values, indices = torch.topk(probs, top_k)

    # Decode tokens
    tokens = tokenizer.batch_decode(indices.unsqueeze(-1))
    probs_list = values.cpu().tolist()

    return tokens, probs_list


def analyze_layer(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layer_idx: int,
    lm_head_weight: Optional[torch.Tensor],
    tokenizer,
    top_k: int = 12,
    norm_weight: Optional[torch.Tensor] = None
) -> Dict:
    """
    Run full analysis on a single layer.

    Returns dict with similarities and logit lens results.
    """
    results = {
        "layer": layer_idx,
        "similarities": {},
        "logit_lens": {},
    }

    # Compute pairwise similarities
    similarities = compute_pairwise_similarities(all_directions, layer_idx)
    results["similarities"] = {f"{k[0]}__vs__{k[1]}": v for k, v in similarities.items()}

    # Run logit lens on each direction (if weight available)
    if lm_head_weight is not None:
        for source, layers in all_directions.items():
            if layer_idx in layers:
                for name, direction in layers[layer_idx].items():
                    full_name = f"{source}/{name}"
                    tokens, probs = logit_lens_for_layer(direction, lm_head_weight, tokenizer, top_k, norm_weight)
                    results["logit_lens"][full_name] = {
                        "tokens": tokens,
                        "probs": probs,
                    }

    return results


def plot_logit_lens_heatmap(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layers: List[int],
    direction_source: str,
    direction_name: str,
    lm_head_weight: torch.Tensor,
    tokenizer,
    output_path: Path,
    top_k: int = 12,
    norm_weight: Optional[torch.Tensor] = None
):
    """
    Plot heatmap showing top-k tokens for each layer.
    Two panels: positive direction (top) and negative direction (bottom).
    Rows = layers, Columns = top-k tokens
    Cell values = softmax probabilities, annotations = token strings
    """
    # Collect data for both positive and negative directions
    pos_token_data = []
    pos_probs_data = []
    neg_token_data = []
    neg_probs_data = []

    for layer_idx in layers:
        if direction_source in all_directions and layer_idx in all_directions[direction_source]:
            if direction_name in all_directions[direction_source][layer_idx]:
                direction = all_directions[direction_source][layer_idx][direction_name]
                # Positive direction
                pos_tokens, pos_probs = logit_lens_for_layer(direction, lm_head_weight, tokenizer, top_k, norm_weight)
                pos_token_data.append(pos_tokens)
                pos_probs_data.append(pos_probs)
                # Negative direction
                neg_tokens, neg_probs = logit_lens_for_layer(-direction, lm_head_weight, tokenizer, top_k, norm_weight)
                neg_token_data.append(neg_tokens)
                neg_probs_data.append(neg_probs)
            else:
                pos_token_data.append([''] * top_k)
                pos_probs_data.append([0.0] * top_k)
                neg_token_data.append([''] * top_k)
                neg_probs_data.append([0.0] * top_k)
        else:
            pos_token_data.append([''] * top_k)
            pos_probs_data.append([0.0] * top_k)
            neg_token_data.append([''] * top_k)
            neg_probs_data.append([0.0] * top_k)

    if not pos_probs_data:
        print(f"No data for {direction_source}/{direction_name}")
        return

    # Convert to arrays
    pos_probs_array = np.array(pos_probs_data)
    pos_token_labels = np.array(pos_token_data)
    neg_probs_array = np.array(neg_probs_data)
    neg_token_labels = np.array(neg_token_data)

    # Clean token labels for display
    pos_cleaned = np.vectorize(clean_token_str)(pos_token_labels)
    neg_cleaned = np.vectorize(clean_token_str)(neg_token_labels)

    # Plot with two panels
    plt.rcParams['font.family'] = 'DejaVu Sans'
    panel_height = max(4, len(layers) * 0.25)
    fig, axes = plt.subplots(2, 1, figsize=(14, panel_height * 2 + 1))

    # Positive direction (top panel)
    sns.heatmap(
        pos_probs_array,
        annot=pos_cleaned,
        fmt='',
        cmap="Reds",
        xticklabels=False,
        yticklabels=[f"L{l}" for l in layers],
        ax=axes[0],
        cbar_kws={'label': 'Softmax Prob'}
    )
    axes[0].set_title(f"Positive Direction (+d): {direction_source}/{direction_name}")
    axes[0].set_xlabel(f"Top {top_k} Tokens")
    axes[0].set_ylabel("Layer")

    # Negative direction (bottom panel)
    sns.heatmap(
        neg_probs_array,
        annot=neg_cleaned,
        fmt='',
        cmap="Blues",
        xticklabels=False,
        yticklabels=[f"L{l}" for l in layers],
        ax=axes[1],
        cbar_kws={'label': 'Softmax Prob'}
    )
    axes[1].set_title(f"Negative Direction (-d): {direction_source}/{direction_name}")
    axes[1].set_xlabel(f"Top {top_k} Tokens")
    axes[1].set_ylabel("Layer")

    save_figure(fig, output_path)


def plot_similarity_across_layers(
    all_directions: Dict[str, Dict[int, Dict[str, np.ndarray]]],
    layers: List[int],
    output_path: Path
):
    """
    Plot how cosine similarity between direction types evolves across layers.
    """
    # Get all unique direction pairs
    all_names = set()
    for source, layer_data in all_directions.items():
        for layer_idx, directions in layer_data.items():
            for name in directions.keys():
                all_names.add(f"{source}/{name}")

    all_names = sorted(all_names)

    if len(all_names) < 2:
        # Skip silently - only one direction type
        return

    # Compute similarities for each layer
    pair_data = defaultdict(list)

    for layer_idx in layers:
        sims = compute_pairwise_similarities(all_directions, layer_idx)
        for (n1, n2), sim in sims.items():
            if n1 < n2:  # Avoid duplicates
                pair_data[(n1, n2)].append((layer_idx, sim))

    # Plot
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig, ax = plt.subplots(figsize=(12, 6))

    for (n1, n2), data in pair_data.items():
        if data:
            xs, ys = zip(*sorted(data))
            # Use source names for clearer labels
            # n1, n2 are like "source/direction_name"
            src1, dir1 = n1.split('/', 1)
            src2, dir2 = n2.split('/', 1)
            # Shorten source names for readability
            src1_short = src1.replace("_entropy", "").replace("introspection_", "intro_")
            src2_short = src2.replace("_entropy", "").replace("introspection_", "intro_")
            label = f"{src1_short} vs {src2_short}"
            ax.plot(xs, ys, 'o-', label=label, alpha=0.7)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Direction Similarity Across Layers")
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    save_figure(fig, output_path)


def main():
    parser = argparse.ArgumentParser(description="Analyze and compare probe directions")
    parser.add_argument("--model-only", action="store_true",
                        help="Only load model, skip analysis")
    parser.add_argument("--layer", type=int, default=None,
                        help="Focus on specific layer")
    parser.add_argument("--metric", type=str, default=None, choices=AVAILABLE_METRICS,
                        help="Only analyze directions for this metric")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip generating plots")
    parser.add_argument("--skip-logit-lens", action="store_true",
                        help="Skip logit lens analysis (no model loading needed)")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if args.metric:
        print(f"Metric filter: {args.metric}")
    if DATASET_FILTER:
        print(f"Dataset filter: {DATASET_FILTER}")

    # Find direction files
    model_short = get_model_short_name(MODEL, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)
    direction_files = find_direction_files(
        OUTPUT_DIR, model_short,
        metric_filter=args.metric,
        dataset_filter=DATASET_FILTER,
        exclude_adapters=(ADAPTER is None)
    )

    if not direction_files:
        print(f"No direction files found in {OUTPUT_DIR} for model {model_short}")
        if args.metric:
            print(f"  (filtered by metric: {args.metric})")
        print("Run one of the probe scripts first:")
        print("  - nexttoken_entropy_probe.py")
        print("  - mc_entropy_probe.py")
        print("  - run_introspection_experiment.py")
        print("  - run_contrastive_direction.py")
        return

    print(f"\nFound {len(direction_files)} direction file(s):")
    for name, path in direction_files.items():
        print(f"  {name}: {path}")

    # Load all directions
    all_directions = {}
    for source, path in direction_files.items():
        all_directions[source] = load_directions(path)
        print(f"  Loaded {source}: {len(all_directions[source])} layers")

    # Determine layers to analyze
    all_layers = set()
    for layers_dict in all_directions.values():
        all_layers.update(layers_dict.keys())
    all_layers = sorted(all_layers)

    if args.layer is not None:
        layers_to_analyze = [args.layer]
    elif LAYERS_TO_ANALYZE is not None:
        layers_to_analyze = LAYERS_TO_ANALYZE
    else:
        layers_to_analyze = all_layers

    print(f"\nLayers available: {len(all_layers)} layers")
    print(f"Layers to analyze: {len(layers_to_analyze)} layers")

    # Load tokenizer and lm_head weight for logit lens (unless skipped)
    tokenizer = None
    lm_head_weight = None
    norm_weight = None

    if not args.skip_logit_lens:
        from transformers import AutoTokenizer
        from dotenv import load_dotenv

        load_dotenv()

        print(f"\nLoading tokenizer: {MODEL}")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL,
            token=os.environ.get("HF_TOKEN")
        )

        if args.model_only:
            print("Tokenizer loaded. Exiting (--model-only specified)")
            return

        # Load lm_head weight and norm weight directly (much faster than loading full model)
        print(f"\nLoading lm_head and norm weights for logit lens...")
        lm_head_weight, norm_weight = load_lm_head_and_norm(MODEL)
    else:
        print("\nSkipping logit lens analysis (--skip-logit-lens)")

    # Run analysis
    output_prefix = get_output_prefix()
    all_results = {}

    for layer_idx in tqdm(layers_to_analyze, desc="Analyzing layers"):
        results = analyze_layer(all_directions, layer_idx, lm_head_weight, tokenizer, TOP_K_TOKENS, norm_weight)
        all_results[layer_idx] = results

    # =========================================================================
    # Console Summary
    # =========================================================================

    print("\n" + "="*80)
    print("DIRECTION ANALYSIS RESULTS")
    print("="*80)

    # --- Direction Similarity Summary ---
    pair_similarities = defaultdict(list)  # (name1, name2) -> [(layer, cosine), ...]

    for layer_idx, results in sorted(all_results.items()):
        for key, sim in results["similarities"].items():
            parts = key.split("__vs__")
            if len(parts) == 2:
                n1, n2 = parts
                if n1 < n2:  # Avoid duplicates
                    pair_similarities[(n1, n2)].append((layer_idx, sim))

    if pair_similarities:
        print(f"\nDIRECTION SIMILARITY ({len(pair_similarities)} pair(s) across {len(layers_to_analyze)} layers):")
        print(f"  {'Pair':<60s}  {'Mean |cos|':>10s}  {'Max |cos|':>10s}  {'Max Layer':>10s}")
        print(f"  {'-'*60}  {'-'*10}  {'-'*10}  {'-'*10}")

        for (n1, n2), layer_sims in sorted(pair_similarities.items()):
            abs_sims = [abs(s) for _, s in layer_sims]
            mean_abs = np.mean(abs_sims)
            max_idx = int(np.argmax(abs_sims))
            max_abs = abs_sims[max_idx]
            max_layer = layer_sims[max_idx][0]

            pair_label = f"{n1} vs {n2}"
            if len(pair_label) > 60:
                pair_label = pair_label[:57] + "..."

            print(f"  {pair_label:<60s}  {mean_abs:>10.3f}  {max_abs:>10.3f}  {'L' + str(max_layer):>10s}")
    else:
        print("\nNo direction pairs found for similarity analysis.")

    # --- Self vs Other Metamcuncert Comparison (Introspection Test) ---
    # Find same-method pairs: probe↔probe and mean_diff↔mean_diff
    # Skip cross-method pairs (probe↔mean_diff) as they mix task AND method differences

    def get_method(name: str) -> str:
        if name.endswith("/probe"):
            return "probe"
        elif name.endswith("/mean_diff"):
            return "mean_diff"
        return "unknown"

    def is_self_other_pair(n1: str, n2: str) -> bool:
        """Check if this is a self-confidence vs other-confidence metamcuncert pair."""
        has_self = "d_meta_mc_uncert_confidence" in n1 or "d_meta_mc_uncert_confidence" in n2
        has_other = "d_meta_mc_uncert_other_confidence" in n1 or "d_meta_mc_uncert_other_confidence" in n2
        # Must have one of each (not both self or both other)
        return has_self and has_other

    # Group by method, only keeping same-method comparisons
    method_pairs = {"probe": None, "mean_diff": None}
    for (n1, n2), layer_sims in pair_similarities.items():
        if not is_self_other_pair(n1, n2):
            continue
        m1, m2 = get_method(n1), get_method(n2)
        if m1 == m2 and m1 in method_pairs:
            method_pairs[m1] = layer_sims

    if any(v is not None for v in method_pairs.values()):
        print("\n" + "-"*80)
        print("SELF vs OTHER METAMCUNCERT (Introspection Test)")
        print("-"*80)
        print("""
Comparing directions that predict MC uncertainty from SELF-confidence vs OTHER-confidence.
Same method applied to both tasks - only the task differs.

Key prediction:
  - High cosine (~1): Same direction → both tasks use surface cues (question difficulty)
  - Low cosine (~0): Different directions → self-confidence may use genuine introspection
""")

        for method, layer_sims in method_pairs.items():
            if layer_sims is None:
                continue

            layer_cosines = sorted(layer_sims, key=lambda x: x[0])
            print(f"  === {method.upper()} method: SELF-CONFIDENCE vs OTHER-CONFIDENCE ===")
            print()
            print(f"  {'Layer':<8s}  {'Cosine':>10s}")
            print(f"  {'-'*8}  {'-'*10}")

            for layer_idx, cos_val in layer_cosines:
                marker = ""
                if abs(cos_val) > 0.8:
                    marker = " ← HIGH (same direction)"
                elif abs(cos_val) < 0.3:
                    marker = " ← LOW (different directions)"
                print(f"  L{layer_idx:<6d}  {cos_val:>10.3f}{marker}")

            all_cos = [c for _, c in layer_cosines]
            early_cos = [c for l, c in layer_cosines if l <= 10]
            late_cos = [c for l, c in layer_cosines if l >= 20]

            print()
            print(f"  Mean: {np.mean(all_cos):.3f}  |  Early (0-10): {np.mean(early_cos):.3f}  |  Late (20+): {np.mean(late_cos):.3f}")
            print()

        # Overall interpretation
        probe_late = mean_diff_late = None
        if method_pairs["probe"] is not None:
            probe_late = np.mean([c for l, c in method_pairs["probe"] if l >= 20])
        if method_pairs["mean_diff"] is not None:
            mean_diff_late = np.mean([c for l, c in method_pairs["mean_diff"] if l >= 20])

        print("  INTERPRETATION:")
        if probe_late is not None and mean_diff_late is not None:
            if mean_diff_late > 0.7 and probe_late < 0.5:
                print("    Method-dependent divergence detected:")
                print(f"    - mean_diff stays aligned ({mean_diff_late:.2f}) → shared surface cues")
                print(f"    - probe diverges ({probe_late:.2f}) → unique component in self-confidence")
                print("    Causal validation needed to determine if unique component matters.")
            elif mean_diff_late > 0.7 and probe_late > 0.7:
                print("    High alignment for both methods → surface cues dominate.")
            elif mean_diff_late < 0.5 and probe_late < 0.5:
                print("    Low alignment for both methods → different mechanisms for self vs other.")
            else:
                print(f"    Mixed pattern: probe={probe_late:.2f}, mean_diff={mean_diff_late:.2f}")

    # --- Logit Lens Summary ---
    if lm_head_weight is not None:
        # Pick representative layers: ~5 evenly spaced
        if len(layers_to_analyze) <= 5:
            repr_layers = list(layers_to_analyze)
        else:
            indices = np.linspace(0, len(layers_to_analyze) - 1, 5, dtype=int)
            repr_layers = [layers_to_analyze[i] for i in indices]

        # Collect all direction type names
        direction_types = set()
        for layer_idx, results in all_results.items():
            direction_types.update(results.get("logit_lens", {}).keys())

        if direction_types:
            print(f"\nLOGIT LENS (top-5 tokens at representative layers):")

            for dir_name in sorted(direction_types):
                print(f"\n  {dir_name}:")
                for layer_idx in repr_layers:
                    if layer_idx in all_results:
                        lens_data = all_results[layer_idx].get("logit_lens", {}).get(dir_name)
                        if lens_data:
                            top5_tokens = lens_data["tokens"][:5]
                            top5_probs = lens_data["probs"][:5]
                            token_strs = [f'"{clean_token_str(t)}" ({p:.3f})' for t, p in zip(top5_tokens, top5_probs)]
                            print(f"    L{layer_idx:<3d}: {', '.join(token_strs)}")
    else:
        print("\nLogit lens skipped (no model weights loaded).")

    # Save results
    results_path = Path(f"{output_prefix}_direction_analysis.json")

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        elif isinstance(obj, tuple):
            return [convert_for_json(v) for v in obj]
        return obj

    # Build output with config
    output_data = {
        "config": get_config_dict(
            model=MODEL,
            adapter=ADAPTER,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            top_k_tokens=TOP_K_TOKENS,
            layers_analyzed=len(layers_to_analyze),
            direction_files={k: str(v) for k, v in direction_files.items()},
            skip_logit_lens=args.skip_logit_lens,
        ),
        "per_layer": all_results,
    }

    # Add similarity summary to output
    if pair_similarities:
        sim_summary = {}
        for (n1, n2), layer_sims in sorted(pair_similarities.items()):
            abs_sims = [abs(s) for _, s in layer_sims]
            mean_abs = float(np.mean(abs_sims))
            max_idx = int(np.argmax(abs_sims))
            max_abs = float(abs_sims[max_idx])
            max_layer = layer_sims[max_idx][0]
            pair_key = f"{n1}__vs__{n2}"
            sim_summary[pair_key] = {
                "mean_abs_cosine": mean_abs,
                "max_abs_cosine": max_abs,
                "max_abs_layer": max_layer,
            }
        output_data["similarity_summary"] = sim_summary

    with open(results_path, "w") as f:
        json.dump(convert_for_json(output_data), f, indent=2)
    print(f"\nSaved analysis results to {results_path}")

    # Generate plots
    if not args.no_plots:
        # Check if we have multiple direction types for similarity plots
        num_direction_types = sum(
            1 for source in all_directions.values()
            for layer_dirs in source.values()
            for _ in layer_dirs.keys()
        ) // len(all_layers) if all_layers else 0

        if num_direction_types >= 2:
            # Similarity across layers
            plot_similarity_across_layers(
                all_directions, all_layers,
                Path(f"{output_prefix}_direction_similarity_across_layers.png")
            )
        else:
            print("\nOnly one direction type found - skipping similarity plot")

        # Logit lens heatmaps for each direction type (if we have weights)
        if lm_head_weight is not None:
            for source, layers_dict in all_directions.items():
                # Get direction names from first available layer
                first_layer = next(iter(layers_dict.keys()))
                for direction_name in layers_dict[first_layer].keys():
                    # Build consistent filename: {output_prefix}_logit_lens_{direction_type}_{dataset}.png
                    # Extract dataset from source (e.g., "d_self_confidence_SimpleMC" -> "SimpleMC")
                    # or use source directly if no dataset embedded

                    # Determine the full direction type name
                    if source.startswith("orthogonal_"):
                        # orthogonal_SimpleMC + self_confidence_unique -> d_self_confidence_unique
                        dataset = source.replace("orthogonal_", "")
                        # Handle legacy names
                        if direction_name == "introspection":
                            dir_type = "d_self_confidence_unique"
                        elif direction_name == "surface":
                            dir_type = "d_other_confidence_unique"
                        else:
                            dir_type = f"d_{direction_name}"
                    elif source.startswith("selfVother_conf_"):
                        # selfVother_conf_SimpleMC -> d_selfVother_conf
                        dataset = source.replace("selfVother_conf_", "")
                        dir_type = "d_selfVother_conf"
                    elif source.startswith("self_vs_other_confidence_"):
                        # Legacy: self_vs_other_confidence_SimpleMC -> d_selfVother_conf
                        dataset = source.replace("self_vs_other_confidence_", "")
                        dir_type = "d_selfVother_conf"
                    elif source.startswith("contrast_"):
                        # Legacy: contrast_SimpleMC -> d_selfVother_conf
                        dataset = source.replace("contrast_", "")
                        dir_type = "d_selfVother_conf"
                    elif source == "consensus":
                        # consensus + d_self_confidence -> d_self_confidence_consensus
                        dataset = "consensus"
                        dir_type = direction_name
                    elif source.startswith("d_self_confidence_"):
                        # d_self_confidence_SimpleMC + mean_diff -> d_self_confidence
                        dataset = source.replace("d_self_confidence_", "")
                        dir_type = "d_self_confidence"
                    elif source.startswith("d_other_confidence_"):
                        # d_other_confidence_SimpleMC + mean_diff -> d_other_confidence
                        dataset = source.replace("d_other_confidence_", "")
                        dir_type = "d_other_confidence"
                    else:
                        # Fallback: use source and direction_name as-is
                        dataset = ""
                        dir_type = f"{source}_{direction_name}" if direction_name not in source else source

                    if dataset:
                        filename = f"{output_prefix}_logit_lens_{dir_type}_{dataset}.png"
                    else:
                        filename = f"{output_prefix}_logit_lens_{dir_type}.png"

                    plot_logit_lens_heatmap(
                        all_directions, all_layers, source, direction_name,
                        lm_head_weight, tokenizer,
                        Path(filename),
                        top_k=TOP_K_TOKENS,
                        norm_weight=norm_weight
                    )

    print("\n" + "="*80)
    print("DIRECTION ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
