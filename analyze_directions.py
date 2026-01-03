"""
Analyze and compare probe directions across different experiments.

This script:
1. Loads direction files from various probe experiments:
   - Next-token uncertainty probes (entropy, top_prob, margin, logit_gap, top_logit)
   - MC uncertainty probes (entropy, top_prob, margin, logit_gap, top_logit)
   - Introspection probes (trained on direct MC prompts, tested on meta prompts)
   - Contrastive directions
2. Computes pairwise cosine similarities between directions
3. Runs logit lens analysis (project directions through unembedding)
4. Generates visualizations

Direction types and their relationships:
- mc_{metric}_{dataset}: Trained on MC questions to predict uncertainty metric
- introspection_{metric}_{dataset}: Also trained on MC questions (direct prompts) to
  predict the same uncertainty metric. These should be very similar to mc directions
  for the same dataset/metric. The introspection experiment additionally tests whether
  these directions transfer to meta-cognition prompts ("How confident are you...?").
- nexttoken_{metric}: Trained on diverse next-token prediction to predict uncertainty
- contrastive: Difference between high/low uncertainty activations (not a probe)

Usage:
    python analyze_directions.py                    # Auto-detect directions in outputs/
    python analyze_directions.py --model-only       # Only load model, skip analysis (for debugging)
    python analyze_directions.py --layer 15         # Focus on specific layer
    python analyze_directions.py --skip-logit-lens  # Skip logit lens (faster, no model needed)
    python analyze_directions.py --metric entropy   # Only analyze entropy directions
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

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = BASE_MODEL_NAME

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Analysis config
TOP_K_TOKENS = 12  # Number of top tokens to show in logit lens
LAYERS_TO_ANALYZE = None  # None = all layers, or list like [10, 15, 20]

# Available uncertainty metrics (same as in probe scripts)
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}")
    return str(OUTPUTS_DIR / f"{model_short}")


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


def find_direction_files(output_dir: Path, model_short: str, metric_filter: Optional[str] = None) -> Dict[str, Path]:
    """
    Find all direction files for a given model.

    Args:
        output_dir: Directory to search
        model_short: Short model name for pattern matching
        metric_filter: If specified, only include files for this metric

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

    # Contrastive directions: {model}_{dataset}_{metric}_contrastive_directions.npz
    for metric in AVAILABLE_METRICS:
        if metric_filter and metric != metric_filter:
            continue

        contrastive_pattern = f"{model_short}*_{metric}_contrastive_directions.npz"
        for path in output_dir.glob(contrastive_pattern):
            dataset = extract_dataset_from_npz(path)
            if dataset is None:
                # Try to extract from filename: {model}_{dataset}_{metric}_contrastive_directions.npz
                # Remove prefix and suffix to get dataset
                name = path.name
                prefix = f"{model_short}_"
                suffix = f"_{metric}_contrastive_directions.npz"
                if name.startswith(prefix) and name.endswith(suffix):
                    dataset = name[len(prefix):-len(suffix)]
                    # Handle adapter names in prefix
                    if "_adapter-" in dataset:
                        # Format: adapter-{adapter}_{dataset}
                        parts = dataset.split("_", 1)
                        if len(parts) > 1:
                            dataset = parts[1]

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

    return direction_files


def load_directions(path: Path) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Load directions from a .npz file.

    Args:
        path: Path to .npz file

    Returns dict mapping layer_idx -> {direction_name: direction_vector}
    """
    data = np.load(path)

    directions = defaultdict(dict)

    for key in data.files:
        # Parse key format: "layer_N_name" or "layer_N"
        parts = key.split("_")
        if parts[0] == "layer" and len(parts) >= 2:
            layer_idx = int(parts[1])
            if len(parts) > 2:
                direction_name = "_".join(parts[2:])
            else:
                # For files with just "layer_N" keys, use "probe" as direction name
                # The source_name (like "introspection_logit_gap_TriviaMC") already
                # encodes the full context, so we just need a short direction name
                direction_name = "probe"
            directions[layer_idx][direction_name] = data[key]

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
    Rows = layers, Columns = top-k tokens
    Cell values = softmax probabilities, annotations = token strings
    """
    token_data = []
    probs_data = []

    for layer_idx in layers:
        if direction_source in all_directions and layer_idx in all_directions[direction_source]:
            if direction_name in all_directions[direction_source][layer_idx]:
                direction = all_directions[direction_source][layer_idx][direction_name]
                tokens, probs = logit_lens_for_layer(direction, lm_head_weight, tokenizer, top_k, norm_weight)
                token_data.append(tokens)
                probs_data.append(probs)
            else:
                token_data.append([''] * top_k)
                probs_data.append([0.0] * top_k)
        else:
            token_data.append([''] * top_k)
            probs_data.append([0.0] * top_k)

    if not probs_data:
        print(f"No data for {direction_source}/{direction_name}")
        return

    # Convert to arrays
    probs_array = np.array(probs_data)
    token_labels = np.array(token_data)

    # Clean token labels for display
    cleaned_token_labels = np.vectorize(clean_token_str)(token_labels)

    # Plot
    plt.rcParams['font.family'] = 'DejaVu Sans'
    fig_height = max(8, len(layers) * 0.25)
    fig, ax = plt.subplots(figsize=(14, fig_height))

    sns.heatmap(
        probs_array,
        annot=cleaned_token_labels,
        fmt='',
        cmap="Reds",
        xticklabels=False,
        yticklabels=[f"L{l}" for l in layers],
        ax=ax,
        cbar_kws={'label': 'Softmax Probability'}
    )

    ax.set_title(f"Logit Lens: {direction_source}/{direction_name}\n(Top {top_k} tokens per layer)")
    ax.set_xlabel(f"Top {top_k} Tokens")
    ax.set_ylabel("Layer")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved logit lens heatmap to {output_path}")
    plt.close()


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
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved similarity-across-layers plot to {output_path}")
    plt.close()


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

    # Find direction files
    model_short = get_model_short_name(BASE_MODEL_NAME)
    direction_files = find_direction_files(OUTPUTS_DIR, model_short, metric_filter=args.metric)

    if not direction_files:
        print(f"No direction files found in {OUTPUTS_DIR} for model {model_short}")
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

        print(f"\nLoading tokenizer: {BASE_MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_NAME,
            token=os.environ.get("HF_TOKEN")
        )

        if args.model_only:
            print("Tokenizer loaded. Exiting (--model-only specified)")
            return

        # Load lm_head weight and norm weight directly (much faster than loading full model)
        print(f"\nLoading lm_head and norm weights for logit lens...")
        lm_head_weight, norm_weight = load_lm_head_and_norm(BASE_MODEL_NAME)
    else:
        print("\nSkipping logit lens analysis (--skip-logit-lens)")

    # Run analysis
    output_prefix = get_output_prefix()
    all_results = {}

    for layer_idx in tqdm(layers_to_analyze, desc="Analyzing layers"):
        results = analyze_layer(all_directions, layer_idx, lm_head_weight, tokenizer, TOP_K_TOKENS, norm_weight)
        all_results[layer_idx] = results

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

    with open(results_path, "w") as f:
        json.dump(convert_for_json(all_results), f, indent=2)
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
                    # Avoid redundant suffixes (e.g., nexttoken_entropy_entropy)
                    # Also skip generic "probe" suffix - source already identifies it
                    if direction_name in source or source.endswith(f"_{direction_name}") or direction_name == "probe":
                        filename = f"{output_prefix}_logit_lens_{source}.png"
                    else:
                        filename = f"{output_prefix}_logit_lens_{source}_{direction_name}.png"
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
