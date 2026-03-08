"""
Analyze cosine similarities between key direction vectors.

This script computes pairwise cosine similarities between:
- d_mc: MC uncertainty direction (from identify_mc_correlate.py)
- d_metamcuncert: Uncertainty recomputed on meta-task (from test_meta_transfer.py)
- d_confdir: Confidence direction from meta-task (from test_meta_transfer.py)
- d_mc_answer: Answer centroid from MC task (from identify_mc_correlate.py)
- d_meta_answer: Answer centroid from meta-task (from test_meta_transfer.py)

Uses mean_diff method only for simplicity.

Prerequisites:
    1. identify_mc_correlate.py - produces d_mc and d_mc_answer directions
    2. test_meta_transfer.py with FIND_MC_UNCERTAINTY_DIRECTIONS=True - produces d_metamcuncert
    3. test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True - produces d_confdir
    4. test_meta_transfer.py with FIND_META_MCQ_CLASSIFIER=True - produces d_meta_answer

Inputs:
    outputs/{model_dir}/working/{dataset}_mc_{metric}_directions.npz
    outputs/{model_dir}/working/{dataset}_mc_answer_directions.npz
    outputs/{model_dir}/working/{dataset}_meta_{task}_mcuncert_directions_final.npz
    outputs/{model_dir}/working/{dataset}_meta_{task}_confdir_{target}_directions_final.npz
    outputs/{model_dir}/working/{dataset}_meta_{task}_metamcq_directions_final.npz

Outputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_direction_cosines_{metric}.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_direction_cosines_{metric}.png

Shared parameters: MODEL, LOAD_IN_4BIT, LOAD_IN_8BIT must match across scripts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from core.model_utils import get_model_dir_name
from core.config_utils import get_output_path, find_output_file, get_config_dict
from core.plotting import save_figure, GRID_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

DATASET = "TriviaMC_difficulty_filtered"
META_TASK = "delegate"
METRIC = "logit_gap"
METHOD = "mean_diff"  # Focus on mean_diff only

# Confdir target for filename matching
CONFDIR_TARGET = "logit_margin"  # or "confidences" for P(Answer) target

# =============================================================================
# DIRECTION SPECIFICATIONS
# =============================================================================

@dataclass
class DirectionSpec:
    """Specification for loading a direction type."""
    name: str
    file_pattern: str
    key_pattern: str  # Uses {layer} placeholder
    description: str


def get_direction_specs(dataset: str, task: str, metric: str, confdir_target: str) -> Dict[str, DirectionSpec]:
    """Get specifications for all direction types."""
    return {
        "d_mc": DirectionSpec(
            name="d_mc",
            file_pattern=f"{dataset}_mc_{metric}_directions.npz",
            key_pattern=f"{METHOD}_layer_{{layer}}",
            description="MC uncertainty direction",
        ),
        "d_metamcuncert": DirectionSpec(
            name="d_metamcuncert",
            file_pattern=f"{dataset}_meta_{task}_mcuncert_directions_final.npz",
            key_pattern=f"{METHOD}_{metric}_layer_{{layer}}",
            description="Uncertainty recomputed on meta-task",
        ),
        "d_confdir": DirectionSpec(
            name="d_confdir",
            file_pattern=f"{dataset}_meta_{task}_confdir_{confdir_target}_directions_final.npz",
            key_pattern=f"{METHOD}_layer_{{layer}}",
            description=f"Confidence direction ({confdir_target})",
        ),
        "d_mc_answer": DirectionSpec(
            name="d_mc_answer",
            file_pattern=f"{dataset}_mc_answer_directions.npz",
            key_pattern="centroid_layer_{layer}",
            description="Answer centroid direction (from MC task)",
        ),
        "d_meta_answer": DirectionSpec(
            name="d_meta_answer",
            file_pattern=f"{dataset}_meta_{task}_metamcq_directions_final.npz",
            key_pattern="centroid_layer_{layer}",
            description="Answer centroid direction (from meta-task)",
        ),
    }


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def load_direction(spec: DirectionSpec, model_dir: str) -> Optional[Dict[int, np.ndarray]]:
    """Load direction vectors for all layers.

    Returns:
        Dict mapping layer index to direction vector, or None if file not found.
    """
    path = find_output_file(spec.file_pattern, model_dir=model_dir)
    if not path.exists():
        return None

    data = np.load(path)
    directions = {}
    layer = 0
    while True:
        key = spec.key_pattern.format(layer=layer)
        if key not in data.files:
            break
        directions[layer] = data[key]
        layer += 1

    return directions if directions else None


def load_all_directions(
    dataset: str, task: str, metric: str, confdir_target: str, model_dir: str
) -> Dict[str, Dict[int, np.ndarray]]:
    """Load all direction types.

    Returns:
        Dict mapping direction name to {layer: direction_vector}
    """
    specs = get_direction_specs(dataset, task, metric, confdir_target)
    directions = {}

    for name, spec in specs.items():
        loaded = load_direction(spec, model_dir)
        if loaded is not None:
            directions[name] = loaded

    return directions


# =============================================================================
# COSINE SIMILARITY COMPUTATION
# =============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_pairwise_cosines(
    directions: Dict[str, Dict[int, np.ndarray]],
    pairs: List[Tuple[str, str]],
) -> Dict[str, Dict[int, float]]:
    """Compute cosine similarity per layer for each pair.

    Args:
        directions: {direction_name: {layer: vector}}
        pairs: List of (name1, name2) tuples

    Returns:
        {pair_key: {layer: cosine_sim}} where pair_key is "name1__vs__name2"
    """
    results = {}

    for name1, name2 in pairs:
        if name1 not in directions or name2 not in directions:
            continue

        dir1 = directions[name1]
        dir2 = directions[name2]

        # Find common layers
        common_layers = sorted(set(dir1.keys()) & set(dir2.keys()))
        if not common_layers:
            continue

        pair_key = f"{name1}__vs__{name2}"
        results[pair_key] = {}

        for layer in common_layers:
            cos = cosine_similarity(dir1[layer], dir2[layer])
            results[pair_key][layer] = cos

    return results


def summarize_cosines(cosines: Dict[str, Dict[int, float]]) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics for cosine similarities.

    Returns:
        {pair_key: {mean_abs, max_abs, max_layer, mean_signed}}
    """
    summaries = {}

    for pair_key, per_layer in cosines.items():
        if not per_layer:
            continue

        values = list(per_layer.values())
        abs_values = [abs(v) for v in values]
        layers = list(per_layer.keys())

        max_abs_idx = np.argmax(abs_values)

        summaries[pair_key] = {
            "mean_abs": float(np.mean(abs_values)),
            "max_abs": float(np.max(abs_values)),
            "max_layer": int(layers[max_abs_idx]),
            "mean_signed": float(np.mean(values)),
        }

    return summaries


def interpret_cosine(mean_abs: float) -> str:
    """Provide interpretation of cosine similarity magnitude."""
    if mean_abs >= 0.7:
        return "strong alignment (likely same signal)"
    elif mean_abs >= 0.4:
        return "moderate alignment"
    elif mean_abs >= 0.15:
        return "weak alignment"
    else:
        return "near-orthogonal (different signals)"


# =============================================================================
# PLOTTING
# =============================================================================

PAIR_COLORS = {
    "d_mc__vs__d_metamcuncert": "tab:blue",
    "d_mc__vs__d_confdir": "tab:orange",
    "d_mc__vs__d_mc_answer": "tab:green",
    "d_metamcuncert__vs__d_confdir": "tab:red",
    "d_metamcuncert__vs__d_meta_answer": "tab:cyan",
    "d_mc_answer__vs__d_meta_answer": "tab:purple",
    "d_mc_answer__vs__d_confdir": "tab:brown",
    "d_meta_answer__vs__d_confdir": "tab:pink",
}


def plot_cosines(
    cosines: Dict[str, Dict[int, float]],
    summaries: Dict[str, Dict[str, float]],
    output_path: Path,
):
    """Plot cosine similarity curves across layers."""
    if not cosines:
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    for pair_key, per_layer in cosines.items():
        if not per_layer:
            continue

        layers = sorted(per_layer.keys())
        values = [per_layer[l] for l in layers]

        # Make label more readable
        parts = pair_key.split("__vs__")
        label = f"{parts[0]} vs {parts[1]}"
        mean_abs = summaries[pair_key]["mean_abs"]
        label = f"{label} (|cos|={mean_abs:.3f})"

        color = PAIR_COLORS.get(pair_key, "gray")
        ax.plot(layers, values, label=label, color=color, linewidth=1.5)

    ax.axhline(0, color="black", linestyle=":", alpha=0.3)
    ax.axhline(0.7, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")
    ax.axhline(-0.7, color="gray", linestyle="--", alpha=0.3, label="_nolegend_")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Direction Cosine Similarities Across Layers")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=GRID_ALPHA)

    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)

    print("=" * 80)
    print("Direction Cosine Analysis")
    print("=" * 80)
    print(f"Config: dataset={DATASET}, task={META_TASK}, metric={METRIC}")
    print()

    # Load directions
    directions = load_all_directions(DATASET, META_TASK, METRIC, CONFDIR_TARGET, model_dir)

    if not directions:
        print("ERROR: No direction files found.")
        print("Run identify_mc_correlate.py and test_meta_transfer.py first.")
        return

    print("Loaded directions:")
    for name, dir_dict in directions.items():
        print(f"  {name}: {len(dir_dict)} layers")
    print()

    # Define pairs to compare
    pairs = [
        ("d_mc", "d_metamcuncert"),           # Transfer vs Recomputation (uncertainty)
        ("d_mc", "d_confdir"),                # MC Uncertainty vs Confidence
        ("d_mc", "d_mc_answer"),              # MC Uncertainty vs MC Answer
        ("d_metamcuncert", "d_confdir"),      # Recomputed Uncertainty vs Confidence
        ("d_metamcuncert", "d_meta_answer"),  # Recomputed Uncertainty vs Meta Answer
        ("d_mc_answer", "d_meta_answer"),     # Transfer vs Recomputation (answer)
        ("d_mc_answer", "d_confdir"),         # MC Answer vs Confidence
        ("d_meta_answer", "d_confdir"),       # Meta Answer vs Confidence
    ]

    # Compute cosines
    cosines = compute_pairwise_cosines(directions, pairs)
    summaries = summarize_cosines(cosines)

    if not summaries:
        print("ERROR: Could not compute any cosine similarities.")
        print("Check that direction files have compatible layers.")
        return

    # Print results
    print("Cosine Similarities (mean |cos| across layers):")
    for pair_key, stats in summaries.items():
        parts = pair_key.split("__vs__")
        pair_label = f"{parts[0]} vs {parts[1]}"
        interp = interpret_cosine(stats["mean_abs"])
        print(f"  {pair_label:35s}: {stats['mean_abs']:.3f} -> {interp}")
    print()

    # Save JSON output
    output_json = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            meta_task=META_TASK,
            metric=METRIC,
            method=METHOD,
            confdir_target=CONFDIR_TARGET,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "directions_loaded": list(directions.keys()),
        "comparisons": {},
    }

    for pair_key, per_layer in cosines.items():
        output_json["comparisons"][pair_key] = {
            "summary": summaries[pair_key],
            "interpretation": interpret_cosine(summaries[pair_key]["mean_abs"]),
            "per_layer": {str(k): v for k, v in per_layer.items()},
        }

    json_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_cosines_{METRIC}.json", model_dir=model_dir
    )
    with open(json_path, "w") as f:
        json.dump(output_json, f, indent=2)

    # Plot
    plot_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_cosines_{METRIC}.png", model_dir=model_dir
    )
    plot_cosines(cosines, summaries, plot_path)

    print(f"Output: {json_path.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
