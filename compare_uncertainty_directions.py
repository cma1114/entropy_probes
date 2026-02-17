"""
Compare all output uncertainty directions across datasets and tasks.

Computes a 12×12 cosine similarity matrix comparing:
- MC entropy directions (3): one per dataset
- Meta output uncertainty directions (9): 3 datasets × 3 meta tasks

Inputs:
    outputs/{model_dir}/working/{dataset}_mc_entropy_directions.npz
    outputs/{model_dir}/working/{dataset}_meta_{task}_metauncert_directions_{pos}.npz

Outputs:
    outputs/{model_dir}/results/uncertainty_directions_comparison.json
    outputs/{model_dir}/results/uncertainty_directions_comparison.png

Run after:
    identify_mc_correlate.py (for MC entropy directions)
    test_meta_transfer.py with FIND_META_OUTPUT_UNCERTAINTY_DIRECTIONS=True (for all 3 meta tasks)
"""

import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns

from core.model_utils import get_model_dir_name
from core.config_utils import get_output_path, find_output_file, get_config_dict
from core.plotting import save_figure

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]

# Short names for plot labels
DATASET_SHORT = {
    "TriviaMC_difficulty_filtered": "Trivia",
    "PopMC_0_difficulty_filtered": "Pop",
    "SimpleMC": "Simple",
}

META_TASKS = ["confidence", "other_confidence", "delegate"]
META_TASK_SHORT = {
    "confidence": "self",
    "other_confidence": "other",
    "delegate": "delegate",
}

PROBE_POSITION = "final"
MC_METRIC = "entropy"

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_model_dir() -> str:
    return get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def load_mc_entropy_direction(dataset: str, model_dir: str) -> dict[int, np.ndarray]:
    """Load MC entropy direction for a dataset."""
    directions = {}
    path = find_output_file(f"{dataset}_mc_{MC_METRIC}_directions.npz", model_dir=model_dir)
    if path.exists():
        data = np.load(path)
        for key in data.files:
            if key.startswith("mean_diff_layer_"):
                layer = int(key.replace("mean_diff_layer_", ""))
                directions[layer] = normalize(data[key])
    return directions


def load_meta_uncertainty_direction(dataset: str, meta_task: str, metric: str, model_dir: str) -> dict[int, np.ndarray]:
    """Load meta output uncertainty direction for a dataset, meta task, and metric."""
    directions = {}
    path = find_output_file(f"{dataset}_meta_{meta_task}_metauncert_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
    if path.exists():
        data = np.load(path)
        # New format: mean_diff_{metric}_layer_{layer}
        prefix = f"mean_diff_{metric}_layer_"
        for key in data.files:
            if key.startswith(prefix):
                layer = int(key.replace(prefix, ""))
                directions[layer] = normalize(data[key])
    return directions


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(v1, v2))


def main():
    model_dir = get_model_dir()
    print(f"Model: {MODEL}")
    print(f"Model dir: {model_dir}")
    print(f"Datasets: {DATASETS}")
    print(f"Meta tasks: {META_TASKS}")
    print()

    # Load all directions
    print("Loading directions...")
    all_directions = {}  # {name: {layer: vec}}

    # MC entropy directions
    for dataset in DATASETS:
        name = f"MC_{DATASET_SHORT[dataset]}"
        dirs = load_mc_entropy_direction(dataset, model_dir)
        if dirs:
            all_directions[name] = dirs
            print(f"  {name}: {len(dirs)} layers")
        else:
            print(f"  {name}: NOT FOUND")

    # Meta output uncertainty directions
    for dataset in DATASETS:
        for meta_task in META_TASKS:
            name = f"{META_TASK_SHORT[meta_task]}_{DATASET_SHORT[dataset]}"
            dirs = load_meta_uncertainty_direction(dataset, meta_task, MC_METRIC, model_dir)
            if dirs:
                all_directions[name] = dirs
                print(f"  {name}: {len(dirs)} layers")
            else:
                print(f"  {name}: NOT FOUND")

    if len(all_directions) < 2:
        print("\nNot enough directions found to compare.")
        return

    # Get common layers
    all_layers = set.intersection(*[set(d.keys()) for d in all_directions.values()])
    layers = sorted(all_layers)
    print(f"\nCommon layers: {len(layers)}")

    # Compute similarity matrix at each layer
    names = list(all_directions.keys())
    n = len(names)

    print(f"\nComputing {n}×{n} similarity matrix...")

    # Per-layer similarities
    sim_by_layer = {layer: np.zeros((n, n)) for layer in layers}
    for layer in layers:
        for i, name_i in enumerate(names):
            for j, name_j in enumerate(names):
                v_i = all_directions[name_i][layer]
                v_j = all_directions[name_j][layer]
                sim_by_layer[layer][i, j] = cosine_similarity(v_i, v_j)

    # Average across layers
    sim_mean = np.mean([sim_by_layer[l] for l in layers], axis=0)
    sim_std = np.std([sim_by_layer[l] for l in layers], axis=0)

    # Find best layer (highest average off-diagonal similarity)
    best_layer = None
    best_avg = -1
    for layer in layers:
        mask = ~np.eye(n, dtype=bool)
        avg_sim = sim_by_layer[layer][mask].mean()
        if avg_sim > best_avg:
            best_avg = avg_sim
            best_layer = layer

    print(f"Best layer (highest avg similarity): {best_layer} (avg={best_avg:.3f})")

    # Print summary
    print("\nMean similarity matrix (averaged across layers):")
    print("  " + "  ".join(f"{name:>8}" for name in names))
    for i, name_i in enumerate(names):
        row = "  ".join(f"{sim_mean[i, j]:8.3f}" for j in range(n))
        print(f"{name_i:>12}: {row}")

    # Group analysis: MC vs Meta
    mc_indices = [i for i, name in enumerate(names) if name.startswith("MC_")]
    meta_indices = [i for i, name in enumerate(names) if not name.startswith("MC_")]

    if mc_indices and meta_indices:
        # Within MC
        mc_pairs = [(i, j) for i in mc_indices for j in mc_indices if i < j]
        within_mc = np.mean([sim_mean[i, j] for i, j in mc_pairs]) if mc_pairs else 0

        # Within Meta
        meta_pairs = [(i, j) for i in meta_indices for j in meta_indices if i < j]
        within_meta = np.mean([sim_mean[i, j] for i, j in meta_pairs]) if meta_pairs else 0

        # Between MC and Meta
        between_pairs = [(i, j) for i in mc_indices for j in meta_indices]
        between = np.mean([sim_mean[i, j] for i, j in between_pairs]) if between_pairs else 0

        print(f"\nGroup analysis:")
        print(f"  Within MC directions: {within_mc:.3f}")
        print(f"  Within Meta directions: {within_meta:.3f}")
        print(f"  Between MC and Meta: {between:.3f}")

    # Save results
    results = {
        "config": get_config_dict(
            model=MODEL,
            datasets=DATASETS,
            meta_tasks=META_TASKS,
            mc_metric=MC_METRIC,
            probe_position=PROBE_POSITION,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "direction_names": names,
        "n_layers": len(layers),
        "best_layer": best_layer,
        "similarity_mean": sim_mean.tolist(),
        "similarity_std": sim_std.tolist(),
        "similarity_at_best_layer": sim_by_layer[best_layer].tolist(),
    }

    if mc_indices and meta_indices:
        results["group_analysis"] = {
            "within_mc": float(within_mc),
            "within_meta": float(within_meta),
            "between_mc_meta": float(between),
        }

    results_path = get_output_path("uncertainty_directions_comparison.json", model_dir=model_dir)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Plot heatmap
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Mean across layers
    ax1 = axes[0]
    sns.heatmap(sim_mean, ax=ax1, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                xticklabels=names, yticklabels=names)
    ax1.set_title("Cosine Similarity (mean across layers)")
    ax1.tick_params(axis='x', rotation=45)
    ax1.tick_params(axis='y', rotation=0)

    # Best layer
    ax2 = axes[1]
    sns.heatmap(sim_by_layer[best_layer], ax=ax2, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1,
                xticklabels=names, yticklabels=names)
    ax2.set_title(f"Cosine Similarity (layer {best_layer})")
    ax2.tick_params(axis='x', rotation=45)
    ax2.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plot_path = get_output_path("uncertainty_directions_comparison.png", model_dir=model_dir)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
