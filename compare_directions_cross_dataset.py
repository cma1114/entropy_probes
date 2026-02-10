"""
Cross-dataset comparison of all direction types.

Computes:
1. Pairwise cosine similarities between datasets for each direction type
2. Consensus vectors: average direction across datasets (captures commonality)
3. Per-dataset alignment to consensus (how much each dataset shares the common signal)

Direction Types:
    d_self_confidence         - predicts self-confidence (from meta_confidence)
    d_other_confidence        - predicts other-confidence (from meta_other_confidence)
    d_self_confidence_unique      - d_self_confidence with d_other_confidence projected out
    d_other_confidence_unique     - d_other_confidence with d_self_confidence projected out
    d_selfVother_conf             - mean(self_activation - other_activation) for paired questions
    d_mc_{metric}                 - predicts MC answer metric (from identify_mc_correlate.py, e.g. logit_gap, entropy)
    d_meta_mc_uncert              - predicts MC uncertainty from meta-task activations (metamcuncert)

Inputs:
    outputs/{model}_{dataset}_meta_confidence_metaconfdir_directions.npz     (d_self_confidence)
    outputs/{model}_{dataset}_meta_other_confidence_metaconfdir_directions.npz  (d_other_confidence)
    outputs/{model}_{dataset}_orthogonal_directions.npz   (d_self_confidence_unique, d_other_confidence_unique)
    outputs/{model}_{dataset}_selfVother_conf_directions.npz  (d_selfVother_conf)
    outputs/{model}_{dataset}_mc_{metric}_directions.npz  (d_mc_{metric})
    outputs/{model}_{dataset}_meta_confidence_metamcuncert_directions.npz  (d_meta_mc_uncert)

Outputs:
    outputs/{model}_cross_dataset_comparison.json
    outputs/{model}_cross_dataset_similarity.png
    outputs/{model}_consensus_directions.npz

Run after:
    test_meta_transfer.py with META_TASK="confidence" and FIND_CONFIDENCE_DIRECTIONS=True
    test_meta_transfer.py with META_TASK="other_confidence" and FIND_CONFIDENCE_DIRECTIONS=True
    compute_orthogonal_directions.py
    compute_contrast_direction.py
"""

import numpy as np
from pathlib import Path
from itertools import combinations
import json
import matplotlib.pyplot as plt
import seaborn as sns

from core.model_utils import get_model_short_name

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_SHORT = "Llama-3.1-8B-Instruct"
ADAPTER = None #"Tristan-Day/ect_20251222_215412_v0uei7y1_2000"###

# Datasets to compare
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]

OUTPUT_DIR = Path("outputs")


def get_output_prefix() -> str:
    """Get output filename prefix, including adapter if configured."""
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        return f"{MODEL_SHORT}_adapter-{adapter_short}"
    return MODEL_SHORT


def get_base_name(dataset: str) -> str:
    """Get base name for a dataset, including adapter if configured."""
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        return f"{MODEL_SHORT}_adapter-{adapter_short}_{dataset}"
    return f"{MODEL_SHORT}_{dataset}"

# All direction types
# MC_METRIC should match the METRICS setting in identify_mc_correlate.py
MC_METRIC = "logit_gap"  # or "entropy", "top_prob", etc.
DIRECTION_TYPES = [
    "d_self_confidence",
    "d_other_confidence",
    "d_self_confidence_unique",
    "d_other_confidence_unique",
    "d_selfVother_conf",
    f"d_mc_{MC_METRIC}",
    "d_meta_mc_uncert",  # metamcuncert: meta activations → MC uncertainty
]


# =============================================================================
# FUNCTIONS
# =============================================================================

def normalize(v: np.ndarray) -> np.ndarray:
    """Normalize vector to unit length."""
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v)
    if norm > 0:
        v = v / norm
    return v


def load_all_directions(base_name: str) -> dict[str, dict[int, np.ndarray]]:
    """
    Load all direction types from their respective source files.

    Args:
        base_name: Full base name including model and adapter if applicable
                   (e.g., "Llama-3.1-8B-Instruct_TriviaMC_difficulty_filtered")

    Returns:
        {"d_self_confidence": {layer: vec}, "d_other_confidence": {layer: vec}, ...}
    """
    directions = {dt: {} for dt in DIRECTION_TYPES}

    # d_self_confidence from meta_confidence directions
    self_path = OUTPUT_DIR / f"{base_name}_meta_confidence_metaconfdir_directions.npz"
    if self_path.exists():
        data = np.load(self_path)
        for key in data.files:
            if key.startswith("mean_diff_layer_"):
                layer = int(key.replace("mean_diff_layer_", ""))
                directions["d_self_confidence"][layer] = normalize(data[key])

    # d_other_confidence from meta_other_confidence directions
    other_path = OUTPUT_DIR / f"{base_name}_meta_other_confidence_metaconfdir_directions.npz"
    if other_path.exists():
        data = np.load(other_path)
        for key in data.files:
            if key.startswith("mean_diff_layer_"):
                layer = int(key.replace("mean_diff_layer_", ""))
                directions["d_other_confidence"][layer] = normalize(data[key])

    # d_self_confidence_unique, d_other_confidence_unique from orthogonal directions
    ortho_path = OUTPUT_DIR / f"{base_name}_orthogonal_directions.npz"
    if ortho_path.exists():
        data = np.load(ortho_path)
        for key in data.files:
            # New format
            if key.startswith("self_confidence_unique_layer_"):
                layer = int(key.replace("self_confidence_unique_layer_", ""))
                directions["d_self_confidence_unique"][layer] = normalize(data[key])
            elif key.startswith("other_confidence_unique_layer_"):
                layer = int(key.replace("other_confidence_unique_layer_", ""))
                directions["d_other_confidence_unique"][layer] = normalize(data[key])
            # Legacy format (for backward compatibility)
            elif key.startswith("introspection_layer_"):
                layer = int(key.replace("introspection_layer_", ""))
                if layer not in directions["d_self_confidence_unique"]:
                    directions["d_self_confidence_unique"][layer] = normalize(data[key])
            elif key.startswith("surface_layer_"):
                layer = int(key.replace("surface_layer_", ""))
                if layer not in directions["d_other_confidence_unique"]:
                    directions["d_other_confidence_unique"][layer] = normalize(data[key])

    # d_selfVother_conf from selfVother_conf directions
    svo_path = OUTPUT_DIR / f"{base_name}_selfVother_conf_directions.npz"
    if svo_path.exists():
        data = np.load(svo_path)
        for key in data.files:
            if key.startswith("selfVother_conf_layer_"):
                layer = int(key.replace("selfVother_conf_layer_", ""))
                directions["d_selfVother_conf"][layer] = normalize(data[key])
    else:
        # Legacy: try old self_vs_other_confidence or contrast format
        for legacy_suffix, legacy_prefix in [
            ("_self_vs_other_confidence_directions.npz", "self_vs_other_confidence_layer_"),
            ("_contrast_directions.npz", "contrast_layer_"),
        ]:
            legacy_path = OUTPUT_DIR / f"{base_name}{legacy_suffix}"
            if legacy_path.exists():
                data = np.load(legacy_path)
                for key in data.files:
                    if key.startswith(legacy_prefix):
                        layer = int(key.replace(legacy_prefix, ""))
                        directions["d_selfVother_conf"][layer] = normalize(data[key])
                break

    # d_mc_{metric} from mc directions (from identify_mc_correlate.py)
    mc_path = OUTPUT_DIR / f"{base_name}_mc_{MC_METRIC}_directions.npz"
    if mc_path.exists():
        data = np.load(mc_path)
        for key in data.files:
            if key.startswith("mean_diff_layer_"):
                layer = int(key.replace("mean_diff_layer_", ""))
                directions[f"d_mc_{MC_METRIC}"][layer] = normalize(data[key])

    # d_meta_mc_uncert from metamcuncert directions (meta activations → MC uncertainty)
    mmu_path = OUTPUT_DIR / f"{base_name}_meta_confidence_metamcuncert_directions.npz"
    if mmu_path.exists():
        data = np.load(mmu_path)
        for key in data.files:
            if key.startswith("mean_diff_layer_"):
                layer = int(key.replace("mean_diff_layer_", ""))
                directions["d_meta_mc_uncert"][layer] = normalize(data[key])

    # Check we have at least some directions
    has_any = any(len(d) > 0 for d in directions.values())
    if not has_any:
        raise FileNotFoundError(f"No direction files found for {base_name}")

    return directions


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def compute_pairwise_similarities(
    all_directions: dict[str, dict[str, dict[int, np.ndarray]]]
) -> dict:
    """
    Compute pairwise cosine similarities between all dataset pairs.

    Args:
        all_directions: {dataset: {direction_type: {layer: vec}}}

    Returns:
        {
            (dataset1, dataset2): {
                direction_type: {layer: cosine_similarity}
            }
        }
    """
    datasets = list(all_directions.keys())
    results = {}

    for ds1, ds2 in combinations(datasets, 2):
        results[(ds1, ds2)] = {}

        for dtype in DIRECTION_TYPES:
            results[(ds1, ds2)][dtype] = {}

            # Find common layers for this direction type
            layers1 = set(all_directions[ds1][dtype].keys())
            layers2 = set(all_directions[ds2][dtype].keys())
            common_layers = sorted(layers1 & layers2)

            for layer in common_layers:
                v1 = all_directions[ds1][dtype][layer]
                v2 = all_directions[ds2][dtype][layer]
                results[(ds1, ds2)][dtype][layer] = cosine_similarity(v1, v2)

    return results


def compute_consensus_directions(
    all_directions: dict[str, dict[str, dict[int, np.ndarray]]]
) -> dict[str, dict[int, np.ndarray]]:
    """
    Compute consensus direction by averaging across datasets.

    IMPORTANT: Consensus is computed independently for each direction type.
    This means if one dataset is missing d_contrast, it won't affect
    the consensus for d_self/d_other/etc.

    For each direction type and layer:
    1. Find datasets that have this direction type
    2. Stack their direction vectors
    3. Average and normalize

    Returns:
        {direction_type: {layer: consensus_vector}}
    """
    datasets = list(all_directions.keys())
    consensus = {}

    for dtype in DIRECTION_TYPES:
        # Find datasets that have this direction type
        available = [ds for ds in datasets if all_directions[ds].get(dtype)]
        if len(available) < 2:
            print(f"  Skipping {dtype}: only {len(available)} dataset(s) have it")
            consensus[dtype] = {}
            continue

        # Find layers common to all available datasets FOR THIS DIRECTION TYPE
        common_layers = set(all_directions[available[0]][dtype].keys())
        for ds in available[1:]:
            common_layers &= set(all_directions[ds][dtype].keys())

        if not common_layers:
            print(f"  Skipping {dtype}: no common layers across datasets")
            consensus[dtype] = {}
            continue

        consensus[dtype] = {}
        for layer in sorted(common_layers):
            vecs = np.stack([all_directions[ds][dtype][layer] for ds in available])
            avg = vecs.mean(axis=0)
            consensus[dtype][layer] = (avg / np.linalg.norm(avg)).astype(np.float32)

        print(f"  {dtype}: consensus from {len(available)} datasets, {len(consensus[dtype])} layers")

    return consensus


def compute_alignment_to_consensus(
    all_directions: dict[str, dict[str, dict[int, np.ndarray]]],
    consensus: dict[str, dict[int, np.ndarray]]
) -> dict:
    """
    Compute how well each dataset's directions align with the consensus.

    Returns:
        {dataset: {direction_type: {layer: cosine_to_consensus}}}
    """
    results = {}

    for ds, directions in all_directions.items():
        results[ds] = {}
        for dtype in DIRECTION_TYPES:
            results[ds][dtype] = {}
            if dtype not in consensus or not consensus[dtype]:
                continue
            for layer in consensus[dtype].keys():
                if layer in directions[dtype]:
                    v = directions[dtype][layer]
                    c = consensus[dtype][layer]
                    results[ds][dtype][layer] = cosine_similarity(v, c)

    return results


def plot_cross_dataset_similarity(
    pairwise: dict,
    alignment: dict,
    output_path: Path
):
    """Plot heatmaps of cross-dataset similarity."""
    datasets = list(alignment.keys())
    n_datasets = len(datasets)

    # Filter to direction types that have data
    active_types = [dt for dt in DIRECTION_TYPES
                    if any(dt in alignment[ds] and alignment[ds][dt] for ds in datasets)]

    if not active_types:
        print("No direction types with data for plotting")
        return

    # Get layers from first non-empty direction type
    layers = None
    for dtype in active_types:
        for ds in datasets:
            if dtype in alignment[ds] and alignment[ds][dtype]:
                layers = sorted(alignment[ds][dtype].keys())
                break
        if layers:
            break

    if layers is None:
        print("No layers found for plotting")
        return

    n_types = len(active_types)
    fig, axes = plt.subplots(2, n_types, figsize=(5 * n_types, 10))
    if n_types == 1:
        axes = axes.reshape(2, 1)

    for col, dtype in enumerate(active_types):
        # Top row: pairwise similarity matrix (averaged across layers)
        ax = axes[0, col]

        sim_matrix = np.zeros((n_datasets, n_datasets))
        for i, ds1 in enumerate(datasets):
            for j, ds2 in enumerate(datasets):
                if i == j:
                    sim_matrix[i, j] = 1.0
                elif (ds1, ds2) in pairwise and dtype in pairwise[(ds1, ds2)]:
                    sims = list(pairwise[(ds1, ds2)][dtype].values())
                    sim_matrix[i, j] = np.mean(sims) if sims else 0
                elif (ds2, ds1) in pairwise and dtype in pairwise[(ds2, ds1)]:
                    sims = list(pairwise[(ds2, ds1)][dtype].values())
                    sim_matrix[i, j] = np.mean(sims) if sims else 0

        sns.heatmap(
            sim_matrix,
            annot=True,
            fmt=".2f",
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=[ds[:15] for ds in datasets],
            yticklabels=[ds[:15] for ds in datasets],
            ax=ax
        )
        ax.set_title(f"{dtype}: Pairwise Similarity (mean)")

        # Bottom row: alignment to consensus by layer
        ax = axes[1, col]

        alignment_matrix = np.zeros((n_datasets, len(layers)))
        for i, ds in enumerate(datasets):
            for j, layer in enumerate(layers):
                if dtype in alignment[ds] and layer in alignment[ds][dtype]:
                    alignment_matrix[i, j] = alignment[ds][dtype][layer]

        sns.heatmap(
            alignment_matrix,
            cmap="RdYlGn",
            vmin=-1,
            vmax=1,
            xticklabels=[f"L{l}" for l in layers[::4]],
            yticklabels=[ds[:15] for ds in datasets],
            ax=ax
        )
        ax.set_xticks(np.arange(0, len(layers), 4) + 0.5)
        ax.set_title(f"{dtype}: Alignment to Consensus")
        ax.set_xlabel("Layer")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path.name}")


def main():
    print(f"Model: {MODEL_SHORT}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Datasets: {DATASETS}")
    print(f"Direction types: {DIRECTION_TYPES}")

    output_prefix = get_output_prefix()

    # Load all directions
    print("\nLoading directions...")
    all_directions = {}
    for ds in DATASETS:
        try:
            base_name = get_base_name(ds)
            all_directions[ds] = load_all_directions(base_name)
            counts = {dt: len(all_directions[ds][dt]) for dt in DIRECTION_TYPES}
            present = [f"{dt}={counts[dt]}" for dt in DIRECTION_TYPES if counts[dt] > 0]
            print(f"  {ds}: {', '.join(present)}")
        except FileNotFoundError as e:
            print(f"  {ds}: SKIPPED - {e}")

    if len(all_directions) < 2:
        print("\nNeed at least 2 datasets to compare. Exiting.")
        return

    # Compute pairwise similarities
    print("\nComputing pairwise similarities...")
    pairwise = compute_pairwise_similarities(all_directions)

    for (ds1, ds2), sims in pairwise.items():
        print(f"\n  {ds1} vs {ds2}:")
        for dtype in DIRECTION_TYPES:
            if dtype in sims and sims[dtype]:
                values = list(sims[dtype].values())
                print(f"    {dtype}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

    # Compute consensus directions
    print("\nComputing consensus directions...")
    consensus = compute_consensus_directions(all_directions)

    # Compute alignment to consensus
    print("\nComputing alignment to consensus...")
    alignment = compute_alignment_to_consensus(all_directions, consensus)

    for ds in all_directions.keys():
        print(f"\n  {ds}:")
        for dtype in DIRECTION_TYPES:
            if dtype in alignment[ds] and alignment[ds][dtype]:
                values = list(alignment[ds][dtype].values())
                print(f"    {dtype}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

    # Save consensus directions
    consensus_path = OUTPUT_DIR / f"{output_prefix}_consensus_directions.npz"
    save_data = {
        "_metadata_model": MODEL_SHORT,
        "_metadata_adapter": ADAPTER,
        "_metadata_datasets": json.dumps(DATASETS),
        "_metadata_direction_types": json.dumps(DIRECTION_TYPES),
    }
    for dtype in DIRECTION_TYPES:
        for layer, vec in consensus.get(dtype, {}).items():
            save_data[f"{dtype}_layer_{layer}"] = vec

    np.savez_compressed(consensus_path, **save_data)
    print(f"\nSaved consensus directions: {consensus_path.name}")

    # Save JSON results
    results = {
        "config": {
            "model": MODEL_SHORT,
            "adapter": ADAPTER,
            "datasets": DATASETS,
            "direction_types": DIRECTION_TYPES,
        },
        "pairwise_similarity": {
            f"{ds1}_vs_{ds2}": {
                dtype: {str(l): v for l, v in sims.items()}
                for dtype, sims in type_sims.items()
            }
            for (ds1, ds2), type_sims in pairwise.items()
        },
        "alignment_to_consensus": {
            ds: {
                dtype: {str(l): v for l, v in layer_sims.items()}
                for dtype, layer_sims in type_sims.items()
            }
            for ds, type_sims in alignment.items()
        },
        "summary": {
            "pairwise_mean": {
                f"{ds1}_vs_{ds2}": {
                    dtype: float(np.mean(list(sims.values()))) if sims else None
                    for dtype, sims in type_sims.items()
                }
                for (ds1, ds2), type_sims in pairwise.items()
            },
            "alignment_mean": {
                ds: {
                    dtype: float(np.mean(list(layer_sims.values()))) if layer_sims else None
                    for dtype, layer_sims in type_sims.items()
                }
                for ds, type_sims in alignment.items()
            },
        },
    }

    results_path = OUTPUT_DIR / f"{output_prefix}_cross_dataset_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results: {results_path.name}")

    # Plot
    plot_path = OUTPUT_DIR / f"{output_prefix}_cross_dataset_similarity.png"
    plot_cross_dataset_similarity(pairwise, alignment, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
