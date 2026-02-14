"""
Stage 4. Compares uncertainty vs answer vs confidence directions to test whether
D2M transfer is confounded by answer encoding, and whether confidence directions
are aligned with uncertainty directions or specific to self vs other.

Inputs:
    outputs/{base}_mc_{metric}_directions.npz                  Uncertainty directions
    outputs/{base}_mc_answer_directions.npz                    Answer directions
    outputs/{base}_meta_{task}_confidence_directions.npz       Confidence directions

Outputs:
    outputs/{base}_direction_comparison.json    Full comparison metrics
    outputs/{base}_direction_comparison.png     Multi-panel comparison visualization

Run after: identify_mc_correlate.py (with FIND_ANSWER_DIRECTIONS=True),
           test_meta_transfer.py (with FIND_CONFIDENCE_DIRECTIONS=True)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List

from core.plotting import save_figure, GRID_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC_difficulty_filtered"
UNCERTAINTY_METRIC = "logit_gap"  # Which uncertainty metric to compare

# --- Script-specific ---
# Which confidence task(s) to compare
# Options: "delegate", "confidence", "other_confidence"
# If multiple, will compare self vs other
CONFIDENCE_TASKS = ["delegate", "other_confidence"]

# --- Output ---
OUTPUT_DIR = Path(__file__).parent / "outputs"


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================


def load_directions(path: Path, key_prefix: str = "layer_") -> Dict[int, np.ndarray]:
    """Load direction vectors from an npz file."""
    if not path.exists():
        return {}

    data = np.load(path)
    directions = {}

    for key in data.keys():
        if key.startswith(key_prefix):
            try:
                layer = int(key.replace(key_prefix, "").replace("mean_diff_", ""))
                directions[layer] = np.asarray(data[key], dtype=np.float32)
            except ValueError:
                continue

    return directions


def load_uncertainty_directions(base_name: str, metric: str, output_dir: Path) -> Dict[int, np.ndarray]:
    """Load uncertainty directions (mean_diff) from identify_mc_correlate.py output."""
    path = output_dir / f"{base_name}_mc_{metric}_directions.npz"
    if not path.exists():
        print(f"Warning: Uncertainty directions not found: {path}")
        return {}

    data = np.load(path)
    directions = {}

    for key in data.keys():
        if key.startswith("mean_diff_layer_"):
            layer = int(key.replace("mean_diff_layer_", ""))
            directions[layer] = np.asarray(data[key], dtype=np.float32)

    return directions


def load_answer_directions(base_name: str, output_dir: Path) -> Dict[int, np.ndarray]:
    """Load answer directions from identify_mc_answer_correlate.py output."""
    path = output_dir / f"{base_name}_mc_answer_directions.npz"
    return load_directions(path, "layer_")


def load_confidence_directions(base_name: str, meta_task: str, output_dir: Path) -> Dict[int, np.ndarray]:
    """Load confidence directions from identify_confidence_correlate.py output.

    Tries to load mean_diff directions first (matching uncertainty pattern),
    falls back to probe, then legacy format.
    """
    path = output_dir / f"{base_name}_{meta_task}_confidence_directions.npz"
    if not path.exists():
        return {}

    data = np.load(path)
    directions = {}

    # Try mean_diff first (matching uncertainty directions pattern)
    for key in data.keys():
        if key.startswith("mean_diff_layer_"):
            layer = int(key.replace("mean_diff_layer_", ""))
            directions[layer] = np.asarray(data[key], dtype=np.float32)

    if directions:
        return directions

    # Fall back to probe
    for key in data.keys():
        if key.startswith("probe_layer_"):
            layer = int(key.replace("probe_layer_", ""))
            directions[layer] = np.asarray(data[key], dtype=np.float32)

    if directions:
        return directions

    # Fall back to legacy format (layer_N)
    return load_directions(path, "layer_")


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================


def cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute cosine similarity between two direction vectors."""
    n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(d1, d2) / (n1 * n2))


def compare_direction_pair(
    dirs_a: Dict[int, np.ndarray],
    dirs_b: Dict[int, np.ndarray],
    name_a: str = "A",
    name_b: str = "B"
) -> Dict:
    """Compare two sets of directions at each layer."""
    common_layers = sorted(set(dirs_a.keys()) & set(dirs_b.keys()))

    if not common_layers:
        return {"error": f"No common layers between {name_a} and {name_b}"}

    result = {
        "name_a": name_a,
        "name_b": name_b,
        "n_layers": len(common_layers),
        "by_layer": {},
        "summary": {}
    }

    cos_sims = []
    abs_cos_sims = []

    for layer in common_layers:
        cos_sim = cosine_similarity(dirs_a[layer], dirs_b[layer])
        cos_sims.append(cos_sim)
        abs_cos_sims.append(abs(cos_sim))

        result["by_layer"][layer] = {
            "cosine_similarity": cos_sim,
            "abs_cosine_similarity": abs(cos_sim)
        }

    result["summary"] = {
        "mean_cosine": float(np.mean(cos_sims)),
        "std_cosine": float(np.std(cos_sims)),
        "max_abs_cosine": float(np.max(abs_cos_sims)),
        "max_abs_cosine_layer": int(common_layers[np.argmax(abs_cos_sims)]),
        "mean_abs_cosine": float(np.mean(abs_cos_sims)),
    }

    return result


def compare_all_direction_types(
    uncertainty_dirs: Dict[int, np.ndarray],
    answer_dirs: Dict[int, np.ndarray],
    confidence_dirs: Dict[str, Dict[int, np.ndarray]],  # {task_name: {layer: dir}}
) -> Dict:
    """Comprehensive comparison of all direction types."""
    results = {
        "comparisons": {},
        "interpretations": {}
    }

    # 1. Uncertainty vs Answer
    if uncertainty_dirs and answer_dirs:
        results["comparisons"]["uncertainty_vs_answer"] = compare_direction_pair(
            uncertainty_dirs, answer_dirs,
            "uncertainty", "answer"
        )

    # 2. Uncertainty vs each confidence task
    for task, conf_dirs in confidence_dirs.items():
        if uncertainty_dirs and conf_dirs:
            key = f"uncertainty_vs_{task}_confidence"
            results["comparisons"][key] = compare_direction_pair(
                uncertainty_dirs, conf_dirs,
                "uncertainty", f"{task}_confidence"
            )

    # 3. Answer vs each confidence task
    for task, conf_dirs in confidence_dirs.items():
        if answer_dirs and conf_dirs:
            key = f"answer_vs_{task}_confidence"
            results["comparisons"][key] = compare_direction_pair(
                answer_dirs, conf_dirs,
                "answer", f"{task}_confidence"
            )

    # 4. Self vs Other confidence (if both available)
    task_names = list(confidence_dirs.keys())
    if len(task_names) >= 2:
        for i, task_a in enumerate(task_names):
            for task_b in task_names[i+1:]:
                if confidence_dirs[task_a] and confidence_dirs[task_b]:
                    key = f"{task_a}_vs_{task_b}_confidence"
                    results["comparisons"][key] = compare_direction_pair(
                        confidence_dirs[task_a], confidence_dirs[task_b],
                        f"{task_a}_confidence", f"{task_b}_confidence"
                    )

    # Generate interpretations
    results["interpretations"] = generate_interpretations(results["comparisons"])

    return results


def generate_interpretations(comparisons: Dict) -> Dict:
    """Generate human-readable interpretations of the comparisons."""
    interp = {}

    # Uncertainty vs Answer
    if "uncertainty_vs_answer" in comparisons:
        comp = comparisons["uncertainty_vs_answer"]
        if "summary" in comp:
            mean_cos = comp["summary"]["mean_cosine"]
            if abs(mean_cos) > 0.7:
                interp["confound_risk"] = "HIGH: Uncertainty and answer directions are highly correlated. D2M transfer may be confounded by answer encoding."
            elif abs(mean_cos) > 0.4:
                interp["confound_risk"] = "MODERATE: Some correlation between uncertainty and answer directions. Some confounding possible."
            else:
                interp["confound_risk"] = "LOW: Uncertainty and answer directions are relatively orthogonal. D2M transfer is likely not confounded by answer encoding."

    # Self vs Other confidence
    self_other_key = None
    for key in comparisons.keys():
        if "other" in key.lower() and "confidence" in key.lower():
            if "delegate" in key or "confidence" in key.split("_")[0]:
                self_other_key = key
                break

    if self_other_key:
        comp = comparisons[self_other_key]
        if "summary" in comp:
            mean_cos = comp["summary"]["mean_cosine"]
            if abs(mean_cos) > 0.7:
                interp["self_vs_other"] = "HIGH SIMILARITY: Self-confidence and other-confidence directions are similar. Introspection may not be self-specific."
            elif abs(mean_cos) > 0.4:
                interp["self_vs_other"] = "MODERATE SIMILARITY: Some overlap between self and other confidence directions."
            else:
                interp["self_vs_other"] = "LOW SIMILARITY: Self-confidence and other-confidence directions differ. This supports genuine self-introspection."

    # Uncertainty vs Confidence
    for key in comparisons.keys():
        if key.startswith("uncertainty_vs_") and "confidence" in key:
            comp = comparisons[key]
            if "summary" in comp:
                mean_cos = comp["summary"]["mean_cosine"]
                task = key.replace("uncertainty_vs_", "").replace("_confidence", "")
                if abs(mean_cos) > 0.5:
                    interp[f"introspection_{task}"] = f"ALIGNED: Uncertainty and {task} confidence directions are aligned. Supports genuine introspection."
                else:
                    interp[f"introspection_{task}"] = f"ORTHOGONAL: Uncertainty and {task} confidence directions differ. Confidence may be based on different features."

    return interp


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_comparison_results(
    results: Dict,
    num_layers: int,
    output_path: Path
):
    """Create multi-panel comparison visualization."""
    comparisons = results.get("comparisons", {})
    if not comparisons:
        print("No comparisons to plot")
        return

    # Determine layout
    n_comparisons = len(comparisons)
    n_cols = min(3, n_comparisons)
    n_rows = (n_comparisons + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle("Direction Type Comparisons", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))
    colors = {
        "uncertainty_vs_answer": "tab:red",
        "uncertainty_vs_delegate_confidence": "tab:blue",
        "uncertainty_vs_other_confidence": "tab:purple",
        "delegate_vs_other_confidence": "tab:green",
        "answer_vs_delegate_confidence": "tab:orange",
        "answer_vs_other_confidence": "tab:brown",
    }

    for idx, (comp_name, comp_data) in enumerate(comparisons.items()):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        if "error" in comp_data or "by_layer" not in comp_data:
            ax.text(0.5, 0.5, f"No data for\n{comp_name}", ha='center', va='center')
            ax.set_title(comp_name.replace("_", " ").title())
            continue

        # Get cosine similarities
        comp_layers = sorted(comp_data["by_layer"].keys())
        cos_sims = [comp_data["by_layer"][l]["cosine_similarity"] for l in comp_layers]

        color = colors.get(comp_name, "tab:gray")

        # Plot
        ax.plot(comp_layers, cos_sims, 'o-', color=color, markersize=3)
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
        ax.axhline(y=-0.5, color='gray', linestyle='--', alpha=0.3)

        # Mark max absolute similarity
        max_layer = comp_data["summary"]["max_abs_cosine_layer"]
        max_val = comp_data["by_layer"][max_layer]["cosine_similarity"]
        ax.scatter([max_layer], [max_val], s=100, c=color, marker='*', zorder=5)

        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine Similarity")
        ax.set_ylim(-1.1, 1.1)

        # Title with summary stats
        mean_cos = comp_data["summary"]["mean_cosine"]
        max_abs = comp_data["summary"]["max_abs_cosine"]
        title = comp_name.replace("_", " ").title()
        ax.set_title(f"{title}\nmean={mean_cos:.3f}, max|cos|={max_abs:.3f} (L{max_layer})", fontsize=9)

        ax.grid(True, alpha=GRID_ALPHA)

    # Hide empty subplots
    for idx in range(n_comparisons, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].axis('off')

    save_figure(fig, output_path)


def print_summary(results: Dict):
    """Print summary of comparisons."""
    print("\n" + "=" * 70)
    print("DIRECTION TYPE COMPARISON SUMMARY")
    print("=" * 70)

    comparisons = results.get("comparisons", {})
    for comp_name, comp_data in comparisons.items():
        if "error" in comp_data:
            print(f"\n{comp_name}: {comp_data['error']}")
            continue

        if "summary" not in comp_data:
            continue

        summary = comp_data["summary"]
        print(f"\n{comp_name.replace('_', ' ').upper()}:")
        # Bootstrap CI on mean |cosine| by resampling per-layer values
        by_layer = comp_data.get("by_layer", {})
        if by_layer:
            abs_cos_values = np.array([v["abs_cosine_similarity"] for v in by_layer.values()])
            rng = np.random.RandomState(42)
            boot_means = [np.mean(rng.choice(abs_cos_values, size=len(abs_cos_values), replace=True))
                          for _ in range(2000)]
            ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
            print(f"  Mean |cosine|: {summary['mean_abs_cosine']:.3f} [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]")
        else:
            print(f"  Mean cosine similarity: {summary['mean_cosine']:.3f} (std={summary['std_cosine']:.3f})")
        print(f"  Max |cosine|: {summary['max_abs_cosine']:.3f} at layer {summary['max_abs_cosine_layer']}")

    # Interpretations
    interpretations = results.get("interpretations", {})
    if interpretations:
        print("\n" + "-" * 70)
        print("INTERPRETATIONS:")
        print("-" * 70)
        for key, interp in interpretations.items():
            print(f"\n{key}:")
            print(f"  {interp}")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Input base: {INPUT_BASE_NAME}")
    print(f"Uncertainty metric: {UNCERTAINTY_METRIC}")
    print(f"Confidence tasks: {CONFIDENCE_TASKS}")
    print()

    # Load all direction types
    print("Loading directions...")

    uncertainty_dirs = load_uncertainty_directions(
        INPUT_BASE_NAME, UNCERTAINTY_METRIC, OUTPUT_DIR
    )
    print(f"  Uncertainty ({UNCERTAINTY_METRIC}): {len(uncertainty_dirs)} layers")

    answer_dirs = load_answer_directions(INPUT_BASE_NAME, OUTPUT_DIR)
    print(f"  Answer: {len(answer_dirs)} layers")

    confidence_dirs = {}
    for task in CONFIDENCE_TASKS:
        dirs = load_confidence_directions(INPUT_BASE_NAME, task, OUTPUT_DIR)
        if dirs:
            confidence_dirs[task] = dirs
        print(f"  Confidence ({task}): {len(dirs)} layers")

    # Determine number of layers
    all_dir_counts = [len(d) for d in [uncertainty_dirs, answer_dirs] + list(confidence_dirs.values()) if d]
    if not all_dir_counts:
        print("\nError: No directions loaded. Run identification scripts first.")
        return

    num_layers = max(all_dir_counts)
    print(f"\nAnalyzing up to {num_layers} layers")

    # Compare all direction types
    print("\nComparing direction types...")
    results = compare_all_direction_types(
        uncertainty_dirs,
        answer_dirs,
        confidence_dirs
    )

    # Add metadata
    results["config"] = {
        "input_base": INPUT_BASE_NAME,
        "uncertainty_metric": UNCERTAINTY_METRIC,
        "confidence_tasks": CONFIDENCE_TASKS,
        "num_layers": num_layers,
    }

    # Save results
    results_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_direction_comparison.json"
    print(f"\nSaving results to {results_path}...")

    # Convert numpy types for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    # Plot results
    plot_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_direction_comparison.png"
    print(f"Plotting comparison...")
    plot_comparison_results(results, num_layers, plot_path)

    # Print summary
    print_summary(results)

    print("\nOutput files:")
    print(f"  {results_path.name}")
    print(f"  {plot_path.name}")


if __name__ == "__main__":
    main()
