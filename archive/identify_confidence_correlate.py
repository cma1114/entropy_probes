"""
Identify meta-judgment confidence directions from meta-task activations.

Trains regression probes to predict stated confidence from meta-task activations.
This finds directions that encode the model's expressed confidence, which may
differ from directions that encode actual uncertainty.

Loads from test_meta_transfer.py outputs:
- {model}_{dataset}_meta_{META_TASK}_activations.npz: Meta task activations
  Contains: layer activations, stated confidence values, uncertainty metrics

Outputs:
- {model}_{dataset}_{META_TASK}_confidence_directions.npz: Direction vectors per layer
- {model}_{dataset}_{META_TASK}_confidence_results.json: R² metrics per layer
- {model}_{dataset}_{META_TASK}_confidence_results.png: Layer-wise R² plot

Configuration is set at the top of the script - no CLI args needed.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

from core import get_model_short_name
from core.confidence_directions import (
    find_confidence_directions_both_methods,
    compare_confidence_to_uncertainty,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base name for input files from test_meta_transfer.py
# Will load: {INPUT_BASE_NAME}_meta_{META_TASK}_activations.npz
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC_difficulty_filtered"

# Which meta-task to analyze
# Options: "delegate", "confidence", "other_confidence"
META_TASK = "delegate"

# Train/test split (should match other scripts for consistency)
TRAIN_SPLIT = 0.8
SEED = 42

# Probe parameters
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100
MEAN_DIFF_QUANTILE = 0.25
N_BOOTSTRAP = 100  # Bootstrap iterations for confidence intervals

# If set to a metric name (e.g., "logit_gap"), loads uncertainty directions
# from {INPUT_BASE_NAME}_mc_{metric}_directions.npz and computes cosine
# similarity between confidence directions and uncertainty directions at
# each layer. Set to None to skip.
COMPARE_UNCERTAINTY_METRIC = "logit_gap"

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_r2_results(results: dict, comparison: dict = None, output_path: Path = None):
    """Plot R² across layers for both methods with confidence intervals and optional uncertainty comparison."""
    fits = results["fits"]
    methods = list(fits.keys())
    layers = sorted(fits[methods[0]].keys())

    colors = {"probe": "tab:blue", "mean_diff": "tab:orange"}

    n_panels = 3 if comparison else 2
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    # Panel 1: Both methods comparison
    ax1 = axes[0]
    for method in methods:
        method_fits = fits[method]
        test_r2 = [method_fits[l]["test_r2"] for l in layers]

        # Check for bootstrap CIs (percentile-based)
        has_ci = "test_r2_ci_low" in method_fits[layers[0]]
        if has_ci:
            ci_low = [method_fits[l]["test_r2_ci_low"] for l in layers]
            ci_high = [method_fits[l]["test_r2_ci_high"] for l in layers]
            color = colors.get(method, "tab:gray")
            ax1.fill_between(layers, ci_low, ci_high, alpha=0.2, color=color)

        best_layer = max(layers, key=lambda l: method_fits[l]["test_r2"])
        best_r2 = method_fits[best_layer]["test_r2"]
        ci_str = ""
        if has_ci:
            ci_str = f" [{method_fits[best_layer]['test_r2_ci_low']:.3f}, {method_fits[best_layer]['test_r2_ci_high']:.3f}]"

        color = colors.get(method, "tab:gray")
        ax1.plot(layers, test_r2, 'o-', label=f'{method} (best: L{best_layer}, {best_r2:.3f}{ci_str})',
                 color=color, markersize=4)

    ax1.axhline(y=0, color='red', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Test R²')
    ax1.set_title('Confidence Prediction by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: Cosine similarity between methods
    ax2 = axes[1]
    if "comparison" in results:
        cos_sims = [results["comparison"][l]["cosine_sim"] for l in layers]
        ax2.plot(layers, cos_sims, 'o-', color='tab:purple', markersize=4)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        mean_cos = np.mean(cos_sims)
        ax2.set_title(f'Probe vs Mean-Diff Direction Similarity\n(mean cosine = {mean_cos:.3f})')
    else:
        ax2.set_title('Direction Comparison (not available)')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    # Panel 3: Cosine similarity with uncertainty (if available)
    if comparison:
        ax3 = axes[2]
        cos_sim = [comparison[l]["cosine_similarity"] for l in layers]
        abs_cos = [comparison[l]["abs_cosine_similarity"] for l in layers]

        ax3.plot(layers, cos_sim, 'o-', label='Cosine Similarity', color='tab:green', markersize=4)
        ax3.plot(layers, abs_cos, '--', label='|Cosine Similarity|', color='tab:green', alpha=0.5)
        ax3.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax3.axhline(y=1, color='gray', linestyle=':', alpha=0.3)
        ax3.axhline(y=-1, color='gray', linestyle=':', alpha=0.3)

        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_title('Confidence vs Uncertainty Direction Similarity')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary(results: dict, comparison: dict = None, num_layers: int = None):
    """Print summary statistics for both methods."""
    fits = results["fits"]
    methods = list(fits.keys())

    print("\n" + "=" * 60)
    print("CONFIDENCE DIRECTION SUMMARY")
    print("=" * 60)

    for method in methods:
        method_fits = fits[method]
        layers = sorted(method_fits.keys())

        # Best layer
        best_layer = max(layers, key=lambda l: method_fits[l]["test_r2"])
        best_r2 = method_fits[best_layer]["test_r2"]
        best_pearson = method_fits[best_layer]["test_pearson"]

        # Check for CI
        has_ci = "test_r2_ci_low" in method_fits[best_layer]
        ci_str = ""
        if has_ci:
            ci_str = f" [{method_fits[best_layer]['test_r2_ci_low']:.3f}, {method_fits[best_layer]['test_r2_ci_high']:.3f}]"

        print(f"\n{method.upper()} METHOD:")
        print(f"  Best layer: {best_layer}")
        print(f"  Train R²: {method_fits[best_layer]['train_r2']:.3f}")
        print(f"  Test R²: {best_r2:.3f}{ci_str}")
        print(f"  Test Pearson: {best_pearson:.3f}")
        if "shuffled_r2" in method_fits[best_layer]:
            print(f"  Shuffled R²: {method_fits[best_layer]['shuffled_r2']:.3f}")

        # Early vs late comparison
        if num_layers:
            n_layers = len(layers)
            early_layers = layers[:n_layers // 4]
            late_layers = layers[3 * n_layers // 4:]

            early_r2 = np.mean([method_fits[l]["test_r2"] for l in early_layers])
            late_r2 = np.mean([method_fits[l]["test_r2"] for l in late_layers])

            print(f"  Early layers: {early_r2:.3f}, Late layers: {late_r2:.3f}")

    # Method comparison
    if "comparison" in results:
        layers = sorted(results["comparison"].keys())
        cos_sims = [results["comparison"][l]["cosine_sim"] for l in layers]
        print(f"\nMETHOD COMPARISON:")
        print(f"  Mean cosine similarity: {np.mean(cos_sims):.3f}")
        print(f"  Max |cosine|: {np.max(np.abs(cos_sims)):.3f}")

    # Comparison with uncertainty direction
    if comparison:
        # Use probe method for comparison (primary)
        probe_fits = fits["probe"]
        layers = sorted(probe_fits.keys())
        best_layer = max(layers, key=lambda l: probe_fits[l]["test_r2"])
        best_cos = comparison[best_layer]["cosine_similarity"]
        print(f"\nUNCERTAINTY DIRECTION COMPARISON (at probe's best layer {best_layer}):")
        print(f"  Cosine similarity: {best_cos:.3f}")
        print(f"  Interpretation: {'aligned' if best_cos > 0.5 else 'weakly aligned' if best_cos > 0.2 else 'orthogonal' if abs(best_cos) < 0.2 else 'anti-aligned'}")

        # Find layer with max absolute similarity
        max_sim_layer = max(layers, key=lambda l: comparison[l]["abs_cosine_similarity"])
        max_sim = comparison[max_sim_layer]["cosine_similarity"]
        print(f"  Max |similarity| at layer {max_sim_layer}: {max_sim:.3f}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Input base: {INPUT_BASE_NAME}")
    print(f"Meta task: {META_TASK}")
    print(f"Train/test split: {TRAIN_SPLIT}")
    print(f"Probe alpha: {PROBE_ALPHA}")
    print(f"PCA components: {PROBE_PCA_COMPONENTS}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    if COMPARE_UNCERTAINTY_METRIC:
        print(f"Compare to uncertainty: {COMPARE_UNCERTAINTY_METRIC}")
    print()

    # Load meta-task activations
    activations_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_meta_{META_TASK}_activations.npz"
    print(f"Loading meta-task activations from {activations_path}...")
    if not activations_path.exists():
        raise FileNotFoundError(
            f"Activations file not found: {activations_path}\n"
            f"Run test_meta_transfer.py with META_TASK='{META_TASK}' first."
        )

    act_data = np.load(activations_path)

    # Detect format: multi-position (layer_N_posname) vs legacy (layer_N)
    layer_keys = [k for k in act_data.keys() if k.startswith("layer_")]
    has_positions = any("_" in k.replace("layer_", "", 1) for k in layer_keys)

    meta_activations_by_layer = {}
    if has_positions:
        # Multi-position format: extract layers for "final" position (last token)
        # Keys look like: layer_0_final, layer_1_final, ...
        position_to_use = "final"
        position_keys = [k for k in layer_keys if k.endswith(f"_{position_to_use}")]
        if not position_keys:
            # Fall back to first position found
            positions = set()
            for k in layer_keys:
                parts = k.split("_")
                if len(parts) >= 3:
                    positions.add("_".join(parts[2:]))
            position_to_use = sorted(positions)[0] if positions else None
            position_keys = [k for k in layer_keys if k.endswith(f"_{position_to_use}")]
            print(f"  Warning: 'final' position not found, using '{position_to_use}'")

        num_layers = len(position_keys)
        print(f"  Found {num_layers} layers (position: {position_to_use})")
        for i in range(num_layers):
            meta_activations_by_layer[i] = act_data[f"layer_{i}_{position_to_use}"]
    else:
        # Legacy format: layer_0, layer_1, ...
        num_layers = len(layer_keys)
        print(f"  Found {num_layers} layers")
        for i in range(num_layers):
            meta_activations_by_layer[i] = act_data[f"layer_{i}"]

    n_samples = meta_activations_by_layer[0].shape[0]
    print(f"  Shape per layer: {meta_activations_by_layer[0].shape}")

    # Load stated confidence values
    if "stated_confidence" in act_data:
        stated_confidences = act_data["stated_confidence"]
    elif "confidences" in act_data:
        stated_confidences = act_data["confidences"]
    else:
        # Try to find P_answer for delegate task
        confidence_keys = [k for k in act_data.keys() if "confidence" in k.lower() or "p_answer" in k.lower()]
        if confidence_keys:
            stated_confidences = act_data[confidence_keys[0]]
            print(f"  Using confidence key: {confidence_keys[0]}")
        else:
            raise KeyError(
                f"No confidence values found in {activations_path}\n"
                f"Available keys: {list(act_data.keys())}"
            )

    print(f"  Stated confidence: mean={stated_confidences.mean():.3f}, std={stated_confidences.std():.3f}")

    # Create train/test split
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED,
        shuffle=True
    )
    print(f"\nTrain/test split: {len(train_idx)}/{len(test_idx)}")

    # Find confidence directions using both methods
    print(f"\nTraining confidence probes ({N_BOOTSTRAP} bootstrap iterations)...")
    results = find_confidence_directions_both_methods(
        meta_activations_by_layer,
        stated_confidences,
        train_idx,
        test_idx,
        alpha=PROBE_ALPHA,
        n_components=PROBE_PCA_COMPONENTS,
        mean_diff_quantile=MEAN_DIFF_QUANTILE,
        n_bootstrap=N_BOOTSTRAP,
        train_split=TRAIN_SPLIT,
        seed=SEED,
    )

    # Load uncertainty directions for comparison (if requested)
    comparison = None
    if COMPARE_UNCERTAINTY_METRIC:
        uncertainty_dir_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_{COMPARE_UNCERTAINTY_METRIC}_directions.npz"
        if uncertainty_dir_path.exists():
            print(f"\nLoading uncertainty directions from {uncertainty_dir_path}...")
            unc_data = np.load(uncertainty_dir_path)

            # Load mean_diff directions (usually more robust)
            uncertainty_directions = {}
            for layer in range(num_layers):
                key = f"mean_diff_layer_{layer}"
                if key in unc_data:
                    uncertainty_directions[layer] = unc_data[key]

            if uncertainty_directions:
                print(f"  Comparing to {len(uncertainty_directions)} layers of {COMPARE_UNCERTAINTY_METRIC} directions")
                # Compare using probe directions (primary method)
                comparison = compare_confidence_to_uncertainty(
                    results["directions"]["probe"],
                    uncertainty_directions
                )
            else:
                print(f"  No mean_diff directions found in {uncertainty_dir_path}")
        else:
            print(f"\nWarning: Uncertainty directions not found at {uncertainty_dir_path}")
            print("  Skipping comparison. Run identify_mc_correlate.py first.")

    # Save directions file (both methods, like uncertainty directions)
    directions_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_{META_TASK}_confidence_directions.npz"
    print(f"\nSaving directions to {directions_path}...")

    dir_save = {
        "_metadata_input_base": INPUT_BASE_NAME,
        "_metadata_meta_task": META_TASK,
    }
    for method in ["probe", "mean_diff"]:
        for layer in range(num_layers):
            dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]

    np.savez(directions_path, **dir_save)

    # Save probe objects (for transfer tests)
    probes_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_{META_TASK}_confidence_probes.joblib"
    print(f"Saving probes to {probes_path}...")
    probe_save = {
        "metadata": {
            "input_base": INPUT_BASE_NAME,
            "meta_task": META_TASK,
            "train_split": TRAIN_SPLIT,
            "probe_alpha": PROBE_ALPHA,
            "pca_components": PROBE_PCA_COMPONENTS,
            "seed": SEED,
        },
        "probes": results["probes"],
    }
    joblib.dump(probe_save, probes_path)

    # Save results JSON
    results_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_{META_TASK}_confidence_results.json"
    print(f"Saving results to {results_path}...")

    results_json = {
        "config": {
            "input_base": INPUT_BASE_NAME,
            "meta_task": META_TASK,
            "train_split": TRAIN_SPLIT,
            "probe_alpha": PROBE_ALPHA,
            "pca_components": PROBE_PCA_COMPONENTS,
            "mean_diff_quantile": MEAN_DIFF_QUANTILE,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "stats": {
            "n_samples": n_samples,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "confidence_mean": float(stated_confidences.mean()),
            "confidence_std": float(stated_confidences.std()),
            "confidence_min": float(stated_confidences.min()),
            "confidence_max": float(stated_confidences.max()),
        },
        "results": {},
        "comparison": {},
    }

    # Results per method
    for method in ["probe", "mean_diff"]:
        results_json["results"][method] = {}
        for layer in range(num_layers):
            layer_info = {}
            for k, v in results["fits"][method][layer].items():
                if isinstance(v, np.floating):
                    layer_info[k] = float(v)
                elif isinstance(v, np.integer):
                    layer_info[k] = int(v)
                else:
                    layer_info[k] = v
            results_json["results"][method][layer] = layer_info

    # Method comparison
    for layer in range(num_layers):
        results_json["comparison"][layer] = {
            "cosine_sim": float(results["comparison"][layer]["cosine_sim"])
        }

    # Add uncertainty comparison if available
    if comparison:
        results_json["uncertainty_comparison"] = {
            "metric": COMPARE_UNCERTAINTY_METRIC,
            "by_layer": {
                layer: {k: float(v) for k, v in comp.items()}
                for layer, comp in comparison.items()
            }
        }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # Plot results
    plot_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_{META_TASK}_confidence_results.png"
    print(f"\nPlotting results...")
    plot_r2_results(results, comparison, plot_path)

    # Print summary
    print_summary(results, comparison, num_layers)

    print("\nOutput files:")
    print(f"  {directions_path.name}")
    print(f"  {probes_path.name}")
    print(f"  {results_path.name}")
    print(f"  {plot_path.name}")


if __name__ == "__main__":
    main()
