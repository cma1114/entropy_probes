"""
Identify MC answer choice directions from direct task activations.

Trains 4-class LogisticRegression to predict which answer (A/B/C/D) the model
selected, then extracts direction vectors via PCA on class coefficients.

This provides a confound control for D2M transfer tests:
- If D2M transfer is just detecting answer encoding, answer directions should
  transfer as well as uncertainty directions
- If genuine introspection exists, uncertainty directions should transfer better

Loads from identify_mc_correlate.py outputs:
- {model}_{dataset}_mc_activations.npz: Direct task activations
- {model}_{dataset}_mc_dataset.json: Questions with model_answer field

Outputs:
- {model}_{dataset}_mc_answer_directions.npz: Direction vectors per layer
- {model}_{dataset}_mc_answer_results.json: Accuracy metrics per layer
- {model}_{dataset}_mc_answer_results.png: Layer-wise accuracy plot

Configuration is set at the top of the script - no CLI args needed.
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from sklearn.model_selection import train_test_split

from core import get_model_short_name
from core.answer_directions import (
    find_answer_directions_both_methods,
    encode_answers,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base name for input files from identify_mc_correlate.py
# Will load: {INPUT_BASE_NAME}_mc_activations.npz and {INPUT_BASE_NAME}_mc_dataset.json
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC_difficulty_filtered"

# Train/test split (should match identify_mc_correlate.py for consistency)
TRAIN_SPLIT = 0.8
SEED = 42

# Classifier parameters
N_PCA_COMPONENTS = 256  # More components for 4-class classification
N_BOOTSTRAP = 100  # Bootstrap iterations for confidence intervals

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_accuracy_results(results: dict, output_path: Path):
    """Plot accuracy across layers for both methods with confidence intervals."""
    fits = results["fits"]
    methods = list(fits.keys())
    layers = sorted(fits[methods[0]].keys())

    colors = {"probe": "tab:blue", "centroid": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Both methods comparison
    ax1 = axes[0]
    for method in methods:
        method_fits = fits[method]
        test_acc = [method_fits[l]["test_accuracy"] for l in layers]

        # Check for bootstrap std
        has_std = "test_accuracy_std" in method_fits[layers[0]]
        if has_std:
            test_acc_std = [method_fits[l]["test_accuracy_std"] for l in layers]
            color = colors.get(method, "tab:gray")
            ax1.fill_between(
                layers,
                np.array(test_acc) - np.array(test_acc_std),
                np.array(test_acc) + np.array(test_acc_std),
                alpha=0.2, color=color
            )

        best_layer = max(layers, key=lambda l: method_fits[l]["test_accuracy"])
        best_acc = method_fits[best_layer]["test_accuracy"]
        best_std_str = ""
        if has_std:
            best_std_str = f" ± {method_fits[best_layer]['test_accuracy_std']:.1%}"

        color = colors.get(method, "tab:gray")
        ax1.plot(layers, test_acc, 'o-', label=f'{method} (best: L{best_layer}, {best_acc:.1%}{best_std_str})',
                 color=color, markersize=4)

    ax1.axhline(y=0.25, color='red', linestyle=':', alpha=0.5, label='Chance (25%)')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('MC Answer Classification by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    # Panel 2: Cosine similarity between methods
    ax2 = axes[1]
    if "comparison" in results:
        cos_sims = [results["comparison"][l]["cosine_sim"] for l in layers]
        ax2.plot(layers, cos_sims, 'o-', color='tab:purple', markersize=4)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)

        mean_cos = np.mean(cos_sims)
        ax2.set_title(f'Probe vs Centroid Direction Similarity\n(mean cosine = {mean_cos:.3f})')
    else:
        ax2.set_title('Direction Comparison (not available)')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1.1, 1.1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_path}")


def print_summary(results: dict, num_layers: int):
    """Print summary statistics for both methods."""
    fits = results["fits"]
    methods = list(fits.keys())

    print("\n" + "=" * 60)
    print("MC ANSWER CLASSIFICATION SUMMARY")
    print("=" * 60)

    for method in methods:
        method_fits = fits[method]
        layers = sorted(method_fits.keys())

        # Best layer
        best_layer = max(layers, key=lambda l: method_fits[l]["test_accuracy"])
        best_acc = method_fits[best_layer]["test_accuracy"]
        train_acc = method_fits[best_layer]["train_accuracy"]

        # Check for std
        has_std = "test_accuracy_std" in method_fits[best_layer]
        std_str = ""
        if has_std:
            std_str = f" ± {method_fits[best_layer]['test_accuracy_std']:.1%}"

        print(f"\n{method.upper()} METHOD:")
        print(f"  Best layer: {best_layer}")
        print(f"  Train accuracy: {train_acc:.1%}")
        print(f"  Test accuracy: {best_acc:.1%}{std_str}")
        print(f"  Chance: 25%")

        # Early vs late comparison
        n_layers = len(layers)
        early_layers = layers[:n_layers // 4]
        late_layers = layers[3 * n_layers // 4:]

        early_acc = np.mean([method_fits[l]["test_accuracy"] for l in early_layers])
        late_acc = np.mean([method_fits[l]["test_accuracy"] for l in late_layers])

        print(f"  Early layers: {early_acc:.1%}, Late layers: {late_acc:.1%}")

    # Comparison
    if "comparison" in results:
        layers = sorted(results["comparison"].keys())
        cos_sims = [results["comparison"][l]["cosine_sim"] for l in layers]
        print(f"\nMETHOD COMPARISON:")
        print(f"  Mean cosine similarity: {np.mean(cos_sims):.3f}")
        print(f"  Max |cosine|: {np.max(np.abs(cos_sims)):.3f}")

    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Input base: {INPUT_BASE_NAME}")
    print(f"Train/test split: {TRAIN_SPLIT}")
    print(f"PCA components: {N_PCA_COMPONENTS}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print()

    # Load activations
    activations_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_activations.npz"
    print(f"Loading activations from {activations_path}...")
    if not activations_path.exists():
        raise FileNotFoundError(
            f"Activations file not found: {activations_path}\n"
            "Run identify_mc_correlate.py first to generate activations."
        )

    act_data = np.load(activations_path)

    # Find number of layers
    layer_keys = [k for k in act_data.keys() if k.startswith("layer_")]
    num_layers = len(layer_keys)
    print(f"  Found {num_layers} layers")

    activations_by_layer = {}
    for i in range(num_layers):
        activations_by_layer[i] = act_data[f"layer_{i}"]
    print(f"  Shape per layer: {activations_by_layer[0].shape}")

    # Load dataset JSON for model answers
    dataset_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_dataset.json"
    print(f"\nLoading dataset from {dataset_path}...")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {dataset_path}\n"
            "Run identify_mc_correlate.py first to generate dataset."
        )

    with open(dataset_path) as f:
        dataset = json.load(f)

    questions = dataset["data"]
    n_samples = len(questions)
    print(f"  Found {n_samples} questions")

    # Extract model answers
    model_answers = [q["predicted_answer"] for q in questions]
    print(f"  Answer distribution: {dict(zip(*np.unique(model_answers, return_counts=True)))}")

    # Encode answers to integers
    encoded_answers, answer_mapping = encode_answers(model_answers)
    print(f"  Answer mapping: {answer_mapping}")

    # Create train/test split
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED,
        shuffle=True
    )
    print(f"\nTrain/test split: {len(train_idx)}/{len(test_idx)}")

    # Find answer directions using both methods
    print(f"\nTraining MC answer classifiers ({N_BOOTSTRAP} bootstrap iterations)...")
    results = find_answer_directions_both_methods(
        activations_by_layer,
        encoded_answers,
        train_idx,
        test_idx,
        n_components=N_PCA_COMPONENTS,
        random_state=SEED,
        n_bootstrap=N_BOOTSTRAP,
        train_split=TRAIN_SPLIT,
    )

    # Save directions file (both methods, like uncertainty directions)
    directions_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_answer_directions.npz"
    print(f"\nSaving directions to {directions_path}...")

    dir_save = {
        "_metadata_input_base": INPUT_BASE_NAME,
        "_metadata_n_classes": len(answer_mapping),
        "_metadata_answer_mapping": json.dumps(answer_mapping),
    }
    for method in ["probe", "centroid"]:
        for layer in range(num_layers):
            dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]

    np.savez(directions_path, **dir_save)

    # Save probe objects (for transfer tests)
    probes_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_answer_probes.joblib"
    print(f"Saving probes to {probes_path}...")
    probe_save = {
        "metadata": {
            "input_base": INPUT_BASE_NAME,
            "n_classes": len(answer_mapping),
            "answer_mapping": answer_mapping,
            "train_split": TRAIN_SPLIT,
            "n_pca_components": N_PCA_COMPONENTS,
            "seed": SEED,
        },
        "probes": results["probes"],
    }
    joblib.dump(probe_save, probes_path)

    # Save results JSON
    results_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_answer_results.json"
    print(f"Saving results to {results_path}...")

    results_json = {
        "config": {
            "input_base": INPUT_BASE_NAME,
            "train_split": TRAIN_SPLIT,
            "n_pca_components": N_PCA_COMPONENTS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "n_classes": len(answer_mapping),
            "answer_mapping": answer_mapping,
        },
        "stats": {
            "n_samples": n_samples,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "answer_distribution": {k: int(v) for k, v in zip(*np.unique(model_answers, return_counts=True))},
        },
        "results": {},
        "comparison": {},
    }

    # Results per method
    for method in ["probe", "centroid"]:
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

    # Comparison
    for layer in range(num_layers):
        results_json["comparison"][layer] = {
            "cosine_sim": float(results["comparison"][layer]["cosine_sim"])
        }

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # Plot results
    plot_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_answer_results.png"
    print(f"\nPlotting results...")
    plot_accuracy_results(results, plot_path)

    # Print summary
    print_summary(results, num_layers)

    print("\nOutput files:")
    print(f"  {directions_path.name}")
    print(f"  {probes_path.name}")
    print(f"  {results_path.name}")
    print(f"  {plot_path.name}")


if __name__ == "__main__":
    main()
