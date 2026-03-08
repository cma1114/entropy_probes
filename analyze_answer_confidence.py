"""
Compare d_mc (holistic uncertainty) vs d_answer classifier confidence.

Tests the relationship between Hypothesis 1 and Hypothesis 2:
- H1: Holistic "I know" direction (d_mc predicts entropy/logit_gap)
- H2: Answer predominance (classifier confidence from ABCD distribution)

Key question: Are H1 and H2 measuring the same thing or different things?

If same:
- correlation(d_mc_projection, classifier_confidence) should be high
- Both should predict stated confidence equally well

If different:
- Low correlation between d_mc and classifier confidence
- Each provides independent signal for predicting confidence

Prerequisites (scripts that must be run first):
    1. identify_mc_correlate.py with FIND_ANSWER_DIRECTIONS=True
       - Produces: {dataset}_mc_activations.npz (MC task activations)
       - Produces: {dataset}_mc_{metric}_directions.npz (d_mc uncertainty directions)
       - Produces: {dataset}_mc_answer_directions.npz (d_answer classifier)
       - Produces: {dataset}_mc_results.json (metrics and best layers)

    2. test_meta_transfer.py with META_TASK = "delegate" (or other)
       - Produces: {dataset}_meta_{task}_activations.npz (meta task activations)
       - Produces: {dataset}_meta_{task}_transfer_results_{pos}.json

Inputs:
    outputs/{model_dir}/working/{dataset}_mc_activations.npz
    outputs/{model_dir}/working/{dataset}_mc_{metric}_directions.npz
    outputs/{model_dir}/working/{dataset}_mc_answer_directions.npz
    outputs/{model_dir}/results/{dataset}_mc_results.json
    outputs/{model_dir}/working/{dataset}_meta_{task}_activations.npz

Outputs:
    outputs/{model_dir}/results/{dataset}_answer_confidence_analysis.json
    outputs/{model_dir}/results/{dataset}_answer_confidence_analysis.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from tqdm import tqdm

from core.model_utils import get_model_dir_name
from core.config_utils import get_output_path, find_output_file, get_config_dict
from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA
from core.answer_directions import (
    train_mc_answer_classifier,
    compute_classifier_confidence,
)
from core.probes import apply_probe_direction

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

DATASET = "TriviaMC_difficulty_filtered"
META_TASK = "delegate"
PROBE_POSITION = "final"
METRIC = "logit_gap"

SEED = 42
TRAIN_SPLIT = 0.8
PROBE_PCA_COMPONENTS = 100

# Layers to analyze (None = analyze best layer from MC results)
LAYERS = None

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def load_mc_activations(base_name: str, model_dir: str) -> dict:
    """Load MC task activations and metrics."""
    act_path = find_output_file(f"{base_name}_mc_activations.npz", model_dir=model_dir)
    if not act_path.exists():
        raise FileNotFoundError(f"MC activations not found: {act_path}")

    data = np.load(act_path)
    return {
        "activations": {int(k.split("_")[1]): data[k] for k in data.files if k.startswith("layer_")},
        "metrics": {k: data[k] for k in data.files if not k.startswith("layer_") and not k.startswith("_")},
    }


def load_mc_directions(base_name: str, metric: str, model_dir: str) -> dict:
    """Load d_mc directions."""
    path = find_output_file(f"{base_name}_mc_{metric}_directions.npz", model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"MC directions not found: {path}")

    data = np.load(path)
    return {
        int(k.split("_")[2]): data[k]
        for k in data.files
        if k.startswith("probe_layer_")
    }


def load_answer_directions(base_name: str, model_dir: str) -> dict:
    """Load d_answer directions and classifiers."""
    path = find_output_file(f"{base_name}_mc_answer_directions.npz", model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Answer directions not found: {path}")

    data = np.load(path)
    return {
        int(k.split("_")[2]): data[k]
        for k in data.files
        if k.startswith("probe_layer_")
    }


def load_meta_activations(base_name: str, task: str, model_dir: str) -> dict:
    """Load meta-task activations."""
    path = find_output_file(f"{base_name}_meta_{task}_activations.npz", model_dir=model_dir)
    if not path.exists():
        return None

    data = np.load(path)
    # Handle multi-position format
    result = {}
    for k in data.files:
        if k.startswith("layer_"):
            parts = k.split("_")
            layer = int(parts[1])
            if len(parts) > 2:
                pos = parts[2]
            else:
                pos = "final"
            if pos not in result:
                result[pos] = {}
            result[pos][layer] = data[k]

    return result


def analyze_layer(
    mc_activations: np.ndarray,
    mc_answers: np.ndarray,
    mc_uncertainty: np.ndarray,
    d_mc: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    rng: np.random.Generator,
) -> dict:
    """
    Analyze relationship between d_mc projection and classifier confidence.

    Returns metrics comparing:
    1. d_mc projection → uncertainty (our baseline)
    2. classifier confidence → uncertainty
    3. d_mc projection ↔ classifier confidence (are they the same?)
    """
    # Train answer classifier on train set
    X_train = mc_activations[train_idx]
    y_train = mc_answers[train_idx]

    scaler, pca, clf = train_mc_answer_classifier(X_train, y_train, n_components=PROBE_PCA_COMPONENTS)

    # Get classifier confidence for all samples
    probs, max_probs, entropies = compute_classifier_confidence(
        mc_activations, scaler, pca, clf, use_separate_scaling=False
    )

    # classifier_confidence = max_prob (higher = more confident)
    # classifier_uncertainty = entropy (higher = more uncertain)
    classifier_confidence = max_probs
    classifier_uncertainty = entropies

    # Compute d_mc projection
    d_mc_norm = d_mc / np.linalg.norm(d_mc)
    mc_mean = np.mean(mc_activations, axis=0)
    mc_centered = mc_activations - mc_mean
    d_mc_projection = mc_centered @ d_mc_norm

    # Test set metrics
    test_unc = mc_uncertainty[test_idx]
    test_d_mc = d_mc_projection[test_idx]
    test_clf_conf = classifier_confidence[test_idx]
    test_clf_unc = classifier_uncertainty[test_idx]

    # Correlations with actual uncertainty
    r_dmc_unc, p_dmc_unc = pearsonr(test_d_mc, test_unc)
    r_clfconf_unc, p_clfconf_unc = pearsonr(test_clf_conf, -test_unc)  # Negate for confidence
    r_clfunc_unc, p_clfunc_unc = pearsonr(test_clf_unc, test_unc)

    # Correlation between d_mc and classifier metrics
    r_dmc_clfconf, p_dmc_clfconf = pearsonr(test_d_mc, test_clf_conf)
    r_dmc_clfunc, p_dmc_clfunc = pearsonr(test_d_mc, test_clf_unc)

    return {
        # d_mc vs actual uncertainty
        "d_mc_vs_uncertainty_r": float(r_dmc_unc),
        "d_mc_vs_uncertainty_p": float(p_dmc_unc),

        # Classifier confidence vs actual uncertainty
        "clf_conf_vs_uncertainty_r": float(r_clfconf_unc),
        "clf_conf_vs_uncertainty_p": float(p_clfconf_unc),

        # Classifier uncertainty vs actual uncertainty
        "clf_unc_vs_uncertainty_r": float(r_clfunc_unc),
        "clf_unc_vs_uncertainty_p": float(p_clfunc_unc),

        # d_mc vs classifier (are they the same?)
        "d_mc_vs_clf_conf_r": float(r_dmc_clfconf),
        "d_mc_vs_clf_conf_p": float(p_dmc_clfconf),
        "d_mc_vs_clf_unc_r": float(r_dmc_clfunc),
        "d_mc_vs_clf_unc_p": float(p_dmc_clfunc),

        # Raw values for plotting
        "test_d_mc": test_d_mc.tolist(),
        "test_clf_conf": test_clf_conf.tolist(),
        "test_clf_unc": test_clf_unc.tolist(),
        "test_uncertainty": test_unc.tolist(),
    }


def plot_answer_confidence_analysis(analysis: dict, output_path: Path):
    """Plot comparison of d_mc vs classifier confidence."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    test_d_mc = np.array(analysis["test_d_mc"])
    test_clf_conf = np.array(analysis["test_clf_conf"])
    test_clf_unc = np.array(analysis["test_clf_unc"])
    test_unc = np.array(analysis["test_uncertainty"])

    # Top-left: d_mc vs uncertainty
    ax = axes[0, 0]
    ax.scatter(test_d_mc, test_unc, alpha=0.5, s=20)
    ax.set_xlabel("d_mc projection")
    ax.set_ylabel("MC Uncertainty (logit_gap)")
    ax.set_title(f"d_mc vs Uncertainty\nr={analysis['d_mc_vs_uncertainty_r']:.3f}")
    ax.grid(True, alpha=GRID_ALPHA)

    # Top-right: classifier confidence vs uncertainty
    ax = axes[0, 1]
    ax.scatter(test_clf_conf, test_unc, alpha=0.5, s=20, color="orange")
    ax.set_xlabel("Classifier Confidence (max prob)")
    ax.set_ylabel("MC Uncertainty (logit_gap)")
    ax.set_title(f"Classifier Confidence vs Uncertainty\nr={analysis['clf_conf_vs_uncertainty_r']:.3f}")
    ax.grid(True, alpha=GRID_ALPHA)

    # Bottom-left: d_mc vs classifier confidence
    ax = axes[1, 0]
    ax.scatter(test_d_mc, test_clf_conf, alpha=0.5, s=20, color="green")
    ax.set_xlabel("d_mc projection")
    ax.set_ylabel("Classifier Confidence (max prob)")
    ax.set_title(f"d_mc vs Classifier Confidence\nr={analysis['d_mc_vs_clf_conf_r']:.3f}")
    ax.grid(True, alpha=GRID_ALPHA)

    # Bottom-right: Summary comparison
    ax = axes[1, 1]
    labels = ["d_mc→Unc", "Clf_conf→Unc", "Clf_unc→Unc", "d_mc↔Clf"]
    values = [
        abs(analysis["d_mc_vs_uncertainty_r"]),
        abs(analysis["clf_conf_vs_uncertainty_r"]),
        abs(analysis["clf_unc_vs_uncertainty_r"]),
        abs(analysis["d_mc_vs_clf_unc_r"]),
    ]
    colors = ["tab:blue", "tab:orange", "tab:red", "tab:green"]

    bars = ax.bar(labels, values, color=colors)
    ax.set_ylabel("|Correlation|")
    ax.set_title("Summary: Correlation Magnitudes")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=GRID_ALPHA, axis='y')

    # Add interpretation
    dmc_clf_corr = abs(analysis["d_mc_vs_clf_unc_r"])
    if dmc_clf_corr > 0.7:
        interpretation = "SAME SIGNAL: d_mc ≈ classifier confidence"
    elif dmc_clf_corr < 0.3:
        interpretation = "DIFFERENT SIGNALS: d_mc and classifier confidence measure different things"
    else:
        interpretation = "PARTIAL OVERLAP: Some shared variance, some unique"

    ax.text(0.5, 0.95, interpretation, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET

    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print(f"Meta task: {META_TASK}")
    print(f"Metric: {METRIC}")
    print()

    rng = np.random.default_rng(SEED)

    # Load data
    print("Loading MC activations and directions...")
    mc_data = load_mc_activations(base_name, model_dir)
    d_mc_directions = load_mc_directions(base_name, METRIC, model_dir)

    activations = mc_data["activations"]
    mc_uncertainty = mc_data["metrics"].get(METRIC, mc_data["metrics"].get("logit_gap"))
    mc_answers = mc_data["metrics"].get("model_answers", mc_data["metrics"].get("answers"))

    # Encode answers if they're strings
    if mc_answers.dtype.kind in ('U', 'S', 'O'):
        from core.answer_directions import encode_answers
        mc_answers = encode_answers(mc_answers)

    n_samples = len(mc_uncertainty)
    print(f"  Loaded {n_samples} samples, {len(activations)} layers")

    # Train/test split
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Determine which layers to analyze
    if LAYERS:
        layers_to_analyze = LAYERS
    else:
        # Use best layer from MC results
        results_path = find_output_file(f"{base_name}_mc_results.json", model_dir=model_dir)
        if results_path.exists():
            with open(results_path) as f:
                mc_results = json.load(f)
            best_layer = mc_results.get("metrics", {}).get(METRIC, {}).get("best_layer", 20)
        else:
            best_layer = 20
        layers_to_analyze = [best_layer]

    print(f"  Analyzing layers: {layers_to_analyze}")

    # Analyze each layer
    all_results = {}
    for layer in tqdm(layers_to_analyze, desc="Analyzing layers"):
        if layer not in activations or layer not in d_mc_directions:
            continue

        result = analyze_layer(
            mc_activations=activations[layer],
            mc_answers=mc_answers,
            mc_uncertainty=mc_uncertainty,
            d_mc=d_mc_directions[layer],
            train_idx=train_idx,
            test_idx=test_idx,
            rng=rng,
        )
        all_results[layer] = result

    # Print summary
    print("\n" + "=" * 70)
    print("ANSWER CONFIDENCE ANALYSIS")
    print("=" * 70)

    for layer, result in all_results.items():
        print(f"\nLayer {layer}:")
        print(f"  d_mc → uncertainty:           r = {result['d_mc_vs_uncertainty_r']:.3f}")
        print(f"  Classifier conf → uncertainty: r = {result['clf_conf_vs_uncertainty_r']:.3f}")
        print(f"  Classifier unc → uncertainty:  r = {result['clf_unc_vs_uncertainty_r']:.3f}")
        print(f"  d_mc ↔ classifier unc:         r = {result['d_mc_vs_clf_unc_r']:.3f}")

        dmc_clf_corr = abs(result["d_mc_vs_clf_unc_r"])
        if dmc_clf_corr > 0.7:
            print(f"\n  CONCLUSION: H1 ≈ H2 (high correlation {dmc_clf_corr:.2f})")
            print("    d_mc and classifier confidence measure the same thing")
        elif dmc_clf_corr < 0.3:
            print(f"\n  CONCLUSION: H1 ≠ H2 (low correlation {dmc_clf_corr:.2f})")
            print("    d_mc and classifier confidence provide independent signals")
        else:
            print(f"\n  CONCLUSION: Partial overlap (correlation {dmc_clf_corr:.2f})")

    # Save results
    summary = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            meta_task=META_TASK,
            metric=METRIC,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "layers_analyzed": layers_to_analyze,
        "results_by_layer": {
            str(k): {kk: vv for kk, vv in v.items() if not kk.startswith("test_")}
            for k, v in all_results.items()
        },
    }

    summary_path = get_output_path(f"{base_name}_answer_confidence_analysis.json", model_dir=model_dir)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")

    # Plot for best layer
    if all_results:
        best_layer = max(all_results.keys(), key=lambda l: abs(all_results[l]["d_mc_vs_uncertainty_r"]))
        plot_path = get_output_path(f"{base_name}_answer_confidence_analysis.png", model_dir=model_dir)
        plot_answer_confidence_analysis(all_results[best_layer], plot_path)
        print(f"Plot: {plot_path}")


if __name__ == "__main__":
    main()
