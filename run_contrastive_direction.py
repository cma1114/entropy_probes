"""
Find introspection mapping direction using contrastive approach.

Instead of regression, this script:
1. Loads introspection data (direct entropies, stated confidences, meta activations)
2. Identifies well-calibrated examples:
   - High confidence + low entropy (model correctly confident)
   - Low confidence + high entropy (model correctly uncertain)
3. Identifies miscalibrated examples:
   - High confidence + high entropy (overconfident)
   - Low confidence + low entropy (underconfident)
4. Computes direction = mean(well_calibrated) - mean(miscalibrated)
5. Tests direction via steering/ablation

This is equivalent to the regression approach but uses a "hard" selection
rather than soft weighting.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_run_name,
    BatchedExtractor,
    compute_introspection_scores,
    compute_contrastive_direction,
    extract_probe_direction,
    train_introspection_mapping_probe,
    steering_context,
    ablation_context,
    generate_orthogonal_directions,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME
SEED = 42

# Contrastive selection thresholds
# Use top/bottom quantiles of introspection score
TOP_QUANTILE = 0.25  # Top 25% = well-calibrated
BOTTOM_QUANTILE = 0.25  # Bottom 25% = miscalibrated

# Which layer to use for direction extraction
TARGET_LAYER = None  # Will be set to best layer from probe results, or middle layer

np.random.seed(SEED)
torch.manual_seed(SEED)


def load_introspection_data(run_name: str = None) -> dict:
    """
    Load previously collected introspection data.

    Looks for:
    - introspection_paired_data.json (or with run_name prefix)
    - introspection_meta_activations.npz
    """
    # Try to find data files
    if run_name:
        paired_path = Path(f"{run_name}_introspection_paired_data.json")
        acts_path = Path(f"{run_name}_introspection_meta_activations.npz")
    else:
        paired_path = Path("introspection_paired_data.json")
        acts_path = Path("introspection_meta_activations.npz")

    if not paired_path.exists():
        raise FileNotFoundError(
            f"Could not find {paired_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    if not acts_path.exists():
        raise FileNotFoundError(
            f"Could not find {acts_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    # Load paired data
    print(f"Loading paired data from {paired_path}...")
    with open(paired_path) as f:
        paired_data = json.load(f)

    # Extract arrays
    direct_entropies = np.array([d["direct_entropy"] for d in paired_data])
    stated_confidences = np.array([d["stated_confidence"] for d in paired_data])

    # Load meta activations
    print(f"Loading meta activations from {acts_path}...")
    acts_data = np.load(acts_path)

    # activations are stored as layer_0, layer_1, etc.
    layer_keys = sorted([k for k in acts_data.keys() if k.startswith("layer_")])
    num_layers = len(layer_keys)

    meta_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in layer_keys
    }

    print(f"Loaded {len(paired_data)} examples with {num_layers} layers")
    print(f"Entropy range: [{direct_entropies.min():.3f}, {direct_entropies.max():.3f}]")
    print(f"Confidence range: [{stated_confidences.min():.3f}, {stated_confidences.max():.3f}]")

    return {
        "paired_data": paired_data,
        "direct_entropies": direct_entropies,
        "stated_confidences": stated_confidences,
        "meta_activations": meta_activations,
        "num_layers": num_layers,
    }


def compute_contrastive_direction_with_details(
    meta_activations: np.ndarray,
    introspection_scores: np.ndarray,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25
) -> dict:
    """
    Compute contrastive direction with detailed statistics.

    Well-calibrated: high introspection score
      = (high confidence AND low entropy) OR (low confidence AND high entropy)

    Miscalibrated: low introspection score
      = (high confidence AND high entropy) OR (low confidence AND low entropy)

    Returns direction and detailed info about selected examples.
    """
    n_samples = len(introspection_scores)

    # Compute thresholds
    high_threshold = np.quantile(introspection_scores, 1 - top_quantile)
    low_threshold = np.quantile(introspection_scores, bottom_quantile)

    # Select examples
    well_calibrated_mask = introspection_scores >= high_threshold
    miscalibrated_mask = introspection_scores <= low_threshold

    well_calibrated_acts = meta_activations[well_calibrated_mask]
    miscalibrated_acts = meta_activations[miscalibrated_mask]

    # Compute direction
    well_calibrated_mean = well_calibrated_acts.mean(axis=0)
    miscalibrated_mean = miscalibrated_acts.mean(axis=0)

    direction = well_calibrated_mean - miscalibrated_mean
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm

    return {
        "direction": direction_normalized,
        "direction_magnitude": direction_norm,
        "n_well_calibrated": int(well_calibrated_mask.sum()),
        "n_miscalibrated": int(miscalibrated_mask.sum()),
        "high_threshold": float(high_threshold),
        "low_threshold": float(low_threshold),
        "well_calibrated_scores_mean": float(introspection_scores[well_calibrated_mask].mean()),
        "miscalibrated_scores_mean": float(introspection_scores[miscalibrated_mask].mean()),
    }


def analyze_selected_examples(
    paired_data: list,
    introspection_scores: np.ndarray,
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    high_threshold: float,
    low_threshold: float
) -> dict:
    """
    Analyze the characteristics of selected well-calibrated vs miscalibrated examples.
    """
    well_mask = introspection_scores >= high_threshold
    misc_mask = introspection_scores <= low_threshold

    # Z-scores for interpretation
    entropy_z = (direct_entropies - direct_entropies.mean()) / direct_entropies.std()
    conf_z = (stated_confidences - stated_confidences.mean()) / stated_confidences.std()

    print("\n" + "="*60)
    print("SELECTED EXAMPLES ANALYSIS")
    print("="*60)

    print(f"\nWell-calibrated examples (n={well_mask.sum()}):")
    print(f"  Mean entropy z-score: {entropy_z[well_mask].mean():.2f}")
    print(f"  Mean confidence z-score: {conf_z[well_mask].mean():.2f}")
    print(f"  Correlation (entropy, confidence): {stats.pearsonr(entropy_z[well_mask], conf_z[well_mask])[0]:.3f}")

    # Break down by type
    high_conf_low_ent = well_mask & (conf_z > 0) & (entropy_z < 0)
    low_conf_high_ent = well_mask & (conf_z < 0) & (entropy_z > 0)
    print(f"  High confidence + low entropy: {high_conf_low_ent.sum()}")
    print(f"  Low confidence + high entropy: {low_conf_high_ent.sum()}")

    print(f"\nMiscalibrated examples (n={misc_mask.sum()}):")
    print(f"  Mean entropy z-score: {entropy_z[misc_mask].mean():.2f}")
    print(f"  Mean confidence z-score: {conf_z[misc_mask].mean():.2f}")
    print(f"  Correlation (entropy, confidence): {stats.pearsonr(entropy_z[misc_mask], conf_z[misc_mask])[0]:.3f}")

    # Break down by type
    high_conf_high_ent = misc_mask & (conf_z > 0) & (entropy_z > 0)
    low_conf_low_ent = misc_mask & (conf_z < 0) & (entropy_z < 0)
    print(f"  High confidence + high entropy (overconfident): {high_conf_high_ent.sum()}")
    print(f"  Low confidence + low entropy (underconfident): {low_conf_low_ent.sum()}")

    return {
        "well_calibrated": {
            "n": int(well_mask.sum()),
            "entropy_z_mean": float(entropy_z[well_mask].mean()),
            "confidence_z_mean": float(conf_z[well_mask].mean()),
            "n_high_conf_low_ent": int(high_conf_low_ent.sum()),
            "n_low_conf_high_ent": int(low_conf_high_ent.sum()),
        },
        "miscalibrated": {
            "n": int(misc_mask.sum()),
            "entropy_z_mean": float(entropy_z[misc_mask].mean()),
            "confidence_z_mean": float(conf_z[misc_mask].mean()),
            "n_overconfident": int(high_conf_high_ent.sum()),
            "n_underconfident": int(low_conf_low_ent.sum()),
        }
    }


def compare_to_regression(
    meta_activations: np.ndarray,
    introspection_scores: np.ndarray,
    contrastive_direction: np.ndarray
) -> dict:
    """
    Compare contrastive direction to regression-based direction.
    """
    print("\n" + "="*60)
    print("COMPARISON: CONTRASTIVE VS REGRESSION")
    print("="*60)

    # Train regression probe
    probe, regression_direction = train_introspection_mapping_probe(
        meta_activations,
        introspection_scores,
        alpha=1000.0,
        use_pca=False
    )

    # Normalize both
    contrastive_norm = contrastive_direction / np.linalg.norm(contrastive_direction)
    regression_norm = regression_direction / np.linalg.norm(regression_direction)

    # Compute similarity
    cosine_sim = np.dot(contrastive_norm, regression_norm)

    print(f"Cosine similarity between directions: {cosine_sim:.4f}")
    print(f"Angle between directions: {np.degrees(np.arccos(np.clip(cosine_sim, -1, 1))):.1f}°")

    # Test how well each direction predicts introspection score
    contrastive_proj = meta_activations @ contrastive_norm
    regression_proj = meta_activations @ regression_norm

    contrastive_corr = stats.pearsonr(contrastive_proj, introspection_scores)[0]
    regression_corr = stats.pearsonr(regression_proj, introspection_scores)[0]

    print(f"Contrastive direction correlation with scores: {contrastive_corr:.4f}")
    print(f"Regression direction correlation with scores: {regression_corr:.4f}")

    return {
        "cosine_similarity": float(cosine_sim),
        "angle_degrees": float(np.degrees(np.arccos(np.clip(cosine_sim, -1, 1)))),
        "contrastive_correlation": float(contrastive_corr),
        "regression_correlation": float(regression_corr),
    }


def run_layer_analysis(
    meta_activations: dict,
    introspection_scores: np.ndarray,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25
) -> dict:
    """
    Compute contrastive direction for each layer and analyze.
    """
    print("\n" + "="*60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("="*60)

    results = {}

    for layer_idx in tqdm(sorted(meta_activations.keys())):
        acts = meta_activations[layer_idx]

        # Compute contrastive direction
        dir_info = compute_contrastive_direction_with_details(
            acts, introspection_scores, top_quantile, bottom_quantile
        )

        # Test how well projection correlates with score
        proj = acts @ dir_info["direction"]
        corr, pval = stats.pearsonr(proj, introspection_scores)

        results[layer_idx] = {
            **dir_info,
            "projection_correlation": float(corr),
            "projection_pvalue": float(pval),
        }

    # Print summary
    print(f"\n{'Layer':<8} {'Dir Mag':<12} {'Proj Corr':<12} {'p-value':<12}")
    print("-" * 44)
    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        print(f"{layer_idx:<8} {r['direction_magnitude']:<12.4f} "
              f"{r['projection_correlation']:<12.4f} {r['projection_pvalue']:<12.2e}")

    # Find best layer
    best_layer = max(results.keys(), key=lambda l: abs(results[l]["projection_correlation"]))
    print(f"\nBest layer: {best_layer} (correlation = {results[best_layer]['projection_correlation']:.4f})")

    return results


def plot_results(
    introspection_scores: np.ndarray,
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    layer_results: dict,
    high_threshold: float,
    low_threshold: float,
    output_path: str = "contrastive_direction_results.png"
):
    """Plot analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Introspection score distribution with selection thresholds
    ax = axes[0, 0]
    ax.hist(introspection_scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(high_threshold, color='green', linestyle='--',
               label=f'Well-calibrated threshold ({high_threshold:.2f})')
    ax.axvline(low_threshold, color='red', linestyle='--',
               label=f'Miscalibrated threshold ({low_threshold:.2f})')
    ax.set_xlabel('Introspection Score')
    ax.set_ylabel('Count')
    ax.set_title('Introspection Score Distribution')
    ax.legend()

    # 2. Entropy vs Confidence with calibration coloring
    ax = axes[0, 1]
    colors = ['red' if s <= low_threshold else 'green' if s >= high_threshold else 'gray'
              for s in introspection_scores]
    ax.scatter(direct_entropies, stated_confidences, c=colors, alpha=0.5, s=20)
    ax.set_xlabel('Direct Entropy')
    ax.set_ylabel('Stated Confidence')
    ax.set_title('Entropy vs Confidence (green=well-calibrated, red=miscalibrated)')

    # Add trend line
    z = np.polyfit(direct_entropies, stated_confidences, 1)
    p = np.poly1d(z)
    x_line = np.linspace(direct_entropies.min(), direct_entropies.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, label='Overall trend')
    ax.legend()

    # 3. Direction magnitude by layer
    ax = axes[1, 0]
    layers = sorted(layer_results.keys())
    magnitudes = [layer_results[l]["direction_magnitude"] for l in layers]
    ax.plot(layers, magnitudes, 'o-')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Direction Magnitude')
    ax.set_title('Contrastive Direction Magnitude by Layer')
    ax.grid(True, alpha=0.3)

    # 4. Projection correlation by layer
    ax = axes[1, 1]
    correlations = [layer_results[l]["projection_correlation"] for l in layers]
    ax.plot(layers, correlations, 'o-')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Introspection Score')
    ax.set_title('Direction Projection Correlation by Layer')
    ax.grid(True, alpha=0.3)

    # Highlight best layer
    best_layer = max(layers, key=lambda l: abs(layer_results[l]["projection_correlation"]))
    ax.scatter([best_layer], [layer_results[best_layer]["projection_correlation"]],
               color='red', s=100, zorder=5, label=f'Best: layer {best_layer}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Contrastive selection: top {TOP_QUANTILE*100:.0f}% vs bottom {BOTTOM_QUANTILE*100:.0f}%")

    # Load data
    data = load_introspection_data()

    direct_entropies = data["direct_entropies"]
    stated_confidences = data["stated_confidences"]
    meta_activations = data["meta_activations"]
    paired_data = data["paired_data"]
    num_layers = data["num_layers"]

    # Compute introspection scores
    print("\nComputing introspection scores...")
    introspection_scores = compute_introspection_scores(direct_entropies, stated_confidences)

    print(f"Introspection score range: [{introspection_scores.min():.3f}, {introspection_scores.max():.3f}]")
    print(f"Mean: {introspection_scores.mean():.3f}, Std: {introspection_scores.std():.3f}")

    # Run layer-by-layer analysis
    layer_results = run_layer_analysis(
        meta_activations, introspection_scores, TOP_QUANTILE, BOTTOM_QUANTILE
    )

    # Get best layer for detailed analysis
    best_layer = max(layer_results.keys(), key=lambda l: abs(layer_results[l]["projection_correlation"]))

    # Detailed analysis on best layer
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS (Layer {best_layer})")
    print(f"{'='*60}")

    best_acts = meta_activations[best_layer]
    best_result = layer_results[best_layer]

    # Analyze selected examples
    example_analysis = analyze_selected_examples(
        paired_data,
        introspection_scores,
        direct_entropies,
        stated_confidences,
        best_result["high_threshold"],
        best_result["low_threshold"]
    )

    # Compare to regression approach
    comparison = compare_to_regression(
        best_acts, introspection_scores, best_result["direction"]
    )

    # Plot results
    plot_results(
        introspection_scores,
        direct_entropies,
        stated_confidences,
        layer_results,
        best_result["high_threshold"],
        best_result["low_threshold"]
    )

    # Save results
    results = {
        "config": {
            "top_quantile": TOP_QUANTILE,
            "bottom_quantile": BOTTOM_QUANTILE,
            "seed": SEED,
        },
        "best_layer": best_layer,
        "layer_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "direction"}
            for k, v in layer_results.items()
        },
        "example_analysis": example_analysis,
        "regression_comparison": comparison,
    }

    output_path = "contrastive_direction_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Save directions
    directions = {
        f"layer_{k}": v["direction"]
        for k, v in layer_results.items()
    }
    np.savez_compressed("contrastive_directions.npz", **directions)
    print("Directions saved to contrastive_directions.npz")


if __name__ == "__main__":
    main()
