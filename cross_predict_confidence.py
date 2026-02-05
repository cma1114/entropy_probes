"""
Analysis. Tests whether self-confidence probes are specific to self-confidence or
also predict other-confidence equally well, determining if the model has genuinely
distinct representations for "how confident am I?" vs "how hard is this for others?"

Inputs:
    outputs/{base}_meta_{task}_confidence_directions.npz      Confidence directions (self and other)
    outputs/{base}_meta_{task}_activations.npz                Activations for confidence tasks

Outputs:
    outputs/{base}_cross_confidence_results.json    Full cross-prediction metrics per layer
    outputs/{base}_cross_confidence_results.png     Multi-panel visualization

Run after: test_meta_transfer.py (with FIND_CONFIDENCE_DIRECTIONS=True for both
           self and other tasks)
"""

from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import joblib

from core.directions import apply_probe_centered
from core.config_utils import get_config_dict
from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

# Base name for input files
INPUT_BASE_NAME = "Llama-3.3-70B-Instruct_TriviaMC_difficulty_filtered"

# Train/test split (must match identify_confidence_correlate.py)
TRAIN_SPLIT = 0.8
SEED = 42

# Bootstrap for confidence intervals (resample predictions, NOT refit)
N_BOOTSTRAP = 100

# Output
OUTPUT_DIR = Path(__file__).parent / "outputs"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_activations_and_confidence(npz_path: Path, position: str = "final"):
    """Load activations and confidence values from npz file."""
    data = np.load(npz_path)

    # Detect layer keys
    layer_keys = [k for k in data.keys() if k.startswith("layer_")]
    has_positions = any("_" in k.replace("layer_", "", 1) for k in layer_keys)

    activations = {}
    if has_positions:
        # Multi-position format: layer_0_final, layer_1_final, ...
        position_keys = [k for k in layer_keys if k.endswith(f"_{position}")]
        if not position_keys:
            # Fall back to first available position
            positions = set()
            for k in layer_keys:
                parts = k.split("_")
                if len(parts) >= 3:
                    positions.add("_".join(parts[2:]))
            position = sorted(positions)[0] if positions else None
            position_keys = [k for k in layer_keys if k.endswith(f"_{position}")]
            print(f"  Warning: 'final' not found, using '{position}'")

        num_layers = len(position_keys)
        for i in range(num_layers):
            activations[i] = data[f"layer_{i}_{position}"]
    else:
        # Legacy format: layer_0, layer_1, ...
        num_layers = len(layer_keys)
        for i in range(num_layers):
            activations[i] = data[f"layer_{i}"]

    # Load confidence values
    if "stated_confidence" in data:
        confidences = data["stated_confidence"]
    elif "confidences" in data:
        confidences = data["confidences"]
    else:
        confidence_keys = [k for k in data.keys() if "confidence" in k.lower() or "p_answer" in k.lower()]
        if confidence_keys:
            confidences = data[confidence_keys[0]]
        else:
            raise KeyError(f"No confidence values found in {npz_path}")

    return activations, confidences, num_layers


def bootstrap_r2(predictions: np.ndarray, targets: np.ndarray, n_bootstrap: int, seed: int):
    """Bootstrap R2 confidence intervals by resampling predictions."""
    rng = np.random.default_rng(seed)
    n = len(predictions)
    r2_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        pred_boot = predictions[idx]
        targ_boot = targets[idx]

        # Compute R2
        ss_res = np.sum((targ_boot - pred_boot) ** 2)
        ss_tot = np.sum((targ_boot - targ_boot.mean()) ** 2)
        if ss_tot > 0:
            r2 = 1 - ss_res / ss_tot
        else:
            r2 = np.nan
        r2_samples.append(r2)

    r2_samples = np.array(r2_samples)
    return {
        "mean": float(np.nanmean(r2_samples)),
        "std": float(np.nanstd(r2_samples)),
        "ci_low": float(np.nanpercentile(r2_samples, 2.5)),
        "ci_high": float(np.nanpercentile(r2_samples, 97.5)),
    }


# =============================================================================
# VISUALIZATION
# =============================================================================


def plot_cross_prediction_results(results: dict, num_layers: int, output_path: Path):
    """Plot cross-prediction results in a 4-panel figure."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Self vs Other Confidence Cross-Prediction", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))

    # Extract data - use Pearson correlation (scale-invariant) instead of R²
    self_to_self = [results["by_layer"][l]["self_to_self"]["pearson"] for l in layers]
    self_to_other = [results["by_layer"][l]["self_to_other"]["pearson"] for l in layers]
    other_to_self = [results["by_layer"][l]["other_to_self"]["pearson"] for l in layers]
    other_to_other = [results["by_layer"][l]["other_to_other"]["pearson"] for l in layers]

    # Panel 1: Layer-wise Pearson correlation (top-left)
    ax1 = axes[0, 0]
    ax1.set_title("Cross-Prediction Pearson r by Layer", fontsize=11)
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax1.plot(layers, self_to_self, 'o-', color='tab:blue', linewidth=2, markersize=4, label='self→self')
    ax1.plot(layers, self_to_other, 's--', color='tab:blue', linewidth=1.5, markersize=4, alpha=0.6, label='self→other')
    ax1.plot(layers, other_to_other, 'o-', color='tab:orange', linewidth=2, markersize=4, label='other→other')
    ax1.plot(layers, other_to_self, 's--', color='tab:orange', linewidth=1.5, markersize=4, alpha=0.6, label='other→self')

    ax1.set_xlabel("Layer Index")
    ax1.set_ylabel("Pearson r")
    ax1.set_ylim(-1.1, 1.1)
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=GRID_ALPHA)

    # Panel 2: Transfer matrix heatmap at best layer (top-right)
    ax2 = axes[0, 1]
    best_layer = results["summary"]["best_layer"]
    ax2.set_title(f"Transfer Matrix at Best Layer (L{best_layer})", fontsize=11)

    matrix = np.array([
        [results["by_layer"][best_layer]["self_to_self"]["pearson"],
         results["by_layer"][best_layer]["self_to_other"]["pearson"]],
        [results["by_layer"][best_layer]["other_to_self"]["pearson"],
         results["by_layer"][best_layer]["other_to_other"]["pearson"]]
    ])

    im = ax2.imshow(matrix, cmap='RdYlGn', vmin=-1.0, vmax=1.0)
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['Test: self', 'Test: other'])
    ax2.set_yticklabels(['Train: self', 'Train: other'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')

    plt.colorbar(im, ax=ax2, label='Pearson r')

    # Panel 3: Specificity ratios by layer (bottom-left)
    ax3 = axes[1, 0]
    ax3.set_title("Specificity Ratios by Layer", fontsize=11)
    ax3.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1, label='No specificity')

    self_spec = []
    other_spec = []
    for l in layers:
        s2s = results["by_layer"][l]["self_to_self"]["pearson"]
        s2o = results["by_layer"][l]["self_to_other"]["pearson"]
        o2o = results["by_layer"][l]["other_to_other"]["pearson"]
        o2s = results["by_layer"][l]["other_to_self"]["pearson"]

        # Specificity = |within| / |cross| (use absolute values since Pearson can be negative)
        if abs(s2o) > 0.01:
            self_spec.append(abs(s2s) / abs(s2o))
        else:
            self_spec.append(np.nan)

        if abs(o2s) > 0.01:
            other_spec.append(abs(o2o) / abs(o2s))
        else:
            other_spec.append(np.nan)

    ax3.plot(layers, self_spec, 'o-', color='tab:blue', linewidth=2, markersize=4, label='self specificity')
    ax3.plot(layers, other_spec, 'o-', color='tab:orange', linewidth=2, markersize=4, label='other specificity')

    ax3.set_xlabel("Layer Index")
    ax3.set_ylabel("Specificity Ratio (|within| / |cross|)")
    ax3.set_ylim(0, max(5, np.nanmax(self_spec + other_spec) * 1.1) if self_spec or other_spec else 5)
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=GRID_ALPHA)

    # Panel 4: Summary text (bottom-right)
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary = results["summary"]
    interpretation = results["interpretation"]

    summary_text = f"""
SUMMARY (Best Layer: {summary['best_layer']})
{'='*40}

Within-Task Pearson r:
  self→self:   {summary['self_to_self_pearson']:.3f}
  other→other: {summary['other_to_other_pearson']:.3f}

Cross-Task Pearson r:
  self→other:  {summary['self_to_other_pearson']:.3f}
  other→self:  {summary['other_to_self_pearson']:.3f}

Specificity Ratios:
  self:  {summary['self_specificity']:.2f}x ({'>>' if summary['self_specificity'] > 1.5 else '≈'} 1)
  other: {summary['other_specificity']:.2f}x ({'>>' if summary['other_specificity'] > 1.5 else '≈'} 1)

{'='*40}
INTERPRETATION: {interpretation}
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, family='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print(f"Input base: {INPUT_BASE_NAME}")
    print(f"Train/test split: {TRAIN_SPLIT}")
    print(f"Bootstrap iterations: {N_BOOTSTRAP}")
    print()

    # Construct file paths
    self_probes_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_confidence_confidence_probes.joblib"
    other_probes_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_other_confidence_confidence_probes.joblib"
    self_acts_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_meta_confidence_activations.npz"
    other_acts_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_meta_other_confidence_activations.npz"

    # Check all required files exist
    missing = []
    for path, desc in [
        (self_probes_path, "self-confidence probes"),
        (other_probes_path, "other-confidence probes"),
        (self_acts_path, "self-confidence activations"),
        (other_acts_path, "other-confidence activations"),
    ]:
        if not path.exists():
            missing.append(f"  {desc}: {path}")

    if missing:
        print("ERROR: Missing required files:")
        print("\n".join(missing))
        print("\nRun these scripts first:")
        print("  1. test_meta_transfer.py with META_TASK='confidence'")
        print("  2. test_meta_transfer.py with META_TASK='other_confidence'")
        print("  3. identify_confidence_correlate.py with META_TASK='confidence'")
        print("  4. identify_confidence_correlate.py with META_TASK='other_confidence'")
        return

    # Load probes
    print(f"Loading self-confidence probes from {self_probes_path}...")
    self_probe_data = joblib.load(self_probes_path)
    self_probes = self_probe_data["probes"]
    print(f"  Found {len(self_probes)} layers")

    print(f"Loading other-confidence probes from {other_probes_path}...")
    other_probe_data = joblib.load(other_probes_path)
    other_probes = other_probe_data["probes"]
    print(f"  Found {len(other_probes)} layers")

    # Load activations
    print(f"\nLoading self-confidence activations from {self_acts_path}...")
    self_acts, self_conf, num_layers_self = load_activations_and_confidence(self_acts_path)
    print(f"  Shape: {self_acts[0].shape}, confidence: mean={self_conf.mean():.3f}, std={self_conf.std():.3f}")

    print(f"Loading other-confidence activations from {other_acts_path}...")
    other_acts, other_conf, num_layers_other = load_activations_and_confidence(other_acts_path)
    print(f"  Shape: {other_acts[0].shape}, confidence: mean={other_conf.mean():.3f}, std={other_conf.std():.3f}")

    num_layers = min(num_layers_self, num_layers_other, len(self_probes), len(other_probes))
    print(f"\nUsing {num_layers} layers")

    # Create train/test split (matching other scripts)
    n_self = len(self_conf)
    n_other = len(other_conf)
    print(f"  Self samples: {n_self}, Other samples: {n_other}")

    # Use same split as identify_confidence_correlate.py
    _, test_idx_self = train_test_split(
        np.arange(n_self), train_size=TRAIN_SPLIT, random_state=SEED, shuffle=True
    )
    _, test_idx_other = train_test_split(
        np.arange(n_other), train_size=TRAIN_SPLIT, random_state=SEED, shuffle=True
    )
    print(f"  Test set sizes: self={len(test_idx_self)}, other={len(test_idx_other)}")

    # Compute 2x2 transfer matrix at each layer
    print(f"\nComputing cross-prediction matrix...")
    results_by_layer = {}

    for layer in range(num_layers):
        if layer not in self_probes or layer not in other_probes:
            continue

        # Get probe components
        self_scaler = self_probes[layer]["scaler"]
        self_pca = self_probes[layer]["pca"]
        self_ridge = self_probes[layer]["probe"]

        other_scaler = other_probes[layer]["scaler"]
        other_pca = other_probes[layer]["pca"]
        other_ridge = other_probes[layer]["probe"]

        # Get test data
        X_self_test = self_acts[layer][test_idx_self]
        y_self_test = self_conf[test_idx_self]
        X_other_test = other_acts[layer][test_idx_other]
        y_other_test = other_conf[test_idx_other]

        # Compute 2x2 matrix
        # self→self: self probes on self data
        s2s = apply_probe_centered(X_self_test, y_self_test, self_scaler, self_pca, self_ridge)

        # self→other: self probes on other data
        s2o = apply_probe_centered(X_other_test, y_other_test, self_scaler, self_pca, self_ridge)

        # other→self: other probes on self data
        o2s = apply_probe_centered(X_self_test, y_self_test, other_scaler, other_pca, other_ridge)

        # other→other: other probes on other data
        o2o = apply_probe_centered(X_other_test, y_other_test, other_scaler, other_pca, other_ridge)

        # Bootstrap CIs (resample predictions)
        s2s_boot = bootstrap_r2(s2s["predictions"], y_self_test, N_BOOTSTRAP, SEED + layer)
        s2o_boot = bootstrap_r2(s2o["predictions"], y_other_test, N_BOOTSTRAP, SEED + layer + 1000)
        o2s_boot = bootstrap_r2(o2s["predictions"], y_self_test, N_BOOTSTRAP, SEED + layer + 2000)
        o2o_boot = bootstrap_r2(o2o["predictions"], y_other_test, N_BOOTSTRAP, SEED + layer + 3000)

        results_by_layer[layer] = {
            "self_to_self": {
                "r2": float(s2s["r2"]),
                "pearson": float(s2s["pearson"]),
                "mae": float(s2s["mae"]),
                "r2_ci_low": s2s_boot["ci_low"],
                "r2_ci_high": s2s_boot["ci_high"],
            },
            "self_to_other": {
                "r2": float(s2o["r2"]),
                "pearson": float(s2o["pearson"]),
                "mae": float(s2o["mae"]),
                "r2_ci_low": s2o_boot["ci_low"],
                "r2_ci_high": s2o_boot["ci_high"],
            },
            "other_to_self": {
                "r2": float(o2s["r2"]),
                "pearson": float(o2s["pearson"]),
                "mae": float(o2s["mae"]),
                "r2_ci_low": o2s_boot["ci_low"],
                "r2_ci_high": o2s_boot["ci_high"],
            },
            "other_to_other": {
                "r2": float(o2o["r2"]),
                "pearson": float(o2o["pearson"]),
                "mae": float(o2o["mae"]),
                "r2_ci_low": o2o_boot["ci_low"],
                "r2_ci_high": o2o_boot["ci_high"],
            },
        }

        if layer % 10 == 0:
            print(f"  Layer {layer}: s→s={s2s['pearson']:.3f}, s→o={s2o['pearson']:.3f}, o→s={o2s['pearson']:.3f}, o→o={o2o['pearson']:.3f}")

    # Find best layer (by average within-task Pearson)
    best_layer = max(
        results_by_layer.keys(),
        key=lambda l: (results_by_layer[l]["self_to_self"]["pearson"] + results_by_layer[l]["other_to_other"]["pearson"]) / 2
    )
    best_data = results_by_layer[best_layer]

    # Compute specificity ratios at best layer using Pearson (scale-invariant)
    s2s_pearson = best_data["self_to_self"]["pearson"]
    s2o_pearson = best_data["self_to_other"]["pearson"]
    o2s_pearson = best_data["other_to_self"]["pearson"]
    o2o_pearson = best_data["other_to_other"]["pearson"]

    # Use absolute values for specificity since Pearson can be negative
    self_specificity = abs(s2s_pearson) / abs(s2o_pearson) if abs(s2o_pearson) > 0.01 else float("inf")
    other_specificity = abs(o2o_pearson) / abs(o2s_pearson) if abs(o2s_pearson) > 0.01 else float("inf")

    # Generate interpretation
    if self_specificity > 1.5 and other_specificity > 1.5:
        interpretation = "Strong evidence for distinct self vs other representations"
    elif self_specificity > 1.2 or other_specificity > 1.2:
        interpretation = "Moderate evidence for distinct representations"
    else:
        interpretation = "Self and other confidence share similar representations"

    summary = {
        "best_layer": int(best_layer),
        "self_to_self_pearson": float(s2s_pearson),
        "self_to_other_pearson": float(s2o_pearson),
        "other_to_self_pearson": float(o2s_pearson),
        "other_to_other_pearson": float(o2o_pearson),
        "self_specificity": float(self_specificity),
        "other_specificity": float(other_specificity),
    }

    print(f"\n{'='*60}")
    print("CROSS-PREDICTION SUMMARY")
    print(f"{'='*60}")
    print(f"Best layer: {best_layer}")
    print(f"  self→self:   r={s2s_pearson:.3f}")
    print(f"  self→other:  r={s2o_pearson:.3f}")
    print(f"  other→self:  r={o2s_pearson:.3f}")
    print(f"  other→other: r={o2o_pearson:.3f}")
    print(f"\nSpecificity ratios (|within|/|cross|):")
    print(f"  self:  {self_specificity:.2f}x")
    print(f"  other: {other_specificity:.2f}x")
    print(f"\nInterpretation: {interpretation}")
    print(f"{'='*60}")

    # Compile full results
    results = {
        "config": get_config_dict(
            input_base=INPUT_BASE_NAME,
            train_split=TRAIN_SPLIT,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
            n_self_samples=int(n_self),
            n_other_samples=int(n_other),
            n_self_test=int(len(test_idx_self)),
            n_other_test=int(len(test_idx_other)),
            num_layers=int(num_layers),
        ),
        "by_layer": results_by_layer,
        "summary": summary,
        "interpretation": interpretation,
    }

    # Save JSON results
    results_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_confidence_cross_prediction.json"
    print(f"\nSaving results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot results
    plot_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_confidence_cross_prediction.png"
    print(f"Plotting results...")
    plot_cross_prediction_results(results, num_layers, plot_path)

    print("\nOutput files:")
    print(f"  {results_path.name}")
    print(f"  {plot_path.name}")


if __name__ == "__main__":
    main()
