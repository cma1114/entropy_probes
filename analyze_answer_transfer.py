"""
Answer Transfer Analysis: Test whether answer classifier confidence predicts delegation.

Hypothesis: If the model delegates when it lacks a clear answer signal, then
answer classifier confidence (from MC activations) should correlate with
delegation behavior (logit_margin) when applied to meta-task activations.

Inputs:
    outputs/{model_dir}/working/{dataset}_mc_answer_probes.joblib   Answer classifiers
    outputs/{model_dir}/working/{dataset}_mc_activations.npz        Direct-task activations
    outputs/{model_dir}/working/{dataset}_meta_{task}_activations.npz  Meta-task activations
    outputs/{model_dir}/results/{dataset}_mc_results.json           Per-question MC data

Outputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_answer_transfer_results_{pos}.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_answer_transfer_results_{pos}.png

Run after: identify_mc_correlate.py (with FIND_ANSWER_DIRECTIONS=True), test_meta_transfer.py
"""

import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from core import get_model_dir_name
from core.config_utils import get_config_dict, get_output_path, find_output_file
from core.directions import _safe_scale
from core.logging_utils import print_run_header, print_key_findings, print_run_footer
from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
DATASET = "TriviaMC_difficulty_filtered"
META_TASK = "delegate"  # delegate, confidence, other_confidence

# --- Quantization (must match Stage 1) ---
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

# --- Analysis ---
PROBE_POSITION = "final"  # Position to analyze
TARGET = "logit_margin"   # Target variable: "logit_margin", "p_answer", or "stated_confidence"

# --- Shared parameters (must match other scripts) ---
SEED = 42
TRAIN_SPLIT = 0.8
N_BOOTSTRAP = 100


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _bootstrap_corr_std(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> float:
    """Bootstrap std for Pearson r by resampling paired examples."""
    rng = np.random.RandomState(seed)
    n = len(a)
    if n < 3:
        return 0.0
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        aa = a[idx]
        bb = b[idx]
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            continue
        r, _ = pearsonr(aa, bb)
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.std(vals)) if len(vals) > 1 else 0.0


def plot_answer_transfer(
    results_by_layer: Dict,
    best_layer: int,
    answer_confidence_best: np.ndarray,
    target: np.ndarray,
    test_idx: np.ndarray,
    output_path: Path,
    target_name: str = "logit_margin",
):
    """Create 3-panel visualization of answer transfer results."""
    layers = sorted(results_by_layer.keys())

    d2d_acc = [results_by_layer[l]["d2d_accuracy"] for l in layers]
    d2m_acc = [results_by_layer[l]["d2m_accuracy"] for l in layers]
    conf_corr = [results_by_layer[l]["confidence_vs_target"]["pearson_r"] for l in layers]
    conf_ci_lo = [results_by_layer[l]["confidence_vs_target"]["ci_low"] for l in layers]
    conf_ci_hi = [results_by_layer[l]["confidence_vs_target"]["ci_high"] for l in layers]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Panel 1: D→D and D→M accuracy by layer
    ax = axes[0]
    ax.plot(layers, d2d_acc, 'o-', label='D→D (sanity)', alpha=0.7)
    ax.plot(layers, d2m_acc, 's-', label='D→M (transfer)', alpha=0.7)
    ax.axhline(0.25, color='gray', linestyle='--', alpha=0.5, label='Chance (25%)')
    ax.axvline(best_layer, color='red', linestyle=':', alpha=0.5, label=f'Best L{best_layer}')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title("Answer Classifier Transfer")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_ylim(0, 1)

    # Panel 2: Correlation with target by layer
    ax = axes[1]
    ax.plot(layers, conf_corr, 'o-', color='tab:blue')
    ax.fill_between(layers, conf_ci_lo, conf_ci_hi, alpha=CI_ALPHA, color='tab:blue')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(best_layer, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r")
    ax.set_title(f"Answer Confidence ↔ {target_name}")
    ax.grid(True, alpha=GRID_ALPHA)

    # Panel 3: Scatter at best layer
    ax = axes[2]
    valid_mask = ~np.isnan(target[test_idx])
    x = answer_confidence_best[test_idx][valid_mask]
    y = target[test_idx][valid_mask]
    ax.scatter(x, y, alpha=0.5, s=20)

    # Add regression line
    if len(x) > 2:
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x.min(), x.max(), 100)
        best_r = results_by_layer[best_layer]["confidence_vs_target"]["pearson_r"]
        ax.plot(x_line, p(x_line), 'r-', alpha=0.7, label=f'r={best_r:.3f}')
        ax.legend()

    ax.set_xlabel("Answer Confidence (max prob)")
    ax.set_ylabel(target_name)
    ax.set_title(f"Layer {best_layer} (best D→M)")
    ax.grid(True, alpha=GRID_ALPHA)

    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET
    base_output = f"{base_name}_meta_{META_TASK}"

    print_run_header(
        "analyze_answer_transfer.py",
        "Answer Transfer",
        "Test if answer classifier confidence predicts delegation",
        {"model": MODEL, "dataset": DATASET, "meta_task": META_TASK, "target": TARGET}
    )

    # =========================================================================
    # 1. LOAD CACHED DATA
    # =========================================================================

    print("Loading cached data...")

    # Load answer probes from Stage 1
    answer_probes_path = find_output_file(f"{base_name}_mc_answer_probes.joblib", model_dir=model_dir)
    if answer_probes_path is None:
        raise FileNotFoundError(
            f"Answer probes not found. Run identify_mc_correlate.py with FIND_ANSWER_DIRECTIONS=True first."
        )
    answer_probes_data = joblib.load(answer_probes_path)
    answer_probes = answer_probes_data["probes"]  # {layer: {"scaler", "pca", "clf"}}
    answer_mapping = answer_probes_data["metadata"]["answer_mapping"]  # {"A": 0, ...}
    num_layers = len(answer_probes)
    print(f"  Loaded answer probes: {num_layers} layers")

    # Load direct activations (for D→D sanity check)
    direct_path = find_output_file(f"{base_name}_mc_activations.npz", model_dir=model_dir)
    if direct_path is None:
        raise FileNotFoundError(f"Direct activations not found: {base_name}_mc_activations.npz")
    direct_data = np.load(direct_path)
    direct_activations = {i: direct_data[f"layer_{i}"] for i in range(num_layers)}
    n_samples = direct_activations[0].shape[0]
    print(f"  Loaded direct activations: {n_samples} samples")

    # Load MC results (for ground truth answers)
    mc_results_path = find_output_file(f"{base_name}_mc_results.json", model_dir=model_dir)
    if mc_results_path is None:
        raise FileNotFoundError(f"MC results not found: {base_name}_mc_results.json")
    with open(mc_results_path) as f:
        mc_results = json.load(f)
    predicted_answers = [q["predicted_answer"] for q in mc_results["dataset"]["data"]]
    y_answer = np.array([answer_mapping[a] for a in predicted_answers])  # (n_samples,)
    print(f"  Loaded MC results: {len(y_answer)} answers")

    # Load meta activations (cached by test_meta_transfer.py)
    meta_path = find_output_file(f"{base_output}_activations.npz", model_dir=model_dir)
    if meta_path is None:
        raise FileNotFoundError(
            f"Meta activations not found. Run test_meta_transfer.py first to cache them."
        )
    meta_data = np.load(meta_path)
    confidences = meta_data["confidences"]  # P(Answer) for delegate, stated_conf for others
    logit_margins = meta_data["logit_margins"]  # log(P(Answer)/(1-P(Answer))) or NaN

    # Load meta activations for the specified position
    meta_activations = {}
    for layer in range(num_layers):
        key = f"layer_{layer}_{PROBE_POSITION}"
        if key in meta_data:
            meta_activations[layer] = meta_data[key]
        else:
            raise KeyError(f"Position '{PROBE_POSITION}' not found in meta activations cache")

    # Validate sample counts match
    n_meta = meta_activations[0].shape[0]
    if n_meta != n_samples:
        raise ValueError(
            f"Sample count mismatch: direct={n_samples}, meta={n_meta}. "
            f"Ensure both were run on the same dataset."
        )
    print(f"  Loaded meta activations: {len(meta_activations)} layers, position={PROBE_POSITION}")

    # Select target variable
    if TARGET == "logit_margin":
        target = logit_margins
        target_name = "logit_margin"
    elif TARGET == "p_answer":
        target = confidences
        target_name = "P(Answer)"
    elif TARGET == "stated_confidence":
        target = confidences
        target_name = "stated_confidence"
    else:
        raise ValueError(f"Unknown TARGET: {TARGET}")

    print(f"  Target: {target_name}, mean={np.nanmean(target):.3f}, std={np.nanstd(target):.3f}")

    # Train/test split (must match Stage 1 - uses sklearn)
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED,
        shuffle=True
    )
    train_idx = np.sort(train_idx)
    test_idx = np.sort(test_idx)
    print(f"  Train/test split: {len(train_idx)}/{len(test_idx)}")

    # =========================================================================
    # 2. COMPUTE PER-QUESTION ANSWER CONFIDENCE
    # =========================================================================

    print("\nAnalyzing layers...")
    results_by_layer = {}
    answer_confidence_by_layer = {}
    answer_margin_by_layer = {}

    for layer in tqdm(range(num_layers), desc="Layers"):
        clf_info = answer_probes[layer]
        scaler = clf_info["scaler"]
        pca = clf_info["pca"]
        clf = clf_info["clf"]

        # --- D→D accuracy (sanity check) ---
        X_direct = direct_activations[layer]
        X_direct_scaled = scaler.transform(X_direct.astype(np.float32))
        X_direct_pca = pca.transform(X_direct_scaled)
        y_direct_pred = clf.predict(X_direct_pca)
        d2d_accuracy = (y_direct_pred[test_idx] == y_answer[test_idx]).mean()

        # --- D→M: Apply to meta activations with centered scaling ---
        X_meta = meta_activations[layer].astype(np.float32)
        X_meta_centered = X_meta - X_meta.mean(axis=0)
        X_meta_scaled = X_meta_centered / _safe_scale(scaler.scale_)
        X_meta_pca = pca.transform(X_meta_scaled)

        # D→M accuracy
        y_meta_pred = clf.predict(X_meta_pca)
        d2m_accuracy = (y_meta_pred[test_idx] == y_answer[test_idx]).mean()

        # Per-question confidence metrics
        proba = clf.predict_proba(X_meta_pca)  # (n_samples, 4)
        answer_confidence = proba.max(axis=1)   # Max probability
        sorted_proba = np.sort(proba, axis=1)
        answer_margin = sorted_proba[:, -1] - sorted_proba[:, -2]  # Top-2 gap

        answer_confidence_by_layer[layer] = answer_confidence
        answer_margin_by_layer[layer] = answer_margin

        # --- Correlation with target (test set only) ---
        valid_mask = ~np.isnan(target[test_idx])
        target_test = target[test_idx][valid_mask]
        conf_test = answer_confidence[test_idx][valid_mask]
        margin_test = answer_margin[test_idx][valid_mask]

        if len(target_test) < 3:
            conf_corr, conf_p = np.nan, np.nan
            margin_corr, margin_p = np.nan, np.nan
            conf_std, margin_std = 0.0, 0.0
        else:
            conf_corr, conf_p = pearsonr(conf_test, target_test)
            margin_corr, margin_p = pearsonr(margin_test, target_test)
            conf_std = _bootstrap_corr_std(conf_test, target_test, N_BOOTSTRAP, SEED + layer)
            margin_std = _bootstrap_corr_std(margin_test, target_test, N_BOOTSTRAP, SEED + 1000 + layer)

        results_by_layer[layer] = {
            "d2d_accuracy": float(d2d_accuracy),
            "d2m_accuracy": float(d2m_accuracy),
            "answer_confidence_mean": float(answer_confidence.mean()),
            "answer_confidence_std": float(answer_confidence.std()),
            "answer_margin_mean": float(answer_margin.mean()),
            "answer_margin_std": float(answer_margin.std()),
            "confidence_vs_target": {
                "pearson_r": float(conf_corr) if np.isfinite(conf_corr) else None,
                "pearson_p": float(conf_p) if np.isfinite(conf_p) else None,
                "pearson_std": float(conf_std),
                "ci_low": float(conf_corr - 1.96 * conf_std) if np.isfinite(conf_corr) else None,
                "ci_high": float(conf_corr + 1.96 * conf_std) if np.isfinite(conf_corr) else None,
            },
            "margin_vs_target": {
                "pearson_r": float(margin_corr) if np.isfinite(margin_corr) else None,
                "pearson_p": float(margin_p) if np.isfinite(margin_p) else None,
                "pearson_std": float(margin_std),
                "ci_low": float(margin_corr - 1.96 * margin_std) if np.isfinite(margin_corr) else None,
                "ci_high": float(margin_corr + 1.96 * margin_std) if np.isfinite(margin_corr) else None,
            },
        }

    # =========================================================================
    # 3. FIND BEST LAYER & SUMMARY
    # =========================================================================

    best_layer = max(
        results_by_layer.keys(),
        key=lambda l: results_by_layer[l]["d2m_accuracy"]
    )
    best_d2m = results_by_layer[best_layer]["d2m_accuracy"]
    best_corr = results_by_layer[best_layer]["confidence_vs_target"]["pearson_r"]
    best_ci_lo = results_by_layer[best_layer]["confidence_vs_target"]["ci_low"]
    best_ci_hi = results_by_layer[best_layer]["confidence_vs_target"]["ci_high"]

    # Spearman correlation at best layer
    valid_mask = ~np.isnan(target[test_idx])
    conf_best = answer_confidence_by_layer[best_layer][test_idx][valid_mask]
    target_best = target[test_idx][valid_mask]
    spearman_r, spearman_p = spearmanr(conf_best, target_best)

    # =========================================================================
    # 4. OUTPUT
    # =========================================================================

    output = {
        "config": get_config_dict(
            model=MODEL,
            adapter=ADAPTER,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            dataset=DATASET,
            meta_task=META_TASK,
            target=TARGET,
            probe_position=PROBE_POSITION,
            train_split=TRAIN_SPLIT,
            n_bootstrap=N_BOOTSTRAP,
            seed=SEED,
        ),
        "summary": {
            "n_samples": n_samples,
            "n_test": len(test_idx),
            "best_layer": best_layer,
            "best_d2m_accuracy": best_d2m,
            "behavioral_correlation": {
                "metric": "answer_confidence",
                "target": target_name,
                "pearson_r": float(best_corr) if best_corr is not None else None,
                "ci_low": float(best_ci_lo) if best_ci_lo is not None else None,
                "ci_high": float(best_ci_hi) if best_ci_hi is not None else None,
                "spearman_r": float(spearman_r),
                "spearman_p": float(spearman_p),
            },
        },
        "by_layer": {str(k): v for k, v in results_by_layer.items()},
        "per_question": [
            {
                "idx": int(i),
                "answer_confidence": float(answer_confidence_by_layer[best_layer][i]),
                "answer_margin": float(answer_margin_by_layer[best_layer][i]),
                "target_value": float(target[i]) if not np.isnan(target[i]) else None,
            }
            for i in test_idx
        ],
    }

    # Save JSON
    output_path = get_output_path(
        f"{base_output}_answer_transfer_results_{PROBE_POSITION}.json",
        model_dir=model_dir
    )
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Plot
    plot_path = output_path.with_suffix(".png")
    plot_answer_transfer(
        results_by_layer,
        best_layer,
        answer_confidence_by_layer[best_layer],
        target,
        test_idx,
        plot_path,
        target_name=target_name,
    )

    # Console summary
    key_findings = {
        "Best D→M accuracy": f"{best_d2m:.1%} at layer {best_layer}",
        f"Correlation (r) with {target_name}": f"{best_corr:.3f} [{best_ci_lo:.3f}, {best_ci_hi:.3f}]" if best_corr else "N/A",
        "Spearman ρ": f"{spearman_r:.3f} (p={spearman_p:.2e})",
    }
    print_key_findings(key_findings)
    print_run_footer([output_path, plot_path], None)


if __name__ == "__main__":
    main()
