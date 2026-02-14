"""
Stage 2. Test direct-to-meta (D->M) transfer of uncertainty directions, and
optionally find confidence directions from meta-task activations. Tests the core
introspection hypothesis: does the same uncertainty representation appear when
the model reports its confidence?

Uses two scaling approaches for D->M transfer:
1. Centered Scaler (Rigorous): Center meta with own mean, scale with direct's std
2. Separate Scaler (Upper Bound): Refit scaler on meta data (domain adaptation)

Inputs:
    outputs/{model_dir}/working/{dataset}_mc_activations.npz      Direct activations (from Stage 1)
    outputs/{model_dir}/working/{dataset}_mc_{metric}_directions.npz  Uncertainty directions
    outputs/{model_dir}/results/{dataset}_mc_results.json         Consolidated results

Outputs (per-position, where {pos} = "final", "options_newline", etc.):
    outputs/{model_dir}/results/{dataset}_meta_{task}_transfer_results_{pos}.json   Transfer R², CIs
    outputs/{model_dir}/results/{dataset}_meta_{task}_transfer_results_{pos}.npz    Transfer data
    outputs/{model_dir}/results/{dataset}_meta_{task}_transfer_results_{pos}.png     Transfer plots
    outputs/{model_dir}/working/{dataset}_meta_{task}_activations.npz               Meta activations (all positions)
    outputs/{model_dir}/working/{dataset}_meta_{task}_confdir_directions_{pos}.npz  Confidence directions
    outputs/{model_dir}/results/{dataset}_meta_{task}_confdir_results_{pos}.json    Confidence results
    outputs/{model_dir}/working/{dataset}_meta_{task}_mcuncert_directions_{pos}.npz MC uncertainty dirs
    outputs/{model_dir}/results/{dataset}_meta_{task}_mcuncert_results_{pos}.json   MC uncertainty results
    outputs/{model_dir}/results/{dataset}_meta_{task}_position_comparison.png       Position comparison (if multi-position)

    where {model_dir} = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)

Shared parameters (must match across scripts):
    SEED, PROBE_ALPHA, PROBE_PCA_COMPONENTS, TRAIN_SPLIT, MEAN_DIFF_QUANTILE

Run after: identify_mc_correlate.py
"""

from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
import joblib
import random

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    get_model_dir_name,
    should_use_chat_template,
    BatchedExtractor,
    apply_probe_shared,
    apply_probe_centered,
    apply_probe_separate,
    metric_sign_for_confidence,
    print_run_header,
    print_key_findings,
    print_run_footer,
    format_r2_with_ci,
)
from core.directions import probe_direction
from core.config_utils import get_config_dict, get_output_path, find_output_file, glob_outputs
from core.plotting import save_figure, METHOD_COLORS, GRID_ALPHA, CI_ALPHA
from core.answer_directions import (
    apply_answer_classifier_centered,
    apply_answer_classifier_separate,
    encode_answers,
)
from core.confidence_directions import (
    find_confidence_directions_both_methods,
    find_mc_uncertainty_directions_from_meta,
    compare_confidence_to_uncertainty,
)
from tasks import (
    format_stated_confidence_prompt,
    format_answer_or_delegate_prompt,
    format_other_confidence_prompt,
    get_stated_confidence_signal,
    get_answer_or_delegate_signal,
    get_other_confidence_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
    OTHER_CONFIDENCE_OPTIONS,
    find_mc_positions,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
#
# CACHING WARNING: Meta-task activations and P(Answer)/confidence values are
# cached at: outputs/{model_dir}/working/{dataset}_meta_{META_TASK}_activations.npz
#
# If you modify the prompts in tasks.py (e.g., teammate performance), you MUST
# delete the cache file to force re-extraction. Otherwise the script will
# silently use stale data from the old prompts.
#
# =============================================================================

# --- Model & Data ---
# Must match the identify_mc_correlate.py settings that produced Stage 1 files
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DATASET = "TriviaMC_difficulty_filtered"
ADAPTER = None  # Optional: must match identify step if used
METRICS = ["logit_gap", "entropy"]  # Which metrics to test transfer for
META_TASK = "confidence"  # "confidence", "other_confidence", or "delegate"

# --- Quantization ---
LOAD_IN_4BIT = False  # Set True for 70B+ models
LOAD_IN_8BIT = False

# --- Experiment ---
SEED = 42                    # Must match across scripts
BATCH_SIZE = 4 if META_TASK == "delegate" else 8  # Delegate uses longer prompts
N_BOOTSTRAP = 100            # Bootstrap iterations for confidence intervals

# --- Direction-finding (must match across scripts) ---
PROBE_ALPHA = 1000.0         # Must match across scripts
PROBE_PCA_COMPONENTS = 100   # Must match across scripts
TRAIN_SPLIT = 0.8            # Must match across scripts

# --- Confidence directions (optional, merged from identify_confidence_correlate.py) ---
FIND_CONFIDENCE_DIRECTIONS = True  # Train probes on stated confidence from meta activations
MEAN_DIFF_QUANTILE = 0.25          # Must match across scripts
COMPARE_UNCERTAINTY_METRIC = METRICS[0]  # Compare confidence vs uncertainty dirs (None to skip)

# --- MC Uncertainty directions from meta activations ---
# Train probes on meta-task activations to predict MC uncertainty (logit_gap, entropy)
# This finds d_meta→mc_uncertainty and compares to d_mc_uncertainty via cosine similarity
FIND_MC_UNCERTAINTY_DIRECTIONS = True
MC_UNCERTAINTY_METRICS = ["logit_gap", "entropy"]  # Which MC metrics to predict from meta activations

# --- Script-specific ---
# Token positions to probe for transfer
# question_mark: "?" at end of embedded MC question
# question_newline: newline after "?"
# options_newline: newline after last MC option (D: ...)
# final: last token (current behavior)
PROBE_POSITIONS = ["final"]  # ["question_mark", "question_newline", "options_newline", "final"]

# --- Output ---
# Uses centralized path management from core.config_utils

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def compute_behavioral_stats(
    confidences: np.ndarray,
    option_probs: np.ndarray = None,
    meta_task: str = "delegate"
) -> dict:
    """
    Compute behavioral statistics for the meta-task.

    Args:
        confidences: (n_samples,) array of confidence signals
                     For delegate: P(Answer) in [0, 1]
                     For confidence: expected confidence from weighted sum
        option_probs: (n_samples, n_options) array of raw option probabilities
                      For delegate: (n, 2) for ["1", "2"]
                      For confidence: (n, 8) for ["S", "T", ..., "Z"]
        meta_task: "delegate" or "confidence"

    Returns:
        dict with behavioral statistics
    """
    stats = {"n_samples": len(confidences)}

    if meta_task == "delegate":
        # For delegate task: confidences = P(Answer)
        delegation_rate = float((confidences < 0.5).mean())
        answer_rate = float((confidences >= 0.5).mean())
        stats["delegation_rate"] = delegation_rate
        stats["answer_rate"] = answer_rate
        stats["P_answer_mean"] = float(confidences.mean())
        stats["P_answer_std"] = float(confidences.std())
        stats["P_answer_min"] = float(confidences.min())
        stats["P_answer_max"] = float(confidences.max())
        stats["P_answer_median"] = float(np.median(confidences))
        stats["P_answer_quantiles"] = {
            "q10": float(np.percentile(confidences, 10)),
            "q25": float(np.percentile(confidences, 25)),
            "q50": float(np.percentile(confidences, 50)),
            "q75": float(np.percentile(confidences, 75)),
            "q90": float(np.percentile(confidences, 90)),
        }
    else:
        # For confidence task: confidences = weighted sum of S-Z midpoints
        stats["confidence_mean"] = float(confidences.mean())
        stats["confidence_std"] = float(confidences.std())
        stats["confidence_min"] = float(confidences.min())
        stats["confidence_max"] = float(confidences.max())
        stats["confidence_median"] = float(np.median(confidences))
        stats["confidence_quantiles"] = {
            "q10": float(np.percentile(confidences, 10)),
            "q25": float(np.percentile(confidences, 25)),
            "q50": float(np.percentile(confidences, 50)),
            "q75": float(np.percentile(confidences, 75)),
            "q90": float(np.percentile(confidences, 90)),
        }

        # If we have raw option probs, compute response distribution
        if option_probs is not None:
            option_names = list(STATED_CONFIDENCE_OPTIONS.keys())  # ["S", "T", ..., "Z"]

            # Mean probability mass on each option across samples
            mean_probs = option_probs.mean(axis=0)
            stats["mean_option_probs"] = {
                name: float(mean_probs[i]) for i, name in enumerate(option_names)
            }

            # Modal response distribution (argmax for each sample)
            modal_responses = np.argmax(option_probs, axis=1)
            modal_counts = np.bincount(modal_responses, minlength=len(option_names))
            stats["modal_response_counts"] = {
                name: int(modal_counts[i]) for i, name in enumerate(option_names)
            }
            stats["modal_response_rates"] = {
                name: float(modal_counts[i] / len(confidences))
                for i, name in enumerate(option_names)
            }

    return stats


def load_probes(probes_path: Path) -> dict:
    """Load probe pipeline from a _probes.joblib file."""
    data = joblib.load(probes_path)
    return data


def _find_directions_npz(dataset: str, metric: str, model_dir: str) -> Path:
    """Locate the *_directions.npz file produced by identify_mc_correlate.py for a metric."""
    candidate_names = [
        f"{dataset}_mc_{metric}_directions.npz",
        f"{dataset}_{metric}_directions.npz",
    ]
    for name in candidate_names:
        path = find_output_file(name, model_dir=model_dir)
        if path.exists():
            return path
    # Fallback: loose glob (keeps things robust to naming tweaks)
    matches = sorted(glob_outputs(f"*{metric}*directions*.npz", model_dir=model_dir))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"Could not find directions npz for metric='{metric}'. Tried: {candidate_names} and glob."
    )


def load_mean_diff_directions(directions_path: Path, num_layers: int) -> dict[int, np.ndarray]:
    """Load mean-diff direction vectors from a *_directions.npz file."""
    data = np.load(directions_path)
    dirs: dict[int, np.ndarray] = {}
    for layer in range(num_layers):
        key = f"mean_diff_layer_{layer}"
        if key not in data:
            continue
        v = np.asarray(data[key], dtype=np.float32)
        n = float(np.linalg.norm(v))
        if n > 0:
            v = v / n
        dirs[layer] = v
    return dirs


def _bootstrap_corr_std(a: np.ndarray, b: np.ndarray, n_bootstrap: int = 100, seed: int = 42) -> float:
    """Cheap bootstrap std for Pearson r by resampling paired examples."""
    rng = np.random.RandomState(seed)
    a = np.asarray(a)
    b = np.asarray(b)
    n = len(a)
    vals = []
    for _ in range(int(n_bootstrap)):
        idx = rng.choice(n, n, replace=True)
        aa = a[idx]
        bb = b[idx]
        if np.std(aa) < 1e-12 or np.std(bb) < 1e-12:
            continue
        r, _ = pearsonr(aa, bb)
        if np.isfinite(r):
            vals.append(float(r))
    return float(np.std(vals)) if len(vals) > 1 else 0.0

def bootstrap_transfer_r2(
    activations: np.ndarray,
    targets: np.ndarray,
    scaler,
    pca,
    ridge,
    scaling: str,  # "centered" or "separate"
    n_bootstrap: int = 100,
    n_boot: int | None = None,
    seed: int = 42,) -> tuple:
    """
    Bootstrap uncertainty for transfer performance.

    Important: vanilla R² on bootstrap resamples can explode negative when the resampled
    `targets` have unusually low variance (SS_tot ≈ 0). That can make plots look
    like "garbage" even when the underlying predictions are fine.

    So we compute a *stable* out-of-sample R² using a fixed denominator equal to the
    variance of the *original* targets (the non-resampled test set):

        R²_stable = 1 - MSE_boot / Var(targets_original)

    This preserves the "R² scale" (<= 1, negative when worse than mean predictor),
    but avoids pathological -100/-1000 values caused purely by bootstrap variance collapse.

    Returns:
        (mean_r2, std_r2): mean and std over bootstrap replicates of R²_stable
    """
    rng = np.random.RandomState(seed)

    # Backward-compat alias: allow callers to pass n_boot
    if n_boot is not None:
        n_bootstrap = int(n_boot)

    targets = np.asarray(targets, dtype=np.float64)
    n_samples = targets.shape[0]

    var_full = float(np.var(targets))
    if not np.isfinite(var_full) or var_full < 1e-12:
        # Degenerate target variance: R² is not meaningful; fall back to MSE scale.
        var_full = 1e-12

    r2s = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, n_samples, replace=True)
        acts_boot = activations[idx]
        y_boot = targets[idx]

        if scaling == "centered":
            result = apply_probe_centered(acts_boot, y_boot, scaler, pca, ridge)
        else:
            result = apply_probe_separate(acts_boot, y_boot, pca, ridge)

        y_pred = np.asarray(result["predictions"], dtype=np.float64)

        mse = float(np.mean((y_boot - y_pred) ** 2))
        r2 = 1.0 - (mse / var_full)

        # Guard tiny numerical overshoot (should be <= 1 for this definition)
        if r2 > 1.0 and r2 < 1.0 + 1e-6:
            r2 = 1.0

        if np.isfinite(r2):
            r2s.append(r2)

    if len(r2s) == 0:
        return float("nan"), float("nan")

    r2s = np.array(r2s, dtype=np.float64)
    return float(r2s.mean()), float(r2s.std(ddof=0))


def bootstrap_r2_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 100,
    n_boot: int | None = None,
    seed: int = 42,) -> tuple:
    """
    Cheap bootstrap uncertainty for R² when the model is already trained.

    We resample *test examples only* and recompute a stable R² using a fixed
    denominator Var(y_true_full) to avoid pathological negative explosions when
    a bootstrap resample happens to have very low target variance.

        R²_stable = 1 - MSE_boot / Var(y_true_full)

    Returns:
        (mean_r2, std_r2) over bootstrap replicates.
    """
    rng = np.random.RandomState(seed)

    # Backward-compat alias: allow callers to pass n_boot
    if n_boot is not None:
        n_bootstrap = int(n_boot)

    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    n = y_true.shape[0]
    if n == 0:
        return float("nan"), float("nan")

    var_full = float(np.var(y_true))
    if not np.isfinite(var_full) or var_full < 1e-12:
        var_full = 1e-12

    r2s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        mse = float(np.mean((yt - yp) ** 2))
        r2 = 1.0 - (mse / var_full)
        if r2 > 1.0 and r2 < 1.0 + 1e-6:
            r2 = 1.0
        if np.isfinite(r2):
            r2s.append(r2)

    if len(r2s) == 0:
        return float("nan"), float("nan")
    r2s = np.array(r2s, dtype=np.float64)
    return float(r2s.mean()), float(r2s.std(ddof=0))


def bootstrap_accuracy_from_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42,
) -> tuple:
    """
    Bootstrap CI for classification accuracy by resampling test examples.

    Resamples test indices and recomputes accuracy on each resample.

    Returns:
        (mean_acc, std_acc) over bootstrap replicates.
    """
    rng = np.random.RandomState(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")

    accs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        acc = float((y_true[idx] == y_pred[idx]).mean())
        accs.append(acc)

    accs = np.array(accs)
    return float(accs.mean()), float(accs.std(ddof=0))


def load_dataset(dataset_path: Path) -> dict:
    """Load consolidated mc_results.json with questions and metric values."""
    with open(dataset_path) as f:
        data = json.load(f)

    # Extract questions from nested structure (consolidated format)
    questions = data["dataset"]["data"]

    # Extract metric values as arrays
    metric_values = {}
    for item in questions:
        for key, val in item.items():
            if key in ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]:
                if key not in metric_values:
                    metric_values[key] = []
                metric_values[key].append(val)

    # Convert to numpy
    for key in metric_values:
        metric_values[key] = np.array(metric_values[key])

    return {
        "config": data["config"],
        "stats": data["dataset"]["stats"],
        "questions": questions,
        "metric_values": metric_values,
    }


def get_meta_format_fn(meta_task: str):
    """Get the prompt formatting function for a meta task."""
    if meta_task == "confidence":
        return format_stated_confidence_prompt
    elif meta_task == "delegate":
        return format_answer_or_delegate_prompt
    elif meta_task == "other_confidence":
        return format_other_confidence_prompt
    else:
        raise ValueError(f"Unknown meta task: {meta_task}. Valid options: confidence, delegate, other_confidence")


def get_meta_signal_fn(meta_task: str):
    """Get the signal extraction function for a meta task."""
    if meta_task == "confidence":
        return lambda probs, mapping: get_stated_confidence_signal(probs)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    elif meta_task == "other_confidence":
        return lambda probs, mapping: get_other_confidence_signal(probs)
    else:
        raise ValueError(f"Unknown meta task: {meta_task}. Valid options: confidence, delegate, other_confidence")


def get_meta_options(meta_task: str):
    """Get option tokens for a meta task."""
    if meta_task == "confidence":
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    elif meta_task == "other_confidence":
        return list(OTHER_CONFIDENCE_OPTIONS.keys())
    else:
        raise ValueError(f"Unknown meta task: {meta_task}. Valid options: confidence, delegate, other_confidence")


def plot_combined_transfer_results_consolidated(
    probe_results: dict,
    probe_direct_r2: dict,
    probe_direct_r2_std: dict,
    mean_diff_results: dict,
    mean_diff_direct_r2: dict,
    mean_diff_direct_r2_std: dict,
    behavioral: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
    metrics: list,
    position: str,
    answer_d2d: dict = None,
    answer_d2m: dict = None,
):
    """
    Consolidated transfer plot with all metrics in one figure.

    Creates a grid with one row per metric, 4 columns:
    - Column 1: Transferred signal → stated confidence (Pearson r)
    - Column 2: Probe Transfer (D→D vs D→M R²)
    - Column 3: Mean-diff Transfer (D→D vs D→M R²)
    - Column 4: Transfer Ratio (D→M / D→D)

    Colors by method (not metric):
    - probe = blue (solid for D→M, dotted for D→D)
    - mean_diff = orange (solid for D→M, dotted for D→D)
    - answer = black (solid for D→M, dotted for D→D)
    """
    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    fig, axes = plt.subplots(n_metrics, 4, figsize=(20, 5 * n_metrics), squeeze=False)
    fig.suptitle(f"Transfer Analysis ({position}): {meta_task}", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))

    PROBE_COLOR = METHOD_COLORS["probe"]
    MEANDIFF_COLOR = METHOD_COLORS["mean_diff"]
    ANSWER_COLOR = "black"
    CHANCE_LEVEL = 0.25

    R2_PLOT_FLOOR = -0.5
    def _clip_r2_for_plot(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, R2_PLOT_FLOOR, 1.0)

    def _safe_ratio(d2m, d2d, min_denom=0.01):
        ratio = np.full_like(d2m, np.nan)
        valid = (d2d > min_denom) & np.isfinite(d2m)
        ratio[valid] = d2m[valid] / d2d[valid]
        return np.clip(ratio, 0, 2.0)

    for row, metric in enumerate(metrics):
        if not probe_results or metric not in probe_results:
            continue

        # === Column 1: Transferred signal → stated confidence ===
        ax1 = axes[row, 0]
        ax1.set_title(f"Signal → Confidence ({metric})", fontsize=10)

        if metric in probe_results:
            vals = np.array([probe_results[metric].get(l, {}).get("centered", {}).get("pred_conf_pearson", np.nan) for l in layers], dtype=float)
            ax1.plot(layers, vals, '-', color=PROBE_COLOR, linewidth=2, label='probe')
            stds = np.array([probe_results[metric].get(l, {}).get("centered", {}).get("pred_conf_pearson_std", 0.0) for l in layers], dtype=float)
            if np.any(stds > 0):
                ax1.fill_between(layers, vals - stds, vals + stds, color=PROBE_COLOR, alpha=CI_ALPHA, linewidth=0)

        if mean_diff_results and metric in mean_diff_results:
            vals = np.array([mean_diff_results[metric].get(l, {}).get("centered", {}).get("pred_conf_pearson", np.nan) for l in layers], dtype=float)
            ax1.plot(layers, vals, '--', color=MEANDIFF_COLOR, linewidth=2, label='mean-diff')
            stds = np.array([mean_diff_results[metric].get(l, {}).get("centered", {}).get("pred_conf_pearson_std", 0.0) for l in layers], dtype=float)
            if np.any(stds > 0):
                ax1.fill_between(layers, vals - stds, vals + stds, color=MEANDIFF_COLOR, alpha=CI_ALPHA, linewidth=0)

        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Pearson r')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=GRID_ALPHA)

        # === Column 2: Probe Transfer (D→D vs D→M) ===
        ax2 = axes[row, 1]
        ax2.set_title(f"Probe Transfer ({metric})", fontsize=10)

        if metric in probe_direct_r2:
            d2d_r2 = _clip_r2_for_plot(np.array([probe_direct_r2[metric].get(l, 0) for l in layers]))
            ax2.plot(layers, d2d_r2, ':', color=PROBE_COLOR, linewidth=2, alpha=0.6, label='D→D')
            d2d_std = np.array([probe_direct_r2_std.get(metric, {}).get(l, 0) for l in layers])
            if np.any(d2d_std > 0):
                ax2.fill_between(layers, _clip_r2_for_plot(d2d_r2 - d2d_std),
                               _clip_r2_for_plot(d2d_r2 + d2d_std), color=PROBE_COLOR, alpha=CI_ALPHA * 0.5)

        if metric in probe_results:
            centered_r2 = np.array([probe_results[metric].get(l, {}).get("centered", {}).get("r2", np.nan) for l in layers], dtype=float)
            ax2.plot(layers, _clip_r2_for_plot(centered_r2), '-', color=PROBE_COLOR, linewidth=2, label='D→M')
            centered_std = np.array([probe_results[metric].get(l, {}).get("centered", {}).get("r2_std", 0) for l in layers])
            if np.any(centered_std > 0):
                ax2.fill_between(layers, _clip_r2_for_plot(centered_r2 - centered_std),
                               _clip_r2_for_plot(centered_r2 + centered_std), color=PROBE_COLOR, alpha=CI_ALPHA)

        if answer_d2d and answer_d2m:
            ax2b = ax2.twinx()
            d2d_acc = np.array([answer_d2d.get(l, {}).get("accuracy", np.nan) for l in layers])
            d2m_acc = np.array([answer_d2m.get(l, {}).get("centered", {}).get("accuracy", np.nan) for l in layers])
            ax2b.plot(layers, d2d_acc, ':', color=ANSWER_COLOR, linewidth=2, alpha=0.6, label='D→D ans')
            ax2b.plot(layers, d2m_acc, '-', color=ANSWER_COLOR, linewidth=2, label='D→M ans')
            ax2b.axhline(y=CHANCE_LEVEL, color='gray', linestyle=':', alpha=0.3)
            ax2b.set_ylabel('Answer Acc', color=ANSWER_COLOR, fontsize=8)
            ax2b.set_ylim(0, 1.0)

        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('R²', color=PROBE_COLOR)
        ax2.legend(loc='upper left', fontsize=8)
        ax2.grid(True, alpha=GRID_ALPHA)

        # === Column 3: Mean-diff Transfer (D→D vs D→M) ===
        ax3 = axes[row, 2]
        ax3.set_title(f"Mean-diff Transfer ({metric})", fontsize=10)

        if metric in mean_diff_direct_r2:
            d2d_r2 = _clip_r2_for_plot(np.array([mean_diff_direct_r2[metric].get(l, 0) for l in layers]))
            ax3.plot(layers, d2d_r2, ':', color=MEANDIFF_COLOR, linewidth=2, alpha=0.6, label='D→D')
            d2d_std = np.array([mean_diff_direct_r2_std.get(metric, {}).get(l, 0) for l in layers])
            if np.any(d2d_std > 0):
                ax3.fill_between(layers, _clip_r2_for_plot(d2d_r2 - d2d_std),
                               _clip_r2_for_plot(d2d_r2 + d2d_std), color=MEANDIFF_COLOR, alpha=CI_ALPHA * 0.5)

        if mean_diff_results and metric in mean_diff_results:
            centered_r2 = np.array([mean_diff_results[metric].get(l, {}).get("centered", {}).get("r2", np.nan) for l in layers], dtype=float)
            ax3.plot(layers, _clip_r2_for_plot(centered_r2), '-', color=MEANDIFF_COLOR, linewidth=2, label='D→M')
            centered_std = np.array([mean_diff_results[metric].get(l, {}).get("centered", {}).get("r2_std", 0) for l in layers])
            if np.any(centered_std > 0):
                ax3.fill_between(layers, _clip_r2_for_plot(centered_r2 - centered_std),
                               _clip_r2_for_plot(centered_r2 + centered_std), color=MEANDIFF_COLOR, alpha=CI_ALPHA)

        if answer_d2d and answer_d2m:
            ax3b = ax3.twinx()
            d2d_acc = np.array([answer_d2d.get(l, {}).get("accuracy", np.nan) for l in layers])
            d2m_acc = np.array([answer_d2m.get(l, {}).get("centered", {}).get("accuracy", np.nan) for l in layers])
            ax3b.plot(layers, d2d_acc, ':', color=ANSWER_COLOR, linewidth=2, alpha=0.6, label='D→D ans')
            ax3b.plot(layers, d2m_acc, '-', color=ANSWER_COLOR, linewidth=2, label='D→M ans')
            ax3b.axhline(y=CHANCE_LEVEL, color='gray', linestyle=':', alpha=0.3)
            ax3b.set_ylabel('Answer Acc', color=ANSWER_COLOR, fontsize=8)
            ax3b.set_ylim(0, 1.0)

        ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Layer')
        ax3.set_ylabel('R²', color=MEANDIFF_COLOR)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=GRID_ALPHA)

        # === Column 4: Transfer Ratio ===
        ax4 = axes[row, 3]
        ax4.set_title(f"Transfer Ratio ({metric})", fontsize=10)

        if metric in probe_direct_r2 and metric in probe_results:
            d2d_r2 = np.array([probe_direct_r2[metric].get(l, 0) for l in layers], dtype=float)
            d2m_r2 = np.array([probe_results[metric].get(l, {}).get("centered", {}).get("r2", np.nan) for l in layers], dtype=float)
            ax4.plot(layers, _safe_ratio(d2m_r2, d2d_r2), '-', color=PROBE_COLOR, linewidth=2, label='probe')

        if metric in mean_diff_direct_r2 and mean_diff_results and metric in mean_diff_results:
            d2d_r2 = np.array([mean_diff_direct_r2[metric].get(l, 0) for l in layers], dtype=float)
            d2m_r2 = np.array([mean_diff_results[metric].get(l, {}).get("centered", {}).get("r2", np.nan) for l in layers], dtype=float)
            ax4.plot(layers, _safe_ratio(d2m_r2, d2d_r2), '--', color=MEANDIFF_COLOR, linewidth=2, label='mean-diff')

        if answer_d2d and answer_d2m:
            d2d_acc = np.array([answer_d2d.get(l, {}).get("accuracy", np.nan) for l in layers], dtype=float)
            d2m_acc = np.array([answer_d2m.get(l, {}).get("centered", {}).get("accuracy", np.nan) for l in layers], dtype=float)
            ax4.plot(layers, _safe_ratio(d2m_acc, d2d_acc, min_denom=0.1), '-', color=ANSWER_COLOR, linewidth=1.5, label='answer')

        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        ax4.axhline(y=0, color='gray', linestyle=':', alpha=0.3)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('D→M / D→D')
        ax4.set_ylim(-0.1, 2.1)
        ax4.legend(loc='upper left', fontsize=8)
        ax4.grid(True, alpha=GRID_ALPHA)

    # Add behavioral correlation text box
    behav_lines = ["Metric ↔ Confidence:"]
    for metric in metrics:
        r_val = behavioral.get(metric, {}).get('test_pearson_r', float('nan'))
        behav_lines.append(f"  {metric}: r={r_val:.3f}")
    fig.text(0.02, 0.02, "\n".join(behav_lines), fontsize=8, family='monospace',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    save_figure(fig, output_path)


def plot_confidence_directions(
    conf_results: dict,
    conf_unc_comparison: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
    compare_metric: str = None,
):
    """
    Plot meta-task confidence direction results (trained on meta activations).

    Shows probe and mean-diff R² by layer, with cosine similarity between methods,
    and optionally comparison to transferred uncertainty directions.

    Args:
        conf_results: Results from find_confidence_directions_both_methods()
        conf_unc_comparison: Dict {"probe": {layer: {...}}, "mean_diff": {layer: {...}}}
            comparing confidence directions to uncertainty directions
        num_layers: Number of model layers
        output_path: Where to save the figure
        meta_task: Name of meta-task for title
        compare_metric: Which uncertainty metric was compared (e.g., "entropy")
    """
    layers = list(range(num_layers))

    # Determine layout based on whether we have uncertainty comparison
    n_cols = 3 if conf_unc_comparison else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    if n_cols == 2:
        axes = list(axes) + [None]  # pad for uniform indexing
    fig.suptitle(f"Meta-task Confidence Directions: {meta_task}", fontsize=14, fontweight='bold')

    # Panel 1: R² by layer for probe and mean-diff
    ax1 = axes[0]
    ax1.set_title("Confidence R² by Layer", fontsize=10)

    for method, color, ls in [("probe", "tab:blue", "-"), ("mean_diff", "tab:orange", "--")]:
        fits = conf_results["fits"][method]
        r2_vals = [fits[l]["test_r2"] for l in layers]
        ci_low = [fits[l].get("test_r2_ci_low", np.nan) for l in layers]
        ci_high = [fits[l].get("test_r2_ci_high", np.nan) for l in layers]

        best_layer = max(layers, key=lambda l: fits[l]["test_r2"])
        best_r2 = fits[best_layer]["test_r2"]

        ax1.plot(layers, r2_vals, ls, color=color, linewidth=2,
                 label=f'{method} (L{best_layer}: {best_r2:.3f})')
        if not all(np.isnan(ci_low)):
            ax1.fill_between(layers, ci_low, ci_high, color=color, alpha=CI_ALPHA)

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² (test set)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=GRID_ALPHA)

    # Panel 2: Cosine similarity between probe and mean-diff directions
    ax2 = axes[1]
    ax2.set_title("Probe vs Mean-diff Cosine Similarity", fontsize=10)

    cosine_sims = [conf_results["comparison"][l]["cosine_sim"] for l in layers]
    ax2.plot(layers, cosine_sims, '-', color='tab:purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=GRID_ALPHA)

    # Panel 3: Comparison with transferred uncertainty directions (if available)
    if conf_unc_comparison and axes[2] is not None:
        ax3 = axes[2]
        metric_label = compare_metric if compare_metric else "Uncertainty"
        ax3.set_title(f"Confidence vs {metric_label} Directions", fontsize=10)

        # Plot cosine similarity with uncertainty directions
        # conf_unc_comparison has structure: {"probe": {layer: {"cosine_similarity": ...}}, "mean_diff": {...}}
        for method, color, ls in [("probe", "tab:blue", "-"), ("mean_diff", "tab:orange", "--")]:
            method_comparison = conf_unc_comparison.get(method, {})
            sims = [method_comparison.get(l, {}).get("cosine_similarity", np.nan) for l in layers]
            if not all(np.isnan(sims)):
                ax3.plot(layers, sims, ls, color=color, linewidth=2, label=f'{method}')

        ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_ylim(-1.1, 1.1)
        ax3.legend(loc='upper left', fontsize=8)
        ax3.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_mc_uncertainty_from_meta(
    mc_uncert_results: dict,
    mc_dir_comparison: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
    mc_metric: str,
):
    """
    Plot MC uncertainty direction results (trained on meta activations → MC uncertainty).

    Shows R² for predicting MC uncertainty from meta-task activations, cosine similarity
    between probe and mean-diff methods, and importantly the comparison to original
    d_mc_uncertainty directions.

    Args:
        mc_uncert_results: Results from find_mc_uncertainty_directions_from_meta()
        mc_dir_comparison: Dict {"probe": {layer: {...}}, "mean_diff": {layer: {...}}}
            comparing new directions to original MC uncertainty directions
        num_layers: Number of model layers
        output_path: Where to save the figure
        meta_task: Name of meta-task for title (source of activations)
        mc_metric: Which MC metric was predicted (e.g., "logit_gap")
    """
    layers = list(range(num_layers))

    # Always 3 panels: R², method comparison, MC direction comparison
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Meta→MC Uncertainty Directions: {meta_task} activations → {mc_metric}",
                 fontsize=14, fontweight='bold')

    # Panel 1: R² by layer for probe and mean-diff
    ax1 = axes[0]
    ax1.set_title(f"R² Predicting {mc_metric} from Meta Activations", fontsize=10)

    for method, color, ls in [("probe", "tab:blue", "-"), ("mean_diff", "tab:orange", "--")]:
        fits = mc_uncert_results["fits"][method]
        r2_vals = [fits[l]["test_r2"] for l in layers]
        ci_low = [fits[l].get("test_r2_ci_low", np.nan) for l in layers]
        ci_high = [fits[l].get("test_r2_ci_high", np.nan) for l in layers]
        shuffled = [fits[l].get("shuffled_r2", np.nan) for l in layers]

        best_layer = max(layers, key=lambda l: fits[l]["test_r2"])
        best_r2 = fits[best_layer]["test_r2"]

        ax1.plot(layers, r2_vals, ls, color=color, linewidth=2,
                 label=f'{method} (L{best_layer}: {best_r2:.3f})')
        if not all(np.isnan(ci_low)):
            ax1.fill_between(layers, ci_low, ci_high, color=color, alpha=CI_ALPHA)

    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² (test set)')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, alpha=GRID_ALPHA)

    # Panel 2: Cosine similarity between probe and mean-diff directions
    ax2 = axes[1]
    ax2.set_title("Probe vs Mean-diff Cosine Similarity", fontsize=10)

    cosine_sims = [mc_uncert_results["comparison"][l]["cosine_sim"] for l in layers]
    ax2.plot(layers, cosine_sims, '-', color='tab:purple', linewidth=2)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Cosine Similarity')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=GRID_ALPHA)

    # Panel 3: KEY COMPARISON - cosine similarity to original d_mc_uncertainty
    ax3 = axes[2]
    ax3.set_title(f"Comparison to d_mc_{mc_metric}\n(high = same direction)", fontsize=10)

    if mc_dir_comparison:
        for method, color, ls in [("probe", "tab:blue", "-"), ("mean_diff", "tab:orange", "--")]:
            method_comparison = mc_dir_comparison.get(method, {})
            sims = [method_comparison.get(l, {}).get("cosine_similarity", np.nan) for l in layers]
            abs_sims = [method_comparison.get(l, {}).get("abs_cosine_similarity", np.nan) for l in layers]
            if not all(np.isnan(sims)):
                # Find layer with best R² for this method
                fits = mc_uncert_results["fits"][method]
                best_layer = max(layers, key=lambda l: fits[l]["test_r2"])
                best_cos = method_comparison.get(best_layer, {}).get("cosine_similarity", np.nan)
                ax3.plot(layers, sims, ls, color=color, linewidth=2,
                         label=f'{method} (L{best_layer}: {best_cos:.3f})')

        ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax3.axhline(y=0.6, color='green', linestyle='--', alpha=0.3, label='Strong similarity')
        ax3.axhline(y=-0.6, color='green', linestyle='--', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, "MC directions\nnot available",
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)

    ax3.set_xlabel('Layer Index')
    ax3.set_ylabel('Cosine Similarity')
    ax3.set_ylim(-1.1, 1.1)
    if mc_dir_comparison:
        ax3.legend(loc='upper left', fontsize=8)
    ax3.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_mc_uncertainty_from_meta_consolidated(
    all_mcuncert_for_plot: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
):
    """
    Plot MC uncertainty direction results for all metrics in a consolidated figure.

    Creates a grid with one row per metric, 3 columns:
    - Column 1: R² predicting MC uncertainty from meta activations
    - Column 2: Probe vs mean-diff cosine similarity
    - Column 3: Comparison to original d_mc_uncertainty

    Args:
        all_mcuncert_for_plot: Dict {metric: (mc_uncert_results, mc_dir_comparison)}
        num_layers: Number of model layers
        output_path: Where to save the figure
        meta_task: Name of meta-task for title
    """
    metrics = list(all_mcuncert_for_plot.keys())
    n_metrics = len(metrics)
    layers = list(range(num_layers))

    fig, axes = plt.subplots(n_metrics, 3, figsize=(15, 4 * n_metrics), squeeze=False)
    fig.suptitle(f"MC Uncertainty Directions from {meta_task} Activations",
                 fontsize=14, fontweight='bold', y=1.02)

    for row, mc_metric in enumerate(metrics):
        mc_uncert_results, mc_dir_comparison = all_mcuncert_for_plot[mc_metric]

        # Panel 1: R² by layer for probe and mean-diff
        ax1 = axes[row, 0]
        ax1.set_title(f"R² Predicting {mc_metric}", fontsize=10)

        for method, color, ls in [("probe", "tab:blue", "-"), ("mean_diff", "tab:orange", "--")]:
            fits = mc_uncert_results["fits"][method]
            r2_vals = [fits[l]["test_r2"] for l in layers]
            ci_low = [fits[l].get("test_r2_ci_low", np.nan) for l in layers]
            ci_high = [fits[l].get("test_r2_ci_high", np.nan) for l in layers]

            best_layer = max(layers, key=lambda l: fits[l]["test_r2"])
            best_r2 = fits[best_layer]["test_r2"]

            ax1.plot(layers, r2_vals, ls, color=color, linewidth=2,
                     label=f'{method} (L{best_layer}: {best_r2:.3f})')
            if not all(np.isnan(ci_low)):
                ax1.fill_between(layers, ci_low, ci_high, color=color, alpha=CI_ALPHA)

        ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax1.set_xlabel('Layer Index')
        ax1.set_ylabel('R² (test set)')
        ax1.legend(loc='upper left', fontsize=8)
        ax1.grid(True, alpha=GRID_ALPHA)

        # Panel 2: Cosine similarity between probe and mean-diff directions
        ax2 = axes[row, 1]
        ax2.set_title("Probe vs Mean-diff Similarity", fontsize=10)

        cosine_sims = [mc_uncert_results["comparison"][l]["cosine_sim"] for l in layers]
        ax2.plot(layers, cosine_sims, '-', color='tab:purple', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax2.set_xlabel('Layer Index')
        ax2.set_ylabel('Cosine Similarity')
        ax2.set_ylim(-1.1, 1.1)
        ax2.grid(True, alpha=GRID_ALPHA)

        # Panel 3: Comparison to original d_mc_uncertainty
        ax3 = axes[row, 2]
        ax3.set_title(f"vs d_mc_{mc_metric}", fontsize=10)

        if mc_dir_comparison:
            for method, color, ls in [("probe", "tab:blue", "-"), ("mean_diff", "tab:orange", "--")]:
                method_comparison = mc_dir_comparison.get(method, {})
                sims = [method_comparison.get(l, {}).get("cosine_similarity", np.nan) for l in layers]
                if not all(np.isnan(sims)):
                    fits = mc_uncert_results["fits"][method]
                    best_layer = max(layers, key=lambda l: fits[l]["test_r2"])
                    best_cos = method_comparison.get(best_layer, {}).get("cosine_similarity", np.nan)
                    ax3.plot(layers, sims, ls, color=color, linewidth=2,
                             label=f'{method} (L{best_layer}: {best_cos:.3f})')

            ax3.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
            ax3.axhline(y=0.6, color='green', linestyle='--', alpha=0.3)
            ax3.axhline(y=-0.6, color='green', linestyle='--', alpha=0.3)
            ax3.legend(loc='upper left', fontsize=8)
        else:
            ax3.text(0.5, 0.5, "MC directions\nnot available",
                     ha='center', va='center', transform=ax3.transAxes, fontsize=12)

        ax3.set_xlabel('Layer Index')
        ax3.set_ylabel('Cosine Similarity')
        ax3.set_ylim(-1.1, 1.1)
        ax3.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_position_comparison(
    transfer_results_by_pos: dict,
    mean_diff_transfer_by_pos: dict,
    num_layers: int,
    output_path: Path,
    meta_task: str,
):
    """
    Plot transfer R² across layers comparing different token positions.

    Creates a 2x2 grid:
    - Top row: Probe-based transfer for each metric
    - Bottom row: Mean-diff transfer for each metric

    Each panel shows one metric with lines for each position.
    """
    # Clip extreme negative R² values for display (same floor as other plots)
    R2_PLOT_FLOOR = -0.5  # Tighter floor for position comparison since we care about positive values
    R2_PLOT_CEIL = 1.0

    def _clip_r2(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=float)
        arr = np.where(np.isfinite(arr), arr, np.nan)
        return np.clip(arr, R2_PLOT_FLOOR, R2_PLOT_CEIL)

    metrics = set()
    for pos_data in transfer_results_by_pos.values():
        metrics.update(pos_data.keys())
    metrics = sorted(metrics)

    if len(metrics) == 0:
        print("  No metrics found for position comparison plot")
        return

    positions = list(transfer_results_by_pos.keys())
    pos_colors = {
        "question_mark": "tab:blue",
        "question_newline": "tab:cyan",
        "options_newline": "tab:green",
        "final": "tab:red",
    }
    # More readable position labels
    pos_labels = {
        "question_mark": "question ?",
        "question_newline": "question \\n",
        "options_newline": "options \\n",
        "final": "final",
    }

    # Use 2 rows x N cols where N = number of metrics
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, n_metrics, figsize=(6 * n_metrics, 10), squeeze=False)
    fig.suptitle(f"Position Comparison: {meta_task}", fontsize=14, fontweight='bold')

    layers = list(range(num_layers))

    # Top row: probe-based
    for col, metric in enumerate(metrics):
        ax = axes[0, col]
        ax.set_title(f"Probe Transfer: {metric}", fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        for pos in positions:
            if metric not in transfer_results_by_pos.get(pos, {}):
                continue
            color = pos_colors.get(pos, "tab:gray")
            display_name = pos_labels.get(pos, pos)

            r2_vals = []
            for l in layers:
                if l in transfer_results_by_pos[pos][metric]:
                    r2_vals.append(transfer_results_by_pos[pos][metric][l]["centered"]["r2"])
                else:
                    r2_vals.append(np.nan)
            r2_vals = np.array(r2_vals, dtype=float)

            # Find best layer BEFORE clipping (use true values)
            finite = np.isfinite(r2_vals)
            if finite.any():
                best_layer = int(np.argmax(np.where(finite, r2_vals, -np.inf)))
                best_r2 = r2_vals[best_layer]
                label = f"{display_name} (L{best_layer}: {best_r2:.3f})"
            else:
                label = display_name

            # Clip for plotting
            r2_vals_clipped = _clip_r2(r2_vals)
            ax.plot(layers, r2_vals_clipped, '-', label=label, color=color, linewidth=2)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('R²')
        ax.set_ylim(R2_PLOT_FLOOR, R2_PLOT_CEIL)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=GRID_ALPHA)

    # Bottom row: mean-diff
    for col, metric in enumerate(metrics):
        ax = axes[1, col]
        ax.set_title(f"Mean-Diff Transfer: {metric}", fontsize=11)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

        for pos in positions:
            if metric not in mean_diff_transfer_by_pos.get(pos, {}):
                continue
            color = pos_colors.get(pos, "tab:gray")
            display_name = pos_labels.get(pos, pos)

            r2_vals = []
            for l in layers:
                if l in mean_diff_transfer_by_pos[pos][metric]:
                    r2_vals.append(mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"])
                else:
                    r2_vals.append(np.nan)
            r2_vals = np.array(r2_vals, dtype=float)

            # Find best layer BEFORE clipping (use true values)
            finite = np.isfinite(r2_vals)
            if finite.any():
                best_layer = int(np.argmax(np.where(finite, r2_vals, -np.inf)))
                best_r2 = r2_vals[best_layer]
                label = f"{display_name} (L{best_layer}: {best_r2:.3f})"
            else:
                label = display_name

            # Clip for plotting
            r2_vals_clipped = _clip_r2(r2_vals)
            ax.plot(layers, r2_vals_clipped, '-', label=label, color=color, linewidth=2)

        ax.set_xlabel('Layer Index')
        ax.set_ylabel('R²')
        ax.set_ylim(R2_PLOT_FLOOR, R2_PLOT_CEIL)
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=GRID_ALPHA)

    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Model directory for organizing outputs
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET  # Model prefix now in directory, not filename
    base_output = f"{base_name}_meta_{META_TASK}"  # e.g., "TriviaMC_meta_confidence"

    # Load dataset from Stage 1
    dataset_path = find_output_file(f"{base_name}_mc_results.json", model_dir=model_dir)
    dataset = load_dataset(dataset_path)

    # Verify model matches
    model_name = dataset['config']['model']
    model_short = get_model_short_name(model_name)
    dataset_name = dataset['config']['dataset']

    # Console header
    config = {
        "model": model_short,
        "dataset": dataset_name,
        "task": META_TASK,
        "metric": METRICS[0] if len(METRICS) == 1 else f"{len(METRICS)} metrics",
    }
    print_run_header("test_meta_transfer.py", 2, "D→M transfer + confidence directions", config)

    print(f"Loading dataset ({len(dataset['questions'])} questions)...")

    # Load direct activations (needed to train probes with proper train/test split)
    direct_activations_path = find_output_file(f"{base_name}_mc_activations.npz", model_dir=model_dir)
    if not direct_activations_path.exists():
        raise ValueError(f"Direct activations not found: {direct_activations_path}\n"
                        f"Run identify_mc_correlate.py first.")

    print(f"Loading direct activations...")
    direct_loaded = np.load(direct_activations_path)

    # Reconstruct activations_by_layer
    layer_keys = [k for k in direct_loaded.files if k.startswith("layer_")]
    num_layers = len(layer_keys)
    direct_activations = {i: direct_loaded[f"layer_{i}"] for i in range(num_layers)}

    # Create train/test split (same split for direct and meta)
    n_questions = len(dataset['questions'])
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED
    )

    # Determine output paths (read paths for cache check, write paths for new files)
    # Activations file contains ALL positions, no position suffix
    # Position comparison compares positions, no position suffix
    # Per-position outputs (transfer_results, confdir, mcuncert) get _{pos} suffix inside the position loop
    activations_read_path = find_output_file(f"{base_output}_activations.npz", model_dir=model_dir)
    activations_path = get_output_path(f"{base_output}_activations.npz", model_dir=model_dir)
    plot_path_positions = get_output_path(f"{base_output}_position_comparison.png", model_dir=model_dir)

    # Check for cached activations
    use_cache = False
    if activations_read_path.exists():
        # Peek at cache to see what positions it has
        with np.load(activations_read_path) as peek:
            cached_positions = set()
            for key in peek.files:
                if key.startswith("layer_"):
                    parts = key.split("_")
                    if len(parts) > 2:  # Multi-position: layer_N_posname
                        cached_positions.add("_".join(parts[2:]))
                    else:  # Legacy: layer_N means "final" only
                        cached_positions.add("final")

            missing = set(PROBE_POSITIONS) - cached_positions
            if missing:
                print(f"Cache missing positions {missing}, re-extracting...")
            else:
                use_cache = True

    if use_cache:
        print(f"Loading cached activations...")
        loaded = np.load(activations_read_path)

        # Detect format: multi-position (layer_N_posname) vs legacy (layer_N)
        has_positions = any("_" in k.replace("layer_", "", 1) for k in loaded.files if k.startswith("layer_"))

        if has_positions:
            # Multi-position format: {position: {layer: array}}
            meta_activations = {pos: {} for pos in PROBE_POSITIONS}
            position_valid_arrays = {}
            for key in loaded.files:
                if key.startswith("layer_"):
                    parts = key.split("_")
                    layer = int(parts[1])
                    pos_name = "_".join(parts[2:])
                    if pos_name in meta_activations:
                        meta_activations[pos_name][layer] = loaded[key]
                elif key.startswith("valid_"):
                    pos_name = key[6:]  # Remove "valid_" prefix
                    position_valid_arrays[pos_name] = loaded[key]
                elif key == "confidences":
                    confidences = loaded[key]
                elif key == "option_probs":
                    option_probs = loaded[key]
            # If no option_probs found (old cache), set to None
            if "option_probs" not in loaded.files:
                option_probs = None
            # If no validity masks found (old format), assume all valid
            if not position_valid_arrays:
                n_samples = len(confidences)
                position_valid_arrays = {pos: np.ones(n_samples, dtype=bool) for pos in PROBE_POSITIONS}
        else:
            # Legacy format: {layer: array} - wrap in "final" position
            legacy_activations = {}
            for key in loaded.files:
                if key.startswith("layer_"):
                    layer = int(key.split("_")[1])
                    legacy_activations[layer] = loaded[key]
                elif key == "confidences":
                    confidences = loaded[key]
                elif key == "option_probs":
                    option_probs = loaded[key]
            meta_activations = {"final": legacy_activations}
            # Legacy format: no option_probs
            if "option_probs" not in loaded.files:
                option_probs = None
            # Legacy format: only final position, all valid
            n_samples = len(confidences)
            position_valid_arrays = {"final": np.ones(n_samples, dtype=bool)}

    if not use_cache:
        # Load model with appropriate quantization
        load_4bit = LOAD_IN_4BIT
        if load_4bit is None:
            load_4bit = "70B" in model_name or "70b" in model_name

        print("Loading model...")
        model, tokenizer, num_layers_model = load_model_and_tokenizer(
            model_name,
            adapter_path=ADAPTER,
            load_in_4bit=load_4bit,
            load_in_8bit=LOAD_IN_8BIT,
        )
        use_chat_template = should_use_chat_template(model_name, tokenizer)

        if num_layers_model != num_layers:
            print(f"Warning: model has {num_layers_model} layers but probes have {num_layers}")
            num_layers = min(num_layers, num_layers_model)

        # Get questions
        questions = dataset["questions"]

        # Compute trial_idx mapping to match OLD workflow
        # OLD workflow (run_introspection_experiment.py:2615-2616) applies a second shuffle:
        #   random.seed(SEED); random.shuffle(questions)
        # This affects which trial_idx each question gets, which determines the 1/2 mapping
        # in delegate prompts. We don't actually shuffle (to preserve alignment with direct
        # activations), but we compute what trial_idx each question WOULD have in OLD order.
        indexed_questions = list(enumerate(questions))  # [(orig_idx, question), ...]
        random.seed(SEED)
        random.shuffle(indexed_questions)
        # old_trial_idx[i] = what trial_idx question at position i would have in OLD workflow
        old_trial_idx = [None] * len(questions)
        for new_pos, (orig_idx, _) in enumerate(indexed_questions):
            old_trial_idx[orig_idx] = new_pos

        # Get meta task setup
        format_fn = get_meta_format_fn(META_TASK)
        signal_fn = get_meta_signal_fn(META_TASK)
        meta_options = get_meta_options(META_TASK)
        option_token_ids = [tokenizer.encode(k, add_special_tokens=False)[0] for k in meta_options]

        # Extract meta activations at multiple token positions
        print(f"Extracting meta activations...")

        # Initialize storage: {position: {layer: [activations]}}
        all_activations = {
            pos: {layer: [] for layer in range(num_layers)}
            for pos in PROBE_POSITIONS
        }
        all_confidences = []
        all_option_probs = []  # Raw probability distributions over options
        all_mappings = []
        # Track which examples have valid positions for each position name
        # (when find_mc_positions fails, it returns only {"final": -1})
        position_valid = {pos: [] for pos in PROBE_POSITIONS}

        with BatchedExtractor(model, num_layers) as extractor:
            for batch_start in tqdm(range(0, len(questions), BATCH_SIZE)):
                batch_questions = questions[batch_start:batch_start + BATCH_SIZE]

                prompts = []
                batch_mappings = []
                batch_positions = []  # List of position dicts per item
                for i, q in enumerate(batch_questions):
                    # Use old_trial_idx to match OLD workflow's 1/2 mapping for delegate prompts
                    trial_idx = old_trial_idx[batch_start + i]
                    if META_TASK == "delegate":
                        prompt, _, mapping = format_fn(q, tokenizer, trial_index=trial_idx, use_chat_template=use_chat_template)
                        batch_mappings.append(mapping)
                    else:
                        prompt, _ = format_fn(q, tokenizer, use_chat_template=use_chat_template)
                        batch_mappings.append(None)
                    prompts.append(prompt)

                    # Find token positions for this prompt
                    positions = find_mc_positions(prompt, tokenizer, q)
                    batch_positions.append(positions)

                encoded = tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,  # Prompts already have special tokens from chat template
                )
                input_ids = encoded["input_ids"].to(model.device)
                attention_mask = encoded["attention_mask"].to(model.device)

                # Build token_positions dict: {pos_name: [idx_for_each_batch_item]}
                # Also track validity: a position is valid if it exists in the dict
                # (find_mc_positions returns only {"final": -1} on failure)
                token_positions = {}
                for pos_name in PROBE_POSITIONS:
                    token_positions[pos_name] = [
                        bp.get(pos_name, -1) for bp in batch_positions
                    ]
                    # Track validity for this batch
                    for bp in batch_positions:
                        position_valid[pos_name].append(pos_name in bp)

                layer_acts_by_pos, probs, _, _ = extractor.extract_batch(
                    input_ids, attention_mask, option_token_ids, token_positions
                )

                # Store activations per position
                for pos_name in PROBE_POSITIONS:
                    for item_acts in layer_acts_by_pos[pos_name]:
                        for layer, act in item_acts.items():
                            all_activations[pos_name][layer].append(act)

                for p, mapping in zip(probs, batch_mappings):
                    confidence = signal_fn(p, mapping)
                    all_confidences.append(confidence)
                    all_option_probs.append(p)
                    all_mappings.append(mapping)

        # Stack activations: {position: {layer: np.array}}
        meta_activations = {
            pos: {layer: np.stack(acts) for layer, acts in pos_acts.items()}
            for pos, pos_acts in all_activations.items()
        }
        confidences = np.array(all_confidences)
        option_probs = np.stack(all_option_probs)  # (n_samples, n_options)
        # Convert validity masks to arrays
        position_valid_arrays = {pos: np.array(valid) for pos, valid in position_valid.items()}

        # Report validity stats (warnings only)
        for pos in PROBE_POSITIONS:
            n_valid = position_valid_arrays[pos].sum()
            n_total = len(position_valid_arrays[pos])
            if n_valid < n_total:
                print(f"Warning: {pos} has {n_valid}/{n_total} valid positions")

        # Save activations for future runs
        print(f"Saving activations...")
        save_dict = {"confidences": confidences, "option_probs": option_probs}
        for pos_name, pos_acts in meta_activations.items():
            for layer, acts in pos_acts.items():
                save_dict[f"layer_{layer}_{pos_name}"] = acts
        # Save validity masks
        for pos_name, valid_arr in position_valid_arrays.items():
            save_dict[f"valid_{pos_name}"] = valid_arr
        np.savez_compressed(activations_path, **save_dict)

    # Get positions available in loaded data
    positions_available = list(meta_activations.keys())
    first_pos = positions_available[0]
    first_layer = list(meta_activations[first_pos].keys())[0]
    target_name = "P(Answer)" if META_TASK == "delegate" else "Stated confidence" if META_TASK == "confidence" else "Other confidence"

    # Train probes and test transfer for each metric and position
    print(f"Training probes and testing transfer...")

    # Results structure: {position: {metric: {layer: {...}}}}
    transfer_results_by_pos = {pos: {} for pos in positions_available}
    direct_r2 = {}
    direct_r2_std = {}
    metrics_tested = [m for m in METRICS if m in dataset["metric_values"]]

    # Helper functions defined once
    def _safe_corr(a, b):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if a.size == 0 or b.size == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        if np.std(a) < 1e-12 or np.std(b) < 1e-12:
            return float("nan"), float("nan"), float("nan"), float("nan")
        r, p = pearsonr(a, b)
        rs, ps = spearmanr(a, b)
        return float(r), float(p), float(rs), float(ps)

    def _bootstrap_corr_std(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> float:
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

    for metric in metrics_tested:
        direct_values = dataset["metric_values"][metric]

        # Split data
        y_train = direct_values[train_idx]
        y_test = direct_values[test_idx]
        conf_test = confidences[test_idx]
        metric_sign = metric_sign_for_confidence(metric)

        direct_r2[metric] = {}
        direct_r2_std[metric] = {}

        # Initialize results for each position
        for pos in positions_available:
            transfer_results_by_pos[pos][metric] = {}

        for layer in tqdm(range(num_layers), desc=f"  {metric}"):
            X_direct_train = direct_activations[layer][train_idx]
            X_direct_test = direct_activations[layer][test_idx]

            # Train probe on direct_train (same probe for all positions)
            _, probe_info = probe_direction(
                X_direct_train, y_train,
                alpha=PROBE_ALPHA,
                pca_components=PROBE_PCA_COMPONENTS,
                bootstrap_splits=None,  # Fit on all train data
                return_probe=True,
            )

            scaler = probe_info["scaler"]
            pca = probe_info["pca"]
            ridge = probe_info["ridge"]

            # D→D: Test on direct_test (sanity check) - same for all positions
            d2d_result = apply_probe_shared(
                X_direct_test, y_test, scaler, pca, ridge
            )
            direct_r2[metric][layer] = d2d_result["r2"]
            _, d2d_std = bootstrap_r2_from_predictions(
                y_test, d2d_result["predictions"],
                n_boot=N_BOOTSTRAP, seed=SEED + layer
            )
            direct_r2_std[metric][layer] = float(d2d_std) if np.isfinite(d2d_std) else 0.0

            # D→M: Test transfer at each position
            for pos in positions_available:
                if layer not in meta_activations[pos]:
                    continue

                # Get validity mask for this position (only use examples with valid positions)
                pos_valid = position_valid_arrays.get(pos, np.ones(len(test_idx), dtype=bool))
                valid_test_mask = pos_valid[test_idx]
                n_valid = valid_test_mask.sum()

                if n_valid < 10:  # Need minimum samples for meaningful statistics
                    continue

                # Filter to valid examples only
                X_meta_test = meta_activations[pos][layer][test_idx][valid_test_mask]
                y_test_valid = y_test[valid_test_mask]
                conf_test_valid = conf_test[valid_test_mask]

                # D→M: Centered scaling (rigorous transfer test)
                centered_result = apply_probe_centered(
                    X_meta_test, y_test_valid, scaler, pca, ridge
                )

                # D→M: Separate scaling (upper bound)
                separate_result = apply_probe_separate(
                    X_meta_test, y_test_valid, pca, ridge
                )

                # Correlate probe predictions with stated confidence
                cen_r, cen_p, cen_rs, cen_ps = _safe_corr(centered_result["predictions"] * metric_sign, conf_test_valid)
                centered_result["pred_conf_pearson"] = cen_r
                centered_result["pred_conf_p"] = cen_p
                centered_result["pred_conf_spearman"] = cen_rs
                centered_result["pred_conf_spearman_p"] = cen_ps

                centered_result["pred_conf_pearson_std"] = _bootstrap_corr_std(
                    centered_result["predictions"] * metric_sign,
                    conf_test_valid,
                    n_boot=N_BOOTSTRAP,
                    seed=SEED + 10000 + layer,
                )

                sep_r, sep_p, sep_rs, sep_ps = _safe_corr(separate_result["predictions"] * metric_sign, conf_test_valid)
                separate_result["pred_conf_pearson"] = sep_r
                separate_result["pred_conf_p"] = sep_p
                separate_result["pred_conf_spearman"] = sep_rs
                separate_result["pred_conf_spearman_p"] = sep_ps

                # Bootstrap CIs for transfer R² (resample test set only)
                _, centered_std = bootstrap_transfer_r2(
                    X_meta_test, y_test_valid,
                    scaler, pca, ridge, "centered",
                    n_boot=N_BOOTSTRAP, seed=SEED + layer
                )
                centered_std = float(centered_std) if np.isfinite(centered_std) else 0.0
                _, separate_std = bootstrap_transfer_r2(
                    X_meta_test, y_test_valid,
                    scaler, pca, ridge, "separate",
                    n_boot=N_BOOTSTRAP, seed=SEED + layer
                )
                separate_std = float(separate_std) if np.isfinite(separate_std) else 0.0

                centered_result["r2_std"] = centered_std
                separate_result["r2_std"] = separate_std

                transfer_results_by_pos[pos][metric][layer] = {
                    "centered": centered_result,
                    "separate": separate_result,
                }


    # For backward compatibility, use "final" position for existing code
    transfer_results = transfer_results_by_pos.get("final", transfer_results_by_pos.get(positions_available[0], {}))

    # Behavioral correlation: metric vs meta-task target
    # For confidence task: correlation between metric and stated confidence
    # For delegate task: correlation between metric and P(Answer)
    meta_target_name = "P(Answer)" if META_TASK == "delegate" else "stated_confidence" if META_TASK == "confidence" else "other_confidence"

    behavioral = {"meta_target": meta_target_name}
    for metric in metrics_tested:
        direct_values = dataset["metric_values"][metric]
        sign = metric_sign_for_confidence(metric)

        corr, p_value = pearsonr(direct_values * sign, confidences)
        spearman_corr, spearman_p = spearmanr(direct_values * sign, confidences)

        # Bootstrap CI for full dataset correlation
        full_std = _bootstrap_corr_std(direct_values * sign, confidences, n_boot=N_BOOTSTRAP, seed=SEED)

        behavioral[metric] = {
            "pearson_r": float(corr),
            "pearson_p": float(p_value),
            "pearson_r_std": float(full_std),
            "spearman_r": float(spearman_corr),
            "spearman_p": float(spearman_p),
        }

        sign_str = "(inverted)" if sign < 0 else ""
        ci_lo = corr - 1.96 * full_std
        ci_hi = corr + 1.96 * full_std
        print(f"  {metric} {sign_str}: r={corr:.3f} [{ci_lo:.3f}, {ci_hi:.3f}] (p={p_value:.2e}), ρ={spearman_corr:.3f}")


    # Test-set baseline: correlation between (signed) raw metric values and meta-task target
    # on the held-out test set (same indices used for probe evaluation).
    conf_test = confidences[test_idx]
    for metric in metrics_tested:
        direct_test = dataset["metric_values"][metric][test_idx]
        sign = metric_sign_for_confidence(metric)

        test_r, test_p = pearsonr(direct_test * sign, conf_test)
        test_rs, test_rs_p = spearmanr(direct_test * sign, conf_test)

        # Bootstrap CI for test set correlation
        test_std = _bootstrap_corr_std(direct_test * sign, conf_test, n_boot=N_BOOTSTRAP, seed=SEED)

        behavioral[metric]["test_pearson_r"] = float(test_r)
        behavioral[metric]["test_pearson_p"] = float(test_p)
        behavioral[metric]["test_pearson_r_std"] = float(test_std)
        behavioral[metric]["test_spearman_r"] = float(test_rs)
        behavioral[metric]["test_spearman_p"] = float(test_rs_p)

        sign_str = "(inverted)" if sign < 0 else ""
        test_ci_lo = test_r - 1.96 * test_std
        test_ci_hi = test_r + 1.96 * test_std
        print(f"  {metric} {sign_str} test-set: r={test_r:.3f} [{test_ci_lo:.3f}, {test_ci_hi:.3f}], ρ={test_rs:.3f}")

    # =============================================================================
    # MEAN-DIFF TRANSFER (precomputed directions)
    # =============================================================================

    # Results by position: {position: {metric: {layer: {...}}}}
    mean_diff_transfer_by_pos: dict = {pos: {} for pos in positions_available}
    mean_diff_direct_r2: dict = {}
    mean_diff_direct_r2_std: dict = {}

    conf_test = confidences[test_idx]

    for metric in metrics_tested:
        # Locate and load directions file for this metric
        try:
            directions_path = _find_directions_npz(base_name, metric, model_dir)
        except FileNotFoundError as e:
            print(f"  Warning: {e}")
            continue
        mean_dirs = load_mean_diff_directions(directions_path, num_layers)

        if len(mean_dirs) == 0:
            print(f"  Warning: no mean_diff_layer_* keys found in {directions_path.name}; skipping.")
            continue

        direct_values = dataset["metric_values"][metric]
        y_train = direct_values[train_idx]
        y_test = direct_values[test_idx]

        for pos in positions_available:
            mean_diff_transfer_by_pos[pos][metric] = {}
        mean_diff_direct_r2[metric] = {}
        mean_diff_direct_r2_std[metric] = {}

        metric_sign = metric_sign_for_confidence(metric)

        for layer in tqdm(range(num_layers), desc=f"  {metric} mean-diff"):
            if layer not in mean_dirs:
                continue
            d = mean_dirs[layer]

            X_direct_train = direct_activations[layer][train_idx]
            X_direct_test = direct_activations[layer][test_idx]

            # 1) Compute 1D scores by projection on direct data
            s_train = X_direct_train @ d
            s_test = X_direct_test @ d

            # 2) Score standardization stats from DIRECT train (stable)
            s_mu = float(np.mean(s_train))
            s_std = float(np.std(s_train))
            if not np.isfinite(s_std) or s_std < 1e-8:
                s_std = 1e-8

            z_train = (s_train - s_mu) / s_std

            # 3) Fit a 1D calibrator on DIRECT train: y ≈ a*z + b
            from sklearn.linear_model import Ridge
            cal = Ridge(alpha=1e-6, fit_intercept=True)
            cal.fit(z_train.reshape(-1, 1), y_train)

            # D→D: evaluate on DIRECT test using DIRECT stats
            z_test = (s_test - s_mu) / s_std
            yhat_test = cal.predict(z_test.reshape(-1, 1))

            from sklearn.metrics import r2_score, mean_absolute_error
            d2d_r2 = float(r2_score(y_test, yhat_test))

            mean_diff_direct_r2[metric][layer] = d2d_r2
            _, d2d_std = bootstrap_r2_from_predictions(
                y_test, yhat_test,
                n_bootstrap=N_BOOTSTRAP,
                seed=SEED + 20000 + layer,
            )
            mean_diff_direct_r2_std[metric][layer] = d2d_std

            # D→M: Test transfer at each position
            for pos in positions_available:
                if layer not in meta_activations[pos]:
                    continue

                # Get validity mask for this position
                pos_valid = position_valid_arrays.get(pos, np.ones(len(test_idx), dtype=bool))
                valid_test_mask = pos_valid[test_idx]
                n_valid = valid_test_mask.sum()

                if n_valid < 10:
                    continue

                # Filter to valid examples only
                X_meta_test = meta_activations[pos][layer][test_idx][valid_test_mask]
                y_test_valid = y_test[valid_test_mask]
                conf_test_valid = conf_test[valid_test_mask]

                s_meta = X_meta_test @ d

                # D→M Centered: center META scores with their own mean, but scale with DIRECT std
                z_meta = (s_meta - float(np.mean(s_meta))) / s_std
                yhat_meta = cal.predict(z_meta.reshape(-1, 1))

                cen_r2 = float(r2_score(y_test_valid, yhat_meta))
                cen_mae = float(mean_absolute_error(y_test_valid, yhat_meta))
                cen_pear, _ = pearsonr(y_test_valid, yhat_meta)

                centered_result = {
                    "r2": cen_r2,
                    "mae": cen_mae,
                    "pearson": float(cen_pear),
                    "predictions": yhat_meta,
                }

                # Bootstrap for centered R²
                rng = np.random.RandomState(SEED + 30000 + layer)
                n = len(y_test_valid)
                vals_r2 = []
                vals_pc = []
                for _ in range(N_BOOTSTRAP):
                    idx = rng.choice(n, n, replace=True)
                    sm = (X_meta_test[idx] @ d)
                    zm = (sm - float(np.mean(sm))) / s_std
                    yhat_b = cal.predict(zm.reshape(-1, 1))
                    r2_b = 1.0 - float(np.mean((y_test_valid[idx] - yhat_b) ** 2)) / float(np.var(y_test_valid))
                    if np.isfinite(r2_b):
                        vals_r2.append(r2_b)
                    if np.std(yhat_b) > 1e-12 and np.std(conf_test_valid[idx]) > 1e-12:
                        r_b, _ = pearsonr(yhat_b * metric_sign, conf_test_valid[idx])
                        if np.isfinite(r_b):
                            vals_pc.append(float(r_b))

                centered_result["r2_std"] = float(np.std(vals_r2)) if len(vals_r2) > 1 else 0.0

                # Prediction→confidence correlation
                pc_r, pc_p, pc_rs, pc_ps = _safe_corr(yhat_meta * metric_sign, conf_test_valid)
                centered_result["pred_conf_pearson"] = pc_r
                centered_result["pred_conf_p"] = pc_p
                centered_result["pred_conf_spearman"] = pc_rs
                centered_result["pred_conf_spearman_p"] = pc_ps
                centered_result["pred_conf_pearson_std"] = float(np.std(vals_pc)) if len(vals_pc) > 1 else 0.0

                mean_diff_transfer_by_pos[pos][metric][layer] = {"centered": centered_result}

        # Summary by position

    # =============================================================================
    # ANSWER DIRECTION D2M TRANSFER
    # =============================================================================
    # This tests whether answer directions (trained to predict A/B/C/D from direct
    # activations) can also predict answers in meta-task activations.

    answer_transfer_results = {}
    answer_probes_path = find_output_file(f"{base_name}_mc_answer_probes.joblib", model_dir=model_dir)

    if answer_probes_path.exists():
        print(f"\nLoading answer probes from {answer_probes_path}...")
        answer_probe_data = joblib.load(answer_probes_path)
        answer_probes = answer_probe_data["probes"]
        answer_metadata = answer_probe_data["metadata"]
        answer_mapping = answer_metadata["answer_mapping"]

        # Get model answers from dataset
        model_answers = [q["predicted_answer"] for q in dataset["questions"]]
        encoded_answers, _ = encode_answers(model_answers)
        y_answer_test = encoded_answers[test_idx]

        # Answer D2D (sanity check): apply classifiers to direct test activations
        print(f"\n  Testing D→D (direct test) answer prediction...")
        answer_d2d_results = {}
        for layer in tqdm(range(num_layers), desc="    D→D answer"):
            if layer not in answer_probes:
                continue
            clf_info = answer_probes[layer]
            X_direct_test = direct_activations[layer][test_idx]

            # D→D: Use shared scaler
            X_scaled = clf_info["scaler"].transform(X_direct_test.astype(np.float32))
            X_pca = clf_info["pca"].transform(X_scaled)
            y_pred = clf_info["clf"].predict(X_pca)
            d2d_acc = float((y_pred == y_answer_test).mean())
            _, d2d_std = bootstrap_accuracy_from_predictions(y_answer_test, y_pred, N_BOOTSTRAP, SEED + layer)
            answer_d2d_results[layer] = {"accuracy": d2d_acc, "accuracy_std": d2d_std}


        # Answer D2M: apply classifiers to meta activations
        print(f"\n  Testing D→M (meta test) answer prediction...")
        answer_d2m_results_by_pos = {pos: {} for pos in positions_available}

        for layer in tqdm(range(num_layers), desc="    D→M answer"):
            if layer not in answer_probes:
                continue
            clf_info = answer_probes[layer]

            for pos in positions_available:
                if layer not in meta_activations[pos]:
                    continue

                # Get validity mask
                pos_valid = position_valid_arrays.get(pos, np.ones(len(test_idx), dtype=bool))
                valid_test_mask = pos_valid[test_idx]
                n_valid = valid_test_mask.sum()

                if n_valid < 10:
                    continue

                X_meta_test = meta_activations[pos][layer][test_idx][valid_test_mask]
                y_answer_test_valid = y_answer_test[valid_test_mask]

                # D2M Centered: center meta with own mean, scale with direct's std
                centered_result = apply_answer_classifier_centered(
                    X_meta_test, y_answer_test_valid,
                    clf_info["scaler"], clf_info["pca"], clf_info["clf"]
                )
                _, centered_std = bootstrap_accuracy_from_predictions(
                    y_answer_test_valid, np.array(centered_result["predictions"]),
                    N_BOOTSTRAP, SEED + layer
                )
                centered_result["accuracy_std"] = centered_std

                # D2M Separate: use meta's own standardization
                separate_result = apply_answer_classifier_separate(
                    X_meta_test, y_answer_test_valid,
                    clf_info["pca"], clf_info["clf"]
                )
                _, separate_std = bootstrap_accuracy_from_predictions(
                    y_answer_test_valid, np.array(separate_result["predictions"]),
                    N_BOOTSTRAP, SEED + layer + 1000
                )
                separate_result["accuracy_std"] = separate_std

                answer_d2m_results_by_pos[pos][layer] = {
                    "centered": centered_result,
                    "separate": separate_result,
                }

        # Store for saving
        answer_transfer_results = {
            "d2d": answer_d2d_results,
            "d2m_by_position": answer_d2m_results_by_pos,
            "answer_mapping": answer_mapping,
        }

    else:
        print(f"\n  No answer classifiers found at {answer_probes_path}")
        print("  Run identify_mc_answer_correlate.py first.")

    # For backward compatibility, use "final" position for legacy results
    mean_diff_transfer_results = mean_diff_transfer_by_pos.get("final", mean_diff_transfer_by_pos.get(positions_available[0], {}))

    # Plot combined transfer results - one consolidated plot per position (all metrics)
    # Combines probe (solid), mean-diff (dashed), and answer (dotted) on same figure
    print(f"Plotting transfer results...")
    for pos in positions_available:
        if not transfer_results_by_pos[pos]:
            continue

        # Get answer data for this position if available
        answer_d2d = None
        answer_d2m = None
        if answer_transfer_results:
            answer_d2d = answer_transfer_results.get("d2d")
            answer_d2m = answer_transfer_results.get("d2m_by_position", {}).get(pos)

        pos_plot_path = get_output_path(f"{base_output}_transfer_results_{pos}.png", model_dir=model_dir)
        plot_combined_transfer_results_consolidated(
            probe_results=transfer_results_by_pos[pos],
            probe_direct_r2=direct_r2,
            probe_direct_r2_std=direct_r2_std,
            mean_diff_results=mean_diff_transfer_by_pos.get(pos, {}),
            mean_diff_direct_r2=mean_diff_direct_r2,
            mean_diff_direct_r2_std=mean_diff_direct_r2_std,
            behavioral=behavioral,
            num_layers=num_layers,
            output_path=pos_plot_path,
            meta_task=META_TASK,
            metrics=metrics_tested,
            position=pos,
            answer_d2d=answer_d2d,
            answer_d2m=answer_d2m,
        )

    # Plot position comparison (only if multiple positions)
    if len(positions_available) > 1:
        plot_position_comparison(
            transfer_results_by_pos,
            mean_diff_transfer_by_pos,
            num_layers,
            plot_path_positions,
            META_TASK,
        )

    # Build key findings for console output
    key_findings = {}
    output_files = [activations_path]  # Will add per-position files in loop

    # =========================================================================
    # SAVE PER-POSITION RESULTS
    # =========================================================================
    # Each position gets its own set of output files:
    # - transfer_results_{pos}.json/npz
    # - confdir_*_{pos}.* (if FIND_CONFIDENCE_DIRECTIONS)
    # - mcuncert_*_{pos}.* (if FIND_MC_UNCERTAINTY_DIRECTIONS)
    print(f"Saving per-position results...")

    for pos in positions_available:
        if not transfer_results_by_pos[pos]:
            continue

        # Define per-position output paths
        results_json_path = get_output_path(f"{base_output}_transfer_results_{pos}.json", model_dir=model_dir)
        results_npz_path = get_output_path(f"{base_output}_transfer_results_{pos}.npz", model_dir=model_dir)

        # Build position-specific JSON
        results_json = {
            "format_version": 2,
            "config": get_config_dict(
                model=model_name,
                dataset=dataset['config']['dataset'],
                meta_task=META_TASK,
                position=pos,
                num_questions=len(dataset['questions']),
                num_layers=num_layers,
                input_model_dir=model_dir,
                input_dataset=base_name,
                train_split=TRAIN_SPLIT,
                n_train=len(train_idx),
                n_test=len(test_idx),
                seed=SEED,
                probe_alpha=PROBE_ALPHA,
                probe_pca_components=PROBE_PCA_COMPONENTS,
                load_in_4bit=LOAD_IN_4BIT,
                load_in_8bit=LOAD_IN_8BIT,
            ),
            "meta_target_stats": compute_behavioral_stats(
                confidences, option_probs, META_TASK
            ),
            "behavioral": behavioral,
            "transfer": {},
            "mean_diff_transfer": {},
        }

        # Add probe transfer results for this position
        for metric in metrics_tested:
            if metric not in transfer_results_by_pos[pos] or not transfer_results_by_pos[pos][metric]:
                continue
            layers_available = list(transfer_results_by_pos[pos][metric].keys())
            if not layers_available:
                continue
            best_layer_cen = max(layers_available, key=lambda l: transfer_results_by_pos[pos][metric][l]["centered"]["r2"])
            best_layer_sep = max(layers_available, key=lambda l: transfer_results_by_pos[pos][metric][l]["separate"]["r2"])

            results_json["transfer"][metric] = {
                "d2m_centered": {
                    "best_layer": best_layer_cen,
                    "best_r2": transfer_results_by_pos[pos][metric][best_layer_cen]["centered"]["r2"],
                    "best_r2_std": transfer_results_by_pos[pos][metric][best_layer_cen]["centered"].get("r2_std", 0.0),
                    "best_pearson": transfer_results_by_pos[pos][metric][best_layer_cen]["centered"]["pearson"],
                },
                "d2m_separate": {
                    "best_layer": best_layer_sep,
                    "best_r2": transfer_results_by_pos[pos][metric][best_layer_sep]["separate"]["r2"],
                    "best_r2_std": transfer_results_by_pos[pos][metric][best_layer_sep]["separate"].get("r2_std", 0.0),
                    "best_pearson": transfer_results_by_pos[pos][metric][best_layer_sep]["separate"]["pearson"],
                },
                "per_layer": {
                    l: {
                        "centered_r2": transfer_results_by_pos[pos][metric][l]["centered"]["r2"],
                        "centered_r2_std": transfer_results_by_pos[pos][metric][l]["centered"].get("r2_std", 0.0),
                        "centered_pearson": transfer_results_by_pos[pos][metric][l]["centered"]["pearson"],
                        "centered_pred_conf_pearson": transfer_results_by_pos[pos][metric][l]["centered"].get("pred_conf_pearson"),
                        "separate_r2": transfer_results_by_pos[pos][metric][l]["separate"]["r2"],
                        "separate_r2_std": transfer_results_by_pos[pos][metric][l]["separate"].get("r2_std", 0.0),
                        "separate_pearson": transfer_results_by_pos[pos][metric][l]["separate"]["pearson"],
                        "separate_pred_conf_pearson": transfer_results_by_pos[pos][metric][l]["separate"].get("pred_conf_pearson"),
                    }
                    for l in layers_available
                },
            }

            # Add D→D results if available (from training)
            if metric in direct_r2 and direct_r2[metric]:
                best_d2d = max(direct_r2[metric].keys(), key=lambda l: direct_r2[metric][l])
                results_json["transfer"][metric]["d2d"] = {
                    "best_layer": best_d2d,
                    "best_r2": direct_r2[metric][best_d2d],
                }
                for l in direct_r2[metric].keys():
                    if l in results_json["transfer"][metric]["per_layer"]:
                        results_json["transfer"][metric]["per_layer"][l]["d2d_r2"] = direct_r2[metric][l]

        # Add mean-diff transfer results for this position
        for metric in metrics_tested:
            if metric not in mean_diff_transfer_by_pos[pos] or not mean_diff_transfer_by_pos[pos][metric]:
                continue
            layers_available = list(mean_diff_transfer_by_pos[pos][metric].keys())
            if not layers_available:
                continue
            best_layer = max(layers_available, key=lambda l: mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"])

            results_json["mean_diff_transfer"][metric] = {
                "best_layer": best_layer,
                "best_r2": mean_diff_transfer_by_pos[pos][metric][best_layer]["centered"]["r2"],
                "per_layer": {
                    l: {
                        "centered_r2": mean_diff_transfer_by_pos[pos][metric][l]["centered"]["r2"],
                        "centered_r2_std": mean_diff_transfer_by_pos[pos][metric][l]["centered"].get("r2_std", 0.0),
                        "centered_pearson": mean_diff_transfer_by_pos[pos][metric][l]["centered"]["pearson"],
                        "centered_pred_conf_pearson": mean_diff_transfer_by_pos[pos][metric][l]["centered"].get("pred_conf_pearson"),
                    }
                    for l in layers_available
                },
            }

        # Add answer direction transfer results for this position
        if answer_transfer_results:
            pos_answer_data = answer_transfer_results.get("d2m_by_position", {}).get(pos, {})
            if pos_answer_data:
                results_json["answer_transfer"] = {
                    "answer_mapping": answer_transfer_results.get("answer_mapping", {}),
                    "d2d": {},
                    "d2m": {},
                }

                # D2D results (same across positions)
                for layer, data in answer_transfer_results.get("d2d", {}).items():
                    results_json["answer_transfer"]["d2d"][layer] = {
                        "accuracy": data["accuracy"],
                        "accuracy_std": data.get("accuracy_std", 0),
                    }

                # D2M for this position
                for layer, layer_data in pos_answer_data.items():
                    results_json["answer_transfer"]["d2m"][layer] = {
                        "centered_accuracy": layer_data["centered"]["accuracy"],
                        "centered_accuracy_std": layer_data["centered"].get("accuracy_std", 0),
                        "separate_accuracy": layer_data["separate"]["accuracy"],
                        "separate_accuracy_std": layer_data["separate"].get("accuracy_std", 0),
                    }

                # Best answer transfer for this position
                if pos_answer_data:
                    best_layer = max(pos_answer_data.keys(), key=lambda l: pos_answer_data[l]["centered"]["accuracy"])
                    results_json["answer_transfer"]["best_layer"] = best_layer
                    results_json["answer_transfer"]["best_centered_accuracy"] = pos_answer_data[best_layer]["centered"]["accuracy"]

        # Add per-question paired data
        results_json["per_question"] = []
        for i, q in enumerate(dataset["questions"]):
            item = {
                "question": q.get("question", ""),
                "correct_answer": q.get("correct_answer", ""),
                "stated_confidence": float(confidences[i]),
            }
            for metric in metrics_tested:
                if metric in dataset["metric_values"]:
                    item[metric] = float(dataset["metric_values"][metric][i])
            results_json["per_question"].append(item)

        # Save JSON
        with open(results_json_path, "w") as f:
            json.dump(results_json, f, indent=2)

        # Save NPZ
        save_dict = {
            "model": model_name,
            "dataset": dataset['config']['dataset'],
            "meta_task": META_TASK,
            "position": pos,
            "metrics": np.array(metrics_tested),
            "num_questions": len(dataset['questions']),
            "num_layers": num_layers,
            "confidences": confidences,
        }

        for metric in metrics_tested:
            if metric not in transfer_results_by_pos[pos]:
                continue
            for layer in transfer_results_by_pos[pos][metric].keys():
                save_dict[f"transfer_{metric}_layer{layer}_centered_r2"] = transfer_results_by_pos[pos][metric][layer]["centered"]["r2"]
                save_dict[f"transfer_{metric}_layer{layer}_centered_r2_std"] = transfer_results_by_pos[pos][metric][layer]["centered"].get("r2_std", 0.0)
                save_dict[f"transfer_{metric}_layer{layer}_centered_pred_conf_pearson"] = transfer_results_by_pos[pos][metric][layer]["centered"].get("pred_conf_pearson", np.nan)
                save_dict[f"transfer_{metric}_layer{layer}_separate_r2"] = transfer_results_by_pos[pos][metric][layer]["separate"]["r2"]
                save_dict[f"transfer_{metric}_layer{layer}_separate_r2_std"] = transfer_results_by_pos[pos][metric][layer]["separate"].get("r2_std", 0.0)
                save_dict[f"transfer_{metric}_layer{layer}_separate_pred_conf_pearson"] = transfer_results_by_pos[pos][metric][layer]["separate"].get("pred_conf_pearson", np.nan)
                # Add D→D from training if available
                if metric in direct_r2 and layer in direct_r2[metric]:
                    save_dict[f"d2d_{metric}_layer{layer}_r2"] = direct_r2[metric][layer]
                if metric in direct_r2_std and layer in direct_r2_std[metric]:
                    save_dict[f"d2d_{metric}_layer{layer}_r2_std"] = direct_r2_std[metric][layer]

            save_dict[f"behavioral_{metric}_pearson_r"] = behavioral[metric]["pearson_r"]
            save_dict[f"behavioral_{metric}_spearman_r"] = behavioral[metric]["spearman_r"]

            if "test_pearson_r" in behavioral[metric]:
                save_dict[f"behavioral_{metric}_test_pearson_r"] = behavioral[metric]["test_pearson_r"]
            if "test_spearman_r" in behavioral[metric]:
                save_dict[f"behavioral_{metric}_test_spearman_r"] = behavioral[metric]["test_spearman_r"]

        np.savez(results_npz_path, **save_dict)

        # Add to output files
        output_files.extend([results_json_path, results_npz_path])

        # =================================================================
        # CONFIDENCE DIRECTIONS FOR THIS POSITION (optional)
        # =================================================================
        if FIND_CONFIDENCE_DIRECTIONS:
            # Use already-loaded activations for this position
            meta_activations_by_layer = meta_activations.get(pos, {})

            # Skip if no activations for this position
            if not meta_activations_by_layer:
                print(f"  Skipping confidence probes ({pos}) - no activations available")
            else:
                # Find confidence directions using both methods
                print(f"  Training confidence probes ({pos})...")
                conf_results = find_confidence_directions_both_methods(
                    meta_activations_by_layer,
                    confidences,
                    train_idx,
                    test_idx,
                    alpha=PROBE_ALPHA,
                    n_components=PROBE_PCA_COMPONENTS,
                    mean_diff_quantile=MEAN_DIFF_QUANTILE,
                    n_bootstrap=N_BOOTSTRAP,
                    train_split=TRAIN_SPLIT,
                    seed=SEED,
                )

                # Compare to uncertainty directions (if requested)
                conf_unc_comparison = None
                if COMPARE_UNCERTAINTY_METRIC:
                    uncertainty_dir_path = find_output_file(f"{base_name}_mc_{COMPARE_UNCERTAINTY_METRIC}_directions.npz", model_dir=model_dir)
                    if uncertainty_dir_path.exists():
                        unc_data = np.load(uncertainty_dir_path)

                        # Compare BOTH probe and mean_diff confidence directions to uncertainty directions
                        conf_unc_comparison = {}
                        for method in ["probe", "mean_diff"]:
                            unc_dirs = {}
                            for layer in range(num_layers):
                                key = f"{method}_layer_{layer}"
                                if key in unc_data:
                                    unc_dirs[layer] = unc_data[key]

                            if unc_dirs:
                                conf_unc_comparison[method] = compare_confidence_to_uncertainty(
                                    conf_results["directions"][method],
                                    unc_dirs
                                )

                        if not conf_unc_comparison:
                            conf_unc_comparison = None

                # Save meta-confidence directions
                conf_dir_path = get_output_path(f"{base_output}_confdir_directions_{pos}.npz", model_dir=model_dir)
                dir_save = {
                    "_metadata_input_base": f"{model_dir}/{base_name}",
                    "_metadata_meta_task": META_TASK,
                    "_metadata_position": pos,
                }
                for method in ["probe", "mean_diff"]:
                    for layer in range(num_layers):
                        dir_save[f"{method}_layer_{layer}"] = conf_results["directions"][method][layer]
                np.savez(conf_dir_path, **dir_save)

                # Save meta-confidence probe objects
                conf_probes_path = get_output_path(f"{base_output}_confdir_probes_{pos}.joblib", model_dir=model_dir)
                probe_save = {
                    "metadata": {
                        "input_base": f"{model_dir}/{base_name}",
                        "meta_task": META_TASK,
                        "position": pos,
                        "train_split": TRAIN_SPLIT,
                        "probe_alpha": PROBE_ALPHA,
                        "pca_components": PROBE_PCA_COMPONENTS,
                        "seed": SEED,
                    },
                    "probes": conf_results["probes"],
                }
                joblib.dump(probe_save, conf_probes_path)

                # Save meta-confidence results JSON
                conf_results_path = get_output_path(f"{base_output}_confdir_results_{pos}.json", model_dir=model_dir)
                conf_json = {
                    "config": get_config_dict(
                        input_base=f"{model_dir}/{base_name}",
                        meta_task=META_TASK,
                        position=pos,
                        train_split=TRAIN_SPLIT,
                        probe_alpha=PROBE_ALPHA,
                        pca_components=PROBE_PCA_COMPONENTS,
                        mean_diff_quantile=MEAN_DIFF_QUANTILE,
                        n_bootstrap=N_BOOTSTRAP,
                        seed=SEED,
                        load_in_4bit=LOAD_IN_4BIT,
                        load_in_8bit=LOAD_IN_8BIT,
                    ),
                    "stats": {
                        "n_samples": len(confidences),
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                        "confidence_mean": float(confidences.mean()),
                        "confidence_std": float(confidences.std()),
                        "confidence_min": float(confidences.min()),
                        "confidence_max": float(confidences.max()),
                    },
                    "results": {},
                    "comparison": {},
                }
                for method in ["probe", "mean_diff"]:
                    conf_json["results"][method] = {}
                    for layer in range(num_layers):
                        layer_info = {}
                        for k, v in conf_results["fits"][method][layer].items():
                            if isinstance(v, np.floating):
                                layer_info[k] = float(v)
                            elif isinstance(v, np.integer):
                                layer_info[k] = int(v)
                            else:
                                layer_info[k] = v
                        conf_json["results"][method][layer] = layer_info
                for layer in range(num_layers):
                    conf_json["comparison"][layer] = {
                        "cosine_sim": float(conf_results["comparison"][layer]["cosine_sim"])
                    }
                if conf_unc_comparison:
                    conf_json["uncertainty_comparison"] = {
                        "metric": COMPARE_UNCERTAINTY_METRIC,
                        "by_method": {}
                    }
                    for method, method_comp in conf_unc_comparison.items():
                        conf_json["uncertainty_comparison"]["by_method"][method] = {
                            layer: {k: float(v) for k, v in comp.items()}
                            for layer, comp in method_comp.items()
                        }
                with open(conf_results_path, "w") as f:
                    json.dump(conf_json, f, indent=2)

                # Plot meta-confidence directions
                conf_plot_path = get_output_path(f"{base_output}_confdir_results_{pos}.png", model_dir=model_dir)
                plot_confidence_directions(
                    conf_results=conf_results,
                    conf_unc_comparison=conf_unc_comparison,
                    num_layers=num_layers,
                    output_path=conf_plot_path,
                    meta_task=META_TASK,
                    compare_metric=COMPARE_UNCERTAINTY_METRIC,
                )

                # Add confdir files to output list
                output_files.extend([conf_dir_path, conf_results_path, conf_plot_path])

        # =================================================================
        # MC UNCERTAINTY DIRECTIONS FOR THIS POSITION (optional)
        # =================================================================
        if FIND_MC_UNCERTAINTY_DIRECTIONS:
            # Use already-loaded activations for this position
            meta_activations_by_layer = meta_activations.get(pos, {})

            if not meta_activations_by_layer:
                print(f"  Skipping MC uncertainty probes ({pos}) - no activations available")
            else:
                # Initialize consolidated structures for all metrics
                all_mcuncert_directions = {
                    "_metadata_input_base": f"{model_dir}/{base_name}",
                    "_metadata_meta_task": META_TASK,
                    "_metadata_position": pos,
                    "_metadata_metrics": json.dumps(MC_UNCERTAINTY_METRICS),
                }
                all_mcuncert_probes = {
                    "metadata": {
                        "input_base": f"{model_dir}/{base_name}",
                        "meta_task": META_TASK,
                        "position": pos,
                        "metrics": MC_UNCERTAINTY_METRICS,
                        "train_split": TRAIN_SPLIT,
                        "probe_alpha": PROBE_ALPHA,
                        "pca_components": PROBE_PCA_COMPONENTS,
                        "seed": SEED,
                    },
                    "probes": {},
                }
                all_mcuncert_json = {
                    "config": get_config_dict(
                        input_base=f"{model_dir}/{base_name}",
                        meta_task=META_TASK,
                        position=pos,
                        metrics=MC_UNCERTAINTY_METRICS,
                        train_split=TRAIN_SPLIT,
                        probe_alpha=PROBE_ALPHA,
                        pca_components=PROBE_PCA_COMPONENTS,
                        mean_diff_quantile=MEAN_DIFF_QUANTILE,
                        n_bootstrap=N_BOOTSTRAP,
                        seed=SEED,
                        load_in_4bit=LOAD_IN_4BIT,
                        load_in_8bit=LOAD_IN_8BIT,
                    ),
                    "stats": {
                        "n_samples": len(train_idx) + len(test_idx),
                        "n_train": len(train_idx),
                        "n_test": len(test_idx),
                    },
                    "metrics": {},
                }
                all_mcuncert_for_plot = {}

                # Loop over each MC uncertainty metric
                for mc_metric in MC_UNCERTAINTY_METRICS:
                    mc_uncertainty = dataset["metric_values"][mc_metric]

                    print(f"  Training MC uncertainty probes ({pos}, {mc_metric})...")
                    mc_uncert_results = find_mc_uncertainty_directions_from_meta(
                        meta_activations_by_layer,
                        mc_uncertainty,
                        train_idx,
                        test_idx,
                        alpha=PROBE_ALPHA,
                        n_components=PROBE_PCA_COMPONENTS,
                        mean_diff_quantile=MEAN_DIFF_QUANTILE,
                        n_bootstrap=N_BOOTSTRAP,
                        seed=SEED,
                    )

                    # Compare to original d_mc_uncertainty
                    mc_dir_comparison = None
                    mc_directions_path = find_output_file(f"{base_name}_mc_{mc_metric}_directions.npz", model_dir=model_dir)
                    if mc_directions_path.exists():
                        mc_data = np.load(mc_directions_path)
                        mc_dir_comparison = {}
                        for method in ["probe", "mean_diff"]:
                            mc_dirs = {}
                            for layer in range(num_layers):
                                key = f"{method}_layer_{layer}"
                                if key in mc_data:
                                    mc_dirs[layer] = mc_data[key]
                            if mc_dirs:
                                mc_dir_comparison[method] = compare_confidence_to_uncertainty(
                                    mc_uncert_results["directions"][method],
                                    mc_dirs
                                )
                        if not mc_dir_comparison:
                            mc_dir_comparison = None

                    # Add to consolidated structures
                    for method in ["probe", "mean_diff"]:
                        for layer in range(num_layers):
                            all_mcuncert_directions[f"{method}_{mc_metric}_layer_{layer}"] = mc_uncert_results["directions"][method][layer]
                    all_mcuncert_probes["probes"][mc_metric] = mc_uncert_results["probes"]

                    metric_results = {
                        "target_stats": {
                            "mean": float(mc_uncertainty.mean()),
                            "std": float(mc_uncertainty.std()),
                            "min": float(mc_uncertainty.min()),
                            "max": float(mc_uncertainty.max()),
                        },
                        "results": {},
                        "comparison_within_methods": {},
                    }
                    for method in ["probe", "mean_diff"]:
                        metric_results["results"][method] = {}
                        for layer in range(num_layers):
                            layer_info = {}
                            for k, v in mc_uncert_results["fits"][method][layer].items():
                                if isinstance(v, np.floating):
                                    layer_info[k] = float(v)
                                elif isinstance(v, np.integer):
                                    layer_info[k] = int(v)
                                else:
                                    layer_info[k] = v
                            metric_results["results"][method][layer] = layer_info
                    for layer in range(num_layers):
                        metric_results["comparison_within_methods"][layer] = {
                            "cosine_sim": float(mc_uncert_results["comparison"][layer]["cosine_sim"])
                        }
                    if mc_dir_comparison:
                        metric_results["comparison_to_mc_directions"] = {"by_method": {}}
                        for method, method_comp in mc_dir_comparison.items():
                            metric_results["comparison_to_mc_directions"]["by_method"][method] = {
                                layer: {k: float(v) for k, v in comp.items()}
                                for layer, comp in method_comp.items()
                            }
                    all_mcuncert_json["metrics"][mc_metric] = metric_results
                    all_mcuncert_for_plot[mc_metric] = (mc_uncert_results, mc_dir_comparison)

                # Save per-position mcuncert files
                mcuncert_dir_path = get_output_path(f"{base_output}_mcuncert_directions_{pos}.npz", model_dir=model_dir)
                np.savez(mcuncert_dir_path, **all_mcuncert_directions)

                mcuncert_probes_path = get_output_path(f"{base_output}_mcuncert_probes_{pos}.joblib", model_dir=model_dir)
                joblib.dump(all_mcuncert_probes, mcuncert_probes_path)

                mcuncert_results_path = get_output_path(f"{base_output}_mcuncert_results_{pos}.json", model_dir=model_dir)
                with open(mcuncert_results_path, "w") as f:
                    json.dump(all_mcuncert_json, f, indent=2)

                mcuncert_plot_path = get_output_path(f"{base_output}_mcuncert_results_{pos}.png", model_dir=model_dir)
                plot_mc_uncertainty_from_meta_consolidated(
                    all_mcuncert_for_plot=all_mcuncert_for_plot,
                    num_layers=num_layers,
                    output_path=mcuncert_plot_path,
                    meta_task=META_TASK,
                )

                # Add mcuncert files to output list
                output_files.extend([mcuncert_dir_path, mcuncert_results_path, mcuncert_plot_path])

    # Collect key findings for console summary
    meta_stats = compute_behavioral_stats(confidences, option_probs, META_TASK)
    if META_TASK == "delegate":
        if "delegation_rate" in meta_stats:
            key_findings["Delegation rate"] = f"{meta_stats['delegation_rate']:.1%}"

    for metric in metrics_tested:
        if metric not in transfer_results or not transfer_results[metric]:
            continue
        layers_available = list(transfer_results[metric].keys())
        if not layers_available:
            continue

        # Behavioral correlation
        beh = behavioral[metric]
        r_val = beh['pearson_r']
        key_findings[f"Behavioral ({metric})"] = f"r={r_val:.3f}"

        # D→D (test set)
        d2d_r2_val = None
        if metric in direct_r2 and direct_r2[metric]:
            best_d2d = max(direct_r2[metric].keys(), key=lambda l: direct_r2[metric][l])
            d2d_r2_val = direct_r2[metric][best_d2d]

        # D→M transfer
        best_layer_cen = max(layers_available, key=lambda l: transfer_results[metric][l]["centered"]["r2"])
        centered_r2 = transfer_results[metric][best_layer_cen]["centered"]["r2"]
        centered_std = transfer_results[metric][best_layer_cen]["centered"]["r2_std"]
        key_findings[f"D→M ({metric})"] = format_r2_with_ci(centered_r2, centered_r2 - 1.96*centered_std, centered_r2 + 1.96*centered_std) + f" at L{best_layer_cen}"

        # Transfer ratio
        if d2d_r2_val and d2d_r2_val > 0:
            transfer_ratio = centered_r2 / d2d_r2_val
            if transfer_ratio > 0.6:
                strength = "Strong"
            elif transfer_ratio > 0.3:
                strength = "Moderate"
            elif transfer_ratio > 0.1:
                strength = "Weak"
            else:
                strength = "No"
            key_findings[f"Transfer ({metric})"] = f"{transfer_ratio:.0%} ({strength})"

    # Collect remaining output files (plots already added per-position in the loop above)
    for pos in positions_available:
        output_files.append(get_output_path(f"{base_output}_transfer_results_{pos}.png", model_dir=model_dir))
    if len(positions_available) > 1:
        output_files.append(plot_path_positions)
    # Note: confdir and mcuncert files are added inside the position loop above

    # Console output
    print_key_findings(key_findings)
    print_run_footer(output_files)


if __name__ == "__main__":
    main()
