"""
Mechanistic analysis: How do models produce confidence judgments that track their internal uncertainty?

Tests the key hypotheses from the plan:

1. Transfer vs Re-computation (includes output connection analysis):
   - Cosine similarity: Are d_mc and metamcuncert the same direction?
   - R² comparison: How well does each predict MC uncertainty from meta activations?
     * Transfer R²: d_mc (trained on MC) applied to meta activations → predicts MC logit_gap
     * Recomputed R²: metamcuncert (trained on meta) → predicts MC logit_gap
   - If transfer R² ≈ recomputed R²: transfer is sufficient
   - If recomputed R² >> transfer R²: recomputation adds value beyond transfer
   - Output connection: Does projection onto direction correlate with meta output?
     * Behavioral baseline: MC uncertainty → meta output (P(Answer), logit_margin)
     * Transfer (d_mc): projection → meta output
     * Recomputed (metamcuncert): projection → meta output

2. Introspection vs Surface Features:
   - Compares self-confidence vs other-confidence (human difficulty) transfer R²
   - If similar: model reads surface difficulty cues
   - If self >> other: genuine introspection

Prerequisites (scripts that must be run first):
    1. identify_mc_correlate.py
       - Produces: {dataset}_mc_*_directions.npz (d_mc directions)

    2. test_meta_transfer.py with META_TASK="delegate" and FIND_MC_UNCERTAINTY_DIRECTIONS=True
       - Produces: {dataset}_meta_delegate_transfer_results_final.json (transfer R²)
       - Produces: {dataset}_meta_delegate_mcuncert_results_final.json (recomputed R²)
       - Produces: {dataset}_meta_delegate_mcuncert_directions.npz (metamcuncert)
       - Produces: {dataset}_meta_delegate_activations.npz (for computing metamcuncert → output)

    3. test_meta_transfer.py with META_TASK="confidence"
       - Produces: {dataset}_meta_confidence_transfer_results_final.json

    4. test_meta_transfer.py with META_TASK="other_confidence"
       - Produces: {dataset}_meta_other_confidence_transfer_results_final.json

    5. analyze_direction_cosines.py (run once per task: delegate, confidence, other_confidence)
       - Produces: {dataset}_meta_{task}_direction_cosines_{metric}.json (pairwise cosine similarities)

Inputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_direction_cosines_{metric}.json (cosine sims per task)
    outputs/{model_dir}/results/{dataset}_meta_{task}_transfer_results_final.json (transfer R²)
    outputs/{model_dir}/results/{dataset}_meta_{task}_mcuncert_results_final.json (recomputed R²)
    outputs/{model_dir}/working/{dataset}_meta_{task}_mcuncert_directions_final.npz (for output correlation)
    outputs/{model_dir}/working/{dataset}_meta_{task}_activations.npz (for output correlation)

Outputs:
    outputs/{model_dir}/results/{dataset}_hypothesis_tests_summary_{metric}.json
    outputs/{model_dir}/results/{dataset}_hypothesis_tests_transfer_vs_recomputation_{metric}.png
    outputs/{model_dir}/results/{dataset}_hypothesis_tests_self_vs_other_{metric}.png

Notes on inference:
    Per-layer quantities are not independent, so p-values and CIs are heuristics.
    Test 1 compares R² values; it does not have a principled null model for cosine similarity.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, t as t_dist

from core.config_utils import find_output_file, get_config_dict, get_output_path
from core.model_utils import get_model_dir_name
from core.plotting import GRID_ALPHA, save_figure, TASK_COLORS

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

DATASET = "TriviaMC_difficulty_filtered"
METRIC = "logit_gap"
DIRECTION_METHODS = ("probe", "mean_diff")  # Compare both methods side-by-side

BOOTSTRAP_SAMPLES = 10_000
PERMUTATION_SAMPLES = 50_000
RNG_SEED = 0
TRANSFER_HIGH_THRESHOLD = 0.70
TRANSFER_LOW_THRESHOLD = 0.30
TASKS_FOR_TEST1 = ("delegate", "confidence", "other_confidence")


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SummaryStats:
    n: int
    mean: float
    median: float
    sd: Optional[float]
    min: float
    max: float
    ci_mean: Optional[Tuple[float, float]] = None
    ci_median: Optional[Tuple[float, float]] = None


@dataclass
class R2Comparison:
    """Comparison of d_mc transfer R² vs metamcuncert R² (recomputed direction)."""
    layers: List[int]
    transfer_r2_per_layer: Dict[int, float]  # d_mc applied to meta activations
    recomputed_r2_per_layer: Dict[int, float]  # metamcuncert trained on meta activations
    diff_per_layer: Dict[int, float]  # recomputed - transfer (positive = recomputation adds value)
    transfer_summary: SummaryStats
    recomputed_summary: SummaryStats
    diff_summary: SummaryStats
    paired_test: Dict[str, Any]


@dataclass
class MethodResult:
    """Results for a single direction method (probe or mean_diff)."""
    cosine_per_layer: Dict[int, float]
    abs_cosine_per_layer: Dict[int, float]
    summary_signed: SummaryStats
    summary_abs: SummaryStats
    r2_comparison: Optional[R2Comparison]
    # Output connection: correlation between direction projection and meta output
    # For both P(Answer) and logit_margin targets
    transfer_output_p_answer: Dict[int, float]  # d_mc → P(Answer)
    transfer_output_logit_margin: Dict[int, float]  # d_mc → logit_margin
    recomputed_output_p_answer: Dict[int, float]  # metamcuncert → P(Answer)
    recomputed_output_logit_margin: Dict[int, float]  # metamcuncert → logit_margin
    # Additivity analysis: do d_mc and metamcuncert contribute independently?
    additivity_p_answer: Optional[AdditivityResult]  # For P(Answer) target
    additivity_logit_margin: Optional[AdditivityResult]  # For logit_margin target
    interpretation: str


@dataclass
class TransferTaskResult:
    """Results for a single meta-task, with separate results per direction method."""
    task: str
    layers: List[int]
    by_method: Dict[str, MethodResult]  # Keyed by "probe" or "mean_diff"
    # Behavioral baselines: correlation between MC uncertainty and meta output
    behavioral_p_answer: Dict[str, Any]  # {pearson_r, spearman_r, p_value}
    behavioral_logit_margin: Dict[str, Any]  # {pearson_r, spearman_r, p_value}
    pooled_interpretation: str = ""  # Overall interpretation across methods
    notes: List[str] = field(default_factory=list)


@dataclass
class SelfOtherResult:
    layers: List[int]
    self_r2_per_layer: Dict[int, float]
    other_r2_per_layer: Dict[int, float]
    diff_r2_per_layer: Dict[int, float]
    self_r2_summary: SummaryStats
    other_r2_summary: SummaryStats
    diff_r2_summary: SummaryStats
    paired_test: Dict[str, Any]
    behavioral: Dict[str, Any]
    interpretation: str
    notes: List[str] = field(default_factory=list)


@dataclass
class LayerAdditivityResult:
    """Per-layer regression results for additivity analysis."""
    r2_transfer: float  # R² using d_mc projection alone
    r2_recomputed: float  # R² using metamcuncert projection alone
    r2_combined: float  # R² using both projections
    beta_transfer: float  # Coefficient for d_mc in combined model
    beta_recomputed: float  # Coefficient for metamcuncert in combined model
    p_transfer: float  # p-value for d_mc coefficient
    p_recomputed: float  # p-value for metamcuncert coefficient
    unique_var_transfer: float  # ΔR² when adding d_mc to metamcuncert
    unique_var_recomputed: float  # ΔR² when adding metamcuncert to d_mc


@dataclass
class AdditivityResult:
    """Results of additivity analysis: do d_mc and metamcuncert contribute independently?"""
    output_target: str  # "confidences" or "logit_margins"
    layers: List[int]
    per_layer: Dict[int, LayerAdditivityResult]
    # Summary statistics across layers
    mean_r2_transfer: float
    mean_r2_recomputed: float
    mean_r2_combined: float
    mean_unique_var_transfer: float  # Average unique contribution of d_mc
    mean_unique_var_recomputed: float  # Average unique contribution of metamcuncert
    # Best layer results
    best_layer: int
    best_r2_combined: float
    # Interpretation
    additivity_ratio: float  # (R²_combined - max) / min; 0=redundant, 1=fully additive
    interpretation: str


# =============================================================================
# GENERIC HELPERS
# =============================================================================


def json_ready(obj: Any) -> Any:
    """Recursively convert numpy / dataclass objects into JSON-safe Python types."""
    if hasattr(obj, "__dataclass_fields__"):
        return json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.floating,)):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float,)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _finite_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return arr
    return arr[np.isfinite(arr)]


def bootstrap_ci(
    values: Sequence[float],
    statistic,
    rng: np.random.Generator,
    n_boot: int = BOOTSTRAP_SAMPLES,
    alpha: float = 0.05,
) -> Optional[Tuple[float, float]]:
    arr = _finite_array(values)
    if arr.size == 0:
        return None
    if arr.size == 1:
        x = float(statistic(arr))
        return (x, x)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    sampled = arr[idx]
    stats = np.asarray([statistic(row) for row in sampled], dtype=float)
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return (lo, hi)


def summarize_values(values: Sequence[float], rng: np.random.Generator) -> SummaryStats:
    arr = _finite_array(values)
    if arr.size == 0:
        raise ValueError("Cannot summarize an empty set of values")
    sd = float(np.std(arr, ddof=1)) if arr.size >= 2 else None
    return SummaryStats(
        n=int(arr.size),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        sd=sd,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
        ci_mean=bootstrap_ci(arr, np.mean, rng),
        ci_median=bootstrap_ci(arr, np.median, rng),
    )


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    rng: np.random.Generator,
    n_perm: int = PERMUTATION_SAMPLES,
) -> Dict[str, Any]:
    """
    Two-sided paired sign-flip permutation test on mean(a - b).

    Exact enumeration is used for small n; Monte Carlo otherwise.
    """
    x = _finite_array(a)
    y = _finite_array(b)
    if x.size != y.size:
        raise ValueError("Paired test requires equal-length arrays")
    if x.size == 0:
        raise ValueError("Paired test requires at least one observation")

    diffs = x - y
    observed = float(np.mean(diffs))
    abs_observed = abs(observed)
    n = diffs.size

    if np.allclose(diffs, 0.0):
        return {
            "n": int(n),
            "observed_mean_diff": observed,
            "p_value_two_sided": 1.0,
            "method": "degenerate_all_zero",
            "cohens_dz": 0.0,
        }

    # Cohen's dz for paired designs.
    sd = float(np.std(diffs, ddof=1)) if n >= 2 else 0.0
    cohens_dz = float(observed / sd) if sd > 0 else None

    if n <= 18:
        # Exact enumeration over all sign flips.
        total = 1 << n
        extreme = 0
        for mask in range(total):
            signs = np.ones(n, dtype=float)
            for bit in range(n):
                if (mask >> bit) & 1:
                    signs[bit] = -1.0
            stat = abs(float(np.mean(diffs * signs)))
            if stat >= abs_observed - 1e-15:
                extreme += 1
        p_value = extreme / total
        method = f"exact_sign_flip_{total}_enumerations"
    else:
        signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
        perm_stats = np.abs(np.mean(signs * diffs[None, :], axis=1))
        extreme = int(np.sum(perm_stats >= abs_observed - 1e-15))
        p_value = (extreme + 1) / (n_perm + 1)
        method = f"monte_carlo_sign_flip_{n_perm}"

    return {
        "n": int(n),
        "observed_mean_diff": observed,
        "p_value_two_sided": float(p_value),
        "method": method,
        "cohens_dz": cohens_dz,
    }


def safe_int_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
    result: Dict[int, Any] = {}
    for k, v in d.items():
        try:
            result[int(k)] = v
        except (TypeError, ValueError):
            continue
    return result


def format_ci(ci: Optional[Tuple[float, float]], digits: int = 3) -> str:
    if ci is None:
        return "n/a"
    return f"[{ci[0]:.{digits}f}, {ci[1]:.{digits}f}]"


# =============================================================================
# LOADING / EXTRACTION
# =============================================================================


def load_json_file(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def load_direction_cosines(dataset: str, task: str, metric: str, model_dir: str) -> Optional[Dict[str, Any]]:
    """Load direction cosine analysis for a specific task.

    Args:
        dataset: Dataset name
        task: Meta-task name (delegate, confidence, other_confidence)
        metric: Uncertainty metric (logit_gap, entropy)
        model_dir: Model directory name

    Returns:
        Loaded JSON or None if not found
    """
    path = find_output_file(f"{dataset}_meta_{task}_direction_cosines_{metric}.json", model_dir=model_dir)
    if not path.exists():
        return None
    return load_json_file(path)


def load_transfer_results(base_name: str, task: str, model_dir: str) -> Optional[Dict[str, Any]]:
    path = find_output_file(f"{base_name}_meta_{task}_transfer_results_final.json", model_dir=model_dir)
    if not path.exists():
        return None
    return load_json_file(path)


def load_mcuncert_results(base_name: str, task: str, model_dir: str) -> Optional[Dict[str, Any]]:
    """Load metamcuncert results (direction trained on meta activations to predict MC uncertainty)."""
    path = find_output_file(f"{base_name}_meta_{task}_mcuncert_results_final.json", model_dir=model_dir)
    if not path.exists():
        return None
    return load_json_file(path)


def extract_cosine_sim_per_layer(
    direction_cosines: Optional[Dict[str, Any]],
    method: str,
) -> Dict[int, float]:
    """Extract cosine similarity between d_mc and metamcuncert directions.

    Args:
        direction_cosines: Loaded direction_cosines JSON (from analyze_direction_cosines.py)
        method: Direction method (only "mean_diff" supported by new format)

    Returns:
        Dict mapping layer -> cosine similarity
    """
    if direction_cosines is None:
        return {}

    # New format only supports mean_diff
    if method != "mean_diff":
        return {}

    comparisons = direction_cosines.get("comparisons", {})
    mc_vs_meta = comparisons.get("d_mc__vs__d_metamcuncert", {})
    per_layer_raw = mc_vs_meta.get("per_layer", {})

    result: Dict[int, float] = {}
    for layer_str, value in per_layer_raw.items():
        try:
            layer = int(layer_str)
            value_f = float(value)
            if math.isfinite(value_f):
                result[layer] = value_f
        except (TypeError, ValueError):
            continue

    return dict(sorted(result.items()))


def extract_transfer_r2_per_layer(
    transfer_results: Dict[str, Any],
    metric: str,
    method: str = "probe",
) -> Dict[int, float]:
    """Extract per-layer transfer R² for a specific method.

    Args:
        transfer_results: Loaded transfer results JSON
        metric: Uncertainty metric (logit_gap, entropy)
        method: Direction method - "probe" uses "transfer" section,
                "mean_diff" uses "mean_diff_transfer" section

    Returns:
        Dict mapping layer -> R² value
    """
    if not transfer_results:
        return {}

    # Map method to JSON key
    transfer_key = "transfer" if method == "probe" else "mean_diff_transfer"
    metric_block = transfer_results.get(transfer_key, {}).get(metric, {})
    per_layer = metric_block.get("per_layer", {})

    result: Dict[int, float] = {}
    for layer, layer_data in safe_int_keys(per_layer).items():
        if not isinstance(layer_data, dict):
            continue
        value = layer_data.get("centered_r2")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            result[layer] = value_f

    return dict(sorted(result.items()))


def extract_mcuncert_r2_per_layer(
    mcuncert_results: Dict[str, Any],
    metric: str,
    method: str = "probe",
) -> Dict[int, float]:
    """Extract per-layer R² from metamcuncert results (recomputed direction).

    Args:
        mcuncert_results: Loaded mcuncert results JSON
        metric: Uncertainty metric (logit_gap, entropy)
        method: Direction method (probe or mean_diff)

    Returns:
        Dict mapping layer -> R² value
    """
    if not mcuncert_results:
        return {}

    # Structure: metrics.{metric}.results.{method}.{layer}.test_r2
    metric_block = mcuncert_results.get("metrics", {}).get(metric, {})
    method_results = metric_block.get("results", {}).get(method, {})

    result: Dict[int, float] = {}
    for layer, layer_data in safe_int_keys(method_results).items():
        if not isinstance(layer_data, dict):
            continue
        value = layer_data.get("test_r2")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            result[layer] = value_f

    return dict(sorted(result.items()))


def extract_behavioral_block(transfer_results: Optional[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    if not transfer_results:
        return {}
    behavioral = transfer_results.get("behavioral", {})
    metric_block = behavioral.get(metric, {}) if isinstance(behavioral, dict) else {}
    return metric_block if isinstance(metric_block, dict) else {}


def extract_pred_conf_pearson_per_layer(
    transfer_results: Dict[str, Any],
    metric: str,
    method: str = "probe",
) -> Dict[int, float]:
    """Extract per-layer correlation between direction projection and meta output.

    This tests whether the uncertainty direction is connected to the model's output.

    Args:
        transfer_results: Loaded transfer results JSON
        metric: Uncertainty metric (logit_gap, entropy)
        method: Direction method - "probe" uses "transfer", "mean_diff" uses "mean_diff_transfer"

    Returns:
        Dict mapping layer -> pred_conf_pearson (correlation with meta output)
    """
    if not transfer_results:
        return {}

    transfer_key = "transfer" if method == "probe" else "mean_diff_transfer"
    metric_block = transfer_results.get(transfer_key, {}).get(metric, {})
    per_layer = metric_block.get("per_layer", {})

    result: Dict[int, float] = {}
    for layer, layer_data in safe_int_keys(per_layer).items():
        if not isinstance(layer_data, dict):
            continue
        value = layer_data.get("centered_pred_conf_pearson")
        if value is None:
            continue
        try:
            value_f = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value_f):
            result[layer] = value_f

    return dict(sorted(result.items()))


def extract_behavioral_logit_margin_block(
    transfer_results: Optional[Dict[str, Any]], metric: str
) -> Dict[str, Any]:
    """Extract behavioral correlation with logit_margin instead of P(Answer)."""
    if not transfer_results:
        return {}
    behavioral = transfer_results.get("behavioral_logit_margin", {})
    metric_block = behavioral.get(metric, {}) if isinstance(behavioral, dict) else {}
    return metric_block if isinstance(metric_block, dict) else {}


def compute_metamcuncert_output_correlation(
    dataset: str,
    task: str,
    metric: str,
    method: str,
    model_dir: str,
    output_target: str = "confidences",
) -> Dict[int, float]:
    """Compute correlation between metamcuncert projection and meta output.

    Checks for pre-computed data first (future-proofing), then falls back to
    computing from directions + activations.

    Args:
        dataset: Dataset name
        task: Meta-task name (delegate, confidence, other_confidence)
        metric: Uncertainty metric (logit_gap, entropy)
        method: Direction method (probe or mean_diff)
        model_dir: Model directory name
        output_target: Which meta output to correlate with ("confidences" or "logit_margins")

    Returns:
        Dict mapping layer -> Pearson r correlation
    """
    # Future-proofing: Check if mcuncert results have pre-computed pred_conf_pearson
    mcuncert_results = load_mcuncert_results(dataset, task, model_dir)
    if mcuncert_results:
        method_results = (
            mcuncert_results.get("metrics", {})
            .get(metric, {})
            .get("results", {})
            .get(method, {})
        )
        # Check for pre-computed values (not yet implemented in test_meta_transfer.py)
        has_precomputed = any(
            isinstance(v, dict) and "pred_conf_pearson" in v
            for v in method_results.values()
        )
        if has_precomputed:
            # Extract pre-computed values if they exist
            result = {}
            for layer_str, layer_data in method_results.items():
                try:
                    layer = int(layer_str)
                except (TypeError, ValueError):
                    continue
                if isinstance(layer_data, dict) and "pred_conf_pearson" in layer_data:
                    val = layer_data["pred_conf_pearson"]
                    if val is not None and math.isfinite(float(val)):
                        result[layer] = float(val)
            if result:
                return dict(sorted(result.items()))

    # Fall back to computing from directions + activations
    dir_path = find_output_file(
        f"{dataset}_meta_{task}_mcuncert_directions_final.npz", model_dir=model_dir
    )
    if not dir_path.exists():
        return {}

    act_path = find_output_file(
        f"{dataset}_meta_{task}_activations.npz", model_dir=model_dir
    )
    if not act_path.exists():
        return {}

    directions = np.load(dir_path)
    activations = np.load(act_path)

    if output_target not in activations.files:
        return {}
    meta_output = activations[output_target]

    # Handle NaN values (logit_margins can be NaN for non-delegate tasks)
    valid_mask = ~np.isnan(meta_output)
    if not np.any(valid_mask):
        return {}

    result: Dict[int, float] = {}
    layer = 0
    while True:
        dir_key = f"{method}_{metric}_layer_{layer}"
        act_key = f"layer_{layer}_final"

        if dir_key not in directions.files or act_key not in activations.files:
            break

        direction = directions[dir_key]
        acts = activations[act_key]
        projections = acts @ direction

        proj_valid = projections[valid_mask]
        output_valid = meta_output[valid_mask]

        if len(proj_valid) > 2:
            r, _ = pearsonr(proj_valid, output_valid)
            if math.isfinite(r):
                result[layer] = float(r)

        layer += 1

    return dict(sorted(result.items()))


def compute_transfer_output_correlation(
    dataset: str,
    task: str,
    metric: str,
    method: str,
    model_dir: str,
    output_target: str = "confidences",
) -> Dict[int, float]:
    """Compute d_mc projection → meta output correlation.

    Uses pre-computed pred_conf_pearson if available (for P(Answer)),
    otherwise computes from MC directions + meta activations.

    Args:
        dataset: Dataset name
        task: Meta-task name (delegate, confidence, other_confidence)
        metric: Uncertainty metric (logit_gap, entropy)
        method: Direction method (probe or mean_diff)
        model_dir: Model directory name
        output_target: Which meta output to correlate with ("confidences" or "logit_margins")

    Returns:
        Dict mapping layer -> Pearson r correlation
    """
    # For P(Answer), try to use pre-computed from JSON first
    if output_target == "confidences":
        transfer_results = load_transfer_results(dataset, task, model_dir)
        precomputed = extract_pred_conf_pearson_per_layer(transfer_results, metric, method)
        if precomputed:
            return precomputed

    # Fall back to computing from scratch using MC directions + meta activations
    # Load d_mc direction from MC task
    dir_path = find_output_file(f"{dataset}_mc_{metric}_directions.npz", model_dir=model_dir)
    if not dir_path.exists():
        return {}

    # Load meta activations
    act_path = find_output_file(f"{dataset}_meta_{task}_activations.npz", model_dir=model_dir)
    if not act_path.exists():
        return {}

    directions = np.load(dir_path)
    activations = np.load(act_path)

    if output_target not in activations.files:
        return {}
    meta_output = activations[output_target]

    # Handle NaN values (logit_margins can be NaN for non-delegate tasks)
    valid_mask = ~np.isnan(meta_output)
    if not np.any(valid_mask):
        return {}

    result: Dict[int, float] = {}
    layer = 0
    while True:
        # MC direction keys use format: {method}_layer_{layer}
        dir_key = f"{method}_layer_{layer}"
        act_key = f"layer_{layer}_final"

        if dir_key not in directions.files or act_key not in activations.files:
            break

        direction = directions[dir_key]
        acts = activations[act_key]
        projections = acts @ direction

        proj_valid = projections[valid_mask]
        output_valid = meta_output[valid_mask]

        if len(proj_valid) > 2:
            r, _ = pearsonr(proj_valid, output_valid)
            if math.isfinite(r):
                result[layer] = float(r)

        layer += 1

    return dict(sorted(result.items()))


def compute_additivity_analysis(
    dataset: str,
    task: str,
    metric: str,
    method: str,
    model_dir: str,
    output_target: str = "confidences",
) -> Optional[AdditivityResult]:
    """Analyze whether d_mc and metamcuncert contribute independently to meta output.

    Uses multiple regression to test additivity:
    - R²_transfer: output ~ d_mc_projection
    - R²_recomputed: output ~ metamcuncert_projection
    - R²_combined: output ~ d_mc_projection + metamcuncert_projection

    If R²_combined ≈ max(R²_transfer, R²_recomputed): redundant (same signal)
    If R²_combined ≈ R²_transfer + R²_recomputed: fully additive/independent

    Args:
        dataset: Dataset name
        task: Meta-task name
        metric: Uncertainty metric (logit_gap, entropy)
        method: Direction method (probe or mean_diff)
        model_dir: Model directory name
        output_target: Which meta output ("confidences" or "logit_margins")

    Returns:
        AdditivityResult with per-layer regression statistics
    """
    # Load d_mc directions from MC task
    mc_dir_path = find_output_file(f"{dataset}_mc_{metric}_directions.npz", model_dir=model_dir)
    if not mc_dir_path.exists():
        return None

    # Load metamcuncert directions
    meta_dir_path = find_output_file(
        f"{dataset}_meta_{task}_mcuncert_directions_final.npz", model_dir=model_dir
    )
    if not meta_dir_path.exists():
        return None

    # Load meta activations
    act_path = find_output_file(f"{dataset}_meta_{task}_activations.npz", model_dir=model_dir)
    if not act_path.exists():
        return None

    mc_directions = np.load(mc_dir_path)
    meta_directions = np.load(meta_dir_path)
    activations = np.load(act_path)

    if output_target not in activations.files:
        return None
    meta_output = activations[output_target]

    # Handle NaN values
    valid_mask = ~np.isnan(meta_output)
    if np.sum(valid_mask) < 10:
        return None

    y = meta_output[valid_mask]
    n = len(y)

    per_layer: Dict[int, LayerAdditivityResult] = {}
    layer = 0

    while True:
        # Key formats differ between MC and meta directions
        mc_key = f"{method}_layer_{layer}"
        meta_key = f"{method}_{metric}_layer_{layer}"
        act_key = f"layer_{layer}_final"

        if mc_key not in mc_directions.files or meta_key not in meta_directions.files:
            break
        if act_key not in activations.files:
            break

        acts = activations[act_key][valid_mask]
        d_mc = mc_directions[mc_key]
        d_meta = meta_directions[meta_key]

        # Compute projections
        proj_mc = acts @ d_mc
        proj_meta = acts @ d_meta

        # Standardize for comparable coefficients
        proj_mc_std = (proj_mc - np.mean(proj_mc)) / (np.std(proj_mc) + 1e-10)
        proj_meta_std = (proj_meta - np.mean(proj_meta)) / (np.std(proj_meta) + 1e-10)
        y_std = (y - np.mean(y)) / (np.std(y) + 1e-10)

        # Single regressions (R² = correlation² for single predictor)
        r_mc, _ = pearsonr(proj_mc, y)
        r_meta, _ = pearsonr(proj_meta, y)
        r2_transfer = r_mc ** 2
        r2_recomputed = r_meta ** 2

        # Combined regression: y ~ proj_mc + proj_meta
        X = np.column_stack([np.ones(n), proj_mc_std, proj_meta_std])
        try:
            # OLS: beta = (X'X)^-1 X'y
            XtX_inv = np.linalg.inv(X.T @ X)
            beta = XtX_inv @ X.T @ y_std
            y_pred = X @ beta
            ss_res = np.sum((y_std - y_pred) ** 2)
            ss_tot = np.sum((y_std - np.mean(y_std)) ** 2)
            r2_combined = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            # Standard errors and p-values for coefficients
            mse = ss_res / (n - 3)  # 3 parameters: intercept + 2 predictors
            se = np.sqrt(np.diag(XtX_inv) * mse)
            t_stats = beta / (se + 1e-10)

            # Two-tailed p-values from t-distribution
            p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n - 3))

            beta_transfer = float(beta[1])
            beta_recomputed = float(beta[2])
            p_transfer = float(p_values[1])
            p_recomputed = float(p_values[2])

        except np.linalg.LinAlgError:
            # Singular matrix - projections are collinear, can't estimate coefficients
            r2_combined = max(r2_transfer, r2_recomputed)
            beta_transfer = float("nan")
            beta_recomputed = float("nan")
            p_transfer = float("nan")
            p_recomputed = float("nan")

        # Unique variance contributions (clamped to non-negative)
        unique_var_transfer = max(0.0, r2_combined - r2_recomputed)  # ΔR² when adding d_mc
        unique_var_recomputed = max(0.0, r2_combined - r2_transfer)  # ΔR² when adding metamcuncert

        per_layer[layer] = LayerAdditivityResult(
            r2_transfer=float(r2_transfer),
            r2_recomputed=float(r2_recomputed),
            r2_combined=float(r2_combined),
            beta_transfer=beta_transfer,
            beta_recomputed=beta_recomputed,
            p_transfer=p_transfer,
            p_recomputed=p_recomputed,
            unique_var_transfer=float(unique_var_transfer),
            unique_var_recomputed=float(unique_var_recomputed),
        )

        layer += 1

    if not per_layer:
        return None

    layers = sorted(per_layer.keys())

    # Summary statistics
    mean_r2_transfer = float(np.mean([per_layer[l].r2_transfer for l in layers]))
    mean_r2_recomputed = float(np.mean([per_layer[l].r2_recomputed for l in layers]))
    mean_r2_combined = float(np.mean([per_layer[l].r2_combined for l in layers]))
    mean_unique_transfer = float(np.mean([per_layer[l].unique_var_transfer for l in layers]))
    mean_unique_recomputed = float(np.mean([per_layer[l].unique_var_recomputed for l in layers]))

    # Best layer by combined R²
    best_layer = max(layers, key=lambda l: per_layer[l].r2_combined)
    best_r2_combined = per_layer[best_layer].r2_combined

    # Additivity ratio: how much of the "potential" additive gain is realized?
    # If fully redundant: combined = max(transfer, recomputed), ratio = 0
    # If fully additive: combined = transfer + recomputed, ratio = 1
    max_single = max(mean_r2_transfer, mean_r2_recomputed)
    min_single = min(mean_r2_transfer, mean_r2_recomputed)
    if min_single > 0.001:
        additivity_ratio = (mean_r2_combined - max_single) / min_single
        additivity_ratio = float(np.clip(additivity_ratio, 0, 1))
    else:
        additivity_ratio = 0.0

    # Interpretation
    if additivity_ratio > 0.5:
        interpretation = "substantially_additive"
    elif additivity_ratio > 0.2:
        interpretation = "partially_additive"
    elif additivity_ratio > 0.05:
        interpretation = "mostly_redundant_some_unique"
    else:
        interpretation = "largely_redundant"

    return AdditivityResult(
        output_target=output_target,
        layers=layers,
        per_layer=per_layer,
        mean_r2_transfer=mean_r2_transfer,
        mean_r2_recomputed=mean_r2_recomputed,
        mean_r2_combined=mean_r2_combined,
        mean_unique_var_transfer=mean_unique_transfer,
        mean_unique_var_recomputed=mean_unique_recomputed,
        best_layer=best_layer,
        best_r2_combined=best_r2_combined,
        additivity_ratio=additivity_ratio,
        interpretation=interpretation,
    )


# =============================================================================
# ANALYSIS
# =============================================================================


def interpret_transfer(summary_abs: SummaryStats) -> str:
    lo_hi = summary_abs.ci_mean
    if lo_hi is not None and lo_hi[0] > TRANSFER_HIGH_THRESHOLD:
        return "strong_support_for_transfer"
    if lo_hi is not None and lo_hi[1] < TRANSFER_LOW_THRESHOLD:
        return "strong_support_for_recomputation"
    if summary_abs.mean >= TRANSFER_HIGH_THRESHOLD:
        return "suggestive_transfer_but_ci_overlaps_boundary"
    if summary_abs.mean <= TRANSFER_LOW_THRESHOLD:
        return "suggestive_recomputation_but_ci_overlaps_boundary"
    return "mixed_or_inconclusive"


def compute_r2_comparison(
    transfer_r2: Dict[int, float],
    recomputed_r2: Dict[int, float],
    rng: np.random.Generator,
) -> Optional[R2Comparison]:
    """Compare d_mc transfer R² vs metamcuncert R² per layer."""
    common_layers = sorted(set(transfer_r2.keys()) & set(recomputed_r2.keys()))
    if not common_layers:
        return None

    transfer_vals = np.array([transfer_r2[l] for l in common_layers], dtype=float)
    recomputed_vals = np.array([recomputed_r2[l] for l in common_layers], dtype=float)
    diff_vals = recomputed_vals - transfer_vals  # positive = recomputed predicts better

    return R2Comparison(
        layers=common_layers,
        transfer_r2_per_layer={l: transfer_r2[l] for l in common_layers},
        recomputed_r2_per_layer={l: recomputed_r2[l] for l in common_layers},
        diff_per_layer={l: float(recomputed_r2[l] - transfer_r2[l]) for l in common_layers},
        transfer_summary=summarize_values(transfer_vals, rng),
        recomputed_summary=summarize_values(recomputed_vals, rng),
        diff_summary=summarize_values(diff_vals, rng),
        paired_test=paired_permutation_test(recomputed_vals, transfer_vals, rng),
    )


def interpret_r2_comparison(r2_comp: Optional[R2Comparison]) -> str:
    if r2_comp is None:
        return "no_r2_data_available"

    diff_ci = r2_comp.diff_summary.ci_mean
    transfer_mean = r2_comp.transfer_summary.mean
    recomputed_mean = r2_comp.recomputed_summary.mean

    # Recomputed significantly exceeds transfer
    if diff_ci is not None and diff_ci[0] > 0:
        if transfer_mean < 0.1:
            return "recomputation_dominant_transfer_negligible"
        elif recomputed_mean > transfer_mean * 1.5:
            return "both_contribute_recomputation_substantially_better"
        else:
            return "both_contribute_recomputation_modestly_better"

    # Similar performance
    if diff_ci is not None and diff_ci[0] <= 0 <= diff_ci[1]:
        if transfer_mean > 0.3 and recomputed_mean > 0.3:
            return "both_strong_no_significant_difference"
        elif transfer_mean < 0.1 and recomputed_mean < 0.1:
            return "both_weak"
        else:
            return "similar_moderate_performance"

    # Transfer exceeds recomputed (unusual)
    if diff_ci is not None and diff_ci[1] < 0:
        return "transfer_exceeds_recomputed_unexpected"

    return "inconclusive"


def analyze_transfer_vs_recomputation(
    dataset: str,
    metric: str,
    model_dir: str,
    rng: np.random.Generator,
    methods: Tuple[str, ...] = DIRECTION_METHODS,
) -> Dict[str, TransferTaskResult]:
    """Analyze transfer vs recomputation for each meta-task, comparing methods side-by-side.

    For each task and method:
    - Cosine similarity between d_mc and metamcuncert
    - R² comparison: transfer (d_mc → meta) vs recomputed (metamcuncert)
    """
    results: Dict[str, TransferTaskResult] = {}

    for task in TASKS_FOR_TEST1:
        # Load data once per task
        transfer_results = load_transfer_results(dataset, task, model_dir)
        mcuncert_results = load_mcuncert_results(dataset, task, model_dir)
        direction_cosines = load_direction_cosines(dataset, task, metric, model_dir)

        by_method: Dict[str, MethodResult] = {}
        all_layers: set = set()
        task_notes: List[str] = []

        for method in methods:
            # Extract cosine similarity for this method (only mean_diff supported)
            cosine_per_layer = extract_cosine_sim_per_layer(direction_cosines, method)
            if not cosine_per_layer:
                if method == "mean_diff":
                    print(f"    Warning: No cosine similarity data for {task}/{method}")
                continue

            layers = list(cosine_per_layer.keys())
            all_layers.update(layers)
            signed_vals = np.array([cosine_per_layer[l] for l in layers], dtype=float)
            abs_vals = np.abs(signed_vals)
            abs_cosine_per_layer = {l: float(abs(cosine_per_layer[l])) for l in layers}

            summary_signed = summarize_values(signed_vals, rng)
            summary_abs = summarize_values(abs_vals, rng)

            # Extract transfer R² for this method (probe uses "transfer", mean_diff uses "mean_diff_transfer")
            transfer_r2 = extract_transfer_r2_per_layer(transfer_results, metric, method) if transfer_results else {}

            # Extract recomputed R² for this method
            recomputed_r2 = extract_mcuncert_r2_per_layer(mcuncert_results, metric, method) if mcuncert_results else {}

            # R² comparison: method-specific transfer vs method-specific recomputed
            r2_comparison = compute_r2_comparison(transfer_r2, recomputed_r2, rng)

            # Output connection: correlation between direction projection and meta output
            # Compute for both P(Answer) and logit_margin targets
            # d_mc (transfer) → meta outputs
            transfer_output_p_answer = compute_transfer_output_correlation(
                dataset, task, metric, method, model_dir, output_target="confidences"
            )
            transfer_output_logit_margin = compute_transfer_output_correlation(
                dataset, task, metric, method, model_dir, output_target="logit_margins"
            )
            # metamcuncert (recomputed) → meta outputs
            recomputed_output_p_answer = compute_metamcuncert_output_correlation(
                dataset, task, metric, method, model_dir, output_target="confidences"
            )
            recomputed_output_logit_margin = compute_metamcuncert_output_correlation(
                dataset, task, metric, method, model_dir, output_target="logit_margins"
            )

            # Additivity analysis: do d_mc and metamcuncert contribute independently?
            additivity_p_answer = compute_additivity_analysis(
                dataset, task, metric, method, model_dir, output_target="confidences"
            )
            additivity_logit_margin = compute_additivity_analysis(
                dataset, task, metric, method, model_dir, output_target="logit_margins"
            )

            # Interpretation
            if r2_comparison is not None:
                r2_interp = interpret_r2_comparison(r2_comparison)
                cosine_interp = interpret_transfer(summary_abs)
                interpretation = f"cosine:{cosine_interp}|r2:{r2_interp}"
            else:
                interpretation = interpret_transfer(summary_abs)
                if method == methods[0]:  # Only note once
                    task_notes.append("No R² comparison available (missing mcuncert or transfer results).")

            by_method[method] = MethodResult(
                cosine_per_layer=cosine_per_layer,
                abs_cosine_per_layer=abs_cosine_per_layer,
                summary_signed=summary_signed,
                summary_abs=summary_abs,
                r2_comparison=r2_comparison,
                transfer_output_p_answer=transfer_output_p_answer,
                transfer_output_logit_margin=transfer_output_logit_margin,
                recomputed_output_p_answer=recomputed_output_p_answer,
                recomputed_output_logit_margin=recomputed_output_logit_margin,
                additivity_p_answer=additivity_p_answer,
                additivity_logit_margin=additivity_logit_margin,
                interpretation=interpretation,
            )

        if not by_method:
            continue

        # Extract behavioral baselines (once per task)
        behavioral_p_answer = extract_behavioral_block(transfer_results, metric)
        behavioral_logit_margin = extract_behavioral_logit_margin_block(transfer_results, metric)

        # Overall interpretation across methods
        if len(by_method) > 1 and all(m.summary_abs.n >= 5 for m in by_method.values()):
            # Compare methods
            method_means = {m: data.summary_abs.mean for m, data in by_method.items()}
            best_method = max(method_means, key=method_means.get)
            worst_method = min(method_means, key=method_means.get)
            diff = method_means[best_method] - method_means[worst_method]
            if diff > 0.1:
                pooled_interp = f"{best_method}_substantially_better"
            elif diff > 0.02:
                pooled_interp = f"{best_method}_modestly_better"
            else:
                pooled_interp = "methods_similar"
        else:
            pooled_interp = "single_method_or_insufficient_data"

        results[task] = TransferTaskResult(
            task=task,
            layers=sorted(all_layers),
            by_method=by_method,
            behavioral_p_answer=behavioral_p_answer,
            behavioral_logit_margin=behavioral_logit_margin,
            pooled_interpretation=pooled_interp,
            notes=task_notes,
        )

    return results


def interpret_self_vs_other(result: SelfOtherResult) -> str:
    ci = result.diff_r2_summary.ci_mean
    if ci is not None and ci[0] > 0:
        return "self_transfer_exceeds_other"
    if ci is not None and ci[1] < 0:
        return "other_transfer_exceeds_self"
    return "no_significant_difference"


def analyze_self_vs_other(
    self_transfer: Dict[str, Any],
    other_transfer: Dict[str, Any],
    metric: str,
    rng: np.random.Generator,
) -> SelfOtherResult:
    self_r2 = extract_transfer_r2_per_layer(self_transfer, metric)
    other_r2 = extract_transfer_r2_per_layer(other_transfer, metric)

    common_layers = sorted(set(self_r2.keys()) & set(other_r2.keys()))
    if not common_layers:
        raise ValueError("No common layers found between self and other transfer results")

    self_vals = np.array([self_r2[l] for l in common_layers], dtype=float)
    other_vals = np.array([other_r2[l] for l in common_layers], dtype=float)
    diff_vals = self_vals - other_vals

    self_summary = summarize_values(self_vals, rng)
    other_summary = summarize_values(other_vals, rng)
    diff_summary = summarize_values(diff_vals, rng)
    paired_test = paired_permutation_test(self_vals, other_vals, rng)

    self_behavioral = extract_behavioral_block(self_transfer, metric)
    other_behavioral = extract_behavioral_block(other_transfer, metric)

    behavioral_summary = {
        "self": {
            "pearson_r": json_ready(self_behavioral.get("pearson_r")),
            "spearman_r": json_ready(self_behavioral.get("spearman_r")),
            "n": json_ready(self_behavioral.get("n")),
            "p_value": json_ready(self_behavioral.get("p_value")),
        },
        "other": {
            "pearson_r": json_ready(other_behavioral.get("pearson_r")),
            "spearman_r": json_ready(other_behavioral.get("spearman_r")),
            "n": json_ready(other_behavioral.get("n")),
            "p_value": json_ready(other_behavioral.get("p_value")),
        },
    }

    # Descriptive ratio only; no formal test from the summary JSON alone.
    self_pr = self_behavioral.get("pearson_r")
    other_pr = other_behavioral.get("pearson_r")
    try:
        if self_pr is not None and other_pr is not None and float(other_pr) != 0:
            behavioral_summary["pearson_ratio_self_over_other"] = float(self_pr) / float(other_pr)
        else:
            behavioral_summary["pearson_ratio_self_over_other"] = None
    except (TypeError, ValueError, ZeroDivisionError):
        behavioral_summary["pearson_ratio_self_over_other"] = None

    notes: List[str] = []
    notes.append(
        "Behavioral correlations are reported descriptively; this script cannot test the difference in those correlations without the underlying paired observations."
    )
    if len(common_layers) < 5:
        notes.append("Very few common layers available; paired layer analysis is unstable.")

    result = SelfOtherResult(
        layers=common_layers,
        self_r2_per_layer={l: self_r2[l] for l in common_layers},
        other_r2_per_layer={l: other_r2[l] for l in common_layers},
        diff_r2_per_layer={l: float(self_r2[l] - other_r2[l]) for l in common_layers},
        self_r2_summary=self_summary,
        other_r2_summary=other_summary,
        diff_r2_summary=diff_summary,
        paired_test=paired_test,
        behavioral=behavioral_summary,
        interpretation="pending",
        notes=notes,
    )
    result.interpretation = interpret_self_vs_other(result)
    return result


def aggregate_transfer_over_tasks(
    results: Dict[str, TransferTaskResult],
    rng: np.random.Generator,
) -> Optional[Dict[str, Any]]:
    """Aggregate cosine similarity across tasks, separately per method."""
    if not results:
        return None

    # Collect all methods across tasks
    all_methods: set = set()
    for task_result in results.values():
        all_methods.update(task_result.by_method.keys())

    by_method: Dict[str, Dict[str, Any]] = {}
    for method in sorted(all_methods):
        pooled_abs: List[float] = []
        pooled_signed: List[float] = []

        for task_result in results.values():
            if method in task_result.by_method:
                method_data = task_result.by_method[method]
                pooled_signed.extend(method_data.cosine_per_layer.values())
                pooled_abs.extend(method_data.abs_cosine_per_layer.values())

        if pooled_abs:
            summary_signed = summarize_values(pooled_signed, rng)
            summary_abs = summarize_values(pooled_abs, rng)
            by_method[method] = {
                "n_task_layer_points": len(pooled_abs),
                "summary_signed": summary_signed,
                "summary_abs": summary_abs,
                "interpretation": interpret_transfer(summary_abs),
            }

    if not by_method:
        return None

    return {
        "by_method": by_method,
        "note": "Pooling task-layer points ignores dependence across adjacent layers and across tasks; treat as descriptive.",
    }


# =============================================================================
# PLOTTING
# =============================================================================

def plot_transfer_vs_recomputation(
    analysis: Dict[str, TransferTaskResult],
    pooled: Optional[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Combined plot: cosine (top), R² comparison, output connection 2×2, additivity 2×2.

    Output connection and additivity use 2×2 grids per task:
    - Rows: probe, mean_diff
    - Cols: P(Answer), logit_margin
    """
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    # Find tasks that have R² data, output connection data, or additivity data
    tasks_with_r2: List[str] = []
    tasks_with_output: List[str] = []
    tasks_with_additivity: List[str] = []
    for task, data in analysis.items():
        for method_data in data.by_method.values():
            if method_data.r2_comparison is not None:
                if task not in tasks_with_r2:
                    tasks_with_r2.append(task)
            has_output = (
                method_data.transfer_output_p_answer or method_data.transfer_output_logit_margin or
                method_data.recomputed_output_p_answer or method_data.recomputed_output_logit_margin
            )
            if has_output:
                if task not in tasks_with_output:
                    tasks_with_output.append(task)
            has_additivity = (
                method_data.additivity_p_answer is not None or
                method_data.additivity_logit_margin is not None
            )
            if has_additivity:
                if task not in tasks_with_additivity:
                    tasks_with_additivity.append(task)

    n_r2_panels = len(tasks_with_r2)
    n_output_tasks = len(tasks_with_output)
    n_additivity_tasks = len(tasks_with_additivity)

    # For 2×2 grids, we need 4 columns per task, or just use nested gridspec
    n_task_cols = max(n_r2_panels, n_output_tasks, n_additivity_tasks, 1)

    # Layout: up to 4 rows - cosine, R², output connection (2×2), additivity (2×2)
    # Output and additivity rows are taller because they contain 2×2 grids
    has_r2 = n_r2_panels > 0
    has_output = n_output_tasks > 0
    has_additivity = n_additivity_tasks > 0

    n_rows = 1 + (1 if has_r2 else 0) + (1 if has_output else 0) + (1 if has_additivity else 0)
    # Height ratios: cosine=1, R²=1, output=2 (for 2×2), additivity=2 (for 2×2)
    height_ratios = [1]
    if has_r2:
        height_ratios.append(1)
    if has_output:
        height_ratios.append(2)  # Taller for 2×2 grid
    if has_additivity:
        height_ratios.append(2)  # Taller for 2×2 grid

    fig = plt.figure(figsize=(6 * n_task_cols, 3 * sum(height_ratios)))
    gs_main = fig.add_gridspec(n_rows, n_task_cols, height_ratios=height_ratios, hspace=0.4, wspace=0.3)

    # --- Row 0: Cosine similarity (spans all columns) ---
    ax_cosine = fig.add_subplot(gs_main[0, :])
    method_styles = {"probe": "-", "mean_diff": "--"}

    for task, data in analysis.items():
        color = TASK_COLORS.get(task, "gray")
        for method, method_data in data.by_method.items():
            layers = sorted(method_data.cosine_per_layer.keys())
            cosines = [method_data.cosine_per_layer[l] for l in layers]
            linestyle = method_styles.get(method, "-")

            label = f"{task}/{method} (|cos|={method_data.summary_abs.mean:.2f})"
            ax_cosine.plot(layers, cosines, label=label, color=color, linestyle=linestyle, linewidth=2)

    ax_cosine.axhline(TRANSFER_HIGH_THRESHOLD, color="red", linestyle=":", alpha=0.45, label="transfer threshold")
    ax_cosine.axhline(TRANSFER_LOW_THRESHOLD, color="orange", linestyle=":", alpha=0.45, label="recomp threshold")
    ax_cosine.axhline(0.0, color="gray", linestyle="-", alpha=0.3)
    ax_cosine.set_xlabel("Layer")
    ax_cosine.set_ylabel("Cosine similarity")
    ax_cosine.set_title("Direction Alignment: cos(d_mc, metamcuncert)", fontsize=11)
    ax_cosine.grid(True, alpha=GRID_ALPHA)
    ax_cosine.legend(loc="lower right", fontsize=8, ncol=2)

    # Add pooled summary text
    if pooled is not None and "by_method" in pooled:
        text_lines = []
        for method, method_pooled in pooled["by_method"].items():
            pooled_abs: SummaryStats = method_pooled["summary_abs"]
            text_lines.append(f"{method}: |cos|={pooled_abs.mean:.3f} {format_ci(pooled_abs.ci_mean)}")
        text = "Pooled:\n" + "\n".join(text_lines)
        ax_cosine.text(
            0.02, 0.98, text,
            transform=ax_cosine.transAxes,
            verticalalignment="top",
            fontsize=8,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
        )

    # Track current row index
    row_idx = 1

    # --- Row 1 (if exists): R² comparison per task ---
    if has_r2:
        method_linestyles = {"probe": "-", "mean_diff": "--"}
        for idx, task in enumerate(tasks_with_r2):
            ax = fig.add_subplot(gs_main[row_idx, idx])
            task_data = analysis[task]

            text_lines = []
            for method, method_data in task_data.by_method.items():
                r2 = method_data.r2_comparison
                if r2 is None:
                    continue

                linestyle = method_linestyles.get(method, "-")
                layers = r2.layers
                transfer_vals = [r2.transfer_r2_per_layer[l] for l in layers]
                recomputed_vals = [r2.recomputed_r2_per_layer[l] for l in layers]

                ax.plot(layers, transfer_vals, label=f"Transfer ({method})",
                        color="tab:blue", linewidth=2, linestyle=linestyle)
                ax.plot(layers, recomputed_vals, label=f"Recomputed ({method})",
                        color="tab:red", linewidth=2, linestyle=linestyle)

                p_val = r2.paired_test["p_value_two_sided"]
                text_lines.append(f"{method}: Δ={r2.diff_summary.mean:+.3f} (p={p_val:.2g})")

            ax.axhline(0.0, color="gray", alpha=0.3)
            ax.set_xlabel("Layer")
            ax.set_ylabel("R²")
            ax.set_title(f"R² Comparison: {task}", fontsize=10)
            ax.grid(True, alpha=GRID_ALPHA)
            ax.legend(loc="best", fontsize=7)

            if text_lines:
                ax.text(0.02, 0.98, "\n".join(text_lines), transform=ax.transAxes,
                        verticalalignment="top", fontsize=8,
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55))
        row_idx += 1

    # --- Row 2 (if exists): Output connection 2×2 grids per task ---
    if has_output:
        methods = ["probe", "mean_diff"]
        targets = [("P(Ans)", "confidences", "transfer_output_p_answer", "recomputed_output_p_answer"),
                   ("LM", "logit_margins", "transfer_output_logit_margin", "recomputed_output_logit_margin")]

        for task_idx, task in enumerate(tasks_with_output):
            task_data = analysis[task]
            # Create 2×2 nested grid for this task
            gs_inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[row_idx, task_idx],
                                                hspace=0.3, wspace=0.25)

            for m_idx, method in enumerate(methods):
                if method not in task_data.by_method:
                    continue
                method_data = task_data.by_method[method]

                for t_idx, (target_name, _, transfer_attr, recomp_attr) in enumerate(targets):
                    ax = fig.add_subplot(gs_inner[m_idx, t_idx])

                    t_data = getattr(method_data, transfer_attr, {})
                    r_data = getattr(method_data, recomp_attr, {})

                    if t_data:
                        layers = sorted(t_data.keys())
                        vals = [t_data[l] for l in layers]
                        ax.plot(layers, vals, label="d_mc", color="tab:blue", linewidth=1.5)

                    if r_data:
                        layers = sorted(r_data.keys())
                        vals = [r_data[l] for l in layers]
                        ax.plot(layers, vals, label="metamc", color="tab:red", linewidth=1.5)

                    ax.axhline(0.0, color="gray", alpha=0.3)
                    ax.set_title(f"{method}/{target_name}", fontsize=8)
                    ax.grid(True, alpha=GRID_ALPHA)
                    if m_idx == 0 and t_idx == 0:
                        ax.legend(loc="best", fontsize=6)
                    if m_idx == 1:
                        ax.set_xlabel("Layer", fontsize=7)
                    if t_idx == 0:
                        ax.set_ylabel("Pearson r", fontsize=7)
                    ax.tick_params(labelsize=6)

        row_idx += 1

    # --- Row 3 (if exists): Additivity 2×2 grids per task ---
    if has_additivity:
        methods = ["probe", "mean_diff"]
        targets = [("P(Ans)", "additivity_p_answer"), ("LM", "additivity_logit_margin")]

        for task_idx, task in enumerate(tasks_with_additivity):
            task_data = analysis[task]
            # Create 2×2 nested grid for this task
            gs_inner = GridSpecFromSubplotSpec(2, 2, subplot_spec=gs_main[row_idx, task_idx],
                                                hspace=0.3, wspace=0.25)

            for m_idx, method in enumerate(methods):
                if method not in task_data.by_method:
                    continue
                method_data = task_data.by_method[method]

                for t_idx, (target_name, attr_name) in enumerate(targets):
                    ax = fig.add_subplot(gs_inner[m_idx, t_idx])
                    add_result = getattr(method_data, attr_name, None)

                    if add_result is not None:
                        layers = add_result.layers
                        r2_t = [add_result.per_layer[l].r2_transfer for l in layers]
                        r2_r = [add_result.per_layer[l].r2_recomputed for l in layers]
                        r2_c = [add_result.per_layer[l].r2_combined for l in layers]

                        # Plot breakdown
                        ax.plot(layers, r2_t, label="R²(d_mc)", color="tab:blue", linewidth=1.5, linestyle="--")
                        ax.plot(layers, r2_r, label="R²(metamc)", color="tab:red", linewidth=1.5, linestyle="--")
                        ax.plot(layers, r2_c, label="R²(combined)", color="tab:purple", linewidth=2, linestyle="-")

                        # Fill additive gain
                        r2_max = [max(r2_t[i], r2_r[i]) for i in range(len(layers))]
                        ax.fill_between(layers, r2_max, r2_c, alpha=0.2, color="tab:purple")

                        # Text annotation
                        ax.text(0.98, 0.02, f"ratio={add_result.additivity_ratio:.2f}",
                                transform=ax.transAxes, ha="right", va="bottom", fontsize=6,
                                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

                    ax.axhline(0.0, color="gray", alpha=0.3)
                    ax.set_title(f"{method}/{target_name}", fontsize=8)
                    ax.grid(True, alpha=GRID_ALPHA)
                    if m_idx == 0 and t_idx == 0:
                        ax.legend(loc="upper left", fontsize=5)
                    if m_idx == 1:
                        ax.set_xlabel("Layer", fontsize=7)
                    if t_idx == 0:
                        ax.set_ylabel("R²", fontsize=7)
                    ax.tick_params(labelsize=6)

    fig.suptitle("Test 1: Transfer vs Re-computation", fontsize=12, fontweight="bold")
    save_figure(fig, output_path)


def plot_self_vs_other(analysis: SelfOtherResult, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    layers = analysis.layers
    self_r2 = [analysis.self_r2_per_layer[l] for l in layers]
    other_r2 = [analysis.other_r2_per_layer[l] for l in layers]
    diff_r2 = [analysis.diff_r2_per_layer[l] for l in layers]

    ax.plot(layers, self_r2, label=f"Self / introspection (mean={analysis.self_r2_summary.mean:.3f})", linewidth=2)
    ax.plot(layers, other_r2, label=f"Other / surface features (mean={analysis.other_r2_summary.mean:.3f})", linewidth=2)
    ax.plot(layers, diff_r2, label=f"Difference: self - other (mean={analysis.diff_r2_summary.mean:.3f})", linewidth=1.5)
    ax.axhline(0.0, color="gray", alpha=0.4)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Transfer R²")
    ax.set_title("Self vs Other: Layerwise Transfer R²", fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend()

    ax = axes[1]
    categories = ["Self mean R²", "Other mean R²", "Mean diff"]
    vals = [analysis.self_r2_summary.mean, analysis.other_r2_summary.mean, analysis.diff_r2_summary.mean]
    cis = [analysis.self_r2_summary.ci_mean, analysis.other_r2_summary.ci_mean, analysis.diff_r2_summary.ci_mean]
    x = np.arange(len(categories))
    ax.bar(x, vals, width=0.6)
    yerr_low = [0 if ci is None else v - ci[0] for v, ci in zip(vals, cis)]
    yerr_high = [0 if ci is None else ci[1] - v for v, ci in zip(vals, cis)]
    ax.errorbar(x, vals, yerr=[yerr_low, yerr_high], fmt="none", capsize=5)
    ax.axhline(0.0, color="gray", alpha=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel("Estimate")
    ax.set_title("Bootstrap mean estimates with 95% CIs", fontsize=11)
    ax.grid(True, alpha=GRID_ALPHA, axis="y")

    test = analysis.paired_test
    text = (
        f"Paired sign-flip p = {test['p_value_two_sided']:.4g}\n"
        f"Mean diff = {test['observed_mean_diff']:.4f}\n"
        f"Cohen's dz = {json_ready(test['cohens_dz'])}\n"
        f"Interpretation: {analysis.interpretation}"
    )
    ax.text(
        0.5,
        0.98,
        text,
        transform=ax.transAxes,
        horizontalalignment="center",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.55),
    )

    save_figure(fig, output_path)


# =============================================================================
# REPORTING
# =============================================================================


def print_transfer_report(results: Dict[str, TransferTaskResult], pooled: Optional[Dict[str, Any]]) -> None:
    if not results:
        print("  No usable direction-alignment comparisons found for Test 1.")
        return

    for task, data in results.items():
        print(f"\n  {task}:")
        print(f"    Layers analyzed: {len(data.layers)}")

        # Behavioral baselines (MC uncertainty → meta output)
        p_ans_r = data.behavioral_p_answer.get("pearson_r")
        logit_margin_r = data.behavioral_logit_margin.get("pearson_r")
        if p_ans_r is not None or logit_margin_r is not None:
            print(f"\n    Behavioral baseline (MC uncertainty → meta output):")
            if p_ans_r is not None:
                print(f"      → P(Answer): r={p_ans_r:.3f}")
            if logit_margin_r is not None:
                print(f"      → logit_margin: r={logit_margin_r:.3f}")

        for method, method_data in data.by_method.items():
            print(f"\n    [{method}]")

            # Cosine similarity (direction alignment)
            print(f"      Direction alignment (cosine similarity):")
            print(
                f"        Mean |cos|: {method_data.summary_abs.mean:.3f} "
                f"(95% bootstrap CI {format_ci(method_data.summary_abs.ci_mean)})"
            )
            print(f"        Signed mean cos: {method_data.summary_signed.mean:.3f}")

            # R² comparison (transfer vs recomputation performance)
            if method_data.r2_comparison is not None:
                r2 = method_data.r2_comparison
                print(f"      Predictive performance (R²):")
                print(
                    f"        Transfer R² (d_mc → meta): {r2.transfer_summary.mean:.3f} "
                    f"(95% CI {format_ci(r2.transfer_summary.ci_mean)})"
                )
                print(
                    f"        Recomputed R² (metamcuncert): {r2.recomputed_summary.mean:.3f} "
                    f"(95% CI {format_ci(r2.recomputed_summary.ci_mean)})"
                )
                print(
                    f"        Difference (recomputed - transfer): {r2.diff_summary.mean:.3f} "
                    f"(95% CI {format_ci(r2.diff_summary.ci_mean)})"
                )
                print(
                    f"        Paired test p-value: {r2.paired_test['p_value_two_sided']:.4g} "
                    f"[{r2.paired_test['method']}]"
                )
                if r2.paired_test.get('cohens_dz') is not None:
                    print(f"        Cohen's dz: {r2.paired_test['cohens_dz']:.3f}")
            else:
                print(f"      Predictive performance: No R² data available")

            # Output connection (direction → meta output)
            has_any_output = (
                method_data.transfer_output_p_answer or method_data.transfer_output_logit_margin or
                method_data.recomputed_output_p_answer or method_data.recomputed_output_logit_margin
            )
            if has_any_output:
                print(f"      Output connection (direction → meta output):")
                # P(Answer) target
                t_p = method_data.transfer_output_p_answer
                r_p = method_data.recomputed_output_p_answer
                if t_p or r_p:
                    print(f"        → P(Answer):")
                    if t_p:
                        best_l = max(t_p.keys(), key=lambda l: abs(t_p[l]))
                        print(f"            Transfer (d_mc): best r={t_p[best_l]:.3f} at L{best_l}, mean |r|={np.mean([abs(v) for v in t_p.values()]):.3f}")
                    if r_p:
                        best_l = max(r_p.keys(), key=lambda l: abs(r_p[l]))
                        print(f"            Recomputed (metamcuncert): best r={r_p[best_l]:.3f} at L{best_l}, mean |r|={np.mean([abs(v) for v in r_p.values()]):.3f}")
                # logit_margin target
                t_lm = method_data.transfer_output_logit_margin
                r_lm = method_data.recomputed_output_logit_margin
                if t_lm or r_lm:
                    print(f"        → logit_margin:")
                    if t_lm:
                        best_l = max(t_lm.keys(), key=lambda l: abs(t_lm[l]))
                        print(f"            Transfer (d_mc): best r={t_lm[best_l]:.3f} at L{best_l}, mean |r|={np.mean([abs(v) for v in t_lm.values()]):.3f}")
                    if r_lm:
                        best_l = max(r_lm.keys(), key=lambda l: abs(r_lm[l]))
                        print(f"            Recomputed (metamcuncert): best r={r_lm[best_l]:.3f} at L{best_l}, mean |r|={np.mean([abs(v) for v in r_lm.values()]):.3f}")

            # Additivity analysis: do d_mc and metamcuncert contribute independently?
            add_p = method_data.additivity_p_answer
            add_lm = method_data.additivity_logit_margin
            if add_p is not None or add_lm is not None:
                print(f"      Additivity (do directions contribute independently?):")
                for target_name, add_result in [("P(Answer)", add_p), ("logit_margin", add_lm)]:
                    if add_result is None:
                        continue
                    best = add_result.per_layer[add_result.best_layer]
                    print(f"        → {target_name}:")
                    print(f"            R²_transfer={add_result.mean_r2_transfer:.3f}, "
                          f"R²_recomputed={add_result.mean_r2_recomputed:.3f}, "
                          f"R²_combined={add_result.mean_r2_combined:.3f}")
                    print(f"            Unique variance: d_mc adds {add_result.mean_unique_var_transfer:.3f}, "
                          f"metamcuncert adds {add_result.mean_unique_var_recomputed:.3f}")
                    # Handle NaN betas (from collinear projections)
                    if math.isnan(best.beta_transfer) or math.isnan(best.beta_recomputed):
                        print(f"            Best layer L{add_result.best_layer}: collinear projections (betas not estimable)")
                    else:
                        print(f"            Best layer L{add_result.best_layer}: "
                              f"β_transfer={best.beta_transfer:.2f} (p={best.p_transfer:.2g}), "
                              f"β_recomputed={best.beta_recomputed:.2f} (p={best.p_recomputed:.2g})")
                    print(f"            Additivity ratio: {add_result.additivity_ratio:.2f} → {add_result.interpretation}")

            print(f"      Interpretation: {method_data.interpretation}")

        print(f"\n    Methods comparison: {data.pooled_interpretation}")
        for note in data.notes:
            print(f"    Note: {note}")

    if pooled is not None and "by_method" in pooled:
        print("\n  Pooled across task-layer points:")
        for method, method_pooled in pooled["by_method"].items():
            pooled_abs: SummaryStats = method_pooled["summary_abs"]
            print(f"    [{method}] Mean |cos|: {pooled_abs.mean:.3f} (95% CI {format_ci(pooled_abs.ci_mean)})")
            print(f"      Interpretation: {method_pooled['interpretation']}")
        print(f"    Note: {pooled['note']}")



def print_self_other_report(result: SelfOtherResult) -> None:
    print(f"\n  Common layers analyzed: {len(result.layers)}")
    print(
        f"  Self mean transfer R²: {result.self_r2_summary.mean:.4f} "
        f"(95% bootstrap CI {format_ci(result.self_r2_summary.ci_mean, 4)})"
    )
    print(
        f"  Other mean transfer R²: {result.other_r2_summary.mean:.4f} "
        f"(95% bootstrap CI {format_ci(result.other_r2_summary.ci_mean, 4)})"
    )
    print(
        f"  Mean paired difference (self - other): {result.diff_r2_summary.mean:.4f} "
        f"(95% bootstrap CI {format_ci(result.diff_r2_summary.ci_mean, 4)})"
    )
    print(
        f"  Paired sign-flip p-value: {result.paired_test['p_value_two_sided']:.4g} "
        f"[{result.paired_test['method']}]"
    )
    print(f"  Cohen's dz: {json_ready(result.paired_test['cohens_dz'])}")
    print(f"  Interpretation: {result.interpretation}")

    self_pr = result.behavioral["self"].get("pearson_r")
    other_pr = result.behavioral["other"].get("pearson_r")
    ratio = result.behavioral.get("pearson_ratio_self_over_other")
    print("\n  Behavioral correlations (descriptive only):")
    print(f"    Self Pearson r: {self_pr}")
    print(f"    Other Pearson r: {other_pr}")
    print(f"    Self/other Pearson ratio: {ratio}")
    for note in result.notes:
        print(f"    Note: {note}")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET

    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print(f"Model dir: {model_dir}")
    print()

    summary: Dict[str, Any] = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            metric=METRIC,
            direction_methods=DIRECTION_METHODS,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "analysis_notes": [
            "Bootstrap CIs and permutation tests are based on layer-level quantities and should be treated as heuristic because layers are not independent.",
            "Test 1 is descriptive without explicit control directions or another principled null model.",
        ],
    }

    # -------------------------------------------------------------------------
    # Test 1
    # -------------------------------------------------------------------------
    print("=" * 70)
    print("TEST 1: TRANSFER vs RE-COMPUTATION")
    print("=" * 70)

    try:
        test1_results = analyze_transfer_vs_recomputation(DATASET, METRIC, model_dir, rng)
        test1_pooled = aggregate_transfer_over_tasks(test1_results, rng)
        print_transfer_report(test1_results, test1_pooled)

        if test1_results:
            plot_path = get_output_path(f"{base_name}_hypothesis_tests_transfer_vs_recomputation_{METRIC}.png", model_dir=model_dir)
            plot_transfer_vs_recomputation(test1_results, test1_pooled, plot_path)
            print(f"\n  Plot: {plot_path}")

        summary["test1_transfer_vs_recomputation"] = {
            "per_task": test1_results,
            "pooled": test1_pooled,
        }
    except Exception as exc:
        summary["test1_transfer_vs_recomputation"] = {"error": str(exc)}
        print(f"  Test 1 failed: {exc}")

    # -------------------------------------------------------------------------
    # Test 2
    # -------------------------------------------------------------------------
    print()
    print("=" * 70)
    print("TEST 2: INTROSPECTION vs SURFACE FEATURES")
    print("=" * 70)

    try:
        self_transfer = load_transfer_results(base_name, "confidence", model_dir)
        other_transfer = load_transfer_results(base_name, "other_confidence", model_dir)

        if self_transfer is None or other_transfer is None:
            missing = []
            if self_transfer is None:
                missing.append("confidence")
            if other_transfer is None:
                missing.append("other_confidence")
            raise FileNotFoundError(f"Missing transfer result files for: {', '.join(missing)}")

        test2_result = analyze_self_vs_other(self_transfer, other_transfer, METRIC, rng)
        print_self_other_report(test2_result)

        plot_path = get_output_path(f"{base_name}_hypothesis_tests_self_vs_other_{METRIC}.png", model_dir=model_dir)
        plot_self_vs_other(test2_result, plot_path)
        print(f"\n  Plot: {plot_path}")

        summary["test2_introspection_vs_surface"] = test2_result
    except Exception as exc:
        summary["test2_introspection_vs_surface"] = {"error": str(exc)}
        print(f"  Test 2 failed: {exc}")

    summary_path = get_output_path(f"{base_name}_hypothesis_tests_summary_{METRIC}.json", model_dir=model_dir)
    with open(summary_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
