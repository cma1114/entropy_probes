"""
Investigates the puzzle: Why does ablating uncertainty INCREASE confidence projection?

Hypotheses tested:
1. Negative correlation: Uncertainty and confidence projections are negatively correlated
   across samples (if high uncertainty -> low confidence pattern was learned)
2. Inhibition model: High-uncertainty samples show bigger confidence increases after ablation
3. Direction overlap: Confidence direction has significant component along -uncertainty direction

Uses cached activations and directions from prior runs - no model loading needed.

Inputs:
    outputs/{base}_mc_activations.npz                   MC task activations
    outputs/{base}_meta_{task}_activations.npz          Meta-task activations
    outputs/{base}_mc_{metric}_directions.npz           Uncertainty directions
    outputs/{base}_meta_{task}_metaconfdir_directions.npz  Confidence directions
    outputs/{base}_cross_direction_{metric}_results.json   Cross-direction ablation results

Outputs:
    outputs/{base}_uncertainty_confidence_analysis.json    Full analysis results
    outputs/{base}_uncertainty_confidence_analysis.png     Visualization

Run after: identify_mc_correlate.py, test_meta_transfer.py, run_cross_direction_causality.py
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
from scipy import stats

from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA, DPI, DIRECTION_COLORS
from core.config_utils import get_config_dict

# =============================================================================
# CONFIGURATION
# =============================================================================

INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC_difficulty_filtered"
META_TASK = "confidence"
METRIC = "logit_gap"  # Uncertainty metric

# Key layers to focus analysis (from prior results: L14-15 are causal peak)
FOCUS_LAYERS = [13, 14, 15, 16, 17]

# Method to use for directions
METHOD = "mean_diff"  # "probe" or "mean_diff"

# Bootstrap settings
BOOTSTRAP_N = 2000
SEED = 42

# Output
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_activations(path: Path) -> Dict[int, np.ndarray]:
    """Load activations from npz file."""
    if not path.exists():
        print(f"  Warning: {path} not found")
        return {}

    data = np.load(path)
    activations = {}

    for key in data.files:
        if key.startswith("layer_"):
            try:
                layer = int(key.replace("layer_", ""))
                activations[layer] = data[key].astype(np.float32)
            except ValueError:
                continue

    return activations


def load_directions(path: Path, method: str = "mean_diff") -> Dict[int, np.ndarray]:
    """Load directions from npz file with sign alignment."""
    if not path.exists():
        print(f"  Warning: {path} not found")
        return {}

    data = np.load(path)
    directions = {}
    prefix = f"{method}_layer_"

    # Collect layer -> key mapping, sorted by layer
    layer_keys = []
    for key in data.files:
        if key.startswith(prefix):
            try:
                layer = int(key.replace(prefix, ""))
                layer_keys.append((layer, key))
            except ValueError:
                continue

    layer_keys.sort(key=lambda x: x[0])

    # Chain alignment for sign consistency
    reference_direction = None
    for layer, key in layer_keys:
        direction = data[key].astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        if reference_direction is None:
            reference_direction = direction.copy()
        else:
            if reference_direction @ direction < 0:
                direction = -direction
            reference_direction = direction.copy()

        directions[layer] = direction

    return directions


def load_cross_direction_results(base_name: str, metric: str) -> Optional[Dict]:
    """Load cross-direction ablation results."""
    path = OUTPUT_DIR / f"{base_name}_cross_direction_{metric}_results.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return None

    with open(path) as f:
        return json.load(f)


def load_dataset(base_name: str) -> List[Dict]:
    """Load question dataset with metric values."""
    path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        return []

    with open(path) as f:
        data = json.load(f)

    return data.get("data", data.get("questions", data.get("items", [])))


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def compute_projection_correlation(
    acts: np.ndarray,
    unc_dir: np.ndarray,
    conf_dir: np.ndarray,
    n_bootstrap: int = 2000,
) -> Dict:
    """
    Test #1: Compute sample-level correlation between uncertainty and confidence projections.

    If negative, this supports the hypothesis that:
    - The confidence direction learned "absence of uncertainty" = high confidence
    - Ablating uncertainty removes the negative signal, shifting toward high confidence
    """
    # Project each sample onto both directions
    unc_proj = acts @ unc_dir
    conf_proj = acts @ conf_dir

    # Pearson correlation
    r, p = stats.pearsonr(unc_proj, conf_proj)

    # Bootstrap CI for correlation
    rng = np.random.RandomState(SEED)
    n = len(unc_proj)
    boot_rs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_r, _ = stats.pearsonr(unc_proj[idx], conf_proj[idx])
        boot_rs.append(boot_r)

    boot_rs = np.array(boot_rs)
    ci_lo = np.percentile(boot_rs, 2.5)
    ci_hi = np.percentile(boot_rs, 97.5)

    return {
        "correlation": float(r),
        "p_value": float(p),
        "ci_low": float(ci_lo),
        "ci_high": float(ci_hi),
        "n_samples": n,
        "unc_proj_mean": float(np.mean(unc_proj)),
        "unc_proj_std": float(np.std(unc_proj)),
        "conf_proj_mean": float(np.mean(conf_proj)),
        "conf_proj_std": float(np.std(conf_proj)),
        "interpretation": (
            "NEGATIVE: Supports inhibition hypothesis - uncertainty suppresses confidence"
            if r < -0.1 else
            "POSITIVE: Against inhibition hypothesis - uncertainty tracks with confidence"
            if r > 0.1 else
            "NEAR ZERO: Directions are approximately orthogonal"
        ),
    }


def analyze_trial_level_ablation(
    cross_results: Dict,
    ablate_layer: int,
    measure_layer: int,
) -> Optional[Dict]:
    """
    Test #3: Check if high-uncertainty samples show bigger confidence increases after ablation.

    If the inhibition model is correct:
    - Samples with high baseline uncertainty projection should show larger delta_confidence
    - Because they had more "inhibition" to remove

    NOTE: This analysis requires sample-level data which may not be in the saved results.
    If not available, we return guidance on how to collect it.
    """
    # The cross-direction results contain aggregate statistics, not sample-level
    # Check if we have per-sample data
    key = f"uncertainty_L{ablate_layer}_to_confidence_L{measure_layer}"

    if key not in cross_results.get("results", {}):
        return {
            "error": f"No result for {key}",
            "guidance": "Run run_cross_direction_causality.py with SAVE_SAMPLE_LEVEL=True"
        }

    result = cross_results["results"][key]

    # Check for sample-level data
    if "per_sample" not in result:
        return {
            "error": "Sample-level data not saved in cross-direction results",
            "aggregate_delta": result.get("delta_mean"),
            "aggregate_ci": [result.get("delta_ci_low"), result.get("delta_ci_high")],
            "guidance": (
                "To test the inhibition hypothesis at sample level:\n"
                "1. Modify run_cross_direction_causality.py to save per-sample projections\n"
                "2. Correlate baseline_uncertainty_proj with delta_confidence_proj\n"
                "3. Positive correlation = inhibition model supported"
            )
        }

    # If we have sample-level data, do the analysis
    per_sample = result["per_sample"]
    baseline_unc = np.array(per_sample["baseline_uncertainty_proj"])
    delta_conf = np.array(per_sample["delta_confidence_proj"])

    r, p = stats.pearsonr(baseline_unc, delta_conf)

    return {
        "correlation_unc_delta_conf": float(r),
        "p_value": float(p),
        "n_samples": len(baseline_unc),
        "interpretation": (
            "POSITIVE: High-uncertainty samples gain more confidence - supports inhibition"
            if r > 0.1 else
            "NEGATIVE: High-uncertainty samples gain less confidence - against inhibition"
            if r < -0.1 else
            "NEAR ZERO: Effect is uniform across samples"
        ),
    }


def analyze_per_sample_correlation(
    per_sample_path: Path,
    unc_dirs: Dict[int, np.ndarray],
    conf_dirs: Dict[int, np.ndarray],
    n_bootstrap: int = 2000,
) -> Dict:
    """
    Analyze per-sample correlation between baseline uncertainty projection and confidence delta.

    This tests the arithmetic explanation:
    - If samples with more negative uncertainty projection show larger confidence increases,
      it confirms that ablation "adds back" the negative projection.

    Args:
        per_sample_path: Path to npz file from run_cross_direction_causality.py
        unc_dirs: Dict mapping layer -> uncertainty direction at that layer
        conf_dirs: Dict mapping layer -> confidence direction at that layer

    Returns:
        Dict with correlation results and interpretation
    """
    import re

    if not per_sample_path.exists():
        return {
            "error": f"Per-sample data not found at {per_sample_path}",
            "guidance": "Run run_cross_direction_causality.py with SAVE_PER_SAMPLE_DATA=True"
        }

    data = np.load(per_sample_path)

    results = {}

    # Find all uncertainty->confidence pairs
    for key in data.files:
        if "_baseline_ablate_proj" in key and "uncertainty" in key and "confidence" in key:
            # Parse the key to get layer info
            # Format: uncertainty_L{X}_to_confidence_L{Y}_baseline_ablate_proj
            prefix = key.replace("_baseline_ablate_proj", "")

            # Parse layer numbers from prefix
            match = re.match(r"(\w+)_L(\d+)_to_(\w+)_L(\d+)", prefix)
            if not match:
                continue

            ablate_type, ablate_layer_str, measure_type, measure_layer_str = match.groups()
            ablate_layer = int(ablate_layer_str)
            measure_layer = int(measure_layer_str)

            # Get all arrays for this pair
            baseline_ablate_proj = data[f"{prefix}_baseline_ablate_proj"]
            delta_measure_proj = data[f"{prefix}_delta_measure_proj"]

            if len(baseline_ablate_proj) != len(delta_measure_proj):
                continue

            # Compute correlation
            r, p = stats.pearsonr(baseline_ablate_proj, delta_measure_proj)

            # Bootstrap CI
            rng = np.random.RandomState(SEED)
            n = len(baseline_ablate_proj)
            boot_rs = []
            for _ in range(n_bootstrap):
                idx = rng.choice(n, size=n, replace=True)
                boot_r, _ = stats.pearsonr(baseline_ablate_proj[idx], delta_measure_proj[idx])
                boot_rs.append(boot_r)
            boot_rs = np.array(boot_rs)

            # Get directions at the CORRECT layers for this pair
            unc_dir = unc_dirs.get(ablate_layer)
            conf_dir = conf_dirs.get(measure_layer)

            # Compute cosine and linear prediction if directions available
            if unc_dir is not None and conf_dir is not None:
                cos_uc = float(unc_dir @ conf_dir)
                # Linear theory: delta_conf = -(baseline_unc_proj) * cos(u, c)
                # Check if actual slope matches theoretical slope
                slope, _, r_sq, _, _ = stats.linregress(baseline_ablate_proj, delta_measure_proj)
                theoretical_slope = -cos_uc
                slope_ratio = slope / theoretical_slope if abs(theoretical_slope) > 1e-6 else float('nan')
            else:
                cos_uc = float('nan')
                slope = float('nan')
                theoretical_slope = float('nan')
                slope_ratio = float('nan')
                r_sq = float('nan')

            results[prefix] = {
                "correlation": float(r),
                "p_value": float(p),
                "ci_low": float(np.percentile(boot_rs, 2.5)),
                "ci_high": float(np.percentile(boot_rs, 97.5)),
                "n_samples": n,
                "baseline_ablate_mean": float(np.mean(baseline_ablate_proj)),
                "delta_measure_mean": float(np.mean(delta_measure_proj)),
                "ablate_layer": ablate_layer,
                "measure_layer": measure_layer,
                "cos_unc_conf": cos_uc,
                "actual_slope": float(slope),
                "theoretical_slope": float(theoretical_slope),
                "slope_ratio": float(slope_ratio),  # Should be ~1.0 if linear model holds
                "r_squared": float(r_sq),
                "interpretation": (
                    "CONFIRMS ARITHMETIC: More negative baseline -> more positive delta"
                    if r < -0.3 else
                    "PARTIAL: Weak negative correlation"
                    if r < 0 else
                    "UNEXPECTED: Positive correlation"
                ),
            }

    if not results:
        return {"error": "No valid uncertainty->confidence pairs found in npz file"}

    return results


def compute_direction_decomposition(
    unc_dir: np.ndarray,
    conf_dir: np.ndarray,
    n_null: int = 1000,
) -> Dict:
    """
    Test #2: Decompose confidence direction into uncertainty-aligned and orthogonal components.

    If confidence ≈ -uncertainty + other_stuff:
    - cosine(conf, unc) should be negative
    - The component of conf along unc should explain variance

    Includes null distribution: cosine similarity between conf_dir and random directions
    to establish what "orthogonal" actually means in this high-dimensional space.
    """
    # Cosine similarity (already normalized)
    cosine = float(unc_dir @ conf_dir)

    # Project confidence onto uncertainty subspace
    # conf = (conf · unc) * unc + orthogonal_component
    proj_onto_unc = cosine * unc_dir
    orthogonal_component = conf_dir - proj_onto_unc

    # Variance explained by uncertainty direction
    # |proj_onto_unc|² / |conf_dir|² = cosine²
    variance_explained = cosine ** 2

    # Norm of orthogonal component (should be sqrt(1 - cosine²) for unit vectors)
    orthogonal_norm = np.linalg.norm(orthogonal_component)

    # Null distribution: cosine between conf_dir and random unit vectors
    # For d-dimensional space, expected |cosine| ~ 1/sqrt(d), but we compute empirically
    rng = np.random.RandomState(SEED)
    d = len(conf_dir)
    null_cosines = []
    for _ in range(n_null):
        random_dir = rng.randn(d).astype(np.float32)
        random_dir /= np.linalg.norm(random_dir)
        null_cosines.append(float(conf_dir @ random_dir))

    null_cosines = np.array(null_cosines)
    null_mean = float(np.mean(null_cosines))
    null_std = float(np.std(null_cosines))

    # Z-score: how many SDs from random is the observed cosine?
    z_score = (cosine - null_mean) / null_std if null_std > 1e-10 else 0.0

    # Significance thresholds based on z-score (|z| > 2 is ~p < 0.05)
    if z_score < -2.0:
        interpretation = (
            f"SIGNIFICANTLY ANTI-ALIGNED: cosine={cosine:.3f}, z={z_score:.1f}\n"
            f"  {variance_explained*100:.1f}% of confidence direction is along -uncertainty\n"
            f"  (SUPPORTS inhibition hypothesis)"
        )
    elif z_score > 2.0:
        interpretation = (
            f"SIGNIFICANTLY ALIGNED: cosine={cosine:.3f}, z={z_score:.1f}\n"
            f"  {variance_explained*100:.1f}% of confidence direction is along +uncertainty"
        )
    else:
        interpretation = (
            f"NOT SIGNIFICANTLY DIFFERENT FROM RANDOM: cosine={cosine:.3f}, z={z_score:.1f}\n"
            f"  Directions capture different information"
        )

    return {
        "cosine_similarity": cosine,
        "variance_explained_by_uncertainty": float(variance_explained),
        "orthogonal_component_norm": float(orthogonal_norm),
        "null_distribution": {
            "mean": null_mean,
            "std": null_std,
            "expected_for_d": f"~1/sqrt({d}) = {1/np.sqrt(d):.4f}",
        },
        "z_score": float(z_score),
        "interpretation": interpretation,
    }


def analyze_steering_sign_consistency(
    metric: str,
) -> Dict:
    """
    Explain the expected relationship between steering and confidence.

    For logit_gap metric:
    - Higher logit_gap = more confident (larger margin)
    - +direction should point toward higher logit_gap
    - So +steering along logit_gap direction should INCREASE confidence

    For entropy metric:
    - Higher entropy = less confident
    - +direction points toward higher entropy
    - So +steering along entropy direction should DECREASE confidence
    """
    if metric in ["logit_gap", "margin", "top_prob", "top_logit"]:
        return {
            "metric": metric,
            "metric_meaning": "Higher = more confident",
            "direction_meaning": "+direction = toward higher metric = more confident",
            "steering_effect": "+steering should INCREASE stated confidence",
            "ablation_effect": "Removing direction should make model LESS confident (if causal)",
        }
    elif metric == "entropy":
        return {
            "metric": metric,
            "metric_meaning": "Higher = less confident (more uncertain)",
            "direction_meaning": "+direction = toward higher entropy = less confident",
            "steering_effect": "+steering should DECREASE stated confidence",
            "ablation_effect": "Removing direction should make model MORE confident (if causal)",
        }
    else:
        return {"error": f"Unknown metric: {metric}"}


def synthesize_findings(
    projection_results: Dict[int, Dict],
    decomposition_results: Dict[int, Dict],
    cross_ablation_results: Dict,
    steering_info: Dict,
) -> Dict:
    """
    Synthesize all findings into a coherent interpretation.
    """
    findings = {
        "hypothesis_tested": (
            "Inhibition model: Uncertainty acts as an inhibitory signal on confidence. "
            "The confidence direction learned 'absence of uncertainty' = high confidence. "
            "Thus ablating uncertainty removes suppression, increasing confidence projection."
        ),
        "evidence": [],
        "interpretation": "",
    }

    # Check projection correlation evidence (prefer MC domain-matched results)
    for layer, layer_data in projection_results.items():
        # Use MC results if available (domain-matched), else meta-task
        if "mc_task" in layer_data:
            result = layer_data["mc_task"]
            domain = "MC"
        elif "meta_task" in layer_data:
            result = layer_data["meta_task"]
            domain = "meta"
        else:
            continue

        if "correlation" in result:
            r = result["correlation"]
            if r < -0.2:
                findings["evidence"].append(
                    f"L{layer}: SUPPORTS - Sample projections are negatively correlated (r={r:.3f}, {domain})"
                )
            elif r > 0.2:
                findings["evidence"].append(
                    f"L{layer}: AGAINST - Sample projections are positively correlated (r={r:.3f}, {domain})"
                )

    # Check direction decomposition evidence (use z-scores for significance)
    for layer, result in decomposition_results.items():
        cos = result.get("cosine_similarity", 0)
        z = result.get("z_score", 0)

        # Use z-scores: |z| > 2 is approximately p < 0.05
        if z < -2.0:
            findings["evidence"].append(
                f"L{layer}: SUPPORTS - Confidence direction significantly anti-aligned (cos={cos:.3f}, z={z:.1f})"
            )
        elif z > 2.0:
            findings["evidence"].append(
                f"L{layer}: MIXED - Confidence direction significantly aligned (cos={cos:.3f}, z={z:.1f})"
            )
        # Don't add evidence for non-significant results

    # Check cross-ablation effect
    if cross_ablation_results:
        for key, result in cross_ablation_results.get("results", {}).items():
            if "uncertainty" in key and "confidence" in key:
                delta = result.get("delta_mean", 0)
                if delta > 0.05:
                    findings["evidence"].append(
                        f"{key}: SUPPORTS - Ablating uncertainty INCREASES confidence (Δ={delta:.3f})"
                    )
                elif delta < -0.05:
                    findings["evidence"].append(
                        f"{key}: AGAINST - Ablating uncertainty DECREASES confidence (Δ={delta:.3f})"
                    )

    # Check steering sign
    steering_sign = steering_info.get("steering_effect", "")
    if "INCREASE" in steering_sign:
        findings["evidence"].append(
            f"Steering: +{steering_info['metric']} direction -> higher confidence (by convention)"
        )

    # Generate interpretation
    n_supports = sum(1 for e in findings["evidence"] if "SUPPORTS" in e)
    n_against = sum(1 for e in findings["evidence"] if "AGAINST" in e)

    if n_supports > n_against:
        findings["interpretation"] = (
            f"EVIDENCE SUPPORTS INHIBITION MODEL ({n_supports} supporting, {n_against} against):\n"
            "The confidence direction appears to encode 'absence of uncertainty' rather than "
            "a separate confidence construct. Ablating uncertainty removes this inhibitory signal, "
            "causing confidence projections to increase. This is consistent with the model learning "
            "that 'no uncertainty signal' = 'high confidence'."
        )
    elif n_against > n_supports:
        findings["interpretation"] = (
            f"EVIDENCE AGAINST INHIBITION MODEL ({n_supports} supporting, {n_against} against):\n"
            "The relationship between uncertainty and confidence directions suggests they capture "
            "related but distinct information. The ablation effect may be due to shared variance "
            "rather than direct inhibition."
        )
    else:
        findings["interpretation"] = (
            f"MIXED EVIDENCE ({n_supports} supporting, {n_against} against):\n"
            "Results are inconclusive. The relationship between uncertainty and confidence "
            "may vary by layer or involve more complex interactions."
        )

    return findings


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_analysis_results(
    projection_results: Dict[int, Dict],
    decomposition_results: Dict[int, Dict],
    cross_ablation_results: Optional[Dict],
    focus_layers: List[int],
    output_path: Path,
    mc_acts: Optional[Dict[int, np.ndarray]] = None,
    unc_dirs: Optional[Dict[int, np.ndarray]] = None,
    conf_dirs: Optional[Dict[int, np.ndarray]] = None,
):
    """Create multi-panel visualization of the analysis."""
    fig = plt.figure(figsize=(14, 10))

    # Get colors from plotting module (with fallbacks)
    unc_color = DIRECTION_COLORS.get("uncertainty", "tab:orange")
    conf_color = DIRECTION_COLORS.get("confidence", "tab:green")

    # Layout: 2x2 grid
    # [0,0]: Projection correlation by layer
    # [0,1]: Direction decomposition by layer
    # [1,0]: Scatter of uncertainty vs confidence projection (focus layer)
    # [1,1]: Cross-ablation delta summary

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Panel 1: Projection correlation by layer (both domains if available)
    layers = sorted(projection_results.keys())

    # Extract correlations - handle nested structure (meta_task/mc_task)
    correlations_meta = []
    ci_los_meta = []
    ci_his_meta = []
    correlations_mc = []
    ci_los_mc = []
    ci_his_mc = []

    for l in layers:
        layer_data = projection_results[l]
        if "meta_task" in layer_data:
            correlations_meta.append(layer_data["meta_task"].get("correlation", np.nan))
            ci_los_meta.append(layer_data["meta_task"].get("ci_low", np.nan))
            ci_his_meta.append(layer_data["meta_task"].get("ci_high", np.nan))
        else:
            correlations_meta.append(np.nan)
            ci_los_meta.append(np.nan)
            ci_his_meta.append(np.nan)

        if "mc_task" in layer_data:
            correlations_mc.append(layer_data["mc_task"].get("correlation", np.nan))
            ci_los_mc.append(layer_data["mc_task"].get("ci_low", np.nan))
            ci_his_mc.append(layer_data["mc_task"].get("ci_high", np.nan))
        else:
            correlations_mc.append(np.nan)
            ci_los_mc.append(np.nan)
            ci_his_mc.append(np.nan)

    # Plot both if available
    has_mc = any(not np.isnan(x) for x in correlations_mc)
    has_meta = any(not np.isnan(x) for x in correlations_meta)

    if has_mc:
        ax1.plot(layers, correlations_mc, 'o-', color='tab:blue', markersize=4, label='MC (domain-matched)')
        ax1.fill_between(layers, ci_los_mc, ci_his_mc, alpha=CI_ALPHA, color='tab:blue')
    if has_meta:
        ax1.plot(layers, correlations_meta, 's--', color='tab:purple', markersize=3, alpha=0.7, label='Meta-task')
        ax1.fill_between(layers, ci_los_meta, ci_his_meta, alpha=CI_ALPHA * 0.5, color='tab:purple')

    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    for fl in focus_layers:
        if fl in layers:
            ax1.axvline(fl, color='green', linestyle=':', alpha=0.3)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (unc_proj, conf_proj)")
    ax1.set_title("Projection Correlation: Directions Capture Shared Variance\n(r > 0 means aligned, not opposing)")
    ax1.grid(True, alpha=GRID_ALPHA)
    ax1.legend(loc='upper right')

    # Panel 2: Direction decomposition (cosine similarity + z-scores)
    layers_d = sorted(decomposition_results.keys())
    cosines = [decomposition_results[l].get("cosine_similarity", np.nan) for l in layers_d]
    z_scores = [decomposition_results[l].get("z_score", np.nan) for l in layers_d]

    ax2.plot(layers_d, cosines, 'o-', color=unc_color, markersize=4, label='Cosine(conf, unc)')
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    for fl in focus_layers:
        if fl in layers_d:
            ax2.axvline(fl, color='green', linestyle=':', alpha=0.3)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Direction Alignment: cos > 0 = Positively Aligned\n(Both point toward 'more confident')")
    ax2.grid(True, alpha=GRID_ALPHA)

    # Add secondary y-axis for z-scores
    ax2b = ax2.twinx()
    ax2b.plot(layers_d, z_scores, 's--', color='tab:red', markersize=3, alpha=0.6, label='Z-score')
    ax2b.axhline(2, color='tab:red', linestyle=':', alpha=0.3)
    ax2b.axhline(-2, color='tab:red', linestyle=':', alpha=0.3)
    ax2b.set_ylabel("Z-score (vs random)", color='tab:red')
    ax2b.tick_params(axis='y', labelcolor='tab:red')

    # Panel 3: Actual scatter plot for a focus layer
    scatter_plotted = False
    if mc_acts and unc_dirs and conf_dirs and focus_layers:
        # Use first focus layer that has all data
        for fl in focus_layers:
            if fl in mc_acts and fl in unc_dirs and fl in conf_dirs:
                unc_proj = mc_acts[fl] @ unc_dirs[fl]
                conf_proj = mc_acts[fl] @ conf_dirs[fl]
                r, _ = stats.pearsonr(unc_proj, conf_proj)

                ax3.scatter(unc_proj, conf_proj, alpha=0.3, s=10, c='tab:blue')
                ax3.set_xlabel("Uncertainty Projection")
                ax3.set_ylabel("Confidence Projection")
                ax3.set_title(f"Layer {fl}: Strong Positive Correlation (r = {r:.3f})\n(High uncertainty ↔ high confidence projection)")

                # Add regression line
                z = np.polyfit(unc_proj, conf_proj, 1)
                p = np.poly1d(z)
                x_line = np.linspace(unc_proj.min(), unc_proj.max(), 100)
                ax3.plot(x_line, p(x_line), 'r--', alpha=0.7, label=f'r={r:.3f}')
                ax3.legend()
                ax3.grid(True, alpha=GRID_ALPHA)
                scatter_plotted = True
                break

    if not scatter_plotted:
        ax3.text(0.5, 0.5,
                 "Scatter plot requires MC activations.\n\n"
                 "Interpretation:\n"
                 "• Negative r: High uncertainty → low confidence\n"
                 "• Positive r: High uncertainty → high confidence\n"
                 "• Zero r: Independent representations",
                 ha='center', va='center', transform=ax3.transAxes,
                 fontsize=10, wrap=True)
        ax3.set_title("Projection Relationship")
        ax3.axis('off')

    # Panel 4: Cross-ablation - L14 only, raw vs normalized
    # Shows that the "effect" is almost entirely a norm artifact
    ABLATION_LAYER = 14  # Focus on L14 ablation

    if cross_ablation_results and "results" in cross_ablation_results:
        # Extract L14 uncertainty→confidence effects with both raw and normalized
        raw_data = []  # (meas_layer, delta, sig)
        norm_data = []  # (meas_layer, delta, sig)

        for key, result in cross_ablation_results["results"].items():
            if f"uncertainty_L{ABLATION_LAYER}_to_confidence" in key:
                # Parse measurement layer
                parts = key.split("_")
                try:
                    meas_layer = int(parts[4].replace("L", ""))

                    # Raw effect
                    raw_delta = result.get("delta_mean", 0)
                    raw_sig = result.get("significant_fdr", False)
                    raw_data.append((meas_layer, raw_delta, raw_sig))

                    # Normalized effect
                    normalized = result.get("normalized", {})
                    if normalized:
                        norm_delta = normalized.get("delta_mean", 0)
                        norm_sig = normalized.get("significant_fdr", False)
                        norm_data.append((meas_layer, norm_delta, norm_sig))
                except (IndexError, ValueError):
                    continue

        if raw_data:
            # Sort by measurement layer
            raw_data.sort(key=lambda x: x[0])
            norm_data.sort(key=lambda x: x[0])

            raw_layers = [x[0] for x in raw_data]
            raw_deltas = [x[1] for x in raw_data]
            raw_sigs = [x[2] for x in raw_data]

            # Plot raw effect
            ax4.plot(raw_layers, raw_deltas, 'o-', color='tab:red',
                    markersize=5, linewidth=2, label='Raw Δ (artifact)')

            # Mark significant raw points
            for ml, d, sig in zip(raw_layers, raw_deltas, raw_sigs):
                if sig:
                    ax4.scatter([ml], [d], s=80, c='tab:red', marker='*', zorder=5)

            # Plot normalized effect if available
            if norm_data:
                norm_layers = [x[0] for x in norm_data]
                norm_deltas = [x[1] for x in norm_data]
                norm_sigs = [x[2] for x in norm_data]

                ax4.plot(norm_layers, norm_deltas, 's-', color='tab:blue',
                        markersize=4, linewidth=2, label='Normalized Δ (real)')

                # Mark significant normalized points (there shouldn't be any)
                for ml, d, sig in zip(norm_layers, norm_deltas, norm_sigs):
                    if sig:
                        ax4.scatter([ml], [d], s=80, c='tab:blue', marker='*', zorder=5)

            ax4.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax4.set_xlabel("Measurement Layer")
            ax4.set_ylabel("Δ Confidence Projection")
            ax4.set_title(f"Ablate L{ABLATION_LAYER} Uncertainty: Raw vs Normalized Effect\n(Normalized effect is ~10x smaller and non-significant)")
            ax4.legend(loc='upper left', fontsize=9)
            ax4.grid(True, alpha=GRID_ALPHA)
        else:
            ax4.text(0.5, 0.5, f"No L{ABLATION_LAYER} uncertainty→confidence\nablation data found",
                    ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title("Cross-Ablation Effects")
    else:
        ax4.text(0.5, 0.5, "Cross-ablation results not available.\n\nRun run_cross_direction_causality.py first.",
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title("Cross-Ablation Effects")

    plt.tight_layout()
    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("UNCERTAINTY-CONFIDENCE RELATIONSHIP ANALYSIS")
    print("=" * 70)
    print(f"\nInput: {INPUT_BASE_NAME}")
    print(f"Meta-task: {META_TASK}")
    print(f"Metric: {METRIC}")
    print(f"Method: {METHOD}")
    print(f"Focus layers: {FOCUS_LAYERS}")

    # Load directions
    print("\nLoading directions...")
    unc_dir_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_{METRIC}_directions.npz"
    conf_dir_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_meta_{META_TASK}_metaconfdir_directions.npz"

    unc_dirs = load_directions(unc_dir_path, METHOD)
    conf_dirs = load_directions(conf_dir_path, METHOD)

    print(f"  Uncertainty directions: {len(unc_dirs)} layers")
    print(f"  Confidence directions: {len(conf_dirs)} layers")

    if not unc_dirs or not conf_dirs:
        print("\nError: Missing directions. Run required scripts first.")
        return

    # Load activations (both MC and meta-task for domain comparison)
    print("\nLoading activations...")
    meta_acts_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_meta_{META_TASK}_activations.npz"
    mc_acts_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_mc_activations.npz"

    meta_acts = load_activations(meta_acts_path)
    mc_acts = load_activations(mc_acts_path)

    print(f"  Meta-task activations: {len(meta_acts)} layers")
    print(f"  MC-task activations: {len(mc_acts)} layers (for domain-matched comparison)")

    # Load cross-direction results
    print("\nLoading cross-direction ablation results...")
    cross_results = load_cross_direction_results(INPUT_BASE_NAME, METRIC)
    if cross_results:
        print(f"  Loaded {len(cross_results.get('results', {}))} ablation results")

    # Get common layers
    common_layers = sorted(set(unc_dirs.keys()) & set(conf_dirs.keys()))
    if meta_acts:
        common_layers = sorted(set(common_layers) & set(meta_acts.keys()))
    print(f"\nAnalyzing {len(common_layers)} common layers")

    # ==========================================================================
    # TEST #1: Sample-level projection correlation
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST #1: SAMPLE-LEVEL PROJECTION CORRELATION")
    print("=" * 70)
    print("If negative: supports inhibition hypothesis")
    print("(high uncertainty projection → low confidence projection)")
    print("\nNote: Testing on BOTH meta-task and MC activations for domain comparison")
    print("  - MC activations: domain-matched (unc directions trained on MC)")
    print("  - Meta-task activations: cross-domain (where we observe the ablation effect)")

    projection_results = {}
    for layer in common_layers:
        layer_result = {}

        # Compute on meta-task activations (cross-domain)
        if layer in meta_acts:
            result_meta = compute_projection_correlation(
                meta_acts[layer],
                unc_dirs[layer],
                conf_dirs[layer],
                n_bootstrap=BOOTSTRAP_N,
            )
            layer_result["meta_task"] = result_meta

        # Compute on MC activations (domain-matched, more reliable)
        if layer in mc_acts:
            result_mc = compute_projection_correlation(
                mc_acts[layer],
                unc_dirs[layer],
                conf_dirs[layer],
                n_bootstrap=BOOTSTRAP_N,
            )
            layer_result["mc_task"] = result_mc

        if layer_result:
            projection_results[layer] = layer_result

        if layer in FOCUS_LAYERS and layer_result:
            print(f"  Layer {layer}:")
            if "meta_task" in layer_result:
                r = layer_result["meta_task"]["correlation"]
                ci = f"[{layer_result['meta_task']['ci_low']:.3f}, {layer_result['meta_task']['ci_high']:.3f}]"
                print(f"    Meta-task: r = {r:+.3f} {ci}")
            if "mc_task" in layer_result:
                r = layer_result["mc_task"]["correlation"]
                ci = f"[{layer_result['mc_task']['ci_low']:.3f}, {layer_result['mc_task']['ci_high']:.3f}]"
                print(f"    MC-task:   r = {r:+.3f} {ci} (domain-matched)")

    # ==========================================================================
    # TEST #2: Direction decomposition
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST #2: DIRECTION DECOMPOSITION")
    print("=" * 70)
    print("If cosine < 0: confidence direction is anti-aligned with uncertainty")
    print("(conf ≈ -unc + orthogonal_stuff)")

    decomposition_results = {}
    for layer in common_layers:
        result = compute_direction_decomposition(unc_dirs[layer], conf_dirs[layer])
        decomposition_results[layer] = result

        if layer in FOCUS_LAYERS:
            cos = result["cosine_similarity"]
            var = result["variance_explained_by_uncertainty"]
            print(f"  Layer {layer}: cosine = {cos:+.3f}, variance from unc = {var*100:.1f}%")
            print(f"             {result['interpretation']}")

    # ==========================================================================
    # TEST #3: Trial-level ablation analysis
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST #3: TRIAL-LEVEL ABLATION ANALYSIS")
    print("=" * 70)
    print("If high-uncertainty samples show bigger confidence increase: supports inhibition")

    trial_results = {}
    if cross_results:
        for abl_l in FOCUS_LAYERS:
            for meas_l in FOCUS_LAYERS:
                if meas_l <= abl_l:
                    continue
                result = analyze_trial_level_ablation(cross_results, abl_l, meas_l)
                if result:
                    key = f"L{abl_l}_to_L{meas_l}"
                    trial_results[key] = result

                    if "error" not in result:
                        print(f"  {key}: r(baseline_unc, delta_conf) = {result['correlation_unc_delta_conf']:.3f}")
                    else:
                        # Show aggregate result if available
                        delta = result.get("aggregate_delta")
                        if delta is not None:
                            print(f"  {key}: aggregate Δconf = {delta:+.3f} (sample-level data not saved)")
    else:
        print("  Cross-direction results not available")

    # ==========================================================================
    # STEERING SIGN CHECK
    # ==========================================================================
    print("\n" + "=" * 70)
    print("STEERING SIGN CONVENTIONS")
    print("=" * 70)

    steering_info = analyze_steering_sign_consistency(METRIC)
    for key, value in steering_info.items():
        print(f"  {key}: {value}")

    # ==========================================================================
    # SYNTHESIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    # ==========================================================================
    # TEST #4: PER-SAMPLE CORRELATION (if data available)
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST #4: PER-SAMPLE ABLATION CORRELATION")
    print("=" * 70)
    print("Tests arithmetic explanation: more negative baseline_unc -> more positive delta_conf")
    print()

    per_sample_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_cross_direction_{METRIC}_per_sample.npz"
    per_sample_results = {}

    if per_sample_path.exists():
        per_sample_results = analyze_per_sample_correlation(
            per_sample_path,
            unc_dirs,  # Pass full dict so cosine is computed at correct layers
            conf_dirs,
            n_bootstrap=BOOTSTRAP_N,
        )

        if "error" not in per_sample_results:
            for key, result in per_sample_results.items():
                print(f"  {key} (L{result['ablate_layer']}->L{result['measure_layer']}):")
                print(f"    Correlation (baseline_unc, delta_conf): r = {result['correlation']:+.3f}")
                print(f"    CI: [{result['ci_low']:.3f}, {result['ci_high']:.3f}]")
                print(f"    Baseline unc mean: {result['baseline_ablate_mean']:.4f}")
                print(f"    Delta conf mean: {result['delta_measure_mean']:.4f}")
                print(f"    Cosine(unc@L{result['ablate_layer']}, conf@L{result['measure_layer']}): {result['cos_unc_conf']:.3f}")
                print(f"    Actual slope: {result['actual_slope']:.4f}, Theoretical: {result['theoretical_slope']:.4f}")
                print(f"    Slope ratio (actual/theoretical): {result['slope_ratio']:.2f}")
                print(f"    -> {result['interpretation']}")
                print()
        else:
            print(f"  {per_sample_results.get('error', 'Unknown error')}")
            print(f"  {per_sample_results.get('guidance', '')}")
    else:
        print(f"  Per-sample data not found at {per_sample_path}")
        print("  Run run_cross_direction_causality.py with SAVE_PER_SAMPLE_DATA=True to generate")

    # ==========================================================================
    # SYNTHESIS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SYNTHESIS")
    print("=" * 70)

    synthesis = synthesize_findings(
        projection_results,
        decomposition_results,
        cross_results,
        steering_info,
    )

    print(f"\nHYPOTHESIS: {synthesis['hypothesis_tested']}")
    print(f"\nEVIDENCE:")
    for e in synthesis["evidence"]:
        print(f"  • {e}")
    print(f"\nINTERPRETATION:\n{synthesis['interpretation']}")

    # ==========================================================================
    # SAVE RESULTS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    results = {
        "config": get_config_dict(
            input_base=INPUT_BASE_NAME,
            meta_task=META_TASK,
            metric=METRIC,
            method=METHOD,
            focus_layers=FOCUS_LAYERS,
            bootstrap_n=BOOTSTRAP_N,
        ),
        "projection_correlations": {str(k): v for k, v in projection_results.items()},
        "direction_decomposition": {str(k): v for k, v in decomposition_results.items()},
        "trial_level_analysis": trial_results,
        "per_sample_correlation": per_sample_results if per_sample_results else None,
        "steering_conventions": steering_info,
        "synthesis": synthesis,
    }

    json_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_uncertainty_confidence_analysis.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Plot results
    plot_path = OUTPUT_DIR / f"{INPUT_BASE_NAME}_uncertainty_confidence_analysis.png"
    plot_analysis_results(
        projection_results,
        decomposition_results,
        cross_results,
        FOCUS_LAYERS,
        plot_path,
        mc_acts=mc_acts,
        unc_dirs=unc_dirs,
        conf_dirs=conf_dirs,
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
