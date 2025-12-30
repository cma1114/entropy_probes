"""
Activation patching experiments: Test causal role of full activations.

Unlike steering (which adds a 1D direction), patching swaps the complete
activation pattern from one example into another. This tests whether the
full activation (not just a linear projection) causally determines behavior.

For each question pair (source=low metric, target=high metric):
1. Run source question normally → baseline confidence
2. Run source question with target's activations patched at layer L → patched confidence
3. Measure: Does patching shift source's confidence toward target's confidence?

If patching B's activations into A makes A behave like B, then layer L's
full activation pattern is causally involved in determining confidence.

This is more robust than steering to non-linear encodings. If uncertainty
is encoded categorically (e.g., low=[1,0,0], mid=[0,1,0], high=[0,0,1]),
steering along a probe direction won't work, but patching will.

Usage:
    python run_activation_patching.py --metric logit_gap
    python run_activation_patching.py --metric entropy --n-pairs 200
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import random

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from core.steering import (
    BatchPatchingHook,
    generate_orthogonal_directions,
)
from tasks import (
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    response_to_confidence,
)

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME
DATASET_NAME = "SimpleMC"

# Meta-judgment task: "confidence" or "delegate"
META_TASK = "confidence"

# Patching configuration
NUM_PATCH_PAIRS = 100  # Number of source→target pairs to test per layer
PAIRING_METHOD = "extremes"  # "extremes", "random", or "quartile"
BATCH_SIZE = 8  # Forward pass batch size

# Layer selection
PATCHING_LAYERS = None  # None = auto-select based on probe transfer

# Metric to use for selecting pairs
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
METRIC = "logit_gap"

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Option tokens (cached at startup)
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS
_CACHED_TOKEN_IDS = {"meta_options": None, "delegate_options": None}


def initialize_token_cache(tokenizer):
    """Precompute option token IDs once."""
    _CACHED_TOKEN_IDS["meta_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS
    ]
    _CACHED_TOKEN_IDS["delegate_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in DELEGATE_OPTIONS
    ]


def get_output_prefix() -> str:
    """Generate output filename prefix."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_patching{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_patching{task_suffix}")


# =============================================================================
# PAIR CREATION
# =============================================================================


def create_patch_pairs(
    metric_values: np.ndarray,
    n_pairs: int,
    method: str = "extremes",
    seed: int = 42
) -> List[Tuple[int, int]]:
    """
    Create source→target pairs for patching.

    Each pair is (source_idx, target_idx) where we'll patch target's
    activations into source and measure the effect.

    Args:
        metric_values: (n_samples,) metric values for each question
        n_pairs: Number of pairs to create
        method: Pairing strategy:
            - "extremes": Pair low-metric sources with high-metric targets
            - "random": Random pairs from different quartiles
            - "quartile": Systematic quartile-to-quartile pairs

    Returns:
        List of (source_idx, target_idx) tuples
    """
    rng = np.random.RandomState(seed)
    n = len(metric_values)
    sorted_idx = np.argsort(metric_values)

    pairs = []

    if method == "extremes":
        # Pair bottom quartile sources with top quartile targets
        n_quartile = n // 4
        low_indices = sorted_idx[:n_quartile]
        high_indices = sorted_idx[-n_quartile:]

        # Create pairs: low source → high target
        for _ in range(n_pairs):
            source = rng.choice(low_indices)
            target = rng.choice(high_indices)
            pairs.append((source, target))

    elif method == "random":
        # Random pairs from different halves
        n_half = n // 2
        low_half = sorted_idx[:n_half]
        high_half = sorted_idx[n_half:]

        for _ in range(n_pairs):
            source = rng.choice(low_half)
            target = rng.choice(high_half)
            pairs.append((source, target))

    elif method == "quartile":
        # Systematic: Q1→Q4, Q2→Q4, Q1→Q3, etc.
        n_quartile = n // 4
        q1 = sorted_idx[:n_quartile]
        q2 = sorted_idx[n_quartile:2*n_quartile]
        q3 = sorted_idx[2*n_quartile:3*n_quartile]
        q4 = sorted_idx[3*n_quartile:]

        # Primary: Q1→Q4 (low to high)
        n_primary = n_pairs // 2
        for _ in range(n_primary):
            pairs.append((rng.choice(q1), rng.choice(q4)))

        # Secondary: Q1→Q3, Q2→Q4
        n_secondary = (n_pairs - n_primary) // 2
        for _ in range(n_secondary):
            pairs.append((rng.choice(q1), rng.choice(q3)))
        for _ in range(n_pairs - n_primary - n_secondary):
            pairs.append((rng.choice(q2), rng.choice(q4)))

    else:
        raise ValueError(f"Unknown pairing method: {method}")

    return pairs


def create_random_pairs(n_samples: int, n_pairs: int, seed: int = 42) -> List[Tuple[int, int]]:
    """Create random pairs for control comparison."""
    rng = np.random.RandomState(seed)
    pairs = []
    for _ in range(n_pairs):
        source = rng.randint(0, n_samples)
        target = rng.randint(0, n_samples)
        while target == source:
            target = rng.randint(0, n_samples)
        pairs.append((source, target))
    return pairs


# =============================================================================
# PATCHING EXPERIMENT
# =============================================================================


def format_meta_prompt_for_question(
    question: Dict,
    tokenizer,
    use_chat_template: bool,
    trial_idx: int = 0
) -> Tuple[str, List[str], Optional[Dict]]:
    """Format meta prompt based on META_TASK."""
    if META_TASK == "delegate":
        prompt, options, mapping = format_answer_or_delegate_prompt(
            question, tokenizer, use_chat_template, trial_idx
        )
        return prompt, options, mapping
    else:
        prompt, options = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
        return prompt, options, None


def compute_confidence_from_logits(
    logits: torch.Tensor,
    option_token_ids: List[int],
    mapping: Optional[Dict] = None
) -> float:
    """Compute confidence value from logits over option tokens."""
    option_logits = logits[option_token_ids]
    probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    if META_TASK == "delegate":
        options = DELEGATE_OPTIONS
    else:
        options = META_OPTIONS

    response = options[np.argmax(probs)]
    return response_to_confidence(response, probs, mapping, task_type=META_TASK)


def run_patching_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    cached_activations: Dict[int, np.ndarray],
    metric_values: np.ndarray,
    layers: List[int],
    patch_pairs: List[Tuple[int, int]],
    use_chat_template: bool,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Run activation patching experiment.

    For each layer and each (source, target) pair:
    1. Run source normally → baseline confidence
    2. Run source with target's activations patched → patched confidence
    3. Compute shift = (patched - baseline) / (target_baseline - baseline)

    Args:
        model: The transformer model
        tokenizer: Tokenizer
        questions: List of question dicts
        cached_activations: {layer_idx: (n_questions, hidden_dim)} pre-extracted activations
        metric_values: (n_questions,) metric values for each question
        layers: Which layers to test
        patch_pairs: List of (source_idx, target_idx) pairs
        use_chat_template: Whether to use chat template
        batch_size: Forward pass batch size

    Returns:
        Dict with results per layer and per pair
    """
    model.eval()

    # Get option token IDs
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]

    # Get access to model layers
    if hasattr(model, 'get_base_model'):
        model_layers = model.get_base_model().model.layers
    else:
        model_layers = model.model.layers

    results = {
        "layers": layers,
        "n_pairs": len(patch_pairs),
        "pairs": patch_pairs,
        "metric": METRIC,
        "layer_results": {},
    }

    # Pre-compute all baselines (no patching)
    print("Computing baselines for all questions...")
    all_prompts = []
    all_mappings = []
    for i, q in enumerate(questions):
        prompt, _, mapping = format_meta_prompt_for_question(q, tokenizer, use_chat_template, i)
        all_prompts.append(prompt)
        all_mappings.append(mapping)

    # Batch compute baselines
    all_baseline_confidences = []
    for batch_start in tqdm(range(0, len(questions), batch_size), desc="Baselines"):
        batch_end = min(batch_start + batch_size, len(questions))
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_mappings = all_mappings[batch_start:batch_end]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

        for i in range(len(batch_prompts)):
            final_logits = outputs.logits[i, -1, :]
            conf = compute_confidence_from_logits(final_logits, option_token_ids, batch_mappings[i])
            all_baseline_confidences.append(conf)

        del inputs, outputs
        torch.cuda.empty_cache()

    all_baseline_confidences = np.array(all_baseline_confidences)
    print(f"Baseline confidence: mean={all_baseline_confidences.mean():.3f}, std={all_baseline_confidences.std():.3f}")

    # Run patching for each layer
    for layer_idx in tqdm(layers, desc="Layers"):
        layer_activations = cached_activations[layer_idx]  # (n_questions, hidden_dim)

        pair_results = []

        # Process pairs in batches
        for batch_start in tqdm(range(0, len(patch_pairs), batch_size),
                                desc=f"Layer {layer_idx}", leave=False):
            batch_end = min(batch_start + batch_size, len(patch_pairs))
            batch_pairs = patch_pairs[batch_start:batch_end]

            # Prepare batch: source prompts and target activations
            batch_prompts = []
            batch_target_acts = []
            batch_mappings = []
            batch_pair_info = []

            for source_idx, target_idx in batch_pairs:
                batch_prompts.append(all_prompts[source_idx])
                batch_target_acts.append(layer_activations[target_idx])
                batch_mappings.append(all_mappings[source_idx])
                batch_pair_info.append({
                    "source_idx": source_idx,
                    "target_idx": target_idx,
                    "source_metric": float(metric_values[source_idx]),
                    "target_metric": float(metric_values[target_idx]),
                    "source_baseline": float(all_baseline_confidences[source_idx]),
                    "target_baseline": float(all_baseline_confidences[target_idx]),
                })

            batch_target_acts = torch.tensor(np.array(batch_target_acts), dtype=torch.float32)

            # Tokenize batch
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(DEVICE)

            # Register patching hook
            hook = BatchPatchingHook(batch_target_acts, position="last")
            handle = model_layers[layer_idx].register_forward_hook(hook)

            try:
                with torch.no_grad():
                    outputs = model(**inputs, use_cache=False)

                # Compute patched confidences
                for i in range(len(batch_prompts)):
                    final_logits = outputs.logits[i, -1, :]
                    patched_conf = compute_confidence_from_logits(
                        final_logits, option_token_ids, batch_mappings[i]
                    )

                    info = batch_pair_info[i]
                    info["patched_confidence"] = float(patched_conf)

                    # Compute shift metrics
                    source_base = info["source_baseline"]
                    target_base = info["target_baseline"]
                    gap = target_base - source_base

                    info["confidence_shift"] = float(patched_conf - source_base)
                    if abs(gap) > 0.01:
                        info["normalized_shift"] = float((patched_conf - source_base) / gap)
                    else:
                        info["normalized_shift"] = 0.0

                    pair_results.append(info)

            finally:
                handle.remove()

            del inputs, outputs
            torch.cuda.empty_cache()

        # Aggregate layer results
        shifts = [r["confidence_shift"] for r in pair_results]
        norm_shifts = [r["normalized_shift"] for r in pair_results]

        results["layer_results"][layer_idx] = {
            "pairs": pair_results,
            "mean_shift": float(np.mean(shifts)),
            "std_shift": float(np.std(shifts)),
            "mean_normalized_shift": float(np.mean(norm_shifts)),
            "std_normalized_shift": float(np.std(norm_shifts)),
            "n_positive_shift": int(np.sum(np.array(shifts) > 0)),
            "n_pairs": len(pair_results),
        }

    # Add baseline info
    results["baselines"] = {
        "mean": float(all_baseline_confidences.mean()),
        "std": float(all_baseline_confidences.std()),
        "all": all_baseline_confidences.tolist(),
    }

    return results


# =============================================================================
# ANALYSIS AND VISUALIZATION
# =============================================================================


def analyze_patching_results(results: Dict) -> Dict:
    """Compute summary statistics for patching experiment."""
    analysis = {
        "layers": results["layers"],
        "n_pairs": results["n_pairs"],
        "metric": results.get("metric", "unknown"),
        "layer_effects": {},
    }

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        # Effect size: how much did patching shift confidence toward target?
        # normalized_shift of 1.0 = patching made source identical to target
        # normalized_shift of 0.0 = patching had no effect
        mean_norm = lr["mean_normalized_shift"]
        std_norm = lr["std_normalized_shift"]

        # Statistical test: is mean_norm significantly > 0?
        from scipy import stats
        norm_shifts = [p["normalized_shift"] for p in lr["pairs"]]
        t_stat, p_value = stats.ttest_1samp(norm_shifts, 0)

        analysis["layer_effects"][layer_idx] = {
            "mean_shift": lr["mean_shift"],
            "mean_normalized_shift": mean_norm,
            "std_normalized_shift": std_norm,
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "significant_p05": p_value < 0.05,
            "effect_size": mean_norm,  # Normalized shift is already effect size
        }

    # Find best layer
    best_layer = max(
        results["layers"],
        key=lambda l: analysis["layer_effects"][l]["mean_normalized_shift"]
    )
    analysis["best_layer"] = best_layer
    analysis["best_effect"] = analysis["layer_effects"][best_layer]["mean_normalized_shift"]

    return analysis


def print_summary(analysis: Dict):
    """Print patching experiment summary."""
    print("\n" + "=" * 70)
    print("ACTIVATION PATCHING RESULTS")
    print("=" * 70)

    print(f"\nMetric: {analysis['metric']}")
    print(f"Pairs tested per layer: {analysis['n_pairs']}")

    print("\n--- Normalized Shift by Layer ---")
    print("(1.0 = source becomes identical to target, 0.0 = no effect)")
    print(f"{'Layer':<8} {'Mean':<10} {'Std':<10} {'p-value':<10} {'Sig?':<6}")
    print("-" * 50)

    for layer_idx in analysis["layers"]:
        e = analysis["layer_effects"][layer_idx]
        sig = "✓" if e["significant_p05"] else ""
        print(f"{layer_idx:<8} {e['mean_normalized_shift']:<10.3f} "
              f"{e['std_normalized_shift']:<10.3f} {e['p_value']:<10.4f} {sig:<6}")

    best = analysis["best_layer"]
    best_effect = analysis["best_effect"]
    best_p = analysis["layer_effects"][best]["p_value"]

    print(f"\nBest layer: {best} (normalized shift = {best_effect:.3f}, p = {best_p:.4f})")

    if best_effect > 0.3 and best_p < 0.05:
        print("\n✓ STRONG causal effect!")
        print("  Patching high-metric activations into low-metric questions")
        print("  significantly shifts confidence toward the high-metric pattern.")
    elif best_effect > 0.1 and best_p < 0.05:
        print("\n✓ Moderate causal effect detected.")
    elif best_effect > 0:
        print("\n⚠ Weak or non-significant effect.")
    else:
        print("\n✗ No patching effect detected.")


def plot_results(analysis: Dict, results: Dict, output_path: str):
    """Create visualization of patching results."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    layers = analysis["layers"]

    # Plot 1: Normalized shift by layer
    ax1 = axes[0]
    shifts = [analysis["layer_effects"][l]["mean_normalized_shift"] for l in layers]
    stds = [analysis["layer_effects"][l]["std_normalized_shift"] for l in layers]
    ax1.bar(range(len(layers)), shifts, yerr=stds, alpha=0.7)
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Normalized Shift")
    ax1.set_title("Patching Effect by Layer")
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter of individual pair effects for best layer
    ax2 = axes[1]
    best_layer = analysis["best_layer"]
    pairs = results["layer_results"][best_layer]["pairs"]
    metric_gaps = [p["target_metric"] - p["source_metric"] for p in pairs]
    conf_shifts = [p["confidence_shift"] for p in pairs]
    ax2.scatter(metric_gaps, conf_shifts, alpha=0.5)
    ax2.set_xlabel(f"Metric Gap (target - source)")
    ax2.set_ylabel("Confidence Shift")
    ax2.set_title(f"Layer {best_layer}: Metric Gap vs Confidence Shift")
    ax2.grid(True, alpha=0.3)

    # Fit and plot regression line
    if len(metric_gaps) > 2:
        from scipy import stats
        slope, intercept, r_value, _, _ = stats.linregress(metric_gaps, conf_shifts)
        x_line = np.array([min(metric_gaps), max(metric_gaps)])
        ax2.plot(x_line, slope * x_line + intercept, 'r-', alpha=0.8,
                 label=f'r={r_value:.2f}')
        ax2.legend()

    # Plot 3: Summary text
    ax3 = axes[2]
    ax3.axis('off')

    best_effect = analysis["layer_effects"][best_layer]
    summary = f"""
ACTIVATION PATCHING SUMMARY

Metric: {analysis['metric']}
Pairs per layer: {analysis['n_pairs']}

Best Layer: {best_layer}
  Normalized shift: {best_effect['mean_normalized_shift']:.3f} ± {best_effect['std_normalized_shift']:.3f}
  p-value: {best_effect['p_value']:.4f}
  Significant: {'Yes' if best_effect['significant_p05'] else 'No'}

Interpretation:
"""
    if best_effect['mean_normalized_shift'] > 0.3:
        summary += """  ✓ Strong causal effect
  Full activation pattern matters
  for confidence judgments."""
    elif best_effect['mean_normalized_shift'] > 0.1:
        summary += """  ✓ Moderate causal effect
  Activation pattern partially
  determines confidence."""
    else:
        summary += """  ⚠ Weak/no effect
  Activation pattern may not be
  primary determinant."""

    ax3.text(0.1, 0.9, summary, transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


# =============================================================================
# MAIN
# =============================================================================


def main():
    global METRIC

    parser = argparse.ArgumentParser(description="Run activation patching experiments")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Metric for pair selection (default: {METRIC})")
    parser.add_argument("--n-pairs", type=int, default=NUM_PATCH_PAIRS,
                        help=f"Number of patch pairs per layer (default: {NUM_PATCH_PAIRS})")
    parser.add_argument("--method", type=str, default=PAIRING_METHOD,
                        choices=["extremes", "random", "quartile"],
                        help=f"Pairing method (default: {PAIRING_METHOD})")
    args = parser.parse_args()

    METRIC = args.metric
    n_pairs = args.n_pairs
    pairing_method = args.method

    print(f"Device: {DEVICE}")
    print(f"Metric: {METRIC}")
    print(f"Pairs per layer: {n_pairs}")
    print(f"Pairing method: {pairing_method}")
    print(f"Meta-judgment task: {META_TASK}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Paths for input data
    # We load from run_introspection_experiment.py outputs
    introspection_prefix = str(OUTPUTS_DIR / f"{get_model_short_name(BASE_MODEL_NAME)}_{DATASET_NAME}_introspection")
    if META_TASK == "delegate":
        introspection_prefix += "_delegate"

    paired_data_path = f"{introspection_prefix}_paired_data.json"
    direct_activations_path = f"{introspection_prefix}_direct_activations.npz"
    probe_results_path = f"{introspection_prefix}_{METRIC}_results.json"

    # Load paired data
    print(f"\nLoading paired data from {paired_data_path}...")
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)

    questions = paired_data["questions"]
    print(f"Loaded {len(questions)} questions")

    # Load metric values
    if "direct_metrics" in paired_data and METRIC in paired_data["direct_metrics"]:
        metric_values = np.array(paired_data["direct_metrics"][METRIC])
    else:
        raise ValueError(f"Metric {METRIC} not found in paired data")

    print(f"Metric range: [{metric_values.min():.3f}, {metric_values.max():.3f}]")

    # Load cached activations
    print(f"\nLoading cached activations from {direct_activations_path}...")
    acts_data = np.load(direct_activations_path)
    cached_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in acts_data.files if k.startswith("layer_")
    }
    print(f"Loaded {len(cached_activations)} layers, shape: {cached_activations[0].shape}")

    # Load probe results to select layers
    print(f"\nLoading probe results from {probe_results_path}...")
    with open(probe_results_path, "r") as f:
        probe_results = json.load(f)

    # Select layers with good direct→meta transfer
    layers = PATCHING_LAYERS
    if layers is None:
        layer_candidates = []
        if "probe_results" in probe_results:
            for layer_str, lr in probe_results["probe_results"].items():
                d2m_r2 = lr.get("direct_to_meta_fixed", {}).get("r2", 0)
                if d2m_r2 > 0.1:
                    layer_candidates.append((int(layer_str), d2m_r2))
        layer_candidates.sort(key=lambda x: -x[1])
        layers = [l[0] for l in layer_candidates[:10]]  # Top 10 layers

    if not layers:
        # Fallback: use middle-to-late layers
        all_layers = sorted(cached_activations.keys())
        n_layers = len(all_layers)
        layers = all_layers[n_layers // 3: 2 * n_layers // 3]

    layers = sorted(layers)
    print(f"Selected {len(layers)} layers: {layers}")

    # Load model
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
    )
    use_chat_template = should_use_chat_template(tokenizer)
    initialize_token_cache(tokenizer)

    # Create patch pairs
    print(f"\nCreating {n_pairs} patch pairs using '{pairing_method}' method...")
    patch_pairs = create_patch_pairs(metric_values, n_pairs, method=pairing_method, seed=SEED)

    # Print pair statistics
    source_metrics = [metric_values[s] for s, t in patch_pairs]
    target_metrics = [metric_values[t] for s, t in patch_pairs]
    print(f"Source metric: mean={np.mean(source_metrics):.3f}, std={np.std(source_metrics):.3f}")
    print(f"Target metric: mean={np.mean(target_metrics):.3f}, std={np.std(target_metrics):.3f}")

    # Run patching experiment
    print("\nRunning patching experiment...")
    results = run_patching_experiment(
        model, tokenizer, questions, cached_activations, metric_values,
        layers, patch_pairs, use_chat_template, BATCH_SIZE
    )

    # Save results
    results_path = f"{output_prefix}_{METRIC}_patching_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved results to {results_path}")

    # Analyze
    analysis = analyze_patching_results(results)

    analysis_path = f"{output_prefix}_{METRIC}_patching_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved analysis to {analysis_path}")

    # Print summary
    print_summary(analysis)

    # Plot
    plot_path = f"{output_prefix}_{METRIC}_patching_results.png"
    plot_results(analysis, results, plot_path)

    print("\n✓ Activation patching experiment complete!")


if __name__ == "__main__":
    main()
