"""
Analyze causal ordering of uncertainty representations in meta-task activations.

Tests the hypothesis: If MC uncertainty causally influences meta-judgment decisions,
it should appear in activations BEFORE meta output uncertainty and meta confidence.

Compares layer-by-layer R² for four signals in meta-task activations:
1. D→M transfer: MC uncertainty direction (from MC task) applied to meta activations
2. mcuncert: Direction trained on meta activations to predict MC uncertainty
3. metauncert: Direction trained on meta activations to predict meta output uncertainty
4. confdir: Direction trained on meta activations to predict stated confidence

Prediction if MC uncertainty is causal:
- D→M transfer and/or mcuncert should peak at EARLIER layers
- metauncert should peak at MIDDLE layers
- confdir should peak at LATER layers

If they all peak at the same layer, or confdir peaks before MC signals,
this argues against MC uncertainty being causal for meta-judgments.

Inputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_transfer_results_{pos}.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_confdir_results_{pos}.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_mcuncert_results_{pos}.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_metauncert_results_{pos}.json

Outputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_causal_ordering.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_causal_ordering.png

Run after:
    test_meta_transfer.py with all direction-finding flags enabled
"""

import json
import numpy as np
import matplotlib.pyplot as plt

from core.model_utils import get_model_dir_name
from core.config_utils import get_output_path, find_output_file, get_config_dict
from core.plotting import GRID_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

DATASET = "TriviaMC_difficulty_filtered"
META_TASK = "delegate"  # or "other_confidence", "delegate"
PROBE_POSITION = "final"

# Which metric to use for D→M transfer, mcuncert, and metauncert
MC_METRIC = "logit_gap"

# For delegate task: which confdir target to load ("p_answer" or "logit_margin")
DELEGATE_CONFDIR_TARGET = "logit_margin"

# =============================================================================
# FUNCTIONS
# =============================================================================

def get_model_dir() -> str:
    return get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)


def load_transfer_r2(base_name: str, meta_task: str, pos: str, metric: str, model_dir: str) -> dict:
    """Load D→M transfer R² by layer for both methods (probe and mean_diff)."""
    path = find_output_file(f"{base_name}_meta_{meta_task}_transfer_results_{pos}.json", model_dir=model_dir)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    result = {}

    # Probe-based transfer: transfer[metric]["per_layer"][layer]["centered_r2"]
    if "transfer" in data and metric in data["transfer"]:
        per_layer = data["transfer"][metric].get("per_layer", {})
        r2_by_layer = {}
        for layer_str, layer_data in per_layer.items():
            layer = int(layer_str)
            r2_by_layer[layer] = layer_data.get("centered_r2", 0)
        if r2_by_layer:
            result["probe"] = r2_by_layer

    # Mean-diff-based transfer: mean_diff_transfer[metric]["per_layer"][layer]["centered_r2"]
    if "mean_diff_transfer" in data and metric in data["mean_diff_transfer"]:
        per_layer = data["mean_diff_transfer"][metric].get("per_layer", {})
        r2_by_layer = {}
        for layer_str, layer_data in per_layer.items():
            layer = int(layer_str)
            r2_by_layer[layer] = layer_data.get("centered_r2", 0)
        if r2_by_layer:
            result["mean_diff"] = r2_by_layer

    return result if result else None


def load_confdir_r2(base_name: str, meta_task: str, pos: str, model_dir: str, delegate_target: str = None) -> dict:
    """Load confidence direction R² by layer for both methods."""
    # Determine filename suffix based on meta task and target
    if meta_task == "delegate":
        target = delegate_target or "p_answer"
        suffix = f"_{target}"
    else:
        suffix = ""

    path = find_output_file(f"{base_name}_meta_{meta_task}_confdir{suffix}_results_{pos}.json", model_dir=model_dir)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    result = {}
    if "results" not in data:
        return None

    for method in ["probe", "mean_diff"]:
        if method in data["results"]:
            r2_by_layer = {}
            for layer_str, layer_data in data["results"][method].items():
                layer = int(layer_str)
                r2_by_layer[layer] = layer_data.get("test_r2", 0)
            if r2_by_layer:
                result[method] = r2_by_layer

    return result if result else None


def load_mcuncert_r2(base_name: str, meta_task: str, pos: str, metric: str, model_dir: str) -> dict:
    """Load meta→MC uncertainty R² by layer for both methods."""
    path = find_output_file(f"{base_name}_meta_{meta_task}_mcuncert_results_{pos}.json", model_dir=model_dir)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if "metrics" not in data or metric not in data["metrics"]:
        return None

    metric_data = data["metrics"][metric]
    if "results" not in metric_data:
        return None

    result = {}
    for method in ["probe", "mean_diff"]:
        if method in metric_data["results"]:
            r2_by_layer = {}
            for layer_str, layer_data in metric_data["results"][method].items():
                layer = int(layer_str)
                r2_by_layer[layer] = layer_data.get("test_r2", 0)
            if r2_by_layer:
                result[method] = r2_by_layer

    return result if result else None


def load_metauncert_r2(base_name: str, meta_task: str, pos: str, metric: str, model_dir: str) -> dict:
    """Load meta output uncertainty R² by layer for both methods."""
    path = find_output_file(f"{base_name}_meta_{meta_task}_metauncert_results_{pos}.json", model_dir=model_dir)
    if not path.exists():
        return None

    with open(path) as f:
        data = json.load(f)

    if "metrics" not in data or metric not in data["metrics"]:
        return None

    metric_data = data["metrics"][metric]
    if "results" not in metric_data:
        return None

    result = {}
    for method in ["probe", "mean_diff"]:
        if method in metric_data["results"]:
            r2_by_layer = {}
            for layer_str, layer_data in metric_data["results"][method].items():
                layer = int(layer_str)
                r2_by_layer[layer] = layer_data.get("test_r2", 0)
            if r2_by_layer:
                result[method] = r2_by_layer

    return result if result else None


def find_peak_layer(r2_by_layer: dict) -> tuple:
    """Find layer with highest R² and return (layer, r2)."""
    if not r2_by_layer:
        return None, None
    best_layer = max(r2_by_layer.keys(), key=lambda l: r2_by_layer[l])
    return best_layer, r2_by_layer[best_layer]


def main():
    model_dir = get_model_dir()
    base_name = DATASET

    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print(f"Meta task: {META_TASK}")
    print(f"MC metric: {MC_METRIC}")
    print()

    # Load all R² curves
    print("Loading R² curves...")

    signals = {}

    # 1. D→M transfer - both methods
    transfer_r2 = load_transfer_r2(base_name, META_TASK, PROBE_POSITION, MC_METRIC, model_dir)
    if transfer_r2:
        for method, r2_dict in transfer_r2.items():
            name = f"D→M_{method}"
            signals[name] = r2_dict
            print(f"  {name}: {len(r2_dict)} layers")
    else:
        print("  D→M transfer: NOT FOUND")

    # 2. mcuncert (meta→MC) - both methods
    mcuncert_r2 = load_mcuncert_r2(base_name, META_TASK, PROBE_POSITION, MC_METRIC, model_dir)
    if mcuncert_r2:
        for method, r2_dict in mcuncert_r2.items():
            name = f"mcuncert_{method}"
            signals[name] = r2_dict
            print(f"  {name}: {len(r2_dict)} layers")
    else:
        print("  mcuncert: NOT FOUND")

    # 3. metauncert (meta output uncertainty) - both methods
    metauncert_r2 = load_metauncert_r2(base_name, META_TASK, PROBE_POSITION, MC_METRIC, model_dir)
    if metauncert_r2:
        for method, r2_dict in metauncert_r2.items():
            name = f"metauncert_{method}"
            signals[name] = r2_dict
            print(f"  {name}: {len(r2_dict)} layers")
    else:
        print("  metauncert: NOT FOUND")

    # 4. confdir - both methods
    confdir_r2 = load_confdir_r2(base_name, META_TASK, PROBE_POSITION, model_dir, delegate_target=DELEGATE_CONFDIR_TARGET)
    if confdir_r2:
        for method, r2_dict in confdir_r2.items():
            name = f"confdir_{method}"
            signals[name] = r2_dict
            print(f"  {name}: {len(r2_dict)} layers")
    else:
        print("  confdir: NOT FOUND")

    if len(signals) < 2:
        print("\nNot enough signals found to compare.")
        return

    # Get common layers
    all_layers = set.intersection(*[set(s.keys()) for s in signals.values()])
    layers = sorted(all_layers)
    print(f"\nCommon layers: {len(layers)}")

    # Find peak layers
    print("\nPeak layers:")
    peaks = {}
    for name, r2_dict in signals.items():
        peak_layer, peak_r2 = find_peak_layer(r2_dict)
        peaks[name] = {"layer": peak_layer, "r2": peak_r2}
        print(f"  {name}: layer {peak_layer} (R²={peak_r2:.3f})")

    # Check causal ordering
    print("\nCausal ordering analysis:")
    ordered_by_peak = sorted(peaks.items(), key=lambda x: x[1]["layer"])
    print("  Order by peak layer (earliest → latest):")
    for i, (name, info) in enumerate(ordered_by_peak):
        print(f"    {i+1}. {name}: layer {info['layer']}")

    # Check if MC signals peak before meta signals
    # MC signals: D→M transfer, mcuncert (both methods)
    # Meta signals: metauncert, confdir (both methods)
    mc_signal_patterns = ["D→M_probe", "D→M_mean_diff", "mcuncert_probe", "mcuncert_mean_diff"]
    meta_signal_patterns = ["metauncert_probe", "metauncert_mean_diff",
                           "confdir_probe", "confdir_mean_diff"]

    mc_peaks = [peaks[s]["layer"] for s in mc_signal_patterns if s in peaks]
    meta_peaks = [peaks[s]["layer"] for s in meta_signal_patterns if s in peaks]

    if mc_peaks and meta_peaks:
        mc_avg = np.mean(mc_peaks)
        meta_avg = np.mean(meta_peaks)
        print(f"\n  MC signals average peak: layer {mc_avg:.1f}")
        print(f"  Meta signals average peak: layer {meta_avg:.1f}")

        if mc_avg < meta_avg:
            print("  ✓ MC signals peak BEFORE meta signals (consistent with causal hypothesis)")
        else:
            print("  ✗ MC signals do NOT peak before meta signals (inconsistent with causal hypothesis)")

    # Save results
    results = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            meta_task=META_TASK,
            mc_metric=MC_METRIC,
            probe_position=PROBE_POSITION,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "signals": list(signals.keys()),
        "n_layers": len(layers),
        "peaks": {name: {"layer": int(info["layer"]), "r2": float(info["r2"])}
                  for name, info in peaks.items()},
        "r2_by_layer": {name: {int(l): float(r) for l, r in r2_dict.items()}
                        for name, r2_dict in signals.items()},
    }

    if mc_peaks and meta_peaks:
        results["causal_analysis"] = {
            "mc_signals": [s for s in mc_signal_patterns if s in peaks],
            "meta_signals": [s for s in meta_signal_patterns if s in peaks],
            "mc_avg_peak": float(mc_avg),
            "meta_avg_peak": float(meta_avg),
            "mc_before_meta": bool(mc_avg < meta_avg),
        }

    results_path = get_output_path(f"{base_name}_meta_{META_TASK}_causal_ordering_{MC_METRIC}_{PROBE_POSITION}.json", model_dir=model_dir)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {results_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    # Colors: each signal type gets distinct hue; probe=solid, mean_diff=dashed
    colors = {
        "D→M_probe": "tab:purple",         # MC: purple
        "D→M_mean_diff": "violet",         # MC: violet (dashed)
        "mcuncert_probe": "tab:blue",      # MC: blue
        "mcuncert_mean_diff": "tab:cyan",  # MC: cyan (dashed)
        "metauncert_probe": "tab:green",   # Meta: green
        "metauncert_mean_diff": "lime",    # Meta: lime (dashed)
        "confdir_probe": "tab:red",        # Meta: red
        "confdir_mean_diff": "tab:orange", # Meta: orange (dashed)
    }

    linestyles = {
        "D→M_probe": "-",
        "D→M_mean_diff": "--",
        "mcuncert_probe": "-",
        "mcuncert_mean_diff": "--",
        "metauncert_probe": "-",
        "metauncert_mean_diff": "--",
        "confdir_probe": "-",
        "confdir_mean_diff": "--",
    }

    for name, r2_dict in signals.items():
        r2_vals = [r2_dict.get(l, 0) for l in layers]
        ax.plot(layers, r2_vals, label=name, color=colors.get(name, "gray"),
                linestyle=linestyles.get(name, "-"), linewidth=2)

        # Mark peak
        peak_layer = peaks[name]["layer"]
        peak_r2 = peaks[name]["r2"]
        ax.scatter([peak_layer], [peak_r2], color=colors.get(name, "gray"), s=100, zorder=5)

    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_ylim(-0.5, 1.0)
    ax.set_title(f"Causal Ordering: {DATASET} / {META_TASK} / {MC_METRIC}\n(dots = peak layer)")
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    plot_path = get_output_path(f"{base_name}_meta_{META_TASK}_causal_ordering_{MC_METRIC}_{PROBE_POSITION}.png", model_dir=model_dir)
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()
