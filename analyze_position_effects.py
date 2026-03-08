"""
Analyze where signals (d_mc, d_answer) first appear in meta-task activations.

Tests the token position hypothesis: Both d_mc and d_answer might appear at
different positions in the meta-task prompt. This matters for understanding
whether uncertainty is computed early (during question processing) or late
(at final token).

Positions analyzed:
- question_mark: "?" at end of embedded MC question
- question_newline: newline after "?"
- options_newline: newline after last MC option (D: ...)
- final: last token (where output is generated)

Prerequisites (scripts that must be run first):
    1. identify_mc_correlate.py
       - Produces: {dataset}_mc_*_directions.npz (d_mc directions for transfer)

    2. test_meta_transfer.py with MULTI-POSITION config:
       - Set PROBE_POSITIONS = ["question_mark", "question_newline", "options_newline", "final"]
       - Run with META_TASK = "delegate" (or other meta task)
       - Produces per-position files:
           {dataset}_meta_{task}_transfer_results_question_mark.json
           {dataset}_meta_{task}_transfer_results_question_newline.json
           {dataset}_meta_{task}_transfer_results_options_newline.json
           {dataset}_meta_{task}_transfer_results_final.json
       - Optional: Set FIND_META_MCQ_CLASSIFIER=True for d_answer analysis
           {dataset}_meta_{task}_metamcq_results_{pos}.json

Inputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_transfer_results_{pos}.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_metamcq_results_{pos}.json (optional)

Outputs:
    outputs/{model_dir}/results/{dataset}_meta_{task}_position_analysis.json
    outputs/{model_dir}/results/{dataset}_meta_{task}_position_analysis_{metric}.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from core.model_utils import get_model_dir_name
from core.config_utils import get_output_path, find_output_file, get_config_dict
from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

DATASET = "TriviaMC_difficulty_filtered"
META_TASK = "delegate"  # "confidence", "other_confidence", or "delegate"
METRICS = ["logit_gap", "entropy"]

POSITIONS = ["question_mark", "question_newline", "options_newline", "final"]

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def load_transfer_results_by_position(base_name: str, task: str, model_dir: str) -> Dict[str, dict]:
    """Load transfer results for all positions."""
    results = {}
    for pos in POSITIONS:
        path = find_output_file(f"{base_name}_meta_{task}_transfer_results_{pos}.json", model_dir=model_dir)
        if path.exists():
            with open(path) as f:
                results[pos] = json.load(f)
    return results


def load_mcq_results_by_position(base_name: str, task: str, model_dir: str) -> Dict[str, dict]:
    """Load Meta-MCQ (answer classifier) results for all positions."""
    results = {}
    for pos in POSITIONS:
        path = find_output_file(f"{base_name}_meta_{task}_metamcq_results_{pos}.json", model_dir=model_dir)
        if path.exists():
            with open(path) as f:
                results[pos] = json.load(f)
    return results


def extract_best_r2_per_position(transfer_results: Dict[str, dict], metric: str) -> Dict[str, dict]:
    """
    Extract best R² and best layer for each position.

    Returns: {position: {"best_r2": float, "best_layer": int, "r2_by_layer": dict}}
    """
    result = {}
    for pos, data in transfer_results.items():
        transfer = data.get("transfer", {}).get(metric, {})
        per_layer = transfer.get("per_layer", {})

        if not per_layer:
            continue

        r2_by_layer = {}
        for layer_str, layer_data in per_layer.items():
            layer = int(layer_str)
            r2_by_layer[layer] = layer_data.get("centered_r2", 0)

        if r2_by_layer:
            best_layer = max(r2_by_layer.keys(), key=lambda l: r2_by_layer[l])
            result[pos] = {
                "best_r2": r2_by_layer[best_layer],
                "best_layer": best_layer,
                "r2_by_layer": r2_by_layer,
            }

    return result


def extract_best_accuracy_per_position(mcq_results: Dict[str, dict]) -> Dict[str, dict]:
    """
    Extract best accuracy and best layer for each position (for answer classifier).

    Returns: {position: {"best_accuracy": float, "best_layer": int, "accuracy_by_layer": dict}}
    """
    result = {}
    for pos, data in mcq_results.items():
        # Check both probe and centroid methods
        for method in ["probe", "centroid"]:
            method_data = data.get("results", {}).get(method, {})
            if not method_data:
                continue

            acc_by_layer = {}
            for layer_str, layer_data in method_data.items():
                layer = int(layer_str)
                acc_by_layer[layer] = layer_data.get("test_accuracy", 0)

            if acc_by_layer:
                best_layer = max(acc_by_layer.keys(), key=lambda l: acc_by_layer[l])
                key = f"{pos}_{method}"
                result[key] = {
                    "best_accuracy": acc_by_layer[best_layer],
                    "best_layer": best_layer,
                    "accuracy_by_layer": acc_by_layer,
                    "position": pos,
                    "method": method,
                }

    return result


def plot_position_comparison(
    transfer_by_position: Dict[str, dict],
    mcq_by_position: Dict[str, dict],
    output_path: Path,
    metric: str
):
    """Plot R² and accuracy comparison across positions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Transfer R² by position
    ax = axes[0]
    position_order = ["question_mark", "question_newline", "options_newline", "final"]
    position_labels = ["question_mark", "question\\nnewline", "options\\nnewline", "final"]

    # Get all layers
    all_layers = set()
    for data in transfer_by_position.values():
        all_layers.update(data.get("r2_by_layer", {}).keys())
    layers = sorted(all_layers)

    colors = {"question_mark": "tab:blue", "question_newline": "tab:cyan",
              "options_newline": "tab:green", "final": "tab:purple"}

    for pos in position_order:
        if pos not in transfer_by_position:
            continue
        data = transfer_by_position[pos]
        r2_by_layer = data["r2_by_layer"]
        r2_values = [r2_by_layer.get(l, 0) for l in layers]
        ax.plot(layers, r2_values, label=f"{pos} (best={data['best_r2']:.3f}@L{data['best_layer']})",
                color=colors[pos], linewidth=2)

    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Transfer R² ({metric})")
    ax.set_title("d_mc Transfer R² by Token Position")
    ax.legend(loc="best")
    ax.grid(True, alpha=GRID_ALPHA)

    # Right: MCQ accuracy by position
    ax = axes[1]
    if mcq_by_position:
        for key, data in mcq_by_position.items():
            pos = data["position"]
            method = data["method"]
            acc_by_layer = data["accuracy_by_layer"]
            acc_values = [acc_by_layer.get(l, 0) for l in layers]

            linestyle = "-" if method == "probe" else "--"
            ax.plot(layers, acc_values,
                    label=f"{pos}/{method} (best={data['best_accuracy']:.3f}@L{data['best_layer']})",
                    color=colors.get(pos, "gray"), linestyle=linestyle, linewidth=2)

        ax.axhline(0.25, color="red", linestyle=":", alpha=0.5, label="Chance (25%)")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Test Accuracy")
        ax.set_title("d_answer Classifier Accuracy by Token Position")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=GRID_ALPHA)
    else:
        ax.text(0.5, 0.5, "No MCQ results available", ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    save_figure(fig, output_path)


def main():
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET

    print(f"Model: {MODEL}")
    print(f"Dataset: {DATASET}")
    print(f"Meta task: {META_TASK}")
    print(f"Model dir: {model_dir}")
    print()

    # Load results
    transfer_results = load_transfer_results_by_position(base_name, META_TASK, model_dir)
    mcq_results = load_mcq_results_by_position(base_name, META_TASK, model_dir)

    print(f"Found transfer results for positions: {list(transfer_results.keys())}")
    print(f"Found MCQ results for positions: {list(mcq_results.keys())}")

    if not transfer_results:
        print("\nNo multi-position transfer results found.")
        print("Run test_meta_transfer.py with PROBE_POSITIONS = ['question_mark', 'question_newline', 'options_newline', 'final']")
        return

    # Analyze each metric
    all_analyses = {}

    for metric in METRICS:
        print(f"\n{'='*70}")
        print(f"METRIC: {metric}")
        print(f"{'='*70}")

        transfer_by_position = extract_best_r2_per_position(transfer_results, metric)
        mcq_by_position = extract_best_accuracy_per_position(mcq_results)

        print("\nTransfer R² (d_mc → meta):")
        for pos in POSITIONS:
            if pos in transfer_by_position:
                data = transfer_by_position[pos]
                print(f"  {pos:20s}: best R²={data['best_r2']:.4f} at layer {data['best_layer']}")
            else:
                print(f"  {pos:20s}: NOT FOUND")

        if mcq_by_position:
            print("\nMCQ Accuracy (d_answer from meta):")
            for pos in POSITIONS:
                for method in ["probe", "centroid"]:
                    key = f"{pos}_{method}"
                    if key in mcq_by_position:
                        data = mcq_by_position[key]
                        print(f"  {pos}/{method:8s}: best acc={data['best_accuracy']:.4f} at layer {data['best_layer']}")

        # Interpretation
        print("\nInterpretation:")
        if transfer_by_position:
            # Check where signal first appears
            best_r2_by_pos = {pos: data["best_r2"] for pos, data in transfer_by_position.items()}
            earliest_signal = min(POSITIONS, key=lambda p: POSITIONS.index(p)
                                  if p in best_r2_by_pos and best_r2_by_pos[p] > 0.1 else 999)

            if earliest_signal != "final":
                print(f"  d_mc signal appears at '{earliest_signal}' (EARLY computation)")
                print("  → Uncertainty is encoded during question processing, not just at output")
            else:
                print(f"  d_mc signal only strong at 'final' (LATE computation)")
                print("  → Uncertainty computed at output generation time")

        all_analyses[metric] = {
            "transfer_by_position": {
                pos: {"best_r2": data["best_r2"], "best_layer": data["best_layer"]}
                for pos, data in transfer_by_position.items()
            },
            "mcq_by_position": {
                key: {"best_accuracy": data["best_accuracy"], "best_layer": data["best_layer"]}
                for key, data in mcq_by_position.items()
            } if mcq_by_position else None,
        }

        # Plot
        plot_path = get_output_path(f"{base_name}_meta_{META_TASK}_position_analysis_{metric}.png", model_dir=model_dir)
        plot_position_comparison(transfer_by_position, mcq_by_position, plot_path, metric)
        print(f"\nPlot: {plot_path}")

    # Save summary
    summary = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            meta_task=META_TASK,
            metrics=METRICS,
            positions=POSITIONS,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "analyses": all_analyses,
    }

    summary_path = get_output_path(f"{base_name}_meta_{META_TASK}_position_analysis.json", model_dir=model_dir)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    main()
