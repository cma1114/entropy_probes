"""
Plot confidence impact from ablation results.

Shows mean Δconf by layer with bootstrap 95% CI - how much ablation
shifts reported confidence (not the correlation with MC uncertainty).

Inputs:
    outputs/{model_dir}/results/{base}_ablation_{task}_{dir_suffix}_{method}_results.json

Outputs:
    outputs/{model_dir}/results/{base}_ablation_{task}_{dir_suffix}_{method}_confidence_{pos}.png
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from core.config_utils import find_output_file, get_output_path
from core.model_utils import get_model_dir_name
from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA, CONDITION_COLORS

# =============================================================================
# CONFIGURATION
# =============================================================================

# Path to ablation results JSON (if None, construct from components below)
RESULTS_PATH = None

# Or specify components to find the file automatically
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False
DATASET = "TriviaMC_difficulty_filtered"
META_TASK = "delegate"
DIRECTION_TYPE = "uncertainty"
METRIC = "logit_gap"
METHOD = "mean_diff"
POSITION = "final"  # Which position to plot (or "all" for all positions)

# Cross-dataset (set if directions came from different dataset)
DIRECTION_DATASET = None

# Confidence signal used (only affects filename for delegate task)
# Set to "logit_margin" if that's what was used in run_ablation_causality.py
CONFIDENCE_SIGNAL = "logit_margin"  # "prob" or "logit_margin"


# =============================================================================
# MAIN
# =============================================================================

def plot_confidence_impact(analysis: dict, position: str, output_path: Path):
    """
    Create single-panel figure showing mean Δconf by layer with bootstrap 95% CI.
    """
    per_layer = analysis.get("per_layer", {})
    if not per_layer:
        print(f"  No per_layer data for {position}")
        return

    layers = sorted([int(l) for l in per_layer.keys()])
    if not layers:
        return

    # Extract data
    delta_conf_mean = np.array([per_layer[str(l)]["delta_conf_mean"] for l in layers], dtype=np.float32)

    # Get bootstrap CI if available
    has_ci = "delta_conf_mean_ci95" in per_layer[str(layers[0])]
    if has_ci:
        ci_lo = np.array([per_layer[str(l)]["delta_conf_mean_ci95"][0] for l in layers], dtype=np.float32)
        ci_hi = np.array([per_layer[str(l)]["delta_conf_mean_ci95"][1] for l in layers], dtype=np.float32)

    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(layers))

    # Reference line at y=0
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # CI band (if available) and line
    if has_ci:
        ax.fill_between(x, ci_lo, ci_hi, color=CONDITION_COLORS["ablated"], alpha=CI_ALPHA)
    ax.plot(x, delta_conf_mean, 'o-', color=CONDITION_COLORS["ablated"],
            markersize=4, linewidth=1.5)

    ax.set_xticks(x[::2])
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), 2)])
    ax.set_xlabel('Layer')
    ax.set_ylabel('Δ Confidence (ablated − baseline)')
    ax.grid(True, alpha=GRID_ALPHA)

    # Title with key info
    metric = analysis.get("metric", "unknown")
    peak_idx = int(np.argmin(delta_conf_mean))
    peak_layer = layers[peak_idx]
    peak_val = float(delta_conf_mean[peak_idx])

    if has_ci:
        peak_ci_str = f" [{ci_lo[peak_idx]:.3f}, {ci_hi[peak_idx]:.3f}]"
    else:
        peak_ci_str = ""

    ax.set_title(f'Confidence Impact: {DIRECTION_TYPE} {metric} ({position})\n'
                 f'Peak: L{peak_layer} Δconf = {peak_val:.3f}{peak_ci_str}',
                 fontsize=11)

    save_figure(fig, output_path)


def main():
    # Find results file
    if RESULTS_PATH:
        results_path = Path(RESULTS_PATH)
    else:
        model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)

        # Build filename
        dir_suffix = f"{DIRECTION_TYPE}_{METRIC}" if DIRECTION_TYPE == "uncertainty" else DIRECTION_TYPE
        cross_suffix = f"_from_{DIRECTION_DATASET}" if DIRECTION_DATASET else ""
        signal_suffix = f"_{CONFIDENCE_SIGNAL}" if META_TASK == "delegate" and CONFIDENCE_SIGNAL != "prob" else ""
        filename = f"{DATASET}_ablation_{META_TASK}_{dir_suffix}_{METHOD}{cross_suffix}{signal_suffix}_results.json"

        results_path = find_output_file(filename, model_dir=model_dir)

    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    print(f"Loading: {results_path}")
    with open(results_path, "r") as f:
        data = json.load(f)

    # Determine output path
    output_base = results_path.stem.replace("_results", "_confidence")
    output_dir = results_path.parent

    # Plot for each position
    if "by_position" in data:
        positions = list(data["by_position"].keys()) if POSITION == "all" else [POSITION]
        for pos in positions:
            if pos not in data["by_position"]:
                print(f"  Position '{pos}' not in results, skipping")
                continue

            analysis = data["by_position"][pos]
            output_path = output_dir / f"{output_base}_{pos}.png"
            print(f"Plotting {pos}...")
            plot_confidence_impact(analysis, pos, output_path)
    else:
        # Legacy format (no by_position)
        output_path = output_dir / f"{output_base}.png"
        print("Plotting (legacy format)...")
        plot_confidence_impact(data, "final", output_path)

    print("Done.")


if __name__ == "__main__":
    main()
