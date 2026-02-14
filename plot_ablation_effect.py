#!/usr/bin/env python3
"""
Plot layer-wise ablation effects for a specific direction pair.

Usage:
    python plot_ablation_effect.py <ablate_type> <measure_type> <ablate_layer>

Example:
    python plot_ablation_effect.py uncertainty confidence 14

Configure MODEL, DATASET below to specify which results file to use.

This loads the cross-direction results JSON and plots:
- Panel 1: Raw projections for baseline (all layers) and ablated (post-ablation)
- Panel 2: Raw delta with error bars (significance indicator)
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from core.config_utils import get_output_path, find_output_file
from core.model_utils import get_model_dir_name

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
DATASET = "TriviaMC_difficulty_filtered"
METRIC = "logit_gap"

# Uses centralized path management from core.config_utils


def load_results(results_path: Path) -> dict:
    """Load results JSON file."""
    with open(results_path) as f:
        return json.load(f)


def extract_ablation_data(
    results: dict,
    ablate_type: str,
    measure_type: str,
    ablate_layer: int
) -> dict:
    """
    Extract baseline and ablated projections for a specific ablation.

    Returns dict mapping measure_layer -> {...} for layers with ablation data.
    """
    data_by_layer = {}

    # Build the key prefix we're looking for
    prefix = f"{ablate_type}_L{ablate_layer}_to_{measure_type}_L"

    for key, value in results.get("results", {}).items():
        if key.startswith(prefix):
            try:
                measure_layer = int(key.split("_L")[-1])
                data_by_layer[measure_layer] = value
            except (ValueError, IndexError):
                continue

    return data_by_layer


def plot_ablation_effect(
    results: dict,
    ablate_type: str,
    measure_type: str,
    ablate_layer: int,
    output_path: Path = None,
    show: bool = True
):
    """
    Create two-panel plot showing raw projections and delta.

    Panel 1: Raw projections (trajectories without CIs)
      - Baseline: shown for ALL layers (from baseline_projections)
      - Ablated: shown for layers > ablation_layer (from results)
      - No CIs (overlapping CIs are misleading for significance)

    Panel 2: Raw delta with error bars (significance indicator)
      - Error bars show if effect is significant (don't cross zero = significant)
    """
    # Extract ablation data
    data_by_layer = extract_ablation_data(results, ablate_type, measure_type, ablate_layer)

    if not data_by_layer:
        print(f"No ablation data found for {ablate_type} L{ablate_layer} -> {measure_type}")
        return

    # Get baseline projections for ALL layers
    baseline_proj_data = results.get("baseline_projections", {}).get(measure_type, {})

    if not baseline_proj_data:
        print(f"No baseline projections found for {measure_type}")
        print("Note: Re-run run_cross_direction_causality.py to generate baseline projections for all layers")
        return

    # Convert string keys to int
    baseline_proj_data = {int(k): v for k, v in baseline_proj_data.items()}

    # Determine layer range
    all_layers = sorted(set(baseline_proj_data.keys()) | set(data_by_layer.keys()))
    max_layer = max(all_layers)

    # Extract baseline trajectory for ALL layers (raw projections)
    baseline_layers = sorted(baseline_proj_data.keys())
    baseline_raw_mean = [baseline_proj_data[l]["mean"] for l in baseline_layers]

    # Extract ablated trajectory for post-ablation layers only (raw projections)
    ablated_layers = sorted([l for l in data_by_layer.keys() if l > ablate_layer])
    if ablated_layers:
        ablated_raw_mean = []
        raw_delta_means = []
        raw_delta_ci_low = []
        raw_delta_ci_high = []

        for l in ablated_layers:
            d = data_by_layer[l]

            # Ablated raw projection
            ablated_raw_mean.append(d.get("ablated_mean", 0))

            # Raw delta for panel 2
            raw_delta_means.append(d.get("delta_mean", 0))
            raw_delta_ci_low.append(d.get("delta_ci_low", d.get("delta_mean", 0)))
            raw_delta_ci_high.append(d.get("delta_ci_high", d.get("delta_mean", 0)))

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    # ==========================================================================
    # Panel 1: Raw projections (no CI bands - significance shown in Panel 2)
    # ==========================================================================

    # Baseline trajectory (all layers)
    ax1.plot(baseline_layers, baseline_raw_mean, 'b-', linewidth=2,
             label='Baseline', marker='o', markersize=4)

    # Ablated trajectory (post-ablation layers only)
    if ablated_layers:
        ax1.plot(ablated_layers, ablated_raw_mean, 'r-', linewidth=2,
                 label=f'After ablating {ablate_type}', marker='s', markersize=4)

    # Vertical line at ablation point
    ax1.axvline(ablate_layer, color='green', linestyle='-', linewidth=2,
                alpha=0.8, label=f'Ablation (L{ablate_layer})')

    ax1.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    ax1.set_ylabel(f'{measure_type.capitalize()} projection')
    ax1.set_title(f'Effect of Ablating {ablate_type} at Layer {ablate_layer}')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # ==========================================================================
    # Panel 2: Raw delta with error bars (NOT normalized - interpretable scale)
    # ==========================================================================

    if ablated_layers:
        colors = ['green' if d < 0 else 'orange' for d in raw_delta_means]
        ax2.bar(ablated_layers, raw_delta_means, color=colors, alpha=0.7,
                edgecolor='black', linewidth=0.5)

        # Error bars
        yerr_low = [d - ci_l for d, ci_l in zip(raw_delta_means, raw_delta_ci_low)]
        yerr_high = [ci_h - d for d, ci_h in zip(raw_delta_means, raw_delta_ci_high)]
        ax2.errorbar(ablated_layers, raw_delta_means, yerr=[yerr_low, yerr_high],
                     fmt='none', color='black', capsize=2, linewidth=1)

        # Summary annotation
        mean_delta = np.mean(raw_delta_means)
        direction = "increases" if mean_delta > 0 else "decreases"
        ax2.text(0.98, 0.95, f'Mean delta = {mean_delta:+.4f}\n(ablation {direction} {measure_type})',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=9)

    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(ablate_layer, color='green', linestyle='-', linewidth=2, alpha=0.8)

    ax2.set_xlabel('Measurement Layer')
    ax2.set_ylabel('Raw delta\n(ablated - baseline)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")

    if show:
        plt.show()
    else:
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Plot layer-wise ablation effects for a specific direction pair.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('ablate_type', type=str,
                        help='Direction type to ablate (e.g., uncertainty, answer, confidence)')
    parser.add_argument('measure_type', type=str,
                        help='Direction type to measure (e.g., uncertainty, answer, confidence)')
    parser.add_argument('ablate_layer', type=int,
                        help='Layer at which to ablate')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for figure (shows interactively if not specified)')
    parser.add_argument('--no-show', action='store_true',
                        help='Do not display figure interactively')

    args = parser.parse_args()

    # Build results path from configuration
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    results_filename = f"{DATASET}_cross_direction_{METRIC}_results.json"
    results_path = find_output_file(results_filename, model_dir=model_dir)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_filename}")
        print(f"Edit MODEL, DATASET, and METRIC in this script to point to the correct file.")
        sys.exit(1)

    print(f"Loading: {results_path}")
    results = load_results(results_path)

    # Check for baseline projections
    if "baseline_projections" not in results:
        print("\nWarning: No baseline_projections in JSON file.")
        print("Re-run run_cross_direction_causality.py to generate baseline projections for all layers.")

    # Check for ablation data
    data = extract_ablation_data(results, args.ablate_type, args.measure_type, args.ablate_layer)

    if not data:
        print(f"\nNo data found for: {args.ablate_type} L{args.ablate_layer} -> {args.measure_type}")
        print("\nAvailable ablation types:", set(k.split("_L")[0] for k in results.get("results", {}).keys()))

        # Show available layers for this ablate_type
        prefix = f"{args.ablate_type}_L"
        available_layers = set()
        for k in results.get("results", {}).keys():
            if k.startswith(prefix):
                try:
                    layer = int(k.split("_L")[1].split("_")[0])
                    available_layers.add(layer)
                except:
                    pass
        if available_layers:
            print(f"Available ablation layers for {args.ablate_type}: {sorted(available_layers)}")
        sys.exit(1)

    print(f"Found ablation data for {len(data)} measurement layers")

    # Report baseline projections availability
    baseline_data = results.get("baseline_projections", {}).get(args.measure_type, {})
    if baseline_data:
        layers = sorted(int(k) for k in baseline_data.keys())
        print(f"Baseline projections available for layers {min(layers)}-{max(layers)}")

    # Generate output path if not specified
    output_path = None
    if args.output:
        output_path = Path(args.output)
    elif not args.no_show:
        # Auto-generate output path following codebase convention
        output_path = get_output_path(f"{DATASET}_cross_direction_{METRIC}_ablation_effect.png", model_dir=model_dir)

    # Plot
    plot_ablation_effect(
        results,
        args.ablate_type,
        args.measure_type,
        args.ablate_layer,
        output_path=output_path,
        show=not args.no_show
    )


if __name__ == "__main__":
    main()
