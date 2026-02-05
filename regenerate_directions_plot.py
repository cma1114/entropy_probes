"""Regenerate directions summary plot from cached output files."""

import json
import numpy as np
from pathlib import Path

from core.plotting import plot_directions_summary

# =============================================================================
# CONFIGURATION - edit these to match your run
# =============================================================================
OUTPUT_DIR = Path("outputs")
BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC_difficulty_filtered"  # Edit this
METRICS = ["logit_gap"]  # Edit this to match what was run

# =============================================================================
# LOAD DATA
# =============================================================================

def load_results():
    """Load cached results and reconstruct data structures for plotting."""
    uncertainty_results = {}

    for metric in METRICS:
        directions_path = OUTPUT_DIR / f"{BASE_NAME}_mc_{metric}_directions.npz"
        results_path = OUTPUT_DIR / f"{BASE_NAME}_mc_{metric}_results.json"

        if not directions_path.exists() or not results_path.exists():
            print(f"Missing files for {metric}, skipping...")
            continue

        # Load directions
        dir_data = np.load(directions_path)

        # Load results JSON
        with open(results_path) as f:
            results_json = json.load(f)

        # Reconstruct the structure expected by plot_directions_summary
        num_layers = len(results_json["results"]["probe"])

        directions = {"probe": {}, "mean_diff": {}}
        for layer in range(num_layers):
            directions["probe"][layer] = dir_data[f"probe_layer_{layer}"]
            directions["mean_diff"][layer] = dir_data[f"mean_diff_layer_{layer}"]

        # Convert layer keys from strings to ints in fits
        fits = {"probe": {}, "mean_diff": {}}
        for method in ["probe", "mean_diff"]:
            for layer_str, layer_data in results_json["results"][method].items():
                fits[method][int(layer_str)] = layer_data

        # Compute comparison (probe vs mean_diff cosine similarity)
        comparison = {}
        for layer in range(num_layers):
            cos_sim = float(np.dot(directions["probe"][layer], directions["mean_diff"][layer]))
            comparison[layer] = {"cosine_sim": cos_sim}

        uncertainty_results[metric] = {
            "directions": directions,
            "fits": fits,
            "comparison": comparison,
        }

        print(f"Loaded {metric}: {num_layers} layers")

    # Load answer directions if available
    answer_results = None
    answer_dir_path = OUTPUT_DIR / f"{BASE_NAME}_mc_answer_directions.npz"
    answer_results_path = OUTPUT_DIR / f"{BASE_NAME}_mc_answer_results.json"

    if answer_dir_path.exists() and answer_results_path.exists():
        ans_dir_data = np.load(answer_dir_path)
        with open(answer_results_path) as f:
            ans_json = json.load(f)

        num_layers = len(ans_json["results"]["probe"])

        directions = {"probe": {}, "centroid": {}}
        for layer in range(num_layers):
            directions["probe"][layer] = ans_dir_data[f"probe_layer_{layer}"]
            directions["centroid"][layer] = ans_dir_data[f"centroid_layer_{layer}"]

        fits = {"probe": {}, "centroid": {}}
        for method in ["probe", "centroid"]:
            for layer_str, layer_data in ans_json["results"][method].items():
                fits[method][int(layer_str)] = layer_data

        comparison = {}
        for layer_str, comp_data in ans_json["comparison"].items():
            comparison[int(layer_str)] = comp_data

        answer_results = {
            "directions": directions,
            "fits": fits,
            "comparison": comparison,
        }
        print(f"Loaded answer directions: {num_layers} layers")
    else:
        print("No answer directions found")

    return uncertainty_results, answer_results


def main():
    uncertainty_results, answer_results = load_results()

    if not uncertainty_results:
        print("No data loaded, exiting")
        return

    output_path = OUTPUT_DIR / f"{BASE_NAME}_mc_directions.png"

    # These args are unused but kept for API compatibility
    plot_directions_summary(
        uncertainty_results=uncertainty_results,
        answer_results=answer_results,
        metrics_dict={},  # unused
        metadata=[],  # unused
        metric_info_map={},  # unused
        output_path=output_path,
        title_prefix="MC",
    )

    print(f"\nRegenerated: {output_path}")


if __name__ == "__main__":
    main()
