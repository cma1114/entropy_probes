"""
Diagnose why directions with ~0.4 cosine similarity achieve equal prediction performance.

Hypothesis: Both directions contain a shared signal component. The orthogonal parts are noise.
Test: Extract the shared component and see if it predicts as well as the full directions.
"""

from pathlib import Path
import numpy as np
from scipy.stats import pearsonr
import json

OUTPUT_DIR = Path(__file__).parent / "outputs"
MODEL_PREFIX = "Llama-3.3-70B-Instruct"


def load_directions(base_name: str, metric: str, method: str = "mean_diff") -> dict:
    """Load direction vectors from a *_directions.npz file."""
    directions_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"
    if not directions_path.exists():
        raise FileNotFoundError(f"Directions not found: {directions_path}")

    data = np.load(directions_path)
    directions = {}

    for key in data.files:
        if key.startswith(f"{method}_layer_"):
            layer = int(key.split("_")[-1])
            vec = np.asarray(data[key], dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            directions[layer] = vec

    return directions


def load_activations_and_metric(base_name: str, metric: str):
    """Load activations and metric values."""
    path = OUTPUT_DIR / f"{base_name}_mc_activations.npz"
    data = np.load(path)

    activations = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_")[1])
            activations[layer] = data[key].astype(np.float32)

    return activations, data[metric]


def analyze_direction_components():
    """
    For each dataset pair:
    1. Load directions from both datasets
    2. Decompose into shared and orthogonal components
    3. Test prediction performance of each component
    """

    datasets = [
        f"{MODEL_PREFIX}_PopMC_0_difficulty_filtered",
        f"{MODEL_PREFIX}_SimpleMC",
        f"{MODEL_PREFIX}_TriviaMC"
    ]

    metric = "logit_gap"
    method = "probe"
    test_layer = 50  # Layer with good performance

    print(f"Analyzing {metric} {method} at layer {test_layer}")
    print("=" * 80)

    for i, ds_a in enumerate(datasets):
        for ds_b in datasets[i+1:]:
            print(f"\n{ds_a.replace(MODEL_PREFIX + '_', '')} vs {ds_b.replace(MODEL_PREFIX + '_', '')}")
            print("-" * 60)

            # Load directions
            dirs_a = load_directions(ds_a, metric, method)
            dirs_b = load_directions(ds_b, metric, method)

            if test_layer not in dirs_a or test_layer not in dirs_b:
                print("  Layer not found")
                continue

            dir_a = dirs_a[test_layer]
            dir_b = dirs_b[test_layer]

            # Cosine similarity
            cos_sim = np.dot(dir_a, dir_b)
            print(f"  Cosine similarity: {cos_sim:.4f}")

            # Decompose dir_b into component parallel to dir_a and orthogonal
            # dir_b = (dir_b · dir_a) * dir_a + orthogonal
            parallel_component = cos_sim * dir_a  # Component of B along A
            orthogonal_component = dir_b - parallel_component  # Component of B orthogonal to A

            # Normalize orthogonal component
            orth_norm = np.linalg.norm(orthogonal_component)
            if orth_norm > 1e-10:
                orthogonal_component = orthogonal_component / orth_norm

            print(f"  ||parallel||: {np.linalg.norm(parallel_component):.4f}")
            print(f"  ||orthogonal||: {orth_norm:.4f}")

            # Load test data (dataset B)
            acts_b, metric_b = load_activations_and_metric(ds_b, metric)
            X = acts_b[test_layer]
            y = metric_b

            # Test prediction with different directions
            pred_full_a = X @ dir_a
            pred_full_b = X @ dir_b
            pred_parallel = X @ (parallel_component / (np.linalg.norm(parallel_component) + 1e-10))
            pred_orthogonal = X @ orthogonal_component

            r_full_a, _ = pearsonr(pred_full_a, y)
            r_full_b, _ = pearsonr(pred_full_b, y)
            r_parallel, _ = pearsonr(pred_parallel, y)
            r_orthogonal, _ = pearsonr(pred_orthogonal, y)

            print(f"\n  Prediction performance on dataset B:")
            print(f"    Full dir_A (cross):     r = {r_full_a:.4f}")
            print(f"    Full dir_B (within):    r = {r_full_b:.4f}")
            print(f"    Parallel to A only:     r = {r_parallel:.4f}")
            print(f"    Orthogonal to A only:   r = {r_orthogonal:.4f}")

            # Key question: does the parallel component (shared part) explain most of the prediction?
            print(f"\n  Analysis:")
            print(f"    Shared component explains {100 * r_parallel / r_full_b:.1f}% of within performance")
            print(f"    Orthogonal component explains {100 * r_orthogonal / r_full_b:.1f}% of within performance")

            # Also test: what if we use ONLY the orthogonal component of B?
            # This is the part of B that's "unique" to B and not shared with A
            print(f"\n  If directions share signal, parallel should predict well and orthogonal should not.")


def analyze_subspace_hypothesis():
    """
    Alternative hypothesis: signal lives in a subspace.
    Test by checking if combining directions improves prediction.
    """
    print("\n\n" + "=" * 80)
    print("SUBSPACE HYPOTHESIS: Does combining directions improve prediction?")
    print("=" * 80)

    datasets = [
        f"{MODEL_PREFIX}_PopMC_0_difficulty_filtered",
        f"{MODEL_PREFIX}_SimpleMC",
        f"{MODEL_PREFIX}_TriviaMC"
    ]

    metric = "logit_gap"
    method = "probe"
    test_layer = 50

    # Load all directions
    all_dirs = {}
    for ds in datasets:
        dirs = load_directions(ds, metric, method)
        if test_layer in dirs:
            all_dirs[ds] = dirs[test_layer]

    # Test on each dataset
    for ds in datasets:
        print(f"\nTesting on {ds.replace(MODEL_PREFIX + '_', '')}:")

        acts, metric_vals = load_activations_and_metric(ds, metric)
        X = acts[test_layer]
        y = metric_vals

        # Single direction (within)
        dir_within = all_dirs[ds]
        r_within, _ = pearsonr(X @ dir_within, y)
        print(f"  Within direction alone: r = {r_within:.4f}")

        # Average of all directions
        avg_dir = np.mean([all_dirs[d] for d in all_dirs], axis=0)
        avg_dir = avg_dir / np.linalg.norm(avg_dir)
        r_avg, _ = pearsonr(X @ avg_dir, y)
        print(f"  Average of all directions: r = {r_avg:.4f}")

        # Optimal linear combination (cheating - uses test labels)
        # This gives upper bound on what's achievable in the span of the directions
        from sklearn.linear_model import Ridge
        dir_matrix = np.column_stack([all_dirs[d] for d in all_dirs])
        projections = X @ dir_matrix  # (n_samples, n_directions)
        ridge = Ridge(alpha=1.0)
        ridge.fit(projections, y)
        pred_optimal = ridge.predict(projections)
        r_optimal, _ = pearsonr(pred_optimal, y)
        print(f"  Optimal combination (upper bound): r = {r_optimal:.4f}")

        print(f"  Improvement from combination: {r_optimal - r_within:.4f}")


if __name__ == "__main__":
    analyze_direction_components()
    analyze_subspace_hypothesis()
