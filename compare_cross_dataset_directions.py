"""
Compare direction vectors computed from different datasets using permutation null baseline.

Tests whether directions (mean-diff or probe) capture a shared signal across datasets,
or are just fitting to dataset-specific noise in activation space.

Approach:
1. Auto-discover all datasets for a given model
2. Run pairwise comparisons between all dataset pairs
3. For each pair: compute cross-dataset cosine similarity per layer
4. Generate permutation null by shuffling metric labels and recomputing directions
5. Compare observed similarity against null distribution

Inputs (from identify_mc_correlate.py):
- {model}_{dataset}_mc_{metric}_directions.npz: Direction vectors
- {model}_{dataset}_mc_activations.npz: Activations and metric values

Outputs:
- Cross-dataset comparison JSON with statistical results per pair
- Visualization plot
"""

from pathlib import Path
from itertools import combinations
import json
import numpy as np
from scipy.stats import percentileofscore
from tqdm import tqdm
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

# Model prefix to search for (auto-discovers all datasets for this model)
MODEL_PREFIX = "Llama-3.3-70B-Instruct"

# Metrics to compare
METRICS = ["entropy", "logit_gap"]

# Direction methods to test
METHODS = ["mean_diff", "probe"]

# Permutation null parameters
N_PERMUTATIONS = 100
SEED = 42

# Mean-diff quantile (must match identify_mc_correlate.py)
MEAN_DIFF_QUANTILE = 0.25

# Output directory
OUTPUT_DIR = Path(__file__).parent / "outputs"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def load_directions(base_name: str, metric: str, method: str = "mean_diff") -> dict:
    """
    Load direction vectors from a *_directions.npz file.

    Args:
        base_name: e.g., "Llama-3.3-70B-Instruct_TriviaMC"
        metric: e.g., "entropy" or "top_logit"
        method: "mean_diff" or "probe"

    Returns:
        {layer: normalized_direction_vector}
    """
    directions_path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"
    if not directions_path.exists():
        raise FileNotFoundError(f"Directions not found: {directions_path}")

    data = np.load(directions_path)
    directions = {}

    # Keys are {method}_layer_{layer}
    for key in data.files:
        if key.startswith(f"{method}_layer_"):
            layer = int(key.split("_")[-1])
            vec = np.asarray(data[key], dtype=np.float32)
            # Normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-10:
                vec = vec / norm
            directions[layer] = vec

    return directions


def load_activations_and_metrics(base_name: str, metric: str) -> tuple:
    """
    Load activations and metric values from *_mc_activations.npz.

    Returns:
        activations: {layer: (n_samples, hidden_dim)}
        metric_values: (n_samples,)
    """
    activations_path = OUTPUT_DIR / f"{base_name}_mc_activations.npz"
    if not activations_path.exists():
        raise FileNotFoundError(f"Activations not found: {activations_path}")

    data = np.load(activations_path)

    # Load activations per layer
    activations = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_")[1])
            activations[layer] = data[key]

    # Load metric values
    if metric not in data.files:
        raise ValueError(f"Metric '{metric}' not found in {activations_path}")
    metric_values = data[metric]

    return activations, metric_values


def compute_mean_diff_direction(X: np.ndarray, y: np.ndarray, quantile: float = 0.25) -> np.ndarray:
    """
    Compute mean-diff direction: mean(top_quantile) - mean(bottom_quantile), normalized.

    Args:
        X: (n_samples, hidden_dim) activations
        y: (n_samples,) metric values
        quantile: fraction for top/bottom groups

    Returns:
        Normalized direction vector (hidden_dim,)
    """
    n = len(y)
    n_group = max(1, int(n * quantile))

    sorted_idx = np.argsort(y)
    low_idx = sorted_idx[:n_group]
    high_idx = sorted_idx[-n_group:]

    mean_low = X[low_idx].mean(axis=0)
    mean_high = X[high_idx].mean(axis=0)

    direction = mean_high - mean_low
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm

    return direction.astype(np.float32)


def compute_probe_direction(X: np.ndarray, y: np.ndarray, alpha: float = 1000.0, pca_components: int = 100) -> np.ndarray:
    """
    Compute probe direction via Ridge regression on PCA-reduced activations.

    Args:
        X: (n_samples, hidden_dim) activations
        y: (n_samples,) metric values
        alpha: Ridge regularization
        pca_components: Number of PCA components

    Returns:
        Normalized direction vector (hidden_dim,)
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(pca_components, X.shape[0], X.shape[1]))
    X_pca = pca.fit_transform(X_scaled)

    # Ridge regression
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_pca, y)

    # Transform weights back to full space
    # direction in PCA space -> original space
    weights_pca = ridge.coef_  # (pca_components,)
    direction = pca.components_.T @ weights_pca  # (hidden_dim, pca_components) @ (pca_components,)

    # Apply scaler transform (direction is in centered/scaled space)
    # To get interpretable direction: unscale
    direction = direction / (scaler.scale_ + 1e-10)

    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm

    return direction.astype(np.float32)


class ProbeDirectionBatch:
    """
    Efficiently compute probe directions for many label permutations.

    Pre-computes the PCA transformation and Ridge solve matrix once,
    then computes directions for all permutations via batched matrix multiplication.
    """

    def __init__(self, X: np.ndarray, alpha: float = 1000.0, pca_components: int = 100):
        """
        Pre-compute reusable components from activations.

        Args:
            X: (n_samples, hidden_dim) activations
            alpha: Ridge regularization
            pca_components: Number of PCA components
        """
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        n_components = min(pca_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X_scaled)  # (n, k)

        # Pre-compute Ridge solve matrix: (X'X + αI)⁻¹X'
        # For Ridge: w = (X'X + αI)⁻¹X'y
        # We pre-compute M = (X'X + αI)⁻¹X' so that w = M @ y
        XtX = self.X_pca.T @ self.X_pca  # (k, k)
        XtX_reg = XtX + alpha * np.eye(XtX.shape[0])
        XtX_reg_inv = np.linalg.inv(XtX_reg)  # (k, k)
        self.solve_matrix = XtX_reg_inv @ self.X_pca.T  # (k, n)

        # For back-projection: direction = pca.components_.T @ weights / scale
        self.back_proj = self.pca.components_.T / (self.scaler.scale_[:, None] + 1e-10)  # (d, k)

    def compute_directions(self, Y: np.ndarray) -> np.ndarray:
        """
        Compute normalized probe directions for multiple label vectors.

        Args:
            Y: (n_samples, n_permutations) matrix of label permutations

        Returns:
            (n_permutations, hidden_dim) normalized direction vectors
        """
        # Compute all Ridge weights: (k, n) @ (n, p) = (k, p)
        weights_pca = self.solve_matrix @ Y  # (k, n_perms)

        # Back-project to full space: (d, k) @ (k, p) = (d, p)
        directions = self.back_proj @ weights_pca  # (d, n_perms)

        # Normalize each column
        norms = np.linalg.norm(directions, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)
        directions = directions / norms

        return directions.T.astype(np.float32)  # (n_perms, d)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors (assumed already normalized)."""
    return float(np.dot(a, b))


# =============================================================================
# DATASET DISCOVERY
# =============================================================================


def discover_datasets(model_prefix: str) -> list:
    """
    Auto-discover all datasets available for a given model.

    Looks for files matching: {model_prefix}_{dataset}_mc_activations.npz

    Returns:
        List of full base names, e.g., ["Llama-3.3-70B-Instruct_PopMC", "Llama-3.3-70B-Instruct_TriviaMC"]
    """
    pattern = f"{model_prefix}_*_mc_activations.npz"
    matches = sorted(OUTPUT_DIR.glob(pattern))

    datasets = []
    for path in matches:
        # Extract base name by removing "_mc_activations.npz"
        base_name = path.name.replace("_mc_activations.npz", "")
        datasets.append(base_name)

    return datasets


def extract_dataset_name(base_name: str, model_prefix: str) -> str:
    """Extract just the dataset part from a base name."""
    if base_name.startswith(model_prefix + "_"):
        return base_name[len(model_prefix) + 1:]
    return base_name


# =============================================================================
# PAIRWISE COMPARISON
# =============================================================================


def compare_pair(
    dataset_a: str,
    dataset_b: str,
    metrics: list,
    methods: list,
    n_permutations: int,
    seed: int,
    output_prefix: str,
) -> dict:
    """
    Compare directions between two datasets.

    Processes all metrics for each method before moving to next method.
    This means all fast mean_diff runs complete before slow probe runs start.

    Saves results incrementally after each (method, metric) completes.

    Args:
        output_prefix: Base path for output files (without extension)

    Returns:
        Full results dict (also saved to {output_prefix}.json)
    """
    results_path = Path(f"{output_prefix}.json")

    # Load existing results if present (for resuming)
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"    Loaded existing results from {results_path.name}")
    else:
        results = {
            "config": {
                "dataset_a": dataset_a,
                "dataset_b": dataset_b,
                "metrics": metrics,
                "methods": methods,
                "n_permutations": n_permutations,
                "seed": seed,
                "mean_diff_quantile": MEAN_DIFF_QUANTILE,
            },
            "comparisons": {},
        }

    # Cache loaded activations to avoid reloading for each method
    activations_cache = {}

    # Process methods in order (mean_diff first = fast results first)
    for method in methods:
        print(f"\n  {'#'*50}")
        print(f"  METHOD: {method.upper()}")
        print(f"  {'#'*50}")

        for metric in metrics:
            # Initialize metric dict if needed
            if metric not in results["comparisons"]:
                results["comparisons"][metric] = {}

            # Skip if already completed
            if method in results["comparisons"][metric]:
                print(f"\n    Skipping {metric}/{method} (already completed)")
                continue

            print(f"\n    --- {metric} ---")

            # Load activations (with caching)
            cache_key = metric
            if cache_key not in activations_cache:
                try:
                    acts_A, y_A = load_activations_and_metrics(dataset_a, metric)
                    acts_B, y_B = load_activations_and_metrics(dataset_b, metric)
                    activations_cache[cache_key] = (acts_A, y_A, acts_B, y_B)
                except FileNotFoundError as e:
                    print(f"      Warning: {e}")
                    continue
            else:
                acts_A, y_A, acts_B, y_B = activations_cache[cache_key]

            layers = sorted(acts_A.keys())
            print(f"      {len(y_A)} x {len(y_B)} samples, {len(layers)} layers")

            # Load observed directions
            try:
                dirs_A = load_directions(dataset_a, metric, method)
                dirs_B = load_directions(dataset_b, metric, method)
            except FileNotFoundError as e:
                print(f"      Warning: {e}")
                continue

            # Compute observed cross-dataset similarity
            observed_sim = {}
            for layer in layers:
                if layer in dirs_A and layer in dirs_B:
                    observed_sim[layer] = cosine_similarity(dirs_A[layer], dirs_B[layer])

            # Generate permutation null
            print(f"      Generating {n_permutations} permutation samples...")
            null_sims = {layer: [] for layer in layers}

            rng = np.random.RandomState(seed)

            if method == "mean_diff":
                # Original loop-based approach (fast enough)
                for _ in tqdm(range(n_permutations), desc=f"        permutations"):
                    y_A_perm = rng.permutation(y_A)
                    y_B_perm = rng.permutation(y_B)

                    for layer in layers:
                        d_A_perm = compute_mean_diff_direction(acts_A[layer], y_A_perm, MEAN_DIFF_QUANTILE)
                        d_B_perm = compute_mean_diff_direction(acts_B[layer], y_B_perm, MEAN_DIFF_QUANTILE)
                        null_sims[layer].append(cosine_similarity(d_A_perm, d_B_perm))

            else:  # probe - use batched computation
                # Pre-generate all permuted label matrices
                Y_A_perms = np.column_stack([rng.permutation(y_A) for _ in range(n_permutations)])  # (n_A, n_perms)
                Y_B_perms = np.column_stack([rng.permutation(y_B) for _ in range(n_permutations)])  # (n_B, n_perms)

                for layer in tqdm(layers, desc=f"        layers"):
                    # Pre-compute PCA/Ridge solve matrix once per layer
                    batch_A = ProbeDirectionBatch(acts_A[layer])
                    batch_B = ProbeDirectionBatch(acts_B[layer])

                    # Compute all permuted directions at once
                    dirs_A_perm = batch_A.compute_directions(Y_A_perms)  # (n_perms, d)
                    dirs_B_perm = batch_B.compute_directions(Y_B_perms)  # (n_perms, d)

                    # Compute all pairwise cosine similarities (dot product since normalized)
                    sims = np.sum(dirs_A_perm * dirs_B_perm, axis=1)  # (n_perms,)
                    null_sims[layer] = sims.tolist()

            # Statistical comparison per layer
            method_results = {"per_layer": {}}
            print(f"\n      Layer-by-layer results:")
            print(f"      {'Layer':>6} | {'Observed':>8} | {'Null Mean':>10} | {'Null Std':>8} | {'p-value':>8} | {'%ile':>6}")
            print(f"      {'-'*6}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}")

            significant_layers = []
            for layer in layers:
                if layer not in observed_sim:
                    continue

                obs = observed_sim[layer]
                null = np.array(null_sims[layer])

                null_mean = float(np.mean(null))
                null_std = float(np.std(null))

                # One-tailed p-value: fraction of null >= observed (plus correction)
                p_value = (np.sum(null >= obs) + 1) / (len(null) + 1)
                percentile = percentileofscore(null, obs)

                # 95% CI of null
                null_5 = float(np.percentile(null, 5))
                null_95 = float(np.percentile(null, 95))

                method_results["per_layer"][layer] = {
                    "observed": obs,
                    "null_mean": null_mean,
                    "null_std": null_std,
                    "null_5th": null_5,
                    "null_95th": null_95,
                    "p_value": float(p_value),
                    "percentile": float(percentile),
                }

                sig_marker = "*" if p_value < 0.05 else ""
                if p_value < 0.05:
                    significant_layers.append(layer)

                print(f"      {layer:>6} | {obs:>8.4f} | {null_mean:>10.4f} | {null_std:>8.4f} | {p_value:>8.4f} | {percentile:>5.1f}% {sig_marker}")

            # Summary
            if significant_layers:
                best_layer = max(significant_layers, key=lambda l: observed_sim[l])
                print(f"\n      Significant layers (p<0.05): {significant_layers}")
                print(f"      Best significant layer: L{best_layer} (obs={observed_sim[best_layer]:.4f})")
            else:
                print(f"\n      No significant layers (p<0.05)")

            method_results["significant_layers"] = significant_layers
            method_results["n_significant"] = len(significant_layers)

            results["comparisons"][metric][method] = method_results

            # Save after each (method, metric) completes
            print(f"      Saving checkpoint to {results_path.name}...")
            with open(results_path, "w") as f:
                json.dump(results, f, indent=2)

            # Generate plot immediately for this (method, metric)
            method_plot_path = Path(f"{output_prefix}_{metric}_{method}.png")
            print(f"      Generating plot: {method_plot_path.name}...")
            _plot_single_method(method_results, layers, method, metric, dataset_a, dataset_b, method_plot_path)

            # Generate combined plot for this metric (updates as each method completes)
            _generate_combined_metric_plot(
                results["comparisons"][metric], metric, dataset_a, dataset_b, output_prefix
            )

    return results


def _plot_single_method(method_results: dict, layers: list, method: str, metric: str,
                        dataset_a: str, dataset_b: str, output_path: Path):
    """Plot cross-dataset similarity for a single method immediately after it completes."""
    per_layer = method_results["per_layer"]
    if not per_layer:
        return

    observed = np.array([per_layer.get(l, {}).get("observed", np.nan) for l in layers])
    null_mean = np.array([per_layer.get(l, {}).get("null_mean", np.nan) for l in layers])
    null_5th = np.array([per_layer.get(l, {}).get("null_5th", np.nan) for l in layers])
    null_95th = np.array([per_layer.get(l, {}).get("null_95th", np.nan) for l in layers])

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.suptitle(f"Cross-Dataset Direction Similarity: {metric} ({method})\n{dataset_a} vs {dataset_b}",
                 fontsize=11, fontweight='bold')

    # Plot null distribution band
    ax.fill_between(layers, null_5th, null_95th, alpha=0.3, color='gray', label='Null 5-95%')
    ax.plot(layers, null_mean, '--', color='gray', linewidth=1, label='Null mean')

    # Plot observed
    ax.plot(layers, observed, '-', color='tab:blue', linewidth=2, label='Observed')

    # Highlight significant layers
    significant = observed > null_95th
    if np.any(significant):
        sig_layers = np.array(layers)[significant]
        sig_obs = observed[significant]
        ax.scatter(sig_layers, sig_obs, color='red', s=30, zorder=5, label='p<0.05')

    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def _generate_combined_metric_plot(metric_data: dict, metric: str, dataset_a: str,
                                   dataset_b: str, output_prefix: str):
    """Generate combined plot for all methods of a metric from saved results."""
    methods = list(metric_data.keys())
    if not methods:
        return

    # Extract layers from first method's results
    first_method = methods[0]
    layers = sorted([int(l) for l in metric_data[first_method]["per_layer"].keys()])

    n_cols = len(methods)
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 4), squeeze=False)
    fig.suptitle(f"Cross-Dataset Direction Similarity: {metric}\n{dataset_a} vs {dataset_b}",
                 fontsize=12, fontweight='bold')

    for col, method in enumerate(methods):
        ax = axes[0, col]
        per_layer = metric_data[method]["per_layer"]

        observed = np.array([per_layer.get(str(l), per_layer.get(l, {})).get("observed", np.nan) for l in layers])
        null_mean = np.array([per_layer.get(str(l), per_layer.get(l, {})).get("null_mean", np.nan) for l in layers])
        null_5th = np.array([per_layer.get(str(l), per_layer.get(l, {})).get("null_5th", np.nan) for l in layers])
        null_95th = np.array([per_layer.get(str(l), per_layer.get(l, {})).get("null_95th", np.nan) for l in layers])

        # Plot null distribution band
        ax.fill_between(layers, null_5th, null_95th, alpha=0.3, color='gray', label='Null 5-95%')
        ax.plot(layers, null_mean, '--', color='gray', linewidth=1, label='Null mean')

        # Plot observed
        ax.plot(layers, observed, '-', color='tab:blue', linewidth=2, label='Observed')

        # Highlight significant layers
        significant = observed > null_95th
        if np.any(significant):
            sig_layers = np.array(layers)[significant]
            sig_obs = observed[significant]
            ax.scatter(sig_layers, sig_obs, color='red', s=30, zorder=5, label='p<0.05')

        ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title(f"{method}")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(f"{output_prefix}_{metric}.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"    Saved combined plot: {output_path.name}")


# =============================================================================
# CROSS-PAIR SYNTHESIS
# =============================================================================


def synthesize_across_pairs(all_results: dict, metrics: list, methods: list, output_dir: Path, model_prefix: str):
    """
    Synthesize findings across all dataset pairs.

    For each (metric, method), identifies:
    - Layers significant in ALL pairs (most robust)
    - Layers significant in MAJORITY of pairs
    - Mean observed similarity across pairs per layer
    - Consistency of the direction (do pairs agree on which layers are best?)
    """
    pairs = list(all_results.keys())
    n_pairs = len(pairs)

    if n_pairs < 2:
        print("\n  Only 1 pair - skipping cross-pair synthesis.")
        return {}

    print("\n" + "=" * 70)
    print("CROSS-PAIR SYNTHESIS")
    print("=" * 70)
    print(f"Synthesizing across {n_pairs} dataset pairs...")

    synthesis = {"n_pairs": n_pairs, "pairs": pairs, "by_metric": {}}

    # Get layer list from first pair's first metric
    first_pair = pairs[0]
    first_metric = metrics[0]
    if first_metric in all_results[first_pair].get("comparisons", {}):
        first_method = methods[0]
        if first_method in all_results[first_pair]["comparisons"][first_metric]:
            layers = sorted([
                int(l) for l in
                all_results[first_pair]["comparisons"][first_metric][first_method]["per_layer"].keys()
            ])
        else:
            print("  Warning: Could not determine layers from results.")
            return synthesis
    else:
        print("  Warning: No comparison data found.")
        return synthesis

    for metric in metrics:
        print(f"\n  --- {metric} ---")
        synthesis["by_metric"][metric] = {}

        for method in methods:
            print(f"\n    {method}:")

            # Collect per-layer stats across pairs
            layer_obs_sims = {l: [] for l in layers}  # observed similarities
            layer_sig_counts = {l: 0 for l in layers}  # count of pairs where significant

            for pair_key in pairs:
                pair_data = all_results[pair_key]
                if "comparisons" not in pair_data:
                    continue
                if metric not in pair_data["comparisons"]:
                    continue
                if method not in pair_data["comparisons"][metric]:
                    continue

                method_data = pair_data["comparisons"][metric][method]
                sig_layers = set(method_data.get("significant_layers", []))

                for layer in layers:
                    layer_key = str(layer) if str(layer) in method_data["per_layer"] else layer
                    if layer_key in method_data["per_layer"]:
                        obs = method_data["per_layer"][layer_key].get("observed", np.nan)
                        layer_obs_sims[layer].append(obs)
                        if layer in sig_layers:
                            layer_sig_counts[layer] += 1

            # Compute summary stats per layer
            layer_summary = {}
            for layer in layers:
                obs_list = layer_obs_sims[layer]
                if obs_list:
                    layer_summary[layer] = {
                        "mean_observed": float(np.mean(obs_list)),
                        "std_observed": float(np.std(obs_list)),
                        "min_observed": float(np.min(obs_list)),
                        "max_observed": float(np.max(obs_list)),
                        "n_pairs_significant": layer_sig_counts[layer],
                        "frac_pairs_significant": layer_sig_counts[layer] / n_pairs,
                    }

            # Find robust layers
            all_sig = [l for l in layers if layer_sig_counts[l] == n_pairs]
            majority_sig = [l for l in layers if layer_sig_counts[l] > n_pairs / 2]

            # Best layers by mean observed similarity
            if layer_summary:
                best_by_mean = sorted(layers, key=lambda l: layer_summary.get(l, {}).get("mean_observed", -1), reverse=True)[:10]
            else:
                best_by_mean = []

            synthesis["by_metric"][metric][method] = {
                "per_layer": layer_summary,
                "layers_significant_in_all_pairs": all_sig,
                "layers_significant_in_majority": majority_sig,
                "n_layers_all_pairs": len(all_sig),
                "n_layers_majority": len(majority_sig),
                "top_10_by_mean_similarity": best_by_mean,
            }

            print(f"      Layers significant in ALL {n_pairs} pairs: {len(all_sig)}")
            if all_sig:
                print(f"        {all_sig[:15]}{'...' if len(all_sig) > 15 else ''}")
            print(f"      Layers significant in majority (>{n_pairs//2}): {len(majority_sig)}")
            print(f"      Top 10 by mean similarity: {best_by_mean}")

    # Generate synthesis plot
    _plot_synthesis(synthesis, metrics, methods, layers, output_dir, model_prefix)

    return synthesis


def _plot_synthesis(synthesis: dict, metrics: list, methods: list, layers: list,
                    output_dir: Path, model_prefix: str):
    """Plot cross-pair synthesis: mean similarity ± std across pairs, with consistency markers."""
    n_pairs = synthesis["n_pairs"]

    for metric in metrics:
        if metric not in synthesis["by_metric"]:
            continue

        n_methods = len(methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5), squeeze=False)
        fig.suptitle(f"Cross-Dataset Direction Similarity: {metric}\n(Mean ± Std across {n_pairs} dataset pairs)",
                     fontsize=12, fontweight='bold')

        for col, method in enumerate(methods):
            if method not in synthesis["by_metric"][metric]:
                continue

            ax = axes[0, col]
            method_data = synthesis["by_metric"][metric][method]
            per_layer = method_data["per_layer"]

            mean_obs = np.array([per_layer.get(l, {}).get("mean_observed", np.nan) for l in layers])
            std_obs = np.array([per_layer.get(l, {}).get("std_observed", 0) for l in layers])

            # Plot mean ± std
            ax.fill_between(layers, mean_obs - std_obs, mean_obs + std_obs, alpha=0.3, color='tab:blue')
            ax.plot(layers, mean_obs, '-', color='tab:blue', linewidth=2, label='Mean observed')

            # Mark layers significant in all pairs
            all_sig = method_data.get("layers_significant_in_all_pairs", [])
            if all_sig:
                sig_mask = np.array([l in all_sig for l in layers])
                ax.scatter(np.array(layers)[sig_mask], mean_obs[sig_mask],
                           color='red', s=40, zorder=5, label=f'Sig in all {n_pairs} pairs')

            # Mark layers significant in majority but not all
            majority_sig = method_data.get("layers_significant_in_majority", [])
            majority_only = [l for l in majority_sig if l not in all_sig]
            if majority_only:
                maj_mask = np.array([l in majority_only for l in layers])
                ax.scatter(np.array(layers)[maj_mask], mean_obs[maj_mask],
                           color='orange', s=30, zorder=4, marker='s', label='Sig in majority')

            ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
            ax.set_xlabel('Layer')
            ax.set_ylabel('Cosine Similarity')
            ax.set_title(f"{method}")
            ax.legend(loc='upper left', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        output_path = output_dir / f"{model_prefix}_cross_dataset_synthesis_{metric}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\n  Saved synthesis plot: {output_path.name}")


# =============================================================================
# MAIN ANALYSIS
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("CROSS-DATASET DIRECTION COMPARISON")
    print("=" * 70)
    print(f"Model prefix: {MODEL_PREFIX}")
    print(f"Metrics: {METRICS}")
    print(f"Methods: {METHODS}")
    print(f"N permutations: {N_PERMUTATIONS}")
    print()

    # Auto-discover datasets
    print("Discovering datasets...")
    datasets = discover_datasets(MODEL_PREFIX)

    if len(datasets) < 2:
        print(f"  Found {len(datasets)} dataset(s): {datasets}")
        print("  Need at least 2 datasets for pairwise comparison. Exiting.")
        return

    print(f"  Found {len(datasets)} datasets:")
    for ds in datasets:
        ds_name = extract_dataset_name(ds, MODEL_PREFIX)
        print(f"    - {ds_name} ({ds})")

    # Generate all pairs
    pairs = list(combinations(datasets, 2))
    print(f"\n  Will compare {len(pairs)} pair(s):")
    for ds_a, ds_b in pairs:
        name_a = extract_dataset_name(ds_a, MODEL_PREFIX)
        name_b = extract_dataset_name(ds_b, MODEL_PREFIX)
        print(f"    - {name_a} vs {name_b}")

    # Run pairwise comparisons
    all_results = {}

    for pair_idx, (dataset_a, dataset_b) in enumerate(pairs):
        name_a = extract_dataset_name(dataset_a, MODEL_PREFIX)
        name_b = extract_dataset_name(dataset_b, MODEL_PREFIX)

        print(f"\n{'#'*70}")
        print(f"# PAIR {pair_idx + 1}/{len(pairs)}: {name_a} vs {name_b}")
        print(f"{'#'*70}")

        output_prefix = str(OUTPUT_DIR / f"{MODEL_PREFIX}_{name_a}_vs_{name_b}_cross_dataset")

        results = compare_pair(
            dataset_a, dataset_b,
            METRICS, METHODS,
            N_PERMUTATIONS, SEED,
            output_prefix,
        )

        pair_key = f"{name_a}_vs_{name_b}"
        all_results[pair_key] = results

    # Synthesize findings across all pairs
    synthesis = synthesize_across_pairs(all_results, METRICS, METHODS, OUTPUT_DIR, MODEL_PREFIX)

    # Save combined summary with synthesis
    summary_path = OUTPUT_DIR / f"{MODEL_PREFIX}_cross_dataset_summary.json"
    summary = {
        "config": {
            "model_prefix": MODEL_PREFIX,
            "datasets": datasets,
            "metrics": METRICS,
            "methods": METHODS,
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
        },
        "pairs": all_results,
        "synthesis": synthesis,
    }
    print(f"\nSaving summary to {summary_path}...")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final takeaway
    if synthesis and "by_metric" in synthesis:
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for metric in METRICS:
            if metric not in synthesis["by_metric"]:
                continue
            print(f"\n{metric}:")
            for method in METHODS:
                if method not in synthesis["by_metric"][metric]:
                    continue
                data = synthesis["by_metric"][metric][method]
                n_all = data.get("n_layers_all_pairs", 0)
                n_maj = data.get("n_layers_majority", 0)
                top_layers = data.get("top_10_by_mean_similarity", [])[:5]
                print(f"  {method}: {n_all} layers sig in ALL pairs, {n_maj} in majority")
                print(f"    Best layers: {top_layers}")

    print("\nDone.")


if __name__ == "__main__":
    main()
