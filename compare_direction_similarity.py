"""
Compare direction vector similarity across datasets and models.

Tests whether uncertainty directions (mean_diff or probe) capture a shared
signal across:
1. Different datasets (same model) - Tests generalization of learned signal
2. Different models (same dataset) - Tests whether representations are similar

Uses permutation-based null baselines to assess statistical significance.

Inputs (from identify_mc_correlate.py):
    outputs/{model}_{dataset}_mc_{metric}_directions.npz   Direction vectors
    outputs/{model}_{dataset}_mc_activations.npz           Activations + metric values

Outputs:
    outputs/direction_similarity.json                      Full comparison results
    outputs/{model}_cross_dataset_similarity.png           Per-model synthesis
    outputs/cross_model_similarity_{dataset}.png           Per-dataset cross-model
    outputs/{pair}_direction_similarity.png                Per-pair detail plots

Shared parameters (must match across scripts):
    SEED, PROBE_ALPHA, PROBE_PCA_COMPONENTS, MEAN_DIFF_QUANTILE

Run after: identify_mc_correlate.py (on all desired model/dataset combinations)
"""

from pathlib import Path
from itertools import combinations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
from scipy.stats import percentileofscore
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.config_utils import get_config_dict, get_output_path, find_output_file, glob_outputs
from core.model_utils import get_model_dir_name
from core.plotting import (
    save_figure, METHOD_COLORS, GRID_ALPHA, CI_ALPHA,
    DPI, MARKER_SIZE, LINE_WIDTH,
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Cross-Dataset Mode (single model, multiple datasets) ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter
LOAD_IN_4BIT = False  # Set True for 70B+ models
LOAD_IN_8BIT = False

# --- Cross-Model Mode (multiple models, single dataset) ---
# Set CROSS_MODEL_DATASET to enable cross-model comparison
# Each entry: (model_path, adapter_or_none, load_in_4bit, load_in_8bit)
CROSS_MODEL_DATASET = None  # e.g., "TriviaMC_difficulty_filtered"
CROSS_MODEL_CONFIGS = [
    # ("meta-llama/Llama-3.1-8B-Instruct", None, False, False),
    # ("meta-llama/Llama-3.3-70B-Instruct", None, True, False),
]

# --- Experiment Parameters ---
METRICS = ["entropy", "logit_gap"]
METHODS = ["mean_diff", "probe"]  # mean_diff first (faster)
N_PERMUTATIONS = 100
SEED = 42  # Must match across scripts

# --- Direction-finding (must match across scripts) ---
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100
MEAN_DIFF_QUANTILE = 0.25

# --- Output ---
# Uses centralized path management from core.config_utils


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class DirectionSet:
    """Container for directions and activations from one model/dataset."""
    model: str
    dataset: str
    base_name: str
    directions: Dict[str, Dict[str, Dict[int, np.ndarray]]]  # {metric: {method: {layer: dir}}}
    activations: Dict[int, np.ndarray]  # {layer: (n_samples, hidden_dim)}
    metric_values: Dict[str, np.ndarray]  # {metric: values}
    hidden_dim: int
    n_layers: int


# =============================================================================
# DISCOVERY FUNCTIONS
# =============================================================================


def get_model_dir_for_config(
    model: str,
    adapter: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
) -> str:
    """Get model directory name for a model config."""
    return get_model_dir_name(model, adapter, load_in_4bit, load_in_8bit)


def get_model_dir() -> str:
    """Get model directory for the primary configured model (MODEL, ADAPTER, etc.)."""
    return get_model_dir_for_config(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)


def discover_datasets_for_model_dir(model_dir: str) -> List[str]:
    """
    Discover all datasets available for a given model directory.

    Returns:
        List of dataset names
    """
    pattern = glob_outputs("*_mc_activations.npz", model_dir=model_dir)

    datasets = []
    for path in pattern:
        # Extract dataset: {dataset}_mc_activations.npz
        dataset = path.stem.replace("_mc_activations", "")
        if dataset not in datasets:
            datasets.append(dataset)

    return sorted(datasets)


def discover_datasets() -> List[str]:
    """Discover datasets for the primary configured model."""
    return discover_datasets_for_model_dir(get_model_dir())


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================


def load_directions(base_name: str, metric: str, method: str, model_dir: str = None) -> Dict[int, np.ndarray]:
    """
    Load direction vectors from a *_directions.npz file.

    Returns:
        {layer: normalized_direction_vector}
    """
    directions_path = find_output_file(f"{base_name}_mc_{metric}_directions.npz", model_dir=model_dir)
    if not directions_path.exists():
        return {}

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


def load_activations_and_metrics(base_name: str, metrics: List[str], model_dir: str = None) -> Tuple[Dict[int, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load activations and metric values from *_mc_activations.npz.

    Returns:
        activations: {layer: (n_samples, hidden_dim)}
        metric_values: {metric: (n_samples,)}
    """
    path = find_output_file(f"{base_name}_mc_activations.npz", model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Activations not found: {path}")

    data = np.load(path)

    activations = {}
    for key in data.files:
        if key.startswith("layer_"):
            layer = int(key.split("_")[1])
            activations[layer] = data[key].astype(np.float32)

    metric_values = {}
    for metric in metrics:
        if metric in data.files:
            metric_values[metric] = data[metric].astype(np.float32)

    return activations, metric_values


def load_direction_set(
    dataset: str,
    metrics: List[str],
    methods: List[str],
    model_dir: Optional[str] = None,
) -> Optional[DirectionSet]:
    """
    Load all directions and activations for a dataset.

    Args:
        dataset: Dataset name
        metrics: List of metrics to load
        methods: List of methods to load
        model_dir: Model directory (uses configured model if None)

    Returns None if required files are missing.
    """
    if model_dir is None:
        model_dir = get_model_dir()

    try:
        activations, metric_values = load_activations_and_metrics(dataset, metrics, model_dir=model_dir)
    except FileNotFoundError:
        return None

    if not activations:
        return None

    # Get dimensions from first layer
    first_layer = min(activations.keys())
    hidden_dim = activations[first_layer].shape[1]
    n_layers = len(activations)

    # Load all direction combinations
    directions: Dict[str, Dict[str, Dict[int, np.ndarray]]] = {}
    for metric in metrics:
        directions[metric] = {}
        for method in methods:
            dirs = load_directions(dataset, metric, method, model_dir=model_dir)
            if dirs:
                directions[metric][method] = dirs

    return DirectionSet(
        model=model_dir,
        dataset=dataset,
        base_name=dataset,
        directions=directions,
        activations=activations,
        metric_values=metric_values,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    )


# =============================================================================
# PERMUTATION NULL COMPUTATION
# =============================================================================


def compute_mean_diff_direction(X: np.ndarray, y: np.ndarray, quantile: float = MEAN_DIFF_QUANTILE) -> np.ndarray:
    """
    Compute mean-diff direction: mean(top_quantile) - mean(bottom_quantile), normalized.
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


class ProbeDirectionBatch:
    """
    Efficiently compute probe directions for many label permutations.

    Pre-computes the PCA transformation and Ridge solve matrix once,
    then computes directions for all permutations via batched matrix multiplication.
    """

    def __init__(self, X: np.ndarray, alpha: float = PROBE_ALPHA, pca_components: int = PROBE_PCA_COMPONENTS):
        """Pre-compute reusable components from activations."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        n_components = min(pca_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X_scaled)  # (n, k)

        # Pre-compute Ridge solve matrix: (X'X + alphaI)^-1 X'
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


def compute_permutation_null(
    acts_A: Dict[int, np.ndarray],
    y_A: np.ndarray,
    acts_B: Dict[int, np.ndarray],
    y_B: np.ndarray,
    method: str,
    n_permutations: int,
    seed: int,
    layers: Optional[List[int]] = None,
) -> Dict[int, np.ndarray]:
    """
    Generate permutation null distribution for cross-comparison.

    For each permutation:
    1. Shuffle y_A labels -> compute direction_A_perm
    2. Shuffle y_B labels -> compute direction_B_perm
    3. Compute cosine_sim(direction_A_perm, direction_B_perm)

    Returns:
        {layer: (n_permutations,) array of null cosine similarities}
    """
    if layers is None:
        layers = sorted(set(acts_A.keys()) & set(acts_B.keys()))

    rng = np.random.RandomState(seed)
    null_sims: Dict[int, List[float]] = {layer: [] for layer in layers}

    if method == "mean_diff":
        # Loop-based approach (fast enough for mean_diff)
        for _ in tqdm(range(n_permutations), desc=f"    permutations ({method})", leave=False):
            y_A_perm = rng.permutation(y_A)
            y_B_perm = rng.permutation(y_B)

            for layer in layers:
                d_A_perm = compute_mean_diff_direction(acts_A[layer], y_A_perm)
                d_B_perm = compute_mean_diff_direction(acts_B[layer], y_B_perm)
                null_sims[layer].append(float(np.dot(d_A_perm, d_B_perm)))

    else:  # probe - use batched computation
        # Pre-generate all permuted label matrices
        Y_A_perms = np.column_stack([rng.permutation(y_A) for _ in range(n_permutations)])
        Y_B_perms = np.column_stack([rng.permutation(y_B) for _ in range(n_permutations)])

        for layer in tqdm(layers, desc=f"    layers ({method})", leave=False):
            batch_A = ProbeDirectionBatch(acts_A[layer])
            batch_B = ProbeDirectionBatch(acts_B[layer])

            dirs_A_perm = batch_A.compute_directions(Y_A_perms)  # (n_perms, d)
            dirs_B_perm = batch_B.compute_directions(Y_B_perms)  # (n_perms, d)

            # Compute all cosine similarities (dot product since normalized)
            sims = np.sum(dirs_A_perm * dirs_B_perm, axis=1)  # (n_perms,)
            null_sims[layer] = sims.tolist()

    return {layer: np.array(sims) for layer, sims in null_sims.items()}


# =============================================================================
# COMPARISON FUNCTIONS
# =============================================================================


def cosine_similarity(d1: np.ndarray, d2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors (assumed normalized)."""
    return float(np.dot(d1, d2))


def compare_direction_pair(
    dirs_A: Dict[int, np.ndarray],
    dirs_B: Dict[int, np.ndarray],
    null_sims: Dict[int, np.ndarray],
) -> Dict:
    """
    Compare two sets of directions with pre-computed permutation null.

    Returns per-layer stats and summary.
    """
    common_layers = sorted(set(dirs_A.keys()) & set(dirs_B.keys()) & set(null_sims.keys()))

    if not common_layers:
        return {"error": "No common layers"}

    per_layer = {}
    observed_abs = []
    p_values = []
    significant_layers = []

    for layer in common_layers:
        obs = cosine_similarity(dirs_A[layer], dirs_B[layer])
        null = null_sims[layer]

        null_mean = float(np.mean(null))
        null_std = float(np.std(null))

        # Two-tailed p-value: fraction of |null| >= |observed|
        p_value = (np.sum(np.abs(null) >= abs(obs)) + 1) / (len(null) + 1)
        percentile = percentileofscore(np.abs(null), abs(obs))

        null_5 = float(np.percentile(null, 5))
        null_95 = float(np.percentile(null, 95))

        per_layer[layer] = {
            "observed_cosine": obs,
            "abs_observed_cosine": abs(obs),
            "null_mean": null_mean,
            "null_std": null_std,
            "null_5th": null_5,
            "null_95th": null_95,
            "p_value": float(p_value),
            "percentile": float(percentile),
        }

        observed_abs.append(abs(obs))
        p_values.append(p_value)
        if p_value < 0.05:
            significant_layers.append(layer)

    # Summary statistics
    summary = {
        "mean_abs_cosine": float(np.mean(observed_abs)),
        "std_abs_cosine": float(np.std(observed_abs)),
        "max_abs_cosine": float(np.max(observed_abs)),
        "max_abs_cosine_layer": int(common_layers[np.argmax(observed_abs)]),
        "n_significant": len(significant_layers),
        "significant_layers": significant_layers,
        "n_layers": len(common_layers),
    }

    return {"per_layer": per_layer, "summary": summary}


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================


def run_cross_dataset_analysis(
    datasets: List[str],
    metrics: List[str],
    methods: List[str],
    n_permutations: int,
    seed: int,
    model_dir: str = None,
) -> Dict:
    """
    Run cross-dataset comparison for the configured model across multiple datasets.
    """
    if len(datasets) < 2:
        return {"error": "Need at least 2 datasets"}

    if model_dir is None:
        model_dir = get_model_dir()

    # Load all direction sets
    direction_sets: Dict[str, DirectionSet] = {}
    for dataset in datasets:
        ds = load_direction_set(dataset, metrics, methods, model_dir=model_dir)
        if ds is not None:
            direction_sets[dataset] = ds

    if len(direction_sets) < 2:
        return {"error": "Could not load at least 2 datasets"}

    results = {}
    pairs = list(combinations(sorted(direction_sets.keys()), 2))

    for ds_A, ds_B in pairs:
        pair_key = f"{ds_A}_vs_{ds_B}"
        print(f"\n  {pair_key}")
        results[pair_key] = {}

        set_A = direction_sets[ds_A]
        set_B = direction_sets[ds_B]

        for metric in metrics:
            results[pair_key][metric] = {}

            if metric not in set_A.metric_values or metric not in set_B.metric_values:
                results[pair_key][metric]["error"] = f"Metric {metric} not found"
                continue

            y_A = set_A.metric_values[metric]
            y_B = set_B.metric_values[metric]

            for method in methods:
                print(f"    {metric}/{method}...")

                if method not in set_A.directions.get(metric, {}):
                    results[pair_key][metric][method] = {"error": f"No {method} directions for {ds_A}"}
                    continue
                if method not in set_B.directions.get(metric, {}):
                    results[pair_key][metric][method] = {"error": f"No {method} directions for {ds_B}"}
                    continue

                dirs_A = set_A.directions[metric][method]
                dirs_B = set_B.directions[metric][method]

                # Compute permutation null
                null_sims = compute_permutation_null(
                    set_A.activations, y_A,
                    set_B.activations, y_B,
                    method, n_permutations, seed,
                )

                # Compare
                comparison = compare_direction_pair(dirs_A, dirs_B, null_sims)
                results[pair_key][metric][method] = comparison

                # Print summary
                if "summary" in comparison:
                    s = comparison["summary"]
                    print(f"      mean |cos|={s['mean_abs_cosine']:.3f}, "
                          f"max |cos|={s['max_abs_cosine']:.3f} (L{s['max_abs_cosine_layer']}), "
                          f"n_sig={s['n_significant']}")

                    # Generate per-pair plot
                    plot_path = get_output_path(f"{pair_key}_{metric}_{method}_similarity.png", model_dir=model_dir)
                    plot_pair_comparison(comparison, pair_key, metric, method, plot_path)

    return results


def run_cross_model_analysis(
    dataset: str,
    model_configs: List[Tuple],  # [(model, adapter, 4bit, 8bit), ...]
    metrics: List[str],
    methods: List[str],
    n_permutations: int,
    seed: int,
) -> Dict:
    """
    Run cross-model comparison for a single dataset across multiple models.

    Args:
        dataset: Dataset name to compare across
        model_configs: List of (model_path, adapter, load_in_4bit, load_in_8bit) tuples
    """
    if len(model_configs) < 2:
        return {"error": "Need at least 2 models"}

    # Load all direction sets
    direction_sets: Dict[str, DirectionSet] = {}
    for config in model_configs:
        model, adapter, load_4bit, load_8bit = config
        model_dir = get_model_dir_for_config(model, adapter, load_4bit, load_8bit)
        ds = load_direction_set(dataset, metrics, methods, model_dir=model_dir)
        if ds is not None:
            direction_sets[model_dir] = ds
        else:
            print(f"  Warning: Could not load {model_dir}/{dataset}")

    if len(direction_sets) < 2:
        return {"error": "Could not load at least 2 models"}

    results = {}
    pairs = list(combinations(sorted(direction_sets.keys()), 2))

    for model_A, model_B in pairs:
        pair_key = f"{model_A}_vs_{model_B}"
        print(f"\n  {pair_key}")

        set_A = direction_sets[model_A]
        set_B = direction_sets[model_B]

        # Check dimension compatibility
        if set_A.hidden_dim != set_B.hidden_dim:
            print(f"    SKIPPED: dimension mismatch ({set_A.hidden_dim} vs {set_B.hidden_dim})")
            results[pair_key] = {
                "skipped": True,
                "reason": f"dimension mismatch ({set_A.hidden_dim} vs {set_B.hidden_dim})",
            }
            continue

        results[pair_key] = {}

        for metric in metrics:
            results[pair_key][metric] = {}

            if metric not in set_A.metric_values or metric not in set_B.metric_values:
                results[pair_key][metric]["error"] = f"Metric {metric} not found"
                continue

            y_A = set_A.metric_values[metric]
            y_B = set_B.metric_values[metric]

            for method in methods:
                print(f"    {metric}/{method}...")

                if method not in set_A.directions.get(metric, {}):
                    results[pair_key][metric][method] = {"error": f"No {method} directions"}
                    continue
                if method not in set_B.directions.get(metric, {}):
                    results[pair_key][metric][method] = {"error": f"No {method} directions"}
                    continue

                dirs_A = set_A.directions[metric][method]
                dirs_B = set_B.directions[metric][method]

                # Compute permutation null
                null_sims = compute_permutation_null(
                    set_A.activations, y_A,
                    set_B.activations, y_B,
                    method, n_permutations, seed,
                )

                # Compare
                comparison = compare_direction_pair(dirs_A, dirs_B, null_sims)
                results[pair_key][metric][method] = comparison

                if "summary" in comparison:
                    s = comparison["summary"]
                    print(f"      mean |cos|={s['mean_abs_cosine']:.3f}, "
                          f"max |cos|={s['max_abs_cosine']:.3f} (L{s['max_abs_cosine_layer']}), "
                          f"n_sig={s['n_significant']}")

                    # Generate per-pair plot
                    plot_path = get_output_path(f"cross_model_{dataset}_{pair_key}_{metric}_{method}_similarity.png")
                    plot_pair_comparison(comparison, pair_key, metric, method, plot_path)

    return results


# =============================================================================
# SYNTHESIS FUNCTIONS
# =============================================================================


def synthesize_cross_dataset(results: Dict, metrics: List[str], methods: List[str]) -> Dict:
    """Synthesize cross-dataset results across all models."""
    synthesis = {}

    for model, model_results in results.items():
        if "error" in model_results:
            continue

        synthesis[model] = {"by_metric": {}}

        for metric in metrics:
            synthesis[model]["by_metric"][metric] = {"by_method": {}}

            for method in methods:
                # Collect stats across all pairs for this model/metric/method
                all_mean_abs = []
                all_n_sig = []

                for pair_key, pair_data in model_results.items():
                    if metric not in pair_data:
                        continue
                    if method not in pair_data[metric]:
                        continue
                    if "summary" not in pair_data[metric][method]:
                        continue

                    s = pair_data[metric][method]["summary"]
                    all_mean_abs.append(s["mean_abs_cosine"])
                    all_n_sig.append(s["n_significant"])

                if all_mean_abs:
                    synthesis[model]["by_metric"][metric]["by_method"][method] = {
                        "mean_abs_cosine_across_pairs": float(np.mean(all_mean_abs)),
                        "std_abs_cosine_across_pairs": float(np.std(all_mean_abs)),
                        "mean_n_significant": float(np.mean(all_n_sig)),
                        "n_pairs": len(all_mean_abs),
                    }

    return synthesis


def synthesize_cross_model(results: Dict, metrics: List[str], methods: List[str]) -> Dict:
    """Synthesize cross-model results across all datasets."""
    synthesis = {}

    for dataset, dataset_results in results.items():
        if "error" in dataset_results:
            continue

        synthesis[dataset] = {"by_metric": {}, "skipped_pairs": []}

        for pair_key, pair_data in dataset_results.items():
            if pair_data.get("skipped"):
                synthesis[dataset]["skipped_pairs"].append({
                    "pair": pair_key,
                    "reason": pair_data.get("reason", "unknown"),
                })
                continue

        for metric in metrics:
            synthesis[dataset]["by_metric"][metric] = {"by_method": {}}

            for method in methods:
                all_mean_abs = []
                all_n_sig = []

                for pair_key, pair_data in dataset_results.items():
                    if pair_data.get("skipped"):
                        continue
                    if metric not in pair_data:
                        continue
                    if method not in pair_data[metric]:
                        continue
                    if "summary" not in pair_data[metric][method]:
                        continue

                    s = pair_data[metric][method]["summary"]
                    all_mean_abs.append(s["mean_abs_cosine"])
                    all_n_sig.append(s["n_significant"])

                if all_mean_abs:
                    synthesis[dataset]["by_metric"][metric]["by_method"][method] = {
                        "mean_abs_cosine_across_pairs": float(np.mean(all_mean_abs)),
                        "std_abs_cosine_across_pairs": float(np.std(all_mean_abs)),
                        "mean_n_significant": float(np.mean(all_n_sig)),
                        "n_pairs": len(all_mean_abs),
                    }

    return synthesis


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_pair_comparison(
    comparison_data: Dict,
    pair_name: str,
    metric: str,
    method: str,
    output_path: Path,
):
    """Plot single pair comparison: observed vs null across layers."""
    if "per_layer" not in comparison_data:
        return

    per_layer = comparison_data["per_layer"]
    layers = sorted([int(l) for l in per_layer.keys()])

    observed = np.array([per_layer[l]["observed_cosine"] for l in layers])
    observed_abs = np.array([per_layer[l]["abs_observed_cosine"] for l in layers])
    null_mean = np.array([per_layer[l]["null_mean"] for l in layers])
    null_5th = np.array([per_layer[l]["null_5th"] for l in layers])
    null_95th = np.array([per_layer[l]["null_95th"] for l in layers])
    p_values = [per_layer[l]["p_value"] for l in layers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Direction Similarity: {pair_name}\n{metric} / {method}", fontsize=12, fontweight='bold')

    # Left: Raw cosine similarity
    ax = axes[0]
    ax.fill_between(layers, null_5th, null_95th, alpha=0.3, color='gray', label='Null 5-95%')
    ax.plot(layers, null_mean, '--', color='gray', linewidth=1, label='Null mean')
    ax.plot(layers, observed, '-', color=METHOD_COLORS.get(method, 'tab:blue'),
            linewidth=LINE_WIDTH, label='Observed')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Cosine Similarity')
    ax.set_title('Raw Cosine Similarity')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)

    # Right: Absolute cosine with significance markers
    ax = axes[1]
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax.plot(layers, observed_abs, '-', color=METHOD_COLORS.get(method, 'tab:blue'),
            linewidth=LINE_WIDTH, label='|Observed|')

    # Mark significant layers
    sig_layers = [l for l, p in zip(layers, p_values) if p < 0.05]
    sig_vals = [observed_abs[layers.index(l)] for l in sig_layers]
    if sig_layers:
        ax.scatter(sig_layers, sig_vals, color='red', s=50, zorder=5,
                   label=f'p<0.05 (n={len(sig_layers)})', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Layer')
    ax.set_ylabel('|Cosine Similarity|')
    ax.set_title('Absolute Cosine (Significant Layers Marked)')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=GRID_ALPHA)

    save_figure(fig, output_path)


def plot_cross_dataset_synthesis(
    cross_dataset_results: Dict,
    model: str,
    metrics: List[str],
    methods: List[str],
    output_path: Path,
):
    """Plot synthesis heatmap for cross-dataset comparisons within a model."""
    pairs = [k for k in cross_dataset_results.keys() if not k.startswith("error")]
    if not pairs:
        return

    n_rows = len(metrics)
    n_cols = len(methods)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)
    fig.suptitle(f"Cross-Dataset Direction Similarity: {model}", fontsize=14, fontweight='bold')

    for row, metric in enumerate(metrics):
        for col, method in enumerate(methods):
            ax = axes[row, col]

            # Collect data for heatmap
            pair_names = []
            mean_abs_cosines = []

            for pair_key in pairs:
                if metric in cross_dataset_results[pair_key]:
                    if method in cross_dataset_results[pair_key][metric]:
                        data = cross_dataset_results[pair_key][metric][method]
                        if "summary" in data:
                            pair_names.append(pair_key.replace("_vs_", "\nvs\n"))
                            mean_abs_cosines.append(data["summary"]["mean_abs_cosine"])

            if not mean_abs_cosines:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                ax.set_title(f"{metric} / {method}")
                continue

            # Bar plot
            x = np.arange(len(pair_names))
            bars = ax.bar(x, mean_abs_cosines, color=METHOD_COLORS.get(method, 'tab:blue'), alpha=0.7)
            ax.set_xticks(x)
            ax.set_xticklabels(pair_names, fontsize=8)
            ax.set_ylabel('Mean |Cosine Similarity|')
            ax.set_title(f"{metric} / {method}")
            ax.set_ylim(0, 1)
            ax.grid(True, axis='y', alpha=GRID_ALPHA)

            # Add value labels
            for bar, val in zip(bars, mean_abs_cosines):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    save_figure(fig, output_path)


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================


def print_summary(results: Dict):
    """Print comprehensive summary to console."""
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Handle cross-dataset mode
    cross_dataset = results.get("cross_dataset", {})
    if cross_dataset and "error" not in cross_dataset:
        print(f"\nMode: Cross-Dataset")
        print(f"Model: {results.get('model_short', 'unknown')}")
        print(f"Datasets: {', '.join(results.get('datasets', []))}")
        print()

        for pair_key, pair_data in cross_dataset.items():
            print(f"{pair_key}:")
            for metric in METRICS:
                if metric not in pair_data:
                    continue
                for method in METHODS:
                    if method not in pair_data[metric]:
                        continue
                    if "summary" not in pair_data[metric][method]:
                        continue
                    s = pair_data[metric][method]["summary"]
                    print(f"  {metric}/{method}: "
                          f"mean |cos|={s['mean_abs_cosine']:.3f}, "
                          f"n_sig={s['n_significant']}/{s['n_layers']}")
        return

    # Handle cross-model mode
    cross_model = results.get("cross_model", {})
    if cross_model and "error" not in cross_model:
        print(f"\nMode: Cross-Model")
        print(f"Dataset: {results.get('dataset', 'unknown')}")
        print(f"Models: {', '.join(results.get('model_configs', []))}")
        print()

        for pair_key, pair_data in cross_model.items():
            if pair_data.get("skipped"):
                print(f"{pair_key}: SKIPPED ({pair_data.get('reason', 'unknown')})")
                continue

            print(f"{pair_key}:")
            for metric in METRICS:
                if metric not in pair_data:
                    continue
                for method in METHODS:
                    if method not in pair_data[metric]:
                        continue
                    if "summary" not in pair_data[metric][method]:
                        continue
                    s = pair_data[metric][method]["summary"]
                    print(f"  {metric}/{method}: "
                          f"mean |cos|={s['mean_abs_cosine']:.3f}, "
                          f"n_sig={s['n_significant']}/{s['n_layers']}")
        return

    print("No results to summarize")


# =============================================================================
# MAIN
# =============================================================================


def main():

    print("=" * 70)
    print("DIRECTION SIMILARITY ANALYSIS")
    print("=" * 70)
    print(f"Metrics: {METRICS}")
    print(f"Methods: {METHODS}")
    print(f"N permutations: {N_PERMUTATIONS}")
    print()

    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj

    # Determine mode
    cross_model_mode = CROSS_MODEL_DATASET is not None and len(CROSS_MODEL_CONFIGS) >= 2

    if cross_model_mode:
        # ===== CROSS-MODEL MODE =====
        print("Mode: CROSS-MODEL")
        print(f"Dataset: {CROSS_MODEL_DATASET}")
        print(f"Models ({len(CROSS_MODEL_CONFIGS)}):")
        for model, adapter, load_4bit, load_8bit in CROSS_MODEL_CONFIGS:
            model_dir = get_model_dir_for_config(model, adapter, load_4bit, load_8bit)
            print(f"  - {model_dir}")
        print()

        print("=" * 70)
        print("CROSS-MODEL ANALYSIS")
        print("=" * 70)

        cross_model_results = run_cross_model_analysis(
            CROSS_MODEL_DATASET,
            CROSS_MODEL_CONFIGS,
            METRICS, METHODS, N_PERMUTATIONS, SEED
        )

        if "error" in cross_model_results:
            print(f"Error: {cross_model_results['error']}")
            return

        # Save results
        results = {
            "config": get_config_dict(
                mode="cross_model",
                dataset=CROSS_MODEL_DATASET,
                model_configs=[
                    {"model": m, "adapter": a, "load_in_4bit": b4, "load_in_8bit": b8}
                    for m, a, b4, b8 in CROSS_MODEL_CONFIGS
                ],
                metrics=METRICS,
                methods=METHODS,
                n_permutations=N_PERMUTATIONS,
                seed=SEED,
                probe_alpha=PROBE_ALPHA,
                probe_pca_components=PROBE_PCA_COMPONENTS,
                mean_diff_quantile=MEAN_DIFF_QUANTILE,
            ),
            "dataset": CROSS_MODEL_DATASET,
            "model_configs": [
                get_model_dir_for_config(m, a, b4, b8)
                for m, a, b4, b8 in CROSS_MODEL_CONFIGS
            ],
            "cross_model": cross_model_results,
        }

        # Cross-model results go to global results (no specific model_dir)
        results_path = get_output_path(f"cross_model_{CROSS_MODEL_DATASET}_direction_similarity.json")

    else:
        # ===== CROSS-DATASET MODE =====
        model_dir = get_model_dir()

        print("Mode: CROSS-DATASET")
        print(f"Model: {MODEL}")
        if ADAPTER:
            print(f"Adapter: {ADAPTER}")
        if LOAD_IN_4BIT:
            print("Quantization: 4-bit")
        elif LOAD_IN_8BIT:
            print("Quantization: 8-bit")
        print(f"Model dir: {model_dir}")
        print()

        # Discover available datasets for this model
        print("Discovering available datasets...")
        datasets = discover_datasets()

        if not datasets:
            print(f"No datasets found for {model_dir} in outputs/")
            return

        print(f"  Found {len(datasets)} datasets: {', '.join(datasets)}")

        if len(datasets) < 2:
            print("\nNeed at least 2 datasets for cross-dataset comparison")
            return

        print("\n" + "=" * 70)
        print("CROSS-DATASET ANALYSIS")
        print("=" * 70)

        cross_dataset_results = run_cross_dataset_analysis(
            datasets, METRICS, METHODS, N_PERMUTATIONS, SEED, model_dir=model_dir
        )

        if "error" in cross_dataset_results:
            print(f"Error: {cross_dataset_results['error']}")
            return

        # Generate synthesis plot
        plot_path = get_output_path("cross_dataset_similarity.png", model_dir=model_dir)
        plot_cross_dataset_synthesis(
            cross_dataset_results, model_dir, METRICS, METHODS, plot_path
        )

        # Save results
        results = {
            "config": get_config_dict(
                mode="cross_dataset",
                model=MODEL,
                adapter=ADAPTER,
                load_in_4bit=LOAD_IN_4BIT,
                load_in_8bit=LOAD_IN_8BIT,
                metrics=METRICS,
                methods=METHODS,
                n_permutations=N_PERMUTATIONS,
                seed=SEED,
                probe_alpha=PROBE_ALPHA,
                probe_pca_components=PROBE_PCA_COMPONENTS,
                mean_diff_quantile=MEAN_DIFF_QUANTILE,
            ),
            "model_dir": model_dir,
            "datasets": datasets,
            "cross_dataset": cross_dataset_results,
        }

        results_path = get_output_path("direction_similarity.json", model_dir=model_dir)

    print(f"\nSaving results to {results_path}...")
    with open(results_path, "w") as f:
        json.dump(convert_for_json(results), f, indent=2)

    # Print summary
    print_summary(results)

    print("\n" + "=" * 70)
    print("Done.")
    print(f"Output: {results_path}")


if __name__ == "__main__":
    main()
