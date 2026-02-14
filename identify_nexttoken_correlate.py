"""
Stage 1 (alternative). Find next-token uncertainty directions and output token directions
from model activations during diverse text prediction.

Tests whether activations encode uncertainty metrics (entropy, logit_gap, etc.) using
both probe and mean_diff direction-finding methods. Optionally also finds output token
directions that predict which token the model selected (parallel to MC answer directions).

Inputs:
    outputs/{model_dir}/working/nexttoken_entropy_dataset.json  Stratified next-token dataset
                                                                 (produced by build_nexttoken_dataset.py)

Outputs:
    outputs/{model_dir}/results/nexttoken_results.json           Consolidated results
    outputs/{model_dir}/results/nexttoken_{metric}_results.png   R² curves per metric
    outputs/{model_dir}/results/nexttoken_entropy_distribution.png  Entropy distribution
    outputs/{model_dir}/working/nexttoken_activations.npz        Cached activations
    outputs/{model_dir}/working/nexttoken_{metric}_directions.npz Direction vectors

    where {model_dir} = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)

Shared parameters (must match across scripts):
    SEED, PROBE_ALPHA, PROBE_PCA_COMPONENTS, TRAIN_SPLIT, MEAN_DIFF_QUANTILE

Run after: build_nexttoken_dataset.py
"""

from pathlib import Path
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, Tuple
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from core import (
    load_model_and_tokenizer,
    get_model_short_name,
    get_model_dir_name,
    compute_nexttoken_metrics,
    find_directions,
    METRIC_INFO,
    DEVICE,
    setup_run_logger,
    print_run_header,
    print_key_findings,
    print_run_footer,
    format_best_layer,
)
from core.config_utils import get_config_dict, get_output_path, find_output_file
from core.plotting import save_figure, METHOD_COLORS, GRID_ALPHA, CI_ALPHA

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter
DATASET_PATH = None  # Auto-detect based on model, or set explicitly
METRICS = ["entropy", "logit_gap"]  # Which metrics to analyze

# --- Quantization ---
LOAD_IN_4BIT = False  # Set True for 70B+ models
LOAD_IN_8BIT = False

# --- Experiment ---
SEED = 42                    # Must match across scripts
BATCH_SIZE = 8
MAX_PROMPT_LENGTH = 500
N_BOOTSTRAP = 100            # Bootstrap iterations for confidence intervals
N_JOBS = -1                  # Parallel jobs: -1 = all cores, 1 = sequential
CHECKPOINT_INTERVAL = 200    # Checkpointing for large datasets

# --- Direction-finding (must match across scripts) ---
PROBE_ALPHA = 1000.0         # Must match across scripts
PROBE_PCA_COMPONENTS = 100   # Must match across scripts
TRAIN_SPLIT = 0.8            # Must match across scripts
MEAN_DIFF_QUANTILE = 0.25    # Must match across scripts

# --- Output Token Directions ---
FIND_OUTPUT_TOKEN_DIRECTIONS = True  # Find directions predicting output token (parallel to MC answer directions)

# --- Output ---
# Paths are constructed via get_output_path() which routes to outputs/results/ or outputs/working/

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_entropy_distribution(
    entropies: np.ndarray,
    output_path: Path
):
    """Plot entropy distribution for next-token prediction."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Overall entropy histogram (percentage)
    ax1 = axes[0]
    ax1.hist(entropies, bins=30, edgecolor='black', alpha=0.7,
             weights=np.ones(len(entropies)) / len(entropies) * 100)
    ax1.axvline(entropies.mean(), color='red', linestyle='--',
                label=f'Mean: {entropies.mean():.3f}')
    ax1.axvline(np.median(entropies), color='orange', linestyle='--',
                label=f'Median: {np.median(entropies):.3f}')
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Percentage')
    ax1.set_title(f'Next-Token Entropy Distribution (n={len(entropies)})')
    ax1.legend()
    ax1.grid(True, alpha=GRID_ALPHA)

    # 2. CDF
    ax2 = axes[1]
    sorted_ent = np.sort(entropies)
    cdf = np.arange(1, len(sorted_ent) + 1) / len(sorted_ent)
    ax2.plot(sorted_ent, cdf, linewidth=2)
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Entropy CDF')
    ax2.grid(True, alpha=GRID_ALPHA)
    # Mark quartiles
    for q, label in [(0.25, 'Q1'), (0.5, 'Median'), (0.75, 'Q3')]:
        val = np.percentile(entropies, q * 100)
        ax2.axhline(q, color='gray', linestyle=':', alpha=0.5)
        ax2.axvline(val, color='gray', linestyle=':', alpha=0.5)

    save_figure(fig, output_path)


def plot_results(
    all_results: dict,
    metric: str,
    output_path: Path
):
    """Plot R² across layers for both methods with confidence intervals."""
    results = all_results[metric]
    layers = sorted(results["fits"]["probe"].keys())

    # Single panel - just the R² curves
    fig, ax = plt.subplots(figsize=(10, 5))

    for method in ["probe", "mean_diff"]:
        fits = results["fits"][method]
        color = METHOD_COLORS.get(method, "tab:gray")
        r2_values = [fits[l]["r2"] for l in layers]

        # Check for std (bootstrap)
        has_std = "r2_std" in fits[layers[0]]
        if has_std:
            r2_std = [fits[l]["r2_std"] for l in layers]
            ax.fill_between(
                layers,
                np.array(r2_values) - np.array(r2_std),
                np.array(r2_values) + np.array(r2_std),
                alpha=CI_ALPHA, color=color
            )

        # Find best layer for this method
        best_layer = max(layers, key=lambda l: fits[l]["r2"])
        best_r2 = fits[best_layer]["r2"]
        label = f'{method} (best: L{best_layer}, R²={best_r2:.3f})'

        ax.plot(layers, r2_values, 'o-', label=label, color=color, markersize=4)

    ax.set_xlabel('Layer Index')
    ax.set_ylabel('R² Score')
    ax.set_title(f'{metric} Predictability by Layer (Next-Token)')
    ax.legend()
    ax.grid(True, alpha=GRID_ALPHA)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    save_figure(fig, output_path)


def log_diagnostic_summary(logger, metrics_dict: dict, all_results: dict):
    """Log diagnostic summary to file."""
    logger.section("Diagnostic Summary")

    # Distribution stats for each metric
    for metric_name, values in metrics_dict.items():
        variance = values.var()
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        logger.info(f"{metric_name.upper()} Distribution:")
        logger.info(f"  Mean: {values.mean():.3f}, Std: {values.std():.3f}, Variance: {variance:.4f}")
        logger.info(f"  Median: {np.median(values):.3f}, IQR: {iqr:.3f}")
        logger.info(f"  Range: [{values.min():.3f}, {values.max():.3f}]")

    # Early vs late layer comparison
    logger.info(f"\nLayer R² Comparison (early vs late):")
    for metric_name, results in all_results.items():
        layers = sorted(results["fits"]["probe"].keys())
        n_layers = len(layers)
        early_layers = layers[:n_layers // 4]
        late_layers = layers[3 * n_layers // 4:]

        for method in ["probe", "mean_diff"]:
            fits = results["fits"][method]
            early_r2 = np.mean([fits[l]["r2"] for l in early_layers])
            late_r2 = np.mean([fits[l]["r2"] for l in late_layers])
            r2_increase = late_r2 - early_r2
            logger.info(f"  {metric_name}/{method}: early={early_r2:.3f}, late={late_r2:.3f}, delta={r2_increase:+.3f}")


# =============================================================================
# TOKEN PREDICTION FUNCTIONS
# =============================================================================


def top_k_accuracy(proba: np.ndarray, y_true: np.ndarray, k: int, classes: np.ndarray) -> float:
    """Compute top-k accuracy: fraction where true label is in top-k predictions."""
    top_k_indices = np.argsort(proba, axis=1)[:, -k:]
    top_k_classes = classes[top_k_indices]
    correct = np.any(top_k_classes == y_true[:, None], axis=1)
    return float(np.mean(correct))


def _train_token_probe_for_layer(
    layer_idx: int,
    X: np.ndarray,
    tokens: np.ndarray,
    train_split: float,
    seed: int,
    use_pca: bool,
    pca_components: int,
) -> Tuple[int, Dict]:
    """
    Train logistic regression probe to predict output token.

    Returns:
        (layer_idx, results_dict) where results_dict contains accuracy metrics.
    """
    rng = np.random.RandomState(seed)
    n = len(tokens)

    indices = np.arange(n)
    rng.shuffle(indices)
    split_idx = int(n * train_split)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = tokens[train_idx], tokens[test_idx]

    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Optional PCA
    if use_pca:
        n_components = min(pca_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
        pca = PCA(n_components=n_components, svd_solver='randomized', random_state=seed)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
    else:
        X_train_pca = X_train_scaled
        X_test_pca = X_test_scaled

    # Train classifier
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver='lbfgs')
    clf.fit(X_train_pca, y_train)

    top1_acc = clf.score(X_test_pca, y_test)
    proba = clf.predict_proba(X_test_pca)
    top5_acc = top_k_accuracy(proba, y_test, k=5, classes=clf.classes_)
    top10_acc = top_k_accuracy(proba, y_test, k=10, classes=clf.classes_)

    return layer_idx, {
        "top1_accuracy": float(top1_acc),
        "top5_accuracy": float(top5_acc),
        "top10_accuracy": float(top10_acc),
        "n_classes": len(clf.classes_),
        "n_train": len(y_train),
        "n_test": len(y_test),
    }


def run_token_prediction_probe(
    activations: Dict[int, np.ndarray],
    predicted_tokens: np.ndarray,
    train_split: float = TRAIN_SPLIT,
    seed: int = SEED,
    use_pca: bool = True,
    pca_components: int = PROBE_PCA_COMPONENTS,
    n_jobs: int = N_JOBS,
) -> Dict[int, Dict]:
    """
    Train logistic regression probes to predict output token at each layer.

    Returns:
        {layer_idx: {"top1_accuracy": ..., "top5_accuracy": ..., "top10_accuracy": ..., ...}}
    """
    layer_indices = sorted(activations.keys())

    results_list = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(_train_token_probe_for_layer)(
            layer_idx,
            activations[layer_idx],
            predicted_tokens,
            train_split,
            seed,
            use_pca,
            pca_components,
        )
        for layer_idx in tqdm(layer_indices, desc="Token probes")
    )
    return {layer_idx: result for layer_idx, result in results_list}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def find_dataset_path(model_name: str, model_dir: str) -> Path:
    """Find the stratified dataset file."""
    if DATASET_PATH:
        return Path(DATASET_PATH)

    # Try model-dir path first (new structure)
    model_dir_path = find_output_file("nexttoken_entropy_dataset.json", model_dir=model_dir)
    if model_dir_path.exists():
        return model_dir_path

    # Fall back to generic (shared dataset)
    generic = find_output_file("entropy_dataset.json")
    if generic.exists():
        return generic

    raise FileNotFoundError(
        f"Could not find dataset. Tried: {model_dir_path}, {generic}. "
        "Run build_nexttoken_dataset.py first."
    )


def load_dataset(path: Path, logger=None) -> list:
    """Load the stratified entropy dataset."""
    if logger:
        logger.info(f"Loading dataset from {path}...")
    with open(path) as f:
        raw = json.load(f)

    # Handle both formats
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
        config = raw.get("config")
        if config and logger:
            logger.info(f"  Config: {config}")
    else:
        data = raw

    if logger:
        logger.info(f"  Loaded {len(data)} samples")
    return data


def extract_activations_and_metrics(
    dataset: list,
    model,
    tokenizer,
    num_layers: int,
    checkpoint_path: Path
) -> tuple:
    """
    Extract activations and compute metrics for all samples.

    Returns:
        activations_by_layer: {layer: (n_samples, hidden_dim)}
        metrics_dict: {metric_name: (n_samples,)}
        predicted_tokens: (n_samples,)
    """
    # Check for checkpoint
    start_idx = 0
    all_activations = {i: [] for i in range(num_layers)}
    all_metrics = {m: [] for m in ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]}
    all_predicted_tokens = []

    if checkpoint_path.exists():
        tqdm.write(f"Loading checkpoint from {checkpoint_path}...")
        ckpt = np.load(checkpoint_path, allow_pickle=True)
        if "processed_count" in ckpt.files:
            start_idx = int(ckpt["processed_count"])
            for i in range(num_layers):
                key = f"layer_{i}"
                if key in ckpt.files:
                    all_activations[i] = list(ckpt[key])
            for m in all_metrics:
                if m in ckpt.files:
                    all_metrics[m] = list(ckpt[m])
            if "predicted_tokens" in ckpt.files:
                all_predicted_tokens = list(ckpt["predicted_tokens"])
            tqdm.write(f"  Resuming from sample {start_idx}")

    # Set up hooks
    activations_cache = {}
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations_cache[layer_idx] = hidden[:, -1, :].detach()
        return hook

    if hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        layers = model.model.layers

    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(make_hook(i)))

    model.eval()
    total = len(dataset)

    try:
        for batch_start in tqdm(range(start_idx, total, BATCH_SIZE)):
            batch = dataset[batch_start:batch_start + BATCH_SIZE]
            texts = [item["text"] for item in batch]

            # Tokenize
            encoded = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LENGTH,
                add_special_tokens=False
            )
            input_ids = encoded["input_ids"].to(model.device)
            attention_mask = encoded["attention_mask"].to(model.device)

            activations_cache.clear()

            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # Get logits at last position
            batch_logits = outputs.logits[:, -1, :].cpu().numpy()

            for b in range(len(batch)):
                logits = batch_logits[b]

                # Compute metrics
                metrics = compute_nexttoken_metrics(logits)
                for m, v in metrics.items():
                    all_metrics[m].append(v)

                all_predicted_tokens.append(int(np.argmax(logits)))

            # Store activations
            for layer_idx, acts in activations_cache.items():
                all_activations[layer_idx].extend(acts.cpu().numpy())

            # Checkpoint
            processed = batch_start + len(batch)
            if processed % CHECKPOINT_INTERVAL < BATCH_SIZE and processed < total:
                save_dict = {f"layer_{i}": np.array(a) for i, a in all_activations.items()}
                for m, v in all_metrics.items():
                    save_dict[m] = np.array(v)
                save_dict["predicted_tokens"] = np.array(all_predicted_tokens)
                save_dict["processed_count"] = processed
                np.savez_compressed(checkpoint_path, **save_dict)
                tqdm.write(f"  Checkpoint: {processed}/{total}")

            # Memory cleanup
            del encoded, input_ids, attention_mask, outputs
            if batch_start % 50 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

    finally:
        for h in hooks:
            h.remove()

    # Convert to arrays
    activations_by_layer = {i: np.array(a) for i, a in all_activations.items()}
    metrics_dict = {m: np.array(v) for m, v in all_metrics.items()}
    predicted_tokens = np.array(all_predicted_tokens)

    # Cleanup checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        tqdm.write("  Removed checkpoint")

    return activations_by_layer, metrics_dict, predicted_tokens


# =============================================================================
# MAIN
# =============================================================================


def main():
    # Model directory for organizing outputs
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = "nexttoken"  # Model prefix now in directory, not filename

    checkpoint_path = get_output_path(f"{base_name}_checkpoint.npz", model_dir=model_dir)

    # Setup logging
    logger = setup_run_logger(None, base_name, "run", model_dir=model_dir)
    config = {
        "model": MODEL,
        "adapter": ADAPTER,
        "metrics": METRICS,
    }
    print_run_header("identify_nexttoken_correlate.py", 1, "Find next-token uncertainty directions", config)

    logger.info(f"Model: {MODEL}")
    if ADAPTER:
        logger.info(f"Adapter: {ADAPTER}")
    logger.info(f"Metrics: {METRICS}")
    logger.info(f"Bootstrap iterations: {N_BOOTSTRAP}")
    logger.info(f"Model dir: {model_dir}")
    logger.info(f"Output base: {base_name}")

    # Find dataset
    dataset_path = find_dataset_path(MODEL, model_dir)

    # Load dataset
    dataset = load_dataset(dataset_path, logger)

    # Check if activations already exist (resume from crash)
    # Check both new and legacy locations for migration compatibility
    activations_read_path = find_output_file(f"{base_name}_activations.npz", model_dir=model_dir)
    activations_path = get_output_path(f"{base_name}_activations.npz", model_dir=model_dir)  # For writing
    if activations_read_path.exists():
        logger.info(f"Found existing activations: {activations_read_path}")
        logger.info("Loading from file (skipping model load and extraction)...")
        loaded = np.load(activations_read_path)

        # Reconstruct activations_by_layer
        layer_keys = [k for k in loaded.files if k.startswith("layer_")]
        num_layers = len(layer_keys)
        activations_by_layer = {i: loaded[f"layer_{i}"] for i in range(num_layers)}

        # Reconstruct metrics_dict
        metrics_dict = {}
        for m in METRICS:
            if m in loaded.files:
                metrics_dict[m] = loaded[m]

        # Get predicted tokens
        predicted_tokens = loaded["predicted_tokens"] if "predicted_tokens" in loaded.files else None

        logger.info(f"  Loaded {num_layers} layers, {len(metrics_dict)} metrics")
    else:
        # Load model
        logger.info("Loading model...")
        model, tokenizer, num_layers = load_model_and_tokenizer(
            MODEL,
            adapter_path=ADAPTER,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT
        )
        logger.info(f"  Layers: {num_layers}")
        logger.info(f"  Device: {DEVICE}")

        # Extract activations and metrics
        logger.info(f"Extracting activations (batch_size={BATCH_SIZE})...")
        activations_by_layer, metrics_dict, predicted_tokens = extract_activations_and_metrics(
            dataset, model, tokenizer, num_layers, checkpoint_path
        )

        logger.info(f"Activations shape: {activations_by_layer[0].shape}")
        logger.info("Metric statistics:")
        for m, v in metrics_dict.items():
            logger.info(f"  {m}: mean={v.mean():.3f}, std={v.std():.3f}, range=[{v.min():.3f}, {v.max():.3f}]")

        # Save activations file
        logger.info(f"Saving activations to {activations_path}...")
        act_save = {f"layer_{i}": activations_by_layer[i] for i in range(num_layers)}
        for m_name, m_values in metrics_dict.items():
            act_save[m_name] = m_values
        act_save["predicted_tokens"] = predicted_tokens
        np.savez_compressed(activations_path, **act_save)

    # Build dataset stats for consolidated JSON
    dataset_stats = {}
    for m_name, m_values in metrics_dict.items():
        dataset_stats[f"{m_name}_mean"] = float(m_values.mean())
        dataset_stats[f"{m_name}_std"] = float(m_values.std())

    # Plot entropy distribution
    if "entropy" in metrics_dict:
        entropy_plot_path = get_output_path(f"{base_name}_entropy_distribution.png", model_dir=model_dir)
        logger.info("Plotting entropy distribution...")
        plot_entropy_distribution(metrics_dict["entropy"], entropy_plot_path)

    # Find directions for each metric
    logger.section("FINDING DIRECTIONS")

    metrics_to_analyze = {m: metrics_dict[m] for m in METRICS}
    all_results = {}
    key_findings = {}
    output_files = []

    for metric in METRICS:
        logger.info(f"\n--- {metric.upper()} ({N_BOOTSTRAP} bootstrap iterations) ---")
        target_values = metrics_to_analyze[metric]

        results = find_directions(
            activations_by_layer,
            target_values,
            methods=["probe", "mean_diff"],
            probe_alpha=PROBE_ALPHA,
            probe_pca_components=PROBE_PCA_COMPONENTS,
            probe_n_bootstrap=N_BOOTSTRAP,
            probe_train_split=TRAIN_SPLIT,
            mean_diff_quantile=MEAN_DIFF_QUANTILE,
            seed=SEED,
            n_jobs=N_JOBS,
            return_scaler=True,  # Save scaler info for transfer tests
        )

        all_results[metric] = results

        # Log and collect summary per method
        for method in ["probe", "mean_diff"]:
            fits = results["fits"][method]
            best_layer = max(fits.keys(), key=lambda l: fits[l]["r2"])
            best_r2 = fits[best_layer]["r2"]
            best_corr = fits[best_layer]["corr"]

            ci_low = fits[best_layer].get("r2_ci_low")
            ci_high = fits[best_layer].get("r2_ci_high")
            if ci_low is None and "r2_std" in fits[best_layer]:
                ci_low = best_r2 - 1.96 * fits[best_layer]['r2_std']
                ci_high = best_r2 + 1.96 * fits[best_layer]['r2_std']

            avg_r2 = np.mean([f["r2"] for f in fits.values()])

            logger.info(f"  {method:12s}: best layer={best_layer:2d} (R²={best_r2:.3f}, r={best_corr:.3f}), avg R²={avg_r2:.3f}")
            key_findings[f"{metric}/{method}"] = format_best_layer(method, best_layer, best_r2, ci_low, ci_high)

        # Method comparison
        if results["comparison"]:
            mid_layer = num_layers // 2
            cos_sim = results["comparison"][mid_layer]["cosine_sim"]
            logger.info(f"  probe vs mean_diff cosine similarity (layer {mid_layer}): {cos_sim:.3f}")

        # Save directions file for this metric
        directions_path = get_output_path(f"{base_name}_{metric}_directions.npz", model_dir=model_dir)
        dir_save = {
            "_metadata_dataset": str(dataset_path),
            "_metadata_model": MODEL,
            "_metadata_metric": metric,
        }
        for method in ["probe", "mean_diff"]:
            for layer in range(num_layers):
                dir_save[f"{method}_layer_{layer}"] = results["directions"][method][layer]
                # Save scaler info for probe method (for centered transfer)
                if method == "probe" and "scaler_scale" in results["fits"][method][layer]:
                    dir_save[f"{method}_scaler_scale_{layer}"] = results["fits"][method][layer]["scaler_scale"]
                    dir_save[f"{method}_scaler_mean_{layer}"] = results["fits"][method][layer]["scaler_mean"]
        np.savez(directions_path, **dir_save)
        output_files.append(directions_path)
        logger.info(f"  Saved directions: {directions_path.name}")

        # Plot results for this metric
        plot_path = get_output_path(f"{base_name}_{metric}_results.png", model_dir=model_dir)
        plot_results(all_results, metric, plot_path)
        output_files.append(plot_path)

    # Diagnostic summary to log file
    log_diagnostic_summary(logger, metrics_to_analyze, all_results)

    # =========================================================================
    # OUTPUT TOKEN DIRECTIONS (parallel to MC answer directions)
    # =========================================================================

    token_results_data = None
    if FIND_OUTPUT_TOKEN_DIRECTIONS and predicted_tokens is not None:
        logger.section("OUTPUT TOKEN DIRECTIONS")

        token_results = run_token_prediction_probe(
            activations_by_layer,
            predicted_tokens,
            train_split=TRAIN_SPLIT,
            seed=SEED,
            use_pca=True,
            pca_components=PROBE_PCA_COMPONENTS,
            n_jobs=N_JOBS,
        )

        # Find best layer
        best_layer = max(token_results.keys(), key=lambda l: token_results[l]["top1_accuracy"])
        best_top1 = token_results[best_layer]["top1_accuracy"]
        best_top5 = token_results[best_layer]["top5_accuracy"]
        best_top10 = token_results[best_layer]["top10_accuracy"]
        n_classes = token_results[best_layer]["n_classes"]

        logger.info(f"TOKEN PREDICTION RESULTS:")
        logger.info(f"  Best layer: {best_layer}")
        logger.info(f"  Top-1 accuracy: {best_top1:.3f} ({best_top1*100:.1f}%)")
        logger.info(f"  Top-5 accuracy: {best_top5:.3f} ({best_top5*100:.1f}%)")
        logger.info(f"  Top-10 accuracy: {best_top10:.3f} ({best_top10*100:.1f}%)")
        logger.info(f"  Number of unique tokens: {n_classes}")

        # Layerwise summary to log
        logger.table(
            ["Layer", "Top-1", "Top-5", "Top-10"],
            [[layer, f"{r['top1_accuracy']:.3f}", f"{r['top5_accuracy']:.3f}", f"{r['top10_accuracy']:.3f}"]
             for layer, r in sorted(token_results.items())],
            "Token Prediction by Layer"
        )

        # Add to key findings
        key_findings["token/probe"] = f"Top-1={best_top1:.3f}, Top-5={best_top5:.3f} at layer {best_layer}"

        # Store for consolidated JSON
        token_results_data = {
            "summary": {
                "best_layer": int(best_layer),
                "best_top1_accuracy": float(best_top1),
                "best_top5_accuracy": float(best_top5),
                "best_top10_accuracy": float(best_top10),
                "n_classes": int(n_classes),
            },
            "per_layer": {int(l): r for l, r in token_results.items()},
        }

    # =========================================================================
    # SAVE CONSOLIDATED RESULTS JSON
    # =========================================================================

    results_path = get_output_path(f"{base_name}_results.json", model_dir=model_dir)
    consolidated_results = {
        "format_version": 2,
        "config": get_config_dict(
            model=MODEL,
            adapter=ADAPTER,
            dataset_path=str(dataset_path),
            num_samples=len(dataset),
            metrics=METRICS,
            seed=SEED,
            train_split=TRAIN_SPLIT,
            probe_alpha=PROBE_ALPHA,
            pca_components=PROBE_PCA_COMPONENTS,
            n_bootstrap=N_BOOTSTRAP,
            mean_diff_quantile=MEAN_DIFF_QUANTILE,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "dataset": {
            "num_samples": len(dataset),
            "stats": dataset_stats,
        },
        "metrics": {},
    }

    # Add per-metric results
    for metric in METRICS:
        target_values = metrics_to_analyze[metric]
        results = all_results[metric]

        metric_section = {
            "stats": {
                "mean": float(target_values.mean()),
                "std": float(target_values.std()),
                "min": float(target_values.min()),
                "max": float(target_values.max()),
                "variance": float(target_values.var()),
                "median": float(np.median(target_values)),
                "iqr": float(np.percentile(target_values, 75) - np.percentile(target_values, 25)),
            },
            "results": {},
        }

        for method in ["probe", "mean_diff"]:
            method_results = {}
            for layer in range(num_layers):
                layer_info = {}
                for k, v in results["fits"][method][layer].items():
                    # Skip numpy arrays (scaler_scale, scaler_mean) - those go in .npz
                    if isinstance(v, np.ndarray):
                        continue
                    # Convert numpy scalars to Python types
                    if isinstance(v, np.floating):
                        layer_info[k] = float(v)
                    elif isinstance(v, np.integer):
                        layer_info[k] = int(v)
                    else:
                        layer_info[k] = v
                method_results[layer] = layer_info
            metric_section["results"][method] = method_results

        consolidated_results["metrics"][metric] = metric_section

    # Add token results if computed
    if token_results_data is not None:
        consolidated_results["token"] = token_results_data

    with open(results_path, "w") as f:
        json.dump(consolidated_results, f, indent=2)
    output_files.append(results_path)
    logger.info(f"Saved consolidated results: {results_path.name}")

    # Add other output files
    output_files.append(activations_path)
    if "entropy" in metrics_dict:
        output_files.append(get_output_path(f"{base_name}_entropy_distribution.png", model_dir=model_dir))

    # Print minimal console output
    print_key_findings(key_findings)
    print_run_footer(output_files, logger.log_file)


if __name__ == "__main__":
    main()
