"""
Run entropy prediction experiment on diverse text.

This script:
1. Loads the stratified dataset from build_nexttoken_dataset.py
2. Extracts activations from all layers of the model
3. Trains linear probes to predict entropy from each layer
4. Evaluates and visualizes results

Supports resuming from checkpoint if interrupted during extraction.

Usage:
    python nexttoken_entropy_probe.py             # Full run
    python nexttoken_entropy_probe.py --plot-only # Load saved activations, retrain probes, plot
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_model_short_name,
    LinearProbe,
    extract_activations_only,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_PATH = None  # Auto-detect based on model name, or set explicitly
MAX_PROMPT_LENGTH = 500
SEED = 42
CHECKPOINT_INTERVAL = 200  # Save checkpoint every N prompts

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100
N_BOOTSTRAP = 100  # Number of bootstrap iterations for confidence intervals

np.random.seed(SEED)
torch.manual_seed(SEED)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_nexttoken")
    return str(OUTPUTS_DIR / f"{model_short}_nexttoken")


def find_dataset_path() -> Path:
    """Find the dataset file, auto-detecting based on model name."""
    if DATASET_PATH:
        return Path(DATASET_PATH)

    # Try model-specific path in outputs directory first
    model_short = get_model_short_name(BASE_MODEL_NAME)
    model_specific = OUTPUTS_DIR / f"{model_short}_nexttoken_entropy_dataset.json"
    if model_specific.exists():
        return model_specific

    # Fall back to generic name in outputs
    generic = OUTPUTS_DIR / "entropy_dataset.json"
    if generic.exists():
        return generic

    raise FileNotFoundError(
        f"Could not find dataset. Tried: {model_specific}, {generic}. "
        "Run build_nexttoken_dataset.py first."
    )


def load_dataset(path: Path) -> Tuple[List[Dict], Optional[Dict]]:
    """
    Load the entropy dataset.

    Returns (data, config) where config may be None for old-format files.
    """
    print(f"Loading dataset from {path}...")
    with open(path) as f:
        raw = json.load(f)

    # Handle both old format (list) and new format (dict with config)
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
        config = raw.get("config")
    else:
        data = raw
        config = None

    print(f"Loaded {len(data)} prompts")
    if config:
        print(f"  Dataset config: {config}")

    return data, config


def extract_all_activations(
    dataset: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    checkpoint_path: Path
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Extract activations from all layers for all prompts.
    Supports resuming from checkpoint.
    """
    # Check for existing checkpoint
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = np.load(checkpoint_path, allow_pickle=True)

        # Handle both old and new checkpoint formats
        if "processed_count" in checkpoint.files:
            start_idx = int(checkpoint["processed_count"])
            all_layer_activations = {
                int(k.split("_")[1]): list(checkpoint[k])
                for k in checkpoint.files if k.startswith("layer_")
            }
            all_entropies = list(checkpoint["entropies"])
            print(f"Resuming from prompt {start_idx}/{len(dataset)}")
        else:
            # Old checkpoint format without processed_count - start fresh
            print("Warning: Old checkpoint format detected, starting fresh extraction")
            checkpoint_path.unlink()  # Remove incompatible checkpoint
            start_idx = 0
            all_layer_activations = {i: [] for i in range(num_layers)}
            all_entropies = []
    else:
        start_idx = 0
        all_layer_activations = {i: [] for i in range(num_layers)}
        all_entropies = []

    print(f"Extracting activations from {num_layers} layers...")
    model.eval()

    for i in tqdm(range(start_idx, len(dataset))):
        item = dataset[i]
        text = item["text"]
        entropy = item["entropy"]

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH
        )
        input_ids = inputs["input_ids"].to(DEVICE)
        attention_mask = inputs["attention_mask"].to(DEVICE)

        # Extract activations
        layer_acts = extract_activations_only(
            model, input_ids, attention_mask, num_layers
        )

        # Store
        for layer_idx, act in layer_acts.items():
            all_layer_activations[layer_idx].append(act)
        all_entropies.append(entropy)

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(
                all_layer_activations, all_entropies, i + 1, checkpoint_path
            )

        # Clear memory
        del inputs, input_ids, attention_mask
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    # Convert to numpy arrays
    activations = {
        layer_idx: np.array(acts)
        for layer_idx, acts in all_layer_activations.items()
    }
    entropies = np.array(all_entropies)

    print(f"Extracted activations shape (per layer): {activations[0].shape}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file")

    return activations, entropies


def save_checkpoint(
    layer_activations: Dict[int, List],
    entropies: List[float],
    processed_count: int,
    checkpoint_path: Path
):
    """Save extraction checkpoint."""
    save_dict = {
        f"layer_{i}": np.array(acts)
        for i, acts in layer_activations.items()
    }
    save_dict["entropies"] = np.array(entropies)
    save_dict["processed_count"] = np.array(processed_count)

    np.savez_compressed(checkpoint_path, **save_dict)
    print(f"  Checkpoint saved: {processed_count} prompts")


def _train_probe_for_layer(
    layer_idx: int,
    X: np.ndarray,
    entropies: np.ndarray,
    n_bootstrap: int,
    train_split: float,
    seed: int,
    use_pca: bool,
    pca_components: int,
    alpha: float
) -> Tuple[int, Dict, np.ndarray]:
    """Train probe for a single layer with bootstrap. Used for parallel execution.

    Returns (layer_idx, results_dict, direction_vector).
    Direction is extracted from a final probe trained on full training set.
    """
    rng = np.random.RandomState(seed + layer_idx)  # Reproducible per-layer
    n = len(entropies)

    test_r2s = []
    test_maes = []

    # Bootstrap for confidence intervals
    for _ in range(n_bootstrap):
        # Random split
        indices = np.arange(n)
        rng.shuffle(indices)
        split_idx = int(n * train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = entropies[train_idx]
        y_test = entropies[test_idx]

        # Train probe
        probe = LinearProbe(
            alpha=alpha,
            use_pca=use_pca,
            pca_components=pca_components
        )
        probe.fit(X_train, y_train)

        # Evaluate
        test_eval = probe.evaluate(X_test, y_test)
        test_r2s.append(test_eval["r2"])
        test_maes.append(test_eval["mae"])

    # Train final probe on canonical split for direction extraction
    rng_final = np.random.RandomState(seed)  # Same seed across layers for consistent split
    indices = np.arange(n)
    rng_final.shuffle(indices)
    split_idx = int(n * train_split)
    train_idx = indices[:split_idx]

    final_probe = LinearProbe(
        alpha=alpha,
        use_pca=use_pca,
        pca_components=pca_components
    )
    final_probe.fit(X[train_idx], entropies[train_idx])
    direction = final_probe.get_direction()  # Always in original space

    return layer_idx, {
        "test_r2_mean": float(np.mean(test_r2s)),
        "test_r2_std": float(np.std(test_r2s)),
        "test_mae_mean": float(np.mean(test_maes)),
        "test_mae_std": float(np.std(test_maes)),
    }, direction


def run_all_probes(
    activations: Dict[int, np.ndarray],
    entropies: np.ndarray,
    n_jobs: int = -1,
    use_pca: bool = USE_PCA,
    pca_components: int = PCA_COMPONENTS,
    alpha: float = PROBE_ALPHA
) -> Tuple[Dict[int, Dict], Dict[int, np.ndarray]]:
    """Train probes for all layers with bootstrap confidence intervals.

    Returns:
        results: Dict mapping layer index to result dict with R², MAE stats
        directions: Dict mapping layer index to normalized direction vectors (in original space)
    """
    pca_str = f"PCA={pca_components}" if use_pca else "no PCA"
    print(f"\nTraining probes for {len(activations)} layers "
          f"({N_BOOTSTRAP} bootstrap iterations, {pca_str}, parallel across layers)...")

    layer_indices = sorted(activations.keys())

    # Run in parallel across layers
    results_list = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_train_probe_for_layer)(
            layer_idx,
            activations[layer_idx],
            entropies,
            N_BOOTSTRAP,
            TRAIN_SPLIT,
            SEED,
            use_pca,
            pca_components,
            alpha
        )
        for layer_idx in layer_indices
    )

    # Convert list of (layer_idx, result, direction) tuples to dicts
    results = {layer_idx: result for layer_idx, result, _ in results_list}
    directions = {layer_idx: direction for layer_idx, _, direction in results_list}

    return results, directions


def print_results(results: Dict[int, Dict]):
    """Print summary of results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Layer':<8} {'Test R²':<20} {'Test MAE':<20}")
    print("-"*80)

    for layer_idx in sorted(results.keys()):
        res = results[layer_idx]
        r2_str = f"{res['test_r2_mean']:.4f} ± {res['test_r2_std']:.4f}"
        mae_str = f"{res['test_mae_mean']:.4f} ± {res['test_mae_std']:.4f}"
        print(f"{layer_idx:<8} {r2_str:<20} {mae_str:<20}")

    print("="*80)

    # Find best layer
    best_layer = max(results.keys(), key=lambda l: results[l]["test_r2_mean"])
    best_r2 = results[best_layer]["test_r2_mean"]
    best_std = results[best_layer]["test_r2_std"]
    print(f"\nBest layer: {best_layer} (Test R² = {best_r2:.4f} ± {best_std:.4f})")


def plot_results(results: Dict[int, Dict], output_path: Path):
    """Plot R² across layers with confidence intervals."""
    layers = sorted(results.keys())
    test_r2_mean = [results[l]["test_r2_mean"] for l in layers]
    test_r2_std = [results[l]["test_r2_std"] for l in layers]
    test_mae_mean = [results[l]["test_mae_mean"] for l in layers]
    test_mae_std = [results[l]["test_mae_std"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R² plot with error bands
    ax1.plot(layers, test_r2_mean, 'o-', label='Test R²', color='tab:blue')
    ax1.fill_between(
        layers,
        np.array(test_r2_mean) - np.array(test_r2_std),
        np.array(test_r2_mean) + np.array(test_r2_std),
        alpha=0.3, color='tab:blue'
    )
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Entropy Predictability by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE plot with error bands
    ax2.plot(layers, test_mae_mean, 'o-', label='Test MAE', color='tab:orange')
    ax2.fill_between(
        layers,
        np.array(test_mae_mean) - np.array(test_mae_std),
        np.array(test_mae_mean) + np.array(test_mae_std),
        alpha=0.3, color='tab:orange'
    )
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Prediction Error by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def plot_entropy_distribution(entropies: np.ndarray, output_path: Path):
    """Plot entropy distribution histogram."""
    _, ax = plt.subplots(figsize=(8, 5))

    ax.hist(entropies, bins=30, edgecolor='black', alpha=0.7,
            weights=np.ones(len(entropies)) / len(entropies) * 100)
    ax.axvline(entropies.mean(), color='red', linestyle='--',
               label=f'Mean: {entropies.mean():.3f}')
    ax.axvline(np.median(entropies), color='orange', linestyle='--',
               label=f'Median: {np.median(entropies):.3f}')
    ax.set_xlabel('Entropy')
    ax.set_ylabel('Percentage')
    ax.set_title(f'Next-Token Entropy Distribution (n={len(entropies)})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy distribution plot saved to {output_path}")


def load_activations(activations_path: Path) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """Load activations from saved file."""
    print(f"Loading activations from {activations_path}...")
    data = np.load(activations_path)

    activations = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }
    entropies = data["entropies"]

    print(f"Loaded {len(activations)} layers, {len(entropies)} samples")
    return activations, entropies


def main():
    parser = argparse.ArgumentParser(description="Train entropy probes on next-token prediction")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction, load saved activations and retrain probes")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Define output paths
    activations_path = Path(f"{output_prefix}_activations.npz")
    results_path = Path(f"{output_prefix}_entropy_probe.json")
    directions_path = Path(f"{output_prefix}_entropy_directions.npz")
    plot_path = Path(f"{output_prefix}_entropy_probe.png")
    entropy_dist_path = Path(f"{output_prefix}_entropy_distribution.png")
    checkpoint_path = Path(f"{output_prefix}_checkpoint.npz")

    if args.plot_only:
        # Load existing activations
        if not activations_path.exists():
            raise FileNotFoundError(
                f"Activations file not found: {activations_path}. "
                "Run without --plot-only first."
            )
        activations, entropies = load_activations(activations_path)
    else:
        # Full run: load model and extract activations
        model, tokenizer, num_layers = load_model_and_tokenizer(
            BASE_MODEL_NAME,
            adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
        )

        # Find and load dataset
        dataset_path = find_dataset_path()
        dataset, dataset_config = load_dataset(dataset_path)

        # Extract activations (with checkpointing)
        activations, entropies = extract_all_activations(
            dataset, model, tokenizer, num_layers, checkpoint_path
        )

        # Save activations
        print("\nSaving activations...")
        np.savez_compressed(
            activations_path,
            **{f"layer_{i}": acts for i, acts in activations.items()},
            entropies=entropies
        )
        print(f"Saved to {activations_path}")

    # Train probes (bootstrap for confidence intervals) and extract directions
    results, directions = run_all_probes(activations, entropies)

    # Save directions
    np.savez_compressed(
        directions_path,
        **{f"layer_{i}_entropy": d for i, d in directions.items()}
    )
    print(f"Saved directions to {directions_path}")

    # Save results with metadata
    output_data = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "train_split": TRAIN_SPLIT,
            "probe_alpha": PROBE_ALPHA,
            "use_pca": USE_PCA,
            "pca_components": PCA_COMPONENTS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "results": {
            str(k): v for k, v in results.items()
        }
    }

    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {results_path}")

    # Print and plot results
    print_results(results)
    plot_results(results, plot_path)
    plot_entropy_distribution(entropies, entropy_dist_path)


if __name__ == "__main__":
    main()
