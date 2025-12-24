"""
Run entropy prediction experiment on diverse text.

This script:
1. Loads the stratified dataset from build_dataset.py
2. Extracts activations from all layers of the model
3. Trains linear probes to predict entropy from each layer
4. Evaluates and visualizes results

Supports resuming from checkpoint if interrupted during extraction.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_model_short_name,
    LinearProbe,
    extract_activations_only,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_PATH = None  # Auto-detect based on model name, or set explicitly
MAX_PROMPT_LENGTH = 500
SEED = 42
CHECKPOINT_INTERVAL = 200  # Save checkpoint every N prompts

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100

np.random.seed(SEED)
torch.manual_seed(SEED)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    return f"{model_short}_nexttoken"


def find_dataset_path() -> Path:
    """Find the dataset file, auto-detecting based on model name."""
    if DATASET_PATH:
        return Path(DATASET_PATH)

    # Try model-specific path first
    model_short = get_model_short_name(BASE_MODEL_NAME)
    model_specific = Path(f"{model_short}_nexttoken_entropy_dataset.json")
    if model_specific.exists():
        return model_specific

    # Fall back to generic name
    generic = Path("entropy_dataset.json")
    if generic.exists():
        return generic

    raise FileNotFoundError(
        f"Could not find dataset. Tried: {model_specific}, {generic}. "
        "Run build_dataset.py first."
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
        start_idx = int(checkpoint["processed_count"])
        all_layer_activations = {
            int(k.split("_")[1]): list(checkpoint[k])
            for k in checkpoint.files if k.startswith("layer_")
        }
        all_entropies = list(checkpoint["entropies"])
        print(f"Resuming from prompt {start_idx}/{len(dataset)}")
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


def run_all_probes(
    activations: Dict[int, np.ndarray],
    entropies: np.ndarray
) -> Dict[int, Dict]:
    """Train probes for all layers."""
    print(f"\nTraining probes for {len(activations)} layers...")

    # Split data
    n = len(entropies)
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_idx = int(n * TRAIN_SPLIT)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    results = {}

    for layer_idx in tqdm(sorted(activations.keys())):
        X = activations[layer_idx]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = entropies[train_idx]
        y_test = entropies[test_idx]

        # Train probe
        probe = LinearProbe(
            alpha=PROBE_ALPHA,
            use_pca=USE_PCA,
            pca_components=PCA_COMPONENTS
        )
        probe.fit(X_train, y_train)

        # Evaluate
        train_eval = probe.evaluate(X_train, y_train)
        test_eval = probe.evaluate(X_test, y_test)

        results[layer_idx] = {
            "train_r2": train_eval["r2"],
            "test_r2": test_eval["r2"],
            "train_mae": train_eval["mae"],
            "test_mae": test_eval["mae"],
            "pca_variance_explained": (
                float(probe.pca.explained_variance_ratio_.sum())
                if USE_PCA and probe.pca else None
            ),
        }

    return results


def print_results(results: Dict[int, Dict]):
    """Print summary of results."""
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Layer':<8} {'Train R²':<12} {'Test R²':<12} {'Train MAE':<12} {'Test MAE':<12}")
    print("-"*80)

    for layer_idx in sorted(results.keys()):
        res = results[layer_idx]
        print(f"{layer_idx:<8} {res['train_r2']:<12.4f} {res['test_r2']:<12.4f} "
              f"{res['train_mae']:<12.4f} {res['test_mae']:<12.4f}")

    print("="*80)

    # Find best layer
    best_layer = max(results.keys(), key=lambda l: results[l]["test_r2"])
    best_r2 = results[best_layer]["test_r2"]
    print(f"\nBest layer: {best_layer} (Test R² = {best_r2:.4f})")

    # First layer with R² > 0.5
    for layer_idx in sorted(results.keys()):
        if results[layer_idx]["test_r2"] > 0.5:
            print(f"First layer with R² > 0.5: {layer_idx} "
                  f"(R² = {results[layer_idx]['test_r2']:.4f})")
            break


def plot_results(results: Dict[int, Dict], output_path: Path):
    """Plot R² across layers."""
    layers = sorted(results.keys())
    train_r2 = [results[l]["train_r2"] for l in layers]
    test_r2 = [results[l]["test_r2"] for l in layers]
    train_mae = [results[l]["train_mae"] for l in layers]
    test_mae = [results[l]["test_mae"] for l in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # R² plot
    ax1.plot(layers, train_r2, 'o-', label='Train R²', alpha=0.7)
    ax1.plot(layers, test_r2, 'o-', label='Test R²', alpha=0.7)
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='R² = 0.5')
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Entropy Predictability by Layer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE plot
    ax2.plot(layers, train_mae, 'o-', label='Train MAE', alpha=0.7)
    ax2.plot(layers, test_mae, 'o-', label='Test MAE', alpha=0.7)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Prediction Error by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    print(f"Device: {DEVICE}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Define output paths
    activations_path = Path(f"{output_prefix}_activations.npz")
    results_path = Path(f"{output_prefix}_entropy_probe.json")
    plot_path = Path(f"{output_prefix}_entropy_probe.png")
    checkpoint_path = Path(f"{output_prefix}_checkpoint.npz")

    # Load model
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

    # Train probes
    results = run_all_probes(activations, entropies)

    # Save results with metadata
    output_data = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "dataset_path": str(dataset_path),
            "dataset_size": len(dataset),
            "train_split": TRAIN_SPLIT,
            "probe_alpha": PROBE_ALPHA,
            "use_pca": USE_PCA,
            "pca_components": PCA_COMPONENTS,
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


if __name__ == "__main__":
    main()
