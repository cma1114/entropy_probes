"""
Run entropy prediction experiment on multiple-choice questions.

This script:
1. Loads MC questions from dataset file
2. Formats them and runs through the model to get probability distributions over answer tokens
3. Computes entropy from those probabilities
4. Extracts activations from all layers
5. Trains linear probes to predict entropy from each layer

Usage:
    python mc_entropy_probe.py             # Full run
    python mc_entropy_probe.py --plot-only # Load saved activations, retrain probes, plot
"""

import argparse
import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import random

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_model_short_name,
    LinearProbe,
    compute_entropy_from_probs,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"###
DATASET_NAME = "SimpleMC"
NUM_QUESTIONS = 447 if DATASET_NAME.startswith("GP") else 500
MAX_PROMPT_LENGTH = 2000  # Multiple choice prompts can be longer
SEED = 42

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

# MC setup prompt
MC_SETUP_PROMPT = "I'm going to ask you a series of multiple-choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text."


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_mc")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_mc")


def load_questions(dataset_name: str, num_questions: int = None) -> List[Dict]:
    """Load MC questions using load_and_format_dataset."""
    from load_and_format_datasets import load_and_format_dataset

    questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    return questions


def _present_question(question_data: Dict) -> str:
    """Format a question for display."""
    formatted_question = ""
    formatted_question += "-" * 30 + "\n"
    formatted_question += "Question:\n"
    formatted_question += question_data["question"] + "\n"

    if "options" in question_data:
        formatted_question += "-" * 10 + "\n"
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"

    formatted_question += "-" * 30
    return formatted_question


def format_mc_prompt(question: Dict, tokenizer) -> Tuple[str, List[str], List[int]]:
    """
    Format a multiple-choice question and get option token IDs.

    Returns:
        Tuple of (full_prompt, option_keys, option_token_ids)
    """
    # Get the formatted question text
    q_text = _present_question(question)

    # Get option keys from the question
    options = list(question["options"].keys())

    # Format the prompt for choice
    options_str = (
        " or ".join(options)
        if len(options) == 2
        else ", ".join(options[:-1]) + f", or {options[-1]}"
    )

    llm_prompt = q_text + f"\nYour choice ({options_str}): "

    # Apply chat template
    messages = [
        {"role": "system", "content": MC_SETUP_PROMPT},
        {"role": "user", "content": llm_prompt}
    ]
    full_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Get token IDs for answer options
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]

    return full_prompt, options, option_token_ids


def extract_mc_activations_and_entropy(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    batch_size: int = 8
) -> Tuple[Dict[int, np.ndarray], np.ndarray, List[Dict]]:
    """
    Extract activations and compute entropy in a single forward pass per question.

    Args:
        questions: List of question dicts
        model: The model
        tokenizer: The tokenizer
        num_layers: Number of layers
        batch_size: Batch size for forward passes (default 8, reduce for large models)

    Returns:
        activations: Dict mapping layer_idx -> array of shape (num_questions, hidden_dim)
        entropies: Array of shape (num_questions,)
        metadata: List of dicts with question info, probabilities, etc.
    """
    print(f"Processing {len(questions)} questions with batch_size={batch_size}...")

    # Initialize storage
    all_layer_activations = {i: [] for i in range(num_layers)}
    all_entropies = []
    metadata = []

    # Set up hooks for activation extraction
    # Key optimization: store only last-token activations per batch item
    activations_cache = {}
    current_last_indices = None  # Will be set per batch
    hooks = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            nonlocal current_last_indices
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Extract only last-token activations for each batch item
            # current_last_indices: (batch_size,) tensor of last token positions
            batch_size_actual = hidden_states.shape[0]
            # Use advanced indexing to get last token for each batch item
            last_token_acts = hidden_states[
                torch.arange(batch_size_actual, device=hidden_states.device),
                current_last_indices[:batch_size_actual]
            ]  # Shape: (batch_size, hidden_dim)
            activations_cache[layer_idx] = last_token_acts.detach()
        return hook

    # Get layers
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        layers = base.model.layers
    else:
        layers = model.model.layers

    # Register hooks
    for i, layer in enumerate(layers):
        handle = layer.register_forward_hook(make_hook(i))
        hooks.append(handle)

    model.eval()

    # Pre-format all prompts
    print("Formatting prompts...")
    formatted_data = []
    for question in questions:
        full_prompt, options, option_token_ids = format_mc_prompt(question, tokenizer)
        formatted_data.append({
            "prompt": full_prompt,
            "options": options,
            "option_token_ids": option_token_ids,
            "question": question
        })

    try:
        # Process in batches
        for batch_start in tqdm(range(0, len(formatted_data), batch_size)):
            batch_end = min(batch_start + batch_size, len(formatted_data))
            batch_items = formatted_data[batch_start:batch_end]
            batch_prompts = [item["prompt"] for item in batch_items]

            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_LENGTH,
                padding=True  # Pad to longest in batch
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            # Compute last token indices
            # With left-padding, the last real token is always at the end of the sequence
            # So the last token index is simply seq_len - 1 for all items in the batch
            seq_len = input_ids.shape[1]
            current_last_indices = torch.full(
                (input_ids.shape[0],), seq_len - 1, device=input_ids.device, dtype=torch.long
            )

            # Clear cache
            activations_cache.clear()

            # Forward pass
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            # Single CPU transfer: stack all layers and transfer at once
            # activations_cache[layer_idx] is (batch_size, hidden_dim)
            stacked = torch.stack([activations_cache[i] for i in range(num_layers)], dim=0)
            # stacked shape: (num_layers, batch_size, hidden_dim)
            stacked_cpu = stacked.cpu().numpy()

            # Distribute to per-layer storage
            for layer_idx in range(num_layers):
                # stacked_cpu[layer_idx] is (batch_size, hidden_dim)
                for batch_item_idx in range(len(batch_items)):
                    all_layer_activations[layer_idx].append(stacked_cpu[layer_idx, batch_item_idx])

            # Compute entropy for each item in batch
            for batch_item_idx, item in enumerate(batch_items):
                option_token_ids = item["option_token_ids"]
                options = item["options"]
                question = item["question"]

                # Get logits at last token position for this batch item
                last_idx = current_last_indices[batch_item_idx].item()
                final_logits = outputs.logits[batch_item_idx, last_idx, :]
                option_logits = final_logits[option_token_ids]
                probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
                entropy = compute_entropy_from_probs(probs)

                # Compute accuracy
                predicted_idx = np.argmax(probs)
                predicted_answer = options[predicted_idx]
                correct_answer = question.get("correct_answer", "")
                is_correct = predicted_answer == correct_answer

                all_entropies.append(entropy)
                metadata.append({
                    "question_id": batch_start + batch_item_idx,
                    "question": question.get("question", ""),
                    "correct_answer": correct_answer,
                    "predicted_answer": predicted_answer,
                    "is_correct": is_correct,
                    "options": options,
                    "probabilities": probs.tolist(),
                    "entropy": entropy,
                })

            # Clear memory
            del inputs, input_ids, attention_mask, outputs, stacked, stacked_cpu
            if (batch_start + batch_size) % 100 == 0:
                torch.cuda.empty_cache()

    finally:
        # Remove hooks
        for handle in hooks:
            handle.remove()

    # Convert to numpy arrays
    activations = {
        layer_idx: np.array(acts)
        for layer_idx, acts in all_layer_activations.items()
    }
    entropies = np.array(all_entropies)

    # Compute accuracy statistics
    correct_count = sum(1 for m in metadata if m["is_correct"])
    accuracy = correct_count / len(metadata)

    print(f"Extracted activations shape (per layer): {activations[0].shape}")
    print(f"Entropy range: [{entropies.min():.3f}, {entropies.max():.3f}]")
    print(f"Entropy mean: {entropies.mean():.3f}, std: {entropies.std():.3f}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{len(metadata)})")

    return activations, entropies, metadata


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
    rng = np.random.RandomState(seed + layer_idx)
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


def plot_entropy_distribution(
    entropies: np.ndarray,
    metadata: List[Dict],
    output_path: Path
):
    """Plot entropy distribution with accuracy breakdown."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Overall entropy histogram (percentage)
    ax1 = axes[0]
    ax1.hist(entropies, bins=30, edgecolor='black', alpha=0.7, weights=np.ones(len(entropies)) / len(entropies) * 100)
    ax1.axvline(entropies.mean(), color='red', linestyle='--', label=f'Mean: {entropies.mean():.3f}')
    ax1.axvline(np.median(entropies), color='orange', linestyle='--', label=f'Median: {np.median(entropies):.3f}')
    ax1.set_xlabel('Entropy')
    ax1.set_ylabel('Percentage')
    ax1.set_title(f'MC Entropy Distribution (n={len(entropies)})')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Entropy by correctness (percentage within each group)
    ax2 = axes[1]
    correct_entropies = [m["entropy"] for m in metadata if m["is_correct"]]
    incorrect_entropies = [m["entropy"] for m in metadata if not m["is_correct"]]

    if correct_entropies:
        ax2.hist(correct_entropies, bins=20, alpha=0.6, label=f'Correct (n={len(correct_entropies)})',
                 color='green', weights=np.ones(len(correct_entropies)) / len(correct_entropies) * 100)
    if incorrect_entropies:
        ax2.hist(incorrect_entropies, bins=20, alpha=0.6, label=f'Incorrect (n={len(incorrect_entropies)})',
                 color='red', weights=np.ones(len(incorrect_entropies)) / len(incorrect_entropies) * 100)
    ax2.set_xlabel('Entropy')
    ax2.set_ylabel('Percentage')
    ax2.set_title('Entropy by Correctness')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Accuracy vs entropy bins
    ax3 = axes[2]
    n_bins = 10
    entropy_bins = np.linspace(entropies.min(), entropies.max(), n_bins + 1)
    bin_accuracies = []
    bin_centers = []
    bin_counts = []

    for i in range(n_bins):
        bin_mask = (entropies >= entropy_bins[i]) & (entropies < entropy_bins[i + 1])
        if i == n_bins - 1:  # Include right edge in last bin
            bin_mask = (entropies >= entropy_bins[i]) & (entropies <= entropy_bins[i + 1])

        bin_items = [m for j, m in enumerate(metadata) if bin_mask[j]]
        if len(bin_items) > 0:
            acc = sum(1 for m in bin_items if m["is_correct"]) / len(bin_items)
            bin_accuracies.append(acc)
            bin_centers.append((entropy_bins[i] + entropy_bins[i + 1]) / 2)
            bin_counts.append(len(bin_items))

    ax3.bar(bin_centers, bin_accuracies, width=(entropy_bins[1] - entropy_bins[0]) * 0.8,
            alpha=0.7, edgecolor='black')
    ax3.set_xlabel('Entropy')
    ax3.set_ylabel('Accuracy')
    ax3.set_title('Accuracy vs Entropy')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    # Add count labels on bars
    for x, y, c in zip(bin_centers, bin_accuracies, bin_counts):
        ax3.text(x, y + 0.02, f'n={c}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Entropy distribution plot saved to {output_path}")


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
    ax1.set_title('MC Entropy Predictability by Layer')
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


def print_diagnostic_summary(entropies: np.ndarray, results: Dict[int, Dict]):
    """Print diagnostic summary to help identify anomalous patterns."""
    print("\n" + "="*80)
    print("DIAGNOSTIC SUMMARY")
    print("="*80)

    # Entropy distribution diagnostics
    pct_near_zero = (entropies < 0.1).sum() / len(entropies) * 100
    pct_low = (entropies < 0.3).sum() / len(entropies) * 100
    variance = entropies.var()
    iqr = np.percentile(entropies, 75) - np.percentile(entropies, 25)

    print(f"\nEntropy Distribution:")
    print(f"  Mean: {entropies.mean():.3f}, Std: {entropies.std():.3f}, Variance: {variance:.4f}")
    print(f"  Median: {np.median(entropies):.3f}, IQR: {iqr:.3f}")
    print(f"  Range: [{entropies.min():.3f}, {entropies.max():.3f}]")
    print(f"  % near zero (<0.1): {pct_near_zero:.1f}%")
    print(f"  % low (<0.3): {pct_low:.1f}%")

    # Early vs late layer comparison
    layers = sorted(results.keys())
    n_layers = len(layers)
    early_layers = layers[:n_layers // 4]  # First quarter
    late_layers = layers[3 * n_layers // 4:]  # Last quarter

    early_r2 = np.mean([results[l]["test_r2_mean"] for l in early_layers])
    late_r2 = np.mean([results[l]["test_r2_mean"] for l in late_layers])
    r2_increase = late_r2 - early_r2

    print(f"\nLayer R² Comparison:")
    print(f"  Early layers (0-{early_layers[-1]}): mean R² = {early_r2:.4f}")
    print(f"  Late layers ({late_layers[0]}-{late_layers[-1]}): mean R² = {late_r2:.4f}")
    print(f"  R² increase (late - early): {r2_increase:.4f}")

    # Flag anomalies
    print(f"\nAnomaly Flags:")
    flags = []

    if pct_near_zero > 40:
        flags.append(f"  ⚠ HIGH % NEAR-ZERO ENTROPY ({pct_near_zero:.1f}%) - probe may be trivially predictive")

    if variance < 0.05:
        flags.append(f"  ⚠ LOW ENTROPY VARIANCE ({variance:.4f}) - limited signal to predict")

    if early_r2 > 0.5 and r2_increase < 0.1:
        flags.append(f"  ⚠ UNIFORMLY HIGH R² - early layers already predictive (early={early_r2:.3f})")

    if late_r2 < 0.3:
        flags.append(f"  ⚠ UNIFORMLY LOW R² - even late layers poorly predictive (late={late_r2:.3f})")

    if abs(r2_increase) < 0.1 and early_r2 > 0.3:
        flags.append(f"  ⚠ FLAT R² CURVE - no clear emergence pattern")

    if not flags:
        flags.append("  ✓ No anomalies detected - typical emergence pattern")

    for flag in flags:
        print(flag)

    print("="*80)


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
    parser = argparse.ArgumentParser(description="Train entropy probes on MC questions")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip extraction, load saved activations and retrain probes")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size for forward passes (default 8, reduce for large models)")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (recommended for 70B+ models)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Define output paths
    activations_path = Path(f"{output_prefix}_activations.npz")
    dataset_path = Path(f"{output_prefix}_entropy_dataset.json")
    results_path = Path(f"{output_prefix}_entropy_probe.json")
    directions_path = Path(f"{output_prefix}_entropy_directions.npz")
    plot_path = Path(f"{output_prefix}_entropy_probe.png")
    entropy_dist_path = Path(f"{output_prefix}_entropy_distribution.png")

    metadata = None  # Will be loaded or computed

    if args.plot_only:
        # Load existing activations
        if not activations_path.exists():
            raise FileNotFoundError(
                f"Activations file not found: {activations_path}. "
                "Run without --plot-only first."
            )
        activations, entropies = load_activations(activations_path)

        # Load metadata for entropy distribution plot
        if dataset_path.exists():
            with open(dataset_path) as f:
                dataset_data = json.load(f)
                metadata = dataset_data.get("data", [])
    else:
        # Full run: load model and extract activations
        model, tokenizer, num_layers = load_model_and_tokenizer(
            BASE_MODEL_NAME,
            adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )

        # Load questions
        questions = load_questions(DATASET_NAME, NUM_QUESTIONS)

        # Shuffle with fixed seed for reproducibility
        random.seed(SEED)
        random.shuffle(questions)

        # Extract activations and compute entropies in single pass
        activations, entropies, metadata = extract_mc_activations_and_entropy(
            questions, model, tokenizer, num_layers, batch_size=args.batch_size
        )

        # Save activations
        print("\nSaving activations...")
        np.savez_compressed(
            activations_path,
            **{f"layer_{i}": acts for i, acts in activations.items()},
            entropies=entropies
        )
        print(f"Saved to {activations_path}")

        # Compute accuracy stats for saving
        correct_count = sum(1 for m in metadata if m["is_correct"])
        accuracy = correct_count / len(metadata)

        # Save dataset with metadata
        output_data = {
            "config": {
                "dataset": DATASET_NAME,
                "num_questions": NUM_QUESTIONS,
                "base_model": BASE_MODEL_NAME,
                "seed": SEED,
            },
            "stats": {
                "accuracy": accuracy,
                "correct_count": correct_count,
                "total_count": len(metadata),
                "entropy_mean": float(entropies.mean()),
                "entropy_std": float(entropies.std()),
                "entropy_min": float(entropies.min()),
                "entropy_max": float(entropies.max()),
            },
            "data": metadata
        }
        with open(dataset_path, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved dataset to {dataset_path}")

    # Train probes (bootstrap for confidence intervals) and extract directions
    results, directions = run_all_probes(activations, entropies)

    # Save directions (include metadata for robust parsing by analysis scripts)
    directions_data = {f"layer_{i}_entropy": d for i, d in directions.items()}
    # Store dataset name as metadata so analysis scripts don't need to parse filenames
    directions_data["_metadata_dataset"] = np.array(DATASET_NAME)
    directions_data["_metadata_model"] = np.array(BASE_MODEL_NAME)
    np.savez_compressed(directions_path, **directions_data)
    print(f"Saved directions to {directions_path}")

    # Compute accuracy from metadata if available
    accuracy_stats = None
    if metadata:
        correct_count = sum(1 for m in metadata if m.get("is_correct", False))
        accuracy_stats = {
            "accuracy": correct_count / len(metadata),
            "correct_count": correct_count,
            "total_count": len(metadata),
        }

    # Save results with metadata
    output_data = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "dataset": DATASET_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "train_split": TRAIN_SPLIT,
            "probe_alpha": PROBE_ALPHA,
            "use_pca": USE_PCA,
            "pca_components": PCA_COMPONENTS,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        },
        "entropy_stats": {
            "mean": float(entropies.mean()),
            "std": float(entropies.std()),
            "min": float(entropies.min()),
            "max": float(entropies.max()),
            "variance": float(entropies.var()),
            "pct_near_zero": float((entropies < 0.1).sum() / len(entropies) * 100),
            "pct_near_max": float((entropies > 1.3).sum() / len(entropies) * 100),
            "median": float(np.median(entropies)),
            "iqr": float(np.percentile(entropies, 75) - np.percentile(entropies, 25)),
        },
        "results": {
            str(k): v for k, v in results.items()
        }
    }

    # Add accuracy if available
    if accuracy_stats:
        output_data["accuracy"] = accuracy_stats

    with open(results_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved results to {results_path}")

    # Print and plot results
    print_results(results)
    plot_results(results, plot_path)

    # Print diagnostic summary for anomaly detection
    print_diagnostic_summary(entropies, results)

    # Plot entropy distribution if metadata available
    if metadata:
        plot_entropy_distribution(entropies, metadata, entropy_dist_path)


if __name__ == "__main__":
    main()
