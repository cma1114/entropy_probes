"""
Build a stratified dataset for entropy prediction experiments.

This script:
1. Samples diverse text from multiple HuggingFace datasets
2. Runs a pilot to compute actual output entropy for each prompt
3. Stratifies by entropy and samples evenly across deciles
4. Saves the final dataset

Supports resuming from checkpoints if interrupted.
"""

import torch
import numpy as np
from datasets import load_dataset
import json
from pathlib import Path
from tqdm import tqdm
import random
from typing import List, Dict

from core import (
    DEVICE,
    load_model_and_tokenizer,
    get_run_name,
    get_model_short_name,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
PILOT_SIZE = 10000  # Number of prompts to sample in pilot
FINAL_SIZE = 5000   # Number of prompts in final dataset (500 per decile)
MIN_PROMPT_LENGTH = 20  # tokens
MAX_PROMPT_LENGTH = 500  # tokens
CHECKPOINT_INTERVAL = 500  # Save checkpoint every N prompts
SEED = 42

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_nexttoken")
    return str(OUTPUTS_DIR / f"{model_short}_nexttoken")


def load_diverse_texts(num_samples: int) -> List[str]:
    """
    Load diverse text samples from multiple sources.

    Returns a list of text strings sampled from:
    - Wikipedia
    - Code (The Stack)
    - FineWeb (web crawl)
    - C4 (web crawl)
    """
    print("Loading diverse text sources...")
    all_texts = []
    samples_per_source = num_samples // 4

    # Wikipedia
    print("  Loading Wikipedia...")
    try:
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        wiki_texts = []
        for i, item in enumerate(wiki):
            if i >= samples_per_source:
                break
            text = item["text"]
            if len(text) > 100:
                wiki_texts.append(text)
        all_texts.extend(wiki_texts)
        print(f"    Loaded {len(wiki_texts)} Wikipedia samples")
    except Exception as e:
        print(f"    Warning: Could not load Wikipedia: {e}")

    # Code
    print("  Loading code samples...")
    try:
        code = load_dataset("bigcode/the-stack-smol", data_dir="data/python", split="train", streaming=True)
        code_texts = []
        for i, item in enumerate(code):
            if i >= samples_per_source:
                break
            text = item["content"]
            if len(text) > 100:
                code_texts.append(text)
        all_texts.extend(code_texts)
        print(f"    Loaded {len(code_texts)} code samples")
    except Exception as e:
        print(f"    Warning: Could not load code: {e}")

    # FineWeb
    print("  Loading FineWeb...")
    try:
        fineweb = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
        fineweb_texts = []
        for i, item in enumerate(fineweb):
            if i >= samples_per_source:
                break
            text = item["text"]
            if len(text) > 100:
                fineweb_texts.append(text)
        all_texts.extend(fineweb_texts)
        print(f"    Loaded {len(fineweb_texts)} FineWeb samples")
    except Exception as e:
        print(f"    Warning: Could not load FineWeb: {e}")

    # C4
    print("  Loading C4...")
    try:
        c4 = load_dataset("allenai/c4", "en", split="train", streaming=True)
        c4_texts = []
        for i, item in enumerate(c4):
            if i >= samples_per_source:
                break
            text = item["text"]
            if len(text) > 100:
                c4_texts.append(text)
        all_texts.extend(c4_texts)
        print(f"    Loaded {len(c4_texts)} C4 samples")
    except Exception as e:
        print(f"    Warning: Could not load C4: {e}")

    print(f"Total texts loaded: {len(all_texts)}")
    return all_texts


def create_prompts(texts: List[str], tokenizer, num_prompts: int) -> List[Dict]:
    """
    Create prompts of varying lengths from text samples.

    Returns list of dicts with 'text' and 'prompt_length' keys.
    """
    print(f"Creating {num_prompts} prompts...")
    prompts = []

    for _ in tqdm(range(num_prompts)):
        # Pick random text
        text = random.choice(texts)

        # Tokenize
        tokens = tokenizer.encode(text, add_special_tokens=False)

        # Skip if too short
        if len(tokens) < MIN_PROMPT_LENGTH + 1:
            continue

        # Pick random prompt length
        max_len = min(len(tokens) - 1, MAX_PROMPT_LENGTH)
        prompt_length = random.randint(MIN_PROMPT_LENGTH, max_len)

        # Extract prompt
        prompt_tokens = tokens[:prompt_length]
        prompt_text = tokenizer.decode(prompt_tokens)

        prompts.append({
            "text": prompt_text,
            "prompt_length": prompt_length
        })

    print(f"Created {len(prompts)} valid prompts")
    return prompts


def compute_entropy(logits: torch.Tensor) -> float:
    """Compute entropy of next-token distribution."""
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum().item()
    return entropy


def load_checkpoint(checkpoint_path: Path) -> List[Dict]:
    """Load checkpoint if it exists."""
    if checkpoint_path.exists():
        print(f"Loading checkpoint from {checkpoint_path}...")
        with open(checkpoint_path) as f:
            return json.load(f)
    return []


def save_checkpoint(results: List[Dict], checkpoint_path: Path):
    """Save checkpoint."""
    with open(checkpoint_path, "w") as f:
        json.dump(results, f)
    print(f"  Checkpoint saved: {len(results)} prompts")


def run_pilot_inference(
    prompts: List[Dict],
    model,
    tokenizer,
    checkpoint_path: Path
) -> List[Dict]:
    """
    Run inference on prompts and compute output entropy.
    Supports resuming from checkpoint.
    """
    # Load existing results if any
    results = load_checkpoint(checkpoint_path)
    start_idx = len(results)

    if start_idx > 0:
        print(f"Resuming from prompt {start_idx}/{len(prompts)}")

    print(f"Running pilot inference on {len(prompts) - start_idx} remaining prompts...")
    model.eval()

    for i in tqdm(range(start_idx, len(prompts))):
        prompt_data = prompts[i]
        text = prompt_data["text"]

        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_PROMPT_LENGTH
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        # Forward pass
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get logits at last position
        last_token_idx = inputs["attention_mask"].sum() - 1
        last_logits = logits[0, last_token_idx, :].cpu()

        # Compute entropy
        entropy = compute_entropy(last_logits)

        results.append({
            **prompt_data,
            "entropy": entropy
        })

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(results, checkpoint_path)

        # Clear memory
        del inputs, outputs, logits
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    # Final save
    save_checkpoint(results, checkpoint_path)

    return results


def stratify_and_sample(
    prompts_with_entropy: List[Dict],
    num_samples: int,
    num_bins: int = 10
) -> List[Dict]:
    """
    Stratify prompts by entropy into bins and sample evenly.
    """
    print(f"Stratifying by entropy into {num_bins} bins...")

    # Sort by entropy
    sorted_prompts = sorted(prompts_with_entropy, key=lambda x: x["entropy"])

    # Get entropy range
    entropies = [p["entropy"] for p in sorted_prompts]
    min_entropy = min(entropies)
    max_entropy = max(entropies)
    print(f"  Entropy range: [{min_entropy:.3f}, {max_entropy:.3f}]")

    # Create bins using percentiles
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(entropies, percentiles)

    # Assign each prompt to a bin
    bins = [[] for _ in range(num_bins)]
    for prompt in sorted_prompts:
        entropy = prompt["entropy"]
        bin_idx = np.searchsorted(bin_edges[1:], entropy)
        bin_idx = min(bin_idx, num_bins - 1)
        bins[bin_idx].append(prompt)

    # Print bin statistics
    print("  Bin statistics:")
    for i, bin_prompts in enumerate(bins):
        if bin_prompts:
            bin_entropies = [p["entropy"] for p in bin_prompts]
            print(f"    Bin {i}: {len(bin_prompts)} prompts, "
                  f"entropy [{min(bin_entropies):.3f}, {max(bin_entropies):.3f}]")

    # Sample evenly from each bin
    samples_per_bin = num_samples // num_bins
    stratified_sample = []

    for i, bin_prompts in enumerate(bins):
        if len(bin_prompts) >= samples_per_bin:
            sampled = random.sample(bin_prompts, samples_per_bin)
        else:
            print(f"    Warning: Bin {i} has only {len(bin_prompts)} prompts, using all")
            sampled = bin_prompts
        stratified_sample.extend(sampled)

    print(f"Final dataset size: {len(stratified_sample)}")
    return stratified_sample


def main():
    print(f"Device: {DEVICE}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Define output paths
    samples_raw_path = Path(f"{output_prefix}_samples_raw.json")
    checkpoint_path = Path(f"{output_prefix}_checkpoint.json")
    final_output = Path(f"{output_prefix}_entropy_dataset.json")

    # Load model
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
    )
    print(f"Model has {num_layers} layers")

    # Step 1: Load diverse texts
    texts = load_diverse_texts(PILOT_SIZE)

    # Step 2: Create prompts
    prompts = create_prompts(texts, tokenizer, PILOT_SIZE)

    # Step 3: Run pilot inference (with checkpointing)
    prompts_with_entropy = run_pilot_inference(
        prompts, model, tokenizer, checkpoint_path
    )

    # Save raw samples (before stratification)
    with open(samples_raw_path, "w") as f:
        json.dump(prompts_with_entropy, f, indent=2)
    print(f"Saved raw samples to {samples_raw_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print("Removed checkpoint file")

    # Step 4: Stratify and sample
    final_dataset = stratify_and_sample(prompts_with_entropy, FINAL_SIZE)

    # Save final dataset with metadata
    output_data = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "pilot_size": PILOT_SIZE,
            "final_size": FINAL_SIZE,
            "min_prompt_length": MIN_PROMPT_LENGTH,
            "max_prompt_length": MAX_PROMPT_LENGTH,
            "seed": SEED,
        },
        "data": final_dataset
    }

    with open(final_output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved final dataset to {final_output}")

    # Print summary statistics
    entropies = [p["entropy"] for p in final_dataset]
    print("\nFinal dataset statistics:")
    print(f"  Size: {len(final_dataset)}")
    print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
    print(f"  Entropy mean: {np.mean(entropies):.3f}")
    print(f"  Entropy std: {np.std(entropies):.3f}")


if __name__ == "__main__":
    main()
