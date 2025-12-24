"""
Run entropy prediction experiment on multiple-choice questions.

This script:
1. Loads MC questions from dataset file
2. Formats them exactly like capabilities_test.py does
3. Runs them through the model to get probability distributions over answer tokens (A, B, C, D)
4. Computes entropy from those probabilities
5. Extracts activations from all layers
6. Trains linear probes to predict entropy from each layer
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import random

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_NAME = "SimpleMC"  # Dataset to load: SimpleMC, MMLU, GPQA, etc.
NUM_QUESTIONS = 500
MAX_PROMPT_LENGTH = 2000  # Multiple choice prompts can be longer
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = MODEL_NAME.split("/")[-1]
    return f"{model_short}_{DATASET_NAME}_mc"

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0  # Ridge regression regularization
USE_PCA = True
PCA_COMPONENTS = 100

np.random.seed(SEED)
torch.manual_seed(SEED)

# MC setup prompt (from capabilities_test.py)
MC_SETUP_PROMPT = "I'm going to ask you a series of multiple-choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text."


def load_questions(dataset_name: str, num_questions: int = None) -> List[Dict]:
    """Load MC questions using load_and_format_dataset."""
    from load_and_format_datasets import load_and_format_dataset

    questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    return questions


def _present_question(question_data: Dict) -> str:
    """
    Format a question for display exactly as base_game_class.py does.

    Args:
        question_data: Dict with 'question' and 'options' keys
                      options must be a dict {A: opt1, B: opt2, ...}

    Returns:
        Formatted question string
    """
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


def format_mc_question(question: Dict) -> Tuple[str, List[str]]:
    """
    Format a multiple-choice question exactly as capabilities_test.py does.

    Args:
        question: Dict with 'question' and 'options' keys
                  options must be a dict {A: opt1, B: opt2, ...}

    Returns:
        Tuple of (formatted_prompt, option_keys)
    """
    # Get the formatted question text (what would be shown to user)
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

    return llm_prompt, options


def compute_entropy_from_probs(probs: List[float]) -> float:
    """
    Compute entropy from a probability distribution.

    Args:
        probs: List of probabilities

    Returns:
        Entropy value (float)
    """
    probs = np.array(probs)
    # Normalize to sum to 1
    probs = probs / probs.sum()
    # Filter out zero probabilities to avoid log(0)
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy)


def get_answer_option_probs(
    model,
    tokenizer,
    full_prompt: str,
    option_keys: List[str]
) -> List[float]:
    """
    Get probability distribution over answer option tokens (A, B, C, D, etc.).

    Args:
        model: The language model
        tokenizer: The tokenizer
        full_prompt: The full formatted prompt (setup + question + options + "Your choice (A, B, C, or D): ")
        option_keys: List of option letters (e.g., ['A', 'B', 'C', 'D'])

    Returns:
        List of probabilities for each option
    """
    # Apply chat template
    messages = [
        {"role": "system", "content": MC_SETUP_PROMPT},
        {"role": "user", "content": full_prompt}
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)

    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs, use_cache=False)

    # Get logits at last position
    final_logits = outputs.logits[0, -1, :]  # [vocab_size]

    # Get token IDs for answer options
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in option_keys]

    # Extract logits for these tokens
    option_logits = final_logits[option_token_ids]

    # Convert to probabilities
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    return option_probs.tolist()


def build_dataset_with_entropies(
    questions: List[Dict],
    model,
    tokenizer
) -> List[Dict]:
    """
    Process all questions: format them, get probabilities, compute entropies.

    Returns:
        List of dicts with 'prompt', 'entropy', 'probabilities', etc.
    """
    print(f"Processing {len(questions)} questions to compute entropies...")

    dataset = []

    for i, question in enumerate(tqdm(questions)):
        # Format question
        full_question, option_keys = format_mc_question(question)

        # Get probabilities over answer options
        probs = get_answer_option_probs(model, tokenizer, full_question, option_keys)

        # Compute entropy
        entropy = compute_entropy_from_probs(probs)

        # Create full prompt for activation extraction (setup + question)
        messages = [
            {"role": "system", "content": MC_SETUP_PROMPT},
            {"role": "user", "content": full_question}
        ]
        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        dataset.append({
            "prompt": full_prompt,
            "entropy": entropy,
            "probabilities": probs,
            "question_id": i,
            "question": question.get("question", ""),
            "correct_answer": question.get("correct_answer", ""),
            "options": option_keys
        })

        # Clear memory periodically
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    print(f"Processed {len(dataset)} questions")

    # Print entropy statistics
    entropies = [d["entropy"] for d in dataset]
    print(f"  Entropy range: [{min(entropies):.3f}, {max(entropies):.3f}]")
    print(f"  Entropy mean: {np.mean(entropies):.3f}")
    print(f"  Entropy std: {np.std(entropies):.3f}")

    return dataset


class ActivationExtractor:
    """Extract activations from all layers of a model."""

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.activations = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # Store the output (residual stream after this layer)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook

    def register_hooks(self):
        """Register forward hooks on all layers."""
        # Handle both regular and PEFT models
        if hasattr(self.model, 'get_base_model'):
            # PEFT model - use get_base_model()
            base = self.model.get_base_model()
            layers = base.model.layers
        else:
            # Regular model
            layers = self.model.model.layers

        for i, layer in enumerate(layers):
            hook = self._make_hook(i)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Extract activations from all layers for a single input.

        Returns:
            Dict mapping layer_idx -> activation tensor at last token position
        """
        self.activations = {}

        with torch.no_grad():
            _ = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Extract last token position activations from each layer
        last_token_idx = attention_mask.sum() - 1
        layer_activations = {}

        for layer_idx, acts in self.activations.items():
            # acts shape: (batch_size=1, seq_len, hidden_dim)
            last_token_act = acts[0, last_token_idx, :].cpu().numpy()
            layer_activations[layer_idx] = last_token_act

        return layer_activations


def extract_all_activations(
    dataset: List[Dict],
    model,
    tokenizer,
    num_layers: int
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Extract activations from all layers for all prompts in dataset.

    Returns:
        activations: Dict mapping layer_idx -> array of shape (num_prompts, hidden_dim)
        entropies: Array of shape (num_prompts,)
    """
    print(f"Extracting activations from {num_layers} layers for {len(dataset)} prompts...")

    extractor = ActivationExtractor(model, num_layers)
    extractor.register_hooks()

    # Initialize storage
    all_layer_activations = {i: [] for i in range(num_layers)}
    all_entropies = []

    model.eval()

    try:
        for item in tqdm(dataset):
            prompt = item["prompt"]
            entropy = item["entropy"]

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_PROMPT_LENGTH
            )
            input_ids = inputs["input_ids"].to(DEVICE)
            attention_mask = inputs["attention_mask"].to(DEVICE)

            # Extract activations
            layer_acts = extractor.extract(input_ids, attention_mask)

            # Store
            for layer_idx, act in layer_acts.items():
                all_layer_activations[layer_idx].append(act)
            all_entropies.append(entropy)

            # Clear memory
            del inputs, input_ids, attention_mask
            if len(all_entropies) % 100 == 0:
                torch.cuda.empty_cache()

    finally:
        extractor.remove_hooks()

    # Convert to numpy arrays
    activations = {
        layer_idx: np.array(acts)
        for layer_idx, acts in all_layer_activations.items()
    }
    entropies = np.array(all_entropies)

    print(f"Extracted activations shape (per layer): {activations[0].shape}")
    print(f"Entropies shape: {entropies.shape}")

    return activations, entropies


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict:
    """
    Train a linear probe to predict entropy from activations.

    Returns:
        Dict with metrics: r2, mae, predictions
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if enabled
    if USE_PCA:
        n_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_test_reduced = pca.transform(X_test_scaled)

        # Use reduced features
        X_train_final = X_train_reduced
        X_test_final = X_test_reduced
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled

    # Train Ridge regression probe
    probe = Ridge(alpha=PROBE_ALPHA)
    probe.fit(X_train_final, y_train)

    # Evaluate
    y_pred_train = probe.predict(X_train_final)
    y_pred_test = probe.predict(X_test_final)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    return {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "predictions": y_pred_test,
        "probe": probe,
        "pca_variance_explained": pca.explained_variance_ratio_.sum() if USE_PCA else None
    }


def run_all_probes(
    activations: Dict[int, np.ndarray],
    entropies: np.ndarray
) -> Dict[int, Dict]:
    """
    Train probes for all layers.

    Returns:
        Dict mapping layer_idx -> probe results
    """
    print(f"\nTraining probes for {len(activations)} layers...")

    # Split data
    indices = np.arange(len(entropies))
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED
    )

    results = {}

    for layer_idx in tqdm(sorted(activations.keys())):
        X = activations[layer_idx]

        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = entropies[train_idx]
        y_test = entropies[test_idx]

        probe_results = train_probe(X_train, y_train, X_test, y_test)
        results[layer_idx] = probe_results

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

    # Check when it first becomes "good" (e.g., R² > 0.5)
    good_threshold = 0.5
    for layer_idx in sorted(results.keys()):
        if results[layer_idx]["test_r2"] > good_threshold:
            print(f"First layer with R² > {good_threshold}: {layer_idx} (R² = {results[layer_idx]['test_r2']:.4f})")
            break


def plot_results(results: Dict[int, Dict], output_path: str):
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
    ax1.set_title('Entropy Predictability by Layer (Multiple Choice)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # MAE plot
    ax2.plot(layers, train_mae, 'o-', label='Train MAE', alpha=0.7)
    ax2.plot(layers, test_mae, 'o-', label='Test MAE', alpha=0.7)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title('Prediction Error by Layer (Multiple Choice)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Loading model: {BASE_MODEL_NAME}")

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        token=HF_TOKEN
    )

    if MODEL_NAME != BASE_MODEL_NAME:
        try:
            from peft import PeftModel
            print("Loading fine-tuned model")
            model = PeftModel.from_pretrained(model, MODEL_NAME)
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            exit()

    # Get number of layers (handle PEFT model structure)
    if hasattr(model, 'get_base_model'):
        # PEFT model - the base model is wrapped, access via get_base_model()
        base = model.get_base_model()
        num_layers = len(base.model.layers)
    else:
        # Regular model
        num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")

    # Load questions
    questions = load_questions(DATASET_NAME, NUM_QUESTIONS)

    # Apply the same post-load shuffle that capabilities_test.py does
    random.seed(SEED)
    random.shuffle(questions)

    # Build dataset: format questions, get probabilities, compute entropies
    dataset = build_dataset_with_entropies(questions, model, tokenizer)

    # Save dataset with entropies in capabilities_test.py format
    print("\nSaving dataset with entropies...")

    # Convert to capabilities_test.py format: results dict keyed by question ID
    results_dict = {}
    for item in dataset:
        q_id = questions[item["question_id"]].get("id", f"q_{item['question_id']}")
        results_dict[q_id] = {
            "question": item["question"],
            "options": questions[item["question_id"]].get("options", {}),
            "correct_answer": item["correct_answer"],
            "probabilities": item["probabilities"],
            "entropy": item["entropy"],
        }

    # Save in same format as capabilities_test.py
    output_data = {
        "subject_id": f"{DATASET_NAME}_{MODEL_NAME.split('/')[-1]}",
        "timestamp": __import__("time").time(),
        "results": results_dict,
        "run_parameters": {
            "dataset_name": DATASET_NAME,
            "num_questions": NUM_QUESTIONS,
            "model_name": MODEL_NAME,
            "base_model_name": BASE_MODEL_NAME,
            "seed": SEED,
        }
    }

    # Generate output paths
    output_prefix = get_output_prefix()
    dataset_path = f"{output_prefix}_entropy_dataset.json"
    activations_path = f"{output_prefix}_activations.npz"
    probe_path = f"{output_prefix}_entropy_probe.json"
    plot_path = f"{output_prefix}_entropy_probe.png"

    with open(dataset_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Saved to {dataset_path}")

    # Extract activations from all layers
    activations, entropies = extract_all_activations(
        dataset, model, tokenizer, num_layers
    )

    # Save activations for later analysis
    print("\nSaving activations...")
    np.savez_compressed(
        activations_path,
        **{f"layer_{i}": acts for i, acts in activations.items()},
        entropies=entropies
    )
    print(f"Saved to {activations_path}")

    # Train probes
    results = run_all_probes(activations, entropies)

    # Save results
    results_to_save = {
        layer_idx: {k: v for k, v in res.items() if k != "probe"}
        for layer_idx, res in results.items()
    }
    with open(probe_path, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for layer_idx, res in results_to_save.items():
            results_json[layer_idx] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in res.items()
            }
        json.dump(results_json, f, indent=2)
    print(f"Saved results to {probe_path}")

    # Print and plot results
    print_results(results)
    plot_results(results, plot_path)


if __name__ == "__main__":
    main()
