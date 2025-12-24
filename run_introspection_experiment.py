"""
Run introspection experiment: Test whether models use internal entropy signals for meta-cognition.

This script:
1. Loads MC questions and runs them through the model in two modes:
   - Direct: Ask the MC question directly, compute entropy over A/B/C/D
   - Meta: Ask "How confident are you that you know the answer to [Q]?"
2. Extracts activations from both prompt types
3. Trains a linear probe on direct activations → direct entropy
4. Tests whether that probe transfers to meta activations → direct entropy
   (If it does, the model may be "introspecting" on an internal entropy signal)

Key insight: If the model is truly introspecting when answering meta-questions,
it should internally access the same representations it would use for the direct case.
A probe trained on direct data should therefore transfer to meta data.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
from dotenv import load_dotenv
import random
import pickle

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_NAME = "SimpleMC"
NUM_QUESTIONS = 500
MAX_PROMPT_LENGTH = 2500  # Meta prompts are longer due to confidence scale
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100

# Option to load pre-trained probe from run_mc_experiment.py
LOAD_PRETRAINED_PROBE = False
PRETRAINED_PROBE_PATH = "mc_probe_trained.pkl"  # If saved from run_mc_experiment

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# ============================================================================
# PROMPT TEMPLATES (from capabilities_test.py)
# ============================================================================

# Direct MC setup
MC_SETUP_PROMPT = "I'm going to ask you a series of multiple-choice questions. For each one, select the answer you think is best. Respond only with the letter of your choice; do NOT output any other text."

# Meta (nested=Self) setup - confidence about own knowledge
META_SETUP_PROMPT = "I'm going to show you a question, and I want you to tell me your level of confidence that you know the correct answer to this question. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."

META_QUESTION_PROMPT = "How confident are you that you know the correct answer to this question?"

# Confidence scale options (from capabilities_test.py nested="Self")
META_OPTION_DICT = {
    "S": "<5%", "T": "5-10%", "U": "10-20%", "V": "20-40%",
    "W": "40-60%", "X": "60-80%", "Y": "80-90%", "Z": ">90%"
}

META_RANGE_MIDPOINTS = {
    "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
    "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
}


def is_base_model(model_name: str) -> bool:
    """Check if model is a base model (not instruction-tuned)."""
    model_lower = model_name.lower()
    # Base models typically don't have these suffixes
    instruct_indicators = ['instruct', 'chat', '-it', 'rlhf', 'sft', 'dpo']
    return not any(ind in model_lower for ind in instruct_indicators)


def has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    try:
        # Try to apply chat template with a simple message
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True
        )
        return True
    except Exception:
        return False


# ============================================================================
# QUESTION LOADING AND FORMATTING
# ============================================================================

def load_questions(dataset_name: str, num_questions: int = None) -> List[Dict]:
    """Load MC questions using load_and_format_dataset."""
    from load_and_format_datasets import load_and_format_dataset

    questions = load_and_format_dataset(dataset_name, num_questions_needed=num_questions)

    if questions is None:
        raise ValueError(f"Failed to load dataset: {dataset_name}")

    return questions


def _present_question(question_data: Dict) -> str:
    """Format a question for display (from base_game_class.py)."""
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


def _present_nested_question(question_data: Dict, outer_question: str, outer_options: Dict) -> str:
    """Format a nested/meta question for display (from base_game_class.py)."""
    formatted_question = ""
    formatted_question += "-" * 30 + "\n"

    formatted_question += outer_question + "\n"
    formatted_question += "-" * 10 + "\n"

    formatted_question += question_data["question"] + "\n"
    if "options" in question_data:
        for key, value in question_data["options"].items():
            formatted_question += f"  {key}: {value}\n"
    formatted_question += "-" * 10 + "\n"

    if outer_options:
        for key, value in outer_options.items():
            formatted_question += f"  {key}: {value}\n"

    formatted_question += "-" * 30
    return formatted_question


def format_direct_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> Tuple[str, List[str]]:
    """
    Format a direct MC question.

    Args:
        question: The question dict with 'question' and 'options'
        tokenizer: The tokenizer
        use_chat_template: If True, use chat template. If False, use plain text format for base models.

    Returns:
        Tuple of (full_prompt, option_keys)
    """
    q_text = _present_question(question)
    options = list(question["options"].keys())

    options_str = (
        " or ".join(options)
        if len(options) == 2
        else ", ".join(options[:-1]) + f", or {options[-1]}"
    )

    llm_prompt = q_text + f"\nYour choice ({options_str}): "

    if use_chat_template and has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": MC_SETUP_PROMPT},
            {"role": "user", "content": llm_prompt}
        ]

        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Plain text format for base models
        full_prompt = f"{MC_SETUP_PROMPT}\n\n{llm_prompt}"

    return full_prompt, options


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> Tuple[str, List[str]]:
    """
    Format a meta/confidence question.

    Args:
        question: The question dict with 'question' and 'options'
        tokenizer: The tokenizer
        use_chat_template: If True, use chat template. If False, use plain text format for base models.

    Returns:
        Tuple of (full_prompt, meta_option_keys)
    """
    q_text = _present_nested_question(question, META_QUESTION_PROMPT, META_OPTION_DICT)
    meta_options = list(META_OPTION_DICT.keys())

    options_str = ", ".join(meta_options[:-1]) + f", or {meta_options[-1]}"

    llm_prompt = q_text + f"\nYour choice ({options_str}): "

    if use_chat_template and has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": META_SETUP_PROMPT},
            {"role": "user", "content": llm_prompt}
        ]

        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        # Plain text format for base models
        full_prompt = f"{META_SETUP_PROMPT}\n\n{llm_prompt}"

    return full_prompt, meta_options


# ============================================================================
# ENTROPY AND PROBABILITY COMPUTATION
# ============================================================================

def compute_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute entropy from a probability distribution."""
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy)


# ============================================================================
# BATCHED ACTIVATION + LOGIT EXTRACTION
# ============================================================================

class BatchedExtractor:
    """Extract activations and logits in a single batched forward pass."""

    def __init__(self, model, num_layers: int):
        self.model = model
        self.num_layers = num_layers
        self.activations = {}
        self.hooks = []

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            self.activations[layer_idx] = hidden_states.detach()
        return hook

    def register_hooks(self):
        if hasattr(self.model, 'get_base_model'):
            base = self.model.get_base_model()
            layers = base.model.layers
        else:
            layers = self.model.model.layers

        for i, layer in enumerate(layers):
            hook = self._make_hook(i)
            handle = layer.register_forward_hook(hook)
            self.hooks.append(handle)

    def remove_hooks(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def extract_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_token_ids: List[int]
    ) -> Tuple[List[Dict[int, np.ndarray]], List[np.ndarray], List[float]]:
        """
        Extract activations AND compute option probabilities in one forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            option_token_ids: List of token IDs for the options

        Returns:
            layer_activations: List of {layer_idx: activation} dicts, one per batch item
            option_probs: List of probability arrays, one per batch item
            entropies: List of entropy values, one per batch item
        """
        self.activations = {}
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Get last token index for each item in batch
        seq_lengths = attention_mask.sum(dim=1)  # (batch_size,)

        # Extract activations at last token for each batch item
        all_layer_activations = []
        for batch_idx in range(batch_size):
            last_idx = seq_lengths[batch_idx] - 1
            item_activations = {}
            for layer_idx, acts in self.activations.items():
                item_activations[layer_idx] = acts[batch_idx, last_idx, :].cpu().numpy()
            all_layer_activations.append(item_activations)

        # Extract logits and compute probabilities for each batch item
        all_probs = []
        all_entropies = []
        for batch_idx in range(batch_size):
            last_idx = seq_lengths[batch_idx] - 1
            final_logits = outputs.logits[batch_idx, last_idx, :]
            option_logits = final_logits[option_token_ids]
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            entropy = compute_entropy_from_probs(probs)
            all_probs.append(probs)
            all_entropies.append(entropy)

        return all_layer_activations, all_probs, all_entropies


# ============================================================================
# MAIN DATA COLLECTION
# ============================================================================

# Batch size for processing (adjust based on GPU memory)
BATCH_SIZE = 4  # Conservative default; increase if you have more VRAM


def collect_paired_data(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool = True,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Collect activations and entropies for both direct and meta prompts.

    Uses batched processing with combined activation+logit extraction
    for ~4x speedup over the naive approach.

    Returns dict with:
        - direct_activations: {layer_idx: np.array of shape (n_questions, hidden_dim)}
        - meta_activations: {layer_idx: np.array of shape (n_questions, hidden_dim)}
        - direct_entropies: np.array of shape (n_questions,)
        - direct_probs: list of prob arrays
        - meta_entropies: np.array (entropy over confidence options)
        - meta_probs: list of prob arrays over S-Z
        - meta_responses: list of predicted confidence letters
        - questions: the question data
    """
    print(f"Collecting paired data for {len(questions)} questions (batch_size={batch_size})...")

    extractor = BatchedExtractor(model, num_layers)
    extractor.register_hooks()

    # Storage
    direct_layer_acts = {i: [] for i in range(num_layers)}
    meta_layer_acts = {i: [] for i in range(num_layers)}
    direct_entropies = []
    direct_probs_list = []
    meta_entropies = []
    meta_probs_list = []
    meta_responses = []

    model.eval()

    # Pre-compute option token IDs (same for all questions of each type)
    # Direct options vary by question, but meta options are always S-Z
    meta_options = list(META_OPTION_DICT.keys())
    meta_option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in meta_options]

    try:
        # Process in batches
        num_batches = (len(questions) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]

            # ============ DIRECT CASE (batched) ============
            direct_prompts = []
            direct_options_list = []
            for q in batch_questions:
                prompt, options = format_direct_prompt(q, tokenizer, use_chat_template)
                direct_prompts.append(prompt)
                direct_options_list.append(options)

            # For direct case, options can vary per question (A/B/C/D vs A/B etc.)
            # We need to handle this - tokenize each and get token IDs
            # Most MC questions have A/B/C/D, so we'll assume that for batching
            # and fall back to per-item processing if options differ
            first_options = direct_options_list[0]
            all_same_options = all(opts == first_options for opts in direct_options_list)

            if all_same_options:
                # Batch process - all questions have same options
                direct_option_token_ids = [
                    tokenizer.encode(opt, add_special_tokens=False)[0] for opt in first_options
                ]

                inputs = tokenizer(
                    direct_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_PROMPT_LENGTH
                ).to(DEVICE)

                batch_acts, batch_probs, batch_entropies = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    direct_option_token_ids
                )

                for i, (acts, probs, entropy) in enumerate(zip(batch_acts, batch_probs, batch_entropies)):
                    for layer_idx, act in acts.items():
                        direct_layer_acts[layer_idx].append(act)
                    direct_probs_list.append(probs.tolist())
                    direct_entropies.append(entropy)

                del inputs
            else:
                # Fall back to per-item processing if options differ
                for i, (prompt, options) in enumerate(zip(direct_prompts, direct_options_list)):
                    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=MAX_PROMPT_LENGTH
                    ).to(DEVICE)

                    batch_acts, batch_probs, batch_entropies = extractor.extract_batch(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        option_token_ids
                    )

                    for layer_idx, act in batch_acts[0].items():
                        direct_layer_acts[layer_idx].append(act)
                    direct_probs_list.append(batch_probs[0].tolist())
                    direct_entropies.append(batch_entropies[0])

                    del inputs

            # ============ META CASE (batched) ============
            meta_prompts = []
            for q in batch_questions:
                prompt, _ = format_meta_prompt(q, tokenizer, use_chat_template)
                meta_prompts.append(prompt)

            inputs = tokenizer(
                meta_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_PROMPT_LENGTH
            ).to(DEVICE)

            batch_acts, batch_probs, batch_entropies = extractor.extract_batch(
                inputs["input_ids"],
                inputs["attention_mask"],
                meta_option_token_ids
            )

            for i, (acts, probs, entropy) in enumerate(zip(batch_acts, batch_probs, batch_entropies)):
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(probs.tolist())
                meta_entropies.append(entropy)
                # Get the model's confidence response (highest prob option)
                meta_response = meta_options[np.argmax(probs)]
                meta_responses.append(meta_response)

            del inputs

            # Clear memory periodically
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

    finally:
        extractor.remove_hooks()

    # Convert to numpy arrays
    direct_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in direct_layer_acts.items()
    }
    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }

    print(f"Direct activations shape (per layer): {direct_activations[0].shape}")
    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")

    return {
        "direct_activations": direct_activations,
        "meta_activations": meta_activations,
        "direct_entropies": np.array(direct_entropies),
        "direct_probs": direct_probs_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "questions": questions
    }


# ============================================================================
# PROBE TRAINING AND EVALUATION
# ============================================================================

def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    return_components: bool = False
) -> Dict:
    """
    Train a linear probe to predict entropy from activations.

    If return_components=True, also returns the scaler, pca, and probe objects
    for applying to new data.
    """
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Apply PCA if enabled
    pca = None
    if USE_PCA:
        n_components = min(PCA_COMPONENTS, X_train.shape[0], X_train.shape[1])
        pca = PCA(n_components=n_components)
        X_train_final = pca.fit_transform(X_train_scaled)
        X_test_final = pca.transform(X_test_scaled)
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

    result = {
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "predictions": y_pred_test,
        "pca_variance_explained": pca.explained_variance_ratio_.sum() if USE_PCA else None
    }

    if return_components:
        result["scaler"] = scaler
        result["pca"] = pca
        result["probe"] = probe

    return result


def apply_trained_probe(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge
) -> Dict:
    """Apply a pre-trained probe to new data."""
    X_scaled = scaler.transform(X)

    if pca is not None:
        X_final = pca.transform(X_scaled)
    else:
        X_final = X_scaled

    y_pred = probe.predict(X_final)

    r2 = r2_score(y, y_pred)
    mae = mean_absolute_error(y, y_pred)

    return {
        "r2": r2,
        "mae": mae,
        "predictions": y_pred
    }


def run_introspection_analysis(
    direct_activations: Dict[int, np.ndarray],
    meta_activations: Dict[int, np.ndarray],
    direct_entropies: np.ndarray,
    pretrained_probe_path: Optional[str] = None
) -> Dict:
    """
    Run the full introspection analysis:
    1. Train probe on direct activations → direct entropy
    2. Test on held-out direct data (sanity check)
    3. Test on meta activations → direct entropy (THE KEY TEST)
    4. Shuffled baseline control

    Returns results for all layers.
    """
    print(f"\nRunning introspection analysis across {len(direct_activations)} layers...")

    # Split by question index (same split for both direct and meta)
    n_questions = len(direct_entropies)
    indices = np.arange(n_questions)
    train_idx, test_idx = train_test_split(
        indices,
        train_size=TRAIN_SPLIT,
        random_state=SEED
    )

    print(f"Train set: {len(train_idx)} questions, Test set: {len(test_idx)} questions")

    results = {}

    for layer_idx in tqdm(sorted(direct_activations.keys()), desc="Training probes"):
        X_direct = direct_activations[layer_idx]
        X_meta = meta_activations[layer_idx]
        y = direct_entropies

        # Split
        X_direct_train = X_direct[train_idx]
        X_direct_test = X_direct[test_idx]
        X_meta_test = X_meta[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # 1. Train on direct, test on direct (sanity check)
        direct_results = train_probe(
            X_direct_train, y_train,
            X_direct_test, y_test,
            return_components=True
        )

        # 2. Apply direct-trained probe to meta activations (THE KEY TEST)
        meta_results = apply_trained_probe(
            X_meta_test, y_test,
            direct_results["scaler"],
            direct_results["pca"],
            direct_results["probe"]
        )

        # 3. Shuffled baseline: shuffle y_test for meta
        shuffled_y = y_test.copy()
        np.random.shuffle(shuffled_y)
        shuffled_results = apply_trained_probe(
            X_meta_test, shuffled_y,
            direct_results["scaler"],
            direct_results["pca"],
            direct_results["probe"]
        )

        # 4. Train on meta, test on meta (does meta have ANY signal?)
        meta_to_meta_results = train_probe(
            X_meta[train_idx], y_train,
            X_meta_test, y_test,
            return_components=False
        )

        results[layer_idx] = {
            "direct_to_direct": {
                "train_r2": direct_results["train_r2"],
                "test_r2": direct_results["test_r2"],
                "train_mae": direct_results["train_mae"],
                "test_mae": direct_results["test_mae"],
            },
            "direct_to_meta": {
                "r2": meta_results["r2"],
                "mae": meta_results["mae"],
                "predictions": meta_results["predictions"].tolist(),
            },
            "shuffled_baseline": {
                "r2": shuffled_results["r2"],
                "mae": shuffled_results["mae"],
            },
            "meta_to_meta": {
                "train_r2": meta_to_meta_results["train_r2"],
                "test_r2": meta_to_meta_results["test_r2"],
            },
            "pca_variance_explained": direct_results["pca_variance_explained"]
        }

    return results, test_idx


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_behavioral_introspection(
    meta_responses: List[str],
    direct_entropies: np.ndarray,
    test_idx: np.ndarray
) -> Dict:
    """
    Analyze correlation between model's stated confidence and direct entropy.

    If the model is introspecting, its stated confidence should negatively
    correlate with direct entropy (high confidence = low entropy).
    """
    # Convert meta responses to midpoint values
    stated_confidence = np.array([
        META_RANGE_MIDPOINTS.get(r, 0.5) for r in meta_responses
    ])

    # Use test set only for fair comparison with probe results
    test_confidence = stated_confidence[test_idx]
    test_entropy = direct_entropies[test_idx]

    # Correlation (should be negative if introspecting correctly)
    correlation = np.corrcoef(test_confidence, test_entropy)[0, 1]

    # Also compute on full dataset
    full_correlation = np.corrcoef(stated_confidence, direct_entropies)[0, 1]

    return {
        "test_correlation": correlation,
        "full_correlation": full_correlation,
        "test_confidence_mean": float(test_confidence.mean()),
        "test_confidence_std": float(test_confidence.std()),
        "test_entropy_mean": float(test_entropy.mean()),
        "test_entropy_std": float(test_entropy.std()),
    }


def print_results(results: Dict, behavioral: Dict):
    """Print summary of results."""
    print("\n" + "=" * 100)
    print("INTROSPECTION EXPERIMENT RESULTS")
    print("=" * 100)

    print("\n--- Behavioral Analysis ---")
    print(f"Correlation (stated confidence vs direct entropy):")
    print(f"  Full dataset:  {behavioral['full_correlation']:.4f}")
    print(f"  Test set only: {behavioral['test_correlation']:.4f}")
    print(f"  (Negative correlation suggests introspection; positive suggests miscalibration)")

    print("\n--- Probe Analysis by Layer ---")
    print(f"{'Layer':<8} {'Direct→Direct':<15} {'Direct→Meta':<15} {'Meta→Meta':<15} {'Shuffled':<12}")
    print(f"{'':8} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<12}")
    print("-" * 100)

    for layer_idx in sorted(results.keys()):
        res = results[layer_idx]
        d2d = res["direct_to_direct"]["test_r2"]
        d2m = res["direct_to_meta"]["r2"]
        m2m = res["meta_to_meta"]["test_r2"]
        shuf = res["shuffled_baseline"]["r2"]
        print(f"{layer_idx:<8} {d2d:<15.4f} {d2m:<15.4f} {m2m:<15.4f} {shuf:<12.4f}")

    print("=" * 100)

    # Summary statistics
    layers = sorted(results.keys())

    best_d2d_layer = max(layers, key=lambda l: results[l]["direct_to_direct"]["test_r2"])
    best_d2d = results[best_d2d_layer]["direct_to_direct"]["test_r2"]

    best_d2m_layer = max(layers, key=lambda l: results[l]["direct_to_meta"]["r2"])
    best_d2m = results[best_d2m_layer]["direct_to_meta"]["r2"]

    best_m2m_layer = max(layers, key=lambda l: results[l]["meta_to_meta"]["test_r2"])
    best_m2m = results[best_m2m_layer]["meta_to_meta"]["test_r2"]

    print(f"\nBest Direct→Direct: Layer {best_d2d_layer} (R² = {best_d2d:.4f})")
    print(f"Best Direct→Meta:   Layer {best_d2m_layer} (R² = {best_d2m:.4f})")
    print(f"Best Meta→Meta:     Layer {best_m2m_layer} (R² = {best_m2m:.4f})")

    # Transfer ratio
    if best_d2d > 0:
        transfer_ratio = best_d2m / best_d2d
        print(f"\nTransfer ratio (best D→M / best D→D): {transfer_ratio:.2%}")
        if transfer_ratio > 0.5:
            print("  → Strong evidence for introspection!")
        elif transfer_ratio > 0.25:
            print("  → Moderate evidence for introspection")
        else:
            print("  → Weak or no evidence for introspection")


def plot_results(results: Dict, behavioral: Dict, output_path: str = "introspection_results.png"):
    """Create visualization of results."""
    layers = sorted(results.keys())

    d2d_r2 = [results[l]["direct_to_direct"]["test_r2"] for l in layers]
    d2m_r2 = [results[l]["direct_to_meta"]["r2"] for l in layers]
    m2m_r2 = [results[l]["meta_to_meta"]["test_r2"] for l in layers]
    shuffled_r2 = [results[l]["shuffled_baseline"]["r2"] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All R² curves
    ax1 = axes[0, 0]
    ax1.plot(layers, d2d_r2, 'o-', label='Direct→Direct (sanity check)', linewidth=2)
    ax1.plot(layers, d2m_r2, 's-', label='Direct→Meta (introspection test)', linewidth=2)
    ax1.plot(layers, m2m_r2, '^-', label='Meta→Meta (signal existence)', linewidth=2, alpha=0.7)
    ax1.plot(layers, shuffled_r2, 'x--', label='Shuffled baseline', linewidth=1, alpha=0.5, color='gray')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Probe Performance: Can We Predict Direct Entropy?')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Transfer ratio by layer
    ax2 = axes[0, 1]
    transfer_ratios = [d2m_r2[i] / max(d2d_r2[i], 0.001) if d2d_r2[i] > 0.01 else 0
                       for i in range(len(layers))]
    ax2.bar(layers, transfer_ratios, alpha=0.7, color='green')
    ax2.axhline(y=1.0, color='red', linestyle='--', label='Perfect transfer')
    ax2.axhline(y=0.5, color='orange', linestyle='--', label='50% transfer')
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('Transfer Ratio (D→M / D→D)')
    ax2.set_title('Introspection Transfer by Layer')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Direct→Direct vs Direct→Meta scatter
    ax3 = axes[1, 0]
    ax3.scatter(d2d_r2, d2m_r2, c=layers, cmap='viridis', s=100, alpha=0.7)
    ax3.plot([0, max(d2d_r2)], [0, max(d2d_r2)], 'r--', label='y=x (perfect transfer)')
    ax3.set_xlabel('Direct→Direct R²')
    ax3.set_ylabel('Direct→Meta R²')
    ax3.set_title('Transfer Quality (points colored by layer)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax3.collections[0], ax=ax3, label='Layer')

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    summary_text = f"""
INTROSPECTION EXPERIMENT SUMMARY

Behavioral Correlation:
  Full dataset:  {behavioral['full_correlation']:.4f}
  Test set only: {behavioral['test_correlation']:.4f}
  (Negative = introspection, Positive = miscalibration)

Best Layer Results:
  Direct→Direct: Layer {max(layers, key=lambda l: results[l]['direct_to_direct']['test_r2'])}
    R² = {max(d2d_r2):.4f}

  Direct→Meta: Layer {max(layers, key=lambda l: results[l]['direct_to_meta']['r2'])}
    R² = {max(d2m_r2):.4f}

  Meta→Meta: Layer {max(layers, key=lambda l: results[l]['meta_to_meta']['test_r2'])}
    R² = {max(m2m_r2):.4f}

Transfer Ratio: {max(d2m_r2) / max(max(d2d_r2), 0.001):.2%}

Interpretation:
  If Direct→Meta R² is high AND close to Direct→Direct R²,
  this suggests the model accesses similar internal entropy
  representations when answering meta-questions (introspection).
"""
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

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
            print(f"Loading fine-tuned model: {MODEL_NAME}")
            model = PeftModel.from_pretrained(model, MODEL_NAME)
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            exit()

    # Get number of layers
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        num_layers = len(base.model.layers)
    else:
        num_layers = len(model.model.layers)
    print(f"Model has {num_layers} layers")

    # Load questions
    print(f"\nLoading {NUM_QUESTIONS} questions from {DATASET_NAME}...")
    questions = load_questions(DATASET_NAME, NUM_QUESTIONS)
    # Re-seed immediately before shuffle to match capabilities_test.py exactly
    random.seed(SEED)
    random.shuffle(questions)
    print(f"Loaded {len(questions)} questions")

    # Determine whether to use chat template
    use_chat_template = has_chat_template(tokenizer) and not is_base_model(BASE_MODEL_NAME)
    print(f"Using chat template: {use_chat_template}")

    # Collect paired data (direct and meta for each question)
    data = collect_paired_data(questions, model, tokenizer, num_layers, use_chat_template)

    # Save activations
    print("\nSaving activations...")
    np.savez_compressed(
        "introspection_direct_activations.npz",
        **{f"layer_{i}": acts for i, acts in data["direct_activations"].items()},
        entropies=data["direct_entropies"]
    )
    np.savez_compressed(
        "introspection_meta_activations.npz",
        **{f"layer_{i}": acts for i, acts in data["meta_activations"].items()},
        entropies=data["meta_entropies"]
    )
    print("Saved activations to introspection_*_activations.npz")

    # Save paired data (for reproducibility and further analysis)
    paired_data = {
        "direct_entropies": data["direct_entropies"].tolist(),
        "direct_probs": data["direct_probs"],
        "meta_entropies": data["meta_entropies"].tolist(),
        "meta_probs": data["meta_probs"],
        "meta_responses": data["meta_responses"],
        "questions": [
            {
                "id": q.get("id", f"q_{i}"),
                "question": q.get("question", ""),
                "correct_answer": q.get("correct_answer", ""),
                "options": q.get("options", {})
            }
            for i, q in enumerate(data["questions"])
        ],
        "config": {
            "model_name": MODEL_NAME,
            "base_model_name": BASE_MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "num_questions": NUM_QUESTIONS,
            "seed": SEED,
        }
    }
    with open("introspection_paired_data.json", "w") as f:
        json.dump(paired_data, f, indent=2)
    print("Saved paired data to introspection_paired_data.json")

    # Run introspection analysis
    results, test_idx = run_introspection_analysis(
        data["direct_activations"],
        data["meta_activations"],
        data["direct_entropies"]
    )

    # Behavioral analysis
    behavioral = analyze_behavioral_introspection(
        data["meta_responses"],
        data["direct_entropies"],
        test_idx
    )

    # Save results
    results_to_save = {
        "probe_results": {
            str(layer_idx): {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in layer_results.items()
                if not isinstance(v, dict) or k in ["direct_to_direct", "direct_to_meta", "shuffled_baseline", "meta_to_meta"]
            }
            for layer_idx, layer_results in results.items()
        },
        "behavioral": behavioral,
        "test_indices": test_idx.tolist(),
    }

    # Properly serialize nested dicts
    for layer_idx in results_to_save["probe_results"]:
        for key in ["direct_to_direct", "direct_to_meta", "shuffled_baseline", "meta_to_meta"]:
            if key in results_to_save["probe_results"][layer_idx]:
                inner = results_to_save["probe_results"][layer_idx][key]
                for k, v in inner.items():
                    if isinstance(v, np.ndarray):
                        inner[k] = v.tolist()

    with open("introspection_probe_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    print("Saved results to introspection_probe_results.json")

    # Print and plot results
    print_results(results, behavioral)
    plot_results(results, behavioral)

    print("\n✓ Introspection experiment complete!")


if __name__ == "__main__":
    main()
