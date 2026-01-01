"""
Run introspection experiment: Test whether models use internal uncertainty signals for meta-cognition.

This script:
1. Loads MC questions and runs them through the model in two modes:
   - Direct: Ask the MC question directly, compute uncertainty metrics over A/B/C/D
   - Meta: Ask "How confident are you that you know the answer to [Q]?"
2. Extracts activations from both prompt types
3. Computes multiple uncertainty metrics (all saved, one probed per run):
   - Prob-based (nonlinear): entropy, top_prob, margin
   - Logit-based (linear): logit_gap, top_logit
4. Trains a linear probe on direct activations → selected metric
5. Tests whether that probe transfers to meta activations → direct metric
   (If it does, the model may be "introspecting" on an internal uncertainty signal)

Key insight: If the model is truly introspecting when answering meta-questions,
it should internally access the same representations it would use for the direct case.
A probe trained on direct data should therefore transfer to meta data.

Usage:
    python run_introspection_experiment.py --metric logit_gap   # Probe logit_gap (default)
    python run_introspection_experiment.py --metric entropy     # Probe entropy
"""

import argparse
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

from core.model_utils import load_model_and_tokenizer, DEVICE, HF_TOKEN
from tasks import (
    # Direct MC task
    MC_SETUP_PROMPT,
    format_direct_prompt,
    # Confidence task
    STATED_CONFIDENCE_SETUP,
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    STATED_CONFIDENCE_QUESTION,
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    # Other-confidence task (control: estimate human difficulty)
    OTHER_CONFIDENCE_SETUP,
    OTHER_CONFIDENCE_QUESTION,
    format_other_confidence_prompt,
    get_other_confidence_signal,
    # Delegate task
    ANSWER_OR_DELEGATE_SETUP,
    ANSWER_OR_DELEGATE_SYSPROMPT,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    # Unified conversion
    response_to_confidence,
)

load_dotenv()

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"###BASE_MODEL_NAME  # 
DATASET_NAME = "TriviaMC"
NUM_QUESTIONS = 447 if DATASET_NAME.startswith("GP") else 500
# DEVICE imported from core.model_utils
SEED = 42

# Meta-judgment task type:
#   "confidence" - Explicit confidence rating on S-Z scale (default)
#   "delegate"   - Answer vs Delegate choice; confidence = P(Answer)
META_TASK = "confidence"

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Probe training config
TRAIN_SPLIT = 0.8
PROBE_ALPHA = 1000.0
USE_PCA = True
PCA_COMPONENTS = 100

# Option to load pre-trained probe from run_mc_experiment.py
LOAD_PRETRAINED_PROBE = False
PRETRAINED_PROBE_PATH = "mc_probe_trained.pkl"  # If saved from run_mc_experiment

# Available uncertainty metrics (same as mc_entropy_probe.py):
# Prob-based (nonlinear targets - may be harder for linear probes):
#   entropy   - Shannon entropy -sum(p * log(p))
#   top_prob  - P(argmax) - probability of most likely answer
#   margin    - P(top) - P(second) - prob gap between top two
# Logit-based (linear targets - better aligned with linear probes):
#   logit_gap - z(top) - z(second) - logit gap between top two
#   top_logit - z(top) - mean(z) - centered top logit
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
METRIC = "entropy"  # Which metric to probe (set via --metric flag)

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from a model path."""
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_output_prefix(metric: str = None) -> str:
    """Generate output filename prefix based on config.

    Args:
        metric: If provided, include metric in prefix (for metric-specific outputs).
                If None, return base prefix (for shared outputs like activations).
    """
    model_short = get_model_short_name(BASE_MODEL_NAME)
    # Include meta task type in output prefix for clarity
    task_suffix = f"_{META_TASK}" if META_TASK != "confidence" else ""
    metric_suffix = f"_{metric}" if metric else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}{metric_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}{metric_suffix}")


# ============================================================================
# BACKWARD COMPATIBILITY ALIASES (now imported from tasks.py)
# ============================================================================

# Confidence task aliases
META_SETUP_PROMPT = STATED_CONFIDENCE_SETUP
META_QUESTION_PROMPT = STATED_CONFIDENCE_QUESTION
META_OPTION_DICT = STATED_CONFIDENCE_OPTIONS
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS

# Delegate task aliases
DELEGATE_SETUP_PROMPT = ANSWER_OR_DELEGATE_SETUP
DELEGATE_SYSPROMPT = ANSWER_OR_DELEGATE_SYSPROMPT
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS


# Import utility functions from core (instead of duplicating)
from core.model_utils import is_base_model, has_chat_template


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


# Use formatting functions from tasks.py (imported at top)
# format_direct_prompt - imported directly
# Local wrappers for meta tasks to maintain backward compatibility


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> Tuple[str, List[str]]:
    """Format a meta/confidence question using centralized tasks.py logic."""
    return format_stated_confidence_prompt(question, tokenizer, use_chat_template)


def format_delegate_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    trial_index: int = 0
) -> Tuple[str, List[str], Dict[str, str]]:
    """Format a delegate question using centralized tasks.py logic."""
    return format_answer_or_delegate_prompt(
        question, tokenizer, trial_index=trial_index,
        alternate_mapping=True, use_chat_template=use_chat_template
    )


def get_meta_prompt_formatter():
    """Return the appropriate prompt formatter based on META_TASK setting."""
    if META_TASK == "delegate":
        return format_delegate_prompt
    else:
        return format_meta_prompt


def get_meta_options():
    """Return the meta options based on META_TASK setting."""
    if META_TASK == "delegate":
        return DELEGATE_OPTIONS
    else:
        return list(META_OPTION_DICT.keys())


def local_response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """
    Convert a meta response to a confidence value.

    Wrapper around tasks.response_to_confidence that passes the correct task_type.
    """
    task_type = "delegate" if META_TASK == "delegate" else "confidence"
    return response_to_confidence(response, probs, mapping, task_type)


# ============================================================================
# ENTROPY AND PROBABILITY COMPUTATION
# ============================================================================

def compute_entropy_from_probs(probs: np.ndarray) -> float:
    """Compute entropy from a probability distribution."""
    probs = probs / probs.sum()
    probs = probs[probs > 0]
    entropy = -(probs * np.log(probs)).sum()
    return float(entropy)


def compute_uncertainty_metrics(probs: np.ndarray, logits: np.ndarray = None) -> Dict[str, float]:
    """
    Compute multiple uncertainty metrics from probability and logit distributions.

    Args:
        probs: Probability distribution over answer options (sums to 1)
        logits: Raw logits for answer options (before softmax). If None, logit-based
                metrics will be computed from log(probs) as an approximation.

    Returns:
        Dict with keys: entropy, top_prob, margin, logit_gap, top_logit
    """
    # === Prob-based metrics (nonlinear) ===

    # Entropy: -sum(p * log(p))
    entropy = compute_entropy_from_probs(probs)

    # Top probability: P(argmax)
    top_prob = float(np.max(probs))

    # Margin: P(top) - P(second)
    sorted_probs = np.sort(probs)[::-1]  # Descending
    margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else float(sorted_probs[0])

    # === Logit-based metrics (linear - better for linear probes) ===

    # If logits not provided, approximate from log-probs
    # (This loses the constant offset but preserves gaps)
    if logits is None:
        logits = np.log(probs + 1e-10)

    # Sort logits descending
    sorted_logits = np.sort(logits)[::-1]

    # Logit gap: z(top) - z(second)
    # This is the cleanest linear target - invariant to temperature/scale shifts
    logit_gap = float(sorted_logits[0] - sorted_logits[1]) if len(sorted_logits) > 1 else float(sorted_logits[0])

    # Top logit (centered): z(top) - mean(z)
    # Subtracting mean makes it invariant to adding a constant to all logits
    top_logit = float(sorted_logits[0] - np.mean(logits))

    return {
        "entropy": entropy,
        "top_prob": top_prob,
        "margin": margin,
        "logit_gap": logit_gap,
        "top_logit": top_logit,
    }


# ============================================================================
# KV CACHE UTILITIES FOR PREFIX SHARING
# ============================================================================

try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None


def extract_cache_tensors(past_key_values):
    """Extract immutable tensors from cache object."""
    keys, values = [], []
    try:
        num_layers = len(past_key_values)
    except TypeError:
        if hasattr(past_key_values, "to_legacy_cache"):
            return extract_cache_tensors(past_key_values.to_legacy_cache())
        raise ValueError(f"Cannot determine length of cache: {type(past_key_values)}")
    for i in range(num_layers):
        k, v = past_key_values[i]
        keys.append(k)
        values.append(v)
    return keys, values


def create_fresh_cache(key_tensors, value_tensors, expand_size=1):
    """Reconstruct a fresh cache object from tensors."""
    if DynamicCache is not None:
        cache = DynamicCache()
        for i, (k, v) in enumerate(zip(key_tensors, value_tensors)):
            if expand_size > 1:
                k = k.repeat_interleave(expand_size, dim=0)
                v = v.repeat_interleave(expand_size, dim=0)
            cache.update(k, v, i)
        return cache
    else:
        layers = []
        for k, v in zip(key_tensors, value_tensors):
            if expand_size > 1:
                k = k.repeat_interleave(expand_size, dim=0)
                v = v.repeat_interleave(expand_size, dim=0)
            layers.append((k, v))
        return tuple(layers)


# ============================================================================
# BATCHED ACTIVATION + LOGIT EXTRACTION
# ============================================================================

class BatchedExtractor:
    """Extract activations and logits in a single batched forward pass.

    Optimized to:
    1. Store only last-token activations in hooks (reduces memory by seq_len×)
    2. Do single CPU transfer per batch (reduces GPU syncs from layers×batch to 1)
    """

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
            # With left-padding, last position (-1) is always the final real token
            # Store only last-token activations: (batch_size, hidden_dim)
            self.activations[layer_idx] = hidden_states[:, -1, :].detach()
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
    ) -> Tuple[List[Dict[int, np.ndarray]], List[np.ndarray], List[np.ndarray], List[Dict[str, float]]]:
        """
        Extract activations AND compute option probabilities/metrics in one forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            option_token_ids: List of token IDs for the options

        Returns:
            layer_activations: List of {layer_idx: activation} dicts, one per batch item
            option_probs: List of probability arrays, one per batch item
            option_logits: List of logit arrays, one per batch item
            all_metrics: List of metric dicts, one per batch item
        """
        self.activations = {}
        batch_size = input_ids.shape[0]

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

        # Single CPU transfer: stack all layers and transfer at once
        # self.activations[layer_idx] is already (batch_size, hidden_dim) from optimized hook
        stacked = torch.stack([self.activations[i] for i in range(self.num_layers)], dim=0)
        # stacked shape: (num_layers, batch_size, hidden_dim)
        stacked_cpu = stacked.cpu().numpy()

        # Distribute to per-batch-item dicts
        all_layer_activations = []
        for batch_idx in range(batch_size):
            item_activations = {
                layer_idx: stacked_cpu[layer_idx, batch_idx]
                for layer_idx in range(self.num_layers)
            }
            all_layer_activations.append(item_activations)

        # Extract logits and compute probabilities/metrics for each batch item
        all_probs = []
        all_logits = []
        all_metrics = []
        for batch_idx in range(batch_size):
            final_logits = outputs.logits[batch_idx, -1, :]
            option_logits = final_logits[option_token_ids]
            option_logits_np = option_logits.cpu().numpy()
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            metrics = compute_uncertainty_metrics(probs, option_logits_np)
            all_probs.append(probs)
            all_logits.append(option_logits_np)
            all_metrics.append(metrics)

        return all_layer_activations, all_probs, all_logits, all_metrics

    def compute_prefix_cache(self, input_ids: torch.Tensor):
        """Run shared prefix once to get cache snapshot (no hooks)."""
        with torch.inference_mode():
            outputs = self.model(input_ids=input_ids, use_cache=True)
        return extract_cache_tensors(outputs.past_key_values)

    def extract_batch_with_cache(
        self,
        suffix_ids: torch.Tensor,
        prefix_cache_data: Tuple,
        option_token_ids: List[int],
        pad_token_id: int = 0
    ) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray], List[Dict]]:
        """
        Extract activations AND compute option probabilities/metrics using KV cache prefix.

        Args:
            suffix_ids: (batch_size, suffix_len) - left-padded suffix token IDs
            prefix_cache_data: (keys, values) tuple from compute_prefix_cache
            option_token_ids: List of token IDs for the options
            pad_token_id: Token ID used for padding (default 0)

        Returns:
            Same as extract_batch
        """
        self.activations = {}
        batch_size = suffix_ids.shape[0]

        keys, values = prefix_cache_data
        prefix_len = keys[0].shape[2]
        suffix_len = suffix_ids.shape[1]

        # Build attention mask for prefix + suffix
        mask = torch.ones((batch_size, prefix_len + suffix_len), dtype=torch.long, device=suffix_ids.device)
        # Handle left-padding in suffix
        mask[:, prefix_len:] = (suffix_ids != pad_token_id).long()

        with torch.no_grad():
            outputs = self.model(
                input_ids=suffix_ids,
                attention_mask=mask,
                past_key_values=create_fresh_cache(keys, values, expand_size=batch_size),
                use_cache=False
            )

        # Single CPU transfer: stack all layers and transfer at once
        stacked = torch.stack([self.activations[i] for i in range(self.num_layers)], dim=0)
        stacked_cpu = stacked.cpu().numpy()

        # Distribute to per-batch-item dicts
        all_layer_activations = []
        for batch_idx in range(batch_size):
            item_activations = {
                layer_idx: stacked_cpu[layer_idx, batch_idx]
                for layer_idx in range(self.num_layers)
            }
            all_layer_activations.append(item_activations)

        # Extract logits and compute probabilities/metrics for each batch item
        all_probs = []
        all_logits = []
        all_metrics = []
        for batch_idx in range(batch_size):
            final_logits = outputs.logits[batch_idx, -1, :]
            option_logits = final_logits[option_token_ids]
            option_logits_np = option_logits.cpu().numpy()
            probs = torch.softmax(option_logits, dim=-1).cpu().numpy()
            metrics = compute_uncertainty_metrics(probs, option_logits_np)
            all_probs.append(probs)
            all_logits.append(option_logits_np)
            all_metrics.append(metrics)

        return all_layer_activations, all_probs, all_logits, all_metrics


# ============================================================================
# MAIN DATA COLLECTION
# ============================================================================

# Batch size for processing (adjust based on GPU memory)
BATCH_SIZE = 4  # Conservative default; increase if you have more VRAM


def find_common_prefix_length(input_ids_list: List[List[int]]) -> int:
    """Find the length of the common token prefix across all sequences."""
    if not input_ids_list:
        return 0
    ref_ids = input_ids_list[0]
    min_len = min(len(ids) for ids in input_ids_list)
    common_len = 0
    for i in range(min_len):
        if all(ids[i] == ref_ids[i] for ids in input_ids_list):
            common_len += 1
        else:
            break
    return common_len


def process_prompts_with_prefix_cache(
    prompts: List[str],
    options_list: List[List[str]],
    tokenizer,
    extractor: BatchedExtractor,
    batch_size: int,
    desc: str,
    collect_activations: bool = True
) -> Tuple[List[Dict], List[np.ndarray], List[np.ndarray], List[Dict], List[str]]:
    """
    Process prompts efficiently with prefix caching.

    When prompts share a common prefix (e.g., same system message and instructions),
    computes the KV cache for the prefix once and reuses it for all suffixes.

    Args:
        prompts: List of full prompt strings
        options_list: List of option lists (one per prompt)
        tokenizer: The tokenizer
        extractor: BatchedExtractor instance (with hooks registered)
        batch_size: Batch size for processing
        desc: Progress bar description
        collect_activations: Whether to collect layer activations

    Returns:
        (layer_activations, probs, logits, metrics, responses)
    """
    # 1. Tokenize all prompts
    encodings = tokenizer(prompts, add_special_tokens=False)
    input_ids_list = encodings["input_ids"]

    # 2. Find common prefix
    common_len = find_common_prefix_length(input_ids_list)

    # 3. Compute prefix cache if prefix is substantial (>20 tokens)
    MIN_PREFIX_FOR_CACHE = 20
    if common_len > MIN_PREFIX_FOR_CACHE:
        print(f"  Found common prefix ({common_len} tokens). Computing prefix cache...")
        prefix_ids = torch.tensor([input_ids_list[0][:common_len]], device=DEVICE)
        prefix_cache = extractor.compute_prefix_cache(prefix_ids)
        suffixes = [ids[common_len:] for ids in input_ids_list]
        use_cache = True
    else:
        if common_len > 0:
            print(f"  Common prefix too short ({common_len} tokens). Using standard processing.")
        suffixes = input_ids_list
        prefix_cache = None
        use_cache = False

    # Get pad token id from tokenizer
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # 4. Process in batches
    results_acts = []
    results_probs = []
    results_logits = []
    results_metrics = []
    results_responses = []

    for b in tqdm(range(0, len(prompts), batch_size), desc=desc):
        batch_suffixes = suffixes[b:b+batch_size]
        batch_opts = options_list[b:b+batch_size]
        actual_batch_size = len(batch_suffixes)

        # Left-pad suffixes to same length
        max_len = max(len(s) for s in batch_suffixes)
        padded = torch.full((actual_batch_size, max_len), pad_token_id, dtype=torch.long, device=DEVICE)
        for i, s in enumerate(batch_suffixes):
            padded[i, max_len-len(s):] = torch.tensor(s, dtype=torch.long, device=DEVICE)

        # Get option token IDs (assume all items in batch have same options)
        opt_ids = [tokenizer.encode(o, add_special_tokens=False)[0] for o in batch_opts[0]]

        if use_cache:
            acts, probs, logits, metrics = extractor.extract_batch_with_cache(
                padded, prefix_cache, opt_ids, pad_token_id=pad_token_id
            )
        else:
            # Build full inputs with attention mask
            mask = (padded != pad_token_id).long()
            acts, probs, logits, metrics = extractor.extract_batch(padded, mask, opt_ids)

        if collect_activations:
            results_acts.extend(acts)
        results_probs.extend(probs)
        results_logits.extend(logits)
        results_metrics.extend(metrics)

        # Determine responses based on argmax
        for i, p in enumerate(probs):
            results_responses.append(batch_opts[i][np.argmax(p)])

    return results_acts, results_probs, results_logits, results_metrics, results_responses


def collect_paired_data(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool = True,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Collect activations and uncertainty metrics for both direct and meta prompts.

    Uses batched processing with combined activation+logit extraction
    and KV cache prefix sharing for meta prompts (~2-3x overall speedup).

    Returns dict with:
        - direct_activations: {layer_idx: np.array of shape (n_questions, hidden_dim)}
        - meta_activations: {layer_idx: np.array of shape (n_questions, hidden_dim)}
        - direct_metrics: {metric_name: np.array of shape (n_questions,)}
        - direct_probs: list of prob arrays
        - direct_logits: list of logit arrays
        - meta_entropies: np.array (entropy over confidence options)
        - meta_probs: list of prob arrays over S-Z (or [P("1"), P("2")] for delegate)
        - meta_responses: list of predicted confidence letters (or "1"/"2" for delegate)
        - meta_mappings: list of mappings for delegate task (None for confidence)
        - questions: the question data
    """
    print(f"Collecting paired data for {len(questions)} questions (batch_size={batch_size})...")

    extractor = BatchedExtractor(model, num_layers)
    extractor.register_hooks()

    # Storage
    direct_layer_acts = {i: [] for i in range(num_layers)}
    direct_metrics_lists = {metric: [] for metric in AVAILABLE_METRICS}
    direct_probs_list = []
    direct_logits_list = []

    model.eval()

    # Pre-compute option token IDs for meta task
    meta_options = get_meta_options()

    try:
        # ============ PHASE 1: DIRECT PROMPTS ============
        print("\nProcessing direct prompts...")
        num_batches = (len(questions) + batch_size - 1) // batch_size

        for batch_idx in tqdm(range(num_batches), desc="Direct prompts"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(questions))
            batch_questions = questions[start_idx:end_idx]

            direct_prompts = []
            direct_options_list = []
            for q in batch_questions:
                prompt, options = format_direct_prompt(q, tokenizer, use_chat_template)
                direct_prompts.append(prompt)
                direct_options_list.append(options)

            # Check if all questions have same options (most MC questions do)
            first_options = direct_options_list[0]
            all_same_options = all(opts == first_options for opts in direct_options_list)

            if all_same_options:
                # Batch process
                direct_option_token_ids = [
                    tokenizer.encode(opt, add_special_tokens=False)[0] for opt in first_options
                ]

                inputs = tokenizer(
                    direct_prompts,
                    return_tensors="pt",
                    padding=True,
                ).to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    direct_option_token_ids
                )

                for acts, probs, logits, metrics in zip(batch_acts, batch_probs, batch_logits, batch_metrics):
                    for layer_idx, act in acts.items():
                        direct_layer_acts[layer_idx].append(act)
                    direct_probs_list.append(probs.tolist())
                    direct_logits_list.append(logits.tolist())
                    for metric_name, metric_val in metrics.items():
                        direct_metrics_lists[metric_name].append(metric_val)

                del inputs
            else:
                # Fall back to per-item processing
                for prompt, options in zip(direct_prompts, direct_options_list):
                    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
                    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                    batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                        inputs["input_ids"],
                        inputs["attention_mask"],
                        option_token_ids
                    )

                    for layer_idx, act in batch_acts[0].items():
                        direct_layer_acts[layer_idx].append(act)
                    direct_probs_list.append(batch_probs[0].tolist())
                    direct_logits_list.append(batch_logits[0].tolist())
                    for metric_name, metric_val in batch_metrics[0].items():
                        direct_metrics_lists[metric_name].append(metric_val)

                    del inputs

            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()

        # ============ PHASE 2: META PROMPTS ============
        print("\nProcessing meta prompts...")

        if META_TASK == "delegate":
            # Delegate task: process individually due to alternating mapping
            meta_layer_acts = {i: [] for i in range(num_layers)}
            meta_probs_list = []
            meta_entropies = []
            meta_responses = []
            meta_mappings = []
            meta_option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in meta_options]

            for trial_idx, q in enumerate(tqdm(questions, desc="Delegate prompts")):
                prompt, _, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_idx)
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    meta_option_token_ids
                )

                for layer_idx, act in batch_acts[0].items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(batch_probs[0].tolist())
                meta_entropies.append(batch_metrics[0]["entropy"])
                meta_response = meta_options[np.argmax(batch_probs[0])]
                meta_responses.append(meta_response)
                meta_mappings.append(mapping)

                del inputs

                if (trial_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        else:
            # Confidence task: use prefix caching (all prompts share same prefix)
            meta_prompts = []
            meta_options_list = []
            for q in questions:
                prompt, opts = format_meta_prompt(q, tokenizer, use_chat_template)
                meta_prompts.append(prompt)
                meta_options_list.append(opts)

            # Use prefix caching for efficiency
            meta_acts, meta_probs_raw, _, meta_metrics_raw, meta_responses = process_prompts_with_prefix_cache(
                meta_prompts,
                meta_options_list,
                tokenizer,
                extractor,
                batch_size,
                desc="Meta prompts (with prefix cache)"
            )

            # Convert to expected format
            meta_layer_acts = {i: [] for i in range(num_layers)}
            for acts in meta_acts:
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)

            meta_probs_list = [p.tolist() for p in meta_probs_raw]
            meta_entropies = [m["entropy"] for m in meta_metrics_raw]
            meta_mappings = [None] * len(questions)  # No mapping for confidence task

    finally:
        extractor.remove_hooks()

    # Convert to numpy arrays
    direct_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in direct_layer_acts.items()
    }
    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }
    direct_metrics = {
        metric: np.array(values) for metric, values in direct_metrics_lists.items()
    }

    print(f"\nDirect activations shape (per layer): {direct_activations[0].shape}")
    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")
    print(f"\nDirect uncertainty metrics:")
    for metric_name, values in direct_metrics.items():
        print(f"  {metric_name}: range=[{values.min():.3f}, {values.max():.3f}], "
              f"mean={values.mean():.3f}, std={values.std():.3f}")

    return {
        "direct_activations": direct_activations,
        "meta_activations": meta_activations,
        "direct_metrics": direct_metrics,
        "direct_probs": direct_probs_list,
        "direct_logits": direct_logits_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "meta_mappings": meta_mappings,
        "questions": questions
    }


def collect_meta_only(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool,
    mc_data: Dict,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Collect only meta prompt data, reusing direct activations from mc_entropy_probe.py.

    This is much faster than collect_paired_data when MC data already exists.
    Uses KV cache prefix sharing for additional speedup on confidence task.
    """
    print(f"Collecting meta data only for {len(questions)} questions (reusing direct activations)...")

    extractor = BatchedExtractor(model, num_layers)
    extractor.register_hooks()

    model.eval()

    # Meta options depend on META_TASK
    meta_options = get_meta_options()

    try:
        if META_TASK == "delegate":
            # Delegate task: process individually due to alternating mapping
            meta_layer_acts = {i: [] for i in range(num_layers)}
            meta_probs_list = []
            meta_entropies = []
            meta_responses = []
            meta_mappings = []
            meta_option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in meta_options]

            for trial_idx, q in enumerate(tqdm(questions, desc="Delegate prompts")):
                prompt, _, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_idx)
                inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

                batch_acts, batch_probs, batch_logits, batch_metrics = extractor.extract_batch(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    meta_option_token_ids
                )

                for layer_idx, act in batch_acts[0].items():
                    meta_layer_acts[layer_idx].append(act)
                meta_probs_list.append(batch_probs[0].tolist())
                meta_entropies.append(batch_metrics[0]["entropy"])
                meta_response = meta_options[np.argmax(batch_probs[0])]
                meta_responses.append(meta_response)
                meta_mappings.append(mapping)

                del inputs

                if (trial_idx + 1) % 50 == 0:
                    torch.cuda.empty_cache()
        else:
            # Confidence task: use prefix caching (all prompts share same prefix)
            meta_prompts = []
            meta_options_list = []
            for q in questions:
                prompt, opts = format_meta_prompt(q, tokenizer, use_chat_template)
                meta_prompts.append(prompt)
                meta_options_list.append(opts)

            # Use prefix caching for efficiency
            meta_acts, meta_probs_raw, _, meta_metrics_raw, meta_responses = process_prompts_with_prefix_cache(
                meta_prompts,
                meta_options_list,
                tokenizer,
                extractor,
                batch_size,
                desc="Meta prompts (with prefix cache)"
            )

            # Convert to expected format
            meta_layer_acts = {i: [] for i in range(num_layers)}
            for acts in meta_acts:
                for layer_idx, act in acts.items():
                    meta_layer_acts[layer_idx].append(act)

            meta_probs_list = [p.tolist() for p in meta_probs_raw]
            meta_entropies = [m["entropy"] for m in meta_metrics_raw]
            meta_mappings = [None] * len(questions)

    finally:
        extractor.remove_hooks()

    meta_activations = {
        layer_idx: np.array(acts) for layer_idx, acts in meta_layer_acts.items()
    }

    print(f"Meta activations shape (per layer): {meta_activations[0].shape}")

    # Build direct data from mc_data
    # mc_data may have "direct_metrics" dict (new format) or just "direct_entropies" (old format)
    direct_probs_list = [m.get("probabilities", []) for m in mc_data["metadata"]]
    direct_logits_list = [m.get("logits", []) for m in mc_data["metadata"]]

    # Handle both old (entropies only) and new (all metrics) mc_data formats
    if "direct_metrics" in mc_data:
        direct_metrics = mc_data["direct_metrics"]
    else:
        # Old format: only has entropies, need to compute metrics from metadata
        direct_metrics = {metric: [] for metric in AVAILABLE_METRICS}
        for m in mc_data["metadata"]:
            probs = np.array(m.get("probabilities", []))
            logits = np.array(m.get("logits", [])) if m.get("logits") else None
            if len(probs) > 0:
                item_metrics = compute_uncertainty_metrics(probs, logits)
                for metric_name, metric_val in item_metrics.items():
                    direct_metrics[metric_name].append(metric_val)
        direct_metrics = {k: np.array(v) for k, v in direct_metrics.items()}

    return {
        "direct_activations": mc_data["direct_activations"],
        "meta_activations": meta_activations,
        "direct_metrics": direct_metrics,
        "direct_probs": direct_probs_list,
        "direct_logits": direct_logits_list,
        "meta_entropies": np.array(meta_entropies),
        "meta_probs": meta_probs_list,
        "meta_responses": meta_responses,
        "meta_mappings": meta_mappings,
        "questions": questions
    }


def collect_other_confidence(
    questions: List[Dict],
    model,
    tokenizer,
    num_layers: int,
    use_chat_template: bool,
    batch_size: int = BATCH_SIZE
) -> Dict:
    """
    Collect other-confidence (human difficulty estimation) responses.

    This is a control task: asks model to estimate what % of college-educated
    people would know the answer (instead of asking about its own confidence).

    If the model is truly introspecting on its own uncertainty, the self-confidence
    task should correlate more strongly with its actual uncertainty metrics than
    this other-confidence task.

    Uses KV cache prefix sharing for efficiency.

    Returns dict with:
        - other_probs: list of prob arrays over S-Z options
        - other_responses: list of predicted confidence letters
        - other_signals: list of expected confidence values (weighted avg of midpoints)
    """
    print(f"\nCollecting other-confidence (control) data for {len(questions)} questions...")

    extractor = BatchedExtractor(model, num_layers)
    extractor.register_hooks()

    model.eval()

    try:
        # Format all prompts
        other_prompts = []
        other_options_list = []
        for q in questions:
            prompt, opts = format_other_confidence_prompt(q, tokenizer, use_chat_template)
            other_prompts.append(prompt)
            other_options_list.append(opts)

        # Use prefix caching for efficiency (no activations needed)
        _, other_probs_raw, _, _, other_responses = process_prompts_with_prefix_cache(
            other_prompts,
            other_options_list,
            tokenizer,
            extractor,
            batch_size,
            desc="Other-confidence (with prefix cache)",
            collect_activations=False
        )

        # Convert to expected format and compute signals
        other_probs_list = [p.tolist() for p in other_probs_raw]
        other_signals = [get_other_confidence_signal(p) for p in other_probs_raw]

    finally:
        extractor.remove_hooks()

    print(f"Other-confidence signals: mean={np.mean(other_signals):.3f}, std={np.std(other_signals):.3f}")

    return {
        "other_probs": other_probs_list,
        "other_responses": other_responses,
        "other_signals": np.array(other_signals),
    }


# ============================================================================
# PROBE TRAINING AND EVALUATION
# ============================================================================

def extract_direction(
    scaler: StandardScaler,
    pca: Optional[PCA],
    probe: Ridge
) -> np.ndarray:
    """
    Extract normalized direction from trained probe in original activation space.

    Maps the probe weights back through PCA (if used) and standardization
    to get the direction in the original activation space.
    """
    coef = probe.coef_

    if pca is not None:
        # Map from PCA space back to scaled space
        direction_scaled = pca.components_.T @ coef
    else:
        direction_scaled = coef

    # Undo standardization scaling
    direction_original = direction_scaled / scaler.scale_

    # Normalize to unit length
    direction_original = direction_original / np.linalg.norm(direction_original)

    return direction_original


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
    """Apply a pre-trained probe to new data using the original scaler."""
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


def apply_probe_with_separate_scaling(
    X: np.ndarray,
    y: np.ndarray,
    pca: Optional[PCA],
    probe: Ridge
) -> Dict:
    """
    Apply a pre-trained probe to new data with SEPARATE standardization.

    This fixes the distribution shift problem: instead of applying the original
    scaler (fit on direct activations) to meta activations, we standardize
    meta activations using their own statistics. This puts both in a comparable
    standardized space without extreme z-scores.

    The assumption is that the probe learned "pattern X maps to entropy Y" in
    standardized space, and both direct and meta activations have similar
    relative structure even if their absolute statistics differ.
    """
    # Standardize X using its own statistics (not the original scaler)
    new_scaler = StandardScaler()
    X_scaled = new_scaler.fit_transform(X)

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
    pretrained_probe_path: Optional[str] = None,
    extract_directions: bool = True
) -> Dict:
    """
    Run the full introspection analysis:
    1. Train probe on direct activations → direct entropy
    2. Test on held-out direct data (sanity check)
    3. Test on meta activations → direct entropy (THE KEY TEST)
    4. Shuffled baseline control

    If extract_directions=True, also extracts the entropy probe direction
    from each layer for use in steering/ablation experiments.

    Returns (results, test_idx, directions) where directions is a dict
    mapping layer_idx -> direction vector (or None if extract_directions=False).
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
    directions = {} if extract_directions else None

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
        # 2a. Original approach: use same scaler (causes distribution shift issues)
        meta_results_shared_scaler = apply_trained_probe(
            X_meta_test, y_test,
            direct_results["scaler"],
            direct_results["pca"],
            direct_results["probe"]
        )

        # 2b. Fixed approach: standardize meta activations separately
        meta_results_separate_scaler = apply_probe_with_separate_scaling(
            X_meta_test, y_test,
            direct_results["pca"],
            direct_results["probe"]
        )

        # 3. Shuffled baseline: train probe on shuffled labels, test on real labels
        # This gives the expected R² under the null hypothesis (no real signal)
        shuffled_y_train = y_train.copy()
        np.random.shuffle(shuffled_y_train)
        shuffled_results = train_probe(
            X_direct_train, shuffled_y_train,
            X_direct_test, y_test,
            return_components=False
        )

        # 4. Train on meta, test on meta (does meta have ANY signal?)
        meta_to_meta_results = train_probe(
            X_meta[train_idx], y_train,
            X_meta_test, y_test,
            return_components=False
        )

        # 5. Extract entropy direction from direct→direct probe (for steering)
        if extract_directions:
            directions[layer_idx] = extract_direction(
                direct_results["scaler"],
                direct_results["pca"],
                direct_results["probe"]
            )

        results[layer_idx] = {
            "direct_to_direct": {
                "train_r2": direct_results["train_r2"],
                "test_r2": direct_results["test_r2"],
                "train_mae": direct_results["train_mae"],
                "test_mae": direct_results["test_mae"],
                "predictions": direct_results["predictions"].tolist(),
            },
            "direct_to_meta": {
                # Keep original (shared scaler) for backwards compatibility
                "r2": meta_results_shared_scaler["r2"],
                "mae": meta_results_shared_scaler["mae"],
                "predictions": meta_results_shared_scaler["predictions"].tolist(),
            },
            "direct_to_meta_fixed": {
                # Separate scaling - the corrected transfer test
                "r2": meta_results_separate_scaler["r2"],
                "mae": meta_results_separate_scaler["mae"],
                "predictions": meta_results_separate_scaler["predictions"].tolist(),
            },
            "shuffled_baseline": {
                "r2": shuffled_results["test_r2"],
                "mae": shuffled_results["test_mae"],
            },
            "meta_to_meta": {
                "train_r2": meta_to_meta_results["train_r2"],
                "test_r2": meta_to_meta_results["test_r2"],
            },
            "pca_variance_explained": direct_results["pca_variance_explained"]
        }

    return results, test_idx, directions


# ============================================================================
# ANALYSIS AND VISUALIZATION
# ============================================================================

def analyze_behavioral_introspection(
    meta_responses: List[str],
    direct_entropies: np.ndarray,
    test_idx: np.ndarray,
    meta_probs: List[List[float]] = None,
    meta_mappings: List[Dict[str, str]] = None,
    direct_probs: List[List[float]] = None,
    questions: List[Dict] = None
) -> Dict:
    """
    Analyze correlation between model's stated confidence and direct entropy.

    If the model is introspecting, its stated confidence should negatively
    correlate with direct entropy (high confidence = low entropy).

    For delegate task, confidence = P(Answer) from meta_probs.
    For confidence task, confidence = weighted average of midpoints (expected value).

    For delegate task, also computes:
    - Delegation rate: fraction of questions delegated
    - Self-answer accuracy: accuracy when model chooses to answer itself
    - Teammate accuracy when delegated (fixed at 50%)
    """
    # Convert meta responses to confidence values
    # For confidence task, use weighted average of midpoints (expected value)
    # For delegate task, use P(Answer)
    if META_TASK == "delegate":
        stated_confidence = np.array([
            local_response_to_confidence(r, np.array(p) if p else None, m)
            for r, p, m in zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses),
                meta_mappings or [None] * len(meta_responses)
            )
        ])
    else:
        # Confidence task: weighted average of midpoints
        from tasks import get_stated_confidence_signal
        stated_confidence = np.array([
            get_stated_confidence_signal(np.array(p)) if p else 0.5
            for p in meta_probs
        ])

    # Split into test set (same split used for probe evaluation)
    test_confidence = stated_confidence[test_idx]
    test_entropy = direct_entropies[test_idx]

    # Subsample-to-m interval approach for CIs
    # This gives intervals calibrated to test size, centered on point estimates
    from scipy import stats
    n_subsamples = 200  # K iterations
    n = len(direct_entropies)
    m = len(test_idx)  # test size

    # Helper: Fisher z transform and inverse
    def fisher_z(r):
        # Clip to avoid inf at r=±1
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def fisher_z_inv(z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    # Full correlation (point estimate on all data - stable)
    full_r = np.corrcoef(stated_confidence, direct_entropies)[0, 1]
    full_z = fisher_z(full_r)

    # Test correlation (point estimate on specific test_idx)
    test_r = np.corrcoef(test_confidence, test_entropy)[0, 1]
    n_test = len(test_confidence)

    # Subsample-to-m interval for FULL correlation
    # Subsample m items, compute correlation, get deviation from full_r in z-space
    full_subsample_deviations = []
    for k in range(n_subsamples):
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=m, replace=False)
        sub_r = np.corrcoef(stated_confidence[idx], direct_entropies[idx])[0, 1]
        if not np.isnan(sub_r):
            sub_z = fisher_z(sub_r)
            full_subsample_deviations.append(sub_z - full_z)

    # Get percentiles of deviations, then map back to correlation space
    dev_lower = np.percentile(full_subsample_deviations, 2.5)
    dev_upper = np.percentile(full_subsample_deviations, 97.5)
    full_ci_lower = fisher_z_inv(full_z + dev_lower)
    full_ci_upper = fisher_z_inv(full_z + dev_upper)
    full_ci_std = np.std([fisher_z_inv(full_z + d) for d in full_subsample_deviations])

    # Subsample-to-m interval for TEST correlation (random test subsets)
    # This shows how test correlation varies across different random splits
    test_subsample_correlations = []
    for k in range(n_subsamples):
        rng = np.random.default_rng(k)
        idx = rng.choice(n, size=m, replace=False)
        sub_r = np.corrcoef(stated_confidence[idx], direct_entropies[idx])[0, 1]
        if not np.isnan(sub_r):
            test_subsample_correlations.append(sub_r)

    test_ci_mean = np.mean(test_subsample_correlations)
    test_ci_std = np.std(test_subsample_correlations)
    test_ci_lower = np.percentile(test_subsample_correlations, 2.5)
    test_ci_upper = np.percentile(test_subsample_correlations, 97.5)

    # P-values using t-distribution
    if abs(full_r) < 1 and n > 2:
        t_stat = full_r * np.sqrt((n - 2) / (1 - full_r**2))
        full_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        full_pvalue = np.nan

    if abs(test_r) < 1 and n_test > 2:
        t_stat = test_r * np.sqrt((n_test - 2) / (1 - test_r**2))
        test_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n_test-2))
    else:
        test_pvalue = np.nan

    # Partial correlation (always computed - equals Pearson when no controls)
    import pandas as pd
    from logres_helpers import partial_correlation_on_decision

    # Control variables for partial correlation (currently empty, ready for future use)
    control_series_list = []

    full_partial_result = partial_correlation_on_decision(
        dv_series=pd.Series(direct_entropies, name='entropy'),
        iv_series=pd.Series(stated_confidence, name='confidence'),
        control_series_list=control_series_list
    )

    result = {
        "full_correlation": float(full_r),
        "full_correlation_pvalue": float(full_pvalue),
        "test_correlation": float(test_r),
        "test_correlation_pvalue": float(test_pvalue),
        # Subsample-to-m interval for full: centered on full_r, width calibrated to test size
        "full_correlation_ci95": [float(full_ci_lower), float(full_ci_upper)],
        "full_correlation_ci_std": float(full_ci_std),
        # Subsample-to-m interval for test: distribution of correlations at test size
        "test_correlation_ci95": [float(test_ci_lower), float(test_ci_upper)],
        "test_correlation_ci_mean": float(test_ci_mean),
        "test_correlation_ci_std": float(test_ci_std),
        "n_subsamples": n_subsamples,
        # Partial correlation (equals Pearson when no controls)
        "partial_correlation": float(full_partial_result['correlation']),
        "partial_correlation_ci95": [float(full_partial_result['ci_lower']), float(full_partial_result['ci_upper'])],
        "partial_correlation_pvalue": float(full_partial_result['p_value']),
        "partial_correlation_controls": [s.name for s in control_series_list],
        "n_samples_full": n,
        "n_samples_test": n_test,
        "test_confidence_mean": float(test_confidence.mean()),
        "test_confidence_std": float(test_confidence.std()),
        "test_entropy_mean": float(test_entropy.mean()),
        "test_entropy_std": float(test_entropy.std()),
    }

    # Delegate-specific metrics
    if META_TASK == "delegate" and meta_mappings is not None:
        # Determine delegation decisions based on response and mapping
        delegated = []
        self_answers = []
        for i, (response, mapping) in enumerate(zip(meta_responses, meta_mappings)):
            if mapping is not None:
                decision = mapping.get(response, "Unknown")
                is_delegated = (decision == "Delegate")
                delegated.append(is_delegated)
                if not is_delegated:
                    self_answers.append(i)

        delegation_rate = sum(delegated) / len(delegated) if delegated else 0.0
        result["delegation_rate"] = float(delegation_rate)
        result["num_delegated"] = sum(delegated)
        result["num_self_answered"] = len(self_answers)

        # Compute self-answer accuracy if we have the data
        if direct_probs is not None and questions is not None and self_answers:
            self_correct = 0
            for idx in self_answers:
                if idx < len(direct_probs) and idx < len(questions):
                    probs = direct_probs[idx]
                    q = questions[idx]
                    if probs and "correct_answer" in q and "options" in q:
                        options = list(q["options"].keys())
                        predicted_answer = options[np.argmax(probs)]
                        if predicted_answer == q["correct_answer"]:
                            self_correct += 1

            self_answer_accuracy = self_correct / len(self_answers)
            result["self_answer_accuracy"] = float(self_answer_accuracy)
            result["self_correct"] = self_correct

            # Teammate accuracy is fixed at 50% (by design of the game)
            result["teammate_accuracy"] = 0.5

            # Team score: self-answered correct + delegated * 0.5
            team_score = self_correct + sum(delegated) * 0.5
            result["team_score"] = float(team_score)
            result["team_score_normalized"] = float(team_score / len(delegated)) if delegated else 0.0

    return result


def analyze_other_confidence_control(
    other_signals: np.ndarray,
    self_confidence: np.ndarray,
    direct_metric: np.ndarray,
    test_idx: np.ndarray
) -> Dict:
    """
    Analyze other-confidence (human difficulty estimation) as a control.

    Compares:
    1. Correlation of self-confidence vs direct metric (introspection)
    2. Correlation of other-confidence vs direct metric (control)
    3. Correlation between self and other confidence

    If the model is truly introspecting, self-confidence should correlate
    more strongly with its own uncertainty than other-confidence does.
    """
    from scipy import stats

    n = len(direct_metric)
    n_test = len(test_idx)

    # Helper: Fisher z transform
    def fisher_z(r):
        r = np.clip(r, -0.9999, 0.9999)
        return 0.5 * np.log((1 + r) / (1 - r))

    def fisher_z_inv(z):
        return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

    # 1. Self-confidence vs direct metric (already computed in behavioral analysis,
    #    but we recompute for completeness and to use for comparison)
    self_r = np.corrcoef(self_confidence, direct_metric)[0, 1]
    if abs(self_r) < 1 and n > 2:
        t_stat = self_r * np.sqrt((n - 2) / (1 - self_r**2))
        self_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        self_pvalue = np.nan

    # 2. Other-confidence vs direct metric
    other_r = np.corrcoef(other_signals, direct_metric)[0, 1]
    if abs(other_r) < 1 and n > 2:
        t_stat = other_r * np.sqrt((n - 2) / (1 - other_r**2))
        other_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        other_pvalue = np.nan

    # 3. Self vs Other confidence correlation
    self_other_r = np.corrcoef(self_confidence, other_signals)[0, 1]
    if abs(self_other_r) < 1 and n > 2:
        t_stat = self_other_r * np.sqrt((n - 2) / (1 - self_other_r**2))
        self_other_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
    else:
        self_other_pvalue = np.nan

    # 4. Compare correlations: is self-confidence significantly more correlated
    #    with direct metric than other-confidence?
    # Use Steiger's Z-test for comparing dependent correlations
    # (self vs metric) vs (other vs metric), where self and other are correlated
    r12 = self_r  # self vs metric
    r13 = other_r  # other vs metric
    r23 = self_other_r  # self vs other

    # Steiger's Z formula for comparing dependent correlations
    if abs(r12) < 1 and abs(r13) < 1 and abs(r23) < 1:
        # Average correlation
        r_avg = (r12 + r13) / 2

        # Hotelling-Williams t-test approximation
        f = (1 - r23) / (2 * (1 - r_avg**2)) if abs(r_avg) < 1 else 1
        det = 1 - r12**2 - r13**2 - r23**2 + 2*r12*r13*r23

        if det > 0:
            t_stat = (r12 - r13) * np.sqrt((n - 3) * (1 + r23) / (2 * det))
            diff_pvalue = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-3))
        else:
            diff_pvalue = np.nan
    else:
        diff_pvalue = np.nan

    # Bootstrap CI for the difference in correlations
    n_bootstrap = 1000
    diff_samples = []
    for b in range(n_bootstrap):
        rng = np.random.default_rng(b)
        idx = rng.choice(n, size=n, replace=True)
        self_r_b = np.corrcoef(self_confidence[idx], direct_metric[idx])[0, 1]
        other_r_b = np.corrcoef(other_signals[idx], direct_metric[idx])[0, 1]
        if not np.isnan(self_r_b) and not np.isnan(other_r_b):
            diff_samples.append(self_r_b - other_r_b)

    if diff_samples:
        diff_mean = np.mean(diff_samples)
        diff_ci_lower = np.percentile(diff_samples, 2.5)
        diff_ci_upper = np.percentile(diff_samples, 97.5)
    else:
        diff_mean = self_r - other_r
        diff_ci_lower = diff_ci_upper = np.nan

    return {
        # Self-confidence analysis
        "self_vs_metric_r": float(self_r),
        "self_vs_metric_pvalue": float(self_pvalue),
        # Other-confidence analysis
        "other_vs_metric_r": float(other_r),
        "other_vs_metric_pvalue": float(other_pvalue),
        # Self vs Other correlation
        "self_vs_other_r": float(self_other_r),
        "self_vs_other_pvalue": float(self_other_pvalue),
        # Comparison
        "correlation_difference": float(self_r - other_r),
        "correlation_difference_ci95": [float(diff_ci_lower), float(diff_ci_upper)],
        "correlation_difference_pvalue": float(diff_pvalue) if not np.isnan(diff_pvalue) else None,
        # Descriptives
        "self_confidence_mean": float(self_confidence.mean()),
        "self_confidence_std": float(self_confidence.std()),
        "other_confidence_mean": float(other_signals.mean()),
        "other_confidence_std": float(other_signals.std()),
        "n_samples": n,
    }


def print_results(results: Dict, behavioral: Dict, other_confidence_analysis: Dict = None):
    """Print summary of results."""
    print("\n" + "=" * 100)
    print("INTROSPECTION EXPERIMENT RESULTS")
    print("=" * 100)

    print("\n--- Behavioral Analysis ---")
    n_full = behavioral.get('n_samples_full', '?')
    n_test = behavioral.get('n_samples_test', '?')
    n_subsamples = behavioral.get('n_subsamples', '?')
    print(f"Correlation (stated confidence vs direct entropy):")

    # Full dataset correlation with subsample-to-m CI
    full_p = behavioral.get('full_correlation_pvalue')
    p_str = f", p = {full_p:.2e}" if full_p is not None and not np.isnan(full_p) else ""
    full_ci = behavioral.get('full_correlation_ci95', [None, None])
    full_ci_std = behavioral.get('full_correlation_ci_std')
    if full_ci[0] is not None:
        print(f"  Full  (n={n_full}):  r = {behavioral['full_correlation']:.4f} ± {full_ci_std:.4f}  [95% CI: {full_ci[0]:.4f}, {full_ci[1]:.4f}]{p_str}")
    else:
        print(f"  Full  (n={n_full}):  r = {behavioral['full_correlation']:.4f}{p_str}")

    # Test set correlation with subsample CI
    test_p = behavioral.get('test_correlation_pvalue')
    p_str = f", p = {test_p:.2e}" if test_p is not None and not np.isnan(test_p) else ""
    test_ci = behavioral.get('test_correlation_ci95', [None, None])
    test_ci_std = behavioral.get('test_correlation_ci_std')
    if test_ci[0] is not None:
        print(f"  Test  (n={n_test}):  r = {behavioral['test_correlation']:.4f} ± {test_ci_std:.4f}  [95% CI: {test_ci[0]:.4f}, {test_ci[1]:.4f}]{p_str}")
    else:
        print(f"  Test  (n={n_test}):  r = {behavioral['test_correlation']:.4f}{p_str}")

    print(f"  (CIs from {n_subsamples} subsamples to test size, centered on point estimate)")

    # Partial correlation
    partial_r = behavioral.get('partial_correlation')
    partial_ci = behavioral.get('partial_correlation_ci95', [None, None])
    partial_p = behavioral.get('partial_correlation_pvalue')
    controls = behavioral.get('partial_correlation_controls', [])
    if partial_r is not None:
        p_str = f", p = {partial_p:.2e}" if partial_p is not None and not np.isnan(partial_p) else ""
        ctrl_str = f" (controlling for {', '.join(controls)})" if controls else ""
        print(f"  Partial{ctrl_str}: r = {partial_r:.4f}  [95% CI: {partial_ci[0]:.4f}, {partial_ci[1]:.4f}]{p_str}")

    print(f"  (Negative correlation suggests introspection; positive suggests miscalibration)")

    # Other-confidence control analysis (only for confidence task)
    if other_confidence_analysis is not None:
        print("\n--- Other-Confidence Control (Human Difficulty Estimation) ---")
        self_r = other_confidence_analysis['self_vs_metric_r']
        other_r = other_confidence_analysis['other_vs_metric_r']
        diff = other_confidence_analysis['correlation_difference']
        diff_ci = other_confidence_analysis['correlation_difference_ci95']
        diff_p = other_confidence_analysis.get('correlation_difference_pvalue')

        self_p = other_confidence_analysis['self_vs_metric_pvalue']
        other_p = other_confidence_analysis['other_vs_metric_pvalue']

        self_p_str = f", p = {self_p:.2e}" if self_p is not None and not np.isnan(self_p) else ""
        other_p_str = f", p = {other_p:.2e}" if other_p is not None and not np.isnan(other_p) else ""

        print(f"  Self-confidence vs {METRIC}:    r = {self_r:.4f}{self_p_str}")
        print(f"  Other-confidence vs {METRIC}:   r = {other_r:.4f}{other_p_str}")
        print(f"  Self vs Other confidence:       r = {other_confidence_analysis['self_vs_other_r']:.4f}")
        print(f"")
        print(f"  Difference (self - other):      Δr = {diff:.4f}  [95% CI: {diff_ci[0]:.4f}, {diff_ci[1]:.4f}]")

        if diff_p is not None:
            print(f"  Steiger's test p-value:         p = {diff_p:.4e}")
            if diff_p < 0.05 and diff < 0:
                print(f"  → Self-confidence significantly MORE correlated with {METRIC} than other-confidence")
                print(f"    This suggests the model is introspecting on its own uncertainty,")
                print(f"    not just assessing question difficulty.")
            elif diff_p < 0.05 and diff > 0:
                print(f"  → Self-confidence significantly LESS correlated with {METRIC} than other-confidence")
                print(f"    This is unexpected - the model may be using question difficulty as a proxy.")
            else:
                print(f"  → No significant difference between self and other confidence correlations")
        else:
            print(f"  → Could not compute significance test")

    # Delegate-specific summary statistics
    if META_TASK == "delegate" and "delegation_rate" in behavioral:
        print("\n--- Delegate Task Summary ---")
        print(f"  Delegation rate:      {behavioral['delegation_rate']:.1%} ({behavioral['num_delegated']} delegated, {behavioral['num_self_answered']} self-answered)")
        if "self_answer_accuracy" in behavioral:
            print(f"  Self-answer accuracy: {behavioral['self_answer_accuracy']:.1%} ({behavioral['self_correct']}/{behavioral['num_self_answered']} correct)")
            print(f"  Teammate accuracy:    {behavioral['teammate_accuracy']:.1%} (by design)")
            print(f"  Team score:           {behavioral['team_score']:.1f} / {behavioral['num_delegated'] + behavioral['num_self_answered']} ({behavioral['team_score_normalized']:.1%})")

    print("\n--- Probe Analysis by Layer ---")
    print(f"{'Layer':<8} {'Direct→Direct':<15} {'D→M (fixed)':<15} {'D→M (orig)':<15} {'Meta→Meta':<15} {'Shuffled':<12}")
    print(f"{'':8} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<15} {'R²':<12}")
    print("-" * 110)

    for layer_idx in sorted(results.keys()):
        res = results[layer_idx]
        d2d = res["direct_to_direct"]["test_r2"]
        d2m_fixed = res["direct_to_meta_fixed"]["r2"]
        d2m_orig = res["direct_to_meta"]["r2"]
        m2m = res["meta_to_meta"]["test_r2"]
        shuf = res["shuffled_baseline"]["r2"]
        print(f"{layer_idx:<8} {d2d:<15.4f} {d2m_fixed:<15.4f} {d2m_orig:<15.4f} {m2m:<15.4f} {shuf:<12.4f}")

    print("=" * 110)

    # Summary statistics
    layers = sorted(results.keys())

    best_d2d_layer = max(layers, key=lambda l: results[l]["direct_to_direct"]["test_r2"])
    best_d2d = results[best_d2d_layer]["direct_to_direct"]["test_r2"]

    best_d2m_fixed_layer = max(layers, key=lambda l: results[l]["direct_to_meta_fixed"]["r2"])
    best_d2m_fixed = results[best_d2m_fixed_layer]["direct_to_meta_fixed"]["r2"]

    best_m2m_layer = max(layers, key=lambda l: results[l]["meta_to_meta"]["test_r2"])
    best_m2m = results[best_m2m_layer]["meta_to_meta"]["test_r2"]

    print(f"\nBest Direct→Direct:      Layer {best_d2d_layer} (R² = {best_d2d:.4f})")
    print(f"Best Direct→Meta (fixed): Layer {best_d2m_fixed_layer} (R² = {best_d2m_fixed:.4f})")
    print(f"Best Meta→Meta:          Layer {best_m2m_layer} (R² = {best_m2m:.4f})")

    # Transfer ratio using fixed D→M
    if best_d2d > 0:
        transfer_ratio = best_d2m_fixed / best_d2d
        print(f"\nTransfer ratio (best D→M fixed / best D→D): {transfer_ratio:.2%}")
        if transfer_ratio > 0.5:
            print("  → Strong evidence for introspection!")
        elif transfer_ratio > 0.25:
            print("  → Moderate evidence for introspection")
        elif transfer_ratio > 0:
            print("  → Weak evidence for introspection")
        else:
            print("  → No evidence for introspection (negative transfer)")


def plot_results(
    results: Dict,
    behavioral: Dict,
    direct_entropies: np.ndarray,
    test_idx: np.ndarray,
    output_path: str = "introspection_results.png"
):
    """Create visualization of results."""
    layers = sorted(results.keys())

    d2d_r2 = [results[l]["direct_to_direct"]["test_r2"] for l in layers]
    d2m_r2_orig = [results[l]["direct_to_meta"]["r2"] for l in layers]
    d2m_r2_fixed = [results[l]["direct_to_meta_fixed"]["r2"] for l in layers]
    m2m_r2 = [results[l]["meta_to_meta"]["test_r2"] for l in layers]
    shuffled_r2 = [results[l]["shuffled_baseline"]["r2"] for l in layers]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: All R² curves (using fixed D→M)
    ax1 = axes[0, 0]
    ax1.plot(layers, d2d_r2, 'o-', label='Direct→Direct', linewidth=2)
    ax1.plot(layers, d2m_r2_fixed, 's-', label='Direct→Meta (transfer test)', linewidth=2)
    ax1.plot(layers, m2m_r2, '^-', label='Meta→Meta', linewidth=2, alpha=0.7)
    ax1.plot(layers, shuffled_r2, 'x--', label='Shuffled baseline', linewidth=1, alpha=0.5, color='gray')
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax1.set_xlabel('Layer Index')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Probe Performance: Can We Predict Direct Entropy?')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Comparison of original vs fixed D→M scaling
    ax2 = axes[0, 1]
    ax2.plot(layers, d2m_r2_fixed, 's-', label='D→M (separate scaling)', linewidth=2, color='C1')
    ax2.plot(layers, d2m_r2_orig, 'x--', label='D→M (shared scaling - broken)', linewidth=1.5, alpha=0.7, color='C3')
    ax2.plot(layers, d2d_r2, 'o-', label='D→D (reference)', linewidth=1.5, alpha=0.5, color='C0')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('R² Score')
    ax2.set_title('Scaling Fix: Shared vs Separate Standardization')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)

    # Plot 3: Prediction scatter for best D→D layer
    ax3 = axes[1, 0]
    best_d2d_layer = max(layers, key=lambda l: results[l]["direct_to_direct"]["test_r2"])
    best_d2d_r2 = results[best_d2d_layer]["direct_to_direct"]["test_r2"]
    predictions = np.array(results[best_d2d_layer]["direct_to_direct"]["predictions"])
    actual_entropy = direct_entropies[test_idx]

    ax3.scatter(actual_entropy, predictions, alpha=0.5, s=30, color='C0')
    # Reference line: y=x (perfect prediction)
    min_val = min(actual_entropy.min(), predictions.min())
    max_val = max(actual_entropy.max(), predictions.max())
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x (perfect)', alpha=0.7)
    ax3.set_xlabel('Actual Entropy')
    ax3.set_ylabel('Predicted Entropy')
    ax3.set_title(f'Prediction Quality (Layer {best_d2d_layer}, R²={best_d2d_r2:.3f})')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Summary text
    ax4 = axes[1, 1]
    ax4.axis('off')

    best_d2d_layer = max(layers, key=lambda l: results[l]['direct_to_direct']['test_r2'])
    best_d2m_layer = max(layers, key=lambda l: results[l]['direct_to_meta_fixed']['r2'])
    best_m2m_layer = max(layers, key=lambda l: results[l]['meta_to_meta']['test_r2'])

    transfer_ratio = max(d2m_r2_fixed) / max(max(d2d_r2), 0.001)

    # Format correlation strings with subsample-to-m CIs
    full_p = behavioral.get('full_correlation_pvalue')
    full_p_str = f"p={full_p:.1e}" if full_p is not None and not np.isnan(full_p) else ""

    full_ci = behavioral.get('full_correlation_ci95', [None, None])
    full_ci_std = behavioral.get('full_correlation_ci_std', 0)
    test_ci = behavioral.get('test_correlation_ci95', [None, None])
    test_ci_std = behavioral.get('test_correlation_ci_std', 0)

    n_full = behavioral.get('n_samples_full', '?')
    n_test = behavioral.get('n_samples_test', '?')

    full_ci_str = f"±{full_ci_std:.3f} [{full_ci[0]:.3f}, {full_ci[1]:.3f}]" if full_ci[0] is not None else ""
    test_ci_str = f"±{test_ci_std:.3f} [{test_ci[0]:.3f}, {test_ci[1]:.3f}]" if test_ci[0] is not None else ""

    summary_text = f"""
INTROSPECTION EXPERIMENT SUMMARY

Behavioral Correlation (stated conf vs entropy):
  Full (n={n_full}):  r = {behavioral['full_correlation']:.4f} {full_ci_str}  {full_p_str}
  Test (n={n_test}):  r = {behavioral['test_correlation']:.4f} {test_ci_str}
  (Negative = model reports low confidence when uncertain)

Best Layer Results:
  Direct→Direct: Layer {best_d2d_layer}  (R² = {max(d2d_r2):.4f})
  Direct→Meta:   Layer {best_d2m_layer}  (R² = {max(d2m_r2_fixed):.4f})
  Meta→Meta:     Layer {best_m2m_layer}  (R² = {max(m2m_r2):.4f})

Transfer Ratio (D→M / D→D): {transfer_ratio:.1%}

Interpretation:
  Transfer ratio near 100% = entropy probe transfers well
  to meta-judgment activations (evidence for introspection).

  Transfer ratio near 0% = meta activations encode entropy
  differently than direct activations (no transfer).
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


def get_mc_prefix() -> str:
    """Get prefix for mc_entropy_probe.py output files."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_mc")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_mc")


def try_load_mc_data() -> Optional[Dict]:
    """
    Try to load existing MC data from mc_entropy_probe.py output.

    Returns dict with direct_activations, direct_metrics, metadata
    if files exist and config matches. Returns None otherwise.

    direct_metrics is a dict mapping metric names to numpy arrays.
    """
    mc_prefix = get_mc_prefix()
    activations_path = Path(f"{mc_prefix}_activations.npz")
    dataset_path = Path(f"{mc_prefix}_dataset.json")

    if not activations_path.exists() or not dataset_path.exists():
        return None

    print(f"\nFound existing MC data: {mc_prefix}")

    # Load and verify config
    with open(dataset_path) as f:
        dataset_data = json.load(f)

    config = dataset_data.get("config", {})
    if config.get("dataset") != DATASET_NAME:
        print(f"  Dataset mismatch: {config.get('dataset')} vs {DATASET_NAME}")
        return None
    if config.get("num_questions") != NUM_QUESTIONS:
        print(f"  Question count mismatch: {config.get('num_questions')} vs {NUM_QUESTIONS}")
        return None
    if config.get("seed") != SEED:
        print(f"  Seed mismatch: {config.get('seed')} vs {SEED}")
        return None

    # Load activations
    print(f"  Loading activations from {activations_path}...")
    acts_data = np.load(activations_path)
    direct_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in acts_data.files if k.startswith("layer_")
    }

    # Load all metrics (new format stores: entropy, top_prob, margin, logit_gap, top_logit)
    # Fall back to "entropies" key for backward compatibility with old files
    direct_metrics = {}
    metric_keys = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
    for key in metric_keys:
        if key in acts_data.files:
            direct_metrics[key] = acts_data[key]
    # Backward compatibility: old files have "entropies" key
    if not direct_metrics and "entropies" in acts_data.files:
        direct_metrics["entropy"] = acts_data["entropies"]

    # Get metadata (includes questions, probs, etc.)
    metadata = dataset_data.get("data", [])

    print(f"  Loaded {len(direct_activations)} layers, {len(direct_metrics.get('entropy', []))} questions")
    print(f"  Metrics available: {list(direct_metrics.keys())}")
    print(f"  Reusing direct activations from mc_entropy_probe.py!")

    return {
        "direct_activations": direct_activations,
        "direct_metrics": direct_metrics,
        "metadata": metadata,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    global METRIC

    parser = argparse.ArgumentParser(description="Run introspection experiment")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Uncertainty metric to probe (default: {METRIC})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"Batch size for forward passes (default {BATCH_SIZE})")
    parser.add_argument("--load-in-4bit", action="store_true",
                        help="Load model in 4-bit quantization (recommended for 70B+ models)")
    parser.add_argument("--load-in-8bit", action="store_true",
                        help="Load model in 8-bit quantization")
    args = parser.parse_args()

    METRIC = args.metric
    print(f"Device: {DEVICE}")
    print(f"Metric: {METRIC}")
    print(f"Meta-judgment task: {META_TASK}")

    # Print delegate parameters if using delegate task
    if META_TASK == "delegate":
        print("\n--- Delegate Task Parameters ---")
        print("  (Matching delegate_game_from_capabilities.py)")
        print("  decision_only: True")
        print("  alternate_decision_mapping: True")
        print("  use_phase1_summary: True")
        print("  use_phase1_history: False")
        print("  use_examples: True")
        print("  teammate_accuracy: 50%")
        print("  Options: 1/2 (alternating mapping per trial)")
        print("")

    # Check for existing MC data first
    mc_data = try_load_mc_data()

    # Load model and tokenizer using shared utility
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit
    )

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
    # If we have existing MC data, only run meta prompts
    if mc_data is not None:
        data = collect_meta_only(questions, model, tokenizer, num_layers, use_chat_template, mc_data, batch_size=args.batch_size)
    else:
        data = collect_paired_data(questions, model, tokenizer, num_layers, use_chat_template, batch_size=args.batch_size)

    # Generate output prefixes
    # Base prefix for shared files (activations, paired data)
    base_prefix = get_output_prefix()
    # Metric-specific prefix for probe results and directions
    metric_prefix = get_output_prefix(METRIC)
    print(f"Base output prefix: {base_prefix}")
    print(f"Metric output prefix: {metric_prefix}")

    # Get the selected metric's values
    direct_target = data["direct_metrics"][METRIC]

    # Save activations with ALL metrics (so we can retrain with different metrics later)
    print("\nSaving activations...")
    np.savez_compressed(
        f"{base_prefix}_direct_activations.npz",
        **{f"layer_{i}": acts for i, acts in data["direct_activations"].items()},
        **data["direct_metrics"]  # Save all metrics
    )
    np.savez_compressed(
        f"{base_prefix}_meta_activations.npz",
        **{f"layer_{i}": acts for i, acts in data["meta_activations"].items()},
        entropy=data["meta_entropies"]  # Meta always uses entropy
    )
    print(f"Saved activations to {base_prefix}_*_activations.npz")

    # Generate example prompts for verification (first 2 questions)
    example_prompts = []
    for i in range(min(2, len(questions))):
        q = questions[i]
        direct_prompt, direct_options = format_direct_prompt(q, tokenizer, use_chat_template)
        if META_TASK == "delegate":
            meta_prompt, meta_options_list, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_index=i)
        else:
            meta_prompt, meta_options_list = format_meta_prompt(q, tokenizer, use_chat_template)
            mapping = None
        example_prompts.append({
            "question_index": i,
            "question_text": q.get("question", ""),
            "direct_prompt": direct_prompt,
            "direct_options": direct_options,
            "meta_prompt": meta_prompt,
            "meta_options": meta_options_list,
            "meta_mapping": mapping,
        })

    # Collect other-confidence control data (only for confidence task)
    # Do this before saving paired_data so we can include it
    other_data = None
    if META_TASK == "confidence":
        print("\n" + "=" * 60)
        print("Running OTHER-CONFIDENCE control task...")
        print("(Asks model to estimate what % of college-educated people would know the answer)")
        print("=" * 60)

        other_data = collect_other_confidence(
            questions, model, tokenizer, num_layers, use_chat_template, batch_size=args.batch_size
        )

        # Add example prompts for other-confidence
        for i in range(min(2, len(questions))):
            q = questions[i]
            other_prompt, _ = format_other_confidence_prompt(q, tokenizer, use_chat_template)
            if i < len(example_prompts):
                example_prompts[i]["other_confidence_prompt"] = other_prompt

    # Save paired data (for reproducibility and further analysis)
    paired_data = {
        "direct_metrics": {k: v.tolist() for k, v in data["direct_metrics"].items()},
        "direct_probs": data["direct_probs"],
        "direct_logits": data.get("direct_logits", []),
        "meta_entropies": data["meta_entropies"].tolist(),
        "meta_probs": data["meta_probs"],
        "meta_responses": data["meta_responses"],
        "meta_mappings": data.get("meta_mappings"),  # Store mappings for delegate task
        "questions": [
            {
                "id": q.get("id", f"q_{i}"),
                "question": q.get("question", ""),
                "correct_answer": q.get("correct_answer", ""),
                "options": q.get("options", {})
            }
            for i, q in enumerate(data["questions"])
        ],
        "example_prompts": example_prompts,  # For verification of prompt formatting
        "config": {
            "model_name": MODEL_NAME,
            "base_model_name": BASE_MODEL_NAME,
            "dataset_name": DATASET_NAME,
            "num_questions": NUM_QUESTIONS,
            "seed": SEED,
            "meta_task": META_TASK,
            "metric": METRIC,
            # Delegate task parameters (matches delegate_game_from_capabilities.py)
            "delegate_params": {
                "decision_only": True,
                "alternate_decision_mapping": True,
                "use_phase1_summary": True,
                "use_phase1_history": False,
                "use_examples": True,
                "teammate_accuracy": 0.5,
            } if META_TASK == "delegate" else None,
        }
    }

    # Add other-confidence data if collected
    if other_data is not None:
        paired_data["other_confidence"] = {
            "probs": other_data["other_probs"],
            "responses": other_data["other_responses"],
            "signals": other_data["other_signals"].tolist(),
        }

    with open(f"{base_prefix}_paired_data.json", "w") as f:
        json.dump(paired_data, f, indent=2)
    print(f"Saved paired data to {base_prefix}_paired_data.json")

    # Run introspection analysis with selected metric
    print(f"\nRunning introspection analysis with metric: {METRIC}")
    results, test_idx, directions = run_introspection_analysis(
        data["direct_activations"],
        data["meta_activations"],
        direct_target,  # Use selected metric
        extract_directions=True
    )

    # Save directions for steering/ablation experiments (metric-specific filename)
    if directions is not None:
        directions_data = {
            f"layer_{layer_idx}": direction
            for layer_idx, direction in directions.items()
        }
        directions_data["_metadata_metric"] = np.array(METRIC)
        directions_data["_metadata_dataset"] = np.array(DATASET_NAME)
        directions_data["_metadata_model"] = np.array(BASE_MODEL_NAME)
        np.savez_compressed(
            f"{metric_prefix}_directions.npz",
            **directions_data
        )
        print(f"Saved {METRIC} directions to {metric_prefix}_directions.npz")

    # Behavioral analysis (uses selected METRIC for correlation with stated confidence)
    behavioral = analyze_behavioral_introspection(
        data["meta_responses"],
        data["direct_metrics"][METRIC],  # Use the selected metric for behavioral correlation
        test_idx,
        data["meta_probs"],
        data.get("meta_mappings"),
        data["direct_probs"],
        data["questions"]
    )

    # Other-confidence control analysis (only for confidence task)
    # other_data was collected earlier, now we analyze it
    other_confidence_analysis = None
    if META_TASK == "confidence" and other_data is not None:
        # Compute self-confidence signals for comparison
        self_confidence = np.array([
            get_stated_confidence_signal(np.array(p)) if p else 0.5
            for p in data["meta_probs"]
        ])

        other_confidence_analysis = analyze_other_confidence_control(
            other_data["other_signals"],
            self_confidence,
            data["direct_metrics"][METRIC],
            test_idx
        )

    # Save results (metric-specific filename)
    results_to_save = {
        "config": {
            "metric": METRIC,
            "meta_task": META_TASK,
            "model": BASE_MODEL_NAME,
            "dataset": DATASET_NAME,
        },
        "metric_stats": {
            "mean": float(direct_target.mean()),
            "std": float(direct_target.std()),
            "min": float(direct_target.min()),
            "max": float(direct_target.max()),
        },
        "probe_results": {
            str(layer_idx): {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in layer_results.items()
                if not isinstance(v, dict) or k in ["direct_to_direct", "direct_to_meta", "direct_to_meta_fixed", "shuffled_baseline", "meta_to_meta"]
            }
            for layer_idx, layer_results in results.items()
        },
        "behavioral": behavioral,
        "other_confidence_analysis": other_confidence_analysis,  # None if not confidence task
        "test_indices": test_idx.tolist(),
    }

    # Properly serialize nested dicts
    for layer_idx in results_to_save["probe_results"]:
        for key in ["direct_to_direct", "direct_to_meta", "direct_to_meta_fixed", "shuffled_baseline", "meta_to_meta"]:
            if key in results_to_save["probe_results"][layer_idx]:
                inner = results_to_save["probe_results"][layer_idx][key]
                for k, v in inner.items():
                    if isinstance(v, np.ndarray):
                        inner[k] = v.tolist()

    with open(f"{metric_prefix}_results.json", "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved results to {metric_prefix}_results.json")

    # Print and plot results
    print_results(results, behavioral, other_confidence_analysis)
    plot_results(
        results, behavioral,
        direct_target, test_idx,
        output_path=f"{metric_prefix}_results.png"
    )

    print(f"\n✓ Introspection experiment complete! (metric: {METRIC})")


if __name__ == "__main__":
    main()
