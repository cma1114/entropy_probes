"""
Find introspection mapping direction using contrastive approach.

Instead of regression, this script:
1. Loads introspection data (direct entropies, stated confidences, meta activations)
2. Selects only calibrated examples (where confidence correctly tracks entropy)
3. Within calibrated examples, contrasts:
   - High confidence + low entropy (correctly confident)
   - Low confidence + high entropy (correctly uncertain)
4. Computes direction = mean(high_conf_low_ent) - mean(low_conf_high_ent)
5. Evaluates direction quality (how well it predicts introspection score)
6. Runs steering/ablation experiments to test causality

This captures the confidence axis within calibrated examples - steering along this
direction should shift confidence while maintaining calibration.
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats
import os
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional
import random
from transformers import AutoTokenizer, AutoModelForCausalLM

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

from core import (
    DEVICE,
    load_model_and_tokenizer,
    compute_introspection_scores,
    generate_orthogonal_directions,
)
from core.steering import SteeringHook, AblationHook
from core.probes import (
    compute_cluster_centroids,
    compute_cluster_directions,
    compute_caa_direction,
    compare_directions,
)
from tasks import (
    # Confidence task
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    # Delegate task
    ANSWER_OR_DELEGATE_SETUP,
    ANSWER_OR_DELEGATE_SYSPROMPT,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    # Unified conversion
    response_to_confidence as tasks_response_to_confidence,
)

# Configuration
BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_NAME = "SimpleMC"
SEED = 42

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


def get_model_short_name(model_name: str) -> str:
    """Extract a short, filesystem-safe name from a model path."""
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_output_prefix() -> str:
    """Generate output filename prefix based on config."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_contrastive")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_contrastive")

# Contrastive selection thresholds
# Use top/bottom quantiles of introspection score
TOP_QUANTILE = 0.25  # Top 25% = well-calibrated
BOTTOM_QUANTILE = 0.25  # Bottom 25% = miscalibrated

# Which layer to use for direction extraction
TARGET_LAYER = None  # Will be set to best layer from probe results, or middle layer

# Steering/ablation configuration
RUN_STEERING = True  # Set to False to skip steering experiments
META_TASK = "confidence"  # "confidence" or "delegate" - must match run_introspection_experiment.py
STEERING_LAYERS = None  # None = auto-select based on projection correlation
STEERING_MULTIPLIERS = [-2.0, -1.0, 0.0, 1.0, 2.0]
NUM_STEERING_QUESTIONS = 50
NUM_CONTROL_DIRECTIONS = 2
MIN_PROJECTION_CORR = 0.5 # Minimum projection correlation for layer selection
STEERING_BATCH_SIZE = 16  # Batch size for steering/ablation (increase for more GPU mem)

# Direction comparison mode
COMPARE_DIRECTIONS = True  # Set to True to compare different direction types
N_CLUSTERS = 3  # Number of clusters for cluster-based directions (low/mid/high)
CLUSTER_METHOD = "quantile"  # "quantile" (group by metric percentiles) or "kmeans"

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Backward compatibility aliases (now imported from tasks.py)
META_OPTION_DICT = STATED_CONFIDENCE_OPTIONS
META_RANGE_MIDPOINTS = STATED_CONFIDENCE_MIDPOINTS
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())

DELEGATE_SETUP_PROMPT = ANSWER_OR_DELEGATE_SETUP
DELEGATE_SYSPROMPT = ANSWER_OR_DELEGATE_SYSPROMPT
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS


def get_introspection_prefix() -> str:
    """Get prefix for introspection data files (from run_introspection_experiment.py)."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection")


def load_introspection_data(run_name: str = None) -> dict:
    """
    Load previously collected introspection data.

    Looks for:
    - {run_name}_paired_data.json (or computed from config)
    - {run_name}_meta_activations.npz
    """
    # Try to find data files
    if run_name:
        paired_path = Path(f"{run_name}_paired_data.json")
        acts_path = Path(f"{run_name}_meta_activations.npz")
    else:
        prefix = get_introspection_prefix()
        paired_path = Path(f"{prefix}_paired_data.json")
        acts_path = Path(f"{prefix}_meta_activations.npz")

    if not paired_path.exists():
        raise FileNotFoundError(
            f"Could not find {paired_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    if not acts_path.exists():
        raise FileNotFoundError(
            f"Could not find {acts_path}. "
            "Run run_introspection_experiment.py first to generate data."
        )

    # Load paired data
    print(f"Loading paired data from {paired_path}...")
    with open(paired_path) as f:
        paired_data = json.load(f)

    # Extract arrays - handle both old format (list of dicts) and new format (dict with arrays)
    if isinstance(paired_data, list):
        # Old format: list of {"direct_entropy": ..., "stated_confidence": ...}
        direct_entropies = np.array([d["direct_entropy"] for d in paired_data])
        stated_confidences = np.array([d["stated_confidence"] for d in paired_data])
    else:
        # New format from run_introspection_experiment.py
        direct_entropies = np.array(paired_data["direct_entropies"])
        meta_responses = paired_data["meta_responses"]
        meta_probs = paired_data.get("meta_probs")
        meta_mappings = paired_data.get("meta_mappings")
        meta_task = paired_data.get("config", {}).get("meta_task", "confidence")

        # Convert meta responses to confidence values
        if meta_task == "delegate":
            # For delegate task, confidence = P(Answer)
            stated_confidences = []
            for i, (response, probs, mapping) in enumerate(zip(
                meta_responses,
                meta_probs or [None] * len(meta_responses),
                meta_mappings or [None] * len(meta_responses)
            )):
                if probs is not None and len(probs) >= 2 and mapping is not None:
                    if mapping.get("1") == "Answer":
                        stated_confidences.append(float(probs[0]))
                    else:
                        stated_confidences.append(float(probs[1]))
                elif probs is not None and len(probs) >= 1:
                    stated_confidences.append(float(probs[0]))
                else:
                    stated_confidences.append(1.0 if response == "1" else 0.0)
            stated_confidences = np.array(stated_confidences)
        else:
            # For confidence task, use midpoint of chosen range
            META_RANGE_MIDPOINTS = {
                "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
                "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
            }
            stated_confidences = np.array([
                META_RANGE_MIDPOINTS.get(r, 0.5) for r in meta_responses
            ])

    # Load meta activations
    print(f"Loading meta activations from {acts_path}...")
    acts_data = np.load(acts_path)

    # activations are stored as layer_0, layer_1, etc.
    layer_keys = sorted([k for k in acts_data.keys() if k.startswith("layer_")])
    num_layers = len(layer_keys)

    meta_activations = {
        int(k.split("_")[1]): acts_data[k]
        for k in layer_keys
    }

    print(f"Loaded {len(paired_data)} examples with {num_layers} layers")
    print(f"Entropy range: [{direct_entropies.min():.3f}, {direct_entropies.max():.3f}]")
    print(f"Confidence range: [{stated_confidences.min():.3f}, {stated_confidences.max():.3f}]")

    return {
        "paired_data": paired_data,
        "direct_entropies": direct_entropies,
        "stated_confidences": stated_confidences,
        "meta_activations": meta_activations,
        "num_layers": num_layers,
    }


# ============================================================================
# STEERING HELPERS (wrappers around tasks.py functions)
# ============================================================================

# Import utility functions from core
from core.model_utils import is_base_model, has_chat_template


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format a meta/confidence question using centralized tasks.py logic."""
    full_prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return full_prompt


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


def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """
    Convert a meta response to a confidence value.

    Wrapper around tasks.response_to_confidence that passes the correct task_type.
    """
    task_type = "delegate" if META_TASK == "delegate" else "confidence"
    return tasks_response_to_confidence(response, probs, mapping, task_type)


def get_confidence_response(
    model,
    tokenizer,
    question: Dict,
    layer_idx: Optional[int],
    steering_direction: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response with optional steering."""
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    if layer_idx is not None and steering_direction is not None and multiplier != 0.0:
        steering_tensor = torch.tensor(
            steering_direction,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        handle = layer_module.register_forward_hook(hook)

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
        finally:
            handle.remove()
    else:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


def get_batch_confidence_responses(
    model,
    tokenizer,
    questions: List[Dict],
    layer_idx: Optional[int],
    steering_direction: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool,
    batch_size: int = 8
) -> List[Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]]:
    """
    Get confidence responses for a batch of questions with optional steering.

    Much more efficient than calling get_confidence_response one at a time.
    """
    results = []

    # Pre-compute option token IDs
    if META_TASK == "delegate":
        options = DELEGATE_OPTIONS
    else:
        options = META_OPTIONS
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]

    # Set up steering hook once if needed
    handle = None
    if layer_idx is not None and steering_direction is not None and multiplier != 0.0:
        steering_tensor = torch.tensor(
            steering_direction,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        handle = layer_module.register_forward_hook(hook)

    try:
        # Process in batches
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            # Format prompts and collect mappings
            prompts = []
            mappings = []
            for i, q in enumerate(batch_questions):
                trial_idx = batch_start + i
                if META_TASK == "delegate":
                    prompt, _, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_idx)
                else:
                    prompt = format_meta_prompt(q, tokenizer, use_chat_template)
                    mapping = None
                prompts.append(prompt)
                mappings.append(mapping)

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

            # Extract confidence for each item in batch
            # With left-padding, the last position (-1) is always the final real token
            for i in range(len(batch_questions)):
                final_logits = outputs.logits[i, -1, :]
                option_logits = final_logits[option_token_ids]
                option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

                response = options[np.argmax(option_probs)]
                confidence = response_to_confidence(response, option_probs, mappings[i])

                results.append((response, confidence, option_probs, mappings[i]))

            del inputs, outputs

    finally:
        if handle is not None:
            handle.remove()

    return results


def get_batch_confidence_with_ablation(
    model,
    tokenizer,
    questions: List[Dict],
    layer_idx: int,
    ablation_direction: np.ndarray,
    use_chat_template: bool,
    batch_size: int = 8
) -> List[Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]]:
    """
    Get confidence responses for a batch of questions with ablation.
    """
    results = []

    # Pre-compute option token IDs
    if META_TASK == "delegate":
        options = DELEGATE_OPTIONS
    else:
        options = META_OPTIONS
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]

    # Set up ablation hook
    ablation_tensor = torch.tensor(
        ablation_direction,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        layer_module = model.get_base_model().model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = AblationHook(ablation_tensor)
    handle = layer_module.register_forward_hook(hook)

    try:
        # Process in batches
        for batch_start in range(0, len(questions), batch_size):
            batch_end = min(batch_start + batch_size, len(questions))
            batch_questions = questions[batch_start:batch_end]

            # Format prompts and collect mappings
            prompts = []
            mappings = []
            for i, q in enumerate(batch_questions):
                trial_idx = batch_start + i
                if META_TASK == "delegate":
                    prompt, _, mapping = format_delegate_prompt(q, tokenizer, use_chat_template, trial_idx)
                else:
                    prompt = format_meta_prompt(q, tokenizer, use_chat_template)
                    mapping = None
                prompts.append(prompt)
                mappings.append(mapping)

            # Tokenize batch
            inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(DEVICE)

            # Forward pass
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)

            # Extract confidence for each item in batch
            # With left-padding, the last position (-1) is always the final real token
            for i in range(len(batch_questions)):
                final_logits = outputs.logits[i, -1, :]
                option_logits = final_logits[option_token_ids]
                option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

                response = options[np.argmax(option_probs)]
                confidence = response_to_confidence(response, option_probs, mappings[i])

                results.append((response, confidence, option_probs, mappings[i]))

            del inputs, outputs

    finally:
        handle.remove()

    return results


def get_confidence_with_ablation(
    model,
    tokenizer,
    question: Dict,
    layer_idx: int,
    ablation_direction: np.ndarray,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response with ablation (direction zeroed out)."""
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    ablation_tensor = torch.tensor(
        ablation_direction,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        layer_module = model.get_base_model().model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = AblationHook(ablation_tensor)
    handle = layer_module.register_forward_hook(hook)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        handle.remove()

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


# ============================================================================
# STEERING AND ABLATION EXPERIMENTS
# ============================================================================

def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    directions: Dict[int, np.ndarray],
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool,
    batch_size: int = 8
) -> Dict:
    """Run steering experiment with contrastive direction (batched for efficiency)."""
    print(f"\nRunning steering experiment (batched)...")
    print(f"  Layers: {layers}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")
    print(f"  Batch size: {batch_size}")

    # Count total forward passes for progress estimation
    non_zero_mults = [m for m in multipliers if m != 0.0]
    total_conditions = 1 + len(layers) * (len(non_zero_mults) * (1 + num_controls))
    print(f"  Total conditions: {total_conditions} (was {len(questions) * total_conditions} forward passes, now ~{total_conditions * ((len(questions) + batch_size - 1) // batch_size)} batches)")

    results = {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": {},
    }

    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    def compute_results_from_batch(batch_results, start_idx=0):
        """Convert batch results to per-question result dicts."""
        out = []
        for q_idx, (response, confidence, probs, mapping) in enumerate(batch_results):
            actual_idx = start_idx + q_idx if start_idx else q_idx
            entropy = direct_entropies[actual_idx]
            entropy_z = (entropy - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = -entropy_z * confidence_z

            out.append({
                "question_idx": actual_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })
        return out

    # Compute baseline once (no steering)
    print("Computing baseline (no steering)...")
    baseline_batch = get_batch_confidence_responses(
        model, tokenizer, questions, None, None, 0.0, use_chat_template, batch_size
    )
    shared_baseline = compute_results_from_batch(baseline_batch)

    for layer_idx in tqdm(layers, desc="Steering layers"):
        contrastive_dir = directions[layer_idx]
        control_dirs = generate_orthogonal_directions(contrastive_dir, num_controls)

        layer_results = {
            "baseline": shared_baseline,
            "contrastive": {m: [] for m in multipliers},
            "controls": {f"control_{i}": {m: [] for m in multipliers} for i in range(num_controls)},
        }

        # Contrastive direction steering
        for mult in tqdm(multipliers, desc="Contrastive", leave=False):
            if mult == 0.0:
                layer_results["contrastive"][mult] = layer_results["baseline"]
                continue

            batch_results = get_batch_confidence_responses(
                model, tokenizer, questions, layer_idx, contrastive_dir, mult, use_chat_template, batch_size
            )
            layer_results["contrastive"][mult] = compute_results_from_batch(batch_results)

        # Control steering
        for ctrl_idx, ctrl_dir in enumerate(control_dirs):
            for mult in tqdm(multipliers, desc=f"Control {ctrl_idx}", leave=False):
                if mult == 0.0:
                    layer_results["controls"][f"control_{ctrl_idx}"][mult] = layer_results["baseline"]
                    continue

                batch_results = get_batch_confidence_responses(
                    model, tokenizer, questions, layer_idx, ctrl_dir, mult, use_chat_template, batch_size
                )
                layer_results["controls"][f"control_{ctrl_idx}"][mult] = compute_results_from_batch(batch_results)

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    directions: Dict[int, np.ndarray],
    num_controls: int,
    use_chat_template: bool,
    baseline_results: Optional[List[Dict]] = None,
    batch_size: int = 8
) -> Dict:
    """Run ablation experiment with contrastive direction (batched for efficiency)."""
    print(f"\nRunning ablation experiment (batched)...")
    print(f"  Layers: {layers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")
    print(f"  Batch size: {batch_size}")
    if baseline_results is not None:
        print(f"  Reusing baseline from steering experiment")

    results = {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": {},
    }

    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    def compute_results_from_batch(batch_results):
        """Convert batch results to per-question result dicts."""
        out = []
        for q_idx, (response, confidence, _, _) in enumerate(batch_results):
            entropy = direct_entropies[q_idx]
            entropy_z = (entropy - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = -entropy_z * confidence_z

            out.append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })
        return out

    if baseline_results is None:
        print("Computing baseline (no intervention)...")
        baseline_batch = get_batch_confidence_responses(
            model, tokenizer, questions, None, None, 0.0, use_chat_template, batch_size
        )
        baseline_results = compute_results_from_batch(baseline_batch)

    for layer_idx in tqdm(layers, desc="Ablation layers"):
        contrastive_dir = directions[layer_idx]
        control_dirs = generate_orthogonal_directions(contrastive_dir, num_controls)

        layer_results = {
            "baseline": baseline_results,
            "contrastive_ablated": [],
            "controls_ablated": {f"control_{i}": [] for i in range(num_controls)},
        }

        # Contrastive direction ablation
        batch_results = get_batch_confidence_with_ablation(
            model, tokenizer, questions, layer_idx, contrastive_dir, use_chat_template, batch_size
        )
        layer_results["contrastive_ablated"] = compute_results_from_batch(batch_results)

        # Control ablation
        for ctrl_idx, ctrl_dir in enumerate(control_dirs):
            batch_results = get_batch_confidence_with_ablation(
                model, tokenizer, questions, layer_idx, ctrl_dir, use_chat_template, batch_size
            )
            layer_results["controls_ablated"][f"control_{ctrl_idx}"] = compute_results_from_batch(batch_results)

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def analyze_steering_results(results: Dict) -> Dict:
    """Analyze steering experiment results."""
    analysis = {}

    for layer_idx, layer_results in results["layer_results"].items():
        baseline_alignments = [r["alignment"] for r in layer_results["baseline"]]
        baseline_confidences = [r["confidence"] for r in layer_results["baseline"]]
        baseline_mean_alignment = np.mean(baseline_alignments)
        baseline_mean_confidence = np.mean(baseline_confidences)

        layer_analysis = {
            "baseline_mean_alignment": float(baseline_mean_alignment),
            "baseline_mean_confidence": float(baseline_mean_confidence),
            "contrastive": {},
            "controls_mean": {},
        }

        # Contrastive direction effects
        for mult, mult_results in layer_results["contrastive"].items():
            alignments = [r["alignment"] for r in mult_results]
            confidences = [r["confidence"] for r in mult_results]
            layer_analysis["contrastive"][mult] = {
                "mean_alignment": float(np.mean(alignments)),
                "alignment_change": float(np.mean(alignments) - baseline_mean_alignment),
                "mean_confidence": float(np.mean(confidences)),
                "confidence_change": float(np.mean(confidences) - baseline_mean_confidence),
            }

        # Control direction effects (averaged)
        for mult in results["multipliers"]:
            control_alignments = []
            control_confidences = []
            for ctrl_key in layer_results["controls"]:
                ctrl_results = layer_results["controls"][ctrl_key][mult]
                control_alignments.extend([r["alignment"] for r in ctrl_results])
                control_confidences.extend([r["confidence"] for r in ctrl_results])
            layer_analysis["controls_mean"][mult] = {
                "mean_alignment": float(np.mean(control_alignments)),
                "alignment_change": float(np.mean(control_alignments) - baseline_mean_alignment),
                "mean_confidence": float(np.mean(control_confidences)),
                "confidence_change": float(np.mean(control_confidences) - baseline_mean_confidence),
            }

        analysis[layer_idx] = layer_analysis

    return analysis


def analyze_ablation_results(results: Dict) -> Dict:
    """Analyze ablation experiment results."""
    analysis = {}

    for layer_idx, layer_results in results["layer_results"].items():
        baseline_alignments = [r["alignment"] for r in layer_results["baseline"]]
        baseline_confidences = [r["confidence"] for r in layer_results["baseline"]]
        baseline_mean = np.mean(baseline_alignments)

        contrastive_alignments = [r["alignment"] for r in layer_results["contrastive_ablated"]]
        contrastive_confidences = [r["confidence"] for r in layer_results["contrastive_ablated"]]
        contrastive_mean = np.mean(contrastive_alignments)

        control_alignments = []
        control_confidences = []
        for ctrl_key in layer_results["controls_ablated"]:
            ctrl_results = layer_results["controls_ablated"][ctrl_key]
            control_alignments.extend([r["alignment"] for r in ctrl_results])
            control_confidences.extend([r["confidence"] for r in ctrl_results])
        controls_mean = np.mean(control_alignments)

        analysis[layer_idx] = {
            "baseline_mean_alignment": float(baseline_mean),
            "baseline_mean_confidence": float(np.mean(baseline_confidences)),
            "contrastive_ablated_mean_alignment": float(contrastive_mean),
            "contrastive_ablated_mean_confidence": float(np.mean(contrastive_confidences)),
            "contrastive_alignment_change": float(contrastive_mean - baseline_mean),
            "controls_ablated_mean_alignment": float(controls_mean),
            "controls_ablated_mean_confidence": float(np.mean(control_confidences)),
            "controls_alignment_change": float(controls_mean - baseline_mean),
            "contrastive_vs_controls": float((contrastive_mean - baseline_mean) - (controls_mean - baseline_mean)),
        }

    return analysis


def print_steering_summary(analysis: Dict):
    """Print steering analysis summary."""
    print("\n" + "=" * 70)
    print("STEERING ANALYSIS SUMMARY")
    print("=" * 70)

    for layer_idx in sorted(analysis.keys()):
        layer = analysis[layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  Baseline alignment: {layer['baseline_mean_alignment']:.4f}")
        print(f"  Contrastive effects (alignment change):")
        for mult in sorted(layer["contrastive"].keys()):
            change = layer["contrastive"][mult]["alignment_change"]
            print(f"    mult={mult:+.1f}: {change:+.4f}")
        print(f"  Control effects (mean alignment change):")
        for mult in sorted(layer["controls_mean"].keys()):
            change = layer["controls_mean"][mult]["alignment_change"]
            print(f"    mult={mult:+.1f}: {change:+.4f}")


def print_ablation_summary(analysis: Dict):
    """Print ablation analysis summary."""
    print("\n" + "=" * 70)
    print("ABLATION ANALYSIS SUMMARY")
    print("=" * 70)

    for layer_idx in sorted(analysis.keys()):
        layer = analysis[layer_idx]
        print(f"\nLayer {layer_idx}:")
        print(f"  Baseline alignment:      {layer['baseline_mean_alignment']:.4f}")
        print(f"  Contrastive ablated:     {layer['contrastive_ablated_mean_alignment']:.4f} ({layer['contrastive_alignment_change']:+.4f})")
        print(f"  Controls ablated (mean): {layer['controls_ablated_mean_alignment']:.4f} ({layer['controls_alignment_change']:+.4f})")
        print(f"  Contrastive vs Controls: {layer['contrastive_vs_controls']:+.4f}")


# ============================================================================
# CONTRASTIVE DIRECTION ANALYSIS
# ============================================================================

def compute_contrastive_direction_with_details(
    meta_activations: np.ndarray,
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25
) -> dict:
    """
    Compute contrastive direction based on confidence dimension.

    Contrasts correctly high-confidence examples vs correctly low-confidence examples:
    - High confidence group: high confidence AND low entropy (correctly confident)
    - Low confidence group: low confidence AND high entropy (correctly uncertain)

    This captures the confidence axis within calibrated examples only.

    Returns direction and detailed info about selected examples.
    """
    # Z-score normalize
    entropy_z = stats.zscore(direct_entropies)
    confidence_z = stats.zscore(stated_confidences)

    # Introspection score: positive when calibrated (confidence inversely tracks entropy)
    introspection_scores = -entropy_z * confidence_z

    # Only consider well-calibrated examples (positive introspection score)
    calibrated_mask = introspection_scores > 0

    # Within calibrated examples, split by confidence
    # High confidence + low entropy (correctly confident)
    high_conf_low_ent = calibrated_mask & (confidence_z > 0) & (entropy_z < 0)
    # Low confidence + high entropy (correctly uncertain)
    low_conf_high_ent = calibrated_mask & (confidence_z < 0) & (entropy_z > 0)

    high_conf_acts = meta_activations[high_conf_low_ent]
    low_conf_acts = meta_activations[low_conf_high_ent]

    if len(high_conf_acts) == 0 or len(low_conf_acts) == 0:
        raise ValueError(f"Not enough examples: high_conf={len(high_conf_acts)}, low_conf={len(low_conf_acts)}")

    # Compute direction: high confidence - low confidence
    high_conf_mean = high_conf_acts.mean(axis=0)
    low_conf_mean = low_conf_acts.mean(axis=0)

    direction = high_conf_mean - low_conf_mean
    direction_norm = np.linalg.norm(direction)
    direction_normalized = direction / direction_norm

    return {
        "direction": direction_normalized,
        "direction_magnitude": direction_norm,
        "n_high_conf": int(high_conf_low_ent.sum()),
        "n_low_conf": int(low_conf_high_ent.sum()),
        "n_calibrated": int(calibrated_mask.sum()),
        "high_conf_entropy_mean": float(direct_entropies[high_conf_low_ent].mean()),
        "high_conf_confidence_mean": float(stated_confidences[high_conf_low_ent].mean()),
        "low_conf_entropy_mean": float(direct_entropies[low_conf_high_ent].mean()),
        "low_conf_confidence_mean": float(stated_confidences[low_conf_high_ent].mean()),
    }


def analyze_selected_examples(
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray
) -> dict:
    """
    Analyze the characteristics of selected high-confidence vs low-confidence examples.

    Both groups are calibrated (correctly high or correctly low confidence).
    """
    # Z-scores for interpretation
    entropy_z = stats.zscore(direct_entropies)
    conf_z = stats.zscore(stated_confidences)

    # Introspection score: positive when calibrated
    introspection_scores = -entropy_z * conf_z
    calibrated_mask = introspection_scores > 0

    # High confidence + low entropy (correctly confident)
    high_conf_low_ent = calibrated_mask & (conf_z > 0) & (entropy_z < 0)
    # Low confidence + high entropy (correctly uncertain)
    low_conf_high_ent = calibrated_mask & (conf_z < 0) & (entropy_z > 0)

    print("\n" + "="*60)
    print("SELECTED EXAMPLES ANALYSIS")
    print("="*60)

    print(f"\nCalibrated examples (n={calibrated_mask.sum()}):")

    print(f"\nHigh confidence + low entropy (correctly confident, n={high_conf_low_ent.sum()}):")
    if high_conf_low_ent.sum() > 0:
        print(f"  Mean entropy: {direct_entropies[high_conf_low_ent].mean():.3f}")
        print(f"  Mean confidence: {stated_confidences[high_conf_low_ent].mean():.3f}")
        print(f"  Mean entropy z-score: {entropy_z[high_conf_low_ent].mean():.2f}")
        print(f"  Mean confidence z-score: {conf_z[high_conf_low_ent].mean():.2f}")

    print(f"\nLow confidence + high entropy (correctly uncertain, n={low_conf_high_ent.sum()}):")
    if low_conf_high_ent.sum() > 0:
        print(f"  Mean entropy: {direct_entropies[low_conf_high_ent].mean():.3f}")
        print(f"  Mean confidence: {stated_confidences[low_conf_high_ent].mean():.3f}")
        print(f"  Mean entropy z-score: {entropy_z[low_conf_high_ent].mean():.2f}")
        print(f"  Mean confidence z-score: {conf_z[low_conf_high_ent].mean():.2f}")

    return {
        "n_calibrated": int(calibrated_mask.sum()),
        "high_conf_low_ent": {
            "n": int(high_conf_low_ent.sum()),
            "entropy_mean": float(direct_entropies[high_conf_low_ent].mean()) if high_conf_low_ent.sum() > 0 else None,
            "confidence_mean": float(stated_confidences[high_conf_low_ent].mean()) if high_conf_low_ent.sum() > 0 else None,
        },
        "low_conf_high_ent": {
            "n": int(low_conf_high_ent.sum()),
            "entropy_mean": float(direct_entropies[low_conf_high_ent].mean()) if low_conf_high_ent.sum() > 0 else None,
            "confidence_mean": float(stated_confidences[low_conf_high_ent].mean()) if low_conf_high_ent.sum() > 0 else None,
        },
    }


def run_layer_analysis(
    meta_activations: dict,
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    top_quantile: float = 0.25,
    bottom_quantile: float = 0.25
) -> dict:
    """
    Compute contrastive direction for each layer and analyze.

    The contrastive direction is: high-confidence/low-entropy - low-confidence/high-entropy
    (i.e., correctly confident minus correctly uncertain)
    """
    print("\n" + "="*60)
    print("LAYER-BY-LAYER ANALYSIS")
    print("="*60)

    # Compute z-scores for correlation analysis
    confidence_z = stats.zscore(stated_confidences)

    results = {}

    for layer_idx in tqdm(sorted(meta_activations.keys())):
        acts = meta_activations[layer_idx]

        # Compute mean activation norm for normalization
        mean_activation_norm = float(np.linalg.norm(acts, axis=1).mean())

        # Compute contrastive direction
        dir_info = compute_contrastive_direction_with_details(
            acts, direct_entropies, stated_confidences, top_quantile, bottom_quantile
        )

        # Test how well projection correlates with stated confidence
        # (since direction is high_conf - low_conf, projection should predict confidence)
        proj = acts @ dir_info["direction"]
        corr, pval = stats.pearsonr(proj, confidence_z)

        results[layer_idx] = {
            **dir_info,
            "projection_correlation": float(corr),
            "projection_pvalue": float(pval),
            "mean_activation_norm": mean_activation_norm,
        }

    # Print summary
    print(f"\n{'Layer':<8} {'Dir Mag':<12} {'Proj Corr':<12} {'p-value':<12} {'N high':<8} {'N low':<8}")
    print("-" * 60)
    for layer_idx in sorted(results.keys()):
        r = results[layer_idx]
        print(f"{layer_idx:<8} {r['direction_magnitude']:<12.4f} "
              f"{r['projection_correlation']:<12.4f} {r['projection_pvalue']:<12.2e} "
              f"{r['n_high_conf']:<8} {r['n_low_conf']:<8}")

    # Find best layer
    best_layer = max(results.keys(), key=lambda l: abs(results[l]["projection_correlation"]))
    print(f"\nBest layer: {best_layer} (correlation = {results[best_layer]['projection_correlation']:.4f})")

    return results


def plot_results(
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    layer_results: dict,
    output_path: str = "contrastive_direction_results.png"
):
    """Plot analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Compute z-scores and masks for coloring
    entropy_z = stats.zscore(direct_entropies)
    conf_z = stats.zscore(stated_confidences)
    introspection_scores = -entropy_z * conf_z
    calibrated = introspection_scores > 0
    high_conf_low_ent = calibrated & (conf_z > 0) & (entropy_z < 0)
    low_conf_high_ent = calibrated & (conf_z < 0) & (entropy_z > 0)

    # 1. Introspection score distribution
    ax = axes[0, 0]
    ax.hist(introspection_scores, bins=50, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='gray', linestyle='--', label='Calibration boundary')
    ax.set_xlabel('Introspection Score')
    ax.set_ylabel('Count')
    ax.set_title('Introspection Score Distribution (>0 = calibrated)')
    ax.legend()

    # 2. Entropy vs Confidence with contrast group coloring
    ax = axes[0, 1]
    colors = ['green' if high_conf_low_ent[i] else 'blue' if low_conf_high_ent[i] else 'gray'
              for i in range(len(introspection_scores))]
    ax.scatter(direct_entropies, stated_confidences, c=colors, alpha=0.5, s=20)
    ax.set_xlabel('Direct Entropy')
    ax.set_ylabel('Stated Confidence')
    ax.set_title('Entropy vs Confidence\n(green=high conf/low ent, blue=low conf/high ent)')

    # Add trend line
    z = np.polyfit(direct_entropies, stated_confidences, 1)
    p = np.poly1d(z)
    x_line = np.linspace(direct_entropies.min(), direct_entropies.max(), 100)
    ax.plot(x_line, p(x_line), 'b--', alpha=0.5, label='Overall trend')
    ax.legend()

    # 3. Normalized direction magnitude by layer
    ax = axes[1, 0]
    layers = sorted(layer_results.keys())
    magnitudes = [layer_results[l]["direction_magnitude"] for l in layers]
    activation_norms = [layer_results[l]["mean_activation_norm"] for l in layers]
    normalized_magnitudes = [m / n for m, n in zip(magnitudes, activation_norms)]
    ax.plot(layers, normalized_magnitudes, 'o-')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Direction Magnitude / Activation Norm')
    ax.set_title('Normalized Contrastive Direction Magnitude by Layer')
    ax.grid(True, alpha=0.3)

    # 4. Projection correlation by layer
    ax = axes[1, 1]
    correlations = [layer_results[l]["projection_correlation"] for l in layers]
    ax.plot(layers, correlations, 'o-')
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Stated Confidence')
    ax.set_title('Direction Projection Correlation by Layer')
    ax.grid(True, alpha=0.3)

    # Highlight best layer
    best_layer = max(layers, key=lambda l: abs(layer_results[l]["projection_correlation"]))
    ax.scatter([best_layer], [layer_results[best_layer]["projection_correlation"]],
               color='red', s=100, zorder=5, label=f'Best: layer {best_layer}')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to {output_path}")


# ============================================================================
# DIRECTION COMPARISON
# ============================================================================

def compute_all_direction_types(
    meta_activations: Dict[int, np.ndarray],
    metric_values: np.ndarray,
    direct_entropies: np.ndarray,
    stated_confidences: np.ndarray,
    n_clusters: int = 3,
    cluster_method: str = "quantile"
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Compute multiple direction types for each layer.

    Direction types:
    - contrastive: High conf/low entropy - Low conf/high entropy (calibrated examples)
    - caa: Simple mean(high_metric) - mean(low_metric)
    - cluster_low_to_high: Direction from low to high cluster centroid
    - cluster_low_to_mid: Direction from low to mid cluster centroid
    - cluster_mid_to_high: Direction from mid to high cluster centroid

    Args:
        meta_activations: Dict mapping layer_idx to activations (n_samples, hidden_dim)
        metric_values: The metric to use for grouping (e.g., stated_confidences)
        direct_entropies: Entropy values (used for contrastive direction)
        stated_confidences: Confidence values (used for contrastive direction)
        n_clusters: Number of clusters for cluster-based directions
        cluster_method: "quantile" or "kmeans"

    Returns:
        Dict mapping layer_idx to Dict of direction_name -> direction_vector
    """
    print("\n" + "=" * 60)
    print("COMPUTING ALL DIRECTION TYPES")
    print("=" * 60)

    all_directions = {}

    for layer_idx in tqdm(sorted(meta_activations.keys()), desc="Computing directions"):
        acts = meta_activations[layer_idx]
        layer_directions = {}

        # 1. Contrastive direction (calibrated high conf vs low conf)
        try:
            contrastive_info = compute_contrastive_direction_with_details(
                acts, direct_entropies, stated_confidences
            )
            layer_directions["contrastive"] = contrastive_info["direction"]
        except ValueError as e:
            print(f"  Layer {layer_idx}: Could not compute contrastive direction: {e}")

        # 2. CAA direction (simple mean difference)
        try:
            caa_direction, caa_info = compute_caa_direction(
                acts, metric_values, high_quantile=0.25, low_quantile=0.25
            )
            layer_directions["caa"] = caa_direction
        except ValueError as e:
            print(f"  Layer {layer_idx}: Could not compute CAA direction: {e}")

        # 3. Cluster-based directions
        try:
            cluster_info = compute_cluster_centroids(
                acts, metric_values, n_clusters=n_clusters, method=cluster_method
            )
            cluster_dirs = compute_cluster_directions(
                cluster_info["centroids"], normalize=True
            )
            # Add relevant cluster directions
            for dir_name, direction in cluster_dirs.items():
                layer_directions[f"cluster_{dir_name}"] = direction
        except Exception as e:
            print(f"  Layer {layer_idx}: Could not compute cluster directions: {e}")

        all_directions[layer_idx] = layer_directions

    # Print summary
    if all_directions:
        first_layer = list(all_directions.keys())[0]
        dir_types = list(all_directions[first_layer].keys())
        print(f"\nComputed direction types: {dir_types}")
        print(f"Number of layers: {len(all_directions)}")

    return all_directions


def evaluate_direction_quality(
    meta_activations: Dict[int, np.ndarray],
    all_directions: Dict[int, Dict[str, np.ndarray]],
    metric_values: np.ndarray,
    metric_name: str = "confidence"
) -> Dict[int, Dict[str, Dict]]:
    """
    Evaluate quality of each direction type by measuring correlation with metric.

    For each layer and direction type:
    - Project activations onto direction
    - Compute correlation of projection with metric values
    - Compute R² (variance explained)

    Returns:
        Dict mapping layer_idx to Dict of direction_name -> quality_metrics
    """
    print("\n" + "=" * 60)
    print("EVALUATING DIRECTION QUALITY")
    print("=" * 60)

    # Z-score normalize metric
    metric_z = stats.zscore(metric_values)

    quality_results = {}

    for layer_idx in sorted(all_directions.keys()):
        acts = meta_activations[layer_idx]
        layer_results = {}

        for dir_name, direction in all_directions[layer_idx].items():
            # Project onto direction
            proj = acts @ direction

            # Compute correlation with metric
            corr, pval = stats.pearsonr(proj, metric_z)
            r_squared = corr ** 2

            layer_results[dir_name] = {
                "correlation": float(corr),
                "r_squared": float(r_squared),
                "p_value": float(pval),
            }

        quality_results[layer_idx] = layer_results

    # Print summary table
    if quality_results:
        first_layer = list(quality_results.keys())[0]
        dir_types = list(quality_results[first_layer].keys())

        print(f"\n{'Layer':<8}", end="")
        for dt in dir_types:
            short_name = dt[:12]
            print(f"{short_name:<15}", end="")
        print()
        print("-" * (8 + 15 * len(dir_types)))

        for layer_idx in sorted(quality_results.keys()):
            print(f"{layer_idx:<8}", end="")
            for dt in dir_types:
                if dt in quality_results[layer_idx]:
                    corr = quality_results[layer_idx][dt]["correlation"]
                    print(f"{corr:+.4f}       ", end="")
                else:
                    print(f"{'N/A':<15}", end="")
            print()

    return quality_results


def compare_direction_similarities(
    all_directions: Dict[int, Dict[str, np.ndarray]],
    layers: Optional[List[int]] = None
) -> Dict[int, Dict[str, float]]:
    """
    Compare similarity between different direction types at each layer.

    Returns:
        Dict mapping layer_idx to Dict of "dirA_vs_dirB" -> cosine_similarity
    """
    print("\n" + "=" * 60)
    print("DIRECTION SIMILARITY ANALYSIS")
    print("=" * 60)

    if layers is None:
        layers = sorted(all_directions.keys())

    similarity_results = {}

    for layer_idx in layers:
        if layer_idx not in all_directions:
            continue

        layer_dirs = all_directions[layer_idx]
        if len(layer_dirs) < 2:
            continue

        # Use the compare_directions utility from core.probes
        similarities = compare_directions(layer_dirs)
        similarity_results[layer_idx] = similarities

    # Print summary for key comparisons
    if similarity_results:
        first_layer = list(similarity_results.keys())[0]
        comparison_keys = list(similarity_results[first_layer].keys())

        # Focus on key comparisons
        key_comparisons = [k for k in comparison_keys if "contrastive" in k or "caa" in k]
        if key_comparisons:
            print(f"\nKey direction comparisons (cosine similarity):")
            print(f"{'Layer':<8}", end="")
            for comp in key_comparisons[:5]:  # Limit to 5 for readability
                print(f"{comp[:18]:<20}", end="")
            print()
            print("-" * (8 + 20 * min(5, len(key_comparisons))))

            for layer_idx in sorted(similarity_results.keys()):
                print(f"{layer_idx:<8}", end="")
                for comp in key_comparisons[:5]:
                    if comp in similarity_results[layer_idx]:
                        sim = similarity_results[layer_idx][comp]
                        print(f"{sim:.4f}              ", end="")
                    else:
                        print(f"{'N/A':<20}", end="")
                print()

    return similarity_results


def run_direction_comparison_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    all_directions: Dict[int, Dict[str, np.ndarray]],
    multipliers: List[float],
    use_chat_template: bool,
    batch_size: int = 8
) -> Dict:
    """
    Run steering experiment comparing different direction types.

    For each layer and direction type, run steering and measure effect.
    This allows comparing the causal efficacy of different direction computation methods.

    Returns:
        Dict with per-layer, per-direction-type steering results
    """
    print("\n" + "=" * 70)
    print("DIRECTION COMPARISON STEERING EXPERIMENT")
    print("=" * 70)
    print(f"  Layers: {layers}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Batch size: {batch_size}")

    # Get direction types from first layer
    if not all_directions:
        print("No directions to compare!")
        return {}

    first_layer = list(all_directions.keys())[0]
    direction_types = list(all_directions[first_layer].keys())
    print(f"  Direction types: {direction_types}")

    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    def compute_results_from_batch(batch_results):
        """Convert batch results to per-question result dicts."""
        out = []
        for q_idx, (response, confidence, probs, mapping) in enumerate(batch_results):
            entropy = direct_entropies[q_idx]
            entropy_z = (entropy - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = -entropy_z * confidence_z

            out.append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })
        return out

    results = {
        "layers": layers,
        "multipliers": multipliers,
        "direction_types": direction_types,
        "num_questions": len(questions),
        "layer_results": {},
    }

    # Compute baseline once
    print("\nComputing baseline (no steering)...")
    baseline_batch = get_batch_confidence_responses(
        model, tokenizer, questions, None, None, 0.0, use_chat_template, batch_size
    )
    shared_baseline = compute_results_from_batch(baseline_batch)

    for layer_idx in tqdm(layers, desc="Layers"):
        if layer_idx not in all_directions:
            continue

        layer_dirs = all_directions[layer_idx]
        layer_results = {
            "baseline": shared_baseline,
            "direction_results": {},
        }

        for dir_type, direction in layer_dirs.items():
            dir_results = {m: [] for m in multipliers}

            for mult in tqdm(multipliers, desc=f"L{layer_idx}/{dir_type[:10]}", leave=False):
                if mult == 0.0:
                    dir_results[mult] = layer_results["baseline"]
                    continue

                batch_results = get_batch_confidence_responses(
                    model, tokenizer, questions, layer_idx, direction, mult,
                    use_chat_template, batch_size
                )
                dir_results[mult] = compute_results_from_batch(batch_results)

            layer_results["direction_results"][dir_type] = dir_results

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def analyze_direction_comparison_results(results: Dict) -> Dict:
    """
    Analyze direction comparison experiment results.

    For each layer and direction type, compute:
    - Mean confidence change at each multiplier
    - Effect size (confidence change per unit multiplier)
    - Comparison to other direction types
    """
    analysis = {
        "layer_analysis": {},
        "direction_type_summary": {},
    }

    all_direction_types = results.get("direction_types", [])
    multipliers = results.get("multipliers", [])

    # Initialize summary accumulators
    for dt in all_direction_types:
        analysis["direction_type_summary"][dt] = {
            "mean_effect": [],
            "layers": [],
        }

    for layer_idx, layer_results in results.get("layer_results", {}).items():
        baseline_confidences = [r["confidence"] for r in layer_results["baseline"]]
        baseline_mean = np.mean(baseline_confidences)

        layer_analysis = {
            "baseline_mean_confidence": float(baseline_mean),
            "direction_effects": {},
        }

        for dir_type, dir_results in layer_results.get("direction_results", {}).items():
            effects = {}
            effect_per_mult = []

            for mult in multipliers:
                if mult == 0.0:
                    effects[mult] = {
                        "mean_confidence": float(baseline_mean),
                        "confidence_change": 0.0,
                    }
                    continue

                mult_confidences = [r["confidence"] for r in dir_results[mult]]
                mult_mean = np.mean(mult_confidences)
                change = mult_mean - baseline_mean

                effects[mult] = {
                    "mean_confidence": float(mult_mean),
                    "confidence_change": float(change),
                }

                # Track effect per unit multiplier (for effect size)
                effect_per_mult.append(change / abs(mult))

            # Compute overall effect size (mean absolute effect per unit multiplier)
            if effect_per_mult:
                mean_effect = np.mean(np.abs(effect_per_mult))
            else:
                mean_effect = 0.0

            layer_analysis["direction_effects"][dir_type] = {
                "multiplier_effects": effects,
                "mean_effect_per_unit": float(mean_effect),
            }

            # Accumulate for summary
            if dir_type in analysis["direction_type_summary"]:
                analysis["direction_type_summary"][dir_type]["mean_effect"].append(mean_effect)
                analysis["direction_type_summary"][dir_type]["layers"].append(layer_idx)

        analysis["layer_analysis"][layer_idx] = layer_analysis

    # Compute overall summary for each direction type
    for dt in all_direction_types:
        if dt in analysis["direction_type_summary"]:
            effects = analysis["direction_type_summary"][dt]["mean_effect"]
            if effects:
                analysis["direction_type_summary"][dt]["overall_mean_effect"] = float(np.mean(effects))
                analysis["direction_type_summary"][dt]["overall_std_effect"] = float(np.std(effects))
            else:
                analysis["direction_type_summary"][dt]["overall_mean_effect"] = 0.0
                analysis["direction_type_summary"][dt]["overall_std_effect"] = 0.0

    return analysis


def print_direction_comparison_summary(analysis: Dict):
    """Print summary of direction comparison analysis."""
    print("\n" + "=" * 70)
    print("DIRECTION COMPARISON SUMMARY")
    print("=" * 70)

    # Overall direction type ranking
    print("\nOverall Direction Type Effectiveness (mean effect per unit multiplier):")
    print(f"{'Direction Type':<25} {'Mean Effect':<15} {'Std':<15}")
    print("-" * 55)

    summary = analysis.get("direction_type_summary", {})
    sorted_types = sorted(
        summary.keys(),
        key=lambda dt: summary[dt].get("overall_mean_effect", 0),
        reverse=True
    )

    for dt in sorted_types:
        mean_eff = summary[dt].get("overall_mean_effect", 0)
        std_eff = summary[dt].get("overall_std_effect", 0)
        print(f"{dt:<25} {mean_eff:<15.4f} {std_eff:<15.4f}")

    # Per-layer breakdown
    print("\nPer-Layer Effect Comparison:")
    layer_analysis = analysis.get("layer_analysis", {})

    for layer_idx in sorted(layer_analysis.keys()):
        layer = layer_analysis[layer_idx]
        print(f"\n  Layer {layer_idx}:")
        print(f"    {'Direction':<20} {'Effect/mult':<15}")
        print(f"    {'-'*35}")

        effects = layer.get("direction_effects", {})
        sorted_dirs = sorted(
            effects.keys(),
            key=lambda d: effects[d].get("mean_effect_per_unit", 0),
            reverse=True
        )

        for dt in sorted_dirs:
            eff = effects[dt].get("mean_effect_per_unit", 0)
            print(f"    {dt:<20} {eff:.4f}")


def plot_direction_comparison(
    quality_results: Dict[int, Dict[str, Dict]],
    similarity_results: Dict[int, Dict[str, float]],
    steering_analysis: Optional[Dict] = None,
    output_path: str = "direction_comparison.png"
):
    """Plot direction comparison results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Get data
    layers = sorted(quality_results.keys())
    if not layers:
        print("No quality results to plot")
        return

    first_layer = layers[0]
    direction_types = list(quality_results[first_layer].keys())

    # 1. Correlation by layer for each direction type
    ax = axes[0, 0]
    for dt in direction_types:
        correlations = []
        valid_layers = []
        for layer_idx in layers:
            if dt in quality_results[layer_idx]:
                correlations.append(quality_results[layer_idx][dt]["correlation"])
                valid_layers.append(layer_idx)
        if correlations:
            ax.plot(valid_layers, correlations, 'o-', label=dt[:15], alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation with Metric')
    ax.set_title('Direction Quality by Layer\n(Correlation of Projection with Metric)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 2. R² by layer
    ax = axes[0, 1]
    for dt in direction_types:
        r2_values = []
        valid_layers = []
        for layer_idx in layers:
            if dt in quality_results[layer_idx]:
                r2_values.append(quality_results[layer_idx][dt]["r_squared"])
                valid_layers.append(layer_idx)
        if r2_values:
            ax.plot(valid_layers, r2_values, 'o-', label=dt[:15], alpha=0.7)
    ax.set_xlabel('Layer')
    ax.set_ylabel('R² (Variance Explained)')
    ax.set_title('Direction Quality by Layer\n(R² of Projection)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Direction similarity (contrastive vs others)
    ax = axes[1, 0]
    if similarity_results:
        # Find comparisons involving contrastive
        sample_layer = list(similarity_results.keys())[0]
        contrastive_comps = [k for k in similarity_results[sample_layer].keys()
                           if "contrastive" in k and k != "contrastive_vs_contrastive"]

        for comp in contrastive_comps[:5]:  # Limit for readability
            similarities = []
            valid_layers = []
            for layer_idx in layers:
                if layer_idx in similarity_results and comp in similarity_results[layer_idx]:
                    similarities.append(similarity_results[layer_idx][comp])
                    valid_layers.append(layer_idx)
            if similarities:
                ax.plot(valid_layers, similarities, 'o-', label=comp[:20], alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Direction Similarity\n(Contrastive vs Other Methods)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No similarity data', ha='center', va='center')
        ax.set_title('Direction Similarity')

    # 4. Steering effect comparison (if available)
    ax = axes[1, 1]
    if steering_analysis and "layer_analysis" in steering_analysis:
        layer_analysis = steering_analysis["layer_analysis"]
        for dt in direction_types:
            effects = []
            valid_layers = []
            for layer_idx in layers:
                if (layer_idx in layer_analysis and
                    dt in layer_analysis[layer_idx].get("direction_effects", {})):
                    eff = layer_analysis[layer_idx]["direction_effects"][dt].get("mean_effect_per_unit", 0)
                    effects.append(eff)
                    valid_layers.append(layer_idx)
            if effects:
                ax.plot(valid_layers, effects, 'o-', label=dt[:15], alpha=0.7)

        ax.set_xlabel('Layer')
        ax.set_ylabel('Effect per Unit Multiplier')
        ax.set_title('Steering Effect by Direction Type')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No steering data\n(Run with RUN_STEERING=True)', ha='center', va='center')
        ax.set_title('Steering Effect Comparison')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDirection comparison plot saved to {output_path}")


def main():
    print(f"Device: {DEVICE}")
    print(f"Contrastive selection: top {TOP_QUANTILE*100:.0f}% vs bottom {BOTTOM_QUANTILE*100:.0f}%")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Load data
    data = load_introspection_data()

    direct_entropies = data["direct_entropies"]
    stated_confidences = data["stated_confidences"]
    meta_activations = data["meta_activations"]
    paired_data = data["paired_data"]
    num_layers = data["num_layers"]

    # Compute introspection scores
    print("\nComputing introspection scores...")
    introspection_scores = compute_introspection_scores(direct_entropies, stated_confidences)

    print(f"Introspection score range: [{introspection_scores.min():.3f}, {introspection_scores.max():.3f}]")
    print(f"Mean: {introspection_scores.mean():.3f}, Std: {introspection_scores.std():.3f}")

    # Run layer-by-layer analysis
    layer_results = run_layer_analysis(
        meta_activations, direct_entropies, stated_confidences, TOP_QUANTILE, BOTTOM_QUANTILE
    )

    # Get best layer for detailed analysis
    best_layer = max(layer_results.keys(), key=lambda l: abs(layer_results[l]["projection_correlation"]))

    # Detailed analysis on best layer
    print(f"\n{'='*60}")
    print(f"DETAILED ANALYSIS (Layer {best_layer})")
    print(f"{'='*60}")

    best_acts = meta_activations[best_layer]
    best_result = layer_results[best_layer]

    # Analyze selected examples
    example_analysis = analyze_selected_examples(
        direct_entropies,
        stated_confidences
    )

    # Print direction quality summary
    print(f"\n{'='*60}")
    print("DIRECTION QUALITY SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Layer':<8} {'Proj Corr':<12} {'R²':<12} {'p-value':<12}")
    print("-" * 44)
    for layer_idx in sorted(layer_results.keys()):
        r = layer_results[layer_idx]
        corr = r["projection_correlation"]
        r2 = corr ** 2  # R² = correlation² for single predictor
        pval = r["projection_pvalue"]
        print(f"{layer_idx:<8} {corr:<12.4f} {r2:<12.4f} {pval:<12.2e}")

    # Plot results
    plot_results(
        direct_entropies,
        stated_confidences,
        layer_results,
        output_path=f"{output_prefix}_results.png"
    )

    # Save results
    results = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "dataset": DATASET_NAME,
            "top_quantile": TOP_QUANTILE,
            "bottom_quantile": BOTTOM_QUANTILE,
            "seed": SEED,
            "meta_task": META_TASK,
        },
        "best_layer": best_layer,
        "layer_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "direction"}
            for k, v in layer_results.items()
        },
        "example_analysis": example_analysis,
    }

    results_path = f"{output_prefix}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
    print(f"\nResults saved to {results_path}")

    # Save directions
    directions_dict = {
        layer_idx: layer_results[layer_idx]["direction"]
        for layer_idx in layer_results.keys()
    }
    directions_for_save = {
        f"layer_{k}": v
        for k, v in directions_dict.items()
    }
    np.savez_compressed(f"{output_prefix}_directions.npz", **directions_for_save)
    print(f"Directions saved to {output_prefix}_directions.npz")

    # ========================================================================
    # DIRECTION COMPARISON (if enabled)
    # ========================================================================
    all_directions = None
    quality_results = None
    similarity_results = None
    direction_comparison_analysis = None

    if COMPARE_DIRECTIONS:
        print("\n" + "=" * 70)
        print("DIRECTION COMPARISON MODE")
        print("=" * 70)

        # Compute all direction types for each layer
        all_directions = compute_all_direction_types(
            meta_activations,
            metric_values=stated_confidences,
            direct_entropies=direct_entropies,
            stated_confidences=stated_confidences,
            n_clusters=N_CLUSTERS,
            cluster_method=CLUSTER_METHOD
        )

        # Evaluate quality of each direction type
        quality_results = evaluate_direction_quality(
            meta_activations,
            all_directions,
            metric_values=stated_confidences,
            metric_name="confidence"
        )

        # Compare similarity between direction types
        similarity_results = compare_direction_similarities(all_directions)

        # Save all directions
        all_directions_for_save = {}
        for layer_idx, layer_dirs in all_directions.items():
            for dir_name, direction in layer_dirs.items():
                all_directions_for_save[f"layer_{layer_idx}_{dir_name}"] = direction
        np.savez_compressed(f"{output_prefix}_all_directions.npz", **all_directions_for_save)
        print(f"\nAll directions saved to {output_prefix}_all_directions.npz")

        # Save quality results
        quality_results_serializable = {
            str(k): v for k, v in quality_results.items()
        }
        quality_path = f"{output_prefix}_direction_quality.json"
        with open(quality_path, "w") as f:
            json.dump(quality_results_serializable, f, indent=2)
        print(f"Direction quality saved to {quality_path}")

        # Save similarity results
        similarity_results_serializable = {
            str(k): v for k, v in similarity_results.items()
        }
        similarity_path = f"{output_prefix}_direction_similarity.json"
        with open(similarity_path, "w") as f:
            json.dump(similarity_results_serializable, f, indent=2)
        print(f"Direction similarity saved to {similarity_path}")

    # ========================================================================
    # STEERING/ABLATION EXPERIMENTS
    # ========================================================================
    if RUN_STEERING:
        print("\n" + "=" * 70)
        print("STEERING/ABLATION EXPERIMENTS")
        print("=" * 70)

        # Load model for steering
        print(f"\nLoading model: {BASE_MODEL_NAME}")
        adapter_path = MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
        model, tokenizer, _ = load_model_and_tokenizer(BASE_MODEL_NAME, adapter_path)
        use_chat_template = not is_base_model(BASE_MODEL_NAME)

        # Determine which layers to steer
        if STEERING_LAYERS is not None:
            steering_layers = STEERING_LAYERS
        else:
            # Select layers with projection correlation above threshold
            steering_layers = [
                layer_idx for layer_idx, r in layer_results.items()
                if abs(r["projection_correlation"]) > MIN_PROJECTION_CORR
            ]
            # Sort numerically for consistent presentation
            steering_layers.sort()

        if not steering_layers:
            print(f"No layers with projection correlation > {MIN_PROJECTION_CORR}. Skipping steering.")
        else:
            print(f"Selected layers for steering: {steering_layers}")

            # Get questions for steering
            questions = paired_data.get("questions", [])
            if len(questions) > NUM_STEERING_QUESTIONS:
                questions = questions[:NUM_STEERING_QUESTIONS]
                steering_entropies = direct_entropies[:NUM_STEERING_QUESTIONS]
            else:
                steering_entropies = direct_entropies

            print(f"Using {len(questions)} questions for steering")

            # Run steering experiment
            steering_results = run_steering_experiment(
                model, tokenizer, questions, steering_entropies,
                steering_layers, directions_dict, STEERING_MULTIPLIERS,
                NUM_CONTROL_DIRECTIONS, use_chat_template,
                batch_size=STEERING_BATCH_SIZE
            )

            # Analyze steering results
            steering_analysis = analyze_steering_results(steering_results)
            print_steering_summary(steering_analysis)

            # Save steering results
            steering_results_path = f"{output_prefix}_steering_results.json"
            with open(steering_results_path, "w") as f:
                json.dump(steering_results, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"\nSteering results saved to {steering_results_path}")

            steering_analysis_path = f"{output_prefix}_steering_analysis.json"
            with open(steering_analysis_path, "w") as f:
                json.dump(steering_analysis, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"Steering analysis saved to {steering_analysis_path}")

            # Run ablation experiment
            print("\n" + "=" * 70)
            print("RUNNING ABLATION EXPERIMENT")
            print("=" * 70)

            # Reuse baseline from steering
            first_layer = steering_layers[0]
            baseline_from_steering = steering_results["layer_results"][first_layer]["baseline"]

            ablation_results = run_ablation_experiment(
                model, tokenizer, questions, steering_entropies,
                steering_layers, directions_dict, NUM_CONTROL_DIRECTIONS,
                use_chat_template, baseline_results=baseline_from_steering,
                batch_size=STEERING_BATCH_SIZE
            )

            # Analyze ablation results
            ablation_analysis = analyze_ablation_results(ablation_results)
            print_ablation_summary(ablation_analysis)

            # Save ablation results
            ablation_results_path = f"{output_prefix}_ablation_results.json"
            with open(ablation_results_path, "w") as f:
                json.dump(ablation_results, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"\nAblation results saved to {ablation_results_path}")

            ablation_analysis_path = f"{output_prefix}_ablation_analysis.json"
            with open(ablation_analysis_path, "w") as f:
                json.dump(ablation_analysis, f, indent=2,
                          default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
            print(f"Ablation analysis saved to {ablation_analysis_path}")

            # ================================================================
            # DIRECTION COMPARISON STEERING (if enabled)
            # ================================================================
            if COMPARE_DIRECTIONS and all_directions is not None:
                print("\n" + "=" * 70)
                print("DIRECTION COMPARISON STEERING EXPERIMENT")
                print("=" * 70)

                # Run steering experiment comparing all direction types
                direction_comparison_results = run_direction_comparison_experiment(
                    model, tokenizer, questions, steering_entropies,
                    steering_layers, all_directions, STEERING_MULTIPLIERS,
                    use_chat_template, batch_size=STEERING_BATCH_SIZE
                )

                # Analyze results
                direction_comparison_analysis = analyze_direction_comparison_results(
                    direction_comparison_results
                )
                print_direction_comparison_summary(direction_comparison_analysis)

                # Save direction comparison results
                direction_comp_path = f"{output_prefix}_direction_comparison_results.json"
                with open(direction_comp_path, "w") as f:
                    json.dump(direction_comparison_results, f, indent=2,
                              default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
                print(f"\nDirection comparison results saved to {direction_comp_path}")

                direction_comp_analysis_path = f"{output_prefix}_direction_comparison_analysis.json"
                with open(direction_comp_analysis_path, "w") as f:
                    json.dump(direction_comparison_analysis, f, indent=2,
                              default=lambda x: float(x) if isinstance(x, (np.floating, np.float16, np.float32, np.float64)) else x)
                print(f"Direction comparison analysis saved to {direction_comp_analysis_path}")

    # ========================================================================
    # FINAL PLOTS (if direction comparison was enabled)
    # ========================================================================
    if COMPARE_DIRECTIONS and quality_results is not None:
        plot_direction_comparison(
            quality_results,
            similarity_results or {},
            steering_analysis=direction_comparison_analysis,
            output_path=f"{output_prefix}_direction_comparison.png"
        )

    print("\n" + "=" * 70)
    print("CONTRASTIVE DIRECTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
