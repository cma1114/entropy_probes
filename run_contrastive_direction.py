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

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Meta confidence scale (for confidence task)
META_OPTION_DICT = {
    "S": "<5%", "T": "5-10%", "U": "10-20%", "V": "20-40%",
    "W": "40-60%", "X": "60-80%", "Y": "80-90%", "Z": ">90%"
}
META_RANGE_MIDPOINTS = {
    "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
    "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
}
META_OPTIONS = list(META_OPTION_DICT.keys())

META_SETUP_PROMPT = "I'm going to show you a question, and I want you to tell me your level of confidence that you know the correct answer to this question. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
META_QUESTION_PROMPT = "How confident are you that you know the correct answer to this question?"

# Delegate task prompts and options
DELEGATE_SETUP_PROMPT = """You are playing a game where you can either:
1. Answer a multiple choice question yourself, OR
2. Delegate to a teammate who answers 50% of questions correctly

If you think you're MORE likely than 50% to get it right, you should answer yourself.
If you think you're LESS likely than 50% to get it right, you should delegate.

I'm going to show you a question. Based on the question, decide whether to ANSWER yourself or DELEGATE to your teammate.
"""
DELEGATE_SYSPROMPT = "Respond ONLY with '1' or '2'. Do NOT output any other text. Do not offer any explanation or justification.\n"
DELEGATE_OPTIONS = ["1", "2"]


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
# STEERING HELPERS
# ============================================================================

def is_base_model(model_name: str) -> bool:
    """Check if model is a base model (not instruction-tuned)."""
    model_lower = model_name.lower()
    instruct_indicators = ['instruct', 'chat', '-it', 'rlhf', 'sft', 'dpo']
    return not any(ind in model_lower for ind in instruct_indicators)


def has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True
        )
        return True
    except Exception:
        return False


def get_delegate_mapping(trial_index: int) -> Dict[str, str]:
    """Return how digits map for this trial (alternates for position bias control)."""
    if (trial_index % 2) == 1:
        return {"1": "Answer", "2": "Delegate"}
    else:
        return {"1": "Delegate", "2": "Answer"}


def _present_nested_question(question_data: Dict, outer_question: str, outer_options: Dict) -> str:
    """Format a nested/meta question for display."""
    formatted = "-" * 30 + "\n"
    formatted += outer_question + "\n"
    formatted += "-" * 10 + "\n"
    formatted += question_data["question"] + "\n"
    if "options" in question_data:
        for key, value in question_data["options"].items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 10 + "\n"
    if outer_options:
        for key, value in outer_options.items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 30
    return formatted


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format a meta/confidence question."""
    q_text = _present_nested_question(question, META_QUESTION_PROMPT, META_OPTION_DICT)
    options_str = ", ".join(META_OPTIONS[:-1]) + f", or {META_OPTIONS[-1]}"
    llm_prompt = q_text + f"\nYour choice ({options_str}): "

    if use_chat_template and has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": META_SETUP_PROMPT},
            {"role": "user", "content": llm_prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"{META_SETUP_PROMPT}\n\n{llm_prompt}"


def format_delegate_prompt(
    question: Dict,
    tokenizer,
    use_chat_template: bool = True,
    trial_index: int = 0
) -> Tuple[str, List[str], Dict[str, str]]:
    """Format a delegate question with alternating mapping."""
    mapping = get_delegate_mapping(trial_index)

    formatted_question = "-" * 30 + "\n"
    formatted_question += "Question:\n"
    formatted_question += question["question"] + "\n"

    if "options" in question:
        formatted_question += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted_question += f"  {key}: {value}\n"

    formatted_question += "-" * 30 + "\n"
    formatted_question += "Choices:\n"
    formatted_question += f"  1: {mapping['1']}\n"
    formatted_question += f"  2: {mapping['2']}\n"
    formatted_question += "Respond ONLY with '1' or '2'.\n"
    formatted_question += "Your choice ('1' or '2'):"

    if use_chat_template and has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": DELEGATE_SYSPROMPT + DELEGATE_SETUP_PROMPT},
            {"role": "user", "content": formatted_question}
        ]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        full_prompt = f"{DELEGATE_SYSPROMPT}\n{DELEGATE_SETUP_PROMPT}\n\n{formatted_question}"

    return full_prompt, DELEGATE_OPTIONS, mapping


def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """Convert a meta response to a confidence value."""
    if META_TASK == "delegate":
        if probs is not None and len(probs) >= 2 and mapping is not None:
            if mapping.get("1") == "Answer":
                return float(probs[0])
            else:
                return float(probs[1])
        elif probs is not None and len(probs) >= 1:
            return float(probs[0])
        if mapping is not None:
            return 1.0 if mapping.get(response) == "Answer" else 0.0
        return 1.0 if response == "1" else 0.0
    else:
        return META_RANGE_MIDPOINTS.get(response, 0.5)


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

    print("\n" + "=" * 70)
    print("CONTRASTIVE DIRECTION ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
