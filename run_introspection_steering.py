"""
Steering and ablation experiments using probe directions.

This script supports three types of directions:
1. "introspection" - From run_introspection_probe.py (meta activations → introspection_score)
2. "entropy" - From run_introspection_experiment.py (direct activations → entropy)
3. "shared" - From analyze_shared_unique.py (shared MC entropy direction across datasets)

Set DIRECTION_TYPE at the top to choose which direction to use.

For "shared" direction type:
- Loads shared component from *_shared_unique_directions.npz
- Uses META_R2_THRESHOLD to filter layers (only tests layers where direct→meta R² >= threshold)
- Tests whether the shared uncertainty signal (common across datasets) is causal for
  the model's confidence judgments

The script:
1. Loads probe results and directions from probe training
2. Automatically selects layers based on probe performance or transfer R²
3. Runs steering experiments with the probe direction and control directions
4. Runs ablation experiments to test causality (zeroing out the direction)
5. Measures effect on alignment between stated confidence and actual entropy
6. Computes p-values vs random control directions for statistical significance

Ablation tests the hypothesis: if the direction is causal for the model's
confidence judgments, removing it should degrade the correlation between
stated confidence and actual entropy.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import random

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model
DATASET_NAME = "SimpleMC"

# Direction type: "introspection", "entropy", or "shared"
# - "introspection": Uses directions from run_introspection_probe.py
#                    (meta activations → introspection_score)
# - "entropy": Uses directions from run_introspection_experiment.py
#              (direct activations → entropy, for direct→meta transfer test)
# - "shared": Uses shared MC entropy direction from analyze_shared_unique.py
#             (average of MC directions across datasets - tests general uncertainty)
DIRECTION_TYPE = "shared"  # "introspection", "entropy", or "shared"

# Threshold for layer selection when using "shared" direction type
# Only layers where direct→meta R² exceeds this threshold will be tested
META_R2_THRESHOLD = 0.25

# Meta-judgment task: "confidence" or "delegate"
# - "confidence": Explicit confidence rating on S-Z scale
# - "delegate": Answer vs Delegate choice; confidence = P(Answer)
# This should match the META_TASK used in run_introspection_experiment.py
META_TASK = "confidence"  # "confidence" or "delegate"

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
    # Add delegate suffix if using delegate task (matches run_introspection_experiment.py)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


# Steering config
STEERING_LAYERS = None  # None = auto-select from probe results
STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
NUM_STEERING_QUESTIONS = 100
NUM_CONTROL_DIRECTIONS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

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
DELEGATE_OPTIONS = ["1", "2"]  # Mapping alternates per trial


def get_delegate_mapping(trial_index: int) -> Dict[str, str]:
    """
    Return how digits map for this trial.

    Matches run_introspection_experiment.py with alternate mapping:
    - Odd trials (1, 3, 5, ...): 1=Answer, 2=Delegate
    - Even trials (0, 2, 4, ...): 1=Delegate, 2=Answer
    """
    if (trial_index % 2) == 1:  # Odd: 1=Answer
        return {"1": "Answer", "2": "Delegate"}
    else:  # Even: 1=Delegate
        return {"1": "Delegate", "2": "Answer"}


# ============================================================================
# HELPERS
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


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

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
    """
    Format a delegate (Answer vs Delegate) question.

    Returns:
        Tuple of (full_prompt, delegate_option_keys, mapping)
        where mapping is {"1": "Answer"/"Delegate", "2": "Delegate"/"Answer"}
    """
    # Get the mapping for this trial (alternates to control position bias)
    mapping = get_delegate_mapping(trial_index)

    # Format the question with MC options
    formatted_question = ""
    formatted_question += "-" * 30 + "\n"
    formatted_question += "Question:\n"
    formatted_question += question["question"] + "\n"

    if "options" in question:
        formatted_question += "-" * 10 + "\n"
        for key, value in question["options"].items():
            formatted_question += f"  {key}: {value}\n"

    formatted_question += "-" * 30 + "\n"

    # Add delegate choice with the current mapping
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

        full_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    else:
        full_prompt = f"{DELEGATE_SYSPROMPT}\n{DELEGATE_SETUP_PROMPT}\n\n{formatted_question}"

    return full_prompt, DELEGATE_OPTIONS, mapping


def get_meta_options() -> List[str]:
    """Return the meta options based on META_TASK setting."""
    if META_TASK == "delegate":
        return DELEGATE_OPTIONS
    else:
        return META_OPTIONS


def response_to_confidence(
    response: str,
    probs: np.ndarray = None,
    mapping: Dict[str, str] = None
) -> float:
    """
    Convert a meta response to a confidence value.

    For confidence task: Uses META_RANGE_MIDPOINTS lookup
    For delegate task: Uses P(Answer) from the probability distribution,
                       accounting for alternating mapping
    """
    if META_TASK == "delegate":
        # For delegate task, confidence = P(Answer)
        if probs is not None and len(probs) >= 2 and mapping is not None:
            if mapping.get("1") == "Answer":
                return float(probs[0])  # P("1") = P(Answer)
            else:
                return float(probs[1])  # P("2") = P(Answer)
        elif probs is not None and len(probs) >= 1:
            return float(probs[0])
        if mapping is not None:
            return 1.0 if mapping.get(response) == "Answer" else 0.0
        return 1.0 if response == "1" else 0.0
    else:
        # For confidence task, use the midpoint lookup
        return META_RANGE_MIDPOINTS.get(response, 0.5)


# ============================================================================
# STEERING AND ABLATION
# ============================================================================

class SteeringHook:
    """Hook that adds a steering vector to activations."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float):
        # Ensure normalized so multiplier has consistent meaning across directions
        self.steering_vector = steering_vector / steering_vector.norm()
        self.multiplier = multiplier
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            steered = hidden_states + self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:]
        else:
            return output + self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


class AblationHook:
    """
    Hook that removes the component of activations along a direction.

    Projects out the direction: x' = x - (x · d) * d
    This tests whether the direction is causally involved in the behavior.
    """

    def __init__(self, direction: torch.Tensor):
        # Ensure normalized
        self.direction = direction / direction.norm()
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Project out the direction from all tokens
        # hidden_states: (batch, seq_len, hidden_dim)
        # direction: (hidden_dim,)
        proj = (hidden_states @ self.direction).unsqueeze(-1) * self.direction
        ablated = hidden_states - proj

        if isinstance(output, tuple):
            return (ablated,) + output[1:]
        return ablated

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


def generate_orthogonal_directions(direction: np.ndarray, num_directions: int) -> List[np.ndarray]:
    """Generate random directions orthogonal to the given direction."""
    hidden_dim = len(direction)
    orthogonal = []

    for _ in range(num_directions):
        random_vec = np.random.randn(hidden_dim)
        random_vec = random_vec - np.dot(random_vec, direction) * direction
        for prev in orthogonal:
            random_vec = random_vec - np.dot(random_vec, prev) * prev
        random_vec = random_vec / np.linalg.norm(random_vec)
        orthogonal.append(random_vec)

    return orthogonal


def get_confidence_response(
    model,
    tokenizer,
    question: Dict,
    layer_idx: Optional[int],
    steering_vector: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response, optionally with steering.

    Returns (response, confidence, option_probs, mapping) where mapping is only
    set for delegate task.
    """
    # Format prompt based on task type
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    if layer_idx is not None and steering_vector is not None and multiplier != 0.0:
        # Steering
        steering_tensor = torch.tensor(
            steering_vector,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        hook.register(layer_module)

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
        finally:
            hook.remove()
    else:
        # No steering
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


def get_confidence_with_ablation(
    model,
    tokenizer,
    question: Dict,
    layer_idx: int,
    ablation_direction: np.ndarray,
    use_chat_template: bool,
    trial_index: int = 0
) -> Tuple[str, float, np.ndarray, Optional[Dict[str, str]]]:
    """Get confidence response with ablation (direction zeroed out).

    Returns (response, confidence, option_probs, mapping) where mapping is only
    set for delegate task.
    """
    # Format prompt based on task type
    mapping = None
    if META_TASK == "delegate":
        prompt, options, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index)
    else:
        prompt = format_meta_prompt(question, tokenizer, use_chat_template)
        options = META_OPTIONS

    # Create ablation tensor
    ablation_tensor = torch.tensor(
        ablation_direction,
        dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    ).to(DEVICE)

    if hasattr(model, 'get_base_model'):
        layer_module = model.get_base_model().model.layers[layer_idx]
    else:
        layer_module = model.model.layers[layer_idx]

    hook = AblationHook(ablation_tensor)
    hook.register(layer_module)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)
    finally:
        hook.remove()

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    directions: Dict,
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool
) -> Dict:
    """Run steering experiment across layers and directions."""
    print(f"\nRunning steering experiment...")
    print(f"  Layers: {layers}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")

    results = {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": {},
    }

    # Compute entropy stats for alignment calculation
    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    # Compute baseline once (no steering) - shared across all layers
    print("Computing baseline (no steering)...")
    shared_baseline = []
    for q_idx, question in enumerate(tqdm(questions, desc="Baseline", leave=False)):
        response, confidence, probs, mapping = get_confidence_response(
            model, tokenizer, question, None, None, 0.0, use_chat_template, trial_index=q_idx
        )

        entropy = direct_entropies[q_idx]
        entropy_z = (entropy - entropy_mean) / entropy_std
        confidence_z = (confidence - 0.5) / 0.25
        alignment = -entropy_z * confidence_z

        shared_baseline.append({
            "question_idx": q_idx,
            "response": response,
            "confidence": confidence,
            "entropy": float(entropy),
            "alignment": float(alignment),
        })

    for layer_idx in tqdm(layers, desc="Steering layers"):
        introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
        control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)

        layer_results = {
            "baseline": shared_baseline,  # Reuse shared baseline
            "introspection": {m: [] for m in multipliers},
            "controls": {f"control_{i}": {m: [] for m in multipliers} for i in range(num_controls)},
        }

        # Introspection steering
        for mult in tqdm(multipliers, desc="Introspection", leave=False):
            if mult == 0.0:
                layer_results["introspection"][mult] = layer_results["baseline"]
                continue

            for q_idx, question in enumerate(questions):
                response, confidence, _, mapping = get_confidence_response(
                    model, tokenizer, question, layer_idx, introspection_dir, mult, use_chat_template, trial_index=q_idx
                )

                entropy = direct_entropies[q_idx]
                entropy_z = (entropy - entropy_mean) / entropy_std
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -entropy_z * confidence_z

                layer_results["introspection"][mult].append({
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "alignment": float(alignment),
                })

        # Control steering
        for ctrl_idx, ctrl_dir in enumerate(control_dirs):
            for mult in tqdm(multipliers, desc=f"Control {ctrl_idx}", leave=False):
                if mult == 0.0:
                    layer_results["controls"][f"control_{ctrl_idx}"][mult] = layer_results["baseline"]
                    continue

                for q_idx, question in enumerate(questions):
                    response, confidence, _, mapping = get_confidence_response(
                        model, tokenizer, question, layer_idx, ctrl_dir, mult, use_chat_template, trial_index=q_idx
                    )

                    entropy = direct_entropies[q_idx]
                    entropy_z = (entropy - entropy_mean) / entropy_std
                    confidence_z = (confidence - 0.5) / 0.25
                    alignment = -entropy_z * confidence_z

                    layer_results["controls"][f"control_{ctrl_idx}"][mult].append({
                        "question_idx": q_idx,
                        "response": response,
                        "confidence": confidence,
                        "alignment": float(alignment),
                    })

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


# ============================================================================
# ABLATION EXPERIMENT
# ============================================================================

def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    directions: Dict,
    num_controls: int,
    use_chat_template: bool,
    baseline_results: Optional[List[Dict]] = None
) -> Dict:
    """
    Run ablation experiment to test causality of introspection direction.

    For each layer, we:
    1. Collect baseline confidence-entropy correlation (no intervention)
    2. Ablate the introspection direction and measure correlation
    3. Ablate control (random orthogonal) directions and measure correlation

    If the introspection direction is causal, ablating it should degrade the
    correlation more than ablating random directions.

    Args:
        baseline_results: Optional pre-computed baseline results from steering experiment.
                          If provided, skips baseline computation for efficiency.
    """
    print(f"\nRunning ablation experiment...")
    print(f"  Layers: {layers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")
    if baseline_results is not None:
        print(f"  Reusing baseline from steering experiment")

    results = {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": {},
    }

    # Compute entropy stats for alignment calculation
    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    # Compute baseline once if not provided
    if baseline_results is None:
        print("Computing baseline (no intervention)...")
        baseline_results = []
        for q_idx, question in enumerate(tqdm(questions, desc="Baseline", leave=False)):
            response, confidence, probs, mapping = get_confidence_response(
                model, tokenizer, question, None, None, 0.0, use_chat_template, trial_index=q_idx
            )

            entropy = direct_entropies[q_idx]
            entropy_z = (entropy - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = -entropy_z * confidence_z

            baseline_results.append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })

    for layer_idx in tqdm(layers, desc="Ablation layers"):
        introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
        control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)

        layer_results = {
            "baseline": baseline_results,  # Reuse shared baseline
            "introspection_ablated": [],
            "controls_ablated": {f"control_{i}": [] for i in range(num_controls)},
        }

        # Introspection direction ablation
        for q_idx, question in enumerate(tqdm(questions, desc="Ablate introspection", leave=False)):
            response, confidence, _, mapping = get_confidence_with_ablation(
                model, tokenizer, question, layer_idx, introspection_dir, use_chat_template, trial_index=q_idx
            )

            entropy = direct_entropies[q_idx]
            entropy_z = (entropy - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = -entropy_z * confidence_z

            layer_results["introspection_ablated"].append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })

        # Control direction ablations
        for ctrl_idx, ctrl_dir in enumerate(control_dirs):
            for q_idx, question in enumerate(tqdm(questions, desc=f"Ablate control {ctrl_idx}", leave=False)):
                response, confidence, _, mapping = get_confidence_with_ablation(
                    model, tokenizer, question, layer_idx, ctrl_dir, use_chat_template, trial_index=q_idx
                )

                entropy = direct_entropies[q_idx]
                entropy_z = (entropy - entropy_mean) / entropy_std
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -entropy_z * confidence_z

                layer_results["controls_ablated"][f"control_{ctrl_idx}"].append({
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "entropy": float(entropy),
                    "alignment": float(alignment),
                })

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def compute_correlation(confidences: np.ndarray, entropies: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and entropy."""
    # We expect negative correlation: high entropy = low confidence
    if len(confidences) < 2 or np.std(confidences) == 0 or np.std(entropies) == 0:
        return 0.0
    return float(np.corrcoef(confidences, entropies)[0, 1])


def analyze_ablation_results(results: Dict) -> Dict:
    """Compute ablation effect statistics."""
    analysis = {
        "layers": results["layers"],
        "num_questions": results["num_questions"],
        "effects": {},
    }

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        # Extract data
        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_entropy = np.array([r["entropy"] for r in lr["baseline"]])
        baseline_align = np.array([r["alignment"] for r in lr["baseline"]])

        ablated_conf = np.array([r["confidence"] for r in lr["introspection_ablated"]])
        ablated_entropy = np.array([r["entropy"] for r in lr["introspection_ablated"]])
        ablated_align = np.array([r["alignment"] for r in lr["introspection_ablated"]])

        # Compute correlations
        baseline_corr = compute_correlation(baseline_conf, baseline_entropy)
        ablated_corr = compute_correlation(ablated_conf, ablated_entropy)

        # Control ablations
        control_corrs = []
        control_aligns = []
        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_entropy = np.array([r["entropy"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_align = np.array([r["alignment"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_entropy))
            control_aligns.append(ctrl_align.mean())

        avg_control_corr = np.mean(control_corrs)
        avg_control_align = np.mean(control_aligns)

        # Compute statistical significance
        # For a well-calibrated model, baseline correlation should be negative
        # Ablation should make correlation less negative (closer to 0), so correlation_change > 0
        # We want to test if introspection ablation has LARGER positive change than controls
        intro_corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        # P-value: fraction of controls with >= correlation change (degradation)
        n_controls_worse = sum(1 for c in control_corr_changes if c >= intro_corr_change)
        p_value = (n_controls_worse + 1) / (len(control_corrs) + 1)  # +1 for conservative estimate

        analysis["effects"][layer_idx] = {
            "baseline": {
                "correlation": baseline_corr,
                "mean_alignment": float(baseline_align.mean()),
                "mean_confidence": float(baseline_conf.mean()),
            },
            "introspection_ablated": {
                "correlation": ablated_corr,
                "correlation_change": intro_corr_change,
                "mean_alignment": float(ablated_align.mean()),
                "alignment_change": float(ablated_align.mean() - baseline_align.mean()),
                "mean_confidence": float(ablated_conf.mean()),
                "p_value_vs_controls": p_value,
            },
            "control_ablated_avg": {
                "correlation": avg_control_corr,
                "correlation_change": avg_control_corr - baseline_corr,
                "mean_alignment": avg_control_align,
                "alignment_change": avg_control_align - float(baseline_align.mean()),
            },
            "individual_controls": {
                f"control_{i}": {
                    "correlation": control_corrs[i],
                    "correlation_change": control_corrs[i] - baseline_corr,
                }
                for i in range(len(control_corrs))
            },
        }

    return analysis


def analyze_results(results: Dict) -> Dict:
    """Compute summary statistics."""
    analysis = {
        "layers": results["layers"],
        "multipliers": results["multipliers"],
        "effects": {},
    }

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]
        multipliers = results["multipliers"]

        baseline_align = np.mean([r["alignment"] for r in lr["baseline"]])
        baseline_conf = np.mean([r["confidence"] for r in lr["baseline"]])

        effects = {"introspection": {}, "control_avg": {}}

        for mult in multipliers:
            # Introspection
            intro_align = np.mean([r["alignment"] for r in lr["introspection"][mult]])
            intro_conf = np.mean([r["confidence"] for r in lr["introspection"][mult]])
            effects["introspection"][mult] = {
                "alignment": float(intro_align),
                "alignment_change": float(intro_align - baseline_align),
                "confidence": float(intro_conf),
                "confidence_change": float(intro_conf - baseline_conf),
            }

            # Control average
            ctrl_aligns = []
            ctrl_confs = []
            for ctrl_key in lr["controls"]:
                ctrl_aligns.extend([r["alignment"] for r in lr["controls"][ctrl_key][mult]])
                ctrl_confs.extend([r["confidence"] for r in lr["controls"][ctrl_key][mult]])
            effects["control_avg"][mult] = {
                "alignment": float(np.mean(ctrl_aligns)),
                "alignment_change": float(np.mean(ctrl_aligns) - baseline_align),
                "confidence": float(np.mean(ctrl_confs)),
                "confidence_change": float(np.mean(ctrl_confs) - baseline_conf),
            }

        # Compute slopes
        intro_slope = np.polyfit(multipliers, [effects["introspection"][m]["alignment_change"] for m in multipliers], 1)[0]
        ctrl_slope = np.polyfit(multipliers, [effects["control_avg"][m]["alignment_change"] for m in multipliers], 1)[0]

        analysis["effects"][layer_idx] = {
            "by_multiplier": effects,
            "slopes": {
                "introspection": float(intro_slope),
                "control_avg": float(ctrl_slope),
            },
            "baseline_alignment": float(baseline_align),
            "baseline_confidence": float(baseline_conf),
        }

    return analysis


def plot_results(analysis: Dict, output_prefix: str):
    """Create visualizations."""
    layers = analysis["layers"]
    multipliers = analysis["multipliers"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Slopes by layer
    ax1 = axes[0]
    intro_slopes = [analysis["effects"][l]["slopes"]["introspection"] for l in layers]
    ctrl_slopes = [analysis["effects"][l]["slopes"]["control_avg"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width/2, intro_slopes, width, label='Introspection', color='green', alpha=0.7)
    ax1.bar(x + width/2, ctrl_slopes, width, label='Control (avg)', color='gray', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Alignment Slope (Δalign / Δmult)")
    ax1.set_title("Steering Effect on Alignment")
    ax1.legend()

    # Plot 2: Best layer detail
    best_layer = max(layers, key=lambda l: analysis["effects"][l]["slopes"]["introspection"])
    ax2 = axes[1]

    intro_align = [analysis["effects"][best_layer]["by_multiplier"]["introspection"][m]["alignment_change"] for m in multipliers]
    ctrl_align = [analysis["effects"][best_layer]["by_multiplier"]["control_avg"][m]["alignment_change"] for m in multipliers]

    ax2.plot(multipliers, intro_align, 'o-', label='Introspection', linewidth=2, color='green')
    ax2.plot(multipliers, ctrl_align, '^--', label='Control', linewidth=2, color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Steering Multiplier")
    ax2.set_ylabel("Δ Alignment")
    ax2.set_title(f"Alignment Change (Layer {best_layer})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary
    ax3 = axes[2]
    ax3.axis('off')

    intro_slope = analysis["effects"][best_layer]["slopes"]["introspection"]
    ctrl_slope = analysis["effects"][best_layer]["slopes"]["control_avg"]

    summary = f"""
STEERING EXPERIMENT SUMMARY

Best Layer: {best_layer}
  Introspection slope: {intro_slope:.4f}
  Control slope: {ctrl_slope:.4f}
  Difference: {intro_slope - ctrl_slope:.4f}

Interpretation:
"""
    if intro_slope > 0 and intro_slope > ctrl_slope + 0.01:
        summary += """  ✓ Positive introspection steering effect
  Steering increases alignment!
  Effect stronger than controls."""
    elif intro_slope > 0:
        summary += """  ⚠ Weak introspection steering effect
  Steering increases alignment
  but not clearly above controls."""
    else:
        summary += """  ✗ No introspection steering effect
  Steering does not improve alignment."""

    ax3.text(0.1, 0.9, summary, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_steering_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_steering_results.png")
    plt.close()


def print_summary(analysis: Dict):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT RESULTS")
    print("=" * 70)

    print("\n--- Alignment Slopes by Layer ---")
    print(f"{'Layer':<8} {'Introspection':<15} {'Control':<15}")
    print("-" * 40)

    for layer in analysis["layers"]:
        s = analysis["effects"][layer]["slopes"]
        print(f"{layer:<8} {s['introspection']:<15.4f} {s['control_avg']:<15.4f}")

    # Best layer
    best_layer = max(analysis["layers"], key=lambda l: analysis["effects"][l]["slopes"]["introspection"])
    best_intro = analysis["effects"][best_layer]["slopes"]["introspection"]
    best_ctrl = analysis["effects"][best_layer]["slopes"]["control_avg"]

    print(f"\nBest introspection steering: Layer {best_layer}")
    print(f"  Introspection slope: {best_intro:.4f}")
    print(f"  Control slope: {best_ctrl:.4f}")

    if best_intro > 0 and best_intro > best_ctrl + 0.01:
        print("\n✓ Evidence for causal introspection effect!")
    elif best_intro > 0:
        print("\n⚠ Weak effect, not clearly separable from controls")
    else:
        print("\n✗ No introspection steering effect found")


def plot_ablation_results(analysis: Dict, output_prefix: str):
    """Create ablation visualizations."""
    layers = analysis["layers"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Correlation change by layer (bar chart)
    ax1 = axes[0]

    intro_corr_change = [analysis["effects"][l]["introspection_ablated"]["correlation_change"] for l in layers]
    ctrl_corr_change = [analysis["effects"][l]["control_ablated_avg"]["correlation_change"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width/2, intro_corr_change, width, label='Introspection Ablated', color='red', alpha=0.7)
    ax1.bar(x + width/2, ctrl_corr_change, width, label='Control Ablated (avg)', color='gray', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Δ Correlation (conf vs entropy)")
    ax1.set_title("Ablation Effect on Confidence-Entropy Correlation")
    ax1.legend()

    # Plot 2: Alignment change by layer
    ax2 = axes[1]

    intro_align_change = [analysis["effects"][l]["introspection_ablated"]["alignment_change"] for l in layers]
    ctrl_align_change = [analysis["effects"][l]["control_ablated_avg"]["alignment_change"] for l in layers]

    ax2.bar(x - width/2, intro_align_change, width, label='Introspection Ablated', color='red', alpha=0.7)
    ax2.bar(x + width/2, ctrl_align_change, width, label='Control Ablated (avg)', color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Δ Alignment")
    ax2.set_title("Ablation Effect on Introspection Alignment")
    ax2.legend()

    # Plot 3: Summary with p-values
    ax3 = axes[2]
    ax3.axis('off')

    # Find best layer by p-value
    best_layer = min(
        layers,
        key=lambda l: analysis["effects"][l]["introspection_ablated"].get("p_value_vs_controls", 1.0)
    )
    best_pval = analysis["effects"][best_layer]["introspection_ablated"].get("p_value_vs_controls", 1.0)
    best_intro = analysis["effects"][best_layer]["introspection_ablated"]["correlation_change"]
    best_ctrl = analysis["effects"][best_layer]["control_ablated_avg"]["correlation_change"]
    baseline_corr = analysis["effects"][best_layer]["baseline"]["correlation"]

    # Count significant layers
    sig_layers = [l for l in layers
                  if analysis["effects"][l]["introspection_ablated"].get("p_value_vs_controls", 1.0) < 0.05]

    summary = f"""
ABLATION EXPERIMENT SUMMARY

Best Layer by p-value: {best_layer}
  Baseline correlation: {baseline_corr:.4f}
  Ablation Δcorr: {best_intro:.4f}
  Control Δcorr: {best_ctrl:.4f}
  p-value: {best_pval:.4f}

Significant layers (p<0.05): {len(sig_layers)}
  {sig_layers if sig_layers else 'None'}

Interpretation:
"""
    if best_pval < 0.05:
        summary += """  ✓ STATISTICALLY SIGNIFICANT
  Ablating the direction degrades
  calibration more than random
  directions (p < 0.05)!"""
    elif best_pval < 0.10:
        summary += """  ⚠ Marginally significant
  Effect in expected direction
  (p < 0.10) but needs more
  statistical power."""
    else:
        summary += """  ✗ Not statistically significant
  Cannot distinguish effect from
  random control directions."""

    ax3.text(0.1, 0.9, summary, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ablation_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_ablation_results.png")
    plt.close()


def print_ablation_summary(analysis: Dict):
    """Print summary of ablation results."""
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 70)

    print("\n--- Correlation Change by Layer (conf vs entropy) ---")
    print(f"{'Layer':<8} {'Baseline':<12} {'Ablated':<12} {'Ctrl Avg':<12} {'p-value':<10}")
    print("-" * 60)

    for layer in analysis["layers"]:
        e = analysis["effects"][layer]
        p_val = e['introspection_ablated'].get('p_value_vs_controls', float('nan'))
        print(f"{layer:<8} {e['baseline']['correlation']:<12.4f} "
              f"{e['introspection_ablated']['correlation']:<12.4f} "
              f"{e['control_ablated_avg']['correlation']:<12.4f} "
              f"{p_val:<10.4f}")

    print("\n--- Alignment Change by Layer ---")
    print(f"{'Layer':<8} {'Baseline':<12} {'Intro Δ':<12} {'Ctrl Δ':<12}")
    print("-" * 45)

    for layer in analysis["layers"]:
        e = analysis["effects"][layer]
        print(f"{layer:<8} {e['baseline']['mean_alignment']:<12.4f} "
              f"{e['introspection_ablated']['alignment_change']:<12.4f} "
              f"{e['control_ablated_avg']['alignment_change']:<12.4f}")

    # Find most affected layer
    most_affected = min(analysis["layers"],
                        key=lambda l: analysis["effects"][l]["introspection_ablated"]["correlation_change"])
    ma = analysis["effects"][most_affected]

    # Find best layer by p-value
    best_layer_by_pval = min(
        analysis["layers"],
        key=lambda l: analysis["effects"][l]["introspection_ablated"].get("p_value_vs_controls", 1.0)
    )
    best_pval = analysis["effects"][best_layer_by_pval]["introspection_ablated"].get("p_value_vs_controls", 1.0)

    print(f"\nMost affected by introspection ablation: Layer {most_affected}")
    print(f"  Baseline correlation: {ma['baseline']['correlation']:.4f}")
    print(f"  After introspection ablation: {ma['introspection_ablated']['correlation']:.4f}")
    print(f"  After control ablation (avg): {ma['control_ablated_avg']['correlation']:.4f}")
    print(f"  p-value vs controls: {ma['introspection_ablated'].get('p_value_vs_controls', float('nan')):.4f}")

    print(f"\nBest layer by p-value: Layer {best_layer_by_pval} (p={best_pval:.4f})")

    # Count significant layers
    sig_layers = [l for l in analysis["layers"]
                  if analysis["effects"][l]["introspection_ablated"].get("p_value_vs_controls", 1.0) < 0.05]

    intro_change = ma["introspection_ablated"]["correlation_change"]
    ctrl_change = ma["control_ablated_avg"]["correlation_change"]

    # For a well-calibrated model, correlation should be negative
    # Ablating causes correlation to become less negative (closer to 0)
    # So correlation_change > 0 means ablation hurt calibration
    if best_pval < 0.05:
        print(f"\n✓ STATISTICALLY SIGNIFICANT causal effect! (p < 0.05)")
        print(f"  {len(sig_layers)} layer(s) with p < 0.05: {sig_layers}")
        print("  Ablating the direction degrades calibration")
        print("  more than ablating random directions.")
    elif intro_change > ctrl_change + 0.02:
        print("\n⚠ Effect in expected direction but not statistically significant")
        print("  Consider running with more control directions for better power.")
    else:
        print("\n✗ No clear causal effect from ablation")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Device: {DEVICE}")
    print(f"Direction type: {DIRECTION_TYPE}")
    print(f"Meta-judgment task: {META_TASK}")

    # Generate output prefix
    output_prefix = get_output_prefix()
    print(f"Output prefix: {output_prefix}")

    # Compute input/output paths based on direction type
    paired_data_path = f"{output_prefix}_paired_data.json"

    if DIRECTION_TYPE == "shared":
        # Shared MC entropy direction from analyze_shared_unique.py
        # Use same prefix logic as analyze_shared_unique.py (includes adapter if set)
        model_short = get_model_short_name(BASE_MODEL_NAME)
        if MODEL_NAME != BASE_MODEL_NAME:
            adapter_short = get_model_short_name(MODEL_NAME)
            shared_prefix = OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}"
        else:
            shared_prefix = OUTPUTS_DIR / f"{model_short}"
        shared_directions_path = Path(f"{shared_prefix}_shared_unique_directions.npz")
        shared_transfer_path = Path(f"{shared_prefix}_{DATASET_NAME}_shared_unique_transfer.json")
        directions_path = str(shared_directions_path)
        transfer_results_path = str(shared_transfer_path)
        direction_key_template = "layer_{}_shared"
        probe_results_path = None  # Not used for shared directions
    elif DIRECTION_TYPE == "entropy":
        # Entropy directions from run_introspection_experiment.py
        probe_results_path = f"{output_prefix}_probe_results.json"
        directions_path = f"{output_prefix}_entropy_directions.npz"
        direction_key_template = "layer_{}_entropy"
        transfer_results_path = None
    else:
        # Introspection directions from run_introspection_probe.py
        probe_results_path = f"{output_prefix}_probe_results.json"
        directions_path = f"{output_prefix}_probe_directions.npz"
        direction_key_template = "layer_{}_introspection"
        transfer_results_path = None

    # Load probe results or transfer results depending on direction type
    if DIRECTION_TYPE == "shared":
        print(f"\nLoading transfer results from {transfer_results_path}...")
        with open(transfer_results_path, "r") as f:
            transfer_results = json.load(f)
        probe_results = None
    else:
        print(f"\nLoading probe results from {probe_results_path}...")
        with open(probe_results_path, "r") as f:
            probe_results = json.load(f)
        transfer_results = None

    # Load directions
    print(f"Loading directions from {directions_path}...")
    directions_data = np.load(directions_path)
    # Remap keys to consistent format for the rest of the script
    directions = {}
    for k in directions_data.files:
        # Extract layer number and remap to "layer_{idx}_introspection" format
        # (the steering functions expect this format)
        parts = k.split("_")
        layer_idx = parts[1]
        directions[f"layer_{layer_idx}_introspection"] = directions_data[k]

    # Determine layers to steer
    if STEERING_LAYERS is not None:
        layers = STEERING_LAYERS
    else:
        if DIRECTION_TYPE == "shared":
            # For shared directions, select layers where meta R² exceeds threshold
            layer_candidates = []
            layer_list = transfer_results["layers"]
            meta_r2_list = transfer_results["shared"]["meta_r2"]
            for layer_idx, meta_r2 in zip(layer_list, meta_r2_list):
                if meta_r2 >= META_R2_THRESHOLD:
                    layer_candidates.append((layer_idx, meta_r2))
            # Sort by meta R² descending
            layer_candidates.sort(key=lambda x: -x[1])
            layers = [l[0] for l in layer_candidates]
            if not layers:
                print(f"  Warning: No layers with meta R² >= {META_R2_THRESHOLD}")
                print(f"  Using top 5 layers by meta R² instead")
                all_layers = [(l, r) for l, r in zip(layer_list, meta_r2_list)]
                all_layers.sort(key=lambda x: -x[1])
                layers = [l[0] for l in all_layers[:5]]
            layers = sorted(layers)
            print(f"  Meta R² threshold: {META_R2_THRESHOLD}")
            print(f"  Layers above threshold: {len(layers)}")
        elif DIRECTION_TYPE == "entropy":
            # For entropy directions, select layers based on direct→meta transfer R²
            # Collect layers with their R² values for ranking
            layer_candidates = []
            if "probe_results" in probe_results:
                # Find layers with good direct→meta transfer
                for layer_str, lr in probe_results["probe_results"].items():
                    d2m_r2 = lr.get("direct_to_meta_fixed", {}).get("r2", 0)
                    d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
                    # Include layer if it has meaningful transfer (threshold of 0.1)
                    if d2m_r2 > 0.1 and d2d_r2 > 0.05:
                        layer_candidates.append((int(layer_str), d2m_r2))
            # Sort by direct→meta R² descending to prioritize best transfer
            layer_candidates.sort(key=lambda x: -x[1])
            layers = [l[0] for l in layer_candidates]
            # If no good layers found, use layers with best direct→direct
            if not layers:
                all_layers = []
                for layer_str, lr in probe_results.get("probe_results", {}).items():
                    d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
                    all_layers.append((int(layer_str), d2d_r2))
                all_layers.sort(key=lambda x: -x[1])  # Sort by R² descending
                layers = [l[0] for l in all_layers[:5]]  # Top 5
            layers = sorted(layers)
        else:
            # Use significant layers from introspection probe
            layers = set()
            for layer_str, lr in probe_results.get("layer_results", {}).items():
                if lr.get("significant_p05", False):
                    layers.add(int(layer_str))
            if "best_layer" in probe_results:
                layers.add(probe_results["best_layer"]["layer"])
            if not layers:
                all_layers = [int(l) for l in probe_results.get("layer_results", {}).keys()]
                mid = len(all_layers) // 2
                layers = all_layers[max(0, mid-3):mid+4]
            layers = sorted(layers)

    print(f"Steering layers: {layers}")

    # Load paired data
    print(f"\nLoading paired data from {paired_data_path}...")
    with open(paired_data_path, "r") as f:
        paired_data = json.load(f)

    questions = paired_data["questions"][:NUM_STEERING_QUESTIONS]
    direct_entropies = np.array(paired_data["direct_entropies"])[:NUM_STEERING_QUESTIONS]
    print(f"Using {len(questions)} questions")

    # Load model
    print(f"\nLoading model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Left-pad for proper batched generation

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        token=HF_TOKEN
    )

    if MODEL_NAME != BASE_MODEL_NAME:
        from peft import PeftModel
        print(f"Loading adapter: {MODEL_NAME}")
        model = PeftModel.from_pretrained(model, MODEL_NAME)

    # Determine chat template usage
    use_chat_template = has_chat_template(tokenizer) and not is_base_model(BASE_MODEL_NAME)
    print(f"Using chat template: {use_chat_template}")

    # Run experiment
    results = run_steering_experiment(
        model, tokenizer, questions, direct_entropies,
        layers, directions, STEERING_MULTIPLIERS, NUM_CONTROL_DIRECTIONS,
        use_chat_template
    )

    # Analyze
    analysis = analyze_results(results)

    # Add direction_type suffix to output files to distinguish them
    direction_suffix = f"_{DIRECTION_TYPE}" if DIRECTION_TYPE != "introspection" else ""

    # Save results
    output_results = f"{output_prefix}_steering{direction_suffix}_results.json"
    with open(output_results, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved {output_results}")

    output_analysis = f"{output_prefix}_steering{direction_suffix}_analysis.json"
    with open(output_analysis, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved {output_analysis}")

    # Print and plot steering results
    print_summary(analysis)
    plot_results(analysis, f"{output_prefix}{direction_suffix}")

    print("\n✓ Steering experiment complete!")

    # ==========================================================================
    # ABLATION EXPERIMENT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENT")
    print("=" * 70)

    # Extract baseline from steering results (first layer's baseline, they're all the same)
    first_layer = layers[0]
    baseline_from_steering = results["layer_results"][first_layer]["baseline"]

    ablation_results = run_ablation_experiment(
        model, tokenizer, questions, direct_entropies,
        layers, directions, NUM_CONTROL_DIRECTIONS,
        use_chat_template,
        baseline_results=baseline_from_steering
    )

    # Analyze ablation results
    ablation_analysis = analyze_ablation_results(ablation_results)

    # Save ablation results
    ablation_results_path = f"{output_prefix}_ablation{direction_suffix}_results.json"
    with open(ablation_results_path, "w") as f:
        json.dump(ablation_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved {ablation_results_path}")

    ablation_analysis_path = f"{output_prefix}_ablation{direction_suffix}_analysis.json"
    with open(ablation_analysis_path, "w") as f:
        json.dump(ablation_analysis, f, indent=2)
    print(f"Saved {ablation_analysis_path}")

    # Print and plot ablation results
    print_ablation_summary(ablation_analysis)
    plot_ablation_results(ablation_analysis, f"{output_prefix}{direction_suffix}")

    print("\n✓ Ablation experiment complete!")
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
