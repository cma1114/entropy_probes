"""
Steering and ablation experiments using probe directions.

This script supports three types of directions:
1. "introspection" - From run_introspection_probe.py (meta activations → introspection_score)
                     NOTE: Only ablation is run for this direction type (steering skipped).
                     The introspection direction captures calibration quality, not a direct
                     uncertainty signal, so steering doesn't make conceptual sense.
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
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import random

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from tasks import (
    # Confidence task
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    # Delegate task
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    # Unified conversion
    response_to_confidence,
)

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
#              (direct activations → selected metric, for direct→meta transfer test)
# - "shared": Uses shared MC entropy direction from analyze_shared_unique.py
#             (average of MC directions across datasets - tests general uncertainty)
DIRECTION_TYPE = "shared"  # "introspection", "entropy", or "shared"

# Metric to use when DIRECTION_TYPE == "entropy"
# Prob-based (nonlinear): entropy, top_prob, margin
# Logit-based (linear, better for probes): logit_gap, top_logit
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
METRIC = "logit_gap"  # Which metric's directions to use (set via --metric flag)

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
NUM_CONTROL_DIRECTIONS = 20
BATCH_SIZE = 8  # Batch size for baseline/single-direction forward passes

# Expanded batch target for multi-multiplier sweeps.
# When sweeping k multipliers simultaneously, we expand each base batch by k.
# This sets the TARGET total expanded batch size (base_batch * k_mult).
# Higher values = better GPU utilization but more memory.
# With k_mult=6 and EXPANDED_BATCH_TARGET=48, base batch = 8, expanded = 48.
EXPANDED_BATCH_TARGET = 48

# Quantization (for large models like 70B)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Aliases for backward compatibility with local code
META_OPTIONS = list(STATED_CONFIDENCE_OPTIONS.keys())
DELEGATE_OPTIONS = ANSWER_OR_DELEGATE_OPTIONS

# Cached token IDs - populated once at startup to avoid repeated tokenization
_CACHED_TOKEN_IDS = {
    "meta_options": None,      # List of token IDs for S, T, U, V, W, X, Y, Z
    "delegate_options": None,  # List of token IDs for "1", "2"
}


def initialize_token_cache(tokenizer):
    """Precompute option token IDs once to avoid repeated tokenization."""
    _CACHED_TOKEN_IDS["meta_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS
    ]
    _CACHED_TOKEN_IDS["delegate_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in DELEGATE_OPTIONS
    ]
    print(f"  Cached token IDs: meta={_CACHED_TOKEN_IDS['meta_options']}, delegate={_CACHED_TOKEN_IDS['delegate_options']}")


# ============================================================================
# PROMPT FORMATTING WRAPPERS
# ============================================================================

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


def get_meta_options() -> List[str]:
    """Return the meta options based on META_TASK setting."""
    if META_TASK == "delegate":
        return DELEGATE_OPTIONS
    else:
        return META_OPTIONS


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
# STEERING AND ABLATION
# ============================================================================

class SteeringHook:
    """Hook that adds a steering vector to activations."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float, pre_normalized: bool = False):
        # Ensure normalized so multiplier has consistent meaning across directions
        if pre_normalized:
            self.steering_vector = steering_vector
        else:
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

    def set_multiplier(self, multiplier: float):
        """Update multiplier without recreating the hook."""
        self.multiplier = multiplier

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()



class BatchSteeringHook:
    """Hook that adds a *per-example* steering delta to activations.

    This is designed for "multiplier sweep in one pass" by expanding the batch:
    each prompt is duplicated for each multiplier, and this hook adds a different
    delta vector for each expanded example.
    """

    def __init__(self, delta_bh: Optional[torch.Tensor] = None):
        self.delta_bh = delta_bh  # (batch, hidden)
        self.handle = None

    def set_delta(self, delta_bh: torch.Tensor):
        self.delta_bh = delta_bh

    def __call__(self, module, input, output):
        if self.delta_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output

        # hs: (batch, seq, hidden); delta_bh: (batch, hidden)
        # Broadcast delta across sequence length.
        # Must cast both device and dtype for compatibility with device_map="auto"
        delta = self.delta_bh[:, None, :].to(device=hs.device, dtype=hs.dtype)
        hs = hs + delta

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None



class AblationHook:
    """
    Hook that removes the component of activations along a direction.

    Projects out the direction: x' = x - (x · d) * d
    This tests whether the direction is causally involved in the behavior.
    """

    def __init__(self, direction: torch.Tensor, pre_normalized: bool = False):
        # Ensure normalized
        if pre_normalized:
            self.direction = direction
        else:
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


def pretokenize_prompts(
    prompts: List[str],
    tokenizer,
    device: str
) -> Dict:
    """
    Pre-tokenize all prompts once (BPE encoding).

    Returns dict with:
        - input_ids: List of token ID lists (variable length, no padding yet)
        - attention_mask: List of attention mask lists
        - lengths: List of sequence lengths
        - sorted_order: Indices sorted by length (for efficient batching)

    Padding is deferred to batch time to avoid padding short prompts to global max.
    """
    # Tokenize without padding - just BPE encode once
    tokenized = tokenizer(
        prompts,
        padding=False,
        truncation=True,
        return_attention_mask=True
    )

    lengths = [len(ids) for ids in tokenized["input_ids"]]
    # Sort indices by length for efficient batching (similar lengths together)
    sorted_order = sorted(range(len(prompts)), key=lambda i: lengths[i])

    return {
        "input_ids": tokenized["input_ids"],  # List of lists
        "attention_mask": tokenized["attention_mask"],  # List of lists
        "lengths": lengths,
        "sorted_order": sorted_order,
        "device": device,
        "tokenizer": tokenizer,  # Keep reference for padding
    }


def build_padded_gpu_batches(
    cached_inputs: Dict,
    tokenizer,
    device: str,
    batch_size: int,
) -> List[Tuple[List[int], Dict[str, torch.Tensor]]]:
    """Pad each length-sorted batch once and keep tensors on-device.

    This eliminates repeated tokenizer.pad() and CPU→GPU copies for every
    (layer × multiplier × control) forward pass.
    """
    sorted_order = cached_inputs["sorted_order"]
    batches: List[Tuple[List[int], Dict[str, torch.Tensor]]] = []

    for batch_start in range(0, len(sorted_order), batch_size):
        batch_indices = sorted_order[batch_start:batch_start + batch_size]
        batch_input_ids = [cached_inputs["input_ids"][i] for i in batch_indices]
        batch_attention = [cached_inputs["attention_mask"][i] for i in batch_indices]

        batch_inputs = tokenizer.pad(
            {"input_ids": batch_input_ids, "attention_mask": batch_attention},
            return_tensors="pt",
            padding=True,
        )
        # Keep on-device for reuse across many sweeps.
        batch_inputs = {k: v.to(device, non_blocking=True) for k, v in batch_inputs.items()}
        batches.append((batch_indices, batch_inputs))

    return batches


def _get_transformer_and_lm_head(model):
    """Best-effort access to (transformer, lm_head) for fast option-only logits."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    transformer = getattr(base, "model", None)
    lm_head = getattr(base, "lm_head", None)
    if transformer is None or lm_head is None or not hasattr(lm_head, "weight"):
        return None, None
    return transformer, lm_head


def _prepare_option_weight(lm_head, model, option_token_ids: List[int]) -> Optional[torch.Tensor]:
    """Extract lm_head rows for the option token IDs: (n_opt, hidden_dim).

    For models with device_map="auto", the lm_head weight may be on meta device.
    In that case, we do a dummy forward pass to materialize the weight.
    """
    if lm_head is None or not hasattr(lm_head, "weight"):
        return None
    W = lm_head.weight
    if W is None or W.ndim != 2:
        return None

    # Check if weight is on meta device (happens with device_map="auto" for large models)
    if W.device.type == "meta":
        # The lm_head hasn't been materialized yet. We need to run a forward pass
        # to trigger the weight loading. This is a one-time cost.
        print("  Note: lm_head on meta device, triggering materialization...")
        try:
            # Create minimal dummy input to trigger weight loading
            dummy_input = torch.zeros(1, 1, dtype=torch.long, device="cuda")
            with torch.no_grad():
                _ = model(dummy_input, use_cache=False)
            # Now check again
            W = lm_head.weight
            if W.device.type == "meta":
                print("  Warning: lm_head still on meta after forward pass, using slow path")
                return None
        except Exception as e:
            print(f"  Warning: Failed to materialize lm_head: {e}, using slow path")
            return None

    option_ids = torch.tensor(option_token_ids, dtype=torch.long, device=W.device)
    return W.index_select(0, option_ids)


def _compute_batch_option_logits(
    model,
    transformer,
    W_opt: Optional[torch.Tensor],
    option_token_ids: List[int],
    batch_inputs: Dict[str, torch.Tensor],
) -> torch.Tensor:
    """Return (batch, n_opt) logits for the next token.

    Fast path: transformer forward → last_hidden_state[:, -1] → matmul with W_opt.
    Fallback: model forward → full logits → index option_token_ids.
    """
    if transformer is None or W_opt is None:
        outputs = model(**batch_inputs, use_cache=False)
        batch_logits = outputs.logits[:, -1, :]
        # Ensure we're on a real device (not meta) before indexing
        if batch_logits.device.type == "meta":
            raise RuntimeError("Model logits are on meta device - model may not be fully loaded")
        return batch_logits[:, option_token_ids]

    out = transformer(**batch_inputs, use_cache=False, return_dict=True)
    last_h = out.last_hidden_state[:, -1, :]
    # With device_map="auto", lm_head may live on a different device.
    if last_h.device != W_opt.device:
        last_h = last_h.to(W_opt.device)
    return last_h @ W_opt.T


def precompute_direction_tensors(
    directions: Dict,
    layers: List[int],
    num_controls: int,
    device: str,
    dtype: torch.dtype
) -> Dict:
    """
    Precompute normalized direction tensors on GPU for all layers and controls.

    Returns dict with structure:
    {
        layer_idx: {
            "introspection": tensor,  # normalized, on GPU
            "controls": [tensor, ...]  # normalized, on GPU
        }
    }
    """
    cached = {}
    for layer_idx in layers:
        introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
        # Normalize in numpy, then convert to tensor
        introspection_dir = introspection_dir / np.linalg.norm(introspection_dir)
        introspection_tensor = torch.tensor(introspection_dir, dtype=dtype, device=device)

        # Generate and cache control directions
        control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)
        control_tensors = [
            torch.tensor(cd, dtype=dtype, device=device)
            for cd in control_dirs
        ]

        cached[layer_idx] = {
            "introspection": introspection_tensor,
            "controls": control_tensors,
        }

    return cached


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

        # Prepare fast option-only projection
        if META_TASK == "delegate":
            option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        else:
            option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        transformer, lm_head = _get_transformer_and_lm_head(model)
        W_opt = _prepare_option_weight(lm_head, model, option_token_ids)

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            with torch.inference_mode():
                option_logits = _compute_batch_option_logits(
                    model, transformer, W_opt, option_token_ids, inputs
                )[0]
        finally:
            hook.remove()
    else:
        # No steering
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        # Prepare fast option-only projection
        if META_TASK == "delegate":
            option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        else:
            option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        transformer, lm_head = _get_transformer_and_lm_head(model)
        W_opt = _prepare_option_weight(lm_head, model, option_token_ids)
        with torch.inference_mode():
            option_logits = _compute_batch_option_logits(
                model, transformer, W_opt, option_token_ids, inputs
            )[0]

    option_probs = torch.softmax(option_logits, dim=-1).float().cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = local_response_to_confidence(response, option_probs, mapping)

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

    # Prepare fast option-only projection
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
    transformer, lm_head = _get_transformer_and_lm_head(model)
    W_opt = _prepare_option_weight(lm_head, model, option_token_ids)

    try:
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.inference_mode():
            option_logits = _compute_batch_option_logits(
                model, transformer, W_opt, option_token_ids, inputs
            )[0]
    finally:
        hook.remove()

    option_probs = torch.softmax(option_logits, dim=-1).float().cpu().numpy()

    response = options[np.argmax(option_probs)]
    confidence = local_response_to_confidence(response, option_probs, mapping)

    return response, confidence, option_probs, mapping


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    directions: Dict,
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """
    Run steering experiment across layers and directions.

    Args:
        direct_metric_values: The selected uncertainty metric values from direct MC task
                              (e.g., logit_gap, entropy, etc. as specified by METRIC)

    Optimized version:
    - Uses precomputed direction tensors if provided
    - Registers hook once per (layer, direction) and runs all questions
    - Uses cached token IDs
    """
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

    # Compute metric stats for alignment calculation
    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # Pre-format all prompts (avoid repeated work)
    print("Pre-formatting prompts...")
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index=q_idx)
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    # Pre-tokenize all prompts once (BPE only, no padding yet)
    print("Pre-tokenizing prompts...")
    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    print(f"  Tokenized {len(prompts)} prompts, lengths: min={min(cached_inputs['lengths'])}, max={max(cached_inputs['lengths'])}")

    # Get option token IDs
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = DELEGATE_OPTIONS
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = META_OPTIONS

    # Fast path: avoid computing full vocab logits.
    # We do a base-transformer forward, take last hidden state, and project only
    # onto the option token rows of lm_head.
    transformer, lm_head = _get_transformer_and_lm_head(model)
    W_opt = _prepare_option_weight(lm_head, model, option_token_ids)

    # Pad batches once and keep them on-device (reused across all sweeps).
    print("Building padded on-device batches...")
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)
    print(f"  Prepared {len(gpu_batches)} batches on {DEVICE}")

    def run_all_questions():
        """Run all questions in batches and return list of results.

        Uses length-sorted batching: process prompts in order of length so each batch
        has similar-length prompts, minimizing padding waste.
        """
        results_list = [None] * len(prompts)  # Pre-allocate to maintain order

        for batch_indices, batch_inputs in gpu_batches:
            with torch.inference_mode():
                batch_option_logits = _compute_batch_option_logits(
                    model, transformer, W_opt, option_token_ids, batch_inputs
                )
                batch_option_probs = torch.softmax(batch_option_logits, dim=-1).float().cpu().numpy()

            # Process each item in the batch (map back to original indices)
            for i, q_idx in enumerate(batch_indices):
                option_probs = batch_option_probs[i]
                response = options[np.argmax(option_probs)]
                confidence = local_response_to_confidence(response, option_probs, mappings[q_idx])

                metric_val = direct_metric_values[q_idx]
                metric_z = (metric_val - metric_mean) / metric_std
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -metric_z * confidence_z

                results_list[q_idx] = {
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "metric": float(metric_val),  # Selected metric value (e.g., logit_gap)
                    "alignment": float(alignment),
                }

        return results_list

    # Compute baseline once (no steering) - shared across all layers
    print("Computing baseline (no steering)...")
    shared_baseline = run_all_questions()

    # ------------------------------------------------------------------
    # Multiplier-sweep acceleration: run *all non-zero multipliers* in one
    # forward pass per batch by expanding the batch and adding per-example
    # deltas with a BatchSteeringHook.
    # ------------------------------------------------------------------
    nonzero_multipliers = [m for m in multipliers if m != 0.0]
    k_mult = len(nonzero_multipliers)

    gpu_batches_expanded = None
    precomputed_expanded_batches = None
    if k_mult > 0:
        # Compute base batch size from EXPANDED_BATCH_TARGET.
        # This gives much better GPU utilization than the old BATCH_SIZE // k_mult approach.
        # E.g., with k_mult=6 and EXPANDED_BATCH_TARGET=48: base=8, expanded=48 sequences per forward pass.
        expanded_base_bs = max(1, EXPANDED_BATCH_TARGET // k_mult)
        actual_expanded_size = expanded_base_bs * k_mult
        print(f"Building expanded batches for {k_mult} multipliers (base batch={expanded_base_bs}, expanded={actual_expanded_size})...")
        gpu_batches_expanded = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, expanded_base_bs)

        # Precompute expanded inputs once - reused for every direction sweep.
        # This avoids redundant repeat_interleave calls for each layer/direction.
        print(f"Precomputing expanded inputs for {len(gpu_batches_expanded)} batches...")
        precomputed_expanded_batches = []
        for batch_indices, batch_inputs in gpu_batches_expanded:
            expanded_inputs = {
                name: tensor.repeat_interleave(k_mult, dim=0)
                for name, tensor in batch_inputs.items()
            }
            precomputed_expanded_batches.append((batch_indices, expanded_inputs))
        print(f"  Precomputed {len(precomputed_expanded_batches)} expanded batches on {DEVICE}")

    def run_all_questions_multi(layer_module, direction_tensor: torch.Tensor) -> Dict[float, List[Dict]]:
        """Run all questions for all non-zero multipliers in one sweep.

        Returns:
            dict: multiplier -> results_list (same format as run_all_questions()).
        """
        if k_mult == 0:
            return {}

        results_by_mult = {m: [None] * len(prompts) for m in nonzero_multipliers}
        mults_t = torch.tensor(nonzero_multipliers, device=DEVICE, dtype=direction_tensor.dtype)

        hook = BatchSteeringHook()
        hook.register(layer_module)
        try:
            for batch_indices, expanded_inputs in precomputed_expanded_batches:
                B = len(batch_indices)  # Original batch size (before expansion)
                k = k_mult

                # Build per-example deltas aligned with repeat_interleave order:
                # [ex0*m0..mk-1, ex1*m0..mk-1, ...]
                mults_rep = mults_t.repeat(B)  # (B*k,)
                delta_bh = direction_tensor[None, :] * mults_rep[:, None]  # (B*k, hidden)
                hook.set_delta(delta_bh)

                with torch.inference_mode():
                    batch_option_logits = _compute_batch_option_logits(
                        model, transformer, W_opt, option_token_ids, expanded_inputs
                    )
                    batch_option_probs = torch.softmax(batch_option_logits, dim=-1).float().cpu().numpy()

                # Map expanded outputs back to (question, multiplier)
                for i, q_idx in enumerate(batch_indices):
                    base = i * k
                    metric_val = direct_metric_values[q_idx]
                    metric_z = (metric_val - metric_mean) / metric_std

                    for j, mult in enumerate(nonzero_multipliers):
                        option_probs = batch_option_probs[base + j]
                        response = options[np.argmax(option_probs)]
                        confidence = local_response_to_confidence(response, option_probs, mappings[q_idx])

                        confidence_z = (confidence - 0.5) / 0.25
                        alignment = -metric_z * confidence_z

                        results_by_mult[mult][q_idx] = {
                            "question_idx": q_idx,
                            "response": response,
                            "confidence": confidence,
                            "metric": float(metric_val),  # Selected metric value
                            "alignment": float(alignment),
                        }

            return results_by_mult
        finally:
            hook.remove()


    for layer_idx in tqdm(layers, desc="Steering layers"):
        # Get layer module once
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        # Get precomputed tensors or compute them
        if cached_directions and layer_idx in cached_directions:
            introspection_tensor = cached_directions[layer_idx]["introspection"]
            control_tensors = cached_directions[layer_idx]["controls"]
        else:
            introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
            introspection_dir = introspection_dir / np.linalg.norm(introspection_dir)
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            introspection_tensor = torch.tensor(introspection_dir, dtype=dtype, device=DEVICE)
            control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)
            control_tensors = [torch.tensor(cd, dtype=dtype, device=DEVICE) for cd in control_dirs]

        layer_results = {
            "baseline": shared_baseline,
            "introspection": {m: [] for m in multipliers},
            "controls": {f"control_{i}": {m: [] for m in multipliers} for i in range(num_controls)},
        }

        
        # Introspection steering (vectorized over multipliers)
        if 0.0 in layer_results["introspection"]:
            layer_results["introspection"][0.0] = layer_results["baseline"]

        if k_mult > 0:
            multi_results = run_all_questions_multi(layer_module, introspection_tensor)
            for mult, res in multi_results.items():
                layer_results["introspection"][mult] = res


        # Control steering (vectorized over multipliers)
        for ctrl_idx, ctrl_tensor in enumerate(tqdm(control_tensors, desc="Controls", leave=False)):
            ctrl_key = f"control_{ctrl_idx}"
            if 0.0 in layer_results["controls"][ctrl_key]:
                layer_results["controls"][ctrl_key][0.0] = layer_results["baseline"]

            if k_mult > 0:
                multi_results = run_all_questions_multi(layer_module, ctrl_tensor)
                for mult, res in multi_results.items():
                    layer_results["controls"][ctrl_key][mult] = res

        results["layer_results"][layer_idx] = layer_results

    return results


# ============================================================================
# ABLATION EXPERIMENT
# ============================================================================

def run_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    directions: Dict,
    num_controls: int,
    use_chat_template: bool,
    baseline_results: Optional[List[Dict]] = None,
    cached_directions: Optional[Dict] = None
) -> Dict:
    """
    Run ablation experiment to test causality of introspection direction.

    For each layer, we:
    1. Collect baseline confidence-metric correlation (no intervention)
    2. Ablate the introspection direction and measure correlation
    3. Ablate control (random orthogonal) directions and measure correlation

    If the introspection direction is causal, ablating it should degrade the
    correlation more than ablating random directions.

    Args:
        direct_metric_values: The selected uncertainty metric values from direct MC task
                              (e.g., logit_gap, entropy, etc. as specified by METRIC)
        baseline_results: Optional pre-computed baseline results from steering experiment.
                          If provided, skips baseline computation for efficiency.
        cached_directions: Optional precomputed direction tensors on GPU.
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

    # Compute metric stats for alignment calculation
    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # Pre-format all prompts (avoid repeated work)
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_delegate_prompt(question, tokenizer, use_chat_template, trial_index=q_idx)
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    # Pre-tokenize all prompts once (BPE only, no padding yet)
    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)

    # Get option token IDs
    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = DELEGATE_OPTIONS
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = META_OPTIONS

    # Fast path: avoid computing full vocab logits.
    transformer, lm_head = _get_transformer_and_lm_head(model)
    W_opt = _prepare_option_weight(lm_head, model, option_token_ids)

    # Pad batches once and keep them on-device (reused across all sweeps).
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    def run_all_questions():
        """Run all questions in batches and return list of results.

        Uses length-sorted batching: process prompts in order of length so each batch
        has similar-length prompts, minimizing padding waste.
        """
        results_list = [None] * len(prompts)  # Pre-allocate to maintain order

        for batch_indices, batch_inputs in gpu_batches:
            with torch.inference_mode():
                batch_option_logits = _compute_batch_option_logits(
                    model, transformer, W_opt, option_token_ids, batch_inputs
                )
                batch_option_probs = torch.softmax(batch_option_logits, dim=-1).float().cpu().numpy()

            # Process each item in the batch (map back to original indices)
            for i, q_idx in enumerate(batch_indices):
                option_probs = batch_option_probs[i]
                response = options[np.argmax(option_probs)]
                confidence = local_response_to_confidence(response, option_probs, mappings[q_idx])

                metric_val = direct_metric_values[q_idx]
                metric_z = (metric_val - metric_mean) / metric_std
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -metric_z * confidence_z

                results_list[q_idx] = {
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "metric": float(metric_val),  # Selected metric value
                    "alignment": float(alignment),
                }

        return results_list

    # Compute baseline once if not provided
    if baseline_results is None:
        print("Computing baseline (no intervention)...")
        baseline_results = run_all_questions()

    for layer_idx in tqdm(layers, desc="Ablation layers"):
        # Get layer module once
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        # Get precomputed tensors or compute them
        if cached_directions and layer_idx in cached_directions:
            introspection_tensor = cached_directions[layer_idx]["introspection"]
            control_tensors = cached_directions[layer_idx]["controls"]
        else:
            introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
            introspection_dir = introspection_dir / np.linalg.norm(introspection_dir)
            dtype = torch.float16 if DEVICE == "cuda" else torch.float32
            introspection_tensor = torch.tensor(introspection_dir, dtype=dtype, device=DEVICE)
            control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)
            control_tensors = [torch.tensor(cd, dtype=dtype, device=DEVICE) for cd in control_dirs]

        layer_results = {
            "baseline": baseline_results,
            "introspection_ablated": [],
            "controls_ablated": {f"control_{i}": [] for i in range(num_controls)},
        }

        # Introspection direction ablation - register hook once
        hook = AblationHook(introspection_tensor, pre_normalized=True)
        hook.register(layer_module)
        try:
            layer_results["introspection_ablated"] = run_all_questions()
        finally:
            hook.remove()

        # Control direction ablations - register hook once per control
        for ctrl_idx, ctrl_tensor in enumerate(control_tensors):
            hook = AblationHook(ctrl_tensor, pre_normalized=True)
            hook.register(layer_module)
            try:
                layer_results["controls_ablated"][f"control_{ctrl_idx}"] = run_all_questions()
            finally:
                hook.remove()

        results["layer_results"][layer_idx] = layer_results

    return results


def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and uncertainty metric."""
    # We expect negative correlation for entropy-like metrics: high metric = low confidence
    # For logit_gap etc., sign depends on metric definition
    if len(confidences) < 2 or np.std(confidences) == 0 or np.std(metric_values) == 0:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def analyze_ablation_results(results: Dict) -> Dict:
    """Compute ablation effect statistics with proper statistical testing.

    Statistical improvements:
    1. Pooled null distribution: Collect all control effects across all layers
       to build a larger null distribution for more robust p-values
    2. Per-layer permutation test: Compare introspection effect to layer-specific controls
    3. FDR correction: Benjamini-Hochberg correction for multiple layer testing
    4. Bootstrap CIs: 95% confidence intervals on control effects
    """
    analysis = {
        "layers": results["layers"],
        "num_questions": results["num_questions"],
        "num_controls": results["num_controls"],
        "effects": {},
    }

    # First pass: collect all effects for pooled null distribution
    all_control_corr_changes = []  # Pooled across all layers
    layer_data = {}  # Store extracted data for second pass

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        # Extract data - results now use "metric" key instead of "entropy"
        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])
        baseline_align = np.array([r["alignment"] for r in lr["baseline"]])

        ablated_conf = np.array([r["confidence"] for r in lr["introspection_ablated"]])
        ablated_metric = np.array([r["metric"] for r in lr["introspection_ablated"]])
        ablated_align = np.array([r["alignment"] for r in lr["introspection_ablated"]])

        # Compute correlations (confidence vs selected metric)
        baseline_corr = compute_correlation(baseline_conf, baseline_metric)
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)

        # Control ablations
        control_corrs = []
        control_aligns = []
        control_confs = []
        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_metric = np.array([r["metric"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_align = np.array([r["alignment"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_metric))
            control_aligns.append(ctrl_align.mean())
            control_confs.append(ctrl_conf.mean())

        intro_corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        # Add to pooled null distribution
        all_control_corr_changes.extend(control_corr_changes)

        # Store for second pass
        layer_data[layer_idx] = {
            "baseline_corr": baseline_corr,
            "baseline_conf": baseline_conf,
            "baseline_metric": baseline_metric,
            "baseline_align": baseline_align,
            "ablated_corr": ablated_corr,
            "ablated_conf": ablated_conf,
            "ablated_align": ablated_align,
            "intro_corr_change": intro_corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
            "control_aligns": control_aligns,
            "control_confs": control_confs,
        }

    # Convert pooled null to array for efficient computation
    pooled_null = np.array(all_control_corr_changes)

    # Second pass: compute statistics with pooled null
    raw_p_values = []

    for layer_idx in results["layers"]:
        ld = layer_data[layer_idx]

        avg_control_corr = np.mean(ld["control_corrs"])
        avg_control_align = np.mean(ld["control_aligns"])
        std_control_corr = np.std(ld["control_corr_changes"])

        # Per-layer p-value (original method, kept for comparison)
        n_controls_worse_local = sum(1 for c in ld["control_corr_changes"] if c >= ld["intro_corr_change"])
        p_value_local = (n_controls_worse_local + 1) / (len(ld["control_corrs"]) + 1)

        # Pooled p-value: compare to all control effects across all layers
        # This gives much finer granularity (e.g., with 20 controls × 7 layers = 140 samples)
        n_pooled_worse = np.sum(pooled_null >= ld["intro_corr_change"])
        p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)

        # Bootstrap 95% CI for control effect
        n_bootstrap = 1000
        bootstrap_means = []
        ctrl_changes = np.array(ld["control_corr_changes"])
        for _ in range(n_bootstrap):
            boot_sample = np.random.choice(ctrl_changes, size=len(ctrl_changes), replace=True)
            bootstrap_means.append(np.mean(boot_sample))
        ci_low = np.percentile(bootstrap_means, 2.5)
        ci_high = np.percentile(bootstrap_means, 97.5)

        # Effect size: how many SDs away from control mean?
        if std_control_corr > 0:
            effect_size_z = (ld["intro_corr_change"] - np.mean(ld["control_corr_changes"])) / std_control_corr
        else:
            effect_size_z = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "baseline": {
                "correlation": ld["baseline_corr"],
                "mean_alignment": float(ld["baseline_align"].mean()),
                "mean_confidence": float(ld["baseline_conf"].mean()),
            },
            "introspection_ablated": {
                "correlation": ld["ablated_corr"],
                "correlation_change": ld["intro_corr_change"],
                "mean_alignment": float(ld["ablated_align"].mean()),
                "alignment_change": float(ld["ablated_align"].mean() - ld["baseline_align"].mean()),
                "mean_confidence": float(ld["ablated_conf"].mean()),
                "p_value_local": p_value_local,  # Per-layer (old method)
                "p_value_pooled": p_value_pooled,  # Pooled null (more powerful)
                "effect_size_z": effect_size_z,  # Z-score vs controls
            },
            "control_ablated": {
                "correlation_mean": avg_control_corr,
                "correlation_std": float(np.std(ld["control_corrs"])),
                "correlation_change_mean": float(np.mean(ld["control_corr_changes"])),
                "correlation_change_std": std_control_corr,
                "correlation_change_ci95": [float(ci_low), float(ci_high)],
                "mean_alignment": avg_control_align,
                "alignment_change": avg_control_align - float(ld["baseline_align"].mean()),
            },
            "individual_controls": {
                f"control_{i}": {
                    "correlation": ld["control_corrs"][i],
                    "correlation_change": ld["control_corr_changes"][i],
                }
                for i in range(len(ld["control_corrs"]))
            },
        }

    # FDR correction (Benjamini-Hochberg)
    # Sort p-values, compute adjusted p-values
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        # BH adjusted p-value: p * n / rank, but capped at 1 and monotonic
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    # Make monotonic (each p-value >= all smaller p-values)
    prev_adjusted = 0.0
    for layer_idx, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer_idx] = max(fdr_adjusted[layer_idx], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer_idx]

    # Add FDR-adjusted p-values to analysis
    for layer_idx in results["layers"]:
        analysis["effects"][layer_idx]["introspection_ablated"]["p_value_fdr"] = fdr_adjusted[layer_idx]

    # Summary statistics
    significant_layers_pooled = [l for l in results["layers"]
                                  if analysis["effects"][l]["introspection_ablated"]["p_value_pooled"] < 0.05]
    significant_layers_fdr = [l for l in results["layers"]
                              if analysis["effects"][l]["introspection_ablated"]["p_value_fdr"] < 0.05]

    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_pooled_p05": significant_layers_pooled,
        "significant_layers_fdr_p05": significant_layers_fdr,
        "n_significant_pooled": len(significant_layers_pooled),
        "n_significant_fdr": len(significant_layers_fdr),
    }

    return analysis


def get_expected_slope_sign(metric: str) -> int:
    """
    Get the expected sign of the confidence slope for a given metric.

    The probe direction points toward *increasing* the metric value.
    - entropy: HIGH = uncertain → steering +direction should DECREASE confidence → expected slope < 0
    - logit_gap, top_prob, margin, top_logit: HIGH = confident → expected slope > 0

    Returns:
        +1 if +direction should increase confidence
        -1 if +direction should decrease confidence
    """
    if metric == "entropy":
        return -1  # +direction = more uncertain = less confident
    else:
        return +1  # +direction = more confident


def analyze_results(results: Dict, metric: str = None) -> Dict:
    """Compute summary statistics.

    Args:
        results: Raw steering results
        metric: The uncertainty metric used (for sign interpretation). If None,
                sign interpretation is skipped.
    """
    analysis = {
        "layers": results["layers"],
        "multipliers": results["multipliers"],
        "metric": metric,
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

        # Compute slopes - use confidence_change as primary metric
        # (steering should shift confidence systematically, not alignment which conflates two effects)
        intro_conf_slope = np.polyfit(multipliers, [effects["introspection"][m]["confidence_change"] for m in multipliers], 1)[0]
        ctrl_conf_slope = np.polyfit(multipliers, [effects["control_avg"][m]["confidence_change"] for m in multipliers], 1)[0]

        # Keep alignment slopes for reference
        intro_align_slope = np.polyfit(multipliers, [effects["introspection"][m]["alignment_change"] for m in multipliers], 1)[0]
        ctrl_align_slope = np.polyfit(multipliers, [effects["control_avg"][m]["alignment_change"] for m in multipliers], 1)[0]

        analysis["effects"][layer_idx] = {
            "by_multiplier": effects,
            "slopes": {
                "introspection": float(intro_conf_slope),
                "control_avg": float(ctrl_conf_slope),
                "introspection_alignment": float(intro_align_slope),
                "control_avg_alignment": float(ctrl_align_slope),
            },
            "baseline_alignment": float(baseline_align),
            "baseline_confidence": float(baseline_conf),
        }

    # Add sign interpretation metadata
    if metric:
        expected_sign = get_expected_slope_sign(metric)
        analysis["sign_interpretation"] = {
            "metric": metric,
            "expected_slope_sign": expected_sign,
            "expected_slope_sign_str": "positive" if expected_sign > 0 else "negative",
            "explanation": (
                f"For {metric}, +direction should → "
                f"{'higher' if expected_sign > 0 else 'lower'} confidence"
            ),
        }

        # Check best layer's sign
        if analysis["effects"]:
            best_layer = max(
                analysis["layers"],
                key=lambda l: abs(analysis["effects"][l]["slopes"]["introspection"])
            )
            best_slope = analysis["effects"][best_layer]["slopes"]["introspection"]
            actual_sign = 1 if best_slope > 0 else -1
            analysis["sign_interpretation"]["best_layer"] = best_layer
            analysis["sign_interpretation"]["actual_slope_sign"] = actual_sign
            analysis["sign_interpretation"]["sign_matches_expected"] = (actual_sign == expected_sign)

    return analysis


def plot_results(analysis: Dict, output_prefix: str):
    """Create visualizations."""
    layers = analysis["layers"]
    multipliers = analysis["multipliers"]

    if not layers:
        print("  Skipping plot - no layers to visualize")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Confidence slopes by layer
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
    ax1.set_ylabel("Confidence Slope (Δconf / Δmult)")
    ax1.set_title("Steering Effect on Confidence")
    ax1.legend()

    # Plot 2: Best layer detail - show confidence change
    best_layer = max(layers, key=lambda l: abs(analysis["effects"][l]["slopes"]["introspection"]))
    ax2 = axes[1]

    intro_conf = [analysis["effects"][best_layer]["by_multiplier"]["introspection"][m]["confidence_change"] for m in multipliers]
    ctrl_conf = [analysis["effects"][best_layer]["by_multiplier"]["control_avg"][m]["confidence_change"] for m in multipliers]

    ax2.plot(multipliers, intro_conf, 'o-', label='Introspection', linewidth=2, color='green')
    ax2.plot(multipliers, ctrl_conf, '^--', label='Control', linewidth=2, color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Steering Multiplier")
    ax2.set_ylabel("Δ Confidence")
    ax2.set_title(f"Confidence Change (Layer {best_layer})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary
    ax3 = axes[2]
    ax3.axis('off')

    intro_slope = analysis["effects"][best_layer]["slopes"]["introspection"]
    ctrl_slope = analysis["effects"][best_layer]["slopes"]["control_avg"]
    metric = analysis.get("metric", "unknown")

    summary = f"""
STEERING EXPERIMENT SUMMARY

Metric: {metric}
Best Layer: {best_layer}
  Confidence slope: {intro_slope:.4f}
  Control slope: {ctrl_slope:.4f}
  Difference: {abs(intro_slope) - abs(ctrl_slope):.4f}

Interpretation:
"""
    # Check if introspection direction causes systematic confidence shift
    if abs(intro_slope) > abs(ctrl_slope) + 0.01:
        direction_str = "lower" if intro_slope < 0 else "higher"
        summary += f"""  ✓ Steering shifts confidence systematically
  +multiplier → {direction_str} confidence
  Effect stronger than controls."""

        # Check sign against expectation
        if metric and metric != "unknown":
            expected_sign = get_expected_slope_sign(metric)
            actual_sign = 1 if intro_slope > 0 else -1
            if actual_sign == expected_sign:
                summary += f"""

  ✓ Sign correct for {metric}"""
            else:
                summary += f"""

  ⚠ Sign OPPOSITE to expected!"""
    elif abs(intro_slope) > 0.01:
        summary += """  ⚠ Weak steering effect
  Some confidence shift but
  not clearly above controls."""
    else:
        summary += """  ✗ No steering effect detected
  Direction doesn't shift confidence."""

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

    if not analysis["layers"]:
        print("\n⚠ No layers were tested - check layer selection criteria")
        return

    metric = analysis.get("metric")
    if metric:
        expected_sign = get_expected_slope_sign(metric)
        expected_direction = "negative" if expected_sign < 0 else "positive"
        print(f"\nMetric: {metric}")
        print(f"Expected slope sign: {expected_direction} (+direction → {'less' if expected_sign < 0 else 'more'} confident)")

    print("\n--- Confidence Slopes by Layer ---")
    print(f"{'Layer':<8} {'Introspection':<15} {'Control':<15}")
    print("-" * 40)

    for layer in analysis["layers"]:
        s = analysis["effects"][layer]["slopes"]
        print(f"{layer:<8} {s['introspection']:<15.4f} {s['control_avg']:<15.4f}")

    # Best layer - pick largest magnitude slope
    best_layer = max(analysis["layers"], key=lambda l: abs(analysis["effects"][l]["slopes"]["introspection"]))
    best_intro = analysis["effects"][best_layer]["slopes"]["introspection"]
    best_ctrl = analysis["effects"][best_layer]["slopes"]["control_avg"]

    print(f"\nStrongest steering effect: Layer {best_layer}")
    print(f"  Confidence slope: {best_intro:.4f}")
    print(f"  Control slope: {best_ctrl:.4f}")

    if abs(best_intro) > abs(best_ctrl) + 0.01:
        direction = "lower" if best_intro < 0 else "higher"
        print(f"\n✓ Steering systematically shifts confidence {direction}!")
        print("  Effect stronger than random controls.")

        # Check if sign matches expectation
        if metric:
            expected_sign = get_expected_slope_sign(metric)
            actual_sign = 1 if best_intro > 0 else -1
            if actual_sign == expected_sign:
                print(f"  ✓ Sign matches expectation for {metric} (direction transfers correctly)")
            else:
                print(f"  ⚠ Sign is OPPOSITE to expectation for {metric}!")
                print(f"    This suggests the direction may not transfer from direct→meta context,")
                print(f"    or the representation differs between contexts.")
    elif abs(best_intro) > 0.01:
        print("\n⚠ Weak effect, not clearly separable from controls")
    else:
        print("\n✗ No steering effect found")


def plot_ablation_results(analysis: Dict, output_prefix: str):
    """Create improved ablation visualizations.

    Three panels:
    1. Absolute correlations (baseline, introspection-ablated, control-ablated with CI)
    2. Effect size with 95% CI - shows if introspection effect differs from controls
    3. Distribution plot - violin/box of control effects with introspection overlay
    """
    layers = analysis["layers"]

    if not layers:
        print("  Skipping ablation plot - no layers to visualize")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ==========================================================================
    # Plot 1: Absolute correlation values (not deltas)
    # Shows baseline, introspection-ablated, and control-ablated correlations
    # ==========================================================================
    ax1 = axes[0]

    baseline_corrs = [analysis["effects"][l]["baseline"]["correlation"] for l in layers]
    intro_corrs = [analysis["effects"][l]["introspection_ablated"]["correlation"] for l in layers]
    ctrl_corrs = [analysis["effects"][l]["control_ablated"]["correlation_mean"] for l in layers]
    ctrl_stds = [analysis["effects"][l]["control_ablated"]["correlation_std"] for l in layers]

    x = np.arange(len(layers))

    # Plot lines with markers
    ax1.plot(x, baseline_corrs, 'o-', label='Baseline (no ablation)', color='blue', linewidth=2, markersize=8)
    ax1.plot(x, intro_corrs, 's-', label='Introspection ablated', color='red', linewidth=2, markersize=8)
    ax1.errorbar(x, ctrl_corrs, yerr=ctrl_stds, fmt='^--', label='Control ablated (mean±SD)',
                 color='gray', linewidth=1.5, markersize=7, capsize=3, alpha=0.8)

    ax1.axhline(y=0, color='black', linestyle=':', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Correlation (confidence vs metric)")
    ax1.set_title("Correlation Values by Condition")
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Add annotation about expected direction
    ax1.text(0.02, 0.02, "Negative corr = well-calibrated\n(high metric → low confidence)",
             transform=ax1.transAxes, fontsize=8, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 2: Effect size with 95% CI
    # Shows (introspection_change - control_mean) with CI from bootstrap
    # ==========================================================================
    ax2 = axes[1]

    # Compute differential effect: introspection effect minus control effect
    diff_effects = []
    diff_ci_low = []
    diff_ci_high = []
    p_values_fdr = []

    for l in layers:
        intro_change = analysis["effects"][l]["introspection_ablated"]["correlation_change"]
        ctrl_mean = analysis["effects"][l]["control_ablated"]["correlation_change_mean"]
        ctrl_ci = analysis["effects"][l]["control_ablated"]["correlation_change_ci95"]

        # Differential effect
        diff = intro_change - ctrl_mean
        diff_effects.append(diff)

        # CI on the difference (approximate: introspection is fixed, so CI comes from control variance)
        diff_ci_low.append(intro_change - ctrl_ci[1])  # intro - upper_ctrl = lower bound of diff
        diff_ci_high.append(intro_change - ctrl_ci[0])  # intro - lower_ctrl = upper bound of diff

        p_values_fdr.append(analysis["effects"][l]["introspection_ablated"].get("p_value_fdr", 1.0))

    # Color bars by significance
    colors = ['red' if p < 0.05 else 'orange' if p < 0.1 else 'gray' for p in p_values_fdr]

    # Plot bars with error bars
    bars = ax2.bar(x, diff_effects, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.errorbar(x, diff_effects,
                 yerr=[np.array(diff_effects) - np.array(diff_ci_low),
                       np.array(diff_ci_high) - np.array(diff_effects)],
                 fmt='none', color='black', capsize=4, capthick=1.5)

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Differential Effect\n(intro_Δcorr − control_Δcorr)")
    ax2.set_title("Introspection Effect vs Controls (with 95% CI)")
    ax2.grid(True, alpha=0.3, axis='y')

    # Add legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, edgecolor='black', label='p < 0.05 (FDR)'),
        Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='p < 0.10 (FDR)'),
        Patch(facecolor='gray', alpha=0.7, edgecolor='black', label='n.s.'),
    ]
    ax2.legend(handles=legend_elements, loc='best', fontsize=9)

    # Annotation
    ax2.text(0.02, 0.98, "Positive = ablation hurts\nintrospection more than controls",
             transform=ax2.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # ==========================================================================
    # Plot 3: Distribution plot - show where introspection falls in null distribution
    # ==========================================================================
    ax3 = axes[2]

    # Collect control correlation changes for each layer
    positions = []
    control_data = []
    intro_points = []

    for i, l in enumerate(layers):
        ctrl_changes = [
            analysis["effects"][l]["individual_controls"][f"control_{j}"]["correlation_change"]
            for j in range(len(analysis["effects"][l]["individual_controls"]))
        ]
        control_data.append(ctrl_changes)
        intro_points.append(analysis["effects"][l]["introspection_ablated"]["correlation_change"])
        positions.append(i)

    # Create violin plot for controls
    parts = ax3.violinplot(control_data, positions=positions, showmeans=True, showmedians=False)

    # Style the violins
    for pc in parts['bodies']:
        pc.set_facecolor('lightgray')
        pc.set_edgecolor('gray')
        pc.set_alpha(0.7)
    parts['cmeans'].set_color('gray')
    parts['cmeans'].set_linewidth(2)
    parts['cbars'].set_color('gray')
    parts['cmins'].set_color('gray')
    parts['cmaxs'].set_color('gray')

    # Overlay introspection points
    for i, (pos, intro_val, p_val) in enumerate(zip(positions, intro_points, p_values_fdr)):
        color = 'red' if p_val < 0.05 else 'orange' if p_val < 0.1 else 'darkred'
        marker = '*' if p_val < 0.05 else 'o'
        size = 200 if p_val < 0.05 else 100
        ax3.scatter(pos, intro_val, color=color, s=size, marker=marker, zorder=5,
                    edgecolor='black', linewidth=1)

    ax3.axhline(y=0, color='black', linestyle=':', alpha=0.5)
    ax3.set_xticks(positions)
    ax3.set_xticklabels(layers)
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Δ Correlation (ablated − baseline)")
    ax3.set_title("Introspection Effect in Null Distribution")
    ax3.grid(True, alpha=0.3, axis='y')

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor='lightgray', edgecolor='gray', alpha=0.7, label='Control distribution'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='red', markersize=15,
               markeredgecolor='black', label='Introspection (p<0.05)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='darkred', markersize=10,
               markeredgecolor='black', label='Introspection (n.s.)'),
    ]
    ax3.legend(handles=legend_elements, loc='best', fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_ablation_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_ablation_results.png")
    plt.close()

    # ==========================================================================
    # Also create a summary figure with text
    # ==========================================================================
    fig2, ax_summary = plt.subplots(1, 1, figsize=(8, 6))
    ax_summary.axis('off')

    # Get summary statistics
    summary_stats = analysis.get("summary", {})
    sig_pooled = summary_stats.get("significant_layers_pooled_p05", [])
    sig_fdr = summary_stats.get("significant_layers_fdr_p05", [])
    pooled_n = summary_stats.get("pooled_null_size", 0)

    # Find best layer by FDR p-value
    best_layer = min(layers, key=lambda l: analysis["effects"][l]["introspection_ablated"].get("p_value_fdr", 1.0))
    best_stats = analysis["effects"][best_layer]
    best_p_fdr = best_stats["introspection_ablated"].get("p_value_fdr", 1.0)
    best_p_pooled = best_stats["introspection_ablated"].get("p_value_pooled", 1.0)
    best_effect_z = best_stats["introspection_ablated"].get("effect_size_z", 0.0)

    summary_text = f"""
ABLATION ANALYSIS SUMMARY
{'='*50}

Statistical Method:
  • Pooled null distribution: {pooled_n} control effects
    (all layers × all control directions)
  • Multiple comparisons: Benjamini-Hochberg FDR correction
  • Bootstrap 95% CIs on control effects (n=1000)

Results:
  • Layers tested: {len(layers)}
  • Significant (pooled p<0.05): {len(sig_pooled)} layers
    {sig_pooled if sig_pooled else 'None'}
  • Significant (FDR p<0.05): {len(sig_fdr)} layers
    {sig_fdr if sig_fdr else 'None'}

Best Layer: {best_layer}
  • Baseline correlation: {best_stats['baseline']['correlation']:.4f}
  • After introspection ablation: {best_stats['introspection_ablated']['correlation']:.4f}
  • Δcorr (introspection): {best_stats['introspection_ablated']['correlation_change']:.4f}
  • Δcorr (controls mean): {best_stats['control_ablated']['correlation_change_mean']:.4f}
  • Effect size (Z): {best_effect_z:.2f} SD
  • p-value (pooled): {best_p_pooled:.4f}
  • p-value (FDR-adjusted): {best_p_fdr:.4f}

Interpretation:
"""
    if len(sig_fdr) > 0:
        summary_text += f"""  ✓ SIGNIFICANT after FDR correction
  {len(sig_fdr)} layer(s) show introspection ablation
  degrades calibration more than random directions.
  This is evidence for a causal role of the direction."""
    elif len(sig_pooled) > 0:
        summary_text += f"""  ⚠ Significant before FDR correction only
  {len(sig_pooled)} layer(s) significant at pooled p<0.05
  but not after multiple comparison correction.
  Suggestive but not definitive evidence."""
    elif best_p_pooled < 0.1:
        summary_text += """  ⚠ Marginal effect (p < 0.10)
  Some trend toward introspection ablation
  having larger effect, but not significant.
  May need more statistical power."""
    else:
        summary_text += """  ✗ No significant effect detected
  Introspection ablation does not clearly
  differ from control ablations.
  Direction may not be causally involved."""

    ax_summary.text(0.05, 0.95, summary_text, transform=ax_summary.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.9))

    plt.savefig(f"{output_prefix}_ablation_summary.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_ablation_summary.png")
    plt.close()


def print_ablation_summary(analysis: Dict):
    """Print summary of ablation results with improved statistics."""
    print("\n" + "=" * 70)
    print("ABLATION EXPERIMENT RESULTS")
    print("=" * 70)

    if not analysis["layers"]:
        print("\n⚠ No layers were tested - check layer selection criteria")
        return

    # Get summary stats
    summary = analysis.get("summary", {})
    pooled_n = summary.get("pooled_null_size", 0)
    sig_pooled = summary.get("significant_layers_pooled_p05", [])
    sig_fdr = summary.get("significant_layers_fdr_p05", [])

    print(f"\nStatistical method: Pooled null distribution ({pooled_n} control effects)")
    print("                    + Benjamini-Hochberg FDR correction")

    print("\n--- Correlation by Layer ---")
    print(f"{'Layer':<6} {'Baseline':<10} {'Ablated':<10} {'Ctrl±SD':<14} {'Δcorr':<8} {'EffectZ':<8} {'p(pool)':<8} {'p(FDR)':<8}")
    print("-" * 82)

    for layer in analysis["layers"]:
        e = analysis["effects"][layer]
        ctrl_mean = e['control_ablated']['correlation_mean']
        ctrl_std = e['control_ablated']['correlation_std']
        intro_change = e['introspection_ablated']['correlation_change']
        effect_z = e['introspection_ablated'].get('effect_size_z', 0.0)
        p_pooled = e['introspection_ablated'].get('p_value_pooled', float('nan'))
        p_fdr = e['introspection_ablated'].get('p_value_fdr', float('nan'))

        # Mark significant layers
        sig_marker = "**" if p_fdr < 0.05 else "*" if p_pooled < 0.05 else ""

        print(f"{layer:<6} {e['baseline']['correlation']:<10.4f} "
              f"{e['introspection_ablated']['correlation']:<10.4f} "
              f"{ctrl_mean:.3f}±{ctrl_std:.3f}  "
              f"{intro_change:<8.4f} {effect_z:<8.2f} "
              f"{p_pooled:<8.4f} {p_fdr:<8.4f} {sig_marker}")

    # Find best layer by FDR p-value
    best_layer = min(
        analysis["layers"],
        key=lambda l: analysis["effects"][l]["introspection_ablated"].get("p_value_fdr", 1.0)
    )
    best_stats = analysis["effects"][best_layer]
    best_p_fdr = best_stats["introspection_ablated"].get("p_value_fdr", 1.0)
    best_p_pooled = best_stats["introspection_ablated"].get("p_value_pooled", 1.0)
    best_effect_z = best_stats["introspection_ablated"].get("effect_size_z", 0.0)

    print(f"\n--- Summary ---")
    print(f"Significant layers (pooled p<0.05): {len(sig_pooled)} {sig_pooled if sig_pooled else ''}")
    print(f"Significant layers (FDR p<0.05):    {len(sig_fdr)} {sig_fdr if sig_fdr else ''}")

    print(f"\nBest layer by FDR p-value: Layer {best_layer}")
    print(f"  Baseline correlation:     {best_stats['baseline']['correlation']:.4f}")
    print(f"  After introspection abl:  {best_stats['introspection_ablated']['correlation']:.4f}")
    print(f"  Δcorr (introspection):    {best_stats['introspection_ablated']['correlation_change']:.4f}")
    print(f"  Δcorr (controls mean):    {best_stats['control_ablated']['correlation_change_mean']:.4f}")
    print(f"  Effect size (Z-score):    {best_effect_z:.2f} SD from control mean")
    print(f"  p-value (pooled):         {best_p_pooled:.4f}")
    print(f"  p-value (FDR-adjusted):   {best_p_fdr:.4f}")

    # Interpretation
    if len(sig_fdr) > 0:
        print(f"\n✓ SIGNIFICANT CAUSAL EFFECT (FDR-corrected p < 0.05)")
        print(f"  {len(sig_fdr)} layer(s) survive multiple comparison correction.")
        print("  Ablating the introspection direction degrades calibration")
        print("  significantly more than ablating random orthogonal directions.")
        print("  This is evidence the direction is causally involved in confidence.")
    elif len(sig_pooled) > 0:
        print(f"\n⚠ SUGGESTIVE but not FDR-significant")
        print(f"  {len(sig_pooled)} layer(s) significant at pooled p<0.05,")
        print("  but not after multiple comparison correction.")
        print("  May indicate a real but weak effect, or need more power.")
    elif best_p_pooled < 0.1:
        print("\n⚠ MARGINAL TREND (p < 0.10)")
        print("  Some suggestion of effect but not statistically reliable.")
        print("  Consider more control directions or more questions for power.")
    else:
        print("\n✗ NO SIGNIFICANT EFFECT DETECTED")
        print("  Cannot distinguish introspection ablation from random ablations.")
        print("  The direction may not be causally involved in confidence judgments.")


# ============================================================================
# MAIN
# ============================================================================

def main():
    global METRIC

    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Run steering and ablation experiments")
    parser.add_argument("--metric", type=str, choices=AVAILABLE_METRICS, default=METRIC,
                        help=f"Metric to use for directions when DIRECTION_TYPE='entropy' or 'introspection' (default: {METRIC})")
    args = parser.parse_args()
    METRIC = args.metric

    print(f"Device: {DEVICE}")
    print(f"Direction type: {DIRECTION_TYPE}")
    if DIRECTION_TYPE in ("entropy", "introspection"):
        print(f"Metric: {METRIC}")
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
        # Metric directions from run_introspection_experiment.py
        # Note: output file naming changed to include metric (e.g., *_logit_gap_results.json)
        probe_results_path = f"{output_prefix}_{METRIC}_results.json"
        directions_path = f"{output_prefix}_{METRIC}_directions.npz"
        direction_key_template = "layer_{}_{}"  # Will be formatted with metric name
        transfer_results_path = None
    else:
        # Introspection directions from run_introspection_probe.py
        # Note: output file naming changed to include metric (e.g., *_logit_gap_probe_results.json)
        probe_results_path = f"{output_prefix}_{METRIC}_probe_results.json"
        directions_path = f"{output_prefix}_{METRIC}_probe_directions.npz"
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
            # For entropy directions, select layers based on probe performance
            # The probe results file may come from run_introspection_experiment.py ("probe_results" key)
            # or from run_introspection_probe.py ("layer_results" key)
            layer_candidates = []

            if "probe_results" in probe_results:
                # Structure from run_introspection_experiment.py
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
                    for layer_str, lr in probe_results["probe_results"].items():
                        d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
                        all_layers.append((int(layer_str), d2d_r2))
                    all_layers.sort(key=lambda x: -x[1])  # Sort by R² descending
                    layers = [l[0] for l in all_layers[:5]]  # Top 5
            elif "layer_results" in probe_results:
                # Structure from run_introspection_probe.py
                # Use significant layers or best R² layers
                for layer_str, lr in probe_results["layer_results"].items():
                    test_r2 = lr.get("test_r2", 0)
                    if lr.get("significant_p05", False) or test_r2 > 0.1:
                        layer_candidates.append((int(layer_str), test_r2))
                layer_candidates.sort(key=lambda x: -x[1])
                layers = [l[0] for l in layer_candidates]
                # If no good layers found, use top 5 by R²
                if not layers:
                    all_layers = []
                    for layer_str, lr in probe_results["layer_results"].items():
                        test_r2 = lr.get("test_r2", 0)
                        all_layers.append((int(layer_str), test_r2))
                    all_layers.sort(key=lambda x: -x[1])
                    layers = [l[0] for l in all_layers[:5]]
            else:
                # Fallback: use all layers from directions file
                print("  Warning: Unknown probe results structure, using all available layers")
                layers = [int(k.split("_")[1]) for k in directions.keys() if k.startswith("layer_")]
                layers = sorted(layers)[:10]  # Limit to first 10

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

    # Load the metric values for alignment calculation
    # New format has direct_metrics (dict of metric_name -> list of values)
    # Old format has direct_entropies (list of values)
    if "direct_metrics" in paired_data and METRIC in paired_data["direct_metrics"]:
        direct_metric_values = np.array(paired_data["direct_metrics"][METRIC])[:NUM_STEERING_QUESTIONS]
        print(f"Using metric '{METRIC}' from paired data")
    elif "direct_entropies" in paired_data:
        # Backward compatibility: fall back to direct_entropies
        direct_metric_values = np.array(paired_data["direct_entropies"])[:NUM_STEERING_QUESTIONS]
        print(f"Using 'entropy' (backward compatible fallback)")
    else:
        raise ValueError("Paired data missing both 'direct_metrics' and 'direct_entropies'")

    print(f"Using {len(questions)} questions")

    # Load model using centralized utility
    adapter_path = MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=adapter_path,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    # Initialize token ID cache once (avoids repeated tokenization)
    initialize_token_cache(tokenizer)

    # Ensure deterministic inference (no dropout) and a tiny speedup.
    model.eval()

    # Determine chat template usage (check once, not per prompt)
    use_chat_template = should_use_chat_template(BASE_MODEL_NAME, tokenizer)
    print(f"Using chat template: {use_chat_template}")

    # Precompute direction tensors on GPU
    print("Precomputing direction tensors...")
    direction_dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = precompute_direction_tensors(
        directions, layers, NUM_CONTROL_DIRECTIONS, DEVICE, direction_dtype
    )
    print(f"  Cached {len(layers)} layers with {NUM_CONTROL_DIRECTIONS} controls each")

    # Add direction_type suffix to output files to distinguish them
    # Include metric name when using entropy or introspection directions
    if DIRECTION_TYPE == "entropy":
        direction_suffix = f"_{METRIC}"
    elif DIRECTION_TYPE == "introspection":
        direction_suffix = f"_{METRIC}_introspection"
    else:
        direction_suffix = f"_{DIRECTION_TYPE}"

    # Skip steering for introspection directions - steering doesn't make sense conceptually.
    # The introspection direction captures calibration quality (metric-confidence alignment),
    # not a direct uncertainty signal. We can't causally steer toward "being well-calibrated"
    # without knowing a question's actual uncertainty. Ablation still makes sense: removing
    # "awareness of calibration" should degrade the metric-confidence correlation.
    baseline_from_steering = None
    if DIRECTION_TYPE == "introspection":
        print("\n" + "=" * 70)
        print("SKIPPING STEERING EXPERIMENT (introspection directions)")
        print("=" * 70)
        print("Steering with introspection directions doesn't make conceptual sense.")
        print("The direction captures calibration quality, not a causal uncertainty signal.")
        print("Running ablation experiment only...")
    else:
        # Run steering experiment
        results = run_steering_experiment(
            model, tokenizer, questions, direct_metric_values,
            layers, directions, STEERING_MULTIPLIERS, NUM_CONTROL_DIRECTIONS,
            use_chat_template, cached_directions
        )

        # Analyze
        analysis = analyze_results(results, metric=METRIC)

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

        # Extract baseline from steering results for ablation (first layer's baseline, they're all the same)
        first_layer = layers[0]
        baseline_from_steering = results["layer_results"][first_layer]["baseline"]

    # ==========================================================================
    # ABLATION EXPERIMENT
    # ==========================================================================
    print("\n" + "=" * 70)
    print("RUNNING ABLATION EXPERIMENT")
    print("=" * 70)

    # baseline_from_steering is set above (None if we skipped steering, otherwise from steering results)
    # run_ablation_experiment will compute its own baseline if baseline_results=None
    ablation_results = run_ablation_experiment(
        model, tokenizer, questions, direct_metric_values,
        layers, directions, NUM_CONTROL_DIRECTIONS,
        use_chat_template,
        baseline_results=baseline_from_steering,
        cached_directions=cached_directions
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
