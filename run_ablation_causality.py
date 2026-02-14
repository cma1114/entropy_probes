"""
Stage 3. Ablation (necessity) test for uncertainty, answer, or confidence directions.
Tests whether directions are causally necessary for the model's meta-judgments by
ablating each direction and measuring degradation in stated confidence correlation.

Supports multiple direction types via DIRECTION_TYPE:
- "uncertainty": Entropy/logit_gap directions (from identify_mc_correlate.py)
- "answer": MC answer A/B/C/D directions (from identify_mc_correlate.py with FIND_ANSWER_DIRECTIONS=True)
- "confidence": Stated confidence directions (from test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True)
- "metamcuncert": MC uncertainty directions found from meta activations (from test_meta_transfer.py with FIND_MC_UNCERTAINTY_DIRECTIONS=True)

Tests all layers with pooled null distribution + FDR correction.

Inputs:
    outputs/{base}_mc_{metric}_directions.npz            Uncertainty directions
    outputs/{base}_mc_answer_directions.npz              Answer directions (if DIRECTION_TYPE="answer")
    outputs/{base}_meta_{task}_confdir_directions.npz Confidence directions (if DIRECTION_TYPE="confidence")
    outputs/{base}_meta_{task}_mcuncert_directions.npz Meta→MC uncertainty directions (if DIRECTION_TYPE="metamcuncert")
    outputs/{base}_mc_results.json                        Consolidated results (dataset + metrics)

Outputs (one file per method, with per-position plots):
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_results.json
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_{position}.png

    where {base} = {dataset} (model info is in directory path)
          {dir_suffix} = "{direction_type}_{metric}" for uncertainty, else "{direction_type}"
          {method} = "probe" or "mean_diff"
          {position} = token position tested (e.g., "final")

Shared parameters (must match across scripts):
    SEED, TRAIN_SPLIT

Run after: identify_mc_correlate.py
    + test_meta_transfer.py (if using DIRECTION_TYPE="confidence" or "metamcuncert")
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, norm
from sklearn.model_selection import train_test_split

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    get_model_dir_name,
    DEVICE,
)
from core.config_utils import get_config_dict, get_output_path, find_output_file
from core.logging_utils import (
    print_run_header,
    print_key_findings,
    print_run_footer,
)
from core.plotting import save_figure, METHOD_COLORS, GRID_ALPHA, CI_ALPHA, CONDITION_COLORS
from core.steering import generate_orthogonal_directions
from core.steering_experiments import (
    SteeringExperimentConfig,
    BatchAblationHook,
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
    precompute_direction_tensors,
)
from core.metrics import metric_sign_for_confidence
from tasks import (
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    format_answer_or_delegate_prompt,
    get_answer_or_delegate_signal,
    format_other_confidence_prompt,
    get_other_confidence_signal,
    STATED_CONFIDENCE_OPTIONS,
    ANSWER_OR_DELEGATE_OPTIONS,
    OTHER_CONFIDENCE_OPTIONS,
    find_mc_positions,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: LoRA adapter path (must match identify step if used)
DATASET = "TriviaMC_difficulty_filtered"  # Dataset name (model prefix now in directory)
METRIC = "logit_gap"  # Which metric's directions to test (for DIRECTION_TYPE="uncertainty")
META_TASK = "confidence"  # "confidence", "delegate", or "other_confidence"
PROBE_POSITION = "final"  # Position from test_meta_transfer.py outputs

# Direction type to ablate:
# - "uncertainty": Ablate uncertainty directions (from identify_mc_correlate.py)
# - "answer": Ablate MC answer directions (from identify_mc_correlate.py with FIND_ANSWER_DIRECTIONS=True)
# - "confidence": Ablate confidence directions (from test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True)
# - "metamcuncert": Ablate MC uncertainty directions found from meta activations (test_meta_transfer.py)
DIRECTION_TYPE = "uncertainty"

# Descriptions for each direction type (used in summary output)
DIRECTION_DESCRIPTIONS = {
    "uncertainty": {
        "trained_on": "logit_gap/entropy from MC task",
        "interpretation": "Tests if uncertainty signal is necessary for calibrated confidence",
    },
    "answer": {
        "trained_on": "A/B/C/D answer probabilities from MC task",
        "interpretation": "Tests if answer representation affects confidence-uncertainty correlation",
    },
    "confidence": {
        "trained_on": "stated confidence from meta-task (same output being measured)",
        "interpretation": "Tests if confidence expression mechanism affects calibration (partially circular)",
    },
    "metamcuncert": {
        "trained_on": "MC uncertainty predicted from meta-task activations",
        "interpretation": "Tests if meta-task uses uncertainty direction found in its own activations",
    },
}

# Confidence signal used as the meta-task output target.
# - For META_TASK=delegate:
#     * "prob"        -> P(Answer) from the Answer/Delegate decision tokens.
#     * "logit_margin" -> logit(Answer) - logit(Delegate) for those same tokens.
# - For META_TASK=confidence: this script always uses the stated-confidence option probabilities;
#   if you set CONFIDENCE_SIGNAL="logit_margin" we fall back to "prob" for that task.
CONFIDENCE_SIGNAL = "prob"  # "prob" or "logit_margin"

# --- Quantization ---
LOAD_IN_4BIT = False   # Set True for 70B+ models
LOAD_IN_8BIT = False

# --- Experiment ---
SEED = 42                    # Must match across scripts
BATCH_SIZE = 4
NUM_QUESTIONS = 100          # How many questions (ignored if USE_TRANSFER_SPLIT=True)
NUM_CONTROLS = 25            # Random orthogonal directions per layer for null distribution

# Use same train/test split as transfer analysis (recommended for apples-to-apples comparison)
# When True: uses the test set from 80/20 split with SEED, ignoring NUM_QUESTIONS
# When False: uses first NUM_QUESTIONS from dataset (legacy behavior)
USE_TRANSFER_SPLIT = True
TRAIN_SPLIT = 0.8            # Must match across scripts

# --- Direction-finding (must match across scripts) ---
# (No probe training here, but layer selection uses these indirectly)

# --- Script-specific ---
# Expanded batch target for batched ablation.
# When ablating k directions (1 primary + NUM_CONTROLS), we expand each base batch by k.
# Higher values = better GPU utilization but more memory.
EXPANDED_BATCH_TARGET = 96

# Optional: specify layers to test (None = all layers from directions file)
LAYERS = None  # e.g., [20, 25, 30] for quick testing

# Optional: specify which direction methods to test (None = all available)
METHODS = ["mean_diff"]  # e.g., ["mean_diff"] or ["probe"] to test just one

# Token positions within the meta-task prompt where we ablate
PROBE_POSITIONS = ["final"]  # ["question_mark", "question_newline", "options_newline", "final"]

# Extra diagnostics
PRINT_DELTA_DIAGNOSTICS = True
DELTA_DIAGNOSTIC_TOPK = 5  # print this many layers for biggest +Δcorr and biggest -Δcorr

# Bootstrap (over questions) for correlation CIs + bootstrap p-values (BH-FDR across layers)
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_CI_ALPHA = 0.05  # 95% CI

# Layer selection from transfer results (for non-final positions)
TRANSFER_R2_THRESHOLD = 0.3  # Layers with R² >= this are tested for non-final positions
TRANSFER_RESULTS_PATH = None  # Auto-detect from MODEL/DATASET if None

# Control count for non-final positions (final uses NUM_CONTROLS)
NUM_CONTROLS_NONFINAL = 10

# --- Output ---
# Uses centralized path management from core.config_utils

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# TRANSFER RESULTS LOADING (for layer selection)
# =============================================================================

def load_transfer_results(base_name: str, meta_task: str, model_dir: str) -> Optional[Dict]:
    """
    Load transfer results JSON to get per-layer R² values.

    Returns None if file not found.
    """
    path = TRANSFER_RESULTS_PATH
    if path is None:
        path = find_output_file(f"{base_name}_meta_{meta_task}_transfer_results_{PROBE_POSITION}.json", model_dir=model_dir)
    else:
        path = Path(path)

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_layers_from_transfer(
    transfer_data: Dict,
    metric: str,
    position: str,
    r2_threshold: float,
    method: str = "probe",
) -> List[int]:
    """
    Get layers with transfer R² >= threshold for a given metric and position.

    Args:
        transfer_data: Loaded transfer results JSON
        metric: Which metric to check (e.g., "top_logit", "entropy")
        position: Token position (e.g., "final", "question_mark")
        r2_threshold: Minimum R² to include layer
        method: Direction method - "probe" uses transfer_by_position, "mean_diff" uses mean_diff_by_position

    Returns:
        Sorted list of layer indices meeting threshold
    """
    # Select the appropriate section based on method
    if method == "mean_diff":
        section_key = "mean_diff_by_position"
        legacy_key = None  # No legacy fallback for mean_diff
    else:
        section_key = "transfer_by_position"
        legacy_key = "transfer"

    # Try position-specific data first
    if section_key in transfer_data and position in transfer_data[section_key]:
        pos_data = transfer_data[section_key][position]
    elif legacy_key and legacy_key in transfer_data:
        # Fall back to legacy format (final position only, probe only)
        pos_data = transfer_data[legacy_key]
    else:
        return []

    if metric not in pos_data:
        return []

    metric_data = pos_data[metric]
    per_layer = metric_data.get("per_layer", {})

    selected = []
    for layer_str, layer_data in per_layer.items():
        # Check for centered R² (preferred) or d2m_centered_r2 (legacy)
        r2 = layer_data.get("centered_r2") or layer_data.get("d2m_centered_r2", 0)
        if r2 >= r2_threshold:
            selected.append(int(layer_str))

    return sorted(selected)


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def load_directions(
    base_name: str,
    direction_type: str = "uncertainty",
    metric: str = "entropy",
    meta_task: str = "delegate",
    model_dir: str = None
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load direction vectors based on direction type.

    Args:
        base_name: Base name for input files (dataset name)
        direction_type: "uncertainty", "answer", "confidence", or "metamcuncert"
        metric: Uncertainty metric (only used for direction_type="uncertainty")
        meta_task: Meta task (only used for direction_type="confidence" or "metamcuncert")
        model_dir: Model directory name

    Returns:
        Dict mapping method name -> {layer: direction_vector}
        For uncertainty: {"probe": {...}, "mean_diff": {...}}
        For answer: {"answer": {...}}
        For confidence/metamcuncert: {"probe": {...}, "mean_diff": {...}}
    """
    if direction_type == "uncertainty":
        path = find_output_file(f"{base_name}_mc_{metric}_directions.npz", model_dir=model_dir)
    elif direction_type == "answer":
        path = find_output_file(f"{base_name}_mc_answer_directions.npz", model_dir=model_dir)
    elif direction_type == "confidence":
        path = find_output_file(f"{base_name}_meta_{meta_task}_confdir_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
    elif direction_type == "metamcuncert":
        # Consolidated file with keys like probe_{metric}_layer_0
        path = find_output_file(f"{base_name}_meta_{meta_task}_mcuncert_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
    else:
        raise ValueError(f"Unknown direction type: {direction_type}")

    if not path.exists():
        raise FileNotFoundError(f"Directions file not found: {path}")

    data = np.load(path)

    methods: Dict[str, Dict[int, np.ndarray]] = {}

    if direction_type == "uncertainty":
        # Keys are like "probe_layer_0", "mean_diff_layer_5"
        for key in data.files:
            if key.startswith("_"):
                continue  # Skip metadata keys

            parts = key.rsplit("_layer_", 1)
            if len(parts) != 2:
                continue

            method, layer_str = parts
            try:
                layer = int(layer_str)
            except ValueError:
                continue

            if method not in methods:
                methods[method] = {}

            # Normalize direction
            direction = data[key].astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            methods[method][layer] = direction

    elif direction_type == "answer":
        # Keys are like "classifier_layer_0", "centroid_layer_5"
        # (matches uncertainty direction naming convention)
        for key in data.files:
            if key.startswith("_"):
                continue

            # Handle new format: "classifier_layer_0", "centroid_layer_5"
            parts = key.rsplit("_layer_", 1)
            if len(parts) == 2:
                method, layer_str = parts
                try:
                    layer = int(layer_str)
                except ValueError:
                    continue

                if method not in methods:
                    methods[method] = {}

                direction = data[key].astype(np.float32)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                methods[method][layer] = direction

            # Also handle legacy format: "layer_0"
            elif key.startswith("layer_"):
                try:
                    layer = int(key.replace("layer_", ""))
                except ValueError:
                    continue

                if "answer" not in methods:
                    methods["answer"] = {}

                direction = data[key].astype(np.float32)
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                methods["answer"][layer] = direction

    elif direction_type == "confidence":
        # Keys are like "probe_layer_0", "mean_diff_layer_5"
        for key in data.files:
            if key.startswith("_"):
                continue  # Skip metadata keys

            parts = key.rsplit("_layer_", 1)
            if len(parts) != 2:
                continue

            method_name, layer_str = parts
            try:
                layer = int(layer_str)
            except ValueError:
                continue

            direction = data[key].astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            if method_name not in methods:
                methods[method_name] = {}
            methods[method_name][layer] = direction

    elif direction_type == "metamcuncert":
        # Consolidated file with keys like "probe_{metric}_layer_0", "mean_diff_{metric}_layer_5"
        # Filter by the requested metric
        for key in data.files:
            if key.startswith("_"):
                continue  # Skip metadata keys

            # Check if this key is for the requested metric
            # Keys are like "probe_entropy_layer_0" or "mean_diff_logit_gap_layer_5"
            parts = key.rsplit("_layer_", 1)
            if len(parts) != 2:
                continue

            method_metric, layer_str = parts
            try:
                layer = int(layer_str)
            except ValueError:
                continue

            # Parse method and metric from "probe_entropy" or "mean_diff_logit_gap"
            if method_metric.startswith("probe_"):
                method_name = "probe"
                key_metric = method_metric[6:]  # Remove "probe_"
            elif method_metric.startswith("mean_diff_"):
                method_name = "mean_diff"
                key_metric = method_metric[10:]  # Remove "mean_diff_"
            else:
                continue

            # Only include if metric matches
            if key_metric != metric:
                continue

            direction = data[key].astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            if method_name not in methods:
                methods[method_name] = {}
            methods[method_name][layer] = direction

    return methods


def load_dataset(base_name: str, model_dir: str) -> Dict:
    """Load consolidated mc_results.json with questions and metric values."""
    path = find_output_file(f"{base_name}_mc_results.json", model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)
    # Return nested dataset section for compatibility
    return data["dataset"]


# =============================================================================
# META-TASK HELPERS
# =============================================================================

def get_format_fn(meta_task: str):
    """Get prompt formatting function for meta-task."""
    if meta_task == "confidence":
        return format_stated_confidence_prompt
    elif meta_task == "delegate":
        return format_answer_or_delegate_prompt
    elif meta_task == "other_confidence":
        return format_other_confidence_prompt
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_signal_fn(meta_task: str):
    """Get signal extraction function for meta-task.

    Returns a function with signature (probs, mapping) -> float.
    For confidence/other_confidence tasks, mapping is ignored.
    """
    if meta_task == "confidence":
        # Wrap to match (probs, mapping) signature
        return lambda p, m: get_stated_confidence_signal(p)
    elif meta_task == "delegate":
        return get_answer_or_delegate_signal
    elif meta_task == "other_confidence":
        return lambda p, m: get_other_confidence_signal(p)
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


def get_options(meta_task: str) -> List[str]:
    """Get response options for meta-task."""
    if meta_task == "confidence":
        return list(STATED_CONFIDENCE_OPTIONS.keys())
    elif meta_task == "delegate":
        return ANSWER_OR_DELEGATE_OPTIONS
    elif meta_task == "other_confidence":
        return list(OTHER_CONFIDENCE_OPTIONS.keys())
    else:
        raise ValueError(f"Unknown meta_task: {meta_task}")


# =============================================================================
# ABLATION EXPERIMENT
# =============================================================================


# -----------------------------------------------------------------------------
# Confidence signal helpers
# -----------------------------------------------------------------------------
def _extract_probs_logits(out, option_token_ids):
    """Return (probs, logits_np) over the option tokens at the final position."""
    logits = out.logits[:, -1, :][:, option_token_ids]
    logits_np = logits.detach().float().cpu().numpy()
    probs = torch.softmax(logits, dim=-1).detach().float().cpu().numpy()
    return probs, logits_np

def _compute_confidence_used(meta_task: str, probs_row, logits_row, mapping, signal_fn):
    """Return (confidence_used, p_answer, logit_margin)."""
    if meta_task == "delegate":
        # mapping maps "1"/"2" -> "Answer"/"Delegate"
        ans_idx = 0 if mapping.get("1") == "Answer" else 1
        del_idx = 1 - ans_idx
        p_answer = float(probs_row[ans_idx])
        logit_margin = float(logits_row[ans_idx] - logits_row[del_idx])
        sig = str(CONFIDENCE_SIGNAL).lower()
        if sig in {"logit_margin", "margin", "logitdiff", "logit_diff"}:
            return logit_margin, p_answer, logit_margin
        return p_answer, p_answer, logit_margin
    # confidence task: keep the original probability-based signal
    if str(CONFIDENCE_SIGNAL).lower() in {"logit_margin", "margin", "logitdiff", "logit_diff"}:
        # be explicit to avoid silent confusion
        import warnings
        warnings.warn("CONFIDENCE_SIGNAL=logit_margin is only defined for META_TASK=delegate; falling back to prob.")
    conf = float(signal_fn(probs_row, mapping))
    return conf, None, None

def run_ablation_for_method(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    directions: Dict[int, np.ndarray],
    num_controls: int,
    meta_task: str,
    use_chat_template: bool,
    layers: Optional[List[int]] = None,
    position: str = "final",
    original_indices: Optional[np.ndarray] = None,
) -> Dict:
    """
    Run ablation experiment for a single direction method at a specific position.

    Uses batched ablation when EXPANDED_BATCH_TARGET is set: multiple directions
    are ablated in a single forward pass by expanding the batch.

    For position="final", uses KV cache optimization.
    For other positions, uses full forward passes with indexed ablation.

    Args:
        position: Token position to ablate at. One of PROBE_POSITIONS:
            - "final": Last token (uses KV cache)
            - "question_mark": Token after "?" in question
            - "question_newline": Newline after question
            - "options_newline": Newline after MC options
        original_indices: Original dataset indices for each question. Used for
            trial_index in delegate task to match prompt formatting with
            test_meta_transfer.py. If None, uses local indices (legacy behavior).

    Returns dict with per-layer results including baseline, ablated, and controls.
    """
    if layers is None:
        layers = sorted(directions.keys())
    else:
        layers = [l for l in layers if l in directions]

    if not layers:
        return {"error": "No layers to test"}

    metric_mean = float(np.mean(metric_values))
    metric_std = float(np.std(metric_values))
    if metric_std < 1e-10:
        metric_std = 1.0

    # Get formatting functions and options
    format_fn = get_format_fn(meta_task)
    signal_fn = get_signal_fn(meta_task)
    options = get_options(meta_task)

    # Tokenize options
    option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
    ]

    # Format prompts and find token positions
    prompts = []
    mappings = []
    position_indices = []  # Per-prompt token index for intervention
    for q_idx, question in enumerate(questions):
        # Use original dataset index for trial_index to match test_meta_transfer.py
        # This ensures the delegate task uses consistent Answer/Delegate mapping
        trial_idx = int(original_indices[q_idx]) if original_indices is not None else q_idx
        if meta_task == "delegate":
            prompt, _, mapping = format_fn(question, tokenizer, trial_index=trial_idx, use_chat_template=use_chat_template)
        else:
            prompt, _ = format_fn(question, tokenizer, use_chat_template=use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

        # Find token positions for this prompt
        positions = find_mc_positions(prompt, tokenizer, question)
        pos_idx = positions.get(position, -1)
        position_indices.append(pos_idx)

    # Warn if some positions weren't found (will fall back to final token)
    # Note: "final" position is always -1 by design, so don't warn for it
    if position != "final":
        n_valid = sum(1 for idx in position_indices if idx >= 0)
        n_total = len(position_indices)
        if n_valid < n_total:
            print(f"  Warning: {position} position found for {n_valid}/{n_total} prompts (others fall back to final)")

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Check if we can use KV cache (only for "final" position)
    use_kv_cache = (position == "final")

    # Generate control directions for each layer
    print(f"  Generating {num_controls} control directions per layer...")
    controls_by_layer = {}
    for layer in layers:
        controls_by_layer[layer] = generate_orthogonal_directions(
            directions[layer], num_controls, seed=SEED + layer
        )

    # Precompute direction tensors
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = {}
    for layer in layers:
        dir_tensor = torch.tensor(directions[layer], dtype=dtype, device=DEVICE)
        ctrl_tensors = [torch.tensor(c, dtype=dtype, device=DEVICE) for c in controls_by_layer[layer]]
        # Stack all directions: [primary, control_0, control_1, ..., control_N-1]
        all_dirs = torch.stack([dir_tensor] + ctrl_tensors, dim=0)  # (1 + num_controls, hidden_dim)
        cached_directions[layer] = {
            "direction": dir_tensor,
            "controls": ctrl_tensors,
            "all_stacked": all_dirs,
        }

    # Initialize results
    baseline_results = [None] * len(questions)
    layer_results = {}
    for layer in layers:
        layer_results[layer] = {
            "baseline": baseline_results,
            "ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    # Determine batching strategy
    total_directions = 1 + num_controls  # primary + controls
    if EXPANDED_BATCH_TARGET is not None and EXPANDED_BATCH_TARGET > 0:
        directions_per_pass = max(1, EXPANDED_BATCH_TARGET // BATCH_SIZE)
        directions_per_pass = min(directions_per_pass, total_directions)
        use_batched = directions_per_pass > 1
    else:
        directions_per_pass = 1
        use_batched = False

    # Calculate number of passes (same formula for both paths)
    num_passes = (total_directions + directions_per_pass - 1) // directions_per_pass if use_batched else total_directions

    if use_kv_cache:
        # KV cache path: efficient but only works for final position
        if use_batched:
            print(f"  Batched ablation (KV cache): {directions_per_pass} directions per pass, {num_passes} passes per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (KV cache): 1 direction per pass")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions
    else:
        # Full forward path: required for non-final positions (also supports batching)
        if use_batched:
            print(f"  Batched ablation (full forward) at '{position}': {directions_per_pass} dirs/pass, {num_passes} passes/layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (full forward) at '{position}': {total_directions} directions per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions

    print(f"  Total forward passes: {total_forward_passes}")

    pbar = tqdm(total=total_forward_passes, desc=f"  Ablation ({position})")

    for batch_idx, (batch_indices, batch_inputs) in enumerate(gpu_batches):
        B = len(batch_indices)

        if use_kv_cache:
            # KV cache path (position == "final")
            base_step_data = get_kv_cache(model, batch_inputs)
            keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

            inputs_template = {
                "input_ids": base_step_data["input_ids"],
                "attention_mask": base_step_data["attention_mask"],
                "use_cache": True
            }
            if "position_ids" in base_step_data:
                inputs_template["position_ids"] = base_step_data["position_ids"]

            # Compute baseline (no ablation)
            if baseline_results[batch_indices[0]] is None:
                fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                baseline_inputs = inputs_template.copy()
                baseline_inputs["past_key_values"] = fresh_cache

                with torch.inference_mode():
                    out = model(**baseline_inputs)
                    probs, logits_np = _extract_probs_logits(out, option_token_ids)

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                    m_val = metric_values[q_idx]
                    baseline_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": float(conf),
                        "metric": float(m_val),
                        "p_answer": (float(p_answer) if p_answer is not None else None),
                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                    }

            # Run ablation for each layer (KV cache path)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                all_dirs = cached_directions[layer]["all_stacked"]
                hook = BatchAblationHook()
                hook.register(layer_module)

                try:
                    if use_batched:
                        for pass_start in range(0, total_directions, directions_per_pass):
                            pass_end = min(pass_start + directions_per_pass, total_directions)
                            k_dirs = pass_end - pass_start

                            expanded_input_ids = inputs_template["input_ids"].repeat_interleave(k_dirs, dim=0)
                            expanded_attention_mask = inputs_template["attention_mask"].repeat_interleave(k_dirs, dim=0)
                            expanded_inputs = {
                                "input_ids": expanded_input_ids,
                                "attention_mask": expanded_attention_mask,
                                "use_cache": True
                            }
                            if "position_ids" in inputs_template:
                                expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(k_dirs, dim=0)

                            pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_dirs)
                            expanded_inputs["past_key_values"] = pass_cache

                            dirs_for_pass = all_dirs[pass_start:pass_end]
                            dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**expanded_inputs)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
                                    prob_idx = i * k_dirs + j
                                    p = probs[prob_idx]
                                    resp = options[np.argmax(p)]
                                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[prob_idx], mappings[q_idx], signal_fn)
                                    m_val = metric_values[q_idx]
                                    data = {
                                        "question_idx": q_idx,
                                        "response": resp,
                                        "confidence": float(conf),
                                        "metric": float(m_val),
                                        "p_answer": (float(p_answer) if p_answer is not None else None),
                                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                    }
                                    if dir_idx == 0:
                                        layer_results[layer]["ablated"][q_idx] = data
                                    else:
                                        ctrl_key = f"control_{dir_idx - 1}"
                                        layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data

                            pbar.update(1)
                    else:
                        # Sequential KV cache path
                        def run_single_ablation_kv(direction_tensor, result_list, key=None):
                            pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                            current_inputs = inputs_template.copy()
                            current_inputs["past_key_values"] = pass_cache

                            dirs_batch = direction_tensor.unsqueeze(0).expand(B, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**current_inputs)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                p = probs[i]
                                resp = options[np.argmax(p)]
                                conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                                m_val = metric_values[q_idx]
                                data = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                    "p_answer": (float(p_answer) if p_answer is not None else None),
                                    "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                }
                                if key:
                                    result_list[key][q_idx] = data
                                else:
                                    result_list[q_idx] = data
                            pbar.update(1)

                        run_single_ablation_kv(cached_directions[layer]["direction"], layer_results[layer]["ablated"])
                        for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                            run_single_ablation_kv(ctrl_dir, layer_results[layer]["controls_ablated"], key=f"control_{i_c}")
                finally:
                    hook.remove()

        else:
            # Full forward path (position != "final")
            # Build position indices for this batch (adjusted for left-padding)
            batch_pos_indices = []
            seq_len = batch_inputs["input_ids"].shape[1]
            for i, q_idx in enumerate(batch_indices):
                pos = position_indices[q_idx]
                if pos >= 0:
                    # Adjust for left-padding
                    actual_len = int(batch_inputs["attention_mask"][i].sum())
                    pad_offset = seq_len - actual_len
                    adjusted_pos = pos + pad_offset
                else:
                    adjusted_pos = seq_len - 1  # fallback to final
                batch_pos_indices.append(adjusted_pos)
            batch_pos_tensor = torch.tensor(batch_pos_indices, dtype=torch.long, device=DEVICE)

            # Compute baseline (no ablation) - full forward
            if baseline_results[batch_indices[0]] is None:
                with torch.inference_mode():
                    out = model(**batch_inputs, use_cache=False)
                    probs, logits_np = _extract_probs_logits(out, option_token_ids)

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                    m_val = metric_values[q_idx]
                    baseline_results[q_idx] = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": float(conf),
                        "metric": float(m_val),
                        "p_answer": (float(p_answer) if p_answer is not None else None),
                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                    }

            # Run ablation for each layer (full forward path with batched directions)
            for layer in layers:
                if hasattr(model, 'get_base_model'):
                    layer_module = model.get_base_model().model.layers[layer]
                else:
                    layer_module = model.model.layers[layer]

                all_dirs = cached_directions[layer]["all_stacked"]

                if use_batched:
                    # Batched ablation: expand batch by k_dirs directions per pass
                    for pass_start in range(0, total_directions, directions_per_pass):
                        pass_end = min(pass_start + directions_per_pass, total_directions)
                        k_dirs = pass_end - pass_start

                        # Expand inputs by k_dirs
                        expanded_input_ids = batch_inputs["input_ids"].repeat_interleave(k_dirs, dim=0)
                        expanded_attention_mask = batch_inputs["attention_mask"].repeat_interleave(k_dirs, dim=0)
                        expanded_inputs = {
                            "input_ids": expanded_input_ids,
                            "attention_mask": expanded_attention_mask,
                        }

                        # Expand position indices to match expanded batch
                        expanded_pos_tensor = batch_pos_tensor.repeat_interleave(k_dirs)

                        # Build direction tensor: (B * k_dirs, hidden_dim)
                        dirs_for_pass = all_dirs[pass_start:pass_end]
                        dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1).reshape(B * k_dirs, -1)

                        hook = BatchAblationHook(intervention_position="indexed")
                        hook.set_position_indices(expanded_pos_tensor)
                        hook.set_directions(dirs_batch)
                        hook.register(layer_module)

                        try:
                            with torch.inference_mode():
                                out = model(**expanded_inputs, use_cache=False)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            # Store results
                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
                                    prob_idx = i * k_dirs + j
                                    p = probs[prob_idx]
                                    resp = options[np.argmax(p)]
                                    conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[prob_idx], mappings[q_idx], signal_fn)
                                    m_val = metric_values[q_idx]
                                    data = {
                                        "question_idx": q_idx,
                                        "response": resp,
                                        "confidence": float(conf),
                                        "metric": float(m_val),
                                        "p_answer": (float(p_answer) if p_answer is not None else None),
                                        "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                    }
                                    if dir_idx == 0:
                                        layer_results[layer]["ablated"][q_idx] = data
                                    else:
                                        ctrl_key = f"control_{dir_idx - 1}"
                                        layer_results[layer]["controls_ablated"][ctrl_key][q_idx] = data
                        finally:
                            hook.remove()

                        pbar.update(1)
                else:
                    # Sequential ablation (one direction per pass)
                    hook = BatchAblationHook(intervention_position="indexed")
                    hook.set_position_indices(batch_pos_tensor)
                    hook.register(layer_module)

                    try:
                        # Primary direction
                        dirs_batch = cached_directions[layer]["direction"].unsqueeze(0).expand(B, -1)
                        hook.set_directions(dirs_batch)

                        with torch.inference_mode():
                            out = model(**batch_inputs, use_cache=False)
                            probs, logits_np = _extract_probs_logits(out, option_token_ids)

                        for i, q_idx in enumerate(batch_indices):
                            p = probs[i]
                            resp = options[np.argmax(p)]
                            conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                            m_val = metric_values[q_idx]
                            layer_results[layer]["ablated"][q_idx] = {
                                "question_idx": q_idx,
                                "response": resp,
                                "confidence": float(conf),
                                "metric": float(m_val),
                                "p_answer": (float(p_answer) if p_answer is not None else None),
                                "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                            }
                        pbar.update(1)

                        # Control directions
                        for i_c, ctrl_dir in enumerate(cached_directions[layer]["controls"]):
                            dirs_batch = ctrl_dir.unsqueeze(0).expand(B, -1)
                            hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**batch_inputs, use_cache=False)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                p = probs[i]
                                resp = options[np.argmax(p)]
                                conf, p_answer, logit_margin = _compute_confidence_used(meta_task, p, logits_np[i], mappings[q_idx], signal_fn)
                                m_val = metric_values[q_idx]
                                layer_results[layer]["controls_ablated"][f"control_{i_c}"][q_idx] = {
                                    "question_idx": q_idx,
                                    "response": resp,
                                    "confidence": float(conf),
                                    "metric": float(m_val),
                                    "p_answer": (float(p_answer) if p_answer is not None else None),
                                    "logit_margin": (float(logit_margin) if logit_margin is not None else None),
                                }
                            pbar.update(1)
                    finally:
                        hook.remove()

    pbar.close()
    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results,
        "position": position,
    }


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation between confidence and metric."""
    if len(confidences) < 2 or np.std(confidences) < 1e-10 or np.std(metric_values) < 1e-10:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def compute_spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman (rank) correlation."""
    if len(x) < 2 or np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0
    return float(spearmanr(x, y).correlation)



def _bh_fdr(pvals_by_layer: Dict[int, float]) -> Dict[int, float]:
    """Benjamini-Hochberg FDR correction.

    Args:
        pvals_by_layer: mapping layer->raw p

    Returns:
        mapping layer->FDR-adjusted p
    """
    items = sorted(pvals_by_layer.items(), key=lambda kv: kv[1])
    n = len(items)
    if n == 0:
        return {}

    adj = {}
    for rank, (layer, p) in enumerate(items, 1):
        adj[layer] = min(1.0, (p * n) / rank)

    # enforce monotonicity (non-decreasing in the sorted order)
    prev = 0.0
    for layer, p in items:
        if adj[layer] < prev:
            adj[layer] = prev
        prev = adj[layer]
    return adj


def _bootstrap_corr(x: np.ndarray, y: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Vectorized bootstrap Pearson correlation for many resamples.

    x, y are shape (n,). idx is shape (B, n) integer indices.

    Returns: shape (B,) correlations (0.0 where variance degenerates).
    """
    n = x.shape[0]
    if n < 2:
        return np.zeros(idx.shape[0], dtype=np.float32)

    X = x[idx]
    Y = y[idx]

    # center
    Xc = X - X.mean(axis=1, keepdims=True)
    Yc = Y - Y.mean(axis=1, keepdims=True)

    denom_n = float(n - 1)
    cov = (Xc * Yc).sum(axis=1) / denom_n
    sx = np.sqrt((Xc * Xc).sum(axis=1) / denom_n)
    sy = np.sqrt((Yc * Yc).sum(axis=1) / denom_n)

    denom = sx * sy
    out = np.zeros_like(cov, dtype=np.float32)
    ok = denom > 1e-12
    out[ok] = (cov[ok] / denom[ok]).astype(np.float32)
    return out


def analyze_ablation_results(results: Dict, metric: str, base_name: str) -> Dict:
    """Compute ablation effect statistics.

    Adds **bootstrap CIs + bootstrap BH-FDR** over questions (cheap, no extra model runs),
    and (when controls exist) retains the existing pooled-control null stats.
    """
    layers = results.get("layers", [])
    num_controls = results.get("num_controls", 0)

    metric_sign = metric_sign_for_confidence(metric)

    # Determine quantization string
    quant_str = "4bit" if LOAD_IN_4BIT else ("8bit" if LOAD_IN_8BIT else "none")

    # Extract model short name for metadata
    model_short = get_model_short_name(MODEL, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)
    dataset_name = base_name  # Already just the dataset name

    analysis = {
        "confidence_signal": results.get("confidence_signal", CONFIDENCE_SIGNAL),
        "layers": layers,
        "num_questions": results.get("num_questions", 0),
        "num_controls": num_controls,
        "metric": metric,
        "metric_sign": metric_sign,
        "bootstrap": {
            "n": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "ci_alpha": BOOTSTRAP_CI_ALPHA,
        },
        "per_layer": {},
        # Metadata for reproducibility
        "direction_type": DIRECTION_TYPE,
        "model_name": MODEL.split("/")[-1],  # Just the model name, not full path
        "dataset": dataset_name,
        "quantization": quant_str,
    }

    if not layers:
        analysis["summary"] = {
            "pooled_null_size": 0,
            "n_significant_fdr": 0,
            "n_significant_bootstrap_fdr": 0,
            "best_layer": None,
            "best_effect_z": 0.0,
            "best_abs_delta": 0.0,
        }
        return analysis

    # --- Pull baseline arrays once (baseline is identical across layers for a given run) ---
    first_layer = layers[0]
    lr0 = results["layer_results"][first_layer]
    baseline_conf = np.array([r["confidence"] for r in lr0["baseline"]], dtype=np.float32)
    baseline_metric = np.array([r["metric"] for r in lr0["baseline"]], dtype=np.float32)

    # Baseline point estimate
    baseline_corr_point = compute_correlation(baseline_conf, baseline_metric)

    # Bootstrap index matrix (shared across layers)
    n_q = baseline_conf.shape[0]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n_q, size=(BOOTSTRAP_N, n_q), dtype=np.int32)

    # Bootstrap baseline corr (shared)
    boot_base = _bootstrap_corr(baseline_conf, baseline_metric, idx)
    lo = BOOTSTRAP_CI_ALPHA / 2.0
    hi = 1.0 - lo
    base_ci = np.quantile(boot_base, [lo, hi]).astype(np.float32)

    # We'll also need signed metric for Δconf diagnostics
    metric_signed = baseline_metric * float(metric_sign)

    # --- If controls exist, build pooled null of corr changes ---
    pooled_null = []

    # First pass: compute per-layer stats and collect pooled null
    layer_data = {}

    for layer in layers:
        lr = results["layer_results"][layer]

        ablated_conf = np.array([r["confidence"] for r in lr["ablated"]], dtype=np.float32)
        ablated_metric = np.array([r["metric"] for r in lr["ablated"]], dtype=np.float32)

        # Point estimates
        baseline_corr = baseline_corr_point
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)
        corr_change = ablated_corr - baseline_corr

        # --- Bootstrap CIs (sampling uncertainty) ---
        boot_ablt = _bootstrap_corr(ablated_conf, ablated_metric, idx)
        boot_delta = boot_ablt - boot_base

        ablt_ci = np.quantile(boot_ablt, [lo, hi]).astype(np.float32)
        delta_ci = np.quantile(boot_delta, [lo, hi]).astype(np.float32)

        # Two-sided "bootstrap sign" p-value for Δcorr != 0
        # (Equivalent to whether 0 is in the bootstrap distribution mass.)
        frac_ge0 = float(np.mean(boot_delta >= 0.0))
        frac_le0 = float(np.mean(boot_delta <= 0.0))
        p_boot = float(min(1.0, 2.0 * min(frac_ge0, frac_le0)))

        # --- Control ablations (null based on random orthogonal directions) ---
        control_corrs = []
        control_corr_changes = []
        control_delta_corrs = []

        if num_controls > 0 and lr.get("controls_ablated"):
            for ctrl_key, ctrl_list in lr["controls_ablated"].items():
                ctrl_conf = np.array([r["confidence"] for r in ctrl_list], dtype=np.float32)
                ctrl_metric = np.array([r["metric"] for r in ctrl_list], dtype=np.float32)
                c_corr = compute_correlation(ctrl_conf, ctrl_metric)
                control_corrs.append(c_corr)
                control_corr_changes.append(c_corr - baseline_corr)

                # Δconf diagnostics: corr(Δconf, signed metric)
                delta_ctrl = ctrl_conf - baseline_conf
                control_delta_corrs.append(compute_correlation(delta_ctrl, metric_signed))

            pooled_null.extend(control_corr_changes)

        # --- Δconf diagnostics (primary) ---
        delta_conf = ablated_conf - baseline_conf
        delta_conf_mean = float(np.mean(delta_conf))
        delta_conf_std = float(np.std(delta_conf))

        delta_corr_metric = compute_correlation(delta_conf, metric_signed)
        delta_spearman_metric = compute_spearman(delta_conf, metric_signed)

        if np.std(baseline_conf) > 1e-10:
            affine_slope, affine_intercept = np.polyfit(baseline_conf, ablated_conf, 1)
        else:
            affine_slope, affine_intercept = 0.0, float(np.mean(ablated_conf))

        baseline_to_ablated_corr = compute_correlation(baseline_conf, ablated_conf)
        resid = ablated_conf - (affine_slope * baseline_conf + affine_intercept)
        residual_corr_metric = compute_correlation(resid, metric_signed)

        pooled_delta_corr = np.array(control_delta_corrs, dtype=np.float32)
        if pooled_delta_corr.size > 0:
            n_worse = int(np.sum(np.abs(pooled_delta_corr) >= abs(delta_corr_metric)))
            p_value_delta_corr_pooled = float((n_worse + 1) / (pooled_delta_corr.size + 1))
            ctrl_delta_mean = float(np.mean(pooled_delta_corr))
            ctrl_delta_std = float(np.std(pooled_delta_corr))
        else:
            p_value_delta_corr_pooled = 1.0
            ctrl_delta_mean = 0.0
            ctrl_delta_std = 0.0

        # Mean Δconf by metric decile
        if np.std(metric_signed) < 1e-10:
            delta_by_decile = [None] * 10
        else:
            edges = np.quantile(metric_signed, np.linspace(0, 1, 11))
            if np.unique(edges).size < 3:
                delta_by_decile = [None] * 10
            else:
                bin_idx = np.digitize(metric_signed, edges[1:-1], right=True)  # 0..9
                delta_by_decile = [
                    float(np.mean(delta_conf[bin_idx == k])) if np.any(bin_idx == k) else None
                    for k in range(10)
                ]

        # Control summary stats (if any)
        if control_corrs:
            ctrl_corr_mean = float(np.mean(control_corrs))
            ctrl_corr_std = float(np.std(control_corrs))
            ctrl_change_mean = float(np.mean(control_corr_changes))
            ctrl_change_std = float(np.std(control_corr_changes))
        else:
            ctrl_corr_mean = baseline_corr
            ctrl_corr_std = 0.0
            ctrl_change_mean = 0.0
            ctrl_change_std = 0.0

        # Effect size vs controls (if any)
        if ctrl_change_std > 1e-10:
            effect_size_z = float((corr_change - ctrl_change_mean) / ctrl_change_std)
            p_value_parametric = float(2 * norm.sf(abs(effect_size_z)))
        else:
            effect_size_z = 0.0
            p_value_parametric = 1.0

        layer_data[layer] = {
            "baseline_corr": baseline_corr,
            "ablated_corr": ablated_corr,
            "corr_change": corr_change,

            # Bootstrap
            "baseline_corr_ci95": [float(base_ci[0]), float(base_ci[1])],
            "ablated_corr_ci95": [float(ablt_ci[0]), float(ablt_ci[1])],
            "delta_corr_ci95": [float(delta_ci[0]), float(delta_ci[1])],
            "p_value_bootstrap_delta": p_boot,

            # Confidence means
            "baseline_conf_mean": float(np.mean(baseline_conf)),
            "ablated_conf_mean": float(np.mean(ablated_conf)),

            # Controls
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
            "control_corr_mean": ctrl_corr_mean,
            "control_corr_std": ctrl_corr_std,
            "control_change_mean": ctrl_change_mean,
            "control_change_std": ctrl_change_std,
            "effect_size_z": float(effect_size_z),
            "p_value_parametric": float(p_value_parametric),

            # Δconf diagnostics
            "delta_conf_mean": delta_conf_mean,
            "delta_conf_std": delta_conf_std,
            "delta_conf_corr_metric": float(delta_corr_metric),
            "delta_conf_spearman_metric": float(delta_spearman_metric),
            "baseline_to_ablated_conf_corr": float(baseline_to_ablated_corr),
            "affine_slope": float(affine_slope),
            "affine_intercept": float(affine_intercept),
            "residual_corr_metric": float(residual_corr_metric),
            "control_delta_conf_corr_metric_mean": ctrl_delta_mean,
            "control_delta_conf_corr_metric_std": ctrl_delta_std,
            "p_value_delta_corr_pooled": float(p_value_delta_corr_pooled),
            "delta_conf_mean_by_metric_decile": delta_by_decile,
        }

    pooled_null = np.array(pooled_null, dtype=np.float32)

    # Second pass: p-values from pooled-null controls (if controls exist)
    raw_p_controls = {}
    for layer in layers:
        ld = layer_data[layer]
        if pooled_null.size > 0:
            n_worse = int(np.sum(np.abs(pooled_null) >= abs(ld["corr_change"])))
            p_val = float((n_worse + 1) / (pooled_null.size + 1))
        else:
            p_val = 1.0
        raw_p_controls[layer] = p_val

    fdr_controls = _bh_fdr(raw_p_controls)

    # Bootstrap BH-FDR
    raw_p_boot = {layer: layer_data[layer]["p_value_bootstrap_delta"] for layer in layers}
    fdr_boot = _bh_fdr(raw_p_boot)

    # Populate analysis[per_layer]
    for layer in layers:
        ld = layer_data[layer]
        analysis["per_layer"][layer] = {
            "baseline_correlation": ld["baseline_corr"],
            "ablated_correlation": ld["ablated_corr"],
            "correlation_change": ld["corr_change"],

            # Bootstrap
            "baseline_corr_ci95": ld["baseline_corr_ci95"],
            "ablated_corr_ci95": ld["ablated_corr_ci95"],
            "delta_corr_ci95": ld["delta_corr_ci95"],
            "p_value_bootstrap_delta": float(ld["p_value_bootstrap_delta"]),
            "p_value_bootstrap_fdr": float(fdr_boot.get(layer, 1.0)),

            # Controls
            "control_correlation_mean": float(ld["control_corr_mean"]),
            "control_correlation_std": float(ld["control_corr_std"]),
            "control_correlation_change_mean": float(ld["control_change_mean"]),
            "control_correlation_change_std": float(ld["control_change_std"]),
            "control_change_p2.5": float(np.percentile(ld["control_corr_changes"], 2.5)) if ld.get("control_corr_changes") else 0.0,
            "control_change_p97.5": float(np.percentile(ld["control_corr_changes"], 97.5)) if ld.get("control_corr_changes") else 0.0,
            "p_value_pooled": float(raw_p_controls[layer]),
            "p_value_fdr": float(fdr_controls.get(layer, 1.0)),
            "p_value_parametric": float(ld["p_value_parametric"]),
            "effect_size_z": float(ld["effect_size_z"]),

            # Δconf diagnostics
            "baseline_confidence_mean": ld["baseline_conf_mean"],
            "ablated_confidence_mean": ld["ablated_conf_mean"],
            "delta_conf_mean": ld["delta_conf_mean"],
            "delta_conf_std": ld["delta_conf_std"],
            "delta_conf_corr_metric": ld["delta_conf_corr_metric"],
            "delta_conf_spearman_metric": ld["delta_conf_spearman_metric"],
            "baseline_to_ablated_conf_corr": ld["baseline_to_ablated_conf_corr"],
            "affine_slope": ld["affine_slope"],
            "affine_intercept": ld["affine_intercept"],
            "residual_corr_metric": ld["residual_corr_metric"],
            "control_delta_conf_corr_metric_mean": ld["control_delta_conf_corr_metric_mean"],
            "control_delta_conf_corr_metric_std": ld["control_delta_conf_corr_metric_std"],
            "p_value_delta_corr_pooled": ld["p_value_delta_corr_pooled"],
            "delta_conf_mean_by_metric_decile": ld["delta_conf_mean_by_metric_decile"],
        }

    # Summary
    per = analysis["per_layer"]

    sig_controls_fdr = [l for l in layers if per[l]["p_value_fdr"] < 0.05]
    sig_boot_fdr = [l for l in layers if per[l]["p_value_bootstrap_fdr"] < 0.05]

    best_layer_z = max(layers, key=lambda l: abs(per[l]["effect_size_z"]))
    best_layer_abs_delta = max(layers, key=lambda l: abs(per[l]["correlation_change"]))

    analysis["summary"] = {
        "pooled_null_size": int(pooled_null.size),
        "significant_layers_fdr": sig_controls_fdr,
        "n_significant_fdr": len(sig_controls_fdr),
        "significant_layers_bootstrap_fdr": sig_boot_fdr,
        "n_significant_bootstrap_fdr": len(sig_boot_fdr),
        "best_layer": int(best_layer_z),
        "best_effect_z": float(per[best_layer_z]["effect_size_z"]),
        "best_layer_abs_delta": int(best_layer_abs_delta),
        "best_abs_delta": float(per[best_layer_abs_delta]["correlation_change"]),
    }

    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_ablation_results(analysis: Dict, method: str, output_path: Path):
    """Create 3-panel ablation visualization with actual values, delta, and summary.

    Panel 1 (top): Actual correlation values (baseline band + ablated line)
    Panel 2 (middle): Delta with controls (gray band + significance stars)
    Panel 3 (bottom): Summary statistics and interpretation
    """
    layers = analysis.get("layers", [])
    if not layers:
        print(f"  Skipping plot for {method} - no layers")
        return

    # Extract data
    per = analysis["per_layer"]
    real_delta = np.array([per[l]["correlation_change"] for l in layers], dtype=np.float32)
    ctrl_lo = np.array([per[l].get("control_change_p2.5", 0.0) for l in layers], dtype=np.float32)
    ctrl_hi = np.array([per[l].get("control_change_p97.5", 0.0) for l in layers], dtype=np.float32)
    delta_ci_lo = np.array([per[l]["delta_corr_ci95"][0] for l in layers], dtype=np.float32)
    delta_ci_hi = np.array([per[l]["delta_corr_ci95"][1] for l in layers], dtype=np.float32)
    p_fdr = np.array([per[l]["p_value_bootstrap_fdr"] for l in layers], dtype=np.float32)

    # Extract actual correlation values for Panel 1
    baseline_corr_arr = np.array([per[l]["baseline_correlation"] for l in layers], dtype=np.float32)
    ablated_corr_arr = np.array([per[l]["ablated_correlation"] for l in layers], dtype=np.float32)

    # Use paired CIs for Panel 1: ablated_ci = baseline + delta_ci
    # This makes Panel 1 and Panel 2 CIs statistically consistent
    baseline_val = float(baseline_corr_arr[0])  # constant across layers
    ablated_ci_lo_paired = baseline_val + delta_ci_lo
    ablated_ci_hi_paired = baseline_val + delta_ci_hi

    # Create figure with 3 vertically stacked panels
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 0.5], hspace=0.3)
    ax_actual = fig.add_subplot(gs[0])
    ax_delta = fig.add_subplot(gs[1])
    ax_summary = fig.add_subplot(gs[2])
    ax_summary.axis('off')

    x = np.arange(len(layers))

    # ===== Panel 1 (top): Actual correlation values =====
    # Baseline as horizontal line (no CI - it's the reference point)
    ax_actual.axhline(baseline_val, color=CONDITION_COLORS["baseline"],
                      linestyle='-', linewidth=1.5,
                      label=f'Baseline (r={baseline_val:.2f})')

    # Ablated correlation with paired CI band (derived from delta CI)
    ax_actual.fill_between(x, ablated_ci_lo_paired, ablated_ci_hi_paired,
                           color=CONDITION_COLORS["ablated"], alpha=CI_ALPHA)
    ax_actual.plot(x, ablated_corr_arr, 'o-', color=CONDITION_COLORS["ablated"],
                   markersize=4, linewidth=1.5, label='Ablated')

    ax_actual.set_xticks(x[::2])
    ax_actual.set_xticklabels([layers[i] for i in range(0, len(layers), 2)])
    ax_actual.set_xlabel('Layer')
    ax_actual.set_ylabel('Correlation (r)')
    ax_actual.set_title('Calibration: Baseline vs Ablated')
    ax_actual.legend(loc='lower left', fontsize=9)
    ax_actual.grid(True, alpha=GRID_ALPHA)

    # ===== Panel 2 (middle): Delta with controls =====
    # Control band (gray) - only plot if controls exist
    has_controls = np.any(ctrl_lo != 0) or np.any(ctrl_hi != 0)
    if has_controls:
        ax_delta.fill_between(x, ctrl_lo, ctrl_hi, color='gray', alpha=0.3,
                              label='Control 2.5-97.5%')
        # Add annotation explaining the tight control band
        ctrl_range = float(np.mean(ctrl_hi - ctrl_lo))
        ax_delta.annotate(f'Random directions: Δr ≈ 0 (95% within ±{ctrl_range/2:.3f})',
                          xy=(2, float(np.mean(ctrl_hi)) + 0.005),
                          fontsize=8, color='dimgray', style='italic')

    ax_delta.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # Real direction with CI band
    ax_delta.fill_between(x, delta_ci_lo, delta_ci_hi, color=CONDITION_COLORS["ablated"], alpha=CI_ALPHA)
    ax_delta.plot(x, real_delta, 'o-', color=CONDITION_COLORS["ablated"], markersize=4, linewidth=1.5,
                  label='Real direction')

    # Highlight significant layers
    sig_mask = p_fdr < 0.05
    if np.any(sig_mask):
        sig_x = x[sig_mask]
        sig_y = real_delta[sig_mask]
        ax_delta.scatter(sig_x, sig_y, color='gold', s=80, marker='*',
                         zorder=5, edgecolor='black', linewidth=0.5,
                         label='FDR < 0.05')

    # Annotation on peak effect (most negative)
    peak_idx = int(np.argmin(real_delta))
    peak_layer = layers[peak_idx]
    peak_val = float(real_delta[peak_idx])

    # Position annotation to avoid overlap
    text_x = peak_idx + 3 if peak_idx < len(layers) - 5 else peak_idx - 8
    text_y = peak_val - 0.015
    ax_delta.annotate(f'Layer {peak_layer}: Δr = {peak_val:.3f}',
                      xy=(peak_idx, peak_val), xytext=(text_x, text_y),
                      fontsize=9, arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='gray', alpha=0.9))

    ax_delta.set_xticks(x[::2])
    ax_delta.set_xticklabels([layers[i] for i in range(0, len(layers), 2)])
    ax_delta.set_xlabel('Layer')
    ax_delta.set_ylabel('Δ Correlation (ablated − baseline)')
    ax_delta.set_title('Ablation Effect vs Random Controls')
    ax_delta.legend(loc='lower left', fontsize=9)
    ax_delta.grid(True, alpha=GRID_ALPHA)

    # ===== Panel 3 (bottom): Summary statistics =====
    # Build summary statistics
    baseline_corr = float(np.mean(baseline_corr_arr))
    baseline_ci_lo_mean = float(np.mean([per[l]["baseline_corr_ci95"][0] for l in layers]))
    baseline_ci_hi_mean = float(np.mean([per[l]["baseline_corr_ci95"][1] for l in layers]))
    peak_ci = per[peak_layer]["delta_corr_ci95"]
    n_sig = int(np.sum(p_fdr < 0.05))
    ctrl_mean = float(np.mean([per[l]["control_correlation_change_mean"] for l in layers]))

    # Count layers where real effect is outside control band
    if has_controls:
        outside_band = int(np.sum((real_delta < ctrl_lo) | (real_delta > ctrl_hi)))
    else:
        outside_band = n_sig

    meta_task = analysis.get("meta_task", analysis.get("config", {}).get("meta_task", "unknown"))
    num_controls = analysis.get("num_controls", 0)
    bootstrap_n = analysis.get("bootstrap", {}).get("n", 0)
    conf_signal = analysis.get("confidence_signal", "prob")

    # Extract metadata for reproducibility
    model_name = analysis.get("model_name", "unknown")
    dataset = analysis.get("dataset", "unknown")
    quantization = analysis.get("quantization", "unknown")
    direction_type = analysis.get("direction_type", "uncertainty")

    # Extract position from output filename if available
    position = "unknown"
    fname = output_path.stem if output_path else ""
    for pos in ["final", "optionsnewline", "questionnewline", "questionmark"]:
        if pos in fname.lower().replace("_", ""):
            position = pos.replace("newline", "_newline").replace("mark", "_mark")
            break

    # Format quantization for display
    quant_display = f" ({quantization})" if quantization != "none" else ""

    # Horizontal summary format for full-width panel
    summary_text = (
        f"CAUSAL NECESSITY TEST | Model: {model_name}{quant_display} | Dataset: {dataset}\n"
        f"Direction: {method} {direction_type} ({analysis['metric']}) | Task: {meta_task} | Position: {position}\n"
        f"N = {analysis['num_questions']} | Controls = {num_controls} | Bootstrap = {bootstrap_n} | Signal = {conf_signal}\n\n"
        f"BASELINE: r = {baseline_corr:.3f} [{baseline_ci_lo_mean:.2f}, {baseline_ci_hi_mean:.2f}]    |    "
        f"PEAK EFFECT: Layer {peak_layer}, Δr = {peak_val:.3f} [{peak_ci[0]:.3f}, {peak_ci[1]:.3f}]\n"
        f"Significant: {n_sig}/{len(layers)} layers (FDR<0.05)    |    "
        f"Controls: mean Δr ≈ {ctrl_mean:.3f}, outside band: {outside_band} layers\n\n"
        f"INTERPRETATION: ✓ Ablating {direction_type} direction degrades calibration  "
        f"✓ Random directions do not  → Direction is causally necessary"
    )

    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes, fontsize=10,
                    verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor='gray', alpha=0.9))

    fig.suptitle(f'Ablation: {method.upper()} direction ({analysis["metric"]})',
                 fontsize=12, fontweight='bold')

    save_figure(fig, output_path)

def plot_method_comparison(analyses: Dict[str, Dict], output_path: Path):
    """Comparison plot of different direction methods.

    Shows Δcorr with bootstrap 95% CI bands and marks layers significant under bootstrap BH-FDR.
    """
    methods = list(analyses.keys())
    if len(methods) < 2:
        print("  Skipping comparison plot - need at least 2 methods")
        return

    layers = analyses[methods[0]].get("layers", [])
    if not layers:
        print("  Skipping comparison plot - no layers")
        return

    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    fig.suptitle("Method Comparison: Ablation Effects (Δcorr)", fontsize=14)

    x = np.arange(len(layers))
    method_colors = METHOD_COLORS

    # Panel 1: Δcorr with CI bands
    ax1 = axes[0]
    for method in methods:
        per = analyses[method]["per_layer"]
        delta = np.array([per[l]["correlation_change"] for l in layers], dtype=np.float32)
        d_lo = np.array([per[l]["delta_corr_ci95"][0] for l in layers], dtype=np.float32)
        d_hi = np.array([per[l]["delta_corr_ci95"][1] for l in layers], dtype=np.float32)
        p_fdr = np.array([per[l]["p_value_bootstrap_fdr"] for l in layers], dtype=np.float32)

        color = method_colors.get(method, "gray")
        ax1.plot(x, delta, "-", label=method, color=color, linewidth=1.8, alpha=0.85)
        ax1.fill_between(x, d_lo, d_hi, color=color, alpha=CI_ALPHA)

        # Mark significant layers
        sig_x = [i for i, p in enumerate(p_fdr) if p < 0.05]
        sig_y = [delta[i] for i in sig_x]
        if sig_x:
            ax1.scatter(sig_x, sig_y, color=color, s=45, zorder=5, edgecolor="black", linewidth=0.5)

    ax1.axhline(y=0, color="black", linestyle="-", linewidth=1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("ΔCorrelation (Ablated - Baseline)")
    ax1.set_title("Δcorr by Method (filled markers = bootstrap FDR<0.05)")
    ax1.legend()
    ax1.grid(True, alpha=GRID_ALPHA)

    # Panel 2: Summary
    ax2 = axes[1]
    ax2.axis("off")

    comparison_text = (
        "METHOD COMPARISON (bootstrap-FDR)\n"
        + "=" * 50
        + "\n\n"
    )
    for method in methods:
        summary = analyses[method].get("summary", {})
        comparison_text += f"{method.upper()}:\n"
        comparison_text += f"  Significant layers (boot FDR<0.05): {summary.get('n_significant_bootstrap_fdr', 0)}\n"
        comparison_text += (
            f"  Best |Δ| layer: {summary.get('best_layer_abs_delta')} "
            f"(Δ={summary.get('best_abs_delta', 0.0):+.3f})\n\n"
        )

    best_method = max(methods, key=lambda m: analyses[m].get("summary", {}).get("n_significant_bootstrap_fdr", 0))
    comparison_text += f"Method with more boot-FDR-significant layers: {best_method.upper()}\n"

    ax2.text(
        0.1,
        0.9,
        comparison_text,
        transform=ax2.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="gray", alpha=0.9),
    )

    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Model directory for organizing outputs
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET  # Dataset name (model prefix now in directory)

    # Setup output naming
    model_short = get_model_short_name(MODEL)

    config = {
        "model": MODEL.split("/")[-1],
        "dataset": DATASET,
        "task": META_TASK,
        "direction_type": DIRECTION_TYPE,
    }
    print_run_header("run_ablation_causality.py", 3, "Ablation necessity test", config)

    # Key findings for console output
    key_findings = {}
    output_files = []

    # Load directions based on direction type
    print("Loading directions...")
    all_directions = load_directions(
        base_name,
        direction_type=DIRECTION_TYPE,
        metric=METRIC,
        meta_task=META_TASK,
        model_dir=model_dir
    )
    available_methods = list(all_directions.keys())

    # Filter to requested methods (with name mapping for equivalent methods)
    # mean_diff and centroid are conceptually equivalent (difference of class means)
    METHOD_ALIASES = {"mean_diff": "centroid", "centroid": "mean_diff"}
    if METHODS is not None:
        methods = []
        for m in METHODS:
            if m in available_methods:
                methods.append(m)
            elif m in METHOD_ALIASES and METHOD_ALIASES[m] in available_methods:
                methods.append(METHOD_ALIASES[m])
        if not methods:
            raise ValueError(f"No matching methods found. Available: {available_methods}, requested: {METHODS}")
    else:
        methods = available_methods

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(base_name, model_dir)
    all_data = dataset["data"]

    if USE_TRANSFER_SPLIT:
        # Use same 80/20 split as transfer analysis for apples-to-apples comparison
        n_total = len(all_data)
        indices = np.arange(n_total)
        train_idx, test_idx = train_test_split(
            indices, train_size=TRAIN_SPLIT, random_state=SEED
        )
        data_items = [all_data[i] for i in test_idx]
        # Keep original indices for trial_index in delegate prompt formatting
        original_indices = test_idx
    else:
        # Legacy behavior: first NUM_QUESTIONS
        data_items = all_data[:NUM_QUESTIONS]
        # Original indices are just 0..NUM_QUESTIONS-1
        original_indices = np.arange(len(data_items))

    # Extract questions (each item has question, options, correct_answer, etc.)
    questions = data_items
    # Extract metric values from each item
    metric_values = np.array([item[METRIC] for item in data_items])

    # Load transfer results for layer selection (non-final positions)
    transfer_data = load_transfer_results(base_name, META_TASK, model_dir)

    # Determine base layers (all available)
    all_available_layers = sorted(all_directions[methods[0]].keys())

    # Load model
    print("Loading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)

    # Run ablation for each method and position
    # Structure: {position: {method: analysis}}
    all_results_by_pos = {pos: {} for pos in PROBE_POSITIONS}
    all_analyses_by_pos = {pos: {} for pos in PROBE_POSITIONS}

    for position in PROBE_POSITIONS:
        # Determine number of controls for this position
        position_num_controls = NUM_CONTROLS if position == "final" else NUM_CONTROLS_NONFINAL

        for method in methods:
            print(f"Running ablation: {method} @ {position}...")

            # Determine layers for this position AND method
            if LAYERS is not None:
                # Explicit override applies to all positions/methods
                method_layers = LAYERS
            elif position == "final":
                # Final position: use all layers
                method_layers = all_available_layers
            else:
                # Non-final position: select based on transfer R² for THIS method
                if transfer_data is not None:
                    method_layers = get_layers_from_transfer(
                        transfer_data, METRIC, position, TRANSFER_R2_THRESHOLD, method
                    )
                    if not method_layers:
                        # Fall back to "final" position transfer data if position-specific not available
                        method_layers = get_layers_from_transfer(
                            transfer_data, METRIC, "final", TRANSFER_R2_THRESHOLD, method
                        )
                else:
                    method_layers = all_available_layers

                if not method_layers:
                    print(f"  Warning: No layers meet R²≥{TRANSFER_R2_THRESHOLD} for {method}/{METRIC}, using all {len(all_available_layers)} layers")
                    method_layers = all_available_layers

            results = run_ablation_for_method(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                metric_values=metric_values,
                directions=all_directions[method],
                num_controls=position_num_controls,
                meta_task=META_TASK,
                use_chat_template=use_chat_template,
                layers=method_layers,
                position=position,
                original_indices=original_indices,
            )
            all_results_by_pos[position][method] = results

            # Analyze results
            analysis = analyze_ablation_results(results, METRIC, base_name)
            all_analyses_by_pos[position][method] = analysis

        # Incremental save after each position completes (crash protection) - one per method
        # Include direction type to distinguish uncertainty/answer/confidence/metamcuncert
        dir_suffix = f"{DIRECTION_TYPE}_{METRIC}" if DIRECTION_TYPE == "uncertainty" else DIRECTION_TYPE

        for method in methods:
            checkpoint_base = f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{method}"
            checkpoint_path = get_output_path(f"{checkpoint_base}_checkpoint.json", model_dir=model_dir, working=True)

            checkpoint_json = {
                "config": get_config_dict(
                    model=MODEL,
                    dataset=base_name,
                    model_dir=model_dir,
                    direction_type=DIRECTION_TYPE,
                    metric=METRIC,
                    meta_task=META_TASK,
                    confidence_signal=CONFIDENCE_SIGNAL,
                    num_questions=len(questions),
                    use_transfer_split=USE_TRANSFER_SPLIT,
                    seed=SEED,
                    load_in_4bit=LOAD_IN_4BIT,
                    load_in_8bit=LOAD_IN_8BIT,
                    method=method,
                    positions_completed=[p for p in PROBE_POSITIONS if all_analyses_by_pos[p]],
                ),
                "by_position": {},
            }
            for pos in PROBE_POSITIONS:
                if all_analyses_by_pos[pos] and method in all_analyses_by_pos[pos]:
                    analysis = all_analyses_by_pos[pos][method]
                    checkpoint_json["by_position"][pos] = {
                        "per_layer": analysis["per_layer"],
                        "summary": analysis["summary"],
                    }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_json, f, indent=2)

    # Generate output filename
    # Include direction type to distinguish uncertainty/answer/confidence/metamcuncert
    dir_suffix = f"{DIRECTION_TYPE}_{METRIC}" if DIRECTION_TYPE == "uncertainty" else DIRECTION_TYPE

    def get_base_output(method: str) -> str:
        """Get base output path for a specific method (without position - added per file)."""
        base = f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{method}"
        # Add confidence signal to filename when non-default (for delegate task)
        if META_TASK == "delegate" and CONFIDENCE_SIGNAL != "prob":
            base += f"_{CONFIDENCE_SIGNAL}"
        return base

    # Save JSON results - one file per method
    print("\nSaving results...")
    for method in methods:
        method_base_output = get_base_output(method)
        results_path = get_output_path(f"{method_base_output}_results.json", model_dir=model_dir)

        output_json = {
            "config": get_config_dict(
                model=MODEL,
                dataset=base_name,
                model_dir=model_dir,
                direction_type=DIRECTION_TYPE,
                metric=METRIC,
                meta_task=META_TASK,
                confidence_signal=CONFIDENCE_SIGNAL,
                num_questions=len(questions),
                use_transfer_split=USE_TRANSFER_SPLIT,
                seed=SEED,
                num_controls_final=NUM_CONTROLS,
                num_controls_nonfinal=NUM_CONTROLS_NONFINAL,
                transfer_r2_threshold=TRANSFER_R2_THRESHOLD,
                load_in_4bit=LOAD_IN_4BIT,
                load_in_8bit=LOAD_IN_8BIT,
                method=method,
                positions_tested=PROBE_POSITIONS,
            ),
            "by_position": {},
        }

        # Per-position results for this method
        for position in PROBE_POSITIONS:
            analysis = all_analyses_by_pos[position][method]
            output_json["by_position"][position] = {
                "layers": analysis["layers"],
                "num_questions": analysis["num_questions"],
                "num_controls": analysis["num_controls"],
                "metric": analysis["metric"],
                "per_layer": analysis["per_layer"],
                "summary": analysis["summary"],
            }

        # Backward compatibility: keep default position results at top level
        default_position = "final" if "final" in all_analyses_by_pos else PROBE_POSITIONS[0]
        analysis = all_analyses_by_pos[default_position][method]
        output_json["layers"] = analysis["layers"]
        output_json["num_questions"] = analysis["num_questions"]
        output_json["num_controls"] = analysis["num_controls"]
        output_json["metric"] = analysis["metric"]
        output_json["per_layer"] = analysis["per_layer"]
        output_json["summary"] = analysis["summary"]

        with open(results_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"  Saved {results_path.name}")
        output_files.append(results_path)

    # Generate plots - one per method per position
    print("\nGenerating plots...")
    for method in methods:
        method_base_output = get_base_output(method)
        for position in PROBE_POSITIONS:
            plot_path = get_output_path(f"{method_base_output}_{position}.png", model_dir=model_dir)
            plot_ablation_results(all_analyses_by_pos[position][method], method, plot_path)
            output_files.append(plot_path)

    # Collect key findings
    for position in PROBE_POSITIONS:
        for method in methods:
            if method in all_analyses_by_pos[position]:
                analysis = all_analyses_by_pos[position][method]
                summary = analysis.get("summary", {})
                n_sig = summary.get("n_significant_bootstrap_fdr", 0)
                total = len(analysis.get("layers", []))
                best_layer = summary.get("best_layer_abs_delta")
                if n_sig > 0:
                    key_findings[f"{position}/{method}"] = f"{n_sig}/{total} sig, best L{best_layer}"

    # Console output
    print_key_findings(key_findings)
    print_run_footer(output_files)


if __name__ == "__main__":
    main()
