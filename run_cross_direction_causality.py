"""
Stage 3. Information flow analysis: Does the model compute answer -> uncertainty -> confidence?

Tests causal information flow between direction types by ablating direction X at layer N
and measuring effects on direction Y's projection at downstream layers M > N. If X causally
precedes Y in the network's computation, ablating X should reduce Y's projection downstream.

Key design principles:
1. Only ablate at layers where the ablated direction is meaningful (R^2 >= threshold)
2. Only measure at layers where the target direction is meaningful (R^2 >= threshold)
3. Measurement must be downstream of ablation (measure_layer > ablate_layer + buffer)
4. Uses KV cache and batched processing for efficiency
5. Reports effect sizes as Cohen's d for interpretability

Inputs:
    outputs/{base}_mc_{metric}_directions.npz              Uncertainty directions
    outputs/{base}_mc_{metric}_results.json                Uncertainty R^2 per layer
    outputs/{base}_mc_answer_directions.npz                Answer directions
    outputs/{base}_mc_answer_results.json                  Answer accuracy per layer
    outputs/{base}_meta_{task}_metaconfdir_directions.npz  Confidence directions
    outputs/{base}_meta_{task}_metaconfdir_results.json    Confidence R^2 per layer
    outputs/{base}_meta_{task}_metamcuncert_directions.npz Metamcuncert directions (if DIRECTION_TYPES includes "metamcuncert")
    outputs/{base}_meta_{task}_metamcuncert_results.json   Metamcuncert R^2 per layer
    outputs/{base}_mc_dataset.json                         Question metadata

    where {base} = {model_short}_{dataset} or {model_short}_adapter-{adapter_short}_{dataset}

Outputs:
    outputs/{base}_cross_direction_{metric}_results.json   Effect matrix with CIs
    outputs/{base}_cross_direction_{metric}_results.png    Propagation heatmaps + flow diagram

Shared parameters (must match across scripts):
    SEED

Configuration:
    ADAPTER: Optional path to PEFT/LoRA adapter (must match identify step if used)

Run after: identify_mc_correlate.py, test_meta_transfer.py
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec

from core.model_utils import load_model_and_tokenizer, get_model_short_name, should_use_chat_template, DEVICE
from core.config_utils import get_config_dict
from core.plotting import save_figure, GRID_ALPHA, CI_ALPHA, DPI
from core.steering_experiments import (
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
)
from core.steering import AblationHook
from tasks import (
    format_stated_confidence_prompt,
    format_answer_or_delegate_prompt,
    format_other_confidence_prompt,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: LoRA adapter path (must match identify step if used)
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC_difficulty_filtered"
META_TASK = "confidence"  # "confidence", "delegate", or "other_confidence"
METRIC = "logit_gap"  # Uncertainty metric for uncertainty directions

# --- Quantization ---
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# --- Experiment ---
SEED = 42  # Must match across scripts
BATCH_SIZE = 4
NUM_QUESTIONS = 100

# Direction types to test
# - "uncertainty": MC uncertainty directions (d_mc from identify_mc_correlate.py)
# - "answer": MC answer directions (A/B/C/D)
# - "confidence": Stated confidence directions (d_confidence from test_meta_transfer.py)
# - "metamcuncert": MC uncertainty directions found from meta activations (from test_meta_transfer.py)
DIRECTION_TYPES = ["uncertainty", "answer", "confidence", "metamcuncert"]

# Layer selection: only ablate/measure where directions are meaningful
# Direction-specific thresholds (answer uses accuracy with chance=0.25, others use R²)
METRIC_THRESHOLDS = {
    "uncertainty": 0.3,   # R² threshold
    "answer": 0.35,       # Accuracy threshold (chance=0.25)
    "confidence": 0.3,    # R² threshold
    "metamcuncert": 0.3,  # R² threshold
}
R2_THRESHOLD = 0.2  # Default fallback if direction not in METRIC_THRESHOLDS
BUFFER_LAYERS = 1   # Measure at layers >= ablate_layer + buffer

# Effect size threshold for "meaningful" effects
EFFECT_SIZE_THRESHOLD = 0.2  # Cohen's d (legacy, kept for compatibility)
DELTA_THRESHOLD = 0.03  # Minimum |delta_mean| to display in figures (noise floor filter)
MIN_DISPLAY_LAYER = 6   # Visual boundary for "semantic" layers in figures

# Normalized projection settings
MIN_NORM = 1e-6  # Floor for numerical stability in normalization
COMPUTE_NORMALIZED_PROJECTIONS = True  # Enable normalized projection computation
PRESERVE_NORM = True  # Rescale ablated activations to preserve original magnitude (avoids LayerNorm artifacts)

# Per-sample data saving (for correlation analysis)
SAVE_PER_SAMPLE_DATA = True  # Save per-sample projections for uncertainty->confidence
PER_SAMPLE_PAIRS = [("uncertainty", "confidence")]  # Which ablate->measure pairs to save

# Control experiment: ablate random directions to establish baseline
N_CONTROL_DIRECTIONS = 3  # Random directions per ablation layer
RUN_CONTROL_EXPERIMENT = True  # Set False to skip control ablations

# Bootstrap settings
BOOTSTRAP_N = 1000
BOOTSTRAP_CI_ALPHA = 0.05

# Efficiency: batch multiple ablations per forward pass
EXPANDED_BATCH_TARGET = 48  # Expand batches to this size for GPU efficiency

# --- Output ---
OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DirectionInfo:
    """Information about a direction type including its R^2 curve."""
    name: str
    directions: Dict[int, np.ndarray]  # layer -> direction vector
    r2_by_layer: Dict[int, float]      # layer -> R^2 (or accuracy)
    meaningful_layers: List[int]        # layers where R^2 >= threshold
    peak_layer: int                     # layer with highest R^2
    formation_layer: int                # first layer where R^2 >= threshold


# =============================================================================
# R^2 LOADING (D→M: meta-task transfer performance)
# =============================================================================

def load_uncertainty_r2(base_name: str, meta_task: str, metric: str = "logit_gap") -> Dict[int, float]:
    """Load D→M R^2 for uncertainty directions from meta-transfer results.

    This is the transfer performance: how well uncertainty directions (trained on MC task)
    predict uncertainty from meta-task activations.
    """
    path = OUTPUT_DIR / f"{base_name}_meta_{meta_task}_results.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        print(f"  -> Run test_meta_transfer.py first to generate this file")
        return {}

    with open(path) as f:
        data = json.load(f)

    r2_by_layer = {}

    # Structure: transfer[metric]["per_layer"][layer]["d2m_centered_r2"]
    transfer = data.get("transfer", {})
    if metric in transfer and "per_layer" in transfer[metric]:
        per_layer = transfer[metric]["per_layer"]
        for layer_str, layer_data in per_layer.items():
            try:
                layer = int(layer_str)
                r2 = layer_data.get("d2m_centered_r2", 0)
                r2_by_layer[layer] = r2
            except (ValueError, KeyError):
                continue

    return r2_by_layer


def load_answer_r2(base_name: str, meta_task: str) -> Dict[int, float]:
    """Load D→M accuracy for answer directions from meta-transfer results.

    This is the transfer performance: how well answer directions (trained on MC task)
    predict which answer from meta-task activations.
    """
    path = OUTPUT_DIR / f"{base_name}_meta_{meta_task}_results.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        print(f"  -> Run test_meta_transfer.py first to generate this file")
        return {}

    with open(path) as f:
        data = json.load(f)

    r2_by_layer = {}
    # Structure: answer_transfer.d2m_by_position[position][layer]["centered_accuracy"]
    answer_transfer = data.get("answer_transfer", {})
    d2m_by_pos = answer_transfer.get("d2m_by_position", {})

    # Use "final" position (current default), fall back to "centered" or "last"
    for pos in ["final", "centered", "last"]:
        if pos in d2m_by_pos and d2m_by_pos[pos]:
            for layer_str, layer_data in d2m_by_pos[pos].items():
                try:
                    layer = int(layer_str)
                    # Key is "centered_accuracy" not nested
                    acc = layer_data.get("centered_accuracy", 0)
                    r2_by_layer[layer] = max(r2_by_layer.get(layer, -999), acc)
                except (ValueError, KeyError):
                    continue
            break  # Use first available position

    return r2_by_layer


def load_confidence_r2(base_name: str, meta_task: str) -> Dict[int, float]:
    """Load R^2 values for confidence directions from meta-task results."""
    path = OUTPUT_DIR / f"{base_name}_meta_{meta_task}_metaconfdir_results.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        print(f"  -> Run test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True")
        return {}

    with open(path) as f:
        data = json.load(f)

    r2_by_layer = {}
    # Try probe method first, fall back to mean_diff
    for method in ["probe", "mean_diff"]:
        if method in data.get("results", {}):
            for layer_str, layer_data in data["results"][method].items():
                try:
                    layer = int(layer_str)
                    # Confidence directions use "test_r2" key, not "r2"
                    r2 = layer_data.get("test_r2", layer_data.get("r2", 0))
                    r2_by_layer[layer] = max(r2_by_layer.get(layer, -999), r2)
                except (ValueError, KeyError):
                    continue

    return r2_by_layer


def load_metamcuncert_r2(base_name: str, meta_task: str, metric: str = "logit_gap") -> Dict[int, float]:
    """Load R^2 values for metamcuncert directions (meta→MC uncertainty).

    These are directions in meta-task activations that predict MC-task uncertainty.
    """
    path = OUTPUT_DIR / f"{base_name}_meta_{meta_task}_metamcuncert_results.json"
    if not path.exists():
        print(f"  Warning: {path} not found")
        print(f"  -> Run test_meta_transfer.py with FIND_MC_UNCERTAINTY_DIRECTIONS=True")
        return {}

    with open(path) as f:
        data = json.load(f)

    r2_by_layer = {}
    # Try probe method first, fall back to mean_diff
    for method in ["probe", "mean_diff"]:
        if method in data.get("results", {}):
            for layer_str, layer_data in data["results"][method].items():
                try:
                    layer = int(layer_str)
                    r2 = layer_data.get("test_r2", 0)
                    r2_by_layer[layer] = max(r2_by_layer.get(layer, -999), r2)
                except (ValueError, KeyError):
                    continue

    return r2_by_layer


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def load_directions_for_type(
    base_name: str,
    direction_type: str,
    metric: str = "logit_gap",
    meta_task: str = "confidence",
    method: str = "probe"
) -> Dict[int, np.ndarray]:
    """Load direction vectors for a specific type and method.

    IMPORTANT: Uses chain alignment to ensure consecutive layers have positive
    cosine similarity. Each layer is aligned to the previous layer, preventing
    sign flips that would cause diagonal checks to fail while allowing the
    direction to evolve through the network.
    """
    if direction_type == "uncertainty":
        path = OUTPUT_DIR / f"{base_name}_mc_{metric}_directions.npz"
    elif direction_type == "answer":
        path = OUTPUT_DIR / f"{base_name}_mc_answer_directions.npz"
    elif direction_type == "confidence":
        path = OUTPUT_DIR / f"{base_name}_meta_{meta_task}_metaconfdir_directions.npz"
    elif direction_type == "metamcuncert":
        path = OUTPUT_DIR / f"{base_name}_meta_{meta_task}_metamcuncert_directions.npz"
    else:
        raise ValueError(f"Unknown direction type: {direction_type}")

    if not path.exists():
        print(f"  Warning: {direction_type} directions not found at {path}")
        return {}

    data = np.load(path)
    directions = {}

    # Map direction types to expected method prefixes
    if direction_type == "uncertainty":
        prefix = f"{method}_layer_"
    elif direction_type == "answer":
        # Answer uses probe/centroid
        method_map = {"mean_diff": "centroid"}
        actual_method = method_map.get(method, method)
        prefix = f"{actual_method}_layer_"
    elif direction_type == "confidence":
        prefix = f"{method}_layer_"
    elif direction_type == "metamcuncert":
        prefix = f"{method}_layer_"

    # Collect layer -> key mapping, sorted by layer
    layer_keys = []
    for key in data.files:
        if key.startswith("_"):
            continue
        if key.startswith(prefix):
            try:
                layer = int(key.replace(prefix, ""))
                layer_keys.append((layer, key))
            except ValueError:
                continue

    # Sort by layer to process in order (important for sign consistency)
    layer_keys.sort(key=lambda x: x[0])

    # Track reference direction for sign alignment
    reference_direction = None

    for layer, key in layer_keys:
        direction = data[key].astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm

        # Ensure sign consistency with previous layer (chain alignment)
        if reference_direction is None:
            reference_direction = direction.copy()
        else:
            if reference_direction @ direction < 0:  # Opposite sign
                direction = -direction  # Flip to match reference
            reference_direction = direction.copy()  # Update reference for next layer

        directions[layer] = direction

    # Fallback for legacy answer format
    if not directions and direction_type == "answer":
        layer_keys = []
        for key in data.files:
            if key.startswith("layer_"):
                try:
                    layer = int(key.replace("layer_", ""))
                    layer_keys.append((layer, key))
                except ValueError:
                    continue

        layer_keys.sort(key=lambda x: x[0])
        reference_direction = None

        for layer, key in layer_keys:
            direction = data[key].astype(np.float32)
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm

            # Ensure sign consistency with previous layer (chain alignment)
            if reference_direction is None:
                reference_direction = direction.copy()
            else:
                if reference_direction @ direction < 0:
                    direction = -direction
                reference_direction = direction.copy()  # Update reference for next layer

            directions[layer] = direction

    return directions


def load_all_direction_info(
    base_name: str,
    direction_types: List[str],
    metric: str,
    meta_task: str,
    r2_threshold: float,
    method: str = "probe"
) -> Dict[str, DirectionInfo]:
    """Load directions and R^2 curves for all direction types."""
    all_info = {}

    for dir_type in direction_types:
        # Load directions
        directions = load_directions_for_type(base_name, dir_type, metric, meta_task, method)
        if not directions:
            print(f"  {dir_type}: no directions found")
            continue

        # Load R^2 curve (D→M transfer performance on meta-task)
        if dir_type == "uncertainty":
            r2_by_layer = load_uncertainty_r2(base_name, meta_task, metric)
        elif dir_type == "answer":
            r2_by_layer = load_answer_r2(base_name, meta_task)
        elif dir_type == "confidence":
            r2_by_layer = load_confidence_r2(base_name, meta_task)
        elif dir_type == "metamcuncert":
            r2_by_layer = load_metamcuncert_r2(base_name, meta_task, metric)
        else:
            r2_by_layer = {}

        # Diagnostic: show what R² values were loaded
        if r2_by_layer:
            r2_vals = list(r2_by_layer.values())
            print(f"  {dir_type}: R² range [{min(r2_vals):.3f}, {max(r2_vals):.3f}] across {len(r2_by_layer)} layers")
        else:
            print(f"  {dir_type}: WARNING - no R² values loaded!")

        # Use direction-specific threshold (answer uses accuracy, others use R²)
        effective_threshold = METRIC_THRESHOLDS.get(dir_type, r2_threshold)

        # Compute meaningful layers
        meaningful_layers = sorted([l for l, r2 in r2_by_layer.items() if r2 >= effective_threshold])

        if not meaningful_layers:
            print(f"  {dir_type}: WARNING - no layers meet threshold >= {effective_threshold}!")
            if r2_by_layer:
                # Show what we have
                sorted_r2 = sorted(r2_by_layer.items(), key=lambda x: x[1], reverse=True)
                print(f"    Top 5 layers by R²: {sorted_r2[:5]}")
            if not r2_by_layer:
                raise ValueError(
                    f"Failed to load R² values for {dir_type}.\n"
                    f"  This script requires test_meta_transfer.py to be run first.\n"
                    f"  Expected file: {base_name}_meta_{meta_task}_results.json"
                )
            # Fall back to all available layers if we have R² but none meet threshold
            meaningful_layers = sorted(directions.keys())
            print(f"    Falling back to all {len(meaningful_layers)} layers")

        # Find peak and formation layers
        if r2_by_layer:
            peak_layer = max(r2_by_layer.keys(), key=lambda l: r2_by_layer[l])
            # Formation = first layer where R^2 >= threshold
            formation_layer = min(meaningful_layers) if meaningful_layers else 0
        else:
            peak_layer = max(directions.keys())
            formation_layer = min(directions.keys())

        all_info[dir_type] = DirectionInfo(
            name=dir_type,
            directions=directions,
            r2_by_layer=r2_by_layer,
            meaningful_layers=meaningful_layers,
            peak_layer=peak_layer,
            formation_layer=formation_layer,
        )

        print(f"  {dir_type}: {len(directions)} layers, meaningful={len(meaningful_layers)}, "
              f"formation=L{formation_layer}, peak=L{peak_layer} (R^2={r2_by_layer.get(peak_layer, 0):.3f})")

    return all_info


# =============================================================================
# DATASET LOADING
# =============================================================================

def load_questions(base_name: str, num_questions: int) -> List[Dict]:
    """Load question data from MC dataset."""
    path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(path) as f:
        data = json.load(f)

    questions = data.get("data", data.get("questions", data.get("items", [])))
    if not questions:
        raise ValueError(f"No questions loaded from {path}")

    return questions[:num_questions]


# =============================================================================
# EXTRACTION WITH ABLATION
# =============================================================================

class ExtractionHook:
    """Hook to extract activations at specified layer."""

    def __init__(self):
        self.activation = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output
        # Store last token activation
        self.activation = hidden_states[:, -1, :].detach().cpu().numpy()
        return output


def run_with_ablation_and_extraction(
    model,
    tokenizer,
    prompts: List[str],
    ablate_direction: np.ndarray,
    ablate_layer: int,
    extract_layers: List[int],
    batch_size: int = 4,
    also_extract_ablate_layer: bool = False,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    """
    Run forward passes with and without ablation, extracting activations at specified layers.
    Uses KV cache for efficiency.

    Args:
        also_extract_ablate_layer: If True, also extract at ablate_layer during baseline pass
                                   (needed for per-sample correlation analysis)

    Returns:
        baseline_acts: Dict[layer -> (n_questions, hidden_dim)] without ablation
        ablated_acts: Dict[layer -> (n_questions, hidden_dim)] with ablation
    """
    # Get model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        raise ValueError("Cannot find model layers")

    # Pretokenize and build batches
    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, batch_size)

    # Determine which layers to extract during baseline pass
    baseline_extract_layers = list(extract_layers)
    if also_extract_ablate_layer and ablate_layer not in baseline_extract_layers:
        baseline_extract_layers.append(ablate_layer)
        baseline_extract_layers.sort()

    baseline_acts = {layer: [] for layer in baseline_extract_layers}
    ablated_acts = {layer: [] for layer in extract_layers}

    # Process batches with KV cache
    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Processing", leave=False):
        B = len(batch_indices)

        # Get KV cache for this batch
        cache_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = cache_data["past_key_values_data"]

        # --- Baseline pass (no ablation) ---
        extraction_hooks = {}
        handles = []
        for layer in baseline_extract_layers:
            hook = ExtractionHook()
            extraction_hooks[layer] = hook
            handles.append(layers[layer].register_forward_hook(hook))

        # Create fresh cache and run
        fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
        baseline_inputs = {
            "input_ids": cache_data["input_ids"],
            "attention_mask": cache_data["attention_mask"],
            "past_key_values": fresh_cache,
            "use_cache": True,
        }

        with torch.inference_mode():
            model(**baseline_inputs)

        for handle in handles:
            handle.remove()

        for layer in baseline_extract_layers:
            baseline_acts[layer].append(extraction_hooks[layer].activation)

        # --- Ablated pass ---
        extraction_hooks = {}
        handles = []

        # Register ablation hook at ablation layer (uses simpler AblationHook from core.steering)
        ablation_hook = AblationHook(ablate_direction, position="last", preserve_norm=PRESERVE_NORM)
        ablation_handle = layers[ablate_layer].register_forward_hook(ablation_hook)

        # Register extraction hooks at measurement layers
        for layer in extract_layers:
            hook = ExtractionHook()
            extraction_hooks[layer] = hook
            handles.append(layers[layer].register_forward_hook(hook))

        # Create fresh cache and run
        fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
        ablated_inputs = {
            "input_ids": cache_data["input_ids"],
            "attention_mask": cache_data["attention_mask"],
            "past_key_values": fresh_cache,
            "use_cache": True,
        }

        with torch.inference_mode():
            model(**ablated_inputs)

        ablation_handle.remove()
        for handle in handles:
            handle.remove()

        for layer in extract_layers:
            ablated_acts[layer].append(extraction_hooks[layer].activation)

    # Concatenate batches
    # Note: baseline_acts may have ablate_layer if also_extract_ablate_layer=True
    for layer in baseline_acts.keys():
        baseline_acts[layer] = np.concatenate(baseline_acts[layer], axis=0)
    for layer in extract_layers:
        ablated_acts[layer] = np.concatenate(ablated_acts[layer], axis=0)

    return baseline_acts, ablated_acts


def extract_baseline_at_layers(
    model,
    tokenizer,
    prompts: List[str],
    extract_layers: List[int],
    batch_size: int = 4,
) -> Dict[int, np.ndarray]:
    """
    Run forward passes WITHOUT ablation, extracting activations at specified layers.
    Used to get baseline projections for ALL layers (including pre-ablation layers).

    Returns:
        baseline_acts: Dict[layer -> (n_questions, hidden_dim)]
    """
    # Get model layers
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
    elif hasattr(model, 'get_base_model'):
        layers = model.get_base_model().model.layers
    else:
        raise ValueError("Cannot find model layers")

    # Pretokenize and build batches
    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, batch_size)

    baseline_acts = {layer: [] for layer in extract_layers}

    # Process batches
    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Extracting baseline", leave=False):
        # Get KV cache for this batch
        cache_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = cache_data["past_key_values_data"]

        # Register extraction hooks
        extraction_hooks = {}
        handles = []
        for layer in extract_layers:
            hook = ExtractionHook()
            extraction_hooks[layer] = hook
            handles.append(layers[layer].register_forward_hook(hook))

        # Create fresh cache and run
        fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
        baseline_inputs = {
            "input_ids": cache_data["input_ids"],
            "attention_mask": cache_data["attention_mask"],
            "past_key_values": fresh_cache,
            "use_cache": True,
        }

        with torch.inference_mode():
            model(**baseline_inputs)

        for handle in handles:
            handle.remove()

        for layer in extract_layers:
            baseline_acts[layer].append(extraction_hooks[layer].activation)

    # Concatenate batches
    for layer in extract_layers:
        baseline_acts[layer] = np.concatenate(baseline_acts[layer], axis=0)

    return baseline_acts


# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================

def compute_projection(activations: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Compute projection of activations onto direction."""
    return activations @ direction


def bootstrap_effect(
    baseline_proj: np.ndarray,
    ablated_proj: np.ndarray,
    n_bootstrap: int = 1000,
    ci_alpha: float = 0.05,
    seed: int = 42
) -> Dict:
    """
    Compute effect size (Cohen's d) with bootstrap CIs and significance.
    """
    rng = np.random.RandomState(seed)
    n = len(baseline_proj)

    # Compute paired difference
    delta = ablated_proj - baseline_proj
    observed_delta_mean = np.mean(delta)

    # Baseline statistics for Cohen's d
    baseline_std = np.std(baseline_proj)
    if baseline_std < 1e-10:
        baseline_std = 1.0

    observed_cohens_d = observed_delta_mean / baseline_std

    # Bootstrap for CI (use fixed baseline_std for denominator stability)
    bootstrap_deltas = []
    bootstrap_cohens_d = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        boot_delta = np.mean(delta[idx])
        bootstrap_deltas.append(boot_delta)
        bootstrap_cohens_d.append(boot_delta / baseline_std)  # Fixed denominator

    bootstrap_deltas = np.array(bootstrap_deltas)
    bootstrap_cohens_d = np.array(bootstrap_cohens_d)

    # CIs
    delta_ci_low = np.percentile(bootstrap_deltas, 100 * ci_alpha / 2)
    delta_ci_high = np.percentile(bootstrap_deltas, 100 * (1 - ci_alpha / 2))
    d_ci_low = np.percentile(bootstrap_cohens_d, 100 * ci_alpha / 2)
    d_ci_high = np.percentile(bootstrap_cohens_d, 100 * (1 - ci_alpha / 2))

    # P-value via permutation test
    null_deltas = []
    for _ in range(n_bootstrap):
        swap = rng.choice([True, False], size=n)
        null_baseline = np.where(swap, ablated_proj, baseline_proj)
        null_ablated = np.where(swap, baseline_proj, ablated_proj)
        null_deltas.append(np.mean(null_ablated - null_baseline))

    null_deltas = np.array(null_deltas)
    p_value = np.mean(np.abs(null_deltas) >= np.abs(observed_delta_mean))

    return {
        "baseline_mean": float(np.mean(baseline_proj)),
        "baseline_std": float(baseline_std),
        "ablated_mean": float(np.mean(ablated_proj)),
        "delta_mean": float(observed_delta_mean),
        "delta_ci_low": float(delta_ci_low),
        "delta_ci_high": float(delta_ci_high),
        "cohens_d": float(observed_cohens_d),
        "cohens_d_ci_low": float(d_ci_low),
        "cohens_d_ci_high": float(d_ci_high),
        "p_value": float(p_value),
        "n_samples": n,
    }


def generate_random_direction(hidden_dim: int, rng: np.random.RandomState,
                              orthogonal_to: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Generate a random unit vector in activation space.

    Args:
        hidden_dim: Dimensionality of the activation space
        rng: Random state for reproducibility
        orthogonal_to: Optional direction to orthogonalize against

    Returns:
        Normalized random direction
    """
    # Generate random vector from standard normal
    direction = rng.randn(hidden_dim).astype(np.float32)

    # Optionally orthogonalize against the given direction
    if orthogonal_to is not None:
        # Gram-Schmidt: remove component along orthogonal_to
        projection = np.dot(direction, orthogonal_to) * orthogonal_to
        direction = direction - projection

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm
    else:
        # Extremely unlikely, but handle it
        direction = rng.randn(hidden_dim).astype(np.float32)
        direction = direction / np.linalg.norm(direction)

    return direction


def apply_fdr_correction(results: Dict, alpha: float = 0.05) -> Dict:
    """Apply Benjamini-Hochberg FDR correction to p-values."""
    pvals = []
    for key, data in results.items():
        if isinstance(data, dict) and "p_value" in data:
            pvals.append((key, data["p_value"]))

    if not pvals:
        return results

    pvals.sort(key=lambda x: x[1])
    n = len(pvals)

    for i, (key, pval) in enumerate(pvals):
        rank = i + 1
        bh_threshold = alpha * rank / n
        results[key]["p_value_fdr"] = min(pval * n / rank, 1.0)
        results[key]["significant_fdr"] = pval <= bh_threshold

    return results


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_r2_curves(direction_info: Dict[str, DirectionInfo], r2_threshold: float, output_path: Path):
    """Plot R^2 curves for all direction types to show temporal ordering."""
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"uncertainty": "tab:orange", "answer": "tab:blue", "confidence": "tab:green"}

    for dir_type, info in direction_info.items():
        layers = sorted(info.r2_by_layer.keys())
        r2_values = [info.r2_by_layer[l] for l in layers]

        ax.plot(layers, r2_values, 'o-', color=colors.get(dir_type, "gray"),
                label=f"{dir_type} (form=L{info.formation_layer}, peak=L{info.peak_layer})",
                markersize=4, linewidth=1.5)

    ax.axhline(r2_threshold, color='gray', linestyle='--', alpha=0.5,
               label=f"Threshold (R^2={r2_threshold})")

    ax.set_xlabel("Layer")
    ax.set_ylabel("D→M R² (or Accuracy)")
    ax.set_title("Direction Transfer to Meta-Task (D→M)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.set_ylim(bottom=0)

    save_figure(fig, output_path)


def plot_propagation_heatmaps(
    results: Dict,
    direction_info: Dict[str, DirectionInfo],
    output_path: Path,
    delta_threshold: float = DELTA_THRESHOLD
):
    """
    Plot information flow as heatmaps showing how effects propagate downstream.
    Creates a grid of heatmaps for off-diagonal pairs (ablate X -> measure Y where X != Y).

    Only shows effects that are BOTH FDR-significant AND exceed delta_threshold.
    Colors by delta_mean (not Cohen's d) to avoid variance-inflation artifacts.
    """
    # Get all direction types present
    ablate_types = sorted(set(k[0] for k in results.keys() if isinstance(k, tuple)))
    measure_types = sorted(set(k[2] for k in results.keys() if isinstance(k, tuple)))

    # Get off-diagonal pairs
    off_diag_pairs = [(a, m) for a in ablate_types for m in measure_types if a != m]

    if not off_diag_pairs:
        print("  No off-diagonal pairs to plot")
        return

    # Determine grid size
    n_pairs = len(off_diag_pairs)
    n_cols = min(3, n_pairs)
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(5 * n_cols, 4 * n_rows + 1))
    gs = gridspec.GridSpec(n_rows + 1, n_cols, height_ratios=[1] * n_rows + [0.1])

    # Find global color scale based on delta_mean (not Cohen's d)
    all_deltas = [d.get("delta_mean", 0) for d in results.values()
                  if isinstance(d, dict) and "delta_mean" in d]
    if all_deltas:
        abs_deltas = [abs(e) for e in all_deltas if not np.isnan(e)]
        if abs_deltas:
            # Use 95th percentile to set scale
            vmax = np.percentile(abs_deltas, 95) * 1.2
            vmax = max(vmax, 0.05)  # Minimum scale
        else:
            vmax = 0.1
    else:
        vmax = 0.1

    im = None
    for idx, (ablate_type, measure_type) in enumerate(off_diag_pairs):
        row, col = idx // n_cols, idx % n_cols
        ax = fig.add_subplot(gs[row, col])

        # Get layers for this pair
        ablate_layers = sorted(set(k[1] for k in results.keys()
                                   if isinstance(k, tuple) and k[0] == ablate_type and k[2] == measure_type))
        measure_layers = sorted(set(k[3] for k in results.keys()
                                    if isinstance(k, tuple) and k[0] == ablate_type and k[2] == measure_type))

        if not ablate_layers or not measure_layers:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Ablate {ablate_type} -> {measure_type}")
            continue

        # Build effect matrix with DUAL FILTER: significant AND meaningful magnitude
        effect_matrix = np.full((len(measure_layers), len(ablate_layers)), np.nan)
        excluded_matrix = np.zeros_like(effect_matrix, dtype=bool)  # Track filtered-out cells

        for key, data in results.items():
            if not isinstance(key, tuple) or len(key) < 4:
                continue
            if key[0] == ablate_type and key[2] == measure_type:
                ablate_layer, measure_layer = key[1], key[3]
                if ablate_layer in ablate_layers and measure_layer in measure_layers:
                    i = measure_layers.index(measure_layer)
                    j = ablate_layers.index(ablate_layer)

                    is_significant = data.get("significant_fdr", False)
                    delta = data.get("delta_mean", 0)
                    is_meaningful = abs(delta) >= delta_threshold

                    if is_significant and is_meaningful:
                        effect_matrix[i, j] = delta  # Color by delta, not Cohen's d
                    else:
                        excluded_matrix[i, j] = True  # Mark as filtered out

        # Plot heatmap
        im = ax.imshow(effect_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")

        # Mark filtered-out cells with small gray dot
        for i in range(len(measure_layers)):
            for j in range(len(ablate_layers)):
                if excluded_matrix[i, j]:
                    ax.plot(j, i, '.', color='gray', markersize=2, alpha=0.5)

        # Add visual boundary line at MIN_DISPLAY_LAYER
        # Find index positions for the boundary
        for boundary_layer in [MIN_DISPLAY_LAYER]:
            # X-axis boundary (ablation layers)
            if boundary_layer in ablate_layers:
                x_idx = ablate_layers.index(boundary_layer)
                ax.axvline(x=x_idx - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            elif ablate_layers and min(ablate_layers) < boundary_layer < max(ablate_layers):
                # Interpolate position
                x_idx = sum(1 for l in ablate_layers if l < boundary_layer)
                ax.axvline(x=x_idx - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

            # Y-axis boundary (measurement layers)
            if boundary_layer in measure_layers:
                y_idx = measure_layers.index(boundary_layer)
                ax.axhline(y=y_idx - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            elif measure_layers and min(measure_layers) < boundary_layer < max(measure_layers):
                y_idx = sum(1 for l in measure_layers if l < boundary_layer)
                ax.axhline(y=y_idx - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Labels - show every 5th layer to avoid crowding
        step_x = max(1, len(ablate_layers) // 6)
        step_y = max(1, len(measure_layers) // 6)
        ax.set_xticks(range(0, len(ablate_layers), step_x))
        ax.set_xticklabels([str(ablate_layers[i]) for i in range(0, len(ablate_layers), step_x)], fontsize=8)
        ax.set_yticks(range(0, len(measure_layers), step_y))
        ax.set_yticklabels([str(measure_layers[i]) for i in range(0, len(measure_layers), step_y)], fontsize=8)
        ax.set_xlabel("Ablation Layer")
        ax.set_ylabel("Measurement Layer")
        ax.set_title(f"{ablate_type} → {measure_type}")

    # Colorbar
    if im is not None:
        cbar_ax = fig.add_subplot(gs[-1, :])
        cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
        cbar.set_label(f"Δ projection (FDR sig. & |Δ|>{delta_threshold:.2f}; gray=filtered)")

    fig.suptitle("Cross-Direction Causal Effects", fontsize=12, y=0.98)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_flow_summary(
    results: Dict,
    direction_info: Dict[str, DirectionInfo],
    output_path: Path,
    delta_threshold: float = DELTA_THRESHOLD,
    control_mean_delta: float = None
):
    """
    Create a summary diagram showing the strongest causal effects between directions.

    Shows delta_mean values and control multiples instead of Cohen's d.
    Only draws arrows for effects that are both significant AND exceed delta_threshold.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Position directions in a triangle
    positions = {
        "answer": (0.2, 0.5),
        "uncertainty": (0.5, 0.9),
        "confidence": (0.8, 0.5),
    }

    # Draw direction nodes
    for dir_type, (x, y) in positions.items():
        if dir_type in direction_info:
            info = direction_info[dir_type]
            circle = plt.Circle((x, y), 0.1, color='lightblue', ec='black', linewidth=2)
            ax.add_patch(circle)
            ax.text(x, y, f"{dir_type}\nL{info.formation_layer}-{info.peak_layer}",
                    ha='center', va='center', fontsize=9, fontweight='bold')

    # Collect strongest effects for each direction pair (by |delta_mean|, not Cohen's d)
    pair_effects = {}  # (ablate, measure) -> (delta, ablate_layer, measure_layer, sig)

    for key, data in results.items():
        if not isinstance(key, tuple) or len(key) < 4:
            continue
        ablate_type, ablate_layer, measure_type, measure_layer = key

        if ablate_type == measure_type:
            continue  # Skip diagonal

        pair_key = (ablate_type, measure_type)
        delta = data.get("delta_mean", 0)
        sig = data.get("significant_fdr", False)
        is_meaningful = abs(delta) >= delta_threshold

        # Only consider significant AND meaningful effects
        if sig and is_meaningful:
            if pair_key not in pair_effects or abs(delta) > abs(pair_effects[pair_key][0]):
                pair_effects[pair_key] = (delta, ablate_layer, measure_layer, sig)

    # Draw arrows for significant effects
    for (ablate_type, measure_type), (delta, abl_l, meas_l, sig) in pair_effects.items():
        if ablate_type not in positions or measure_type not in positions:
            continue

        x1, y1 = positions[ablate_type]
        x2, y2 = positions[measure_type]

        # Offset for bidirectional arrows
        dx, dy = x2 - x1, y2 - y1
        length = np.sqrt(dx**2 + dy**2)
        if length > 0:
            # Perpendicular offset
            px, py = -dy/length * 0.03, dx/length * 0.03
        else:
            px, py = 0, 0

        # Arrow properties based on delta (negative = reduces, positive = increases)
        color = 'green' if delta < 0 else 'red'
        # Line width based on magnitude relative to threshold
        linewidth = min(3, 1 + abs(delta) / delta_threshold)
        linestyle = '-'

        # Draw arrow
        arrow = FancyArrowPatch(
            (x1 + px + 0.1 * dx/length, y1 + py + 0.1 * dy/length),
            (x2 + px - 0.1 * dx/length, y2 + py - 0.1 * dy/length),
            arrowstyle='->', mutation_scale=15,
            color=color, linewidth=linewidth, linestyle=linestyle,
        )
        ax.add_patch(arrow)

        # Label with delta and control multiple
        mid_x = (x1 + x2) / 2 + px * 2
        mid_y = (y1 + y2) / 2 + py * 2
        if control_mean_delta and control_mean_delta > 0:
            ctrl_mult = abs(delta) / control_mean_delta
            label = f"Δ={delta:+.3f}\n({ctrl_mult:.1f}× ctrl)\nL{abl_l}→L{meas_l}"
        else:
            label = f"Δ={delta:+.3f}\nL{abl_l}→L{meas_l}"
        ax.text(mid_x, mid_y, label, ha='center', va='center', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Information Flow Summary\n(green=reduces, red=increases; |Δ|>{delta_threshold:.2f} & FDR sig)")

    save_figure(fig, output_path)


def plot_diagonal_effects(
    results: Dict,
    direction_info: Dict[str, DirectionInfo],
    output_path: Path,
    delta_threshold: float = DELTA_THRESHOLD
):
    """
    Create heatmaps showing diagonal effects (ablate X -> measure X) across all layer pairs.
    One subplot per direction type. Reveals where ablation has expected vs unexpected effects.

    Only shows effects that are BOTH FDR-significant AND exceed delta_threshold.
    Colors by delta_mean (not Cohen's d) to avoid variance-inflation artifacts.
    """
    dir_types = list(direction_info.keys())
    n_types = len(dir_types)

    if n_types == 0:
        return

    fig, axes = plt.subplots(1, n_types, figsize=(5 * n_types, 4))
    if n_types == 1:
        axes = [axes]

    # Find global color scale based on delta_mean (not Cohen's d)
    all_deltas = []
    for k, v in results.items():
        if isinstance(k, tuple) and k[0] == k[2]:  # Diagonal only
            all_deltas.append(v.get("delta_mean", 0))

    if all_deltas:
        abs_deltas = [abs(d) for d in all_deltas if not np.isnan(d)]
        if abs_deltas:
            vmax = np.percentile(abs_deltas, 95) * 1.2
            vmax = max(vmax, 0.05)  # Minimum scale
        else:
            vmax = 0.1
    else:
        vmax = 0.1

    im = None
    for idx, dir_type in enumerate(dir_types):
        ax = axes[idx]

        # Collect diagonal effects for this type
        diag_effects = {}
        for k, v in results.items():
            if isinstance(k, tuple) and k[0] == dir_type and k[2] == dir_type:
                abl_l, meas_l = k[1], k[3]
                diag_effects[(abl_l, meas_l)] = v

        if not diag_effects:
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{dir_type} (diagonal)")
            continue

        # Get unique layers
        abl_layers = sorted(set(k[0] for k in diag_effects.keys()))
        meas_layers = sorted(set(k[1] for k in diag_effects.keys()))

        # Build matrix with DUAL FILTER
        matrix = np.full((len(meas_layers), len(abl_layers)), np.nan)
        excluded_matrix = np.zeros_like(matrix, dtype=bool)

        for (abl_l, meas_l), data in diag_effects.items():
            if abl_l in abl_layers and meas_l in meas_layers:
                i = meas_layers.index(meas_l)
                j = abl_layers.index(abl_l)

                is_significant = data.get("significant_fdr", False)
                delta = data.get("delta_mean", 0)
                is_meaningful = abs(delta) >= delta_threshold

                if is_significant and is_meaningful:
                    matrix[i, j] = delta  # Color by delta, not Cohen's d
                else:
                    excluded_matrix[i, j] = True

        # Plot heatmap - negative (expected) is blue, positive (unexpected) is red
        im = ax.imshow(matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       aspect="auto", origin="lower")

        # Mark filtered-out cells with gray dots
        for i in range(len(meas_layers)):
            for j in range(len(abl_layers)):
                if excluded_matrix[i, j]:
                    ax.plot(j, i, '.', color='gray', markersize=2, alpha=0.5)

        # Add visual boundary at MIN_DISPLAY_LAYER
        for boundary_layer in [MIN_DISPLAY_LAYER]:
            if abl_layers and min(abl_layers) < boundary_layer < max(abl_layers):
                x_idx = sum(1 for l in abl_layers if l < boundary_layer)
                ax.axvline(x=x_idx - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            if meas_layers and min(meas_layers) < boundary_layer < max(meas_layers):
                y_idx = sum(1 for l in meas_layers if l < boundary_layer)
                ax.axhline(y=y_idx - 0.5, color='black', linestyle='--', linewidth=0.8, alpha=0.5)

        # Axis labels - show every few layers to avoid crowding
        step_x = max(1, len(abl_layers) // 6)
        step_y = max(1, len(meas_layers) // 6)
        ax.set_xticks(range(0, len(abl_layers), step_x))
        ax.set_xticklabels([str(abl_layers[i]) for i in range(0, len(abl_layers), step_x)], fontsize=8)
        ax.set_yticks(range(0, len(meas_layers), step_y))
        ax.set_yticklabels([str(meas_layers[i]) for i in range(0, len(meas_layers), step_y)], fontsize=8)

        ax.set_xlabel("Ablation Layer")
        ax.set_ylabel("Measurement Layer")
        ax.set_title(f"{dir_type}")

    # Add colorbar
    if im is not None:
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label(f"Δ projection (FDR sig. & |Δ|>{delta_threshold:.2f})")

    fig.suptitle("Diagonal: Ablate X → Measure X (expect blue/negative)", fontsize=11)
    save_figure(fig, output_path)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("CROSS-DIRECTION CAUSAL FLOW ANALYSIS")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    print(f"Input: {INPUT_BASE_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Meta-task: {META_TASK}")
    print(f"Direction types: {DIRECTION_TYPES}")
    print(f"R^2 threshold: {R2_THRESHOLD}")
    print(f"Buffer layers: {BUFFER_LAYERS}")
    print(f"Effect size threshold: {EFFECT_SIZE_THRESHOLD}")
    print(f"Questions: {NUM_QUESTIONS}")
    print(f"Bootstrap: {BOOTSTRAP_N} resamples")

    # Load all direction info
    print("\nLoading directions and R^2 curves...")
    direction_info = load_all_direction_info(
        INPUT_BASE_NAME, DIRECTION_TYPES, METRIC, META_TASK, R2_THRESHOLD
    )

    if len(direction_info) < 2:
        raise ValueError(f"Need at least 2 direction types, found: {list(direction_info.keys())}")

    # Cross-layer direction similarity diagnostic (first vs last)
    print("\nCROSS-LAYER DIRECTION SIMILARITY (first vs last):")
    print("  (Low similarity suggests directions differ across layers, which can cause positive diagonal effects)")
    for dir_type, info in direction_info.items():
        layers = sorted(info.meaningful_layers)
        if len(layers) >= 2:
            first_layer = layers[0]
            last_layer = layers[-1]
            if first_layer in info.directions and last_layer in info.directions:
                cosine = float(info.directions[first_layer] @ info.directions[last_layer])
                print(f"  {dir_type}: L{first_layer} vs L{last_layer} cosine = {cosine:.3f}")
            else:
                print(f"  {dir_type}: direction not available at both L{first_layer} and L{last_layer}")
        else:
            print(f"  {dir_type}: only {len(layers)} meaningful layer(s), skipping")

    # Consecutive-layer similarity diagnostic
    print("\nCONSECUTIVE-LAYER DIRECTION SIMILARITY:")
    print("  (Low similarity = directions capture different things at different layers)")
    for dir_type, info in direction_info.items():
        layers = sorted(info.directions.keys())
        if len(layers) < 2:
            print(f"  {dir_type}: insufficient layers")
            continue

        cosines = []
        for i in range(len(layers) - 1):
            l1, l2 = layers[i], layers[i + 1]
            cos = float(info.directions[l1] @ info.directions[l2])
            cosines.append((l1, l2, cos))

        # Summary stats
        cos_vals = [c[2] for c in cosines]
        min_cos = min(cos_vals)
        mean_cos = np.mean(cos_vals)

        # Find any low-similarity transitions
        low_trans = [(l1, l2, c) for l1, l2, c in cosines if c < 0.8]

        print(f"  {dir_type}: mean={mean_cos:.3f}, min={min_cos:.3f}")
        if low_trans:
            for l1, l2, c in low_trans[:3]:  # Show up to 3
                print(f"    L{l1}->L{l2}: {c:.3f}")

    # Check temporal ordering
    print("\nTEMPORAL ORDERING (formation layers):")
    formation_order = sorted(direction_info.items(), key=lambda x: x[1].formation_layer)
    for i, (name, info) in enumerate(formation_order):
        print(f"  {i+1}. {name}: forms at L{info.formation_layer}, peaks at L{info.peak_layer}")

    expected_order = ["answer", "uncertainty", "confidence"]
    actual_order = [name for name, _ in formation_order]
    if actual_order == expected_order:
        print("  -> Consistent with Answer -> Uncertainty -> Confidence hypothesis")
    else:
        print(f"  -> Actual order: {' -> '.join(actual_order)}")

    # Load questions
    print("\nLoading questions...")
    questions = load_questions(INPUT_BASE_NAME, NUM_QUESTIONS)
    print(f"  Loaded {len(questions)} questions")

    # Load model
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Layers: {num_layers}")
    print(f"  Chat template: {use_chat_template}")

    # Prepare prompts (use meta-task prompts for the forward passes)
    print("\nPreparing meta-task prompts...")
    prompts = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "confidence":
            prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template=use_chat_template)
        elif META_TASK == "delegate":
            prompt, _, _ = format_answer_or_delegate_prompt(question, tokenizer, trial_index=q_idx, use_chat_template=use_chat_template)
        elif META_TASK == "other_confidence":
            prompt, _ = format_other_confidence_prompt(question, tokenizer, use_chat_template=use_chat_template)
        else:
            raise ValueError(f"Unknown meta task: {META_TASK}")
        prompts.append(prompt)

    # ==========================================================================
    # EXTRACT BASELINE PROJECTIONS FOR ALL LAYERS
    # ==========================================================================
    print("\n" + "=" * 70)
    print("EXTRACTING BASELINE PROJECTIONS FOR ALL LAYERS")
    print("=" * 70)

    # Get all layers where any direction exists
    all_direction_layers = set()
    for dir_type, dir_info in direction_info.items():
        all_direction_layers.update(dir_info.directions.keys())
    all_direction_layers = sorted(all_direction_layers)
    print(f"  Extracting at {len(all_direction_layers)} layers: {min(all_direction_layers)}-{max(all_direction_layers)}")

    # Extract baseline activations at ALL layers (single forward pass, no ablation)
    baseline_acts_all = extract_baseline_at_layers(
        model, tokenizer, prompts, all_direction_layers, batch_size=BATCH_SIZE
    )

    # Compute and store baseline projections for each direction at each layer
    baseline_projections = {}  # {dir_type: {layer: {mean, std, normalized_mean, normalized_std}}}

    for dir_type, dir_info in direction_info.items():
        baseline_projections[dir_type] = {}
        for layer in all_direction_layers:
            if layer not in dir_info.directions:
                continue

            direction = dir_info.directions[layer]
            acts = baseline_acts_all[layer]

            # Compute raw projection
            proj = compute_projection(acts, direction)

            # Compute normalized projection
            norms = np.maximum(np.linalg.norm(acts, axis=1), MIN_NORM)
            proj_normalized = proj / norms

            # Bootstrap for CIs
            rng = np.random.RandomState(SEED)
            n = len(proj)
            boot_means = []
            boot_norm_means = []
            for _ in range(BOOTSTRAP_N):
                idx = rng.choice(n, size=n, replace=True)
                boot_means.append(np.mean(proj[idx]))
                boot_norm_means.append(np.mean(proj_normalized[idx]))

            baseline_projections[dir_type][layer] = {
                "mean": float(np.mean(proj)),
                "std": float(np.std(proj)),
                "ci_low": float(np.percentile(boot_means, 100 * BOOTSTRAP_CI_ALPHA / 2)),
                "ci_high": float(np.percentile(boot_means, 100 * (1 - BOOTSTRAP_CI_ALPHA / 2))),
                "normalized_mean": float(np.mean(proj_normalized)),
                "normalized_std": float(np.std(proj_normalized)),
                "normalized_ci_low": float(np.percentile(boot_norm_means, 100 * BOOTSTRAP_CI_ALPHA / 2)),
                "normalized_ci_high": float(np.percentile(boot_norm_means, 100 * (1 - BOOTSTRAP_CI_ALPHA / 2))),
                "n_samples": n,
            }

        print(f"  {dir_type}: {len(baseline_projections[dir_type])} layers")

    # Clean up to free memory
    del baseline_acts_all

    # Run cross-direction experiments
    print("\n" + "=" * 70)
    print("RUNNING CROSS-DIRECTION CAUSAL EXPERIMENTS")
    print("=" * 70)

    results = {}

    # Per-sample data for correlation analysis
    per_sample_data = {}  # {(ablate_type, ablate_layer, measure_type, measure_layer): {...}}

    # For each ablation type, test effects on all measurement types
    for ablate_type, ablate_info in direction_info.items():
        ablate_layers = ablate_info.meaningful_layers

        for ablate_layer in ablate_layers:
            if ablate_layer not in ablate_info.directions:
                continue

            print(f"\n--- Ablate {ablate_type} @ layer {ablate_layer} ---")
            ablate_direction = ablate_info.directions[ablate_layer]

            # Determine measurement layers (downstream of ablation, where targets are meaningful)
            all_measure_layers = set()
            for measure_type, measure_info in direction_info.items():
                for ml in measure_info.meaningful_layers:
                    if ml >= ablate_layer + BUFFER_LAYERS:
                        all_measure_layers.add(ml)

            extract_layers = sorted(all_measure_layers)
            if not extract_layers:
                print(f"  No valid measurement layers downstream of L{ablate_layer}")
                continue

            print(f"  Measuring at layers: {extract_layers}")

            # Check if we need per-sample data for this ablation type
            need_per_sample = SAVE_PER_SAMPLE_DATA and any(
                ablate_type == pair[0] for pair in PER_SAMPLE_PAIRS
            )

            # Run forward passes
            baseline_acts, ablated_acts = run_with_ablation_and_extraction(
                model, tokenizer, prompts,
                ablate_direction, ablate_layer, extract_layers,
                batch_size=BATCH_SIZE,
                also_extract_ablate_layer=need_per_sample,
            )

            # Measure projection onto all direction types at each measurement layer
            for measure_type, measure_info in direction_info.items():
                for measure_layer in extract_layers:
                    if measure_layer not in measure_info.directions:
                        continue
                    if measure_layer not in measure_info.meaningful_layers:
                        continue

                    measure_dir = measure_info.directions[measure_layer]

                    baseline_proj = compute_projection(baseline_acts[measure_layer], measure_dir)
                    ablated_proj = compute_projection(ablated_acts[measure_layer], measure_dir)

                    # Bootstrap raw projections (existing)
                    effect = bootstrap_effect(
                        baseline_proj, ablated_proj,
                        n_bootstrap=BOOTSTRAP_N,
                        ci_alpha=BOOTSTRAP_CI_ALPHA,
                        seed=SEED,
                    )

                    # Compute normalized projections if enabled
                    if COMPUTE_NORMALIZED_PROJECTIONS:
                        # Compute baseline activation norms (common reference for normalization)
                        baseline_norms = np.maximum(
                            np.linalg.norm(baseline_acts[measure_layer], axis=1),
                            MIN_NORM
                        )

                        # Normalized projections - divide BOTH by baseline norm for fair comparison
                        # This gives: how much does the projection change, relative to baseline activation magnitude?
                        baseline_normalized = baseline_proj / baseline_norms
                        ablated_normalized = ablated_proj / baseline_norms  # Same denominator!

                        # Bootstrap normalized projections to get proper CIs
                        effect_norm = bootstrap_effect(
                            baseline_normalized, ablated_normalized,
                            n_bootstrap=BOOTSTRAP_N,
                            ci_alpha=BOOTSTRAP_CI_ALPHA,
                            seed=SEED,
                        )

                        # Store norm statistics for diagnostics
                        ablated_norms = np.linalg.norm(ablated_acts[measure_layer], axis=1)
                        effect["norm_stats"] = {
                            "baseline_norm_mean": float(np.mean(baseline_norms)),
                            "baseline_norm_std": float(np.std(baseline_norms)),
                            "ablated_norm_mean": float(np.mean(ablated_norms)),
                            "ablated_norm_std": float(np.std(ablated_norms)),
                        }

                        # Store normalized projection stats with bootstrap CIs
                        effect["normalized"] = {
                            "baseline_mean": effect_norm["baseline_mean"],
                            "baseline_std": effect_norm["baseline_std"],
                            "ablated_mean": effect_norm["ablated_mean"],
                            "delta_mean": effect_norm["delta_mean"],
                            "delta_ci_low": effect_norm["delta_ci_low"],
                            "delta_ci_high": effect_norm["delta_ci_high"],
                            "cohens_d": effect_norm["cohens_d"],
                            "p_value": effect_norm["p_value"],
                        }

                    key = (ablate_type, ablate_layer, measure_type, measure_layer)
                    results[key] = effect

                    # Save per-sample data if this pair is requested
                    if SAVE_PER_SAMPLE_DATA and (ablate_type, measure_type) in PER_SAMPLE_PAIRS:
                        # Get baseline projection onto ablated direction at ablation layer
                        if ablate_layer in baseline_acts:
                            baseline_ablate_proj = compute_projection(
                                baseline_acts[ablate_layer], ablate_direction
                            )
                        else:
                            baseline_ablate_proj = None

                        per_sample_data[key] = {
                            "baseline_ablate_proj": baseline_ablate_proj,  # Proj onto ablated dir at ablate layer
                            "baseline_measure_proj": baseline_proj,  # Proj onto measure dir at measure layer
                            "ablated_measure_proj": ablated_proj,  # After ablation
                            "delta_measure_proj": ablated_proj - baseline_proj,  # Change
                        }

            # Print summary for this ablation
            summary_parts = []
            for measure_type in direction_info.keys():
                # Find strongest effect on this measure type
                relevant = [(k, v) for k, v in results.items()
                            if k[0] == ablate_type and k[1] == ablate_layer and k[2] == measure_type]
                if relevant:
                    best = max(relevant, key=lambda x: abs(x[1]["cohens_d"]))
                    d = best[1]["cohens_d"]
                    sig = "*" if best[1]["p_value"] < 0.05 else ""
                    summary_parts.append(f"{measure_type[:3]}={d:+.2f}{sig}")

            if summary_parts:
                print(f"  Effects: {', '.join(summary_parts)}")

    # Apply FDR correction
    print("\nApplying FDR correction...")
    results = apply_fdr_correction(results, alpha=BOOTSTRAP_CI_ALPHA)
    n_sig = sum(1 for v in results.values() if isinstance(v, dict) and v.get("significant_fdr", False))
    print(f"  {n_sig}/{len(results)} effects significant after FDR correction")

    # ==========================================================================
    # CONTROL EXPERIMENT: Ablate random directions to establish baseline
    # ==========================================================================
    control_effects = []  # Collect all |d| from control ablations
    control_effects_normalized = []  # Collect |delta| from normalized projections
    control_effects_by_gap = {}  # Track by layer gap: {gap: [deltas]}
    control_effects_normalized_by_gap = {}  # Track normalized by gap

    if RUN_CONTROL_EXPERIMENT:
        print("\n" + "=" * 70)
        print("CONTROL EXPERIMENT: Ablating random directions")
        print("=" * 70)
        print(f"  {N_CONTROL_DIRECTIONS} random directions per ablation layer")

        control_rng = np.random.RandomState(SEED + 999)  # Separate seed for control

        # Get hidden dimension from first direction we have
        first_dir_type = list(direction_info.keys())[0]
        first_layer = list(direction_info[first_dir_type].directions.keys())[0]
        hidden_dim = len(direction_info[first_dir_type].directions[first_layer])

        # Collect all ablation layers used in real experiment
        all_ablate_layers = set()
        for k in results.keys():
            if isinstance(k, tuple):
                all_ablate_layers.add(k[1])
        all_ablate_layers = sorted(all_ablate_layers)

        # Run control ablations
        for ctrl_idx in range(N_CONTROL_DIRECTIONS):
            print(f"\n--- Control direction {ctrl_idx + 1}/{N_CONTROL_DIRECTIONS} ---")

            for ablate_layer in all_ablate_layers:
                # Generate random direction (orthogonal to nothing - truly random)
                random_dir = generate_random_direction(hidden_dim, control_rng)

                # Determine measurement layers (same logic as real experiment)
                all_measure_layers = set()
                for measure_type, measure_info in direction_info.items():
                    for ml in measure_info.meaningful_layers:
                        if ml >= ablate_layer + BUFFER_LAYERS:
                            all_measure_layers.add(ml)

                extract_layers = sorted(all_measure_layers)
                if not extract_layers:
                    continue

                # Run forward passes with random direction ablation
                baseline_acts, ablated_acts = run_with_ablation_and_extraction(
                    model, tokenizer, prompts,
                    random_dir, ablate_layer, extract_layers,
                    batch_size=BATCH_SIZE,
                )

                # Measure effects on all real direction types
                for measure_type, measure_info in direction_info.items():
                    for measure_layer in extract_layers:
                        if measure_layer not in measure_info.directions:
                            continue
                        if measure_layer not in measure_info.meaningful_layers:
                            continue

                        measure_dir = measure_info.directions[measure_layer]
                        baseline_proj = compute_projection(baseline_acts[measure_layer], measure_dir)
                        ablated_proj = compute_projection(ablated_acts[measure_layer], measure_dir)

                        effect = bootstrap_effect(
                            baseline_proj, ablated_proj,
                            n_bootstrap=BOOTSTRAP_N,
                            ci_alpha=BOOTSTRAP_CI_ALPHA,
                            seed=SEED,
                        )

                        control_effects.append(abs(effect["cohens_d"]))

                        # Track by layer gap
                        gap = measure_layer - ablate_layer
                        if gap not in control_effects_by_gap:
                            control_effects_by_gap[gap] = []
                        control_effects_by_gap[gap].append(abs(effect["delta_mean"]))

                        # Compute normalized projections for control (same logic as real experiment)
                        if COMPUTE_NORMALIZED_PROJECTIONS:
                            control_baseline_norms = np.maximum(
                                np.linalg.norm(baseline_acts[measure_layer], axis=1),
                                MIN_NORM
                            )
                            control_baseline_normalized = baseline_proj / control_baseline_norms
                            control_ablated_normalized = ablated_proj / control_baseline_norms

                            effect_norm = bootstrap_effect(
                                control_baseline_normalized, control_ablated_normalized,
                                n_bootstrap=BOOTSTRAP_N,
                                ci_alpha=BOOTSTRAP_CI_ALPHA,
                                seed=SEED,
                            )

                            control_effects_normalized.append(abs(effect_norm["delta_mean"]))

                            # Track normalized by gap
                            if gap not in control_effects_normalized_by_gap:
                                control_effects_normalized_by_gap[gap] = []
                            control_effects_normalized_by_gap[gap].append(abs(effect_norm["delta_mean"]))

            print(f"  Collected {len(control_effects)} control effect measurements")

        # Compute control distribution statistics
        if control_effects:
            control_mean = np.mean(control_effects)
            control_std = np.std(control_effects)
            control_p95 = np.percentile(control_effects, 95)
            print(f"\nCONTROL DISTRIBUTION (Cohen's d):")
            print(f"  Mean |d|: {control_mean:.3f}")
            print(f"  Std |d|:  {control_std:.3f}")
            print(f"  95th percentile: {control_p95:.3f}")
            print(f"  Threshold (mean + 2*std): {control_mean + 2 * control_std:.3f}")

        # Compute normalized control statistics
        if control_effects_normalized:
            control_norm_mean = np.mean(control_effects_normalized)
            control_norm_std = np.std(control_effects_normalized)
            control_norm_p95 = np.percentile(control_effects_normalized, 95)
            print(f"\nCONTROL DISTRIBUTION (Normalized |delta|):")
            print(f"  Mean |delta|: {control_norm_mean:.4f}")
            print(f"  Std |delta|:  {control_norm_std:.4f}")
            print(f"  95th percentile: {control_norm_p95:.4f}")
            print(f"  Threshold (mean + 2*std): {control_norm_mean + 2 * control_norm_std:.4f}")
        else:
            control_norm_mean = None
            control_norm_std = None
            control_norm_p95 = None

        # Compute per-gap statistics (key diagnostic: does control effect grow with gap?)
        control_by_gap_stats = {}
        control_by_gap_normalized_stats = {}

        if control_effects_by_gap:
            print(f"\nCONTROL EFFECTS BY LAYER GAP (raw |delta|):")
            for gap in sorted(control_effects_by_gap.keys()):
                deltas = control_effects_by_gap[gap]
                gap_stats = {
                    "mean_abs_delta": float(np.mean(deltas)),
                    "std_abs_delta": float(np.std(deltas)),
                    "n": len(deltas),
                }
                control_by_gap_stats[gap] = gap_stats
                print(f"  gap={gap:2d}: mean |Δ|={gap_stats['mean_abs_delta']:.4f} ± {gap_stats['std_abs_delta']:.4f} (n={gap_stats['n']})")

        if control_effects_normalized_by_gap:
            print(f"\nCONTROL EFFECTS BY LAYER GAP (normalized |delta|):")
            for gap in sorted(control_effects_normalized_by_gap.keys()):
                deltas = control_effects_normalized_by_gap[gap]
                gap_stats = {
                    "mean_abs_delta": float(np.mean(deltas)),
                    "std_abs_delta": float(np.std(deltas)),
                    "n": len(deltas),
                }
                control_by_gap_normalized_stats[gap] = gap_stats
                print(f"  gap={gap:2d}: mean |Δ|={gap_stats['mean_abs_delta']:.6f} ± {gap_stats['std_abs_delta']:.6f} (n={gap_stats['n']})")
    else:
        control_mean = None
        control_std = None
        control_p95 = None
        control_norm_mean = None
        control_norm_std = None
        control_norm_p95 = None
        control_by_gap_stats = {}
        control_by_gap_normalized_stats = {}
        print("\n(Control experiment skipped - set RUN_CONTROL_EXPERIMENT = True to enable)")

    # Compute real effects by gap (before saving to include in JSON)
    real_by_gap_diagonal = {}  # Diagonal: ablate X, measure X
    real_by_gap_cross = {}     # Cross: ablate X, measure Y

    for k, v in results.items():
        if not isinstance(k, tuple):
            continue
        gap = k[3] - k[1]  # measure_layer - ablate_layer
        delta = abs(v.get("delta_mean", 0))

        if k[0] == k[2]:  # Diagonal
            if gap not in real_by_gap_diagonal:
                real_by_gap_diagonal[gap] = []
            real_by_gap_diagonal[gap].append(delta)
        else:  # Cross-direction
            if gap not in real_by_gap_cross:
                real_by_gap_cross[gap] = []
            real_by_gap_cross[gap].append(delta)

    # Compute stats for JSON
    real_by_gap_diagonal_stats = {
        gap: {
            "mean_abs_delta": float(np.mean(deltas)),
            "std_abs_delta": float(np.std(deltas)),
            "n": len(deltas),
        }
        for gap, deltas in real_by_gap_diagonal.items()
    }
    real_by_gap_cross_stats = {
        gap: {
            "mean_abs_delta": float(np.mean(deltas)),
            "std_abs_delta": float(np.std(deltas)),
            "n": len(deltas),
        }
        for gap, deltas in real_by_gap_cross.items()
    }

    # Save results
    print("\nSaving results...")
    model_short = get_model_short_name(MODEL, LOAD_IN_4BIT, LOAD_IN_8BIT)
    # Extract dataset part from INPUT_BASE_NAME
    if "_" in INPUT_BASE_NAME:
        parts = INPUT_BASE_NAME.split("_", 1)
        dataset_name = parts[1] if len(parts) > 1 else INPUT_BASE_NAME
    else:
        dataset_name = INPUT_BASE_NAME
    # Include adapter in output name if used
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        output_base = f"{model_short}_adapter-{adapter_short}_{dataset_name}"
    else:
        output_base = f"{model_short}_{dataset_name}"

    # Convert tuple keys to strings for JSON
    json_results = {
        "config": get_config_dict(
            model=MODEL,
            adapter=ADAPTER,
            input_base=INPUT_BASE_NAME,
            metric=METRIC,
            meta_task=META_TASK,
            direction_types=DIRECTION_TYPES,
            r2_threshold=R2_THRESHOLD,
            buffer_layers=BUFFER_LAYERS,
            effect_size_threshold=EFFECT_SIZE_THRESHOLD,
            num_questions=len(questions),
            bootstrap_n=BOOTSTRAP_N,
            n_control_directions=N_CONTROL_DIRECTIONS if RUN_CONTROL_EXPERIMENT else 0,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "direction_info": {
            name: {
                "formation_layer": info.formation_layer,
                "peak_layer": info.peak_layer,
                "n_meaningful_layers": len(info.meaningful_layers),
                "meaningful_layers": info.meaningful_layers,
                "r2_at_peak": info.r2_by_layer.get(info.peak_layer, 0),
            }
            for name, info in direction_info.items()
        },
        "control_experiment": {
            "n_control_directions": N_CONTROL_DIRECTIONS if RUN_CONTROL_EXPERIMENT else 0,
            "n_measurements": len(control_effects) if control_effects else 0,
            "mean_abs_d": float(control_mean) if control_mean is not None else None,
            "std_abs_d": float(control_std) if control_std is not None else None,
            "p95_abs_d": float(control_p95) if control_p95 is not None else None,
            "threshold_2std": float(control_mean + 2 * control_std) if control_mean is not None else None,
            # Per-gap statistics (key for distinguishing real effects from artifacts)
            "by_layer_gap": {
                str(gap): stats for gap, stats in control_by_gap_stats.items()
            } if control_by_gap_stats else None,
            # Normalized control statistics
            "normalized": {
                "mean_abs_delta": float(control_norm_mean) if control_norm_mean is not None else None,
                "std_abs_delta": float(control_norm_std) if control_norm_std is not None else None,
                "p95_abs_delta": float(control_norm_p95) if control_norm_p95 is not None else None,
                "threshold_2std": float(control_norm_mean + 2 * control_norm_std) if control_norm_mean is not None else None,
                # Per-gap normalized statistics
                "by_layer_gap": {
                    str(gap): stats for gap, stats in control_by_gap_normalized_stats.items()
                } if control_by_gap_normalized_stats else None,
            } if COMPUTE_NORMALIZED_PROJECTIONS else None,
        },
        "baseline_projections": {
            dir_type: {
                str(layer): layer_data
                for layer, layer_data in dir_data.items()
            }
            for dir_type, dir_data in baseline_projections.items()
        },
        # Real effects by layer gap (for comparison with control)
        "real_effects_by_gap": {
            "diagonal": {
                str(gap): stats for gap, stats in real_by_gap_diagonal_stats.items()
            } if real_by_gap_diagonal_stats else None,
            "cross_direction": {
                str(gap): stats for gap, stats in real_by_gap_cross_stats.items()
            } if real_by_gap_cross_stats else None,
        },
        "results": {},
    }

    for key, value in results.items():
        if isinstance(key, tuple):
            str_key = f"{key[0]}_L{key[1]}_to_{key[2]}_L{key[3]}"
            json_results["results"][str_key] = value

    json_path = OUTPUT_DIR / f"{output_base}_cross_direction_{METRIC}_results.json"
    with open(json_path, "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Save per-sample data if collected
    if per_sample_data:
        npz_data = {}
        for key, data in per_sample_data.items():
            prefix = f"{key[0]}_L{key[1]}_to_{key[2]}_L{key[3]}"
            if data["baseline_ablate_proj"] is not None:
                npz_data[f"{prefix}_baseline_ablate_proj"] = data["baseline_ablate_proj"]
            npz_data[f"{prefix}_baseline_measure_proj"] = data["baseline_measure_proj"]
            npz_data[f"{prefix}_ablated_measure_proj"] = data["ablated_measure_proj"]
            npz_data[f"{prefix}_delta_measure_proj"] = data["delta_measure_proj"]

        npz_path = OUTPUT_DIR / f"{output_base}_cross_direction_{METRIC}_per_sample.npz"
        np.savez(npz_path, **npz_data)
        print(f"  Saved per-sample data: {npz_path}")
        print(f"    Pairs saved: {list(per_sample_data.keys())}")

    # Plot R^2 curves
    r2_plot_path = OUTPUT_DIR / f"{output_base}_cross_direction_r2_curves.png"
    plot_r2_curves(direction_info, R2_THRESHOLD, r2_plot_path)

    # Plot propagation heatmaps
    heatmap_path = OUTPUT_DIR / f"{output_base}_cross_direction_{METRIC}_heatmaps.png"
    plot_propagation_heatmaps(results, direction_info, heatmap_path)

    # Plot flow summary
    flow_path = OUTPUT_DIR / f"{output_base}_cross_direction_{METRIC}_flow.png"
    plot_flow_summary(results, direction_info, flow_path,
                      control_mean_delta=control_mean if control_mean is not None else None)

    # Plot diagonal effects heatmaps
    diag_path = OUTPUT_DIR / f"{output_base}_cross_direction_{METRIC}_diagonal.png"
    plot_diagonal_effects(results, direction_info, diag_path)

    # Print interpretation
    print("\n" + "=" * 70)
    print("INFORMATION FLOW ANALYSIS")
    print("=" * 70)

    # Analyze cross-direction effects
    print("\nCROSS-DIRECTION CAUSAL EFFECTS:")
    if control_mean is not None:
        control_threshold = control_mean + 2 * control_std
        print(f"  (Cohen's d, * = FDR sig, + = exceeds control threshold {control_threshold:.2f})")
    else:
        control_threshold = None
        print(f"  (Cohen's d, * = FDR significant)")
    print()

    flow_evidence = []
    for ablate_type in direction_info.keys():
        for measure_type in direction_info.keys():
            if ablate_type == measure_type:
                continue

            # Find effects for this pair
            pair_effects = [(k, v) for k, v in results.items()
                           if isinstance(k, tuple) and k[0] == ablate_type and k[2] == measure_type]

            if not pair_effects:
                print(f"  {ablate_type:11} -> {measure_type:11}: No data")
                continue

            # Find strongest FDR-significant effect
            sig_effects = [(k, v) for k, v in pair_effects if v.get("significant_fdr", False)]

            if sig_effects:
                best = max(sig_effects, key=lambda x: abs(x[1]["cohens_d"]))
                k, v = best
                d = v["cohens_d"]
                abl_l, meas_l = k[1], k[3]
                direction = "reduces" if d < 0 else "increases"

                # Check if exceeds control threshold
                exceeds_ctrl = "+" if (control_threshold and abs(d) > control_threshold) else ""
                print(f"  {ablate_type:11} -> {measure_type:11}: "
                      f"d={d:+.3f}*{exceeds_ctrl} @ L{abl_l}->L{meas_l} ({direction} {measure_type})")

                # Only count as evidence if exceeds control threshold (or no control run)
                if control_threshold is None or abs(d) > control_threshold:
                    flow_evidence.append((ablate_type, measure_type, abl_l, meas_l, d))
            else:
                # Report best non-significant effect
                best = max(pair_effects, key=lambda x: abs(x[1]["cohens_d"]))
                k, v = best
                d = v["cohens_d"]
                abl_l, meas_l = k[1], k[3]
                print(f"  {ablate_type:11} -> {measure_type:11}: "
                      f"d={d:+.3f} @ L{abl_l}->L{meas_l} (not significant)")

    # Diagonal sanity checks - use mean d across all pairs, not max |d|
    print("\nDIAGONAL CHECKS (ablate X -> measure X, expect negative mean d):")
    for dir_type, info in direction_info.items():
        diag_effects = [(k, v) for k, v in results.items()
                       if isinstance(k, tuple) and k[0] == dir_type and k[2] == dir_type]
        if diag_effects:
            # Compute mean Cohen's d across all diagonal pairs
            all_d = [v["cohens_d"] for _, v in diag_effects]
            mean_d = np.mean(all_d)
            std_d = np.std(all_d)
            n_negative = sum(1 for d in all_d if d < 0)
            status = "pass" if mean_d < 0 else "FAIL"

            print(f"  {dir_type}: mean d={mean_d:+.3f} (std={std_d:.3f}), {n_negative}/{len(all_d)} negative [{status}]")

            if status == "FAIL":
                worst = sorted(diag_effects, key=lambda x: x[1]["cohens_d"], reverse=True)[:3]
                for k, v in worst:
                    print(f"      Offender: L{k[1]}->L{k[3]}: d={v['cohens_d']:+.3f}")

    # Effect size by layer gap analysis (diagonal effects only)
    print("\nEFFECT SIZE BY LAYER GAP (diagonal only):")
    gap_effects = {}  # gap -> list of |d| values
    for k, v in results.items():
        if not isinstance(k, tuple) or k[0] != k[2]:  # Skip non-diagonal
            continue
        gap = k[3] - k[1]
        d = abs(v["cohens_d"])
        if gap not in gap_effects:
            gap_effects[gap] = []
        gap_effects[gap].append(d)

    for gap in sorted(gap_effects.keys())[:6]:  # Show first 6 gaps
        effects = gap_effects[gap]
        mean_d = np.mean(effects)
        print(f"  gap={gap}: mean |d|={mean_d:.2f}, n={len(effects)}")

    # Interpret flow
    if flow_evidence:
        print("\nINFERRED INFORMATION FLOW:")

        # Check for Answer -> Uncertainty
        ans_to_unc = [(a, m, al, ml, d) for a, m, al, ml, d in flow_evidence
                      if a == "answer" and m == "uncertainty"]
        unc_to_ans = [(a, m, al, ml, d) for a, m, al, ml, d in flow_evidence
                      if a == "uncertainty" and m == "answer"]

        if ans_to_unc and not unc_to_ans:
            e = ans_to_unc[0]
            print(f"  Answer -> Uncertainty: SUPPORTED (d={e[4]:+.3f} at L{e[2]}->L{e[3]})")
        elif unc_to_ans and not ans_to_unc:
            e = unc_to_ans[0]
            print(f"  Uncertainty -> Answer: unexpected reverse flow (d={e[4]:+.3f})")
        elif ans_to_unc and unc_to_ans:
            a2u = max(ans_to_unc, key=lambda x: abs(x[4]))
            u2a = max(unc_to_ans, key=lambda x: abs(x[4]))
            if abs(a2u[4]) > abs(u2a[4]):
                print(f"  Answer -> Uncertainty: LIKELY (|d|={abs(a2u[4]):.3f} > {abs(u2a[4]):.3f})")
            else:
                print(f"  Answer <-> Uncertainty: BIDIRECTIONAL or shared representation")

        # Check for Uncertainty -> Confidence
        unc_to_conf = [(a, m, al, ml, d) for a, m, al, ml, d in flow_evidence
                       if a == "uncertainty" and m == "confidence"]
        conf_to_unc = [(a, m, al, ml, d) for a, m, al, ml, d in flow_evidence
                       if a == "confidence" and m == "uncertainty"]

        if unc_to_conf and not conf_to_unc:
            e = unc_to_conf[0]
            print(f"  Uncertainty -> Confidence: SUPPORTED (d={e[4]:+.3f} at L{e[2]}->L{e[3]})")
        elif conf_to_unc and not unc_to_conf:
            e = conf_to_unc[0]
            print(f"  Confidence -> Uncertainty: unexpected reverse flow (d={e[4]:+.3f})")
        elif unc_to_conf and conf_to_unc:
            u2c = max(unc_to_conf, key=lambda x: abs(x[4]))
            c2u = max(conf_to_unc, key=lambda x: abs(x[4]))
            if abs(u2c[4]) > abs(c2u[4]):
                print(f"  Uncertainty -> Confidence: LIKELY (|d|={abs(u2c[4]):.3f} > {abs(c2u[4]):.3f})")
            else:
                print(f"  Uncertainty <-> Confidence: BIDIRECTIONAL or shared representation")

    # Summary comparison: real vs control
    if control_mean is not None:
        print("\nREAL vs CONTROL COMPARISON:")
        # Collect all real effect sizes
        real_effects = [abs(v["cohens_d"]) for v in results.values()
                       if isinstance(v, dict) and "cohens_d" in v]
        real_mean = np.mean(real_effects) if real_effects else 0
        real_max = max(real_effects) if real_effects else 0

        # Count how many real effects exceed control threshold
        n_exceed = sum(1 for d in real_effects if d > control_threshold)

        print(f"  Control: mean |d| = {control_mean:.3f}, 95th pctl = {control_p95:.3f}")
        print(f"  Real:    mean |d| = {real_mean:.3f}, max |d| = {real_max:.3f}")
        print(f"  Effects exceeding control threshold: {n_exceed}/{len(real_effects)}")

        if real_max > control_p95:
            print(f"  -> Strongest real effect ({real_max:.2f}) exceeds 95th percentile of control ({control_p95:.2f})")
        else:
            print(f"  -> WARNING: No real effects clearly exceed control distribution")

    # Per-gap comparison: real vs control (key diagnostic for artifact vs real effect)
    if control_by_gap_stats:
        print("\nPER-GAP COMPARISON: REAL vs CONTROL (raw |delta|):")
        print("  (If control stays flat while real grows, effect is real)")
        print("  (If both grow with gap, effect may be artifact)")
        print()

        # Print comparison table (using pre-computed stats)
        all_gaps = sorted(set(control_by_gap_stats.keys()) |
                         set(real_by_gap_diagonal_stats.keys()) |
                         set(real_by_gap_cross_stats.keys()))

        print(f"  {'Gap':>4} | {'Control':>10} | {'Real Diag':>10} | {'Real Cross':>10} | {'Diag/Ctrl':>10}")
        print(f"  {'-'*4}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

        for gap in all_gaps[:10]:  # Show first 10 gaps
            ctrl = control_by_gap_stats.get(gap, {}).get("mean_abs_delta", float('nan'))
            diag = real_by_gap_diagonal_stats.get(gap, {}).get("mean_abs_delta", float('nan'))
            cross = real_by_gap_cross_stats.get(gap, {}).get("mean_abs_delta", float('nan'))
            ratio = diag / ctrl if ctrl > 0 and not np.isnan(diag) else float('nan')

            print(f"  {gap:4d} | {ctrl:10.4f} | {diag:10.4f} | {cross:10.4f} | {ratio:10.1f}x")

        # Compute trends
        if len(control_by_gap_stats) >= 3:
            ctrl_gaps = sorted(control_by_gap_stats.keys())
            ctrl_means = [control_by_gap_stats[g]["mean_abs_delta"] for g in ctrl_gaps]
            ctrl_corr = np.corrcoef(ctrl_gaps, ctrl_means)[0, 1] if len(ctrl_gaps) > 1 else 0

            print(f"\n  Control trend (corr gap vs |Δ|): {ctrl_corr:+.3f}")

            if real_by_gap_diagonal_stats and len(real_by_gap_diagonal_stats) >= 3:
                diag_gaps = sorted(real_by_gap_diagonal_stats.keys())
                diag_means = [real_by_gap_diagonal_stats[g]["mean_abs_delta"] for g in diag_gaps]
                diag_corr = np.corrcoef(diag_gaps, diag_means)[0, 1] if len(diag_gaps) > 1 else 0
                print(f"  Real diagonal trend: {diag_corr:+.3f}")

            if real_by_gap_cross_stats and len(real_by_gap_cross_stats) >= 3:
                cross_gaps = sorted(real_by_gap_cross_stats.keys())
                cross_means = [real_by_gap_cross_stats[g]["mean_abs_delta"] for g in cross_gaps]
                cross_corr = np.corrcoef(cross_gaps, cross_means)[0, 1] if len(cross_gaps) > 1 else 0
                print(f"  Real cross-direction trend: {cross_corr:+.3f}")

            # Interpretation
            print()
            if abs(ctrl_corr) < 0.3:
                print("  -> Control effects are FLAT across gaps (good)")
            else:
                print(f"  -> WARNING: Control effects vary with gap (corr={ctrl_corr:+.2f})")

    print("\n" + "=" * 70)
    print("OUTPUT FILES:")
    print("=" * 70)
    print(f"  {json_path.name}")
    print(f"  {r2_plot_path.name}")
    print(f"  {heatmap_path.name}")
    print(f"  {flow_path.name}")
    print(f"  {diag_path.name}")


if __name__ == "__main__":
    main()
