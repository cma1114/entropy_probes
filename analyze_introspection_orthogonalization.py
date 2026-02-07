"""
Analysis: Isolating True Introspection from Surface Difficulty Cues.

Orthogonalizes self-confidence directions with respect to other-confidence
directions to isolate "pure introspection" (d_introspection) and "pure surface
difficulty cues" (d_surface). Tests whether these residual directions still
work for prediction, steering, and ablation.

Hypothesis:
- d_self = introspection + surface difficulty
- d_other = primarily surface difficulty
- d_introspection = d_self - proj(d_self, d_other) = pure introspection
- d_surface = d_other - proj(d_other, d_self) = pure surface cues

Inputs:
    outputs/{base}_meta_confidence_metaconfdir_directions.npz    Self-confidence directions
    outputs/{base}_meta_other_confidence_metaconfdir_directions.npz  Other-confidence directions
    outputs/{base}_meta_confidence_activations.npz               Self-confidence activations
    outputs/{base}_meta_other_confidence_activations.npz         Other-confidence activations
    outputs/{base}_mc_dataset.json                               Question metadata + metric values

Outputs:
    outputs/{base}_orthogonal_directions.npz                     d_introspection + d_surface vectors
    outputs/{base}_orthogonal_analysis_results.json              Full results
    outputs/{base}_orthogonal_similarity.png                     Cosine similarity by layer
    outputs/{base}_orthogonal_predictive.png                     Predictive power heatmap
    outputs/{base}_orthogonal_steering.png                       Steering effect heatmap
    outputs/{base}_orthogonal_ablation.png                       Ablation effect heatmap

Shared parameters (must match across scripts):
    SEED, TRAIN_SPLIT, PROBE_ALPHA, PROBE_PCA_COMPONENTS, MEAN_DIFF_QUANTILE

Run after:
    test_meta_transfer.py with META_TASK="confidence" and FIND_CONFIDENCE_DIRECTIONS=True
    test_meta_transfer.py with META_TASK="other_confidence" and FIND_CONFIDENCE_DIRECTIONS=True
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from core.config_utils import get_config_dict
from core.plotting import save_figure, GRID_ALPHA
from core.steering import generate_orthogonal_directions
from core.steering_experiments import (
    BatchSteeringHook,
    BatchAblationHook,
    pretokenize_prompts,
    build_padded_gpu_batches,
    get_kv_cache,
    create_fresh_cache,
)
from tasks import (
    format_stated_confidence_prompt,
    format_other_confidence_prompt,
    get_stated_confidence_signal,
    get_other_confidence_signal,
    STATED_CONFIDENCE_OPTIONS,
    OTHER_CONFIDENCE_OPTIONS,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# --- Model & Data ---
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = "Tristan-Day/ect_20251222_215412_v0uei7y1_2000"###None 
INPUT_BASE_NAME = "Llama-3.1-8B-Instruct_TriviaMC_difficulty_filtered"
METRIC = "logit_gap"  # Uncertainty metric for correlation measurement

# --- Quantization ---
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

# --- Experiment ---
SEED = 42                    # Must match across scripts
TRAIN_SPLIT = 0.8            # Must match across scripts
BATCH_SIZE = 4
NUM_CONTROLS = 100#500           # Random orthogonal directions per direction type for null (min p ≈ 0.002)
N_BOOTSTRAP = 100            # For CIs

# --- Direction-finding (must match across scripts) ---
PROBE_ALPHA = 1000.0         # Must match across scripts
PROBE_PCA_COMPONENTS = 100   # Must match across scripts
MEAN_DIFF_QUANTILE = 0.25    # Must match across scripts

# --- Orthogonalization ---
MIN_RESIDUAL_NORM = 0.1      # Flag layers where d_self ~ d_other
METHOD = "mean_diff"         # Direction method to use (parameter, default "mean_diff")

# --- Steering ---
STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
EXPANDED_BATCH_TARGET = 48

# --- Layer Selection for Causal Experiments ---
CAUSAL_LAYER_MODE = "top_k"  # "all", "top_k", or "explicit"
CAUSAL_TOP_K = 8             # Number of top layers by R² (if mode="top_k")
CAUSAL_EXPLICIT_LAYERS = []  # Explicit layer list (if mode="explicit")

# --- Output ---
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# Fixed seed offsets for direction types (replaces unstable hash(dt))
DT_SEED_OFFSET = {"d_self": 11, "d_other": 22, "d_introspection": 33, "d_surface": 44}

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# FDR CORRECTION
# =============================================================================

def _bh_fdr(pvals: Dict) -> Dict:
    """
    Benjamini-Hochberg FDR correction.

    Args:
        pvals: dict mapping key -> raw p-value

    Returns:
        dict mapping key -> FDR-adjusted p-value
    """
    if not pvals:
        return {}

    items = sorted(pvals.items(), key=lambda kv: kv[1])
    n = len(items)

    adj = {}
    for rank, (key, p) in enumerate(items, 1):
        adj[key] = min(1.0, (p * n) / rank)

    # Enforce monotonicity (non-decreasing in sorted order)
    prev = 0.0
    for key, p in items:
        adj[key] = max(adj[key], prev)
        prev = adj[key]

    return adj


# =============================================================================
# TOKEN HELPERS
# =============================================================================

def get_single_token_ids(tokenizer, options: List[str]) -> List[int]:
    """
    Get token IDs for options, asserting each is a single token.

    Args:
        tokenizer: HuggingFace tokenizer
        options: List of option strings (e.g., ["A", "B", "C", "D"])

    Returns:
        List of token IDs

    Raises:
        AssertionError: If any option encodes to multiple tokens
    """
    token_ids = []
    for opt in options:
        ids = tokenizer.encode(opt, add_special_tokens=False)
        assert len(ids) == 1, f"Option '{opt}' encodes to {len(ids)} tokens, expected 1"
        token_ids.append(ids[0])
    return token_ids


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OnlineCorrelation:
    """
    Streaming Pearson correlation accumulator using sufficient statistics.

    Allows computing correlation without storing all data points.
    """
    n: int = 0
    sum_x: float = 0.0
    sum_y: float = 0.0
    sum_xx: float = 0.0
    sum_yy: float = 0.0
    sum_xy: float = 0.0

    def update(self, x: float, y: float):
        """Add a single observation."""
        self.n += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_xx += x * x
        self.sum_yy += y * y
        self.sum_xy += x * y

    def correlation(self) -> float:
        """Compute Pearson correlation from accumulated statistics."""
        if self.n < 2:
            return 0.0
        mean_x = self.sum_x / self.n
        mean_y = self.sum_y / self.n
        # Clamp to non-negative to handle floating-point cancellation errors
        var_x = max(0.0, self.sum_xx / self.n - mean_x ** 2)
        var_y = max(0.0, self.sum_yy / self.n - mean_y ** 2)
        if var_x < 1e-10 or var_y < 1e-10:
            return 0.0
        cov_xy = self.sum_xy / self.n - mean_x * mean_y
        return cov_xy / (np.sqrt(var_x) * np.sqrt(var_y))


@dataclass
class OrthogonalDirections:
    """Container for orthogonalized direction vectors at a single layer."""
    d_self: np.ndarray
    d_other: np.ndarray
    d_introspection: np.ndarray
    d_surface: np.ndarray
    cosine_similarity: float
    residual_norm: float
    degenerate: bool


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def load_confidence_directions(base_name: str, task: str, method: str) -> Dict[int, np.ndarray]:
    """
    Load confidence directions from meta-task confidence direction finding.

    Args:
        base_name: Base name for input files
        task: "confidence" or "other_confidence"
        method: "probe" or "mean_diff"

    Returns:
        Dict mapping layer -> normalized direction vector
    """
    path = OUTPUT_DIR / f"{base_name}_meta_{task}_metaconfdir_directions.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Confidence directions not found: {path}\n"
            f"Run: test_meta_transfer.py with META_TASK='{task}' and FIND_CONFIDENCE_DIRECTIONS=True"
        )

    data = np.load(path)
    directions = {}

    for key in data.files:
        if key.startswith("_"):
            continue  # Skip metadata

        parts = key.rsplit("_layer_", 1)
        if len(parts) != 2:
            continue

        method_name, layer_str = parts
        if method_name != method:
            continue

        try:
            layer = int(layer_str)
        except ValueError:
            continue

        direction = data[key].astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        directions[layer] = direction

    return directions


def load_meta_activations(base_name: str, task: str, position: str = "final") -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load cached meta-task activations.

    Args:
        base_name: Base name for input files
        task: "confidence" or "other_confidence"
        position: Token position to load (default "final")

    Returns:
        activations_by_layer: {layer: (n_samples, hidden_dim)}
        confidences: (n_samples,) stated confidence values
    """
    path = OUTPUT_DIR / f"{base_name}_meta_{task}_activations.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Meta-task activations not found: {path}\n"
            f"Run: test_meta_transfer.py with META_TASK='{task}'"
        )

    data = np.load(path)

    # Load confidences
    confidences = data["confidences"]

    # Load activations by layer
    # Try position-specific format first (layer_{idx}_{position}), then legacy (layer_{idx})
    activations = {}
    for key in data.files:
        # Position-specific format: layer_{idx}_{position}
        if key.startswith("layer_") and key.endswith(f"_{position}"):
            parts = key.replace(f"_{position}", "").replace("layer_", "")
            try:
                layer = int(parts)
                activations[layer] = data[key]
            except ValueError:
                continue
        # Legacy format: layer_{idx}
        elif key.startswith("layer_") and "_" not in key.replace("layer_", ""):
            try:
                layer = int(key.replace("layer_", ""))
                if layer not in activations:  # Don't overwrite position-specific
                    activations[layer] = data[key]
            except ValueError:
                continue

    if not activations:
        raise ValueError(f"No activations found for position '{position}' in {path}")

    return activations, confidences


def load_dataset(base_name: str) -> Dict:
    """Load dataset with questions and metric values."""
    path = OUTPUT_DIR / f"{base_name}_mc_dataset.json"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# ORTHOGONALIZATION
# =============================================================================

def orthogonalize_directions(
    d_self: np.ndarray,
    d_other: np.ndarray,
    min_residual_norm: float = 0.1,
    layer: int = 0,
    seed: int = SEED
) -> OrthogonalDirections:
    """
    Compute orthogonalized directions via Gram-Schmidt projection.

    d_introspection = d_self - proj(d_self, d_other)
    d_surface = d_other - proj(d_other, d_self)

    Args:
        d_self: Self-confidence direction (normalized)
        d_other: Other-confidence direction (normalized)
        min_residual_norm: Minimum norm threshold for degenerate detection
        layer: Layer index (for layer-specific fallback seeds in degenerate cases)
        seed: Base seed (for reproducibility)

    Returns:
        OrthogonalDirections dataclass with all vectors and metadata
    """
    # Ensure unit vectors
    d_self = d_self / np.linalg.norm(d_self)
    d_other = d_other / np.linalg.norm(d_other)

    # Cosine similarity
    cosine = float(np.dot(d_self, d_other))

    # Gram-Schmidt: introspection = self with other removed
    d_introspection = d_self - cosine * d_other
    introspection_norm = np.linalg.norm(d_introspection)

    # Gram-Schmidt: surface = other with self removed
    d_surface = d_other - cosine * d_self
    surface_norm = np.linalg.norm(d_surface)

    # Note: introspection_norm == surface_norm by symmetry (both = sqrt(1 - cosine^2))
    residual_norm = float(introspection_norm)
    degenerate = residual_norm < min_residual_norm

    # Normalize 
    d_introspection = d_introspection / introspection_norm
    d_surface = d_surface / surface_norm

    return OrthogonalDirections(
        d_self=d_self.astype(np.float32),
        d_other=d_other.astype(np.float32),
        d_introspection=d_introspection.astype(np.float32),
        d_surface=d_surface.astype(np.float32),
        cosine_similarity=cosine,
        residual_norm=residual_norm,
        degenerate=degenerate,
    )


def compute_all_orthogonalizations(
    d_self_by_layer: Dict[int, np.ndarray],
    d_other_by_layer: Dict[int, np.ndarray],
    min_residual_norm: float = 0.1,
    seed: int = SEED
) -> Dict[int, OrthogonalDirections]:
    """
    Compute orthogonalized directions for all layers.

    Returns:
        Dict mapping layer -> OrthogonalDirections
    """
    layers = sorted(set(d_self_by_layer.keys()) & set(d_other_by_layer.keys()))

    results = {}
    for layer in layers:
        results[layer] = orthogonalize_directions(
            d_self_by_layer[layer],
            d_other_by_layer[layer],
            min_residual_norm=min_residual_norm,
            layer=layer,
            seed=seed
        )

    return results


# =============================================================================
# PREDICTIVE EVALUATION
# =============================================================================

def evaluate_direction_predictive_power(
    train_activations: np.ndarray,
    train_targets: np.ndarray,
    test_activations: np.ndarray,
    test_targets: np.ndarray,
    direction: np.ndarray,
    n_bootstrap: int = 100,
    seed: int = 42
) -> Dict:
    """
    Evaluate how well a direction predicts a target variable (true out-of-sample).

    Fits OLS on train data, evaluates R² on test data with bootstrap CI.

    Args:
        train_activations: (n_train, hidden_dim) training activations
        train_targets: (n_train,) training target values
        test_activations: (n_test, hidden_dim) test activations
        test_targets: (n_test,) test target values
        direction: (hidden_dim,) unit vector
        n_bootstrap: Number of bootstrap iterations for CI
        seed: Random seed

    Returns:
        Dict with r2, r2_ci, pearson (on test set)
    """
    # Project onto direction
    train_proj = train_activations @ direction
    test_proj = test_activations @ direction

    # Check for degenerate cases
    if np.std(train_proj) < 1e-10 or np.std(train_targets) < 1e-10:
        return {"r2": 0.0, "r2_ci": [0.0, 0.0], "pearson": 0.0}
    if np.std(test_proj) < 1e-10 or np.std(test_targets) < 1e-10:
        return {"r2": 0.0, "r2_ci": [0.0, 0.0], "pearson": 0.0}

    # Fit OLS on TRAIN data: y = slope * proj + intercept
    train_corr, _ = pearsonr(train_proj, train_targets)
    if not np.isfinite(train_corr):
        return {"r2": 0.0, "r2_ci": [0.0, 0.0], "pearson": 0.0}

    train_proj_std = np.std(train_proj)
    train_y_std = np.std(train_targets)
    slope = train_corr * (train_y_std / train_proj_std)
    intercept = np.mean(train_targets) - slope * np.mean(train_proj)

    # Predict on TEST data
    test_predictions = slope * test_proj + intercept

    # Compute test R²
    r2 = r2_score(test_targets, test_predictions)
    if not np.isfinite(r2):
        r2 = 0.0
    r2 = max(-1.0, min(1.0, r2))  # Clip to reasonable range

    # Pearson correlation on TEST set (for reference)
    test_corr, _ = pearsonr(test_proj, test_targets)
    if not np.isfinite(test_corr):
        test_corr = 0.0

    # Bootstrap CI on test set
    rng = np.random.default_rng(seed)
    n_test = len(test_targets)
    r2s = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_test, n_test, replace=True)
        y_true = test_targets[idx]
        y_pred = test_predictions[idx]
        if np.std(y_true) > 1e-10:
            boot_r2 = r2_score(y_true, y_pred)
            if np.isfinite(boot_r2):
                r2s.append(float(np.clip(boot_r2, -1.0, 1.0)))

    if r2s:
        ci_low = float(np.percentile(r2s, 2.5))
        ci_high = float(np.percentile(r2s, 97.5))
    else:
        ci_low, ci_high = 0.0, 0.0

    return {
        "r2": float(r2),
        "r2_ci": [ci_low, ci_high],
        "pearson": float(test_corr),
    }


def compute_predictive_power_matrix(
    ortho_by_layer: Dict[int, OrthogonalDirections],
    self_activations: Dict[int, np.ndarray],
    other_activations: Dict[int, np.ndarray],
    self_confidences: np.ndarray,
    other_confidences: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_bootstrap: int = 100
) -> Dict:
    """
    Compute 4x2 predictive power matrix: (direction_type) x (task).

    Fits OLS on train set, evaluates R² on test set (true out-of-sample prediction).

    Direction types: d_self, d_other, d_introspection, d_surface
    Tasks: self (confidence), other (other_confidence)

    Returns:
        {
            "by_layer": {
                layer: {
                    (dir_type, task): {"r2": ..., "r2_ci": ..., "pearson": ...}
                }
            },
            "summary": {...}
        }
    """
    layers = sorted(ortho_by_layer.keys())

    results = {"by_layer": {}, "summary": {}}

    direction_types = ["d_self", "d_other", "d_introspection", "d_surface"]
    tasks = ["self", "other"]

    for layer in tqdm(layers, desc="Computing predictive power"):
        ortho = ortho_by_layer[layer]
        results["by_layer"][layer] = {}

        for dir_type in direction_types:
            direction = getattr(ortho, dir_type)

            for task in tasks:
                if task == "self":
                    train_acts = self_activations[layer][train_idx]
                    train_targets = self_confidences[train_idx]
                    test_acts = self_activations[layer][test_idx]
                    test_targets = self_confidences[test_idx]
                else:
                    train_acts = other_activations[layer][train_idx]
                    train_targets = other_confidences[train_idx]
                    test_acts = other_activations[layer][test_idx]
                    test_targets = other_confidences[test_idx]

                metrics = evaluate_direction_predictive_power(
                    train_acts, train_targets,
                    test_acts, test_targets,
                    direction,
                    n_bootstrap=n_bootstrap,
                    seed=SEED + layer * 100
                )

                results["by_layer"][layer][(dir_type, task)] = {
                    "r2": metrics["r2"],
                    "r2_ci": metrics["r2_ci"],
                    "pearson": metrics["pearson"],
                }

    # Compute summary statistics
    for dir_type in direction_types:
        for task in tasks:
            key = f"{dir_type}_{task}"
            r2s = [results["by_layer"][l][(dir_type, task)]["r2"]
                   for l in layers if not ortho_by_layer[l].degenerate]
            if r2s:
                results["summary"][key] = {
                    "mean_r2": float(np.mean(r2s)),
                    "max_r2": float(np.max(r2s)),
                    "peak_layer": layers[np.argmax([
                        results["by_layer"][l][(dir_type, task)]["r2"] for l in layers
                    ])],
                }

    return results


# =============================================================================
# META-TASK HELPERS
# =============================================================================

# Task configuration: maps task name to (format_fn, signal_fn, options)
TASK_CONFIG = {
    "self": (
        format_stated_confidence_prompt,
        lambda p, m: get_stated_confidence_signal(p),
        list(STATED_CONFIDENCE_OPTIONS.keys()),
    ),
    "other": (
        format_other_confidence_prompt,
        lambda p, m: get_other_confidence_signal(p),
        list(OTHER_CONFIDENCE_OPTIONS.keys()),
    ),
}


def get_task_config(task: str):
    """Get (format_fn, signal_fn, options) for task."""
    if task not in TASK_CONFIG:
        raise ValueError(f"Unknown task: {task}")
    return TASK_CONFIG[task]


# =============================================================================
# STEERING EXPERIMENTS (Batched by task for efficiency)
# =============================================================================

def run_steering_matrix(
    model,
    tokenizer,
    questions: List[Dict],
    ortho_by_layer: Dict[int, OrthogonalDirections],
    layers: List[int],
    use_chat_template: bool,
) -> Dict:
    """
    Run 4x2 steering experiment matrix with optimized batching.

    Batches all 4 directions together per (layer, task), then batches controls
    in chunks. This reduces forward passes from 136 to ~8 per batch per layer.

    Returns:
        {
            "by_layer": {
                layer: {
                    (dir_type, task): {"slope": ..., "control_slopes": ..., "p_value": ...}
                }
            },
            "summary": {...}
        }
    """
    direction_types = ["d_self", "d_other", "d_introspection", "d_surface"]
    tasks = ["self", "other"]
    n_dirs = len(direction_types)
    n_questions = len(questions)

    results = {"by_layer": {layer: {} for layer in layers}, "summary": {}}

    # Pre-compute parameters
    nonzero_multipliers = [m for m in STEERING_MULTIPLIERS if m != 0.0]
    k_mult = len(nonzero_multipliers)

    # For main directions: batch 4 dirs × k_mult multipliers
    main_expand = n_dirs * k_mult
    main_batch_size = max(1, min(BATCH_SIZE, EXPANDED_BATCH_TARGET // main_expand))

    # For controls: batch in chunks that fit EXPANDED_BATCH_TARGET
    ctrl_per_pass = max(1, EXPANDED_BATCH_TARGET // (k_mult * main_batch_size))

    # Generate per-direction control directions (orthogonal to each direction type)
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    controls_by_layer = {}
    for layer in layers:
        ortho = ortho_by_layer[layer]
        controls_by_layer[layer] = {}
        for dt in direction_types:
            direction = getattr(ortho, dt)
            # Use different seed per direction type for independence
            seed = SEED + layer * 1000 + DT_SEED_OFFSET[dt]
            controls = generate_orthogonal_directions(direction, NUM_CONTROLS, seed=seed)
            controls_by_layer[layer][dt] = [torch.tensor(c, dtype=dtype, device=DEVICE) for c in controls]

    for task in tasks:
        format_fn, signal_fn, options = get_task_config(task)
        option_token_ids = get_single_token_ids(tokenizer, options)

        # Format prompts for this task
        prompts = [format_fn(q, tokenizer, use_chat_template=use_chat_template)[0] for q in questions]
        cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
        gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, main_batch_size)

        # Streaming storage: {layer: {dir_type: {mult: {"sum": float, "count": int}}}}
        layer_accum = {
            layer: {
                dt: {m: {"sum": 0.0, "count": 0} for m in STEERING_MULTIPLIERS}
                for dt in direction_types
            }
            for layer in layers
        }

        # Control streaming storage: {layer: {dir_type: {ctrl_idx: {mult: {"sum": float, "count": int}}}}}
        ctrl_accum = {
            layer: {
                dt: {
                    c: {m: {"sum": 0.0, "count": 0} for m in STEERING_MULTIPLIERS}
                    for c in range(NUM_CONTROLS)
                }
                for dt in direction_types
            }
            for layer in layers
        }

        # Baseline accumulator (for mean computation)
        baseline_accum = {layer: {"sum": 0.0, "count": 0} for layer in layers}

        for layer in tqdm(layers, desc=f"Steering ({task} task)"):
            ortho = ortho_by_layer[layer]

            # Get layer module
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer]
            else:
                layer_module = model.model.layers[layer]

            # Get direction tensors for this layer
            dir_tensors = {dt: torch.tensor(getattr(ortho, dt), dtype=dtype, device=DEVICE)
                          for dt in direction_types}
            ctrl_tensors = controls_by_layer[layer]

            for batch_indices, batch_inputs in gpu_batches:
                B = len(batch_indices)

                # Get KV cache
                base_step_data = get_kv_cache(model, batch_inputs)
                keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

                inputs_template = {
                    "input_ids": base_step_data["input_ids"],
                    "attention_mask": base_step_data["attention_mask"],
                    "use_cache": True
                }
                if "position_ids" in base_step_data:
                    inputs_template["position_ids"] = base_step_data["position_ids"]

                # Baseline (no steering) - compute once per (layer, task, batch)
                # Check if we've already accumulated baseline for this layer+batch
                if baseline_accum[layer]["count"] < n_questions:
                    fresh_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                    baseline_inputs = inputs_template.copy()
                    baseline_inputs["past_key_values"] = fresh_cache

                    with torch.inference_mode():
                        out = model(**baseline_inputs)
                        logits = out.logits[:, -1, :][:, option_token_ids]
                        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                    for i, q_idx in enumerate(batch_indices):
                        conf = float(signal_fn(probs[i], None))
                        baseline_accum[layer]["sum"] += conf
                        baseline_accum[layer]["count"] += 1
                        # Accumulate baseline to multiplier 0 for all directions
                        for dt in direction_types:
                            layer_accum[layer][dt][0.0]["sum"] += conf
                            layer_accum[layer][dt][0.0]["count"] += 1
                        # Accumulate baseline to multiplier 0 for controls
                        for dt in direction_types:
                            for c in range(NUM_CONTROLS):
                                ctrl_accum[layer][dt][c][0.0]["sum"] += conf
                                ctrl_accum[layer][dt][c][0.0]["count"] += 1

                # === Batch all 4 main directions × k_mult multipliers ===
                expand_main = n_dirs * k_mult
                assert B * (n_dirs * k_mult) <= EXPANDED_BATCH_TARGET, "Batch size too large for expanded directions"
                expanded_ids = inputs_template["input_ids"].repeat_interleave(expand_main, dim=0)
                expanded_mask = inputs_template["attention_mask"].repeat_interleave(expand_main, dim=0)
                expanded_inputs = {"input_ids": expanded_ids, "attention_mask": expanded_mask, "use_cache": True}
                if "position_ids" in inputs_template:
                    expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(expand_main, dim=0)

                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=expand_main)
                expanded_inputs["past_key_values"] = pass_cache

                # Build delta tensor: for each sample, apply each direction × each multiplier
                deltas = []
                for _ in range(B):
                    for dt in direction_types:
                        for mult in nonzero_multipliers:
                            deltas.append(dir_tensors[dt] * mult)
                delta_bh = torch.stack(deltas, dim=0)

                hook = BatchSteeringHook()
                hook.set_delta(delta_bh)
                hook.register(layer_module)

                try:
                    with torch.inference_mode():
                        out = model(**expanded_inputs)
                        logits = out.logits[:, -1, :][:, option_token_ids]
                        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                    # Parse results: layout is [sample0_dir0_mult0, sample0_dir0_mult1, ..., sample0_dir1_mult0, ...]
                    for i, q_idx in enumerate(batch_indices):
                        for d_idx, dt in enumerate(direction_types):
                            for m_idx, mult in enumerate(nonzero_multipliers):
                                flat_idx = i * expand_main + d_idx * k_mult + m_idx
                                conf = float(signal_fn(probs[flat_idx], None))
                                layer_accum[layer][dt][mult]["sum"] += conf
                                layer_accum[layer][dt][mult]["count"] += 1
                finally:
                    hook.remove()

                # === Batch ALL controls across direction types together ===
                # Flatten all controls with tracking: [(dt, c_idx, tensor), ...]
                all_ctrl_list = []
                for dt in direction_types:
                    for c_idx, c_tensor in enumerate(controls_by_layer[layer][dt]):
                        all_ctrl_list.append((dt, c_idx, c_tensor))

                total_ctrls = len(all_ctrl_list)
                for ctrl_start in range(0, total_ctrls, ctrl_per_pass):
                    ctrl_end = min(ctrl_start + ctrl_per_pass, total_ctrls)
                    ctrl_chunk = all_ctrl_list[ctrl_start:ctrl_end]
                    n_ctrl = len(ctrl_chunk)

                    expand_ctrl = n_ctrl * k_mult
                    ctrl_ids = inputs_template["input_ids"].repeat_interleave(expand_ctrl, dim=0)
                    ctrl_mask = inputs_template["attention_mask"].repeat_interleave(expand_ctrl, dim=0)
                    ctrl_inputs = {"input_ids": ctrl_ids, "attention_mask": ctrl_mask, "use_cache": True}
                    if "position_ids" in inputs_template:
                        ctrl_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(expand_ctrl, dim=0)

                    ctrl_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=expand_ctrl)
                    ctrl_inputs["past_key_values"] = ctrl_cache

                    # Build delta tensor for controls
                    ctrl_deltas = []
                    for _ in range(B):
                        for dt, c_idx, c_tensor in ctrl_chunk:
                            for mult in nonzero_multipliers:
                                ctrl_deltas.append(c_tensor * mult)
                    ctrl_delta_bh = torch.stack(ctrl_deltas, dim=0)

                    hook = BatchSteeringHook()
                    hook.set_delta(ctrl_delta_bh)
                    hook.register(layer_module)

                    try:
                        with torch.inference_mode():
                            out = model(**ctrl_inputs)
                            logits = out.logits[:, -1, :][:, option_token_ids]
                            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                        # Parse control results - route to correct direction type
                        for i, q_idx in enumerate(batch_indices):
                            for c_local, (dt, c_idx, _) in enumerate(ctrl_chunk):
                                for m_idx, mult in enumerate(nonzero_multipliers):
                                    flat_idx = i * expand_ctrl + c_local * k_mult + m_idx
                                    conf = float(signal_fn(probs[flat_idx], None))
                                    ctrl_accum[layer][dt][c_idx][mult]["sum"] += conf
                                    ctrl_accum[layer][dt][c_idx][mult]["count"] += 1
                    finally:
                        hook.remove()

        # After all layers for this task, compute slopes from streaming accumulators
        for layer in layers:
            # Compute direction slopes and per-direction control slopes
            for dt in direction_types:
                # Helper to compute mean from accumulator
                def accum_mean(acc):
                    return acc["sum"] / acc["count"] if acc["count"] > 0 else 0.0

                # Compute control slopes for this (layer, direction_type)
                control_slopes = []
                for c_idx in range(NUM_CONTROLS):
                    mean_confs = [accum_mean(ctrl_accum[layer][dt][c_idx][m])
                                  for m in STEERING_MULTIPLIERS]
                    slope, _ = np.polyfit(STEERING_MULTIPLIERS, mean_confs, 1)
                    control_slopes.append(float(slope))

                # Compute direction slope
                mean_confs = [accum_mean(layer_accum[layer][dt][m])
                              for m in STEERING_MULTIPLIERS]
                slope, _ = np.polyfit(STEERING_MULTIPLIERS, mean_confs, 1)

                # Compute baseline mean
                baseline_mean = accum_mean(baseline_accum[layer])

                results["by_layer"][layer][(dt, task)] = {
                    "slope": float(slope),
                    "control_slopes": control_slopes,
                    "baseline_conf_mean": float(baseline_mean),
                }

    # Compute p-values using per-(layer, direction, task) null distribution
    for layer in layers:
        for dir_type in direction_types:
            for task in tasks:
                data = results["by_layer"][layer][(dir_type, task)]
                slope_abs = abs(data["slope"])

                # Use per-direction control slopes (not pooled)
                ctrl_abs = np.abs(data["control_slopes"]) if data["control_slopes"] else np.array([0.0])
                n_larger = np.sum(ctrl_abs >= slope_abs)
                p_value = (n_larger + 1) / (len(ctrl_abs) + 1)
                data["p_value"] = float(p_value)

                # Effect size (using signed slopes for proper z-score)
                ctrl_signed = np.array(data["control_slopes"]) if data["control_slopes"] else np.array([0.0])
                if np.std(ctrl_signed) > 1e-10:
                    z_score = (data["slope"] - np.mean(ctrl_signed)) / np.std(ctrl_signed)
                else:
                    z_score = 0.0
                data["effect_z"] = float(z_score)

    # FDR correction per (direction_type, task) combination
    for dir_type in direction_types:
        for task in tasks:
            pvals = {l: results["by_layer"][l][(dir_type, task)]["p_value"] for l in layers}
            pvals_fdr = _bh_fdr(pvals)
            for l in layers:
                results["by_layer"][l][(dir_type, task)]["p_value_fdr"] = pvals_fdr[l]

    # Summary
    for dir_type in direction_types:
        for task in tasks:
            key = f"{dir_type}_{task}"
            slopes = [results["by_layer"][l][(dir_type, task)]["slope"] for l in layers]
            p_values = [results["by_layer"][l][(dir_type, task)]["p_value"] for l in layers]
            p_values_fdr = [results["by_layer"][l][(dir_type, task)]["p_value_fdr"] for l in layers]

            results["summary"][key] = {
                "n_significant": sum(1 for p in p_values if p < 0.05),
                "n_significant_fdr": sum(1 for p in p_values_fdr if p < 0.05),
                "peak_slope": float(np.max(np.abs(slopes))),
                "peak_layer": layers[np.argmax(np.abs(slopes))],
            }

    return results


# =============================================================================
# ABLATION EXPERIMENTS (Batched by task for efficiency)
# =============================================================================
def option_probs_last_token(model, inputs, option_token_ids):
    """
    Return P(option | prompt) for last token only, trying to avoid full [B,T,V] logits.
    """
    fw = dict(inputs)
    fw["use_cache"] = bool(fw.get("use_cache", False))

    # Prefer "keep only last token logits" APIs if available
    try:
        out = model(**fw, logits_to_keep=1)
        logits = out.logits  # [B,1,V] or [B,T,V] but with T small
    except TypeError:
        try:
            print("Model does not support logits_to_keep; trying num_logits_to_keep...")
            out = model(**fw, num_logits_to_keep=1)
            logits = out.logits
        except TypeError:
            print("Model does not support num_logits_to_keep; falling back to full logits.")
            # Fallback: still works, but may be slower/more memory
            out = model(**fw)
            logits = out.logits

    last = logits[:, -1, :]  # [B,V]
    opt_ids = torch.as_tensor(option_token_ids, device=last.device)
    opt_logits = last.index_select(-1, opt_ids)  # [B,nopt]
    probs = torch.softmax(opt_logits, dim=-1).float().cpu().numpy()

    # help peak memory
    del out, logits, last, opt_logits
    return probs

def run_ablation_matrix(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    ortho_by_layer: Dict[int, OrthogonalDirections],
    layers: List[int],
    use_chat_template: bool,
) -> Dict:
    """
    Run 4x2 ablation experiment matrix without storing per-question arrays.

    - Baseline corr computed once per task (streaming).
    - Ablated corr computed per (layer, direction, task) (streaming).
    - Control corrs computed per-control (streaming), but without storing conf lists.
    - Controls kept on CPU, moved to GPU only in the current chunk.
    """
    direction_types = ["d_self", "d_other", "d_introspection", "d_surface"]
    tasks = ["self", "other"]
    n_dirs = len(direction_types)

    results = {"by_layer": {layer: {} for layer in layers}, "summary": {}}

    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    # --- Controls: keep on CPU (numpy arrays), not GPU tensors ---
    controls_by_layer = {}
    for layer in layers:
        ortho = ortho_by_layer[layer]
        controls_by_layer[layer] = {}
        for dt in direction_types:
            direction = getattr(ortho, dt)
            seed = SEED + layer * 1000 + DT_SEED_OFFSET[dt]  # stable & reproducible
            controls = generate_orthogonal_directions(direction, NUM_CONTROLS, seed=seed)
            # keep as numpy arrays on CPU; tensorize per chunk later
            controls_by_layer[layer][dt] = controls

    def safe_corr_from_accum(acc: OnlineCorrelation) -> float:
        r = acc.correlation()
        return float(r) if np.isfinite(r) else 0.0

    for task in tasks:
        format_fn, signal_fn, options = get_task_config(task)
        option_token_ids = get_single_token_ids(tokenizer, options)

        # Format + tokenize prompts once per task
        prompts = [format_fn(q, tokenizer, use_chat_template=use_chat_template)[0] for q in questions]
        cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)

        # Use the base BATCH_SIZE for the "question batch" dimension
        main_expand = len(direction_types)  # 4 directions
        main_batch_size = max(1, min(BATCH_SIZE, EXPANDED_BATCH_TARGET // main_expand))
        gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, main_batch_size)

        # ---- Baseline corr once per task (layer-independent) ----
        baseline_acc = OnlineCorrelation()
        #for batch_indices, batch_inputs in gpu_batches:
        for batch_indices, batch_inputs in tqdm(gpu_batches, leave=False, desc=f"  batches ({task})"):
            with torch.inference_mode():
                #out = model(**batch_inputs, use_cache=False)
                #logits = out.logits[:, -1, :][:, option_token_ids]
                #probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                probs = option_probs_last_token(model, batch_inputs, option_token_ids)
    
            for i, q_idx in enumerate(batch_indices):
                conf = float(signal_fn(probs[i], None))
                baseline_acc.update(conf, float(metric_values[q_idx]))

        baseline_corr = safe_corr_from_accum(baseline_acc)

        # ---- Per-layer ablation ----
        for layer in tqdm(layers, desc=f"Ablation ({task} task)"):
            ortho = ortho_by_layer[layer]

            # Layer module
            if hasattr(model, "get_base_model"):
                layer_module = model.get_base_model().model.layers[layer]
            else:
                layer_module = model.model.layers[layer]

            # Direction tensors on GPU (just 4 of them, OK)
            dir_tensors = {
                dt: torch.tensor(getattr(ortho, dt), dtype=dtype, device=DEVICE)
                for dt in direction_types
            }

            # Streaming accumulators for this layer/task
            ablated_acc = {dt: OnlineCorrelation() for dt in direction_types}
            ctrl_acc = {dt: [OnlineCorrelation() for _ in range(NUM_CONTROLS)] for dt in direction_types}

            for batch_indices, batch_inputs in gpu_batches:
                B = len(batch_indices)
                assert B * len(direction_types) <= EXPANDED_BATCH_TARGET, "Batch size too large for expanded directions in ablation"
                # --- KV cache snapshot for this batch ---
                base_step_data = get_kv_cache(model, batch_inputs)
                keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

                inputs_template = {
                    "input_ids": base_step_data["input_ids"],
                    "attention_mask": base_step_data["attention_mask"],
                    "use_cache": True,
                }
                if "position_ids" in base_step_data:
                    inputs_template["position_ids"] = base_step_data["position_ids"]

                # === Main 4 directions in one expanded pass ===
                main_dirs = [dir_tensors[dt] for dt in direction_types]
                n_main = len(main_dirs)

                expanded_ids = inputs_template["input_ids"].repeat_interleave(n_main, dim=0)
                expanded_mask = inputs_template["attention_mask"].repeat_interleave(n_main, dim=0)
                expanded_inputs = {"input_ids": expanded_ids, "attention_mask": expanded_mask, "use_cache": True}
                if "position_ids" in inputs_template:
                    expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(n_main, dim=0)

                expanded_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=n_main)
                expanded_inputs["past_key_values"] = expanded_cache

                dirs_bh = torch.stack([d for _ in range(B) for d in main_dirs], dim=0)

                hook = BatchAblationHook(dirs_bh)
                hook.register(layer_module)
                try:
                    with torch.inference_mode():
                        #out = model(**expanded_inputs, use_cache=False)
                        #logits = out.logits[:, -1, :][:, option_token_ids]
                        #probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                        probs = option_probs_last_token(model, expanded_inputs, option_token_ids)

                    for i, q_idx in enumerate(batch_indices):
                        y = float(metric_values[q_idx])
                        for d_idx, dt in enumerate(direction_types):
                            flat_idx = i * n_main + d_idx
                            conf = float(signal_fn(probs[flat_idx], None))
                            ablated_acc[dt].update(conf, y)
                finally:
                    hook.remove()

                # === Controls (chunked to avoid VRAM blowups) ===
                # Flatten all controls across direction types, but keep their (dt, c_idx)
                all_ctrl = []
                for dt in direction_types:
                    for c_idx, c_np in enumerate(controls_by_layer[layer][dt]):
                        all_ctrl.append((dt, c_idx, c_np))

                # Critical: ensure expanded batch size <= EXPANDED_BATCH_TARGET
                # expanded_size = B * n_ctrl  ->  n_ctrl_max = floor(EXPANDED_BATCH_TARGET / B)
                n_ctrl_max = max(1, EXPANDED_BATCH_TARGET // max(1, B))

                #for start in range(0, len(all_ctrl), n_ctrl_max):
                for start in tqdm(range(0, len(all_ctrl), n_ctrl_max), leave=False, desc="    ctrl chunks"):
                    chunk = all_ctrl[start:start + n_ctrl_max]
                    n_ctrl = len(chunk)

                    ctrl_ids = inputs_template["input_ids"].repeat_interleave(n_ctrl, dim=0)
                    ctrl_mask = inputs_template["attention_mask"].repeat_interleave(n_ctrl, dim=0)
                    ctrl_inputs = {"input_ids": ctrl_ids, "attention_mask": ctrl_mask, "use_cache": True}
                    if "position_ids" in inputs_template:
                        ctrl_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(n_ctrl, dim=0)

                    ctrl_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=n_ctrl)
                    ctrl_inputs["past_key_values"] = ctrl_cache

                    # tensorize only this chunk (and only once per chunk)
                    ctrl_vecs = [torch.tensor(c_np, dtype=dtype, device=DEVICE) for (_, _, c_np) in chunk]
                    ctrl_bh = torch.stack([v for _ in range(B) for v in ctrl_vecs], dim=0)

                    hook = BatchAblationHook(ctrl_bh)
                    hook.register(layer_module)
                    try:
                        with torch.inference_mode():
                            #out = model(**ctrl_inputs, use_cache=False)
                            #logits = out.logits[:, -1, :][:, option_token_ids]
                            #probs = torch.softmax(logits, dim=-1).float().cpu().numpy()
                            probs = option_probs_last_token(model, ctrl_inputs, option_token_ids)
                        for i, q_idx in enumerate(batch_indices):
                            y = float(metric_values[q_idx])
                            for c_local, (dt, c_idx, _) in enumerate(chunk):
                                flat_idx = i * n_ctrl + c_local
                                conf = float(signal_fn(probs[flat_idx], None))
                                ctrl_acc[dt][c_idx].update(conf, y)
                    finally:
                        hook.remove()

            # ---- Save per-layer results for this task ----
            for dt in direction_types:
                ablated_corr = safe_corr_from_accum(ablated_acc[dt])
                delta_corr = baseline_corr - ablated_corr

                control_delta_corrs = []
                for c_idx in range(NUM_CONTROLS):
                    ctrl_corr = safe_corr_from_accum(ctrl_acc[dt][c_idx])
                    control_delta_corrs.append(float(baseline_corr - ctrl_corr))

                results["by_layer"][layer][(dt, task)] = {
                    "baseline_corr": float(baseline_corr),
                    "ablated_corr": float(ablated_corr),
                    "delta_corr": float(delta_corr),
                    "control_delta_corrs": control_delta_corrs,
                }

            # (Optional) free cached GPU memory between layers
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ---- p-values / effect size / FDR (unchanged logic) ----
    for layer in layers:
        for dir_type in direction_types:
            for task in tasks:
                data = results["by_layer"][layer][(dir_type, task)]
                delta_abs = abs(data["delta_corr"])

                ctrl_abs = np.abs(data["control_delta_corrs"]) if data["control_delta_corrs"] else np.array([0.0])
                n_larger = np.sum(ctrl_abs >= delta_abs)
                p_value = (n_larger + 1) / (len(ctrl_abs) + 1)
                data["p_value"] = float(p_value)

                ctrl_signed = np.array(data["control_delta_corrs"]) if data["control_delta_corrs"] else np.array([0.0])
                if np.std(ctrl_signed) > 1e-10:
                    z_score = (data["delta_corr"] - np.mean(ctrl_signed)) / np.std(ctrl_signed)
                else:
                    z_score = 0.0
                data["effect_z"] = float(z_score)

    for dir_type in direction_types:
        for task in tasks:
            pvals = {l: results["by_layer"][l][(dir_type, task)]["p_value"] for l in layers}
            pvals_fdr = _bh_fdr(pvals)
            for l in layers:
                results["by_layer"][l][(dir_type, task)]["p_value_fdr"] = pvals_fdr[l]

    for dir_type in direction_types:
        for task in tasks:
            key = f"{dir_type}_{task}"
            deltas = [results["by_layer"][l][(dir_type, task)]["delta_corr"] for l in layers]
            p_values = [results["by_layer"][l][(dir_type, task)]["p_value"] for l in layers]
            p_values_fdr = [results["by_layer"][l][(dir_type, task)]["p_value_fdr"] for l in layers]
            results["summary"][key] = {
                "n_significant": sum(1 for p in p_values if p < 0.05),
                "n_significant_fdr": sum(1 for p in p_values_fdr if p < 0.05),
                "peak_delta": float(np.max(np.abs(deltas))),
                "peak_layer": layers[int(np.argmax(np.abs(deltas)))],
            }

    return results

# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_orthogonalization_similarity(
    ortho_by_layer: Dict[int, OrthogonalDirections],
    output_path: Path
):
    """Plot cosine similarity and residual norm across layers."""
    layers = sorted(ortho_by_layer.keys())

    cosines = [ortho_by_layer[l].cosine_similarity for l in layers]
    residuals = [ortho_by_layer[l].residual_norm for l in layers]
    degenerate = [ortho_by_layer[l].degenerate for l in layers]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle("Self vs Other Confidence Direction Similarity", fontsize=14)

    x = np.arange(len(layers))

    # Panel 1: Cosine similarity
    ax1.bar(x, cosines, color='steelblue', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Degenerate threshold')
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("cos(d_self, d_other) by Layer")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=GRID_ALPHA)

    # Mark degenerate layers
    deg_x = [i for i, d in enumerate(degenerate) if d]
    if deg_x:
        ax1.scatter(deg_x, [cosines[i] for i in deg_x], color='red', s=100, marker='x', zorder=5, label='Degenerate')

    # Panel 2: Residual norm (unique variance)
    ax2.bar(x, residuals, color='darkorange', alpha=0.7)
    ax2.axhline(y=MIN_RESIDUAL_NORM, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_ylabel("Residual Norm")
    ax2.set_xlabel("Layer")
    ax2.set_xticks(x)
    ax2.set_xticklabels(layers)
    ax2.set_title(f"||d_self - proj(d_self, d_other)|| = sqrt(1 - cos²)")
    ax2.grid(True, alpha=GRID_ALPHA)

    save_figure(fig, output_path)


def plot_predictive_heatmap(
    predictive_results: Dict,
    ortho_by_layer: Dict[int, OrthogonalDirections],
    output_path: Path
):
    """Plot 4x2 predictive power heatmap."""
    layers = sorted(predictive_results["by_layer"].keys())
    direction_types = ["d_self", "d_other", "d_introspection", "d_surface"]
    tasks = ["self", "other"]

    # Create heatmap data
    n_dir = len(direction_types)
    n_task = len(tasks)
    n_layers = len(layers)

    fig, axes = plt.subplots(n_dir, n_task, figsize=(12, 14), sharex=True, sharey=True)
    fig.suptitle("Predictive Power: R² by (Direction, Task, Layer)", fontsize=14, y=1.02)

    for i, dir_type in enumerate(direction_types):
        for j, task in enumerate(tasks):
            ax = axes[i, j]

            r2s = [predictive_results["by_layer"][l][(dir_type, task)]["r2"] for l in layers]
            degenerate = [ortho_by_layer[l].degenerate for l in layers]

            # Bar plot
            colors = ['gray' if d else 'steelblue' for d in degenerate]
            ax.bar(range(n_layers), r2s, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            # Title and labels
            if i == 0:
                ax.set_title(f"Task: {task}")
            if j == 0:
                ax.set_ylabel(f"{dir_type}\n\nR²")
            if i == n_dir - 1:
                ax.set_xlabel("Layer")
                ax.set_xticks(range(0, n_layers, 4))
                ax.set_xticklabels([layers[k] for k in range(0, n_layers, 4)])

            ax.grid(True, alpha=GRID_ALPHA)
            ax.set_ylim(-0.1, 1.0)

    plt.tight_layout()
    save_figure(fig, output_path)


def plot_causal_heatmap(
    steering_results: Dict,
    ablation_results: Dict,
    output_path_steering: Path,
    output_path_ablation: Path
):
    """Plot 4x2 causal effect heatmaps for steering and ablation."""
    layers = sorted(steering_results["by_layer"].keys())
    direction_types = ["d_self", "d_other", "d_introspection", "d_surface"]
    tasks = ["self", "other"]
    n_dir = len(direction_types)
    n_task = len(tasks)

    # Steering plot (4x2 grid)
    fig, axes = plt.subplots(n_dir, n_task, figsize=(14, 16), sharex=True)
    fig.suptitle("Steering Effects: Slope by (Direction, Task) [green=FDR<0.05]", fontsize=14, y=1.02)

    for i, dir_type in enumerate(direction_types):
        for j, task in enumerate(tasks):
            ax = axes[i, j]

            slopes = [steering_results["by_layer"][l][(dir_type, task)]["slope"] for l in layers]
            p_values_fdr = [steering_results["by_layer"][l][(dir_type, task)]["p_value_fdr"] for l in layers]

            colors = ['green' if p < 0.05 else 'gray' for p in p_values_fdr]
            ax.bar(range(len(layers)), slopes, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            if i == 0:
                ax.set_title(f"Task: {task}")
            if j == 0:
                ax.set_ylabel(f"{dir_type}\n\nSlope")
            if i == n_dir - 1:
                ax.set_xlabel("Layer")
                ax.set_xticks(range(0, len(layers), 4))
                ax.set_xticklabels([layers[k] for k in range(0, len(layers), 4)])

            ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(fig, output_path_steering)

    # Ablation plot (4x2 grid)
    fig, axes = plt.subplots(n_dir, n_task, figsize=(14, 16), sharex=True)
    fig.suptitle("Ablation Effects: Δcorr by (Direction, Task) [green=FDR<0.05]", fontsize=14, y=1.02)

    for i, dir_type in enumerate(direction_types):
        for j, task in enumerate(tasks):
            ax = axes[i, j]

            deltas = [ablation_results["by_layer"][l][(dir_type, task)]["delta_corr"] for l in layers]
            p_values_fdr = [ablation_results["by_layer"][l][(dir_type, task)]["p_value_fdr"] for l in layers]

            colors = ['green' if p < 0.05 else 'gray' for p in p_values_fdr]
            ax.bar(range(len(layers)), deltas, color=colors, alpha=0.7)
            ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

            if i == 0:
                ax.set_title(f"Task: {task}")
            if j == 0:
                ax.set_ylabel(f"{dir_type}\n\nΔcorr")
            if i == n_dir - 1:
                ax.set_xlabel("Layer")
                ax.set_xticks(range(0, len(layers), 4))
                ax.set_xticklabels([layers[k] for k in range(0, len(layers), 4)])

            ax.grid(True, alpha=GRID_ALPHA)

    plt.tight_layout()
    save_figure(fig, output_path_ablation)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("INTROSPECTION VS SURFACE DIFFICULTY ANALYSIS")
    print("=" * 70)
    print(f"\nModel: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Input: {INPUT_BASE_NAME}")
    print(f"Method: {METHOD}")
    print(f"Metric: {METRIC}")

    # Define checkpoint paths early for resumability
    model_short = get_model_short_name(MODEL, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)
    dataset_name = INPUT_BASE_NAME.replace(f"{model_short}_", "", 1)
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        base_output = f"{model_short}_adapter-{adapter_short}_{dataset_name}_orthogonal"
    else:
        base_output = f"{model_short}_{dataset_name}_orthogonal"

    # Checkpoint files for resumability
    steering_checkpoint = OUTPUT_DIR / f"{base_output}_steering_checkpoint.json"
    ablation_checkpoint = OUTPUT_DIR / f"{base_output}_ablation_checkpoint.json"
    print(f"\nCheckpoint files:")
    print(f"  Steering: {steering_checkpoint.name}")
    print(f"  Ablation: {ablation_checkpoint.name}")

    # Load directions
    print("\nLoading confidence directions...")
    d_self_by_layer = load_confidence_directions(INPUT_BASE_NAME, "confidence", METHOD)
    d_other_by_layer = load_confidence_directions(INPUT_BASE_NAME, "other_confidence", METHOD)

    layers = sorted(set(d_self_by_layer.keys()) & set(d_other_by_layer.keys()))
    print(f"  Found {len(layers)} common layers")

    # Load activations
    print("\nLoading meta-task activations...")
    self_activations, self_confidences = load_meta_activations(INPUT_BASE_NAME, "confidence")
    other_activations, other_confidences = load_meta_activations(INPUT_BASE_NAME, "other_confidence")
    print(f"  Self-confidence: {len(self_confidences)} samples")
    print(f"  Other-confidence: {len(other_confidences)} samples")

    # Load dataset for metric values
    print("\nLoading dataset...")
    dataset = load_dataset(INPUT_BASE_NAME)
    all_data = dataset["data"]
    metric_values = np.array([item[METRIC] for item in all_data])

    # Create train/test split (matching other scripts)
    n_samples = len(self_confidences)
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_SPLIT, random_state=SEED)
    print(f"  Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Compute orthogonalizations
    print("\nComputing orthogonalized directions...")
    ortho_by_layer = compute_all_orthogonalizations(
        d_self_by_layer, d_other_by_layer,
        min_residual_norm=MIN_RESIDUAL_NORM
    )

    n_degenerate = sum(1 for l in layers if ortho_by_layer[l].degenerate)
    mean_cosine = np.mean([ortho_by_layer[l].cosine_similarity for l in layers])
    print(f"  Mean cosine similarity: {mean_cosine:.3f}")
    print(f"  Degenerate layers: {n_degenerate}/{len(layers)}")

    # Compute predictive power
    print("\nComputing predictive power matrix...")
    predictive_results = compute_predictive_power_matrix(
        ortho_by_layer,
        self_activations, other_activations,
        self_confidences, other_confidences,
        train_idx, test_idx,
        n_bootstrap=N_BOOTSTRAP
    )

    # Print key predictive results
    print("\n  Predictive R² summary (test set):")
    for key, stats in predictive_results["summary"].items():
        print(f"    {key}: peak R²={stats['max_r2']:.3f} at L{stats['peak_layer']}")

    # Load model for causal experiments
    print("\nLoading model...")
    model, tokenizer, num_layers = load_model_and_tokenizer(
        MODEL,
        adapter_path=ADAPTER,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    use_chat_template = should_use_chat_template(MODEL, tokenizer)
    print(f"  Use chat template: {use_chat_template}")
    print(f"  Device: {DEVICE}")

    # Prepare questions for causal experiments (use test set)
    questions = [all_data[i] for i in test_idx]
    test_metric_values = metric_values[test_idx]

    # Select layers for causal experiments based on config
    non_degenerate = [l for l in layers if not ortho_by_layer[l].degenerate]

    if CAUSAL_LAYER_MODE == "all":
        causal_layers = non_degenerate
        print(f"\nCausal experiments on ALL {len(causal_layers)} non-degenerate layers")
    elif CAUSAL_LAYER_MODE == "top_k":
        # Select top-k layers by d_introspection_self R² (the key hypothesis)
        layer_r2 = [
            (l, predictive_results["by_layer"][l][("d_introspection", "self")]["r2"])
            for l in non_degenerate
        ]
        layer_r2.sort(key=lambda x: x[1], reverse=True)
        causal_layers = [l for l, _ in layer_r2[:CAUSAL_TOP_K]]
        print(f"\nCausal experiments on TOP {len(causal_layers)} layers by d_introspection R²:")
        for l, r2 in layer_r2[:CAUSAL_TOP_K]:
            print(f"    Layer {l}: R²={r2:.3f}")
    elif CAUSAL_LAYER_MODE == "explicit":
        causal_layers = [l for l in CAUSAL_EXPLICIT_LAYERS
                         if l in layers and not ortho_by_layer[l].degenerate]
        print(f"\nCausal experiments on {len(causal_layers)} EXPLICIT layers: {causal_layers}")
    else:
        raise ValueError(f"Unknown CAUSAL_LAYER_MODE: {CAUSAL_LAYER_MODE}")

    # Run steering experiments (with checkpoint support)
    if steering_checkpoint.exists():
        print(f"\nLoading steering results from checkpoint...")
        with open(steering_checkpoint, "r") as f:
            steering_checkpoint_data = json.load(f)
        # Reconstruct steering_results with tuple keys
        steering_results = {"by_layer": {}, "summary": steering_checkpoint_data["summary"]}
        for layer_str, layer_data in steering_checkpoint_data["by_layer"].items():
            layer = int(layer_str)
            steering_results["by_layer"][layer] = {}
            for key_str, val in layer_data.items():
                # Parse "d_self_self" -> ("d_self", "self")
                parts = key_str.rsplit("_", 1)
                dir_type, task = parts[0], parts[1]
                steering_results["by_layer"][layer][(dir_type, task)] = val
        print(f"  Loaded checkpoint with {len(steering_results['by_layer'])} layers")
    else:
        print("\nRunning steering experiments...")
        steering_results = run_steering_matrix(
            model, tokenizer, questions,
            ortho_by_layer, causal_layers, use_chat_template
        )
        # Save checkpoint immediately
        steering_checkpoint_data = {"by_layer": {}, "summary": steering_results["summary"]}
        for layer, layer_data in steering_results["by_layer"].items():
            steering_checkpoint_data["by_layer"][layer] = {}
            for (dir_type, task), val in layer_data.items():
                steering_checkpoint_data["by_layer"][layer][f"{dir_type}_{task}"] = val
        with open(steering_checkpoint, "w") as f:
            json.dump(steering_checkpoint_data, f, indent=2)
        print(f"\n  Saved steering checkpoint: {steering_checkpoint.name}")

    print("\n  Steering summary (FDR-corrected):")
    for key, stats in steering_results["summary"].items():
        print(f"    {key}: {stats['n_significant_fdr']} FDR-significant layers (of {stats['n_significant']} raw)")

    # Run ablation experiments (with checkpoint support)
    # Clear CUDA cache to avoid OOM from memory fragmentation after steering
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"\nCleared CUDA cache before ablation")

    if ablation_checkpoint.exists():
        print(f"\nLoading ablation results from checkpoint...")
        with open(ablation_checkpoint, "r") as f:
            ablation_checkpoint_data = json.load(f)
        # Reconstruct ablation_results with tuple keys
        ablation_results = {"by_layer": {}, "summary": ablation_checkpoint_data["summary"]}
        for layer_str, layer_data in ablation_checkpoint_data["by_layer"].items():
            layer = int(layer_str)
            ablation_results["by_layer"][layer] = {}
            for key_str, val in layer_data.items():
                parts = key_str.rsplit("_", 1)
                dir_type, task = parts[0], parts[1]
                ablation_results["by_layer"][layer][(dir_type, task)] = val
        print(f"  Loaded checkpoint with {len(ablation_results['by_layer'])} layers")
    else:
        print("\nRunning ablation experiments...")
        ablation_results = run_ablation_matrix(
            model, tokenizer, questions, test_metric_values,
            ortho_by_layer, causal_layers, use_chat_template
        )
        # Save checkpoint immediately
        ablation_checkpoint_data = {"by_layer": {}, "summary": ablation_results["summary"]}
        for layer, layer_data in ablation_results["by_layer"].items():
            ablation_checkpoint_data["by_layer"][layer] = {}
            for (dir_type, task), val in layer_data.items():
                ablation_checkpoint_data["by_layer"][layer][f"{dir_type}_{task}"] = val
        with open(ablation_checkpoint, "w") as f:
            json.dump(ablation_checkpoint_data, f, indent=2)
        print(f"\n  Saved ablation checkpoint: {ablation_checkpoint.name}")

    print("\n  Ablation summary (FDR-corrected):")
    for key, stats in ablation_results["summary"].items():
        print(f"    {key}: {stats['n_significant_fdr']} FDR-significant, peak Δcorr={stats['peak_delta']:.3f}")

    # Save directions (base_output computed at start of main())
    print("\nSaving orthogonal directions...")
    directions_path = OUTPUT_DIR / f"{base_output}_directions.npz"
    save_data = {
        "_metadata_model": MODEL,
        "_metadata_dataset": INPUT_BASE_NAME,
        "_metadata_method": METHOD,
    }
    for layer in layers:
        ortho = ortho_by_layer[layer]
        save_data[f"introspection_layer_{layer}"] = ortho.d_introspection
        save_data[f"surface_layer_{layer}"] = ortho.d_surface
        save_data[f"cosine_layer_{layer}"] = np.array([ortho.cosine_similarity])
        save_data[f"residual_norm_layer_{layer}"] = np.array([ortho.residual_norm])
    np.savez_compressed(directions_path, **save_data)
    print(f"  Saved {directions_path.name}")

    # Save JSON results
    print("\nSaving results JSON...")
    results_path = OUTPUT_DIR / f"{base_output}_analysis_results.json"

    # Convert predictive results to JSON-serializable format
    predictive_json = {"by_layer": {}, "summary": predictive_results["summary"]}
    for layer in layers:
        predictive_json["by_layer"][layer] = {}
        for key, val in predictive_results["by_layer"][layer].items():
            dir_type, task = key
            predictive_json["by_layer"][layer][f"{dir_type}_{task}"] = val

    # Convert steering results
    steering_json = {"by_layer": {}, "summary": steering_results["summary"]}
    for layer in causal_layers:
        steering_json["by_layer"][layer] = {}
        for key, val in steering_results["by_layer"][layer].items():
            dir_type, task = key
            steering_json["by_layer"][layer][f"{dir_type}_{task}"] = {
                "slope": val["slope"],
                "p_value": val["p_value"],
                "p_value_fdr": val["p_value_fdr"],
                "effect_z": val["effect_z"],
            }

    # Convert ablation results
    ablation_json = {"by_layer": {}, "summary": ablation_results["summary"]}
    for layer in causal_layers:
        ablation_json["by_layer"][layer] = {}
        for key, val in ablation_results["by_layer"][layer].items():
            dir_type, task = key
            ablation_json["by_layer"][layer][f"{dir_type}_{task}"] = {
                "baseline_corr": val["baseline_corr"],
                "ablated_corr": val["ablated_corr"],
                "delta_corr": val["delta_corr"],
                "p_value": val["p_value"],
                "p_value_fdr": val["p_value_fdr"],
                "effect_z": val["effect_z"],
            }

    output_json = {
        "config": get_config_dict(
            model=MODEL,
            adapter=ADAPTER,
            input_base_name=INPUT_BASE_NAME,
            metric=METRIC,
            method=METHOD,
            seed=SEED,
            train_split=TRAIN_SPLIT,
            min_residual_norm=MIN_RESIDUAL_NORM,
            num_controls=NUM_CONTROLS,
            n_bootstrap=N_BOOTSTRAP,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "orthogonalization": {
            "by_layer": {
                l: {
                    "cosine_similarity": ortho_by_layer[l].cosine_similarity,
                    "residual_norm": ortho_by_layer[l].residual_norm,
                    "shared_variance": ortho_by_layer[l].cosine_similarity ** 2,
                    "unique_variance": 1 - ortho_by_layer[l].cosine_similarity ** 2,
                    "degenerate": ortho_by_layer[l].degenerate,
                }
                for l in layers
            },
            "summary": {
                "mean_cosine": float(mean_cosine),
                "mean_shared_variance": float(np.mean([ortho_by_layer[l].cosine_similarity ** 2 for l in layers])),
                "mean_unique_variance": float(np.mean([1 - ortho_by_layer[l].cosine_similarity ** 2 for l in layers])),
                "n_degenerate": n_degenerate,
                "degenerate_layers": [l for l in layers if ortho_by_layer[l].degenerate],
            },
        },
        "predictive": predictive_json,
        "steering": steering_json,
        "ablation": ablation_json,
        "interpretation": {
            "unique_introspection_exists": float(np.mean([1 - ortho_by_layer[l].cosine_similarity ** 2 for l in layers])) > 0.1,
            "introspection_predictive": predictive_results["summary"].get("d_introspection_self", {}).get("max_r2", 0) > 0.05,
            "surface_predictive": predictive_results["summary"].get("d_surface_other", {}).get("max_r2", 0) > 0.05,
        },
    }

    with open(results_path, "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"  Saved {results_path.name}")

    # Generate plots
    print("\nGenerating plots...")

    similarity_path = OUTPUT_DIR / f"{base_output}_similarity.png"
    plot_orthogonalization_similarity(ortho_by_layer, similarity_path)

    predictive_path = OUTPUT_DIR / f"{base_output}_predictive.png"
    plot_predictive_heatmap(predictive_results, ortho_by_layer, predictive_path)

    steering_path = OUTPUT_DIR / f"{base_output}_steering.png"
    ablation_path = OUTPUT_DIR / f"{base_output}_ablation.png"
    plot_causal_heatmap(steering_results, ablation_results, steering_path, ablation_path)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nMean cosine(d_self, d_other): {mean_cosine:.3f}")
    print(f"Mean shared variance: {output_json['orthogonalization']['summary']['mean_shared_variance']:.1%}")
    print(f"Mean unique variance: {output_json['orthogonalization']['summary']['mean_unique_variance']:.1%}")

    # Print full 4x2 predictive power summary
    print("\nPredictive Power (R²):")
    print("                      self-task    other-task")
    for dir_type in ["d_self", "d_other", "d_introspection", "d_surface"]:
        self_pred = predictive_results["summary"].get(f"{dir_type}_self", {})
        other_pred = predictive_results["summary"].get(f"{dir_type}_other", {})
        self_r2 = f"{self_pred['max_r2']:.3f}" if self_pred else "N/A"
        other_r2 = f"{other_pred['max_r2']:.3f}" if other_pred else "N/A"
        print(f"  {dir_type:18s}  {self_r2:8s}     {other_r2:8s}")

    print(f"\nSteering with d_introspection on self-task: {steering_results['summary'].get('d_introspection_self', {}).get('n_significant_fdr', 0)} FDR-significant layers")
    print(f"Ablating d_introspection on self-task: {ablation_results['summary'].get('d_introspection_self', {}).get('n_significant_fdr', 0)} FDR-significant layers")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  {directions_path.name}")
    print(f"  {results_path.name}")
    print(f"  {similarity_path.name}")
    print(f"  {predictive_path.name}")
    print(f"  {steering_path.name}")
    print(f"  {ablation_path.name}")


if __name__ == "__main__":
    main()
