"""
Stage 3. Ablation (necessity) test for uncertainty, answer, or confidence directions.
Tests whether directions are causally necessary for the model's meta-judgments by
ablating each direction and measuring degradation in stated confidence correlation.

Supports multiple direction types via DIRECTION_TYPE:
- "uncertainty": Entropy/logit_gap directions (from identify_mc_correlate.py)
- "answer": MC answer A/B/C/D directions (from identify_mc_correlate.py with FIND_ANSWER_DIRECTIONS=True)
- "confidence": Stated confidence directions (from test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True)
- "metamcuncert": MC uncertainty directions found from meta activations (from test_meta_transfer.py with FIND_MC_UNCERTAINTY_DIRECTIONS=True)
- "metamcq" / "metamcanswer": MC answer directions found from meta activations (from test_meta_transfer.py with FIND_META_MCQ_DIRECTIONS=True)
- "joint": Ablate the span of multiple configured directions at once

Cross-dataset ablation: Set DIRECTION_DATASET to load directions from a different dataset
than the evaluation dataset (DATASET). Tests whether d_mc generalizes across datasets.

Tests all layers with pooled null distribution + FDR correction.

Inputs:
    outputs/{dir_base}_mc_{metric}_directions.npz         Uncertainty directions
    outputs/{dir_base}_mc_answer_directions.npz           Answer directions (if DIRECTION_TYPE="answer")
    outputs/{dir_base}_meta_{task}_confdir_directions.npz Confidence directions (if DIRECTION_TYPE="confidence")
    outputs/{dir_base}_meta_{task}_mcuncert_directions.npz Meta→MC uncertainty directions (if DIRECTION_TYPE="metamcuncert")
    outputs/{dir_base}_meta_{task}_metamcq_directions.npz Meta→MC answer directions (if DIRECTION_TYPE="metamcq"/"metamcanswer")
    outputs/{base}_mc_results.json                        Consolidated results (dataset + metrics)

    where {dir_base} = DIRECTION_DATASET if set, else DATASET

Outputs (one file per method, with per-position plots):
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_results.json                    (same dataset)
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_from_{dir_base}_results.json   (cross-dataset)
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_cross_layer_propagation.json    (if MEASURE_CROSS_LAYER_PROPAGATION)
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_same_layer_projection.png       (line plot, always generated)
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_propagation_{dir}.png           (heatmaps, only if stride>0 and significant)
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_propagation_{dir}_ranked.txt    (readable top-pair summary)
    outputs/{base}_ablation_{task}_{dir_suffix}_{method}_propagation_{dir}_ranked.png    (readable top-pair bar chart)

    where {base} = {dataset} (model info is in directory path)
          {dir_suffix} = "{direction_type}_{metric}" for uncertainty, else "{direction_type}"
          {method} = "probe" or "mean_diff"
          {position} = token position tested (e.g., "final")
          {dir} = projection direction name (metamcuncert, confdir)

Shared parameters (must match across scripts):
    SEED, TRAIN_SPLIT

Run after: identify_mc_correlate.py
    + test_meta_transfer.py (if using DIRECTION_TYPE="confidence", "metamcuncert", or "metamcq"/"metamcanswer")
"""

import torch
import numpy as np
import json
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
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
from core.steering_experiments import (
    SteeringExperimentConfig,
    BatchAblationHook,
    ActivationCaptureHook,
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
MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None  # Optional: LoRA adapter path (must match identify step if used)
DATASET = "TriviaMC_difficulty_filtered"  # Evaluation dataset (meta-task runs here)
DIRECTION_DATASET = None  # Source of directions (None = use DATASET, or e.g. "PopMC_0_difficulty_filtered")
METRIC = "logit_gap"  # Which metric's directions to test (for uncertainty/metamcuncert)
META_TASK = "delegate"  # "confidence", "delegate", or "other_confidence"
PROBE_POSITION = "options_newline"  # Position from test_meta_transfer.py outputs

# Direction type to ablate:
# - "uncertainty": Ablate uncertainty directions (from identify_mc_correlate.py)
# - "answer": Ablate MC answer directions (from identify_mc_correlate.py with FIND_ANSWER_DIRECTIONS=True)
# - "confidence": Ablate confidence directions (from test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True)
# - "metamcuncert": Ablate MC uncertainty directions found from meta activations (test_meta_transfer.py)
# - "metamcq" / "metamcanswer": Ablate MC answer directions found from meta activations (test_meta_transfer.py)
# - "joint": Ablate the span of multiple directions at once (configured below)
DIRECTION_TYPE = "metamcanswer"

# Optional joint ablation configuration. When provided with 2+ components, the script
# ablates the orthonormal span of all listed directions at each layer.
JOINT_ABLATION_COMPONENTS = [
    {
        "label": "mcuncert",
        "direction_type": "uncertainty",
        "metric": METRIC,
    },
    {
        "label": "metamcuncert",
        "direction_type": "metamcuncert",
        "metric": METRIC,
    },
]

# Optional explicit per-component layer tuples for joint ablations.
# Each entry must provide one layer per JOINT_ABLATION_COMPONENT, in order, either as:
#   - a list/tuple like [26, 30]
#   - or a dict keyed by component label like {"mcuncert": 26, "metamcanswer": 30}
# When None, joint ablation falls back to the original same-layer behavior.
JOINT_ABLATION_LAYER_PAIRS = None

# For DIRECTION_TYPE="confidence": which target was the confdir trained on?
# Must match DELEGATE_CONFDIR_TARGET in test_meta_transfer.py
# - "logit_margin": confdir trained on logit(Answer) - logit(Delegate)
# - "p_answer": confdir trained on P(Answer)
# - None: no suffix (for non-delegate tasks like "confidence" or "other_confidence")
CONFDIR_TARGET = "logit_margin"  # "logit_margin", "p_answer", or None

# --- Mediation Test (Test C) ---
# When True and DIRECTION_TYPE="answer": also measures how ablating MC_Answer
# affects projections onto d_delegate (confdir) direction. Tests the causal link:
# MC_Answer direction → d_delegate_logit_margin direction
# Requires confdir directions to exist (run test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True)
MEASURE_MEDIATION = True

# --- Cross-Layer Propagation Test ---
# When True: measures how ablation at layer L affects projections at downstream layers.
# Default run is configured to ablate d_metamcanswer at options_newline and project onto:
# - d_metamcuncert (meta-task direction predicting MC uncertainty)
# - d_confdir (meta-task decision/confidence readout)
# - stride=0: same-layer only (captures only at ablation layer), generates line plot
# - stride>0: cross-layer mode (captures at L, L+stride, ...), generates line plot + heatmap
# Always generates same-layer line plot; heatmaps only for significant cross-layer effects
MEASURE_CROSS_LAYER_PROPAGATION = True
PROPAGATION_CAPTURE_STRIDE = 5       # 0 = same-layer only; >0 = capture every Nth layer downstream
PROPAGATION_SIGNIFICANCE_THRESHOLD = 2.0  # Effect size threshold: |delta_mean| > threshold * delta_std

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
    "metamcq": {
        "trained_on": "MC answer (A/B/C/D) predicted from meta-task activations",
        "interpretation": "Tests if meta-task contains answer representation in its own context",
    },
    "metamcanswer": {
        "trained_on": "MC answer (A/B/C/D) predicted from meta-task activations",
        "interpretation": "Tests if meta-task contains answer representation in its own context",
    },
    "joint": {
        "trained_on": "orthonormal span of the configured component directions",
        "interpretation": "Tests whether the combined subspace is necessary for the meta-task behavior",
    },
}

# Confidence signal used as the meta-task output target.
# - For META_TASK=delegate:
#     * "prob"        -> P(Answer) from the Answer/Delegate decision tokens.
#     * "logit_margin" -> logit(Answer) - logit(Delegate) for those same tokens.
# - For META_TASK=confidence: this script always uses the stated-confidence option probabilities;
#   if you set CONFIDENCE_SIGNAL="logit_margin" we fall back to "prob" for that task.
CONFIDENCE_SIGNAL = "logit_margin"  # "prob" or "logit_margin"

# --- Quantization ---
LOAD_IN_4BIT = True   # Set True for 70B+ models
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
PROBE_POSITIONS = ["options_newline"]  # ["question_mark", "question_newline", "options_newline", "final"]

# Extra diagnostics
PRINT_DELTA_DIAGNOSTICS = True
DELTA_DIAGNOSTIC_TOPK = 5  # print this many layers for biggest +Δcorr and biggest -Δcorr

# Bootstrap (over questions) for correlation CIs only
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 12345
BOOTSTRAP_CI_ALPHA = 0.05  # 95% CI

# Paired permutation test (swap baseline/ablated labels within question) for Δcorr p-values
PERMUTATION_N = 2000
PERMUTATION_SEED = 23456

# Layer selection from transfer results (for non-final positions)
TRANSFER_R2_THRESHOLD = 0.3  # Layers with R² >= this are tested for non-final positions
TRANSFER_RESULTS_PATH = None  # Auto-detect from MODEL/DATASET if None

# Layer selection for answer directions (based on D2D accuracy)
# When DIRECTION_TYPE="answer", only test layers where answer classifier works
ANSWER_LAYER_SELECTION = True  # Auto-select layers from answer transfer results
ANSWER_D2D_THRESHOLD = 0.8  # D2D accuracy threshold (0.25 = chance for 4-way)

# Control count for non-final positions (final uses NUM_CONTROLS)
NUM_CONTROLS_NONFINAL = 10

# --- Output ---
# Uses centralized path management from core.config_utils

np.random.seed(SEED)
torch.manual_seed(SEED)


# =============================================================================
# TRANSFER RESULTS LOADING (for layer selection)
# =============================================================================

def load_transfer_results(
    base_name: str,
    meta_task: str,
    model_dir: str,
    position: str,
) -> Optional[Dict]:
    """
    Load a position-specific transfer-results JSON to get per-layer R² values.

    Returns None if file not found.
    """
    path = TRANSFER_RESULTS_PATH
    if path is None:
        path = find_output_file(
            f"{base_name}_meta_{meta_task}_transfer_results_{position}.json",
            model_dir=model_dir,
        )
    else:
        raw_path = str(path).format(position=position, pos=position)
        path = Path(raw_path)
        if path.is_dir():
            path = path / f"{base_name}_meta_{meta_task}_transfer_results_{position}.json"
        elif "_transfer_results_" in path.name:
            prefix, suffix = path.name.rsplit("_transfer_results_", 1)
            extension = ""
            if "." in suffix:
                extension = "." + suffix.split(".", 1)[1]
            path = path.with_name(f"{prefix}_transfer_results_{position}{extension}")

    if not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def _get_transfer_metric_section(
    transfer_data: Dict,
    metric: str,
    position: str,
    method: str,
) -> Optional[Dict]:
    """Extract the metric-specific transfer section across supported JSON schemas."""
    if method == "mean_diff":
        current_key = "mean_diff_transfer"
        legacy_key = "mean_diff_by_position"
    else:
        current_key = "transfer"
        legacy_key = "transfer_by_position"

    # Current schema: each position has its own file with top-level transfer sections.
    if current_key in transfer_data and metric in transfer_data[current_key]:
        return transfer_data[current_key][metric]

    # Legacy schema: one file contains all positions under *_by_position.
    if legacy_key in transfer_data and position in transfer_data[legacy_key]:
        pos_data = transfer_data[legacy_key][position]
        if metric in pos_data:
            return pos_data[metric]

    return None


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
        method: Direction method - "probe" uses transfer / transfer_by_position,
                "mean_diff" uses mean_diff_transfer / mean_diff_by_position.

    Returns:
        Sorted list of layer indices meeting threshold
    """
    metric_data = _get_transfer_metric_section(transfer_data, metric, position, method)
    if metric_data is None:
        return []
    per_layer = metric_data.get("per_layer", {})

    selected = []
    for layer_str, layer_data in per_layer.items():
        # Check for centered R² (preferred) or d2m_centered_r2 (legacy)
        r2 = layer_data.get("centered_r2") or layer_data.get("d2m_centered_r2", 0)
        if r2 >= r2_threshold:
            selected.append(int(layer_str))

    return sorted(selected)


def load_answer_transfer_results(base_name: str, meta_task: str, model_dir: str) -> Optional[Dict]:
    """
    Load answer transfer results JSON to get per-layer D2D accuracy.

    Returns None if file not found.
    """
    path = find_output_file(f"{base_name}_meta_{meta_task}_answer_transfer_results_final.json", model_dir=model_dir)

    if path is None or not path.exists():
        return None

    with open(path, "r") as f:
        return json.load(f)


def get_layers_from_answer_transfer(
    answer_transfer_data: Dict,
    d2d_threshold: float = 0.5,
) -> List[int]:
    """
    Get layers where answer classifier works (D2D accuracy >= threshold).

    Args:
        answer_transfer_data: Loaded answer transfer results JSON
        d2d_threshold: Minimum D2D accuracy to include layer (0.25 = chance for 4-way)

    Returns:
        Sorted list of layer indices meeting threshold
    """
    if "by_layer" not in answer_transfer_data:
        return []

    selected = []
    for layer_str, layer_data in answer_transfer_data["by_layer"].items():
        d2d = layer_data.get("d2d_accuracy", 0)
        if d2d >= d2d_threshold:
            selected.append(int(layer_str))

    return sorted(selected)


# =============================================================================
# DIRECTION LOADING
# =============================================================================

def canonicalize_direction_type(direction_type: str) -> str:
    """Map user-facing aliases onto the direction type names used on disk."""
    return {
        "metamcanswer": "metamcq",
    }.get(direction_type, direction_type)


def canonicalize_method_name(method: str) -> str:
    """Treat centroid as the answer-direction analogue of mean_diff."""
    return "mean_diff" if method == "centroid" else method


def choose_method_variant(direction_sets: Dict[str, Dict[int, np.ndarray]], requested_method: str) -> Optional[Tuple[str, Dict[int, np.ndarray]]]:
    """Resolve a requested conceptual method name to the actual stored variant."""
    actual_method = requested_method
    if requested_method not in direction_sets:
        alias = "centroid" if requested_method == "mean_diff" else ("mean_diff" if requested_method == "centroid" else None)
        if alias is None or alias not in direction_sets:
            return None
        actual_method = alias
    return actual_method, direction_sets[actual_method]


def available_canonical_methods(direction_sets: Dict[str, Dict[int, np.ndarray]]) -> List[str]:
    """Return conceptual method names available for a direction set."""
    canonical_to_actual = {}
    for actual_method in direction_sets:
        canonical = canonicalize_method_name(actual_method)
        if canonical not in canonical_to_actual or actual_method == canonical:
            canonical_to_actual[canonical] = actual_method
    return sorted(canonical_to_actual.keys())


def direction_to_basis(direction: np.ndarray) -> np.ndarray:
    """Convert a single direction or basis to a normalized 2D basis array."""
    arr = np.asarray(direction, dtype=np.float32)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        if norm <= 1e-8:
            raise ValueError("Direction has near-zero norm")
        return (arr / norm)[None, :]
    if arr.ndim != 2:
        raise ValueError(f"Expected direction with ndim 1 or 2, got shape {arr.shape}")
    return orthonormalize_rows(arr)


def orthonormalize_rows(vectors: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Build an orthonormal row basis with stable Gram-Schmidt."""
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]

    basis = []
    for vec in arr:
        candidate = vec.copy()
        for prev in basis:
            candidate = candidate - np.dot(candidate, prev) * prev
        norm = float(np.linalg.norm(candidate))
        if norm > tol:
            basis.append(candidate / norm)

    if not basis:
        raise ValueError("Could not construct a non-empty orthonormal basis")

    return np.stack(basis, axis=0).astype(np.float32)


def generate_orthogonal_control_subspaces(reference_basis: np.ndarray, n_subspaces: int, seed: int = 42) -> List[np.ndarray]:
    """Generate random orthonormal control subspaces orthogonal to a reference basis."""
    basis = orthonormalize_rows(reference_basis)
    subspace_dim, hidden_dim = basis.shape
    rng = np.random.RandomState(seed)
    controls: List[np.ndarray] = []

    for _ in range(n_subspaces):
        control_basis = []
        attempts = 0
        while len(control_basis) < subspace_dim:
            attempts += 1
            if attempts > hidden_dim * 4:
                raise RuntimeError("Failed to sample an orthogonal control subspace")

            candidate = rng.randn(hidden_dim)
            for ref_vec in basis:
                candidate = candidate - np.dot(candidate, ref_vec) * ref_vec
            for prev_vec in control_basis:
                candidate = candidate - np.dot(candidate, prev_vec) * prev_vec

            norm = float(np.linalg.norm(candidate))
            if norm <= 1e-6:
                continue
            control_basis.append(candidate / norm)

        controls.append(np.stack(control_basis, axis=0).astype(np.float32))

    return controls


def resolve_joint_component_sets(
    component_specs: List[Dict[str, Any]],
    direction_base: str,
    model_dir: str,
    meta_task: str,
    requested_methods: Optional[List[str]],
) -> Dict[str, Dict[str, Any]]:
    """Resolve per-method component directions for a joint ablation configuration."""
    component_data = []
    for spec in component_specs:
        direction_sets = load_directions(
            direction_base,
            direction_type=spec["direction_type"],
            metric=spec.get("metric", METRIC),
            meta_task=spec.get("meta_task", meta_task),
            model_dir=model_dir,
            confdir_target=spec.get("confdir_target"),
        )
        component_data.append({
            "label": spec["label"],
            "direction_type": spec["direction_type"],
            "direction_sets": direction_sets,
        })

    shared_methods = None
    for component in component_data:
        methods_here = set(available_canonical_methods(component["direction_sets"]))
        shared_methods = methods_here if shared_methods is None else (shared_methods & methods_here)

    if not shared_methods:
        raise ValueError("No shared methods found across joint ablation components")

    if requested_methods is None:
        methods = sorted(shared_methods)
    else:
        methods = [canonicalize_method_name(m) for m in requested_methods if canonicalize_method_name(m) in shared_methods]
        if not methods:
            raise ValueError(
                f"No matching shared methods found for joint ablation. "
                f"Available: {sorted(shared_methods)}, requested: {requested_methods}"
            )

    resolved: Dict[str, Dict[str, Any]] = {}
    component_order = [component["label"] for component in component_data]
    component_types = {
        component["label"]: component["direction_type"]
        for component in component_data
    }

    for method in methods:
        component_directions: Dict[str, Dict[int, np.ndarray]] = {}
        component_methods: Dict[str, str] = {}
        layer_sets = []

        for component in component_data:
            chosen = choose_method_variant(component["direction_sets"], method)
            if chosen is None:
                component_directions = {}
                break
            actual_method, layer_dirs = chosen
            component_directions[component["label"]] = layer_dirs
            component_methods[component["label"]] = actual_method
            layer_sets.append(set(layer_dirs.keys()))

        if not component_directions:
            continue

        common_layers = sorted(set.intersection(*layer_sets)) if layer_sets else []
        resolved[method] = {
            "component_labels": component_order,
            "component_types": component_types,
            "component_methods": component_methods,
            "component_directions": component_directions,
            "common_layers": common_layers,
        }

    if not resolved:
        raise ValueError("Joint ablation components do not share any layers after method resolution")

    return resolved


def build_joint_direction_set(
    component_specs: List[Dict[str, Any]],
    direction_base: str,
    model_dir: str,
    meta_task: str,
    requested_methods: Optional[List[str]],
) -> Tuple[Dict[str, Dict[int, np.ndarray]], Dict[str, Dict[str, Any]]]:
    """Load multiple direction families and combine them into per-layer orthonormal bases."""
    resolved = resolve_joint_component_sets(
        component_specs=component_specs,
        direction_base=direction_base,
        model_dir=model_dir,
        meta_task=meta_task,
        requested_methods=requested_methods,
    )
    joint_directions: Dict[str, Dict[int, np.ndarray]] = {}
    joint_metadata: Dict[str, Dict[str, Any]] = {}

    for method, resolved_method in resolved.items():
        common_layers = resolved_method["common_layers"]
        if not common_layers:
            continue

        method_directions = {}
        layer_ranks = {}
        for layer in common_layers:
            component_vectors = [
                resolved_method["component_directions"][label][layer]
                for label in resolved_method["component_labels"]
            ]
            basis = orthonormalize_rows(np.stack(component_vectors, axis=0))
            method_directions[layer] = basis
            layer_ranks[layer] = int(basis.shape[0])

        joint_directions[method] = method_directions
        joint_metadata[method] = {
            "component_methods": resolved_method["component_methods"],
            "component_labels": resolved_method["component_labels"],
            "component_types": resolved_method["component_types"],
            "layer_ranks": layer_ranks,
        }

    if not joint_directions:
        raise ValueError("Joint ablation components do not share any layers after method resolution")

    return joint_directions, joint_metadata


def is_joint_ablation_enabled() -> bool:
    return DIRECTION_TYPE == "joint"


def has_explicit_joint_layer_pairs() -> bool:
    return is_joint_ablation_enabled() and bool(JOINT_ABLATION_LAYER_PAIRS)


def get_ablation_component_specs() -> List[Dict[str, Any]]:
    if is_joint_ablation_enabled():
        if JOINT_ABLATION_COMPONENTS is None or len(JOINT_ABLATION_COMPONENTS) < 2:
            raise ValueError("DIRECTION_TYPE='joint' requires JOINT_ABLATION_COMPONENTS with at least two entries")
        return JOINT_ABLATION_COMPONENTS

    return [{
        "label": canonicalize_direction_type(DIRECTION_TYPE),
        "direction_type": DIRECTION_TYPE,
        "metric": METRIC,
        "meta_task": META_TASK,
        "confdir_target": CONFDIR_TARGET if DIRECTION_TYPE == "confidence" else None,
    }]


def get_direction_suffix() -> str:
    """Filename-friendly ablation label."""
    if is_joint_ablation_enabled():
        parts = []
        for spec in get_ablation_component_specs():
            label = str(spec["label"])
            direction_type = canonicalize_direction_type(str(spec["direction_type"]))
            metric = spec.get("metric")
            if direction_type in {"uncertainty", "metamcuncert"} and metric:
                label = f"{label}_{metric}"
            parts.append(label)
        return "_plus_".join(parts)

    if DIRECTION_TYPE in {"uncertainty", "metamcuncert"}:
        return f"{DIRECTION_TYPE}_{METRIC}"
    return DIRECTION_TYPE


def get_ablation_title_label() -> str:
    if is_joint_ablation_enabled():
        return " + ".join(str(spec["label"]) for spec in get_ablation_component_specs())
    return DIRECTION_TYPE


def _normalize_joint_layer_pair_entry(entry: Any, component_labels: Sequence[str]) -> List[int]:
    if isinstance(entry, dict):
        missing = [label for label in component_labels if label not in entry]
        if missing:
            raise ValueError(f"Missing joint layer-pair entries for: {missing}")
        return [int(entry[label]) for label in component_labels]

    if not isinstance(entry, (list, tuple)):
        raise ValueError(f"Joint layer pair must be list/tuple or dict, got {type(entry).__name__}")

    if len(entry) != len(component_labels):
        raise ValueError(
            f"Joint layer pair {entry} has {len(entry)} entries, expected {len(component_labels)} "
            f"for components {list(component_labels)}"
        )
    return [int(layer) for layer in entry]


def build_explicit_joint_ablation_conditions(
    resolved_method: Dict[str, Any],
    layer_pairs: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Build explicit multi-layer joint ablation conditions from configured layer tuples."""
    component_labels = list(resolved_method["component_labels"])
    component_directions = resolved_method["component_directions"]
    conditions = []

    for pair_idx, raw_entry in enumerate(layer_pairs):
        pair_layers = _normalize_joint_layer_pair_entry(raw_entry, component_labels)
        layers_by_component = {
            label: layer
            for label, layer in zip(component_labels, pair_layers)
        }

        grouped_vectors: Dict[int, List[np.ndarray]] = {}
        for label, layer in layers_by_component.items():
            if layer not in component_directions[label]:
                raise ValueError(
                    f"Joint layer-pair condition {raw_entry} requests layer {layer} for {label}, "
                    f"but that layer is unavailable for this method"
                )
            grouped_vectors.setdefault(layer, []).append(component_directions[label][layer])

        layer_bases = {
            layer: orthonormalize_rows(np.stack(vectors, axis=0))
            for layer, vectors in grouped_vectors.items()
        }
        display_label = " + ".join(
            f"{label}@L{layers_by_component[label]}"
            for label in component_labels
        )
        conditions.append({
            "key": display_label,
            "display_label": display_label,
            "pair_index": pair_idx,
            "layers_by_component": layers_by_component,
            "ablation_layers": sorted(layer_bases.keys()),
            "layer_bases": layer_bases,
        })

    return conditions


def load_directions(
    base_name: str,
    direction_type: str = "uncertainty",
    metric: str = "entropy",
    meta_task: str = "delegate",
    model_dir: str = None,
    confdir_target: str = None,
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Load direction vectors based on direction type.

    Args:
        base_name: Base name for input files (dataset name)
        direction_type: "uncertainty", "answer", "confidence", "metamcuncert",
                        "metamcq", or "metamcanswer"
        metric: Uncertainty metric (only used for uncertainty/metamcuncert directions)
        meta_task: Meta task (only used for meta-task directions like confidence/metamcuncert/metamcq)
        model_dir: Model directory name
        confdir_target: For direction_type="confidence" with meta_task="delegate":
                        "logit_margin" or "p_answer" (adds suffix to filename)

    Returns:
        Dict mapping method name -> {layer: direction_vector}
        For uncertainty: {"probe": {...}, "mean_diff": {...}}
        For answer: {"answer": {...}}
        For confidence/metamcuncert: {"probe": {...}, "mean_diff": {...}}
    """
    direction_type = canonicalize_direction_type(direction_type)

    if direction_type == "uncertainty":
        path = find_output_file(f"{base_name}_mc_{metric}_directions.npz", model_dir=model_dir)
    elif direction_type == "answer":
        path = find_output_file(f"{base_name}_mc_answer_directions.npz", model_dir=model_dir)
    elif direction_type == "confidence":
        # For delegate task, confdir files have a target suffix (_logit_margin or _p_answer)
        confdir_suffix = f"_{confdir_target}" if confdir_target else ""
        path = find_output_file(f"{base_name}_meta_{meta_task}_confdir{confdir_suffix}_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
    elif direction_type == "metamcuncert":
        # Consolidated file with keys like probe_{metric}_layer_0
        path = find_output_file(f"{base_name}_meta_{meta_task}_mcuncert_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
    elif direction_type == "metamcq":
        # MC answer directions from meta activations (keys like probe_layer_0, centroid_layer_0)
        path = find_output_file(f"{base_name}_meta_{meta_task}_metamcq_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
        if not path.exists():
            path = find_output_file(f"{base_name}_meta_{meta_task}_metaanswer_directions_{PROBE_POSITION}.npz", model_dir=model_dir)
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

    elif direction_type == "metamcq":
        # MC answer directions from meta activations
        # Keys are like "probe_layer_0", "centroid_layer_5"
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


def build_meta_task_prompt_cache(
    tokenizer,
    questions: List[Dict],
    meta_task: str,
    use_chat_template: bool,
    original_indices: Optional[np.ndarray] = None,
) -> Dict:
    """Precompute prompts, mappings, token positions, and padded GPU batches."""
    format_fn = get_format_fn(meta_task)
    prompts = []
    mappings = []
    position_names = ("question_mark", "question_newline", "options_newline", "final")
    position_indices = {name: [] for name in position_names}

    for q_idx, question in enumerate(questions):
        trial_idx = int(original_indices[q_idx]) if original_indices is not None else q_idx
        if meta_task == "delegate":
            prompt, _, mapping = format_fn(
                question,
                tokenizer,
                trial_index=trial_idx,
                use_chat_template=use_chat_template,
            )
        else:
            prompt, _ = format_fn(question, tokenizer, use_chat_template=use_chat_template)
            mapping = None

        prompts.append(prompt)
        mappings.append(mapping)

        positions = find_mc_positions(prompt, tokenizer, question)
        for name in position_names:
            position_indices[name].append(positions.get(name, -1))

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    return {
        "prompts": prompts,
        "mappings": mappings,
        "position_indices": position_indices,
        "cached_inputs": cached_inputs,
        "gpu_batches": gpu_batches,
    }


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
    prompt_cache: Optional[Dict] = None,
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

    if prompt_cache is None:
        prompt_cache = build_meta_task_prompt_cache(
            tokenizer=tokenizer,
            questions=questions,
            meta_task=meta_task,
            use_chat_template=use_chat_template,
            original_indices=original_indices,
        )

    # Get options and signal extraction
    signal_fn = get_signal_fn(meta_task)
    options = get_options(meta_task)

    # Tokenize options
    option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
    ]

    mappings = prompt_cache["mappings"]
    position_indices = prompt_cache["position_indices"].get(position, [-1] * len(questions))

    # Warn if some positions weren't found (will fall back to final token)
    # Note: "final" position is always -1 by design, so don't warn for it
    if position != "final":
        n_valid = sum(1 for idx in position_indices if idx >= 0)
        n_total = len(position_indices)
        if n_valid < n_total:
            print(f"  Warning: {position} position found for {n_valid}/{n_total} prompts (others fall back to final)")

    gpu_batches = prompt_cache["gpu_batches"]

    # Check if we can use KV cache (only for "final" position)
    use_kv_cache = (position == "final")

    # Generate matched-dimension control subspaces for each layer.
    print(f"  Generating {num_controls} control subspaces per layer...")
    controls_by_layer = {}
    basis_dims = {}
    for layer in layers:
        primary_basis = direction_to_basis(directions[layer])
        basis_dims[layer] = int(primary_basis.shape[0])
        controls_by_layer[layer] = generate_orthogonal_control_subspaces(
            primary_basis, num_controls, seed=SEED + layer
        )
    unique_basis_dims = sorted(set(basis_dims.values()))
    if len(unique_basis_dims) == 1:
        print(f"  Primary ablation subspace dimension: {unique_basis_dims[0]}")
    else:
        print(f"  Primary ablation subspace dimension varies by layer: {unique_basis_dims}")

    # Precompute basis tensors
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    cached_directions = {}
    for layer in layers:
        basis_tensor = torch.tensor(direction_to_basis(directions[layer]), dtype=dtype, device=DEVICE)
        ctrl_tensors = [torch.tensor(c, dtype=dtype, device=DEVICE) for c in controls_by_layer[layer]]
        all_bases = torch.stack([basis_tensor] + ctrl_tensors, dim=0)  # (1 + num_controls, k, hidden_dim)
        cached_directions[layer] = {
            "basis": basis_tensor,
            "controls": ctrl_tensors,
            "all_stacked": all_bases,
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
    total_directions = 1 + num_controls  # primary ablation + controls
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
            print(f"  Batched ablation (KV cache): {directions_per_pass} ablation conditions per pass, {num_passes} passes per layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (KV cache): 1 ablation condition per pass")
            total_forward_passes = len(gpu_batches) * len(layers) * total_directions
    else:
        # Full forward path: required for non-final positions (also supports batching)
        if use_batched:
            print(f"  Batched ablation (full forward) at '{position}': {directions_per_pass} conditions/pass, {num_passes} passes/layer")
            total_forward_passes = len(gpu_batches) * len(layers) * num_passes
        else:
            print(f"  Sequential ablation (full forward) at '{position}': {total_directions} conditions per layer")
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
                            dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * k_dirs, dirs_for_pass.shape[1], dirs_for_pass.shape[2])
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

                            dirs_batch = direction_tensor.unsqueeze(0).expand(B, -1, -1)
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

                        run_single_ablation_kv(cached_directions[layer]["basis"], layer_results[layer]["ablated"])
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
                        dirs_batch = dirs_for_pass.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * k_dirs, dirs_for_pass.shape[1], dirs_for_pass.shape[2])

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
                        dirs_batch = cached_directions[layer]["basis"].unsqueeze(0).expand(B, -1, -1)
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
                            dirs_batch = ctrl_dir.unsqueeze(0).expand(B, -1, -1)
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


def run_joint_layer_pair_ablation_for_method(
    model,
    tokenizer,
    questions: List[Dict],
    metric_values: np.ndarray,
    conditions: List[Dict[str, Any]],
    num_controls: int,
    meta_task: str,
    use_chat_template: bool,
    position: str = "final",
    original_indices: Optional[np.ndarray] = None,
    prompt_cache: Optional[Dict] = None,
) -> Dict:
    """Run joint ablations for an explicit list of multi-layer conditions."""
    if not conditions:
        return {"error": "No explicit joint ablation conditions"}

    if prompt_cache is None:
        prompt_cache = build_meta_task_prompt_cache(
            tokenizer=tokenizer,
            questions=questions,
            meta_task=meta_task,
            use_chat_template=use_chat_template,
            original_indices=original_indices,
        )

    signal_fn = get_signal_fn(meta_task)
    options = get_options(meta_task)
    option_token_ids = [
        tokenizer.encode(opt, add_special_tokens=False)[0] for opt in options
    ]

    mappings = prompt_cache["mappings"]
    position_indices = prompt_cache["position_indices"].get(position, [-1] * len(questions))
    if position != "final":
        n_valid = sum(1 for idx in position_indices if idx >= 0)
        n_total = len(position_indices)
        if n_valid < n_total:
            print(f"  Warning: {position} position found for {n_valid}/{n_total} prompts (others fall back to final)")

    gpu_batches = prompt_cache["gpu_batches"]
    use_kv_cache = (position == "final")
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    print(f"  Generating {num_controls} control subspaces per explicit joint condition...")
    cached_conditions: Dict[str, Dict[str, Any]] = {}
    basis_dim_summaries = []
    for cond_idx, condition in enumerate(conditions):
        cond_key = condition["key"]
        per_layer = {}
        layer_ranks = {}
        for layer, basis_np in condition["layer_bases"].items():
            basis_np = direction_to_basis(basis_np)
            layer_ranks[layer] = int(basis_np.shape[0])
            controls_np = generate_orthogonal_control_subspaces(
                basis_np, num_controls, seed=SEED + cond_idx * 1000 + layer
            )
            basis_tensor = torch.tensor(basis_np, dtype=dtype, device=DEVICE)
            ctrl_tensors = [torch.tensor(c, dtype=dtype, device=DEVICE) for c in controls_np]
            all_stacked = torch.stack([basis_tensor] + ctrl_tensors, dim=0)
            per_layer[layer] = {
                "basis": basis_tensor,
                "controls": ctrl_tensors,
                "all_stacked": all_stacked,
            }
        cached_conditions[cond_key] = {
            "condition": condition,
            "per_layer": per_layer,
            "layer_ranks": layer_ranks,
        }
        basis_dim_summaries.append(
            f"{condition['display_label']} -> "
            + ", ".join(f"L{layer}:k={rank}" for layer, rank in sorted(layer_ranks.items()))
        )
    for summary in basis_dim_summaries:
        print(f"    {summary}")

    condition_labels = [condition["key"] for condition in conditions]
    baseline_results = [None] * len(questions)
    layer_results = {}
    for cond_key in condition_labels:
        layer_results[cond_key] = {
            "baseline": baseline_results,
            "ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)},
        }

    total_directions = 1 + num_controls
    if EXPANDED_BATCH_TARGET is not None and EXPANDED_BATCH_TARGET > 0:
        directions_per_pass = max(1, EXPANDED_BATCH_TARGET // BATCH_SIZE)
        directions_per_pass = min(directions_per_pass, total_directions)
        use_batched = directions_per_pass > 1
    else:
        directions_per_pass = 1
        use_batched = False
    num_passes = (total_directions + directions_per_pass - 1) // directions_per_pass if use_batched else total_directions

    if use_kv_cache:
        if use_batched:
            print(
                f"  Batched explicit-pair ablation (KV cache): {directions_per_pass} conditions/pass, "
                f"{num_passes} passes per pair"
            )
            total_forward_passes = len(gpu_batches) * len(conditions) * num_passes
        else:
            print("  Sequential explicit-pair ablation (KV cache): 1 condition per pass")
            total_forward_passes = len(gpu_batches) * len(conditions) * total_directions
    else:
        if use_batched:
            print(
                f"  Batched explicit-pair ablation (full forward) at '{position}': "
                f"{directions_per_pass} conditions/pass, {num_passes} passes per pair"
            )
            total_forward_passes = len(gpu_batches) * len(conditions) * num_passes
        else:
            print(
                f"  Sequential explicit-pair ablation (full forward) at '{position}': "
                f"{total_directions} conditions per pair"
            )
            total_forward_passes = len(gpu_batches) * len(conditions) * total_directions

    print(f"  Total forward passes: {total_forward_passes}")
    pbar = tqdm(total=total_forward_passes, desc=f"  Ablation ({position})")

    def _store_result(result_list: List[Optional[Dict[str, Any]]], q_idx: int, probs_row, logits_row) -> None:
        resp = options[int(np.argmax(probs_row))]
        conf, p_answer, logit_margin = _compute_confidence_used(meta_task, probs_row, logits_row, mappings[q_idx], signal_fn)
        m_val = metric_values[q_idx]
        result_list[q_idx] = {
            "question_idx": q_idx,
            "response": resp,
            "confidence": float(conf),
            "metric": float(m_val),
            "p_answer": (float(p_answer) if p_answer is not None else None),
            "logit_margin": (float(logit_margin) if logit_margin is not None else None),
        }

    def _register_condition_hooks(
        condition_cache: Dict[str, Any],
        position_indices_tensor: Optional[torch.Tensor] = None,
    ) -> Dict[int, BatchAblationHook]:
        hooks = {}
        for layer in condition_cache["condition"]["ablation_layers"]:
            hook = BatchAblationHook(intervention_position="last" if use_kv_cache else "indexed")
            if position_indices_tensor is not None:
                hook.set_position_indices(position_indices_tensor)
            if hasattr(model, "get_base_model"):
                layer_module = model.get_base_model().model.layers[layer]
            else:
                layer_module = model.model.layers[layer]
            hook.register(layer_module)
            hooks[layer] = hook
        return hooks

    for batch_indices, batch_inputs in gpu_batches:
        B = len(batch_indices)

        if use_kv_cache:
            base_step_data = get_kv_cache(model, batch_inputs)
            keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

            inputs_template = {
                "input_ids": base_step_data["input_ids"],
                "attention_mask": base_step_data["attention_mask"],
                "use_cache": True,
            }
            if "position_ids" in base_step_data:
                inputs_template["position_ids"] = base_step_data["position_ids"]

            if baseline_results[batch_indices[0]] is None:
                baseline_inputs = inputs_template.copy()
                baseline_inputs["past_key_values"] = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                with torch.inference_mode():
                    out = model(**baseline_inputs)
                    probs, logits_np = _extract_probs_logits(out, option_token_ids)
                for i, q_idx in enumerate(batch_indices):
                    _store_result(baseline_results, q_idx, probs[i], logits_np[i])

            for cond_key in condition_labels:
                condition_cache = cached_conditions[cond_key]
                hooks = _register_condition_hooks(condition_cache)
                try:
                    if use_batched:
                        for pass_start in range(0, total_directions, directions_per_pass):
                            pass_end = min(pass_start + directions_per_pass, total_directions)
                            k_dirs = pass_end - pass_start

                            expanded_inputs = {
                                "input_ids": inputs_template["input_ids"].repeat_interleave(k_dirs, dim=0),
                                "attention_mask": inputs_template["attention_mask"].repeat_interleave(k_dirs, dim=0),
                                "use_cache": True,
                                "past_key_values": create_fresh_cache(keys_snapshot, values_snapshot, expand_size=k_dirs),
                            }
                            if "position_ids" in inputs_template:
                                expanded_inputs["position_ids"] = inputs_template["position_ids"].repeat_interleave(k_dirs, dim=0)

                            for layer, hook in hooks.items():
                                all_bases = condition_cache["per_layer"][layer]["all_stacked"]
                                bases_for_pass = all_bases[pass_start:pass_end]
                                dirs_batch = (
                                    bases_for_pass.unsqueeze(0)
                                    .expand(B, -1, -1, -1)
                                    .reshape(B * k_dirs, bases_for_pass.shape[1], bases_for_pass.shape[2])
                                )
                                hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**expanded_inputs)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
                                    prob_idx = i * k_dirs + j
                                    target_list = (
                                        layer_results[cond_key]["ablated"]
                                        if dir_idx == 0
                                        else layer_results[cond_key]["controls_ablated"][f"control_{dir_idx - 1}"]
                                    )
                                    _store_result(target_list, q_idx, probs[prob_idx], logits_np[prob_idx])
                            pbar.update(1)
                    else:
                        for dir_idx in range(total_directions):
                            for layer, hook in hooks.items():
                                basis_tensor = (
                                    condition_cache["per_layer"][layer]["basis"]
                                    if dir_idx == 0
                                    else condition_cache["per_layer"][layer]["controls"][dir_idx - 1]
                                )
                                hook.set_directions(basis_tensor.unsqueeze(0).expand(B, -1, -1))

                            current_inputs = inputs_template.copy()
                            current_inputs["past_key_values"] = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)
                            with torch.inference_mode():
                                out = model(**current_inputs)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            target_list = (
                                layer_results[cond_key]["ablated"]
                                if dir_idx == 0
                                else layer_results[cond_key]["controls_ablated"][f"control_{dir_idx - 1}"]
                            )
                            for i, q_idx in enumerate(batch_indices):
                                _store_result(target_list, q_idx, probs[i], logits_np[i])
                            pbar.update(1)
                finally:
                    for hook in hooks.values():
                        hook.remove()
        else:
            batch_pos_indices = []
            seq_len = batch_inputs["input_ids"].shape[1]
            for i, q_idx in enumerate(batch_indices):
                pos = position_indices[q_idx]
                if pos >= 0:
                    actual_len = int(batch_inputs["attention_mask"][i].sum())
                    pad_offset = seq_len - actual_len
                    adjusted_pos = pos + pad_offset
                else:
                    adjusted_pos = seq_len - 1
                batch_pos_indices.append(adjusted_pos)
            batch_pos_tensor = torch.tensor(batch_pos_indices, dtype=torch.long, device=DEVICE)

            if baseline_results[batch_indices[0]] is None:
                with torch.inference_mode():
                    out = model(**batch_inputs, use_cache=False)
                    probs, logits_np = _extract_probs_logits(out, option_token_ids)
                for i, q_idx in enumerate(batch_indices):
                    _store_result(baseline_results, q_idx, probs[i], logits_np[i])

            for cond_key in condition_labels:
                condition_cache = cached_conditions[cond_key]
                hooks = _register_condition_hooks(condition_cache, position_indices_tensor=batch_pos_tensor)
                try:
                    if use_batched:
                        for pass_start in range(0, total_directions, directions_per_pass):
                            pass_end = min(pass_start + directions_per_pass, total_directions)
                            k_dirs = pass_end - pass_start

                            expanded_inputs = {
                                "input_ids": batch_inputs["input_ids"].repeat_interleave(k_dirs, dim=0),
                                "attention_mask": batch_inputs["attention_mask"].repeat_interleave(k_dirs, dim=0),
                            }
                            expanded_pos_tensor = batch_pos_tensor.repeat_interleave(k_dirs)

                            for hook in hooks.values():
                                hook.set_position_indices(expanded_pos_tensor)

                            for layer, hook in hooks.items():
                                all_bases = condition_cache["per_layer"][layer]["all_stacked"]
                                bases_for_pass = all_bases[pass_start:pass_end]
                                dirs_batch = (
                                    bases_for_pass.unsqueeze(0)
                                    .expand(B, -1, -1, -1)
                                    .reshape(B * k_dirs, bases_for_pass.shape[1], bases_for_pass.shape[2])
                                )
                                hook.set_directions(dirs_batch)

                            with torch.inference_mode():
                                out = model(**expanded_inputs, use_cache=False)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            for i, q_idx in enumerate(batch_indices):
                                for j in range(k_dirs):
                                    dir_idx = pass_start + j
                                    prob_idx = i * k_dirs + j
                                    target_list = (
                                        layer_results[cond_key]["ablated"]
                                        if dir_idx == 0
                                        else layer_results[cond_key]["controls_ablated"][f"control_{dir_idx - 1}"]
                                    )
                                    _store_result(target_list, q_idx, probs[prob_idx], logits_np[prob_idx])
                            pbar.update(1)
                    else:
                        for dir_idx in range(total_directions):
                            for layer, hook in hooks.items():
                                hook.set_position_indices(batch_pos_tensor)
                                basis_tensor = (
                                    condition_cache["per_layer"][layer]["basis"]
                                    if dir_idx == 0
                                    else condition_cache["per_layer"][layer]["controls"][dir_idx - 1]
                                )
                                hook.set_directions(basis_tensor.unsqueeze(0).expand(B, -1, -1))

                            with torch.inference_mode():
                                out = model(**batch_inputs, use_cache=False)
                                probs, logits_np = _extract_probs_logits(out, option_token_ids)

                            target_list = (
                                layer_results[cond_key]["ablated"]
                                if dir_idx == 0
                                else layer_results[cond_key]["controls_ablated"][f"control_{dir_idx - 1}"]
                            )
                            for i, q_idx in enumerate(batch_indices):
                                _store_result(target_list, q_idx, probs[i], logits_np[i])
                            pbar.update(1)
                finally:
                    for hook in hooks.values():
                        hook.remove()

    pbar.close()

    condition_metadata = [
        {
            "label": condition["display_label"],
            "layers_by_component": condition["layers_by_component"],
            "ablation_layers": condition["ablation_layers"],
        }
        for condition in conditions
    ]
    return {
        "layers": condition_labels,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results,
        "position": position,
        "condition_type": "joint_layer_pairs",
        "condition_axis_label": "Layer pair",
        "condition_metadata": condition_metadata,
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

    raw_adj = [min(1.0, (p * n) / rank) for rank, (_, p) in enumerate(items, 1)]

    # BH adjusted p-values use a reverse cumulative minimum in sorted-p order.
    monotone_adj = [0.0] * n
    running_min = 1.0
    for idx in range(n - 1, -1, -1):
        running_min = min(running_min, raw_adj[idx])
        monotone_adj[idx] = running_min

    return {
        layer: float(monotone_adj[idx])
        for idx, (layer, _) in enumerate(items)
    }


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


def _corr_rows(x_rows: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute Pearson correlation for each row in x_rows against a fixed y."""
    if x_rows.ndim != 2:
        raise ValueError(f"x_rows must be 2D, got shape {x_rows.shape}")

    n = y.shape[0]
    if n < 2:
        return np.zeros(x_rows.shape[0], dtype=np.float32)

    y = np.asarray(y, dtype=np.float32)
    yc = y - y.mean()
    denom_n = float(n - 1)
    sy = np.sqrt((yc * yc).sum() / denom_n)
    if sy <= 1e-12:
        return np.zeros(x_rows.shape[0], dtype=np.float32)

    x_rows = np.asarray(x_rows, dtype=np.float32)
    xc = x_rows - x_rows.mean(axis=1, keepdims=True)
    cov = (xc * yc[None, :]).sum(axis=1) / denom_n
    sx = np.sqrt((xc * xc).sum(axis=1) / denom_n)

    denom = sx * sy
    out = np.zeros(x_rows.shape[0], dtype=np.float32)
    ok = denom > 1e-12
    out[ok] = (cov[ok] / denom[ok]).astype(np.float32)
    return out


def _paired_permutation_pvalue(
    baseline_conf: np.ndarray,
    ablated_conf: np.ndarray,
    metric_values: np.ndarray,
    n_perm: int,
    seed: int,
) -> float:
    """Two-sided paired permutation p-value for Δcorr under label exchangeability."""
    if n_perm <= 0:
        return 1.0

    observed = compute_correlation(ablated_conf, metric_values) - compute_correlation(baseline_conf, metric_values)
    if baseline_conf.shape[0] < 2:
        return 1.0

    avg_conf = 0.5 * (baseline_conf + ablated_conf)
    half_delta = 0.5 * (ablated_conf - baseline_conf)
    if np.std(half_delta) <= 1e-12:
        return 1.0

    rng = np.random.default_rng(seed)
    signs = rng.integers(0, 2, size=(n_perm, baseline_conf.shape[0]), dtype=np.int8).astype(np.float32)
    signs = (2.0 * signs) - 1.0

    perm_ablated = avg_conf[None, :] + signs * half_delta[None, :]
    perm_baseline = avg_conf[None, :] - signs * half_delta[None, :]
    perm_delta = _corr_rows(perm_ablated, metric_values) - _corr_rows(perm_baseline, metric_values)

    n_extreme = int(np.sum(np.abs(perm_delta) >= abs(observed)))
    return float((n_extreme + 1) / (n_perm + 1))


def _primary_fdr_value(layer_stats: Dict) -> float:
    """Primary layer-wise FDR result, preferring paired-permutation values."""
    return float(layer_stats.get("p_value_permutation_fdr", layer_stats.get("p_value_bootstrap_fdr", 1.0)))


def _primary_sig_count(summary: Dict) -> int:
    """Primary significant-layer count, preferring paired-permutation values."""
    return int(summary.get("n_significant_permutation_fdr", summary.get("n_significant_bootstrap_fdr", 0)))


def _condition_seed_offset(condition: Any) -> int:
    """Stable integer seed offset for layer indices or string-labeled conditions."""
    if isinstance(condition, (int, np.integer)):
        return int(condition)
    return int(zlib.crc32(str(condition).encode("utf-8")) & 0xFFFFFFFF)


def analyze_ablation_results(results: Dict, metric: str, base_name: str) -> Dict:
    """Compute ablation effect statistics.

    Uses bootstrap CIs for estimation and a paired permutation test for Δcorr
    p-values/FDR across layers. When controls exist, retains the pooled-control
    null comparison as a secondary diagnostic.
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
        "condition_type": results.get("condition_type", "layer"),
        "condition_axis_label": results.get("condition_axis_label", "Layer"),
        "condition_metadata": results.get("condition_metadata"),
        "bootstrap": {
            "n": BOOTSTRAP_N,
            "seed": BOOTSTRAP_SEED,
            "ci_alpha": BOOTSTRAP_CI_ALPHA,
        },
        "hypothesis_test": {
            "method": "paired_permutation",
            "n": PERMUTATION_N,
            "seed": PERMUTATION_SEED,
        },
        "per_layer": {},
        # Metadata for reproducibility
        "direction_type": DIRECTION_TYPE,
        "ablation_label": get_ablation_title_label(),
        "model_name": MODEL.split("/")[-1],  # Just the model name, not full path
        "dataset": dataset_name,
        "quantization": quant_str,
        "meta_task": META_TASK,
    }

    if not layers:
        analysis["summary"] = {
            "pooled_null_size": 0,
            "n_significant_fdr": 0,
            "significant_layers_permutation_fdr": [],
            "n_significant_permutation_fdr": 0,
            "significant_layers_bootstrap_fdr": [],
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

        # Two-sided paired permutation p-value for Δcorr != 0
        p_perm = _paired_permutation_pvalue(
            baseline_conf,
            ablated_conf,
            baseline_metric,
            n_perm=PERMUTATION_N,
            seed=PERMUTATION_SEED + _condition_seed_offset(layer),
        )

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

        # Bootstrap CI for mean Δconf (reuse idx matrix from correlation bootstrap)
        boot_delta_conf_means = delta_conf[idx].mean(axis=1)
        delta_conf_ci = np.quantile(boot_delta_conf_means, [lo, hi]).astype(np.float32)

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
            "p_value_permutation_delta": p_perm,

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
            "delta_conf_mean_ci95": [float(delta_conf_ci[0]), float(delta_conf_ci[1])],
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

    # Paired-permutation BH-FDR
    raw_p_perm = {layer: layer_data[layer]["p_value_permutation_delta"] for layer in layers}
    fdr_perm = _bh_fdr(raw_p_perm)

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
            "p_value_permutation_delta": float(ld["p_value_permutation_delta"]),
            "p_value_permutation_fdr": float(fdr_perm.get(layer, 1.0)),
            # Backward-compatible aliases for existing consumers.
            "p_value_bootstrap_delta": float(ld["p_value_permutation_delta"]),
            "p_value_bootstrap_fdr": float(fdr_perm.get(layer, 1.0)),

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
            "delta_conf_mean_ci95": ld["delta_conf_mean_ci95"],
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
    sig_perm_fdr = [l for l in layers if _primary_fdr_value(per[l]) < 0.05]

    best_layer_z = max(layers, key=lambda l: abs(per[l]["effect_size_z"]))
    best_layer_abs_delta = max(layers, key=lambda l: abs(per[l]["correlation_change"]))

    analysis["summary"] = {
        "pooled_null_size": int(pooled_null.size),
        "significant_layers_fdr": sig_controls_fdr,
        "n_significant_fdr": len(sig_controls_fdr),
        "significant_layers_permutation_fdr": sig_perm_fdr,
        "n_significant_permutation_fdr": len(sig_perm_fdr),
        # Backward-compatible aliases for existing consumers.
        "significant_layers_bootstrap_fdr": sig_perm_fdr,
        "n_significant_bootstrap_fdr": len(sig_perm_fdr),
        "best_layer": best_layer_z,
        "best_effect_z": float(per[best_layer_z]["effect_size_z"]),
        "best_layer_abs_delta": best_layer_abs_delta,
        "best_abs_delta": float(per[best_layer_abs_delta]["correlation_change"]),
    }

    return analysis


# =============================================================================
# VISUALIZATION
# =============================================================================

def _condition_axis_label_from_analysis(analysis: Dict[str, Any]) -> str:
    return str(analysis.get("condition_axis_label", "Layer"))


def _condition_axis_label_from_results(results: Dict[str, Any]) -> str:
    return str(results.get("condition_axis_label", "Layer"))


def _format_condition_value(value: Any, axis_label: str) -> str:
    if isinstance(value, (int, np.integer)):
        prefix = "L" if axis_label.lower() == "layer" else ""
        return f"{prefix}{int(value)}"
    return str(value)


def _xtick_step(num_items: int) -> int:
    return 1 if num_items <= 8 else 2


def _xtick_rotation(labels: Sequence[Any]) -> int:
    return 30 if any(len(str(label)) > 8 for label in labels) else 0


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
    p_fdr = np.array([_primary_fdr_value(per[l]) for l in layers], dtype=np.float32)
    metric_sign = float(analysis.get("metric_sign", 1.0))
    aligned_delta = metric_sign * real_delta
    axis_label = _condition_axis_label_from_analysis(analysis)
    tick_step = _xtick_step(len(layers))
    tick_rotation = _xtick_rotation(layers)

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

    ax_actual.set_xticks(x[::tick_step])
    ax_actual.set_xticklabels([layers[i] for i in range(0, len(layers), tick_step)], rotation=tick_rotation, ha='right' if tick_rotation else 'center')
    ax_actual.set_xlabel(axis_label)
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
                         label='Perm FDR < 0.05')

    # Pick the strongest effect in metric-aligned space so entropy-style metrics
    # don't get misread as "helpful" just because raw Δr is positive.
    harmful_idx = int(np.argmin(aligned_delta))
    helpful_idx = int(np.argmax(aligned_delta))
    peak_idx = harmful_idx if aligned_delta[harmful_idx] < 0 else helpful_idx
    peak_layer = layers[peak_idx]
    peak_val = float(real_delta[peak_idx])
    peak_aligned = float(aligned_delta[peak_idx])
    peak_label = "Peak degradation" if peak_aligned < 0 else "Peak increase"

    # Position annotation to avoid overlap
    text_x = peak_idx + 3 if peak_idx < len(layers) - 5 else peak_idx - 8
    text_y = peak_val - 0.015 if peak_val <= 0 else peak_val + 0.015
    ax_delta.annotate(f'{peak_label} {_format_condition_value(peak_layer, axis_label)}: Δr = {peak_val:+.3f}',
                      xy=(peak_idx, peak_val), xytext=(text_x, text_y),
                      fontsize=9, arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
                      bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                               edgecolor='gray', alpha=0.9))

    ax_delta.set_xticks(x[::tick_step])
    ax_delta.set_xticklabels([layers[i] for i in range(0, len(layers), tick_step)], rotation=tick_rotation, ha='right' if tick_rotation else 'center')
    ax_delta.set_xlabel(axis_label)
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
    peak_aligned_ci = sorted([metric_sign * float(peak_ci[0]), metric_sign * float(peak_ci[1])])
    n_sig = int(np.sum(p_fdr < 0.05))
    ctrl_mean = float(np.mean([per[l]["control_correlation_change_mean"] for l in layers]))
    n_sig_degrade = int(np.sum((p_fdr < 0.05) & (aligned_delta < 0)))
    n_sig_improve = int(np.sum((p_fdr < 0.05) & (aligned_delta > 0)))

    # Count layers where real effect is outside control band
    if has_controls:
        outside_band = int(np.sum((real_delta < ctrl_lo) | (real_delta > ctrl_hi)))
    else:
        outside_band = n_sig

    meta_task = analysis.get("meta_task", analysis.get("config", {}).get("meta_task", "unknown"))
    num_controls = analysis.get("num_controls", 0)
    bootstrap_n = analysis.get("bootstrap", {}).get("n", 0)
    permutation_n = analysis.get("hypothesis_test", {}).get("n", 0)
    conf_signal = analysis.get("confidence_signal", "prob")

    # Extract metadata for reproducibility
    model_name = analysis.get("model_name", "unknown")
    dataset = analysis.get("dataset", "unknown")
    quantization = analysis.get("quantization", "unknown")
    direction_type = analysis.get("ablation_label", analysis.get("direction_type", "uncertainty"))

    # Extract position from output filename if available
    position = "unknown"
    fname = output_path.stem if output_path else ""
    for pos in ["final", "optionsnewline", "questionnewline", "questionmark"]:
        if pos in fname.lower().replace("_", ""):
            position = pos.replace("newline", "_newline").replace("mark", "_mark")
            break

    # Format quantization for display
    quant_display = f" ({quantization})" if quantization != "none" else ""

    if n_sig_degrade > 0:
        interpretation = (
            f"Ablating {direction_type} reduces metric-aligned calibration at {n_sig_degrade} layer(s)"
        )
    elif n_sig_improve > 0:
        interpretation = (
            f"Ablating {direction_type} increases metric-aligned calibration at {n_sig_improve} layer(s)"
        )
    else:
        interpretation = "No paired-permutation-FDR-significant change in metric-aligned calibration"
    if has_controls:
        interpretation += "; random directions stay near zero"

    # Horizontal summary format for full-width panel
    summary_text = (
        f"CAUSAL NECESSITY TEST | Model: {model_name}{quant_display} | Dataset: {dataset}\n"
        f"Direction: {method} {direction_type} ({analysis['metric']}) | Task: {meta_task} | Position: {position}\n"
        f"N = {analysis['num_questions']} | Controls = {num_controls} | Bootstrap CI = {bootstrap_n} | Permutations = {permutation_n} | Signal = {conf_signal}\n\n"
        f"BASELINE: r = {baseline_corr:.3f} [{baseline_ci_lo_mean:.2f}, {baseline_ci_hi_mean:.2f}]    |    "
        f"{peak_label.upper()}: {axis_label} {_format_condition_value(peak_layer, axis_label)}, Δaligned = {peak_aligned:+.3f} "
        f"[{peak_aligned_ci[0]:+.3f}, {peak_aligned_ci[1]:+.3f}] (raw Δr = {peak_val:+.3f})\n"
        f"Significant: {n_sig}/{len(layers)} layers (paired-permutation FDR<0.05)    |    "
        f"Controls: mean Δr ≈ {ctrl_mean:.3f}, outside band: {outside_band} layers\n\n"
        f"INTERPRETATION: {interpretation}"
    )

    ax_summary.text(0.5, 0.5, summary_text, transform=ax_summary.transAxes, fontsize=10,
                    verticalalignment='center', horizontalalignment='center',
                    fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='white',
                              edgecolor='gray', alpha=0.9))

    fig.suptitle(f'Ablation: {method.upper()} direction ({analysis["metric"]})',
                 fontsize=12, fontweight='bold')

    save_figure(fig, output_path)


def plot_confidence_impact(analysis: Dict, method: str, position: str, output_path: Path):
    """
    Create single-panel figure showing mean Δconf by layer with bootstrap 95% CI.
    """
    layers = analysis.get("layers", [])
    if not layers:
        print(f"  Skipping confidence plot for {method} - no layers")
        return

    per = analysis["per_layer"]

    # Extract data
    delta_conf_mean = np.array([per[l]["delta_conf_mean"] for l in layers], dtype=np.float32)
    ci_lo = np.array([per[l]["delta_conf_mean_ci95"][0] for l in layers], dtype=np.float32)
    ci_hi = np.array([per[l]["delta_conf_mean_ci95"][1] for l in layers], dtype=np.float32)
    axis_label = _condition_axis_label_from_analysis(analysis)
    tick_step = _xtick_step(len(layers))
    tick_rotation = _xtick_rotation(layers)

    # Create single-panel figure
    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(layers))

    # Reference line at y=0
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)

    # CI band and line
    ax.fill_between(x, ci_lo, ci_hi, color=CONDITION_COLORS["ablated"], alpha=CI_ALPHA)
    ax.plot(x, delta_conf_mean, 'o-', color=CONDITION_COLORS["ablated"],
            markersize=4, linewidth=1.5)

    ax.set_xticks(x[::tick_step])
    ax.set_xticklabels([layers[i] for i in range(0, len(layers), tick_step)], rotation=tick_rotation, ha='right' if tick_rotation else 'center')
    ax.set_xlabel(axis_label)
    ax.set_ylabel('Δ Confidence (ablated − baseline)')
    ax.grid(True, alpha=GRID_ALPHA)

    # Title with key info
    metric = analysis.get("metric", "unknown")
    peak_idx = int(np.argmin(np.abs(delta_conf_mean)))
    peak_layer = layers[peak_idx]
    peak_val = float(delta_conf_mean[peak_idx])
    peak_ci = [float(ci_lo[peak_idx]), float(ci_hi[peak_idx])]
    peak_direction = "decrease" if peak_val < 0 else "increase"

    ax.set_title(f'Confidence Impact: {method.upper()} {metric} ({position})\n'
                 f'Peak {peak_direction}: {_format_condition_value(peak_layer, axis_label)} Δconf = {peak_val:.3f} [{peak_ci[0]:.3f}, {peak_ci[1]:.3f}]',
                 fontsize=11)

    save_figure(fig, output_path)


def plot_method_comparison(analyses: Dict[str, Dict], output_path: Path):
    """Comparison plot of different direction methods.

    Shows Δcorr with bootstrap 95% CI bands and marks layers significant under
    paired-permutation BH-FDR.
    """
    methods = list(analyses.keys())
    if len(methods) < 2:
        print("  Skipping comparison plot - need at least 2 methods")
        return

    layers = analyses[methods[0]].get("layers", [])
    if not layers:
        print("  Skipping comparison plot - no layers")
        return
    axis_label = _condition_axis_label_from_analysis(analyses[methods[0]])
    tick_rotation = _xtick_rotation(layers)

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
        p_fdr = np.array([_primary_fdr_value(per[l]) for l in layers], dtype=np.float32)

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
    ax1.set_xticklabels(layers, rotation=tick_rotation, ha='right' if tick_rotation else 'center')
    ax1.set_xlabel(axis_label)
    ax1.set_ylabel("ΔCorrelation (Ablated - Baseline)")
    ax1.set_title("Δcorr by Method (filled markers = paired-permutation FDR<0.05)")
    ax1.legend()
    ax1.grid(True, alpha=GRID_ALPHA)

    # Panel 2: Summary
    ax2 = axes[1]
    ax2.axis("off")

    comparison_text = (
        "METHOD COMPARISON (paired-permutation FDR)\n"
        + "=" * 50
        + "\n\n"
    )
    for method in methods:
        summary = analyses[method].get("summary", {})
        comparison_text += f"{method.upper()}:\n"
        comparison_text += f"  Significant layers (perm FDR<0.05): {_primary_sig_count(summary)}\n"
        comparison_text += (
            f"  Best |Δ| layer: {summary.get('best_layer_abs_delta')} "
            f"(Δ={summary.get('best_abs_delta', 0.0):+.3f})\n\n"
        )

    best_method = max(methods, key=lambda m: _primary_sig_count(analyses[m].get("summary", {})))
    comparison_text += f"Method with more perm-FDR-significant layers: {best_method.upper()}\n"

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
# MEDIATION TEST (Test C: MC_Answer → d_delegate)
# =============================================================================

def run_mediation_test(
    model,
    tokenizer,
    questions: List[Dict],
    mc_answer_directions: Dict[int, np.ndarray],
    confdir_directions: Dict[int, np.ndarray],
    meta_task: str,
    use_chat_template: bool,
    layers: List[int],
    original_indices: np.ndarray,
    prompt_cache: Optional[Dict] = None,
) -> Dict:
    """
    Test whether ablating MC_Answer direction affects projections onto d_delegate.

    For each layer:
    1. Run baseline (no ablation), capture activations, project onto confdir
    2. Run with MC_Answer ablated, capture activations, project onto confdir
    3. Compute delta = ablated_projection - baseline_projection

    If MC_Answer causally affects d_delegate, we expect ablation to change projections.

    Returns:
        {
            "layers": [...],
            "per_layer": {
                layer: {
                    "baseline_projections": [...],  # Per-question
                    "ablated_projections": [...],
                    "delta_mean": float,
                    "delta_std": float,
                    "pearson_r": float,  # Correlation between baseline and ablated
                }
            }
        }
    """
    # Get layer modules
    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model()
        layer_modules = base_model.model.layers
    else:
        layer_modules = model.model.layers

    if prompt_cache is None:
        prompt_cache = build_meta_task_prompt_cache(
            tokenizer=tokenizer,
            questions=questions,
            meta_task=meta_task,
            use_chat_template=use_chat_template,
            original_indices=original_indices,
        )

    gpu_batches = prompt_cache["gpu_batches"]
    device = next(model.parameters()).device
    results = {"layers": layers, "per_layer": {}}

    # Get model dtype for direction conversion (quantized models use float16)
    model_dtype = next(model.parameters()).dtype
    valid_layers = [l for l in layers if l in mc_answer_directions and l in confdir_directions]
    if not valid_layers:
        return results

    layer_tensors = {}
    for layer in valid_layers:
        mc_dir = torch.from_numpy(mc_answer_directions[layer]).to(device=device, dtype=model_dtype)
        confdir_cpu = torch.from_numpy(confdir_directions[layer]).to(dtype=model_dtype)
        zero_dir = torch.zeros_like(mc_dir)
        layer_tensors[layer] = {
            "mc_dir": mc_dir,
            "confdir_cpu": confdir_cpu,
            "zero_dir": zero_dir,
            "baseline_projs": [],
            "ablated_projs": [],
        }

    total_passes = len(gpu_batches) * len(valid_layers)
    pbar = tqdm(total=total_passes, desc="Mediation test")
    for batch_indices, batch_inputs in gpu_batches:
        B = len(batch_indices)
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True,
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        expanded_inputs_template = {
            "input_ids": inputs_template["input_ids"].repeat_interleave(2, dim=0),
            "attention_mask": inputs_template["attention_mask"].repeat_interleave(2, dim=0),
            "use_cache": True,
        }
        if "position_ids" in inputs_template:
            expanded_inputs_template["position_ids"] = inputs_template["position_ids"].repeat_interleave(2, dim=0)

        for layer in valid_layers:
            capture_hook = ActivationCaptureHook(ablate_direction=None)
            tensors = layer_tensors[layer]
            dirs_pair = torch.stack([tensors["zero_dir"], tensors["mc_dir"]], dim=0)
            dirs_batch = dirs_pair.unsqueeze(0).expand(B, -1, -1).reshape(B * 2, -1)
            capture_hook.set_ablate_direction(dirs_batch)
            capture_hook.register(layer_modules[layer])

            try:
                expanded_inputs = expanded_inputs_template.copy()
                expanded_inputs["past_key_values"] = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=2)
                with torch.inference_mode():
                    _ = model(**expanded_inputs)
            finally:
                capture_hook.remove()

            if capture_hook.captured is None:
                raise RuntimeError(f"No activations captured at layer {layer}")

            captured = capture_hook.captured.to(dtype=model_dtype)
            baseline_acts = captured[0::2]
            ablated_acts = captured[1::2]
            confdir_cpu = tensors["confdir_cpu"]
            tensors["baseline_projs"].extend(torch.matmul(baseline_acts, confdir_cpu).tolist())
            tensors["ablated_projs"].extend(torch.matmul(ablated_acts, confdir_cpu).tolist())
            pbar.update(1)
    pbar.close()

    for layer in valid_layers:
        baseline_arr = np.array(layer_tensors[layer]["baseline_projs"], dtype=np.float32)
        ablated_arr = np.array(layer_tensors[layer]["ablated_projs"], dtype=np.float32)
        deltas = ablated_arr - baseline_arr

        corr, p_val = pearsonr(baseline_arr, ablated_arr)

        results["per_layer"][layer] = {
            "baseline_projections": layer_tensors[layer]["baseline_projs"],
            "ablated_projections": layer_tensors[layer]["ablated_projs"],
            "delta_mean": float(deltas.mean()),
            "delta_std": float(deltas.std()),
            "baseline_mean": float(baseline_arr.mean()),
            "ablated_mean": float(ablated_arr.mean()),
            "pearson_r": float(corr),
            "pearson_p": float(p_val),
        }

    return results


# =============================================================================
# METAMCUNCERT PROJECTION TEST
# =============================================================================
# CROSS-LAYER PROPAGATION TEST
# =============================================================================

class SingleDirectionAblationHook:
    """Hook that ablates a single direction from all examples.

    Unlike ActivationCaptureHook, this hook MODIFIES the output so the ablation
    propagates to downstream layers.
    """
    def __init__(self, direction: torch.Tensor):
        self.direction = direction
        self.handle = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        hs = hs.clone()
        d = self.direction.to(device=hs.device, dtype=hs.dtype)
        last_token = hs[:, -1, :]
        dots = torch.einsum('bh,h->b', last_token, d)
        proj = dots.unsqueeze(-1) * d.unsqueeze(0)
        hs[:, -1, :] = last_token - proj
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_cross_layer_projection_test(
    model,
    tokenizer,
    questions: List[Dict],
    ablate_directions: Dict[int, np.ndarray],
    projection_directions: Dict[str, Dict[int, np.ndarray]],  # {"metamcuncert": {layer: dir}, "confdir": {...}}
    meta_task: str,
    use_chat_template: bool,
    ablation_layers: List[int],
    original_indices: np.ndarray,
    capture_stride: int = 5,
    significance_threshold: float = 2.0,
    prompt_cache: Optional[Dict] = None,
) -> Dict:
    """
    Test how ablation at layer L affects projections at downstream layers.

    For each ablation_layer L:
    1. Determine capture_layers = [L, L+stride, L+2*stride, ..., final_layer]
    2. Register hooks at all capture_layers
    3. Run baseline forward pass → collect all activations
    4. Run ablated forward pass → collect all activations
    5. For each capture_layer and direction type: compute Δproj

    Args:
        ablate_directions: Direction being ablated at each layer
        projection_directions: Dict mapping direction name -> {layer: direction}
            e.g., {"metamcuncert": {...}, "confdir": {...}}
        capture_stride: Capture every Nth layer downstream
        significance_threshold: Flag as significant if |delta_mean| > threshold * delta_std

    Returns:
        {
            "ablation_layers": [...],
            "propagation": {
                ablation_layer: {
                    "capture_layers": [...],
                    "direction_name": {
                        capture_layer: {
                            "baseline_mean", "ablated_mean", "delta_mean", "delta_std",
                            "significant": bool
                        }
                    }
                }
            },
            "significant_pairs": [{"ablate": L, "capture": C, "direction": D, "delta": X}, ...]
        }
    """
    # Get layer modules
    if hasattr(model, 'get_base_model'):
        base_model = model.get_base_model()
        layer_modules = base_model.model.layers
    else:
        layer_modules = model.model.layers

    if prompt_cache is None:
        prompt_cache = build_meta_task_prompt_cache(
            tokenizer=tokenizer,
            questions=questions,
            meta_task=meta_task,
            use_chat_template=use_chat_template,
            original_indices=original_indices,
        )

    gpu_batches = prompt_cache["gpu_batches"]

    num_layers = len(layer_modules)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    results = {
        "ablation_layers": ablation_layers,
        "condition_type": "layer",
        "condition_axis_label": "Layer",
        "propagation": {},
        "significant_pairs": [],
    }

    # Convert all projection directions to tensors (CPU, model dtype)
    proj_dir_tensors = {}
    for dir_name, layer_dirs in projection_directions.items():
        proj_dir_tensors[dir_name] = {
            layer: torch.from_numpy(d).to(dtype=model_dtype)
            for layer, d in layer_dirs.items()
        }

    ablation_setup = {}
    for ablation_layer in ablation_layers:
        if ablation_layer not in ablate_directions:
            continue
        if capture_stride == 0:
            capture_layers = [ablation_layer]
        else:
            capture_layers = list(range(ablation_layer, num_layers, capture_stride))
        capture_layers = [
            cl for cl in capture_layers
            if any(cl in proj_dir_tensors[dn] for dn in proj_dir_tensors)
        ]
        if not capture_layers:
            continue

        ablate_dir = torch.from_numpy(direction_to_basis(ablate_directions[ablation_layer])).to(device=device, dtype=model_dtype)
        ablation_setup[ablation_layer] = {
            "capture_layers": capture_layers,
            "ablate_dir": ablate_dir,
            "zero_dir": torch.zeros_like(ablate_dir),
            "layer_projs": {
            cl: {dn: {"baseline": [], "ablated": []} for dn in proj_dir_tensors}
            for cl in capture_layers
        }
        }

    total_passes = len(gpu_batches) * len(ablation_setup)
    pbar = tqdm(total=total_passes, desc="Cross-layer propagation")

    for batch_indices, batch_inputs in gpu_batches:
        B = len(batch_indices)
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True,
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        expanded_inputs_template = {
            "input_ids": inputs_template["input_ids"].repeat_interleave(2, dim=0),
            "attention_mask": inputs_template["attention_mask"].repeat_interleave(2, dim=0),
            "use_cache": True,
        }
        if "position_ids" in inputs_template:
            expanded_inputs_template["position_ids"] = inputs_template["position_ids"].repeat_interleave(2, dim=0)

        for ablation_layer, setup in ablation_setup.items():
            capture_hooks = {
                cl: ActivationCaptureHook(ablate_direction=None)
                for cl in setup["capture_layers"]
            }
            dirs_pair = torch.stack([setup["zero_dir"], setup["ablate_dir"]], dim=0)
            dirs_batch = dirs_pair.unsqueeze(0).expand(B, -1, -1, -1).reshape(B * 2, dirs_pair.shape[1], dirs_pair.shape[2])

            ablation_hook = BatchAblationHook()
            ablation_hook.set_directions(dirs_batch)
            ablation_hook.register(layer_modules[ablation_layer])

            for cl, hook in capture_hooks.items():
                hook.register(layer_modules[cl])

            try:
                expanded_inputs = expanded_inputs_template.copy()
                expanded_inputs["past_key_values"] = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=2)
                with torch.inference_mode():
                    _ = model(**expanded_inputs)
            finally:
                ablation_hook.remove()

            for cl, hook in capture_hooks.items():
                hook.remove()
                if hook.captured is None:
                    raise RuntimeError(f"No activations captured at layer {cl} for ablation layer {ablation_layer}")

                captured = hook.captured.to(dtype=model_dtype)
                baseline_acts = captured[0::2]
                ablated_acts = captured[1::2]
                for dir_name, dir_tensors in proj_dir_tensors.items():
                    if cl not in dir_tensors:
                        continue
                    dir_tensor = dir_tensors[cl]
                    setup["layer_projs"][cl][dir_name]["baseline"].extend(torch.matmul(baseline_acts, dir_tensor).tolist())
                    setup["layer_projs"][cl][dir_name]["ablated"].extend(torch.matmul(ablated_acts, dir_tensor).tolist())
            pbar.update(1)
    pbar.close()

    # Compute statistics for each (capture_layer, direction) pair
    for ablation_layer, setup in ablation_setup.items():
        capture_layers = setup["capture_layers"]
        layer_projs = setup["layer_projs"]
        layer_results = {"capture_layers": capture_layers}
        for dir_name in proj_dir_tensors:
            layer_results[dir_name] = {}
            for cl in capture_layers:
                if cl not in proj_dir_tensors[dir_name]:
                    continue
                baseline = np.array(layer_projs[cl][dir_name]["baseline"])
                ablated = np.array(layer_projs[cl][dir_name]["ablated"])
                if len(baseline) == 0:
                    continue
                deltas = ablated - baseline
                delta_mean = float(deltas.mean())
                delta_std = float(deltas.std()) if len(deltas) > 1 else 0.0

                # Significance test
                significant = abs(delta_mean) > significance_threshold * delta_std if delta_std > 0 else False

                layer_results[dir_name][cl] = {
                    "baseline_mean": float(baseline.mean()),
                    "ablated_mean": float(ablated.mean()),
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "significant": significant,
                }

                if significant:
                    results["significant_pairs"].append({
                        "ablate": ablation_layer,
                        "capture": cl,
                        "direction": dir_name,
                        "delta": delta_mean,
                    })

        results["propagation"][ablation_layer] = layer_results

    return results


def run_cross_layer_projection_test_for_conditions(
    model,
    tokenizer,
    questions: List[Dict],
    ablation_conditions: List[Dict[str, Any]],
    projection_directions: Dict[str, Dict[int, np.ndarray]],
    meta_task: str,
    use_chat_template: bool,
    original_indices: np.ndarray,
    capture_stride: int = 5,
    significance_threshold: float = 2.0,
    prompt_cache: Optional[Dict] = None,
) -> Dict:
    """Run propagation analysis for explicit multi-layer ablation conditions."""
    if hasattr(model, "get_base_model"):
        base_model = model.get_base_model()
        layer_modules = base_model.model.layers
    else:
        layer_modules = model.model.layers

    if prompt_cache is None:
        prompt_cache = build_meta_task_prompt_cache(
            tokenizer=tokenizer,
            questions=questions,
            meta_task=meta_task,
            use_chat_template=use_chat_template,
            original_indices=original_indices,
        )

    gpu_batches = prompt_cache["gpu_batches"]
    num_layers = len(layer_modules)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    results = {
        "ablation_layers": [condition["key"] for condition in ablation_conditions],
        "condition_type": "joint_layer_pairs",
        "condition_axis_label": "Layer pair",
        "propagation": {},
        "significant_pairs": [],
    }

    proj_dir_tensors = {}
    for dir_name, layer_dirs in projection_directions.items():
        proj_dir_tensors[dir_name] = {
            layer: torch.from_numpy(d).to(dtype=model_dtype)
            for layer, d in layer_dirs.items()
        }

    condition_setup = {}
    for condition in ablation_conditions:
        capture_start = max(condition["ablation_layers"])
        if capture_stride == 0:
            capture_layers = [capture_start]
        else:
            capture_layers = list(range(capture_start, num_layers, capture_stride))
        capture_layers = [
            cl for cl in capture_layers
            if any(cl in proj_dir_tensors[dn] for dn in proj_dir_tensors)
        ]
        if not capture_layers:
            continue

        layer_bases = {
            layer: torch.from_numpy(direction_to_basis(basis)).to(device=device, dtype=model_dtype)
            for layer, basis in condition["layer_bases"].items()
        }
        zero_bases = {
            layer: torch.zeros_like(basis)
            for layer, basis in layer_bases.items()
        }
        condition_setup[condition["key"]] = {
            "condition": condition,
            "capture_layers": capture_layers,
            "layer_bases": layer_bases,
            "zero_bases": zero_bases,
            "layer_projs": {
                cl: {dn: {"baseline": [], "ablated": []} for dn in proj_dir_tensors}
                for cl in capture_layers
            },
        }

    total_passes = len(gpu_batches) * len(condition_setup)
    pbar = tqdm(total=total_passes, desc="Cross-layer propagation")

    for batch_indices, batch_inputs in gpu_batches:
        B = len(batch_indices)
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True,
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        expanded_inputs_template = {
            "input_ids": inputs_template["input_ids"].repeat_interleave(2, dim=0),
            "attention_mask": inputs_template["attention_mask"].repeat_interleave(2, dim=0),
            "use_cache": True,
        }
        if "position_ids" in inputs_template:
            expanded_inputs_template["position_ids"] = inputs_template["position_ids"].repeat_interleave(2, dim=0)

        for cond_key, setup in condition_setup.items():
            capture_hooks = {
                cl: ActivationCaptureHook(ablate_direction=None)
                for cl in setup["capture_layers"]
            }
            ablation_hooks = {}
            for layer in setup["condition"]["ablation_layers"]:
                hook = BatchAblationHook()
                dirs_pair = torch.stack([setup["zero_bases"][layer], setup["layer_bases"][layer]], dim=0)
                dirs_batch = (
                    dirs_pair.unsqueeze(0)
                    .expand(B, -1, -1, -1)
                    .reshape(B * 2, dirs_pair.shape[1], dirs_pair.shape[2])
                )
                hook.set_directions(dirs_batch)
                hook.register(layer_modules[layer])
                ablation_hooks[layer] = hook

            for cl, hook in capture_hooks.items():
                hook.register(layer_modules[cl])

            try:
                expanded_inputs = expanded_inputs_template.copy()
                expanded_inputs["past_key_values"] = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=2)
                with torch.inference_mode():
                    _ = model(**expanded_inputs)
            finally:
                for hook in ablation_hooks.values():
                    hook.remove()

            for cl, hook in capture_hooks.items():
                hook.remove()
                if hook.captured is None:
                    raise RuntimeError(f"No activations captured at layer {cl} for condition {cond_key}")

                captured = hook.captured.to(dtype=model_dtype)
                baseline_acts = captured[0::2]
                ablated_acts = captured[1::2]
                for dir_name, dir_tensors in proj_dir_tensors.items():
                    if cl not in dir_tensors:
                        continue
                    dir_tensor = dir_tensors[cl]
                    setup["layer_projs"][cl][dir_name]["baseline"].extend(torch.matmul(baseline_acts, dir_tensor).tolist())
                    setup["layer_projs"][cl][dir_name]["ablated"].extend(torch.matmul(ablated_acts, dir_tensor).tolist())
            pbar.update(1)
    pbar.close()

    for cond_key, setup in condition_setup.items():
        layer_results = {
            "capture_layers": setup["capture_layers"],
            "ablation_layers": setup["condition"]["ablation_layers"],
            "layers_by_component": setup["condition"]["layers_by_component"],
        }
        for dir_name in proj_dir_tensors:
            layer_results[dir_name] = {}
            for cl in setup["capture_layers"]:
                if cl not in proj_dir_tensors[dir_name]:
                    continue
                baseline = np.array(setup["layer_projs"][cl][dir_name]["baseline"])
                ablated = np.array(setup["layer_projs"][cl][dir_name]["ablated"])
                if len(baseline) == 0:
                    continue
                deltas = ablated - baseline
                delta_mean = float(deltas.mean())
                delta_std = float(deltas.std()) if len(deltas) > 1 else 0.0
                significant = abs(delta_mean) > significance_threshold * delta_std if delta_std > 0 else False

                layer_results[dir_name][cl] = {
                    "baseline_mean": float(baseline.mean()),
                    "ablated_mean": float(ablated.mean()),
                    "delta_mean": delta_mean,
                    "delta_std": delta_std,
                    "significant": significant,
                }

                if significant:
                    results["significant_pairs"].append({
                        "ablate": cond_key,
                        "capture": cl,
                        "direction": dir_name,
                        "delta": delta_mean,
                    })
        results["propagation"][cond_key] = layer_results

    return results


def plot_propagation_heatmap(
    results: Dict,
    direction_name: str,
    output_path: Path,
    title_suffix: str = "",
) -> bool:
    """
    Plot heatmap of ablation propagation effects.

    Only plots if there are significant effects for this direction type.
    Rows = ablation layers (only those with significant effects)
    Cols = capture layers (only those with significant effects)

    Returns True if a figure was generated, False otherwise.
    """
    # Filter significant pairs for this direction
    sig_pairs = [p for p in results["significant_pairs"] if p["direction"] == direction_name]
    if not sig_pairs:
        return False

    # Find unique ablation and capture layers with significant effects
    ablation_layers = sorted(set(p["ablate"] for p in sig_pairs))
    capture_layers = sorted(set(p["capture"] for p in sig_pairs))
    axis_label = _condition_axis_label_from_results(results)

    if not ablation_layers or not capture_layers:
        return False

    # Build delta matrix
    delta_matrix = np.full((len(ablation_layers), len(capture_layers)), np.nan)
    abl_idx = {l: i for i, l in enumerate(ablation_layers)}
    cap_idx = {l: i for i, l in enumerate(capture_layers)}

    for abl_layer in ablation_layers:
        if abl_layer not in results["propagation"]:
            continue
        layer_data = results["propagation"][abl_layer]
        if direction_name not in layer_data:
            continue
        for cap_layer, stats in layer_data[direction_name].items():
            if cap_layer in cap_idx:
                delta_matrix[abl_idx[abl_layer], cap_idx[cap_layer]] = stats["delta_mean"]

    # Plot
    fig, ax = plt.subplots(figsize=(max(8, len(capture_layers) * 0.5), max(6, len(ablation_layers) * 0.4)))

    # Use diverging colormap centered at 0
    vmax = np.nanmax(np.abs(delta_matrix))
    im = ax.imshow(delta_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")

    # Labels
    ax.set_xticks(range(len(capture_layers)))
    ax.set_xticklabels([f"L{l}" for l in capture_layers], rotation=45, ha="right")
    ax.set_yticks(range(len(ablation_layers)))
    ax.set_yticklabels([_format_condition_value(l, axis_label) for l in ablation_layers])
    ax.set_xlabel("Capture Layer (projection measured here)")
    ax.set_ylabel(f"{axis_label} (direction removed here)")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Δ projection (ablated - baseline)")

    # Title
    title = f"Propagation: Ablation → {direction_name} projection"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)

    # Annotate significant cells
    for i, abl in enumerate(ablation_layers):
        for j, cap in enumerate(capture_layers):
            val = delta_matrix[i, j]
            if not np.isnan(val):
                # Check if significant
                is_sig = any(
                    p["ablate"] == abl and p["capture"] == cap and p["direction"] == direction_name
                    for p in sig_pairs
                )
                if is_sig:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if abs(val) > vmax * 0.5 else "black")

    save_figure(fig, output_path)  # save_figure calls tight_layout internally
    return True


def plot_same_layer_projection_effects(
    results: Dict,
    direction_names: List[str],
    output_path: Path,
    title_suffix: str = "",
) -> bool:
    """
    Plot line chart of Δproj vs layer for same-layer captures.

    For each direction type, plots the delta projection (ablated - baseline)
    at the ablation layer itself (capture_layer == ablation_layer).

    Args:
        results: Output from run_cross_layer_projection_test
        direction_names: List of direction names to plot (e.g., ["metamcuncert", "confdir"])
        output_path: Where to save the figure
        title_suffix: Additional title info

    Returns:
        True if figure was generated, False if no data available.
    """
    axis_label = _condition_axis_label_from_results(results)
    if axis_label.lower() != "layer":
        return False

    # Collect same-layer data: {dir_name: {layer: delta_mean, ...}}
    same_layer_data = {dn: {} for dn in direction_names}
    same_layer_stds = {dn: {} for dn in direction_names}
    same_layer_sig = {dn: set() for dn in direction_names}

    for abl_layer, layer_data in results.get("propagation", {}).items():
        abl_layer = int(abl_layer)  # JSON keys are strings
        for dir_name in direction_names:
            if dir_name not in layer_data:
                continue
            # Same-layer = capture at ablation layer
            if abl_layer in layer_data[dir_name]:
                stats = layer_data[dir_name][abl_layer]
                same_layer_data[dir_name][abl_layer] = stats["delta_mean"]
                same_layer_stds[dir_name][abl_layer] = stats["delta_std"]
                if stats.get("significant", False):
                    same_layer_sig[dir_name].add(abl_layer)

    # Check if we have any data
    has_data = any(len(d) > 0 for d in same_layer_data.values())
    if not has_data:
        return False

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors for different directions
    dir_colors = {
        "mcuncert": "tab:green",
        "metamcuncert": "tab:blue",
        "metamcq": "tab:red",
        "metamcanswer": "tab:red",
        "confdir": "tab:orange",
    }

    for dir_name in direction_names:
        data = same_layer_data[dir_name]
        if not data:
            continue

        layers = sorted(data.keys())
        deltas = [data[l] for l in layers]
        stds = [same_layer_stds[dir_name].get(l, 0) for l in layers]

        color = dir_colors.get(dir_name, "tab:gray")

        # Plot line with error bands
        ax.plot(layers, deltas, '-o', color=color, label=dir_name, markersize=4, linewidth=1.5)
        ax.fill_between(
            layers,
            [d - s for d, s in zip(deltas, stds)],
            [d + s for d, s in zip(deltas, stds)],
            color=color,
            alpha=CI_ALPHA,
        )

        # Mark significant layers
        sig_layers = same_layer_sig[dir_name]
        for layer in sig_layers:
            if layer in data:
                ax.scatter([layer], [data[layer]], color=color, s=80, marker='*', zorder=5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel("Layer (ablation = capture)")
    ax.set_ylabel("Δ projection (ablated - baseline)")
    ax.legend(loc="best")
    ax.grid(True, alpha=GRID_ALPHA)

    title = "Same-Layer Projection Effects"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title)

    save_figure(fig, output_path)
    return True


def _extract_propagation_pairs(results: Dict, direction_name: str) -> List[Dict]:
    """Flatten propagation results for one direction into ranked pair rows."""
    rows = []
    axis_label = _condition_axis_label_from_results(results)
    for ablation_key, layer_data in results.get("propagation", {}).items():
        ablation_site = ablation_key
        ablation_layers = layer_data.get("ablation_layers", [])
        if not ablation_layers and axis_label.lower() == "layer":
            try:
                ablation_layers = [int(ablation_key)]
            except (TypeError, ValueError):
                ablation_layers = []
        dir_data = layer_data.get(direction_name, {})
        for capture_key, stats in dir_data.items():
            capture_layer = int(capture_key)
            delta_mean = float(stats.get("delta_mean", 0.0))
            delta_std = float(stats.get("delta_std", 0.0))
            strength = abs(delta_mean) / delta_std if delta_std > 1e-12 else float("inf")
            same_layer = (
                axis_label.lower() == "layer"
                and len(ablation_layers) == 1
                and ablation_layers[0] == capture_layer
            )
            rows.append({
                "ablate": ablation_site,
                "capture": capture_layer,
                "delta_mean": delta_mean,
                "delta_std": delta_std,
                "baseline_mean": float(stats.get("baseline_mean", 0.0)),
                "ablated_mean": float(stats.get("ablated_mean", 0.0)),
                "significant": bool(stats.get("significant", False)),
                "same_layer": same_layer,
                "strength": strength,
                "ablation_layers": ablation_layers,
            })
    rows.sort(key=lambda row: abs(row["delta_mean"]), reverse=True)
    return rows


def write_ranked_propagation_table(
    results: Dict,
    direction_name: str,
    output_path: Path,
    top_k: int = 20,
) -> bool:
    """Write a compact top-k propagation summary table."""
    all_rows = _extract_propagation_pairs(results, direction_name)
    if not all_rows:
        return False

    same_layer_rows = [row for row in all_rows if row["same_layer"]]
    offdiag_rows = [row for row in all_rows if not row["same_layer"]]

    def section_lines(title: str, rows: List[Dict]) -> List[str]:
        lines = [title]
        lines.append("rank  ablate                capture   delta    std   |d|/sd  sig")
        lines.append("----  --------------------  -------  -------  -----  ------  ---")
        if not rows:
            lines.append("(none)")
            return lines
        for rank, row in enumerate(rows[:top_k], 1):
            strength = "inf" if not np.isfinite(row["strength"]) else f"{row['strength']:.2f}"
            lines.append(
                f"{rank:>4}  "
                f"{str(row['ablate']):<20}  "
                f"{row['capture']:>7}  "
                f"{row['delta_mean']:>+7.3f}  "
                f"{row['delta_std']:>5.3f}  "
                f"{strength:>6}  "
                f"{'yes' if row['significant'] else 'no':>3}"
            )
        return lines

    summary_lines = [
        f"Propagation summary for {direction_name}",
        f"Total tested pairs: {len(all_rows)}",
        f"Same-layer pairs: {len(same_layer_rows)}",
        f"Off-diagonal pairs: {len(offdiag_rows)}",
        "",
    ]
    summary_lines.extend(section_lines("Top overall |delta| pairs", all_rows))
    summary_lines.append("")
    summary_lines.extend(section_lines("Top off-diagonal |delta| pairs", offdiag_rows))
    summary_lines.append("")
    summary_lines.extend(section_lines("Top same-layer |delta| pairs", same_layer_rows))

    with open(output_path, "w") as f:
        f.write("\n".join(summary_lines) + "\n")
    return True


def plot_ranked_propagation_pairs(
    results: Dict,
    direction_name: str,
    output_path: Path,
    top_k: int = 15,
    title_suffix: str = "",
) -> bool:
    """Plot readable ranked bar charts for the strongest propagation pairs."""
    all_rows = _extract_propagation_pairs(results, direction_name)
    if not all_rows:
        return False

    sections = [("Top overall |Δ projection|", all_rows[:top_k])]
    offdiag_rows = [row for row in all_rows if not row["same_layer"]]
    if offdiag_rows:
        sections.append(("Top off-diagonal |Δ projection|", offdiag_rows[:top_k]))

    fig_height = max(5, 2.8 * len(sections) + 0.35 * top_k)
    fig, axes = plt.subplots(len(sections), 1, figsize=(12, fig_height))
    if len(sections) == 1:
        axes = [axes]

    for ax, (section_title, rows) in zip(axes, sections):
        labels = [f"{row['ablate']} -> L{row['capture']}" for row in rows]
        values = np.array([row["delta_mean"] for row in rows], dtype=np.float32)
        y = np.arange(len(rows))
        colors = ["tab:red" if value < 0 else "tab:blue" for value in values]

        ax.barh(y, values, color=colors, alpha=0.85)
        ax.axvline(0, color="black", linewidth=1)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Δ projection (ablated - baseline)")
        ax.set_title(section_title)
        ax.grid(True, axis="x", alpha=GRID_ALPHA)

        x_span = float(np.max(np.abs(values))) if len(values) else 0.0
        text_pad = max(0.03 * x_span, 0.02)
        for idx, (value, row) in enumerate(zip(values, rows)):
            text_x = value + text_pad if value >= 0 else value - text_pad
            ha = "left" if value >= 0 else "right"
            suffix = " *" if row["significant"] else ""
            ax.text(text_x, idx, f"{value:+.2f}{suffix}", va="center", ha=ha, fontsize=9)

    title = f"Ranked Propagation Effects: {direction_name}"
    if title_suffix:
        title += f"\n{title_suffix}"
    fig.suptitle(title, fontsize=12, fontweight="bold")
    save_figure(fig, output_path)
    return True


# =============================================================================
# MAIN
# =============================================================================

def main():
    # Model directory for organizing outputs
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    base_name = DATASET  # Evaluation dataset (model prefix now in directory)
    direction_base = DIRECTION_DATASET if DIRECTION_DATASET else DATASET  # Source of directions
    ablation_label = get_ablation_title_label()
    dir_suffix = get_direction_suffix()

    # Cross-dataset suffix for output files (only when datasets differ)
    if DIRECTION_DATASET and DIRECTION_DATASET != DATASET:
        cross_suffix = f"_from_{DIRECTION_DATASET}"
    else:
        cross_suffix = ""

    # Setup output naming
    model_short = get_model_short_name(MODEL)

    config = {
        "model": MODEL.split("/")[-1],
        "dataset": DATASET,
        "direction_dataset": DIRECTION_DATASET or DATASET,
        "task": META_TASK,
        "direction_type": DIRECTION_TYPE,
        "ablation_label": ablation_label,
    }
    if is_joint_ablation_enabled():
        config["ablation_components"] = [spec["label"] for spec in get_ablation_component_specs()]
    print_run_header("run_ablation_causality.py", 3, "Ablation necessity test", config)

    # Key findings for console output
    key_findings = {}
    output_files = []

    # Load directions based on direction type
    joint_method_metadata: Dict[str, Dict[str, Any]] = {}
    resolved_joint_methods: Dict[str, Dict[str, Any]] = {}
    if is_joint_ablation_enabled():
        print(f"Loading joint ablation components from {direction_base}...")
        resolved_joint_methods = resolve_joint_component_sets(
            get_ablation_component_specs(),
            direction_base=direction_base,
            model_dir=model_dir,
            meta_task=META_TASK,
            requested_methods=METHODS,
        )
        all_directions, joint_method_metadata = build_joint_direction_set(
            get_ablation_component_specs(),
            direction_base=direction_base,
            model_dir=model_dir,
            meta_task=META_TASK,
            requested_methods=METHODS,
        )
        methods = list(resolved_joint_methods.keys())
        for method in methods:
            metadata = joint_method_metadata[method]
            component_desc = ", ".join(
                f"{label}:{actual_method}"
                for label, actual_method in metadata["component_methods"].items()
            )
            print(f"  Joint method {method}: {component_desc}")
        if has_explicit_joint_layer_pairs():
            print("  Explicit joint layer pairs:")
            for pair in JOINT_ABLATION_LAYER_PAIRS:
                print(f"    {pair}")
    else:
        print(f"Loading directions from {direction_base}...")
        all_directions = load_directions(
            direction_base,
            direction_type=DIRECTION_TYPE,
            metric=METRIC,
            meta_task=META_TASK,
            model_dir=model_dir,
            confdir_target=CONFDIR_TARGET if DIRECTION_TYPE == "confidence" else None,
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
    transfer_data_by_position = {}
    transfer_positions = set(PROBE_POSITIONS)
    transfer_positions.add("final")  # Always keep final available as fallback
    for position in sorted(transfer_positions):
        transfer_data_by_position[position] = load_transfer_results(
            base_name,
            META_TASK,
            model_dir,
            position=position,
        )

    # Load answer transfer results for layer selection (answer directions)
    answer_transfer_data = None
    answer_selected_layers = None
    if DIRECTION_TYPE == "answer" and ANSWER_LAYER_SELECTION:
        answer_transfer_data = load_answer_transfer_results(base_name, META_TASK, model_dir)
        if answer_transfer_data is not None:
            answer_selected_layers = get_layers_from_answer_transfer(
                answer_transfer_data, ANSWER_D2D_THRESHOLD
            )
            if answer_selected_layers:
                print(f"Answer layer selection: {len(answer_selected_layers)} layers with D2D >= {ANSWER_D2D_THRESHOLD}")
                print(f"  Layers: {answer_selected_layers[0]}-{answer_selected_layers[-1]}")
            else:
                print(f"Warning: No layers meet D2D >= {ANSWER_D2D_THRESHOLD}, using all layers")
                answer_selected_layers = None  # Reset so fallback to all_available_layers kicks in
        else:
            print("Warning: Answer transfer results not found, using all layers")

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

    print("Preparing prompts and tokenization cache...")
    prompt_cache = build_meta_task_prompt_cache(
        tokenizer=tokenizer,
        questions=questions,
        meta_task=META_TASK,
        use_chat_template=use_chat_template,
        original_indices=original_indices,
    )

    # Run ablation for each method and position
    # Structure: {position: {method: analysis}}
    all_results_by_pos = {pos: {} for pos in PROBE_POSITIONS}
    all_analyses_by_pos = {pos: {} for pos in PROBE_POSITIONS}

    for position in PROBE_POSITIONS:
        # Determine number of controls for this position
        position_num_controls = NUM_CONTROLS if position == "final" else NUM_CONTROLS_NONFINAL

        for method in methods:
            print(f"Running ablation: {method} @ {position}...")

            if has_explicit_joint_layer_pairs():
                explicit_conditions = build_explicit_joint_ablation_conditions(
                    resolved_joint_methods[method],
                    JOINT_ABLATION_LAYER_PAIRS,
                )
                results = run_joint_layer_pair_ablation_for_method(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    metric_values=metric_values,
                    conditions=explicit_conditions,
                    num_controls=position_num_controls,
                    meta_task=META_TASK,
                    use_chat_template=use_chat_template,
                    position=position,
                    original_indices=original_indices,
                    prompt_cache=prompt_cache,
                )
            else:
                # Determine layers for this position AND method
                if LAYERS is not None:
                    # Explicit override applies to all positions/methods
                    method_layers = LAYERS
                elif answer_selected_layers is not None:
                    # Answer direction: use layers with significant D2D accuracy
                    method_layers = answer_selected_layers
                elif position == "final":
                    # Final position: use all layers
                    method_layers = all_available_layers
                else:
                    # Non-final position: select based on transfer R² for THIS method
                    transfer_data = transfer_data_by_position.get(position)
                    final_transfer_data = transfer_data_by_position.get("final")
                    if transfer_data is not None:
                        method_layers = get_layers_from_transfer(
                            transfer_data, METRIC, position, TRANSFER_R2_THRESHOLD, method
                        )
                    elif final_transfer_data is not None:
                        method_layers = []
                    else:
                        method_layers = all_available_layers

                    if not method_layers and final_transfer_data is not None:
                        # Fall back to the final-position transfer file if the requested
                        # non-final transfer results are missing or have no qualifying layers.
                        method_layers = get_layers_from_transfer(
                            final_transfer_data, METRIC, "final", TRANSFER_R2_THRESHOLD, method
                        )

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
                    prompt_cache=prompt_cache,
                )
            all_results_by_pos[position][method] = results

            # Analyze results
            analysis = analyze_ablation_results(results, METRIC, base_name)
            all_analyses_by_pos[position][method] = analysis

        # Incremental save after each position completes (crash protection) - one per method
        for method in methods:
            checkpoint_base = f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{method}{cross_suffix}"
            checkpoint_path = get_output_path(f"{checkpoint_base}_checkpoint.json", model_dir=model_dir, working=True)

            checkpoint_json = {
                "config": get_config_dict(
                    model=MODEL,
                    dataset=base_name,
                    model_dir=model_dir,
                    direction_type=DIRECTION_TYPE,
                    ablation_label=ablation_label,
                    ablation_components=get_ablation_component_specs(),
                    ablation_component_methods=joint_method_metadata.get(method),
                    joint_layer_pairs=JOINT_ABLATION_LAYER_PAIRS if has_explicit_joint_layer_pairs() else None,
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
                        "condition_type": analysis.get("condition_type"),
                        "condition_axis_label": analysis.get("condition_axis_label"),
                        "condition_metadata": analysis.get("condition_metadata"),
                    }
            with open(checkpoint_path, "w") as f:
                json.dump(checkpoint_json, f, indent=2)

    def get_base_output(method: str) -> str:
        """Get base output path for a specific method (without position - added per file)."""
        base = f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{method}{cross_suffix}"
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
                ablation_label=ablation_label,
                ablation_components=get_ablation_component_specs(),
                ablation_component_methods=joint_method_metadata.get(method),
                joint_layer_pairs=JOINT_ABLATION_LAYER_PAIRS if has_explicit_joint_layer_pairs() else None,
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
                answer_layer_selection=ANSWER_LAYER_SELECTION if DIRECTION_TYPE == "answer" else None,
                answer_d2d_threshold=ANSWER_D2D_THRESHOLD if DIRECTION_TYPE == "answer" and ANSWER_LAYER_SELECTION else None,
                answer_layers_selected=answer_selected_layers,
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
                "condition_type": analysis.get("condition_type"),
                "condition_axis_label": analysis.get("condition_axis_label"),
                "condition_metadata": analysis.get("condition_metadata"),
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
        output_json["condition_type"] = analysis.get("condition_type")
        output_json["condition_axis_label"] = analysis.get("condition_axis_label")
        output_json["condition_metadata"] = analysis.get("condition_metadata")
        output_json["per_layer"] = analysis["per_layer"]
        output_json["summary"] = analysis["summary"]

        with open(results_path, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"  Saved {results_path.name}")
        output_files.append(results_path)

    # --- Mediation Test (Test C) ---
    # Only run when MEASURE_MEDIATION=True and DIRECTION_TYPE="answer"
    if MEASURE_MEDIATION and DIRECTION_TYPE == "answer":
        print("\n" + "="*60)
        print("MEDIATION TEST: MC_Answer → d_delegate")
        print("="*60)

        # Load confdir (d_delegate) directions
        try:
            confdir_directions = load_directions(
                direction_base,
                direction_type="confidence",
                metric=METRIC,
                meta_task=META_TASK,
                model_dir=model_dir,
                confdir_target=CONFDIR_TARGET,
            )
            # Use probe method from confdir (or mean_diff if available)
            confdir_method = "probe" if "probe" in confdir_directions else list(confdir_directions.keys())[0]
            confdir_dirs = confdir_directions[confdir_method]

            # Use mean_diff from mc_answer directions (or first available method)
            answer_method = "centroid" if "centroid" in all_directions else list(all_directions.keys())[0]
            answer_dirs = all_directions[answer_method]

            # Run mediation test on layers where we have both directions
            mediation_layers = sorted(set(answer_dirs.keys()) & set(confdir_dirs.keys()))
            if LAYERS is not None:
                mediation_layers = [l for l in mediation_layers if l in LAYERS]

            print(f"  Testing {len(mediation_layers)} layers with {answer_method} (answer) → {confdir_method} (confdir)")

            mediation_results = run_mediation_test(
                model=model,
                tokenizer=tokenizer,
                questions=questions,
                mc_answer_directions=answer_dirs,
                confdir_directions=confdir_dirs,
                meta_task=META_TASK,
                use_chat_template=use_chat_template,
                layers=mediation_layers,
                original_indices=original_indices,
                prompt_cache=prompt_cache,
            )

            # Save mediation results
            mediation_path = get_output_path(
                f"{base_name}_mediation_answer_to_confdir_{CONFDIR_TARGET}_results.json",
                model_dir=model_dir
            )
            mediation_output = {
                "config": get_config_dict(
                    model=MODEL,
                    dataset=base_name,
                    model_dir=model_dir,
                    answer_method=answer_method,
                    confdir_method=confdir_method,
                    confdir_target=CONFDIR_TARGET,
                    meta_task=META_TASK,
                    num_questions=len(questions),
                    seed=SEED,
                    load_in_4bit=LOAD_IN_4BIT,
                    load_in_8bit=LOAD_IN_8BIT,
                ),
                "layers": mediation_results["layers"],
                "per_layer": mediation_results["per_layer"],
            }
            with open(mediation_path, "w") as f:
                json.dump(mediation_output, f, indent=2)
            print(f"  Saved {mediation_path.name}")
            output_files.append(mediation_path)

            # Print summary
            for layer in mediation_results["layers"]:
                if layer in mediation_results["per_layer"]:
                    lr = mediation_results["per_layer"][layer]
                    print(f"    Layer {layer}: Δproj = {lr['delta_mean']:.4f} ± {lr['delta_std']:.4f}, "
                          f"r(baseline,ablated) = {lr['pearson_r']:.3f}")

        except FileNotFoundError as e:
            print(f"  Skipping mediation test: {e}")
            print("  (Run test_meta_transfer.py with FIND_CONFIDENCE_DIRECTIONS=True first)")

    # --- Cross-Layer Propagation Test ---
    if MEASURE_CROSS_LAYER_PROPAGATION:
        print("\n" + "="*60)
        print("CROSS-LAYER PROPAGATION TEST")
        print("="*60)

        # Use the first configured ablation method as the preferred projection method
        # so propagation plots compare like with like by default.
        ablate_method = methods[0] if methods else list(all_directions.keys())[0]

        def _select_projection_method(direction_sets: Dict[str, Dict[int, np.ndarray]]) -> Tuple[str, Dict[int, np.ndarray]]:
            if ablate_method in direction_sets:
                return ablate_method, direction_sets[ablate_method]
            if "mean_diff" in direction_sets:
                return "mean_diff", direction_sets["mean_diff"]
            first_method = next(iter(direction_sets))
            return first_method, direction_sets[first_method]

        canonical_direction_type = canonicalize_direction_type(DIRECTION_TYPE)
        if is_joint_ablation_enabled() or canonical_direction_type == "metamcq":
            projection_specs = [
                ("metamcuncert", "metamcuncert", None),
                ("confdir", "confidence", CONFDIR_TARGET),
            ]
        else:
            projection_specs = [
                ("mcuncert", "uncertainty", None),
                ("metamcuncert", "metamcuncert", None),
                ("confdir", "confidence", CONFDIR_TARGET),
            ]

        projection_directions = {}
        for projection_name, projection_type, projection_confdir_target in projection_specs:
            try:
                direction_sets = load_directions(
                    direction_base,
                    direction_type=projection_type,
                    metric=METRIC,
                    meta_task=META_TASK,
                    model_dir=model_dir,
                    confdir_target=projection_confdir_target,
                )
                selected_method, selected_dirs = _select_projection_method(direction_sets)
                projection_directions[projection_name] = selected_dirs
                print(
                    f"  Loaded {projection_name} directions ({selected_method}): "
                    f"{len(projection_directions[projection_name])} layers"
                )
            except FileNotFoundError:
                print(f"  Warning: {projection_name} directions not found, skipping")

        if projection_directions:
            print(f"  Projection directions: {list(projection_directions.keys())}")

            if has_explicit_joint_layer_pairs():
                ablation_conditions = build_explicit_joint_ablation_conditions(
                    resolved_joint_methods[ablate_method],
                    JOINT_ABLATION_LAYER_PAIRS,
                )
                print(f"  Testing {len(ablation_conditions)} explicit ablation conditions, stride={PROPAGATION_CAPTURE_STRIDE}")
                cross_results = run_cross_layer_projection_test_for_conditions(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    ablation_conditions=ablation_conditions,
                    projection_directions=projection_directions,
                    meta_task=META_TASK,
                    use_chat_template=use_chat_template,
                    original_indices=original_indices,
                    capture_stride=PROPAGATION_CAPTURE_STRIDE,
                    significance_threshold=PROPAGATION_SIGNIFICANCE_THRESHOLD,
                    prompt_cache=prompt_cache,
                )
            else:
                ablate_dirs = all_directions[ablate_method]

                # Determine ablation layers
                ablation_layers = sorted(ablate_dirs.keys())
                if LAYERS is not None:
                    ablation_layers = [l for l in ablation_layers if l in LAYERS]

                print(f"  Testing {len(ablation_layers)} ablation layers, stride={PROPAGATION_CAPTURE_STRIDE}")

                cross_results = run_cross_layer_projection_test(
                    model=model,
                    tokenizer=tokenizer,
                    questions=questions,
                    ablate_directions=ablate_dirs,
                    projection_directions=projection_directions,
                    meta_task=META_TASK,
                    use_chat_template=use_chat_template,
                    ablation_layers=ablation_layers,
                    original_indices=original_indices,
                    capture_stride=PROPAGATION_CAPTURE_STRIDE,
                    significance_threshold=PROPAGATION_SIGNIFICANCE_THRESHOLD,
                    prompt_cache=prompt_cache,
                )

            # Save results
            cross_path = get_output_path(
                f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{ablate_method}_cross_layer_propagation.json",
                model_dir=model_dir
            )
            cross_output = {
                "config": get_config_dict(
                    model=MODEL,
                    dataset=base_name,
                    model_dir=model_dir,
                    direction_type=DIRECTION_TYPE,
                    ablation_label=ablation_label,
                    ablation_components=get_ablation_component_specs(),
                    ablation_component_methods=joint_method_metadata.get(ablate_method),
                    joint_layer_pairs=JOINT_ABLATION_LAYER_PAIRS if has_explicit_joint_layer_pairs() else None,
                    ablate_method=ablate_method,
                    projection_directions=list(projection_directions.keys()),
                    meta_task=META_TASK,
                    num_questions=len(questions),
                    capture_stride=PROPAGATION_CAPTURE_STRIDE,
                    significance_threshold=PROPAGATION_SIGNIFICANCE_THRESHOLD,
                    seed=SEED,
                    load_in_4bit=LOAD_IN_4BIT,
                    load_in_8bit=LOAD_IN_8BIT,
                ),
                "ablation_layers": cross_results["ablation_layers"],
                "condition_type": cross_results.get("condition_type"),
                "condition_axis_label": cross_results.get("condition_axis_label"),
                "propagation": cross_results["propagation"],
                "significant_pairs": cross_results["significant_pairs"],
            }
            with open(cross_path, "w") as f:
                json.dump(cross_output, f, indent=2)
            print(f"  Saved {cross_path.name}")
            output_files.append(cross_path)

            # Print summary
            n_sig = len(cross_results["significant_pairs"])
            print(f"\n  Found {n_sig} significant (ablation_layer, capture_layer, direction) pairs")
            if n_sig > 0:
                # Group by direction
                by_dir = {}
                for p in cross_results["significant_pairs"]:
                    by_dir.setdefault(p["direction"], []).append(p)
                for dir_name, pairs in by_dir.items():
                    print(f"    {dir_name}: {len(pairs)} significant pairs")

            # Always generate same-layer line plot (Δproj vs layer)
            line_path = get_output_path(
                f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{ablate_method}_same_layer_projection.png",
                model_dir=model_dir
            )
            if plot_same_layer_projection_effects(
                cross_results,
                list(projection_directions.keys()),
                line_path,
                title_suffix=f"Ablating {ablation_label}/{ablate_method} | {base_name}",
            ):
                print(f"  Generated same-layer line plot: {line_path.name}")
                output_files.append(line_path)

            # Generate heatmaps only when stride > 0 (cross-layer mode)
            for dir_name in projection_directions.keys():
                if PROPAGATION_CAPTURE_STRIDE > 0:
                    heatmap_path = get_output_path(
                        f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{ablate_method}_propagation_{dir_name}.png",
                        model_dir=model_dir
                    )
                    if plot_propagation_heatmap(
                        cross_results,
                        dir_name,
                        heatmap_path,
                        title_suffix=f"Ablating {ablation_label}/{ablate_method} | {base_name}",
                    ):
                        print(f"  Generated heatmap: {heatmap_path.name}")
                        output_files.append(heatmap_path)
                    else:
                        print(f"  No significant effects for {dir_name}, skipping heatmap")

                ranked_table_path = get_output_path(
                    f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{ablate_method}_propagation_{dir_name}_ranked.txt",
                    model_dir=model_dir,
                )
                if write_ranked_propagation_table(cross_results, dir_name, ranked_table_path):
                    print(f"  Saved ranked summary: {ranked_table_path.name}")
                    output_files.append(ranked_table_path)

                ranked_plot_path = get_output_path(
                    f"{base_name}_ablation_{META_TASK}_{dir_suffix}_{ablate_method}_propagation_{dir_name}_ranked.png",
                    model_dir=model_dir,
                )
                if plot_ranked_propagation_pairs(
                    cross_results,
                    dir_name,
                    ranked_plot_path,
                    title_suffix=f"Ablating {ablation_label}/{ablate_method} | {base_name}",
                ):
                    print(f"  Generated ranked propagation plot: {ranked_plot_path.name}")
                    output_files.append(ranked_plot_path)

        else:
            print("  No projection directions available, skipping cross-layer test")

    # Generate plots - one per method per position
    print("\nGenerating plots...")
    for method in methods:
        method_base_output = get_base_output(method)
        for position in PROBE_POSITIONS:
            # Correlation impact plot
            plot_path = get_output_path(f"{method_base_output}_{position}.png", model_dir=model_dir)
            plot_ablation_results(all_analyses_by_pos[position][method], method, plot_path)
            output_files.append(plot_path)

            # Confidence impact plot
            conf_plot_path = get_output_path(f"{method_base_output}_{position}_confidence.png", model_dir=model_dir)
            plot_confidence_impact(all_analyses_by_pos[position][method], method, position, conf_plot_path)
            output_files.append(conf_plot_path)

    # Collect key findings
    for position in PROBE_POSITIONS:
        for method in methods:
            if method in all_analyses_by_pos[position]:
                analysis = all_analyses_by_pos[position][method]
                summary = analysis.get("summary", {})
                n_sig = _primary_sig_count(summary)
                total = len(analysis.get("layers", []))
                best_layer = summary.get("best_layer_abs_delta")
                if n_sig > 0:
                    key_findings[f"{position}/{method}"] = f"{n_sig}/{total} sig, best L{best_layer}"

    # Console output
    print_key_findings(key_findings)
    print_run_footer(output_files)


if __name__ == "__main__":
    main()
