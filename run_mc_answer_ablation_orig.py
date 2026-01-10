"""
MC Answer Probe Causality Experiment.

Tests whether the MC answer probe direction is causally involved in introspection by:
1. Ablating the MC answer direction during meta task execution
2. Measuring impact on D2M transfer R² (entropy probe transfer)
3. Measuring impact on behavioral correlation (stated confidence vs entropy)
4. Computing direction similarity between MC answer and entropy probes

Uses the same ablation infrastructure and statistical approach as run_introspection_steering.py.

Usage:
    python run_mc_answer_ablation.py                     # Default: entropy metric
    python run_mc_answer_ablation.py --metric logit_gap  # Use logit_gap metric
"""

import torch
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import random
from scipy import stats
import argparse
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split

from core.model_utils import (
    load_model_and_tokenizer,
    should_use_chat_template,
    get_model_short_name,
    DEVICE,
)
from tasks import (
    STATED_CONFIDENCE_OPTIONS,
    STATED_CONFIDENCE_MIDPOINTS,
    format_stated_confidence_prompt,
    get_stated_confidence_signal,
    ANSWER_OR_DELEGATE_OPTIONS,
    format_answer_or_delegate_prompt,
    get_delegate_mapping,
    response_to_confidence,
)

# Import DynamicCache safely for KV cache optimization
try:
    from transformers.cache_utils import DynamicCache
except ImportError:
    DynamicCache = None

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME
DATASET_NAME = "SimpleMC"

# Metric for entropy probe (which we measure D2M transfer for)
AVAILABLE_METRICS = ["entropy", "top_prob", "margin", "logit_gap", "top_logit"]
METRIC = "entropy"

# Meta-judgment task (should match run_introspection_experiment.py)
META_TASK = "confidence"  # "confidence" or "delegate"

# Thresholds for layer selection (same as run_introspection_steering.py)
D2M_R2_THRESHOLD = 0.20
D2D_R2_THRESHOLD = D2M_R2_THRESHOLD * 1.5

# Output directory
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)

# Ablation config (same as run_introspection_steering.py)
ABLATION_LAYERS = None  # None = auto-select from probe results
NUM_QUESTIONS = 500  # Number of questions for ablation experiment
NUM_CONTROL_DIRECTIONS = 100  # More controls for tighter p-values
FDR_ALPHA = 0.05
FDR_SAFETY_FACTOR = 25
MIN_CONTROLS_PER_LAYER = 10

BATCH_SIZE = 8
INTERVENTION_POSITION = "last"

# Quantization
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False

SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Cached token IDs
_CACHED_TOKEN_IDS = {
    "meta_options": None,
    "delegate_options": None,
}


def get_output_prefix() -> str:
    """Generate output filename prefix."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    task_suffix = "_delegate" if META_TASK == "delegate" else ""
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection{task_suffix}")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection{task_suffix}")


def get_directions_prefix() -> str:
    """Generate output filename prefix for direction files (task-independent)."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return str(OUTPUTS_DIR / f"{model_short}_adapter-{adapter_short}_{DATASET_NAME}_introspection")
    return str(OUTPUTS_DIR / f"{model_short}_{DATASET_NAME}_introspection")


def initialize_token_cache(tokenizer):
    """Precompute option token IDs once."""
    _CACHED_TOKEN_IDS["meta_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in STATED_CONFIDENCE_OPTIONS.keys()
    ]
    _CACHED_TOKEN_IDS["delegate_options"] = [
        tokenizer.encode(opt, add_special_tokens=False)[0]
        for opt in ANSWER_OR_DELEGATE_OPTIONS
    ]


# =============================================================================
# LOADING FUNCTIONS
# =============================================================================

def load_mc_answer_directions(prefix: str) -> Dict[int, np.ndarray]:
    """Load MC answer probe directions."""
    path = Path(f"{prefix}_mc_answer_directions.npz")
    if not path.exists():
        raise FileNotFoundError(
            f"MC answer directions not found: {path}\n"
            "Run run_introspection_experiment.py first to generate them."
        )
    print(f"Loading MC answer directions from {path}...")
    data = np.load(path)
    directions = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }
    print(f"  Loaded {len(directions)} layer directions")
    return directions


def load_entropy_directions(prefix: str, metric: str) -> Dict[int, np.ndarray]:
    """Load entropy/metric probe directions."""
    path = Path(f"{prefix}_{metric}_directions.npz")
    if not path.exists():
        raise FileNotFoundError(f"Entropy directions not found: {path}")
    print(f"Loading {metric} directions from {path}...")
    data = np.load(path)
    directions = {
        int(k.split("_")[1]): data[k]
        for k in data.files if k.startswith("layer_")
    }
    print(f"  Loaded {len(directions)} layer directions")
    return directions


def load_probe_results(prefix: str, metric: str) -> Dict:
    """Load probe results for layer selection."""
    path = Path(f"{prefix}_{metric}_results.json")
    if not path.exists():
        raise FileNotFoundError(f"Probe results not found: {path}")
    print(f"Loading probe results from {path}...")
    with open(path) as f:
        return json.load(f)


def load_paired_data(prefix: str) -> Dict:
    """Load paired data including questions and direct metrics."""
    # Try base prefix first (without metric suffix)
    base_prefix = prefix.rsplit("_", 1)[0] if "_" in prefix else prefix
    path = Path(f"{base_prefix}_paired_data.json")

    if not path.exists():
        # Try with the full prefix
        path = Path(f"{prefix}_paired_data.json")

    if not path.exists():
        raise FileNotFoundError(
            f"Paired data not found at {path}\n"
            "Run run_introspection_experiment.py first."
        )

    print(f"Loading paired data from {path}...")
    with open(path) as f:
        data = json.load(f)

    # Convert direct_metrics back to numpy arrays
    direct_metrics = {k: np.array(v) for k, v in data["direct_metrics"].items()}

    return {
        "questions": data["questions"],
        "direct_metrics": direct_metrics,
        "direct_probs": data["direct_probs"],
        "meta_probs": data["meta_probs"],
        "meta_responses": data["meta_responses"],
        "meta_mappings": data.get("meta_mappings"),
        "config": data.get("config", {}),
    }


def load_activations(prefix: str, layers: List[int]) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Load saved direct activations for specified layers.

    Returns:
        direct_activations: {layer_idx: activations array}
        entropy_values: Ground truth entropy from direct task
    """
    direct_path = Path(f"{prefix}_direct_activations.npz")

    if not direct_path.exists():
        raise FileNotFoundError(
            f"Activation file not found: {direct_path}\n"
            "Run run_introspection_experiment.py first to generate it."
        )

    print(f"Loading activations from {direct_path}...")
    direct_data = np.load(direct_path)

    direct_activations = {}

    for layer_idx in layers:
        key = f"layer_{layer_idx}"
        if key in direct_data.files:
            direct_activations[layer_idx] = direct_data[key]

    # Get entropy values (saved in direct_activations file)
    if "entropy" in direct_data.files:
        entropy_values = direct_data["entropy"]
    else:
        raise KeyError("Entropy values not found in direct_activations.npz")

    print(f"  Loaded {len(direct_activations)} layers, {len(direct_activations[layers[0]])} samples")

    return direct_activations, entropy_values


def train_entropy_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_components: int = 128
) -> Tuple[StandardScaler, PCA, Ridge]:
    """
    Train entropy probe on direct activations.

    Returns:
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        probe: Fitted Ridge regressor
    """
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    # PCA
    pca = PCA(n_components=min(n_components, X_scaled.shape[1], X_scaled.shape[0]))
    X_pca = pca.fit_transform(X_scaled)

    # Ridge regression
    probe = Ridge(alpha=1.0)
    probe.fit(X_pca, y_train)

    return scaler, pca, probe


def apply_probe_strict(
    X: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    probe: Ridge
) -> np.ndarray:
    """
    Apply a pre-trained probe using the ORIGINAL scaler (no domain adaptation).
    This is the strictest/fairest comparison for ablation experiments.
    """
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return probe.predict(X_pca)


# =============================================================================
# ABLATION INFRASTRUCTURE (adapted from run_introspection_steering.py)
# =============================================================================

class BatchAblationHook:
    """Hook that projects out a per-example direction from activations."""

    def __init__(self, directions_bh: Optional[torch.Tensor] = None):
        self.directions_bh = directions_bh
        self.handle = None

    def set_directions(self, directions_bh: torch.Tensor):
        self.directions_bh = directions_bh

    def __call__(self, module, input, output):
        if self.directions_bh is None:
            return output

        hs = output[0] if isinstance(output, tuple) else output
        # directions_bh should match the batch dimension of hs
        dirs = self.directions_bh.to(device=hs.device, dtype=hs.dtype)

        if INTERVENTION_POSITION == "last":
            hs = hs.clone()
            last_token = hs[:, -1, :]
            dots = torch.einsum('bh,bh->b', last_token, dirs)
            proj = dots.unsqueeze(-1) * dirs
            hs[:, -1, :] = last_token - proj
        else:
            dots = torch.einsum('bsh,bh->bs', hs, dirs)
            proj = dots.unsqueeze(-1) * dirs.unsqueeze(1)
            hs = hs - proj

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class ActivationCaptureHook:
    """Hook that captures activations at the last token position."""

    def __init__(self):
        self.activations = None
        self.handle = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Capture last token activations
        self.activations = hs[:, -1, :].detach().cpu().numpy()
        return output

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# =============================================================================
# KV CACHE HELPERS (from run_introspection_steering.py)
# =============================================================================

def extract_cache_tensors(past_key_values):
    """
    Extract raw tensors from past_key_values (tuple or DynamicCache).
    Returns (key_tensors, value_tensors) where each is a list of tensors.
    """
    keys = []
    values = []

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
    """
    Create a fresh DynamicCache (or tuple) from tensors.
    """
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


def get_kv_cache(model, batch_inputs):
    """
    Run the prefix to generate KV cache tensors.
    Returns dictionary with next-step inputs and 'past_key_values_data' (snapshot).
    """
    input_ids = batch_inputs["input_ids"]
    attention_mask = batch_inputs["attention_mask"]

    # Run Prefix (Tokens 0 to T-1)
    prefix_ids = input_ids[:, :-1]
    prefix_mask = attention_mask[:, :-1]

    with torch.inference_mode():
        outputs = model(
            input_ids=prefix_ids,
            attention_mask=prefix_mask,
            use_cache=True,
        )

    # Extract Immutable Snapshot
    keys, values = extract_cache_tensors(outputs.past_key_values)

    # Prepare next step inputs
    last_ids = input_ids[:, -1:]

    result = {
        "input_ids": last_ids,
        "attention_mask": attention_mask,  # Full mask
        "past_key_values_data": (keys, values),
    }

    if "position_ids" in batch_inputs:
        result["position_ids"] = batch_inputs["position_ids"][:, -1:]

    return result


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


def pretokenize_prompts(prompts: List[str], tokenizer, device: str) -> List[Dict]:
    """Pre-tokenize all prompts."""
    cached = []
    for p in prompts:
        enc = tokenizer(p, return_tensors="pt", padding=False, truncation=True)
        cached.append({
            "input_ids": enc["input_ids"].to(device),
            "attention_mask": enc["attention_mask"].to(device),
        })
    return cached


def build_padded_gpu_batches(
    cached_inputs: List[Dict],
    tokenizer,
    device: str,
    batch_size: int
) -> List[Tuple[List[int], Dict]]:
    """Build padded batches for GPU processing."""
    batches = []
    n = len(cached_inputs)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_indices = list(range(start, end))

        # Find max length in batch
        max_len = max(cached_inputs[i]["input_ids"].shape[1] for i in batch_indices)

        # Pad and stack
        input_ids_list = []
        attention_mask_list = []

        for i in batch_indices:
            ids = cached_inputs[i]["input_ids"]
            mask = cached_inputs[i]["attention_mask"]
            pad_len = max_len - ids.shape[1]

            if pad_len > 0:
                pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
                ids = torch.cat([
                    torch.full((1, pad_len), pad_id, dtype=ids.dtype, device=device),
                    ids
                ], dim=1)
                mask = torch.cat([
                    torch.zeros((1, pad_len), dtype=mask.dtype, device=device),
                    mask
                ], dim=1)

            input_ids_list.append(ids)
            attention_mask_list.append(mask)

        batch_inputs = {
            "input_ids": torch.cat(input_ids_list, dim=0),
            "attention_mask": torch.cat(attention_mask_list, dim=0),
        }

        batches.append((batch_indices, batch_inputs))

    return batches


# =============================================================================
# CORE EXPERIMENT FUNCTIONS
# =============================================================================

def compute_direction_similarity(
    mc_directions: Dict[int, np.ndarray],
    entropy_directions: Dict[int, np.ndarray]
) -> Dict[int, float]:
    """Compute cosine similarity between MC answer and entropy directions."""
    similarities = {}
    common_layers = set(mc_directions.keys()) & set(entropy_directions.keys())

    for layer_idx in sorted(common_layers):
        mc_dir = mc_directions[layer_idx]
        ent_dir = entropy_directions[layer_idx]

        # Normalize (should already be normalized, but ensure)
        mc_dir = mc_dir / np.linalg.norm(mc_dir)
        ent_dir = ent_dir / np.linalg.norm(ent_dir)

        similarities[layer_idx] = float(np.dot(mc_dir, ent_dir))

    return similarities


class CaptureOnlyHook:
    """
    Hook that captures activations WITHOUT any modification.
    Used for fair baseline comparison in ablation experiments.
    """

    def __init__(self):
        self.activations = None
        self.handle = None

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output
        # Capture activations at last token (no modification)
        self.activations = hs[:, -1, :].detach().cpu().numpy()
        return output

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


class AblationWithCaptureHook:
    """
    Hook that both ablates a direction AND captures the post-ablation activations.
    Used for D2M transfer measurement under ablation.
    """

    def __init__(self, direction: Optional[torch.Tensor] = None):
        self.direction = direction  # Single direction tensor (hidden_dim,)
        self.activations = None
        self.handle = None

    def set_direction(self, direction: torch.Tensor):
        self.direction = direction

    def __call__(self, module, input, output):
        hs = output[0] if isinstance(output, tuple) else output

        if self.direction is not None:
            hs = hs.clone()
            d = self.direction.to(device=hs.device, dtype=hs.dtype)

            # Ablate at last token position
            last_token = hs[:, -1, :]  # (B, H)
            dots = torch.sum(last_token * d, dim=-1, keepdim=True)  # (B, 1)
            proj = dots * d.unsqueeze(0)  # (B, H)
            hs[:, -1, :] = last_token - proj

        # Capture post-ablation activations at last token
        self.activations = hs[:, -1, :].detach().cpu().numpy()

        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


def run_d2m_transfer_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_activations: Dict[int, np.ndarray],
    entropy_values: np.ndarray,
    layers: List[int],
    mc_directions: Dict[int, np.ndarray],
    num_controls: int,
    use_chat_template: bool,
    test_indices: np.ndarray
) -> Dict:
    """
    Measure D2M Transfer R² under MC direction ablation.

    Process:
    1. Train entropy probe on direct activations (train set)
    2. Collect FRESH baseline meta activations (no ablation) via inference
    3. Collect meta activations with MC direction ablated
    4. Apply SAME scaler to both and measure R² change
    5. Repeat with control directions
    6. Compute statistics

    This ensures a fair comparison: only difference is the ablation itself.

    Args:
        model: The language model
        tokenizer: Tokenizer
        questions: List of question dicts
        direct_activations: {layer: direct task activations}
        entropy_values: Ground truth entropy values
        layers: List of layer indices to test
        mc_directions: {layer: MC answer direction}
        num_controls: Number of control directions per layer
        use_chat_template: Whether to use chat template
        test_indices: Indices for test set

    Returns:
        Dict with per-layer results
    """
    print(f"\nRunning D2M Transfer ablation experiment (fair baseline)...")
    print(f"  Layers: {layers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Test set size: {len(test_indices)}")
    print(f"  Controls per layer: {num_controls}")

    n_samples = len(questions)
    all_indices = np.arange(n_samples)
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[test_indices] = False
    train_indices = all_indices[train_mask]

    y_train = entropy_values[train_indices]
    y_test = entropy_values[test_indices]

    # Subset questions to test set for ablation inference
    test_questions = [questions[i] for i in test_indices]

    # Format prompts for test questions
    prompts = []
    for q in test_questions:
        prompt = format_meta_prompt(q, tokenizer, use_chat_template)
        prompts.append(prompt)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)
    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    results = {}

    for layer_idx in tqdm(layers, desc="Layers"):
        # Get direct activations for this layer
        X_direct = direct_activations[layer_idx]
        X_train = X_direct[train_indices]

        # Train entropy probe on direct activations
        scaler, pca, probe = train_entropy_probe(X_train, y_train)

        # Get model layer module
        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        def collect_baseline_activations() -> np.ndarray:
            """Run inference WITHOUT ablation and collect activations."""
            hook = CaptureOnlyHook()
            hook.register(layer_module)

            all_activations = []
            try:
                for _, batch_inputs in gpu_batches:
                    with torch.inference_mode():
                        model(**batch_inputs)
                    all_activations.append(hook.activations.copy())
            finally:
                hook.remove()

            return np.concatenate(all_activations, axis=0)

        def collect_ablated_activations(direction_tensor: torch.Tensor) -> np.ndarray:
            """Run inference with ablation hook and collect post-ablation activations."""
            hook = AblationWithCaptureHook(direction_tensor)
            hook.register(layer_module)

            all_activations = []
            try:
                for _, batch_inputs in gpu_batches:
                    with torch.inference_mode():
                        model(**batch_inputs)
                    all_activations.append(hook.activations.copy())
            finally:
                hook.remove()

            return np.concatenate(all_activations, axis=0)

        # Collect FRESH baseline activations (no ablation)
        baseline_activations = collect_baseline_activations()
        baseline_preds = apply_probe_strict(baseline_activations, scaler, pca, probe)
        baseline_r2 = r2_score(y_test, baseline_preds)

        # Prepare MC direction tensor
        mc_dir = mc_directions[layer_idx]
        mc_dir_tensor = torch.tensor(mc_dir, dtype=torch.float16, device=DEVICE)
        mc_dir_tensor = mc_dir_tensor / mc_dir_tensor.norm()

        # Collect activations with MC direction ablated
        mc_ablated_activations = collect_ablated_activations(mc_dir_tensor)
        mc_preds = apply_probe_strict(mc_ablated_activations, scaler, pca, probe)
        mc_ablated_r2 = r2_score(y_test, mc_preds)

        # Generate control directions
        control_dirs = generate_orthogonal_directions(mc_dir, num_controls)
        control_tensors = [
            torch.tensor(d, dtype=torch.float16, device=DEVICE) / torch.tensor(d, dtype=torch.float16, device=DEVICE).norm()
            for d in control_dirs
        ]

        # Collect activations with control directions ablated
        control_r2s = []
        for ctrl_tensor in control_tensors:
            ctrl_ablated_activations = collect_ablated_activations(ctrl_tensor)
            ctrl_preds = apply_probe_strict(ctrl_ablated_activations, scaler, pca, probe)
            ctrl_r2 = r2_score(y_test, ctrl_preds)
            control_r2s.append(ctrl_r2)

        results[layer_idx] = {
            "baseline_r2": float(baseline_r2),
            "mc_ablated_r2": float(mc_ablated_r2),
            "mc_r2_change": float(mc_ablated_r2 - baseline_r2),
            "control_r2s": control_r2s,
            "control_r2_changes": [r2 - baseline_r2 for r2 in control_r2s],
        }

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_test": len(test_indices),
        "num_controls": num_controls,
        "layer_results": results
    }


def analyze_d2m_transfer_ablation_results(results: Dict) -> Dict:
    """Analyze D2M transfer ablation results with statistical testing."""
    analysis = {
        "layers": results["layers"],
        "num_test": results["num_test"],
        "num_controls": results["num_controls"],
        "effects": {},
    }

    # Collect all control effects for pooled null
    all_control_r2_changes = []
    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]
        all_control_r2_changes.extend(lr["control_r2_changes"])

    pooled_null = np.array(all_control_r2_changes)

    # Per-layer statistics
    raw_p_values = []

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        mc_r2_change = lr["mc_r2_change"]
        control_r2_changes = lr["control_r2_changes"]

        # Pooled p-value (one-sided: MC ablation should DECREASE R²)
        # We test if MC ablation has a more negative effect than controls
        n_pooled_worse = np.sum(pooled_null <= mc_r2_change)
        p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)

        # Z-score
        std_control = np.std(control_r2_changes)
        if std_control > 0:
            z_score = (mc_r2_change - np.mean(control_r2_changes)) / std_control
        else:
            z_score = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "baseline_r2": lr["baseline_r2"],
            "ablated_r2": lr["mc_ablated_r2"],
            "r2_change": mc_r2_change,
            "control_r2_mean": float(np.mean(lr["control_r2s"])),
            "control_change_mean": float(np.mean(control_r2_changes)),
            "control_change_std": float(std_control),
            "p_value_pooled": p_value_pooled,
            "z_score": z_score,
        }

    # FDR correction
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    prev_adjusted = 0.0
    for layer_idx, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer_idx] = max(fdr_adjusted[layer_idx], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer_idx]

    for layer_idx in results["layers"]:
        analysis["effects"][layer_idx]["p_value_fdr"] = fdr_adjusted[layer_idx]

    # Summary
    significant_layers = [
        l for l in results["layers"]
        if analysis["effects"][l]["p_value_fdr"] < 0.05
    ]
    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_fdr": significant_layers,
        "n_significant": len(significant_layers),
    }

    return analysis


def compute_correlation(confidences: np.ndarray, metric_values: np.ndarray) -> float:
    """Compute Pearson correlation."""
    if len(confidences) < 2 or np.std(confidences) == 0 or np.std(metric_values) == 0:
        return 0.0
    return float(np.corrcoef(confidences, metric_values)[0, 1])


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool) -> str:
    """Format a meta/confidence question."""
    prompt, _ = format_stated_confidence_prompt(question, tokenizer, use_chat_template)
    return prompt


def local_response_to_confidence(response: str, probs: np.ndarray, mapping: Optional[Dict]) -> float:
    """Convert response to confidence value."""
    if META_TASK == "delegate":
        return response_to_confidence(response, probs, mapping)
    else:
        return get_stated_confidence_signal(probs)


def run_behavioral_ablation_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_metric_values: np.ndarray,
    layers: List[int],
    mc_directions: Dict[int, np.ndarray],
    num_controls: int,
    use_chat_template: bool
) -> Dict:
    """
    Run ablation experiment measuring behavioral correlation change.
    Uses KV cache for efficiency - only processes last token during ablations.

    For each layer:
    1. Compute baseline confidence-entropy correlation (no ablation)
    2. Ablate MC answer direction, measure new correlation
    3. Ablate control directions, measure their correlations
    4. Compute statistics (p-values, z-scores)
    """
    print(f"\nRunning behavioral ablation experiment (KV-cached)...")
    print(f"  Layers: {layers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Controls per layer: {num_controls}")

    metric_mean = direct_metric_values.mean()
    metric_std = direct_metric_values.std()

    # Format prompts
    prompts = []
    mappings = []
    for q_idx, question in enumerate(questions):
        if META_TASK == "delegate":
            prompt, _, mapping = format_answer_or_delegate_prompt(
                question, tokenizer, use_chat_template, trial_index=q_idx
            )
        else:
            prompt = format_meta_prompt(question, tokenizer, use_chat_template)
            mapping = None
        prompts.append(prompt)
        mappings.append(mapping)

    cached_inputs = pretokenize_prompts(prompts, tokenizer, DEVICE)

    if META_TASK == "delegate":
        option_token_ids = _CACHED_TOKEN_IDS["delegate_options"]
        options = list(ANSWER_OR_DELEGATE_OPTIONS)
    else:
        option_token_ids = _CACHED_TOKEN_IDS["meta_options"]
        options = list(STATED_CONFIDENCE_OPTIONS.keys())

    gpu_batches = build_padded_gpu_batches(cached_inputs, tokenizer, DEVICE, BATCH_SIZE)

    # Compute baseline (no ablation)
    print("Computing baseline...")
    baseline_results = [None] * len(questions)
    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Baseline"):
        with torch.inference_mode():
            out = model(**batch_inputs)
            logits = out.logits[:, -1, :][:, option_token_ids]
            probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

        for i, q_idx in enumerate(batch_indices):
            p = probs[i]
            resp = options[np.argmax(p)]
            conf = local_response_to_confidence(resp, p, mappings[q_idx])
            m_val = direct_metric_values[q_idx]
            align = -((m_val - metric_mean) / metric_std) * ((conf - 0.5) / 0.25)
            baseline_results[q_idx] = {
                "question_idx": q_idx,
                "response": resp,
                "confidence": conf,
                "metric": float(m_val),
                "alignment": float(align)
            }

    # Initialize results storage
    layer_results = {}
    for layer_idx in layers:
        layer_results[layer_idx] = {
            "baseline": baseline_results,
            "mc_answer_ablated": [None] * len(questions),
            "controls_ablated": {f"control_{i}": [None] * len(questions) for i in range(num_controls)}
        }

    # Precompute direction tensors for all layers
    cached_directions = {}
    for layer_idx in layers:
        mc_dir = mc_directions[layer_idx]
        mc_dir_tensor = torch.tensor(mc_dir, dtype=torch.float16, device=DEVICE)
        mc_dir_tensor = mc_dir_tensor / mc_dir_tensor.norm()

        control_dirs = generate_orthogonal_directions(mc_dir, num_controls)
        control_tensors = [
            torch.tensor(d, dtype=torch.float16, device=DEVICE) / torch.tensor(d, dtype=torch.float16, device=DEVICE).norm()
            for d in control_dirs
        ]

        cached_directions[layer_idx] = {
            "mc_answer": mc_dir_tensor,
            "controls": control_tensors
        }

    print(f"Processing {len(gpu_batches)} batches (KV-cached)...")

    # Process batches - compute KV cache once per batch, reuse for all ablations
    for batch_indices, batch_inputs in tqdm(gpu_batches, desc="Batches"):
        B = len(batch_indices)

        # Compute KV Cache once for this batch
        base_step_data = get_kv_cache(model, batch_inputs)
        keys_snapshot, values_snapshot = base_step_data["past_key_values_data"]

        inputs_template = {
            "input_ids": base_step_data["input_ids"],
            "attention_mask": base_step_data["attention_mask"],
            "use_cache": True
        }
        if "position_ids" in base_step_data:
            inputs_template["position_ids"] = base_step_data["position_ids"]

        # Iterate over layers
        for layer_idx in layers:
            if hasattr(model, 'get_base_model'):
                layer_module = model.get_base_model().model.layers[layer_idx]
            else:
                layer_module = model.model.layers[layer_idx]

            mc_dir_tensor = cached_directions[layer_idx]["mc_answer"]
            control_tensors = cached_directions[layer_idx]["controls"]

            hook = BatchAblationHook()
            hook.register(layer_module)

            def run_single_ablation(direction_tensor, result_storage, key=None):
                # Reconstruct fresh cache from snapshot
                pass_cache = create_fresh_cache(keys_snapshot, values_snapshot, expand_size=1)

                current_inputs = inputs_template.copy()
                current_inputs["past_key_values"] = pass_cache

                # Set direction for ablation
                dirs_batch = direction_tensor.unsqueeze(0).expand(B, -1)
                hook.set_directions(dirs_batch)

                # Run only last token through model
                with torch.inference_mode():
                    out = model(**current_inputs)
                    logits = out.logits[:, -1, :][:, option_token_ids]
                    probs = torch.softmax(logits, dim=-1).float().cpu().numpy()

                for i, q_idx in enumerate(batch_indices):
                    p = probs[i]
                    resp = options[np.argmax(p)]
                    conf = local_response_to_confidence(resp, p, mappings[q_idx])
                    m_val = direct_metric_values[q_idx]
                    align = -((m_val - metric_mean) / metric_std) * ((conf - 0.5) / 0.25)

                    data = {
                        "question_idx": q_idx,
                        "response": resp,
                        "confidence": conf,
                        "metric": float(m_val),
                        "alignment": float(align)
                    }
                    if key:
                        result_storage[key][q_idx] = data
                    else:
                        result_storage[q_idx] = data

            try:
                # Ablate MC answer direction
                run_single_ablation(mc_dir_tensor, layer_results[layer_idx]["mc_answer_ablated"])

                # Ablate control directions
                for i_c, ctrl_tensor in enumerate(control_tensors):
                    run_single_ablation(ctrl_tensor, layer_results[layer_idx]["controls_ablated"], key=f"control_{i_c}")
            finally:
                hook.remove()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return {
        "layers": layers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": layer_results
    }


def analyze_behavioral_ablation_results(results: Dict) -> Dict:
    """Analyze behavioral ablation results with statistical testing."""
    analysis = {
        "layers": results["layers"],
        "num_questions": results["num_questions"],
        "num_controls": results["num_controls"],
        "effects": {},
    }

    # Collect all control effects for pooled null
    all_control_corr_changes = []
    layer_data = {}

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]

        baseline_conf = np.array([r["confidence"] for r in lr["baseline"]])
        baseline_metric = np.array([r["metric"] for r in lr["baseline"]])

        ablated_conf = np.array([r["confidence"] for r in lr["mc_answer_ablated"]])
        ablated_metric = np.array([r["metric"] for r in lr["mc_answer_ablated"]])

        baseline_corr = compute_correlation(baseline_conf, baseline_metric)
        ablated_corr = compute_correlation(ablated_conf, ablated_metric)

        control_corrs = []
        for ctrl_key in lr["controls_ablated"]:
            ctrl_conf = np.array([r["confidence"] for r in lr["controls_ablated"][ctrl_key]])
            ctrl_metric = np.array([r["metric"] for r in lr["controls_ablated"][ctrl_key]])
            control_corrs.append(compute_correlation(ctrl_conf, ctrl_metric))

        mc_corr_change = ablated_corr - baseline_corr
        control_corr_changes = [c - baseline_corr for c in control_corrs]

        all_control_corr_changes.extend(control_corr_changes)

        layer_data[layer_idx] = {
            "baseline_corr": baseline_corr,
            "ablated_corr": ablated_corr,
            "mc_corr_change": mc_corr_change,
            "control_corrs": control_corrs,
            "control_corr_changes": control_corr_changes,
        }

    pooled_null = np.array(all_control_corr_changes)

    # Second pass: compute statistics
    raw_p_values = []

    for layer_idx in results["layers"]:
        ld = layer_data[layer_idx]

        # Pooled p-value
        n_pooled_worse = np.sum(pooled_null >= ld["mc_corr_change"])
        p_value_pooled = (n_pooled_worse + 1) / (len(pooled_null) + 1)

        # Effect size z-score
        std_control = np.std(ld["control_corr_changes"])
        if std_control > 0:
            z_score = (ld["mc_corr_change"] - np.mean(ld["control_corr_changes"])) / std_control
        else:
            z_score = 0.0

        raw_p_values.append((layer_idx, p_value_pooled))

        analysis["effects"][layer_idx] = {
            "baseline_correlation": ld["baseline_corr"],
            "ablated_correlation": ld["ablated_corr"],
            "correlation_change": ld["mc_corr_change"],
            "control_correlation_mean": float(np.mean(ld["control_corrs"])),
            "control_change_mean": float(np.mean(ld["control_corr_changes"])),
            "control_change_std": float(std_control),
            "p_value_pooled": p_value_pooled,
            "z_score": z_score,
        }

    # FDR correction
    sorted_pvals = sorted(raw_p_values, key=lambda x: x[1])
    n_tests = len(sorted_pvals)
    fdr_adjusted = {}

    for rank, (layer_idx, p_val) in enumerate(sorted_pvals, 1):
        adjusted = min(1.0, p_val * n_tests / rank)
        fdr_adjusted[layer_idx] = adjusted

    prev_adjusted = 0.0
    for layer_idx, _ in sorted(sorted_pvals, key=lambda x: x[1]):
        fdr_adjusted[layer_idx] = max(fdr_adjusted[layer_idx], prev_adjusted)
        prev_adjusted = fdr_adjusted[layer_idx]

    for layer_idx in results["layers"]:
        analysis["effects"][layer_idx]["p_value_fdr"] = fdr_adjusted[layer_idx]

    # Summary
    significant_layers = [
        l for l in results["layers"]
        if analysis["effects"][l]["p_value_fdr"] < 0.05
    ]
    analysis["summary"] = {
        "pooled_null_size": len(pooled_null),
        "significant_layers_fdr": significant_layers,
        "n_significant": len(significant_layers),
    }

    return analysis


# =============================================================================
# PLOTTING
# =============================================================================

def plot_results(
    behavioral_analysis: Dict,
    d2m_analysis: Optional[Dict],
    direction_similarity: Dict[int, float],
    output_prefix: str
):
    """Generate plots for the experiment results."""

    layers = behavioral_analysis["layers"]

    # Determine layout: 3 panels if D2M analysis exists, otherwise 2
    if d2m_analysis is not None:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(layers))
    width = 0.35

    # Panel 1: Behavioral correlation change
    ax1 = axes[0]
    mc_changes = [behavioral_analysis["effects"][l]["correlation_change"] for l in layers]
    ctrl_means = [behavioral_analysis["effects"][l]["control_change_mean"] for l in layers]
    ctrl_stds = [behavioral_analysis["effects"][l]["control_change_std"] for l in layers]
    p_raw_beh = [behavioral_analysis["effects"][l]["p_value_pooled"] for l in layers]
    p_fdr_beh = [behavioral_analysis["effects"][l]["p_value_fdr"] for l in layers]

    ax1.bar(x - width/2, mc_changes, width, label='MC Answer Ablation', color='tab:red', alpha=0.8)
    ax1.bar(x + width/2, ctrl_means, width, yerr=ctrl_stds, label='Control (mean±std)',
            color='tab:gray', alpha=0.6, capsize=3)

    ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Correlation Change (ablated - baseline)')
    ax1.set_title('Behavioral: Confidence-Entropy Correlation')
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.legend(loc='best', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add p-value annotations
    y_range = max(mc_changes) - min(mc_changes) if mc_changes else 1
    for i, l in enumerate(layers):
        p_raw = p_raw_beh[i]
        p_fdr = p_fdr_beh[i]
        y_pos = mc_changes[i]
        # Position annotation above or below bar depending on sign
        offset = 0.05 * y_range if y_pos >= 0 else -0.15 * y_range
        va = 'bottom' if y_pos >= 0 else 'top'
        # Only annotate if p_raw < 0.1 to avoid clutter
        if p_raw < 0.1:
            sig_marker = "*" if p_fdr < 0.05 else ""
            ax1.annotate(f'p={p_raw:.2f}{sig_marker}', (x[i] - width/2, y_pos + offset),
                        ha='center', va=va, fontsize=6, rotation=90)

    # Panel 2: D2M Transfer R² change (if available)
    if d2m_analysis is not None:
        ax2 = axes[1]
        d2m_layers = d2m_analysis["layers"]
        d2m_changes = [d2m_analysis["effects"][l]["r2_change"] for l in d2m_layers]
        d2m_ctrl_means = [d2m_analysis["effects"][l]["control_change_mean"] for l in d2m_layers]
        d2m_ctrl_stds = [d2m_analysis["effects"][l]["control_change_std"] for l in d2m_layers]
        p_raw_d2m = [d2m_analysis["effects"][l]["p_value_pooled"] for l in d2m_layers]
        p_fdr_d2m = [d2m_analysis["effects"][l]["p_value_fdr"] for l in d2m_layers]

        x_d2m = np.arange(len(d2m_layers))

        ax2.bar(x_d2m - width/2, d2m_changes, width, label='MC Answer Ablation', color='tab:orange', alpha=0.8)
        ax2.bar(x_d2m + width/2, d2m_ctrl_means, width, yerr=d2m_ctrl_stds, label='Control (mean±std)',
                color='tab:gray', alpha=0.6, capsize=3)

        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('R² Change (ablated - baseline)')
        ax2.set_title('D2M Transfer: Entropy Probe R²')
        ax2.set_xticks(x_d2m)
        ax2.set_xticklabels(d2m_layers)
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add p-value annotations
        y_range_d2m = max(d2m_changes) - min(d2m_changes) if d2m_changes else 1
        for i, l in enumerate(d2m_layers):
            p_raw = p_raw_d2m[i]
            p_fdr = p_fdr_d2m[i]
            y_pos = d2m_changes[i]
            # Position annotation above or below bar depending on sign
            offset = 0.05 * y_range_d2m if y_pos >= 0 else -0.15 * y_range_d2m
            va = 'bottom' if y_pos >= 0 else 'top'
            # Only annotate if p_raw < 0.1 to avoid clutter
            if p_raw < 0.1:
                sig_marker = "*" if p_fdr < 0.05 else ""
                ax2.annotate(f'p={p_raw:.2f}{sig_marker}', (x_d2m[i] - width/2, y_pos + offset),
                            ha='center', va=va, fontsize=6, rotation=90)

        ax_sim = axes[2]
    else:
        ax_sim = axes[1]

    # Panel 3 (or 2): Direction similarity
    sim_layers = sorted(direction_similarity.keys())
    similarities = [direction_similarity[l] for l in sim_layers]

    ax_sim.plot(sim_layers, similarities, 'o-', color='tab:blue', markersize=6, linewidth=2)
    ax_sim.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax_sim.set_xlabel('Layer')
    ax_sim.set_ylabel('Cosine Similarity')
    ax_sim.set_title('MC Answer vs Entropy Direction')
    ax_sim.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_mc_answer_ablation.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {output_prefix}_mc_answer_ablation.png")


# =============================================================================
# MAIN
# =============================================================================

def main():
    global METRIC

    parser = argparse.ArgumentParser(description="MC Answer Probe Causality Experiment")
    parser.add_argument("--metric", type=str, default=METRIC, choices=AVAILABLE_METRICS,
                        help=f"Metric for entropy probe (default: {METRIC})")
    parser.add_argument("--num-questions", type=int, default=NUM_QUESTIONS,
                        help=f"Number of questions (default: {NUM_QUESTIONS})")
    parser.add_argument("--num-controls", type=int, default=None,
                        help="Number of control directions per layer (default: 25, use 100+ for tighter p-values)")
    parser.add_argument("--similarity-only", action="store_true",
                        help="Only compute direction similarity (skip ablation)")
    args = parser.parse_args()

    METRIC = args.metric
    num_questions = args.num_questions

    # Override NUM_CONTROL_DIRECTIONS if specified
    global NUM_CONTROL_DIRECTIONS
    if args.num_controls is not None:
        NUM_CONTROL_DIRECTIONS = args.num_controls

    print("=" * 70)
    print("MC ANSWER PROBE CAUSALITY EXPERIMENT")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Device: {DEVICE}")

    # Generate paths
    output_prefix = get_output_prefix()
    directions_prefix = get_directions_prefix()

    # Load directions
    mc_directions = load_mc_answer_directions(directions_prefix)
    entropy_directions = load_entropy_directions(directions_prefix, METRIC)

    # Load probe results for layer selection
    probe_results = load_probe_results(output_prefix, METRIC)

    # Load paired data (questions and direct metrics)
    paired_data = load_paired_data(output_prefix)
    questions = paired_data["questions"]
    direct_metrics = paired_data["direct_metrics"]

    if METRIC not in direct_metrics:
        raise ValueError(f"Metric '{METRIC}' not found in paired data. Available: {list(direct_metrics.keys())}")

    direct_metric_values = direct_metrics[METRIC]
    print(f"Loaded {len(questions)} questions with {METRIC} values")
    print(f"  {METRIC} range: [{direct_metric_values.min():.3f}, {direct_metric_values.max():.3f}]")

    # Subsample if needed
    if len(questions) > num_questions:
        np.random.seed(SEED)
        indices = np.random.choice(len(questions), num_questions, replace=False)
        questions = [questions[i] for i in indices]
        direct_metric_values = direct_metric_values[indices]
    print(f"Using {len(questions)} questions")

    # Select layers based on probe performance
    layer_candidates = []
    if "probe_results" in probe_results:
        for layer_str, lr in probe_results["probe_results"].items():
            d2m_r2 = lr.get("direct_to_meta_fixed", {}).get("r2", 0)
            d2d_r2 = lr.get("direct_to_direct", {}).get("test_r2", 0)
            if d2m_r2 >= D2M_R2_THRESHOLD and d2d_r2 >= D2D_R2_THRESHOLD:
                layer_candidates.append((int(layer_str), d2m_r2))

    if layer_candidates:
        layer_candidates.sort(key=lambda x: -x[1])
        layers = [l[0] for l in layer_candidates]
    else:
        # Fallback to MC answer direction layers
        layers = sorted(mc_directions.keys())

    # Filter to layers that have both MC and entropy directions
    layers = [l for l in layers if l in mc_directions and l in entropy_directions]
    print(f"\nSelected {len(layers)} layers: {layers}")

    # Compute direction similarity
    print("\nComputing direction similarities...")
    direction_similarity = compute_direction_similarity(mc_directions, entropy_directions)
    print(f"  Similarity range: [{min(direction_similarity.values()):.3f}, {max(direction_similarity.values()):.3f}]")

    # Print similarity summary
    print("\n" + "=" * 70)
    print("DIRECTION SIMILARITY (MC Answer vs Entropy)")
    print("=" * 70)
    for layer_idx in sorted(direction_similarity.keys()):
        sim = direction_similarity[layer_idx]
        print(f"  Layer {layer_idx:2d}: {sim:+.4f}")

    if args.similarity_only:
        # Save similarity-only results
        results = {
            "config": {
                "base_model": BASE_MODEL_NAME,
                "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
                "dataset": DATASET_NAME,
                "metric": METRIC,
            },
            "direction_similarity": {str(k): v for k, v in direction_similarity.items()},
        }
        results_path = f"{output_prefix}_{METRIC}_mc_answer_similarity.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved similarity results to {results_path}")
        return

    # Determine number of controls
    if NUM_CONTROL_DIRECTIONS is None:
        num_controls = max(MIN_CONTROLS_PER_LAYER, FDR_SAFETY_FACTOR)
    else:
        num_controls = NUM_CONTROL_DIRECTIONS
    print(f"\nControls per layer: {num_controls}")

    # Load activations for D2M transfer experiment
    print("\nLoading activations for D2M transfer analysis...")
    try:
        direct_activations, entropy_values = load_activations(
            directions_prefix, layers
        )
        # Get test indices from probe results
        test_indices = np.array(probe_results.get("test_indices", []))
        if len(test_indices) == 0:
            # Fallback: use 20% of data as test set
            n_samples = len(entropy_values)
            test_indices = np.random.choice(n_samples, int(n_samples * 0.2), replace=False)
        run_d2m_experiment = True
        print(f"  Loaded activations for {len(layers)} layers")
        print(f"  Test set size: {len(test_indices)}")
    except FileNotFoundError as e:
        print(f"  Warning: Could not load activations: {e}")
        print("  Skipping D2M transfer experiment")
        run_d2m_experiment = False
        direct_activations = None
        entropy_values = None
        test_indices = None

    # Load model for ablation experiments
    print("\nLoading model...")
    adapter_path = MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None
    model, tokenizer, num_layers = load_model_and_tokenizer(
        BASE_MODEL_NAME,
        adapter_path=adapter_path,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )
    model.eval()

    initialize_token_cache(tokenizer)
    use_chat_template = should_use_chat_template(BASE_MODEL_NAME, tokenizer)

    # Run behavioral ablation experiment
    print("\n" + "=" * 70)
    print("BEHAVIORAL ABLATION EXPERIMENT")
    print("=" * 70)

    ablation_results = run_behavioral_ablation_experiment(
        model, tokenizer, questions, direct_metric_values,
        layers, mc_directions, num_controls, use_chat_template
    )

    # Analyze results
    behavioral_analysis = analyze_behavioral_ablation_results(ablation_results)

    # Print summary
    print("\n" + "=" * 70)
    print("BEHAVIORAL ABLATION RESULTS")
    print("=" * 85)
    print(f"{'Layer':<8} {'Baseline r':<12} {'Ablated r':<12} {'Delta':<10} {'p (raw)':<10} {'p (FDR)':<10} {'Z-score':<10}")
    print("-" * 85)

    for layer_idx in layers:
        eff = behavioral_analysis["effects"][layer_idx]
        sig = "*" if eff["p_value_fdr"] < 0.05 else ""
        print(f"{layer_idx:<8} {eff['baseline_correlation']:+.4f}      {eff['ablated_correlation']:+.4f}      "
              f"{eff['correlation_change']:+.4f}    {eff['p_value_pooled']:.4f}    {eff['p_value_fdr']:.4f}    {eff['z_score']:+.2f} {sig}")

    print("-" * 70)
    print(f"Significant layers (FDR<0.05): {behavioral_analysis['summary']['significant_layers_fdr']}")

    # Run D2M transfer ablation experiment
    d2m_analysis = None
    if run_d2m_experiment:
        print("\n" + "=" * 70)
        print("D2M TRANSFER ABLATION EXPERIMENT")
        print("=" * 70)

        # Use all questions (not subsampled) for D2M since we need to match activation indices
        all_questions = paired_data["questions"]

        d2m_results = run_d2m_transfer_ablation_experiment(
            model, tokenizer, all_questions,
            direct_activations, entropy_values,
            layers, mc_directions, num_controls, use_chat_template,
            test_indices
        )

        d2m_analysis = analyze_d2m_transfer_ablation_results(d2m_results)

        # Print D2M results
        print("\n" + "=" * 70)
        print("D2M TRANSFER ABLATION RESULTS")
        print("=" * 85)
        print(f"{'Layer':<8} {'Baseline R²':<12} {'Ablated R²':<12} {'Delta':<10} {'p (raw)':<10} {'p (FDR)':<10} {'Z-score':<10}")
        print("-" * 85)

        for layer_idx in layers:
            eff = d2m_analysis["effects"][layer_idx]
            sig = "*" if eff["p_value_fdr"] < 0.05 else ""
            print(f"{layer_idx:<8} {eff['baseline_r2']:.4f}       {eff['ablated_r2']:.4f}       "
                  f"{eff['r2_change']:+.4f}    {eff['p_value_pooled']:.4f}    {eff['p_value_fdr']:.4f}    {eff['z_score']:+.2f} {sig}")

        print("-" * 70)
        print(f"Significant layers (FDR<0.05): {d2m_analysis['summary']['significant_layers_fdr']}")

    # Save results
    results = {
        "config": {
            "base_model": BASE_MODEL_NAME,
            "adapter": MODEL_NAME if MODEL_NAME != BASE_MODEL_NAME else None,
            "dataset": DATASET_NAME,
            "metric": METRIC,
            "num_questions": len(questions),
            "num_controls": num_controls,
            "layers": layers,
        },
        "direction_similarity": {str(k): v for k, v in direction_similarity.items()},
        "behavioral_ablation": behavioral_analysis,
    }

    if d2m_analysis is not None:
        results["d2m_transfer_ablation"] = d2m_analysis

    results_path = f"{output_prefix}_{METRIC}_mc_answer_ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Generate plots
    plot_results(behavioral_analysis, d2m_analysis, direction_similarity, output_prefix + f"_{METRIC}")


if __name__ == "__main__":
    main()