from __future__ import annotations

"""
Orthogonal-component analysis for uncertainty directions.

Goal:
Compare across meta tasks whether the part of d_metamcuncert beyond d_mcuncert
(and vice versa) still predicts the meta-task output.

For each task and layer this script computes:
- cosine(d_mcuncert, d_metamcuncert)
- single-direction fit of d_mcuncert and d_metamcuncert to the meta output
- single-direction fit of d_metamcuncert ⟂ d_mcuncert and d_mcuncert ⟂ d_metamcuncert
- added fit from the orthogonal component beyond the parent direction
- logitlens analysis of original, orthogonalized, and shared direction components

Inputs (per task in META_TASKS):
    {model_dir}/working/{dataset}_mc_{metric}_directions.npz          d_mcuncert (from identify_mc_correlate.py)
    {model_dir}/working/{dataset}_meta_{task}_mcuncert_directions_final.npz   d_metamcuncert (from test_meta_transfer.py)
    {model_dir}/working/{dataset}_meta_{task}_activations.npz         Meta-task activations (from test_meta_transfer.py)

Outputs:
    {model_dir}/results/{dataset}_meta_orthogonal_uncertainty_{metric}.png    Combined figure (rows=tasks, cols=4)
    {model_dir}/results/{dataset}_meta_orthogonal_uncertainty_{metric}.json   JSON summary with per-layer details
    {model_dir}/results/{dataset}_meta_logitlens_{task}_{direction}_{metric}.png  One plot per direction per task
        Directions: d_mcuncert, d_metamcuncert, d_mc_perp_metamc, d_metamc_perp_mc, shared
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from core import DEVICE
from core.config_utils import find_output_file, get_config_dict, get_output_path
from core.model_utils import get_model_dir_name
from core.plotting import GRID_ALPHA, save_figure

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
ADAPTER = None
LOAD_IN_4BIT = True
LOAD_IN_8BIT = False

DATASET = "TriviaMC_difficulty_filtered"
UNCERTAINTY_METRIC = "logit_gap"
METHOD = "mean_diff"
META_TASKS = ("delegate", "confidence", "other_confidence")

DELEGATE_OUTPUT_TARGET = "logit_margins"
NON_DELEGATE_OUTPUT_PREFERENCE = ("computed_confidence", "confidences")

# Colors: blue for d_mcuncert-related, orange for d_metamcuncert-related
COLOR_MC = "tab:blue"
COLOR_METAMC = "tab:orange"
COLOR_COSINE = "tab:purple"

SAVE_FULL_PER_LAYER_DETAILS = True

# Logitlens configuration
RUN_LOGITLENS = True
LOGITLENS_TOP_K = 10
LOGITLENS_LAYERS = None  # None = all layers, or list like [20, 30, 40]

# =============================================================================
# DATACLASSES
# =============================================================================


@dataclass
class DirectionSpec:
    name: str
    description: str
    file_patterns: List[str]
    key_patterns: List[str]


@dataclass
class LoadedDirection:
    name: str
    description: str
    file_path: str
    key_pattern: str
    layers: Dict[int, np.ndarray]


# =============================================================================
# HELPERS
# =============================================================================


def json_ready(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return json_ready(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_ready(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_ready(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if not math.isfinite(x) else x
    return obj


_LAYER_RE_CACHE: Dict[str, re.Pattern] = {}


def key_pattern_to_regex(key_pattern: str) -> re.Pattern:
    if key_pattern in _LAYER_RE_CACHE:
        return _LAYER_RE_CACHE[key_pattern]
    pattern = re.escape(key_pattern).replace(re.escape("{layer}"), r"(\d+)")
    regex = re.compile(rf"^{pattern}$")
    _LAYER_RE_CACHE[key_pattern] = regex
    return regex


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def standardize_columns(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    keep = stds > 0
    Z = np.zeros((X.shape[0], int(np.sum(keep))), dtype=float)
    if np.any(keep):
        Z = (X[:, keep] - means[keep]) / stds[keep]
    return Z, keep


def standardize_vector(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    sd = np.std(y, ddof=0)
    if sd == 0:
        return np.zeros_like(y, dtype=float), False
    return (y - np.mean(y)) / sd, True


def regression_r2(X: np.ndarray, y: np.ndarray) -> Tuple[float, Optional[np.ndarray], List[int]]:
    if X.ndim == 1:
        X = X[:, None]
    Zx, keep_mask = standardize_columns(X)
    Zy, y_ok = standardize_vector(y)
    kept = [i for i, k in enumerate(keep_mask) if k]
    if not y_ok or Zx.shape[1] == 0 or Zx.shape[0] <= Zx.shape[1]:
        return 0.0, None, kept
    beta, *_ = np.linalg.lstsq(Zx, Zy, rcond=None)
    pred = Zx @ beta
    ss_res = float(np.sum((Zy - pred) ** 2))
    ss_tot = float(np.sum((Zy - Zy.mean()) ** 2))
    r2 = 0.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    return float(r2), beta, kept


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n == 0.0:
        return np.zeros_like(v)
    return v / n


def orthogonal_component(v: np.ndarray, onto: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    onto = np.asarray(onto, dtype=float)
    denom = float(np.dot(onto, onto))
    if denom == 0.0:
        return v.copy()
    return v - (float(np.dot(v, onto)) / denom) * onto


def shared_component(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Compute the shared component between two directions (normalized sum, sign-aligned)."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    u = normalize(u)
    v = normalize(v)
    # Align signs since directions are sign-ambiguous (d and -d represent same axis)
    if np.dot(u, v) < 0:
        v = -v
    return normalize(u + v)


# =============================================================================
# LOGITLENS
# =============================================================================


def load_lm_head_and_norm(model_name: str) -> Tuple[torch.Tensor, torch.Tensor, Any]:
    """Load lm_head weight, final norm weight, and tokenizer from model files."""
    print(f"Loading model weights for logitlens: {model_name}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Get weight map from model index
    index_file = hf_hub_download(repo_id=model_name, filename="model.safetensors.index.json")
    with open(index_file) as f:
        index = json.load(f)
    weight_map = index["weight_map"]

    # Find lm_head file
    lm_head_file = weight_map.get("lm_head.weight")
    if not lm_head_file:
        raise ValueError(f"Could not find lm_head.weight in model index")

    # Download and load lm_head
    print(f"  Downloading {lm_head_file}...")
    lm_head_path = hf_hub_download(repo_id=model_name, filename=lm_head_file)
    print(f"  Loading lm_head weight to {DEVICE}...")
    from safetensors import safe_open
    with safe_open(lm_head_path, framework="pt", device=str(DEVICE)) as f:
        lm_head_weight = f.get_tensor("lm_head.weight")
    print(f"  Loaded lm_head weight: {lm_head_weight.shape}, dtype: {lm_head_weight.dtype}")

    # Load norm weight
    norm_weight = None
    norm_file = weight_map.get("model.norm.weight")
    if norm_file:
        if norm_file != lm_head_file:
            norm_path = hf_hub_download(repo_id=model_name, filename=norm_file)
            with safe_open(norm_path, framework="pt", device=str(DEVICE)) as f:
                norm_weight = f.get_tensor("model.norm.weight")
        else:
            with safe_open(lm_head_path, framework="pt", device=str(DEVICE)) as f:
                norm_weight = f.get_tensor("model.norm.weight")
        print(f"  Loaded norm weight: {norm_weight.shape}")

    return lm_head_weight, norm_weight, tokenizer


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Apply RMSNorm to a vector."""
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    return weight * x_normed


def logit_lens_for_direction(
    direction: np.ndarray,
    lm_head_weight: torch.Tensor,
    tokenizer,
    top_k: int = 10,
    norm_weight: Optional[torch.Tensor] = None,
) -> Tuple[List[str], List[float]]:
    """Project a direction through the unembedding matrix."""
    direction_tensor = torch.tensor(
        direction, dtype=lm_head_weight.dtype, device=lm_head_weight.device
    )
    if norm_weight is not None:
        direction_tensor = rms_norm(direction_tensor, norm_weight)
    logits = direction_tensor @ lm_head_weight.T
    probs = torch.nn.functional.softmax(logits, dim=-1)
    values, indices = torch.topk(probs, top_k)
    tokens = tokenizer.batch_decode(indices.unsqueeze(-1))
    return tokens, values.cpu().tolist()


def clean_token_str(s: str) -> str:
    """Clean token string for display."""
    s = re.sub(r'[^\x00-\x7F]+', '', str(s))
    s = s.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
    if len(s) > 12:
        s = s[:10] + '..'
    return s.strip() or '□'


def plot_logitlens_single_direction(
    direction_by_layer: Dict[int, np.ndarray],
    layers: List[int],
    lm_head_weight: torch.Tensor,
    tokenizer,
    output_path: Path,
    direction_name: str,
    top_k: int = 10,
    norm_weight: Optional[torch.Tensor] = None,
) -> None:
    """Plot logitlens heatmap for a single direction type (like analyze_directions.py)."""
    pos_token_data, pos_probs_data = [], []
    neg_token_data, neg_probs_data = [], []

    for layer in layers:
        if layer in direction_by_layer:
            direction = direction_by_layer[layer]
            pos_tokens, pos_probs = logit_lens_for_direction(
                direction, lm_head_weight, tokenizer, top_k, norm_weight
            )
            neg_tokens, neg_probs = logit_lens_for_direction(
                -direction, lm_head_weight, tokenizer, top_k, norm_weight
            )
            pos_token_data.append(pos_tokens)
            pos_probs_data.append(pos_probs)
            neg_token_data.append(neg_tokens)
            neg_probs_data.append(neg_probs)
        else:
            pos_token_data.append([''] * top_k)
            pos_probs_data.append([0.0] * top_k)
            neg_token_data.append([''] * top_k)
            neg_probs_data.append([0.0] * top_k)

    if not pos_probs_data:
        print(f"  No data for {direction_name}")
        return

    pos_probs_array = np.array(pos_probs_data)
    neg_probs_array = np.array(neg_probs_data)
    pos_cleaned = np.vectorize(clean_token_str)(np.array(pos_token_data))
    neg_cleaned = np.vectorize(clean_token_str)(np.array(neg_token_data))

    # Dynamic height based on number of layers (like analyze_directions.py)
    plt.rcParams['font.family'] = 'DejaVu Sans'
    panel_height = max(4, len(layers) * 0.25)
    fig, axes = plt.subplots(2, 1, figsize=(14, panel_height * 2 + 1))

    # Positive direction (top panel)
    sns.heatmap(
        pos_probs_array,
        annot=pos_cleaned,
        fmt='',
        cmap="Reds",
        xticklabels=False,
        yticklabels=[f"L{l}" for l in layers],
        ax=axes[0],
        cbar_kws={'label': 'Softmax Prob'},
    )
    axes[0].set_title(f"Positive Direction (+d): {direction_name}")
    axes[0].set_xlabel(f"Top {top_k} Tokens")
    axes[0].set_ylabel("Layer")

    # Negative direction (bottom panel)
    sns.heatmap(
        neg_probs_array,
        annot=neg_cleaned,
        fmt='',
        cmap="Blues",
        xticklabels=False,
        yticklabels=[f"L{l}" for l in layers],
        ax=axes[1],
        cbar_kws={'label': 'Softmax Prob'},
    )
    axes[1].set_title(f"Negative Direction (-d): {direction_name}")
    axes[1].set_xlabel(f"Top {top_k} Tokens")
    axes[1].set_ylabel("Layer")

    plt.tight_layout()
    save_figure(fig, output_path)


# =============================================================================
# LOADING
# =============================================================================


def infer_output_target(task: str, activations: np.lib.npyio.NpzFile) -> str:
    if task == "delegate":
        if DELEGATE_OUTPUT_TARGET in activations.files:
            return DELEGATE_OUTPUT_TARGET
        raise KeyError(f"Delegate task expected {DELEGATE_OUTPUT_TARGET!r}; available: {sorted(activations.files)}")
    for key in NON_DELEGATE_OUTPUT_PREFERENCE:
        if key in activations.files:
            return key
    raise KeyError(f"Could not infer meta output key. Available: {sorted(activations.files)}")


def get_direction_specs(dataset: str, task: str, uncertainty_metric: str) -> Dict[str, DirectionSpec]:
    return {
        "d_mcuncert": DirectionSpec(
            name="d_mcuncert",
            description="MC uncertainty direction",
            file_patterns=[f"{dataset}_mc_{uncertainty_metric}_directions.npz"],
            key_patterns=[f"{METHOD}_layer_{{layer}}"],
        ),
        "d_metamcuncert": DirectionSpec(
            name="d_metamcuncert",
            description="Meta-task recomputed uncertainty direction",
            file_patterns=[
                f"{dataset}_meta_{task}_mcuncert_directions_final.npz",
                f"{dataset}_meta_{task}_mcuncert_directions.npz",
            ],
            key_patterns=[f"{METHOD}_{uncertainty_metric}_layer_{{layer}}"],
        ),
    }


def load_direction_from_spec(spec: DirectionSpec, model_dir: str) -> Optional[LoadedDirection]:
    for file_pattern in spec.file_patterns:
        path = find_output_file(file_pattern, model_dir=model_dir)
        if not path.exists():
            continue
        data = np.load(path)
        for key_pattern in spec.key_patterns:
            regex = key_pattern_to_regex(key_pattern)
            layers: Dict[int, np.ndarray] = {}
            for key in data.files:
                m = regex.match(key)
                if m:
                    layers[int(m.group(1))] = np.asarray(data[key], dtype=float)
            if layers:
                return LoadedDirection(spec.name, spec.description, str(path), key_pattern, dict(sorted(layers.items())))
    return None


def load_meta_activations(dataset: str, task: str, model_dir: str) -> np.lib.npyio.NpzFile:
    path = find_output_file(f"{dataset}_meta_{task}_activations.npz", model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Meta activations file not found: {path}")
    return np.load(path)


def available_activation_layers(activations: np.lib.npyio.NpzFile) -> Dict[int, str]:
    layers: Dict[int, str] = {}
    regex = re.compile(r"^layer_(\d+)_final$")
    for key in activations.files:
        m = regex.match(key)
        if m:
            layers[int(m.group(1))] = key
    return dict(sorted(layers.items()))


# =============================================================================
# ANALYSIS
# =============================================================================


def summarize_metric(layer_vals: Dict[int, float]) -> Dict[str, Any]:
    if not layer_vals:
        return {"n_layers": 0, "mean": None, "best": None, "best_layer": None}
    layers = sorted(layer_vals)
    vals = np.array([layer_vals[l] for l in layers], dtype=float)
    best_idx = int(np.argmax(vals))
    return {
        "n_layers": len(layers),
        "mean": float(np.mean(vals)),
        "best": float(vals[best_idx]),
        "best_layer": int(layers[best_idx]),
    }


def analyze_task(
    task: str,
    dataset: str,
    uncertainty_metric: str,
    model_dir: str,
) -> Dict[str, Any]:
    directions: Dict[str, LoadedDirection] = {}
    for name, spec in get_direction_specs(dataset, task, uncertainty_metric).items():
        loaded = load_direction_from_spec(spec, model_dir)
        if loaded is None:
            raise FileNotFoundError(f"Could not load required direction {name!r} for task={task}")
        directions[name] = loaded

    activations = load_meta_activations(dataset, task, model_dir)
    output_key = infer_output_target(task, activations)
    y_all = np.asarray(activations[output_key], dtype=float)
    layer_keys = available_activation_layers(activations)

    common_layers = sorted(set(layer_keys) & set(directions["d_mcuncert"].layers) & set(directions["d_metamcuncert"].layers))

    per_layer: Dict[int, Dict[str, Any]] = {}
    cosine_per_layer: Dict[int, float] = {}
    raw_mc_r2: Dict[int, float] = {}
    raw_meta_r2: Dict[int, float] = {}
    meta_perp_mc_r2: Dict[int, float] = {}
    mc_perp_meta_r2: Dict[int, float] = {}
    add_meta_perp_to_mc: Dict[int, float] = {}
    add_mc_perp_to_meta: Dict[int, float] = {}
    combined_r2: Dict[int, float] = {}

    for layer in common_layers:
        acts = np.asarray(activations[layer_keys[layer]], dtype=float)
        u = np.asarray(directions["d_mcuncert"].layers[layer], dtype=float)
        v = np.asarray(directions["d_metamcuncert"].layers[layer], dtype=float)
        if acts.ndim != 2 or acts.shape[1] != u.shape[0] or acts.shape[1] != v.shape[0]:
            continue

        u = normalize(u)
        v = normalize(v)
        v_perp_u = normalize(orthogonal_component(v, u))
        u_perp_v = normalize(orthogonal_component(u, v))

        proj_u = acts @ u
        proj_v = acts @ v
        proj_v_perp_u = acts @ v_perp_u
        proj_u_perp_v = acts @ u_perp_v

        cosine_per_layer[layer] = cosine_similarity(u, v)

        mask_u = finite_mask(proj_u, y_all)
        mask_v = finite_mask(proj_v, y_all)
        mask_vp = finite_mask(proj_v_perp_u, y_all)
        mask_up = finite_mask(proj_u_perp_v, y_all)

        r2_u, _, _ = regression_r2(proj_u[mask_u][:, None], y_all[mask_u])
        r2_v, _, _ = regression_r2(proj_v[mask_v][:, None], y_all[mask_v])
        r2_vp, _, _ = regression_r2(proj_v_perp_u[mask_vp][:, None], y_all[mask_vp])
        r2_up, _, _ = regression_r2(proj_u_perp_v[mask_up][:, None], y_all[mask_up])

        raw_mc_r2[layer] = r2_u
        raw_meta_r2[layer] = r2_v
        meta_perp_mc_r2[layer] = r2_vp
        mc_perp_meta_r2[layer] = r2_up

        mask_pair1 = finite_mask(proj_u, proj_v_perp_u, y_all)
        X1 = np.column_stack([proj_u[mask_pair1], proj_v_perp_u[mask_pair1]])
        y1 = y_all[mask_pair1]
        full1, _, _ = regression_r2(X1, y1)
        base1, _, _ = regression_r2(proj_u[mask_pair1][:, None], y1)
        add1 = max(0.0, full1 - base1)

        mask_pair2 = finite_mask(proj_v, proj_u_perp_v, y_all)
        X2 = np.column_stack([proj_v[mask_pair2], proj_u_perp_v[mask_pair2]])
        y2 = y_all[mask_pair2]
        full2, _, _ = regression_r2(X2, y2)
        base2, _, _ = regression_r2(proj_v[mask_pair2][:, None], y2)
        add2 = max(0.0, full2 - base2)

        combined = max(full1, full2)
        add_meta_perp_to_mc[layer] = add1
        add_mc_perp_to_meta[layer] = add2
        combined_r2[layer] = combined

        per_layer[layer] = {
            "cosine_mc_vs_metamc": cosine_per_layer[layer],
            "single_r2": {
                "d_mcuncert": r2_u,
                "d_metamcuncert": r2_v,
                "d_metamcuncert_perp_d_mcuncert": r2_vp,
                "d_mcuncert_perp_d_metamcuncert": r2_up,
            },
            "added_r2": {
                "add_d_metamcuncert_perp_to_d_mcuncert": add1,
                "add_d_mcuncert_perp_to_d_metamcuncert": add2,
            },
            "combined_r2": combined,
            "residual_norm_fraction": {
                "d_metamcuncert_perp_d_mcuncert": float(np.linalg.norm(orthogonal_component(v, u)) / (np.linalg.norm(v) + 1e-12)),
                "d_mcuncert_perp_d_metamcuncert": float(np.linalg.norm(orthogonal_component(u, v)) / (np.linalg.norm(u) + 1e-12)),
            },
        }

    # Collect directions for logitlens (after normalization and orthogonalization)
    directions_for_logitlens: Dict[str, Dict[int, np.ndarray]] = {
        "d_mcuncert": {},
        "d_metamcuncert": {},
        "d_mc⟂metamc": {},
        "d_metamc⟂mc": {},
        "shared": {},
    }

    for layer in common_layers:
        u = np.asarray(directions["d_mcuncert"].layers[layer], dtype=float)
        v = np.asarray(directions["d_metamcuncert"].layers[layer], dtype=float)
        u = normalize(u)
        v = normalize(v)
        v_perp_u = normalize(orthogonal_component(v, u))
        u_perp_v = normalize(orthogonal_component(u, v))
        sh = shared_component(u, v)

        directions_for_logitlens["d_mcuncert"][layer] = u
        directions_for_logitlens["d_metamcuncert"][layer] = v
        directions_for_logitlens["d_mc⟂metamc"][layer] = u_perp_v
        directions_for_logitlens["d_metamc⟂mc"][layer] = v_perp_u
        directions_for_logitlens["shared"][layer] = sh

    return {
        "task": task,
        "output_target": output_key,
        "loaded_directions": {
            name: {
                "description": d.description,
                "file_path": d.file_path,
                "key_pattern": d.key_pattern,
                "n_layers": len(d.layers),
            }
            for name, d in directions.items()
        },
        "summary": {
            "cosine_mc_vs_metamc": summarize_metric({l: abs(v) for l, v in cosine_per_layer.items()}),
            "raw_d_mcuncert_r2": summarize_metric(raw_mc_r2),
            "raw_d_metamcuncert_r2": summarize_metric(raw_meta_r2),
            "orth_d_metamcuncert_perp_d_mcuncert_r2": summarize_metric(meta_perp_mc_r2),
            "orth_d_mcuncert_perp_d_metamcuncert_r2": summarize_metric(mc_perp_meta_r2),
            "add_d_metamcuncert_perp_to_d_mcuncert": summarize_metric(add_meta_perp_to_mc),
            "add_d_mcuncert_perp_to_d_metamcuncert": summarize_metric(add_mc_perp_to_meta),
            "combined_r2": summarize_metric(combined_r2),
        },
        "per_layer": per_layer,
        "series": {
            "cosine_mc_vs_metamc": cosine_per_layer,
            "raw_d_mcuncert_r2": raw_mc_r2,
            "raw_d_metamcuncert_r2": raw_meta_r2,
            "orth_d_metamcuncert_perp_d_mcuncert_r2": meta_perp_mc_r2,
            "orth_d_mcuncert_perp_d_metamcuncert_r2": mc_perp_meta_r2,
            "add_d_metamcuncert_perp_to_d_mcuncert": add_meta_perp_to_mc,
            "add_d_mcuncert_perp_to_d_metamcuncert": add_mc_perp_to_meta,
            "combined_r2": combined_r2,
        },
        "directions_for_logitlens": directions_for_logitlens,
    }


# =============================================================================
# PLOTTING
# =============================================================================


def plot_task_row(axs: Sequence[plt.Axes], result: Dict[str, Any]) -> None:
    task = result["task"]
    output_key = result["output_target"]
    s = result["series"]

    def _plot_series(ax: plt.Axes, series: Dict[int, float], label: str, **kwargs: Any) -> None:
        xs = sorted(series)
        ys = [series[x] for x in xs]
        ax.plot(xs, ys, label=label, **kwargs)

    # Column 1: Cosine similarity (neutral color)
    ax = axs[0]
    _plot_series(ax, s["cosine_mc_vs_metamc"], "cos(d_mc, d_metamc)", linewidth=2, color=COLOR_COSINE)
    ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(
        f"{task}: cosine geometry\nmean |cos|={result['summary']['cosine_mc_vs_metamc']['mean']:.3f}"
        if result['summary']['cosine_mc_vs_metamc']['mean'] is not None else f"{task}: cosine geometry"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="best")

    # Column 2: Raw directions (blue=mc, orange=metamc)
    ax = axs[1]
    _plot_series(ax, s["raw_d_mcuncert_r2"], "d_mcuncert", linewidth=2, color=COLOR_MC)
    _plot_series(ax, s["raw_d_metamcuncert_r2"], "d_metamcuncert", linewidth=2, color=COLOR_METAMC)
    ax.set_title(
        f"{task}: raw directions → {output_key}\n"
        f"mc mean={result['summary']['raw_d_mcuncert_r2']['mean']:.3f}, meta mean={result['summary']['raw_d_metamcuncert_r2']['mean']:.3f}"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="best")

    # Column 3: Orthogonal components (blue=mc⊥metamc, orange=metamc⊥mc)
    ax = axs[2]
    _plot_series(ax, s["orth_d_mcuncert_perp_d_metamcuncert_r2"], "d_mc ⟂ d_metamc", linewidth=2, color=COLOR_MC)
    _plot_series(ax, s["orth_d_metamcuncert_perp_d_mcuncert_r2"], "d_metamc ⟂ d_mc", linewidth=2, color=COLOR_METAMC)
    ax.set_title(
        f"{task}: orthogonal components → {output_key}\n"
        f"mc⊥meta mean={result['summary']['orth_d_mcuncert_perp_d_metamcuncert_r2']['mean']:.3f}, "
        f"meta⊥mc mean={result['summary']['orth_d_metamcuncert_perp_d_mcuncert_r2']['mean']:.3f}"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="best")

    # Column 4: Added fit (blue=unique mc contribution, orange=unique metamc contribution)
    ax = axs[3]
    _plot_series(ax, s["combined_r2"], "combined R²", linewidth=2.5, color="black")
    _plot_series(ax, s["add_d_mcuncert_perp_to_d_metamcuncert"], "ΔR² unique to d_mc", linewidth=2, color=COLOR_MC)
    _plot_series(ax, s["add_d_metamcuncert_perp_to_d_mcuncert"], "ΔR² unique to d_metamc", linewidth=2, color=COLOR_METAMC)
    ax.set_title(
        f"{task}: unique contributions\n"
        f"mc unique mean={result['summary']['add_d_mcuncert_perp_to_d_metamcuncert']['mean']:.3f}, "
        f"metamc unique mean={result['summary']['add_d_metamcuncert_perp_to_d_mcuncert']['mean']:.3f}"
    )
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² / ΔR²")
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(fontsize=8, loc="best")


def plot_combined(results: Sequence[Dict[str, Any]], output_path: Path, title_suffix: str) -> None:
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 4, figsize=(20, 4.8 * n_rows), squeeze=False)
    for row_idx, result in enumerate(results):
        plot_task_row(axes[row_idx], result)
    fig.suptitle(
        "Orthogonal uncertainty-component analysis\n"
        f"{title_suffix}",
        fontsize=16,
    )
    plt.tight_layout()
    save_figure(fig, output_path)


# =============================================================================
# REPORTING
# =============================================================================


def print_task_summary(result: Dict[str, Any]) -> None:
    task = result["task"]
    summ = result["summary"]
    print(f"{task}:")
    print(f"  Output target: {result['output_target']}")
    print(
        f"  Raw directions: d_mcuncert mean R²={summ['raw_d_mcuncert_r2']['mean']:.3f} "
        f"(best={summ['raw_d_mcuncert_r2']['best']:.3f} @ L{summ['raw_d_mcuncert_r2']['best_layer']}), "
        f"d_metamcuncert mean R²={summ['raw_d_metamcuncert_r2']['mean']:.3f} "
        f"(best={summ['raw_d_metamcuncert_r2']['best']:.3f} @ L{summ['raw_d_metamcuncert_r2']['best_layer']})"
    )
    print(
        f"  Orthogonal pieces: d_metamcuncert⟂d_mcuncert mean R²={summ['orth_d_metamcuncert_perp_d_mcuncert_r2']['mean']:.3f} "
        f"(best={summ['orth_d_metamcuncert_perp_d_mcuncert_r2']['best']:.3f} @ L{summ['orth_d_metamcuncert_perp_d_mcuncert_r2']['best_layer']}), "
        f"d_mcuncert⟂d_metamcuncert mean R²={summ['orth_d_mcuncert_perp_d_metamcuncert_r2']['mean']:.3f} "
        f"(best={summ['orth_d_mcuncert_perp_d_metamcuncert_r2']['best']:.3f} @ L{summ['orth_d_mcuncert_perp_d_metamcuncert_r2']['best_layer']})"
    )
    print(
        f"  Added fit beyond parent direction: add(d_metamc⊥d_mc | d_mc) mean ΔR²={summ['add_d_metamcuncert_perp_to_d_mcuncert']['mean']:.3f} "
        f"(best={summ['add_d_metamcuncert_perp_to_d_mcuncert']['best']:.3f} @ L{summ['add_d_metamcuncert_perp_to_d_mcuncert']['best_layer']}), "
        f"add(d_mc⊥d_metamc | d_metamc) mean ΔR²={summ['add_d_mcuncert_perp_to_d_metamcuncert']['mean']:.3f} "
        f"(best={summ['add_d_mcuncert_perp_to_d_metamcuncert']['best']:.3f} @ L{summ['add_d_mcuncert_perp_to_d_metamcuncert']['best_layer']})"
    )
    print()


def print_cross_task_summary(results: Sequence[Dict[str, Any]]) -> None:
    print("Cross-task comparison:")
    ranked_meta_extra = sorted(
        ((r['task'], r['summary']['add_d_metamcuncert_perp_to_d_mcuncert']['mean']) for r in results),
        key=lambda kv: -999.0 if kv[1] is None else kv[1],
        reverse=True,
    )
    ranked_mc_extra = sorted(
        ((r['task'], r['summary']['add_d_mcuncert_perp_to_d_metamcuncert']['mean']) for r in results),
        key=lambda kv: -999.0 if kv[1] is None else kv[1],
        reverse=True,
    )
    print("  Strongest extra signal in d_metamcuncert beyond d_mcuncert:")
    for task, val in ranked_meta_extra:
        print(f"    {task:16s} mean ΔR²={val:.3f}" if val is not None else f"    {task:16s} n/a")
    print("  Strongest extra signal in d_mcuncert beyond d_metamcuncert:")
    for task, val in ranked_mc_extra:
        print(f"    {task:16s} mean ΔR²={val:.3f}" if val is not None else f"    {task:16s} n/a")
    print()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    model_dir = get_model_dir_name(MODEL, adapter=ADAPTER, load_in_4bit=LOAD_IN_4BIT, load_in_8bit=LOAD_IN_8BIT)

    print("=" * 100)
    print("ORTHOGONAL UNCERTAINTY-COMPONENT ANALYSIS")
    print("=" * 100)
    print(f"Model dir: {model_dir}")
    print(f"Dataset:   {DATASET}")
    print(f"Metric:    {UNCERTAINTY_METRIC}")
    print(f"Method:    {METHOD}")
    print(f"Tasks:     {', '.join(META_TASKS)}")
    print()

    results = [analyze_task(task, DATASET, UNCERTAINTY_METRIC, model_dir) for task in META_TASKS]

    for result in results:
        print_task_summary(result)
    print_cross_task_summary(results)

    title_suffix = f"{DATASET} | uncertainty={UNCERTAINTY_METRIC} | method={METHOD}"
    figure_path = get_output_path(
        f"{DATASET}_meta_orthogonal_uncertainty_{UNCERTAINTY_METRIC}.png",
        model_dir=model_dir,
    )
    plot_combined(results, figure_path, title_suffix)

    summary = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            uncertainty_metric=UNCERTAINTY_METRIC,
            method=METHOD,
            tasks=list(META_TASKS),
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
        ),
        "notes": [
            "d_metamcuncert ⟂ d_mcuncert asks what the meta-task uncertainty direction contains beyond the MCQ-derived uncertainty direction.",
            "d_mcuncert ⟂ d_metamcuncert asks what the MCQ-derived uncertainty direction contains beyond the meta-task uncertainty direction.",
            "Single-direction R^2 on orthogonal components is a useful first-pass test; added-fit delta R^2 is the more direct test of extra predictive signal beyond the parent direction.",
            "This is predictive/correlational analysis, not causal analysis.",
        ],
        "results_by_task": {
            r["task"]: {k: v for k, v in r.items() if k != "directions_for_logitlens"}
            for r in results
        },
        "figures": {"combined_report": str(figure_path)},
    }
    if not SAVE_FULL_PER_LAYER_DETAILS:
        for task in list(summary["results_by_task"].keys()):
            summary["results_by_task"][task].pop("per_layer", None)

    json_path = get_output_path(
        f"{DATASET}_meta_orthogonal_uncertainty_{UNCERTAINTY_METRIC}.json",
        model_dir=model_dir,
    )
    with open(json_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"Combined report: {figure_path}")
    print(f"Summary JSON:    {json_path}")

    # Logitlens analysis
    if RUN_LOGITLENS:
        print()
        print("Running logitlens analysis...")
        lm_head_weight, norm_weight, tokenizer = load_lm_head_and_norm(MODEL)

        for result in results:
            task = result["task"]
            directions_for_logitlens = result["directions_for_logitlens"]

            # Use all layers (like analyze_directions.py)
            all_layers = sorted(directions_for_logitlens["d_mcuncert"].keys())
            if LOGITLENS_LAYERS is not None:
                layers = [l for l in LOGITLENS_LAYERS if l in all_layers]
            else:
                layers = all_layers

            # One plot per direction type (like analyze_directions.py)
            for dir_name, dir_data in directions_for_logitlens.items():
                # Sanitize direction name for filename (replace special chars)
                safe_name = dir_name.replace("⟂", "_perp_")
                logitlens_path = get_output_path(
                    f"{DATASET}_meta_logitlens_{task}_{safe_name}_{UNCERTAINTY_METRIC}.png",
                    model_dir=model_dir,
                )

                plot_logitlens_single_direction(
                    direction_by_layer=dir_data,
                    layers=layers,
                    lm_head_weight=lm_head_weight,
                    tokenizer=tokenizer,
                    output_path=logitlens_path,
                    direction_name=f"{task}/{dir_name}",
                    top_k=LOGITLENS_TOP_K,
                    norm_weight=norm_weight,
                )

                summary["figures"][f"logitlens_{task}_{safe_name}"] = str(logitlens_path)

        # Re-save JSON with logitlens paths
        with open(json_path, "w") as f:
            json.dump(json_ready(summary), f, indent=2)

    print("=" * 100)


if __name__ == "__main__":
    main()
