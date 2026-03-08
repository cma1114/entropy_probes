from __future__ import annotations

"""
Analyze which mean-diff directions uniquely predict meta-task output.

Goal
----
For one dataset/task/output target, load meta activations and candidate directions,
then ask which directions seem distinctively related to the meta output.

Candidate directions (mean_diff-focused)
---------------------------------------
- d_mc: MC uncertainty direction
- d_metamc: meta-task uncertainty direction (recomputed on meta activations)
- d_answer: MC answer direction
- d_metaanswer: meta-task answer direction
- d_confdir: confidence/output direction

What this script reports
------------------------
1. Geometry: pairwise cosine similarities between candidate directions.
2. Functional redundancy: pairwise correlations between direction projections on meta activations.
3. Marginal output connection: per-direction Pearson r / R^2 with the chosen meta output.
4. Unique contribution: full-model leave-one-out delta R^2 for each direction.

Interpretation caveat
---------------------
This is still correlational / predictive. It helps identify which directions carry
unique output-related information, but it does not by itself prove causal use.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

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
META_TASK = "delegate"  # delegate / confidence / other_confidence
UNCERTAINTY_METRIC = "logit_gap"
METHOD = "mean_diff"   # intentionally mean_diff-focused for mechanistic analysis

# Which meta-output target to analyze.
# Use "auto" to pick based on META_TASK and available activation keys.
# Common activation keys seen in this codebase: logit_margins, confidences, computed_confidence.
OUTPUT_TARGET = "auto"
DELEGATE_OUTPUT_TARGET = "logit_margins"
NON_DELEGATE_OUTPUT_PREFERENCE = ("computed_confidence", "confidences")

# Confdir filename tag. "auto" maps logit_margins -> logit_margin, otherwise uses OUTPUT_TARGET.
CONFDIR_FILE_TAG = "auto"

# Candidate directions to include. If a file is missing, the script will skip that direction.
INCLUDE_DIRECTIONS = ("d_mc", "d_metamc", "d_answer", "d_metaanswer", "d_confdir")

# If True, also save a wide CSV-like JSON block with all per-layer values.
SAVE_FULL_PER_LAYER_DETAILS = True

# =============================================================================
# DIRECTION SPECS
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


def infer_output_target(task: str, requested: str, activations: np.lib.npyio.NpzFile) -> str:
    if requested != "auto":
        if requested not in activations.files:
            raise KeyError(
                f"Requested OUTPUT_TARGET={requested!r} not found in activations file. "
                f"Available keys: {sorted(activations.files)}"
            )
        return requested

    if task == "delegate":
        if DELEGATE_OUTPUT_TARGET in activations.files:
            return DELEGATE_OUTPUT_TARGET
        raise KeyError(
            f"Delegate task expected output key {DELEGATE_OUTPUT_TARGET!r}, but activations only contain "
            f"{sorted(activations.files)}"
        )

    for key in NON_DELEGATE_OUTPUT_PREFERENCE:
        if key in activations.files:
            return key

    raise KeyError(
        "Could not infer output target for non-delegate task. "
        f"Tried {NON_DELEGATE_OUTPUT_PREFERENCE}; available keys: {sorted(activations.files)}"
    )


def infer_confdir_file_tag(output_target: str) -> str:
    if CONFDIR_FILE_TAG != "auto":
        return CONFDIR_FILE_TAG
    if output_target == "logit_margins":
        return "logit_margin"
    return output_target


def get_direction_specs(dataset: str, task: str, uncertainty_metric: str, confdir_file_tag: str) -> Dict[str, DirectionSpec]:
    """
    Edit these patterns if your upstream scripts use different filenames.

    The loader tries each file pattern in order, then each key pattern in order,
    and uses the first combination that yields at least one layer.
    """
    return {
        "d_mc": DirectionSpec(
            name="d_mc",
            description="MC uncertainty direction",
            file_patterns=[
                f"{dataset}_mc_{uncertainty_metric}_directions.npz",
            ],
            key_patterns=[
                f"{METHOD}_layer_{{layer}}",
            ],
        ),
        "d_metamc": DirectionSpec(
            name="d_metamc",
            description="Meta-task recomputed uncertainty direction",
            file_patterns=[
                f"{dataset}_meta_{task}_mcuncert_directions_final.npz",
                f"{dataset}_meta_{task}_mcuncert_directions.npz",
            ],
            key_patterns=[
                f"{METHOD}_{uncertainty_metric}_layer_{{layer}}",
            ],
        ),
        "d_answer": DirectionSpec(
            name="d_answer",
            description="MC answer direction",
            file_patterns=[
                f"{dataset}_mc_answer_directions.npz",
                f"{dataset}_mc_answer_centroid_directions.npz",
                f"{dataset}_mcq_directions_final.npz",
            ],
            key_patterns=[
                "centroid_layer_{layer}",
                f"{METHOD}_layer_{{layer}}",
            ],
        ),
        "d_metaanswer": DirectionSpec(
            name="d_metaanswer",
            description="Meta-task answer direction",
            file_patterns=[
                f"{dataset}_meta_{task}_metamcq_directions_final.npz",
                f"{dataset}_meta_{task}_metaanswer_directions_final.npz",
            ],
            key_patterns=[
                "centroid_layer_{layer}",
                f"{METHOD}_layer_{{layer}}",
            ],
        ),
        "d_confdir": DirectionSpec(
            name="d_confdir",
            description=f"Confidence/output direction ({confdir_file_tag})",
            file_patterns=[
                f"{dataset}_meta_{task}_confdir_{confdir_file_tag}_directions_final.npz",
                f"{dataset}_meta_{task}_confdir_{confdir_file_tag}_directions.npz",
            ],
            key_patterns=[
                f"{METHOD}_layer_{{layer}}",
            ],
        ),
    }


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
        if not math.isfinite(x):
            return None
        return x
    return obj


def finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for arr in arrays:
        mask &= np.isfinite(arr)
    return mask


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or y.size < 3:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    r, _ = pearsonr(x, y)
    if not math.isfinite(r):
        return None
    return float(r)


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
    """
    OLS without intercept on standardized X and y.
    Returns (R^2, betas_for_kept_columns, kept_column_indices).
    """
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


_LAYER_RE_CACHE: Dict[str, re.Pattern] = {}


def key_pattern_to_regex(key_pattern: str) -> re.Pattern:
    if key_pattern in _LAYER_RE_CACHE:
        return _LAYER_RE_CACHE[key_pattern]
    pattern = re.escape(key_pattern).replace(re.escape("{layer}"), r"(\d+)")
    regex = re.compile(rf"^{pattern}$")
    _LAYER_RE_CACHE[key_pattern] = regex
    return regex


# =============================================================================
# LOADING
# =============================================================================


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
                match = regex.match(key)
                if not match:
                    continue
                layer = int(match.group(1))
                layers[layer] = np.asarray(data[key], dtype=float)
            if layers:
                return LoadedDirection(
                    name=spec.name,
                    description=spec.description,
                    file_path=str(path),
                    key_pattern=key_pattern,
                    layers=dict(sorted(layers.items())),
                )
    return None


def load_all_directions(dataset: str, task: str, uncertainty_metric: str, confdir_file_tag: str, model_dir: str) -> Dict[str, LoadedDirection]:
    specs = get_direction_specs(dataset, task, uncertainty_metric, confdir_file_tag)
    loaded: Dict[str, LoadedDirection] = {}
    for name in INCLUDE_DIRECTIONS:
        spec = specs[name]
        direction = load_direction_from_spec(spec, model_dir)
        if direction is not None:
            loaded[name] = direction
    return loaded


def load_meta_activations(dataset: str, task: str, model_dir: str) -> np.lib.npyio.NpzFile:
    path = find_output_file(f"{dataset}_meta_{task}_activations.npz", model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Meta activations file not found: {path}")
    return np.load(path)


def available_activation_layers(activations: np.lib.npyio.NpzFile) -> Dict[int, str]:
    layers = {}
    regex = re.compile(r"^layer_(\d+)_final$")
    for key in activations.files:
        m = regex.match(key)
        if m:
            layers[int(m.group(1))] = key
    return dict(sorted(layers.items()))


# =============================================================================
# ANALYSIS
# =============================================================================


def compute_pairwise_direction_cosines(directions: Dict[str, LoadedDirection]) -> Dict[str, Dict[int, float]]:
    pairwise: Dict[str, Dict[int, float]] = {}
    for a, b in combinations(directions.keys(), 2):
        da = directions[a].layers
        db = directions[b].layers
        common = sorted(set(da.keys()) & set(db.keys()))
        if not common:
            continue
        key = f"{a}__vs__{b}"
        vals = {}
        for layer in common:
            if da[layer].shape != db[layer].shape:
                continue
            vals[layer] = cosine_similarity(da[layer], db[layer])
        if vals:
            pairwise[key] = vals
    return pairwise


def summarize_pairwise_cosines(pairwise: Dict[str, Dict[int, float]]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for key, per_layer in pairwise.items():
        layers = sorted(per_layer.keys())
        vals = np.array([per_layer[l] for l in layers], dtype=float)
        abs_vals = np.abs(vals)
        best_idx = int(np.argmax(abs_vals))
        out[key] = {
            "n_layers": int(len(layers)),
            "mean_signed": float(np.mean(vals)),
            "mean_abs": float(np.mean(abs_vals)),
            "best_abs": float(abs_vals[best_idx]),
            "best_layer": int(layers[best_idx]),
        }
    return out


def build_layer_projection_table(
    directions: Dict[str, LoadedDirection],
    activations: np.lib.npyio.NpzFile,
    output_key: str,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Returns:
      per_layer: detailed stats for each layer
      per_direction_summary: aggregated stats by direction
      projection_pair_summary: aggregated pairwise correlations between projected features
    """
    layer_keys = available_activation_layers(activations)
    y_all = np.asarray(activations[output_key], dtype=float)

    per_layer: Dict[int, Dict[str, Any]] = {}
    direction_layers: Dict[str, List[Dict[str, Any]]] = {name: [] for name in directions}
    proj_pair_layers: Dict[str, List[float]] = {}

    candidate_layers = sorted(
        set(layer_keys.keys()) & set().union(*(set(d.layers.keys()) for d in directions.values()))
    )

    for layer in candidate_layers:
        if layer not in layer_keys:
            continue
        acts = np.asarray(activations[layer_keys[layer]], dtype=float)
        if acts.ndim != 2:
            continue

        projections: Dict[str, np.ndarray] = {}
        available_names: List[str] = []
        for name, loaded in directions.items():
            vec = loaded.layers.get(layer)
            if vec is None:
                continue
            if acts.shape[1] != vec.shape[0]:
                continue
            proj = acts @ vec
            projections[name] = proj
            available_names.append(name)

        if not available_names:
            continue

        layer_record: Dict[str, Any] = {
            "layer": layer,
            "directions": {},
            "pairwise_projection_corr": {},
            "full_model": None,
        }

        # Single-direction stats
        for name in available_names:
            proj = projections[name]
            mask = finite_mask(proj, y_all)
            x = proj[mask]
            y = y_all[mask]
            r = safe_pearsonr(x, y)
            if r is None:
                continue
            r2, _, _ = regression_r2(x[:, None], y)
            record = {
                "pearson_r": float(r),
                "abs_r": float(abs(r)),
                "r2": float(r2),
                "n": int(len(y)),
            }
            layer_record["directions"][name] = record
            direction_layers[name].append({"layer": layer, **record})

        # Pairwise projected-feature correlations
        for a, b in combinations(sorted(layer_record["directions"].keys()), 2):
            mask = finite_mask(projections[a], projections[b], y_all)
            pa = projections[a][mask]
            pb = projections[b][mask]
            r_proj = safe_pearsonr(pa, pb)
            if r_proj is None:
                continue
            pair_key = f"{a}__vs__{b}"
            layer_record["pairwise_projection_corr"][pair_key] = float(r_proj)
            proj_pair_layers.setdefault(pair_key, []).append(float(abs(r_proj)))

        # Full multivariate model + leave-one-out unique contributions
        valid_for_full = sorted(layer_record["directions"].keys())
        if len(valid_for_full) >= 2:
            mask = finite_mask(y_all, *[projections[n] for n in valid_for_full])
            y = y_all[mask]
            X = np.column_stack([projections[n][mask] for n in valid_for_full])
            full_r2, beta, kept = regression_r2(X, y)
            kept_names = [valid_for_full[i] for i in kept]
            full_model = {
                "n": int(len(y)),
                "directions": kept_names,
                "r2": float(full_r2),
                "betas": {},
                "leave_one_out_delta_r2": {},
            }
            if beta is not None:
                for name, coef in zip(kept_names, beta):
                    full_model["betas"][name] = float(coef)

                for j, name in enumerate(kept_names):
                    X_loo = np.delete(X[:, kept], j, axis=1)
                    loo_r2, _, _ = regression_r2(X_loo, y)
                    delta = max(0.0, float(full_r2 - loo_r2))
                    full_model["leave_one_out_delta_r2"][name] = delta
            layer_record["full_model"] = full_model

        if layer_record["directions"]:
            per_layer[layer] = layer_record

    # Aggregate by direction
    per_direction_summary: Dict[str, Dict[str, Any]] = {}
    for name, rows in direction_layers.items():
        if not rows:
            continue
        mean_abs_r = float(np.mean([r["abs_r"] for r in rows]))
        mean_r2 = float(np.mean([r["r2"] for r in rows]))
        best_abs = max(rows, key=lambda r: r["abs_r"])
        best_r2 = max(rows, key=lambda r: r["r2"])

        deltas: List[Tuple[int, float]] = []
        betas: List[Tuple[int, float]] = []
        for layer, info in per_layer.items():
            full = info.get("full_model")
            if not full:
                continue
            if name in full.get("leave_one_out_delta_r2", {}):
                deltas.append((layer, float(full["leave_one_out_delta_r2"][name])))
            if name in full.get("betas", {}):
                betas.append((layer, float(full["betas"][name])))

        summary = {
            "n_layers": int(len(rows)),
            "mean_abs_r": mean_abs_r,
            "mean_r2": mean_r2,
            "best_abs_r": float(best_abs["abs_r"]),
            "best_abs_r_signed": float(best_abs["pearson_r"]),
            "best_abs_r_layer": int(best_abs["layer"]),
            "best_r2": float(best_r2["r2"]),
            "best_r2_layer": int(best_r2["layer"]),
        }
        if deltas:
            best_delta = max(deltas, key=lambda t: t[1])
            summary.update(
                {
                    "mean_leave_one_out_delta_r2": float(np.mean([d for _, d in deltas])),
                    "best_leave_one_out_delta_r2": float(best_delta[1]),
                    "best_leave_one_out_layer": int(best_delta[0]),
                }
            )
        else:
            summary["mean_leave_one_out_delta_r2"] = None
            summary["best_leave_one_out_delta_r2"] = None
            summary["best_leave_one_out_layer"] = None

        if betas:
            best_beta = max(betas, key=lambda t: abs(t[1]))
            summary.update(
                {
                    "mean_abs_beta": float(np.mean([abs(b) for _, b in betas])),
                    "best_beta": float(best_beta[1]),
                    "best_beta_layer": int(best_beta[0]),
                }
            )
        else:
            summary["mean_abs_beta"] = None
            summary["best_beta"] = None
            summary["best_beta_layer"] = None

        per_direction_summary[name] = summary

    projection_pair_summary: Dict[str, Dict[str, Any]] = {}
    for pair_key, vals in proj_pair_layers.items():
        projection_pair_summary[pair_key] = {
            "n_layers": int(len(vals)),
            "mean_abs_projection_corr": float(np.mean(vals)),
            "max_abs_projection_corr": float(np.max(vals)),
        }

    return per_layer, per_direction_summary, projection_pair_summary


# =============================================================================
# PLOTTING
# =============================================================================


def pairwise_matrix(direction_names: Sequence[str], pair_summary: Dict[str, Dict[str, Any]], value_key: str) -> np.ndarray:
    n = len(direction_names)
    mat = np.full((n, n), np.nan, dtype=float)
    np.fill_diagonal(mat, 1.0)
    for i, a in enumerate(direction_names):
        for j, b in enumerate(direction_names):
            if i >= j:
                continue
            key1 = f"{a}__vs__{b}"
            key2 = f"{b}__vs__{a}"
            src = pair_summary.get(key1) or pair_summary.get(key2)
            if src and value_key in src and src[value_key] is not None:
                mat[i, j] = float(src[value_key])
                mat[j, i] = float(src[value_key])
    return mat


def draw_heatmap(ax: plt.Axes, mat: np.ndarray, labels: Sequence[str], title: str, vmin: float = 0.0, vmax: float = 1.0) -> None:
    im = ax.imshow(mat, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if math.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_overview(
    direction_names: List[str],
    cosine_summary: Dict[str, Dict[str, Any]],
    proj_pair_summary: Dict[str, Dict[str, Any]],
    per_direction_summary: Dict[str, Dict[str, Any]],
    output_path: Path,
    title_suffix: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    cosine_mat = pairwise_matrix(direction_names, cosine_summary, "mean_abs")
    draw_heatmap(axes[0, 0], cosine_mat, direction_names, "Mean |cos| across common layers")

    proj_corr_mat = pairwise_matrix(direction_names, proj_pair_summary, "mean_abs_projection_corr")
    draw_heatmap(axes[0, 1], proj_corr_mat, direction_names, "Mean |corr(projections)| across layers")

    names = [n for n in direction_names if n in per_direction_summary]
    best_r2 = [per_direction_summary[n]["best_r2"] for n in names]
    axes[1, 0].bar(np.arange(len(names)), best_r2)
    axes[1, 0].set_xticks(np.arange(len(names)))
    axes[1, 0].set_xticklabels(names, rotation=30, ha="right")
    axes[1, 0].set_ylabel("Best single-direction R²")
    axes[1, 0].set_title("Best marginal output fit by direction")
    axes[1, 0].grid(True, axis="y", alpha=GRID_ALPHA)
    for i, n in enumerate(names):
        layer = per_direction_summary[n]["best_r2_layer"]
        axes[1, 0].text(i, best_r2[i], f"L{layer}", ha="center", va="bottom", fontsize=8)

    unique = [per_direction_summary[n]["mean_leave_one_out_delta_r2"] or 0.0 for n in names]
    axes[1, 1].bar(np.arange(len(names)), unique)
    axes[1, 1].set_xticks(np.arange(len(names)))
    axes[1, 1].set_xticklabels(names, rotation=30, ha="right")
    axes[1, 1].set_ylabel("Mean leave-one-out ΔR²")
    axes[1, 1].set_title("Average unique contribution in full model")
    axes[1, 1].grid(True, axis="y", alpha=GRID_ALPHA)
    for i, n in enumerate(names):
        layer = per_direction_summary[n]["best_leave_one_out_layer"]
        if layer is not None:
            axes[1, 1].text(i, unique[i], f"L{layer}", ha="center", va="bottom", fontsize=8)

    fig.suptitle(f"Meta-output direction influence overview\n{title_suffix}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_path)


def plot_per_layer(
    direction_names: List[str],
    per_layer: Dict[int, Dict[str, Any]],
    output_path: Path,
    title_suffix: str,
) -> None:
    if not per_layer:
        return

    layers = sorted(per_layer.keys())
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    for name in direction_names:
        xs, r2s = [], []
        for layer in layers:
            d = per_layer[layer]["directions"].get(name)
            if d is None:
                continue
            xs.append(layer)
            r2s.append(d["r2"])
        if xs:
            axes[0].plot(xs, r2s, label=name, linewidth=2)
    axes[0].set_ylabel("Single-direction R²")
    axes[0].set_title("Marginal output fit by layer")
    axes[0].grid(True, alpha=GRID_ALPHA)
    axes[0].legend(loc="best")

    full_x, full_r2 = [], []
    for layer in layers:
        full = per_layer[layer].get("full_model")
        if full is not None:
            full_x.append(layer)
            full_r2.append(full["r2"])
    if full_x:
        axes[1].plot(full_x, full_r2, color="black", linewidth=2.5, label="full model R²")
    for name in direction_names:
        xs, deltas = [], []
        for layer in layers:
            full = per_layer[layer].get("full_model")
            if full is None:
                continue
            delta = full["leave_one_out_delta_r2"].get(name)
            if delta is None:
                continue
            xs.append(layer)
            deltas.append(delta)
        if xs:
            axes[1].plot(xs, deltas, label=f"ΔR² drop if remove {name}", linewidth=1.7)
    axes[1].set_ylabel("R² / ΔR²")
    axes[1].set_title("Full-model fit and leave-one-out unique contributions")
    axes[1].grid(True, alpha=GRID_ALPHA)
    axes[1].legend(loc="best", fontsize=9)

    for name in direction_names:
        xs, betas = [], []
        for layer in layers:
            full = per_layer[layer].get("full_model")
            if full is None:
                continue
            beta = full["betas"].get(name)
            if beta is None:
                continue
            xs.append(layer)
            betas.append(beta)
        if xs:
            axes[2].plot(xs, betas, label=name, linewidth=2)
    axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Standardized β")
    axes[2].set_title("Full-model standardized coefficients by layer")
    axes[2].grid(True, alpha=GRID_ALPHA)
    axes[2].legend(loc="best")

    fig.suptitle(f"Meta-output direction influence by layer\n{title_suffix}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_path)



# =============================================================================
# BEHAVIORAL-CORRELATION DECOMPOSITION (focused 2-direction add-on)
# =============================================================================


def load_mcq_metric_values(dataset: str, metric: str, model_dir: str, activations: np.lib.npyio.NpzFile, output_key: str) -> np.ndarray:
    mc_results_path = find_output_file(f"{dataset}_mc_results.json", model_dir=model_dir)
    if not mc_results_path.exists():
        raise FileNotFoundError(
            f"MCQ results file not found: {mc_results_path}. Expected per-example MCQ metrics in *_mc_results.json."
        )

    with open(mc_results_path, "r") as f:
        payload = json.load(f)

    try:
        questions = payload["dataset"]["data"]
    except Exception as e:
        raise KeyError(f"Could not read dataset.data from MCQ results file: {mc_results_path}") from e

    vals = []
    for q in questions:
        try:
            vals.append(float(q[metric]))
        except Exception:
            vals.append(float("nan"))
    arr = np.asarray(vals, dtype=float)

    y_len = len(np.asarray(activations[output_key], dtype=float))
    if len(arr) == y_len:
        return arr

    if "valid_final" in activations.files:
        valid = np.asarray(activations["valid_final"]).astype(bool)
        if len(valid) == len(arr) and int(np.sum(valid)) == y_len:
            return arr[valid]

    raise ValueError(
        f"MCQ metric array length mismatch for metric={metric!r}: mc_results has {len(arr)} examples, meta output has {y_len}. Could not align via valid_final."
    )


def regression_residuals(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    if X.ndim == 1:
        X = X[:, None]
    Zx, keep_mask = standardize_columns(X)
    Zy, y_ok = standardize_vector(y)
    kept = [i for i, k in enumerate(keep_mask) if k]
    if not y_ok:
        return np.zeros_like(y, dtype=float), kept
    if Zx.shape[1] == 0 or Zx.shape[0] <= Zx.shape[1]:
        return Zy, kept
    beta, *_ = np.linalg.lstsq(Zx, Zy, rcond=None)
    pred = Zx @ beta
    return Zy - pred, kept


def analyze_behavioral_correlation_two_direction(
    directions: Dict[str, LoadedDirection],
    activations: np.lib.npyio.NpzFile,
    output_key: str,
    mcq_metric_values: np.ndarray,
    dir_a: str = "d_mc",
    dir_b: str = "d_metaanswer",
) -> Dict[str, Any]:
    if dir_a not in directions or dir_b not in directions:
        missing = [d for d in (dir_a, dir_b) if d not in directions]
        raise KeyError(f"Behavioral-correlation add-on requires directions {missing}, but they were not loaded.")

    layer_keys = available_activation_layers(activations)
    y_all = np.asarray(activations[output_key], dtype=float)
    u_all = np.asarray(mcq_metric_values, dtype=float)
    raw_mask = finite_mask(u_all, y_all)
    raw_corr = safe_pearsonr(u_all[raw_mask], y_all[raw_mask])
    if raw_corr is None:
        raise RuntimeError(f"Could not compute raw behavioral correlation for {UNCERTAINTY_METRIC} vs {output_key}.")

    common_layers = sorted(set(layer_keys.keys()) & set(directions[dir_a].layers.keys()) & set(directions[dir_b].layers.keys()))
    per_layer: Dict[int, Dict[str, Any]] = {}

    for layer in common_layers:
        acts = np.asarray(activations[layer_keys[layer]], dtype=float)
        if acts.ndim != 2:
            continue
        va = directions[dir_a].layers[layer]
        vb = directions[dir_b].layers[layer]
        if acts.shape[1] != va.shape[0] or acts.shape[1] != vb.shape[0]:
            continue

        za = acts @ va
        zb = acts @ vb
        mask = finite_mask(u_all, y_all, za, zb)
        if int(np.sum(mask)) < 10:
            continue
        u = u_all[mask]
        y = y_all[mask]
        xa = za[mask]
        xb = zb[mask]

        resid_a, kept_a = regression_residuals(xa[:, None], y)
        remaining_a = safe_pearsonr(u, resid_a)
        resid_b, kept_b = regression_residuals(xb[:, None], y)
        remaining_b = safe_pearsonr(u, resid_b)
        resid_both, kept_both = regression_residuals(np.column_stack([xa, xb]), y)
        remaining_both = safe_pearsonr(u, resid_both)
        if remaining_a is None or remaining_b is None or remaining_both is None:
            continue

        explained_a = float(raw_corr - remaining_a)
        explained_b = float(raw_corr - remaining_b)
        explained_both = float(raw_corr - remaining_both)
        unique_a = float(remaining_b - remaining_both)
        unique_b = float(remaining_a - remaining_both)

        per_layer[layer] = {
            "layer": int(layer),
            "n": int(np.sum(mask)),
            "raw_behavioral_corr": float(raw_corr),
            "single_direction": {
                dir_a: {
                    "remaining_behavioral_corr": float(remaining_a),
                    "explained_behavioral_corr": explained_a,
                    "kept_columns": kept_a,
                },
                dir_b: {
                    "remaining_behavioral_corr": float(remaining_b),
                    "explained_behavioral_corr": explained_b,
                    "kept_columns": kept_b,
                },
            },
            "two_direction_model": {
                "directions": [dir_a, dir_b],
                "remaining_behavioral_corr": float(remaining_both),
                "explained_behavioral_corr": explained_both,
                "leave_one_out_unique_explained_corr": {
                    dir_a: unique_a,
                    dir_b: unique_b,
                },
                "kept_columns": kept_both,
            },
        }

    if not per_layer:
        raise RuntimeError("No valid layers for 2-direction behavioral-correlation analysis.")

    def summarize_single(name: str) -> Dict[str, Any]:
        rows = [(layer, info["single_direction"][name]["explained_behavioral_corr"], info["single_direction"][name]["remaining_behavioral_corr"]) for layer, info in per_layer.items()]
        best = max(rows, key=lambda t: t[1])
        return {
            "mean_explained_behavioral_corr": float(np.mean([r[1] for r in rows])),
            "best_explained_behavioral_corr": float(best[1]),
            "best_explained_layer": int(best[0]),
            "mean_remaining_behavioral_corr": float(np.mean([r[2] for r in rows])),
        }

    full_rows = [(layer, info["two_direction_model"]["explained_behavioral_corr"], info["two_direction_model"]["remaining_behavioral_corr"]) for layer, info in per_layer.items()]
    best_full = max(full_rows, key=lambda t: t[1])

    uniq_summary = {}
    for name in (dir_a, dir_b):
        vals = [(layer, info["two_direction_model"]["leave_one_out_unique_explained_corr"][name]) for layer, info in per_layer.items()]
        best = max(vals, key=lambda t: t[1])
        uniq_summary[name] = {
            "mean_unique_explained_behavioral_corr": float(np.mean([v for _, v in vals])),
            "best_unique_explained_behavioral_corr": float(best[1]),
            "best_unique_layer": int(best[0]),
        }

    return {
        "mcq_metric": UNCERTAINTY_METRIC,
        "output_key": output_key,
        "raw_behavioral_corr": float(raw_corr),
        "directions": [dir_a, dir_b],
        "single_direction_summary": {
            dir_a: summarize_single(dir_a),
            dir_b: summarize_single(dir_b),
        },
        "two_direction_summary": {
            "mean_explained_behavioral_corr": float(np.mean([r[1] for r in full_rows])),
            "best_explained_behavioral_corr": float(best_full[1]),
            "best_explained_layer": int(best_full[0]),
            "mean_remaining_behavioral_corr": float(np.mean([r[2] for r in full_rows])),
            "unique_summary": uniq_summary,
        },
        "per_layer": per_layer,
    }


def plot_behavioral_correlation_two_direction(analysis: Dict[str, Any], output_path: Path, title_suffix: str) -> None:
    dir_a, dir_b = analysis["directions"]
    raw_corr = analysis["raw_behavioral_corr"]
    layers = sorted(int(l) for l in analysis["per_layer"].keys())
    xs = np.array(layers, dtype=int)

    exp_a = np.array([analysis["per_layer"][l]["single_direction"][dir_a]["explained_behavioral_corr"] for l in layers], dtype=float)
    exp_b = np.array([analysis["per_layer"][l]["single_direction"][dir_b]["explained_behavioral_corr"] for l in layers], dtype=float)
    rem_a = np.array([analysis["per_layer"][l]["single_direction"][dir_a]["remaining_behavioral_corr"] for l in layers], dtype=float)
    rem_b = np.array([analysis["per_layer"][l]["single_direction"][dir_b]["remaining_behavioral_corr"] for l in layers], dtype=float)
    exp_both = np.array([analysis["per_layer"][l]["two_direction_model"]["explained_behavioral_corr"] for l in layers], dtype=float)
    rem_both = np.array([analysis["per_layer"][l]["two_direction_model"]["remaining_behavioral_corr"] for l in layers], dtype=float)
    uniq_a = np.array([analysis["per_layer"][l]["two_direction_model"]["leave_one_out_unique_explained_corr"][dir_a] for l in layers], dtype=float)
    uniq_b = np.array([analysis["per_layer"][l]["two_direction_model"]["leave_one_out_unique_explained_corr"][dir_b] for l in layers], dtype=float)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(xs, exp_a, linewidth=2.2, label=f"{dir_a} alone")
    axes[0].plot(xs, exp_b, linewidth=2.2, label=f"{dir_b} alone")
    axes[0].plot(xs, exp_both, color="black", linewidth=2.5, label=f"{dir_a} + {dir_b}")
    axes[0].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Explained behavioral corr")
    axes[0].set_title("How much corr(MCQ uncertainty, meta output) is explained")
    axes[0].grid(True, alpha=GRID_ALPHA)
    axes[0].legend(loc="best")

    axes[1].axhline(raw_corr, color="gray", linestyle="--", linewidth=2, label=f"raw corr = {raw_corr:.3f}")
    axes[1].plot(xs, rem_a, linewidth=2.2, label=f"remaining after {dir_a}")
    axes[1].plot(xs, rem_b, linewidth=2.2, label=f"remaining after {dir_b}")
    axes[1].plot(xs, rem_both, color="black", linewidth=2.5, label=f"remaining after {dir_a} + {dir_b}")
    axes[1].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Remaining behavioral corr")
    axes[1].set_title("Residual behavioral correlation after residualizing meta output")
    axes[1].grid(True, alpha=GRID_ALPHA)
    axes[1].legend(loc="best")

    axes[2].plot(xs, exp_both, color="black", linewidth=2.5, label=f"full explained ({dir_a} + {dir_b})")
    axes[2].plot(xs, uniq_a, linewidth=2.2, label=f"unique explained by {dir_a}")
    axes[2].plot(xs, uniq_b, linewidth=2.2, label=f"unique explained by {dir_b}")
    axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Explained corr")
    axes[2].set_title("2-direction model: total vs unique explained behavioral correlation")
    axes[2].grid(True, alpha=GRID_ALPHA)
    axes[2].legend(loc="best")

    fig.suptitle(f"Behavioral correlation decomposition ({dir_a} + {dir_b})\n{title_suffix}", fontsize=14)
    plt.tight_layout()
    save_figure(fig, output_path)


def print_behavioral_correlation_two_direction_report(analysis: Dict[str, Any]) -> None:
    dir_a, dir_b = analysis["directions"]
    sa = analysis["single_direction_summary"][dir_a]
    sb = analysis["single_direction_summary"][dir_b]
    full = analysis["two_direction_summary"]
    ua = full["unique_summary"][dir_a]
    ub = full["unique_summary"][dir_b]

    print("Behavioral-correlation decomposition (2-direction add-on):")
    print(f"  Raw corr({analysis['mcq_metric']}, {analysis['output_key']}) = {analysis['raw_behavioral_corr']:.3f}")
    print(f"  {dir_a} alone: mean explained corr = {sa['mean_explained_behavioral_corr']:.3f} | best = {sa['best_explained_behavioral_corr']:.3f} @ L{sa['best_explained_layer']}")
    print(f"  {dir_b} alone: mean explained corr = {sb['mean_explained_behavioral_corr']:.3f} | best = {sb['best_explained_behavioral_corr']:.3f} @ L{sb['best_explained_layer']}")
    print(f"  {dir_a} + {dir_b}: mean explained corr = {full['mean_explained_behavioral_corr']:.3f} | best = {full['best_explained_behavioral_corr']:.3f} @ L{full['best_explained_layer']}")
    print(f"  Unique in 2-direction model: {dir_a} = {ua['mean_unique_explained_behavioral_corr']:.3f} mean | {dir_b} = {ub['mean_unique_explained_behavioral_corr']:.3f} mean")
    print()

# =============================================================================
# REPORTING
# =============================================================================


def print_report(
    directions: Dict[str, LoadedDirection],
    output_key: str,
    cosine_summary: Dict[str, Dict[str, Any]],
    proj_pair_summary: Dict[str, Dict[str, Any]],
    per_direction_summary: Dict[str, Dict[str, Any]],
) -> None:
    print("Loaded directions:")
    for name, info in directions.items():
        print(f"  {name:12s} {len(info.layers):3d} layers  |  {Path(info.file_path).name}  |  {info.key_pattern}")
    print()
    print(f"Meta output target: {output_key}")
    print()

    if cosine_summary:
        print("Pairwise direction geometry (mean |cos|):")
        for pair_key, stats in sorted(cosine_summary.items(), key=lambda kv: kv[1]["mean_abs"], reverse=True):
            print(
                f"  {pair_key:28s}  mean|cos|={stats['mean_abs']:.3f}  "
                f"best={stats['best_abs']:.3f} @ L{stats['best_layer']}"
            )
        print()

    if proj_pair_summary:
        print("Pairwise projection redundancy on meta activations (mean |corr|):")
        for pair_key, stats in sorted(proj_pair_summary.items(), key=lambda kv: kv[1]["mean_abs_projection_corr"], reverse=True):
            print(
                f"  {pair_key:28s}  mean|corr|={stats['mean_abs_projection_corr']:.3f}  "
                f"max={stats['max_abs_projection_corr']:.3f}"
            )
        print()

    if per_direction_summary:
        print("Direction summaries:")
        for name, stats in sorted(
            per_direction_summary.items(),
            key=lambda kv: ((kv[1]["mean_leave_one_out_delta_r2"] or -1.0), kv[1]["best_r2"]),
            reverse=True,
        ):
            print(f"  {name}:")
            print(
                f"    Best single R²: {stats['best_r2']:.3f} @ L{stats['best_r2_layer']} | "
                f"Best |r|: {stats['best_abs_r']:.3f} @ L{stats['best_abs_r_layer']}"
            )
            print(
                f"    Mean single R²: {stats['mean_r2']:.3f} | Mean |r|: {stats['mean_abs_r']:.3f}"
            )
            if stats["mean_leave_one_out_delta_r2"] is not None:
                print(
                    f"    Mean leave-one-out ΔR²: {stats['mean_leave_one_out_delta_r2']:.3f} | "
                    f"Best ΔR²: {stats['best_leave_one_out_delta_r2']:.3f} @ L{stats['best_leave_one_out_layer']}"
                )
            if stats["mean_abs_beta"] is not None:
                print(
                    f"    Mean |β| in full model: {stats['mean_abs_beta']:.3f} | "
                    f"Best β: {stats['best_beta']:.3f} @ L{stats['best_beta_layer']}"
                )
        print()

        ranked_unique = [
            (name, stats["mean_leave_one_out_delta_r2"])
            for name, stats in per_direction_summary.items()
            if stats["mean_leave_one_out_delta_r2"] is not None
        ]
        if ranked_unique:
            ranked_unique.sort(key=lambda x: x[1], reverse=True)
            print("Most uniquely informative directions (by mean leave-one-out ΔR²):")
            for name, val in ranked_unique:
                print(f"  {name:12s} {val:.3f}")
            print()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    activations = load_meta_activations(DATASET, META_TASK, model_dir)
    output_key = infer_output_target(META_TASK, OUTPUT_TARGET, activations)
    confdir_file_tag = infer_confdir_file_tag(output_key)

    directions = load_all_directions(DATASET, META_TASK, UNCERTAINTY_METRIC, confdir_file_tag, model_dir)
    if len(directions) < 2:
        raise RuntimeError(
            "Need at least two candidate directions to compare. "
            f"Loaded: {list(directions.keys())}"
        )

    pairwise_cosines = compute_pairwise_direction_cosines(directions)
    cosine_summary = summarize_pairwise_cosines(pairwise_cosines)
    per_layer, per_direction_summary, proj_pair_summary = build_layer_projection_table(directions, activations, output_key)

    mcq_metric_values = load_mcq_metric_values(DATASET, UNCERTAINTY_METRIC, model_dir, activations, output_key)
    behavioral_two_direction = analyze_behavioral_correlation_two_direction(
        directions, activations, output_key, mcq_metric_values, dir_a="d_mc", dir_b="d_metaanswer"
    )

    print("=" * 90)
    print("MEAN-DIFF META-OUTPUT DIRECTION INFLUENCE ANALYSIS")
    print("=" * 90)
    print(f"Model dir: {model_dir}")
    print(f"Dataset:   {DATASET}")
    print(f"Task:      {META_TASK}")
    print(f"Metric:    {UNCERTAINTY_METRIC}")
    print(f"Method:    {METHOD}")
    print()
    print_report(directions, output_key, cosine_summary, proj_pair_summary, per_direction_summary)
    print(f"MCQ uncertainty source: {DATASET}_mc_results.json :: dataset.data[*]['{UNCERTAINTY_METRIC}']")
    print_behavioral_correlation_two_direction_report(behavioral_two_direction)

    direction_names = [name for name in INCLUDE_DIRECTIONS if name in directions]
    title_suffix = f"{DATASET} | {META_TASK} | output={output_key} | method={METHOD}"

    overview_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_influence_overview_{output_key}.png",
        model_dir=model_dir,
    )
    plot_overview(direction_names, cosine_summary, proj_pair_summary, per_direction_summary, overview_path, title_suffix)

    per_layer_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_influence_per_layer_{output_key}.png",
        model_dir=model_dir,
    )
    plot_per_layer(direction_names, per_layer, per_layer_path, title_suffix)

    behavioral_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_behavioral_correlation_two_direction_{output_key}_{UNCERTAINTY_METRIC}.png",
        model_dir=model_dir,
    )
    plot_behavioral_correlation_two_direction(behavioral_two_direction, behavioral_path, title_suffix)

    summary = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            meta_task=META_TASK,
            metric=UNCERTAINTY_METRIC,
            method=METHOD,
            output_target=output_key,
            confdir_file_tag=confdir_file_tag,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            include_directions=list(INCLUDE_DIRECTIONS),
        ),
        "notes": [
            "This analysis is predictive / correlational, not causal.",
            "Mean leave-one-out delta R^2 is the most useful summary of distinct predictive contribution in the full linear model.",
            "Pairwise cosine addresses residual-space geometry; pairwise projection correlation addresses functional redundancy on the chosen meta activations.",
        ],
        "loaded_directions": {
            name: {
                "description": info.description,
                "file_path": info.file_path,
                "key_pattern": info.key_pattern,
                "n_layers": len(info.layers),
            }
            for name, info in directions.items()
        },
        "pairwise_direction_cosines": {
            key: {
                "summary": cosine_summary[key],
                "per_layer": {str(layer): float(val) for layer, val in per_layer_vals.items()},
            }
            for key, per_layer_vals in pairwise_cosines.items()
        },
        "pairwise_projection_redundancy": proj_pair_summary,
        "per_direction_summary": per_direction_summary,
        "behavioral_correlation_two_direction": behavioral_two_direction,
        "figures": {
            "overview": str(overview_path),
            "per_layer": str(per_layer_path),
            "behavioral_correlation_two_direction": str(behavioral_path),
        },
    }
    if SAVE_FULL_PER_LAYER_DETAILS:
        summary["per_layer"] = per_layer

    json_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_influence_summary_{output_key}.json",
        model_dir=model_dir,
    )
    with open(json_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)

    print(f"Overview figure: {overview_path}")
    print(f"Per-layer figure: {per_layer_path}")
    print(f"Behavioral correlation figure: {behavioral_path}")
    print(f"Summary JSON: {json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
