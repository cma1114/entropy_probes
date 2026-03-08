from __future__ import annotations

"""
Layerwise mean-diff direction analysis with one focused six-panel figure:
1) all relevant pairwise cosines across layers
2) single-direction fit to the MCQ uncertainty target
3) single-direction fit to the MCQ answer target (when available)
4) single-direction fit to the meta output target (all directions)
5) full-model leave-one-out unique contribution to the meta output target (all directions)
6) 2-direction model (d_mcuncert + d_metamcanswer) fit and unique contributions to meta output

Plus two behavioral correlation figures that decompose how much of the raw
correlation between MCQ uncertainty and meta output is explained by direction projections:
- 2-direction figure: focused analysis of d_mcuncert and d_metamcanswer
- all-directions figure: analysis using all loaded directions

Notes:
- d_metaconfdir is excluded from the meta-output regression figures because it is target-defined.
- Fig 2 uses the MCQ uncertainty target from the activations file if present.
  If it is not present, it falls back to the old precomputed transfer/mcuncert JSONs,
  in which case only d_mcuncert and d_metamcuncert are available for that figure.
- Fig 3 uses a numeric MCQ-answer target from the activations file if present.
  If not found, the figure is saved with a clear "not available" message.
"""

import json
import math
import re
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
META_TASK = "delegate"
UNCERTAINTY_METRIC = "logit_gap"
METHOD = "mean_diff"

OUTPUT_TARGET = "auto"
DELEGATE_OUTPUT_TARGET = "logit_margins"
NON_DELEGATE_OUTPUT_PREFERENCE = ("computed_confidence", "confidences")
CONFDIR_FILE_TAG = "auto"
MCQ_UNCERTAINTY_TARGET = "auto"
MCQ_ANSWER_TARGET = "auto"

INCLUDE_DIRECTIONS = (
    "d_mcuncert",
    "d_metamcuncert",
    "d_mcanswer",
    "d_metamcanswer",
    "d_metaconfdir",
)

EXCLUDE_FROM_META_OUTPUT_ANALYSIS = {"d_metaconfdir"}

# Directions to use absolute value projection (captures "has any answer" vs signed "which answer")
USE_ABSOLUTE_VALUE_PROJECTION = {"d_mcanswer", "d_metamcanswer"}

COSINE_PAIR_ORDER = [
    ("d_mcuncert", "d_metamcuncert"),
    ("d_mcuncert", "d_metaconfdir"),
    ("d_mcuncert", "d_mcanswer"),
    ("d_metamcuncert", "d_metaconfdir"),
    ("d_metamcuncert", "d_metamcanswer"),
    ("d_mcanswer", "d_metamcanswer"),
    ("d_mcanswer", "d_metaconfdir"),
    ("d_metamcanswer", "d_metaconfdir"),
]

COSINE_CI_BOOTSTRAPS = 100
COSINE_CI_LEVEL = 95.0
COSINE_CI_SEED = 0

SAVE_FULL_PER_LAYER_DETAILS = True

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


def bootstrap_cosine_ci(a: np.ndarray, b: np.ndarray, n_boot: int = COSINE_CI_BOOTSTRAPS, ci_level: float = COSINE_CI_LEVEL, seed: int = COSINE_CI_SEED) -> Tuple[float, float]:
    """Approximate per-layer cosine CI by bootstrap-resampling vector coordinates."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    d = len(a)
    if d == 0 or len(b) != d:
        return (float('nan'), float('nan'))
    if d == 1:
        c = cosine_similarity(a, b)
        return (c, c)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, d, size=(n_boot, d), endpoint=False)
    a_bs = a[idx]
    b_bs = b[idx]
    numer = np.sum(a_bs * b_bs, axis=1)
    denom = np.linalg.norm(a_bs, axis=1) * np.linalg.norm(b_bs, axis=1)
    vals = np.divide(numer, denom, out=np.zeros_like(numer, dtype=float), where=denom > 0)
    alpha = (100.0 - ci_level) / 2.0
    lo, hi = np.percentile(vals, [alpha, 100.0 - alpha])
    return float(lo), float(hi)


def safe_int_keys(d: Dict[Any, Any]) -> Dict[int, Any]:
    out: Dict[int, Any] = {}
    for k, v in d.items():
        try:
            out[int(k)] = v
        except Exception:
            continue
    return dict(sorted(out.items()))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def coerce_numeric_array(arr: np.ndarray) -> Optional[np.ndarray]:
    try:
        out = np.asarray(arr, dtype=float)
    except Exception:
        return None
    if out.ndim != 1:
        return None
    if not np.any(np.isfinite(out)):
        return None
    return out


def infer_output_target(task: str, requested: str, activations: np.lib.npyio.NpzFile) -> str:
    if requested != "auto":
        if requested not in activations.files:
            raise KeyError(f"Requested OUTPUT_TARGET={requested!r} not found. Available: {sorted(activations.files)}")
        return requested
    if task == "delegate":
        if DELEGATE_OUTPUT_TARGET in activations.files:
            return DELEGATE_OUTPUT_TARGET
        raise KeyError(f"Delegate task expected {DELEGATE_OUTPUT_TARGET!r}; available: {sorted(activations.files)}")
    for key in NON_DELEGATE_OUTPUT_PREFERENCE:
        if key in activations.files:
            return key
    raise KeyError(f"Could not infer meta output key. Available: {sorted(activations.files)}")


def infer_confdir_file_tag(output_target: str) -> str:
    if CONFDIR_FILE_TAG != "auto":
        return CONFDIR_FILE_TAG
    return "logit_margin" if output_target == "logit_margins" else output_target


def infer_mcq_uncertainty_key(activations: np.lib.npyio.NpzFile, metric: str, requested: str = "auto") -> Optional[str]:
    if requested != "auto":
        return requested if requested in activations.files else None
    candidates = [metric, f"{metric}s", f"mc_{metric}", f"mc_{metric}s"]
    if metric == "logit_gap":
        candidates += ["logit_gap", "logit_gaps", "mc_logit_gap", "mc_logit_gaps"]
    if metric == "entropy":
        candidates += ["entropy", "entropies", "mc_entropy", "mc_entropies"]
    for key in candidates:
        if key in activations.files and coerce_numeric_array(activations[key]) is not None:
            return key
    return None


def infer_mcq_answer_key(activations: np.lib.npyio.NpzFile, requested: str = "auto") -> Optional[str]:
    if requested != "auto":
        return requested if requested in activations.files and coerce_numeric_array(activations[requested]) is not None else None
    candidates = [
        "mc_answers",
        "mc_answer",
        "answers",
        "answer",
        "answer_idx",
        "mc_answer_idx",
        "chosen_answer",
        "choice_idx",
        "mc_choice_idx",
        "gold_answer_idx",
        "labels",
        "targets",
    ]
    for key in candidates:
        if key in activations.files and coerce_numeric_array(activations[key]) is not None:
            return key
    return None


def get_direction_specs(dataset: str, task: str, uncertainty_metric: str, confdir_file_tag: str) -> Dict[str, DirectionSpec]:
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
        "d_mcanswer": DirectionSpec(
            name="d_mcanswer",
            description="MC answer direction",
            file_patterns=[
                f"{dataset}_mc_answer_directions.npz",
                f"{dataset}_mc_answer_centroid_directions.npz",
                f"{dataset}_mcq_directions_final.npz",
            ],
            key_patterns=["centroid_layer_{layer}", f"{METHOD}_layer_{{layer}}"],
        ),
        "d_metamcanswer": DirectionSpec(
            name="d_metamcanswer",
            description="Meta-task answer direction",
            file_patterns=[
                f"{dataset}_meta_{task}_metamcq_directions_final.npz",
                f"{dataset}_meta_{task}_metaanswer_directions_final.npz",
            ],
            key_patterns=["centroid_layer_{layer}", f"{METHOD}_layer_{{layer}}"],
        ),
        "d_metaconfdir": DirectionSpec(
            name="d_metaconfdir",
            description=f"Confidence/output direction ({confdir_file_tag})",
            file_patterns=[
                # delegate task uses target tag (logit_margin, etc.)
                f"{dataset}_meta_{task}_confdir_{confdir_file_tag}_directions_final.npz",
                f"{dataset}_meta_{task}_confdir_{confdir_file_tag}_directions.npz",
                # confidence/other_confidence tasks: no target tag, but may have position suffix
                f"{dataset}_meta_{task}_confdir_directions_final.npz",
                f"{dataset}_meta_{task}_confdir_directions.npz",
            ],
            key_patterns=[f"{METHOD}_layer_{{layer}}", "mean_diff_layer_{layer}"],
        ),
    }


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
                m = regex.match(key)
                if m:
                    layers[int(m.group(1))] = np.asarray(data[key], dtype=float)
            if layers:
                return LoadedDirection(spec.name, spec.description, str(path), key_pattern, dict(sorted(layers.items())))
    return None


def load_all_directions(dataset: str, task: str, uncertainty_metric: str, confdir_file_tag: str, model_dir: str) -> Dict[str, LoadedDirection]:
    specs = get_direction_specs(dataset, task, uncertainty_metric, confdir_file_tag)
    loaded: Dict[str, LoadedDirection] = {}
    for name in INCLUDE_DIRECTIONS:
        d = load_direction_from_spec(specs[name], model_dir)
        if d is not None:
            loaded[name] = d
    return loaded


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


def load_transfer_results(dataset: str, task: str, model_dir: str) -> Optional[Dict[str, Any]]:
    return load_json(find_output_file(f"{dataset}_meta_{task}_transfer_results_final.json", model_dir=model_dir))


def load_mcuncert_results(dataset: str, task: str, model_dir: str) -> Optional[Dict[str, Any]]:
    return load_json(find_output_file(f"{dataset}_meta_{task}_mcuncert_results_final.json", model_dir=model_dir))


# =============================================================================
# ANALYSIS
# =============================================================================


def compute_pairwise_direction_cosines(directions: Dict[str, LoadedDirection]) -> Dict[str, Dict[int, float]]:
    pairwise: Dict[str, Dict[int, float]] = {}
    for a, b in combinations(directions.keys(), 2):
        da = directions[a].layers
        db = directions[b].layers
        common = sorted(set(da.keys()) & set(db.keys()))
        vals: Dict[int, float] = {}
        for layer in common:
            if da[layer].shape == db[layer].shape:
                vals[layer] = cosine_similarity(da[layer], db[layer])
        if vals:
            pairwise[f"{a}__vs__{b}"] = vals
    return pairwise


def compute_pairwise_cosine_cis(directions: Dict[str, LoadedDirection]) -> Dict[str, Dict[int, Tuple[float, float]]]:
    pairwise_cis: Dict[str, Dict[int, Tuple[float, float]]] = {}
    for a, b in combinations(directions.keys(), 2):
        da = directions[a].layers
        db = directions[b].layers
        common = sorted(set(da.keys()) & set(db.keys()))
        vals: Dict[int, Tuple[float, float]] = {}
        for layer in common:
            if da[layer].shape == db[layer].shape:
                vals[layer] = bootstrap_cosine_ci(da[layer], db[layer], seed=COSINE_CI_SEED + int(layer))
        if vals:
            pairwise_cis[f"{a}__vs__{b}"] = vals
    return pairwise_cis


def summarize_pairwise_cosines(pairwise: Dict[str, Dict[int, float]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for key, per_layer in pairwise.items():
        layers = sorted(per_layer.keys())
        vals = np.array([per_layer[l] for l in layers], dtype=float)
        abs_vals = np.abs(vals)
        best_idx = int(np.argmax(abs_vals))
        out[key] = {
            "n_layers": len(layers),
            "mean_signed": float(np.mean(vals)),
            "mean_abs": float(np.mean(abs_vals)),
            "best_abs": float(abs_vals[best_idx]),
            "best_layer": int(layers[best_idx]),
        }
    return out


def build_projection_store(
    directions: Dict[str, LoadedDirection],
    activations: np.lib.npyio.NpzFile,
    use_abs: Optional[set] = None,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Build projection store, optionally using absolute value for specified directions."""
    layer_keys = available_activation_layers(activations)
    store: Dict[int, Dict[str, np.ndarray]] = {}
    if not directions:
        return store
    use_abs = use_abs or set()
    common_layers = set(layer_keys.keys())
    for d in directions.values():
        common_layers |= set(d.layers.keys())
    for layer in sorted(common_layers):
        act_key = layer_keys.get(layer)
        if act_key is None:
            continue
        acts = np.asarray(activations[act_key], dtype=float)
        if acts.ndim != 2:
            continue
        layer_store: Dict[str, np.ndarray] = {}
        for name, d in directions.items():
            vec = d.layers.get(layer)
            if vec is not None and acts.shape[1] == vec.shape[0]:
                proj = acts @ vec
                if name in use_abs:
                    proj = np.abs(proj)
                layer_store[name] = proj
        if layer_store:
            store[layer] = layer_store
    return store


def analyze_continuous_target(y_all: np.ndarray, projection_store: Dict[int, Dict[str, np.ndarray]], direction_names: Sequence[str]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    per_layer: Dict[int, Dict[str, Any]] = {}
    rows: Dict[str, List[Dict[str, Any]]] = {n: [] for n in direction_names}

    for layer in sorted(projection_store.keys()):
        projections = projection_store[layer]
        names = [n for n in direction_names if n in projections]
        if not names:
            continue
        rec: Dict[str, Any] = {"single": {}, "full_model": None}
        for name in names:
            mask = finite_mask(projections[name], y_all)
            x = np.asarray(projections[name][mask], dtype=float)
            y = np.asarray(y_all[mask], dtype=float)
            if len(y) < 5:
                continue
            r = pearsonr(x, y)[0] if np.std(x) > 0 and np.std(y) > 0 else np.nan
            r2, _, _ = regression_r2(x[:, None], y)
            if math.isfinite(r):
                rec["single"][name] = {"r": float(r), "r2": float(r2), "n": int(len(y))}
                rows[name].append({"layer": layer, "r": float(r), "r2": float(r2)})
        full_names = sorted(rec["single"].keys())
        if len(full_names) >= 2:
            mask = finite_mask(y_all, *[projections[n] for n in full_names])
            y = y_all[mask]
            X = np.column_stack([projections[n][mask] for n in full_names])
            full_r2, beta, kept = regression_r2(X, y)
            kept_names = [full_names[i] for i in kept]
            full = {"r2": float(full_r2), "betas": {}, "leave_one_out_delta_r2": {}, "directions": kept_names}
            if beta is not None:
                for name, coef in zip(kept_names, beta):
                    full["betas"][name] = float(coef)
                Xk = X[:, kept]
                for j, name in enumerate(kept_names):
                    if Xk.shape[1] <= 1:
                        loo_r2 = 0.0
                    else:
                        loo_r2, _, _ = regression_r2(np.delete(Xk, j, axis=1), y)
                    full["leave_one_out_delta_r2"][name] = max(0.0, float(full_r2 - loo_r2))
            rec["full_model"] = full
        if rec["single"]:
            per_layer[layer] = rec

    summary: Dict[str, Dict[str, Any]] = {}
    for name, name_rows in rows.items():
        if not name_rows:
            continue
        best_r2 = max(name_rows, key=lambda r: r["r2"])
        deltas: List[Tuple[int, float]] = []
        betas: List[Tuple[int, float]] = []
        for layer, info in per_layer.items():
            full = info.get("full_model")
            if not full:
                continue
            if name in full["leave_one_out_delta_r2"]:
                deltas.append((layer, float(full["leave_one_out_delta_r2"][name])))
            if name in full["betas"]:
                betas.append((layer, float(full["betas"][name])))
        best_delta = max(deltas, key=lambda t: t[1]) if deltas else None
        summary[name] = {
            "n_layers": len(name_rows),
            "mean_abs_r": float(np.mean([abs(r["r"]) for r in name_rows])),
            "mean_r2": float(np.mean([r["r2"] for r in name_rows])),
            "best_r2": float(best_r2["r2"]),
            "best_r2_layer": int(best_r2["layer"]),
            "mean_leave_one_out_delta_r2": None if not deltas else float(np.mean([d for _, d in deltas])),
            "best_leave_one_out_delta_r2": None if best_delta is None else float(best_delta[1]),
            "best_leave_one_out_layer": None if best_delta is None else int(best_delta[0]),
            "mean_abs_beta": None if not betas else float(np.mean([abs(b) for _, b in betas])),
        }
    return per_layer, summary


def extract_transfer_r2_per_layer(transfer_results: Optional[Dict[str, Any]], metric: str) -> Dict[int, float]:
    if not transfer_results:
        return {}
    transfer_key = "transfer" if METHOD == "probe" else "mean_diff_transfer"
    per_layer = transfer_results.get(transfer_key, {}).get(metric, {}).get("per_layer", {})
    out: Dict[int, float] = {}
    for layer, layer_data in safe_int_keys(per_layer).items():
        if isinstance(layer_data, dict) and layer_data.get("centered_r2") is not None:
            try:
                value = float(layer_data["centered_r2"])
            except Exception:
                continue
            if math.isfinite(value):
                out[layer] = value
    return dict(sorted(out.items()))


def extract_mcuncert_r2_per_layer(mcuncert_results: Optional[Dict[str, Any]], metric: str) -> Dict[int, float]:
    if not mcuncert_results:
        return {}
    probe_results = mcuncert_results.get("metrics", {}).get(metric, {}).get("results", {}).get(METHOD, {})
    out: Dict[int, float] = {}
    for layer, layer_data in safe_int_keys(probe_results).items():
        if isinstance(layer_data, dict) and layer_data.get("test_r2") is not None:
            try:
                value = float(layer_data["test_r2"])
            except Exception:
                continue
            if math.isfinite(value):
                out[layer] = value
    return dict(sorted(out.items()))


def analyze_two_direction_meta_output(
    y_all: np.ndarray,
    projection_store: Dict[int, Dict[str, np.ndarray]],
    dir_a: str,
    dir_b: str,
) -> Dict[int, Dict[str, Any]]:
    """Analyze meta output fit using exactly two directions."""
    per_layer: Dict[int, Dict[str, Any]] = {}

    for layer in sorted(projection_store.keys()):
        projections = projection_store[layer]
        if dir_a not in projections or dir_b not in projections:
            continue

        pa = projections[dir_a]
        pb = projections[dir_b]
        mask = finite_mask(pa, pb, y_all)
        if mask.sum() < 10:
            continue

        xa = pa[mask]
        xb = pb[mask]
        y = y_all[mask]

        # Single-direction R²
        r2_a, _, _ = regression_r2(xa[:, None], y)
        r2_b, _, _ = regression_r2(xb[:, None], y)

        # 2-direction model
        X = np.column_stack([xa, xb])
        r2_both, beta, kept = regression_r2(X, y)

        # Leave-one-out unique contributions
        r2_without_a, _, _ = regression_r2(xb[:, None], y)
        r2_without_b, _, _ = regression_r2(xa[:, None], y)
        unique_a = max(0.0, r2_both - r2_without_a)
        unique_b = max(0.0, r2_both - r2_without_b)

        per_layer[layer] = {
            "n": int(mask.sum()),
            "single": {dir_a: r2_a, dir_b: r2_b},
            "combined_r2": r2_both,
            "unique": {dir_a: unique_a, dir_b: unique_b},
            "betas": {dir_a: float(beta[0]), dir_b: float(beta[1])} if beta is not None and len(beta) == 2 else {},
        }

    return per_layer


def build_mcq_uncertainty_fallback(transfer_results: Optional[Dict[str, Any]], mcuncert_results: Optional[Dict[str, Any]]) -> Tuple[Dict[int, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    transfer_r2 = extract_transfer_r2_per_layer(transfer_results, UNCERTAINTY_METRIC)
    recomputed_r2 = extract_mcuncert_r2_per_layer(mcuncert_results, UNCERTAINTY_METRIC)
    per_layer: Dict[int, Dict[str, Any]] = {}
    for layer in sorted(set(transfer_r2.keys()) | set(recomputed_r2.keys())):
        rec = {"single": {}, "full_model": None}
        if layer in transfer_r2:
            rec["single"]["d_mcuncert"] = {"r": None, "r2": float(transfer_r2[layer]), "n": None}
        if layer in recomputed_r2:
            rec["single"]["d_metamcuncert"] = {"r": None, "r2": float(recomputed_r2[layer]), "n": None}
        if rec["single"]:
            per_layer[layer] = rec
    summary: Dict[str, Dict[str, Any]] = {}
    for name, vals in (("d_mcuncert", transfer_r2), ("d_metamcuncert", recomputed_r2)):
        if vals:
            layers = sorted(vals.keys())
            arr = np.array([vals[l] for l in layers], dtype=float)
            best_idx = int(np.argmax(arr))
            summary[name] = {
                "n_layers": len(layers),
                "mean_abs_r": None,
                "mean_r2": float(np.mean(arr)),
                "best_r2": float(arr[best_idx]),
                "best_r2_layer": int(layers[best_idx]),
                "mean_leave_one_out_delta_r2": None,
                "best_leave_one_out_delta_r2": None,
                "best_leave_one_out_layer": None,
                "mean_abs_beta": None,
            }
    return per_layer, summary


# =============================================================================
# BEHAVIORAL CORRELATION ANALYSIS
# =============================================================================


def load_mcq_metric_values(dataset: str, metric: str, model_dir: str, activations: np.lib.npyio.NpzFile, output_key: str) -> np.ndarray:
    """Load MCQ metric values from mc_results.json, aligned to meta activations."""
    mc_results_path = find_output_file(f"{dataset}_mc_results.json", model_dir=model_dir)
    if not mc_results_path.exists():
        raise FileNotFoundError(f"MCQ results file not found: {mc_results_path}")

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
        f"MCQ metric array length mismatch for metric={metric!r}: mc_results has {len(arr)} examples, "
        f"meta output has {y_len}. Could not align via valid_final."
    )


def regression_residuals(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, List[int]]:
    """Compute residuals from OLS on standardized X and y."""
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


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 3 or y.size < 3:
        return None
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    r, _ = pearsonr(x, y)
    if not math.isfinite(r):
        return None
    return float(r)


def analyze_behavioral_correlation_two_direction(
    directions: Dict[str, LoadedDirection],
    activations: np.lib.npyio.NpzFile,
    output_key: str,
    mcq_metric_values: np.ndarray,
    metric_name: str,
    dir_a: str = "d_mcuncert",
    dir_b: str = "d_metamcanswer",
) -> Optional[Dict[str, Any]]:
    """Analyze how much behavioral correlation is explained by two specific directions."""
    if dir_a not in directions or dir_b not in directions:
        return None

    layer_keys = available_activation_layers(activations)
    y_all = np.asarray(activations[output_key], dtype=float)
    u_all = np.asarray(mcq_metric_values, dtype=float)
    raw_mask = finite_mask(u_all, y_all)
    raw_corr = safe_pearsonr(u_all[raw_mask], y_all[raw_mask])
    if raw_corr is None:
        return None

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

        resid_a, _ = regression_residuals(xa[:, None], y)
        remaining_a = safe_pearsonr(u, resid_a)
        resid_b, _ = regression_residuals(xb[:, None], y)
        remaining_b = safe_pearsonr(u, resid_b)
        resid_both, _ = regression_residuals(np.column_stack([xa, xb]), y)
        remaining_both = safe_pearsonr(u, resid_both)
        if remaining_a is None or remaining_b is None or remaining_both is None:
            continue

        explained_a = float(raw_corr - remaining_a)
        explained_b = float(raw_corr - remaining_b)
        explained_both = float(raw_corr - remaining_both)
        unique_a = float(remaining_b - remaining_both)
        unique_b = float(remaining_a - remaining_both)

        per_layer[layer] = {
            "n": int(np.sum(mask)),
            "raw_behavioral_corr": float(raw_corr),
            "single_direction": {
                dir_a: {"remaining": float(remaining_a), "explained": explained_a},
                dir_b: {"remaining": float(remaining_b), "explained": explained_b},
            },
            "two_direction_model": {
                "remaining": float(remaining_both),
                "explained": explained_both,
                "unique": {dir_a: unique_a, dir_b: unique_b},
            },
        }

    if not per_layer:
        return None

    def summarize_single(name: str) -> Dict[str, Any]:
        rows = [(layer, info["single_direction"][name]["explained"]) for layer, info in per_layer.items()]
        best = max(rows, key=lambda t: t[1])
        return {
            "mean_explained": float(np.mean([r[1] for r in rows])),
            "best_explained": float(best[1]),
            "best_layer": int(best[0]),
        }

    full_rows = [(layer, info["two_direction_model"]["explained"]) for layer, info in per_layer.items()]
    best_full = max(full_rows, key=lambda t: t[1])

    uniq_summary = {}
    for name in (dir_a, dir_b):
        vals = [(layer, info["two_direction_model"]["unique"][name]) for layer, info in per_layer.items()]
        best = max(vals, key=lambda t: t[1])
        uniq_summary[name] = {
            "mean_unique": float(np.mean([v for _, v in vals])),
            "best_unique": float(best[1]),
            "best_layer": int(best[0]),
        }

    return {
        "mcq_metric": metric_name,
        "output_key": output_key,
        "raw_behavioral_corr": float(raw_corr),
        "directions": [dir_a, dir_b],
        "single_summary": {dir_a: summarize_single(dir_a), dir_b: summarize_single(dir_b)},
        "two_direction_summary": {
            "mean_explained": float(np.mean([r[1] for r in full_rows])),
            "best_explained": float(best_full[1]),
            "best_layer": int(best_full[0]),
            "unique": uniq_summary,
        },
        "per_layer": per_layer,
    }


def analyze_behavioral_correlation_all_directions(
    directions: Dict[str, LoadedDirection],
    activations: np.lib.npyio.NpzFile,
    output_key: str,
    mcq_metric_values: np.ndarray,
    metric_name: str,
    exclude_directions: Optional[set] = None,
) -> Optional[Dict[str, Any]]:
    """Analyze how much behavioral correlation is explained by all directions."""
    exclude_directions = exclude_directions or set()
    dir_names = [n for n in directions.keys() if n not in exclude_directions]
    if len(dir_names) < 2:
        return None

    layer_keys = available_activation_layers(activations)
    y_all = np.asarray(activations[output_key], dtype=float)
    u_all = np.asarray(mcq_metric_values, dtype=float)
    raw_mask = finite_mask(u_all, y_all)
    raw_corr = safe_pearsonr(u_all[raw_mask], y_all[raw_mask])
    if raw_corr is None:
        return None

    per_layer: Dict[int, Dict[str, Any]] = {}

    for layer in sorted(layer_keys.keys()):
        acts = np.asarray(activations[layer_keys[layer]], dtype=float)
        if acts.ndim != 2:
            continue

        projections: Dict[str, np.ndarray] = {}
        for name in dir_names:
            vec = directions[name].layers.get(layer)
            if vec is not None and acts.shape[1] == vec.shape[0]:
                projections[name] = acts @ vec

        available = sorted(projections.keys())
        if len(available) < 2:
            continue

        mask = finite_mask(u_all, y_all, *[projections[n] for n in available])
        if int(np.sum(mask)) < 10:
            continue

        u = u_all[mask]
        y = y_all[mask]

        # Single-direction analysis
        single_results: Dict[str, Dict[str, float]] = {}
        for name in available:
            x = projections[name][mask]
            resid, _ = regression_residuals(x[:, None], y)
            remaining = safe_pearsonr(u, resid)
            if remaining is not None:
                single_results[name] = {
                    "remaining": float(remaining),
                    "explained": float(raw_corr - remaining),
                }

        # Full model
        X_full = np.column_stack([projections[n][mask] for n in available])
        resid_full, _ = regression_residuals(X_full, y)
        remaining_full = safe_pearsonr(u, resid_full)
        if remaining_full is None:
            continue

        explained_full = float(raw_corr - remaining_full)

        # Leave-one-out unique contributions
        unique: Dict[str, float] = {}
        for j, name in enumerate(available):
            X_loo = np.delete(X_full, j, axis=1)
            resid_loo, _ = regression_residuals(X_loo, y)
            remaining_loo = safe_pearsonr(u, resid_loo)
            if remaining_loo is not None:
                unique[name] = float(remaining_loo - remaining_full)

        per_layer[layer] = {
            "n": int(np.sum(mask)),
            "raw_behavioral_corr": float(raw_corr),
            "directions_available": available,
            "single_direction": single_results,
            "full_model": {
                "remaining": float(remaining_full),
                "explained": explained_full,
                "unique": unique,
            },
        }

    if not per_layer:
        return None

    # Summaries
    single_summary: Dict[str, Dict[str, Any]] = {}
    for name in dir_names:
        rows = [(layer, info["single_direction"][name]["explained"])
                for layer, info in per_layer.items() if name in info["single_direction"]]
        if rows:
            best = max(rows, key=lambda t: t[1])
            single_summary[name] = {
                "mean_explained": float(np.mean([r[1] for r in rows])),
                "best_explained": float(best[1]),
                "best_layer": int(best[0]),
            }

    full_rows = [(layer, info["full_model"]["explained"]) for layer, info in per_layer.items()]
    best_full = max(full_rows, key=lambda t: t[1])

    unique_summary: Dict[str, Dict[str, Any]] = {}
    for name in dir_names:
        vals = [(layer, info["full_model"]["unique"][name])
                for layer, info in per_layer.items() if name in info["full_model"]["unique"]]
        if vals:
            best = max(vals, key=lambda t: t[1])
            unique_summary[name] = {
                "mean_unique": float(np.mean([v for _, v in vals])),
                "best_unique": float(best[1]),
                "best_layer": int(best[0]),
            }

    return {
        "mcq_metric": metric_name,
        "output_key": output_key,
        "raw_behavioral_corr": float(raw_corr),
        "directions": dir_names,
        "single_summary": single_summary,
        "full_model_summary": {
            "mean_explained": float(np.mean([r[1] for r in full_rows])),
            "best_explained": float(best_full[1]),
            "best_layer": int(best_full[0]),
            "unique": unique_summary,
        },
        "per_layer": per_layer,
    }


# =============================================================================
# PLOTTING
# =============================================================================


def pair_label(a: str, b: str) -> str:
    short = {
        "d_mcuncert": "d_mc",
        "d_metamcuncert": "d_metamcuncert",
        "d_mcanswer": "d_mc_answer",
        "d_metamcanswer": "d_meta_answer",
        "d_metaconfdir": "d_confdir",
    }
    return f"{short.get(a, a)} vs {short.get(b, b)}"


def _plot_cosines_on_axis(ax: plt.Axes, pairwise_cosines: Dict[str, Dict[int, float]], cosine_summary: Dict[str, Dict[str, Any]], pairwise_cosine_cis: Optional[Dict[str, Dict[int, Tuple[float, float]]]] = None) -> None:
    for a, b in COSINE_PAIR_ORDER:
        key = f"{a}__vs__{b}"
        rev = f"{b}__vs__{a}"
        actual = key if key in pairwise_cosines else rev if rev in pairwise_cosines else None
        if actual is None:
            continue
        layers = np.array(sorted(pairwise_cosines[actual].keys()), dtype=int)
        vals = np.array([pairwise_cosines[actual][int(l)] for l in layers], dtype=float)
        # Use |cosine| for pairs involving answer directions (sign is arbitrary for those)
        if a in USE_ABSOLUTE_VALUE_PROJECTION or b in USE_ABSOLUTE_VALUE_PROJECTION:
            vals = np.abs(vals)
        stat = cosine_summary[actual]
        line, = ax.plot(layers, vals, linewidth=2, label=f"{pair_label(a, b)} (|cos|={stat['mean_abs']:.3f})")
        if pairwise_cosine_cis is not None:
            ci_key = actual if actual in pairwise_cosine_cis else key if key in pairwise_cosine_cis else rev if rev in pairwise_cosine_cis else None
            if ci_key is not None:
                lo = np.array([pairwise_cosine_cis[ci_key].get(int(l), (np.nan, np.nan))[0] for l in layers], dtype=float)
                hi = np.array([pairwise_cosine_cis[ci_key].get(int(l), (np.nan, np.nan))[1] for l in layers], dtype=float)
                # Apply |cos| to CIs as well for answer direction pairs
                if a in USE_ABSOLUTE_VALUE_PROJECTION or b in USE_ABSOLUTE_VALUE_PROJECTION:
                    lo = np.abs(lo)
                    hi = np.abs(hi)
                    # Ensure lo <= hi after abs
                    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
                ok = np.isfinite(lo) & np.isfinite(hi)
                if np.any(ok):
                    ax.fill_between(layers[ok], lo[ok], hi[ok], color=line.get_color(), alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cosine similarity")
    ax.set_ylim(-1.0, 1.0)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=9)


def plot_all_cosines(pairwise_cosines: Dict[str, Dict[int, float]], cosine_summary: Dict[str, Dict[str, Any]], output_path: Path, title_suffix: str) -> None:
    fig, ax = plt.subplots(figsize=(16, 10))
    _plot_cosines_on_axis(ax, pairwise_cosines, cosine_summary)
    ax.set_title(f"Direction cosine similarities across layers\n{title_suffix}")
    save_figure(fig, output_path)


def _plot_single_direction_fit_on_axis(
    ax: plt.Axes,
    per_layer: Dict[int, Dict[str, Any]],
    summary: Dict[str, Dict[str, Any]],
    title: str,
    unavailable_text: Optional[str] = None,
    reference_line: Optional[Tuple[str, Dict[int, float]]] = None,
) -> None:
    if unavailable_text is not None:
        ax.axis("off")
        ax.text(0.5, 0.5, unavailable_text, ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return
    for name, stats in summary.items():
        xs = sorted([layer for layer, info in per_layer.items() if name in info["single"]])
        ys = [per_layer[layer]["single"][name]["r2"] for layer in xs]
        ax.plot(xs, ys, linewidth=2.5, label=f"{name} (best={stats['best_r2']:.3f} @ L{stats['best_r2_layer']})")
    # Add reference line (e.g., d_metaconfdir as target-defined ceiling)
    if reference_line is not None:
        ref_name, ref_r2_by_layer = reference_line
        ref_xs = sorted(ref_r2_by_layer.keys())
        ref_ys = [ref_r2_by_layer[l] for l in ref_xs]
        best_r2 = max(ref_ys) if ref_ys else 0
        best_layer = ref_xs[ref_ys.index(best_r2)] if ref_ys else 0
        ax.plot(ref_xs, ref_ys, linewidth=2.5, linestyle="--", color="gray",
                label=f"{ref_name} (best={best_r2:.3f} @ L{best_layer}, target-defined)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=9)


def plot_single_direction_fit(per_layer: Dict[int, Dict[str, Any]], summary: Dict[str, Dict[str, Any]], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    unavailable_text = None if summary else "Target not available"
    _plot_single_direction_fit_on_axis(ax, per_layer, summary, title, unavailable_text)
    save_figure(fig, output_path)


def _plot_meta_output_unique_on_axis(ax: plt.Axes, per_layer: Dict[int, Dict[str, Any]], direction_names: Sequence[str], title: str) -> None:
    full_x, full_r2 = [], []
    for layer in sorted(per_layer.keys()):
        full = per_layer[layer].get("full_model")
        if full is not None:
            full_x.append(layer)
            full_r2.append(full["r2"])
    if full_x:
        ax.plot(full_x, full_r2, color="black", linewidth=3, label="full model R²")
    for name in direction_names:
        xs, ys = [], []
        for layer in sorted(per_layer.keys()):
            full = per_layer[layer].get("full_model")
            if full is None:
                continue
            delta = full["leave_one_out_delta_r2"].get(name)
            if delta is None:
                continue
            xs.append(layer)
            ys.append(delta)
        if xs:
            ax.plot(xs, ys, linewidth=2.5, label=f"ΔR² drop if remove {name}")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R² / ΔR²")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=9)


def plot_meta_output_unique(per_layer: Dict[int, Dict[str, Any]], direction_names: Sequence[str], output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 7))
    _plot_meta_output_unique_on_axis(ax, per_layer, direction_names, title)
    save_figure(fig, output_path)


def _plot_two_direction_meta_on_axis(
    ax: plt.Axes,
    two_dir_per_layer: Dict[int, Dict[str, Any]],
    dir_a: str,
    dir_b: str,
    title: str,
) -> None:
    """Plot 2-direction model R² and unique contributions."""
    if not two_dir_per_layer:
        ax.axis("off")
        ax.text(0.5, 0.5, "2-direction analysis not available", ha="center", va="center", fontsize=14)
        ax.set_title(title)
        return

    layers = sorted(two_dir_per_layer.keys())
    xs = np.array(layers, dtype=int)

    # Combined R²
    combined = np.array([two_dir_per_layer[l]["combined_r2"] for l in layers], dtype=float)
    ax.plot(xs, combined, color="black", linewidth=3, label=f"combined R² ({dir_a} + {dir_b})")

    # Unique contributions
    unique_a = np.array([two_dir_per_layer[l]["unique"][dir_a] for l in layers], dtype=float)
    unique_b = np.array([two_dir_per_layer[l]["unique"][dir_b] for l in layers], dtype=float)
    ax.plot(xs, unique_a, linewidth=2.5, label=f"ΔR² unique to {dir_a}")
    ax.plot(xs, unique_b, linewidth=2.5, label=f"ΔR² unique to {dir_b}")

    ax.set_xlabel("Layer")
    ax.set_ylabel("R² / ΔR²")
    ax.set_title(title)
    ax.grid(True, alpha=GRID_ALPHA)
    ax.legend(loc="best", fontsize=9)


def plot_combined_report(
    pairwise_cosines: Dict[str, Dict[int, float]],
    cosine_summary: Dict[str, Dict[str, Any]],
    pairwise_cosine_cis: Dict[str, Dict[int, Tuple[float, float]]],
    mcq_uncert_per_layer: Dict[int, Dict[str, Any]],
    mcq_uncert_summary: Dict[str, Dict[str, Any]],
    mcq_answer_per_layer: Dict[int, Dict[str, Any]],
    mcq_answer_summary: Dict[str, Dict[str, Any]],
    meta_per_layer: Dict[int, Dict[str, Any]],
    meta_summary: Dict[str, Dict[str, Any]],
    meta_names: Sequence[str],
    output_path: Path,
    title_suffix: str,
    mcq_uncert_title_suffix: str,
    mcq_answer_available: bool,
    confdir_r2_by_layer: Optional[Dict[int, float]] = None,
    two_dir_meta_per_layer: Optional[Dict[int, Dict[str, Any]]] = None,
    two_dir_names: Optional[Tuple[str, str]] = None,
) -> None:
    fig, axes = plt.subplots(6, 1, figsize=(16, 34))

    _plot_cosines_on_axis(axes[0], pairwise_cosines, cosine_summary)
    axes[0].set_title(f"Panel 1: direction cosine similarities across layers\n{title_suffix}")

    _plot_single_direction_fit_on_axis(
        axes[1],
        mcq_uncert_per_layer,
        mcq_uncert_summary,
        f"Panel 2: MCQ uncertainty target fit by layer ({mcq_uncert_title_suffix})",
        None if mcq_uncert_summary else "MCQ uncertainty target not available",
    )

    _plot_single_direction_fit_on_axis(
        axes[2],
        mcq_answer_per_layer,
        mcq_answer_summary,
        "Panel 3: MCQ answer target fit by layer",
        None if mcq_answer_available else "Numeric MCQ answer target not found in activations",
    )

    # Panel 4: include d_metaconfdir as reference line (it's target-defined but useful as ceiling)
    confdir_ref = None
    if confdir_r2_by_layer:
        confdir_ref = ("d_metaconfdir", confdir_r2_by_layer)
    _plot_single_direction_fit_on_axis(
        axes[3],
        meta_per_layer,
        meta_summary,
        "Panel 4: meta output fit by layer (d_metaconfdir excluded)",
        None,
        reference_line=confdir_ref,
    )

    _plot_meta_output_unique_on_axis(
        axes[4],
        meta_per_layer,
        meta_names,
        "Panel 5: meta output unique contribution by layer (d_metaconfdir excluded)",
    )

    # Panel 6: 2-direction model
    dir_a, dir_b = two_dir_names if two_dir_names else ("d_mcuncert", "d_metamcanswer")
    _plot_two_direction_meta_on_axis(
        axes[5],
        two_dir_meta_per_layer or {},
        dir_a,
        dir_b,
        f"Panel 6: 2-direction meta output model ({dir_a} + {dir_b})",
    )

    fig.suptitle(f"Mean-diff direction analysis\n{title_suffix}", fontsize=18, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.987))
    save_figure(fig, output_path)


def plot_behavioral_correlation_two_direction(
    analysis: Dict[str, Any],
    output_path: Path,
    title_suffix: str,
) -> None:
    """Plot behavioral correlation decomposition for two specific directions."""
    dir_a, dir_b = analysis["directions"]
    raw_corr = analysis["raw_behavioral_corr"]
    layers = sorted(int(l) for l in analysis["per_layer"].keys())
    xs = np.array(layers, dtype=int)

    exp_a = np.array([analysis["per_layer"][l]["single_direction"][dir_a]["explained"] for l in layers], dtype=float)
    exp_b = np.array([analysis["per_layer"][l]["single_direction"][dir_b]["explained"] for l in layers], dtype=float)
    rem_a = np.array([analysis["per_layer"][l]["single_direction"][dir_a]["remaining"] for l in layers], dtype=float)
    rem_b = np.array([analysis["per_layer"][l]["single_direction"][dir_b]["remaining"] for l in layers], dtype=float)
    exp_both = np.array([analysis["per_layer"][l]["two_direction_model"]["explained"] for l in layers], dtype=float)
    rem_both = np.array([analysis["per_layer"][l]["two_direction_model"]["remaining"] for l in layers], dtype=float)
    uniq_a = np.array([analysis["per_layer"][l]["two_direction_model"]["unique"][dir_a] for l in layers], dtype=float)
    uniq_b = np.array([analysis["per_layer"][l]["two_direction_model"]["unique"][dir_b] for l in layers], dtype=float)

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
    fig.tight_layout()
    save_figure(fig, output_path)


def plot_behavioral_correlation_all_directions(
    analysis: Dict[str, Any],
    output_path: Path,
    title_suffix: str,
) -> None:
    """Plot behavioral correlation decomposition for all directions."""
    dir_names = analysis["directions"]
    raw_corr = analysis["raw_behavioral_corr"]
    layers = sorted(int(l) for l in analysis["per_layer"].keys())
    xs = np.array(layers, dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

    # Panel 1: Single-direction explained correlation
    for name in dir_names:
        ys = []
        valid_xs = []
        for l in layers:
            info = analysis["per_layer"][l]["single_direction"].get(name)
            if info is not None:
                ys.append(info["explained"])
                valid_xs.append(l)
        if ys:
            axes[0].plot(valid_xs, ys, linewidth=2, label=f"{name}")
    axes[0].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[0].set_ylabel("Explained behavioral corr")
    axes[0].set_title("Single-direction: how much corr(MCQ uncertainty, meta output) is explained")
    axes[0].grid(True, alpha=GRID_ALPHA)
    axes[0].legend(loc="best", fontsize=9)

    # Panel 2: Full model explained + remaining
    full_exp = np.array([analysis["per_layer"][l]["full_model"]["explained"] for l in layers], dtype=float)
    full_rem = np.array([analysis["per_layer"][l]["full_model"]["remaining"] for l in layers], dtype=float)
    axes[1].axhline(raw_corr, color="gray", linestyle="--", linewidth=2, label=f"raw corr = {raw_corr:.3f}")
    axes[1].plot(xs, full_exp, color="black", linewidth=2.5, label="full model explained")
    axes[1].plot(xs, full_rem, color="red", linewidth=2.5, label="full model remaining")
    axes[1].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[1].set_ylabel("Correlation")
    axes[1].set_title("Full model: explained vs remaining behavioral correlation")
    axes[1].grid(True, alpha=GRID_ALPHA)
    axes[1].legend(loc="best")

    # Panel 3: Leave-one-out unique contributions
    axes[2].plot(xs, full_exp, color="black", linewidth=2.5, label="full model explained")
    for name in dir_names:
        ys = []
        valid_xs = []
        for l in layers:
            uniq = analysis["per_layer"][l]["full_model"]["unique"].get(name)
            if uniq is not None:
                ys.append(uniq)
                valid_xs.append(l)
        if ys:
            axes[2].plot(valid_xs, ys, linewidth=2, label=f"unique: {name}")
    axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.5)
    axes[2].set_xlabel("Layer")
    axes[2].set_ylabel("Explained corr")
    axes[2].set_title("Full model: unique contribution by direction (leave-one-out)")
    axes[2].grid(True, alpha=GRID_ALPHA)
    axes[2].legend(loc="best", fontsize=9)

    fig.suptitle(f"Behavioral correlation decomposition (all directions)\n{title_suffix}", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_path)


# =============================================================================
# REPORTING
# =============================================================================


def print_direction_list(directions: Dict[str, LoadedDirection]) -> None:
    print("Loaded directions:")
    for name in INCLUDE_DIRECTIONS:
        if name in directions:
            info = directions[name]
            print(f"  {name:14s} {len(info.layers):3d} layers | {Path(info.file_path).name}")
    print()


def format_best(summary: Dict[str, Dict[str, Any]]) -> str:
    if not summary:
        return "none"
    items = sorted(summary.items(), key=lambda kv: kv[1]["mean_r2"], reverse=True)
    name, stats = items[0]
    return f"{name} (mean R²={stats['mean_r2']:.3f}, best={stats['best_r2']:.3f} @ L{stats['best_r2_layer']})"


def format_best_delta(summary: Dict[str, Dict[str, Any]]) -> str:
    if not summary:
        return "none"
    scored = [(name, stats) for name, stats in summary.items() if stats.get("mean_leave_one_out_delta_r2") is not None]
    if not scored:
        return "none"
    name, stats = max(scored, key=lambda kv: kv[1]["mean_leave_one_out_delta_r2"])
    return f"{name} (mean ΔR²={stats['mean_leave_one_out_delta_r2']:.3f}, best={stats['best_leave_one_out_delta_r2']:.3f} @ L{stats['best_leave_one_out_layer']})"


def print_behavioral_correlation_report(
    two_dir: Optional[Dict[str, Any]],
    all_dir: Optional[Dict[str, Any]],
) -> None:
    """Print behavioral correlation analysis summary."""
    if two_dir is None and all_dir is None:
        print("Behavioral correlation analysis: not available (missing directions or MCQ data)")
        return

    print("Behavioral correlation decomposition:")
    if two_dir is not None:
        dir_a, dir_b = two_dir["directions"]
        raw = two_dir["raw_behavioral_corr"]
        sa = two_dir["single_summary"][dir_a]
        sb = two_dir["single_summary"][dir_b]
        full = two_dir["two_direction_summary"]
        print(f"  Raw corr({two_dir['mcq_metric']}, {two_dir['output_key']}) = {raw:.3f}")
        print(f"  2-direction model ({dir_a} + {dir_b}):")
        print(f"    {dir_a} alone: mean explained = {sa['mean_explained']:.3f}, best = {sa['best_explained']:.3f} @ L{sa['best_layer']}")
        print(f"    {dir_b} alone: mean explained = {sb['mean_explained']:.3f}, best = {sb['best_explained']:.3f} @ L{sb['best_layer']}")
        print(f"    Combined: mean explained = {full['mean_explained']:.3f}, best = {full['best_explained']:.3f} @ L{full['best_layer']}")
        print(f"    Unique: {dir_a} = {full['unique'][dir_a]['mean_unique']:.3f}, {dir_b} = {full['unique'][dir_b]['mean_unique']:.3f}")

    if all_dir is not None:
        full = all_dir["full_model_summary"]
        print(f"  All-directions model:")
        print(f"    Full model: mean explained = {full['mean_explained']:.3f}, best = {full['best_explained']:.3f} @ L{full['best_layer']}")
        uniq_items = sorted(full["unique"].items(), key=lambda kv: kv[1]["mean_unique"], reverse=True)
        if uniq_items:
            top = uniq_items[0]
            print(f"    Top unique contributor: {top[0]} (mean = {top[1]['mean_unique']:.3f})")
    print()


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    model_dir = get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)
    activations = load_meta_activations(DATASET, META_TASK, model_dir)
    output_key = infer_output_target(META_TASK, OUTPUT_TARGET, activations)
    confdir_file_tag = infer_confdir_file_tag(output_key)
    mcq_uncertainty_key = infer_mcq_uncertainty_key(activations, UNCERTAINTY_METRIC, MCQ_UNCERTAINTY_TARGET)
    mcq_answer_key = infer_mcq_answer_key(activations, MCQ_ANSWER_TARGET)

    directions = load_all_directions(DATASET, META_TASK, UNCERTAINTY_METRIC, confdir_file_tag, model_dir)
    if len(directions) < 2:
        raise RuntimeError(f"Need at least two directions. Loaded: {list(directions.keys())}")

    pairwise_cosines = compute_pairwise_direction_cosines(directions)
    pairwise_cosine_cis = compute_pairwise_cosine_cis(directions)
    cosine_summary = summarize_pairwise_cosines(pairwise_cosines)
    # Use |proj| for answer directions (captures "has any answer" vs "which answer")
    projection_store = build_projection_store(directions, activations, use_abs=USE_ABSOLUTE_VALUE_PROJECTION)
    # Signed projection for MCQ answer target (predicting which answer needs signed values)
    projection_store_signed = build_projection_store(directions, activations, use_abs=None)

    # Panel 2: MCQ uncertainty target.
    mcq_uncert_per_layer: Dict[int, Dict[str, Any]] = {}
    mcq_uncert_summary: Dict[str, Dict[str, Any]] = {}
    mcq_uncert_mode = "from_activations"
    if mcq_uncertainty_key is not None:
        y_mcq = coerce_numeric_array(activations[mcq_uncertainty_key])
        assert y_mcq is not None
        mcq_uncert_per_layer, mcq_uncert_summary = analyze_continuous_target(
            y_mcq,
            projection_store,
            [n for n in ("d_mcuncert", "d_metamcuncert") if n in directions],
        )
    else:
        mcq_uncert_mode = "fallback_json"
        transfer_results = load_transfer_results(DATASET, META_TASK, model_dir)
        mcuncert_results = load_mcuncert_results(DATASET, META_TASK, model_dir)
        mcq_uncert_per_layer, mcq_uncert_summary = build_mcq_uncertainty_fallback(transfer_results, mcuncert_results)

    # Panel 3: MCQ answer target (when available).
    # Use signed projection here since we're predicting which answer, not whether any answer.
    mcq_answer_per_layer: Dict[int, Dict[str, Any]] = {}
    mcq_answer_summary: Dict[str, Dict[str, Any]] = {}
    if mcq_answer_key is not None:
        y_answer = coerce_numeric_array(activations[mcq_answer_key])
        assert y_answer is not None
        mcq_answer_per_layer, mcq_answer_summary = analyze_continuous_target(
            y_answer,
            projection_store_signed,
            [n for n in ("d_mcanswer", "d_metamcanswer") if n in directions],
        )

    # Panels 4/5: meta output, excluding target-defined output axis.
    y_meta = coerce_numeric_array(activations[output_key])
    if y_meta is None:
        raise RuntimeError(f"Meta output target {output_key!r} is not numeric.")
    meta_names = [n for n in INCLUDE_DIRECTIONS if n in directions and n not in EXCLUDE_FROM_META_OUTPUT_ANALYSIS]
    meta_per_layer, meta_summary = analyze_continuous_target(y_meta, projection_store, meta_names)

    # Panel 6: 2-direction model (d_mcuncert + d_metamcanswer)
    two_dir_a, two_dir_b = "d_mcuncert", "d_metamcanswer"
    two_dir_meta_per_layer: Dict[int, Dict[str, Any]] = {}
    if two_dir_a in directions and two_dir_b in directions:
        two_dir_meta_per_layer = analyze_two_direction_meta_output(
            y_meta, projection_store, two_dir_a, two_dir_b
        )

    # Compute d_metaconfdir R² as reference (target-defined, so circular but useful as ceiling)
    confdir_r2_by_layer: Dict[int, float] = {}
    if "d_metaconfdir" in directions:
        for layer in sorted(projection_store.keys()):
            if "d_metaconfdir" not in projection_store[layer]:
                continue
            proj = projection_store[layer]["d_metaconfdir"]
            mask = finite_mask(proj, y_meta)
            if mask.sum() < 5:
                continue
            x = proj[mask]
            y = y_meta[mask]
            r2, _, _ = regression_r2(x[:, None], y)
            confdir_r2_by_layer[layer] = float(r2)

    # Behavioral correlation analysis: how much of corr(MCQ uncertainty, meta output) is explained
    behavioral_two_dir: Optional[Dict[str, Any]] = None
    behavioral_all_dir: Optional[Dict[str, Any]] = None
    try:
        mcq_metric_values = load_mcq_metric_values(DATASET, UNCERTAINTY_METRIC, model_dir, activations, output_key)
        behavioral_two_dir = analyze_behavioral_correlation_two_direction(
            directions, activations, output_key, mcq_metric_values, UNCERTAINTY_METRIC,
            dir_a="d_mcuncert", dir_b="d_metamcanswer",
        )
        behavioral_all_dir = analyze_behavioral_correlation_all_directions(
            directions, activations, output_key, mcq_metric_values, UNCERTAINTY_METRIC,
            exclude_directions=EXCLUDE_FROM_META_OUTPUT_ANALYSIS,
        )
    except (FileNotFoundError, KeyError, ValueError) as e:
        print(f"Behavioral correlation analysis skipped: {e}")

    title_suffix = f"{DATASET} | {META_TASK} | output={output_key} | metric={UNCERTAINTY_METRIC} | method={METHOD}"

    combined_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_report_{output_key}_{UNCERTAINTY_METRIC}.png",
        model_dir=model_dir,
    )
    plot_combined_report(
        pairwise_cosines,
        cosine_summary,
        pairwise_cosine_cis,
        mcq_uncert_per_layer,
        mcq_uncert_summary,
        mcq_answer_per_layer,
        mcq_answer_summary,
        meta_per_layer,
        meta_summary,
        meta_names,
        combined_path,
        title_suffix,
        mcq_uncert_mode if mcq_uncertainty_key is None else mcq_uncertainty_key,
        mcq_answer_key is not None,
        confdir_r2_by_layer=confdir_r2_by_layer,
        two_dir_meta_per_layer=two_dir_meta_per_layer,
        two_dir_names=(two_dir_a, two_dir_b),
    )

    # Behavioral correlation figures
    behavioral_two_dir_path: Optional[Path] = None
    behavioral_all_dir_path: Optional[Path] = None
    if behavioral_two_dir is not None:
        behavioral_two_dir_path = get_output_path(
            f"{DATASET}_meta_{META_TASK}_behavioral_corr_2dir_{output_key}_{UNCERTAINTY_METRIC}.png",
            model_dir=model_dir,
        )
        plot_behavioral_correlation_two_direction(behavioral_two_dir, behavioral_two_dir_path, title_suffix)
    if behavioral_all_dir is not None:
        behavioral_all_dir_path = get_output_path(
            f"{DATASET}_meta_{META_TASK}_behavioral_corr_alldir_{output_key}_{UNCERTAINTY_METRIC}.png",
            model_dir=model_dir,
        )
        plot_behavioral_correlation_all_directions(behavioral_all_dir, behavioral_all_dir_path, title_suffix)

    print("=" * 90)
    print("MEAN-DIFF DIRECTION ANALYSIS")
    print("=" * 90)
    print(f"Model dir: {model_dir}")
    print(f"Dataset:   {DATASET}")
    print(f"Task:      {META_TASK}")
    print(f"Metric:    {UNCERTAINTY_METRIC}")
    print(f"Method:    {METHOD}")
    print(f"Meta output target: {output_key}")
    print(f"MCQ uncertainty target: {mcq_uncertainty_key if mcq_uncertainty_key is not None else 'not found; using JSON fallback for d_mcuncert and d_metamcuncert'}")
    print(f"MCQ answer target: {mcq_answer_key if mcq_answer_key is not None else 'not found'}")
    print()
    print_direction_list(directions)

    print(f"Strongest MCQ-uncertainty predictor: {format_best(mcq_uncert_summary)}")
    print(f"Strongest MCQ-answer predictor:      {format_best(mcq_answer_summary)}")
    print(f"Strongest meta-output predictor:     {format_best(meta_summary)}")
    print(f"Strongest meta-output unique term:   {format_best_delta(meta_summary)}")
    print()
    print_behavioral_correlation_report(behavioral_two_dir, behavioral_all_dir)
    print(f"Six-panel report: {combined_path}")
    if behavioral_two_dir_path:
        print(f"Behavioral (2-dir): {behavioral_two_dir_path}")
    if behavioral_all_dir_path:
        print(f"Behavioral (all-dir): {behavioral_all_dir_path}")

    summary = {
        "config": get_config_dict(
            model=MODEL,
            dataset=DATASET,
            meta_task=META_TASK,
            metric=UNCERTAINTY_METRIC,
            method=METHOD,
            output_target=output_key,
            mcq_uncertainty_target=mcq_uncertainty_key,
            mcq_answer_target=mcq_answer_key,
            confdir_file_tag=confdir_file_tag,
            load_in_4bit=LOAD_IN_4BIT,
            load_in_8bit=LOAD_IN_8BIT,
            use_absolute_value_projection=sorted(USE_ABSOLUTE_VALUE_PROJECTION),
        ),
        "loaded_directions": {
            name: {
                "description": d.description,
                "file_path": d.file_path,
                "key_pattern": d.key_pattern,
                "n_layers": len(d.layers),
            }
            for name, d in directions.items()
        },
        "pairwise_direction_cosines": {
            key: {
                "summary": cosine_summary[key],
                "per_layer": {str(layer): float(val) for layer, val in vals.items()},
            }
            for key, vals in pairwise_cosines.items()
        },
        "mcq_uncertainty_analysis": {
            "mode": mcq_uncert_mode,
            "target_key": mcq_uncertainty_key,
            "summary": mcq_uncert_summary,
            "per_layer": mcq_uncert_per_layer if SAVE_FULL_PER_LAYER_DETAILS else None,
        },
        "mcq_answer_analysis": {
            "target_key": mcq_answer_key,
            "summary": mcq_answer_summary,
            "per_layer": mcq_answer_per_layer if SAVE_FULL_PER_LAYER_DETAILS else None,
        },
        "meta_output_analysis": {
            "target_key": output_key,
            "excluded_directions": sorted(EXCLUDE_FROM_META_OUTPUT_ANALYSIS),
            "summary": meta_summary,
            "per_layer": meta_per_layer if SAVE_FULL_PER_LAYER_DETAILS else None,
        },
        "two_direction_meta_output_analysis": {
            "directions": [two_dir_a, two_dir_b],
            "per_layer": two_dir_meta_per_layer if SAVE_FULL_PER_LAYER_DETAILS else None,
        },
        "behavioral_correlation_two_direction": behavioral_two_dir,
        "behavioral_correlation_all_directions": behavioral_all_dir,
        "figures": {
            "combined_report": str(combined_path),
            "behavioral_two_direction": str(behavioral_two_dir_path) if behavioral_two_dir_path else None,
            "behavioral_all_directions": str(behavioral_all_dir_path) if behavioral_all_dir_path else None,
        },
    }

    json_path = get_output_path(
        f"{DATASET}_meta_{META_TASK}_direction_influence_summary_{output_key}_{UNCERTAINTY_METRIC}.json",
        model_dir=model_dir,
    )
    with open(json_path, "w") as f:
        json.dump(json_ready(summary), f, indent=2)
    print(f"Summary JSON: {json_path}")
    print("=" * 90)


if __name__ == "__main__":
    main()
