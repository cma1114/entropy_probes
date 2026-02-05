# -*- coding: utf-8 -*-
"""test_cross_dataset_transfer.py

Cross-dataset transfer: train a direction on dataset A (MC metric), evaluate on dataset B.

This script computes:
  - D2D: A_train direction -> B_test MC activations vs B_test MC metric
  - D2M: A_train direction -> B_test meta activations vs B_test behavioral confidence

Statistical analysis:
  - A *training-procedure permutation null*: permute labels on A_train, retrain direction,
    then evaluate on B_test; use this to compute baseline p-values.
  - 95% Fisher-z confidence intervals on the *observed* Pearson r (cross and within),
    and BH-FDR correction across layers within each panel.

"""

import os
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

# =============================================================================
# CONFIG
# =============================================================================

MODEL_PREFIX = "Llama-3.3-70B-Instruct"
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

# Statistical / plotting configuration
ALPHA = 0.05  # significance threshold
CI_ALPHA = 0.05  # 95% Fisher-z CI for correlations
USE_FDR = True  # apply BH-FDR across layers within each panel
EPS_DENOM = 1e-8  # guard against (near-)constant predictions inflating |r|
D2D_TWO_SIDED = False  # one-sided vs perm-null after training on permuted A labels
D2M_TWO_SIDED = False  # after metric-sign fix, usually want one-sided vs perm-null
PLOT_SHOW_NULL_MEAN = True  # show perm-null mean as a dashed line (diagnostic)
PLOT_SHOW_NULL_BAND = False  # do NOT plot perm-null 5-95 band; use cross/within CIs instead

# Metrics, methods, tasks
METRICS = ["entropy", "logit_gap"]
METHODS = ["mean_diff", "probe"]
META_TASKS = ["delegate", "second_chance"]

# Train/test splits and permutation settings
WITHIN_TRAIN_SPLIT = 0.8
N_PERMUTATIONS = 200
SEED = 42

# Mean-diff and probe settings
MEAN_DIFF_QUANTILE = 0.2
PROBE_ALPHA = 1.0
PROBE_PCA_COMPONENTS = 256

# =============================================================================
# STATISTICS HELPERS
# =============================================================================


def _safe_pearson_r(preds: np.ndarray, y: np.ndarray, eps: float = EPS_DENOM) -> float:
    """Fast, numerically-guarded Pearson r (returns 0.0 if degenerate)."""
    preds = np.asarray(preds, dtype=np.float32).reshape(-1)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    if preds.size != y.size or preds.size < 2:
        return 0.0
    pc = preds - preds.mean()
    yc = y - y.mean()
    denom = float(np.linalg.norm(pc) * np.linalg.norm(yc))
    if denom < eps:
        return 0.0
    r = float(np.dot(pc, yc) / denom)
    if not np.isfinite(r):
        return 0.0
    return max(-1.0, min(1.0, r))


def _fisher_ci(r: float, n: int, alpha: float = CI_ALPHA):
    """Approximate CI for Pearson r via Fisher z-transform."""
    if n is None or n <= 3:
        return (float("nan"), float("nan"))
    r_clip = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r_clip)
    se = 1.0 / np.sqrt(n - 3)

    zcrit = 1.959963984540054  # ~= norm.ppf(0.975)
    if abs(alpha - 0.05) > 1e-12:
        try:
            from scipy.stats import norm

            zcrit = float(norm.ppf(1 - alpha / 2))
        except Exception:
            zcrit = 1.959963984540054

    z_lo = z - zcrit * se
    z_hi = z + zcrit * se
    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction (returns q-values)."""
    pvals = np.asarray(pvals, dtype=np.float64)
    m = pvals.size
    if m == 0:
        return pvals
    order = np.argsort(pvals)
    ranked = pvals[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty_like(q)
    out[order] = q
    return out


def _vectorized_null_rs(preds_perm: np.ndarray, y: np.ndarray, eps: float = EPS_DENOM) -> np.ndarray:
    """Compute Pearson r for many prediction vectors vs y, with guards for degeneracy."""
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    y_centered = y - y.mean()
    y_norm = np.linalg.norm(y_centered)
    if y_norm < eps:
        return np.zeros(preds_perm.shape[1], dtype=np.float32)

    P = np.asarray(preds_perm, dtype=np.float32)
    P_centered = P - P.mean(axis=0, keepdims=True)
    P_norms = np.linalg.norm(P_centered, axis=0)
    denom = P_norms * y_norm

    dot = P_centered.T @ y_centered
    rs = np.zeros(P.shape[1], dtype=np.float32)
    valid = denom >= eps
    rs[valid] = dot[valid] / denom[valid]
    rs = np.clip(rs, -1.0, 1.0)
    rs[~np.isfinite(rs)] = 0.0
    return rs


# =============================================================================
# OPTIMIZED DIRECTION COMPUTATION
# =============================================================================


def compute_mean_diff_direction_vectorized(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_perms: int,
    rng: Optional[np.random.Generator],
    quantile: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean-difference direction and permuted-label versions."""
    X_train = X_train.astype(np.float32)
    y_train = np.asarray(y_train, dtype=np.float32)

    # Real direction
    thresh_low = np.quantile(y_train, quantile)
    thresh_high = np.quantile(y_train, 1 - quantile)
    low_mask = y_train <= thresh_low
    high_mask = y_train >= thresh_high

    mean_low = X_train[low_mask].mean(axis=0)
    mean_high = X_train[high_mask].mean(axis=0)
    real_dir = mean_high - mean_low
    real_dir = real_dir / (np.linalg.norm(real_dir) + 1e-12)

    # Permuted-label directions
    perm_dirs = []
    if n_perms > 0:
        assert rng is not None
        for _ in range(n_perms):
            y_perm = rng.permutation(y_train)
            tl = np.quantile(y_perm, quantile)
            th = np.quantile(y_perm, 1 - quantile)
            lm = y_perm <= tl
            hm = y_perm >= th
            d = X_train[hm].mean(axis=0) - X_train[lm].mean(axis=0)
            d = d / (np.linalg.norm(d) + 1e-12)
            perm_dirs.append(d)

    perm_dirs = np.stack(perm_dirs, axis=0) if len(perm_dirs) else np.zeros((0, X_train.shape[1]), dtype=np.float32)
    return real_dir.astype(np.float32), perm_dirs.astype(np.float32)


class ProbeDirectionManager:
    """Fit scaler+PCA once per layer, then quickly fit Ridge probes for real/permuted labels."""

    def __init__(self, X_train: np.ndarray, alpha: float, n_components: int):
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = self.scaler.fit_transform(X_train)
        self.pca = PCA(n_components=min(n_components, Xs.shape[1]))
        self.X_train_reduced = self.pca.fit_transform(Xs).astype(np.float32)
        self.alpha = alpha

    def compute_directions(
        self,
        y_train: np.ndarray,
        n_perms: int,
        rng: Optional[np.random.Generator],
    ) -> Tuple[np.ndarray, np.ndarray]:
        y_train = np.asarray(y_train, dtype=np.float32)

        # Real
        ridge = Ridge(alpha=self.alpha)
        ridge.fit(self.X_train_reduced, y_train)
        w = ridge.coef_.astype(np.float32)
        # Map back to original space
        dir_reduced = w / (np.linalg.norm(w) + 1e-12)
        dir_orig = self.pca.components_.T @ dir_reduced
        dir_orig = dir_orig / (np.linalg.norm(dir_orig) + 1e-12)

        perm_dirs = []
        if n_perms > 0:
            assert rng is not None
            for _ in range(n_perms):
                y_perm = rng.permutation(y_train)
                ridge_p = Ridge(alpha=self.alpha)
                ridge_p.fit(self.X_train_reduced, y_perm)
                w_p = ridge_p.coef_.astype(np.float32)
                d_red = w_p / (np.linalg.norm(w_p) + 1e-12)
                d_orig = self.pca.components_.T @ d_red
                d_orig = d_orig / (np.linalg.norm(d_orig) + 1e-12)
                perm_dirs.append(d_orig)

        perm_dirs = np.stack(perm_dirs, axis=0) if len(perm_dirs) else np.zeros((0, dir_orig.shape[0]), dtype=np.float32)
        return dir_orig.astype(np.float32), perm_dirs.astype(np.float32)


# =============================================================================
# DATA LOADING
# =============================================================================


def metric_sign_for_confidence(metric: str) -> float:
    """Match prior convention: sign so positive correlation = 'good transfer'."""
    # For entropy, higher entropy -> lower confidence, so flip.
    if metric == "entropy":
        return -1.0
    # For logit_gap, higher gap -> higher confidence, so keep.
    return 1.0


def load_data_package(dataset_name: str, metric: str) -> Dict:
    """Load .npz for dataset and build (layer->acts) dicts for MC and each meta task."""
    fpath = DATA_DIR / f"{MODEL_PREFIX}_{dataset_name}_{metric}.npz"
    data = np.load(fpath, allow_pickle=True)

    # MC activations
    mc_acts = {}
    # If MC activations are saved for multiple token positions (e.g. layer_12_final),
    # filter to *_final by default to avoid silently overwriting per-layer arrays.
    mc_has_pos = any(k.startswith('layer_') and len(k.split('_')) > 2 for k in data.files)
    for key in data.files:
        if not key.startswith('layer_'):
            continue
        if mc_has_pos and not key.endswith('_final'):
            continue
        layer = int(key.split('_')[1])
        mc_acts[layer] = data[key]

    mc_metric = data[metric]

    # Meta tasks
    meta = {}
    for task in META_TASKS:
        acts = {}
        for key in data.files:
            if key.startswith(f"{task}_layer_"):
                parts = key.split("_")
                # {task}_layer_{layer}_{pos}
                layer = int(parts[2])
                pos = parts[3] if len(parts) > 3 else "final"
                if pos == "final":
                    acts[layer] = data[key]

        conf = data[f"{task}_confidence"]
        meta[task] = {"acts": acts, "conf": conf}

    return {"mc_acts": mc_acts, "mc_metric": mc_metric, "meta": meta}


# =============================================================================
# EVALUATION
# =============================================================================


def evaluate_transfer(
    target_acts: np.ndarray,
    target_vals: np.ndarray,
    real_dir: np.ndarray,
    perm_dirs: np.ndarray,
    two_sided: bool,
):
    """Evaluate observed r + CI and a training-procedure perm-null p-value."""
    target_acts = target_acts.astype(np.float32)
    target_vals = np.asarray(target_vals, dtype=np.float32)
    n = int(target_vals.shape[0])

    preds = target_acts @ real_dir
    cross_r = _safe_pearson_r(preds, target_vals)
    cross_ci_low, cross_ci_high = _fisher_ci(cross_r, n, alpha=CI_ALPHA)

    try:
        _, cross_p = pearsonr(np.asarray(preds, dtype=np.float64), np.asarray(target_vals, dtype=np.float64))
        cross_p = float(cross_p) if np.isfinite(cross_p) else 1.0
    except Exception:
        cross_p = 1.0

    if perm_dirs is None or len(perm_dirs) == 0:
        p_value = float("nan")
        null_mean = null_std = null_p5 = null_p95 = float("nan")
        null_n = 0
    else:
        preds_perm = target_acts @ perm_dirs.T
        null_rs = _vectorized_null_rs(preds_perm, target_vals, eps=EPS_DENOM)
        if two_sided:
            p_value = (np.sum(np.abs(null_rs) >= abs(cross_r)) + 1) / (len(null_rs) + 1)
        else:
            p_value = (np.sum(null_rs >= cross_r) + 1) / (len(null_rs) + 1)

        null_mean = float(np.mean(null_rs))
        null_std = float(np.std(null_rs))
        null_p5 = float(np.percentile(null_rs, 5))
        null_p95 = float(np.percentile(null_rs, 95))
        null_n = int(len(null_rs))

    return {
        "cross_r": float(cross_r),
        "cross_ci_low": float(cross_ci_low),
        "cross_ci_high": float(cross_ci_high),
        "cross_p": float(cross_p),
        "p_value": float(p_value) if p_value == p_value else float("nan"),
        "null_stats": {
            "mean": null_mean,
            "std": null_std,
            "p5": null_p5,
            "p95": null_p95,
            "n": null_n,
        },
    }


def get_within_baseline(X_train, y_train, X_test, y_test, method):
    """Computes within-dataset baseline (B->B) and Fisher-z CI on the test correlation."""
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = np.asarray(y_test, dtype=np.float32)
    n = int(y_test.shape[0])

    if method == "mean_diff":
        dr, _ = compute_mean_diff_direction_vectorized(X_train, y_train, 0, None, MEAN_DIFF_QUANTILE)
    else:
        mgr = ProbeDirectionManager(X_train, PROBE_ALPHA, PROBE_PCA_COMPONENTS)
        dr, _ = mgr.compute_directions(y_train, 0, None)

    preds = X_test @ dr
    r = _safe_pearson_r(preds, y_test)
    ci_low, ci_high = _fisher_ci(r, n, alpha=CI_ALPHA)
    return {"within_r": float(r), "within_ci_low": float(ci_low), "within_ci_high": float(ci_high)}


# =============================================================================
# PLOTTING
# =============================================================================


def plot_transfer_results(d2d_results, d2m_results, ds_a, ds_b, out_path: Path):
    """Make 2x4 grid plot for D2D and D2M for a given dataset pair."""
    # Be robust to missing/empty metrics (avoid early-return if the first metric is empty).
    metrics = sorted([m for m in d2d_results.keys() if d2d_results.get(m)])
    methods = sorted({meth for m in metrics for meth in d2d_results[m].keys()})
    if not metrics or not methods:
        return

    fig, axes = plt.subplots(2, 4, figsize=(20, 8), sharex=True, sharey=False)
    fig.suptitle(f"Cross-Dataset Transfer: {ds_a} -> {ds_b}", fontsize=16)

    # Top row: D2D
    for j, metric in enumerate(metrics):
        for i, method in enumerate(methods):
            ax = axes[0, j * 2 + i]
            res = d2d_results[metric][method]
            _plot_single(ax, res, f"d2d: {metric} ({method})")

    # Bottom row: D2M (use first available task per metric unless passed already flattened)
    for j, metric in enumerate(metrics):
        for i, method in enumerate(methods):
            ax = axes[1, j * 2 + i]
            res = d2m_results.get(metric, {}).get(method)
            if res is None:
                ax.set_title(f"d2m: {metric} ({method}) - No data")
                continue
            _plot_single(ax, res, f"d2m: {metric} ({method})")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _plot_single(ax, results, title):
    per_layer = results.get("per_layer", {})
    if not per_layer:
        ax.set_title(f"{title} - No data")
        return

    layers = sorted(per_layer.keys())
    cross_r = np.array([per_layer[l].get("cross_r", np.nan) for l in layers], dtype=np.float32)
    within_r = np.array([per_layer[l].get("within_r", np.nan) for l in layers], dtype=np.float32)

    cross_lo = np.array([per_layer[l].get("cross_ci_low", np.nan) for l in layers], dtype=np.float32)
    cross_hi = np.array([per_layer[l].get("cross_ci_high", np.nan) for l in layers], dtype=np.float32)
    within_lo = np.array([per_layer[l].get("within_ci_low", np.nan) for l in layers], dtype=np.float32)
    within_hi = np.array([per_layer[l].get("within_ci_high", np.nan) for l in layers], dtype=np.float32)

    ax.fill_between(layers, within_lo, within_hi, alpha=0.18, color="green", label="Within 95% CI")
    ax.fill_between(layers, cross_lo, cross_hi, alpha=0.18, color="blue", label="Cross 95% CI")

    ax.plot(layers, within_r, "-", color="green", lw=2, alpha=0.8, label="Within (B->B)")
    ax.plot(layers, cross_r, "-", color="blue", lw=2, label="Cross (A->B)")

    if PLOT_SHOW_NULL_MEAN and all("null_mean" in per_layer[l] for l in layers):
        null_mean = np.array([per_layer[l].get("null_mean", np.nan) for l in layers], dtype=np.float32)
        ax.plot(layers, null_mean, "--", color="gray", lw=1, label="Perm-null mean")

    if PLOT_SHOW_NULL_BAND and all(("null_5th" in per_layer[l] and "null_95th" in per_layer[l]) for l in layers):
        null_5 = np.array([per_layer[l].get("null_5th", np.nan) for l in layers], dtype=np.float32)
        null_95 = np.array([per_layer[l].get("null_95th", np.nan) for l in layers], dtype=np.float32)
        ax.fill_between(layers, null_5, null_95, alpha=0.25, color="gray", label="Perm-null 5-95%")

    p_key = "p_value_fdr" if USE_FDR and any("p_value_fdr" in per_layer[l] for l in layers) else "p_value"
    sig_layers = [l for l in layers if float(per_layer[l].get(p_key, 1.0)) < ALPHA]
    if sig_layers:
        sig_rs = [float(per_layer[l].get("cross_r", np.nan)) for l in sig_layers]
        ax.scatter(sig_layers, sig_rs, color="red", s=28, zorder=5, label=f"{p_key} < {ALPHA:g}")

    ax.axhline(0, color="black", ls=":", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_synthesis(all_d2d, all_d2m, out_path: Path):
    """Optional synthesis plot across pairs (mean cross/within by layer)."""
    # Keep existing behavior (mean ± std across pairs) for now.
    pass


# =============================================================================
# MAIN
# =============================================================================


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = ["SimpleMC", "TriviaMC"]
    pairs = [(datasets[0], datasets[1])]

    rng = np.random.default_rng(SEED)

    all_d2d = {}
    all_d2m = {}

    for ds_a, ds_b in pairs:
        pair_key = f"{ds_a}_vs_{ds_b}"
        print(f"\nProcessing {pair_key}...")

        all_d2d[pair_key] = {}
        all_d2m[pair_key] = {}

        for metric in METRICS:
            data_A = load_data_package(ds_a, metric)
            data_B = load_data_package(ds_b, metric)

            # For fair comparisons across D2D/D2M we restrict to the minimum length
            # that exists across B's MC metric and all meta confidences.
            len_B = len(data_B["mc_metric"])
            for task in META_TASKS:
                len_B = min(len_B, len(data_B["meta"][task]["conf"]))

            all_d2d[pair_key][metric] = {}
            all_d2m[pair_key][metric] = {}

            # Train/test split indices (shared across layers)
            idx_A = np.arange(len(data_A["mc_metric"]))
            idx_B = np.arange(len_B)
            train_idx_A, test_idx_A = train_test_split(idx_A, train_size=WITHIN_TRAIN_SPLIT, random_state=SEED, shuffle=True)
            train_idx_B, test_idx_B = train_test_split(idx_B, train_size=WITHIN_TRAIN_SPLIT, random_state=SEED, shuffle=True)

            for method in METHODS:
                res_d2d = {"per_layer": {}}
                res_d2m = {task: {"per_layer": {}} for task in META_TASKS}

                layers = sorted(set(data_A["mc_acts"].keys()) & set(data_B["mc_acts"].keys()))

                for layer in tqdm(layers, desc=f"{pair_key} {metric} {method}"):
                    X_A_full = data_A["mc_acts"][layer]
                    y_A_full = data_A["mc_metric"]
                    X_A_train = X_A_full[train_idx_A]
                    y_A_train = y_A_full[train_idx_A]

                    # STEP 1: directions on A_train
                    if method == "mean_diff":
                        real_dir, perm_dirs = compute_mean_diff_direction_vectorized(
                            X_A_train, y_A_train, N_PERMUTATIONS, rng, MEAN_DIFF_QUANTILE
                        )
                    else:
                        mgr = ProbeDirectionManager(X_A_train, PROBE_ALPHA, PROBE_PCA_COMPONENTS)
                        real_dir, perm_dirs = mgr.compute_directions(y_A_train, N_PERMUTATIONS, rng)

                    # STEP 2: D2D
                    X_B = data_B["mc_acts"][layer][:len_B]
                    X_B_train, X_B_test = X_B[train_idx_B], X_B[test_idx_B]
                    y_B_train, y_B_test = data_B["mc_metric"][train_idx_B], data_B["mc_metric"][test_idx_B]

                    d2d_eval = evaluate_transfer(X_B_test, y_B_test, real_dir, perm_dirs, two_sided=D2D_TWO_SIDED)
                    within_eval = get_within_baseline(X_B_train, y_B_train, X_B_test, y_B_test, method)
                    within_r = within_eval["within_r"]

                    transfer_eff = d2d_eval["cross_r"] / within_r if abs(within_r) > 1e-6 else 0.0

                    res_d2d["per_layer"][layer] = {
                        "cross_r": d2d_eval["cross_r"],
                        "cross_ci_low": d2d_eval["cross_ci_low"],
                        "cross_ci_high": d2d_eval["cross_ci_high"],
                        "cross_p": d2d_eval["cross_p"],
                        "within_r": within_r,
                        "within_ci_low": within_eval["within_ci_low"],
                        "within_ci_high": within_eval["within_ci_high"],
                        "transfer_efficiency": float(transfer_eff),
                        "p_value": d2d_eval["p_value"],
                        "null_mean": d2d_eval["null_stats"]["mean"],
                        "null_std": d2d_eval["null_stats"]["std"],
                        "null_5th": d2d_eval["null_stats"]["p5"],
                        "null_95th": d2d_eval["null_stats"]["p95"],
                        "null_n": d2d_eval["null_stats"]["n"],
                    }

                    # STEP 3: D2M
                    msign = metric_sign_for_confidence(metric)
                    real_dir_m = real_dir * msign
                    perm_dirs_m = perm_dirs * msign if perm_dirs is not None else None

                    # Within baseline direction for D2M (compute once per layer)
                    if method == "mean_diff":
                        db, _ = compute_mean_diff_direction_vectorized(X_B_train, y_B_train, 0, None, MEAN_DIFF_QUANTILE)
                    else:
                        mgr_b = ProbeDirectionManager(X_B_train, PROBE_ALPHA, PROBE_PCA_COMPONENTS)
                        db, _ = mgr_b.compute_directions(y_B_train, 0, None)

                    for task in META_TASKS:
                        meta_acts = data_B["meta"][task]["acts"].get(layer)
                        if meta_acts is None:
                            continue
                        X_Meta = meta_acts[:len_B].astype(np.float32)
                        X_Meta_test = X_Meta[test_idx_B]
                        conf_test = np.asarray(data_B["meta"][task]["conf"][:len_B][test_idx_B], dtype=np.float32)

                        d2m_eval = evaluate_transfer(X_Meta_test, conf_test, real_dir_m, perm_dirs_m, two_sided=D2M_TWO_SIDED)

                        within_preds = (X_Meta_test @ db) * msign
                        within_d2m_r = _safe_pearson_r(within_preds, conf_test)
                        within_ci_low, within_ci_high = _fisher_ci(within_d2m_r, int(len(conf_test)), alpha=CI_ALPHA)

                        transfer_eff_m = d2m_eval["cross_r"] / within_d2m_r if abs(within_d2m_r) > 1e-6 else 0.0

                        res_d2m[task]["per_layer"][layer] = {
                            "cross_r": d2m_eval["cross_r"],
                            "cross_ci_low": d2m_eval["cross_ci_low"],
                            "cross_ci_high": d2m_eval["cross_ci_high"],
                            "cross_p": d2m_eval["cross_p"],
                            "within_r": float(within_d2m_r),
                            "within_ci_low": float(within_ci_low),
                            "within_ci_high": float(within_ci_high),
                            "transfer_efficiency": float(transfer_eff_m),
                            "p_value": d2m_eval["p_value"],
                            "null_mean": d2m_eval["null_stats"]["mean"],
                            "null_std": d2m_eval["null_stats"]["std"],
                            "null_5th": d2m_eval["null_stats"]["p5"],
                            "null_95th": d2m_eval["null_stats"]["p95"],
                            "null_n": d2m_eval["null_stats"]["n"],
                        }

                # Summaries + FDR (panel-wise across layers)
                d2d_layers = sorted(res_d2d["per_layer"].keys())
                if d2d_layers:
                    p_raw = np.array([res_d2d["per_layer"][l].get("p_value", 1.0) for l in d2d_layers], dtype=np.float64)
                    if USE_FDR:
                        q = _bh_fdr(p_raw)
                        for l, qv in zip(d2d_layers, q):
                            res_d2d["per_layer"][l]["p_value_fdr"] = float(qv)
                        sig_layers = [l for l, qv in zip(d2d_layers, q) if qv < ALPHA]
                    else:
                        sig_layers = [l for l, pv in zip(d2d_layers, p_raw) if pv < ALPHA]
                else:
                    sig_layers = []
                res_d2d["n_significant"] = len(sig_layers)
                res_d2d["significant_layers"] = sig_layers
                if sig_layers:
                    best = max(sig_layers, key=lambda l: res_d2d["per_layer"][l]["cross_r"])
                    res_d2d["best_layer"] = best
                    res_d2d["best_cross_r"] = res_d2d["per_layer"][best]["cross_r"]
                else:
                    res_d2d["best_layer"] = None
                    res_d2d["best_cross_r"] = None

                all_d2d[pair_key][metric][method] = res_d2d

                for task in META_TASKS:
                    task_layers = sorted(res_d2m[task]["per_layer"].keys())
                    if task_layers:
                        p_raw = np.array([res_d2m[task]["per_layer"][l].get("p_value", 1.0) for l in task_layers], dtype=np.float64)
                        if USE_FDR:
                            q = _bh_fdr(p_raw)
                            for l, qv in zip(task_layers, q):
                                res_d2m[task]["per_layer"][l]["p_value_fdr"] = float(qv)
                            sig_layers = [l for l, qv in zip(task_layers, q) if qv < ALPHA]
                        else:
                            sig_layers = [l for l, pv in zip(task_layers, p_raw) if pv < ALPHA]
                    else:
                        sig_layers = []
                    res_d2m[task]["n_significant"] = len(sig_layers)
                    res_d2m[task]["significant_layers"] = sig_layers
                    if sig_layers:
                        best = max(sig_layers, key=lambda l: res_d2m[task]["per_layer"][l]["cross_r"])
                        res_d2m[task]["best_layer"] = best
                        res_d2m[task]["best_cross_r"] = res_d2m[task]["per_layer"][best]["cross_r"]
                    else:
                        res_d2m[task]["best_layer"] = None
                        res_d2m[task]["best_cross_r"] = None

                    if task not in all_d2m[pair_key][metric]:
                        all_d2m[pair_key][metric][task] = {}
                    all_d2m[pair_key][metric][task][method] = res_d2m[task]

        # Plotting - flatten d2m structure for plotting (use first task per metric)
        plot_path = OUTPUT_DIR / f"{MODEL_PREFIX}_{pair_key}_transfer.png"
        d2m_for_plot = {m: {} for m in METRICS}
        for m in METRICS:
            if m in all_d2m[pair_key]:
                for task, task_data in all_d2m[pair_key][m].items():
                    for meth, res in task_data.items():
                        d2m_for_plot[m][meth] = res
                    break

        plot_transfer_results(all_d2d[pair_key], d2m_for_plot, ds_a, ds_b, plot_path)

    results = {
        "config": {
            "model_prefix": MODEL_PREFIX,
            "datasets": datasets,
            "metrics": METRICS,
            "methods": METHODS,
            "meta_tasks": META_TASKS,
            "n_permutations": N_PERMUTATIONS,
            "seed": SEED,
            "within_train_split": WITHIN_TRAIN_SPLIT,
            "alpha": ALPHA,
            "ci_alpha": CI_ALPHA,
            "use_fdr": USE_FDR,
            "eps_denom": EPS_DENOM,
            "d2d_two_sided": D2D_TWO_SIDED,
            "d2m_two_sided": D2M_TWO_SIDED,
            "plot_show_null_mean": PLOT_SHOW_NULL_MEAN,
            "plot_show_null_band": PLOT_SHOW_NULL_BAND,
            "note": (
                "Cross p-values are against the training-procedure perm-null (permute labels on A). "
                "CIs are Fisher-z for observed r. BH-FDR is applied across layers within each panel when USE_FDR=True."
            ),
        },
        "d2d": all_d2d,
        "d2m": all_d2m,
    }

    with open(OUTPUT_DIR / f"{MODEL_PREFIX}_cross_dataset_transfer.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
