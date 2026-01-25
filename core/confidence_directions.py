"""
Meta-Judgment Confidence Direction Finding.

Trains regression probes to predict stated confidence from meta-task activations.
This finds directions that encode the model's expressed confidence, which may
differ from directions that encode actual uncertainty.

Two methods (matching uncertainty direction finding):
1. probe: Ridge regression to find direction that best predicts stated confidence
2. mean_diff: mean(high_confidence_samples) - mean(low_confidence_samples)

Key comparisons:
- Uncertainty direction (from direct task): predicts actual entropy/logit_gap
- Confidence direction (from meta task): predicts stated confidence

If genuine introspection exists, these directions should be correlated.
If the model pattern-matches without introspecting, they may differ.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from .directions import _as_float32, _safe_scale, _sanitize_r2, mean_diff_direction


def _bootstrap_r2_percentiles(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    chunk_boot: int = 2048,
) -> Tuple[float, float]:
    """
    Bootstrap 2.5th and 97.5th percentiles of R² by resampling examples WITH replacement.
    No refitting: y_pred is treated as fixed predictions on the fixed test set.
    Returns (ci_low, ci_high) for 95% CI.

    Percentile CIs are preferred over mean±std because R² is bounded and
    the bootstrap distribution is often skewed.
    """
    if n_bootstrap <= 0:
        return float("nan"), float("nan")

    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    n = y_true.shape[0]

    r2s = np.empty(n_bootstrap, dtype=np.float32)
    done = 0
    while done < n_bootstrap:
        b = min(chunk_boot, n_bootstrap - done)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        y_t = y_true[idx]  # (b, n)
        y_p = y_pred[idx]  # (b, n)
        ss_res = ((y_t - y_p) ** 2).sum(axis=1)
        ss_tot = ((y_t - y_t.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
        # Avoid div by zero
        ss_tot = np.where(ss_tot > 0, ss_tot, 1.0)
        r2_raw = 1.0 - ss_res / ss_tot
        # Clip to [-1, 1] to avoid extreme values from low-variance samples
        r2s[done:done + b] = np.clip(r2_raw, -1.0, 1.0)
        done += b

    return float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5))


def train_confidence_probe(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 1000.0,
    n_components: int = 100,
) -> Tuple[StandardScaler, PCA, Ridge]:
    """
    Train a Ridge regression probe to predict stated confidence from meta activations.

    Args:
        X: (n_samples, hidden_dim) meta-task activations
        y: (n_samples,) stated confidence values (0-1 scale)
        alpha: Ridge regularization strength
        n_components: PCA components

    Returns:
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        probe: Fitted Ridge regressor
    """
    X = _as_float32(X)

    # Standardize
    scaler = StandardScaler()
    scaler.fit(X)
    scaler.scale_ = _safe_scale(scaler.scale_)
    X_scaled = scaler.transform(X)

    # PCA
    n_comp = min(n_components, X.shape[0], X.shape[1])
    pca = PCA(n_components=n_comp)
    X_pca = pca.fit_transform(X_scaled)

    # Ridge regression
    probe = Ridge(alpha=alpha)
    probe.fit(X_pca, y)

    return scaler, pca, probe


def extract_confidence_direction(
    scaler: StandardScaler,
    pca: PCA,
    probe: Ridge
) -> np.ndarray:
    """
    Extract direction vector from a confidence probe.

    Maps the probe coefficients back through PCA and standardization
    to original activation space.

    Args:
        scaler: Fitted StandardScaler from training
        pca: Fitted PCA from training
        probe: Fitted Ridge regressor

    Returns:
        direction: (hidden_dim,) normalized direction vector
    """
    # Project probe coefficients back to original space
    direction = pca.inverse_transform(probe.coef_.reshape(1, -1)).flatten()

    # Undo standardization scaling
    direction = direction / scaler.scale_

    # Normalize to unit length
    direction = direction / np.linalg.norm(direction)

    return direction.astype(np.float32)


def evaluate_confidence_probe(
    X: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    probe: Ridge
) -> Dict:
    """
    Evaluate a confidence probe on held-out data.

    Args:
        X: (n_samples, hidden_dim) activations
        y: (n_samples,) ground truth confidence values
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        probe: Fitted Ridge regressor

    Returns:
        Dict with r2, mae, pearson, spearman, predictions
    """
    X = _as_float32(X)
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)

    y_pred = probe.predict(X_pca)

    r2 = _sanitize_r2(r2_score(y, y_pred))
    mae = mean_absolute_error(y, y_pred)
    pearson, _ = pearsonr(y, y_pred)
    spearman, _ = spearmanr(y, y_pred)

    return {
        "r2": float(r2),
        "mae": float(mae),
        "pearson": float(pearson),
        "spearman": float(spearman),
        "predictions": y_pred
    }


def find_confidence_directions(
    meta_activations_by_layer: Dict[int, np.ndarray],
    stated_confidences: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float = 1000.0,
    n_components: int = 100,
) -> Dict:
    """
    Find confidence directions for all layers.

    Args:
        meta_activations_by_layer: {layer_idx: (n_samples, hidden_dim)}
        stated_confidences: (n_samples,) confidence values (0-1 scale)
        train_idx: Indices for training
        test_idx: Indices for testing
        alpha: Ridge regularization strength
        n_components: PCA components

    Returns:
        {
            "directions": {layer: direction_vector},
            "probes": {layer: {"scaler", "pca", "probe"}},
            "fits": {layer: {"train_r2", "test_r2", "test_pearson", ...}}
        }
    """
    layers = sorted(meta_activations_by_layer.keys())
    y = np.asarray(stated_confidences)

    results = {
        "directions": {},
        "probes": {},
        "fits": {}
    }

    for layer in tqdm(layers, desc="Training confidence probes"):
        X = meta_activations_by_layer[layer]

        # Split
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train probe
        scaler, pca, probe = train_confidence_probe(
            X_train, y_train,
            alpha=alpha,
            n_components=n_components
        )

        # Evaluate on train and test
        train_result = evaluate_confidence_probe(X_train, y_train, scaler, pca, probe)
        test_result = evaluate_confidence_probe(X_test, y_test, scaler, pca, probe)

        # Extract direction
        direction = extract_confidence_direction(scaler, pca, probe)

        # Shuffled baseline - use its own scaler/pca for correct feature space
        rng_shuf = np.random.default_rng(42 + layer)
        y_shuffled = rng_shuf.permutation(y_train)
        scaler_shuf, pca_shuf, probe_shuffled = train_confidence_probe(
            X_train, y_shuffled,
            alpha=alpha,
            n_components=n_components
        )
        X_test_shuf = pca_shuf.transform(scaler_shuf.transform(_as_float32(X_test)))
        shuffled_preds = probe_shuffled.predict(X_test_shuf)
        shuffled_r2 = _sanitize_r2(r2_score(y_test, shuffled_preds))

        results["directions"][layer] = direction
        results["probes"][layer] = {
            "scaler": scaler,
            "pca": pca,
            "probe": probe
        }
        results["fits"][layer] = {
            "train_r2": train_result["r2"],
            "test_r2": test_result["r2"],
            "test_mae": test_result["mae"],
            "test_pearson": test_result["pearson"],
            "test_spearman": test_result["spearman"],
            "shuffled_r2": float(shuffled_r2),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        }

    return results


def find_confidence_directions_both_methods(
    meta_activations_by_layer: Dict[int, np.ndarray],
    stated_confidences: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    alpha: float = 1000.0,
    n_components: int = 100,
    mean_diff_quantile: float = 0.25,
    n_bootstrap: int = 0,
    train_split: float = 0.8,
    seed: int = 42,
) -> Dict:
    """
    Find confidence directions using both methods (matching uncertainty directions).

    Methods:
    - "probe": Ridge regression to predict stated confidence
    - "mean_diff": mean(top 25% confidence) - mean(bottom 25% confidence)

    Args:
        meta_activations_by_layer: {layer_idx: (n_samples, hidden_dim)}
        stated_confidences: (n_samples,) confidence values (0-1 scale)
        train_idx: Indices for training
        test_idx: Indices for testing
        alpha: Ridge regularization strength
        n_components: PCA components for probe
        mean_diff_quantile: Quantile for mean_diff method (default 0.25)
        n_bootstrap: Number of bootstrap iterations for confidence intervals (0 = no bootstrap)
        train_split: Train/test split ratio for bootstrap
        seed: Random seed for bootstrap

    Returns:
        {
            "directions": {"probe": {layer: dir}, "mean_diff": {layer: dir}},
            "probes": {layer: {"scaler", "pca", "probe"}},
            "fits": {"probe": {layer: metrics}, "mean_diff": {layer: metrics}},
            "comparison": {layer: {"cosine_sim": float}}
        }
        If n_bootstrap > 0, fits will include "test_r2_std" for each method.
    """
    layers = sorted(meta_activations_by_layer.keys())
    y = np.asarray(stated_confidences)
    n_samples = len(y)

    # NOTE: Bootstrap is done by resampling test predictions, NOT refitting.
    # This is orders of magnitude faster and gives valid confidence intervals.

    results = {
        "directions": {"probe": {}, "mean_diff": {}},
        "probes": {},
        "fits": {"probe": {}, "mean_diff": {}},
        "comparison": {}
    }

    for layer in tqdm(layers, desc="Training confidence probes"):
        X = meta_activations_by_layer[layer]

        # Split
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Method 1: Probe-based direction
        scaler, pca, probe = train_confidence_probe(
            X_train, y_train,
            alpha=alpha,
            n_components=n_components
        )

        # Evaluate probe
        train_result = evaluate_confidence_probe(X_train, y_train, scaler, pca, probe)
        test_result = evaluate_confidence_probe(X_test, y_test, scaler, pca, probe)

        probe_dir = extract_confidence_direction(scaler, pca, probe)

        # Shuffled baseline for probe - use its own scaler/pca for correct feature space
        rng_shuf = np.random.default_rng(seed + 12_345 + layer)
        y_shuffled = rng_shuf.permutation(y_train)
        scaler_shuf, pca_shuf, probe_shuffled = train_confidence_probe(
            X_train, y_shuffled,
            alpha=alpha,
            n_components=n_components
        )
        X_test_shuf = pca_shuf.transform(scaler_shuf.transform(_as_float32(X_test)))
        shuffled_preds = probe_shuffled.predict(X_test_shuf)
        shuffled_r2 = _sanitize_r2(r2_score(y_test, shuffled_preds))

        probe_fit = {
            "train_r2": train_result["r2"],
            "test_r2": test_result["r2"],
            "test_mae": test_result["mae"],
            "test_pearson": test_result["pearson"],
            "test_spearman": test_result["spearman"],
            "shuffled_r2": float(shuffled_r2),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        }

        # Bootstrap confidence intervals for probe (fast: resample predictions, not refit)
        if n_bootstrap > 0:
            rng = np.random.default_rng(seed + 10_000 + layer)
            ci_low, ci_high = _bootstrap_r2_percentiles(y_test, test_result["predictions"], n_bootstrap, rng)
            probe_fit["test_r2_ci_low"] = ci_low
            probe_fit["test_r2_ci_high"] = ci_high
            probe_fit["n_bootstrap"] = int(n_bootstrap)

        results["directions"]["probe"][layer] = probe_dir
        results["probes"][layer] = {
            "scaler": scaler,
            "pca": pca,
            "probe": probe
        }
        results["fits"]["probe"][layer] = probe_fit

        # Method 2: Mean-diff direction
        # Use the function from directions.py with train/test split
        mean_diff_dir, mean_diff_info = mean_diff_direction(
            X_train, y_train,
            quantile=mean_diff_quantile,
        )

        # Evaluate mean_diff on test set
        test_projections = X_test @ mean_diff_dir
        test_corr, _ = pearsonr(y_test, test_projections)
        if not np.isfinite(test_corr):
            test_corr = 0.0

        # Use OLS mapping from projection -> y so test_r2 = r2_score(y, y_pred_md)
        # OLS slope = corr(y, proj) * std(y) / std(proj)
        proj_mean, y_mean = test_projections.mean(), y_test.mean()
        proj_std, y_std = test_projections.std(), y_test.std()
        if proj_std > 0 and y_std > 0:
            b = test_corr * (y_std / proj_std)
            y_pred_md = y_mean + (test_projections - proj_mean) * b
        else:
            y_pred_md = np.full_like(y_test, y_mean)

        test_r2 = _sanitize_r2(r2_score(y_test, y_pred_md))
        test_mae_md = float(mean_absolute_error(y_test, y_pred_md))

        mean_diff_fit = {
            "train_r2": mean_diff_info["r2"],
            "test_r2": float(test_r2),
            "test_mae": test_mae_md,
            "test_pearson": float(test_corr),
            "quantile": mean_diff_quantile,
            "n_group": mean_diff_info.get("n_low", mean_diff_info.get("n_group", 0)),
        }

        # Bootstrap confidence intervals for mean_diff (fast: resample predictions, not refit)
        # y_pred_md is already computed with OLS mapping above
        if n_bootstrap > 0:
            rng = np.random.default_rng(seed + 20_000 + layer)
            ci_low, ci_high = _bootstrap_r2_percentiles(y_test, y_pred_md, n_bootstrap, rng)
            mean_diff_fit["test_r2_ci_low"] = ci_low
            mean_diff_fit["test_r2_ci_high"] = ci_high
            mean_diff_fit["n_bootstrap"] = int(n_bootstrap)

        results["directions"]["mean_diff"][layer] = mean_diff_dir
        results["fits"]["mean_diff"][layer] = mean_diff_fit

        # Cosine similarity between methods
        cos_sim = float(np.dot(probe_dir, mean_diff_dir))
        results["comparison"][layer] = {"cosine_sim": cos_sim}

    return results


def compare_confidence_to_uncertainty(
    confidence_directions: Dict[int, np.ndarray],
    uncertainty_directions: Dict[int, np.ndarray]
) -> Dict[int, Dict]:
    """
    Compare confidence directions to uncertainty directions at each layer.

    Args:
        confidence_directions: {layer: direction} from meta-task
        uncertainty_directions: {layer: direction} from direct task

    Returns:
        {layer: {"cosine_similarity": float}}
    """
    results = {}

    layers = sorted(set(confidence_directions.keys()) & set(uncertainty_directions.keys()))

    for layer in layers:
        conf_dir = confidence_directions[layer]
        unc_dir = uncertainty_directions[layer]

        # Normalize (should already be normalized, but be safe)
        conf_dir = conf_dir / np.linalg.norm(conf_dir)
        unc_dir = unc_dir / np.linalg.norm(unc_dir)

        cosine_sim = float(np.dot(conf_dir, unc_dir))

        results[layer] = {
            "cosine_similarity": cosine_sim,
            "abs_cosine_similarity": abs(cosine_sim),
        }

    return results


def cross_evaluate_directions(
    activations: np.ndarray,
    uncertainty_direction: np.ndarray,
    confidence_direction: np.ndarray,
    actual_uncertainty: np.ndarray,
    stated_confidence: np.ndarray
) -> Dict:
    """
    Cross-evaluate: does uncertainty direction predict confidence and vice versa?

    Args:
        activations: (n_samples, hidden_dim)
        uncertainty_direction: Direction trained on actual uncertainty
        confidence_direction: Direction trained on stated confidence
        actual_uncertainty: (n_samples,) ground truth uncertainty
        stated_confidence: (n_samples,) ground truth stated confidence

    Returns:
        {
            "unc_dir_predicts_unc": r2/corr,
            "unc_dir_predicts_conf": r2/corr,
            "conf_dir_predicts_conf": r2/corr,
            "conf_dir_predicts_unc": r2/corr
        }
    """
    # Project onto each direction
    unc_proj = activations @ uncertainty_direction
    conf_proj = activations @ confidence_direction

    def compute_metrics(projections, targets):
        corr, _ = pearsonr(projections, targets)
        r2 = float(corr ** 2)
        return {"r2": r2, "corr": float(corr)}

    return {
        "unc_dir_predicts_unc": compute_metrics(unc_proj, actual_uncertainty),
        "unc_dir_predicts_conf": compute_metrics(unc_proj, stated_confidence),
        "conf_dir_predicts_conf": compute_metrics(conf_proj, stated_confidence),
        "conf_dir_predicts_unc": compute_metrics(conf_proj, actual_uncertainty),
    }
