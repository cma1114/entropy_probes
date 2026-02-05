"""
Cross-Dataset Transfer Prediction Test (Optimized & Corrected).

Corrections from previous version:
1. Fixes D2M p-value to be two-sided (abs(null) >= abs(obs)).
2. Fixes RNG initialization (outside layer loop) to ensure independent permutations across layers.
3. Restores missing metrics: transfer_efficiency, significant_layers list, best_layer index.
4. Restores Config block and Synthesis plot.

Statistical improvements:
5. Fisher-z confidence intervals on observed Pearson r (cross and within).
6. BH-FDR correction across layers within each panel.
7. Safer Pearson computation with guards against degenerate cases.
"""

from pathlib import Path
from itertools import combinations
import json
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

from core import metric_sign_for_confidence

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_PREFIX = "Llama-3.3-70B-Instruct"
METRICS = ["entropy", "logit_gap"]
METHODS = ["mean_diff", "probe"]
META_TASKS = ["delegate"]
N_PERMUTATIONS = 100
SEED = 42
MEAN_DIFF_QUANTILE = 0.25
PROBE_ALPHA = 1000.0
PROBE_PCA_COMPONENTS = 100
WITHIN_TRAIN_SPLIT = 0.8
OUTPUT_DIR = Path(__file__).parent / "outputs"

# Statistical configuration
ALPHA = 0.05  # Significance threshold
CI_ALPHA = 0.05  # 95% Fisher-z CI for correlations
USE_FDR = True  # Apply BH-FDR across layers within each panel
EPS_DENOM = 1e-8  # Guard against near-constant predictions


# =============================================================================
# STATISTICS HELPERS
# =============================================================================

def _safe_pearson_r(preds: np.ndarray, y: np.ndarray, eps: float = EPS_DENOM) -> float:
    """Fast, numerically-guarded Pearson r (returns 0.0 if degenerate)."""
    preds = np.asarray(preds, dtype=np.float32).ravel()
    y = np.asarray(y, dtype=np.float32).ravel()
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
    # z-critical for two-tailed alpha
    from scipy.stats import norm
    zcrit = float(norm.ppf(1 - alpha / 2))
    z_lo = z - zcrit * se
    z_hi = z + zcrit * se
    return (float(np.tanh(z_lo)), float(np.tanh(z_hi)))


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (returns q-values)."""
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
    y = np.asarray(y, dtype=np.float32).ravel()
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

def compute_mean_diff_direction_vectorized(X: np.ndarray, y: np.ndarray, n_perms: int, rng, quantile: float = 0.25):
    """
    Computes real direction and N permuted directions simultaneously.

    Returns:
        real_dir: (D,) normalized direction from real labels
        perm_dirs: (n_valid_perms, D) normalized directions from valid permutations
                   May have fewer than n_perms rows if some permutations produced
                   degenerate (zero-norm) directions.
    """
    # Cast to float32 to avoid overflow in float16 arithmetic
    X = X.astype(np.float32)

    # Check for NaN/Inf in input (indicates data loading issue)
    if not np.isfinite(X).all():
        n_nan = np.isnan(X).sum()
        n_inf = np.isinf(X).sum()
        raise ValueError(f"X contains {n_nan} NaN and {n_inf} Inf values - check activation data")
    if not np.isfinite(y).all():
        raise ValueError(f"y contains non-finite values - check metric data")

    n = len(y)
    n_group = max(1, int(n * quantile))

    # 1. Real Direction
    sorted_idx = np.argsort(y)
    real_dir = (X[sorted_idx[-n_group:]].mean(0) - X[sorted_idx[:n_group]].mean(0))
    real_norm = np.linalg.norm(real_dir)

    # Check real direction - if this is zero, something is fundamentally wrong
    if real_norm < 1e-10:
        raise RuntimeError(
            f"compute_mean_diff_direction_vectorized: REAL direction has zero/tiny norm ({real_norm:.2e}). "
            f"X shape: {X.shape}, X dtype: {X.dtype}, X range: [{X.min():.2e}, {X.max():.2e}], "
            f"y unique: {len(np.unique(y))}, n_group: {n_group}"
        )
    real_dir = real_dir / real_norm

    # 2. Permuted Directions (Vectorized)
    if n_perms == 0:
        return real_dir.astype(np.float32), None

    # Generate permutation indices matrix (n_perms, n_samples)
    Y_perms = np.zeros((n_perms, n))
    for i in range(n_perms):
        Y_perms[i] = rng.permutation(y)

    sort_idxs = np.argsort(Y_perms, axis=1)
    low_idxs = sort_idxs[:, :n_group]
    high_idxs = sort_idxs[:, -n_group:]

    # Vectorized mean computation
    mean_low = X[low_idxs].mean(axis=1)   # (Perms, D)
    mean_high = X[high_idxs].mean(axis=1) # (Perms, D)

    diffs = mean_high - mean_low

    # Check for NaN/Inf in intermediate computation (indicates overflow in mean)
    if not np.isfinite(diffs).all():
        n_nan = np.isnan(diffs).sum()
        n_inf = np.isinf(diffs).sum()
        raise RuntimeError(
            f"compute_mean_diff_direction_vectorized: diffs contains {n_nan} NaN and {n_inf} Inf. "
            f"mean_low finite: {np.isfinite(mean_low).all()}, mean_high finite: {np.isfinite(mean_high).all()}"
        )

    norms = np.linalg.norm(diffs, axis=1)  # (n_perms,)

    # Filter out degenerate permutations (zero or tiny norms)
    # This is expected to happen occasionally for random permutations
    valid_mask = norms >= 1e-10
    n_valid = valid_mask.sum()
    n_degenerate = n_perms - n_valid

    if n_degenerate > 0:
        # Log warning if significant fraction of permutations are degenerate
        if n_degenerate > n_perms * 0.1:  # More than 10%
            import warnings
            warnings.warn(
                f"compute_mean_diff_direction_vectorized: {n_degenerate}/{n_perms} permutations "
                f"produced degenerate directions (norm < 1e-10). "
                f"X range: [{X.min():.2e}, {X.max():.2e}], y unique: {len(np.unique(y))}"
            )

    if n_valid == 0:
        raise RuntimeError(
            f"compute_mean_diff_direction_vectorized: ALL {n_perms} permutations produced degenerate directions. "
            f"X shape: {X.shape}, X dtype: {X.dtype}, X range: [{X.min():.2e}, {X.max():.2e}], "
            f"y unique: {len(np.unique(y))}, n_group: {n_group}"
        )

    # Normalize only valid permutations
    valid_diffs = diffs[valid_mask]
    valid_norms = norms[valid_mask, None]  # (n_valid, 1) for broadcasting
    all_perm_dirs = valid_diffs / valid_norms

    return real_dir.astype(np.float32), all_perm_dirs.astype(np.float32)


class ProbeDirectionManager:
    """Manages PCA and Ridge fitting for real and permuted labels efficiently."""
    def __init__(self, X: np.ndarray, alpha: float = 1000.0, pca_components: int = 100):
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        # Cast to float32 to avoid overflow in float16 arithmetic
        X = X.astype(np.float32)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        n_components = min(pca_components, X.shape[0], X.shape[1])
        self.pca = PCA(n_components=n_components)
        self.X_pca = self.pca.fit_transform(X_scaled)

        # Precompute (XtX + alpha*I)^-1 Xt
        XtX = self.X_pca.T @ self.X_pca
        XtX_reg = XtX + alpha * np.eye(XtX.shape[0])
        self.solve_matrix = np.linalg.inv(XtX_reg) @ self.X_pca.T 
        
        self.back_proj = self.pca.components_.T / (self.scaler.scale_[:, None] + 1e-10)

    def compute_directions(self, y_real: np.ndarray, n_perms: int, rng) -> tuple:
        ys = [y_real]
        for _ in range(n_perms):
            ys.append(rng.permutation(y_real))
        Y_mat = np.column_stack(ys) # (samples, 1 + n_perms)
        
        weights_pca = self.solve_matrix @ Y_mat
        directions = self.back_proj @ weights_pca
        
        norms = np.linalg.norm(directions, axis=0, keepdims=True)
        directions = directions / (norms + 1e-10)
        
        directions = directions.astype(np.float32)
        real_dir = directions[:, 0]
        perm_dirs = directions[:, 1:].T if n_perms > 0 else None
        
        return real_dir, perm_dirs

# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_package(dataset_name: str, metric: str, meta_tasks: list):
    """Loads all necessary data for a dataset at once."""
    act_path = OUTPUT_DIR / f"{dataset_name}_mc_activations.npz"
    if not act_path.exists():
        raise FileNotFoundError(f"Activations not found: {act_path}")
    data = np.load(act_path)
    
    mc_acts = {}
    for key in data.files:
        if key.startswith("layer_"):
            mc_acts[int(key.split("_")[1])] = data[key]
    mc_metric = data[metric]
    
    meta_data = {}
    for task in meta_tasks:
        path = OUTPUT_DIR / f"{dataset_name}_transfer_{task}_activations.npz"
        if path.exists():
            m_data = np.load(path)
            acts = {}
            has_pos = any("_" in k.replace("layer_", "", 1) for k in m_data.files if k.startswith("layer_"))
            for key in m_data.files:
                if key.startswith("layer_"):
                    if has_pos and not key.endswith("_final"): continue
                    layer = int(key.split("_")[1])
                    acts[layer] = m_data[key]
            meta_data[task] = {"acts": acts, "conf": m_data["confidences"]}
            
    return {"mc_acts": mc_acts, "mc_metric": mc_metric, "meta": meta_data}

def discover_datasets(model_prefix: str) -> list:
    pattern = f"{model_prefix}_*_mc_activations.npz"
    return sorted([p.name.replace("_mc_activations.npz", "") for p in OUTPUT_DIR.glob(pattern)])

def extract_dataset_name(base_name: str, model_prefix: str) -> str:
    if base_name.startswith(model_prefix + "_"):
        return base_name[len(model_prefix) + 1:]
    return base_name

# =============================================================================
# EVALUATION LOGIC
# =============================================================================

def evaluate_transfer(target_acts: np.ndarray, target_vals: np.ndarray,
                      real_dir: np.ndarray, perm_dirs: np.ndarray, two_sided: bool):
    """
    Applies directions to target activations and computes correlations.
    two_sided: If True, checks abs(null) >= abs(real). If False, checks null >= real.

    Returns dict with cross_r, cross_ci_low, cross_ci_high, cross_p, p_value, and null_stats.
    """
    # Cast to float32 to avoid overflow in float16 arithmetic
    target_acts = target_acts.astype(np.float32)
    target_vals = np.asarray(target_vals, dtype=np.float32)
    n = int(target_vals.shape[0])

    # Real correlation with Fisher CI
    preds = target_acts @ real_dir
    cross_r = _safe_pearson_r(preds, target_vals)
    cross_ci_low, cross_ci_high = _fisher_ci(cross_r, n, alpha=CI_ALPHA)

    try:
        _, cross_p = pearsonr(np.asarray(preds, dtype=np.float64), np.asarray(target_vals, dtype=np.float64))
        cross_p = float(cross_p) if np.isfinite(cross_p) else 1.0
    except Exception:
        cross_p = 1.0

    # Permutation null
    if perm_dirs is None or len(perm_dirs) == 0:
        p_value = float("nan")
        null_mean = null_std = null_p5 = null_p95 = float("nan")
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
        }
    }

def get_within_baseline(X_train, y_train, X_test, y_test, method):
    """Computes within-dataset baseline with Fisher-z CI."""
    # Cast to float32 to avoid overflow in float16 arithmetic
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

def plot_transfer_results(d2d_results, d2m_results, dataset_a, dataset_b, output_path):
    metrics = list(d2d_results.keys())
    methods = list(d2d_results[metrics[0]].keys()) if metrics else []
    
    if not metrics or not methods: return
    
    n_cols = len(metrics) * len(methods)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
    name_a = extract_dataset_name(dataset_a, MODEL_PREFIX)
    name_b = extract_dataset_name(dataset_b, MODEL_PREFIX)
    fig.suptitle(f"Cross-Dataset Transfer: {name_a} -> {name_b}", fontsize=14, fontweight="bold")
    
    col = 0
    for metric in metrics:
        for method in methods:
            # D2D
            ax = axes[0, col]
            if metric in d2d_results and method in d2d_results[metric]:
                _plot_single(ax, d2d_results[metric][method], f"d2d: {metric} ({method})")
            else:
                ax.set_title(f"d2d: {metric} ({method}) - No data")
            
            # D2M
            ax = axes[1, col]
            if metric in d2m_results and method in d2m_results[metric]:
                 _plot_single(ax, d2m_results[metric][method], f"d2m: {metric} ({method})")
            else:
                 ax.set_title(f"d2m: {metric} ({method}) - No data")
            col += 1
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_single(ax, results, title):
    per_layer = results.get("per_layer", {})
    if not per_layer:
        ax.set_title(f"{title} - No data")
        return

    layers = sorted(per_layer.keys())
    cross_r = np.array([per_layer[l]["cross_r"] for l in layers])
    within_r = np.array([per_layer[l]["within_r"] for l in layers])
    null_mean = np.array([per_layer[l]["null_mean"] for l in layers])

    # Get CIs if available
    cross_lo = np.array([per_layer[l].get("cross_ci_low", np.nan) for l in layers])
    cross_hi = np.array([per_layer[l].get("cross_ci_high", np.nan) for l in layers])
    within_lo = np.array([per_layer[l].get("within_ci_low", np.nan) for l in layers])
    within_hi = np.array([per_layer[l].get("within_ci_high", np.nan) for l in layers])

    # Plot Fisher-z CIs for cross and within
    if not np.all(np.isnan(within_lo)):
        ax.fill_between(layers, within_lo, within_hi, alpha=0.18, color="green", label="Within 95% CI")
    if not np.all(np.isnan(cross_lo)):
        ax.fill_between(layers, cross_lo, cross_hi, alpha=0.18, color="blue", label="Cross 95% CI")

    ax.plot(layers, null_mean, "--", color="gray", lw=1, label="Perm-null mean")
    ax.plot(layers, within_r, "-", color="green", lw=2, alpha=0.8, label="Within (B->B)")
    ax.plot(layers, cross_r, "-", color="blue", lw=2, label="Cross (A->B)")

    # Use FDR-corrected p-value if available
    p_key = "p_value_fdr" if USE_FDR and any("p_value_fdr" in per_layer[l] for l in layers) else "p_value"
    sigs = [l for l in layers if float(per_layer[l].get(p_key, 1.0)) < ALPHA]
    if sigs:
        sig_rs = [per_layer[l]["cross_r"] for l in sigs]
        ax.scatter(sigs, sig_rs, color="red", s=28, zorder=5, label=f"{p_key}<{ALPHA:g}")
    
    ax.axhline(0, color="black", ls=":", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

def plot_synthesis(all_d2d, all_d2m, model_prefix, output_path):
    pairs = list(all_d2d.keys())
    if len(pairs) < 2: return

    # Gather metrics/methods
    metrics, methods = set(), set()
    for pair_data in all_d2d.values():
        metrics.update(pair_data.keys())
        for m_data in pair_data.values(): methods.update(m_data.keys())
    metrics, methods = sorted(metrics), sorted(methods)

    n_cols = len(metrics) * len(methods)
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 8), squeeze=False)
    fig.suptitle(f"Cross-Dataset Transfer Synthesis ({len(pairs)} pairs)", fontsize=14, fontweight="bold")

    # Get common layers from first available
    first_pair = pairs[0]
    first_metric = metrics[0]
    first_method = methods[0]
    if first_method in all_d2d[first_pair].get(first_metric, {}):
        layers = sorted(all_d2d[first_pair][first_metric][first_method]["per_layer"].keys())
    else: return

    col = 0
    for metric in metrics:
        for method in methods:
            # D2D
            _plot_synthesis_single(axes[0, col], all_d2d, pairs, metric, method, layers, f"d2d: {metric} ({method})")
            # D2M
            _plot_synthesis_single(axes[1, col], all_d2m, pairs, metric, method, layers, f"d2m: {metric} ({method})")
            col += 1
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

def _plot_synthesis_single(ax, all_results, pairs, metric, method, layers, title):
    layer_cross_rs, layer_within_rs = {l: [] for l in layers}, {l: [] for l in layers}
    
    for pair in pairs:
        if metric not in all_results.get(pair, {}) or method not in all_results[pair][metric]: continue
        per_layer = all_results[pair][metric][method]["per_layer"]
        for l in layers:
            if l in per_layer:
                layer_cross_rs[l].append(per_layer[l]["cross_r"])
                layer_within_rs[l].append(per_layer[l]["within_r"])

    mean_cross = np.array([np.mean(v) if v else np.nan for v in layer_cross_rs.values()])
    std_cross = np.array([np.std(v) if v else 0 for v in layer_cross_rs.values()])
    mean_within = np.array([np.mean(v) if v else np.nan for v in layer_within_rs.values()])

    ax.fill_between(layers, mean_cross - std_cross, mean_cross + std_cross, alpha=0.3, color="blue")
    ax.plot(layers, mean_cross, "-", color="blue", lw=2, label="Cross (mean±std)")
    ax.plot(layers, mean_within, "-", color="green", lw=2, alpha=0.7, label="Within (mean)")
    ax.axhline(0, color="black", ls=":", alpha=0.3)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Pearson r")
    ax.set_title(title)
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

# =============================================================================
# MAIN
# =============================================================================

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    datasets = discover_datasets(MODEL_PREFIX)
    if len(datasets) < 2: return

    pairs = list(combinations(datasets, 2))
    
    # Store results
    all_d2d = {} 
    all_d2m = {}

    for ds_a, ds_b in pairs:
        pair_key = f"{extract_dataset_name(ds_a, MODEL_PREFIX)}_vs_{extract_dataset_name(ds_b, MODEL_PREFIX)}"
        print(f"\nProcessing {pair_key}...")
        
        all_d2d[pair_key] = {}
        all_d2m[pair_key] = {}
        
        for metric in METRICS:
            all_d2d[pair_key][metric] = {}
            all_d2m[pair_key][metric] = {}
            
            # 1. Load Data ONCE
            try:
                data_A = load_data_package(ds_a, metric, META_TASKS)
                data_B = load_data_package(ds_b, metric, META_TASKS)
            except FileNotFoundError as e:
                print(f"Skipping {metric}: {e}")
                continue

            # Split BOTH datasets with same 80/20 ratio for fair comparison
            # Cross: train on A_train, test on B_test
            # Within: train on B_train, test on B_test
            len_A = len(data_A["mc_metric"])
            len_B = len(data_B["mc_metric"])
            for t in META_TASKS:
                if t in data_B["meta"]:
                    len_B = min(len_B, len(data_B["meta"][t]["conf"]))

            indices_A = np.arange(len_A)
            indices_B = np.arange(len_B)
            train_idx_A, _ = train_test_split(indices_A, train_size=WITHIN_TRAIN_SPLIT, random_state=SEED)
            train_idx_B, test_idx_B = train_test_split(indices_B, train_size=WITHIN_TRAIN_SPLIT, random_state=SEED)
            
            # 2. Iterate Methods
            for method in METHODS:
                print(f"  Metric: {metric}, Method: {method}")
                
                # RNG Initialized once per method/pair to match original statistical behavior
                rng = np.random.RandomState(SEED)
                
                res_d2d = {
                    "per_layer": {},
                    "n_significant": 0,
                    "n_A_train": len(train_idx_A),
                    "n_B_train": len(train_idx_B),
                    "n_B_test": len(test_idx_B)
                }
                res_d2m = {
                    t: {
                        "per_layer": {},
                        "n_significant": 0,
                        "n_A_train": len(train_idx_A),
                        "n_B_train": len(train_idx_B),
                        "n_B_test": len(test_idx_B)
                    } for t in META_TASKS if t in data_B["meta"]
                }
                
                layers = sorted(set(data_A["mc_acts"].keys()) & set(data_B["mc_acts"].keys()))
                
                for layer in tqdm(layers, leave=False):
                    # Use A_train for cross-dataset direction (fair comparison with B_train for within)
                    X_A_full = data_A["mc_acts"][layer]
                    y_A_full = data_A["mc_metric"]
                    X_A_train = X_A_full[train_idx_A]
                    y_A_train = y_A_full[train_idx_A]

                    # --- STEP 1: Compute Directions on A_train ---
                    if method == "mean_diff":
                        real_dir, perm_dirs = compute_mean_diff_direction_vectorized(
                            X_A_train, y_A_train, N_PERMUTATIONS, rng, MEAN_DIFF_QUANTILE
                        )
                    else:
                        mgr = ProbeDirectionManager(X_A_train, PROBE_ALPHA, PROBE_PCA_COMPONENTS)
                        real_dir, perm_dirs = mgr.compute_directions(y_A_train, N_PERMUTATIONS, rng)
                    
                    # --- STEP 2: Evaluate D2D ---
                    X_B = data_B["mc_acts"][layer][:len_B]
                    X_B_train, X_B_test = X_B[train_idx_B], X_B[test_idx_B]
                    y_B_train, y_B_test = data_B["mc_metric"][train_idx_B], data_B["mc_metric"][test_idx_B]
                    
                    # two_sided=True for D2D
                    d2d_eval = evaluate_transfer(X_B_test, y_B_test, real_dir, perm_dirs, two_sided=True)
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
                    }

                    # --- STEP 3: Evaluate D2M ---
                    # Get metric sign for correlation interpretation
                    # (matches test_meta_transfer.py convention: positive r = good transfer)
                    msign = metric_sign_for_confidence(metric)

                    for task in res_d2m:
                        meta_acts = data_B["meta"][task]["acts"].get(layer)
                        if meta_acts is None: continue

                        X_Meta = meta_acts[:len_B].astype(np.float32)  # Cast to avoid float16 overflow
                        X_Meta_test = X_Meta[test_idx_B]
                        conf_test = np.asarray(data_B["meta"][task]["conf"][:len_B][test_idx_B], dtype=np.float32)
                        n_meta = int(conf_test.shape[0])

                        # Compute cross d2m: direction from A applied to B's meta activations
                        # Multiply projection by metric_sign so positive correlation = good transfer
                        cross_preds = (X_Meta_test @ real_dir) * msign
                        cross_r = _safe_pearson_r(cross_preds, conf_test)
                        cross_ci_low, cross_ci_high = _fisher_ci(cross_r, n_meta, alpha=CI_ALPHA)

                        try:
                            _, cross_p = pearsonr(np.asarray(cross_preds, dtype=np.float64), np.asarray(conf_test, dtype=np.float64))
                            cross_p = float(cross_p) if np.isfinite(cross_p) else 1.0
                        except Exception:
                            cross_p = 1.0

                        # Permutation test for significance (two-sided for d2m)
                        perm_preds = (X_Meta_test @ perm_dirs.T) * msign
                        null_rs = _vectorized_null_rs(perm_preds, conf_test, eps=EPS_DENOM)
                        p_value = (np.sum(np.abs(null_rs) >= np.abs(cross_r)) + 1) / (len(null_rs) + 1)

                        # Within D2M Baseline (Train on B's MC, Test on B's Meta)
                        within_d2m_r = 0.0
                        within_ci_low_m, within_ci_high_m = float("nan"), float("nan")
                        if len(X_B_train) > 0:
                            if method == "mean_diff":
                                db, _ = compute_mean_diff_direction_vectorized(X_B_train, y_B_train, 0, None, MEAN_DIFF_QUANTILE)
                            else:
                                mb = ProbeDirectionManager(X_B_train, PROBE_ALPHA, PROBE_PCA_COMPONENTS)
                                db, _ = mb.compute_directions(y_B_train, 0, None)

                            # Apply same metric_sign convention
                            p_within = (X_Meta_test @ db) * msign
                            within_d2m_r = _safe_pearson_r(p_within, conf_test)
                            within_ci_low_m, within_ci_high_m = _fisher_ci(within_d2m_r, n_meta, alpha=CI_ALPHA)

                        transfer_eff_m = cross_r / within_d2m_r if abs(within_d2m_r) > 1e-6 else 0.0

                        res_d2m[task]["per_layer"][layer] = {
                            "cross_r": float(cross_r),
                            "cross_ci_low": float(cross_ci_low),
                            "cross_ci_high": float(cross_ci_high),
                            "cross_p": float(cross_p),
                            "within_r": float(within_d2m_r),
                            "within_ci_low": float(within_ci_low_m),
                            "within_ci_high": float(within_ci_high_m),
                            "transfer_efficiency": float(transfer_eff_m),
                            "p_value": float(p_value),
                            "null_mean": float(np.mean(null_rs)),
                            "null_std": float(np.std(null_rs)),
                            "null_5th": float(np.percentile(null_rs, 5)),
                            "null_95th": float(np.percentile(null_rs, 95)),
                        }

                # Save Summaries D2D with FDR correction
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

                # Save Summaries D2M with FDR correction
                for task in res_d2m:
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
                    
                    if task not in all_d2m[pair_key][metric]: all_d2m[pair_key][metric][task] = {}
                    all_d2m[pair_key][metric][task][method] = res_d2m[task]

        # Plotting - flatten d2m structure for plotting
        plot_path = OUTPUT_DIR / f"{MODEL_PREFIX}_{pair_key}_transfer.png"
        d2m_for_plot = {}
        for m in METRICS:
            d2m_for_plot[m] = {}
            if m in all_d2m[pair_key]:
                for task, task_data in all_d2m[pair_key][m].items():
                    for meth, res in task_data.items():
                        d2m_for_plot[m][meth] = res
                    break  # Use first task only
        plot_transfer_results(all_d2d[pair_key], d2m_for_plot, ds_a, ds_b, plot_path)

    # Save Final JSON with Config
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
            "note": (
                "Fair comparison: both cross (A_train) and within (B_train) use 80% for training. "
                "CIs are Fisher-z for observed r. BH-FDR is applied across layers within each panel when USE_FDR=True."
            ),
        },
        "d2d": all_d2d,
        "d2m": all_d2m,
    }

    with open(OUTPUT_DIR / f"{MODEL_PREFIX}_cross_dataset_transfer.json", "w") as f:
        json.dump(results, f, indent=2)

    # Synthesis Plot
    if len(pairs) >= 2:
        synthesis_path = OUTPUT_DIR / f"{MODEL_PREFIX}_cross_dataset_transfer_synthesis.png"
        # Flatten d2m
        d2m_flat = {}
        for pair_key, pair_data in all_d2m.items():
            d2m_flat[pair_key] = {}
            for metric, metric_data in pair_data.items():
                d2m_flat[pair_key][metric] = {}
                for task, task_data in metric_data.items():
                    for method, res in task_data.items():
                        d2m_flat[pair_key][metric][method] = res
                    break

        plot_synthesis(all_d2d, d2m_flat, MODEL_PREFIX, synthesis_path)

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for pair_key in all_d2d.keys():
        print(f"\n{pair_key}:")
        for metric in METRICS:
            print(f"  {metric}:")
            for method in METHODS:
                if method in all_d2d[pair_key].get(metric, {}):
                    d2d = all_d2d[pair_key][metric][method]
                    n = d2d["n_significant"]
                    r = d2d.get("best_cross_r")
                    l = d2d.get("best_layer")
                    if r is not None:
                        print(f"    {method} d2d: {n} sig, best r={r:.3f} (L{l})")
                    else:
                        print(f"    {method} d2d: {n} sig")
    print("\nDone.")

if __name__ == "__main__":
    main()