"""
MC Answer Choice Direction Finding.

Trains multiclass classifiers to predict which MC answer (A/B/C/D) the model chose,
then extracts direction vectors via PCA on class coefficients.

This provides a confound control for D2M transfer tests:
- If D2M transfer is just detecting answer encoding, answer directions should
  transfer as well as uncertainty directions
- If genuine introspection exists, uncertainty directions should transfer better
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from .directions import _as_float32, _safe_scale


def _bootstrap_accuracy_stats_from_preds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
    chunk_boot: int = 2048,
) -> Tuple[float, float]:
    """
    Bootstrap mean/std of accuracy by resampling examples WITH replacement.
    No refitting: y_pred is treated as fixed predictions on the fixed test set.
    Returns (mean, std). Uses ddof=1 for std.
    """
    if n_bootstrap <= 0:
        return float("nan"), float("nan")

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = y_true.shape[0]

    acc = np.empty(n_bootstrap, dtype=np.float32)
    done = 0
    while done < n_bootstrap:
        b = min(chunk_boot, n_bootstrap - done)
        idx = rng.integers(0, n, size=(b, n), dtype=np.int32)
        acc[done:done + b] = (y_pred[idx] == y_true[idx]).mean(axis=1)
        done += b

    return float(acc.mean()), float(acc.std(ddof=1))


def train_mc_answer_classifier(
    X: np.ndarray,
    y: np.ndarray,
    n_components: int = 256,
    random_state: int = 42
) -> Tuple[StandardScaler, PCA, LogisticRegression]:
    """
    Train a multiclass LogisticRegression to predict MC answer (A/B/C/D).

    Args:
        X: (n_samples, hidden_dim) activation matrix
        y: (n_samples,) answer labels (integers 0-3 for A-D)
        n_components: Number of PCA components
        random_state: Random seed for reproducibility

    Returns:
        scaler: Fitted StandardScaler
        pca: Fitted PCA
        clf: Fitted LogisticRegression classifier
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

    # Logistic Regression
    clf = LogisticRegression(max_iter=1000, random_state=random_state)
    clf.fit(X_pca, y)

    return scaler, pca, clf


def extract_answer_direction(
    scaler: StandardScaler,
    pca: PCA,
    clf: LogisticRegression
) -> np.ndarray:
    """
    Extract a single normalized direction from multiclass classifier coefficients.

    For a 4-class classifier, clf.coef_ has shape (4, n_pca_components).
    We extract the first principal component of these 4 class vectors to get
    the dominant axis of variation that distinguishes different MC answer classes.

    Maps the direction back through PCA and standardization to original activation space.

    Args:
        scaler: Fitted StandardScaler from training
        pca: Fitted PCA from training
        clf: Fitted LogisticRegression classifier

    Returns:
        direction: (hidden_dim,) normalized direction vector
    """
    # clf.coef_ is (n_classes, n_pca_components) - one row per class
    # Take first principal component of the class coefficient vectors
    coef_pca = PCA(n_components=1)
    coef_pca.fit(clf.coef_)
    pc1 = coef_pca.components_[0]  # (n_pca_components,)

    # Project back to scaled space
    direction_scaled = pca.components_.T @ pc1  # (hidden_dim,)

    # Undo standardization scaling
    direction_original = direction_scaled / scaler.scale_

    # Normalize to unit length
    direction_original = direction_original / np.linalg.norm(direction_original)

    return direction_original.astype(np.float32)


def apply_answer_classifier_centered(
    X_meta: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    pca: PCA,
    clf: LogisticRegression
) -> Dict:
    """
    Apply answer classifier to meta activations using centered scaling.

    Centers meta data using its own mean, but scales using direct's variance.
    This is the rigorous transfer test.

    Args:
        X_meta: (n_samples, hidden_dim) meta task activations
        y: (n_samples,) ground truth answer labels
        scaler: Fitted StandardScaler from training on direct
        pca: Fitted PCA from training
        clf: Fitted LogisticRegression classifier

    Returns:
        Dict with accuracy and predictions
    """
    X_meta = _as_float32(X_meta)

    # Center using meta's own mean
    meta_mean = np.mean(X_meta, axis=0, dtype=np.float32)
    X_centered = X_meta - meta_mean

    # Scale using direct's variance
    X_scaled = X_centered / _safe_scale(scaler.scale_)

    # Apply PCA and classify
    X_pca = pca.transform(X_scaled)
    y_pred = clf.predict(X_pca)
    accuracy = accuracy_score(y, y_pred)

    return {
        "accuracy": float(accuracy),
        "predictions": y_pred.tolist()
    }


def apply_answer_classifier_separate(
    X_meta: np.ndarray,
    y: np.ndarray,
    pca: PCA,
    clf: LogisticRegression
) -> Dict:
    """
    Apply answer classifier to meta activations using separate scaling.

    Standardizes meta data using its own statistics (domain adaptation).
    This is the upper bound transfer test.

    Args:
        X_meta: (n_samples, hidden_dim) meta task activations
        y: (n_samples,) ground truth answer labels
        pca: Fitted PCA from training
        clf: Fitted LogisticRegression classifier

    Returns:
        Dict with accuracy and predictions
    """
    X_meta = _as_float32(X_meta)

    # Standardize using meta's own statistics
    meta_scaler = StandardScaler()
    meta_scaler.fit(X_meta)
    meta_scaler.scale_ = _safe_scale(meta_scaler.scale_)
    X_scaled = meta_scaler.transform(X_meta)

    # Apply PCA and classify
    X_pca = pca.transform(X_scaled)
    y_pred = clf.predict(X_pca)
    accuracy = accuracy_score(y, y_pred)

    return {
        "accuracy": float(accuracy),
        "predictions": y_pred.tolist()
    }


def find_answer_directions(
    activations_by_layer: Dict[int, np.ndarray],
    model_answers: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_components: int = 256,
    random_state: int = 42,
) -> Dict:
    """
    Find MC answer directions for all layers.

    Args:
        activations_by_layer: {layer_idx: (n_samples, hidden_dim)}
        model_answers: (n_samples,) integer labels for model's predicted answers
        train_idx: Indices for training
        test_idx: Indices for testing
        n_components: PCA components for classifier
        random_state: Random seed

    Returns:
        {
            "directions": {layer: direction_vector},
            "probes": {layer: {"scaler", "pca", "clf"}},
            "fits": {layer: {"train_accuracy", "test_accuracy"}}
        }
    """
    layers = sorted(activations_by_layer.keys())
    y = np.asarray(model_answers)

    results = {
        "directions": {},
        "probes": {},
        "fits": {}
    }

    for layer in tqdm(layers, desc="Training MC answer classifiers"):
        X = activations_by_layer[layer]

        # Split
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Train classifier
        scaler, pca, clf = train_mc_answer_classifier(
            X_train, y_train,
            n_components=n_components,
            random_state=random_state
        )

        # Evaluate
        X_train_scaled = scaler.transform(_as_float32(X_train))
        X_test_scaled = scaler.transform(_as_float32(X_test))
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        train_accuracy = accuracy_score(y_train, clf.predict(X_train_pca))
        test_accuracy = accuracy_score(y_test, clf.predict(X_test_pca))

        # Extract direction
        direction = extract_answer_direction(scaler, pca, clf)

        # Shuffled baseline
        y_shuffled = y_train.copy()
        np.random.RandomState(random_state).shuffle(y_shuffled)
        _, _, clf_shuffled = train_mc_answer_classifier(
            X_train, y_shuffled,
            n_components=n_components,
            random_state=random_state
        )
        shuffled_accuracy = accuracy_score(
            y_test,
            clf_shuffled.predict(pca.transform(scaler.transform(_as_float32(X_test))))
        )

        results["directions"][layer] = direction
        results["probes"][layer] = {
            "scaler": scaler,
            "pca": pca,
            "clf": clf
        }
        results["fits"][layer] = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "shuffled_accuracy": float(shuffled_accuracy),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
            "n_classes": len(np.unique(y_train)),
        }

    return results


def encode_answers(answers: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Encode string answer labels (A, B, C, D) to integers.

    Args:
        answers: List of answer strings

    Returns:
        encoded: (n_samples,) integer array
        mapping: Dict mapping strings to integers
    """
    unique = sorted(set(answers))
    mapping = {label: i for i, label in enumerate(unique)}
    encoded = np.array([mapping[a] for a in answers])
    return encoded, mapping


def decode_answers(encoded: np.ndarray, mapping: Dict[str, int]) -> List[str]:
    """
    Decode integer labels back to strings.

    Args:
        encoded: (n_samples,) integer array
        mapping: Dict mapping strings to integers

    Returns:
        List of answer strings
    """
    reverse_mapping = {v: k for k, v in mapping.items()}
    return [reverse_mapping[i] for i in encoded]


def class_centroid_direction(
    X: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    """
    Compute direction via class centroids (analogous to mean_diff for classification).

    Computes the centroid of each class, then takes the first principal component
    of the centroids to get a single direction that separates the classes.

    Args:
        X: (n_samples, hidden_dim) activation matrix
        y: (n_samples,) class labels (integers)

    Returns:
        direction: (hidden_dim,) normalized direction vector
    """
    X = _as_float32(X)

    # Compute class centroids
    classes = np.unique(y)
    centroids = []
    for c in classes:
        mask = y == c
        if mask.sum() > 0:
            centroids.append(X[mask].mean(axis=0))

    if len(centroids) < 2:
        # Can't compute direction with fewer than 2 classes
        return np.zeros(X.shape[1], dtype=np.float32)

    centroids = np.stack(centroids)  # (n_classes, hidden_dim)

    # Take first principal component of centroids
    centroid_pca = PCA(n_components=1)
    centroid_pca.fit(centroids)
    direction = centroid_pca.components_[0]  # (hidden_dim,)

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction = direction / norm

    return direction.astype(np.float32)


def find_answer_directions_both_methods(
    activations_by_layer: Dict[int, np.ndarray],
    model_answers: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    n_components: int = 256,
    random_state: int = 42,
    n_bootstrap: int = 0,
    train_split: float = 0.8,
) -> Dict:
    """
    Find MC answer directions using both methods (like uncertainty directions).

    Methods:
    - "probe": LogisticRegression + PCA on class coefficients
    - "centroid": Class centroids + PCA (analogous to mean_diff)

    Args:
        activations_by_layer: {layer_idx: (n_samples, hidden_dim)}
        model_answers: (n_samples,) integer labels for model's predicted answers
        train_idx: Indices for training
        test_idx: Indices for testing
        n_components: PCA components for classifier
        random_state: Random seed
        n_bootstrap: Number of bootstrap iterations for confidence intervals (0 = no bootstrap)
        train_split: Train/test split ratio for bootstrap

    Returns:
        {
            "directions": {"probe": {layer: dir}, "centroid": {layer: dir}},
            "probes": {layer: {"scaler", "pca", "clf"}},
            "fits": {"probe": {layer: metrics}, "centroid": {layer: metrics}},
            "comparison": {layer: {"cosine_sim": float}}
        }
        If n_bootstrap > 0, fits will include "test_accuracy_std" for each method.
    """
    layers = sorted(activations_by_layer.keys())
    y = np.asarray(model_answers)
    n_samples = len(y)

    # NOTE: Bootstrap is done by resampling test predictions, NOT refitting.
    # This is orders of magnitude faster and gives valid confidence intervals.

    results = {
        "directions": {"probe": {}, "centroid": {}},
        "probes": {},
        "fits": {"probe": {}, "centroid": {}},
        "comparison": {}
    }

    for layer in tqdm(layers, desc="Training MC answer classifiers"):
        X = activations_by_layer[layer]

        # Split
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        # Method 1: Classifier-based direction
        scaler, pca, clf = train_mc_answer_classifier(
            X_train, y_train,
            n_components=n_components,
            random_state=random_state
        )

        # Evaluate classifier
        X_train_scaled = scaler.transform(_as_float32(X_train))
        X_test_scaled = scaler.transform(_as_float32(X_test))
        X_train_pca = pca.transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)

        y_pred_train = clf.predict(X_train_pca)
        y_pred_test = clf.predict(X_test_pca)
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        probe_dir = extract_answer_direction(scaler, pca, clf)

        # Bootstrap confidence intervals for probe (fast: resample predictions, not refit)
        probe_fit = {
            "train_accuracy": float(train_accuracy),
            "test_accuracy": float(test_accuracy),
            "pca_variance_explained": float(pca.explained_variance_ratio_.sum()),
        }

        if n_bootstrap > 0:
            rng = np.random.default_rng(random_state + 10_000 + layer)
            m, s = _bootstrap_accuracy_stats_from_preds(y_test, y_pred_test, n_bootstrap, rng)
            probe_fit["test_accuracy_mean"] = m
            probe_fit["test_accuracy_std"] = s
            probe_fit["n_bootstrap"] = int(n_bootstrap)

        results["directions"]["probe"][layer] = probe_dir
        results["probes"][layer] = {"scaler": scaler, "pca": pca, "clf": clf}
        results["fits"]["probe"][layer] = probe_fit

        # Method 2: Centroid-based direction
        centroid_dir = class_centroid_direction(X_train, y_train)
        results["directions"]["centroid"][layer] = centroid_dir

        # Evaluate centroid direction (project and measure class separability)
        train_proj = X_train @ centroid_dir
        test_proj = X_test @ centroid_dir

        # Use projection as a 1D feature for simple accuracy estimate
        simple_clf = LR(max_iter=1000, random_state=random_state)
        simple_clf.fit(train_proj.reshape(-1, 1), y_train)
        y_centroid_pred_train = simple_clf.predict(train_proj.reshape(-1, 1))
        y_centroid_pred_test = simple_clf.predict(test_proj.reshape(-1, 1))
        centroid_train_acc = accuracy_score(y_train, y_centroid_pred_train)
        centroid_test_acc = accuracy_score(y_test, y_centroid_pred_test)

        centroid_fit = {
            "train_accuracy": float(centroid_train_acc),
            "test_accuracy": float(centroid_test_acc),
        }

        # Bootstrap confidence intervals for centroid (fast: resample predictions, not refit)
        if n_bootstrap > 0:
            rng = np.random.default_rng(random_state + 20_000 + layer)
            m, s = _bootstrap_accuracy_stats_from_preds(y_test, y_centroid_pred_test, n_bootstrap, rng)
            centroid_fit["test_accuracy_mean"] = m
            centroid_fit["test_accuracy_std"] = s
            centroid_fit["n_bootstrap"] = int(n_bootstrap)

        results["fits"]["centroid"][layer] = centroid_fit

        # Cosine similarity between methods
        cos_sim = float(np.dot(probe_dir, centroid_dir))
        results["comparison"][layer] = {"cosine_sim": cos_sim}

    return results
