"""
Compute orthogonalized directions from d_self_confidence and d_other_confidence.

Outputs:
    d_self_confidence_unique  = d_self_confidence with d_other_confidence projected out
    d_other_confidence_unique = d_other_confidence with d_self_confidence projected out

This is just Gram-Schmidt projection - takes seconds, not hours.
Processes all datasets in DATASETS list in one run.

Inputs (for each dataset):
    outputs/{model}_{dataset}_meta_confidence_confdir_directions.npz    (d_self_confidence)
    outputs/{model}_{dataset}_meta_other_confidence_confdir_directions.npz  (d_other_confidence)

Outputs (for each dataset):
    outputs/{model}_{dataset}_orthogonal_directions.npz

Configuration:
    MODEL: Full model path (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    ADAPTER: Optional path to PEFT/LoRA adapter (must match identify step)
    DATASETS: List of dataset names to process

Run after:
    test_meta_transfer.py with META_TASK="confidence" and FIND_CONFIDENCE_DIRECTIONS=True
    test_meta_transfer.py with META_TASK="other_confidence" and FIND_CONFIDENCE_DIRECTIONS=True
"""

import numpy as np
from pathlib import Path

from core.model_utils import get_model_dir_name
from core.config_utils import get_output_path, find_output_file

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter (must match identify step if used)
LOAD_IN_4BIT = False
LOAD_IN_8BIT = False
PROBE_POSITION = "final"  # Position from test_meta_transfer.py outputs
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]
METHOD = "mean_diff"  # "probe" or "mean_diff" - must match what you want to orthogonalize
MIN_RESIDUAL_NORM = 0.1  # Flag layers where d_self_confidence ≈ d_other_confidence

# Uses centralized path management from core.config_utils


def get_model_dir() -> str:
    """Get model directory name for the configured model."""
    return get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)


def load_directions(dataset: str, task: str, method: str, model_dir: str = None) -> dict[int, np.ndarray]:
    """Load direction vectors from meta confidence direction file."""
    filename = f"{dataset}_meta_{task}_confdir_directions_{PROBE_POSITION}.npz"
    path = find_output_file(filename, model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {filename}")

    data = np.load(path)
    directions = {}
    for key in data.files:
        if key.startswith(f"{method}_layer_"):
            layer = int(key.replace(f"{method}_layer_", ""))
            v = np.asarray(data[key], dtype=np.float32)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            directions[layer] = v
    return directions


def orthogonalize(d_self_confidence: np.ndarray, d_other_confidence: np.ndarray):
    """
    Gram-Schmidt orthogonalization.

    d_self_confidence_unique = d_self_confidence - proj(d_self_confidence, d_other_confidence)
    d_other_confidence_unique = d_other_confidence - proj(d_other_confidence, d_self_confidence)
    """
    # Normalize inputs
    d_self_confidence = d_self_confidence / np.linalg.norm(d_self_confidence)
    d_other_confidence = d_other_confidence / np.linalg.norm(d_other_confidence)

    # Cosine similarity
    cosine = float(np.dot(d_self_confidence, d_other_confidence))

    # Gram-Schmidt
    d_self_confidence_unique = d_self_confidence - cosine * d_other_confidence
    residual_norm = float(np.linalg.norm(d_self_confidence_unique))

    d_other_confidence_unique = d_other_confidence - cosine * d_self_confidence

    # Check for degenerate case (d_self_confidence ≈ d_other_confidence)
    # Flag it but keep the actual residual - don't replace with random noise
    degenerate = residual_norm < MIN_RESIDUAL_NORM

    # Normalize outputs (even if small, preserves the actual direction)
    self_unique_norm = np.linalg.norm(d_self_confidence_unique)
    other_unique_norm = np.linalg.norm(d_other_confidence_unique)

    d_self_confidence_unique = d_self_confidence_unique / self_unique_norm
    d_other_confidence_unique = d_other_confidence_unique / other_unique_norm

    return {
        "d_self_confidence_unique": d_self_confidence_unique.astype(np.float32),
        "d_other_confidence_unique": d_other_confidence_unique.astype(np.float32),
        "cosine": cosine,
        "residual_norm": residual_norm,
        "degenerate": degenerate,
    }


def process_dataset(dataset: str, model_dir: str = None) -> bool:
    """Process a single dataset. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Method: {METHOD}")

    # Load directions
    print("\nLoading directions...")
    try:
        d_self_conf_by_layer = load_directions(dataset, "confidence", METHOD, model_dir=model_dir)
        d_other_conf_by_layer = load_directions(dataset, "other_confidence", METHOD, model_dir=model_dir)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return False

    layers = sorted(set(d_self_conf_by_layer.keys()) & set(d_other_conf_by_layer.keys()))
    print(f"  Layers: {len(layers)}")

    # Orthogonalize
    print("\nOrthogonalizing...")
    save_data = {
        "_metadata_model": MODEL,
        "_metadata_dataset": dataset,
        "_metadata_method": METHOD,
    }

    n_degenerate = 0
    cosines = []

    for layer in layers:
        result = orthogonalize(d_self_conf_by_layer[layer], d_other_conf_by_layer[layer])

        save_data[f"self_confidence_unique_layer_{layer}"] = result["d_self_confidence_unique"]
        save_data[f"other_confidence_unique_layer_{layer}"] = result["d_other_confidence_unique"]
        save_data[f"cosine_layer_{layer}"] = np.array([result["cosine"]])
        save_data[f"residual_norm_layer_{layer}"] = np.array([result["residual_norm"]])

        cosines.append(result["cosine"])
        if result["degenerate"]:
            n_degenerate += 1

    # Save
    output_path = get_output_path(f"{dataset}_orthogonal_directions.npz", model_dir=model_dir)
    np.savez_compressed(output_path, **save_data)
    print(f"\nSaved: {output_path.name}")

    # Summary
    print(f"\nSummary:")
    print(f"  Mean cos(d_self_confidence, d_other_confidence): {np.mean(cosines):.3f}")
    print(f"  Degenerate layers: {n_degenerate}/{len(layers)}")
    return True


def main():
    model_dir = get_model_dir()

    print(f"Model: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Model dir: {model_dir}")
    print(f"Datasets: {DATASETS}")
    print(f"Method: {METHOD}")

    n_success = 0
    n_skipped = 0

    for dataset in DATASETS:
        if process_dataset(dataset, model_dir=model_dir):
            n_success += 1
        else:
            n_skipped += 1

    print(f"\n{'='*60}")
    print(f"Done! Processed {n_success} datasets, skipped {n_skipped}")


if __name__ == "__main__":
    main()
