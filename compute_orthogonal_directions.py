"""
Compute orthogonalized directions (d_introspection, d_surface) from d_self and d_other.

This is just Gram-Schmidt projection - takes seconds, not hours.
Processes all datasets in DATASETS list in one run.

Inputs (for each dataset):
    outputs/{model}_{dataset}_meta_confidence_metaconfdir_directions.npz    (d_self)
    outputs/{model}_{dataset}_meta_other_confidence_metaconfdir_directions.npz  (d_other)

Outputs (for each dataset):
    outputs/{model}_{dataset}_orthogonal_directions.npz

Configuration:
    MODEL_SHORT: Model name (e.g., "Llama-3.1-8B-Instruct")
    ADAPTER: Optional path to PEFT/LoRA adapter (must match identify step)
    DATASETS: List of dataset names to process

Run after:
    test_meta_transfer.py with META_TASK="confidence" and FIND_CONFIDENCE_DIRECTIONS=True
    test_meta_transfer.py with META_TASK="other_confidence" and FIND_CONFIDENCE_DIRECTIONS=True
"""

import numpy as np
from pathlib import Path

from core.model_utils import get_model_short_name

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_SHORT = "Llama-3.1-8B-Instruct"
ADAPTER = None  # Optional: path to PEFT/LoRA adapter (must match identify step if used)
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]
METHOD = "mean_diff"  # "probe" or "mean_diff" - must match what you want to orthogonalize
MIN_RESIDUAL_NORM = 0.1  # Flag layers where d_self ≈ d_other

OUTPUT_DIR = Path("outputs")


def get_base_name(dataset: str) -> str:
    """Get base name for a dataset, including adapter if configured."""
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        return f"{MODEL_SHORT}_adapter-{adapter_short}_{dataset}"
    return f"{MODEL_SHORT}_{dataset}"


def load_directions(base_name: str, task: str, method: str) -> dict[int, np.ndarray]:
    """Load direction vectors from meta confidence direction file."""
    path = OUTPUT_DIR / f"{base_name}_meta_{task}_metaconfdir_directions.npz"
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

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


def orthogonalize(d_self: np.ndarray, d_other: np.ndarray):
    """
    Gram-Schmidt orthogonalization.

    d_introspection = d_self - proj(d_self, d_other)
    d_surface = d_other - proj(d_other, d_self)
    """
    # Normalize inputs
    d_self = d_self / np.linalg.norm(d_self)
    d_other = d_other / np.linalg.norm(d_other)

    # Cosine similarity
    cosine = float(np.dot(d_self, d_other))

    # Gram-Schmidt
    d_introspection = d_self - cosine * d_other
    residual_norm = float(np.linalg.norm(d_introspection))

    d_surface = d_other - cosine * d_self

    # Check for degenerate case (d_self ≈ d_other)
    # Flag it but keep the actual residual - don't replace with random noise
    degenerate = residual_norm < MIN_RESIDUAL_NORM

    # Normalize outputs (even if small, preserves the actual direction)
    intro_norm = np.linalg.norm(d_introspection)
    surf_norm = np.linalg.norm(d_surface)

    d_introspection = d_introspection / intro_norm
    d_surface = d_surface / surf_norm

    return {
        "d_introspection": d_introspection.astype(np.float32),
        "d_surface": d_surface.astype(np.float32),
        "cosine": cosine,
        "residual_norm": residual_norm,
        "degenerate": degenerate,
    }


def process_dataset(base_name: str) -> bool:
    """Process a single dataset. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Dataset: {base_name}")
    print(f"Method: {METHOD}")

    # Load directions
    print("\nLoading directions...")
    try:
        d_self_by_layer = load_directions(base_name, "confidence", METHOD)
        d_other_by_layer = load_directions(base_name, "other_confidence", METHOD)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return False

    layers = sorted(set(d_self_by_layer.keys()) & set(d_other_by_layer.keys()))
    print(f"  Layers: {len(layers)}")

    # Orthogonalize
    print("\nOrthogonalizing...")
    save_data = {
        "_metadata_model": base_name.split("_")[0],
        "_metadata_dataset": "_".join(base_name.split("_")[1:]),
        "_metadata_method": METHOD,
    }

    n_degenerate = 0
    cosines = []

    for layer in layers:
        result = orthogonalize(d_self_by_layer[layer], d_other_by_layer[layer])

        save_data[f"introspection_layer_{layer}"] = result["d_introspection"]
        save_data[f"surface_layer_{layer}"] = result["d_surface"]
        save_data[f"cosine_layer_{layer}"] = np.array([result["cosine"]])
        save_data[f"residual_norm_layer_{layer}"] = np.array([result["residual_norm"]])

        cosines.append(result["cosine"])
        if result["degenerate"]:
            n_degenerate += 1

    # Save
    output_path = OUTPUT_DIR / f"{base_name}_orthogonal_directions.npz"
    np.savez_compressed(output_path, **save_data)
    print(f"\nSaved: {output_path.name}")

    # Summary
    print(f"\nSummary:")
    print(f"  Mean cos(d_self, d_other): {np.mean(cosines):.3f}")
    print(f"  Degenerate layers: {n_degenerate}/{len(layers)}")
    return True


def main():
    print(f"Model: {MODEL_SHORT}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Datasets: {DATASETS}")
    print(f"Method: {METHOD}")

    n_success = 0
    n_skipped = 0

    for dataset in DATASETS:
        base_name = get_base_name(dataset)
        if process_dataset(base_name):
            n_success += 1
        else:
            n_skipped += 1

    print(f"\n{'='*60}")
    print(f"Done! Processed {n_success} datasets, skipped {n_skipped}")


if __name__ == "__main__":
    main()
