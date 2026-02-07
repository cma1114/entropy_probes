"""
Compute contrast direction from paired self vs other confidence activations.

d_contrast = normalize(mean(self_activation - other_activation))

This captures what changes in activation space when the model introspects
(self-confidence) vs externally evaluates (other-confidence) the same questions.

Unlike d_introspection (which orthogonalizes direction vectors), this works
directly on paired activations - same question, same answer options, only
the task framing differs.

Processes all datasets in DATASETS list in one run.

Inputs (for each dataset):
    outputs/{model}_{dataset}_meta_confidence_activations.npz
    outputs/{model}_{dataset}_meta_other_confidence_activations.npz

Outputs (for each dataset):
    outputs/{model}_{dataset}_contrast_directions.npz

Configuration:
    MODEL_SHORT: Model name (e.g., "Llama-3.1-8B-Instruct")
    ADAPTER: Optional path to PEFT/LoRA adapter (must match identify step)
    DATASETS: List of dataset names to process

Run after:
    test_meta_transfer.py with META_TASK="confidence"
    test_meta_transfer.py with META_TASK="other_confidence"
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
POSITION = "final"  # Token position to use for activations

OUTPUT_DIR = Path("outputs")


def get_base_name(dataset: str) -> str:
    """Get base name for a dataset, including adapter if configured."""
    if ADAPTER:
        adapter_short = get_model_short_name(ADAPTER)
        return f"{MODEL_SHORT}_adapter-{adapter_short}_{dataset}"
    return f"{MODEL_SHORT}_{dataset}"


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_activations(base_name: str, task: str, position: str = "final") -> dict[int, np.ndarray]:
    """
    Load activations from meta-task activation file.

    Args:
        base_name: Base name for input files
        task: "confidence" or "other_confidence"
        position: Token position (default "final")

    Returns:
        {layer: (n_samples, hidden_dim)}
    """
    path = OUTPUT_DIR / f"{base_name}_meta_{task}_activations.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Activations not found: {path}\n"
            f"Run: test_meta_transfer.py with META_TASK='{task}'"
        )

    data = np.load(path)
    activations = {}

    for key in data.files:
        # Position-specific format: layer_{idx}_{position}
        if key.startswith("layer_") and key.endswith(f"_{position}"):
            parts = key.replace(f"_{position}", "").replace("layer_", "")
            try:
                layer = int(parts)
                activations[layer] = data[key]
            except ValueError:
                continue
        # Legacy format: layer_{idx}
        elif key.startswith("layer_") and "_" not in key.replace("layer_", ""):
            try:
                layer = int(key.replace("layer_", ""))
                if layer not in activations:  # Don't overwrite position-specific
                    activations[layer] = data[key]
            except ValueError:
                continue

    return activations


def compute_contrast_direction(
    self_acts: np.ndarray,
    other_acts: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Compute contrast direction from paired activations.

    Args:
        self_acts: (n_samples, hidden_dim) self-confidence activations
        other_acts: (n_samples, hidden_dim) other-confidence activations

    Returns:
        direction: (hidden_dim,) normalized contrast direction
        info: dict with statistics
    """
    assert self_acts.shape == other_acts.shape, \
        f"Shape mismatch: {self_acts.shape} vs {other_acts.shape}"

    # Paired difference for each question
    diff = self_acts - other_acts  # (n_samples, hidden_dim)

    # Mean difference across questions
    mean_diff = diff.mean(axis=0)

    # Norm before normalization (indicates effect size)
    raw_norm = float(np.linalg.norm(mean_diff))

    # Normalize to unit length
    if raw_norm > 0:
        direction = mean_diff / raw_norm
    else:
        direction = mean_diff

    # Compute per-sample norms for statistics
    sample_norms = np.linalg.norm(diff, axis=1)

    info = {
        "n_samples": self_acts.shape[0],
        "hidden_dim": self_acts.shape[1],
        "raw_norm": raw_norm,
        "mean_sample_norm": float(sample_norms.mean()),
        "std_sample_norm": float(sample_norms.std()),
    }

    return direction.astype(np.float32), info


def process_dataset(base_name: str) -> bool:
    """Process a single dataset. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Dataset: {base_name}")
    print(f"Position: {POSITION}")

    # Load activations
    print("\nLoading activations...")
    try:
        self_acts = load_activations(base_name, "confidence", POSITION)
        other_acts = load_activations(base_name, "other_confidence", POSITION)
    except FileNotFoundError as e:
        print(f"  SKIPPED: {e}")
        return False

    # Find common layers
    layers = sorted(set(self_acts.keys()) & set(other_acts.keys()))
    print(f"  Layers: {len(layers)}")

    # Verify alignment
    for layer in layers:
        n_self = self_acts[layer].shape[0]
        n_other = other_acts[layer].shape[0]
        if n_self != n_other:
            raise ValueError(
                f"Layer {layer}: n_samples mismatch ({n_self} vs {n_other}). "
                "Activations may not be aligned."
            )

    n_samples = self_acts[layers[0]].shape[0]
    hidden_dim = self_acts[layers[0]].shape[1]
    print(f"  Samples: {n_samples}")
    print(f"  Hidden dim: {hidden_dim}")

    # Compute contrast direction for each layer
    print("\nComputing contrast directions...")
    save_data = {
        "_metadata_model": base_name.split("_")[0],
        "_metadata_dataset": "_".join(base_name.split("_")[1:]),
        "_metadata_position": POSITION,
    }

    raw_norms = []
    for layer in layers:
        direction, info = compute_contrast_direction(
            self_acts[layer], other_acts[layer]
        )
        save_data[f"contrast_layer_{layer}"] = direction
        raw_norms.append(info["raw_norm"])

    # Save
    output_path = OUTPUT_DIR / f"{base_name}_contrast_directions.npz"
    np.savez_compressed(output_path, **save_data)
    print(f"\nSaved: {output_path.name}")

    # Summary
    print(f"\nSummary:")
    print(f"  Mean raw norm: {np.mean(raw_norms):.4f}")
    print(f"  Max raw norm at layer: {layers[np.argmax(raw_norms)]}")
    print(f"  Min raw norm at layer: {layers[np.argmin(raw_norms)]}")
    return True


def main():
    print(f"Model: {MODEL_SHORT}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Datasets: {DATASETS}")
    print(f"Position: {POSITION}")

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
