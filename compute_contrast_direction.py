"""
Compute selfVother_conf direction from paired activations.

d_selfVother_conf = normalize(mean(self_activation - other_activation))

This captures what changes in activation space when the model is asked about
its own confidence vs another entity's confidence on the same questions.

Unlike d_self_confidence_unique (which orthogonalizes direction vectors), this works
directly on paired activations - same question, same answer options, only
the task framing differs.

Processes all datasets in DATASETS list in one run.

Inputs (for each dataset):
    outputs/{model}_{dataset}_meta_confidence_activations.npz
    outputs/{model}_{dataset}_meta_other_confidence_activations.npz

Outputs (for each dataset):
    outputs/{model}_{dataset}_selfVother_conf_directions.npz

Configuration:
    MODEL: Full model path (e.g., "meta-llama/Llama-3.1-8B-Instruct")
    ADAPTER: Optional path to PEFT/LoRA adapter (must match identify step)
    DATASETS: List of dataset names to process

Run after:
    test_meta_transfer.py with META_TASK="confidence"
    test_meta_transfer.py with META_TASK="other_confidence"
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
DATASETS = [
    "TriviaMC_difficulty_filtered",
    "PopMC_0_difficulty_filtered",
    "SimpleMC",
]
POSITION = "final"  # Token position to use for activations

# Uses centralized path management from core.config_utils


def get_model_dir() -> str:
    """Get model directory name for the configured model."""
    return get_model_dir_name(MODEL, ADAPTER, LOAD_IN_4BIT, LOAD_IN_8BIT)


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_activations(dataset: str, task: str, position: str = "final", model_dir: str = None) -> dict[int, np.ndarray]:
    """
    Load activations from meta-task activation file.

    Args:
        dataset: Dataset name
        task: "confidence" or "other_confidence"
        position: Token position (default "final")
        model_dir: Model directory for routing to correct output location

    Returns:
        {layer: (n_samples, hidden_dim)}
    """
    filename = f"{dataset}_meta_{task}_activations.npz"
    path = find_output_file(filename, model_dir=model_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Activations not found: {filename}\n"
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


def compute_selfVother_conf_direction(
    self_acts: np.ndarray,
    other_acts: np.ndarray
) -> tuple[np.ndarray, dict]:
    """
    Compute selfVother_conf direction from paired activations.

    Args:
        self_acts: (n_samples, hidden_dim) self-confidence activations
        other_acts: (n_samples, hidden_dim) other-confidence activations

    Returns:
        direction: (hidden_dim,) normalized direction (self - other)
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


def process_dataset(dataset: str, model_dir: str = None) -> bool:
    """Process a single dataset. Returns True if successful."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset}")
    print(f"Position: {POSITION}")

    # Load activations
    print("\nLoading activations...")
    try:
        self_acts = load_activations(dataset, "confidence", POSITION, model_dir=model_dir)
        other_acts = load_activations(dataset, "other_confidence", POSITION, model_dir=model_dir)
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

    # Compute selfVother_conf direction for each layer
    print("\nComputing selfVother_conf directions...")
    save_data = {
        "_metadata_model": MODEL,
        "_metadata_dataset": dataset,
        "_metadata_position": POSITION,
    }

    raw_norms = []
    for layer in layers:
        direction, info = compute_selfVother_conf_direction(
            self_acts[layer], other_acts[layer]
        )
        save_data[f"selfVother_conf_layer_{layer}"] = direction
        raw_norms.append(info["raw_norm"])

    # Save
    output_path = get_output_path(f"{dataset}_selfVother_conf_directions.npz", model_dir=model_dir)
    np.savez_compressed(output_path, **save_data)
    print(f"\nSaved: {output_path.name}")

    # Summary
    print(f"\nSummary:")
    print(f"  Mean raw norm: {np.mean(raw_norms):.4f}")
    print(f"  Max raw norm at layer: {layers[np.argmax(raw_norms)]}")
    print(f"  Min raw norm at layer: {layers[np.argmin(raw_norms)]}")
    return True


def main():
    model_dir = get_model_dir()

    print(f"Model: {MODEL}")
    if ADAPTER:
        print(f"Adapter: {ADAPTER}")
    print(f"Model dir: {model_dir}")
    print(f"Datasets: {DATASETS}")
    print(f"Position: {POSITION}")

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
