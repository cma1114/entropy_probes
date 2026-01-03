"""
Interpret linear probe directions using Activation Oracles.

This script loads direction vectors from run_contrastive_direction.py outputs
and uses the Activation Oracle (AO) adapter to interpret what concepts they represent.

Usage:
    python act_oracles.py
"""

import json
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Callable
import matplotlib.pyplot as plt
from peft import PeftModel

from core import load_model_and_tokenizer, get_model_short_name

# =============================================================================
# Configuration - match these to your run_contrastive_direction.py settings
# =============================================================================

BASE_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path if using fine-tuned model
DATASET_NAME = "SimpleMC"
METRIC = "top_logit"
META_TASK = "confidence"  # "confidence" or "delegate"
DIRECTION_TYPE = "calibration"  # "calibration" or "contrastive"

# Quantization options for large models
LOAD_IN_4BIT = True   # Recommended for 70B
LOAD_IN_8BIT = False

# Output directory
OUTPUTS_DIR = Path("outputs")

# Activation Oracle adapter paths
AO_ADAPTERS = {
    "8b": "adamkarvonen/checkpoints_latentqa_cls_past_lens_Llama-3_1-8B-Instruct",
    "70b": "adamkarvonen/checkpoints_act_cls_latentqa_pretrain_mix_adding_Llama-3_3-70B-Instruct",
}

# Model configurations
MODEL_CONFIGS = {
    "8b": {
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
        "ao_adapter": AO_ADAPTERS["8b"],
        "num_layers": 32,
        "d_model": 4096,
    },
    "70b": {
        "base_model": "meta-llama/Llama-3.3-70B-Instruct",
        "ao_adapter": AO_ADAPTERS["70b"],
        "num_layers": 80,
        "d_model": 8192,
    },
}

PLACEHOLDER_TOKEN = "?"

# Questions to ask the activation oracle
INTERPRETATION_QUESTIONS = [
    "What concept or phenomenon does this direction represent?",
    "Is this related to uncertainty, confidence, or certainty?",
    "What type of reasoning or mental state does this direction encode?",
]


# =============================================================================
# Path utilities
# =============================================================================

def get_model_prefix() -> str:
    """Get model prefix for filenames."""
    model_short = get_model_short_name(BASE_MODEL_NAME)
    if MODEL_NAME != BASE_MODEL_NAME:
        adapter_short = get_model_short_name(MODEL_NAME)
        return f"{model_short}_adapter-{adapter_short}"
    return model_short


def find_directions_file() -> Path:
    """
    Find available directions file, checking multiple possible sources.

    Searches for direction files from (in order of preference):
    1. run_contrastive_direction.py: {model}_{dataset}_{metric}[_task]_{direction_type}_directions.npz
    2. run_introspection_experiment.py: {model}_{dataset}_introspection[_task]_{metric}_directions.npz
    3. run_introspection_probe.py: {model}_{dataset}_introspection[_task]_{metric}_probe_directions.npz
    4. mc_entropy_probe.py: {model}_{dataset}_mc_{metric}_directions.npz
    5. nexttoken_entropy_probe.py: {model}_{dataset}_{metric}_directions.npz
    """
    model_prefix = get_model_prefix()
    task_suffix = "_delegate" if META_TASK == "delegate" else ""

    # List of patterns to try, in order of preference
    patterns = [
        # run_contrastive_direction.py
        f"{model_prefix}_{DATASET_NAME}_{METRIC}{task_suffix}_{DIRECTION_TYPE}_directions.npz",
        # run_introspection_experiment.py
        f"{model_prefix}_{DATASET_NAME}_introspection{task_suffix}_{METRIC}_directions.npz",
        # run_introspection_probe.py
        f"{model_prefix}_{DATASET_NAME}_introspection{task_suffix}_{METRIC}_probe_directions.npz",
        # mc_entropy_probe.py
        f"{model_prefix}_{DATASET_NAME}_mc_{METRIC}_directions.npz",
        # nexttoken_entropy_probe.py
        f"{model_prefix}_{DATASET_NAME}_{METRIC}_directions.npz",
    ]

    for pattern in patterns:
        path = OUTPUTS_DIR / pattern
        if path.exists():
            return path

    # If no exact match, try glob patterns
    glob_patterns = [
        f"{model_prefix}*{METRIC}*directions.npz",
        f"{model_prefix}*directions.npz",
    ]

    for pattern in glob_patterns:
        matches = list(OUTPUTS_DIR.glob(pattern))
        if matches:
            # Return most recently modified
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

    raise FileNotFoundError(
        f"No directions file found in {OUTPUTS_DIR}\n"
        f"Tried patterns:\n" + "\n".join(f"  - {p}" for p in patterns) +
        f"\n\nRun one of these scripts first:\n"
        f"  - run_contrastive_direction.py (requires run_introspection_experiment.py first)\n"
        f"  - run_introspection_experiment.py\n"
        f"  - run_introspection_probe.py (requires run_introspection_experiment.py first)\n"
        f"  - mc_entropy_probe.py\n"
        f"  - nexttoken_entropy_probe.py"
    )


def load_directions(path: Path) -> dict:
    """Load direction vectors from npz file.

    Returns:
        Dict mapping layer index to direction vector (numpy array)
    """
    if not path.exists():
        raise FileNotFoundError(f"Directions file not found: {path}")

    data = np.load(path)
    directions = {}
    for key in data.files:
        # Keys are "layer_0", "layer_1", etc.
        layer_idx = int(key.split("_")[1])
        directions[layer_idx] = data[key]
    return directions


# =============================================================================
# Activation Oracle Interpreter
# =============================================================================

class ProbeInterpreter:
    def __init__(
        self,
        model_size: str = "70b",
        load_in_4bit: bool = LOAD_IN_4BIT,
        load_in_8bit: bool = LOAD_IN_8BIT,
        device: str = "auto",
    ):
        """
        Initialize the Activation Oracle for interpreting probe directions.

        Args:
            model_size: "8b" or "70b"
            load_in_4bit: Use 4-bit quantization
            load_in_8bit: Use 8-bit quantization
            device: Device to load model on
        """
        if model_size not in MODEL_CONFIGS:
            raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")

        self.config = MODEL_CONFIGS[model_size]
        self.device = device

        # Load base model with quantization
        print(f"Loading base model: {self.config['base_model']}")
        self.model, self.tokenizer, _ = load_model_and_tokenizer(
            self.config["base_model"],
            adapter_path=None,  # AO adapter loaded separately
            load_in_4bit=load_in_4bit,
            load_in_8bit=load_in_8bit,
        )

        # Load AO adapter on top
        print(f"Loading AO adapter: {self.config['ao_adapter']}")
        self.model = PeftModel.from_pretrained(self.model, self.config["ao_adapter"])
        self.model.eval()

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Find placeholder token id
        self.placeholder_id = self.tokenizer.encode(
            PLACEHOLDER_TOKEN, add_special_tokens=False
        )[0]

        print("Ready!")

    def _make_injection_hook(
        self,
        vectors: torch.Tensor,
        placeholder_positions: list,
    ) -> Callable:
        """
        Create a hook that injects vectors at layer 1 using norm-matched addition.

        Injection formula: h'_i = h_i + ||h_i|| * (v_i / ||v_i||)
        """
        def hook(module, input, output):
            hidden_states = output[0]  # (batch, seq, d_model)

            for i, pos in enumerate(placeholder_positions):
                if i >= len(vectors):
                    break

                v = vectors[i].to(hidden_states.device, hidden_states.dtype)
                h = hidden_states[0, pos]  # Original activation at this position

                # Norm-matched addition
                v_normalized = v / (v.norm() + 1e-8)
                h_norm = h.norm()

                hidden_states[0, pos] = h + h_norm * v_normalized

            return (hidden_states,) + output[1:]

        return hook

    def _find_placeholder_positions(self, input_ids: torch.Tensor) -> list:
        """Find positions of placeholder tokens in the input."""
        positions = (input_ids[0] == self.placeholder_id).nonzero(as_tuple=True)[0]
        return positions.tolist()

    def _get_layer_1_module(self):
        """Get the module after which to inject (layer 1)."""
        # For Llama models with PEFT, layers are in model.base_model.model.model.layers
        return self.model.base_model.model.model.layers[1]

    def interpret(
        self,
        vector: torch.Tensor,
        source_layer: int,
        question: str = "What concept or phenomenon does this direction represent?",
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        num_placeholders: int = 1,
    ) -> str:
        """
        Interpret a single probe direction vector.

        Args:
            vector: The probe direction, shape (d_model,)
            source_layer: Which layer of the original model this probe was trained on
            question: Natural language question to ask about the direction
            max_new_tokens: Maximum response length
            temperature: Sampling temperature
            num_placeholders: Number of placeholder tokens (usually 1 for a single direction)

        Returns:
            The AO's interpretation as a string
        """
        if vector.dim() != 1:
            raise ValueError(f"Expected 1D vector, got shape {vector.shape}")

        # Construct the prompt
        placeholders = " ".join([PLACEHOLDER_TOKEN] * num_placeholders)
        prompt = f"Layer {source_layer}: {placeholders} {question}"

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        # Find placeholder positions
        placeholder_positions = self._find_placeholder_positions(input_ids)
        if len(placeholder_positions) == 0:
            raise ValueError("No placeholder tokens found in prompt")

        # Prepare vectors for injection
        vectors = vector.unsqueeze(0) if num_placeholders == 1 else vector
        if vectors.dim() == 1:
            vectors = vectors.unsqueeze(0)

        # Register injection hook at layer 1
        layer_1 = self._get_layer_1_module()
        hook = self._make_injection_hook(vectors, placeholder_positions)
        handle = layer_1.register_forward_hook(hook)

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            # Decode only the generated part
            generated = outputs[0, input_ids.shape[1]:]
            response = self.tokenizer.decode(generated, skip_special_tokens=True)

        finally:
            handle.remove()

        return response.strip()

    def interpret_with_multiple_questions(
        self,
        vector: torch.Tensor,
        source_layer: int,
        questions: list = None,
        **kwargs,
    ) -> dict:
        """
        Interpret a direction with multiple questions.

        Returns:
            Dict mapping question to response
        """
        if questions is None:
            questions = INTERPRETATION_QUESTIONS

        results = {}
        for question in questions:
            results[question] = self.interpret(
                vector=vector,
                source_layer=source_layer,
                question=question,
                **kwargs,
            )
        return results


# =============================================================================
# Visualization
# =============================================================================

def plot_interpretations(
    results: dict,
    output_path: str,
    title_suffix: str = "",
):
    """
    Create a visualization of layer-by-layer interpretations.

    Args:
        results: Dict mapping layer_idx -> {question -> response}
        output_path: Where to save the figure
        title_suffix: Additional text for the title
    """
    # Extract the primary question's responses
    primary_question = INTERPRETATION_QUESTIONS[0]

    layers = sorted(results.keys())
    n_layers = len(layers)

    if n_layers == 0:
        print("No results to plot")
        return

    # Create figure
    fig_height = max(8, n_layers * 0.4)
    _, ax = plt.subplots(figsize=(14, fig_height))

    # Plot each layer's interpretation
    for i, layer_idx in enumerate(layers):
        layer_results = results[layer_idx]
        if isinstance(layer_results, str):
            # Single question format
            interpretation = layer_results
        else:
            # Multi-question format
            interpretation = layer_results.get(primary_question, str(layer_results))

        # Truncate long interpretations
        max_chars = 120
        if len(interpretation) > max_chars:
            interpretation = interpretation[:max_chars] + "..."

        # Clean up whitespace
        interpretation = " ".join(interpretation.split())

        ax.text(
            0.02, i,
            f"Layer {layer_idx:2d}: {interpretation}",
            fontsize=8,
            va='center',
            fontfamily='monospace',
            wrap=True,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, n_layers - 0.5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.invert_yaxis()  # Put layer 0 at top

    model_short = get_model_short_name(BASE_MODEL_NAME)
    title = f"Activation Oracle Interpretations\n{model_short} - {DATASET_NAME} - {DIRECTION_TYPE}"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=12, fontweight='bold')

    # Add border
    for spine in ax.spines.values():
        spine.set_visible(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    plt.close()


def print_interpretations(results: dict):
    """Print interpretations to console in a readable format."""
    print("\n" + "=" * 80)
    print("ACTIVATION ORACLE INTERPRETATIONS")
    print("=" * 80)

    for layer_idx in sorted(results.keys()):
        layer_results = results[layer_idx]
        print(f"\n{'─' * 80}")
        print(f"LAYER {layer_idx}")
        print("─" * 80)

        if isinstance(layer_results, str):
            print(f"  {layer_results}")
        else:
            for question, response in layer_results.items():
                print(f"\n  Q: {question}")
                print(f"  A: {response}")


# =============================================================================
# Main
# =============================================================================

def main():
    print(f"\n{'=' * 70}")
    print("ACTIVATION ORACLE PROBE INTERPRETATION")
    print("=" * 70)
    print(f"Model: {get_model_short_name(BASE_MODEL_NAME)}")
    print(f"Dataset: {DATASET_NAME}")
    print(f"Metric: {METRIC}")
    print(f"Task: {META_TASK}")
    print(f"Direction type: {DIRECTION_TYPE}")

    # Load directions
    directions_path = find_directions_file()
    print(f"\nLoading directions from: {directions_path}")

    try:
        directions = load_directions(directions_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease run run_contrastive_direction.py first to generate directions.")
        return

    print(f"Loaded directions for {len(directions)} layers: {sorted(directions.keys())}")

    # Determine model size from config
    if "70B" in BASE_MODEL_NAME or "70b" in BASE_MODEL_NAME:
        model_size = "70b"
    else:
        model_size = "8b"

    # Initialize interpreter
    print(f"\nInitializing interpreter (model_size={model_size})...")
    interpreter = ProbeInterpreter(
        model_size=model_size,
        load_in_4bit=LOAD_IN_4BIT,
        load_in_8bit=LOAD_IN_8BIT,
    )

    # Select layers to interpret
    # For efficiency, sample a subset if there are many layers
    all_layers = sorted(directions.keys())
    if len(all_layers) > 20:
        # Sample early, middle, and late layers
        n_samples = 15
        indices = np.linspace(0, len(all_layers) - 1, n_samples, dtype=int)
        layers_to_interpret = [all_layers[i] for i in indices]
        print(f"\nSampling {n_samples} layers from {len(all_layers)} total")
    else:
        layers_to_interpret = all_layers

    print(f"Layers to interpret: {layers_to_interpret}")

    # Interpret each layer
    results = {}
    for layer_idx in tqdm(layers_to_interpret, desc="Interpreting layers"):
        vector = torch.from_numpy(directions[layer_idx]).float()

        # Ask multiple questions for richer interpretation
        layer_results = interpreter.interpret_with_multiple_questions(
            vector=vector,
            source_layer=layer_idx,
            questions=INTERPRETATION_QUESTIONS,
        )
        results[layer_idx] = layer_results

    # Print results
    print_interpretations(results)

    # Save results - use the directions file path as base
    output_prefix = str(directions_path).replace("_directions.npz", "").replace("_probe_directions.npz", "")

    # Save as JSON
    json_path = f"{output_prefix}_ao_interpretations.json"
    # Convert keys to strings for JSON
    results_serializable = {str(k): v for k, v in results.items()}
    with open(json_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {json_path}")

    # Create visualization
    png_path = f"{output_prefix}_ao_interpretations.png"
    plot_interpretations(results, png_path)

    print(f"\n{'=' * 70}")
    print("INTERPRETATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
