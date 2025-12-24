"""
Steering experiment using probe directions from run_introspection_probe.py.

This script:
1. Loads probe results and directions from probe training
2. Automatically selects best layers based on significance
3. Runs steering experiments with entropy direction and control directions
4. Measures effect on alignment between stated confidence and actual entropy
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv
import random

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_NAME = BASE_MODEL_NAME  # Set to adapter path for fine-tuned model

PROBE_RESULTS_PATH = "introspection_probe_results.json"
PROBE_DIRECTIONS_PATH = "introspection_probe_directions.npz"
PAIRED_DATA_PATH = "introspection_paired_data.json"
OUTPUT_PREFIX = "introspection"

# Steering config
STEERING_LAYERS = None  # None = auto-select from probe results
STEERING_MULTIPLIERS = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
NUM_STEERING_QUESTIONS = 100
NUM_CONTROL_DIRECTIONS = 3

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# Meta confidence scale
META_OPTION_DICT = {
    "S": "<5%", "T": "5-10%", "U": "10-20%", "V": "20-40%",
    "W": "40-60%", "X": "60-80%", "Y": "80-90%", "Z": ">90%"
}
META_RANGE_MIDPOINTS = {
    "S": 0.025, "T": 0.075, "U": 0.15, "V": 0.3,
    "W": 0.5, "X": 0.7, "Y": 0.85, "Z": 0.95
}
META_OPTIONS = list(META_OPTION_DICT.keys())

META_SETUP_PROMPT = "I'm going to show you a question, and I want you to tell me your level of confidence that you know the correct answer to this question. Respond only with the letter corresponding to the percentage range you choose; do NOT output any other text."
META_QUESTION_PROMPT = "How confident are you that you know the correct answer to this question?"


# ============================================================================
# HELPERS
# ============================================================================

def is_base_model(model_name: str) -> bool:
    """Check if model is a base model (not instruction-tuned)."""
    model_lower = model_name.lower()
    instruct_indicators = ['instruct', 'chat', '-it', 'rlhf', 'sft', 'dpo']
    return not any(ind in model_lower for ind in instruct_indicators)


def has_chat_template(tokenizer) -> bool:
    """Check if tokenizer has a chat template."""
    try:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "test"}],
            tokenize=False,
            add_generation_prompt=True
        )
        return True
    except Exception:
        return False


# ============================================================================
# PROMPT FORMATTING
# ============================================================================

def _present_nested_question(question_data: Dict, outer_question: str, outer_options: Dict) -> str:
    """Format a nested/meta question for display."""
    formatted = "-" * 30 + "\n"
    formatted += outer_question + "\n"
    formatted += "-" * 10 + "\n"
    formatted += question_data["question"] + "\n"
    if "options" in question_data:
        for key, value in question_data["options"].items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 10 + "\n"
    if outer_options:
        for key, value in outer_options.items():
            formatted += f"  {key}: {value}\n"
    formatted += "-" * 30
    return formatted


def format_meta_prompt(question: Dict, tokenizer, use_chat_template: bool = True) -> str:
    """Format a meta/confidence question."""
    q_text = _present_nested_question(question, META_QUESTION_PROMPT, META_OPTION_DICT)
    options_str = ", ".join(META_OPTIONS[:-1]) + f", or {META_OPTIONS[-1]}"
    llm_prompt = q_text + f"\nYour choice ({options_str}): "

    if use_chat_template and has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": META_SETUP_PROMPT},
            {"role": "user", "content": llm_prompt}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        return f"{META_SETUP_PROMPT}\n\n{llm_prompt}"


# ============================================================================
# STEERING
# ============================================================================

class SteeringHook:
    """Hook that adds a steering vector to activations."""

    def __init__(self, steering_vector: torch.Tensor, multiplier: float):
        self.steering_vector = steering_vector
        self.multiplier = multiplier
        self.handle = None

    def __call__(self, module, input, output):
        if isinstance(output, tuple):
            hidden_states = output[0]
            steered = hidden_states + self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)
            return (steered,) + output[1:]
        else:
            return output + self.multiplier * self.steering_vector.unsqueeze(0).unsqueeze(0)

    def register(self, layer_module):
        self.handle = layer_module.register_forward_hook(self)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()


def generate_orthogonal_directions(direction: np.ndarray, num_directions: int) -> List[np.ndarray]:
    """Generate random directions orthogonal to the given direction."""
    hidden_dim = len(direction)
    orthogonal = []

    for _ in range(num_directions):
        random_vec = np.random.randn(hidden_dim)
        random_vec = random_vec - np.dot(random_vec, direction) * direction
        for prev in orthogonal:
            random_vec = random_vec - np.dot(random_vec, prev) * prev
        random_vec = random_vec / np.linalg.norm(random_vec)
        orthogonal.append(random_vec)

    return orthogonal


def get_confidence_response(
    model,
    tokenizer,
    question: Dict,
    layer_idx: Optional[int],
    steering_vector: Optional[np.ndarray],
    multiplier: float,
    use_chat_template: bool
) -> Tuple[str, float, np.ndarray]:
    """Get confidence response, optionally with steering."""
    prompt = format_meta_prompt(question, tokenizer, use_chat_template)

    if layer_idx is not None and steering_vector is not None and multiplier != 0.0:
        # Steering
        steering_tensor = torch.tensor(
            steering_vector,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        ).to(DEVICE)

        if hasattr(model, 'get_base_model'):
            layer_module = model.get_base_model().model.layers[layer_idx]
        else:
            layer_module = model.model.layers[layer_idx]

        hook = SteeringHook(steering_tensor, multiplier)
        hook.register(layer_module)

        try:
            inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
            with torch.no_grad():
                outputs = model(**inputs, use_cache=False)
        finally:
            hook.remove()
    else:
        # No steering
        inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs, use_cache=False)

    final_logits = outputs.logits[0, -1, :]
    option_token_ids = [tokenizer.encode(opt, add_special_tokens=False)[0] for opt in META_OPTIONS]
    option_logits = final_logits[option_token_ids]
    option_probs = torch.softmax(option_logits, dim=-1).cpu().numpy()

    response = META_OPTIONS[np.argmax(option_probs)]
    confidence = META_RANGE_MIDPOINTS[response]

    return response, confidence, option_probs


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_steering_experiment(
    model,
    tokenizer,
    questions: List[Dict],
    direct_entropies: np.ndarray,
    layers: List[int],
    directions: Dict,
    multipliers: List[float],
    num_controls: int,
    use_chat_template: bool
) -> Dict:
    """Run steering experiment across layers and directions."""
    print(f"\nRunning steering experiment...")
    print(f"  Layers: {layers}")
    print(f"  Multipliers: {multipliers}")
    print(f"  Questions: {len(questions)}")
    print(f"  Control directions: {num_controls}")

    results = {
        "layers": layers,
        "multipliers": multipliers,
        "num_questions": len(questions),
        "num_controls": num_controls,
        "layer_results": {},
    }

    # Compute entropy stats for alignment calculation
    entropy_mean = direct_entropies.mean()
    entropy_std = direct_entropies.std()

    for layer_idx in tqdm(layers, desc="Steering layers"):
        introspection_dir = np.array(directions[f"layer_{layer_idx}_introspection"])
        control_dirs = generate_orthogonal_directions(introspection_dir, num_controls)

        layer_results = {
            "baseline": [],
            "introspection": {m: [] for m in multipliers},
            "controls": {f"control_{i}": {m: [] for m in multipliers} for i in range(num_controls)},
        }

        # Baseline (no steering)
        for q_idx, question in enumerate(tqdm(questions, desc="Baseline", leave=False)):
            response, confidence, probs = get_confidence_response(
                model, tokenizer, question, None, None, 0.0, use_chat_template
            )

            entropy = direct_entropies[q_idx]
            entropy_z = (entropy - entropy_mean) / entropy_std
            confidence_z = (confidence - 0.5) / 0.25
            alignment = -entropy_z * confidence_z

            layer_results["baseline"].append({
                "question_idx": q_idx,
                "response": response,
                "confidence": confidence,
                "entropy": float(entropy),
                "alignment": float(alignment),
            })

        # Introspection steering
        for mult in tqdm(multipliers, desc="Introspection", leave=False):
            if mult == 0.0:
                layer_results["introspection"][mult] = layer_results["baseline"]
                continue

            for q_idx, question in enumerate(questions):
                response, confidence, _ = get_confidence_response(
                    model, tokenizer, question, layer_idx, introspection_dir, mult, use_chat_template
                )

                entropy = direct_entropies[q_idx]
                entropy_z = (entropy - entropy_mean) / entropy_std
                confidence_z = (confidence - 0.5) / 0.25
                alignment = -entropy_z * confidence_z

                layer_results["introspection"][mult].append({
                    "question_idx": q_idx,
                    "response": response,
                    "confidence": confidence,
                    "alignment": float(alignment),
                })

        # Control steering
        for ctrl_idx, ctrl_dir in enumerate(control_dirs):
            for mult in tqdm(multipliers, desc=f"Control {ctrl_idx}", leave=False):
                if mult == 0.0:
                    layer_results["controls"][f"control_{ctrl_idx}"][mult] = layer_results["baseline"]
                    continue

                for q_idx, question in enumerate(questions):
                    response, confidence, _ = get_confidence_response(
                        model, tokenizer, question, layer_idx, ctrl_dir, mult, use_chat_template
                    )

                    entropy = direct_entropies[q_idx]
                    entropy_z = (entropy - entropy_mean) / entropy_std
                    confidence_z = (confidence - 0.5) / 0.25
                    alignment = -entropy_z * confidence_z

                    layer_results["controls"][f"control_{ctrl_idx}"][mult].append({
                        "question_idx": q_idx,
                        "response": response,
                        "confidence": confidence,
                        "alignment": float(alignment),
                    })

        results["layer_results"][layer_idx] = layer_results
        torch.cuda.empty_cache()

    return results


def analyze_results(results: Dict) -> Dict:
    """Compute summary statistics."""
    analysis = {
        "layers": results["layers"],
        "multipliers": results["multipliers"],
        "effects": {},
    }

    for layer_idx in results["layers"]:
        lr = results["layer_results"][layer_idx]
        multipliers = results["multipliers"]

        baseline_align = np.mean([r["alignment"] for r in lr["baseline"]])
        baseline_conf = np.mean([r["confidence"] for r in lr["baseline"]])

        effects = {"introspection": {}, "control_avg": {}}

        for mult in multipliers:
            # Introspection
            intro_align = np.mean([r["alignment"] for r in lr["introspection"][mult]])
            intro_conf = np.mean([r["confidence"] for r in lr["introspection"][mult]])
            effects["introspection"][mult] = {
                "alignment": float(intro_align),
                "alignment_change": float(intro_align - baseline_align),
                "confidence": float(intro_conf),
                "confidence_change": float(intro_conf - baseline_conf),
            }

            # Control average
            ctrl_aligns = []
            ctrl_confs = []
            for ctrl_key in lr["controls"]:
                ctrl_aligns.extend([r["alignment"] for r in lr["controls"][ctrl_key][mult]])
                ctrl_confs.extend([r["confidence"] for r in lr["controls"][ctrl_key][mult]])
            effects["control_avg"][mult] = {
                "alignment": float(np.mean(ctrl_aligns)),
                "alignment_change": float(np.mean(ctrl_aligns) - baseline_align),
                "confidence": float(np.mean(ctrl_confs)),
                "confidence_change": float(np.mean(ctrl_confs) - baseline_conf),
            }

        # Compute slopes
        intro_slope = np.polyfit(multipliers, [effects["introspection"][m]["alignment_change"] for m in multipliers], 1)[0]
        ctrl_slope = np.polyfit(multipliers, [effects["control_avg"][m]["alignment_change"] for m in multipliers], 1)[0]

        analysis["effects"][layer_idx] = {
            "by_multiplier": effects,
            "slopes": {
                "introspection": float(intro_slope),
                "control_avg": float(ctrl_slope),
            },
            "baseline_alignment": float(baseline_align),
            "baseline_confidence": float(baseline_conf),
        }

    return analysis


def plot_results(analysis: Dict, output_prefix: str):
    """Create visualizations."""
    layers = analysis["layers"]
    multipliers = analysis["multipliers"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Slopes by layer
    ax1 = axes[0]
    intro_slopes = [analysis["effects"][l]["slopes"]["introspection"] for l in layers]
    ctrl_slopes = [analysis["effects"][l]["slopes"]["control_avg"] for l in layers]

    x = np.arange(len(layers))
    width = 0.35
    ax1.bar(x - width/2, intro_slopes, width, label='Introspection', color='green', alpha=0.7)
    ax1.bar(x + width/2, ctrl_slopes, width, label='Control (avg)', color='gray', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax1.set_xticks(x)
    ax1.set_xticklabels(layers)
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Alignment Slope (Δalign / Δmult)")
    ax1.set_title("Steering Effect on Alignment")
    ax1.legend()

    # Plot 2: Best layer detail
    best_layer = max(layers, key=lambda l: analysis["effects"][l]["slopes"]["introspection"])
    ax2 = axes[1]

    intro_align = [analysis["effects"][best_layer]["by_multiplier"]["introspection"][m]["alignment_change"] for m in multipliers]
    ctrl_align = [analysis["effects"][best_layer]["by_multiplier"]["control_avg"][m]["alignment_change"] for m in multipliers]

    ax2.plot(multipliers, intro_align, 'o-', label='Introspection', linewidth=2, color='green')
    ax2.plot(multipliers, ctrl_align, '^--', label='Control', linewidth=2, color='gray', alpha=0.7)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.3)
    ax2.set_xlabel("Steering Multiplier")
    ax2.set_ylabel("Δ Alignment")
    ax2.set_title(f"Alignment Change (Layer {best_layer})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary
    ax3 = axes[2]
    ax3.axis('off')

    intro_slope = analysis["effects"][best_layer]["slopes"]["introspection"]
    ctrl_slope = analysis["effects"][best_layer]["slopes"]["control_avg"]

    summary = f"""
STEERING EXPERIMENT SUMMARY

Best Layer: {best_layer}
  Introspection slope: {intro_slope:.4f}
  Control slope: {ctrl_slope:.4f}
  Difference: {intro_slope - ctrl_slope:.4f}

Interpretation:
"""
    if intro_slope > 0 and intro_slope > ctrl_slope + 0.01:
        summary += """  ✓ Positive introspection steering effect
  Steering increases alignment!
  Effect stronger than controls."""
    elif intro_slope > 0:
        summary += """  ⚠ Weak introspection steering effect
  Steering increases alignment
  but not clearly above controls."""
    else:
        summary += """  ✗ No introspection steering effect
  Steering does not improve alignment."""

    ax3.text(0.1, 0.9, summary, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_prefix}_steering_results.png", dpi=300, bbox_inches='tight')
    print(f"Saved {output_prefix}_steering_results.png")
    plt.close()


def print_summary(analysis: Dict):
    """Print summary of results."""
    print("\n" + "=" * 70)
    print("STEERING EXPERIMENT RESULTS")
    print("=" * 70)

    print("\n--- Alignment Slopes by Layer ---")
    print(f"{'Layer':<8} {'Introspection':<15} {'Control':<15}")
    print("-" * 40)

    for layer in analysis["layers"]:
        s = analysis["effects"][layer]["slopes"]
        print(f"{layer:<8} {s['introspection']:<15.4f} {s['control_avg']:<15.4f}")

    # Best layer
    best_layer = max(analysis["layers"], key=lambda l: analysis["effects"][l]["slopes"]["introspection"])
    best_intro = analysis["effects"][best_layer]["slopes"]["introspection"]
    best_ctrl = analysis["effects"][best_layer]["slopes"]["control_avg"]

    print(f"\nBest introspection steering: Layer {best_layer}")
    print(f"  Introspection slope: {best_intro:.4f}")
    print(f"  Control slope: {best_ctrl:.4f}")

    if best_intro > 0 and best_intro > best_ctrl + 0.01:
        print("\n✓ Evidence for causal introspection effect!")
    elif best_intro > 0:
        print("\n⚠ Weak effect, not clearly separable from controls")
    else:
        print("\n✗ No introspection steering effect found")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Device: {DEVICE}")

    # Load probe results
    print(f"\nLoading probe results from {PROBE_RESULTS_PATH}...")
    with open(PROBE_RESULTS_PATH, "r") as f:
        probe_results = json.load(f)

    # Load directions
    print(f"Loading directions from {PROBE_DIRECTIONS_PATH}...")
    directions_data = np.load(PROBE_DIRECTIONS_PATH)
    directions = {k: directions_data[k] for k in directions_data.files}

    # Determine layers to steer
    if STEERING_LAYERS is not None:
        layers = STEERING_LAYERS
    else:
        # Use significant layers from probe
        layers = set()

        # Add all significant introspection layers
        for layer_str, lr in probe_results["layer_results"].items():
            if lr["significant_p05"]:
                layers.add(int(layer_str))

        # Always add the best layer even if not significant
        if "best_layer" in probe_results:
            layers.add(probe_results["best_layer"]["layer"])

        # If no significant layers, use middle layers
        if not layers:
            all_layers = [int(l) for l in probe_results["layer_results"].keys()]
            mid = len(all_layers) // 2
            layers = all_layers[max(0, mid-3):mid+4]

        layers = sorted(layers)

    print(f"Steering layers: {layers}")

    # Load paired data
    print(f"\nLoading paired data from {PAIRED_DATA_PATH}...")
    with open(PAIRED_DATA_PATH, "r") as f:
        paired_data = json.load(f)

    questions = paired_data["questions"][:NUM_STEERING_QUESTIONS]
    direct_entropies = np.array(paired_data["direct_entropies"])[:NUM_STEERING_QUESTIONS]
    print(f"Using {len(questions)} questions")

    # Load model
    print(f"\nLoading model: {BASE_MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        device_map="auto",
        token=HF_TOKEN
    )

    if MODEL_NAME != BASE_MODEL_NAME:
        from peft import PeftModel
        print(f"Loading adapter: {MODEL_NAME}")
        model = PeftModel.from_pretrained(model, MODEL_NAME)

    # Determine chat template usage
    use_chat_template = has_chat_template(tokenizer) and not is_base_model(BASE_MODEL_NAME)
    print(f"Using chat template: {use_chat_template}")

    # Run experiment
    results = run_steering_experiment(
        model, tokenizer, questions, direct_entropies,
        layers, directions, STEERING_MULTIPLIERS, NUM_CONTROL_DIRECTIONS,
        use_chat_template
    )

    # Analyze
    analysis = analyze_results(results)

    # Save results
    output_results = f"{OUTPUT_PREFIX}_steering_results.json"
    with open(output_results, "w") as f:
        json.dump(results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\nSaved {output_results}")

    output_analysis = f"{OUTPUT_PREFIX}_steering_analysis.json"
    with open(output_analysis, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"Saved {output_analysis}")

    # Print and plot
    print_summary(analysis)
    plot_results(analysis, OUTPUT_PREFIX)

    print("\n✓ Steering experiment complete!")


if __name__ == "__main__":
    main()
