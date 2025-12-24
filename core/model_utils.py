"""
Model loading and naming utilities for introspection experiments.

Provides consistent model loading, run naming, and detection of model properties
(base vs instruct, chat template availability, etc.)
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from dotenv import load_dotenv
from typing import Optional, Tuple

load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# Device detection
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


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


def get_model_short_name(model_name: str) -> str:
    """
    Extract a short, filesystem-safe name from a model path.

    Examples:
        "meta-llama/Llama-3.1-8B-Instruct" -> "Llama-3.1-8B-Instruct"
        "/path/to/adapter" -> "adapter"
    """
    # Handle HuggingFace paths
    if "/" in model_name:
        parts = model_name.split("/")
        return parts[-1]
    return model_name


def get_run_name(
    base_model: str,
    dataset: str,
    task: str = "probe",
    adapter: Optional[str] = None,
    num_questions: Optional[int] = None,
    seed: Optional[int] = None
) -> str:
    """
    Generate a consistent run name for output files.

    Format: {model_short}[_adapter]_{dataset}_{task}[_n{num}][_s{seed}]

    Args:
        base_model: Base model name/path
        dataset: Dataset name (e.g., "SimpleMC", "GPQA")
        task: Task type (e.g., "probe", "steer", "delegate")
        adapter: Optional adapter path
        num_questions: Optional number of questions
        seed: Optional random seed

    Returns:
        Filesystem-safe run name string
    """
    model_short = get_model_short_name(base_model)

    parts = [model_short]

    if adapter:
        adapter_short = get_model_short_name(adapter)
        parts.append(f"adapter-{adapter_short}")

    parts.append(dataset)
    parts.append(task)

    if num_questions:
        parts.append(f"n{num_questions}")

    if seed is not None:
        parts.append(f"s{seed}")

    return "_".join(parts)


def load_model_and_tokenizer(
    base_model_name: str,
    adapter_path: Optional[str] = None,
    device_map: str = "auto",
    torch_dtype: Optional[torch.dtype] = None
) -> Tuple:
    """
    Load a model and tokenizer, optionally with a PEFT adapter.

    Args:
        base_model_name: HuggingFace model name or path
        adapter_path: Optional path to PEFT adapter
        device_map: Device mapping strategy
        torch_dtype: Data type (auto-detected if None)

    Returns:
        Tuple of (model, tokenizer, num_layers)
    """
    if torch_dtype is None:
        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32

    print(f"Loading model: {base_model_name}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        token=HF_TOKEN
    )

    if adapter_path:
        try:
            from peft import PeftModel
            print(f"Loading adapter: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
        except Exception as e:
            raise RuntimeError(f"Error loading adapter {adapter_path}: {e}")

    # Get number of layers
    if hasattr(model, 'get_base_model'):
        base = model.get_base_model()
        num_layers = len(base.model.layers)
    else:
        num_layers = len(model.model.layers)

    print(f"Model has {num_layers} layers, device: {DEVICE}")

    return model, tokenizer, num_layers


def should_use_chat_template(model_name: str, tokenizer) -> bool:
    """Determine whether to use chat template based on model type and tokenizer."""
    return has_chat_template(tokenizer) and not is_base_model(model_name)
